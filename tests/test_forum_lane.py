"""A forum post's notifications travel on keys of their own (#248).

An old worker (main's queue and worker, before forum support) shares Redis,
consumer groups and guild leases with a new one during a rolling update. It
must never take a forum post's wake: it cannot answer a post, and taking it
would acknowledge and delete it unanswered.
"""

from __future__ import annotations

import subprocess
import sys
import types
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import fakeredis.aioredis
import pytest

from proactive_agent.contracts import NotificationEnvelope
from proactive_agent.keys import (
    FORUM_READY_STREAM_KEY,
    READY_STREAM_KEY,
    batch_key,
    forum_batch_key,
    forum_pending_dropped_key,
    forum_pending_key,
    forum_wake_stream_key,
    ownership_key,
    pending_key,
    wake_stream_key,
)
from proactive_agent.purge import PrivacyPurgeConsumer
from proactive_agent.queue import RedisWakeQueue
from proactive_agent.worker import ProactiveWorker

GUILD = "111"
CHANNEL = "222"
POST = "333"


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


def _main_module(path: str, name: str) -> types.ModuleType:
    """The module as it is on main, before forum support."""
    source = subprocess.check_output(["git", "show", f"main:{path}"], text=True)
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"main:{path}", "exec"), module.__dict__)
    return module


def envelope(channel_id: str, body: str, *, wakes: bool = True) -> NotificationEnvelope:
    return NotificationEnvelope(
        schema_version=1,
        notification_id=uuid4(),
        guild_id=GUILD,
        channel_id=channel_id,
        channel_name="general",
        kind="mention" if wakes else "reaction",
        created_at=datetime.now(UTC),
        body=body,
        message_ids=("444",),
        wakes=wakes,
        passive=False,
        watcher_usage={},
        trace_id=uuid4(),
    )


async def publish(redis_client, item: NotificationEnvelope, *, forum: bool) -> None:
    """What the bot's publish writes (smarter_dev/bot/proactive/redis_queue.py)."""
    payload = item.model_dump_json()
    if not item.wakes:
        key = forum_pending_key(GUILD) if forum else pending_key(GUILD)
        await redis_client.rpush(key, payload)
        return
    stream = forum_wake_stream_key(GUILD) if forum else wake_stream_key(GUILD)
    await redis_client.xadd(stream, {"payload": payload})
    ready = FORUM_READY_STREAM_KEY if forum else READY_STREAM_KEY
    await redis_client.xadd(ready, {"guild_id": GUILD})


class RecordingRuntimes:
    """One guild runtime that records the bodies of each batch it processes."""

    def __init__(self):
        self.batches: list[list[str]] = []

        async def process(batch):
            self.batches.append([item.body for item in batch.notifications])

        self._runtime = SimpleNamespace(process=process)

    async def get(self, _guild_id):
        return self._runtime


def test_the_forum_keys_are_the_bots():
    """Must match smarter_dev/bot/proactive/redis_queue.py."""
    assert FORUM_READY_STREAM_KEY == "proactive:v1:forum-ready"
    assert forum_wake_stream_key(GUILD) == "proactive:v1:{guild:111}:forum-wake"
    assert forum_pending_key(GUILD) == "proactive:v1:{guild:111}:forum-pending"
    assert (
        forum_pending_dropped_key(GUILD)
        == "proactive:v1:{guild:111}:forum-pending-dropped"
    )


async def test_an_old_replica_leaves_a_forum_posts_wake_for_a_new_one(redis_client):
    old_queue_module = _main_module(
        "src/proactive_agent/queue.py", "proactive_agent._main_queue"
    )
    old_worker_module = _main_module(
        "src/proactive_agent/worker.py", "proactive_agent._main_worker"
    )
    await redis_client.set(ownership_key(GUILD), "external")
    old_queue = old_queue_module.RedisWakeQueue(redis_client, consumer_name="old")
    new_queue = RedisWakeQueue(redis_client, consumer_name="new")
    await old_queue.initialize()
    await new_queue.initialize()
    old_runtimes = RecordingRuntimes()
    new_runtimes = RecordingRuntimes()
    old_worker = old_worker_module.ProactiveWorker(old_queue, old_runtimes)
    new_worker = ProactiveWorker(new_queue, new_runtimes)

    await publish(redis_client, envelope(POST, "post context", wakes=False), forum=True)
    await publish(redis_client, envelope(POST, "post question"), forum=True)
    await publish(redis_client, envelope(CHANNEL, "channel question"), forum=False)

    # The old replica gets there first and runs everything it can see.
    old_ready = await old_queue.read_ready(block_ms=1)
    assert [record.guild_id for record in old_ready] == [GUILD]
    await old_worker._run_guild(GUILD, old_ready)
    assert await old_queue.reclaim_ready() == ()

    assert old_runtimes.batches == [["channel question"]]
    assert await redis_client.xlen(forum_wake_stream_key(GUILD)) == 1
    assert await redis_client.llen(forum_pending_key(GUILD)) == 1

    # The new replica then gets the post's wake, context included.
    new_ready = await new_queue.read_ready(block_ms=1)
    await new_worker._run_guild(GUILD, new_ready)

    assert new_runtimes.batches == [["post context", "post question"]]
    assert await redis_client.xlen(forum_wake_stream_key(GUILD)) == 0
    assert await redis_client.xlen(FORUM_READY_STREAM_KEY) == 0
    assert await redis_client.exists(forum_pending_key(GUILD)) == 0


async def test_a_new_worker_runs_both_lanes_as_one_wake(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="new")
    await queue.initialize()
    await publish(
        redis_client, envelope(CHANNEL, "channel context", wakes=False), forum=False
    )
    await publish(redis_client, envelope(POST, "post context", wakes=False), forum=True)
    await publish(redis_client, envelope(CHANNEL, "channel question"), forum=False)
    await publish(redis_client, envelope(POST, "post question"), forum=True)

    ready = await queue.read_ready(block_ms=1)
    assert sorted(record.forum for record in ready) == [False, True]
    batch = await queue.build_batch(GUILD, ready)

    assert [item.body for item in batch.notifications] == [
        "channel context",
        "post context",
        "channel question",
        "post question",
    ]
    await queue.acknowledge(batch)
    assert await redis_client.xlen(wake_stream_key(GUILD)) == 0
    assert await redis_client.xlen(forum_wake_stream_key(GUILD)) == 0
    assert await redis_client.xlen(READY_STREAM_KEY) == 0
    assert await redis_client.xlen(FORUM_READY_STREAM_KEY) == 0
    assert (
        await redis_client.exists(
            batch_key(GUILD, batch.wake_id), forum_batch_key(GUILD, batch.wake_id)
        )
        == 0
    )


async def test_a_failed_attempt_keeps_a_posts_context_for_the_retry(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="new", reclaim_idle_seconds=0)
    await queue.initialize()
    await publish(redis_client, envelope(POST, "post context", wakes=False), forum=True)
    await publish(redis_client, envelope(POST, "post question"), forum=True)
    first = await queue.build_batch(GUILD, await queue.read_ready(block_ms=1))

    assert not await queue.record_failure(first, error="boom", max_attempts=2)
    retry = await queue.build_batch(GUILD, ())

    assert retry.wake_id == first.wake_id
    assert [item.body for item in retry.notifications] == [
        "post context",
        "post question",
    ]


async def test_a_midrun_drain_takes_a_posts_late_notifications(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="new")
    await queue.initialize()
    await publish(redis_client, envelope(CHANNEL, "channel question"), forum=False)
    batch = await queue.build_batch(GUILD, await queue.read_ready(block_ms=1))
    await publish(redis_client, envelope(POST, "late context", wakes=False), forum=True)
    await publish(redis_client, envelope(POST, "late question"), forum=True)

    added, _dropped = await queue.drain_midrun(batch)

    assert [item.body for item in added] == ["late context", "late question"]
    await queue.acknowledge(batch)
    assert await redis_client.xlen(forum_wake_stream_key(GUILD)) == 0


async def test_a_privacy_purge_discards_a_posts_notifications(redis_client):
    await publish(redis_client, envelope(POST, "post context", wakes=False), forum=True)
    await publish(redis_client, envelope(POST, "post question"), forum=True)
    await redis_client.set(forum_pending_dropped_key(GUILD), 3)

    dropped = await PrivacyPurgeConsumer._discard_notifications(
        SimpleNamespace(_redis=redis_client), GUILD
    )

    assert dropped == 2
    assert await redis_client.xlen(forum_wake_stream_key(GUILD)) == 0
    assert (
        await redis_client.exists(
            forum_pending_key(GUILD), forum_pending_dropped_key(GUILD)
        )
        == 0
    )
