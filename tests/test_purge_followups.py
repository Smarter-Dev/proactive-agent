"""Follow-ups: shared-stream deletion and deferred wakes on a tombstoned guild."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from purge_fakes import GUILD, TARGET
from test_purge import Replica, command, honest_model, pending_count, submit
from test_purge_review import envelope, publish

from proactive_agent.history import HistoryUnavailableError
from proactive_agent.keys import (
    DEAD_LETTER_STREAM_KEY,
    PRIVACY_PURGE_STREAM_KEY,
    attempts_key,
    ownership_key,
    wake_stream_key,
)
from proactive_agent.queue import WAKE_GROUP, RedisWakeQueue
from proactive_agent.worker import ProactiveWorker

BOT_GROUP = "smarter-dev-bot"


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


async def bot_group(redis_client):
    await redis_client.xgroup_create(PRIVACY_PURGE_STREAM_KEY, BOT_GROUP, id="0")


# 1. The worker deletes an entry only once every other group is done with it.


async def test_an_entry_the_bot_has_not_read_is_left_in_place(redis_client, world):
    replica = Replica(redis_client, world, honest_model([]), name="a")
    await replica.consumer.initialize()
    await bot_group(redis_client)
    await submit(redis_client, "not json " + TARGET)

    assert await replica.consumer.poll_once() == 1

    assert await pending_count(redis_client) == 0  # the worker's group acked
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1  # bot unread
    await replica.writer.close(timeout=1)


async def test_an_entry_the_bot_still_holds_pending_is_left_in_place(
    redis_client, world
):
    replica = Replica(redis_client, world, honest_model([]), name="a")
    replica.consumer._max_deliveries = 0  # the delivery-limit drop
    await replica.consumer.initialize()
    await bot_group(redis_client)
    await submit(redis_client, command())
    await redis_client.xreadgroup(BOT_GROUP, "bot", {PRIVACY_PURGE_STREAM_KEY: ">"})

    await replica.consumer.poll_once()

    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1  # bot pending
    await replica.writer.close(timeout=1)


async def test_an_entry_the_bot_has_acked_is_deleted(redis_client, world):
    replica = Replica(redis_client, world, honest_model([]), name="a")
    replica.consumer._max_deliveries = 0
    await replica.consumer.initialize()
    await bot_group(redis_client)
    entry_id = await submit(redis_client, command())
    await redis_client.xreadgroup(BOT_GROUP, "bot", {PRIVACY_PURGE_STREAM_KEY: ">"})
    await redis_client.xack(PRIVACY_PURGE_STREAM_KEY, BOT_GROUP, entry_id)

    await replica.consumer.poll_once()

    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_an_unknown_run_is_deleted_whatever_the_bot_did(redis_client, world):
    world.ack_status = 404
    replica = Replica(redis_client, world, honest_model([]), name="a")
    await replica.consumer.initialize()
    await bot_group(redis_client)
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0
    await replica.writer.close(timeout=1)


# 2. A wake on a tombstoned guild is deferred, never counted or dead-lettered.


async def test_a_tombstoned_wake_is_deferred_not_failed(redis_client):
    await redis_client.set(ownership_key(GUILD), "external")
    # Reclaim at once, standing in for the reclaim timeout passing.
    queue = RedisWakeQueue(redis_client, consumer_name="w", reclaim_idle_seconds=0)
    await queue.initialize()
    await publish(redis_client, envelope("someone mentioned the bot"))
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=HistoryUnavailableError("tombstoned")),
        report_failure=AsyncMock(),
        record_unavailable_retries=AsyncMock(),
    )
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    # One attempt would already dead-letter an ordinary failure.
    worker = ProactiveWorker(queue, runtimes, max_attempts=1)
    ready = await queue.read_ready(block_ms=1)

    for _ in range(3):
        await worker._run_guild(GUILD, ready)

    assert runtime.process.await_count == 3
    batch_wake_ids = [
        key async for key in redis_client.scan_iter(match=attempts_key(GUILD, "*"))
    ]
    assert batch_wake_ids == []  # no attempt counted
    assert await redis_client.xlen(DEAD_LETTER_STREAM_KEY) == 0
    runtime.report_failure.assert_not_awaited()
    # Still pending in the wake stream, to be reclaimed later.
    summary = await redis_client.xpending(wake_stream_key(GUILD), WAKE_GROUP)
    assert summary["pending"] == 1
