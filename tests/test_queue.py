from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import fakeredis.aioredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from proactive_agent.contracts import NotificationEnvelope
from proactive_agent.keys import (
    DEAD_LETTER_STREAM_KEY,
    READY_STREAM_KEY,
    ownership_key,
    pending_key,
    wake_stream_key,
)
from proactive_agent.queue import RedisWakeQueue


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


def envelope(guild_id: str, *, body: str, wakes: bool) -> NotificationEnvelope:
    return NotificationEnvelope(
        schema_version=1,
        notification_id=uuid4(),
        guild_id=guild_id,
        channel_id="222",
        channel_name="general",
        kind="mention" if wakes else "reaction",
        created_at=datetime(2026, 9, 1, 16, 0, tzinfo=UTC),
        body=body,
        message_ids=("333",),
        wakes=wakes,
        passive=False,
        watcher_usage={},
        trace_id=uuid4(),
    )


@pytest.mark.asyncio
async def test_idle_ready_stream_socket_timeout_is_an_empty_poll() -> None:
    class TimedOutRedis:
        async def xreadgroup(self, *args, **kwargs):
            raise RedisTimeoutError("idle block elapsed")

    queue = RedisWakeQueue(TimedOutRedis(), consumer_name="worker-1")
    assert await queue.read_ready(block_ms=5_000) == ()


async def publish_wake(redis_client, item: NotificationEnvelope) -> str:
    stream_id = await redis_client.xadd(
        wake_stream_key(item.guild_id), {"payload": item.model_dump_json()}
    )
    await redis_client.xadd(READY_STREAM_KEY, {"guild_id": item.guild_id})
    return stream_id.decode()


@pytest.mark.asyncio
async def test_two_guilds_build_separate_batches(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    await queue.initialize()
    await publish_wake(redis_client, envelope("111", body="guild-one", wakes=True))
    await publish_wake(redis_client, envelope("999", body="guild-two", wakes=True))

    ready = await queue.read_ready(block_ms=1)
    by_guild = {
        guild_id: tuple(record for record in ready if record.guild_id == guild_id)
        for guild_id in {record.guild_id for record in ready}
    }
    first = await queue.build_batch("111", by_guild["111"])
    second = await queue.build_batch("999", by_guild["999"])

    assert [item.body for item in first.notifications] == ["guild-one"]
    assert [item.body for item in second.notifications] == ["guild-two"]
    assert first.guild_id != second.guild_id


@pytest.mark.asyncio
async def test_pending_is_claimed_once_and_survives_retry(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    await queue.initialize()
    pending = envelope("111", body="pending-context", wakes=False)
    await redis_client.rpush(pending_key("111"), pending.model_dump_json())
    await publish_wake(redis_client, envelope("111", body="wake", wakes=True))
    ready = await queue.read_ready(block_ms=1)

    first = await queue.build_batch("111", ready)
    assert [item.body for item in first.notifications] == [
        "pending-context",
        "wake",
    ]
    assert await redis_client.llen(pending_key("111")) == 0

    # A retry reads the durable in-progress list, not the now-empty pending key.
    raw = await redis_client.lrange(
        f"proactive:v1:{{guild:111}}:batch:{first.wake_id}", 0, -1
    )
    assert len(raw) == 1

    await queue.acknowledge(first)
    assert (
        await redis_client.exists(f"proactive:v1:{{guild:111}}:batch:{first.wake_id}")
        == 0
    )


@pytest.mark.asyncio
async def test_guild_lease_serializes_shared_workers(redis_client):
    first_queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    second_queue = RedisWakeQueue(redis_client, consumer_name="worker-2")
    await redis_client.mset(
        {ownership_key("111"): "external", ownership_key("999"): "external"}
    )

    first = await first_queue.acquire_lease("111")
    blocked = await second_queue.acquire_lease("111")
    other_guild = await second_queue.acquire_lease("999")

    assert first is not None
    assert blocked is None
    assert other_guild is not None
    assert await first.renew() is True
    await first.release()
    assert await second_queue.acquire_lease("111") is not None


@pytest.mark.asyncio
async def test_lease_is_fenced_when_bot_owns_the_guild(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    await redis_client.set(ownership_key("111"), "embedded")
    assert await queue.acquire_lease("111") is None

    await redis_client.set(ownership_key("111"), "external")
    lease = await queue.acquire_lease("111")
    assert lease is not None
    await redis_client.set(ownership_key("111"), "embedded")
    assert await lease.renew() is False


@pytest.mark.asyncio
async def test_failed_batch_dead_letters_at_retry_ceiling(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    await queue.initialize()
    await publish_wake(redis_client, envelope("111", body="wake", wakes=True))
    ready = await queue.read_ready(block_ms=1)
    batch = await queue.build_batch("111", ready)

    assert not await queue.record_failure(batch, error="first", max_attempts=2)
    assert await queue.record_failure(batch, error="second", max_attempts=2)

    dead = await redis_client.xrange(DEAD_LETTER_STREAM_KEY)
    assert len(dead) == 1
    assert dead[0][1][b"attempts"] == b"2"
    assert b"wake" in dead[0][1][b"payload"]
    assert (await redis_client.xpending(wake_stream_key("111"), "proactive-agent-v1"))[
        "pending"
    ] == 0


@pytest.mark.asyncio
async def test_acknowledging_one_guild_does_not_ack_another(redis_client):
    queue = RedisWakeQueue(redis_client, consumer_name="worker-1")
    await queue.initialize()
    await publish_wake(redis_client, envelope("111", body="one", wakes=True))
    await publish_wake(redis_client, envelope("999", body="two", wakes=True))
    ready = await queue.read_ready(block_ms=1)
    first_ready = tuple(record for record in ready if record.guild_id == "111")
    second_ready = tuple(record for record in ready if record.guild_id == "999")
    first = await queue.build_batch("111", first_ready)
    second = await queue.build_batch("999", second_ready)

    await queue.acknowledge(first)

    pending_second = await redis_client.xpending(
        wake_stream_key("999"), "proactive-agent-v1"
    )
    assert pending_second["pending"] == 1
    await queue.acknowledge(second)
