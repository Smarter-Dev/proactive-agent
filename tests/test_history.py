from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest

from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import history_key, legacy_history_key


class FakeAPI:
    def __init__(self, durable=None):
        self.durable = durable
        self.puts = []

    async def get_history(self, guild_id):
        return self.durable

    async def put_history(self, snapshot):
        self.puts.append(snapshot)


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


@pytest.mark.asyncio
async def test_redis_hit_avoids_durable_api(redis_client):
    api = FakeAPI(durable=build_snapshot("111", [{"from": "db"}], revision=2))
    repository = GuildHistoryRepository(redis_client, api)
    cached = build_snapshot("111", [{"from": "redis"}], revision=3)
    await repository.cache(cached)

    loaded = await repository.load("111")

    assert loaded == cached


@pytest.mark.asyncio
async def test_redis_miss_restores_postgres_snapshot(redis_client):
    durable = build_snapshot("111", [{"from": "postgres"}], revision=4)
    api = FakeAPI(durable=durable)
    repository = GuildHistoryRepository(redis_client, api)

    loaded = await repository.load("111")

    assert loaded == durable
    assert await redis_client.get(history_key("111")) is not None


@pytest.mark.asyncio
async def test_invalid_redis_snapshot_falls_back_to_postgres(redis_client):
    durable = build_snapshot("111", [{"valid": True}], revision=5)
    api = FakeAPI(durable=durable)
    repository = GuildHistoryRepository(redis_client, api)
    invalid = durable.model_copy(update={"checksum": "0" * 64})
    await redis_client.set(history_key("111"), invalid.model_dump_json())

    loaded = await repository.load("111")

    assert loaded == durable


@pytest.mark.asyncio
async def test_legacy_history_migrates_when_no_durable_copy_exists(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    await redis_client.set(
        legacy_history_key("111"), json.dumps([{"legacy": "message"}])
    )

    loaded = await repository.load("111")

    assert loaded.revision == 1
    assert loaded.history == [{"legacy": "message"}]
    assert await redis_client.get(history_key("111")) is not None


@pytest.mark.asyncio
async def test_live_legacy_history_wins_over_stale_canary_snapshot(redis_client):
    durable = build_snapshot("111", [{"from": "old-canary"}], revision=9)
    api = FakeAPI(durable=durable)
    repository = GuildHistoryRepository(redis_client, api)
    await redis_client.set(
        legacy_history_key("111"), json.dumps([{"from": "embedded-bot"}])
    )

    loaded = await repository.load("111")

    assert loaded.history == [{"from": "embedded-bot"}]
    assert loaded.revision == 9


@pytest.mark.asyncio
async def test_debounce_persists_only_latest_revision(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(
        repository, api, debounce_seconds=0.01, retry_base_seconds=0.001
    )

    first = await writer.save(
        guild_id="111", history=[{"wake": 1}], previous_revision=0
    )
    second = await writer.save(
        guild_id="111", history=[{"wake": 2}], previous_revision=first.revision
    )
    await asyncio.sleep(0.03)

    assert [snapshot.revision for snapshot in api.puts] == [second.revision]
    cached = await repository.load("111")
    assert cached.history == [{"wake": 2}]
    await writer.close()


@pytest.mark.asyncio
async def test_close_flushes_dirty_history_without_waiting_for_debounce(
    redis_client,
):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    await writer.save(guild_id="111", history=[{"wake": 1}], previous_revision=0)

    await writer.close(timeout=1)

    assert len(api.puts) == 1
    assert api.puts[0].history == [{"wake": 1}]
