from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest

from proactive_agent.api import ApplicationAPIError
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import history_key, legacy_history_key, purge_epoch_key


class FakeAPI:
    """Postgres stand-in that enforces the server's monotonic revisions."""

    def __init__(self, durable=None):
        self.durable = durable
        self.puts = []
        self.fail_puts = False
        # Every PUT the writer tried, accepted or not.
        self.attempts = []

    async def get_history(self, guild_id):
        return self.durable

    async def put_history(self, snapshot):
        self.attempts.append(snapshot)
        if self.fail_puts:
            raise ApplicationAPIError(500, "boom")
        if self.durable is not None and snapshot.revision <= self.durable.revision:
            if snapshot.revision == self.durable.revision and (
                snapshot.checksum == self.durable.checksum
            ):
                return
            raise ApplicationAPIError(409, "conflict")
        self.puts.append(snapshot)
        self.durable = snapshot


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
async def test_postgres_snapshot_wins_over_legacy_history(redis_client):
    # Restore order is v1 -> Postgres -> legacy: a purge writes Postgres, so
    # the legacy key must never outrank it.
    durable = build_snapshot("111", [{"from": "postgres"}], revision=9)
    api = FakeAPI(durable=durable)
    repository = GuildHistoryRepository(redis_client, api)
    await redis_client.set(
        legacy_history_key("111"), json.dumps([{"from": "embedded-bot"}])
    )

    loaded = await repository.load("111")

    assert loaded == durable


@pytest.mark.asyncio
async def test_legacy_history_is_never_restored_after_a_purge(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    await redis_client.set(
        legacy_history_key("111"), json.dumps([{"from": "kai 111111111111111111"}])
    )
    await redis_client.incr(purge_epoch_key("111"))

    loaded = await repository.load("111")

    assert loaded.history == []
    assert loaded.revision == 0
    assert await redis_client.get(history_key("111")) is None


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


@pytest.mark.asyncio
async def test_replace_purged_discards_dirty_copy_and_writes_postgres_first(
    redis_client,
):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(
        repository, api, debounce_seconds=0.05, retry_base_seconds=0.001
    )
    stale = await writer.save(
        guild_id="111", history=[{"raw": "kai said x"}], previous_revision=0
    )

    purged, _v1 = await writer.replace_purged(
        "111", [{"note": "clean"}], previous_revision=stale.revision
    )
    await asyncio.sleep(0.1)

    assert [snapshot.history for snapshot in api.puts] == [[{"note": "clean"}]]
    assert purged.revision == stale.revision + 1000
    assert (await repository.load("111")).history == [{"note": "clean"}]
    await writer.close(timeout=1)
    assert [snapshot.history for snapshot in api.attempts] == [[{"note": "clean"}]]


@pytest.mark.asyncio
async def test_replace_purged_goes_above_a_newer_postgres_revision(redis_client):
    api = FakeAPI(durable=build_snapshot("111", [{"x": 1}], revision=7))
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api)

    purged, _v1 = await writer.replace_purged(
        "111", [{"note": "clean"}], previous_revision=3
    )

    assert purged.revision == 7 + 1000
    await writer.close(timeout=1)


@pytest.mark.asyncio
async def test_failed_purged_put_leaves_redis_untouched(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api)
    before = build_snapshot("111", [{"raw": "old"}], revision=4)
    await repository.cache(before)
    api.fail_puts = True

    with pytest.raises(ApplicationAPIError):
        await writer.replace_purged("111", [{"note": "clean"}], previous_revision=4)

    assert await repository.load("111") == before
    await writer.close(timeout=1)


@pytest.mark.asyncio
async def test_restore_puts_back_a_discarded_dirty_copy(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0.01)
    await writer.save(guild_id="111", history=[{"wake": 1}], previous_revision=0)

    taken = writer.discard("111")
    writer.restore(taken)
    await asyncio.sleep(0.05)

    assert [snapshot.history for snapshot in api.puts] == [[{"wake": 1}]]
    await writer.close(timeout=1)


@pytest.mark.asyncio
async def test_superseded_flush_is_dropped_not_retried_forever(redis_client):
    # Another replica's purge wrote revision 5; this replica's stale dirty
    # revision 5 (different content) can never land and must not loop.
    api = FakeAPI(durable=build_snapshot("111", [{"note": "clean"}], revision=5))
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(
        repository, api, debounce_seconds=0.01, retry_base_seconds=0.001
    )
    await writer.save(guild_id="111", history=[{"raw": "kai"}], previous_revision=4)

    await asyncio.wait_for(writer.flush("111"), timeout=1)

    assert api.puts == []
    assert api.durable.history == [{"note": "clean"}]
    await writer.close(timeout=1)


@pytest.mark.asyncio
async def test_a_failed_flush_logs_no_history_text(redis_client, caplog):
    class FailingAPI(FakeAPI):
        async def put_history(self, snapshot):
            if not self.puts:
                self.puts.append(None)
                raise RuntimeError("rejected: what someone said")
            self.puts.append(snapshot)

    api = FailingAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(
        repository, api, debounce_seconds=60, retry_base_seconds=0.001
    )
    await writer.save(
        guild_id="111", history=[{"content": "what someone said"}], previous_revision=0
    )

    with caplog.at_level("ERROR"):
        await writer.close(timeout=1)

    assert "proactive history flush failed guild=111" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "what someone said" not in caplog.text
