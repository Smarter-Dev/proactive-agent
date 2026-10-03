"""Runtimes reload history once a purge bumps the guild's purge epoch."""

from __future__ import annotations

import fakeredis.aioredis
import pytest
from purge_fakes import (
    CHANNEL,
    GUILD,
    TARGET,
    FakeAPI,
    batch,
    dump,
    make_runtime,
    raw_history_with_target,
)

from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import purge_epoch_key
from proactive_agent.runtime import GuildRuntimeRegistry


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


async def test_wake_reloads_history_and_memory_after_an_epoch_bump(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    await repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    runtime = make_runtime(redis_client, api, repository, writer)

    await runtime.process(batch())
    assert TARGET in runtime.engine.seen_histories[0]
    assert api.memory_reads == 1

    # Another replica purges: v1 now holds a clean note, epoch moves.
    writer.discard(GUILD)
    await repository.cache(build_snapshot(GUILD, [], revision=10))
    await redis_client.incr(purge_epoch_key(GUILD))

    await runtime.process(batch())

    assert runtime.engine.seen_histories[1] == "[]"
    assert api.memory_reads == 2
    assert runtime.history_revision == 11
    await writer.close(timeout=1)


async def test_unchanged_epoch_keeps_the_loaded_history(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    await redis_client.incr(purge_epoch_key(GUILD))
    runtime = make_runtime(redis_client, api, repository, writer)

    await runtime.process(batch())
    # A write behind the runtime's back is not reloaded without an epoch bump.
    await repository.cache(build_snapshot(GUILD, [], revision=50))
    await runtime.process(batch())

    assert runtime.history_revision == 2
    assert api.memory_reads == 1
    await writer.close(timeout=1)


async def test_registry_forget_resets_a_cached_runtime(redis_client):
    api = FakeAPI()
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.history_loaded = True
    runtime.memory_refreshed_at = 123.0
    runtime.memory_block = "old"
    runtime.engine.agent_runner.history = raw_history_with_target()

    async def factory(_guild_id):
        return runtime

    registry = GuildRuntimeRegistry(factory)
    await registry.get(GUILD)
    registry.forget(GUILD)
    registry.forget("555555555555555555")

    assert runtime.history_loaded is False
    assert runtime.memory_refreshed_at == 0
    assert runtime.memory_block == ""
    assert runtime.engine.agent_runner.history == []
    await writer.close(timeout=1)
