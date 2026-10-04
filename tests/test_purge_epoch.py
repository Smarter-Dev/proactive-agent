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
    StaleHistoryError,
    build_snapshot,
)
from proactive_agent.keys import history_key, purge_epoch_key
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

    loads = []
    real_load = repository.load

    async def counting_load(guild_id):
        loads.append(guild_id)
        return await real_load(guild_id)

    repository.load = counting_load
    await runtime.process(batch())
    await runtime.process(batch())

    assert loads == [GUILD]
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


async def test_a_stale_writer_cannot_overwrite_a_purged_snapshot(redis_client):
    # The backstop: a wake that lost its lease mid-run, after its start-of-
    # wake check, tries to save revision 4 over the purge's revision 10.
    # Refused, and the runtime drops its stale in-RAM copy.
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    await repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    runtime = make_runtime(redis_client, api, repository, writer)
    purged = build_snapshot(GUILD, [], revision=10)
    real_wake = runtime.engine.wake

    async def purge_lands_mid_wake(**kwargs):
        writer.discard(GUILD)
        assert await repository.cache(purged)
        return await real_wake(**kwargs)

    runtime.engine.wake = purge_lands_mid_wake

    with pytest.raises(StaleHistoryError):
        await runtime.process(batch())

    assert await repository.load(GUILD) == purged
    assert runtime.history_loaded is False
    assert runtime.engine.agent_runner.history == []
    await writer.close(timeout=1)
    assert api.puts == []


async def test_an_unreadable_v1_snapshot_does_not_block_the_postgres_copy(
    redis_client,
):
    api = FakeAPI()
    durable = build_snapshot(GUILD, [{"from": "postgres"}], revision=2)
    api.durable[GUILD] = durable
    repository = GuildHistoryRepository(redis_client, api)
    broken = build_snapshot(GUILD, [{"x": 1}], revision=9).model_copy(
        update={"checksum": "0" * 64}
    )
    await redis_client.set(history_key(GUILD), broken.model_dump_json())

    assert await repository.load(GUILD) == durable
    assert await repository.load_canonical(GUILD) == durable


async def test_alternating_replicas_hand_off_without_stale_saves(redis_client):
    # Wakes alternate between two replicas. Each must notice at wake start
    # that the other stored a newer revision, reload, and save on top of
    # it: no StaleHistoryError, no lost wake.
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    replicas = []
    for _ in range(2):
        repository = GuildHistoryRepository(redis_client, api)
        writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
        replicas.append(make_runtime(redis_client, api, repository, writer))

    for turn in range(6):
        await replicas[turn % 2].process(batch())

    stored = await replicas[0].history_repository.load(GUILD)
    assert stored.revision == 6
    assert len(stored.history) == 12  # every wake's request/response pair
    # The replica that ran the last wake started from the other's save.
    assert replicas[1].engine.seen_histories[-1].count('"wake"') == 5
    assert api.memory_reads == 2  # a hand-off reloads history, not memory
    for runtime in replicas:
        await runtime.history_writer.close(timeout=1)
