"""End-to-end privacy purge of the worker's guild state (fakeredis + fake API)."""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import uuid4

import fakeredis.aioredis
import pytest
from purge_fakes import (
    BYSTANDER,
    CHANNEL,
    GUILD,
    TARGET,
    TARGET_NAME,
    batch,
    dump,
    make_runtime,
    raw_history_with_target,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.agent import name_hits
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import (
    PRIVACY_PURGE_STREAM_KEY,
    history_invalid_key,
    history_key,
    legacy_history_key,
    ownership_key,
    privacy_lock_key,
    purge_epoch_key,
)
from proactive_agent.purge import PURGE_GROUP, PrivacyPurgeConsumer
from proactive_agent.queue import RedisWakeQueue
from proactive_agent.runtime import GuildRuntimeRegistry

SECOND_GUILD = "555555555555555555"


def leaks(text: str) -> bool:
    return TARGET in text or bool(name_hits(text, [TARGET_NAME]))


def _texts(messages: list[ModelMessage]) -> list[str]:
    """Every line of text the model was shown, except the purge prompt."""
    lines: list[str] = []
    for message in messages[:-1]:
        for part in message.parts:
            content = getattr(part, "content", None)
            if isinstance(content, str):
                lines.extend(content.splitlines())
    return lines


def honest_model(calls: list[str]) -> FunctionModel:
    """A fake agent model that really leaves the target out.

    Memory notes keep every line it was shown that does not mention the
    target (so nia's content survives); watch decisions drop entries about
    the target and keep the rest.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = "\n".join(_texts([messages[-1], messages[-1]]))
        if info.output_tools:
            calls.append("watch")
            decisions = []
            for line in prompt.splitlines():
                if line.startswith("- w") and ": " in line:
                    key, text = line[2:].split(": ", 1)
                    action = "drop" if leaks(text) else "keep"
                    decisions.append({"instruction_id": key, "action": action})
            return ModelResponse(
                parts=[
                    ToolCallPart(info.output_tools[0].name, {"decisions": decisions})
                ]
            )
        calls.append("note")
        kept = [
            line
            for line in dict.fromkeys(_texts(messages))
            if line.strip()
            and not leaks(line)
            and not line.startswith("[COMPACTION MEMORY NOTE")
            and not line.startswith("Understood")
        ]
        return ModelResponse(parts=[TextPart("\n".join(kept) or "quiet guild")])

    return FunctionModel(respond)


def leaky_model(calls: list[str]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append("note")
        return ModelResponse(parts=[TextPart(f"remember <@{TARGET}> likes cats")])

    return FunctionModel(respond)


def command(*guilds: str) -> dict:
    return {
        "schema_version": 1,
        "request_id": str(uuid4()),
        "run_id": str(uuid4()),
        "user_id": TARGET,
        "names": [TARGET_NAME, "Kai the Rustacean"],
        "guild_ids": list(guilds or (GUILD,)),
        "created_at": "2026-10-03T12:00:00Z",
    }


class Replica:
    """One worker process: its own writer, queue, runtimes and consumer."""

    def __init__(self, redis_client, api, model, *, name: str, debounce=0.05):
        self.repository = GuildHistoryRepository(redis_client, api)
        self.writer = DebouncedHistoryWriter(
            self.repository, api, debounce_seconds=debounce, retry_base_seconds=0.001
        )
        self.queue = RedisWakeQueue(redis_client, consumer_name=name)
        self.runtime = make_runtime(redis_client, api, self.repository, self.writer)

        async def factory(_guild_id):
            return self.runtime

        self.runtimes = GuildRuntimeRegistry(factory)
        self.consumer = PrivacyPurgeConsumer(
            redis_client,
            api,
            self.queue,
            self.repository,
            self.writer,
            self.runtimes,
            model=model,
            consumer_name=name,
            fence_wait_seconds=2,
            fence_poll_seconds=0.01,
        )

    async def wake(self):
        runtime = await self.runtimes.get(GUILD)
        await runtime.process(batch())
        return runtime


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


async def submit(redis_client, payload) -> str:
    stream_id = await redis_client.xadd(
        PRIVACY_PURGE_STREAM_KEY,
        {"payload": payload if isinstance(payload, str) else json.dumps(payload)},
    )
    return stream_id.decode()


async def pending_count(redis_client) -> int:
    summary = await redis_client.xpending(PRIVACY_PURGE_STREAM_KEY, PURGE_GROUP)
    return summary["pending"]


async def stored_v1(redis_client) -> str:
    return (await redis_client.get(history_key(GUILD))).decode()


async def test_purge_leaves_no_trace_of_the_target_and_keeps_the_bystander(
    redis_client, world, caplog
):
    caplog.set_level(logging.DEBUG)
    api = world
    calls: list[str] = []
    replica = Replica(redis_client, api, honest_model(calls), name="a")
    runtime = await replica.wake()  # in-RAM history now holds kai's words
    assert leaks(runtime.engine.seen_histories[0])
    attempts_before = len(api.put_attempts)
    await replica.consumer.initialize()
    await submit(redis_client, command())

    assert await replica.consumer.poll_once() == 1

    v1 = await stored_v1(redis_client)
    assert not leaks(v1)
    assert "ship Rust 1.95" in v1 and BYSTANDER in v1
    durable = api.durable[GUILD]
    assert not leaks(json.dumps(durable.history))
    assert durable.revision == json.loads(v1)["revision"]
    assert len(json.loads(v1)["history"]) == 2
    # The wake's dirty copy (with kai) was discarded, never flushed.
    await asyncio.sleep(0.15)
    assert all(
        not leaks(json.dumps(snapshot.history))
        for snapshot in api.put_attempts[attempts_before:]
    )
    assert attempts_before == 0
    # Watch instructions: kai's dropped, nia's kept, via the existing API.
    addendum = api.addenda[GUILD][CHANNEL]
    assert not leaks(addendum) and "nia's Rust 1.95" in addendum
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    # The local runtime forgot its copy; the next wake starts clean.
    assert runtime.engine.agent_runner.history == []
    await replica.wake()
    assert not leaks(runtime.engine.seen_histories[-1])
    assert "ship Rust 1.95" in runtime.engine.seen_histories[-1]
    # Acked, XACKed, and nothing identifying was logged or acked.
    assert [(ack["guild_id"], ack["outcome"]) for ack in api.acks] == [
        (GUILD, "purged")
    ]
    assert set(api.acks[0]["stores"]) == {
        "proactive:v1:history",
        "proactive_agent_histories",
        "proactive:guild-history",
        "watch_instructions",
    }
    assert not leaks(api.acks[0]["detail"])
    assert await pending_count(redis_client) == 0
    assert not await redis_client.exists(privacy_lock_key(GUILD))
    assert not any(leaks(record.getMessage()) for record in caplog.records)
    assert calls == ["note", "watch"]
    await replica.writer.close(timeout=1)


async def test_legacy_history_cannot_come_back_after_a_purge(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())
    await replica.consumer.poll_once()

    # The worker owned the legacy key and deleted it once v1 and Postgres
    # held the purged history.
    assert not await redis_client.exists(legacy_history_key(GUILD))
    # Even if a raw legacy copy reappears and both newer stores are lost,
    # it is never restored once the guild has a purge epoch.
    await redis_client.set(
        legacy_history_key(GUILD), json.dumps(dump(raw_history_with_target()))
    )
    await redis_client.delete(history_key(GUILD))
    api.durable.pop(GUILD)
    loaded = await replica.repository.load(GUILD)

    assert loaded.history == []
    await replica.writer.close(timeout=1)


async def test_a_purge_of_another_user_skips_an_already_folded_history(
    redis_client, world
):
    # Rule (b): after a purge the history is just the agent's memory note,
    # so a later purge for someone the note never mentions folds nothing.
    api = world
    calls: list[str] = []
    replica = Replica(redis_client, api, honest_model(calls), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())
    await replica.consumer.poll_once()
    first = await stored_v1(redis_client)

    other = command()
    other["user_id"] = "777777777777777777"
    other["names"] = ["zed"]
    await submit(redis_client, other)
    await replica.consumer.poll_once()

    assert await stored_v1(redis_client) == first
    assert calls == ["note", "watch", "watch"]  # no second fold
    assert [ack["outcome"] for ack in api.acks] == ["purged", "unchanged"]
    assert api.acks[1]["detail"] == "history already attributed and clean"
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    await replica.writer.close(timeout=1)


async def test_a_model_that_keeps_the_id_fails_and_changes_nothing(
    redis_client, world, caplog
):
    caplog.set_level(logging.DEBUG)
    api = world
    calls: list[str] = []
    replica = Replica(redis_client, api, leaky_model(calls), name="a", debounce=0.05)
    await replica.wake()  # leaves a dirty copy at revision 4
    v1_before = await stored_v1(redis_client)
    durable_before = api.durable[GUILD]
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert calls == ["note"] * 3
    assert await stored_v1(redis_client) == v1_before
    assert api.durable[GUILD] == durable_before
    assert not await redis_client.exists(purge_epoch_key(GUILD))
    assert api.addenda[GUILD][CHANNEL] == watch_addendum_texts_unchanged(api)
    [ack] = api.acks
    assert ack["outcome"] == "failed" and ack["stores"] == []
    assert not leaks(ack["detail"])
    assert await pending_count(redis_client) == 0
    assert not any(leaks(record.getMessage()) for record in caplog.records)
    # The discarded dirty copy was put back and still reaches Postgres.
    await asyncio.sleep(0.15)
    assert api.durable[GUILD].revision == 4
    await replica.writer.close(timeout=1)


def watch_addendum_texts_unchanged(api) -> str:
    stored = api.addenda[GUILD][CHANNEL]
    assert TARGET_NAME in stored  # untouched: kai's watch entry still there
    return stored


async def test_failing_summarizer_acks_failed_and_continues_with_next_guild(
    redis_client, world
):
    api = world
    raised = []

    async def broken(messages, **_kwargs):
        raised.append(True)
        raise RuntimeError(f"provider echoed {TARGET}")

    async def no_watch(entries, **_kwargs):
        return entries, 0

    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._compact = broken
    replica.consumer._decide_watch = no_watch
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(SECOND_GUILD, [], revision=0)
    )
    v1_before = await stored_v1(redis_client)
    await replica.consumer.initialize()
    await submit(redis_client, command(GUILD, SECOND_GUILD))

    await replica.consumer.poll_once()

    assert await stored_v1(redis_client) == v1_before
    assert [(ack["guild_id"], ack["outcome"]) for ack in api.acks] == [
        (GUILD, "failed"),
        (SECOND_GUILD, "unchanged"),
    ]
    assert api.acks[0]["detail"] == "purge: RuntimeError"
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_a_stale_runtime_on_another_replica_reloads_after_the_purge(
    redis_client, world
):
    api = world
    replica_a = Replica(redis_client, api, honest_model([]), name="a")
    replica_b = Replica(redis_client, api, honest_model([]), name="b", debounce=0.05)
    runtime_b = await replica_b.wake()  # B holds kai in RAM and a dirty copy
    assert leaks(runtime_b.engine.seen_histories[0])
    await replica_a.consumer.initialize()
    await submit(redis_client, command())

    await replica_a.consumer.poll_once()
    await asyncio.sleep(0.15)  # B's stale dirty flush is rejected, not retried
    await replica_b.wake()

    assert not leaks(runtime_b.engine.seen_histories[-1])
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert not leaks(await stored_v1(redis_client))
    await replica_a.writer.close(timeout=1)
    await replica_b.writer.close(timeout=1)


async def test_bot_owned_guild_is_fenced_by_the_privacy_lock(redis_client, world):
    api = world
    await redis_client.set(ownership_key(GUILD), "bot")
    seen_lock = []
    replica = Replica(redis_client, api, honest_model([]), name="a")
    real_compact = replica.consumer._compact

    async def compact(messages, **kwargs):
        seen_lock.append(await redis_client.exists(privacy_lock_key(GUILD)))
        await redis_client.set(ownership_key(GUILD), "external")
        # Even with ownership flipped, no wake may start mid-purge.
        seen_lock.append(await replica.queue.acquire_lease(GUILD))
        return await real_compact(messages, **kwargs)

    replica.consumer._compact = compact
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert seen_lock == [1, None]
    assert not await redis_client.exists(privacy_lock_key(GUILD))
    assert not leaks(await stored_v1(redis_client))
    # Watch instructions belong to the bot when it owns the guild.
    assert api.acks[0]["stores"] == [
        "proactive_agent_histories",
        "proactive:v1:history",
    ]
    await replica.writer.close(timeout=1)


async def test_busy_guild_lease_times_out_as_failed(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._fence_wait_seconds = 0.05
    held = await replica.queue.acquire_lease(GUILD)
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "failed"
    assert api.acks[0]["detail"] == "fence: guild could not be fenced in time"
    await held.release()
    await replica.writer.close(timeout=1)


async def test_malformed_command_is_dropped_without_logging_it(
    redis_client, world, caplog
):
    caplog.set_level(logging.DEBUG)
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    bad = command()
    bad["unexpected"] = "field"
    await submit(redis_client, bad)
    await submit(redis_client, "not json " + TARGET)

    assert await replica.consumer.poll_once() == 1
    assert await replica.consumer.poll_once() == 1

    assert api.acks == []
    assert await pending_count(redis_client) == 0
    # Nobody else will XDEL a malformed entry: the consumer does.
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0
    assert not any(leaks(record.getMessage()) for record in caplog.records)
    await replica.writer.close(timeout=1)


async def test_unknown_run_is_dropped_after_the_first_404(redis_client, world):
    api = world
    api.ack_status = 404
    calls: list[str] = []
    replica = Replica(redis_client, api, honest_model(calls), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command(GUILD, SECOND_GUILD))

    await replica.consumer.poll_once()

    assert await pending_count(redis_client) == 0
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0  # XDELed
    assert calls == ["note", "watch"]  # the second guild was never started
    await replica.writer.close(timeout=1)


async def test_failed_ack_leaves_the_entry_pending_for_reclaim(redis_client, world):
    api = world
    api.ack_status = 500
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()
    assert await pending_count(redis_client) == 1

    # Reclaimed and redone (idempotently) once the API answers again.
    api.ack_status = 200
    assert await replica.consumer.poll_once() == 1
    assert await pending_count(redis_client) == 0
    assert [ack["outcome"] for ack in api.acks] == ["purged"]
    await replica.writer.close(timeout=1)


async def test_group_reads_entries_written_before_it_existed(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await submit(redis_client, command())
    await replica.consumer.initialize()
    await replica.consumer.initialize()  # BUSYGROUP tolerated

    assert await replica.consumer.poll_once() == 1
    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)


async def test_run_loop_survives_errors_and_stops(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    stop = asyncio.Event()
    await submit(redis_client, command())
    task = asyncio.create_task(replica.consumer.run(stop))
    for _ in range(100):
        if api.acks:
            break
        await asyncio.sleep(0.01)
    stop.set()
    await asyncio.wait_for(task, timeout=10)

    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)


async def test_a_22_digit_guild_id_from_the_contract_is_purged(redis_client, world):
    api = world
    long_guild = "1" * 22
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command(long_guild))

    await replica.consumer.poll_once()

    assert [(ack["guild_id"], ack["outcome"]) for ack in api.acks] == [
        (long_guild, "unchanged")
    ]
    # Nothing was written, so no epoch bump.
    assert not await redis_client.exists(purge_epoch_key(long_guild))
    await replica.writer.close(timeout=1)


async def test_v1_write_failing_after_the_put_falls_back_to_postgres(
    redis_client, world
):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")

    async def broken_cache(_snapshot):
        raise ConnectionError("redis write lost")

    replica.repository.write_purged = broken_cache
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert not await redis_client.exists(history_key(GUILD))
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    assert api.acks[0]["outcome"] == "purged"
    # The next load reads the purged Postgres copy, never the legacy key.
    loaded = await GuildHistoryRepository(redis_client, api).load(GUILD)
    assert loaded == api.durable[GUILD]
    await replica.writer.close(timeout=1)


async def test_v1_write_and_delete_both_failing_acks_failed(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")

    async def broken(*_args):
        raise ConnectionError("redis down")

    real_write, real_forget = (
        replica.repository.write_purged,
        replica.repository.forget,
    )
    replica.repository.write_purged = broken
    replica.repository.forget = broken
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    # Postgres was purged, v1 could be neither written nor deleted: v1 is
    # tombstoned so loads skip it, and the ack names the exact state.
    assert api.acks[0]["outcome"] == "failed"
    assert api.acks[0]["detail"] == ("purge: postgres:purged v1:unpurged v1:tombstoned")
    assert api.acks[0]["stores"] == ["proactive_agent_histories"]
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert leaks(await stored_v1(redis_client))  # still there, but unusable
    assert await redis_client.exists(history_invalid_key(GUILD))
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    # Forced retry: the entry stays pending and no done record was kept.
    assert await pending_count(redis_client) == 1
    loaded = await GuildHistoryRepository(redis_client, api).load(GUILD)
    assert loaded == api.durable[GUILD]
    # A load never clears the tombstone; only the purge's own write does.
    assert leaks(await stored_v1(redis_client))
    assert await redis_client.exists(history_invalid_key(GUILD))

    replica.repository.write_purged = real_write
    replica.repository.forget = real_forget
    await replica.consumer.poll_once()  # the reclaimed retry

    assert api.acks[-1]["outcome"] == "purged"
    assert not leaks(await stored_v1(redis_client))
    assert not await redis_client.exists(history_invalid_key(GUILD))
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_purge_put_conflict_is_retried_once_with_a_fresh_revision(
    redis_client, world
):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    real_get = api.get_history
    reads = []

    async def get_history(guild_id):
        snapshot = await real_get(guild_id)
        reads.append(snapshot.revision)
        if len(reads) == 1:
            # A stale flush lands between the purge's read and its PUT.
            api.durable[guild_id] = build_snapshot(
                guild_id, dump(raw_history_with_target()), revision=4
            )
        return snapshot

    api.get_history = get_history
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert reads == [2, 4]
    assert api.durable[GUILD].revision == 5
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert json.loads(await stored_v1(redis_client))["revision"] == 5
    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)


async def test_a_working_consumers_entry_is_not_reclaimed(redis_client, world):
    # A purge running longer than the reclaim idle time keeps its entry
    # fresh, so a second replica does not start a duplicate purge.
    api = world
    replica_a = Replica(redis_client, api, honest_model([]), name="a")
    replica_b = Replica(redis_client, api, honest_model([]), name="b")
    replica_a.consumer._heartbeat_seconds = 0.01
    replica_b.consumer._reclaim_idle_ms = 50
    real_compact = replica_a.consumer._compact
    release = asyncio.Event()
    started = asyncio.Event()

    async def slow_compact(messages, **kwargs):
        started.set()
        await release.wait()
        return await real_compact(messages, **kwargs)

    replica_a.consumer._compact = slow_compact
    await replica_a.consumer.initialize()
    await submit(redis_client, command())
    working = asyncio.create_task(replica_a.consumer.poll_once())
    await started.wait()

    for _ in range(10):
        await asyncio.sleep(0.03)  # 300 ms in total, 6x the idle limit
        assert await replica_b.consumer.poll_once() == 0

    release.set()
    assert await working == 1
    assert [ack["outcome"] for ack in api.acks] == ["purged"]
    await replica_a.writer.close(timeout=1)
    await replica_b.writer.close(timeout=1)


async def test_a_purge_waits_for_a_running_wake_then_purges_its_save(
    redis_client, world
):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    runtime = await replica.runtimes.get(GUILD)
    engine = runtime.engine
    in_wake = asyncio.Event()
    finish_wake = asyncio.Event()
    real_wake = engine.wake

    async def slow_wake(**kwargs):
        in_wake.set()
        await finish_wake.wait()
        return await real_wake(**kwargs)

    engine.wake = slow_wake
    lease = await replica.queue.acquire_lease(GUILD)

    async def wake_under_lease():
        async with lease:
            await runtime.process(batch())
            return runtime.history_revision

    wake_task = asyncio.create_task(wake_under_lease())
    await in_wake.wait()
    await replica.consumer.initialize()
    await submit(redis_client, command())
    purge_task = asyncio.create_task(replica.consumer.poll_once())
    await asyncio.sleep(0.1)

    assert api.acks == []  # the purge is waiting for the lease
    assert not await redis_client.exists(purge_epoch_key(GUILD))
    finish_wake.set()
    wake_revision = await wake_task
    assert await purge_task == 1

    v1 = json.loads(await stored_v1(redis_client))
    assert v1["revision"] > wake_revision
    assert not leaks(json.dumps(v1))
    assert "ship Rust 1.95" in json.dumps(v1)
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert api.acks[0]["outcome"] == "purged"
    await asyncio.sleep(0.15)  # the wake's dirty copy never reaches Postgres
    assert not leaks(json.dumps(api.durable[GUILD].history))
    await replica.writer.close(timeout=1)
