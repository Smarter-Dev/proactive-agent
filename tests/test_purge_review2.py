"""Second review of the worker purge: tombstone, enforcing reports, fold
rule, delivery handling, legacy deletion and the consumer's alive marker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from purge_fakes import (
    BYSTANDER,
    CHANNEL,
    GUILD,
    TARGET,
    FakeAPI,
    batch,
    bot_group_done,
    dump,
    make_runtime,
    raw_history_with_target,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from test_purge import (
    Replica,
    command,
    honest_model,
    leaks,
    pending_count,
    stored_v1,
    submit,
)
from test_purge_review import ListAPI, counting_note_model, pre_uid_history

from proactive_agent.agent import memory_note_pair
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    HistoryUnavailableError,
    StaleHistoryError,
    build_snapshot,
)
from proactive_agent.keys import (
    PRIVACY_PURGE_STREAM_KEY,
    history_invalid_key,
    history_key,
    legacy_history_key,
    privacy_enforcing_key,
)
from proactive_agent.purge import history_needs_no_fold
from proactive_agent.worker import ProactiveWorker


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


# A. Tombstone: only the purge's own write clears it; nothing else writes
#    v1 while it exists, and no wake runs.


async def test_a_stale_writer_cannot_erase_the_tombstone(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    raw = build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    await repository.cache(raw)
    # A purge rewrote Postgres at revision 4, could not touch v1, and its
    # epoch bump failed too: only the tombstone guards the raw v1.
    api.durable[GUILD] = build_snapshot(GUILD, [], revision=4)
    await redis_client.set(history_invalid_key(GUILD), "1")

    # A writer still holding revision 3 in RAM saves revision 4, 5, ...
    for revision in (3, 4, 10):
        with pytest.raises(StaleHistoryError):
            await writer.save(
                guild_id=GUILD, history=[{"raw": TARGET}], previous_revision=revision
            )
    assert await redis_client.exists(history_invalid_key(GUILD))
    assert await stored_v1(redis_client) == raw.model_dump_json()
    # Loads skip v1 and do not cache over it either.
    assert await repository.load(GUILD) == api.durable[GUILD]
    assert await redis_client.exists(history_invalid_key(GUILD))
    await writer.close(timeout=1)


async def test_a_runtime_refuses_a_wake_while_tombstoned(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    await repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    runtime = make_runtime(redis_client, api, repository, writer)
    await runtime.process(batch())  # loads kai's raw history into RAM
    await redis_client.set(history_invalid_key(GUILD), "1")

    with pytest.raises(HistoryUnavailableError):
        await runtime.process(batch())

    assert len(runtime.engine.seen_histories) == 1  # the model never ran
    assert runtime.engine.agent_runner.history == []
    assert runtime.history_loaded is False
    await writer.close(timeout=1)


# B. Enforcing only once reported; re-checked right before the lease.


async def test_a_failed_report_means_not_enforcing(redis_client):
    api = ListAPI()
    api.blocked = {"revision": 4, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client, replica_id="r1")

    async def broken_eval(*_args, **_kwargs):
        raise ConnectionError("redis write lost")

    real_eval = redis_client.eval
    redis_client.eval = broken_eval
    assert not await blocked.refresh()
    assert not blocked.enforcing
    redis_client.eval = real_eval
    assert await blocked.refresh()
    assert blocked.enforcing


async def test_enforcing_is_rechecked_after_the_semaphore_wait(redis_client):
    blocked = SimpleNamespace(enforcing=True)
    queue = SimpleNamespace(
        externally_owned=AsyncMock(return_value=True),
        acquire_lease=AsyncMock(return_value=None),
    )
    worker = ProactiveWorker(
        queue, SimpleNamespace(), concurrency=1, blocked_users=blocked
    )
    worker._enforcing_poll_seconds = 0.005
    await worker._semaphore.acquire()  # another wake holds the only slot
    task = asyncio.create_task(
        worker._run_guild(GUILD, (SimpleNamespace(stream_id="1-0", guild_id=GUILD),))
    )
    await asyncio.sleep(0.02)
    blocked.enforcing = False  # goes stale while queued on the semaphore
    worker._semaphore.release()
    await asyncio.sleep(0.05)

    queue.acquire_lease.assert_not_awaited()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_a_new_replica_is_in_the_aggregate_from_its_first_report(
    redis_client,
):
    old = ListAPI()
    old.blocked = {"revision": 9, "user_ids": [TARGET]}
    new = ListAPI()
    new.blocked = {"revision": 7, "user_ids": []}
    running = BlockedUsers(old, redis_client, replica_id="old")
    joining = BlockedUsers(new, redis_client, replica_id="new")
    await running.refresh()

    # Interleave: the running replica re-reports while the new one reports
    # for the first time. Whatever the order, the aggregate is the min.
    await asyncio.gather(running.refresh(), joining.refresh(), running.refresh())

    assert await redis_client.get(privacy_enforcing_key("worker")) == b"7"
    assert await redis_client.get(privacy_enforcing_key("worker") + ":new") == b"7"
    assert await redis_client.get(privacy_enforcing_key("worker") + ":old") == b"9"


# D. Fold rule and done records.


def attributed_wake(extra_line: str = "") -> list[dict]:
    brief = (
        "NOTIFICATIONS since your last wake (oldest first):\n"
        "[#general] [11:00 UTC, mention] You were @mentioned by nia "
        f"(username nia, id {BYSTANDER}) in message id=2:\n> ship Rust 1.95"
    )
    tool = f"[id=2] A·nia (uid={BYSTANDER}): ship Rust 1.95" + extra_line
    return dump(
        [
            *memory_note_pair("nia ships Rust"),
            ModelRequest(parts=[UserPromptPart(brief)]),
            ModelRequest(parts=[ToolReturnPart("channel_history", tool, "c1")]),
            ModelResponse(parts=[TextPart("noted")]),
        ]
    )


def test_host_written_lines_do_not_force_a_fold():
    # Brief and notification lines carry no uid=; only member lines must.
    assert history_needs_no_fold(attributed_wake(), TARGET, ["kai"])


def test_a_member_line_without_uid_forces_a_fold():
    assert not history_needs_no_fold(
        attributed_wake("\n[id=3] B·kaizer: my cat"), TARGET, ["kai"]
    )


def test_host_written_text_is_searched_for_id_and_names():
    history = attributed_wake()
    history[2]["parts"][0]["content"] += f"\n(user {TARGET} said hi)"
    assert not history_needs_no_fold(history, TARGET, [])
    assert not history_needs_no_fold(attributed_wake(), TARGET, ["nia"])


async def test_a_new_run_refolds_when_something_matches_again(redis_client):
    api = FakeAPI()
    await redis_client.set(f"proactive:v1:{{guild:{GUILD}}}:owner", "external")
    repository = GuildHistoryRepository(redis_client, api)
    await repository.cache(build_snapshot(GUILD, pre_uid_history(), revision=3))
    calls: list[str] = []
    replica = Replica(redis_client, api, counting_note_model(calls), name="a")
    await replica.consumer.initialize()
    first = command()
    await submit(redis_client, first)
    await replica.consumer.poll_once()
    assert calls == ["note"]

    # Something about the user reappears (say, a wake before the block).
    current = await repository.load(GUILD)
    await repository.cache(
        build_snapshot(
            GUILD,
            [*current.history, *dump(raw_history_with_target())],
            revision=current.revision + 1,
        )
    )
    await submit(
        redis_client, dict(first, run_id="0b6c0f6e-1d4a-4f2e-9a1c-2f6d8b7e9c99")
    )
    await replica.consumer.poll_once()

    assert calls == ["note", "note"]  # the new run folded again
    assert not leaks(await stored_v1(redis_client))
    assert [a["outcome"] for a in api.acks] == ["purged", "purged"]
    await replica.writer.close(timeout=1)


# E/F. Drops XDEL first; a lost done record is reported.


async def test_drop_never_acks_an_entry_it_could_not_delete(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    await submit(redis_client, "not json " + TARGET)
    await bot_group_done(redis_client)
    real_xdel = redis_client.xdel

    async def broken_xdel(*_args):
        raise ConnectionError("redis down")

    redis_client.xdel = broken_xdel
    with pytest.raises(ConnectionError):
        await replica.consumer.poll_once()
    assert await pending_count(redis_client) == 1  # not acked, retried later

    redis_client.xdel = real_xdel
    assert await replica.consumer.poll_once() == 1
    assert await pending_count(redis_client) == 0
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0
    await replica.writer.close(timeout=1)


async def test_a_lost_done_record_is_reported_in_the_ack(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")

    async def broken_hset(*_args):
        raise ConnectionError("redis down")

    redis_client.hset = broken_hset
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "purged"
    assert api.acks[0]["detail"].startswith("done_record=unsaved; ")
    await replica.writer.close(timeout=1)


# H. One entry at a time.


async def test_poll_claims_one_entry_at_a_time(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    # Two entries delivered to this consumer earlier and left pending.
    await submit(redis_client, command())
    await submit(redis_client, command())
    await redis_client.xreadgroup(
        "proactive-agent-workers-v1-privacy", "a", {PRIVACY_PURGE_STREAM_KEY: ">"}
    )
    handled = []
    real_handle = replica.consumer.handle

    async def handle(stream_id, fields):
        handled.append(stream_id)
        return await real_handle(stream_id, fields)

    replica.consumer.handle = handle

    assert await replica.consumer.poll_once() == 1
    assert await replica.consumer.poll_once() == 1
    assert len(set(handled)) == 2
    await replica.writer.close(timeout=1)


# I. The legacy key goes only once v1 and Postgres both hold the result.


async def test_legacy_key_goes_when_v1_was_deleted_instead_of_written(
    redis_client, world
):
    # M3: v1 could not be written but was deleted; Postgres holds the purged
    # copy, so the raw legacy key must go too.
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")

    async def broken(_snapshot):
        raise ConnectionError("redis write lost")

    replica.repository.write_purged = broken
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "purged"  # v1 deleted, Postgres purged
    assert not await redis_client.exists(history_key(GUILD))
    assert not await redis_client.exists(legacy_history_key(GUILD))
    assert "proactive:guild-history" in api.acks[0]["stores"]
    await replica.writer.close(timeout=1)


async def test_a_failed_legacy_delete_fails_and_retries(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0

    async def broken(_guild_id):
        raise ConnectionError("redis write lost")

    real_forget_legacy = replica.repository.forget_legacy
    replica.repository.forget_legacy = broken
    await replica.consumer.initialize()
    run = command()
    await submit(redis_client, run)

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "failed"
    assert "legacy history delete failed" in api.acks[0]["detail"]
    assert await redis_client.exists(legacy_history_key(GUILD))
    assert await pending_count(redis_client) == 1  # kept for a retry
    done_key = f"privacy:v1:purge-done:worker:{run['run_id']}"
    assert not await redis_client.exists(done_key)

    replica.repository.forget_legacy = real_forget_legacy
    await replica.consumer.poll_once()
    assert api.acks[-1]["outcome"] == "purged"
    assert not await redis_client.exists(legacy_history_key(GUILD))
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_legacy_key_is_deleted_after_both_writes(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert not await redis_client.exists(legacy_history_key(GUILD))
    assert "proactive:guild-history" in api.acks[0]["stores"]
    assert "legacy history deleted" in api.acks[0]["detail"]
    assert await redis_client.exists(history_key(GUILD))
    await replica.writer.close(timeout=1)


async def test_bot_owned_legacy_key_is_left_to_the_bot(redis_client, world):
    api = world
    await redis_client.set(f"proactive:v1:{{guild:{GUILD}}}:owner", "bot")
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert await redis_client.exists(legacy_history_key(GUILD))
    await replica.writer.close(timeout=1)


# K. The consumer's alive marker, and surviving Redis errors.


async def test_consumer_marks_itself_alive_and_survives_redis_errors(
    redis_client, world
):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._replica_id = "host-42"
    replica.consumer._error_backoff_seconds = 0.01
    real_poll = replica.consumer.poll_once
    failures = []

    async def flaky_poll(**kwargs):
        if len(failures) < 2:
            failures.append(True)
            raise ConnectionError("redis down")
        return await real_poll(**kwargs)

    replica.consumer.poll_once = flaky_poll
    stop = asyncio.Event()
    await submit(redis_client, command())
    task = asyncio.create_task(replica.consumer.run(stop))
    for _ in range(400):
        if api.acks:
            break
        await asyncio.sleep(0.05)
    key = "privacy:v1:consumer:worker:host-42"
    assert await redis_client.exists(key)
    assert 0 < await redis_client.ttl(key) <= 180
    stop.set()
    await asyncio.wait_for(task, timeout=10)

    assert len(failures) == 2
    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)
