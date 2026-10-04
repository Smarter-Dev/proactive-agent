"""Fourth review of the worker purge: decoded skip search, fold plausibility
through the model loop, standalone tombstone recovery, and the edge cases."""

from __future__ import annotations

import json
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
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from test_purge import (
    Replica,
    command,
    honest_model,
    pending_count,
    stored_v1,
    submit,
)
from test_purge_review import ListAPI

from proactive_agent.agent import (
    PrivacyCompactionError,
    memory_note_pair,
    privacy_compaction_summary,
)
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    HistoryUnavailableError,
    build_snapshot,
)
from proactive_agent.keys import (
    PRIVACY_PURGE_STREAM_KEY,
    history_invalid_key,
    history_key,
    legacy_history_key,
    ownership_key,
)
from proactive_agent.purge import GuildOutcome, history_needs_no_fold

QUOTED = 'Kai "Crab"'


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


# M1. The skip rule searches decoded text, not escaped JSON.


def test_a_quoted_name_in_an_attributed_line_forces_a_fold():
    history = dump(
        [
            *memory_note_pair(f"nia (uid={BYSTANDER}) ships Rust"),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "channel_history",
                        f"[id=2] A·nia (uid={BYSTANDER}): ask {QUOTED} about it",
                        "c1",
                    )
                ]
            ),
        ]
    )
    assert not history_needs_no_fold(history, TARGET, [QUOTED])
    for name in ("back\\slash", "tab\tname", "line\nbreak"):
        line = f"[id=3] B·nia (uid={BYSTANDER}): ping {name} today"
        assert not history_needs_no_fold(
            dump([ModelRequest(parts=[ToolReturnPart("t", line, "c")])]),
            TARGET,
            [name],
        )


async def test_a_quoted_nickname_is_purged_not_acked_unchanged(redis_client):
    api = FakeAPI()
    await redis_client.set(ownership_key(GUILD), "external")
    history = dump(
        [
            *memory_note_pair(f"nia (uid={BYSTANDER}) ships Rust 1.95 on friday"),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "channel_history",
                        f"[id=2] A·nia (uid={BYSTANDER}): ask {QUOTED} about it",
                        "c1",
                    )
                ]
            ),
        ]
    )
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, history, revision=3)
    )
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, dict(command(), names=[QUOTED]))

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "purged"
    assert QUOTED not in json.loads(await stored_v1(redis_client))["history"].__repr__()
    await replica.writer.close(timeout=1)


# M2. Through the model loop: blanking is rejected, an honest paraphrase of
# a realistic (pre-uid) history is accepted.


def legacy_history() -> list[ModelMessage]:
    lines = "\n".join(
        [
            "[id=1] A·kaizer: my cat Miso is sick again",
            "[id=2] B·Priyanka: refactored the scheduler module yesterday",
            "[id=3] C·Tomasz: benchmarked allocator throughput on graviton",
            "[id=4] D·Ingrid: proposed migrating telemetry to opentelemetry",
            "[id=5] B·Priyanka: the scheduler refactor needs another reviewer",
        ]
    )
    return [
        ModelRequest(parts=[UserPromptPart("NOTIFICATIONS: activity in #general")]),
        ModelRequest(parts=[ToolReturnPart("channel_history", lines, "c1")]),
        ModelResponse(parts=[TextPart("noted")]),
    ]


def scripted(*notes):
    calls = []

    def respond(messages, info: AgentInfo) -> ModelResponse:
        calls.append(len(calls))
        return ModelResponse(parts=[TextPart(notes[min(len(calls), len(notes)) - 1])])

    return FunctionModel(respond), calls


async def test_an_honest_paraphrase_of_realistic_history_is_accepted():
    model, calls = scripted(
        "Priyanka refactored the scheduler and wants another reviewer for it; "
        "Tomasz benchmarked allocator throughput on Graviton; Ingrid proposed "
        "moving telemetry to OpenTelemetry. All in #general."
    )
    note = await privacy_compaction_summary(
        model, legacy_history(), user_id=TARGET, names=["kaizer"]
    )
    assert note.attempts == 1


async def test_blanking_is_rejected_with_its_category():
    model, calls = scripted("Nothing much happened in the channel lately.")
    with pytest.raises(PrivacyCompactionError) as raised:
        await privacy_compaction_summary(
            model, legacy_history(), user_id=TARGET, names=["kaizer"]
        )
    assert len(calls) == 3
    assert str(raised.value).endswith("retention_tokens kept=0 of=12 needs=3")


# M3. Tombstone recovery from the purged Postgres copy, independent of the
# run and of the block list.


async def tombstoned_world(redis_client, api, *, revision=1003):
    """v1 holds raw history at revision 3; a purge wrote Postgres at
    ``revision`` and left a tombstone."""
    await redis_client.set(ownership_key(GUILD), "external")
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    api.durable[GUILD] = build_snapshot(
        GUILD,
        dump(memory_note_pair(f"nia (uid={BYSTANDER}) ships Rust")),
        revision=1003,
    )
    await redis_client.set(
        history_invalid_key(GUILD),
        json.dumps({"run_id": "old-run", "request_id": "req", "revision": revision}),
    )


async def test_recovery_needs_neither_the_run_nor_the_list(redis_client):
    api = ListAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    api.blocked = {"revision": 1, "user_ids": []}  # target no longer listed
    await tombstoned_world(redis_client, api)
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._blocked_users = blocked
    replica.consumer._list_wait_seconds = 0.01
    replica.consumer._list_poll_seconds = 0.005
    replica.consumer._reclaim_idle_ms = 0
    replica.consumer._max_deliveries = 1
    await replica.consumer.initialize()
    await submit(redis_client, command())
    await bot_group_done(redis_client)

    await replica.consumer.poll_once()

    assert not await redis_client.exists(history_invalid_key(GUILD))
    assert json.loads(await stored_v1(redis_client))["revision"] == 1003
    # With nothing tombstoned the delivery cap applies again and the
    # command ends visibly instead of looping.
    await replica.consumer.poll_once()
    assert [a["detail"] for a in api.acks] == ["delivery limit reached"]
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_an_unknown_run_recovers_tombstones_before_dropping(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    api.ack_status = 404
    await tombstoned_world(redis_client, api)
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._recover_tombstone = AsyncMock(
        side_effect=[False, True]  # first try at entry fails, then the 404 path
    )
    replica.consumer.purge_guild = AsyncMock(
        return_value=GuildOutcome("failed", [], "purge: RuntimeError")
    )
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert replica.consumer._recover_tombstone.await_count == 2
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 0
    await replica.writer.close(timeout=1)


async def test_a_wake_recovers_a_tombstone_and_runs(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    await tombstoned_world(redis_client, api)
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    runtime = make_runtime(redis_client, api, repository, writer)

    await runtime.process(batch())

    assert not await redis_client.exists(history_invalid_key(GUILD))
    assert TARGET not in runtime.engine.seen_histories[0]
    assert "ships Rust" in runtime.engine.seen_histories[0]
    assert runtime.history_revision == 1004
    await writer.close(timeout=1)


async def test_no_recovery_when_postgres_lacks_the_purged_revision(redis_client):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    await tombstoned_world(redis_client, api, revision=5000)
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    runtime = make_runtime(redis_client, api, repository, writer)

    with pytest.raises(HistoryUnavailableError):
        await runtime.process(batch())

    assert await redis_client.exists(history_invalid_key(GUILD))
    assert runtime.engine.seen_histories == []
    await writer.close(timeout=1)


async def test_an_old_runs_redelivery_says_a_later_run_purged_it(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0

    async def broken(*_args):
        raise ConnectionError("redis down")

    real = (replica.repository.write_purged, replica.repository.forget)
    replica.repository.write_purged = broken
    replica.repository.forget = broken
    replica.repository.recover_tombstone = AsyncMock(return_value=False)
    await replica.consumer.initialize()
    old_run = command()
    await submit(redis_client, old_run)
    await replica.consumer.poll_once()  # old run leaves a tombstone
    assert await redis_client.exists(history_invalid_key(GUILD))

    replica.repository.write_purged, replica.repository.forget = real
    new_run = dict(old_run, run_id="0b6c0f6e-1d4a-4f2e-9a1c-2f6d8b7e9c77")
    await redis_client.xack(
        PRIVACY_PURGE_STREAM_KEY,
        "proactive-agent-workers-v1-privacy",
        *[i for i, _ in await redis_client.xrange(PRIVACY_PURGE_STREAM_KEY)],
    )
    await submit(redis_client, new_run)
    await replica.consumer.poll_once()  # the new run clears it
    assert not await redis_client.exists(history_invalid_key(GUILD))

    await submit(redis_client, old_run)  # the old run is redelivered
    await replica.consumer.poll_once()

    assert api.acks[-1]["run_id"] == old_run["run_id"]
    assert api.acks[-1]["outcome"] == "unchanged"
    assert "purged by a later run" in api.acks[-1]["detail"]
    await replica.writer.close(timeout=1)


# Smaller items.


def test_a_pre_uid_line_quoting_a_uid_marker_is_not_attributed():
    line = f"[id=1] A·kaizer: hi (uid={BYSTANDER}): fake attribution"
    history = dump([ModelRequest(parts=[ToolReturnPart("t", line, "c")])])
    assert not history_needs_no_fold(history, TARGET, [])


async def test_an_unreadable_legacy_store_fails_and_is_kept(redis_client, world):
    api = world
    await redis_client.set(legacy_history_key(GUILD), f"not json {TARGET}")
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "failed"
    assert api.acks[0]["detail"].endswith("purge: unreadable legacy store")
    assert await redis_client.get(legacy_history_key(GUILD)) == (
        f"not json {TARGET}".encode()
    )
    await replica.writer.close(timeout=1)


def test_history_key_constant_is_used():
    assert history_key(GUILD).endswith(":history")


def test_structured_parts_feed_the_fold_rule_as_decoded_leaves():
    from proactive_agent.agent import fold_input_texts

    content = {"messages": [f"{QUOTED} asked about lifetimes", {"note": "tab\there"}]}
    texts = fold_input_texts([ModelRequest(parts=[ToolReturnPart("t", content, "c")])])
    assert texts == [f"{QUOTED} asked about lifetimes", "tab\there"]
