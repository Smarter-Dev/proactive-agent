"""Third review of the worker purge: note plausibility, tombstone lifecycle,
legacy and dead-letter cleanup, consumer group recovery and details."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import fakeredis.aioredis
import pytest
from purge_fakes import (
    BYSTANDER,
    GUILD,
    TARGET,
    bot_group_done,
    dump,
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
    leaks,
    pending_count,
    stored_v1,
    submit,
)

import proactive_agent.agent as agent_module
from proactive_agent.agent import (
    PrivacyCompactionError,
    privacy_compaction_summary,
    privacy_watch_decisions,
)
from proactive_agent.keys import (
    DEAD_LETTER_STREAM_KEY,
    PRIVACY_PURGE_STREAM_KEY,
    history_invalid_key,
    history_key,
    purge_epoch_key,
)
from proactive_agent.purge import _join, history_needs_no_fold


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


def fixed_model(text: str, calls: list) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(text)
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(respond)


# M1. A note that is not a plausible memory fails the step, untouched.

LAZY = "ok."
REFUSAL = (
    "I'm sorry, but I can't help with rewriting memories to remove a person. "
    "Please contact an administrator for this kind of request."
)
NO_BYSTANDERS = (
    "The channel talked about software releases, cats and veterinary visits, "
    "and someone planned to ship a new version on Friday. Nothing else of "
    "note happened in #general during these wakes."
)


@pytest.mark.parametrize("note", [LAZY, REFUSAL, NO_BYSTANDERS])
async def test_an_implausible_note_fails_and_leaves_every_store(
    redis_client, world, note
):
    api = world
    calls: list = []
    replica = Replica(redis_client, api, fixed_model(note, calls), name="a")
    v1_before = await stored_v1(redis_client)
    durable_before = api.durable[GUILD]
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert len(calls) == 3  # asked twice more, then gave up
    assert await stored_v1(redis_client) == v1_before
    assert api.durable[GUILD] == durable_before
    assert not await redis_client.exists(purge_epoch_key(GUILD))
    assert api.acks[0]["outcome"] == "failed"
    assert api.acks[0]["detail"].startswith(
        "purge: purged note implausible after 3 attempts: "
    )
    category = api.acks[0]["detail"].split(": ")[2].split()[0]
    assert category in {"too_short", "refusal", "retention_members"}
    await replica.writer.close(timeout=1)


def member_history(*members: tuple[str, str]) -> list[ModelMessage]:
    lines = "\n".join(
        f"[id={index}] {chr(65 + index)}·{name} (uid={uid}): shipping release "
        f"number {index} of the compiler with notes"
        for index, (uid, name) in enumerate(members)
    )
    return [
        ModelRequest(parts=[UserPromptPart("NOTIFICATIONS: activity")]),
        ModelRequest(parts=[ToolReturnPart("channel_history", lines, "c1")]),
    ]


# The rule itself is covered by tests/test_fold_plausibility.py (shared
# vectors).


# Timeouts on the fold and watch calls.


async def test_a_fold_model_call_times_out(monkeypatch):
    async def slow(messages, info):
        await asyncio.sleep(1)
        return ModelResponse(parts=[TextPart("never")])

    with pytest.raises(PrivacyCompactionError) as raised:
        await privacy_compaction_summary(
            FunctionModel(slow),
            member_history((BYSTANDER, "nia")),
            user_id=TARGET,
            names=[],
            timeout_seconds=0.05,
        )
    assert "timed out" in str(raised.value)


async def test_a_watch_model_call_times_out(monkeypatch):
    monkeypatch.setattr(agent_module, "PRIVACY_MODEL_TIMEOUT_SECONDS", 0.05)

    async def slow(messages, info):
        await asyncio.sleep(1)
        return ModelResponse(parts=[TextPart("never")])

    with pytest.raises(PrivacyCompactionError):
        await privacy_watch_decisions(
            FunctionModel(slow), {"w1": "watch kai"}, user_id=TARGET, names=[]
        )


# M2. A tombstone always has a pending retry and names its run.


async def test_a_tombstone_names_its_run_and_is_never_given_up(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    replica.consumer._max_deliveries = 1

    async def broken(*_args):
        raise ConnectionError("redis down")

    replica.repository.write_purged = broken
    replica.repository.forget = broken
    await replica.consumer.initialize()
    run = command()
    await submit(redis_client, run)

    attempts = []
    for _ in range(6):
        before = len(api.acks)
        await replica.consumer.poll_once()
        attempts.append(len(api.acks) - before)

    raw = await redis_client.get(history_invalid_key(GUILD))
    value = json.loads(raw)
    assert value["run_id"] == run["run_id"]
    assert value["request_id"] == run["request_id"]
    assert value["revision"] == api.durable[GUILD].revision
    assert set(value) == {"run_id", "request_id", "revision"}
    assert TARGET.encode() not in raw
    assert await redis_client.ttl(history_invalid_key(GUILD)) == -1  # no TTL
    # Retried on every delivery past the cap (backoff capped at the 10 min
    # reclaim interval), never given up, never XACKed.
    assert attempts == [1, 1, 1, 1, 1, 1]
    assert await pending_count(redis_client) == 1
    assert all(a["outcome"] == "failed" for a in api.acks)
    assert all(a["detail"].startswith("tombstoned=1; ") for a in api.acks)
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1
    await replica.writer.close(timeout=1)


# M4. The consumer recreates a vanished group and reports alive only after
# a successful poll.


async def test_consumer_recreates_its_group_and_heartbeats_only_when_consuming(
    redis_client, world
):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    consumer = replica.consumer
    consumer._replica_id = "host-1"
    consumer._error_backoff_seconds = 0.01
    alive = "privacy:v1:consumer:worker:host-1"
    stop = asyncio.Event()
    task = asyncio.create_task(consumer.run(stop))
    for _ in range(100):
        if await redis_client.exists(alive):
            break
        await asyncio.sleep(0.01)
    assert await redis_client.exists(alive)

    real_create = redis_client.xgroup_create
    blocked = asyncio.Event()

    async def failing_create(*args, **kwargs):
        if not blocked.is_set():
            raise ConnectionError("still down")
        return await real_create(*args, **kwargs)

    redis_client.xgroup_create = failing_create
    await redis_client.delete(PRIVACY_PURGE_STREAM_KEY)
    await asyncio.sleep(0.05)
    await redis_client.delete(alive)  # as if it had expired
    await asyncio.sleep(0.1)
    assert not await redis_client.exists(alive)  # not consuming, not alive

    blocked.set()
    for _ in range(200):
        if await redis_client.exists(alive):
            break
        await asyncio.sleep(0.01)
    assert await redis_client.exists(alive)
    await submit(redis_client, command())
    for _ in range(300):
        if api.acks:
            break
        await asyncio.sleep(0.02)
    stop.set()
    await asyncio.wait_for(task, timeout=10)

    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)


# L1. Only groups that have really read the entry count.


async def test_an_absent_bot_group_keeps_the_entry(redis_client, world):
    replica = Replica(redis_client, world, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, "not json")

    await replica.consumer.poll_once()

    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_a_group_created_at_dollar_after_the_entry_is_not_done(
    redis_client, world
):
    replica = Replica(redis_client, world, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, "not json")
    await redis_client.xgroup_create(PRIVACY_PURGE_STREAM_KEY, "smarter-dev-bot", "$")

    await replica.consumer.poll_once()

    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1
    await replica.writer.close(timeout=1)


# L2. The guild's dead letters go with its queued notifications.


async def test_a_purge_discards_the_guilds_dead_letters_in_both_shapes(
    redis_client, world
):
    api = world
    other = "666666666666666666"
    verbatim = await redis_client.xadd(
        DEAD_LETTER_STREAM_KEY,
        {
            "guild_id": GUILD,
            "source_stream_id": "1-0",
            "payload": json.dumps({"body": f"kai {TARGET} said"}),
            "error": "boom",
            "attempts": "5",
        },
    )
    ids_only = await redis_client.xadd(
        DEAD_LETTER_STREAM_KEY, {"guild_id": GUILD, "wake_id": "w-1"}
    )
    kept = await redis_client.xadd(
        DEAD_LETTER_STREAM_KEY, {"guild_id": other, "wake_id": "w-2"}
    )
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    remaining = [i for i, _f in await redis_client.xrange(DEAD_LETTER_STREAM_KEY)]
    assert remaining == [kept]
    assert verbatim not in remaining and ids_only not in remaining
    assert "dead_letters_dropped=2" in api.acks[0]["detail"]
    assert "proactive:v1:dead-letter" in api.acks[0]["stores"]
    await replica.writer.close(timeout=1)


# D gaps.


def wake(tool_content) -> list[dict]:
    return dump(
        [
            ModelRequest(parts=[UserPromptPart("NOTIFICATIONS: none")]),
            ModelRequest(parts=[ToolReturnPart("channel_history", tool_content, "c")]),
        ]
    )


def test_uid_marker_counts_only_in_the_rendered_prefix():
    attributed = f"[id=2] A·nia (uid={BYSTANDER}): ship it"
    assert history_needs_no_fold(wake(attributed), TARGET, [])
    in_content = "[id=3] B·kaizer: look (uid=999999999999999999) here"
    assert not history_needs_no_fold(wake(in_content), TARGET, [])
    reply = f"[id=4] [BOT] A·bot (uid={BYSTANDER}) (reply to id=2): done"
    assert history_needs_no_fold(wake(reply), TARGET, [])


def test_member_lines_in_structured_tool_output_are_unattributed():
    structured = {"messages": [f"[id=2] A·nia (uid={BYSTANDER}): ship it"]}
    assert not history_needs_no_fold(wake(structured), TARGET, [])
    assert history_needs_no_fold(wake({"results": ["no member lines"]}), TARGET, [])


# Residual 409: Redis was down for the whole purge write.


async def test_postgres_wins_over_a_stale_v1_left_by_a_redis_outage(
    redis_client, world
):
    api = world
    replica_a = Replica(redis_client, api, honest_model([]), name="a")
    replica_b = Replica(redis_client, api, honest_model([]), name="b", debounce=0.02)
    runtime_b = await replica_b.wake()  # B holds kai's raw history in RAM
    await asyncio.sleep(0.08)  # and flushed it

    async def broken(*_args):
        raise ConnectionError("redis down")

    replica_a.repository.write_purged = broken
    replica_a.repository.forget = broken
    replica_a.repository.invalidate = AsyncMock(return_value=False)
    replica_a.consumer._bump_epoch = AsyncMock()  # the epoch bump fails too
    await replica_a.consumer.initialize()
    await submit(redis_client, command())
    await replica_a.consumer.poll_once()
    assert api.acks[0]["detail"].endswith("v1:not-tombstoned")
    purged = api.durable[GUILD]
    assert not leaks(json.dumps(purged.history))
    assert leaks(await stored_v1(redis_client))  # stale v1, no tombstone

    await replica_b.wake()  # saves on the stale copy; its flush gets 409
    await asyncio.sleep(0.08)
    assert api.durable[GUILD] == purged  # Postgres was never overwritten
    assert not await redis_client.exists(history_key(GUILD))  # v1 dropped

    await replica_b.wake()  # reloads from Postgres
    assert not leaks(runtime_b.engine.seen_histories[-1])
    assert not leaks(await stored_v1(redis_client))
    await replica_a.writer.close(timeout=1)
    await replica_b.writer.close(timeout=1)


# L4. A give-up with no done record and no ack stays pending.


async def test_give_up_keeps_the_entry_when_nothing_could_be_recorded(
    redis_client, world
):
    api = world
    api.ack_status = 500
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._max_deliveries = 0
    await replica.consumer.initialize()
    await submit(redis_client, command())
    await bot_group_done(redis_client)

    async def broken_hset(*_args):
        raise ConnectionError("redis down")

    redis_client.hset = broken_hset

    await replica.consumer.poll_once()

    assert await pending_count(redis_client) == 1
    assert await redis_client.xlen(PRIVACY_PURGE_STREAM_KEY) == 1
    await replica.writer.close(timeout=1)


# L5. Deliveries are counted per stream entry.


async def test_a_republished_entry_starts_its_own_delivery_count(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    replica.consumer._max_deliveries = 2
    real_compact = replica.consumer._compact
    failing = [True]

    async def compact(messages, **kwargs):
        if failing[0]:
            raise RuntimeError("model down")
        return await real_compact(messages, **kwargs)

    replica.consumer._compact = compact
    api.ack_status = 500  # failed outcomes, and no ack gets through
    await replica.consumer.initialize()
    run = command()
    await submit(redis_client, run)
    for _ in range(2):
        await replica.consumer.poll_once()  # deliveries 1 and 2
    assert await pending_count(redis_client) == 1

    failing[0] = False
    api.ack_status = 200
    await redis_client.delete(PRIVACY_PURGE_STREAM_KEY)
    await replica.consumer.initialize()
    await submit(redis_client, run)  # the web re-publishes the same run

    await replica.consumer.poll_once()

    # Delivery 1 of the new entry (a run-keyed count would say 3 > 2 and
    # give up): processed normally.
    assert api.acks[-1]["outcome"] == "purged"
    assert "delivery limit" not in api.acks[-1]["detail"]
    await replica.writer.close(timeout=1)


async def test_unchecked_names_are_reported_first(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, dict(command(), names=["k", "kai"]))

    await replica.consumer.poll_once()

    assert api.acks[0]["detail"].startswith("unchecked_names=1; ")
    await replica.writer.close(timeout=1)


# F. Critical detail segments survive truncation.


def test_critical_detail_segments_are_never_truncated():
    detail = _join(
        "x" * 600,
        "history folded attempts=1; history_name_hits=2",
        "done_record=unsaved",
        "tombstoned=1",
        "watch_name_hits=3",
    )
    assert len(detail) <= 500
    assert detail.startswith(
        "history_name_hits=2; done_record=unsaved; tombstoned=1; watch_name_hits=3"
    )
