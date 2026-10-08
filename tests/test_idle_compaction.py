"""Idle compaction (#89): no verbatim message outlives 2 idle hours."""

from __future__ import annotations

import asyncio
import json
import time

import fakeredis.aioredis
import pytest
from purge_fakes import (
    CHANNEL,
    GUILD,
    FakeAPI,
    batch,
    dump,
    make_runtime,
    raw_history_with_target,
)
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage

from proactive_agent.agent import (
    KimiAgentRunner,
    is_summary_only,
    memory_note_pair,
)
from proactive_agent.contracts import HistorySnapshot
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.idle import IDLE_WINDOW, IdleHistoryCompactor
from proactive_agent.keys import (
    HISTORY_IDLE_INDEX_KEY,
    history_key,
    history_meta_key,
    ownership_key,
)
from proactive_agent.queue import RedisWakeQueue

IDLE = IDLE_WINDOW.total_seconds()
VERBATIM = "my cat Miso is sick"
SPENT = {"input_tokens": 900, "output_tokens": 120, "cache_read_tokens": 30}


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


class Clock:
    """Wall clock shifted forward by ``offset`` seconds."""

    def __init__(self):
        self.offset = 0.0

    def __call__(self) -> float:
        return time.time() + self.offset


class Setup:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.api = FakeAPI()
        self.repository = GuildHistoryRepository(redis_client, self.api)
        self.writer = DebouncedHistoryWriter(
            self.repository, self.api, debounce_seconds=0
        )
        self.queue = RedisWakeQueue(redis_client, consumer_name="test")
        self.clock = Clock()
        self.summaries: list[list] = []

    async def summarize(self, messages) -> tuple[str, dict]:
        self.summaries.append(list(messages))
        return "kai and nia talked about pets and Rust in #general", dict(SPENT)

    def compactor(self, **kwargs) -> IdleHistoryCompactor:
        return IdleHistoryCompactor(
            self.redis,
            self.queue,
            self.repository,
            self.writer,
            summarize=kwargs.pop("summarize", self.summarize),
            api=self.api,
            model_id="agent-model",
            clock=self.clock,
            **kwargs,
        )

    async def save(self, history, *, revision=0, fresh=False):
        return await self.writer.save(
            guild_id=GUILD,
            history=dump(history),
            previous_revision=revision,
            freshly_compacted=fresh,
        )

    async def stored(self):
        """(v1 history, Postgres history) after the debounced flush."""
        await self.writer.flush(GUILD)
        v1 = json.loads(await self.redis.get(history_key(GUILD)))["history"]
        durable = self.api.durable[GUILD].history
        return v1, durable


@pytest.fixture
async def setup(redis_client):
    world = Setup(redis_client)
    await redis_client.set(ownership_key(GUILD), "external")
    yield world
    await world.writer.close(timeout=1)


def compacted_with_tail() -> list:
    """What a wake's 100k compaction stores: note pair + the kept tail."""
    return [
        *memory_note_pair("earlier: kai asked about cats"),
        *raw_history_with_target(),
    ]


def as_text(history: list[dict]) -> str:
    return json.dumps(history)


async def test_idle_history_is_folded_to_summary_only_at_two_hours(setup):
    await setup.save(raw_history_with_target())
    setup.clock.offset = IDLE + 1

    outcomes = await setup.compactor().sweep_once()

    assert outcomes == {GUILD: "folded"}
    assert len(setup.summaries) == 1
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        history = ModelMessagesTypeAdapter.validate_json(json.dumps(copy))
        assert is_summary_only(history)
        assert VERBATIM not in as_text(copy)
    # Summary only: out of the index until the next real write.
    assert await setup.redis.zscore(HISTORY_IDLE_INDEX_KEY, GUILD) is None


async def test_history_inside_the_window_is_untouched(setup):
    # Negative control for the fold: one second short of the window.
    await setup.save(raw_history_with_target())
    before = await setup.redis.get(history_key(GUILD))
    setup.clock.offset = IDLE - 1

    assert await setup.compactor().sweep_once() == {}
    assert setup.summaries == []
    assert await setup.redis.get(history_key(GUILD)) == before


async def test_a_write_restarts_the_idle_clock(setup):
    snapshot = await setup.save(raw_history_with_target())
    setup.clock.offset = IDLE + 1
    # Written again just now (in clock terms, after the history went idle):
    # the index still lists the old time but the meta hash is the authority.
    await setup.redis.zadd(HISTORY_IDLE_INDEX_KEY, {GUILD: time.time() - IDLE - 5})
    await setup.redis.hset(
        f"proactive:v1:{{guild:{GUILD}}}:history-meta",
        "written_at",
        repr(setup.clock() - 10),
    )
    before = await setup.redis.get(history_key(GUILD))

    assert await setup.compactor().compact_guild(GUILD) == "written since"
    assert setup.summaries == []
    assert await setup.redis.get(history_key(GUILD)) == before
    assert snapshot.revision == 1


async def test_freshly_compacted_history_drops_its_tail_without_a_model_call(setup):
    await setup.save(compacted_with_tail(), fresh=True)
    setup.clock.offset = IDLE + 1

    outcomes = await setup.compactor().sweep_once()

    assert outcomes == {GUILD: "tail dropped"}
    assert setup.summaries == []
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        assert is_summary_only(ModelMessagesTypeAdapter.validate_json(json.dumps(copy)))
        assert "earlier: kai asked about cats" in as_text(copy)
        assert VERBATIM not in as_text(copy)


async def test_unflagged_compacted_history_is_folded_by_the_model(setup):
    # Negative control for the flag: the same history without it costs a
    # model call, and the call sees the whole history, tail included.
    await setup.save(compacted_with_tail(), fresh=False)
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "folded"}
    assert len(setup.summaries) == 1
    assert VERBATIM in ModelMessagesTypeAdapter.dump_json(setup.summaries[0]).decode()


async def test_any_later_write_clears_the_fresh_flag(setup):
    snapshot = await setup.save(compacted_with_tail(), fresh=True)
    assert (await setup.repository.idle_state(GUILD))[1] is True
    await setup.save(
        [*compacted_with_tail(), ModelRequest(parts=[UserPromptPart("wake")])],
        revision=snapshot.revision,
    )
    assert (await setup.repository.idle_state(GUILD))[1] is False

    setup.clock.offset = IDLE + 1
    assert await setup.compactor().sweep_once() == {GUILD: "folded"}


async def test_the_clock_survives_a_restart_and_adopts_unindexed_histories(setup):
    # A v1 history from before the index existed: no meta, not indexed.
    await setup.repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=4)
    )
    await setup.redis.zrem(HISTORY_IDLE_INDEX_KEY, GUILD)
    stop = asyncio.Event()
    first = setup.compactor(tick_seconds=0.01)
    task = asyncio.create_task(first.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await task
    assert await setup.redis.zscore(HISTORY_IDLE_INDEX_KEY, GUILD) is not None
    assert setup.summaries == []

    # A new process (nothing carried in RAM) two hours later.
    setup.clock.offset = IDLE + 1
    assert await setup.compactor().sweep_once() == {GUILD: "folded"}


async def test_busy_lease_defers_to_the_next_tick(setup):
    await setup.save(raw_history_with_target())
    lease = await setup.queue.acquire_lease(GUILD)
    assert lease is not None
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "fence busy"}
    assert setup.summaries == []
    await lease.release()
    assert await setup.compactor().sweep_once() == {GUILD: "folded"}


async def test_model_failure_still_leaves_no_verbatim_text(setup):
    await setup.save(compacted_with_tail())
    setup.clock.offset = IDLE + 1

    async def failing(_messages):
        raise RuntimeError("provider down")

    assert await setup.compactor(summarize=failing).sweep_once() == {GUILD: "folded"}
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        assert is_summary_only(ModelMessagesTypeAdapter.validate_json(json.dumps(copy)))
        assert VERBATIM not in as_text(copy)


async def test_a_worker_holding_the_old_history_reloads_the_fold(setup):
    # The runtime loaded the verbatim history before the sweep folded it;
    # its next wake must start from the fold and never write the old copy
    # back to either store.
    setup.api.addenda[GUILD] = {CHANNEL: ""}
    await setup.save(raw_history_with_target())
    runtime = make_runtime(setup.redis, setup.api, setup.repository, setup.writer)
    await runtime.process(batch())
    assert VERBATIM in runtime.engine.seen_histories[-1]

    setup.clock.offset = IDLE + 1
    assert await setup.compactor().sweep_once() == {GUILD: "folded"}
    await runtime.process(batch())

    assert VERBATIM not in runtime.engine.seen_histories[-1]
    v1, durable = await setup.stored()
    assert VERBATIM not in as_text(v1)
    assert VERBATIM not in as_text(durable)


class _FakeAgent:
    def __init__(self):
        self.started_from = None

    async def run(self, brief, *, deps, message_history):
        self.started_from = list(message_history or [])
        messages = [
            *self.started_from,
            ModelRequest(parts=[UserPromptPart(brief)]),
            ModelResponse(parts=[TextPart("done")]),
        ]

        class Result:
            output = "done"

            def all_messages(self):
                return messages

            usage = RunUsage(input_tokens=40, output_tokens=4, requests=1)

        return Result()


async def test_wake_stores_its_compaction_before_the_model_runs():
    stored = []

    async def summarize(_messages):
        return "note", dict(SPENT)

    async def on_compacted(history):
        stored.append(list(history))

    runner = KimiAgentRunner(
        agent=_FakeAgent(),
        summarize=summarize,
        token_limit=10,
        history=raw_history_with_target() * 3,
        on_compacted=on_compacted,
    )
    await runner.wake("brief", deps=None)

    assert len(stored) == 1
    assert stored[0] == runner.agent.started_from
    assert stored[0][0].parts[0].content.startswith("[COMPACTION MEMORY NOTE")


async def test_wake_without_compaction_stores_nothing_early():
    # Negative control: under the token limit no early write happens.
    stored = []

    async def on_compacted(history):
        stored.append(history)

    runner = KimiAgentRunner(
        agent=_FakeAgent(),
        summarize=lambda _messages: None,
        history=raw_history_with_target(),
        on_compacted=on_compacted,
    )
    await runner.wake("brief", deps=None)
    assert stored == []


async def test_a_wake_that_fails_after_compacting_leaves_it_flagged(setup):
    # The real path to a fresh flag at the idle point: the runtime stores the
    # wake's compaction (flagged) and the wake then dies before its own save.
    setup.api.addenda[GUILD] = {CHANNEL: ""}
    snapshot = await setup.save(raw_history_with_target() * 3)
    runtime = make_runtime(setup.redis, setup.api, setup.repository, setup.writer)

    async def wake_that_compacts_then_fails(**_kwargs):
        await runtime.engine.agent_runner.on_compacted(compacted_with_tail())
        raise RuntimeError("model died mid-wake")

    runtime.engine.wake = wake_that_compacts_then_fails
    with pytest.raises(RuntimeError):
        await runtime.process(batch())

    assert runtime.history_revision == snapshot.revision + 1
    assert (await setup.repository.idle_state(GUILD))[1] is True
    setup.clock.offset = IDLE + 1
    assert await setup.compactor().sweep_once() == {GUILD: "tail dropped"}
    assert setup.summaries == []


async def test_a_purge_rewrite_keeps_the_idle_clock(setup):
    # Last written a minute short of the window, then purged: the rest must
    # still fold on time, not 2 hours after the purge.
    snapshot = await setup.save(raw_history_with_target())
    written_at = time.time() - IDLE + 60
    await setup.redis.hset(history_meta_key(GUILD), "written_at", repr(written_at))
    await setup.redis.zadd(HISTORY_IDLE_INDEX_KEY, {GUILD: written_at})

    purged = build_snapshot(
        GUILD,
        dump(memory_note_pair("nia asked about Rust")),
        revision=snapshot.revision + 10,
    )
    await setup.repository.write_purged(purged)

    assert (await setup.repository.idle_state(GUILD))[0] == written_at
    assert await setup.redis.zscore(HISTORY_IDLE_INDEX_KEY, GUILD) == written_at
    setup.clock.offset = 61
    assert await setup.compactor().sweep_once() == {GUILD: "already summary only"}


async def test_an_unreadable_durable_copy_is_replaced_at_the_idle_point(setup):
    # Postgres holds bytes that fail their checksum, and v1 has nothing.
    setup.api.durable[GUILD] = HistorySnapshot(
        guild_id=GUILD,
        revision=3,
        checksum="0" * 64,
        history=dump(raw_history_with_target()),
    )
    await setup.redis.zadd(HISTORY_IDLE_INDEX_KEY, {GUILD: time.time()})
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "unreadable replaced"}
    assert setup.summaries == []
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        assert is_summary_only(ModelMessagesTypeAdapter.validate_json(json.dumps(copy)))
        assert VERBATIM not in as_text(copy)
    assert setup.api.durable[GUILD].revision == 4
    assert await setup.redis.zscore(HISTORY_IDLE_INDEX_KEY, GUILD) is None


async def test_a_history_that_does_not_parse_is_replaced_at_the_idle_point(setup):
    # A valid snapshot whose messages are not model messages.
    await setup.save([{"kind": "unknown", "text": VERBATIM}])
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "unreadable replaced"}
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        assert VERBATIM not in as_text(copy)


async def test_an_unreadable_history_inside_the_window_is_kept(setup):
    # Negative control for the replace: only the idle point removes it.
    await setup.save([{"kind": "unknown", "text": VERBATIM}])
    before = await setup.redis.get(history_key(GUILD))
    setup.clock.offset = IDLE - 60

    assert await setup.compactor().sweep_once() == {}
    assert await setup.redis.get(history_key(GUILD)) == before


async def test_wake_bills_its_compaction_with_its_own_turn():
    runner = KimiAgentRunner(
        agent=_FakeAgent(),
        summarize=lambda _messages: asyncio.sleep(0, ("note", dict(SPENT))),
        token_limit=10,
        history=raw_history_with_target() * 3,
    )
    _note, usage = await runner.wake("brief", deps=None)

    assert usage == {"input_tokens": 940, "output_tokens": 124, "cache_read_tokens": 30}


async def test_wake_without_compaction_bills_only_its_turn():
    runner = KimiAgentRunner(
        agent=_FakeAgent(), summarize=None, history=raw_history_with_target()
    )
    _note, usage = await runner.wake("brief", deps=None)

    assert usage == {"input_tokens": 40, "output_tokens": 4, "cache_read_tokens": 0}


async def test_an_idle_fold_bills_the_agent_model(setup):
    await setup.save(raw_history_with_target())
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "folded"}

    [report] = setup.api.usage_reports
    assert report["guild_id"] == GUILD
    assert report["wake_id"].startswith("idle-")
    assert len(report["wake_id"]) <= 64
    assert report["passive"] is True and report["responses"] == 0
    assert report["entries"] == [
        {"model_id": "agent-model", "operation": "agent", **SPENT}
    ]


async def test_dropping_a_fresh_tail_bills_nothing(setup):
    # Negative control: no model call, no usage row.
    await setup.save(compacted_with_tail(), fresh=True)
    setup.clock.offset = IDLE + 1

    assert await setup.compactor().sweep_once() == {GUILD: "tail dropped"}
    assert setup.api.usage_reports == []


async def test_a_failed_usage_report_still_writes_the_fold(setup):
    await setup.save(raw_history_with_target())
    setup.clock.offset = IDLE + 1
    setup.api.fail_usage = True

    assert await setup.compactor().sweep_once() == {GUILD: "folded"}
    v1, durable = await setup.stored()
    for copy in (v1, durable):
        assert is_summary_only(ModelMessagesTypeAdapter.validate_json(json.dumps(copy)))
