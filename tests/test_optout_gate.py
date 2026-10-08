"""The opt-out gate at the writers (smarter-dev #100).

An opted-out member is on the blocked-users list; nothing the worker writes
(memory notes, saved history) or renders from memory may carry them.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import fakeredis.aioredis
import pytest
from purge_fakes import (
    BYSTANDER,
    BYSTANDER_NAME,
    CHANNEL,
    GUILD,
    TARGET,
    TARGET_NAME,
    FakeAPI,
    batch,
    dump,
    make_runtime,
    raw_history_with_target,
)
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart

from proactive_agent.agent import ToolBudget
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.contracts import BlockedUsersList
from proactive_agent.environment import WakeActions
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import history_key
from proactive_agent.optout_gate import (
    blank_blocked_lines,
    scrub_history,
    scrub_memory,
)
from proactive_agent.parity import REMEMBER_OPTED_OUT, ProactiveDeps, remember
from proactive_agent.runtime import render_memory_block
from proactive_agent.transcript import BLOCKED_LINE

BYSTANDER_LINE = f"[id=2] B·{BYSTANDER_NAME} (uid={BYSTANDER}): ship Rust 1.95"
TARGET_TEXT = "my cat Miso is sick"


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


class GateAPI(FakeAPI):
    """FakeAPI with a typed blocked list, memory notes and a memory bundle."""

    def __init__(self):
        super().__init__()
        self.notes: list[dict] = []
        self.note_result = {"saved": True}
        self.memory = None

    async def get_blocked_users(self):
        return BlockedUsersList.model_validate(await super().get_blocked_users())

    async def save_memory_note(self, guild_id, payload):
        self.notes.append(payload)
        return self.note_result

    async def get_memory(self, guild_id):
        return self.memory


async def blocked_list(redis_client, api, user_ids) -> BlockedUsers:
    api.blocked = {"revision": 1, "user_ids": list(user_ids)}
    blocked = BlockedUsers(api, redis_client)
    assert await blocked.refresh()
    return blocked


async def no_skim(_transcript: str) -> str:
    return ""


def deps_kwargs() -> dict:
    return {
        "actions": WakeActions(),
        "skim_transcript": no_skim,
        "budget": ToolBudget(),
    }


def remember_ctx(api, blocked) -> SimpleNamespace:
    return SimpleNamespace(
        deps=ProactiveDeps(
            guild_id=GUILD,
            discord=None,
            api=api,
            channel_id=CHANNEL,
            blocked_users=blocked,
            **deps_kwargs(),
        )
    )


# -- remember ----------------------------------------------------------------


@pytest.mark.parametrize(
    "note",
    [
        f"<@{TARGET}> has a sick cat",
        f"<@!{TARGET}> has a sick cat",
        f"user {TARGET} has a sick cat",
    ],
)
async def test_remember_refuses_a_note_naming_a_blocked_user(
    redis_client, monkeypatch, note
):
    monkeypatch.delenv("CHAT_MEMORY_ENABLED", raising=False)
    api = GateAPI()
    blocked = await blocked_list(redis_client, api, [TARGET])

    answer = await remember(remember_ctx(api, blocked), note)

    assert answer == REMEMBER_OPTED_OUT
    assert api.notes == []


async def test_remember_posts_a_note_naming_an_unblocked_user(
    redis_client, monkeypatch
):
    monkeypatch.delenv("CHAT_MEMORY_ENABLED", raising=False)
    api = GateAPI()
    blocked = await blocked_list(redis_client, api, [TARGET])
    note = f"<@{BYSTANDER}> ships Rust 1.95 ({BYSTANDER})"

    answer = await remember(remember_ctx(api, blocked), note)

    assert answer != REMEMBER_OPTED_OUT
    assert [posted["content"] for posted in api.notes] == [note]


async def test_server_opted_out_reason_maps_to_the_sentence(redis_client, monkeypatch):
    monkeypatch.delenv("CHAT_MEMORY_ENABLED", raising=False)
    api = GateAPI()
    api.note_result = {"saved": False, "reason": "opted_out"}
    blocked = await blocked_list(redis_client, api, [])

    answer = await remember(remember_ctx(api, blocked), "kai likes cats")

    assert answer == REMEMBER_OPTED_OUT
    assert len(api.notes) == 1


async def test_runtime_deps_carry_the_blocked_list(redis_client, monkeypatch):
    """The wake's tools get the runtime's list, so remember is gated live."""
    monkeypatch.delenv("CHAT_MEMORY_ENABLED", raising=False)
    api = GateAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    blocked = await blocked_list(redis_client, api, [TARGET])
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, blocked_users=blocked)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.blocked_users = blocked
    runtime.image_capabilities = SimpleNamespace(review=None, generate=None)
    await runtime.process(batch())

    deps = runtime.engine.deps_factory(channel_id=CHANNEL, **deps_kwargs())
    answer = await remember(SimpleNamespace(deps=deps), f"<@{TARGET}> is sad")

    assert answer == REMEMBER_OPTED_OUT
    assert api.notes == []
    await writer.close()


# -- history -----------------------------------------------------------------


def stored_leaves(history: list[dict]) -> str:
    return json.dumps(history, ensure_ascii=False)


def tool_return_content(history: list[dict]) -> str:
    return next(
        part["content"]
        for message in history
        for part in message.get("parts", ())
        if part.get("part_kind") == "tool-return"
    )


async def wake_and_store(redis_client, user_ids) -> tuple[list[dict], list[dict]]:
    """One wake on a history where kai and nia spoke; (v1, Postgres) after."""
    api = GateAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    blocked = await blocked_list(redis_client, api, user_ids)
    repository = GuildHistoryRepository(redis_client, api)
    await repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    writer = DebouncedHistoryWriter(
        repository, api, debounce_seconds=0, blocked_users=blocked
    )
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.blocked_users = blocked
    await runtime.process(batch())
    await writer.flush(GUILD)
    v1 = (await repository.load(GUILD)).history
    await writer.close()
    return v1, api.durable[GUILD].history


async def test_saved_history_drops_a_blocked_authors_line(redis_client):
    v1, durable = await wake_and_store(redis_client, [TARGET])

    for stored in (v1, durable):
        text = stored_leaves(stored)
        assert TARGET not in text
        assert TARGET_TEXT not in text
        lines = tool_return_content(stored).split("\n")
        assert lines == [BLOCKED_LINE, BYSTANDER_LINE]


async def test_saved_history_keeps_an_unblocked_authors_line(redis_client):
    v1, durable = await wake_and_store(redis_client, [BYSTANDER])

    for stored in (v1, durable):
        lines = tool_return_content(stored).split("\n")
        assert lines[0] == f"[id=1] A·{TARGET_NAME} (uid={TARGET}): {TARGET_TEXT}"
        assert lines[1] == BLOCKED_LINE
        assert TARGET in stored_leaves(stored)


async def test_history_scrub_drops_continuations_and_keeps_the_input(redis_client):
    api = GateAPI()
    blocked = await blocked_list(redis_client, api, [TARGET])
    content = (
        f"[id=1] A·{TARGET_NAME} (uid={TARGET}): first line\nsecond line\n"
        f"{BYSTANDER_LINE} cc <@{TARGET}>"
    )
    history = dump(
        [ModelRequest(parts=[ToolReturnPart("channel_history", content, "c1")])]
    )
    original = json.dumps(history)

    scrubbed = scrub_history(history, blocked)

    assert tool_return_content(scrubbed).split("\n") == [
        BLOCKED_LINE,
        f"{BYSTANDER_LINE} cc @[blocked user]",
    ]
    assert json.dumps(history) == original
    assert blank_blocked_lines(content, None) == content


async def model_input_over_two_wakes(redis_client, blocked_ids_later):
    """Two wakes on one stored history, nobody blocked on the first; the
    list changes between them. Returns the history the model saw on each."""
    api = GateAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    blocked = await blocked_list(redis_client, api, [])
    repository = GuildHistoryRepository(redis_client, api)
    await repository.cache(
        build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    )
    writer = DebouncedHistoryWriter(repository, api, blocked_users=blocked)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.blocked_users = blocked
    await runtime.process(batch())
    # The member opts out while the runner holds the history in memory.
    api.blocked = {"revision": 2, "user_ids": list(blocked_ids_later)}
    assert await blocked.refresh()
    await runtime.process(batch())
    await writer.close()
    first, second = runtime.engine.seen_histories
    return json.loads(first), json.loads(second)


async def test_history_loaded_before_opt_out_is_blanked_for_the_model(redis_client):
    """Stored before the opt-out: blanked on load as well as in memory."""
    api = GateAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    blocked = await blocked_list(redis_client, api, [TARGET])
    repository = GuildHistoryRepository(redis_client, api)
    stored = build_snapshot(GUILD, dump(raw_history_with_target()), revision=3)
    await repository.cache(stored)
    stored_raw = await redis_client.get(history_key(GUILD))
    # No writer scrub: only the read path may blank what the model sees.
    writer = DebouncedHistoryWriter(repository, api)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.blocked_users = blocked
    real_wake = runtime.engine.wake
    stored_during_wake = []

    async def wake(**kwargs):
        stored_during_wake.append(await redis_client.get(history_key(GUILD)))
        return await real_wake(**kwargs)

    runtime.engine.wake = wake

    await runtime.process(batch())

    seen = json.loads(runtime.engine.seen_histories[0])
    assert tool_return_content(seen).split("\n") == [BLOCKED_LINE, BYSTANDER_LINE]
    assert TARGET not in stored_leaves(seen)
    # The read path never rewrote the stored bytes.
    assert stored_during_wake == [stored_raw]
    await writer.close()


async def test_runner_history_kept_across_wakes_loses_a_new_opt_out(redis_client):
    first, second = await model_input_over_two_wakes(redis_client, [TARGET])

    assert tool_return_content(first).split("\n")[0].endswith(TARGET_TEXT)
    assert tool_return_content(second).split("\n") == [
        BLOCKED_LINE,
        BYSTANDER_LINE,
    ]
    assert TARGET not in stored_leaves(second)


async def test_runner_history_keeps_an_unblocked_authors_line(redis_client):
    first, second = await model_input_over_two_wakes(redis_client, [BYSTANDER])

    target_line = f"[id=1] A·{TARGET_NAME} (uid={TARGET}): {TARGET_TEXT}"
    assert tool_return_content(second).split("\n") == [target_line, BLOCKED_LINE]
    assert tool_return_content(first).split("\n") == [target_line, BYSTANDER_LINE]


# -- memory block ------------------------------------------------------------


BLOB_KEPT = f"<@{BYSTANDER}> ships Rust ({BYSTANDER})\nthe server likes cats\n"
BEHAVIOR_KEPT = "be gentle\n"
NOTE_KEPT = {"channel_name": "general", "content": f"{BYSTANDER} shipped 1.95"}
NOTE_KEPT_PLAIN = {"channel_name": "general", "content": "release went out"}


def memory_bundle() -> dict:
    """Each field mixes a line about kai (TARGET) with lines about nia."""
    return {
        "memory_enabled": True,
        "content": f"<@{TARGET}> has a cat\n{BLOB_KEPT}",
        "behavior": f"{BEHAVIOR_KEPT}go easy on <@!{TARGET}>\n",
        "personality": f"warm toward {TARGET}",
        "notes": [
            {"channel_name": "general", "content": f"{TARGET} asked about vets"},
            NOTE_KEPT,
            {"channel_name": f"dm-{TARGET}", "content": "asked about vets"},
            NOTE_KEPT_PLAIN,
        ],
    }


def note_lines(*notes) -> str:
    return "\n".join(f"- [{note['channel_name']}] {note['content']}" for note in notes)


async def test_memory_block_drops_blocked_lines_and_notes(redis_client):
    api = GateAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    api.memory = memory_bundle()
    blocked = await blocked_list(redis_client, api, [TARGET])
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, blocked_users=blocked)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.blocked_users = blocked

    await runtime.process(batch())

    preamble = runtime.engine.seen_preambles[-1]
    assert TARGET not in preamble
    assert "has a cat" not in preamble and "vets" not in preamble
    assert f"GUILD MEMORY:\n{BLOB_KEPT}" in preamble
    assert f"NOTES YOU KEPT TODAY:\n{note_lines(NOTE_KEPT, NOTE_KEPT_PLAIN)}" in (
        preamble
    )
    # The copy is for the prompt: the fetched bundle is untouched.
    assert api.memory == memory_bundle()
    await writer.close()


async def test_memory_scrub_keeps_other_lines_and_empties_to_none(redis_client):
    api = GateAPI()
    blocked = await blocked_list(redis_client, api, [TARGET])
    bundle = {**memory_bundle(), "content": None}

    scrubbed = scrub_memory(bundle, blocked)

    assert scrubbed["content"] is None
    assert scrubbed["behavior"] == BEHAVIOR_KEPT
    assert scrubbed["personality"] is None
    assert scrubbed["notes"] == [NOTE_KEPT, NOTE_KEPT_PLAIN]
    assert scrub_memory(bundle, None) is bundle


async def test_memory_block_keeps_an_unblocked_id(redis_client):
    api = GateAPI()
    blocked = await blocked_list(redis_client, api, [])

    block = render_memory_block(memory_bundle(), blocked)

    assert f"<@{TARGET}> has a cat\n{BLOB_KEPT}" in block
    assert note_lines(*memory_bundle()["notes"]) in block


async def test_idle_fold_never_shows_the_summariser_a_blocked_author(redis_client):
    from purge_fakes import raw_history_with_target

    from proactive_agent.idle import IDLE_WINDOW, IdleHistoryCompactor
    from proactive_agent.keys import ownership_key
    from proactive_agent.queue import RedisWakeQueue

    for user_ids, seen in (([], True), ([TARGET], False)):
        await redis_client.flushall()
        api = GateAPI()
        blocked = await blocked_list(redis_client, api, user_ids)
        repository = GuildHistoryRepository(redis_client, api)
        # Stored while nobody had opted out.
        writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0)
        await redis_client.set(ownership_key(GUILD), "external")
        await writer.save(
            guild_id=GUILD,
            history=json.loads(
                ModelMessagesTypeAdapter.dump_json(raw_history_with_target())
            ),
            previous_revision=0,
        )
        await writer.flush(GUILD)
        summaries: list = []

        async def summarize(messages, summaries=summaries) -> tuple[str, dict]:
            summaries.append(list(messages))
            return "they talked about pets", {}

        compactor = IdleHistoryCompactor(
            redis_client,
            RedisWakeQueue(redis_client, consumer_name="test"),
            repository,
            writer,
            summarize=summarize,
            api=api,
            model_id="agent-model",
            clock=lambda: time.time() + IDLE_WINDOW.total_seconds() + 1,
            blocked_users=blocked,
        )

        assert await compactor.sweep_once() == {GUILD: "folded"}
        assert (TARGET_TEXT in str(summaries[0])) is seen
        await writer.close(timeout=1)


async def test_idle_sweep_waits_for_the_blocked_list(redis_client):
    from proactive_agent.idle import IDLE_WINDOW, IdleHistoryCompactor
    from proactive_agent.queue import RedisWakeQueue

    api = GateAPI()
    api.blocked = {"revision": 1, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)  # not refreshed yet
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0)
    idle_guild_ids = repository.idle_guild_ids
    asked: list = []

    async def spy(**kwargs):
        asked.append(kwargs)
        return await idle_guild_ids(**kwargs)

    repository.idle_guild_ids = spy
    compactor = IdleHistoryCompactor(
        redis_client,
        RedisWakeQueue(redis_client, consumer_name="test"),
        repository,
        writer,
        summarize=no_skim,
        api=api,
        model_id="agent-model",
        clock=lambda: time.time() + IDLE_WINDOW.total_seconds() + 1,
        blocked_users=blocked,
    )

    assert await compactor.sweep_once() == {}
    assert asked == []
    assert await blocked.refresh()
    await compactor.sweep_once()
    assert len(asked) == 1
    await writer.close(timeout=1)
