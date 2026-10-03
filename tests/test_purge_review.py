"""Purge review fixes: legacy migration, queued notifications, list checks,
durable progress, name-hit reporting."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import fakeredis.aioredis
import pytest
from purge_fakes import (
    BYSTANDER,
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
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
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

from proactive_agent.agent import KimiAgentRunner, memory_note_pair
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.contracts import BlockedUsersList, NotificationEnvelope
from proactive_agent.discord import DiscordREST
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    build_snapshot,
)
from proactive_agent.keys import (
    READY_STREAM_KEY,
    history_key,
    legacy_history_key,
    pending_key,
    purge_epoch_key,
    wake_stream_key,
)
from proactive_agent.parity import build_proactive_agent
from proactive_agent.queue import RedisWakeQueue
from proactive_agent.runtime import GuildRuntime


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


class ListAPI(FakeAPI):
    async def get_blocked_users(self):
        return BlockedUsersList.model_validate(await super().get_blocked_users())


def counting_note_model(calls: list[str]) -> FunctionModel:
    """Returns a different (clean) note every call."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.output_tools:
            calls.append("watch")
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"decisions": []})]
            )
        calls.append("note")
        return ModelResponse(parts=[TextPart(f"note {len(calls)}: nia ships Rust")])

    return FunctionModel(respond)


# 1. Legacy-only guild is folded and migrated, not reset.


async def test_a_legacy_only_guild_is_folded_into_v1_and_postgres(redis_client, world):
    api = world
    await redis_client.delete(history_key(GUILD))
    api.durable.pop(GUILD)
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    v1 = await stored_v1(redis_client)
    assert not leaks(v1) and "ship Rust 1.95" in v1
    assert not leaks(json.dumps(api.durable[GUILD].history))
    assert api.acks[0]["outcome"] == "purged"
    assert api.acks[0]["detail"].startswith("legacy history migrated; history folded")
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    # The guild keeps its (folded) memory instead of starting empty.
    loaded = await GuildHistoryRepository(redis_client, api).load(GUILD)
    assert len(loaded.history) == 2
    await replica.writer.close(timeout=1)


# 2. Queued raw notifications about the user are dropped by the purge, and a
#    wake never puts a blocked user's id into the brief.


def envelope(body: str, *, wakes: bool = True) -> NotificationEnvelope:
    return NotificationEnvelope(
        schema_version=1,
        notification_id=uuid4(),
        guild_id=GUILD,
        channel_id=CHANNEL,
        channel_name="general",
        kind="mention",
        created_at=datetime.now(UTC),
        body=body,
        message_ids=("700000000000000001",),
        wakes=wakes,
        passive=False,
        watcher_usage={},
        trace_id=uuid4(),
    )


KAI_MENTION = (
    f"You were @mentioned by kai (username kai_rs, id {TARGET}) in message "
    "id=700000000000000001:\n> my cat Miso is sick"
)
KAI_SUMMARY = "Watcher summary: Kai asked about the vet (relevant message ids: 1)"
NIA_MENTION = (
    f"You were @mentioned by nia (username nia, id {BYSTANDER}) in message "
    "id=700000000000000002:\n> release notes are up"
)


async def publish(redis_client, item: NotificationEnvelope) -> None:
    await redis_client.xadd(wake_stream_key(GUILD), {"payload": item.model_dump_json()})
    await redis_client.xadd(READY_STREAM_KEY, {"guild_id": GUILD})


def real_runtime(redis_client, api, queue, seen, blocked_users):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart("stayed quiet")])

    engine = AgentEngine(
        agent_runner=KimiAgentRunner(
            agent=build_proactive_agent(FunctionModel(respond), system_prompt="s"),
            summarize=None,
        ),
        skim=SkimRunner(FunctionModel(respond)),
        agent_model_id="fake",
        skim_model_id="fake",
        deps_factory=None,
    )

    async def channel(_channel_id):
        return {"name": "general"}

    async def channel_history(_channel_id, **_kwargs):
        return []

    repository = GuildHistoryRepository(redis_client, api)
    runtime = make_runtime(redis_client, api, repository, None)
    runtime.engine = engine
    runtime.discord = SimpleNamespace(channel=channel, channel_history=channel_history)
    runtime.queue = queue
    runtime.image_capabilities = SimpleNamespace(review=None, generate=None)
    runtime.blocked_users = blocked_users
    return runtime


async def test_queued_notifications_about_the_user_never_reach_the_model(
    redis_client, world
):
    api = ListAPI()
    api.addenda = world.addenda
    api.blocked = {"revision": 1, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    queue = RedisWakeQueue(redis_client, consumer_name="w")
    await queue.initialize()
    # Queued before the user was blocked: a waking mention, a pending
    # summary naming them, a claimed batch from a crashed wake, and nia.
    await publish(redis_client, envelope(KAI_MENTION))
    await publish(redis_client, envelope(NIA_MENTION))
    await redis_client.rpush(
        pending_key(GUILD), envelope(KAI_SUMMARY, wakes=False).model_dump_json()
    )
    batch_list = f"proactive:v1:{{guild:{GUILD}}}:batch:crashed"
    await redis_client.rpush(
        batch_list, envelope(KAI_SUMMARY, wakes=False).model_dump_json()
    )
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert api.acks[0]["outcome"] == "purged"
    assert "notifications_dropped=3" in api.acks[0]["detail"]
    assert await redis_client.llen(pending_key(GUILD)) == 0
    assert await redis_client.llen(batch_list) == 0

    # A mention queued after the purge by a producer that missed the block.
    await publish(redis_client, envelope(f"<@{TARGET}> ping from kai"))
    seen: list = []
    runtime = real_runtime(redis_client, api, queue, seen, blocked)
    runtime.history_writer = replica.writer
    ready = await queue.read_ready(block_ms=1)
    batch = await queue.build_batch(GUILD, ready)
    await runtime.process(batch)

    model_input = ModelMessagesTypeAdapter.dump_json(
        [m for messages in seen for m in messages if isinstance(m, ModelRequest)]
    ).decode()
    assert "release notes are up" in model_input
    assert TARGET not in model_input
    assert "Miso" not in model_input and "Kai" not in model_input
    assert not leaks(await stored_v1(redis_client))
    await replica.writer.close(timeout=1)


# 3. The consumer refuses to purge until its own list blocks the target.


async def test_consumer_waits_until_its_own_list_blocks_the_target(redis_client, world):
    api = world
    lister = ListAPI()
    lister.blocked = {"revision": 1, "user_ids": []}
    blocked = BlockedUsers(lister, redis_client)
    await blocked.refresh()
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._blocked_users = blocked
    replica.consumer._list_wait_seconds = 0.05
    replica.consumer._list_poll_seconds = 0.01
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()
    assert api.acks == []
    assert await pending_count(redis_client) == 1
    assert leaks(await stored_v1(redis_client))

    lister.blocked = {"revision": 2, "user_ids": [TARGET]}
    await replica.consumer.poll_once()  # reclaimed; the refresh now blocks
    assert [ack["outcome"] for ack in api.acks] == ["purged"]
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


# 4. Raw ids in content are scrubbed even when Discord lists no mention.


def test_raw_mentions_and_ids_in_content_are_scrubbed():
    blocked = BlockedUsers(None, None)
    blocked._user_ids = frozenset({TARGET})
    discord = DiscordREST(bot_token="t", blocked_users=blocked)
    code_block = {
        "id": "700000000000000005",
        "timestamp": "2026-10-03T11:00:00+00:00",
        "author": {"id": BYSTANDER, "username": "nia"},
        "content": f"```\nping <@{TARGET}> and <@!{TARGET}>\n```",
        "mentions": [],
    }
    webhook = {
        "id": "700000000000000006",
        "timestamp": "2026-10-03T11:01:00+00:00",
        "author": {"id": "888888888888888888", "username": "hook", "bot": True},
        "webhook_id": "888888888888888888",
        "content": f"New issue by <@{TARGET}> (user {TARGET}); cc <@{BYSTANDER}>",
    }

    first = discord._message(code_block, {})
    second = discord._message(webhook, {})

    assert TARGET not in first.content and TARGET not in second.content
    assert first.content == "```\nping @[blocked user] and @[blocked user]\n```"
    assert second.content == (
        f"New issue by @[blocked user] (user [blocked user]); cc <@{BYSTANDER}>"
    )


# 5. Durable per-request progress, delivery cap, and when a fold is skipped.


async def test_ack_outage_never_refolds_a_finished_guild(redis_client, world):
    api = world
    api.ack_status = 500
    calls: list[str] = []
    replica = Replica(redis_client, api, counting_note_model(calls), name="a")
    replica.consumer._reclaim_idle_ms = 0
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()
    after_first = await stored_v1(redis_client)
    calls_first = list(calls)
    assert calls_first[0] == "note"
    assert await pending_count(redis_client) == 1

    api.ack_status = 200
    await replica.consumer.poll_once()

    assert calls == calls_first  # redelivery: no model call at all
    assert await stored_v1(redis_client) == after_first
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    assert [ack["outcome"] for ack in api.acks] == ["purged"]
    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


async def test_deliveries_are_capped(redis_client, world):
    api = world
    api.ack_status = 500
    replica = Replica(redis_client, api, honest_model([]), name="a")
    replica.consumer._reclaim_idle_ms = 0
    replica.consumer._max_deliveries = 2
    await replica.consumer.initialize()
    await submit(redis_client, command())

    for _ in range(2):
        await replica.consumer.poll_once()
        assert await pending_count(redis_client) == 1
    await replica.consumer.poll_once()

    assert await pending_count(redis_client) == 0
    await replica.writer.close(timeout=1)


def pre_uid_history() -> list[dict]:
    """Old transcript lines (before `uid=`) under a nickname nobody listed."""
    return dump(
        [
            ModelRequest(parts=[UserPromptPart("NOTIFICATIONS: activity")]),
            ModelResponse(parts=[ToolCallPart("channel_history", {}, "c1")]),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "channel_history",
                        "[id=1] A·kaizer: my cat Miso is sick\n"
                        "[id=2] B·nia: ship Rust 1.95",
                        "c1",
                    )
                ]
            ),
            ModelResponse(parts=[TextPart("noted")]),
        ]
    )


async def test_pre_uid_history_is_folded_once_per_request(redis_client):
    api = FakeAPI()
    await redis_client.set(f"proactive:v1:{{guild:{GUILD}}}:owner", "external")
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, pre_uid_history(), revision=3)
    )
    calls: list[str] = []
    replica = Replica(redis_client, api, counting_note_model(calls), name="a")
    await replica.consumer.initialize()
    first_run = command()
    await submit(redis_client, first_run)

    await replica.consumer.poll_once()

    v1 = await stored_v1(redis_client)
    assert "kaizer" not in v1 and "Miso" not in v1
    assert calls == ["note"]
    revision = json.loads(v1)["revision"]

    rerun = dict(first_run, run_id=str(uuid4()))  # same request, new run
    await submit(redis_client, rerun)
    await replica.consumer.poll_once()

    assert calls == ["note"]  # no model call
    assert json.loads(await stored_v1(redis_client))["revision"] == revision
    assert await redis_client.get(purge_epoch_key(GUILD)) == b"1"
    assert [(ack["run_id"], ack["outcome"]) for ack in api.acks] == [
        (first_run["run_id"], "purged"),
        (rerun["run_id"], "purged"),
    ]
    await replica.writer.close(timeout=1)


async def test_a_fully_attributed_history_without_the_user_is_not_folded(
    redis_client,
):
    api = FakeAPI()
    await redis_client.set(f"proactive:v1:{{guild:{GUILD}}}:owner", "external")
    history = dump(
        [
            *memory_note_pair("nia ships Rust; zara reviews"),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "channel_history",
                        f"[id=2] B·nia (uid={BYSTANDER}): ship Rust 1.95\n"
                        "[BLOCKED BY USER]",
                        "c1",
                    )
                ]
            ),
        ]
    )
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, history, revision=3)
    )
    calls: list[str] = []
    replica = Replica(redis_client, api, counting_note_model(calls), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert calls == []
    assert json.loads(await stored_v1(redis_client))["revision"] == 3
    assert api.puts == []
    assert not await redis_client.exists(purge_epoch_key(GUILD))
    assert api.acks[0]["outcome"] == "unchanged"
    await replica.writer.close(timeout=1)


async def test_a_raw_line_without_uid_is_folded_even_without_any_hit(redis_client):
    api = FakeAPI()
    await redis_client.set(f"proactive:v1:{{guild:{GUILD}}}:owner", "external")
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, pre_uid_history(), revision=3)
    )
    calls: list[str] = []
    replica = Replica(redis_client, api, counting_note_model(calls), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert calls == ["note"]
    assert api.acks[0]["outcome"] == "purged"
    await replica.writer.close(timeout=1)


# 7. Names that survived the retry are reported in the ack.


async def test_name_hits_after_retry_are_reported(redis_client, world):
    api = world

    def stubborn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(f"{TARGET_NAME.upper()} liked Rust")])

    async def decide_watch(entries, **_kwargs):
        return dict(entries), 2

    replica = Replica(redis_client, api, FunctionModel(stubborn), name="a")
    replica.consumer._decide_watch = decide_watch
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    ack = api.acks[0]
    assert ack["outcome"] == "purged"
    assert "history folded attempts=2 name_hits=1" in ack["detail"]
    assert "watch channels_rewritten=0 name_hits=2" in ack["detail"]
    assert TARGET_NAME not in ack["detail"].lower().replace("name_hits", "")
    await replica.writer.close(timeout=1)


# 8. An epoch bump refreshes a cached memory block even when history is
#    not loaded at that moment.


async def test_epoch_bump_refreshes_memory_when_history_is_not_loaded(
    redis_client,
):
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=60)
    runtime: GuildRuntime = make_runtime(redis_client, api, repository, writer)
    await runtime.process(batch())
    assert api.memory_reads == 1
    # A hand-off reload left history unloaded but the memory block cached.
    runtime.history_loaded = False
    await redis_client.incr(purge_epoch_key(GUILD))

    await runtime.process(batch())

    assert api.memory_reads == 2
    await writer.close(timeout=1)


async def test_legacy_key_left_alone_once_an_epoch_exists(redis_client, world):
    # A legacy key next to an existing epoch is never migrated back in.
    api = world
    await redis_client.delete(history_key(GUILD))
    api.durable.pop(GUILD)
    await redis_client.incr(purge_epoch_key(GUILD))
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert not await redis_client.exists(history_key(GUILD))
    assert await redis_client.exists(legacy_history_key(GUILD))
    assert api.acks[0]["detail"].startswith("history empty")
    await replica.writer.close(timeout=1)


def test_raw_history_fixture_mentions_the_target():
    assert TARGET in json.dumps(dump(raw_history_with_target()))
