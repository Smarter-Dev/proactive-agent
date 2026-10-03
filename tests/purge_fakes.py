"""Shared fakes for the privacy purge tests (fakeredis + an in-memory API)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

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

from proactive_agent.api import ApplicationAPIError
from proactive_agent.contracts import EnabledChannel, NotificationEnvelope
from proactive_agent.queue import StreamNotification, WakeBatch
from proactive_agent.runtime import ActionJournal, GuildRuntime
from proactive_agent.types import ActivationResult

TARGET = "111111111111111111"
TARGET_NAME = "kai"
BYSTANDER = "222222222222222222"
BYSTANDER_NAME = "nia"
GUILD = "333333333333333333"
CHANNEL = "444444444444444444"


class FakeAPI:
    """In-memory stand-in for the smarter-dev bot API."""

    def __init__(self):
        self.durable: dict[str, object] = {}
        self.puts = []
        self.put_attempts = []
        self.fail_puts = False
        self.addenda: dict[str, dict[str, str]] = {}
        self.addendum_writes = []
        self.acks = []
        self.ack_status = 200
        self.memory_reads = 0
        self.blocked = {"revision": 0, "user_ids": []}
        self.blocked_failures = 0

    async def get_history(self, guild_id):
        return self.durable.get(guild_id)

    async def put_history(self, snapshot):
        self.put_attempts.append(snapshot)
        if self.fail_puts:
            raise ApplicationAPIError(500, "boom")
        current = self.durable.get(snapshot.guild_id)
        if current is not None and snapshot.revision <= current.revision:
            if (
                snapshot.revision == current.revision
                and snapshot.checksum == current.checksum
            ):
                return
            raise ApplicationAPIError(409, "conflict")
        self.puts.append(snapshot)
        self.durable[snapshot.guild_id] = snapshot

    async def list_enabled_channels(self, guild_id):
        return tuple(
            EnabledChannel(channel_id=channel_id, watch_addendum=addendum)
            for channel_id, addendum in self.addenda.get(guild_id, {}).items()
        )

    async def set_watch_addendum(
        self, *, guild_id, channel_id, enabled, watch_addendum
    ):
        self.addendum_writes.append((guild_id, channel_id, watch_addendum))
        self.addenda.setdefault(guild_id, {})[channel_id] = watch_addendum
        return {}

    async def get_memory(self, guild_id):
        self.memory_reads += 1
        return {"content": f"memory read {self.memory_reads}"}

    async def record_usage(self, **kwargs):
        return None

    async def post_privacy_ack(
        self, run_id, *, component, guild_id, outcome, stores, detail
    ):
        if self.ack_status == 404:
            return False
        if self.ack_status >= 400:
            raise ApplicationAPIError(self.ack_status, "ack failed")
        self.acks.append(
            {
                "run_id": run_id,
                "component": component,
                "guild_id": guild_id,
                "outcome": outcome,
                "stores": list(stores),
                "detail": detail,
            }
        )
        return True

    async def get_blocked_users(self):
        if self.blocked_failures:
            self.blocked_failures -= 1
            raise ApplicationAPIError(None, "connection refused")
        return dict(self.blocked)


def raw_history_with_target() -> list[ModelMessage]:
    """A wake where both kai and nia spoke, as tool output and in the brief."""
    return [
        ModelRequest(
            parts=[
                UserPromptPart(
                    f"NOTIFICATIONS: [mention] {TARGET_NAME} (uid={TARGET}) asked "
                    "about cats"
                )
            ]
        ),
        ModelResponse(parts=[ToolCallPart("channel_history", {}, "c1")]),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "channel_history",
                    f"[id=1] A·{TARGET_NAME} (uid={TARGET}): my cat Miso is sick\n"
                    f"[id=2] B·{BYSTANDER_NAME} (uid={BYSTANDER}): ship Rust 1.95",
                    "c1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart(f"told {TARGET_NAME} to see a vet")]),
    ]


def dump(history: list[ModelMessage]) -> list[dict]:
    return json.loads(ModelMessagesTypeAdapter.dump_json(history))


def notification() -> NotificationEnvelope:
    return NotificationEnvelope(
        schema_version=1,
        notification_id=uuid4(),
        guild_id=GUILD,
        channel_id=CHANNEL,
        channel_name="general",
        kind="mention",
        created_at=datetime.now(UTC),
        body="someone mentioned the bot",
        message_ids=("1",),
        wakes=True,
        passive=False,
        watcher_usage={},
        trace_id=uuid4(),
    )


def batch() -> WakeBatch:
    item = StreamNotification(stream_id="1-0", envelope=notification())
    return WakeBatch(
        guild_id=GUILD,
        wake_id=str(item.envelope.notification_id),
        ready_ids=[],
        waking=[item],
        pending=[],
        dropped=0,
    )


class FakeEngine:
    """Records the history and memory each wake starts from."""

    agent_model_id = "fake"

    def __init__(self):
        self.agent_runner = SimpleNamespace(history=[])
        self.deps_factory = None
        self.last_deps = None
        self.seen_histories: list[str] = []
        self.seen_preambles: list[str] = []

    async def wake(self, *, brief_preamble="", **_kwargs):
        self.seen_histories.append(
            ModelMessagesTypeAdapter.dump_json(self.agent_runner.history).decode()
        )
        self.seen_preambles.append(brief_preamble)
        self.agent_runner.history = [
            *self.agent_runner.history,
            ModelRequest(parts=[UserPromptPart("wake")]),
            ModelResponse(parts=[TextPart("nothing to do")]),
        ]
        return ActivationResult(
            responses=[],
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            model_id="fake",
        )


def make_runtime(redis_client, api, repository, writer) -> GuildRuntime:
    discord = SimpleNamespace()

    async def channel(_channel_id):
        return {"name": "general"}

    discord.channel = channel
    return GuildRuntime(
        guild_id=GUILD,
        engine=FakeEngine(),
        history_repository=repository,
        history_writer=writer,
        api=api,
        discord=discord,
        redis=redis_client,
        queue=SimpleNamespace(),
        journal=ActionJournal(redis_client),
        bot_user_id="999999999999999999",
        guild_name="guild",
        summarize_web=None,
        image_capabilities=None,
        media_reader=None,
        author_handler=None,
    )


def watch_addendum() -> str:
    expires = (datetime.now(UTC) + timedelta(hours=2)).isoformat()
    return json.dumps(
        [
            {
                "id": "w1",
                "text": f"wake when {TARGET_NAME} reports back",
                "expires_at": expires,
            },
            {
                "id": "w2",
                "text": "watch for nia's Rust 1.95 release",
                "expires_at": expires,
            },
        ]
    )
