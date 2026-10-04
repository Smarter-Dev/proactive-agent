"""One persistent, isolated proactive agent runtime per Discord guild."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from proactive_agent.agent import OPERATING_POLICY_BRIEF
from proactive_agent.contracts import ControlCommand, NotificationEnvelope
from proactive_agent.engine import AgentEngine, render_notifications
from proactive_agent.environment import ChannelEnvironment, InstructionStore
from proactive_agent.history import (
    DebouncedHistoryWriter,
    GuildHistoryRepository,
    HistoryUnavailableError,
    StaleHistoryError,
)
from proactive_agent.keys import (
    checkpoint_key,
    control_stream_key,
    history_invalid_key,
    history_key,
    purge_epoch_key,
)
from proactive_agent.parity import ProactiveDeps
from proactive_agent.queue import RedisWakeQueue, WakeBatch
from proactive_agent.response_fitting import split_for_discord
from proactive_agent.types import ActivationResult

MEMORY_REFRESH_SECONDS = 3600
_SNOWFLAKE_PATTERN = re.compile(r"(?<![0-9])([0-9]{15,22})(?![0-9])")
# One round trip at wake start: the guild's purge epoch and the revision of
# the v1 history snapshot (matched as text, never decoding the history).
_STORE_STATE_LUA = """
local epoch = redis.call('GET', KEYS[1]) or ''
local current = redis.call('GET', KEYS[2])
local revision = ''
if current then
  revision = string.match(current, '"revision":(%d+)') or ''
end
local invalid = tostring(redis.call('EXISTS', KEYS[3]))
return {epoch, revision, invalid}
"""
HISTORY_FETCH_LIMIT = 60
# One dropped wake is worth saying out loud; a broken model or a poisoned
# history would otherwise repeat it every few minutes for hours.
FAILURE_NOTICE_INTERVAL_SECONDS = 6 * 60 * 60
FAILURE_NOTICE_ERROR_LIMIT = 1500


def failure_notice_text(error: str) -> str:
    """The single message a guild sees when a wake is given up on."""
    detail = error.replace("```", "'''")[:FAILURE_NOTICE_ERROR_LIMIT]
    return (
        "\u26a0\ufe0f I hit an error while waking up and gave up on that wake "
        "after several retries, so I may have missed something here. This one "
        "needs a look at my logs.\n"
        f"```{detail}```\n"
        "I'll stay quiet about any further failures for the next 6 hours."
    )


def _decode_or_none(value) -> str | None:
    if value is None or value in (b"", ""):
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


def render_memory_block(memory: dict | None) -> str:
    if not memory or memory.get("memory_enabled") is False:
        return ""
    sections = []
    content = memory.get("content")
    if content:
        stamp = (
            f" (dreamed {memory['updated_at'][:10]})"
            if memory.get("updated_at")
            else ""
        )
        sections.append(f"GUILD MEMORY{stamp}:\n{content}")
    notes = memory.get("notes") or ()
    if notes:
        lines = "\n".join(
            f"- [{note.get('channel_name') or note.get('channel_id') or 'somewhere'}] "
            f"{note.get('content', '')}"
            for note in notes
        )
        sections.append(f"NOTES YOU KEPT TODAY:\n{lines}")
    if not sections:
        return ""
    return "YOUR MEMORY (refreshed at most hourly):\n" + "\n\n".join(sections)


class ActionJournal:
    """Best-effort idempotency checkpoints for one wake's Discord actions."""

    def __init__(self, redis_client):
        self._redis = redis_client

    async def completed(self, guild_id: str, wake_id: str, action_id: str) -> bool:
        return bool(
            await self._redis.hexists(checkpoint_key(guild_id, wake_id), action_id)
        )

    async def record(
        self,
        guild_id: str,
        wake_id: str,
        action_id: str,
        result_id: str,
    ) -> None:
        key = checkpoint_key(guild_id, wake_id)
        async with self._redis.pipeline(transaction=True) as pipeline:
            pipeline.hset(key, action_id, result_id)
            pipeline.expire(key, 7 * 24 * 60 * 60)
            await pipeline.execute()


@dataclass
class GuildRuntime:
    guild_id: str
    engine: AgentEngine
    history_repository: GuildHistoryRepository
    history_writer: DebouncedHistoryWriter
    api: object
    discord: object
    redis: object
    queue: RedisWakeQueue
    journal: ActionJournal
    bot_user_id: str
    guild_name: str
    summarize_web: object
    image_capabilities: object
    media_reader: object
    author_handler: object
    history_loaded: bool = False
    history_revision: int = 0
    memory_block: str = ""
    memory_refreshed_at: float = 0
    # The guild's purge-epoch value the in-RAM history was loaded under.
    purge_epoch: str | None = None
    store_synced: bool = False
    # Anything with is_blocked(user_id); notification bodies naming a
    # blocked user's id never reach the brief. None blocks nobody.
    blocked_users: object = None

    async def process(self, batch: WakeBatch) -> ActivationResult:
        if batch.guild_id != self.guild_id:
            raise ValueError("wake batch crossed guild runtime boundary")
        await self._sync_with_store()
        enabled_rows = await self.api.list_enabled_channels(self.guild_id)
        enabled_channels: dict[str, str] = {}
        instruction_stores: dict[str, InstructionStore] = {}
        persisted_addenda: dict[str, str] = {}
        notifications = self._without_blocked(batch.notifications)
        for row in enabled_rows:
            channel = await self.discord.channel(row.channel_id)
            channel_name = channel.get("name") or row.channel_id
            enabled_channels[row.channel_id] = channel_name
            store = InstructionStore.from_stored(
                OPERATING_POLICY_BRIEF, row.watch_addendum
            )
            persisted_addenda[row.channel_id] = store.to_stored()
            for expired in store.prune_expired(now=datetime.now(UTC)):
                notifications.append(
                    self._local_notification(
                        batch,
                        kind="instruction_expired",
                        channel_id=row.channel_id,
                        channel_name=channel_name,
                        body=(
                            f"Watch instruction {expired.instruction_id} expired: "
                            f'"{expired.text}"'
                        ),
                    )
                )
            instruction_stores[row.channel_id] = store
        if not enabled_channels:
            return ActivationResult(
                responses=[],
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                model_id=self.engine.agent_model_id,
                details={"agent": {"note": "no enabled channels"}},
            )

        await self._load_history()
        if (
            not self.memory_refreshed_at
            or time.monotonic() - self.memory_refreshed_at >= MEMORY_REFRESH_SECONDS
        ):
            self.memory_block = render_memory_block(
                await self.api.get_memory(self.guild_id)
            )
            self.memory_refreshed_at = time.monotonic()

        async def channel_envs(channel_id: str) -> ChannelEnvironment:
            messages = await self.discord.channel_history(
                channel_id,
                guild_id=self.guild_id,
                limit=HISTORY_FETCH_LIMIT,
            )
            return ChannelEnvironment(visible=messages, bot_user_id=self.bot_user_id)

        async def request_mode(channel_id: str, mode: str, minutes: int) -> str:
            command = ControlCommand(
                command_id=uuid4(),
                guild_id=self.guild_id,
                channel_id=channel_id,
                mode=mode,
                minutes=max(0, minutes),
                created_at=datetime.now(UTC),
                trace_id=batch.waking[0].envelope.trace_id,
            )
            await self.redis.xadd(
                control_stream_key(), {"payload": command.model_dump_json()}
            )
            if mode == "active":
                return f"Monitoring mode set to active for {minutes} minutes."
            return "Monitoring mode set to passive."

        async def drain_notifications() -> str:
            arrived, dropped = await self.queue.drain_midrun(batch)
            arrived = tuple(self._without_blocked(arrived))
            if not arrived:
                return "No new notifications."
            return render_notifications(arrived, dropped)

        def deps_factory(**kwargs):
            return ProactiveDeps(
                guild_id=self.guild_id,
                discord=self.discord,
                api=self.api,
                summarize_web=self.summarize_web,
                describe_media=self.media_reader,
                review_image_prompt=self.image_capabilities.review,
                generate_image_bytes=self.image_capabilities.generate,
                author_handler=self.author_handler,
                request_mode=request_mode,
                drain_notifications=drain_notifications,
                **kwargs,
            )

        self.engine.deps_factory = deps_factory
        result = await self.engine.wake(
            notifications=tuple(notifications),
            dropped=batch.dropped,
            enabled_channels=enabled_channels,
            instruction_stores=instruction_stores,
            channel_envs=channel_envs,
            brief_preamble=self.memory_block,
        )
        responses = await self._dispatch(batch, result, enabled_channels)
        await self._persist_instructions(
            enabled_channels, instruction_stores, persisted_addenda
        )
        serialized_history = json.loads(
            ModelMessagesTypeAdapter.dump_json(self.engine.agent_runner.history)
        )
        snapshot = await self._save_history(serialized_history)
        self.history_revision = snapshot.revision
        await self._record_usage(batch, result, responses)
        return result

    async def report_failure(self, batch: WakeBatch, error: str) -> str | None:
        """Say in Discord that a wake was dropped, at most once per window.

        Returns the channel posted in, or None when there was nowhere to
        post or the window is still held by an earlier failure.
        """
        channel_id = await self._failure_notice_channel(batch)
        if channel_id is None:
            return None
        # Claim last: an unclaimable window must not be spent on a notice
        # that was never going to be sent.
        if not await self.queue.claim_failure_notice(
            self.guild_id, ttl_seconds=FAILURE_NOTICE_INTERVAL_SECONDS
        ):
            return None
        await self.discord.send_message(channel_id, failure_notice_text(error))
        return channel_id

    async def _failure_notice_channel(self, batch: WakeBatch) -> str | None:
        """The channel the wake was for, else any the agent is enabled in."""
        enabled = [
            row.channel_id
            for row in await self.api.list_enabled_channels(self.guild_id)
        ]
        if not enabled:
            return None
        candidates = [item.envelope.channel_id for item in batch.waking]
        candidates += [envelope.channel_id for envelope in batch.notifications]
        return next(
            (channel_id for channel_id in candidates if channel_id in enabled),
            enabled[0],
        )

    async def _save_history(self, history: list[dict]):
        try:
            return await self.history_writer.save(
                guild_id=self.guild_id,
                history=history,
                previous_revision=self.history_revision,
            )
        except StaleHistoryError:
            # Someone else (another replica, or a purge) stored a newer
            # revision; this copy is stale and must be reloaded, not kept.
            self.forget_history()
            raise

    def _without_blocked(self, notifications) -> list[NotificationEnvelope]:
        """Drop notifications whose body names a blocked user's id.

        The producer stops emitting them; this is the backstop for anything
        queued before the user was blocked.
        """
        if self.blocked_users is None:
            return list(notifications)
        kept = []
        for notification in notifications:
            ids = _SNOWFLAKE_PATTERN.findall(notification.body)
            if any(self.blocked_users.is_blocked(user_id) for user_id in ids):
                continue
            kept.append(notification)
        return kept

    def forget_history(self) -> None:
        """Drop the in-RAM history and memory so the next wake reloads both."""
        self.history_loaded = False
        self.engine.agent_runner.history = []
        self.memory_block = ""
        self.memory_refreshed_at = 0

    async def _sync_with_store(self) -> None:
        """One Redis call per wake, before the model runs.

        - The purge epoch moved: a purge replaced the guild's history, so
          history AND memory are reloaded (saving on the stale in-RAM copy
          would bring the purged user's words back).
        - The history-invalid tombstone exists: the in-RAM copy is dropped
          and the wake refused before the model runs.
        - The stored v1 revision differs from the one this runtime holds:
          another replica ran the guild's last wake, so history is reloaded.
          Without this the wake would run and only then have its save
          refused by the compare-on-revision backstop.
        """
        raw_epoch, raw_revision, raw_invalid = await self.redis.eval(
            _STORE_STATE_LUA,
            3,
            purge_epoch_key(self.guild_id),
            history_key(self.guild_id),
            history_invalid_key(self.guild_id),
        )
        epoch = _decode_or_none(raw_epoch)
        revision = _decode_or_none(raw_revision)
        if _decode_or_none(raw_invalid) == "1":
            # A purge rewrote Postgres but could not replace v1. The Postgres
            # copy is the purged one: under this wake's guild lease, restore
            # v1 from it and carry on. Only when Postgres does not hold the
            # purged revision does the wake defer (no model call, no Discord
            # action) and the guild stay tombstoned.
            self.forget_history()
            self.store_synced = False
            self.history_writer.discard(self.guild_id)
            if not await self.history_repository.recover_tombstone(self.guild_id):
                raise HistoryUnavailableError(
                    f"proactive history tombstoned for guild {self.guild_id}"
                )
            self.purge_epoch = epoch
            self.store_synced = True
            return
        if self.store_synced and epoch != self.purge_epoch:
            # Also when history is not loaded: a cached memory block from
            # before the purge must be refetched too.
            self.forget_history()
        elif self.history_loaded and revision != str(self.history_revision):
            self.history_loaded = False
            self.engine.agent_runner.history = []
        self.purge_epoch = epoch
        self.store_synced = True

    async def _load_history(self) -> None:
        if self.history_loaded:
            return
        snapshot = await self.history_repository.load(self.guild_id)
        self.engine.agent_runner.history = list(
            ModelMessagesTypeAdapter.validate_json(
                json.dumps(snapshot.history).encode()
            )
        )
        self.history_revision = snapshot.revision
        self.history_loaded = True

    async def record_unavailable_retries(
        self, batch: WakeBatch, *, retries: int, recovered: bool
    ) -> None:
        """Put a worker error in the next agent wake's history without waking it."""
        await self._load_history()
        marker = f"[WORKER UNAVAILABLE wake={batch.wake_id}]"
        history = self.engine.agent_runner.history
        if any(
            isinstance(message, ModelRequest)
            and any(
                isinstance(part, UserPromptPart)
                and isinstance(part.content, str)
                and part.content.startswith(marker)
                for part in message.parts
            )
            for message in history
        ):
            return
        outcome = (
            "The same wake eventually completed."
            if recovered
            else "The wake was abandoned and may have missed an action."
        )
        note = (
            f"{marker}\n"
            "This is a worker-generated error record, not a Discord user message "
            "or a request to respond now. A model request returned HTTP 503 "
            f"UNAVAILABLE. The worker retried {retries} time(s) after the "
            f"initial attempt. {outcome} Use this context on the next real wake."
        )
        updated = [
            *history,
            ModelRequest(parts=[UserPromptPart(note)]),
            ModelResponse(parts=[TextPart("Worker error recorded for the next wake.")]),
        ]
        snapshot = await self._save_history(
            json.loads(ModelMessagesTypeAdapter.dump_json(updated))
        )
        self.engine.agent_runner.history = updated
        self.history_revision = snapshot.revision

    async def _dispatch(
        self,
        batch: WakeBatch,
        result: ActivationResult,
        enabled_channels: dict[str, str],
    ) -> int:
        dispatched = 0
        for response_index, response in enumerate(result.responses):
            if response.channel_id not in enabled_channels:
                continue
            sent_any = False
            for part_index, part in enumerate(split_for_discord(response.content)):
                action_id = f"response:{response_index}:part:{part_index}"
                if await self.journal.completed(
                    self.guild_id, batch.wake_id, action_id
                ):
                    sent_any = True
                    continue
                sent = await self.discord.send_message(
                    response.channel_id,
                    part,
                    reply_to_id=(response.reply_to_id if part_index == 0 else None),
                )
                await self.journal.record(
                    self.guild_id,
                    batch.wake_id,
                    action_id,
                    str(sent.get("id", "sent")),
                )
                sent_any = True
            if sent_any:
                dispatched += 1
        for index, reaction in enumerate(result.reactions):
            if reaction.channel_id not in enabled_channels:
                continue
            action_id = f"reaction:{index}"
            if await self.journal.completed(self.guild_id, batch.wake_id, action_id):
                continue
            await self.discord.add_reaction(
                reaction.channel_id, reaction.message_id, reaction.emoji
            )
            await self.journal.record(
                self.guild_id, batch.wake_id, action_id, "reacted"
            )
        deps = self.engine.last_deps
        if isinstance(deps, ProactiveDeps):
            images_by_channel: dict[str, list] = {}
            for pending in deps.pending_images:
                if pending.channel_id in enabled_channels:
                    images_by_channel.setdefault(pending.channel_id, []).append(pending)
            for channel_id, pending_images in images_by_channel.items():
                action_id = f"images:{channel_id}"
                if await self.journal.completed(
                    self.guild_id, batch.wake_id, action_id
                ):
                    continue
                reply_to_id = next(
                    (
                        response.reply_to_id
                        for response in result.responses
                        if response.channel_id == channel_id and response.reply_to_id
                    ),
                    None,
                )
                sent = await self.discord.send_files(
                    channel_id,
                    [
                        (image.filename, image.data, image.mime_type)
                        for image in pending_images
                    ],
                    reply_to_id=reply_to_id,
                )
                await self.journal.record(
                    self.guild_id,
                    batch.wake_id,
                    action_id,
                    str(sent.get("id", "sent")),
                )
        return dispatched

    async def _persist_instructions(
        self,
        enabled_channels: dict[str, str],
        instruction_stores: dict[str, InstructionStore],
        persisted_addenda: dict[str, str],
    ) -> None:
        for channel_id, store in instruction_stores.items():
            stored = store.to_stored()
            if stored == persisted_addenda[channel_id]:
                continue
            await self.api.set_watch_addendum(
                guild_id=self.guild_id,
                channel_id=channel_id,
                enabled=channel_id in enabled_channels,
                watch_addendum=stored,
            )

    async def _record_usage(
        self, batch: WakeBatch, result: ActivationResult, responses: int
    ) -> None:
        entries = [
            {
                "model_id": model_id,
                "operation": "agent",
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_tokens": usage.get("cache_read_tokens", 0),
            }
            for model_id, usage in (result.usage_by_model or {}).items()
        ]
        if entries:
            await self.api.record_usage(
                guild_id=self.guild_id,
                wake_id=batch.wake_id,
                metered_at=datetime.now(UTC),
                passive=batch.passive,
                responses=responses,
                entries=entries,
            )

    def _local_notification(
        self,
        batch: WakeBatch,
        *,
        kind: str,
        channel_id: str,
        channel_name: str,
        body: str,
    ) -> NotificationEnvelope:
        first = batch.waking[0].envelope
        return NotificationEnvelope(
            schema_version=1,
            notification_id=uuid4(),
            guild_id=self.guild_id,
            channel_id=channel_id,
            channel_name=channel_name,
            kind=kind,
            created_at=datetime.now(UTC),
            body=body,
            message_ids=(),
            wakes=False,
            passive=first.passive,
            watcher_usage={},
            trace_id=first.trace_id,
        )


class GuildRuntimeRegistry:
    """Caches one independently stateful runtime for each guild."""

    def __init__(self, factory):
        self._factory = factory
        self._runtimes: dict[str, GuildRuntime] = {}

    async def get(self, guild_id: str) -> GuildRuntime:
        runtime = self._runtimes.get(guild_id)
        if runtime is None:
            runtime = await self._factory(guild_id)
            self._runtimes[guild_id] = runtime
        return runtime

    def forget(self, guild_id: str) -> None:
        """Make a cached runtime reload its history and memory next wake."""
        runtime = self._runtimes.get(guild_id)
        if runtime is not None:
            runtime.forget_history()

    @property
    def guild_ids(self) -> frozenset[str]:
        return frozenset(self._runtimes)
