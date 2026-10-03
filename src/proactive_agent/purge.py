"""Per-user privacy purge of the worker's guild state (purge contract v1, #79).

One PurgeCommand arrives per purge run on ``privacy:v1:purge``. For each guild
the worker fences the guild, has the agent's own model rewrite its memory
without the user (the whole history folds into one memory note), replaces the
stored history synchronously, lets the model purge its watch instructions,
bumps the guild's purge epoch and acks the outcome to the bot API.

The command carries the target's id and names. They go into the purge
prompts and nowhere else: logs name run/request/guild ids only, ack details
are content-free, and exception text (which can echo payloads or model
output) is never logged or acked, only exception type names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from proactive_agent.agent import (
    OPERATING_POLICY_BRIEF,
    PrivacyCompactionError,
    PrivacyNote,
    privacy_compaction_summary,
    privacy_watch_decisions,
    purge_agent_history,
)
from proactive_agent.contracts import PurgeCommand
from proactive_agent.environment import InstructionStore
from proactive_agent.history import canonical_history
from proactive_agent.keys import (
    PRIVACY_PURGE_STREAM_KEY,
    lease_key,
    privacy_lock_key,
    purge_epoch_key,
)

logger = logging.getLogger(__name__)

PURGE_GROUP = "proactive-agent-workers-v1-privacy"
PURGE_COMPONENT = "worker"
RECLAIM_IDLE_MS = 10 * 60 * 1000
PRIVACY_LOCK_SECONDS = 600
FENCE_WAIT_SECONDS = 15 * 60
FENCE_POLL_SECONDS = 0.5

STORE_REDIS_HISTORY = "proactive:v1:history"
STORE_POSTGRES_HISTORY = "proactive_agent_histories"
STORE_WATCH = "watch_instructions"

_RENEW_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_RELEASE_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

CompactFn = Callable[..., Awaitable[PrivacyNote]]
WatchFn = Callable[..., Awaitable[dict[str, str | None]]]


class PurgeFailed(Exception):
    """A guild purge failed; the message is a content-free ack detail."""


@dataclass
class GuildOutcome:
    outcome: str
    stores: list[str] = field(default_factory=list)
    detail: str = ""


class PrivacyLock:
    """``privacy-lock`` fence for a guild the external worker does not own."""

    def __init__(self, redis_client, *, guild_id: str, token: str, ttl_seconds: int):
        self._redis = redis_client
        self.guild_id = guild_id
        self.token = token
        self.ttl_seconds = ttl_seconds

    async def renew(self) -> bool:
        return bool(
            await self._redis.eval(
                _RENEW_LOCK_LUA,
                1,
                privacy_lock_key(self.guild_id),
                self.token,
                self.ttl_seconds * 1000,
            )
        )

    async def release(self) -> None:
        await self._redis.eval(
            _RELEASE_LOCK_LUA, 1, privacy_lock_key(self.guild_id), self.token
        )


@dataclass
class Fence:
    """Either the guild lease (external owner) or the privacy lock."""

    holder: object
    external: bool
    lost: bool = False


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _error_detail(prefix: str, error: BaseException) -> str:
    """Content-free description: the exception type and HTTP status only."""
    if isinstance(error, PurgeFailed | PrivacyCompactionError):
        return f"{prefix}: {error}"[:500]
    status = getattr(error, "status_code", None)
    suffix = f" status={status}" if status is not None else ""
    return f"{prefix}: {type(error).__name__}{suffix}"[:500]


class PrivacyPurgeConsumer:
    """Reads purge commands and purges each guild's worker-side state."""

    def __init__(
        self,
        redis_client,
        api,
        queue,
        repository,
        writer,
        runtimes,
        *,
        model=None,
        consumer_name: str,
        compact: CompactFn | None = None,
        decide_watch: WatchFn | None = None,
        fence_wait_seconds: float = FENCE_WAIT_SECONDS,
        fence_poll_seconds: float = FENCE_POLL_SECONDS,
        reclaim_idle_ms: int = RECLAIM_IDLE_MS,
    ):
        if model is None and (compact is None or decide_watch is None):
            raise ValueError("a model is required unless both steps are injected")
        self._redis = redis_client
        self._api = api
        self._queue = queue
        self._repository = repository
        self._writer = writer
        self._runtimes = runtimes
        self._consumer_name = consumer_name
        self._compact = compact or (
            lambda messages, **kwargs: privacy_compaction_summary(
                model, messages, **kwargs
            )
        )
        self._decide_watch = decide_watch or (
            lambda entries, **kwargs: privacy_watch_decisions(model, entries, **kwargs)
        )
        self._fence_wait_seconds = fence_wait_seconds
        self._fence_poll_seconds = fence_poll_seconds
        self._reclaim_idle_ms = reclaim_idle_ms

    async def initialize(self) -> None:
        try:
            await self._redis.xgroup_create(
                PRIVACY_PURGE_STREAM_KEY, PURGE_GROUP, id="0", mkstream=True
            )
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise

    async def run(self, stop: asyncio.Event) -> None:
        initialized = False
        while not stop.is_set():
            try:
                if not initialized:
                    await self.initialize()
                    initialized = True
                await self.poll_once(block_ms=5_000)
                # Let other tasks run even if the server answered at once.
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "privacy purge consumer error type=%s", type(error).__name__
                )
                try:
                    await asyncio.wait_for(stop.wait(), timeout=5)
                except TimeoutError:
                    pass

    async def poll_once(self, *, block_ms: int = 0) -> int:
        """Handle reclaimed entries, else at most one new one. Returns count."""
        reclaimed = await self._redis.xautoclaim(
            PRIVACY_PURGE_STREAM_KEY,
            PURGE_GROUP,
            self._consumer_name,
            self._reclaim_idle_ms,
            "0-0",
            count=10,
        )
        entries = list(reclaimed[1]) if reclaimed else []
        if not entries:
            try:
                records = await self._redis.xreadgroup(
                    PURGE_GROUP,
                    self._consumer_name,
                    {PRIVACY_PURGE_STREAM_KEY: ">"},
                    count=1,
                    block=block_ms or None,
                )
            except RedisTimeoutError:
                records = []
            for _stream, stream_entries in records or ():
                entries.extend(stream_entries)
        for stream_id, fields in entries:
            await self.handle(_decode(stream_id), fields)
        return len(entries)

    async def handle(self, stream_id: str, fields) -> None:
        raw = None
        if fields:
            raw = fields.get(b"payload", fields.get("payload"))
        try:
            if raw is None:
                raise ValueError("missing payload")
            command = PurgeCommand.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            # The error text would echo the payload (the target's id and
            # names), so only the stream id is logged.
            logger.warning("privacy purge command malformed, dropped id=%s", stream_id)
            await self._ack(stream_id)
            return
        run_id = str(command.run_id)
        logger.info(
            "privacy purge started run=%s request=%s guilds=%d",
            run_id,
            command.request_id,
            len(command.guild_ids),
        )
        all_acked = True
        for guild_id in dict.fromkeys(command.guild_ids):
            result = await self.purge_guild(command, guild_id)
            logger.info(
                "privacy purge guild finished run=%s guild=%s outcome=%s",
                run_id,
                guild_id,
                result.outcome,
            )
            try:
                accepted = await self._api.post_privacy_ack(
                    run_id,
                    component=PURGE_COMPONENT,
                    guild_id=guild_id,
                    outcome=result.outcome,
                    stores=result.stores,
                    detail=result.detail,
                )
            except Exception as error:
                all_acked = False
                logger.error(
                    "privacy purge ack failed run=%s guild=%s type=%s",
                    run_id,
                    guild_id,
                    type(error).__name__,
                )
                continue
            if not accepted:
                logger.warning("privacy purge run unknown, dropped run=%s", run_id)
                await self._ack(stream_id)
                return
        if all_acked:
            await self._ack(stream_id)
            logger.info("privacy purge acknowledged run=%s", run_id)
        # Otherwise the entry stays pending and is reclaimed after the idle
        # timeout; every step is idempotent, so the retry redoes all guilds.

    async def _ack(self, stream_id: str) -> None:
        # XACK only: the producer XDELs the entry (it carries the target)
        # once every component acked every guild.
        await self._redis.xack(PRIVACY_PURGE_STREAM_KEY, PURGE_GROUP, stream_id)

    async def purge_guild(self, command: PurgeCommand, guild_id: str) -> GuildOutcome:
        """Purge one guild. Never raises; failures come back as an outcome."""
        stores: list[str] = []
        if not guild_id.isdigit() or len(guild_id) > 20:
            return GuildOutcome("failed", stores, "guild id outside supported range")
        try:
            fence = await self._acquire_fence(guild_id)
        except Exception as error:
            return GuildOutcome("failed", stores, _error_detail("fence", error))
        renew_task = asyncio.create_task(self._renew(fence))
        try:
            detail = await self._purge_fenced(command, guild_id, fence, stores)
        except Exception as error:
            logger.warning(
                "privacy purge guild failed run=%s guild=%s type=%s",
                command.run_id,
                guild_id,
                type(error).__name__,
            )
            return GuildOutcome("failed", stores, _error_detail("purge", error))
        finally:
            renew_task.cancel()
            await asyncio.gather(renew_task, return_exceptions=True)
            try:
                await fence.holder.release()
            except Exception as error:
                logger.warning(
                    "privacy purge fence release failed guild=%s type=%s",
                    guild_id,
                    type(error).__name__,
                )
        return GuildOutcome("purged" if stores else "unchanged", stores, detail)

    async def _purge_fenced(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
    ) -> str:
        details: list[str] = []
        taken = self._writer.discard(guild_id)
        try:
            history_detail = await self._purge_history(command, guild_id, fence, stores)
        except BaseException:
            self._writer.restore(taken)
            raise
        details.append(history_detail)
        # Always bump, even with nothing to fold: the epoch is also what stops
        # the legacy embedded-bot key from ever being restored for the guild,
        # and it makes every replica drop its in-RAM copy.
        await self._redis.incr(purge_epoch_key(guild_id))
        self._runtimes.forget(guild_id)
        if fence.external:
            details.append(await self._purge_watch(command, guild_id, fence, stores))
        return "; ".join(item for item in details if item)[:500]

    async def _purge_history(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
    ) -> str:
        snapshot = await self._repository.load_canonical(guild_id)
        if snapshot is None or not snapshot.history:
            return "history empty"
        history: list[ModelMessage] = list(
            ModelMessagesTypeAdapter.validate_json(json.dumps(snapshot.history))
        )
        notes: list[PrivacyNote] = []

        async def summarize(messages: list[ModelMessage]) -> str:
            note = await self._compact(
                messages, user_id=command.user_id, names=list(command.names)
            )
            notes.append(note)
            return note.text

        purged = await purge_agent_history(history, summarize=summarize)
        serialized = json.loads(ModelMessagesTypeAdapter.dump_json(purged))
        # Belt and braces over the note validator: nothing written may hold
        # the id, whichever summarizer produced it.
        if command.user_id.encode() in canonical_history(serialized):
            raise PurgeFailed("purged history still held the user id")
        if fence.lost:
            raise PurgeFailed("fence lost before the history write")
        await self._writer.replace_purged(
            guild_id, serialized, previous_revision=snapshot.revision
        )
        stores.extend([STORE_POSTGRES_HISTORY, STORE_REDIS_HISTORY])
        note = notes[-1] if notes else None
        if note is None:
            return "history folded"
        detail = f"history folded in {note.attempts} model attempt(s)"
        if note.name_hits:
            detail += f"; {note.name_hits} name match(es) kept after retry"
        return detail

    async def _purge_watch(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
    ) -> str:
        rows = await self._api.list_enabled_channels(guild_id)
        changed = 0
        for row in rows:
            store = InstructionStore.from_stored(
                OPERATING_POLICY_BRIEF, row.watch_addendum
            )
            entries = {entry.instruction_id: entry.text for entry in store.entries}
            if not entries:
                continue
            decided = await self._decide_watch(
                entries, user_id=command.user_id, names=list(command.names)
            )
            # An entry without a decision is kept as it is.
            kept = [
                dataclass_replace(entry, text=text)
                for entry in store.entries
                if (text := decided.get(entry.instruction_id, entry.text)) is not None
            ]
            if kept == store.entries:
                continue
            store.entries = kept
            stored = store.to_stored()
            if fence.lost:
                raise PurgeFailed("fence lost before the watch instruction write")
            await self._api.set_watch_addendum(
                guild_id=guild_id,
                channel_id=row.channel_id,
                enabled=True,
                watch_addendum=stored,
            )
            changed += 1
        if changed:
            stores.append(STORE_WATCH)
            return f"watch instructions rewritten in {changed} channel(s)"
        return ""

    async def _acquire_fence(self, guild_id: str) -> Fence:
        deadline = time.monotonic() + self._fence_wait_seconds
        while True:
            if await self._queue.externally_owned(guild_id):
                lease = await self._queue.acquire_lease(guild_id)
                if lease is not None:
                    return Fence(lease, external=True)
            else:
                token = uuid4().hex
                if await self._redis.set(
                    privacy_lock_key(guild_id), token, nx=True, ex=PRIVACY_LOCK_SECONDS
                ):
                    lock = PrivacyLock(
                        self._redis,
                        guild_id=guild_id,
                        token=token,
                        ttl_seconds=PRIVACY_LOCK_SECONDS,
                    )
                    # Ownership may just have moved to the bot while a worker
                    # wake still runs; no new lease can start while the lock
                    # is held, so wait out the one in flight.
                    while await self._redis.exists(lease_key(guild_id)):
                        if time.monotonic() >= deadline:
                            await lock.release()
                            raise PurgeFailed("a running wake kept the guild lease")
                        await asyncio.sleep(self._fence_poll_seconds)
                    return Fence(lock, external=False)
            if time.monotonic() >= deadline:
                raise PurgeFailed("guild could not be fenced in time")
            await asyncio.sleep(self._fence_poll_seconds)

    async def _renew(self, fence: Fence) -> None:
        holder = fence.holder
        while True:
            await asyncio.sleep(min(5, max(1, holder.ttl_seconds / 3)))
            if not await holder.renew():
                fence.lost = True
                return
