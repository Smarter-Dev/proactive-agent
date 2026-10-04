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
import re
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
    name_hits,
    privacy_compaction_summary,
    privacy_watch_decisions,
    purge_agent_history,
)
from proactive_agent.contracts import PurgeCommand
from proactive_agent.environment import InstructionStore
from proactive_agent.history import PartialPurgeError, canonical_history
from proactive_agent.keys import (
    PRIVACY_PURGE_STREAM_KEY,
    batch_key_pattern,
    lease_key,
    pending_dropped_key,
    pending_key,
    privacy_consumer_key,
    privacy_lock_key,
    purge_deliveries_key,
    purge_done_key,
    purge_epoch_key,
    wake_stream_key,
)
from proactive_agent.queue import WAKE_GROUP

logger = logging.getLogger(__name__)

PURGE_GROUP = "proactive-agent-workers-v1-privacy"
PURGE_COMPONENT = "worker"
RECLAIM_IDLE_MS = 10 * 60 * 1000
# Re-claim the entry being worked on this often, well inside RECLAIM_IDLE_MS,
# so a long multi-guild purge is never reclaimed by another replica.
HEARTBEAT_SECONDS = 60
PRIVACY_LOCK_SECONDS = 600
# A command delivered more often than this acks its unfinished guilds
# "failed" and is acknowledged, so an outage cannot loop it forever.
MAX_DELIVERIES = 5
DONE_TTL_SECONDS = 7 * 24 * 60 * 60
# Finished guilds are remembered per run, so a redelivery of the same run
# never folds a guild twice. A new run of the request inspects every guild
# again (rule (b) in history_needs_no_fold decides whether to fold).
PURGE_DONE_TTL_SECONDS = 30 * 24 * 60 * 60
LIST_WAIT_SECONDS = 5 * 60
CONSUMER_ALIVE_SECONDS = 180
LIST_POLL_SECONDS = 5
FENCE_WAIT_SECONDS = 15 * 60
FENCE_POLL_SECONDS = 0.5

STORE_REDIS_HISTORY = "proactive:v1:history"
STORE_POSTGRES_HISTORY = "proactive_agent_histories"
STORE_WATCH = "watch_instructions"
STORE_NOTIFICATIONS = "proactive:v1:notifications"
STORE_LEGACY_HISTORY = "proactive:guild-history"

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
WatchFn = Callable[..., Awaitable[tuple[dict[str, str | None], int]]]


# A transcript line for one member's Discord message, as rendered by
# render_transcript_line (or the skim model's `[id=…] name (user id …)`).
_MEMBER_LINE = re.compile(r"^\[id=[0-9]+\] ")


def _raw_parts(history: list[dict]):
    """Text of every part that came from outside the model, by part."""
    for message in history:
        for part in message.get("parts", ()):
            if part.get("part_kind") not in ("user-prompt", "tool-return"):
                continue
            content = part.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            if text.startswith("[COMPACTION MEMORY NOTE"):
                continue  # the agent's own note, written about named people
            yield text


def history_needs_no_fold(history: list[dict], user_id: str, names: list[str]) -> bool:
    """True only if folding could not remove anything about the user.

    - Nothing anywhere may hold the user id or a checked name. That search
      covers host-written text too: wake briefs, notification bodies and
      tool output that is not a member's message.
    - Every member-message line (`[id=…] …`) must carry its author's id
      (`(uid=…)`, or `(user id …)` in skim output). A member line without
      it predates author ids and could name the user under a nickname
      nobody listed, so the history is folded.
    """
    serialized = canonical_history(history).decode()
    if user_id in serialized or name_hits(serialized, names):
        return False
    for text in _raw_parts(history):
        for line in text.splitlines():
            line = line.strip()
            if (
                _MEMBER_LINE.match(line)
                and "(uid=" not in line
                and "(user id " not in line
            ):
                return False
    return True


class PurgeFailed(Exception):
    """A guild purge failed; the message is a content-free ack detail."""


@dataclass
class GuildOutcome:
    outcome: str
    stores: list[str] = field(default_factory=list)
    detail: str = ""
    # Set when Postgres was purged but v1 could not be replaced: the entry
    # is kept pending so the purge is retried, and no done record is kept.
    retry: bool = False


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


def _stream_id_key(stream_id: str) -> tuple[int, int]:
    milliseconds, _, sequence = stream_id.partition("-")
    return int(milliseconds), int(sequence or 0)


def _join(detail: str, extra: str) -> str:
    return ("; ".join(item for item in (detail, extra) if item))[:500]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _error_detail(prefix: str, error: BaseException) -> str:
    """Content-free description: the exception type and HTTP status only."""
    if isinstance(error, PurgeFailed | PrivacyCompactionError | PartialPurgeError):
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
        heartbeat_seconds: float = HEARTBEAT_SECONDS,
        blocked_users=None,
        max_deliveries: int = MAX_DELIVERIES,
        list_wait_seconds: float = LIST_WAIT_SECONDS,
        list_poll_seconds: float = LIST_POLL_SECONDS,
        replica_id: str | None = None,
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
        self._heartbeat_seconds = heartbeat_seconds
        self._blocked_users = blocked_users
        self._max_deliveries = max_deliveries
        self._list_wait_seconds = list_wait_seconds
        self._list_poll_seconds = list_poll_seconds
        self._replica_id = replica_id or consumer_name
        self._error_backoff_seconds = 5.0

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
                # Alive marker for the web admin, idle polls included.
                await self._mark_alive()
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
                    await asyncio.wait_for(
                        stop.wait(), timeout=self._error_backoff_seconds
                    )
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
            # One at a time: only the entry in hand is heartbeated, so
            # nothing else may sit claimed and idle behind it.
            count=1,
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
            stream_id = _decode(stream_id)
            heartbeat = asyncio.create_task(self._heartbeat(stream_id))
            try:
                await self.handle(stream_id, fields)
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
        return len(entries)

    async def _mark_alive(self) -> None:
        await self._redis.set(
            privacy_consumer_key(PURGE_COMPONENT, self._replica_id),
            "1",
            ex=CONSUMER_ALIVE_SECONDS,
        )

    async def _heartbeat(self, stream_id: str) -> None:
        """Keep the owned entry's idle time near zero while it is handled,
        and the consumer's alive marker fresh during a long purge."""
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                await self._mark_alive()
                await self._redis.xclaim(
                    PRIVACY_PURGE_STREAM_KEY,
                    PURGE_GROUP,
                    self._consumer_name,
                    0,
                    [stream_id],
                    justid=True,
                )
            except Exception as error:
                logger.warning(
                    "privacy purge heartbeat failed id=%s type=%s",
                    stream_id,
                    type(error).__name__,
                )

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
            await self._drop(stream_id)
            return
        run_id = str(command.run_id)
        guild_ids = list(dict.fromkeys(command.guild_ids))
        deliveries = await self._count_delivery(run_id)
        logger.info(
            "privacy purge started run=%s request=%s guilds=%d delivery=%d",
            run_id,
            command.request_id,
            len(guild_ids),
            deliveries,
        )
        done = await self._done(run_id)
        if deliveries > self._max_deliveries:
            await self._give_up(stream_id, command, guild_ids, done)
            return
        if not await self._enforcing_target(command):
            # Leave the entry pending: it is reclaimed and retried, and the
            # delivery cap ends it if the list never catches up.
            logger.warning(
                "privacy purge waiting for the blocked users list run=%s", run_id
            )
            return
        all_acked = True
        for guild_id in guild_ids:
            if guild_id in done:
                # Finished on an earlier delivery of this run: never fold
                # again, just re-post the ack.
                result = done[guild_id]
            else:
                result = await self.purge_guild(command, guild_id)
                if result.retry:
                    # Forced retry: keep the entry pending, no done record.
                    all_acked = False
                elif result.outcome != "failed" and not await self._record_done(
                    run_id, guild_id, result
                ):
                    result.detail = _join(result.detail, "done_record=unsaved")
            logger.info(
                "privacy purge guild finished run=%s guild=%s outcome=%s",
                run_id,
                guild_id,
                result.outcome,
            )
            accepted = await self._post_ack(run_id, guild_id, result)
            if accepted is None:
                all_acked = False
                continue
            if not accepted:
                logger.warning("privacy purge run unknown, dropped run=%s", run_id)
                await self._drop(stream_id, unknown_run=True)
                return
        if all_acked:
            await self._ack(stream_id)
            logger.info("privacy purge acknowledged run=%s", run_id)
        # Otherwise the entry stays pending and is reclaimed after the idle
        # timeout; finished guilds are skipped on the next delivery.

    async def _post_ack(self, run_id: str, guild_id: str, result) -> bool | None:
        """True accepted, False unknown run (404), None not delivered."""
        try:
            return await self._api.post_privacy_ack(
                run_id,
                component=PURGE_COMPONENT,
                guild_id=guild_id,
                outcome=result.outcome,
                stores=result.stores,
                detail=result.detail,
            )
        except Exception as error:
            logger.error(
                "privacy purge ack failed run=%s guild=%s type=%s",
                run_id,
                guild_id,
                type(error).__name__,
            )
            return None

    async def _count_delivery(self, run_id: str) -> int:
        key = purge_deliveries_key(run_id)
        count = int(await self._redis.incr(key))
        await self._redis.expire(key, DONE_TTL_SECONDS)
        return count

    async def _give_up(self, stream_id, command, guild_ids, done) -> None:
        """Delivery limit reached: report what we can, then drop the entry.

        The entry is XDELed even when the ack endpoint is down or the run
        is unknown, so it never stays stranded holding the id and names.
        What is left is content-free: this log line and a "failed" done
        record per unfinished guild.
        """
        run_id = str(command.run_id)
        logger.error(
            "privacy purge delivery limit reached run=%s limit=%d",
            run_id,
            self._max_deliveries,
        )
        unknown_run = False
        for guild_id in guild_ids:
            result = done.get(guild_id)
            if result is None:
                result = GuildOutcome("failed", [], "delivery limit reached")
                await self._record_done(run_id, guild_id, result)
            if await self._post_ack(run_id, guild_id, result) is False:
                unknown_run = True
                break
        await self._drop(stream_id, unknown_run=unknown_run)

    async def _done(self, run_id: str) -> dict[str, GuildOutcome]:
        raw = await self._redis.hgetall(purge_done_key(PURGE_COMPONENT, run_id))
        done = {}
        for guild_id, value in raw.items():
            try:
                data = json.loads(value)
                done[_decode(guild_id)] = GuildOutcome(
                    data["outcome"], list(data["stores"]), data["detail"]
                )
            except (ValueError, KeyError, TypeError):
                continue
        return done

    async def _record_done(
        self, run_id: str, guild_id: str, result: GuildOutcome
    ) -> bool:
        """Record a finished guild (tried twice). False if it was not saved."""
        key = purge_done_key(PURGE_COMPONENT, run_id)
        value = json.dumps(
            {
                "outcome": result.outcome,
                "stores": result.stores,
                "detail": result.detail,
            }
        )
        for _attempt in range(2):
            try:
                await self._redis.hset(key, guild_id, value)
                await self._redis.expire(key, PURGE_DONE_TTL_SECONDS)
                return True
            except Exception as error:
                logger.warning(
                    "privacy purge done record failed guild=%s type=%s",
                    guild_id,
                    type(error).__name__,
                )
        return False

    async def _enforcing_target(self, command: PurgeCommand) -> bool:
        """Wait until THIS process's own list blocks the target.

        Otherwise a wake on this replica could re-ingest the user's messages
        right after their purge.
        """
        if self._blocked_users is None:
            return True
        deadline = time.monotonic() + self._list_wait_seconds
        while True:
            if self._blocked_users.enforcing and self._blocked_users.is_blocked(
                command.user_id
            ):
                return True
            await self._blocked_users.refresh()
            if self._blocked_users.enforcing and self._blocked_users.is_blocked(
                command.user_id
            ):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._list_poll_seconds)

    async def _ack(self, stream_id: str) -> None:
        # XACK only: the producer XDELs the entry (it carries the target)
        # once every component acked every guild.
        await self._redis.xack(PRIVACY_PURGE_STREAM_KEY, PURGE_GROUP, stream_id)

    async def _drop(self, stream_id: str, *, unknown_run: bool = False) -> None:
        """Finish with an entry that will not be retried (malformed, unknown
        run, delivery limit); it may still carry the id and names.

        The entry is XDELed only for a run the server answered 404 for, or
        once every other consumer group (the bot's) has read and acked it;
        otherwise only this group acks it and the web deletes it when the
        run ends or closes. When it is deleted, XDEL goes first: if that
        fails the entry stays pending and is retried, never left acked but
        undeleted by mistake.
        """
        if unknown_run or await self._others_done_with(stream_id):
            await self._redis.xdel(PRIVACY_PURGE_STREAM_KEY, stream_id)
        await self._ack(stream_id)

    async def _others_done_with(self, stream_id: str) -> bool:
        """Every other group on the stream has delivered and acked the entry."""
        entry = _stream_id_key(stream_id)
        for group in await self._redis.xinfo_groups(PRIVACY_PURGE_STREAM_KEY):
            name = _decode(group["name"])
            if name == PURGE_GROUP:
                continue
            if _stream_id_key(_decode(group["last-delivered-id"])) < entry:
                return False
            pending = await self._redis.xpending_range(
                PRIVACY_PURGE_STREAM_KEY, name, min=stream_id, max=stream_id, count=1
            )
            if pending:
                return False
        return True

    async def purge_guild(self, command: PurgeCommand, guild_id: str) -> GuildOutcome:
        """Purge one guild. Never raises; failures come back as an outcome."""
        stores: list[str] = []
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
            return GuildOutcome(
                "failed",
                stores,
                _error_detail("purge", error),
                retry=isinstance(error, PartialPurgeError),
            )
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
        dropped = await self._discard_notifications(guild_id)
        if dropped:
            stores.append(STORE_NOTIFICATIONS)
            details.append(f"notifications_dropped={dropped}")
        taken = self._writer.discard(guild_id)
        try:
            history_detail, wrote = await self._purge_history(
                command, guild_id, fence, stores
            )
        except PartialPurgeError:
            # Postgres already holds the purged copy: make every replica
            # reload (best effort), then fail with the exact store state.
            stores.append(STORE_POSTGRES_HISTORY)
            await self._bump_epoch(guild_id, best_effort=True)
            raise
        except BaseException:
            self._writer.restore(taken)
            raise
        if not wrote:
            self._writer.restore(taken)
        details.append(history_detail)
        if wrote:
            # Every replica drops its in-RAM copy, and the legacy key can
            # never be restored for the guild again.
            await self._bump_epoch(guild_id)
        if fence.external:
            details.append(await self._purge_watch(command, guild_id, fence, stores))
        return "; ".join(item for item in details if item)[:500]

    async def _bump_epoch(self, guild_id: str, *, best_effort: bool = False) -> None:
        try:
            await self._redis.incr(purge_epoch_key(guild_id))
        except Exception:
            if not best_effort:
                raise
        self._runtimes.forget(guild_id)

    async def _discard_notifications(self, guild_id: str) -> int:
        """Discard every notification queued for the guild before the purge.

        Wake stream entries, the pending list and claimed batch lists are
        raw copies of channel activity made before the user was blocked,
        and pending triggers rather than memory, so all of them go, without
        trying to recognise the user by name. Anything queued after this
        point was produced under the block list. The dead-letter stream is
        not touched.
        """
        dropped = 0
        stream = wake_stream_key(guild_id)
        entry_ids = [
            _decode(entry_id) for entry_id, _fields in await self._redis.xrange(stream)
        ]
        if entry_ids:
            try:
                await self._redis.xack(stream, WAKE_GROUP, *entry_ids)
            except ResponseError:
                pass  # no consumer group yet: nothing was delivered
            await self._redis.xdel(stream, *entry_ids)
            dropped += len(entry_ids)
        keys = [pending_key(guild_id), pending_dropped_key(guild_id)]
        async for key in self._redis.scan_iter(match=batch_key_pattern(guild_id)):
            keys.append(_decode(key))
        for key in keys:
            if not key.endswith("dropped"):
                dropped += int(await self._redis.llen(key))
            await self._redis.delete(key)
        return dropped

    async def _purge_history(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
    ) -> tuple[str, bool]:
        """Returns (detail, whether the stored history was rewritten)."""
        names = list(command.names)
        tombstoned = await self._repository.is_invalid(guild_id)
        legacy = await self._repository.load_legacy(guild_id)
        snapshot = await self._repository.load_canonical(guild_id)
        migrated = False
        if (
            snapshot is None
            and legacy is not None
            and not await self._redis.exists(purge_epoch_key(guild_id))
        ):
            # Only the pre-split legacy key holds this guild's memory. Bumping
            # the epoch without carrying it over would reset the agent's
            # memory, so it is migrated (folded first if needed).
            snapshot = legacy
            migrated = True
        # The worker owns the legacy key of an external guild: it is deleted
        # once v1 and Postgres both hold the purged history.
        drop_legacy = fence.external and legacy is not None
        if snapshot is None or not snapshot.history:
            if drop_legacy:
                await self._repository.forget_legacy(guild_id)
                stores.append(STORE_LEGACY_HISTORY)
                return "history empty; legacy history deleted", False
            return "history empty", False
        clean = history_needs_no_fold(snapshot.history, command.user_id, names)
        # A clean history is still written when the write itself is needed:
        # to migrate legacy, to clear a tombstone, or before deleting legacy.
        if clean and not (migrated or tombstoned or drop_legacy):
            return "history already attributed and clean", False
        notes: list[PrivacyNote] = []

        async def summarize(messages: list[ModelMessage]) -> str:
            note = await self._compact(messages, user_id=command.user_id, names=names)
            notes.append(note)
            return note.text

        if clean:
            serialized = snapshot.history
        else:
            history: list[ModelMessage] = list(
                ModelMessagesTypeAdapter.validate_json(json.dumps(snapshot.history))
            )
            purged = await purge_agent_history(history, summarize=summarize)
            serialized = json.loads(ModelMessagesTypeAdapter.dump_json(purged))
        # Belt and braces over the note validator: nothing written may hold
        # the id, whichever summarizer produced it.
        if command.user_id.encode() in canonical_history(serialized):
            raise PurgeFailed("purged history still held the user id")
        if fence.lost:
            raise PurgeFailed("fence lost before the history write")
        _snapshot, v1_written = await self._writer.replace_purged(
            guild_id, serialized, previous_revision=snapshot.revision
        )
        stores.extend([STORE_POSTGRES_HISTORY, STORE_REDIS_HISTORY])
        details = []
        if migrated:
            details.append("legacy history migrated")
        note = notes[-1] if notes else None
        if note is None:
            details.append("history kept")
        else:
            details.append(
                f"history folded attempts={note.attempts} "
                f"history_name_hits={note.name_hits}"
            )
        if drop_legacy and v1_written:
            await self._repository.forget_legacy(guild_id)
            stores.append(STORE_LEGACY_HISTORY)
            details.append("legacy history deleted")
        return "; ".join(details), True

    async def _purge_watch(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
    ) -> str:
        rows = await self._api.list_enabled_channels(guild_id)
        changed = 0
        name_hits_kept = 0
        for row in rows:
            store = InstructionStore.from_stored(
                OPERATING_POLICY_BRIEF, row.watch_addendum
            )
            entries = {entry.instruction_id: entry.text for entry in store.entries}
            if not entries:
                continue
            decided, hits = await self._decide_watch(
                entries, user_id=command.user_id, names=list(command.names)
            )
            name_hits_kept += hits
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
        if changed or name_hits_kept:
            return (
                f"watch channels_rewritten={changed} watch_name_hits={name_hits_kept}"
            )
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
