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
    MEMBER_LINE_PREFIX,
    OPERATING_POLICY_BRIEF,
    PrivacyCompactionError,
    PrivacyNote,
    name_hits,
    privacy_compaction_summary,
    privacy_watch_decisions,
    purge_agent_history,
    string_leaves,
    unchecked_names,
)
from proactive_agent.contracts import PurgeCommand
from proactive_agent.environment import InstructionStore
from proactive_agent.history import PartialPurgeError
from proactive_agent.keys import (
    DEAD_LETTER_STREAM_KEY,
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
    tombstone_cleared_key,
    wake_stream_key,
)
from proactive_agent.queue import WAKE_GROUP

logger = logging.getLogger(__name__)

PURGE_GROUP = "proactive-agent-workers-v1-privacy"
PURGE_COMPONENT = "worker"
# Every consumer group of privacy:v1:purge; an entry is deleted only once all
# of them are done with it (unless its run is unknown to the server).
EXPECTED_GROUPS = ("smarter-dev-bot", PURGE_GROUP)
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
STORE_DEAD_LETTERS = "proactive:v1:dead-letter"

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


# The skim model's verbatim line shape: `[id=<msg>] <name> (user id <author>): `.
_SKIM_LINE_PREFIX = re.compile(
    r"^\[id=[0-9]+\] [^\n]{1,200}? \(user id [0-9]{1,22}\): "
)


def history_needs_no_fold(history: list[dict], user_id: str, names: list[str]) -> bool:
    """True only if folding could not remove anything about the user.

    - Nothing anywhere may hold the user id or a checked name. That search
      covers host-written text too: wake briefs, notification bodies and
      tool output that is not a member's message.
    - Every member-message line (starting `[id=`) must carry its author's id
      exactly where render_transcript_line writes it (MEMBER_LINE_PREFIX),
      or in the skim model's `(user id …): ` prefix. `(uid=` anywhere else
      on the line (say, in the message text) does not count. Such a line
      without it predates author ids and could name the user under a
      nickname nobody listed, so the history is folded.
    - Member lines inside structured (non-string) tool output cannot be
      checked line by line and count as unattributed.
    """
    # Searched over the decoded strings, never the JSON text: escaping would
    # hide a name holding a quote, backslash, tab or newline.
    for leaf in string_leaves(history):
        if user_id in leaf or name_hits(leaf, names):
            return False
    for message in history:
        for part in message.get("parts", ()):
            if part.get("part_kind") not in ("user-prompt", "tool-return"):
                continue
            content = part.get("content")
            if not isinstance(content, str):
                if any("[id=" in leaf for leaf in string_leaves(content)):
                    return False
                continue
            if content.startswith("[COMPACTION MEMORY NOTE"):
                continue  # the agent's own note, written about named people
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("[id=") and not (
                    MEMBER_LINE_PREFIX.match(line) or _SKIM_LINE_PREFIX.match(line)
                ):
                    return False
    return True


class PurgeFailed(Exception):
    """A guild purge failed; the message is a content-free ack detail."""


class PurgeRetry(PurgeFailed):
    """Failed after Postgres was already purged: keep the entry pending."""


@dataclass
class GuildOutcome:
    outcome: str
    stores: list[str] = field(default_factory=list)
    detail: str = ""
    # Structured ack fields (Ack v1); the detail string is display only.
    name_hits: dict[str, int] = field(default_factory=dict)
    tombstoned: bool = False
    unchecked_names: int = 0
    done_record: str = "not_written"
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


# Detail segments the web parses; they go first and are never truncated.
_CRITICAL_SEGMENTS = (
    "tombstoned=",
    "unchecked_names=",
    "done_record=",
    "history_name_hits=",
    "watch_name_hits=",
)
DETAIL_LIMIT = 500


def _join(*parts: str) -> str:
    """`; `-joined ack detail, critical segments first, at most 500 chars.

    Only the non-critical tail is ever cut.
    """
    segments = [
        segment.strip()
        for part in parts
        if part
        for segment in part.split("; ")
        if segment.strip()
    ]
    critical = list(
        dict.fromkeys(s for s in segments if s.startswith(_CRITICAL_SEGMENTS))
    )
    rest = [s for s in segments if not s.startswith(_CRITICAL_SEGMENTS)]
    detail = "; ".join(critical)
    for segment in rest:
        candidate = f"{detail}; {segment}" if detail else segment
        if len(candidate) > DETAIL_LIMIT:
            room = DETAIL_LIMIT - len(detail) - (2 if detail else 0)
            if room > 0:
                detail = (f"{detail}; " if detail else "") + segment[:room]
            break
        detail = candidate
    return detail


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _error_detail(prefix: str, error: BaseException) -> str:
    """Content-free description: the exception type and HTTP status only."""
    if isinstance(error, PurgeFailed | PrivacyCompactionError | PartialPurgeError):
        return f"{prefix}: {error}"[:DETAIL_LIMIT]
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
                if not initialized:
                    await self.initialize()
                    initialized = True
                await self.poll_once(block_ms=5_000)
                # Alive marker for the web admin, written only after a
                # successful poll (idle polls included).
                await self._mark_alive()
                # Let other tasks run even if the server answered at once.
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except ResponseError as error:
                if "NOGROUP" not in str(error):
                    logger.error(
                        "privacy purge consumer error type=%s", type(error).__name__
                    )
                else:
                    # The stream or group vanished: recreate it (id 0,
                    # MKSTREAM) before reporting alive again.
                    logger.warning("privacy purge consumer group missing, recreating")
                    initialized = False
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=self._error_backoff_seconds
                    )
                except TimeoutError:
                    pass
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
        deliveries = await self._count_delivery(stream_id)
        logger.info(
            "privacy purge started run=%s request=%s guilds=%d delivery=%d",
            run_id,
            command.request_id,
            len(guild_ids),
            deliveries,
        )
        done = await self._done(run_id)
        # Tombstones first, independent of the run and of the block list: the
        # Postgres copy of a tombstoned guild is already the purged one.
        await self._recover_tombstones(guild_ids)
        tombstoned = await self._tombstoned(guild_ids)
        if deliveries > self._max_deliveries and not tombstoned:
            await self._give_up(stream_id, command, guild_ids, done)
            return
        # Past the cap with a guild still tombstoned: never given up, never
        # XACKed; retried on every delivery, i.e. every reclaim interval
        # (10 min), which is also the backoff cap.
        if not await self._enforcing_target(command):
            logger.warning(
                "privacy purge waiting for the blocked users list run=%s", run_id
            )
            # A guild that stays tombstoned is reported failed, visibly.
            for guild_id in tombstoned:
                accepted = await self._post_ack(
                    run_id,
                    guild_id,
                    GuildOutcome(
                        "failed",
                        [],
                        _join("tombstoned=1", "postgres copy not the purged revision"),
                        tombstoned=True,
                        unchecked_names=unchecked_names(list(command.names)),
                    ),
                )
                if accepted is False:
                    await self._drop(stream_id, unknown_run=True)
                    return
            # Otherwise the entry stays pending and is retried; the delivery
            # cap ends it if the list never catches up.
            return
        all_acked = True
        for guild_id in guild_ids:
            if guild_id in done:
                # Finished on an earlier delivery of this run: never fold
                # again, just re-post the ack.
                result = done[guild_id]
                result.done_record = "replayed"
            else:
                result = await self.purge_guild(command, guild_id)
                if result.retry:
                    # Forced retry: keep the entry pending, no done record.
                    all_acked = False
                elif result.outcome != "failed":
                    if await self._record_done(run_id, guild_id, result):
                        result.done_record = "written"
                    else:
                        result.done_record = "not_written"
                        result.detail = _join("done_record=unsaved", result.detail)
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
                # The run is gone, the purged Postgres copies are not: recover
                # any tombstone before forgetting the command.
                await self._recover_tombstones(guild_ids)
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
                name_hits=dict(result.name_hits),
                tombstoned=result.tombstoned,
                unchecked_names=result.unchecked_names,
                done_record=result.done_record,
            )
        except Exception as error:
            logger.error(
                "privacy purge ack failed run=%s guild=%s type=%s",
                run_id,
                guild_id,
                type(error).__name__,
            )
            return None

    async def _tombstoned(self, guild_ids: list[str]) -> list[str]:
        found = []
        for guild_id in guild_ids:
            try:
                if await self._repository.is_invalid(guild_id):
                    found.append(guild_id)
            except Exception:
                found.append(guild_id)  # cannot tell: never give up on it
        return found

    async def _recover_tombstones(self, guild_ids: list[str]) -> None:
        """Restore v1 from the purged Postgres copy wherever a tombstone is
        found, under the guild's fence. Failures leave it tombstoned."""
        for guild_id in await self._tombstoned(guild_ids):
            try:
                await self._recover_tombstone(guild_id)
            except Exception as error:
                logger.warning(
                    "privacy purge tombstone recovery failed guild=%s type=%s",
                    guild_id,
                    type(error).__name__,
                )

    async def _recover_tombstone(self, guild_id: str) -> bool:
        fence = await self._acquire_fence(guild_id)
        try:
            old = await self._repository.tombstone(guild_id)
            if old is None:
                return True
            self._writer.discard(guild_id)
            if not await self._repository.recover_tombstone(guild_id):
                return False
            self._runtimes.forget(guild_id)
            await self._note_tombstone_cleared(old, guild_id, "recovered from postgres")
            logger.info("privacy purge tombstone recovered guild=%s", guild_id)
            return True
        finally:
            await fence.holder.release()

    async def _note_tombstone_cleared(self, old: dict, guild_id: str, how: str) -> None:
        """Remember, for the run that left a tombstone, how it was cleared."""
        run_id = old.get("run_id") if isinstance(old, dict) else None
        if not run_id:
            return
        key = tombstone_cleared_key(str(run_id))
        try:
            await self._redis.hset(key, guild_id, how)
            await self._redis.expire(key, PURGE_DONE_TTL_SECONDS)
        except Exception as error:
            logger.warning(
                "privacy purge tombstone note failed guild=%s type=%s",
                guild_id,
                type(error).__name__,
            )

    async def _count_delivery(self, stream_id: str) -> int:
        """Deliveries of this stream entry (a re-published entry starts at 1)."""
        key = purge_deliveries_key(stream_id)
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
        unrecorded = False
        for guild_id in guild_ids:
            result = done.get(guild_id)
            saved = True
            if result is None:
                result = GuildOutcome(
                    "failed",
                    [],
                    "delivery limit reached",
                    tombstoned=guild_id in await self._tombstoned([guild_id]),
                    unchecked_names=unchecked_names(list(command.names)),
                )
                saved = await self._record_done(run_id, guild_id, result)
                result.done_record = "written" if saved else "not_written"
            else:
                result.done_record = "replayed"
            accepted = await self._post_ack(run_id, guild_id, result)
            if accepted is False:
                unknown_run = True
                break
            if accepted is None and not saved:
                unrecorded = True
        if unrecorded and not unknown_run:
            # Neither an ack nor a done record exists for some guild: the
            # failure would be invisible, so keep the entry pending.
            logger.error("privacy purge give-up left pending run=%s", run_id)
            return
        await self._drop(stream_id, unknown_run=unknown_run)

    async def _done(self, run_id: str) -> dict[str, GuildOutcome]:
        raw = await self._redis.hgetall(purge_done_key(PURGE_COMPONENT, run_id))
        done = {}
        for guild_id, value in raw.items():
            try:
                data = json.loads(value)
                done[_decode(guild_id)] = GuildOutcome(
                    data["outcome"],
                    list(data["stores"]),
                    data["detail"],
                    name_hits={
                        str(k): int(v) for k, v in data.get("name_hits", {}).items()
                    },
                    tombstoned=bool(data.get("tombstoned", False)),
                    unchecked_names=int(data.get("unchecked_names", 0)),
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
                "name_hits": result.name_hits,
                "tombstoned": result.tombstoned,
                "unchecked_names": result.unchecked_names,
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
        """Every other expected group has delivered and acked the entry.

        An expected group that does not exist is not done. A group counts as
        having read the entry only if its last-delivered-id is at or past it,
        it reports entries-read, it has at least one consumer, and the entry
        is not in its PEL. Groups are always created at id 0; a group created
        at `$` after the entry is indistinguishable from one that read it, so
        a group with no consumers or no entries-read counts as not done.
        """
        entry = _stream_id_key(stream_id)
        groups = {
            _decode(group["name"]): group
            for group in await self._redis.xinfo_groups(PRIVACY_PURGE_STREAM_KEY)
        }
        for name in EXPECTED_GROUPS:
            if name == PURGE_GROUP:
                continue
            group = groups.get(name)
            if group is None:
                return False
            if _stream_id_key(_decode(group["last-delivered-id"])) < entry:
                return False
            if group.get("entries-read") is None or not group.get("consumers"):
                return False
            pending = await self._redis.xpending_range(
                PRIVACY_PURGE_STREAM_KEY, name, min=stream_id, max=stream_id, count=1
            )
            if pending:
                return False
        return True

    async def purge_guild(self, command: PurgeCommand, guild_id: str) -> GuildOutcome:
        """Purge one guild. Never raises; failures come back as an outcome."""
        result = await self._purge_guild(command, guild_id)
        if result.outcome == "unchanged":
            try:
                how = await self._redis.hget(
                    tombstone_cleared_key(str(command.run_id)), guild_id
                )
            except Exception:
                how = None
            if how is not None:
                cleared = _decode(how)
                label = (
                    "purged by a later run"
                    if cleared == "later run"
                    else "recovered from postgres"
                )
                result.detail = _join(label, result.detail)
        unchecked = unchecked_names(list(command.names))
        result.unchecked_names = unchecked
        if unchecked:
            # Names too short to match deterministically: the model saw
            # them, the checks could not; the web shows the count.
            result.detail = _join(f"unchecked_names={unchecked}", result.detail)
        return result

    async def _purge_guild(self, command: PurgeCommand, guild_id: str) -> GuildOutcome:
        stores: list[str] = []
        hits: dict[str, int] = {}
        try:
            fence = await self._acquire_fence(guild_id)
        except Exception as error:
            return GuildOutcome("failed", stores, _error_detail("fence", error))
        renew_task = asyncio.create_task(self._renew(fence))
        try:
            detail = await self._purge_fenced(command, guild_id, fence, stores, hits)
        except Exception as error:
            logger.warning(
                "privacy purge guild failed run=%s guild=%s type=%s",
                command.run_id,
                guild_id,
                type(error).__name__,
            )
            retry = isinstance(error, PartialPurgeError | PurgeRetry)
            detail = _error_detail("purge", error)
            try:
                tombstoned = await self._repository.is_invalid(guild_id)
            except Exception:
                tombstoned = isinstance(error, PartialPurgeError)
            if tombstoned:
                retry = True
                detail = _join("tombstoned=1", detail)
            return GuildOutcome(
                "failed",
                stores,
                detail,
                name_hits=hits,
                tombstoned=tombstoned,
                retry=retry,
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
        return GuildOutcome(
            "purged" if stores else "unchanged", stores, detail, name_hits=hits
        )

    async def _purge_fenced(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
        hits: dict[str, int],
    ) -> str:
        details: list[str] = []
        dropped = await self._discard_notifications(guild_id)
        if dropped:
            stores.append(STORE_NOTIFICATIONS)
            details.append(f"notifications_dropped={dropped}")
        dead = await self._discard_dead_letters(guild_id)
        if dead:
            stores.append(STORE_DEAD_LETTERS)
            details.append(f"dead_letters_dropped={dead}")
        taken = self._writer.discard(guild_id)
        try:
            history_detail, wrote = await self._purge_history(
                command, guild_id, fence, stores, hits
            )
        except (PartialPurgeError, PurgeRetry) as error:
            # Postgres already holds the purged copy: the discarded dirty
            # copy is never restored, every replica reloads (best effort),
            # and the step fails with the exact store state.
            if isinstance(error, PartialPurgeError):
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
            details.append(
                await self._purge_watch(command, guild_id, fence, stores, hits)
            )
        return _join(*details)

    async def _bump_epoch(self, guild_id: str, *, best_effort: bool = False) -> None:
        try:
            await self._redis.incr(purge_epoch_key(guild_id))
        except Exception:
            if not best_effort:
                raise
        self._runtimes.forget(guild_id)

    async def _discard_dead_letters(self, guild_id: str) -> int:
        """Discard the guild's dead-lettered wakes from before the purge.

        Same rule as queued notifications, no name matching: every entry of
        the dead-letter stream whose `guild_id` field is this guild and
        that exists now. Only the `guild_id` field and the entry id are
        read, so it works on the verbatim and on the ids-only shape.
        """
        dropped = 0
        start = "-"
        while True:
            entries = await self._redis.xrange(
                DEAD_LETTER_STREAM_KEY, min=start, max="+", count=500
            )
            if not entries:
                return dropped
            matching = [
                _decode(entry_id)
                for entry_id, fields in entries
                if _decode(fields.get(b"guild_id", fields.get("guild_id", b"")))
                == guild_id
            ]
            if matching:
                dropped += int(
                    await self._redis.xdel(DEAD_LETTER_STREAM_KEY, *matching)
                )
            if len(entries) < 500:
                return dropped
            start = "(" + _decode(entries[-1][0])

    async def _discard_notifications(self, guild_id: str) -> int:
        """Discard every notification queued for the guild before the purge.

        Wake stream entries, the pending list and claimed batch lists are
        raw copies of channel activity made before the user was blocked,
        and pending triggers rather than memory, so all of them go, without
        trying to recognise the user by name. Anything queued after this
        point was produced under the block list. They are not restored if
        the fold then fails, and a retry discards whatever was queued after
        the first attempt too. Dead letters: _discard_dead_letters.
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
        hits: dict[str, int],
    ) -> tuple[str, bool]:
        """Returns (detail, whether the stored history was rewritten)."""
        names = list(command.names)
        old_tombstone = await self._repository.tombstone(guild_id)
        tombstoned = old_tombstone is not None
        legacy_state, legacy = await self._repository.legacy_state(guild_id)
        snapshot = await self._repository.load_canonical(guild_id)
        if legacy_state == "unreadable" and (
            fence.external
            or (
                snapshot is None
                and not await self._redis.exists(purge_epoch_key(guild_id))
            )
        ):
            # Its bytes may hold the user, but nothing can be folded from
            # them; never deleted, never acked purged.
            raise PurgeFailed("unreadable legacy store")
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
        hits["history"] = 0
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
        if any(command.user_id in leaf for leaf in string_leaves(serialized)):
            raise PurgeFailed("purged history still held the user id")
        if fence.lost:
            raise PurgeFailed("fence lost before the history write")
        _snapshot, _v1_written = await self._writer.replace_purged(
            guild_id,
            serialized,
            previous_revision=snapshot.revision,
            tombstone={
                "run_id": str(command.run_id),
                "request_id": str(command.request_id),
            },
        )
        stores.extend([STORE_POSTGRES_HISTORY, STORE_REDIS_HISTORY])
        if tombstoned and str(old_tombstone.get("run_id", "")) != str(command.run_id):
            await self._note_tombstone_cleared(old_tombstone, guild_id, "later run")
        details = []
        if migrated:
            details.append("legacy history migrated")
        note = notes[-1] if notes else None
        if note is None:
            details.append("history kept")
        else:
            details.append(f"history folded attempts={note.attempts}")
            details.append(f"history_name_hits={note.name_hits}")
            hits["history"] = note.name_hits
        if drop_legacy:
            # Postgres holds the purged copy and v1 holds it too or was
            # deleted: the raw legacy key must not outlive a reported purge.
            try:
                await self._repository.forget_legacy(guild_id)
            except Exception as error:
                raise PurgeRetry(
                    "legacy history delete failed after postgres:purged"
                ) from error
            stores.append(STORE_LEGACY_HISTORY)
            details.append("legacy history deleted")
        return _join(*details), True

    async def _purge_watch(
        self,
        command: PurgeCommand,
        guild_id: str,
        fence: Fence,
        stores: list[str],
        hits: dict[str, int],
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
            decided, watch_hits = await self._decide_watch(
                entries, user_id=command.user_id, names=list(command.names)
            )
            name_hits_kept += watch_hits
            hits["watch"] = name_hits_kept
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
            return _join(
                f"watch channels_rewritten={changed}",
                f"watch_name_hits={name_hits_kept}",
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
