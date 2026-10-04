"""Redis-hot, REST-durable guild history with debounced write-behind."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from pydantic import ValidationError

from proactive_agent.contracts import HistorySnapshot
from proactive_agent.keys import (
    history_invalid_key,
    history_key,
    legacy_history_key,
    purge_epoch_key,
)

logger = logging.getLogger(__name__)

# Set the v1 snapshot only over an older revision, and never while the
# guild's history-invalid tombstone exists (only a purge's own write may
# replace v1 then). HistorySnapshot JSON puts "revision" before "history",
# so the first match is the snapshot's own field, read without decoding a
# potentially large history in Lua.
_CACHE_IF_NEWER_LUA = """
if redis.call('EXISTS', KEYS[2]) == 1 then
  return 0
end
local current = redis.call('GET', KEYS[1])
if current then
  local revision = tonumber(string.match(current, '"revision":(%d+)'))
  if revision and revision >= tonumber(ARGV[2]) then
    return 0
  end
end
redis.call('SET', KEYS[1], ARGV[1])
return 1
"""

# A privacy purge's own v1 write: unconditional, and the only write that
# clears the tombstone.
_WRITE_PURGED_LUA = """
redis.call('SET', KEYS[1], ARGV[1])
redis.call('DEL', KEYS[2])
return 1
"""

# Postgres wins over a v1 copy that fell behind it: delete v1 only if its
# revision is below the durable one (a newer v1 is left alone).
_DELETE_IF_OLDER_LUA = """
local current = redis.call('GET', KEYS[1])
if not current then
  return 0
end
local revision = tonumber(string.match(current, '"revision":(%d+)'))
if revision and revision < tonumber(ARGV[1]) then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

# A purged history is written this many revisions above both stores, so a
# writer still holding the pre-purge v1 copy can never reach the purged
# Postgres revision: its flushes get 409 and Postgres wins (see flush).
PURGE_REVISION_GAP = 1000

_DELETE_IF_EQUAL_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class PartialPurgeError(RuntimeError):
    """Postgres holds the purged history but v1 could not be replaced.

    The message names exactly which stores were touched; it holds no
    content and is used as the purge ack detail.
    """


class HistoryUnavailableError(RuntimeError):
    """The guild's v1 history is tombstoned after a partial purge.

    No wake may run until the purge is retried and clears it.
    """


class StaleHistoryError(RuntimeError):
    """The v1 key already holds this revision or a newer one.

    A writer that lost its guild lease (or ran before a purge) must not
    overwrite what the current owner stored.
    """


def canonical_history(history: list[dict]) -> bytes:
    return json.dumps(
        history,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def history_checksum(history: list[dict]) -> str:
    return hashlib.sha256(canonical_history(history)).hexdigest()


def build_snapshot(
    guild_id: str, history: list[dict], *, revision: int
) -> HistorySnapshot:
    return HistorySnapshot(
        guild_id=guild_id,
        revision=revision,
        checksum=history_checksum(history),
        history=history,
    )


def snapshot_is_valid(snapshot: HistorySnapshot) -> bool:
    return snapshot.checksum == history_checksum(snapshot.history)


class GuildHistoryRepository:
    """Load Redis first and use the application API as durable fallback."""

    def __init__(self, redis_client, api):
        self._redis = redis_client
        self._api = api

    async def load(self, guild_id: str) -> HistorySnapshot:
        """v1 Redis, then the Postgres recovery copy, then the legacy key.

        The legacy embedded-bot key is never restored once the guild has a
        purge epoch: it may still hold a purged user's raw history.
        """
        snapshot = await self.load_canonical(guild_id)
        if snapshot is not None:
            return snapshot
        if not await self._redis.exists(purge_epoch_key(guild_id)):
            legacy = await self.load_legacy(guild_id)
            if legacy is not None:
                await self.cache(legacy)
                return legacy
        return build_snapshot(guild_id, [], revision=0)

    async def load_canonical(self, guild_id: str) -> HistorySnapshot | None:
        """v1 Redis, else the Postgres copy (cached into v1); never legacy.

        A history-invalid tombstone skips v1: it may hold history a purge
        already removed from Postgres. (The Postgres copy is then not cached:
        only the purge's own write may replace v1.)
        """
        if not await self._redis.exists(history_invalid_key(guild_id)):
            cached = await self._load_redis(guild_id)
            if cached is not None:
                return cached
        durable = await self._api.get_history(guild_id)
        if durable is not None:
            if not snapshot_is_valid(durable):
                raise ValueError(
                    f"durable proactive history checksum failed for guild {guild_id}"
                )
            await self.cache(durable)
            return durable
        return None

    async def cache(self, snapshot: HistorySnapshot) -> bool:
        """Set the v1 key only if it holds an older revision (or nothing).

        Returns False, writing nothing, when the stored revision is equal or
        newer.
        """
        if not snapshot_is_valid(snapshot):
            raise ValueError("refusing to cache history with an invalid checksum")
        return bool(
            await self._redis.eval(
                _CACHE_IF_NEWER_LUA,
                2,
                history_key(snapshot.guild_id),
                history_invalid_key(snapshot.guild_id),
                snapshot.model_dump_json(),
                snapshot.revision,
            )
        )

    async def _load_redis(self, guild_id: str) -> HistorySnapshot | None:
        raw = await self._redis.get(history_key(guild_id))
        if not raw:
            return None
        try:
            snapshot = HistorySnapshot.model_validate_json(raw)
        except ValidationError:
            logger.warning("invalid Redis proactive history guild=%s", guild_id)
            await self._drop_unreadable(guild_id, raw)
            return None
        if snapshot.guild_id != guild_id or not snapshot_is_valid(snapshot):
            logger.warning("mismatched Redis proactive history guild=%s", guild_id)
            await self._drop_unreadable(guild_id, raw)
            return None
        return snapshot

    async def write_purged(self, snapshot: HistorySnapshot) -> None:
        """The purge's own v1 write: replaces v1 and clears the tombstone."""
        if not snapshot_is_valid(snapshot):
            raise ValueError("refusing to cache history with an invalid checksum")
        await self._redis.eval(
            _WRITE_PURGED_LUA,
            2,
            history_key(snapshot.guild_id),
            history_invalid_key(snapshot.guild_id),
            snapshot.model_dump_json(),
        )

    async def is_invalid(self, guild_id: str) -> bool:
        return bool(await self._redis.exists(history_invalid_key(guild_id)))

    async def forget_legacy(self, guild_id: str) -> None:
        await self._redis.delete(legacy_history_key(guild_id))

    async def forget(self, guild_id: str) -> None:
        """Delete the v1 key so the next load reads the Postgres copy."""
        await self._redis.delete(history_key(guild_id))

    async def forget_if_older(self, guild_id: str, revision: int) -> bool:
        """Delete v1 if it holds a revision below ``revision``."""
        return bool(
            await self._redis.eval(
                _DELETE_IF_OLDER_LUA, 1, history_key(guild_id), revision
            )
        )

    async def invalidate(self, guild_id: str, tombstone: dict | None = None) -> bool:
        """Best effort: mark v1 unreadable until the purge's own write.

        The value names the purge run and request (never the target) so the
        web can list outstanding tombstones; the key has no TTL.
        """
        value = json.dumps(
            {
                "run_id": (tombstone or {}).get("run_id", ""),
                "request_id": (tombstone or {}).get("request_id", ""),
            }
        )
        try:
            await self._redis.set(history_invalid_key(guild_id), value)
        except Exception as error:
            logger.error(
                "proactive history tombstone failed guild=%s type=%s",
                guild_id,
                type(error).__name__,
            )
            return False
        return True

    async def _drop_unreadable(self, guild_id: str, raw) -> None:
        # An unreadable snapshot's revision would otherwise block caching
        # the Postgres copy that replaces it.
        await self._redis.eval(_DELETE_IF_EQUAL_LUA, 1, history_key(guild_id), raw)

    async def load_legacy(self, guild_id: str) -> HistorySnapshot | None:
        """The pre-split embedded bot's history key, as a revision-1 snapshot."""
        raw = await self._redis.get(legacy_history_key(guild_id))
        if not raw:
            return None
        try:
            history = json.loads(raw)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(history, list) or not all(
            isinstance(item, dict) for item in history
        ):
            return None
        return build_snapshot(guild_id, history, revision=1)


class DebouncedHistoryWriter:
    """Cache every revision immediately and persist only the newest dirty one."""

    def __init__(
        self,
        repository: GuildHistoryRepository,
        api,
        *,
        debounce_seconds: float = 5,
        retry_base_seconds: float = 1,
    ):
        self._repository = repository
        self._api = api
        self._debounce_seconds = debounce_seconds
        self._retry_base_seconds = retry_base_seconds
        self._dirty: dict[str, HistorySnapshot] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._closed = False

    async def save(
        self,
        *,
        guild_id: str,
        history: list[dict],
        previous_revision: int,
    ) -> HistorySnapshot:
        if self._closed:
            raise RuntimeError("history writer is closed")
        snapshot = build_snapshot(guild_id, history, revision=previous_revision + 1)
        if not await self._repository.cache(snapshot):
            raise StaleHistoryError(
                f"proactive history revision {snapshot.revision} is stale "
                f"for guild {guild_id}"
            )
        current = self._dirty.get(guild_id)
        if current is None or snapshot.revision >= current.revision:
            self._dirty[guild_id] = snapshot
        task = self._tasks.get(guild_id)
        if task is not None:
            task.cancel()
        self._tasks[guild_id] = asyncio.create_task(self._flush_after_delay(guild_id))
        return snapshot

    def discard(self, guild_id: str) -> HistorySnapshot | None:
        """Drop the guild's dirty copy and pending flush without writing it.

        A purge calls this first: the dirty copy may hold the purged user's
        words and must never reach Postgres afterwards. Returns the copy so
        a failed purge can put it back.
        """
        task = self._tasks.pop(guild_id, None)
        if task is not None:
            task.cancel()
        return self._dirty.pop(guild_id, None)

    def restore(self, snapshot: HistorySnapshot | None) -> None:
        """Put back a dirty copy taken by discard() when a purge failed."""
        if snapshot is None or self._closed:
            return
        current = self._dirty.get(snapshot.guild_id)
        if current is not None and current.revision >= snapshot.revision:
            return
        self._dirty[snapshot.guild_id] = snapshot
        task = self._tasks.get(snapshot.guild_id)
        if task is not None:
            task.cancel()
        self._tasks[snapshot.guild_id] = asyncio.create_task(
            self._flush_after_delay(snapshot.guild_id)
        )

    async def replace_purged(
        self,
        guild_id: str,
        history: list[dict],
        *,
        previous_revision: int,
        tombstone: dict | None = None,
    ) -> tuple[HistorySnapshot, bool]:
        """Synchronously replace the guild's history after a privacy purge.

        Any dirty copy is discarded, the Postgres recovery copy is written
        first at a revision above both stores, then the v1 Redis key.
        - The PUT fails: nothing was written; the error propagates.
        - The PUT hits 409 (a stale flush landed after the revision read):
          the revision is read again and the PUT retried once.
        - The v1 SET fails after the PUT: the v1 key is deleted so loads
          fall back to the purged Postgres copy. If that delete fails too,
          a history-invalid tombstone is set (loads skip v1, every other
          v1 write is refused) and PartialPurgeError is raised.
        Returns the snapshot and whether v1 now holds it.
        """
        self.discard(guild_id)
        for attempt in range(2):
            durable = await self._api.get_history(guild_id)
            revision = (
                max(previous_revision, durable.revision if durable else 0)
                + PURGE_REVISION_GAP
            )
            snapshot = build_snapshot(guild_id, history, revision=revision)
            try:
                await self._api.put_history(snapshot)
            except Exception as error:
                if getattr(error, "status_code", None) == 409 and attempt == 0:
                    logger.warning(
                        "proactive purged history conflicted, retrying guild=%s",
                        guild_id,
                    )
                    continue
                raise
            break
        try:
            await self._repository.write_purged(snapshot)
            cached = True
        except Exception as error:
            logger.warning(
                "proactive purged history cache failed guild=%s type=%s",
                guild_id,
                type(error).__name__,
            )
            cached = False
        if not cached:
            try:
                await self._repository.forget(guild_id)
            except Exception as error:
                logger.error(
                    "proactive purged history v1 delete failed guild=%s type=%s",
                    guild_id,
                    type(error).__name__,
                )
                tombstoned = await self._repository.invalidate(guild_id, tombstone)
                raise PartialPurgeError(
                    "postgres:purged v1:unpurged "
                    + ("v1:tombstoned" if tombstoned else "v1:not-tombstoned")
                ) from error
            logger.warning(
                "proactive purged history left only in Postgres guild=%s", guild_id
            )
        return snapshot, cached

    async def flush(self, guild_id: str) -> None:
        attempt = 0
        while snapshot := self._dirty.get(guild_id):
            try:
                await self._api.put_history(snapshot)
            except Exception as error:
                if getattr(error, "status_code", None) == 409:
                    # Postgres already holds a newer or conflicting revision
                    # (a purge). Retrying can never succeed, and a v1 copy
                    # below the durable revision is stale: Postgres wins, so
                    # v1 is deleted and the next wake reloads from Postgres.
                    logger.warning(
                        "proactive history flush superseded guild=%s revision=%d",
                        guild_id,
                        snapshot.revision,
                    )
                    await self._postgres_wins(guild_id)
                    latest = self._dirty.get(guild_id)
                    if latest is not None and latest.revision == snapshot.revision:
                        self._dirty.pop(guild_id, None)
                    continue
                attempt += 1
                logger.exception(
                    "proactive history flush failed guild=%s revision=%d",
                    guild_id,
                    snapshot.revision,
                )
                await asyncio.sleep(
                    min(60, self._retry_base_seconds * (2 ** (attempt - 1)))
                )
                continue
            latest = self._dirty.get(guild_id)
            if latest is not None and latest.revision == snapshot.revision:
                self._dirty.pop(guild_id, None)
            attempt = 0

    async def _postgres_wins(self, guild_id: str) -> None:
        try:
            durable = await self._api.get_history(guild_id)
            if durable is not None and await self._repository.forget_if_older(
                guild_id, durable.revision
            ):
                logger.warning(
                    "proactive v1 history behind postgres, dropped guild=%s",
                    guild_id,
                )
        except Exception as error:
            logger.warning(
                "proactive history postgres check failed guild=%s type=%s",
                guild_id,
                type(error).__name__,
            )

    async def close(self, *, timeout: float = 10) -> None:
        self._closed = True
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
        if not self._dirty:
            return
        try:
            await asyncio.wait_for(
                asyncio.gather(*(self.flush(guild_id) for guild_id in self._dirty)),
                timeout=timeout,
            )
        except TimeoutError:
            logger.error(
                "proactive history shutdown flush timed out; Redis retains dirty guilds=%s",
                sorted(self._dirty),
            )

    async def _flush_after_delay(self, guild_id: str) -> None:
        try:
            await asyncio.sleep(self._debounce_seconds)
            await self.flush(guild_id)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            if self._tasks.get(guild_id) is current:
                self._tasks.pop(guild_id, None)
