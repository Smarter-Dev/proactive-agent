"""Redis-hot, REST-durable guild history with debounced write-behind."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from pydantic import ValidationError

from proactive_agent.contracts import HistorySnapshot
from proactive_agent.keys import history_key, legacy_history_key, purge_epoch_key

logger = logging.getLogger(__name__)


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
            legacy = await self._load_legacy(guild_id)
            if legacy is not None:
                await self.cache(legacy)
                return legacy
        return build_snapshot(guild_id, [], revision=0)

    async def load_canonical(self, guild_id: str) -> HistorySnapshot | None:
        """v1 Redis, else the Postgres copy (cached into v1); never legacy."""
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

    async def cache(self, snapshot: HistorySnapshot) -> None:
        if not snapshot_is_valid(snapshot):
            raise ValueError("refusing to cache history with an invalid checksum")
        await self._redis.set(
            history_key(snapshot.guild_id), snapshot.model_dump_json()
        )

    async def _load_redis(self, guild_id: str) -> HistorySnapshot | None:
        raw = await self._redis.get(history_key(guild_id))
        if not raw:
            return None
        try:
            snapshot = HistorySnapshot.model_validate_json(raw)
        except ValidationError:
            logger.warning("invalid Redis proactive history guild=%s", guild_id)
            return None
        if snapshot.guild_id != guild_id or not snapshot_is_valid(snapshot):
            logger.warning("mismatched Redis proactive history guild=%s", guild_id)
            return None
        return snapshot

    async def _load_legacy(self, guild_id: str) -> HistorySnapshot | None:
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
        await self._repository.cache(snapshot)
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
        self, guild_id: str, history: list[dict], *, previous_revision: int
    ) -> HistorySnapshot:
        """Synchronously replace the guild's history after a privacy purge.

        Any dirty copy is discarded, the Postgres recovery copy is written
        first at a revision above both stores, then the v1 Redis key. If the
        PUT fails nothing was written; the caller treats the purge as failed.
        """
        self.discard(guild_id)
        durable = await self._api.get_history(guild_id)
        revision = max(previous_revision, durable.revision if durable else 0) + 1
        snapshot = build_snapshot(guild_id, history, revision=revision)
        await self._api.put_history(snapshot)
        await self._repository.cache(snapshot)
        return snapshot

    async def flush(self, guild_id: str) -> None:
        attempt = 0
        while snapshot := self._dirty.get(guild_id):
            try:
                await self._api.put_history(snapshot)
            except Exception as error:
                if getattr(error, "status_code", None) == 409:
                    # Postgres already holds a newer revision (for example a
                    # purge written by another replica). Retrying an older
                    # copy can never succeed; Redis keeps the live history.
                    logger.warning(
                        "proactive history flush superseded guild=%s revision=%d",
                        guild_id,
                        snapshot.revision,
                    )
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
