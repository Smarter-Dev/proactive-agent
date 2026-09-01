"""Redis-hot, REST-durable guild history with debounced write-behind."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging

from pydantic import ValidationError

from proactive_agent.contracts import HistorySnapshot
from proactive_agent.keys import history_key, legacy_history_key

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
        cached = await self._load_redis(guild_id)
        if cached is not None:
            return cached

        # During the split rollout the integrated bot's existing Redis key is
        # still the freshest live history. Prefer it to a potentially older
        # Postgres snapshot left by an earlier canary.
        legacy = await self._load_legacy(guild_id)
        if legacy is not None:
            durable = await self._api.get_history(guild_id)
            if durable is not None:
                if not snapshot_is_valid(durable):
                    raise ValueError(
                        f"durable proactive history checksum failed for guild {guild_id}"
                    )
                # Keep the legacy content but inherit the durable revision so
                # the next write is a valid monotonic replacement.
                legacy = build_snapshot(
                    guild_id, legacy.history, revision=durable.revision
                )
            await self.cache(legacy)
            return legacy

        durable = await self._api.get_history(guild_id)
        if durable is not None:
            if not snapshot_is_valid(durable):
                raise ValueError(
                    f"durable proactive history checksum failed for guild {guild_id}"
                )
            await self.cache(durable)
            return durable

        return build_snapshot(guild_id, [], revision=0)

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

    async def flush(self, guild_id: str) -> None:
        attempt = 0
        while snapshot := self._dirty.get(guild_id):
            try:
                await self._api.put_history(snapshot)
            except Exception:
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
