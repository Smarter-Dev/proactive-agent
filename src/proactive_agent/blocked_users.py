"""Process-wide list of Discord users whose messages must not reach the model.

Today a privacy purge adds its user here so a wake cannot re-ingest the
messages that were just purged; the opt-out feature (#74) takes the same list
over. The list is refreshed in the background; a failed refresh keeps the
last list read, and until the first fetch succeeds the worker processes no
wakes at all. Logs never name a user on the list.
"""

from __future__ import annotations

import asyncio
import logging

from proactive_agent.keys import privacy_enforcing_key

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 60
ENFORCING_TTL_SECONDS = 180
RETRY_BASE_SECONDS = 1
RETRY_MAX_SECONDS = 30


class BlockedUsers:
    def __init__(
        self,
        api,
        redis_client,
        *,
        component: str = "worker",
        refresh_seconds: float = REFRESH_SECONDS,
        retry_base_seconds: float = RETRY_BASE_SECONDS,
        retry_max_seconds: float = RETRY_MAX_SECONDS,
    ):
        self._api = api
        self._redis = redis_client
        self._component = component
        self._refresh_seconds = refresh_seconds
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._user_ids: frozenset[str] = frozenset()
        self.revision: int | None = None
        self.loaded = asyncio.Event()

    def is_blocked(self, user_id: str | int | None) -> bool:
        return user_id is not None and str(user_id) in self._user_ids

    async def refresh(self) -> bool:
        """Fetch the list once. False (and the old list kept) on failure."""
        try:
            listing = await self._api.get_blocked_users()
        except Exception as error:
            status = getattr(error, "status_code", None)
            logger.warning(
                "blocked users refresh failed type=%s status=%s loaded=%s",
                type(error).__name__,
                status,
                self.loaded.is_set(),
            )
            return False
        if self.revision is not None and listing.revision < self.revision:
            logger.warning(
                "blocked users refresh ignored an older revision=%d current=%d",
                listing.revision,
                self.revision,
            )
        else:
            self._user_ids = frozenset(listing.user_ids)
            self.revision = listing.revision
        self.loaded.set()
        try:
            await self._redis.set(
                privacy_enforcing_key(self._component),
                str(self.revision),
                ex=ENFORCING_TTL_SECONDS,
            )
        except Exception as error:
            logger.warning(
                "blocked users enforcing marker failed type=%s", type(error).__name__
            )
        return True

    async def run(self, stop: asyncio.Event) -> None:
        failures = 0
        while not stop.is_set():
            if await self.refresh():
                failures = 0
                delay = self._refresh_seconds
            else:
                failures += 1
                delay = (
                    self._refresh_seconds
                    if self.loaded.is_set()
                    else min(
                        self._retry_max_seconds,
                        self._retry_base_seconds * 2 ** (failures - 1),
                    )
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def wait_loaded(self, stop: asyncio.Event) -> bool:
        """Block until the first list is loaded; False if stopped first."""
        if self.loaded.is_set():
            return True
        loaded = asyncio.create_task(self.loaded.wait())
        stopped = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({loaded, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            loaded.cancel()
            stopped.cancel()
        return self.loaded.is_set()
