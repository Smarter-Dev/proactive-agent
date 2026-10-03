"""Process-wide list of Discord users whose messages must not reach the model.

Today a privacy purge adds its user here so a wake cannot re-ingest the
messages that were just purged; the opt-out feature (#74) takes the same list
over. The list is refreshed in the background; a failed refresh keeps the
last list read, and until the first fetch succeeds the worker processes no
wakes at all. The same holds whenever this process has gone longer than
ENFORCING_TTL_SECONDS without a successful fetch. Logs never name a user on the list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from proactive_agent.keys import (
    privacy_enforcing_key,
    privacy_enforcing_replica_key,
)

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
        clock=time.monotonic,
        replica_id: str | None = None,
    ):
        self.replica_id = replica_id or uuid4().hex
        self._api = api
        self._redis = redis_client
        self._component = component
        self._refresh_seconds = refresh_seconds
        self._retry_base_seconds = retry_base_seconds
        self._retry_max_seconds = retry_max_seconds
        self._user_ids: frozenset[str] = frozenset()
        self.revision: int | None = None
        self.loaded = asyncio.Event()
        self._clock = clock
        self._last_success: float | None = None

    @property
    def enforcing(self) -> bool:
        """True while THIS process's last successful fetch is fresh.

        The shared enforcing key can be kept alive by another replica, so
        each replica judges itself: past ENFORCING_TTL_SECONDS without a
        successful fetch it must stop taking wakes until one succeeds.
        """
        return (
            self._last_success is not None
            and self._clock() - self._last_success < ENFORCING_TTL_SECONDS
        )

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
            # A database restore can move the revision back. The server's
            # answer is still the list to enforce; freezing on the old one
            # would stop following it for good.
            logger.warning(
                "blocked users revision went back revision=%d previous=%d",
                listing.revision,
                self.revision,
            )
        self._user_ids = frozenset(listing.user_ids)
        self.revision = listing.revision
        self._last_success = self._clock()
        self.loaded.set()
        try:
            await self._report_enforcing()
        except Exception as error:
            logger.warning(
                "blocked users enforcing marker failed type=%s", type(error).__name__
            )
        return True

    async def _report_enforcing(self) -> None:
        """Report this replica, then publish the minimum over live replicas.

        The purge job reads only the component key, so it must never claim
        a revision some live replica has not loaded yet. Replica keys expire
        after ENFORCING_TTL_SECONDS, the same window after which a replica
        that cannot fetch stops taking wakes.
        """
        own = privacy_enforcing_replica_key(self._component, self.replica_id)
        await self._redis.set(own, str(self.revision), ex=ENFORCING_TTL_SECONDS)
        pattern = f"{privacy_enforcing_key(self._component)}:*"
        keys = [key async for key in self._redis.scan_iter(match=pattern)]
        revisions = []
        for value in await self._redis.mget(keys) if keys else ():
            if value is None:
                continue
            try:
                revisions.append(int(value))
            except ValueError:
                continue
        if not revisions:
            revisions = [self.revision]
        await self._redis.set(
            privacy_enforcing_key(self._component),
            str(min(revisions)),
            ex=ENFORCING_TTL_SECONDS,
        )

    async def run(self, stop: asyncio.Event) -> None:
        failures = 0
        while not stop.is_set():
            if await self.refresh():
                failures = 0
                delay = self._refresh_seconds
            else:
                failures += 1
                # Retry faster than the refresh interval so a short outage
                # does not cost the replica its enforcing window.
                delay = min(
                    self._refresh_seconds,
                    self._retry_max_seconds,
                    self._retry_base_seconds * 2 ** (failures - 1),
                )
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def wait_enforcing(
        self, stop: asyncio.Event, *, poll_seconds: float = 1
    ) -> bool:
        """Block until this process is enforcing; False if stopped first."""
        while not self.enforcing:
            if stop.is_set():
                return False
            try:
                await asyncio.wait_for(stop.wait(), timeout=poll_seconds)
            except TimeoutError:
                pass
        return True
