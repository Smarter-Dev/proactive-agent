"""Process-wide list of Discord users whose messages must not reach the model.

Today a privacy purge adds its user here so a wake cannot re-ingest the
messages that were just purged; the opt-out feature (#74) takes the same list
over. The list is refreshed in the background. Until the first fetch (and
this process's enforcing report) succeeds the worker processes no wakes at
all; after that it keeps going on the last list it loaded however long
fetches fail (Zech's decision), still renewing its per-process enforcing key
with the revision it holds. Logs never name a user on the list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import uuid4

from proactive_agent.keys import (
    privacy_enforcing_key,
    privacy_enforcing_replica_key,
    privacy_enforcing_replicas_key,
)

logger = logging.getLogger(__name__)

REFRESH_SECONDS = 60
ENFORCING_TTL_SECONDS = 180
RETRY_BASE_SECONDS = 1
RETRY_MAX_SECONDS = 30

# KEYS: own replica key, registry set of replica keys, component key.
# Replica keys are read by name from the registry (single-node Redis).
_REPORT_ENFORCING_LUA = """
redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
redis.call('SADD', KEYS[2], KEYS[1])
redis.call('EXPIRE', KEYS[2], 86400)
local lowest = nil
for _, key in ipairs(redis.call('SMEMBERS', KEYS[2])) do
  local value = tonumber(redis.call('GET', key))
  if value == nil then
    redis.call('SREM', KEYS[2], key)
  elseif lowest == nil or value < lowest then
    lowest = value
  end
end
redis.call('SET', KEYS[3], tostring(lowest), 'EX', ARGV[2])
return lowest
"""


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
        self._stale_logged_at: float | None = None

    @property
    def enforcing(self) -> bool:
        """True once this process has loaded the list and reported it.

        Zech's decision: after the first load the worker always keeps going
        on the last list it loaded, however long fetches fail (that only
        happens in a broader failure). With no list ever loaded it takes no
        wakes (cold start).
        """
        return self._last_success is not None

    def is_blocked(self, user_id: str | int | None) -> bool:
        return user_id is not None and str(user_id) in self._user_ids

    async def refresh(self) -> bool:
        """Fetch the list once. False (and the old list kept) on failure."""
        try:
            listing = await self._api.get_blocked_users()
        except Exception as error:
            if self._last_success is None:
                logger.warning(
                    "blocked users refresh failed type=%s status=%s loaded=False",
                    type(error).__name__,
                    getattr(error, "status_code", None),
                )
                return False
            await self._keep_reporting_stale()
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
        # Taken before the report, so this replica's enforcing window never
        # outlives the key it wrote.
        reported_at = self._clock()
        try:
            await self._report_enforcing()
        except Exception as error:
            # Unreported means not enforcing: the purge job could not see
            # this replica, so it must not take wakes on the strength of it.
            logger.warning(
                "blocked users enforcing marker failed type=%s", type(error).__name__
            )
            return False
        self._last_success = reported_at
        self._stale_logged_at = None
        self.loaded.set()
        return True

    async def _keep_reporting_stale(self) -> None:
        """A fetch failed after the first load: keep the last list.

        The per-process key is still renewed with the revision this process
        actually holds, so it expires only when the process is gone or Redis
        is unreachable, and the published minimum stays at that revision (a
        purge needing a newer one waits). Logged at most once a minute,
        with the age and revision only.
        """
        now = self._clock()
        if self._stale_logged_at is None or now - self._stale_logged_at >= 60:
            self._stale_logged_at = now
            logger.warning(
                "blocked users list stale age=%ds revision=%s",
                int(now - self._last_success),
                self.revision,
            )
        try:
            await self._report_enforcing()
        except Exception as error:
            logger.debug(
                "blocked users enforcing marker failed type=%s", type(error).__name__
            )

    async def _report_enforcing(self) -> None:
        """Report this replica and publish the minimum over live replicas.

        One script, so no concurrent report can overwrite the aggregate with
        a minimum computed before this replica's key existed: the aggregate
        never exceeds the revision of a replica that is taking wakes.
        Replica keys expire after ENFORCING_TTL_SECONDS and are renewed every
        refresh cycle, fetch failures included, so one expires only when its
        process is gone or Redis is unreachable.
        """
        await self._redis.eval(
            _REPORT_ENFORCING_LUA,
            3,
            privacy_enforcing_replica_key(self._component, self.replica_id),
            privacy_enforcing_replicas_key(self._component),
            privacy_enforcing_key(self._component),
            str(self.revision),
            ENFORCING_TTL_SECONDS,
        )

    async def run(self, stop: asyncio.Event) -> None:
        failures = 0
        while not stop.is_set():
            if await self.refresh():
                failures = 0
                delay = self._refresh_seconds
            else:
                failures += 1
                # Retry faster than the refresh interval; each failed cycle
                # still renews this process's key with the held revision.
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
