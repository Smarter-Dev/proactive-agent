"""Tells the bot what this worker can run, by keys that expire without it."""

from __future__ import annotations

import asyncio

from proactive_agent.keys import FORUM_POSTS_CAPABILITY_KEY

# The bot stops routing forum posts at most this long after the last worker
# that runs them stops.
CAPABILITY_TTL_SECONDS = 180
CAPABILITY_REFRESH_SECONDS = 60


class CapabilityBeacon:
    """Keeps the forum-posts capability key set while this worker runs."""

    def __init__(
        self,
        redis_client,
        *,
        ttl_seconds: int = CAPABILITY_TTL_SECONDS,
        refresh_seconds: float = CAPABILITY_REFRESH_SECONDS,
    ):
        self._redis = redis_client
        self._ttl_seconds = ttl_seconds
        self._refresh_seconds = refresh_seconds

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self._redis.set(
                FORUM_POSTS_CAPABILITY_KEY, b"1", ex=self._ttl_seconds
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._refresh_seconds)
            except TimeoutError:
                pass
