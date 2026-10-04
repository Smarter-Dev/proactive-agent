"""Fixtures shared by the privacy purge tests."""

from __future__ import annotations

import asyncio
import json

import pytest
from purge_fakes import (
    CHANNEL,
    GUILD,
    FakeAPI,
    dump,
    raw_history_with_target,
    watch_addendum,
)

from proactive_agent.history import GuildHistoryRepository, build_snapshot
from proactive_agent.keys import legacy_history_key, ownership_key


@pytest.fixture
async def world(redis_client):
    """A guild the worker owns, with the target in every store."""
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: watch_addendum()}
    raw = dump(raw_history_with_target())
    api.durable[GUILD] = build_snapshot(GUILD, raw[:2], revision=2)
    await redis_client.set(ownership_key(GUILD), "external")
    await redis_client.set(legacy_history_key(GUILD), json.dumps(raw))
    await GuildHistoryRepository(redis_client, api).cache(
        build_snapshot(GUILD, raw, revision=3)
    )
    yield api
    await asyncio.sleep(0)
