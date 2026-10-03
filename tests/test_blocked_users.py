"""The blocked-users list: refreshed, kept on failure, required before wakes."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx
import pytest
from purge_fakes import TARGET, FakeAPI

from proactive_agent.api import ApplicationAPI
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.contracts import BlockedUsersList
from proactive_agent.keys import privacy_enforcing_key
from proactive_agent.worker import ProactiveWorker


class ListAPI(FakeAPI):
    async def get_blocked_users(self):
        return BlockedUsersList.model_validate(await super().get_blocked_users())


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


async def test_refresh_loads_the_list_and_marks_the_worker_enforcing(redis_client):
    api = ListAPI()
    api.blocked = {"revision": 7, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)

    assert await blocked.refresh()

    assert blocked.is_blocked(TARGET) and blocked.is_blocked(int(TARGET))
    assert not blocked.is_blocked("222222222222222222")
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"7"
    assert 0 < await redis_client.ttl(privacy_enforcing_key("worker")) <= 180


async def test_failed_refresh_keeps_the_last_list_and_logs_no_ids(redis_client, caplog):
    caplog.set_level(logging.DEBUG)
    api = ListAPI()
    api.blocked = {"revision": 1, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    api.blocked_failures = 1

    assert not await blocked.refresh()

    assert blocked.is_blocked(TARGET)
    assert blocked.revision == 1
    assert not any(TARGET in record.getMessage() for record in caplog.records)


async def test_an_older_revision_never_replaces_a_newer_list(redis_client):
    api = ListAPI()
    api.blocked = {"revision": 5, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    api.blocked = {"revision": 4, "user_ids": []}

    await blocked.refresh()

    assert blocked.is_blocked(TARGET)
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"5"


async def test_cold_start_retries_with_backoff_until_the_first_list(redis_client):
    api = ListAPI()
    api.blocked_failures = 3
    blocked = BlockedUsers(
        api, redis_client, retry_base_seconds=0.001, retry_max_seconds=0.004
    )
    stop = asyncio.Event()
    task = asyncio.create_task(blocked.run(stop))

    assert await asyncio.wait_for(blocked.wait_loaded(stop), timeout=2)
    assert api.blocked_failures == 0
    assert await redis_client.exists(privacy_enforcing_key("worker"))
    stop.set()
    await asyncio.wait_for(task, timeout=1)


async def test_cold_start_404_keeps_retrying(redis_client):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "not found"})

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    blocked = BlockedUsers(
        api, redis_client, retry_base_seconds=0.001, retry_max_seconds=0.002
    )
    stop = asyncio.Event()
    task = asyncio.create_task(blocked.run(stop))
    await asyncio.sleep(0.05)

    assert not blocked.loaded.is_set()
    assert not task.done()
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    await api.close()


async def test_worker_processes_no_wakes_before_the_list_is_loaded(redis_client):
    api = ListAPI()
    blocked = BlockedUsers(api, redis_client)

    async def read_ready(**_kwargs):
        await asyncio.sleep(0.005)  # a real XREADGROUP blocks
        return ()

    queue = SimpleNamespace(
        initialize=AsyncMock(),
        reclaim_ready=AsyncMock(return_value=()),
        read_ready=AsyncMock(side_effect=read_ready),
    )
    stop = asyncio.Event()
    worker = ProactiveWorker(queue, SimpleNamespace(), blocked_users=blocked)
    task = asyncio.create_task(worker.run(stop))
    await asyncio.sleep(0.02)

    queue.initialize.assert_not_awaited()
    queue.read_ready.assert_not_awaited()
    await blocked.refresh()
    await asyncio.sleep(0.02)
    queue.initialize.assert_awaited_once()
    stop.set()
    await asyncio.wait_for(task, timeout=1)


async def test_blocked_users_client_path_and_shape():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"revision": 3, "user_ids": [TARGET]})

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    try:
        listing = await api.get_blocked_users()
    finally:
        await api.close()

    assert seen[0].url.path == "/api/privacy/blocked-users"
    assert listing.revision == 3 and listing.user_ids == [TARGET]
