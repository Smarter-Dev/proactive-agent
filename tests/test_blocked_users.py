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


async def test_a_lower_revision_after_a_restore_still_replaces_the_list(
    redis_client, caplog
):
    api = ListAPI()
    api.blocked = {"revision": 5, "user_ids": [TARGET]}
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    api.blocked = {"revision": 2, "user_ids": ["222222222222222222"]}

    assert await blocked.refresh()

    assert not blocked.is_blocked(TARGET)
    assert blocked.is_blocked("222222222222222222")
    assert blocked.revision == 2
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"2"
    assert "revision went back" in caplog.text


async def test_the_aggregate_is_the_minimum_over_live_replicas(redis_client):
    api_new, api_old = ListAPI(), ListAPI()
    api_new.blocked = {"revision": 5, "user_ids": [TARGET]}
    api_old.blocked = {"revision": 3, "user_ids": []}
    current = BlockedUsers(api_new, redis_client, replica_id="a")
    lagging = BlockedUsers(api_old, redis_client, replica_id="b")

    await lagging.refresh()
    await current.refresh()

    # The replica still on revision 3 holds the component key down, so the
    # purge job cannot believe every worker blocks the revision-5 user.
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"3"
    assert await redis_client.get(privacy_enforcing_key("worker") + ":a") == b"5"
    assert await redis_client.ttl(privacy_enforcing_key("worker") + ":b") <= 180

    api_old.blocked = {"revision": 5, "user_ids": [TARGET]}
    await lagging.refresh()
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"5"

    # A replica that died stops counting once its own key expires.
    api_new.blocked = {"revision": 6, "user_ids": [TARGET]}
    await redis_client.delete(privacy_enforcing_key("worker") + ":b")
    await current.refresh()
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"6"


async def test_cold_start_retries_with_backoff_until_the_first_list(redis_client):
    api = ListAPI()
    api.blocked_failures = 3
    blocked = BlockedUsers(
        api, redis_client, retry_base_seconds=0.001, retry_max_seconds=0.004
    )
    stop = asyncio.Event()
    task = asyncio.create_task(blocked.run(stop))

    assert await asyncio.wait_for(
        blocked.wait_enforcing(stop, poll_seconds=0.001), timeout=2
    )
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
    worker._enforcing_poll_seconds = 0.005
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


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


async def test_a_replica_with_failing_refreshes_stops_enforcing_alone(redis_client):
    # Replica A keeps the shared enforcing key fresh; replica B's fetches
    # fail. B must stop taking wakes once its own list is 180 s old.
    api_a, api_b = ListAPI(), ListAPI()
    clock = Clock()
    replica_a = BlockedUsers(api_a, redis_client, clock=clock)
    replica_b = BlockedUsers(api_b, redis_client, clock=clock)
    await replica_a.refresh()
    await replica_b.refresh()
    assert replica_a.enforcing and replica_b.enforcing

    api_b.blocked_failures = 100
    clock.now += 179
    assert not await replica_b.refresh()
    assert replica_b.enforcing  # a short gap keeps the last list
    clock.now += 2
    await replica_a.refresh()
    assert not await replica_b.refresh()

    assert await redis_client.exists(privacy_enforcing_key("worker"))
    assert replica_a.enforcing
    assert not replica_b.enforcing

    api_b.blocked_failures = 0
    assert await replica_b.refresh()
    assert replica_b.enforcing


async def test_a_non_enforcing_replica_takes_no_wake_lease(redis_client):
    clock = Clock()
    blocked = BlockedUsers(ListAPI(), redis_client, clock=clock)
    await blocked.refresh()
    clock.now += 181
    queue = SimpleNamespace(
        externally_owned=AsyncMock(return_value=True),
        acquire_lease=AsyncMock(return_value=None),
    )
    worker = ProactiveWorker(queue, SimpleNamespace(), blocked_users=blocked)
    worker._enforcing_poll_seconds = 0.005
    ready = (SimpleNamespace(stream_id="1-0", guild_id="111"),)
    task = asyncio.create_task(worker._run_guild("111", ready))
    await asyncio.sleep(0.05)

    queue.acquire_lease.assert_not_awaited()
    clock.now -= 181  # a fetch succeeded again
    await asyncio.sleep(0.05)
    queue.acquire_lease.assert_awaited()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_the_wake_loop_pauses_while_not_enforcing(redis_client):
    clock = Clock()
    blocked = BlockedUsers(ListAPI(), redis_client, clock=clock)
    await blocked.refresh()

    async def read_ready(**_kwargs):
        await asyncio.sleep(0.005)
        return ()

    queue = SimpleNamespace(
        initialize=AsyncMock(),
        reclaim_ready=AsyncMock(return_value=()),
        read_ready=AsyncMock(side_effect=read_ready),
    )
    stop = asyncio.Event()
    worker = ProactiveWorker(queue, SimpleNamespace(), blocked_users=blocked)
    worker._enforcing_poll_seconds = 0.005
    task = asyncio.create_task(worker.run(stop))
    await asyncio.sleep(0.03)
    clock.now += 181
    await asyncio.sleep(0.03)
    reads = queue.read_ready.await_count
    await asyncio.sleep(0.05)

    assert queue.read_ready.await_count == reads
    stop.set()
    await asyncio.wait_for(task, timeout=1)
