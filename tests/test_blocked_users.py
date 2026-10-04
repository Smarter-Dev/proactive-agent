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


async def test_a_loaded_replica_keeps_going_through_a_long_outage(redis_client, caplog):
    # Zech's decision: after the first load the worker keeps taking wakes on
    # the last list however long fetches fail, and keeps reporting the
    # revision it holds.
    api = ListAPI()
    api.blocked = {"revision": 4, "user_ids": [TARGET]}
    clock = Clock()
    blocked = BlockedUsers(api, redis_client, clock=clock, replica_id="host-1")
    assert await blocked.refresh()
    api.blocked_failures = 10_000
    own = privacy_enforcing_key("worker") + ":host-1"

    for _ in range(25):  # 12.5 minutes of failing 30 s cycles
        clock.now += 30
        # fakeredis does not follow the fake clock: expire the key by hand
        # so each cycle has to renew it.
        await redis_client.delete(own)
        assert not await blocked.refresh()

    assert blocked.enforcing
    assert blocked.is_blocked(TARGET)
    assert await redis_client.get(own) == b"4"
    assert 170 < await redis_client.ttl(own) <= 180
    assert await redis_client.get(privacy_enforcing_key("worker")) == b"4"
    stale = [r.getMessage() for r in caplog.records if "list stale" in r.getMessage()]
    assert len(stale) == 13  # at most once a minute
    assert stale[-1].endswith("revision=4")
    assert all(TARGET not in r.getMessage() for r in caplog.records)

    queue = SimpleNamespace(
        externally_owned=AsyncMock(return_value=True),
        acquire_lease=AsyncMock(return_value=None),
    )
    worker = ProactiveWorker(queue, SimpleNamespace(), blocked_users=blocked)
    worker._enforcing_poll_seconds = 0.005
    ready = (SimpleNamespace(stream_id="1-0", guild_id="111"),)
    task = asyncio.create_task(worker._run_guild("111", ready))
    await asyncio.sleep(0.02)
    queue.acquire_lease.assert_awaited()  # still takes wakes
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_a_stale_replica_holds_the_published_minimum_down(redis_client):
    old, new = ListAPI(), ListAPI()
    old.blocked = {"revision": 3, "user_ids": []}
    new.blocked = {"revision": 5, "user_ids": [TARGET]}
    stale = BlockedUsers(old, redis_client, replica_id="old")
    fresh = BlockedUsers(new, redis_client, replica_id="new")
    await stale.refresh()
    old.blocked_failures = 100

    for _ in range(10):
        await redis_client.delete(privacy_enforcing_key("worker") + ":old")
        await stale.refresh()  # fails, still renews revision 3
        await fresh.refresh()

    assert await redis_client.get(privacy_enforcing_key("worker")) == b"3"


async def test_cold_start_still_takes_no_wakes(redis_client):
    api = ListAPI()
    api.blocked_failures = 100
    blocked = BlockedUsers(api, redis_client, replica_id="host-1")
    for _ in range(5):
        assert not await blocked.refresh()
    assert not blocked.enforcing
    assert not await redis_client.exists(privacy_enforcing_key("worker") + ":host-1")
    queue = SimpleNamespace(
        externally_owned=AsyncMock(return_value=True),
        acquire_lease=AsyncMock(return_value=None),
    )
    worker = ProactiveWorker(queue, SimpleNamespace(), blocked_users=blocked)
    worker._enforcing_poll_seconds = 0.005
    ready = (SimpleNamespace(stream_id="1-0", guild_id="111"),)
    task = asyncio.create_task(worker._run_guild("111", ready))
    await asyncio.sleep(0.03)
    queue.acquire_lease.assert_not_awaited()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
