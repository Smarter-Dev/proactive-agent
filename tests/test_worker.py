from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from proactive_agent.worker import GuildLeaseLostError, ProactiveWorker


class FakeLease:
    guild_id = "111"
    ttl_seconds = 30

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def renew(self):
        return True


def queue_and_batch():
    batch = SimpleNamespace(guild_id="111", wake_id="wake-1")
    queue = SimpleNamespace(
        externally_owned=AsyncMock(return_value=True),
        discard_embedded_ready=AsyncMock(),
        acquire_lease=AsyncMock(return_value=FakeLease()),
        build_batch=AsyncMock(return_value=batch),
        acknowledge=AsyncMock(),
        acknowledge_ready=AsyncMock(),
        record_failure=AsyncMock(return_value=False),
    )
    return queue, batch


async def test_embedded_owner_discards_stale_external_ready() -> None:
    queue, _batch = queue_and_batch()
    queue.externally_owned.return_value = False
    runtimes = SimpleNamespace(get=AsyncMock())
    worker = ProactiveWorker(queue, runtimes)
    ready = (SimpleNamespace(stream_id="1-0", guild_id="111"),)

    await worker._run_guild("111", ready)

    queue.discard_embedded_ready.assert_awaited_once_with("111", ready)
    queue.acquire_lease.assert_not_awaited()
    runtimes.get.assert_not_awaited()


async def test_successful_wake_is_acknowledged():
    queue, batch = queue_and_batch()
    runtime = SimpleNamespace(process=AsyncMock())
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    await worker._run_guild("111", ())

    runtime.process.assert_awaited_once_with(batch)
    queue.acknowledge.assert_awaited_once_with(batch)
    queue.record_failure.assert_not_awaited()


async def test_lost_lease_cancels_processing_and_leaves_batch_for_retry():
    queue, batch = queue_and_batch()
    cancelled = asyncio.Event()

    async def process(_batch):
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    runtime = SimpleNamespace(process=process)
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    async def lose_lease(_lease):
        raise GuildLeaseLostError("lost")

    worker._renew_lease = lose_lease
    await worker._run_guild("111", ())

    assert cancelled.is_set()
    queue.acknowledge.assert_not_awaited()
    queue.record_failure.assert_awaited_once_with(
        batch, error="GuildLeaseLostError: lost", max_attempts=5
    )


async def test_dead_lettered_wake_announces_itself():
    queue, batch = queue_and_batch()
    queue.record_failure.return_value = True
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=RuntimeError("boom")),
        report_failure=AsyncMock(return_value="channel-1"),
    )
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    await worker._run_guild("111", ())

    runtime.report_failure.assert_awaited_once_with(batch, "RuntimeError: boom")


async def test_failure_below_the_ceiling_stays_silent():
    queue, _batch = queue_and_batch()
    queue.record_failure.return_value = False
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=RuntimeError("boom")),
        report_failure=AsyncMock(),
    )
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    await worker._run_guild("111", ())

    runtime.report_failure.assert_not_awaited()


async def test_a_broken_announcement_does_not_escape_the_worker():
    queue, _batch = queue_and_batch()
    queue.record_failure.return_value = True
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=RuntimeError("boom")),
        report_failure=AsyncMock(side_effect=RuntimeError("discord down")),
    )
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    await worker._run_guild("111", ())

    queue.record_failure.assert_awaited_once()


async def test_a_runtime_that_never_loaded_cannot_be_asked_to_announce():
    queue, batch = queue_and_batch()
    queue.record_failure.return_value = True
    runtimes = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("no runtime")))
    worker = ProactiveWorker(queue, runtimes)

    await worker._run_guild("111", ())

    queue.record_failure.assert_awaited_once_with(
        batch, error="RuntimeError: no runtime", max_attempts=5
    )
