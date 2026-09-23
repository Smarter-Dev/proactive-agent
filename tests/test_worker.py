from __future__ import annotations

import asyncio
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

from pydantic_ai.exceptions import ModelHTTPError

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
    batch = SimpleNamespace(
        guild_id="111",
        wake_id="wake-1",
        notifications=("one", "two"),
        dropped=3,
    )
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


async def test_unavailable_retries_at_30_and_60_before_recovery():
    queue, batch = queue_and_batch()
    runtime = SimpleNamespace(
        process=AsyncMock(
            side_effect=[
                ModelHTTPError(503, "gemini-3.8-flash"),
                ModelHTTPError(503, "gemini-3.8-flash"),
                None,
            ]
        ),
        record_unavailable_retries=AsyncMock(),
    )
    worker = ProactiveWorker(
        queue, SimpleNamespace(get=AsyncMock(return_value=runtime))
    )
    worker._sleep_retry = AsyncMock()

    await worker._run_guild("111", ())

    assert runtime.process.await_count == 3
    assert worker._sleep_retry.await_args_list == [call(30), call(60)]
    runtime.record_unavailable_retries.assert_awaited_once_with(
        batch, retries=2, recovered=True
    )
    queue.acknowledge.assert_awaited_once_with(batch)
    queue.record_failure.assert_not_awaited()


async def test_unavailable_exhausts_three_retries_then_dead_letters():
    queue, batch = queue_and_batch()
    queue.record_failure.return_value = True
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=ModelHTTPError(503, "gemini-3.8-flash")),
        record_unavailable_retries=AsyncMock(),
        report_failure=AsyncMock(return_value=None),
    )
    worker = ProactiveWorker(
        queue, SimpleNamespace(get=AsyncMock(return_value=runtime))
    )
    worker._sleep_retry = AsyncMock()

    await worker._run_guild("111", ())

    assert runtime.process.await_count == 4
    assert worker._sleep_retry.await_args_list == [call(30), call(60), call(120)]
    runtime.record_unavailable_retries.assert_awaited_once_with(
        batch, retries=3, recovered=False
    )
    queue.acknowledge.assert_not_awaited()
    queue.record_failure.assert_awaited_once_with(
        batch,
        error="UnavailableRetriesExhausted: model returned HTTP 503 after 3 retries",
        max_attempts=1,
    )
    runtime.report_failure.assert_awaited_once()


async def test_history_note_failure_does_not_repeat_a_completed_wake():
    queue, batch = queue_and_batch()
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=[ModelHTTPError(503, "gemini-3.8-flash"), None]),
        record_unavailable_retries=AsyncMock(side_effect=RuntimeError("cache down")),
    )
    worker = ProactiveWorker(
        queue, SimpleNamespace(get=AsyncMock(return_value=runtime))
    )
    worker._sleep_retry = AsyncMock()

    await worker._run_guild("111", ())

    assert runtime.process.await_count == 2
    queue.acknowledge.assert_awaited_once_with(batch)
    queue.record_failure.assert_not_awaited()


async def test_other_model_http_error_keeps_existing_retry_policy():
    queue, _batch = queue_and_batch()
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=ModelHTTPError(500, "gemini-3.8-flash")),
        record_unavailable_retries=AsyncMock(),
    )
    worker = ProactiveWorker(
        queue, SimpleNamespace(get=AsyncMock(return_value=runtime))
    )
    worker._sleep_retry = AsyncMock()

    await worker._run_guild("111", ())

    assert runtime.process.await_count == 1
    worker._sleep_retry.assert_not_awaited()
    runtime.record_unavailable_retries.assert_not_awaited()
    assert queue.record_failure.await_args.kwargs["max_attempts"] == 5


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


def completion_lines(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "wake completed" in record.getMessage()
    ]


async def test_a_completed_wake_says_so(caplog):
    queue, _batch = queue_and_batch()
    runtime = SimpleNamespace(process=AsyncMock())
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    with caplog.at_level(logging.INFO, logger="proactive_agent.worker"):
        await worker._run_guild("111", ())

    lines = completion_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert "guild=111" in line
    assert "wake=wake-1" in line
    assert "notifications=2" in line
    assert "dropped=3" in line
    assert re.search(r"duration=\d+\.\d+s", line)


async def test_the_logged_duration_measures_the_wake(caplog):
    queue, _batch = queue_and_batch()

    async def slow_process(_batch):
        await asyncio.sleep(0.05)

    runtime = SimpleNamespace(process=slow_process)
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    with caplog.at_level(logging.INFO, logger="proactive_agent.worker"):
        await worker._run_guild("111", ())

    seconds = float(re.search(r"duration=(\d+\.\d+)s", completion_lines(caplog)[0])[1])
    assert seconds >= 0.05


async def test_a_failed_wake_is_never_logged_as_completed(caplog):
    queue, _batch = queue_and_batch()
    runtime = SimpleNamespace(
        process=AsyncMock(side_effect=RuntimeError("boom")),
        report_failure=AsyncMock(),
    )
    runtimes = SimpleNamespace(get=AsyncMock(return_value=runtime))
    worker = ProactiveWorker(queue, runtimes)

    with caplog.at_level(logging.INFO, logger="proactive_agent.worker"):
        await worker._run_guild("111", ())

    assert completion_lines(caplog) == []


async def test_a_wake_with_nothing_to_process_is_not_logged_as_completed(caplog):
    queue, _batch = queue_and_batch()
    queue.build_batch.return_value = None
    runtimes = SimpleNamespace(get=AsyncMock())
    worker = ProactiveWorker(queue, runtimes)

    with caplog.at_level(logging.INFO, logger="proactive_agent.worker"):
        await worker._run_guild("111", ())

    assert completion_lines(caplog) == []
