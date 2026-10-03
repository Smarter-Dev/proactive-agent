"""Shared worker scheduler; concurrency across guilds, serialization within."""

from __future__ import annotations

import asyncio
import logging
import time

from pydantic_ai.exceptions import ModelHTTPError

from proactive_agent.queue import ReadyRecord

logger = logging.getLogger(__name__)
UNAVAILABLE_RETRY_DELAYS = (30, 60, 120)


class GuildLeaseLostError(RuntimeError):
    pass


class UnavailableRetriesExhausted(RuntimeError):
    def __init__(self, retries: int):
        self.retries = retries
        super().__init__(f"model returned HTTP 503 after {retries} retries")


class ProactiveWorker:
    def __init__(
        self,
        queue,
        runtimes,
        *,
        concurrency: int = 8,
        max_attempts: int = 5,
        services: tuple = (),
    ):
        self._queue = queue
        # Long-running side loops (privacy purges, ...) with run(stop).
        self._services = tuple(services)
        self._runtimes = runtimes
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_attempts = max_attempts
        self._tasks: set[asyncio.Task] = set()
        self._sleep_retry = asyncio.sleep

    async def run(self, stop: asyncio.Event) -> None:
        services = [
            asyncio.create_task(service.run(stop)) for service in self._services
        ]
        try:
            await self._run_wakes(stop)
        finally:
            stop.set()
            await asyncio.gather(*services, return_exceptions=True)

    async def _run_wakes(self, stop: asyncio.Event) -> None:
        await self._queue.initialize()
        while not stop.is_set():
            reclaimed = await self._queue.reclaim_ready()
            ready = reclaimed or await self._queue.read_ready(block_ms=5_000)
            by_guild: dict[str, list[ReadyRecord]] = {}
            for record in ready:
                by_guild.setdefault(record.guild_id, []).append(record)
            for guild_id, records in by_guild.items():
                task = asyncio.create_task(self._run_guild(guild_id, tuple(records)))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _run_guild(self, guild_id: str, ready: tuple[ReadyRecord, ...]) -> None:
        while True:
            if not await self._queue.externally_owned(guild_id):
                await self._queue.discard_embedded_ready(guild_id, ready)
                return
            async with self._semaphore:
                lease = await self._queue.acquire_lease(guild_id)
                if lease is not None:
                    await self._run_guild_with_lease(guild_id, ready, lease)
                    return
            # This worker already owns the ready records. Keep them hot while
            # another wake for the guild finishes instead of abandoning them
            # until the stream's multi-minute reclaim timeout.
            await asyncio.sleep(0.25)

    async def _run_guild_with_lease(
        self, guild_id: str, ready: tuple[ReadyRecord, ...], lease
    ) -> None:
        async with lease:
            batch = await self._queue.build_batch(guild_id, ready)
            if batch is None:
                await self._queue.acknowledge_ready(ready)
                return
            renew_task = asyncio.create_task(self._renew_lease(lease))
            process_task = None
            runtime = None
            started = time.monotonic()
            try:
                runtime = await self._runtimes.get(guild_id)
                process_task = asyncio.create_task(
                    self._process_with_unavailable_retries(runtime, batch)
                )
                done, _pending = await asyncio.wait(
                    {process_task, renew_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if renew_task in done:
                    # Propagates GuildLeaseLostError and cancels in-flight
                    # Discord/API work before another replica takes over.
                    await renew_task
                await process_task
                await self._queue.acknowledge(batch)
                # The only line a healthy wake writes. Without it an idle
                # agent and a dead one leave identical logs.
                logger.info(
                    "proactive guild wake completed guild=%s wake=%s notifications=%d dropped=%d duration=%.2fs",
                    guild_id,
                    batch.wake_id,
                    len(batch.notifications),
                    batch.dropped,
                    time.monotonic() - started,
                )
            except Exception as error:
                logger.exception(
                    "proactive guild wake failed guild=%s wake=%s",
                    guild_id,
                    batch.wake_id,
                )
                detail = f"{type(error).__name__}: {error}"
                unavailable = isinstance(error, UnavailableRetriesExhausted)
                dead_lettered = await self._queue.record_failure(
                    batch,
                    error=detail,
                    max_attempts=1 if unavailable else self._max_attempts,
                )
                if dead_lettered:
                    if unavailable and runtime is not None:
                        await self._record_unavailable_note(
                            runtime, batch, retries=error.retries, recovered=False
                        )
                    logger.error(
                        "proactive guild wake dead-lettered guild=%s wake=%s",
                        guild_id,
                        batch.wake_id,
                    )
                    await self._announce_failure(runtime, batch, detail)
                # Below the ceiling records remain pending for reclaim.
            finally:
                renew_task.cancel()
                cleanup_tasks = [renew_task]
                if process_task is not None:
                    process_task.cancel()
                    cleanup_tasks.append(process_task)
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def _process_with_unavailable_retries(self, runtime, batch) -> None:
        for retries in range(len(UNAVAILABLE_RETRY_DELAYS) + 1):
            try:
                await runtime.process(batch)
            except ModelHTTPError as error:
                if error.status_code != 503:
                    raise
                if retries == len(UNAVAILABLE_RETRY_DELAYS):
                    raise UnavailableRetriesExhausted(retries) from error
                delay = UNAVAILABLE_RETRY_DELAYS[retries]
                logger.warning(
                    "proactive model unavailable guild=%s wake=%s retry=%d delay=%ds",
                    batch.guild_id,
                    batch.wake_id,
                    retries + 1,
                    delay,
                )
                await self._sleep_retry(delay)
            else:
                if retries:
                    await self._record_unavailable_note(
                        runtime, batch, retries=retries, recovered=True
                    )
                return

    async def _record_unavailable_note(
        self, runtime, batch, *, retries: int, recovered: bool
    ) -> None:
        try:
            await runtime.record_unavailable_retries(
                batch, retries=retries, recovered=recovered
            )
        except Exception:
            logger.exception(
                "proactive unavailable history note failed guild=%s wake=%s",
                batch.guild_id,
                batch.wake_id,
            )

    async def _announce_failure(self, runtime, batch, detail: str) -> None:
        """Tell the guild a wake was dropped, without ever raising.

        A dead-lettered wake is already the bad path; a failure to announce
        it must not replace the logged cause with its own traceback.
        """
        if runtime is None:
            return
        try:
            channel_id = await runtime.report_failure(batch, detail)
        except Exception:
            logger.exception(
                "proactive failure notice failed guild=%s wake=%s",
                batch.guild_id,
                batch.wake_id,
            )
            return
        if channel_id is not None:
            logger.info(
                "proactive failure notice sent guild=%s wake=%s channel=%s",
                batch.guild_id,
                batch.wake_id,
                channel_id,
            )

    async def _renew_lease(self, lease) -> None:
        while True:
            await asyncio.sleep(min(5, max(1, lease.ttl_seconds / 3)))
            if not await lease.renew():
                raise GuildLeaseLostError(
                    f"lost proactive guild lease guild={lease.guild_id}"
                )
