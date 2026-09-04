"""Shared worker scheduler; concurrency across guilds, serialization within."""

from __future__ import annotations

import asyncio
import logging

from proactive_agent.queue import ReadyRecord

logger = logging.getLogger(__name__)


class GuildLeaseLostError(RuntimeError):
    pass


class ProactiveWorker:
    def __init__(self, queue, runtimes, *, concurrency: int = 8, max_attempts: int = 5):
        self._queue = queue
        self._runtimes = runtimes
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_attempts = max_attempts
        self._tasks: set[asyncio.Task] = set()

    async def run(self, stop: asyncio.Event) -> None:
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
            try:
                runtime = await self._runtimes.get(guild_id)
                process_task = asyncio.create_task(runtime.process(batch))
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
            except Exception as error:
                logger.exception(
                    "proactive guild wake failed guild=%s wake=%s",
                    guild_id,
                    batch.wake_id,
                )
                detail = f"{type(error).__name__}: {error}"
                dead_lettered = await self._queue.record_failure(
                    batch,
                    error=detail,
                    max_attempts=self._max_attempts,
                )
                if dead_lettered:
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
