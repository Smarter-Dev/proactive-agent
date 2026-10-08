"""Idle compaction: no verbatim message outlives a quiet guild by long.

Once a guild's history has not been written for ``IDLE_WINDOW`` (2 hours,
the same window the bot uses, ``AGENT_VERBATIM_IDLE_WINDOW`` there), the
history becomes its memory note alone:

- a history flagged fresh (a wake stored its compaction, note + kept tail,
  and nothing since) drops the tail, with no model call;
- any other history is folded whole by the agent's own model, as a privacy
  purge folds it, but with the ordinary compaction prompt.

It runs as a worker service, whether or not any wake comes, every
``SWEEP_TICK``. The clock is the ``written_at`` the history write stores
beside the v1 key (``keys.history_meta_key``), indexed in
``keys.HISTORY_IDLE_INDEX_KEY``; both live in Redis, so a restart resumes it.
v1 histories written before the index existed start their clock at the
service's first pass. A privacy purge's rewrite keeps the clock where it was.
A history that no longer parses is replaced at the idle point by a note that
it was dropped.

A fold's tokens are billed like a wake's: one usage report under the agent
model (operation ``agent``), with its own ``idle-`` wake id.

The fold is written like a wake's: v1 at once (a new revision, so every
replica's in-RAM copy is reloaded before its next wake) and Postgres through
the debounced writer. The guild is fenced like a purge fences it: the guild
lease when the worker owns the guild, else the privacy lock; a fence that is
busy is retried on the next tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from proactive_agent.agent import (
    ZERO_USAGE,
    add_usage,
    is_summary_only,
    leading_note_pair,
    memory_note_pair,
    purge_agent_history,
)
from proactive_agent.errors import exception_trace
from proactive_agent.history import HistoryChecksumError
from proactive_agent.keys import lease_key, privacy_lock_key
from proactive_agent.optout_gate import scrub_history
from proactive_agent.purge import PrivacyLock

logger = logging.getLogger(__name__)

IDLE_WINDOW = timedelta(hours=2)
SWEEP_TICK = timedelta(minutes=1)
SUMMARY_TIMEOUT_SECONDS = 120
FENCE_SECONDS = 600
UNSUMMARIZED_NOTE = (
    "[Earlier history could not be summarized when it went idle and was dropped.]"
)
UNREADABLE_NOTE = (
    "[Earlier history could not be read when it went idle and was dropped.]"
)


class IdleHistoryCompactor:
    """Worker service (``run(stop)``) folding idle guild histories."""

    def __init__(
        self,
        redis_client,
        queue,
        repository,
        writer,
        *,
        summarize: Callable[[list[ModelMessage]], Awaitable[tuple[str, dict]]],
        api,
        model_id: str,
        clock: Callable[[], float] = time.time,
        tick_seconds: float = SWEEP_TICK.total_seconds(),
        blocked_users=None,
    ):
        self._redis = redis_client
        self._queue = queue
        self._repository = repository
        self._writer = writer
        self._summarize = summarize
        self._api = api
        self._model_id = model_id
        self._clock = clock
        self._tick_seconds = tick_seconds
        self._blocked_users = blocked_users

    async def run(self, stop: asyncio.Event) -> None:
        await self._repository.index_unindexed(now=self._clock())
        while not stop.is_set():
            await self.sweep_once()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._tick_seconds)
            except TimeoutError:
                pass

    async def sweep_once(self) -> dict[str, str]:
        if self._blocked_users is not None and not self._blocked_users.enforcing:
            # No list yet, so nobody would be scrubbed from the fold (#100).
            # The next tick tries again.
            logger.info("proactive idle compaction deferred: blocked users not loaded")
            return {}
        cutoff = self._clock() - IDLE_WINDOW.total_seconds()
        outcomes: dict[str, str] = {}
        for guild_id in await self._repository.idle_guild_ids(written_before=cutoff):
            try:
                outcomes[guild_id] = await self.compact_guild(guild_id)
            except Exception as error:
                # Types and frames only: a model error can quote the history.
                logger.error(
                    "proactive idle compaction failed guild=%s\n%s",
                    guild_id,
                    exception_trace(error),
                )
                continue
            logger.info(
                "proactive idle compaction guild=%s outcome=%s",
                guild_id,
                outcomes[guild_id],
            )
        return outcomes

    async def compact_guild(self, guild_id: str) -> str:
        fence = await self._try_fence(guild_id)
        if fence is None:
            return "fence busy"
        renew = asyncio.create_task(self._renew(fence))
        try:
            return await self._compact_fenced(guild_id)
        finally:
            renew.cancel()
            await asyncio.gather(renew, return_exceptions=True)
            await fence.release()

    @staticmethod
    async def _renew(fence) -> None:
        # Losing the fence mid-fold is caught by the revision check on save.
        while True:
            await asyncio.sleep(min(5, max(1, fence.ttl_seconds / 3)))
            if not await fence.renew():
                return

    async def _try_fence(self, guild_id: str):
        if await self._queue.externally_owned(guild_id):
            return await self._queue.acquire_lease(guild_id)
        token = uuid4().hex
        if not await self._redis.set(
            privacy_lock_key(guild_id), token, nx=True, ex=FENCE_SECONDS
        ):
            return None
        fence = PrivacyLock(
            self._redis, guild_id=guild_id, token=token, ttl_seconds=FENCE_SECONDS
        )
        if await self._redis.exists(lease_key(guild_id)):
            # Ownership just moved to the bot while a worker wake still runs.
            await fence.release()
            return None
        return fence

    async def _compact_fenced(self, guild_id: str) -> str:
        if await self._repository.is_invalid(guild_id):
            # A purge left v1 tombstoned; it is retried and rewrites history.
            return "tombstoned"
        written_at, fresh = await self._repository.idle_state(guild_id)
        if written_at is None:
            await self._repository.forget_idle(guild_id)
            return "not indexed"
        if self._clock() - written_at < IDLE_WINDOW.total_seconds():
            await self._repository.reindex(guild_id, written_at)
            return "written since"
        try:
            snapshot = await self._repository.load_canonical(guild_id)
            if snapshot is None or not snapshot.history:
                await self._repository.forget_idle(guild_id)
                return "empty"
            # The summariser never reads someone who opted out (#100).
            history: list[ModelMessage] = list(
                ModelMessagesTypeAdapter.validate_json(
                    json.dumps(scrub_history(snapshot.history, self._blocked_users))
                )
            )
        except (HistoryChecksumError, ValidationError):
            return await self._replace_unreadable(guild_id)
        if is_summary_only(history):
            await self._repository.forget_idle(guild_id)
            return "already summary only"
        note_pair = leading_note_pair(history) if fresh else None
        if note_pair is not None:
            compacted, outcome = note_pair, "tail dropped"
        else:
            compacted, usage = await self._fold(history)
            outcome = "folded"
            await self._record_usage(guild_id, usage)
        await self._writer.save(
            guild_id=guild_id,
            history=json.loads(ModelMessagesTypeAdapter.dump_json(compacted)),
            previous_revision=snapshot.revision,
        )
        # Summary only: nothing for the sweep until the next real write.
        await self._repository.forget_idle(guild_id)
        return outcome

    async def _replace_unreadable(self, guild_id: str) -> str:
        """Verbatim bytes nobody can use, and no purge can rewrite what it
        cannot parse. The worker cannot delete its Postgres row, so both
        copies become a note that the history was dropped."""
        revision = await self._repository.stored_revision(guild_id)
        await self._writer.save(
            guild_id=guild_id,
            history=json.loads(
                ModelMessagesTypeAdapter.dump_json(memory_note_pair(UNREADABLE_NOTE))
            ),
            previous_revision=revision,
        )
        await self._repository.forget_idle(guild_id)
        return "unreadable replaced"

    async def _fold(
        self, history: list[ModelMessage]
    ) -> tuple[list[ModelMessage], dict[str, int]]:
        usage = dict(ZERO_USAGE)

        async def summarize(messages: list[ModelMessage]) -> str:
            summary, spent = await self._summarize(messages)
            add_usage(usage, spent)
            return summary

        try:
            folded = await asyncio.wait_for(
                purge_agent_history(history, summarize=summarize),
                timeout=SUMMARY_TIMEOUT_SECONDS,
            )
        except Exception as error:
            # The bound holds even when the model does not answer: keep the
            # note the history already had, or say the rest was dropped.
            logger.error(
                "proactive idle summary failed, dropping verbatim history\n%s",
                exception_trace(error),
            )
            folded = leading_note_pair(history) or memory_note_pair(UNSUMMARIZED_NOTE)
        return folded, usage

    async def _record_usage(self, guild_id: str, usage: dict[str, int]) -> None:
        if not any(usage.values()):
            return
        try:
            await self._api.record_usage(
                guild_id=guild_id,
                wake_id=f"idle-{uuid4().hex}",
                metered_at=datetime.now(UTC),
                passive=True,
                responses=0,
                entries=[{"model_id": self._model_id, "operation": "agent", **usage}],
            )
        except Exception as error:
            # The fold is still written: losing a usage row beats keeping
            # verbatim history past the idle window.
            logger.error(
                "proactive idle usage report failed guild=%s\n%s",
                guild_id,
                exception_trace(error),
            )
