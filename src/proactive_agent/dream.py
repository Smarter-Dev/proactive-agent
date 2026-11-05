"""The nightly dream, run by the agent itself (smarter-dev #454).

Once the night is over, each guild with notes (or a memory row not yet dreamed
tonight) gets one turn in the agent's own stored history: the web hands over
what the dream reads (``GET .../chat-memory/dream``), the agent answers with
structured output, and the answer is submitted back. All validation lives in
the web application; a refusal comes back with a reason that quotes nothing
stored, and goes back to the agent in the same conversation, at most
``MAX_RETRIES`` times. The whole exchange stays in the agent's history.

The guild is fenced like an idle fold: the guild lease when the worker owns the
guild, else the privacy lock, so a dream never runs beside a wake or a purge on
the same guild. A busy fence is retried on the next tick. History is read
through the opt-out scrub, and written through the history writer's.

Off unless ``PROACTIVE_DREAM_ENABLED`` is set; the smarter-dev cron dreams
meanwhile. A failed night is retried after ``COOLDOWN_SECONDS`` and given up on
after ``MAX_NIGHT_ATTEMPTS`` tries, leaving the notes for tomorrow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models import Model

from proactive_agent.agent import (
    HISTORY_TOKEN_LIMIT,
    ZERO_USAGE,
    add_usage,
    compact_agent_history,
    drop_other_models_claude_thinking,
)
from proactive_agent.errors import exception_trace
from proactive_agent.history import HistoryChecksumError
from proactive_agent.idle import renew_fence, try_fence
from proactive_agent.keys import dream_attempts_key, dream_cooldown_key
from proactive_agent.optout_gate import scrub_history
from proactive_agent.usage_meter import wake_meter

logger = logging.getLogger(__name__)

DREAM_PROMPT = (Path(__file__).parent / "prompts" / "dream_prompt.md").read_text(
    encoding="utf-8"
)

SWEEP_TICK_SECONDS = 60.0
MAX_RETRIES = 2
MAX_NIGHT_ATTEMPTS = 3
COOLDOWN_SECONDS = 30 * 60
TURN_TIMEOUT_SECONDS = 300
ATTEMPT_TTL_SECONDS = 3 * 24 * 60 * 60


class IdentityUpdate(BaseModel):
    before: str | None = None
    after: str
    evidence: str


class DreamAnswer(BaseModel):
    """The structured answer; the field names are the web's submission's."""

    memory: str
    identity_updates: list[IdentityUpdate] = Field(default_factory=list)
    identity_moves: list[str] = Field(default_factory=list)
    behavior: str | None = None
    personality: str | None = None
    facts: str | None = None
    personality_reason: str | None = None


def opening_prompt(inputs: dict) -> str:
    return inputs["user_message"]


def refusal_prompt(message: str) -> str:
    return (
        "That answer was not saved. "
        f"{message}\n\nAnswer again with the whole structured output."
    )


def restart_prompt(inputs: dict) -> str:
    return (
        "What you read has changed since (memory or tonight's notes moved), so "
        "that answer was not saved. Here is the night again; redo it from this.\n\n"
        + inputs["user_message"]
    )


class DreamService:
    """Worker service (``run(stop)``) dreaming each guild once the night ends."""

    def __init__(
        self,
        redis_client,
        queue,
        repository,
        writer,
        *,
        model: Model,
        summarize,
        api,
        model_id: str,
        blocked_users=None,
        clock: Callable[[], float] = time.time,
        tick_seconds: float = SWEEP_TICK_SECONDS,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._queue = queue
        self._repository = repository
        self._writer = writer
        self._model = model
        self._summarize = summarize
        self._api = api
        self._model_id = model_id
        self._blocked_users = blocked_users
        self._clock = clock
        self._tick_seconds = tick_seconds
        self._enabled = enabled

    async def run(self, stop: asyncio.Event) -> None:
        if not self._enabled:
            await stop.wait()
            return
        while not stop.is_set():
            try:
                await self.sweep_once()
            except Exception as error:
                logger.error("proactive dream sweep failed\n%s", exception_trace(error))
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._tick_seconds)
            except TimeoutError:
                pass

    async def sweep_once(self) -> dict[str, str]:
        if self._blocked_users is not None and not self._blocked_users.enforcing:
            # No list yet, so nobody would be scrubbed from the history (#100).
            return {}
        listed = await self._api.dream_guilds()
        if not listed:
            return {}
        cutoff = str(listed["cutoff"])
        outcomes: dict[str, str] = {}
        for guild_id in listed["guild_ids"]:
            if await self._redis.exists(dream_cooldown_key(guild_id)):
                continue
            attempts_key = dream_attempts_key(guild_id, cutoff)
            if int(await self._redis.get(attempts_key) or 0) >= MAX_NIGHT_ATTEMPTS:
                continue
            try:
                outcomes[guild_id] = await self.dream_guild(guild_id)
            except Exception as error:
                # Types and frames only: a model error can quote the history.
                outcomes[guild_id] = "failed"
                logger.error(
                    "proactive dream failed guild=%s\n%s",
                    guild_id,
                    exception_trace(error),
                )
            if outcomes[guild_id] not in ("dreamed", "nothing to dream"):
                await self._redis.incr(attempts_key)
                await self._redis.expire(attempts_key, ATTEMPT_TTL_SECONDS)
                await self._redis.set(
                    dream_cooldown_key(guild_id), "1", ex=COOLDOWN_SECONDS
                )
            logger.info(
                "proactive dream guild=%s outcome=%s", guild_id, outcomes[guild_id]
            )
        return outcomes

    async def dream_guild(self, guild_id: str) -> str:
        fence = await try_fence(self._redis, self._queue, guild_id)
        if fence is None:
            return "fence busy"
        renew = asyncio.create_task(renew_fence(fence))
        try:
            return await self._dream_fenced(guild_id)
        finally:
            renew.cancel()
            await asyncio.gather(renew, return_exceptions=True)
            await fence.release()

    async def _dream_fenced(self, guild_id: str) -> str:
        if await self._repository.is_invalid(guild_id):
            # A purge left v1 tombstoned; it is retried and rewrites history.
            return "tombstoned"
        inputs = await self._api.dream_inputs(guild_id)
        if inputs is None or not inputs.get("needed"):
            return "nothing to dream"
        snapshot = await self._repository.load_canonical(guild_id)
        try:
            history: list[ModelMessage] = (
                list(
                    ModelMessagesTypeAdapter.validate_json(
                        json.dumps(scrub_history(snapshot.history, self._blocked_users))
                    )
                )
                if snapshot is not None
                else []
            )
        except (HistoryChecksumError, ValidationError):
            return "unreadable history"

        usage = dict(ZERO_USAGE)

        async def summarize(messages: list[ModelMessage]) -> str:
            summary, spent = await self._summarize(messages)
            add_usage(usage, spent)
            return summary

        history = await compact_agent_history(
            history, token_limit=HISTORY_TOKEN_LIMIT, summarize=summarize
        )
        agent = Agent(self._model, output_type=DreamAnswer, instructions=DREAM_PROMPT)
        new_messages: list[ModelMessage] = []
        prompt = opening_prompt(inputs)
        outcome = "refused"
        try:
            async with wake_meter() as meter:
                try:
                    for _ in range(MAX_RETRIES + 1):
                        sent = drop_other_models_claude_thinking(
                            [*history, *new_messages],
                            getattr(self._model, "model_name", None),
                        )
                        result = await asyncio.wait_for(
                            agent.run(
                                prompt,
                                message_history=sent or None,
                                model=meter.wrap(self._model),
                            ),
                            timeout=TURN_TIMEOUT_SECONDS,
                        )
                        new_messages.extend(result.new_messages())
                        verdict = await self._api.submit_dream(
                            guild_id, submission(inputs, result.output)
                        )
                        if verdict.get("accepted"):
                            outcome = "dreamed"
                            break
                        if verdict.get("reason") == "stale":
                            inputs = await self._api.dream_inputs(guild_id)
                            if inputs is None or not inputs.get("needed"):
                                outcome = "nothing to dream"
                                break
                            prompt = restart_prompt(inputs)
                        else:
                            prompt = refusal_prompt(str(verdict.get("message") or ""))
                finally:
                    for spent in meter.usage_by_model().values():
                        add_usage(usage, spent)
        finally:
            if new_messages:
                await self._keep_exchange(guild_id, snapshot, history, new_messages)
            await self._record_usage(guild_id, usage)
        return outcome

    async def _keep_exchange(self, guild_id, snapshot, history, new_messages) -> None:
        """The dream stays in the agent's history, refused turns included."""
        revision = (
            snapshot.revision
            if snapshot is not None
            else await self._repository.stored_revision(guild_id)
        )
        await self._writer.save(
            guild_id=guild_id,
            history=json.loads(
                ModelMessagesTypeAdapter.dump_json([*history, *new_messages])
            ),
            previous_revision=revision,
        )

    async def _record_usage(self, guild_id: str, usage: dict) -> None:
        if not any(value for key, value in usage.items() if key != "requests"):
            return
        try:
            await self._api.record_usage(
                guild_id=guild_id,
                wake_id=f"dream-{uuid4().hex}",
                metered_at=datetime.now(UTC),
                passive=True,
                responses=0,
                entries=[{"model_id": self._model_id, "operation": "agent", **usage}],
            )
        except Exception as error:
            logger.error(
                "proactive dream usage report failed guild=%s\n%s",
                guild_id,
                exception_trace(error),
            )


def submission(inputs: dict, answer: DreamAnswer) -> dict:
    """The answer plus what it was made from, which the web checks is current."""
    return {
        **answer.model_dump(),
        "cutoff": inputs["cutoff"],
        "base_revision": inputs["base_revision"],
        "note_ids": [note["id"] for note in inputs["notes"]],
    }
