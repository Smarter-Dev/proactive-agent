"""Agent-only half of the former two-pass in-process adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC

from pydantic_ai import Agent
from pydantic_ai.models import Model

from proactive_agent.agent import AgentDeps, KimiAgentRunner, ToolBudget
from proactive_agent.contracts import NotificationEnvelope
from proactive_agent.environment import (
    ChannelEnvironment,
    InstructionStore,
    WakeActions,
)
from proactive_agent.types import ActivationResult

SKIM_SYSTEM_PROMPT = """\
You skim Discord transcripts for a chat agent. Summarize what is happening
in a short paragraph, then list the load-bearing messages VERBATIM as
`[id=<message id>] <display name> (user id <author id>): <content>` lines.
Keep it brief; the agent can look up anything by id."""


def _usage_dict(usage) -> dict[str, int]:
    if callable(usage):
        usage = usage()
    return {
        "input_tokens": usage.input_tokens or 0,
        "output_tokens": usage.output_tokens or 0,
        "cache_read_tokens": usage.cache_read_tokens or 0,
    }


def _merge_usage(usage_by_model: dict, model_id: str, usage: dict) -> None:
    entry = usage_by_model.setdefault(
        model_id, {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
    )
    for key, value in usage.items():
        entry[key] += value


@dataclass
class SkimRunner:
    model: Model | str
    _agent: Agent = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._agent = Agent(
            self.model, output_type=str, system_prompt=SKIM_SYSTEM_PROMPT
        )

    async def skim(self, transcript: str) -> tuple[str, dict]:
        result = await self._agent.run(transcript)
        return result.output, _usage_dict(result.usage)


def render_channel_instructions(
    instruction_stores: dict[str, InstructionStore],
    enabled_channels: dict[str, str],
) -> str:
    sections = ["YOUR WATCH INSTRUCTIONS BY CHANNEL:"]
    for channel_id, channel_name in enabled_channels.items():
        label = channel_name or channel_id
        store = instruction_stores.get(channel_id)
        sections.append(f"WATCH INSTRUCTIONS FOR #{label} (channel_id={channel_id}):")
        if store is None or not store.entries:
            sections.append("- none set")
            continue
        sections.extend(
            f"- {entry.instruction_id} "
            f"(expires {entry.expires_at:%H:%M} UTC): {entry.text}"
            for entry in store.entries
        )
    return "\n".join(sections)


def render_notifications(
    notifications: tuple[NotificationEnvelope, ...], dropped: int = 0
) -> str:
    lines = ["NOTIFICATIONS since your last wake (oldest first):"]
    if dropped:
        lines.append(f"({dropped} older notifications were dropped)")
    for notification in notifications:
        stamp = notification.created_at.astimezone(UTC).strftime("%H:%M")
        channel = notification.channel_name or notification.channel_id
        channel_prefix = f"[#{channel}] " if channel else ""
        lines.append(
            f"{channel_prefix}[{stamp} UTC, {notification.kind}] {notification.body}"
        )
    return "\n".join(lines)


def build_wake_brief(
    notifications: tuple[NotificationEnvelope, ...],
    dropped: int,
    *,
    instruction_stores: dict[str, InstructionStore],
    enabled_channels: dict[str, str],
) -> str:
    return f"""\
{render_notifications(notifications, dropped)}

{render_channel_instructions(instruction_stores, enabled_channels)}

A notification is a lead, not the full story — pull context with your tools
when it isn't enough. Act per your policy (or deliberately don't).

Before you finish: the watcher will not wake you again except for direct
engagement or activity it independently judges interesting. If anything here
deserves a follow-up you would otherwise never hear about — someone said
they'd report back, a thread you want to see resolved, a topic worth
catching — call set_watch_instruction now with a TTL. Then finish with a
one-sentence note on what you did and why."""


@dataclass
class AgentEngine:
    agent_runner: KimiAgentRunner
    skim: SkimRunner
    agent_model_id: str
    skim_model_id: str
    deps_factory: Callable
    last_deps: AgentDeps | None = field(default=None, init=False, repr=False)

    async def wake(
        self,
        *,
        notifications: tuple[NotificationEnvelope, ...],
        dropped: int,
        enabled_channels: dict[str, str],
        instruction_stores: dict[str, InstructionStore],
        channel_envs: object | dict[str, ChannelEnvironment],
        brief_preamble: str = "",
    ) -> ActivationResult:
        usage_by_model: dict[str, dict] = {}
        actions = WakeActions()

        async def skim_transcript(transcript: str) -> str:
            text, usage = await self.skim.skim(transcript)
            _merge_usage(usage_by_model, self.skim_model_id, usage)
            return text

        deps = self.deps_factory(
            enabled_channels=enabled_channels,
            channel_envs=channel_envs,
            actions=actions,
            instruction_stores=instruction_stores,
            skim_transcript=skim_transcript,
            budget=ToolBudget(),
        )
        if not isinstance(deps, AgentDeps):
            raise TypeError("deps_factory must return AgentDeps")
        self.last_deps = deps
        brief = build_wake_brief(
            notifications,
            dropped,
            instruction_stores=instruction_stores,
            enabled_channels=enabled_channels,
        )
        if brief_preamble:
            brief = f"{brief_preamble}\n\n{brief}"
        note, usage = await self.agent_runner.wake(brief, deps)
        _merge_usage(usage_by_model, self.agent_model_id, usage)
        totals = {
            key: sum(entry[key] for entry in usage_by_model.values())
            for key in ("input_tokens", "output_tokens", "cache_read_tokens")
        }
        return ActivationResult(
            responses=list(actions.sent),
            input_tokens=totals["input_tokens"],
            output_tokens=totals["output_tokens"],
            cache_read_tokens=totals["cache_read_tokens"],
            model_id=self.agent_model_id,
            reactions=tuple(actions.reactions),
            usage_by_model=usage_by_model,
            details={
                "agent": {"tool_calls": deps.budget.used, "note": str(note)[:300]},
                "watch_instruction_updates": sum(
                    store.updates for store in instruction_stores.values()
                ),
            },
        )
