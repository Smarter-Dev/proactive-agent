"""The guild system prompt survives every history fold (#456).

pydantic-ai adds a ``system_prompt`` only to an empty history, so the live
agent sends the guild prompt as ``instructions``: every run carries it,
first. Each fold path rebuilds the history without any
system part; the next run must still send the prompt.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.agent import (
    KimiAgentRunner,
    build_kimi_agent,
    compact_agent_history,
    purge_agent_history,
)

PROMPT = "GUILD SYSTEM PROMPT: identity, rules, FACTS guidance"


def _turns(count: int):
    messages = []
    for index in range(count):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"wake {index}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"done {index}")]))
    return messages


class Seen:
    def __init__(self):
        self.requests: list[tuple[str | None, list]] = []

    def model(self):
        def respond(messages, info: AgentInfo):
            self.requests.append((info.instructions, list(messages)))
            return ModelResponse(parts=[TextPart(content="ok")])

        return FunctionModel(respond)


async def _summarize(_messages):
    return "a summary"


async def _wake(history, seen: Seen, **runner_args):
    agent = build_kimi_agent(seen.model(), system_prompt=PROMPT)
    runner = KimiAgentRunner(
        agent=agent,
        summarize=lambda m: _summary_with_usage(),
        history=history,
        **runner_args,
    )
    await runner.wake("next", deps=None)
    return runner


async def _summary_with_usage():
    return "a summary", {}


def _system_parts(messages) -> int:
    return sum(
        isinstance(part, SystemPromptPart)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
    )


def _assert_prompt_sent_once(seen: Seen):
    instructions, messages = seen.requests[-1]
    assert instructions is not None and instructions.startswith(PROMPT)
    assert _system_parts(messages) == 0


async def test_a_first_wake_sends_the_prompt():
    seen = Seen()
    await _wake([], seen)
    _assert_prompt_sent_once(seen)


async def test_after_compaction_the_next_run_sends_the_prompt():
    folded = await compact_agent_history(
        _turns(6), token_limit=1, summarize=_summarize, keep_messages=2
    )
    assert folded != _turns(6) and _system_parts(folded) == 0
    seen = Seen()
    await _wake(folded, seen)
    _assert_prompt_sent_once(seen)


@pytest.mark.parametrize("fold", ["idle", "privacy"])
async def test_after_a_whole_history_fold_the_next_run_sends_the_prompt(fold):
    # The idle sweep and the privacy purge both fold through purge_agent_history.
    folded = await purge_agent_history(_turns(3), summarize=_summarize)
    assert _system_parts(folded) == 0
    seen = Seen()
    await _wake(folded, seen)
    _assert_prompt_sent_once(seen)


async def test_a_run_inside_a_runner_that_compacts_itself_sends_the_prompt():
    seen = Seen()
    await _wake(_turns(6), seen, token_limit=1)
    _assert_prompt_sent_once(seen)


async def test_an_old_history_that_already_holds_the_prompt_sends_it_once():
    old = _turns(2)
    old[0] = ModelRequest(parts=[SystemPromptPart(content=PROMPT), *old[0].parts])
    seen = Seen()
    runner = await _wake(old, seen)
    _assert_prompt_sent_once(seen)
    # Stored history is not rewritten; only what is sent.
    assert _system_parts(runner.history) == 1
