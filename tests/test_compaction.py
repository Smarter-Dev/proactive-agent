"""Compaction must never split a tool call from its return.

pydantic-ai rejects a run whose message history ends on an unanswered
ToolCallPart, so a boundary that lands mid-turn wedges every later wake:
the same history compacts the same way and raises the same UserError
forever.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from proactive_agent.agent import compact_agent_history, fold_boundary


def wake_turn(label: str, tool_calls: int) -> list[ModelMessage]:
    """One wake: a user prompt, some tool round trips, a closing note."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(f"wake {label}")])
    ]
    for index in range(tool_calls):
        call_id = f"{label}-{index}"
        messages.append(
            ModelResponse(
                parts=[ToolCallPart("read_notifications", {}, tool_call_id=call_id)]
            )
        )
        messages.append(
            ModelRequest(
                parts=[ToolReturnPart("read_notifications", "ok", tool_call_id=call_id)]
            )
        )
    messages.append(ModelResponse(parts=[TextPart(f"done {label}")]))
    return messages


def ends_on_unanswered_call(messages: list[ModelMessage]) -> bool:
    """Mirror of the pydantic-ai precondition compaction kept violating."""
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, ModelResponse) and any(
        isinstance(part, ToolCallPart) for part in last.parts
    )


async def test_folded_half_never_ends_on_an_unanswered_tool_call():
    # Three tool round trips leave the ideal cut (len - 8) on a tool-return
    # request, which is what wedged the production agent.
    history = wake_turn("a", 3) + wake_turn("b", 3) + wake_turn("c", 3)
    folded: list[list[ModelMessage]] = []

    async def summarize(messages):
        folded.append(messages)
        return "note"

    result = await compact_agent_history(
        history, token_limit=0, summarize=summarize, keep_messages=8
    )

    assert not ends_on_unanswered_call(folded[0])
    assert isinstance(result[0], ModelRequest)
    # The summary note, its acknowledgement, then the kept tail.
    assert result[2:] == history[len(folded[0]) :]


@pytest.mark.parametrize("tool_calls", range(6))
async def test_boundary_starts_a_turn_for_any_turn_length(tool_calls: int):
    history = wake_turn("a", tool_calls) + wake_turn("b", tool_calls)

    cut = fold_boundary(history, keep_messages=8)

    assert cut is not None
    assert history[cut].parts[0].content.startswith("wake ")
    assert not ends_on_unanswered_call(history[:cut])


async def test_final_turn_longer_than_keep_messages_folds_earlier_turns():
    # No turn starts within the last 8 messages, so the boundary falls back
    # to the start of the final turn rather than folding the whole history.
    history = wake_turn("a", 2) + wake_turn("b", 9)

    cut = fold_boundary(history, keep_messages=8)

    assert cut == len(wake_turn("a", 2))
    assert not ends_on_unanswered_call(history[:cut])


async def test_single_turn_history_is_left_alone():
    history = wake_turn("a", 9)
    calls = []

    async def summarize(messages):
        calls.append(messages)
        return "note"

    result = await compact_agent_history(
        history, token_limit=0, summarize=summarize, keep_messages=8
    )

    assert result == history
    assert calls == []


async def test_history_under_the_limit_is_not_compacted():
    history = wake_turn("a", 3)
    calls = []

    async def summarize(messages):
        calls.append(messages)
        return "note"

    result = await compact_agent_history(
        history, token_limit=1_000_000, summarize=summarize, keep_messages=8
    )

    assert result == history
    assert calls == []
