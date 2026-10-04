"""Privacy compaction: the agent's own model rewrites memory without one user."""

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
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.agent import (
    PrivacyCompactionError,
    name_hits,
    privacy_compaction_summary,
    purge_agent_history,
)

TARGET = "111111111111111111"
BYSTANDER = "222222222222222222"


def history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart("wake 1: notifications")]),
        ModelResponse(
            parts=[ToolCallPart("channel_history", {"channel_id": "9"}, "c1")]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "channel_history",
                    f"[id=1] A·kai (uid={TARGET}): my cat is called Miso\n"
                    f"[id=2] B·nia (uid={BYSTANDER}): I ship Rust 1.95 friday",
                    "c1",
                )
            ]
        ),
        ModelResponse(parts=[TextPart("noted both")]),
    ]


def last_prompt(messages: list[ModelMessage]) -> str:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return "\n".join(
        part.content for part in request.parts if isinstance(part, UserPromptPart)
    )


def scripted(*notes: str):
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart(notes[min(len(seen), len(notes)) - 1])])

    return FunctionModel(respond), seen


async def test_prompt_names_the_user_and_the_whole_history_rides_along():
    model, seen = scripted("nia ships Rust 1.95 on friday, she said so in #general.")

    note = await privacy_compaction_summary(
        model, history(), user_id=TARGET, names=["kai"]
    )

    assert note.text == "nia ships Rust 1.95 on friday, she said so in #general."
    assert note.attempts == 1 and note.name_hits == 0
    prompt = last_prompt(seen[0])
    assert TARGET in prompt and '"kai"' in prompt
    assert "my cat is called Miso" in str(seen[0])


async def test_id_in_note_is_asked_again_then_accepted_when_clean():
    model, seen = scripted(
        f"<@{TARGET}> has a cat",
        "nia ships Rust 1.95 on friday, she said so in #general.",
    )

    note = await privacy_compaction_summary(
        model, history(), user_id=TARGET, names=["kai"]
    )

    assert note.text == "nia ships Rust 1.95 on friday, she said so in #general."
    assert note.attempts == 2
    assert "still contains the user id" in last_prompt(seen[1])


async def test_id_after_two_retries_raises_without_content():
    model, seen = scripted(*[f"{TARGET} again"] * 3)

    with pytest.raises(PrivacyCompactionError) as raised:
        await privacy_compaction_summary(
            model, history(), user_id=TARGET, names=["kai"]
        )

    assert len(seen) == 3
    assert TARGET not in str(raised.value)
    assert "kai" not in str(raised.value)


async def test_empty_note_never_becomes_a_blank_memory():
    model, _seen = scripted("  ", "\n", " ")

    with pytest.raises(PrivacyCompactionError) as raised:
        await privacy_compaction_summary(model, history(), user_id=TARGET, names=[])
    assert TARGET not in str(raised.value)


async def test_name_is_asked_about_once_then_reported():
    model, seen = scripted(
        "Kai likes cats; nia ships Rust 1.95 on friday in #general.",
        "KAI still here; nia ships Rust 1.95 on friday in #general.",
    )

    note = await privacy_compaction_summary(
        model, history(), user_id=TARGET, names=["kai"]
    )

    assert len(seen) == 2
    assert note.text.startswith("KAI still here")
    assert note.name_hits == 1


def test_name_hits_are_whole_word_and_case_insensitive():
    assert name_hits("Kai said hi", ["kai"]) == ["kai"]
    assert name_hits("kaiser and kaizen", ["kai"]) == []
    assert name_hits("ask Kai the Rustacean.", ["Kai the Rustacean"]) == [
        "Kai the Rustacean"
    ]
    assert name_hits("nothing", []) == []


async def test_purge_folds_everything_into_the_note_pair():
    folded: list[list[ModelMessage]] = []

    async def summarize(messages):
        folded.append(messages)
        return "nia ships Rust"

    original = history()
    result = await purge_agent_history(original, summarize=summarize)

    assert folded == [original]
    assert len(result) == 2
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)
    assert "nia ships Rust" in result[0].parts[0].content
    assert TARGET not in str(result)


async def test_purge_of_empty_history_calls_nothing():
    async def summarize(_messages):
        raise AssertionError("must not run")

    assert await purge_agent_history([], summarize=summarize) == []
