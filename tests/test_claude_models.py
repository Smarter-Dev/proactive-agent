"""Claude runs on Anthropic's own API: routing, caching, tool choice, usage,
and thinking blocks that are only replayed where they are still valid.

Every model call here goes to ``claude_fakes.ClaudeServer``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from claude_fakes import (
    ClaudeServer,
    cache_controls,
    text_reply,
    thinking_blocks,
    tool_reply,
)
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel

from proactive_agent import models
from proactive_agent.agent import (
    KimiAgentRunner,
    compact_agent_history,
    drop_other_models_claude_thinking,
    self_compaction_summary,
    strip_claude_thinking,
    usage_dict,
)
from proactive_agent.runtime import GuildRuntime

ONE_HOUR = {"type": "ephemeral", "ttl": "1h"}


@pytest.fixture
def no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_ENDPOINT", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)


def serve(monkeypatch, *replies) -> ClaudeServer:
    """Route every AnthropicProvider build_model makes to a fake server."""
    server = ClaudeServer(replies)
    keys: list[str] = []

    def provider(*, api_key):
        keys.append(api_key)
        return server.provider(api_key)

    monkeypatch.setattr(models, "AnthropicProvider", provider)
    server.keys = keys
    return server


def thinking(text: str, model: str = "claude-opus-5-5") -> ModelResponse:
    return ModelResponse(
        parts=[
            ThinkingPart(
                content=text, signature=f"sig-{text}", provider_name="anthropic"
            ),
            TextPart(f"answer after {text}"),
        ],
        model_name=model,
        provider_name="anthropic",
    )


def turn(prompt: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(prompt)])


def kinds(message: ModelResponse) -> list[str]:
    return [part.part_kind for part in message.parts]


# --- routing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model_id", ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
)
def test_claude_ids_go_to_anthropic_on_the_claude_key(
    monkeypatch, no_keys, model_id
) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "from-claude-name")
    server = serve(monkeypatch)

    model = models.build_model(model_id)

    assert isinstance(model, AnthropicModel)
    assert model.model_name == model_id
    assert server.keys == ["from-claude-name"]
    assert model.profile["supports_forced_tool_choice"] is False
    assert model.settings == {
        "anthropic_cache_tool_definitions": "1h",
        "anthropic_cache_instructions": "1h",
        "anthropic_cache_messages": "1h",
    }


def test_anthropic_api_key_wins_over_claude_api_key(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "canonical")
    monkeypatch.setenv("CLAUDE_API_KEY", "fallback")
    server = serve(monkeypatch)

    models.build_model("claude-opus-5-5")

    assert server.keys == ["canonical"]


def test_openrouter_claude_ids_are_refused(no_keys) -> None:
    with pytest.raises(ValueError, match="claude-opus-5-5"):
        models.build_model("anthropic/claude-sonnet-5.5")


def test_bare_non_claude_ids_still_go_to_zen(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen")

    assert not isinstance(models.build_model("kimi-k3"), AnthropicModel)


# --- the request Opus 5.5 receives ------------------------------------------


class Decisions(BaseModel):
    keep: list[str]


async def test_structured_output_is_cached_and_never_forces_a_tool(
    monkeypatch, no_keys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server = serve(
        monkeypatch,
        tool_reply(
            "final_result",
            {"keep": ["w1"]},
            input_tokens=12,
            cache_read=2_000,
            cache_write_1h=3_000,
        ),
    )
    agent = Agent(
        models.build_model("claude-opus-5-5"),
        output_type=Decisions,
        system_prompt="You decide which watch instructions to keep.",
    )

    @agent.tool_plain
    def list_channels() -> str:
        """List the enabled channels."""
        return "#general"

    result = await agent.run("w1: watch for release questions")

    assert result.output == Decisions(keep=["w1"])
    (body,) = server.requests
    # Opus, Sonnet and Haiku 5.5 answer "any"/"tool" with a 400.
    assert body.get("tool_choice", {"type": "auto"})["type"] == "auto"
    # One 1-hour breakpoint each on the tools, the system prompt and the
    # newest message, inside Anthropic's limit of four.
    assert body["tools"][-1]["cache_control"] == ONE_HOUR
    assert body["system"][-1]["cache_control"] == ONE_HOUR
    assert body["messages"][-1]["content"][-1]["cache_control"] == ONE_HOUR
    assert cache_controls(body) == [ONE_HOUR] * 3
    # Opus 5.5 rejects sampling parameters and disabled thinking.
    assert not {"temperature", "top_p", "top_k"} & body.keys()
    assert body.get("thinking", {}).get("type") != "disabled"
    assert "budget_tokens" not in body.get("thinking", {})
    # Cache reads and writes are both inside input_tokens.
    assert usage_dict(result.usage) == {
        "input_tokens": 5_012,
        "output_tokens": 5,
        "cache_read_tokens": 2_000,
        "cache_write_tokens": 3_000,
    }


async def test_a_wake_reports_its_cache_writes(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    serve(monkeypatch, text_reply("nothing to do", cache_write_1h=4_096))
    runner = KimiAgentRunner(
        agent=Agent(models.build_model("claude-opus-5-5"), system_prompt="guild"),
        summarize=AsyncMock(),
    )

    _, usage = await runner.wake("brief", deps=None)

    assert usage["cache_write_tokens"] == 4_096


# --- thinking blocks ---------------------------------------------------------


def test_strip_keeps_text_tool_calls_and_other_providers_thinking() -> None:
    gemini = ModelResponse(
        parts=[
            ThinkingPart(content="g", provider_name="google-gla"),
            ToolCallPart("lookup", {"id": "1"}),
        ]
    )
    history = [turn("a"), thinking("one"), turn("b"), gemini]

    stripped = strip_claude_thinking(history)

    assert kinds(stripped[1]) == ["text"]
    assert stripped[3] is gemini
    assert kinds(history[1]) == ["thinking", "text"]  # input untouched


def test_strip_returns_the_same_list_when_nothing_changes() -> None:
    history = [turn("a"), ModelResponse(parts=[TextPart("x")])]

    assert strip_claude_thinking(history) is history


def test_strip_from_an_index_leaves_earlier_blocks() -> None:
    history = [turn("a"), thinking("one"), turn("b"), thinking("two")]

    stripped = strip_claude_thinking(history, start=2)

    assert kinds(stripped[1]) == ["thinking", "text"]
    assert kinds(stripped[3]) == ["text"]


async def test_compaction_strips_thinking_from_the_kept_tail() -> None:
    history = [turn("a"), thinking("one"), turn("b" * 4000), thinking("two")]

    compacted = await compact_agent_history(
        history,
        token_limit=10,
        summarize=AsyncMock(return_value="note"),
        keep_messages=2,
    )

    assert "COMPACTION MEMORY NOTE" in compacted[0].parts[0].content
    assert compacted[-1].parts == [TextPart("answer after two")]


async def test_the_summarizer_never_receives_claude_thinking(
    monkeypatch, no_keys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server = serve(monkeypatch, text_reply("memory note"))
    history = [
        turn("a"),
        ModelResponse(
            parts=[
                ThinkingPart(content="t", signature="sig", provider_name="anthropic"),
                ToolCallPart("lookup", {"id": "1"}, tool_call_id="c1"),
            ],
            model_name="claude-opus-5-5",
            provider_name="anthropic",
        ),
        ModelRequest(parts=[ToolReturnPart("lookup", "found", tool_call_id="c1")]),
        thinking("two"),
    ]

    note, _ = await self_compaction_summary(
        models.build_model("claude-opus-5-5"), history
    )

    assert note == "memory note"
    assert thinking_blocks(server.requests[0]) == []
    assert kinds(history[1]) == ["thinking", "tool-call"]  # stored copy intact


def test_a_model_switch_drops_the_earlier_models_leading_blocks() -> None:
    history = [
        turn("a"),
        thinking("old", model="claude-sonnet-5-5"),
        turn("b"),
        thinking("new", model="claude-opus-5-5"),
    ]

    kept = drop_other_models_claude_thinking(history, "claude-opus-5-5")

    assert kinds(kept[1]) == ["text"]
    assert kept[3] is history[3]


def test_same_model_history_is_left_alone() -> None:
    history = [turn("a"), thinking("one"), turn("b"), thinking("two")]

    assert drop_other_models_claude_thinking(history, "claude-opus-5-5") is history


async def test_a_wake_on_a_new_claude_model_sends_no_foreign_thinking(
    monkeypatch, no_keys
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    server = serve(monkeypatch, text_reply("ok"))
    runner = KimiAgentRunner(
        agent=Agent(models.build_model("claude-opus-5-5"), system_prompt="guild"),
        summarize=AsyncMock(),
        history=[turn("a"), thinking("old", model="claude-sonnet-5-5")],
    )

    await runner.wake("brief", deps=None)

    assert thinking_blocks(server.requests[0]) == []


def test_blanking_an_opted_out_member_drops_thinking_after_the_edit() -> None:
    before = thinking("before")
    history = [
        turn("first wake"),
        before,
        turn("[id=1] <@999123456789012345> hello"),
        thinking("after"),
    ]
    runner = SimpleNamespace(history=history)
    guild = GuildRuntime(
        guild_id="111",
        engine=SimpleNamespace(agent_runner=runner),
        history_repository=None,
        history_writer=None,
        api=None,
        discord=None,
        redis=None,
        queue=None,
        journal=None,
        bot_user_id="bot",
        guild_name="guild",
        summarize_web=None,
        image_capabilities=None,
        media_reader=None,
        author_handler=None,
        blocked_users=SimpleNamespace(
            is_blocked=lambda user_id, message_id=None: user_id == "999123456789012345"
        ),
    )

    guild._scrub_runner_history()

    scrubbed = runner.history
    assert "999123456789012345" not in scrubbed[2].parts[0].content
    assert kinds(scrubbed[1]) == ["thinking", "text"]
    assert kinds(scrubbed[3]) == ["text"]
    # Round-trips like any stored history.
    ModelMessagesTypeAdapter.validate_python(
        ModelMessagesTypeAdapter.dump_python(scrubbed)
    )
