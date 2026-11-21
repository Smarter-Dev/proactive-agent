"""Internal model defaults: Claude unless Claude cannot do the job (#490)."""

from __future__ import annotations

import json

import httpx2 as httpx
import pytest
from anthropic import AsyncAnthropic
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from proactive_agent.capabilities import (
    HandlerAuthor,
    HandlerPlan,
    MediaReader,
    _thinking_settings,
)
from proactive_agent.config import Settings
from proactive_agent.credentials import encode_credential
from proactive_agent.models import build_model


def _keys(monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_ENDPOINT", raising=False)
    monkeypatch.setenv("KMSVP8", encode_credential("sk-ant-test"))
    monkeypatch.setenv("IN8OID", encode_credential("gemini-test"))


def test_defaults_are_claude_except_what_claude_cannot_do() -> None:
    s = Settings(_env_file=None)
    assert s.proactive_handler_model == "claude-opus-5-5"
    assert s.proactive_image_reviewer_model == "claude-haiku-5-5"
    # Audio: Claude cannot hear it.
    assert s.proactive_media_reader_model == "gemini-3.5-flash-lite"
    # Gemini is unfunded, so the summarizer fallback is Claude too.
    assert s.proactive_web_summarizer_model.startswith("claude-")
    assert s.proactive_web_summarizer_fallback_model == "claude-sonnet-5-5"
    assert s.proactive_media_reader_image_model == "claude-haiku-5-5"


def test_default_models_route_to_their_providers(monkeypatch) -> None:
    _keys(monkeypatch)
    s = Settings(_env_file=None)
    assert isinstance(build_model(s.proactive_handler_model), AnthropicModel)
    assert isinstance(build_model(s.proactive_image_reviewer_model), AnthropicModel)
    assert isinstance(build_model(s.proactive_media_reader_model), GoogleModel)


def test_thinking_settings_follow_the_provider(monkeypatch) -> None:
    _keys(monkeypatch)
    claude = _thinking_settings(build_model("claude-opus-5-5"), "MEDIUM")
    assert claude["anthropic_thinking"] == {"type": "adaptive"}
    assert claude["anthropic_effort"] == "medium"
    assert "budget_tokens" not in str(claude)
    gemini = _thinking_settings(build_model("gemini-3.5-flash-lite"), "LOW")
    assert gemini["google_thinking_config"] == {"thinking_level": "LOW"}


def test_thinking_settings_keep_effort_on_openai_dialect_models() -> None:
    model = OpenAIChatModel("gpt-x", provider=OpenAIProvider(api_key="k"))
    assert _thinking_settings(model, "LOW")["openai_reasoning_effort"] == "low"


def _capturing_claude(sent: list[dict]) -> AnthropicModel:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            500, json={"type": "error", "error": {"type": "api_error", "message": "x"}}
        )

    client = AsyncAnthropic(
        api_key="k",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return AnthropicModel(
        "claude-opus-5-5", provider=AnthropicProvider(anthropic_client=client)
    )


def _schema_properties(node) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            found |= set(props)
        for v in node.values():
            found |= _schema_properties(v)
    elif isinstance(node, list):
        for v in node:
            found |= _schema_properties(v)
    return found


@pytest.mark.anyio
async def test_claude_handler_author_request_keeps_the_settings_fields() -> None:
    sent: list[dict] = []
    author = HandlerAuthor(api=None, model=_capturing_claude(sent), discord=None)
    with pytest.raises(ModelHTTPError):
        await author._author.run("make a daily digest")
    assert sent, "no request captured"
    assert sent[0]["thinking"] == {"type": "adaptive"}
    props = _schema_properties(sent[0])
    for field in (
        "interval_seconds",
        "daily_time",
        "start_at",
        "delay_seconds",
        "fire_at",
        "include_bot_messages",
    ):
        assert field in props, field


def test_handler_settings_round_trip_without_unset_fields() -> None:
    plan = HandlerPlan(feasible=True, settings={"daily_time": "15:00"})
    assert plan.settings.to_dict() == {"daily_time": "15:00"}


def test_claude_handler_agents_get_extra_output_retries(monkeypatch) -> None:
    _keys(monkeypatch)
    model = build_model("claude-opus-5-5")
    author = HandlerAuthor(api=None, model=model, discord=None)
    assert author._author._max_output_retries == 3
    assert author._judge._max_output_retries == 3


def test_media_reader_routes_images_to_claude_and_audio_to_gemini(monkeypatch) -> None:
    _keys(monkeypatch)
    reader = MediaReader(
        build_model("gemini-3.5-flash-lite"),
        image_model=build_model("claude-haiku-5-5"),
    )
    assert isinstance(reader._image_agent.model, AnthropicModel)
    assert isinstance(reader._agent.model, GoogleModel)
