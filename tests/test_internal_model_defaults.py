"""Internal model defaults: Claude unless Claude cannot do the job (#490)."""

from __future__ import annotations

from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel

from proactive_agent.capabilities import _thinking_settings
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
    # Primary summarizer is Claude, so the fallback is another vendor.
    assert s.proactive_web_summarizer_model.startswith("claude-")
    assert s.proactive_web_summarizer_fallback_model == "gemini-3.5-flash-lite"


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
