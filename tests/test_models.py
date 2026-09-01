from __future__ import annotations

from types import SimpleNamespace

from pydantic_ai.models.openai import OpenAIChatModel

from proactive_agent.agent import usage_dict
from proactive_agent.models import build_model


def test_litellm_proxy_routes_both_model_families(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_ENDPOINT", "https://proxy.example.test")
    monkeypatch.setenv("LITELLM_API_KEY", "secret")

    for model_id in ("glm-5.3-flash", "gemini/gemini-3.7-flash"):
        model = build_model(model_id)
        assert isinstance(model, OpenAIChatModel)
        assert model.model_name == model_id
        assert str(model.base_url) == "https://proxy.example.test/v1/"


def test_usage_dict_accepts_method_and_property_result_apis() -> None:
    usage = SimpleNamespace(
        input_tokens=3,
        output_tokens=2,
        cache_read_tokens=1,
    )

    assert (
        usage_dict(usage)
        == usage_dict(lambda: usage)
        == {
            "input_tokens": 3,
            "output_tokens": 2,
            "cache_read_tokens": 1,
        }
    )
