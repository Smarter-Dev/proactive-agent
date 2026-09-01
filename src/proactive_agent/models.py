"""Provider routing for the standalone agent and skim models."""

from __future__ import annotations

import os

from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider


def build_model(model_id: str) -> Model:
    """Build the same provider class used by the integrated proactive bot."""
    litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "").rstrip("/")
    litellm_api_key = os.getenv("LITELLM_API_KEY", "")
    if litellm_endpoint and litellm_api_key:
        if not litellm_endpoint.endswith("/v1"):
            litellm_endpoint += "/v1"
        return OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(
                base_url=litellm_endpoint,
                api_key=litellm_api_key,
            ),
            profile=OpenAIModelProfile(
                openai_supports_tool_choice_required=False,
                openai_chat_supports_multiple_system_messages=False,
            ),
        )
    if model_id.startswith("gemini-"):
        return GoogleModel(
            model_id,
            provider=GoogleProvider(
                api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
            ),
        )
    if model_id == "kimi-k3" or "/" not in model_id:
        return OpenAIChatModel(
            model_id,
            provider=OpenAIProvider(
                base_url=os.getenv(
                    "OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1"
                ),
                api_key=os.getenv("OPENCODE_ZEN_API_KEY") or "",
            ),
            profile=OpenAIModelProfile(
                openai_supports_tool_choice_required=False,
                openai_chat_supports_multiple_system_messages=False,
            ),
        )
    return OpenAIChatModel(
        model_id,
        provider=OpenRouterProvider(
            api_key=os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPEN_ROUTER_API_KEY")
            or os.getenv("OPEN_ROUTER")
            or ""
        ),
    )
