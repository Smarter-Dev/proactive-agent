"""Provider routing for the standalone agent and skim models."""

from __future__ import annotations

import os

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

# Prompt caching on every Claude request: the tool definitions, the system
# prompt and the newest message, three of Anthropic's four breakpoints, all on
# the 1-hour TTL. The guild system prompt and tool set never change and the
# history is append-only, so each wake reads back everything up to its new
# brief. Wakes come a median 15 minutes apart and 62% within an hour, so a
# 5-minute entry on the history would usually be gone by the next wake. One
# TTL also makes every cache write a 1-hour write, which is how smarter-dev's
# llm_pricing prices them.
ANTHROPIC_CACHE_TTL = "1h"
ANTHROPIC_CACHE_SETTINGS = AnthropicModelSettings(
    anthropic_cache_tool_definitions=ANTHROPIC_CACHE_TTL,
    anthropic_cache_instructions=ANTHROPIC_CACHE_TTL,
    anthropic_cache_messages=ANTHROPIC_CACHE_TTL,
)


def anthropic_api_key() -> str:
    """The direct Claude API key: ANTHROPIC_API_KEY, else CLAUDE_API_KEY."""
    return os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""


def build_claude_model(model_id: str) -> AnthropicModel:
    """A Claude model on Anthropic's own API, cached and never tool-forced.

    Opus, Sonnet and Haiku 5.5 answer a forced tool_choice ("any"/"tool")
    with a 400, so structured output (WatchDecisions, HandlerPlan, ...) is
    offered its output tool on "auto" for every Claude id. Thinking is left at
    the model's default: adaptive on the 5.5 generation, which rejects an
    explicit disabled setting and sampling parameters alike, and nothing here
    sends either.
    """
    provider = AnthropicProvider(api_key=anthropic_api_key())
    return AnthropicModel(
        model_id,
        provider=provider,
        profile=merge_profile(
            provider.model_profile(model_id),
            ModelProfile(supports_forced_tool_choice=False),
        ),
        settings=ANTHROPIC_CACHE_SETTINGS,
    )


def reasoning_settings(
    model: Model, effort: str, *, cache_messages: bool = True
) -> ModelSettings:
    """Reasoning ``effort`` in ``model``'s own dialect.

    A Claude model gets adaptive thinking at that effort, never an OpenAI
    ``openai_reasoning_effort`` it would not read. ``cache_messages=False`` is
    for an agent that answers in one request (a web summary): it drops the
    message cache breakpoint, which would only bill the prompt at the 1-hour
    write rate; the system-prompt breakpoint stays.
    """
    if isinstance(model, AnthropicModel):
        settings = AnthropicModelSettings(
            anthropic_thinking={"type": "adaptive"}, anthropic_effort=effort
        )
        if not cache_messages:
            settings["anthropic_cache_messages"] = False
        return settings
    return OpenAIChatModelSettings(openai_reasoning_effort=effort)


def build_model(model_id: str) -> Model:
    """Build the same provider class used by the integrated proactive bot."""
    # Claude first: Anthropic serves it directly, with the funded key and
    # 1-hour prompt caching, even when a LiteLLM proxy is configured.
    if model_id.startswith("claude-"):
        return build_claude_model(model_id)
    if model_id.startswith("anthropic/"):
        # OpenRouter's Claude ids would skip the direct key and its caching,
        # and price as OpenRouter rows. Fail at startup with the fix instead.
        raise ValueError(
            f"{model_id!r}: Claude models are served directly by Anthropic; "
            "use the direct id (e.g. claude-opus-5-5)"
        )
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
