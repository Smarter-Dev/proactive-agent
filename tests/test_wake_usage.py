"""Token counts the provider sent must reach the usage ledger (#102).

pydantic-ai 1.107 with genai-prices 0.1.5 dropped every Gemini and GLM count
into ``details`` and reported 0 input / 0 output for each wake. These tests
run the production model classes from ``build_model`` against provider
responses served at the HTTP transport, so a library bump that loses the
counts again fails here before it reaches billing.

The fixtures are hand-built in the wire shape of a one-word "Reply OK" call
(Gemini generateContent and an OpenRouter chat completion); their token
fields are the ones the worker pod saw in ``details``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import fakeredis.aioredis
import httpx
import httpx2
import pytest
from purge_fakes import CHANNEL, GUILD, FakeAPI, batch, make_runtime
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from proactive_agent.agent import KimiAgentRunner, self_compaction_summary, usage_dict
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.models import build_model
from proactive_agent.parity import build_proactive_agent

FIXTURES = Path(__file__).parent / "fixtures"
GEMINI = "gemini-3.8-flash"
GLM = "z-ai/glm-5.3-flash"
# input, output: Gemini bills thoughts as output (1 + 69 thoughts).
EXPECTED = {GEMINI: (8, 70), GLM: (12, 38)}


@pytest.fixture
def providers(monkeypatch):
    """Serve the recorded responses to every outgoing model request.

    Both transports are patched: SDKs on legacy httpx must not reach the
    network either."""
    requests: list[str] = []

    def respond(host: str):
        requests.append(host)
        if host == "generativelanguage.googleapis.com":
            return (FIXTURES / "gemini_generate_content.json").read_bytes()
        if host == "openrouter.ai":
            return (FIXTURES / "openrouter_chat_completion.json").read_bytes()
        raise AssertionError(f"unexpected model request to {host}")

    for client in (httpx, httpx2):

        async def handle(self, request, client=client):
            return client.Response(
                200,
                content=respond(request.url.host),
                headers={"content-type": "application/json"},
            )

        monkeypatch.setattr(client.AsyncHTTPTransport, "handle_async_request", handle)
    monkeypatch.delenv("LITELLM_ENDPOINT", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    return requests


@pytest.mark.parametrize("model_id", [GEMINI, GLM])
async def test_a_reply_ok_call_reports_its_token_counts(providers, model_id):
    result = await Agent(build_model(model_id), output_type=str).run("Reply OK")

    assert result.output == "OK"
    assert providers, "the call never reached the fixture transport"
    usage = usage_dict(result.usage)
    assert (usage["input_tokens"], usage["output_tokens"]) == EXPECTED[model_id]


async def test_the_test_model_reports_tokens():
    result = await Agent(TestModel(), output_type=str).run("Reply OK")

    usage = usage_dict(result.usage)
    assert usage["input_tokens"] > 0 and usage["output_tokens"] > 0


@pytest.mark.parametrize("model_id", [GEMINI, GLM])
async def test_self_compaction_reports_its_token_counts(providers, model_id):
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart("hi")])]

    summary, usage = await self_compaction_summary(build_model(model_id), history)

    assert summary == "OK"
    assert (usage["input_tokens"], usage["output_tokens"]) == EXPECTED[model_id]


@pytest.mark.parametrize("model_id", [GEMINI, GLM])
async def test_a_wake_never_records_zero_tokens_the_provider_reported(
    providers, model_id
):
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.engine = AgentEngine(
        agent_runner=KimiAgentRunner(
            agent=build_proactive_agent(build_model(model_id), system_prompt="s"),
            summarize=None,
        ),
        skim=SkimRunner(build_model(model_id)),
        agent_model_id=model_id,
        skim_model_id=model_id,
        deps_factory=None,
    )
    runtime.image_capabilities = SimpleNamespace(review=None, generate=None)

    await runtime.process(batch())
    await writer.close(timeout=1)

    assert providers, "the wake never called the model"
    [report] = api.usage_reports
    [entry] = report["entries"]
    assert entry["model_id"] == model_id and entry["operation"] == "agent"
    assert (entry["input_tokens"], entry["output_tokens"]) == EXPECTED[model_id]


async def test_the_fixtures_carry_the_counts_the_guard_expects():
    # The guard is only as good as its fixtures: if a fixture lost its usage
    # block, a zero-token wake would match a zero expectation.
    gemini = json.loads((FIXTURES / "gemini_generate_content.json").read_text())
    glm = json.loads((FIXTURES / "openrouter_chat_completion.json").read_text())
    assert gemini["usageMetadata"]["promptTokenCount"] == EXPECTED[GEMINI][0]
    assert glm["usage"]["prompt_tokens"] == EXPECTED[GLM][0]
    assert all(min(counts) > 0 for counts in EXPECTED.values())
