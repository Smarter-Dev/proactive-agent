"""A wake's usage report carries each provider request, its capabilities'
spend, and what a failed wake was billed for (#211).

The wake runs through the real GuildRuntime, AgentEngine and KimiAgentRunner;
only the models are FunctionModels answering with known per-request usage, and
the bot API is the in-memory FakeAPI that keeps every usage report.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import fakeredis.aioredis
import jsonschema
import pytest
from purge_fakes import CHANNEL, GUILD, FakeAPI, batch, make_runtime
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from proactive_agent.agent import KimiAgentRunner
from proactive_agent.capabilities import WebSummarizer
from proactive_agent.contracts import NotificationEnvelope, TokenUsage
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.parity import ProactiveDeps

AGENT = "claude-opus-5-5"
SUMMARIZER = "claude-haiku-5-5"
FIRST = RequestUsage(input_tokens=120_000, output_tokens=40, cache_read_tokens=100_000)
SECOND = RequestUsage(input_tokens=121_000, output_tokens=60, cache_write_tokens=900)
SUMMARY = RequestUsage(input_tokens=150_000, output_tokens=300)


def request(usage: RequestUsage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def agent_model(*, fail_second: bool = False) -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart("read_page", {"url": "https://example.test"})],
                usage=FIRST,
            )
        if fail_second:
            raise ModelHTTPError(500, AGENT)
        return ModelResponse(parts=[TextPart("done")], usage=SECOND)

    return FunctionModel(respond, model_name=AGENT)


def summarizer_model() -> FunctionModel:
    return FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[TextPart("a summary")], usage=SUMMARY
        ),
        model_name=SUMMARIZER,
    )


def build_agent(model: FunctionModel) -> Agent:
    agent = Agent(model, deps_type=ProactiveDeps)

    @agent.tool
    async def read_page(ctx: RunContext[ProactiveDeps], url: str) -> str:
        return await ctx.deps.summarize_web(
            instruction="summarize", content="page text", title="t", url=url
        )

    return agent


@pytest.fixture
def runtime_and_api():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = FakeAPI()
    api.addenda[GUILD] = {CHANNEL: ""}
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.summarize_web = WebSummarizer(summarizer_model())
    runtime.image_capabilities = SimpleNamespace(review=None, generate=None)

    def use(model: FunctionModel):
        runtime.engine = AgentEngine(
            agent_runner=KimiAgentRunner(
                agent=build_agent(model), summarize=None, model_id=AGENT
            ),
            skim=SkimRunner(summarizer_model()),
            agent_model_id=AGENT,
            skim_model_id=SUMMARIZER,
            deps_factory=None,
        )
        return runtime

    yield use, api, writer


async def test_a_wake_reports_each_request_and_its_capabilities(runtime_and_api):
    use, api, writer = runtime_and_api
    wake = batch()

    await use(agent_model()).process(wake)
    await writer.close(timeout=1)

    [report] = api.usage_reports
    entries = {entry["model_id"]: entry for entry in report["entries"]}
    assert entries[AGENT]["requests"] == [request(FIRST), request(SECOND)]
    assert entries[AGENT]["input_tokens"] == FIRST.input_tokens + SECOND.input_tokens
    assert entries[AGENT]["cache_write_tokens"] == SECOND.cache_write_tokens
    # The web summarizer a tool called is billed with the wake.
    assert entries[SUMMARIZER]["requests"] == [request(SUMMARY)]
    assert entries[SUMMARIZER]["input_tokens"] == SUMMARY.input_tokens
    assert {entry["operation"] for entry in report["entries"]} == {"agent"}
    # One attempt's own id, under the wake's.
    assert report["wake_id"].startswith(f"{wake.wake_id}:")
    assert len(report["wake_id"]) <= 64


async def test_a_wake_that_fails_reports_what_the_provider_billed(runtime_and_api):
    use, api, writer = runtime_and_api
    runtime = use(agent_model(fail_second=True))

    with pytest.raises(ModelHTTPError):
        await runtime.process(batch())
    with pytest.raises(ModelHTTPError):
        await runtime.process(batch())
    await writer.close(timeout=1)

    first, retry = api.usage_reports
    for report in (first, retry):
        entries = {entry["model_id"]: entry for entry in report["entries"]}
        assert entries[AGENT]["requests"] == [request(FIRST)]
        assert entries[SUMMARIZER]["requests"] == [request(SUMMARY)]
        assert report["responses"] == 0
    # The retry ran the model again: its report must not replay the first's.
    assert first["wake_id"] != retry["wake_id"]
    assert runtime.engine.agent_runner.unreported_usage == {}


async def test_watcher_usage_with_requests_crosses_the_contract():
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "contracts/proactive/v1/notification.schema.json"
        ).read_text()
    )
    usage = {
        "input_tokens": 3,
        "output_tokens": 1,
        "cache_read_tokens": 0,
        "requests": [request(RequestUsage(input_tokens=3, output_tokens=1))],
    }
    envelope = NotificationEnvelope(
        schema_version=1,
        notification_id="00000000-0000-4000-8000-000000000001",
        guild_id=GUILD,
        channel_id=CHANNEL,
        channel_name="general",
        kind="mention",
        created_at="2026-10-18T12:00:00Z",
        body="hi",
        message_ids=(),
        wakes=True,
        passive=False,
        watcher_usage={"typesafe:jev-1.13.0": usage},
        trace_id="00000000-0000-4000-8000-000000000002",
    )
    jsonschema.validate(envelope.model_dump(mode="json"), schema)
    # An old bot's usage, without requests, still validates.
    assert TokenUsage(input_tokens=1).requests is None
    with pytest.raises(ValueError):
        TokenUsage(input_tokens=1, unknown=2)
