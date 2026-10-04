"""Rewrite checks privacy:v1: the only checks on a purge rewrite.

1. A non-empty input must come back non-empty, else the step fails.
2. The output must not hold the user id: asked again, then the step fails.
3. A listed name left after the re-ask is stored and reported.
4. A model timeout or error fails the step.
A failing step leaves the stored bytes untouched and acks `failed`.
"""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest
from purge_fakes import BYSTANDER, CHANNEL, GUILD, TARGET, TARGET_NAME
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from test_purge import Replica, command, stored_v1, submit

import proactive_agent.agent as agent_module
from proactive_agent.agent import PrivacyCompactionError, privacy_watch_decisions


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


def note_model(note, calls: list):
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.output_tools:  # the watch-instruction step: keep everything
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, {"decisions": []})]
            )
        calls.append(len(calls))
        if isinstance(note, BaseException):
            raise note
        if note == "sleep":
            await asyncio.sleep(1)
        return ModelResponse(parts=[TextPart(note)])

    return FunctionModel(respond)


async def run_purge(redis_client, api, model):
    replica = Replica(redis_client, api, model, name="a")
    v1_before = await stored_v1(redis_client)
    durable_before = api.durable[GUILD]
    await replica.consumer.initialize()
    await submit(redis_client, command())
    await replica.consumer.poll_once()
    await replica.writer.close(timeout=1)
    return v1_before, durable_before


async def assert_failed_untouched(redis_client, api, before, detail):
    v1_before, durable_before = before
    assert await stored_v1(redis_client) == v1_before
    assert api.durable[GUILD] == durable_before
    assert api.acks[0]["outcome"] == "failed"
    assert api.acks[0]["detail"] == detail


async def test_check1_an_empty_rewrite_fails_untouched(redis_client, world):
    calls: list = []
    before = await run_purge(redis_client, world, note_model("   ", calls))
    assert len(calls) == 1  # fails at once, no retry
    await assert_failed_untouched(redis_client, world, before, "purge: empty_output")


async def test_check2_the_user_id_is_asked_again_then_fails(redis_client, world):
    calls: list = []
    model = note_model(f"remember <@{TARGET}> likes cats", calls)
    before = await run_purge(redis_client, world, model)
    assert len(calls) == 3  # two retries
    await assert_failed_untouched(redis_client, world, before, "purge: target_id")


async def test_check3_a_name_left_after_the_retry_is_stored_and_reported(
    redis_client, world
):
    calls: list = []
    note = f"{TARGET_NAME.upper()} liked Rust; nia ships Rust 1.95 on friday."
    await run_purge(redis_client, world, note_model(note, calls))
    assert len(calls) == 2  # one re-ask about names
    ack = world.acks[0]
    assert ack["outcome"] == "purged"
    assert ack["name_hits"]["history"] == 1
    assert (
        note
        in json.loads(await stored_v1(redis_client))["history"][0]["parts"][0][
            "content"
        ]
    )


async def test_check4_a_timeout_fails_untouched(redis_client, world, monkeypatch):
    monkeypatch.setattr(agent_module, "PRIVACY_MODEL_TIMEOUT_SECONDS", 0.05)
    before = await run_purge(redis_client, world, note_model("sleep", []))
    await assert_failed_untouched(redis_client, world, before, "purge: timeout")


async def test_check4_a_model_error_fails_untouched(redis_client, world):
    before = await run_purge(
        redis_client, world, note_model(RuntimeError("provider down"), [])
    )
    await assert_failed_untouched(redis_client, world, before, "purge: RuntimeError")


def watch_model(decisions_per_call: list[list[dict]], calls: list):
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        decisions = decisions_per_call[min(len(calls), len(decisions_per_call) - 1)]
        calls.append(len(calls))
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {"decisions": decisions})]
        )

    return FunctionModel(respond)


ENTRIES = {"w1": f"wake when {TARGET_NAME} reports back", "w2": "watch nia's release"}


async def test_watch_check1_an_empty_rewrite_fails():
    model = watch_model(
        [[{"instruction_id": "w1", "action": "rewrite", "text": " "}]], []
    )
    with pytest.raises(PrivacyCompactionError, match="empty_output"):
        await privacy_watch_decisions(model, ENTRIES, user_id=TARGET, names=["kai"])


async def test_watch_check1_an_explicit_drop_is_allowed():
    model = watch_model([[{"instruction_id": "w1", "action": "drop"}]], [])
    outcome, hits = await privacy_watch_decisions(
        model, ENTRIES, user_id=TARGET, names=["kai"]
    )
    assert outcome == {"w1": None, "w2": "watch nia's release"}
    assert hits == 0


async def test_watch_check2_the_user_id_is_asked_again_then_fails():
    calls: list = []
    leak = [{"instruction_id": "w1", "action": "rewrite", "text": f"ping <@{TARGET}>"}]
    with pytest.raises(PrivacyCompactionError, match="target_id"):
        await privacy_watch_decisions(
            watch_model([leak], calls), ENTRIES, user_id=TARGET, names=[]
        )
    assert len(calls) == 3


async def test_watch_check3_a_name_left_after_the_retry_is_reported():
    calls: list = []
    outcome, hits = await privacy_watch_decisions(
        watch_model([[]], calls), ENTRIES, user_id=TARGET, names=["kai"]
    )
    assert len(calls) == 2
    assert hits == 1
    assert outcome == ENTRIES


async def test_watch_check4_a_timeout_fails(monkeypatch):
    monkeypatch.setattr(agent_module, "PRIVACY_MODEL_TIMEOUT_SECONDS", 0.05)

    async def slow(messages, info):
        await asyncio.sleep(1)

    with pytest.raises(PrivacyCompactionError, match="timeout"):
        await privacy_watch_decisions(
            FunctionModel(slow), ENTRIES, user_id=TARGET, names=[]
        )


def test_constants_used():
    assert BYSTANDER and CHANNEL
