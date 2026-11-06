"""The agent-run nightly dream (smarter-dev #454)."""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest
from purge_fakes import GUILD, FakeAPI, dump, raw_history_with_target
from pydantic_ai.messages import ModelResponse, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent import dream as dream_module
from proactive_agent.dream import (
    DreamAnswer,
    DreamService,
    dream_instructions,
    submission,
)
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.keys import (
    dream_attempts_key,
    dream_cooldown_key,
    history_key,
    lease_key,
    ownership_key,
)
from proactive_agent.queue import RedisWakeQueue

INPUTS = {
    "guild_id": GUILD,
    "cutoff": "2026-11-05T00:00:00Z",
    "needed": True,
    "base_revision": 3,
    "inputs_digest": "d1",
    "notes": [{"id": "n1", "line": "00:10Z #dev — the shader shipped"}],
    "user_message": "# Today\n00:10Z #dev — the shader shipped",
}


class DreamAPI(FakeAPI):
    def __init__(self, verdicts):
        super().__init__()
        self.verdicts = list(verdicts)
        self.submitted: list[dict] = []
        self.inputs = dict(INPUTS)
        self.guilds = [GUILD]

    async def dream_guilds(self):
        return {"cutoff": INPUTS["cutoff"], "guild_ids": list(self.guilds)}

    async def dream_inputs(self, guild_id):
        return self.inputs

    async def submit_dream(self, guild_id, payload):
        self.submitted.append(payload)
        return self.verdicts.pop(0)


class Brain:
    """A model answering every turn with the same structured answer."""

    def __init__(self, **answer):
        self.answer = {"memory": "## Ongoing Context\nthe shader shipped"} | answer
        self.prompts: list[str] = []
        self.instructions: list[str | None] = []

    def model(self) -> FunctionModel:
        async def respond(messages, info: AgentInfo):
            self.prompts.append(str(messages[-1].parts[-1].content))
            self.instructions.append(
                info.instructions if hasattr(info, "instructions") else None
            )
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, self.answer)])

        return FunctionModel(respond)


@pytest.fixture
def redis_client():
    return fakeredis.aioredis.FakeRedis(decode_responses=False)


async def build(redis_client, verdicts, brain, **kwargs):
    api = DreamAPI(verdicts)
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api, debounce_seconds=0)
    queue = RedisWakeQueue(redis_client, consumer_name="test")
    await redis_client.set(ownership_key(GUILD), "external")

    async def summarize(messages):
        return "earlier summary", {}

    async def system_prompt_for(guild_id):
        return "GUILD SYSTEM PROMPT"

    kwargs.setdefault("clock", lambda: 2e9)  # long after INPUTS' cutoff
    service = DreamService(
        redis_client,
        queue,
        repository,
        writer,
        model=brain.model(),
        summarize=summarize,
        api=api,
        model_id="agent-model",
        system_prompt_for=system_prompt_for,
        **kwargs,
    )
    return service, api, writer


async def test_accepted_dream_is_submitted_with_what_it_read_and_kept_in_history(
    redis_client,
):
    brain = Brain(facts="Server is Pine.")
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    await writer.save(
        guild_id=GUILD, history=dump(raw_history_with_target()), previous_revision=0
    )

    assert await service.sweep_once() == {GUILD: "dreamed"}

    [sent] = api.submitted
    assert sent["facts"] == "Server is Pine."
    assert sent["base_revision"] == 3 and sent["note_ids"] == ["n1"]
    assert sent["cutoff"] == INPUTS["cutoff"]
    await writer.flush(GUILD)
    stored = json.loads(await redis_client.get(history_key(GUILD)))["history"]
    assert "the shader shipped" in json.dumps(stored)
    await writer.close(timeout=1)


async def test_a_refusal_goes_back_in_the_same_conversation(redis_client):
    brain = Brain()
    verdicts = [
        {
            "accepted": False,
            "reason": "invalid",
            "message": "`facts` is over its limit.",
        },
        {"accepted": True},
    ]
    service, api, writer = await build(redis_client, verdicts, brain)

    assert await service.sweep_once() == {GUILD: "dreamed"}

    assert len(api.submitted) == 2
    assert "`facts` is over its limit." in brain.prompts[1]
    await writer.close(timeout=1)


async def test_retries_are_capped_and_the_night_is_cooled_down(redis_client):
    brain = Brain()
    refusal = {"accepted": False, "reason": "invalid", "message": "no"}
    service, api, writer = await build(redis_client, [refusal] * 5, brain)

    assert await service.sweep_once() == {GUILD: "refused"}
    assert len(api.submitted) == 3  # first try plus two retries
    # The cooldown keeps the next tick away from the guild.
    assert await service.sweep_once() == {}
    assert len(api.submitted) == 3
    await writer.close(timeout=1)


async def test_a_stale_answer_rereads_the_night(redis_client):
    brain = Brain()
    verdicts = [
        {"accepted": False, "reason": "stale", "message": "moved"},
        {"accepted": True},
    ]
    service, api, writer = await build(redis_client, verdicts, brain)
    api.inputs = INPUTS | {"base_revision": 3}
    original = api.dream_inputs

    async def moving(guild_id):
        inputs = await original(guild_id)
        # The first read is the old night; the re-read carries a new digest.
        return inputs if not api.submitted else inputs | {"inputs_digest": "d2"}

    api.dream_inputs = moving

    assert await service.sweep_once() == {GUILD: "dreamed"}
    assert "has changed" in brain.prompts[1]
    assert [sent["inputs_digest"] for sent in api.submitted] == ["d1", "d2"]
    await writer.close(timeout=1)


async def test_a_busy_fence_leaves_the_guild_alone(redis_client):
    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    await redis_client.set(lease_key(GUILD), "someone-else", ex=60)

    outcomes = await service.sweep_once()

    assert outcomes == {GUILD: "fence busy"}
    assert api.submitted == [] and brain.prompts == []
    # A busy fence is not a failed night: no attempt, no cooldown.
    assert not await redis_client.exists(dream_cooldown_key(GUILD))
    assert not await redis_client.exists(dream_attempts_key(GUILD, INPUTS["cutoff"]))
    await writer.close(timeout=1)


async def test_a_quiet_guild_never_calls_the_model(redis_client):
    brain = Brain()
    service, api, writer = await build(redis_client, [], brain)
    api.inputs = {"needed": False, "reason": "no_notes"}

    assert await service.sweep_once() == {GUILD: "nothing to dream"}
    assert brain.prompts == []
    await writer.close(timeout=1)


async def test_disabled_service_does_nothing(redis_client):
    import asyncio

    brain = Brain()
    service, api, writer = await build(redis_client, [], brain, enabled=False)
    stop = asyncio.Event()
    task = asyncio.create_task(service.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await task

    assert brain.prompts == []
    await writer.close(timeout=1)


def test_submission_carries_the_inputs_not_the_models_claims():
    sent = submission(INPUTS, DreamAnswer(memory="m", facts=None))
    assert sent["facts"] is None and sent["note_ids"] == ["n1"]
    assert sent["base_revision"] == 3
    assert sent["inputs_digest"] == "d1"


def test_memory_block_renders_the_durable_blocks_ahead_of_the_blob():
    from proactive_agent.runtime import render_memory_block

    block = render_memory_block(
        {
            "personality": "Dry.",
            "behavior": "Wait to be asked.",
            "facts": "Server is Pine.",
            "content": "kai likes rust",
        },
        identity=True,
    )

    assert (
        block.index("PERSONALITY")
        < block.index("BEHAVIOR")
        < block.index("FACTS")
        < block.index("GUILD MEMORY")
    )
    assert "Server is Pine." in block


def test_facts_alone_make_a_memory_block():
    from proactive_agent.runtime import render_memory_block

    assert "Server is Pine." in render_memory_block({"facts": "Server is Pine."})


def test_personality_and_behavior_stay_out_until_the_flag_is_on():
    from proactive_agent.runtime import render_memory_block

    memory = {
        "personality": "Dry.",
        "behavior": "Wait to be asked.",
        "facts": "Server is Pine.",
        "content": "kai likes rust",
    }

    off = render_memory_block(memory)

    assert "Dry." not in off and "Wait to be asked." not in off
    assert "Server is Pine." in off and "kai likes rust" in off


async def test_a_guild_the_worker_does_not_own_is_left_alone(redis_client):
    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    await redis_client.delete(ownership_key(GUILD))

    assert await service.sweep_once() == {}
    assert api.submitted == [] and brain.prompts == []
    await writer.close(timeout=1)


async def test_an_empty_history_keeps_the_guild_system_prompt(redis_client):
    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)

    assert await service.sweep_once() == {GUILD: "dreamed"}

    assert "GUILD SYSTEM PROMPT" in (brain.instructions[0] or "")
    await writer.flush(GUILD)
    stored = json.loads(await redis_client.get(history_key(GUILD)))["history"]
    assert "SystemPromptPart" not in json.dumps(stored)
    assert "system-prompt" not in json.dumps(stored)
    await writer.close(timeout=1)


@pytest.mark.parametrize("reason", ["no_notes", "memory_disabled"])
async def test_an_unfixable_refusal_ends_the_night_without_an_attempt(
    redis_client, reason
):
    brain = Brain()
    verdicts = [{"accepted": False, "reason": reason, "message": ""}]
    service, api, writer = await build(redis_client, verdicts, brain)

    assert await service.sweep_once() == {GUILD: "nothing to dream"}

    assert len(api.submitted) == 1 and len(brain.prompts) == 1
    assert not await redis_client.exists(dream_cooldown_key(GUILD))
    await writer.close(timeout=1)


async def test_nothing_is_dreamed_before_the_cutoff(redis_client):
    brain = Brain()
    # 23:55 the night before: the web has snapped the cutoff to midnight.
    before = 1_762_300_800 - 300  # 2025-11-05T00:00Z minus five minutes
    service, api, writer = await build(
        redis_client,
        [{"accepted": True}],
        brain,
        clock=lambda: before,
    )
    api.inputs = INPUTS | {"cutoff": "2025-11-05T00:00:00Z"}

    assert await service.sweep_once() == {}
    assert brain.prompts == [] and api.submitted == []
    await writer.close(timeout=1)


async def test_a_set_stop_ends_the_sweep_between_guilds(redis_client):
    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    stop = asyncio.Event()
    stop.set()

    assert await service.sweep_once(stop) == {}
    assert brain.prompts == []
    await writer.close(timeout=1)


def test_the_prompt_is_rendered_with_the_webs_limits():
    text = dream_instructions({"personality": 111, "behavior": 222, "facts": 333})

    assert "at most 111 characters" in text
    assert "at most 222 characters" in text
    assert "at most 333 characters" in text
    assert "[[" not in text
    assert "at most 750 characters" in dream_instructions(None)


async def test_a_slow_run_times_out_keeps_finished_turns_and_reports_usage(
    redis_client, monkeypatch
):
    monkeypatch.setattr(dream_module, "RUN_TIMEOUT_SECONDS", 0.3)
    calls = 0

    async def respond(messages, info: AgentInfo):
        nonlocal calls
        calls += 1
        if calls > 1:
            await asyncio.sleep(5)
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {"memory": "## Ongoing Context\nfirst"})]
        )

    verdicts = [{"accepted": False, "reason": "invalid", "message": "too long"}]
    brain = Brain()
    service, api, writer = await build(redis_client, verdicts, brain)
    service._model = FunctionModel(respond)

    assert await service.sweep_once() == {GUILD: "timed out"}

    assert await redis_client.exists(dream_cooldown_key(GUILD))
    # The turn that finished before the timeout is billed and kept.
    assert len(api.submitted) == 1
    [report] = api.usage_reports
    assert report["guild_id"] == GUILD and report["entries"]
    await writer.flush(GUILD)
    stored = json.loads(await redis_client.get(history_key(GUILD)))["history"]
    assert "first" in json.dumps(stored)
    await writer.close(timeout=1)


async def test_a_history_written_during_the_dream_is_not_overwritten(redis_client):
    """No history at load time; a live wake writes the first one mid-dream."""
    service_box = {}
    live = dump(raw_history_with_target())

    async def respond(messages, info: AgentInfo):
        _, _, writer = service_box["parts"]
        await writer.save(
            guild_id=GUILD,
            history=live,
            previous_revision=0,
        )
        await writer.flush(GUILD)
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {"memory": "## Ongoing Context\nx"})]
        )

    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    service_box["parts"] = (service, api, writer)
    service._model = FunctionModel(respond)

    assert await service.sweep_once() == {GUILD: "dreamed"}

    await writer.flush(GUILD)
    stored = json.loads(await redis_client.get(history_key(GUILD)))["history"]
    assert "GUILD SYSTEM PROMPT" not in json.dumps(stored)  # the live write stands
    assert stored == live
    await writer.close(timeout=1)


def test_a_zero_limit_is_kept_and_memory_and_facts_are_templated():
    text = dream_instructions({"memory": 1500, "facts": 0})

    assert "at most 0 characters" in text
    assert "Hard cap: 1500 characters" in text
    assert "[[" not in text
    assert "Hard cap: 2000 characters" in dream_instructions(None)


async def test_thinking_is_not_kept_in_the_history(redis_client):
    async def respond(messages, info: AgentInfo):
        tool = info.output_tools[0]
        return ModelResponse(
            parts=[
                ThinkingPart(content="secret reasoning", signature="sig"),
                ToolCallPart(tool.name, {"memory": "## Ongoing Context\nx"}),
            ]
        )

    brain = Brain()
    service, api, writer = await build(redis_client, [{"accepted": True}], brain)
    service._model = FunctionModel(respond)

    assert await service.sweep_once() == {GUILD: "dreamed"}

    await writer.flush(GUILD)
    stored = json.loads(await redis_client.get(history_key(GUILD)))["history"]
    assert "secret reasoning" not in json.dumps(stored)
    await writer.close(timeout=1)
