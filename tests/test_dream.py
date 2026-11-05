"""The agent-run nightly dream (smarter-dev #454)."""

from __future__ import annotations

import json

import fakeredis.aioredis
import pytest
from purge_fakes import GUILD, FakeAPI, dump, raw_history_with_target
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.dream import DreamAnswer, DreamService, submission
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.keys import history_key, ownership_key, privacy_lock_key
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

    service = DreamService(
        redis_client,
        queue,
        repository,
        writer,
        model=brain.model(),
        summarize=summarize,
        api=api,
        model_id="agent-model",
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
    await redis_client.delete(ownership_key(GUILD))
    await redis_client.set(privacy_lock_key(GUILD), "someone-else", ex=60)

    outcomes = await service.sweep_once()

    assert outcomes == {GUILD: "fence busy"}
    assert api.submitted == [] and brain.prompts == []
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
