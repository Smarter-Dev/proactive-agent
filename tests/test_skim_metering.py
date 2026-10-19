"""A skim the agent's tool runs is billed with the wake, a failed one too (#211).

The wake runs the real AgentEngine, KimiAgentRunner and production agent
(``build_proactive_agent``); its ``skim_messages`` tool reads a mocked Discord
channel and calls the real SkimRunner. Only the models are FunctionModels.
"""

from __future__ import annotations

from purge_fakes import CHANNEL, GUILD
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from test_blocked_messages import BOT, NIA_MESSAGE, discord_client

from proactive_agent.agent import KimiAgentRunner
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.environment import ChannelEnvironment, InstructionStore
from proactive_agent.parity import ProactiveDeps, build_proactive_agent

AGENT = "claude-opus-5-5"
SKIM = "glm-5.3-flash"
CALL = RequestUsage(input_tokens=30_000, output_tokens=40)
REPLY = RequestUsage(input_tokens=31_000, output_tokens=20)
SKIMMED = RequestUsage(input_tokens=90_000, output_tokens=4_096)


def tokens(usage: RequestUsage) -> dict:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def agent_model() -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "skim_messages",
                        {"channel_id": CHANNEL, "around_message_id": NIA_MESSAGE},
                    )
                ],
                usage=CALL,
            )
        return ModelResponse(parts=[TextPart("stayed quiet")], usage=REPLY)

    return FunctionModel(respond, model_name=AGENT)


def skim_model(*, cut_off: bool) -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        if cut_off:
            # Thinking only, cut off at its output limit: billed, then the
            # run raises and the tool hands the error back to the agent.
            return ModelResponse(
                parts=[ThinkingPart("hm")], usage=SKIMMED, finish_reason="length"
            )
        return ModelResponse(parts=[TextPart("nia shares notes")], usage=SKIMMED)

    return FunctionModel(respond, model_name=SKIM)


async def wake(*, cut_off: bool):
    discord = discord_client(None)
    engine = AgentEngine(
        agent_runner=KimiAgentRunner(
            agent=build_proactive_agent(agent_model(), system_prompt="sys"),
            summarize=None,
            model_id=AGENT,
        ),
        skim=SkimRunner(skim_model(cut_off=cut_off)),
        agent_model_id=AGENT,
        skim_model_id=SKIM,
        deps_factory=lambda **kwargs: ProactiveDeps(
            guild_id=GUILD, discord=discord, api=None, **kwargs
        ),
    )

    async def channel_envs(channel_id: str) -> ChannelEnvironment:
        return ChannelEnvironment(
            visible=await discord.channel_history(channel_id, guild_id=GUILD),
            bot_user_id=BOT,
        )

    return await engine.wake(
        notifications=(),
        dropped=0,
        enabled_channels={CHANNEL: "general"},
        instruction_stores={CHANNEL: InstructionStore(seed="seed")},
        channel_envs=channel_envs,
    )


async def test_a_skim_that_fails_is_billed_with_the_wake():
    result = await wake(cut_off=True)

    assert result.usage_by_model[SKIM]["requests"] == [tokens(SKIMMED)]
    assert result.usage_by_model[SKIM]["input_tokens"] == SKIMMED.input_tokens
    assert result.usage_by_model[AGENT]["requests"] == [tokens(CALL), tokens(REPLY)]
    assert result.input_tokens == sum(
        usage.input_tokens for usage in (CALL, REPLY, SKIMMED)
    )


async def test_a_skim_that_succeeds_is_billed_once():
    result = await wake(cut_off=False)

    assert result.usage_by_model[SKIM]["requests"] == [tokens(SKIMMED)]
    assert result.usage_by_model[SKIM]["output_tokens"] == SKIMMED.output_tokens
