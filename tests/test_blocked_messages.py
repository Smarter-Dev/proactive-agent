"""A blocked user's Discord messages never reach model input.

After a purge adds the target to the blocked-users list, a real wake fetches
the channel through DiscordREST: the model and the skim model must see only
`[BLOCKED BY USER]` where the target spoke, and every tool must refuse the
blocked message's id without saying anything about it.
"""

from __future__ import annotations

import fakeredis.aioredis
import httpx
from purge_fakes import BYSTANDER, CHANNEL, GUILD, TARGET, FakeAPI
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.agent import (
    AgentDeps,
    KimiAgentRunner,
    build_kimi_agent,
)
from proactive_agent.blocked_users import BlockedUsers
from proactive_agent.contracts import BlockedUsersList
from proactive_agent.discord import DiscordREST
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.environment import ChannelEnvironment, InstructionStore
from proactive_agent.transcript import render_transcript_line, speaker_tags
from proactive_agent.types import BlockedMessage

BOT = "999999999999999999"
KAI_MESSAGE = "700000000000000001"
NIA_MESSAGE = "700000000000000002"
NIA_REPLY = "700000000000000003"
KAI_SECRET = "my cat Miso is sick"
KAI_TIMESTAMP = "2026-10-03T11:11:11.000000+00:00"


def discord_messages() -> list[dict]:
    # Newest first, as Discord returns them.
    return [
        {
            "id": NIA_REPLY,
            "timestamp": "2026-10-03T11:13:00.000000+00:00",
            "author": {"id": BYSTANDER, "username": "nia", "global_name": "Nia"},
            "content": f"<@{TARGET}> hope Miso feels better, ship Rust 1.95",
            "mentions": [{"id": TARGET}],
            "message_reference": {"message_id": KAI_MESSAGE},
            "referenced_message": {
                "id": KAI_MESSAGE,
                "author": {"id": TARGET, "username": "kai_rs"},
                "content": KAI_SECRET,
            },
        },
        {
            "id": NIA_MESSAGE,
            "timestamp": "2026-10-03T11:12:00.000000+00:00",
            "author": {"id": BYSTANDER, "username": "nia"},
            "content": "release notes are up",
        },
        {
            "id": KAI_MESSAGE,
            "timestamp": KAI_TIMESTAMP,
            "author": {
                "id": TARGET,
                "username": "kai_rs",
                "global_name": "Kai the Rustacean",
            },
            "member": {"nick": "kai", "roles": ["1"]},
            "content": KAI_SECRET,
            "mentions": [{"id": BYSTANDER}],
            "attachments": [{"url": "https://cdn.discord.test/kai.png"}],
            "sticker_items": [{"id": "5"}],
        },
    ]


def discord_client(blocked_users) -> DiscordREST:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith(f"/channels/{CHANNEL}/messages"):
            return httpx.Response(200, json=discord_messages())
        if path.endswith(f"/channels/{CHANNEL}/messages/{KAI_MESSAGE}"):
            return httpx.Response(200, json=discord_messages()[2])
        if path.endswith(f"/guilds/{GUILD}/roles"):
            return httpx.Response(200, json=[{"id": "1", "name": "Rustaceans"}])
        raise AssertionError(path)

    return DiscordREST(
        bot_token="t",
        api_base="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        blocked_users=blocked_users,
    )


class ListAPI(FakeAPI):
    async def get_blocked_users(self):
        return BlockedUsersList.model_validate(await super().get_blocked_users())


def scripted_agent(seen: list[list[ModelMessage]]) -> FunctionModel:
    """Read the channel, then try every way to reach kai's message."""
    calls = [
        ("channel_history", {"channel_id": CHANNEL, "limit": 20}),
        ("lookup_message", {"channel_id": CHANNEL, "message_id": KAI_MESSAGE}),
        (
            "skim_messages",
            {"channel_id": CHANNEL, "around_message_id": NIA_MESSAGE},
        ),
        (
            "reply_to_message",
            {"channel_id": CHANNEL, "message_id": KAI_MESSAGE, "content": "hi"},
        ),
        (
            "react_to_message",
            {"channel_id": CHANNEL, "message_id": KAI_MESSAGE, "emoji": "x"},
        ),
    ]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        step = len(seen) - 1
        if step < len(calls):
            name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(name, args, f"call-{step}")])
        return ModelResponse(parts=[TextPart("stayed quiet")])

    return FunctionModel(respond)


def skim_model(seen: list[list[ModelMessage]]) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(list(messages))
        return ModelResponse(parts=[TextPart("nia shares release notes")])

    return FunctionModel(respond)


async def wake_with(discord: DiscordREST):
    agent_seen: list[list[ModelMessage]] = []
    skim_seen: list[list[ModelMessage]] = []
    runner = KimiAgentRunner(
        agent=build_kimi_agent(scripted_agent(agent_seen), system_prompt="sys"),
        summarize=None,
    )
    engine = AgentEngine(
        agent_runner=runner,
        skim=SkimRunner(skim_model(skim_seen)),
        agent_model_id="fake",
        skim_model_id="fake-skim",
        deps_factory=AgentDeps,
    )

    async def channel_envs(channel_id: str) -> ChannelEnvironment:
        return ChannelEnvironment(
            visible=await discord.channel_history(channel_id, guild_id=GUILD),
            bot_user_id=BOT,
        )

    result = await engine.wake(
        notifications=(),
        dropped=0,
        enabled_channels={CHANNEL: "general"},
        instruction_stores={CHANNEL: InstructionStore(seed="seed")},
        channel_envs=channel_envs,
    )
    # What the models were given: prompts and tool returns. The scripted
    # agent's own tool calls (model output) are left out on purpose.
    model_input = ModelMessagesTypeAdapter.dump_json(
        [
            message
            for messages in agent_seen + skim_seen
            for message in messages
            if not isinstance(message, ModelResponse)
        ]
    ).decode()
    tool_outputs = {
        part.tool_name: part.content
        for message in agent_seen[-1]
        for part in getattr(message, "parts", ())
        if isinstance(part, ToolReturnPart)
    }
    return model_input, tool_outputs, result


IDENTIFYING = [
    TARGET,
    KAI_SECRET,
    KAI_MESSAGE,
    "kai",
    "Kai the Rustacean",
    "kai_rs",
    "11:11",
    "kai.png",
    "Rustaceans",
]


async def test_purged_user_messages_do_not_reach_model_input():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = ListAPI()
    blocked = BlockedUsers(api, redis_client)
    await blocked.refresh()
    discord = discord_client(blocked)

    # Before the purge kai is ordinary channel content.
    before, _tools, _result = await wake_with(discord)
    assert KAI_SECRET in before and TARGET in before

    # The purge adds kai to the list; the next refresh picks it up.
    api.blocked = {"revision": 2, "user_ids": [TARGET]}
    await blocked.refresh()
    model_input, tools, result = await wake_with(discord)

    # The only place kai's message id may appear is the refusal echoing the
    # id the scripted model asked for.
    refusal = f"No message with id {KAI_MESSAGE} is visible."
    for value in IDENTIFYING:
        assert value not in model_input.replace(refusal, ""), value
    assert "[BLOCKED BY USER]" in model_input
    assert "release notes are up" in model_input
    assert "hope Miso feels better" in model_input
    assert "@[blocked user]" in model_input

    history_lines = tools["channel_history"].splitlines()
    assert history_lines[0] == "[BLOCKED BY USER]"
    assert history_lines[1].startswith(f"[id={NIA_MESSAGE}] A·nia (uid={BYSTANDER})")
    assert history_lines[2].startswith(f"[id={NIA_REPLY}] A·Nia (uid={BYSTANDER})")
    assert tools["lookup_message"] == refusal
    assert tools["reply_to_message"] == refusal
    assert tools["react_to_message"] == refusal
    assert tools["skim_messages"] == "nia shares release notes"
    assert result.responses == [] and result.reactions == ()
    await discord.close()


async def test_fetch_message_of_a_blocked_author_is_a_bare_placeholder():
    blocked = BlockedUsers(None, None)
    blocked._user_ids = frozenset({TARGET})
    discord = discord_client(blocked)

    message = await discord.fetch_message(CHANNEL, KAI_MESSAGE, guild_id=GUILD)

    assert message == BlockedMessage()
    assert message.to_record() == {"blocked": True}
    await discord.close()


def test_blocked_record_renders_alone_and_takes_no_speaker_tag():
    nia = {
        "id": "1",
        "author_id": BYSTANDER,
        "author_display": "nia",
        "is_bot": False,
        "content": "hi",
        "reply_to_id": None,
    }
    records = [BlockedMessage().to_record(), nia]
    tags = speaker_tags(records)

    assert tags == {BYSTANDER: "A"}
    assert render_transcript_line(records[0], tags) == "[BLOCKED BY USER]"
    env = ChannelEnvironment(visible=[BlockedMessage()], bot_user_id=BOT)
    assert env.lookup("None") is None and env.lookup(None) is None
    assert env.render(env.history(5)) == "[BLOCKED BY USER]"
