"""channel_details (#278): a channel's description, read on demand.

The brief names each channel by id and name only. The free channel_details
tool reads one's description, or a forum's guidelines and tags, and for a
thread or forum post its parent's too. It reads only the channels the agent
is enabled in, the threads and posts under them, and their parents.
"""

from __future__ import annotations

from types import SimpleNamespace

import fakeredis.aioredis
from purge_fakes import CHANNEL, TARGET, batch
from test_forum_posts import (
    FakeEngine,
    addendum,
    forum_row,
    make_runtime,
    wake_batch,
)
from test_recipes import RecipeAPI, real_runtime, tool_results

from proactive_agent.agent import GUILD_AGENT_SYSTEM_PROMPT
from proactive_agent.contracts import EnabledChannel
from proactive_agent.discord import DiscordRESTError
from proactive_agent.runtime import CHANNEL_TEXT_MAX_CHARS

GUILD = "333333333333333333"
TEXT = "444444444444444444"
BARE = "444444444444444445"
FORUM = "555555555555555555"
POST = "666666666666666666"
POST_OFF = "666666666666666668"
THREAD = "777777777777777777"
PRIVATE = "123123123123123123"
PRIVATE_THREAD = "123123123123123124"
TAG_HELP = "888888888888888881"
TAG_PY = "888888888888888882"

CHANNELS = {
    TEXT: {
        "id": TEXT,
        "name": "general",
        "type": 0,
        "topic": "Chat about anything.\nNo spam, please.",
    },
    BARE: {"id": BARE, "name": "random", "type": 0, "topic": None},
    FORUM: {
        "id": FORUM,
        "name": "help-forum",
        "type": 15,
        "topic": "Post one question per thread.\nTag it with its language.",
        "available_tags": [
            {"id": TAG_HELP, "name": "help"},
            {"id": TAG_PY, "name": "python"},
        ],
    },
    POST: {
        "id": POST,
        "name": "How do I sort a dict?",
        "type": 11,
        "parent_id": FORUM,
        "applied_tags": [TAG_PY],
    },
    POST_OFF: {"id": POST_OFF, "name": "Leave me be", "type": 11, "parent_id": FORUM},
    THREAD: {"id": THREAD, "name": "side chat", "type": 11, "parent_id": TEXT},
    PRIVATE: {"id": PRIVATE, "name": "mods", "type": 0, "topic": "Secret plans."},
    PRIVATE_THREAD: {
        "id": PRIVATE_THREAD,
        "name": "mod thread",
        "type": 11,
        "parent_id": PRIVATE,
    },
}


def text_row(channel_id: str = TEXT) -> EnabledChannel:
    return EnabledChannel(channel_id=channel_id, watch_addendum=addendum("watch"))


async def details_for(rows, *, own_rows=None, blocked=None, channels=CHANNELS):
    """A wake with these rows enabled; returns the wake's channel_details."""
    engine = FakeEngine()
    runtime = make_runtime(rows, own_rows=own_rows, engine=engine)
    fetched: list[str] = []

    async def channel(channel_id):
        fetched.append(channel_id)
        if channel_id not in channels:
            raise DiscordRESTError(404, "Unknown Channel")
        return channels[channel_id]

    runtime.discord.channel = channel
    runtime.blocked_users = blocked
    # A wake about the text channel and the post, so either kind of row has
    # a channel to wake for.
    await runtime.process(wake_batch(TEXT, POST))
    return engine.last_deps.channel_details, runtime, fetched


async def test_a_text_channel_shows_its_whole_description():
    details, _, _ = await details_for([text_row()])

    assert await details(TEXT) == (
        f"#general (channel_id {TEXT}): text channel\n"
        "Description:\nChat about anything.\nNo spam, please."
    )


async def test_a_text_channel_without_a_topic_says_none():
    details, _, _ = await details_for([text_row(TEXT), text_row(BARE)])

    assert await details(BARE) == (
        f"#random (channel_id {BARE}): text channel\nDescription: none"
    )


async def test_a_forum_shows_its_guidelines_and_tags():
    details, _, _ = await details_for([forum_row()])

    assert await details(FORUM) == (
        f"#help-forum (channel_id {FORUM}): forum\n"
        "Guidelines:\nPost one question per thread.\nTag it with its language.\n"
        "Tags: help, python"
    )


async def test_a_forum_post_shows_its_tags_and_its_forums_details():
    details, _, _ = await details_for([forum_row()])

    assert await details(POST) == (
        f"#How do I sort a dict? (channel_id {POST}): forum post\n"
        "Tags: python\n\n"
        "In forum:\n"
        f"#help-forum (channel_id {FORUM}): forum\n"
        "Guidelines:\nPost one question per thread.\nTag it with its language.\n"
        "Tags: help, python"
    )


async def test_a_thread_in_a_text_channel_shows_its_parents_details():
    details, _, _ = await details_for([text_row()])

    assert await details(THREAD) == (
        f"#side chat (channel_id {THREAD}): thread\n\n"
        "In:\n"
        f"#general (channel_id {TEXT}): text channel\n"
        "Description:\nChat about anything.\nNo spam, please."
    )


async def test_a_channel_that_is_not_enabled_is_refused():
    details, _, _ = await details_for([text_row()])

    for channel_id in (PRIVATE, PRIVATE_THREAD, "999", "not an id"):
        result = await details(channel_id)
        assert result == (
            f"Channel {channel_id} is not one you can read: only the channels "
            "you are enabled in, their threads and posts, and their parents."
        )
        assert "Secret plans" not in result
        assert "mod" not in result


async def test_a_post_switched_off_on_its_own_is_refused():
    details, _, _ = await details_for(
        [forum_row()], own_rows={POST_OFF: {"enabled": False}}
    )

    assert (await details(POST_OFF)).startswith(f"Channel {POST_OFF} is not one")
    assert "forum post" in await details(POST)


async def test_an_enabled_rows_channel_is_not_fetched_again():
    details, _, fetched = await details_for([text_row()])
    before = len(fetched)

    await details(TEXT)
    await details(THREAD)

    # The thread is fetched; its parent comes from the wake's own fetch.
    assert fetched[before:] == [THREAD]


async def test_an_opted_out_members_id_is_scrubbed():
    channels = {
        **CHANNELS,
        TEXT: {**CHANNELS[TEXT], "topic": f"Ask <@{TARGET}> or {TARGET} for help."},
        THREAD: {**CHANNELS[THREAD], "owner_id": TARGET},
    }
    blocked = SimpleNamespace(
        is_blocked=lambda user_id, _message_id=None: str(user_id) == TARGET
    )
    details, _, _ = await details_for([text_row()], blocked=blocked, channels=channels)

    for result in (await details(TEXT), await details(THREAD)):
        assert TARGET not in result
    assert "Ask @[blocked user] or [blocked user] for help." in await details(TEXT)
    # A thread's name is its starter's words, hidden like a post's title.
    assert "side chat" not in await details(THREAD)
    assert (await details(THREAD)).startswith("#a post by someone who opted out")


async def test_a_long_description_is_cut_with_a_note():
    long = "x" * (CHANNEL_TEXT_MAX_CHARS + 250)
    channels = {**CHANNELS, TEXT: {**CHANNELS[TEXT], "topic": long}}
    details, _, _ = await details_for([text_row()], channels=channels)

    result = await details(TEXT)

    assert "x" * CHANNEL_TEXT_MAX_CHARS + "\n[cut here: 250 more characters]" in result
    assert "x" * (CHANNEL_TEXT_MAX_CHARS + 1) not in result


async def test_the_tool_is_free_and_reaches_the_model():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    seen: list = []
    runtime, writer = real_runtime(
        redis_client,
        RecipeAPI([]),
        [("channel_details", {"channel_id": CHANNEL}), None],
        seen,
    )

    async def channel(channel_id):
        return {"id": channel_id, "name": "general", "type": 0, "topic": "Be kind."}

    runtime.discord.channel = channel

    await runtime.process(batch())

    (result,) = tool_results(runtime.engine.agent_runner.history, "channel_details")
    assert result == (
        f"#general (channel_id {CHANNEL}): text channel\nDescription:\nBe kind."
    )
    assert runtime.engine.last_deps.budget.used == 0
    await writer.close()


def test_the_system_prompt_names_the_tool():
    assert "channel_details tool reads one's description" in " ".join(
        GUILD_AGENT_SYSTEM_PROMPT.split()
    )
