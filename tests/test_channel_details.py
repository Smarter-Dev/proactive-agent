"""channel_details, recipe-named channels, close_and_lock_post (#278).

The brief names each channel by id and name only. The free channel_details
tool reads one's description, or a forum's guidelines and tags, and for a
thread or forum post its parent's too. The agent knows about the channels
it is enabled in, the threads and posts under them, their parents, and the
channels an enabled recipe names by id: those it may post to and read, but
never watches. close_and_lock_post locks a post of an enabled forum at once
and closes it after the wake's messages are sent.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import fakeredis.aioredis
import httpx
from purge_fakes import CHANNEL, TARGET, batch
from test_forum_posts import (
    FakeEngine,
    addendum,
    forum_row,
    make_runtime,
    wake_batch,
)
from test_recipes import (
    WELCOME_ID,
    RecipeAPI,
    opened_history,
    preview_of,
    real_runtime,
    tool_results,
    welcome,
)

from proactive_agent.agent import GUILD_AGENT_SYSTEM_PROMPT
from proactive_agent.contracts import EnabledChannel
from proactive_agent.discord import DiscordREST, DiscordRESTError
from proactive_agent.recipes import recipe_channel_ids
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
MOD_CHAT = "720663441966366850"
VOICE = "131313131313131313"
ELSEWHERE = "141414141414141414"
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
    PRIVATE: {
        "id": PRIVATE,
        "name": "mods",
        "type": 0,
        "topic": "Secret plans.",
    },
    PRIVATE_THREAD: {
        "id": PRIVATE_THREAD,
        "name": "mod thread",
        "type": 11,
        "parent_id": PRIVATE,
    },
}


def text_row(channel_id: str = TEXT) -> EnabledChannel:
    return EnabledChannel(channel_id=channel_id, watch_addendum=addendum("watch"))


class ActingEngine(FakeEngine):
    """A FakeEngine whose wake also runs ``act(deps)``, as a tool would."""

    def __init__(self, act, respond_in=()):
        super().__init__(respond_in=respond_in)
        self.act = act
        self.outcomes: list = []

    async def wake(self, **kwargs):
        result = await super().wake(**kwargs)
        self.outcomes.append(await self.act(self.last_deps))
        return result


async def wake_for(
    rows,
    *,
    own_rows=None,
    blocked=None,
    channels=CHANNELS,
    engine=None,
    calls=None,
    edit_errors=(),
):
    """One wake with these rows enabled; returns its deps, the runtime and
    the channel ids fetched one by one."""
    engine = engine or FakeEngine()
    runtime = make_runtime(rows, own_rows=own_rows, engine=engine)
    fetched: list[str] = []
    calls = [] if calls is None else calls
    edit_errors = list(edit_errors)

    async def channel(channel_id):
        fetched.append(channel_id)
        if channel_id not in channels:
            raise DiscordRESTError(404, "Unknown Channel")
        return channels[channel_id]

    async def edit_channel(channel_id, fields, *, reason=None):
        calls.append(("edit", channel_id, fields, reason))
        error = edit_errors.pop(0) if edit_errors else None
        if error is not None:
            raise error
        return {"id": channel_id}

    async def send_message(channel_id, content, **_kwargs):
        calls.append(("send", channel_id))
        return {"id": "1"}

    runtime.discord.channel = channel
    runtime.discord.edit_channel = edit_channel
    runtime.discord.send_message = send_message
    runtime.blocked_users = blocked
    # A wake about the text channel and the post, so either kind of row has
    # a channel to wake for.
    await runtime.process(wake_batch(TEXT, POST))
    return engine.last_deps, runtime, fetched


async def details_for(rows, **kwargs):
    """A wake with these rows enabled; returns the wake's channel_details."""
    deps, runtime, fetched = await wake_for(rows, **kwargs)
    return deps.channel_details, runtime, fetched


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
            "you are enabled in, their threads and posts, their parents, and "
            "channels your recipes name."
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


# -- Closing a post ----------------------------------------------------------------


def closer(reason="Off-topic:   selling   accounts."):
    async def act(deps):
        return await deps.close_post(POST, reason)

    return act


async def test_a_post_is_locked_now_and_closed_after_its_messages():
    calls: list = []
    engine = ActingEngine(closer(), respond_in=(POST,))

    await wake_for([forum_row()], engine=engine, calls=calls)

    assert engine.outcomes == [
        "Post locked: only moderators can post in it now. It closes once this "
        "wake's messages are sent, so explain why in the post now if you have not."
    ]
    reason = "Off-topic: selling accounts."
    assert calls == [
        ("edit", POST, {"locked": True}, reason),
        ("send", POST),
        ("edit", POST, {"archived": True, "locked": True}, reason),
    ]


async def test_only_a_post_of_an_enabled_forum_can_be_closed():
    calls: list = []

    async def act(deps):
        return await deps.close_post(TEXT, "spam")

    engine = ActingEngine(act)
    await wake_for([text_row(), forum_row()], engine=engine, calls=calls)

    assert engine.outcomes == [
        f"Channel {TEXT} is not a post in a forum you are enabled in; only those "
        "can be closed."
    ]
    assert calls == []


async def test_a_lock_discord_refuses_says_why_and_closes_nothing():
    calls: list = []
    engine = ActingEngine(closer(), respond_in=(POST,))

    await wake_for(
        [forum_row()],
        engine=engine,
        calls=calls,
        edit_errors=[DiscordRESTError(403, "PATCH -> 403 (code 50013)")],
    )

    assert engine.outcomes == [
        f"Discord refused to lock post {POST}: the bot lacks permission there (it "
        "needs Manage Threads). The post is unchanged; tell a moderator instead."
    ]
    assert calls == [
        ("edit", POST, {"locked": True}, "Off-topic: selling accounts."),
        ("send", POST),
    ]


async def test_a_close_that_fails_after_the_lock_is_logged_and_the_wake_goes_on(
    caplog,
):
    calls: list = []
    engine = ActingEngine(closer(), respond_in=(POST,))

    with caplog.at_level(logging.WARNING, "proactive_agent.runtime"):
        _, runtime, _ = await wake_for(
            [forum_row()],
            engine=engine,
            calls=calls,
            edit_errors=[None, DiscordRESTError(503, "PATCH -> 503 (code None)")],
        )

    assert [call[0] for call in calls] == ["edit", "send", "edit"]
    assert "proactive post close failed" in caplog.text
    recorded = [call.args[2] for call in runtime.journal.record.await_args_list]
    assert f"close:{POST}" not in recorded


async def test_a_close_already_done_is_not_repeated_on_a_retried_wake():
    calls: list = []
    engine = ActingEngine(closer())
    deps, runtime, _ = await wake_for([forum_row()], engine=engine, calls=calls)
    recorded = [call.args[2] for call in runtime.journal.record.await_args_list]
    assert recorded == [f"close:{POST}"]

    calls.clear()
    runtime.journal.completed = AsyncMock(return_value=True)
    await runtime.process(wake_batch(TEXT, POST))

    # The retry locks again (harmless) but does not close twice.
    assert calls == [("edit", POST, {"locked": True}, "Off-topic: selling accounts.")]


async def test_the_audit_log_reason_travels_as_a_header():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": POST})

    client = DiscordREST(bot_token="t", transport=httpx.MockTransport(handler))
    await client.edit_channel(POST, {"locked": True}, reason="Spam — again, ünïcode")
    await client.close()

    (request,) = seen
    assert request.method == "PATCH"
    assert request.url.path.endswith(f"/channels/{POST}")
    assert request.read() == b'{"locked":true}'
    assert unquote(request.headers["X-Audit-Log-Reason"]) == "Spam — again, ünïcode"


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


async def test_closing_spends_budget_and_a_text_channel_is_refused():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    seen: list = []
    runtime, writer = real_runtime(
        redis_client,
        RecipeAPI([]),
        [("close_and_lock_post", {"channel_id": CHANNEL, "reason": "spam"}), None],
        seen,
    )
    runtime.discord.edit_channel = AsyncMock()

    await runtime.process(batch())

    (closed,) = tool_results(runtime.engine.agent_runner.history, "close_and_lock_post")
    assert closed == (
        f"Channel {CHANNEL} is not a post in a forum you are enabled in; only "
        "those can be closed."
    )
    runtime.discord.edit_channel.assert_not_awaited()
    assert runtime.engine.last_deps.budget.used == 1
    await writer.close()


# -- Channels a recipe names --------------------------------------------------------

ESCALATE_TEXT = (
    "Use this when a forum post is off-topic.\n\n"
    f"Reply in the post. If it is egregious, close it and tell <#{MOD_CHAT}>."
)


def recipe_runtime(script, *, channels=None):
    """The real tool loop, with a recipe naming mod-chat and a Discord that
    knows these channels."""
    channels = channels or {
        CHANNEL: {"id": CHANNEL, "guild_id": GUILD, "name": "general", "type": 0},
        MOD_CHAT: {"id": MOD_CHAT, "guild_id": GUILD, "name": "mod-chat", "type": 0},
    }
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI([welcome(text=ESCALATE_TEXT)])
    seen: list = []
    runtime, writer = real_runtime(redis_client, api, script, seen)
    fetched: list[str] = []

    async def channel(channel_id):
        fetched.append(channel_id)
        if channel_id not in channels:
            raise DiscordRESTError(404, "Unknown Channel")
        return channels[channel_id]

    runtime.discord.channel = channel
    runtime.discord.send_message = AsyncMock(return_value={"id": "1"})
    runtime.fetched = fetched
    return runtime, writer, api


def sent_to(runtime) -> list[str]:
    return [call.args[0] for call in runtime.discord.send_message.await_args_list]


async def test_the_agent_can_post_to_a_channel_its_recipe_names():
    runtime, writer, _ = recipe_runtime(
        [
            ("open_recipe", {"name": "Welcome"}),
            ("send_channel_message", {"channel_id": MOD_CHAT, "content": "FYI"}),
            ("channel_details", {"channel_id": MOD_CHAT}),
            None,
        ]
    )

    await runtime.process(batch())

    history = runtime.engine.agent_runner.history
    assert tool_results(history, "send_channel_message") == ["Message sent."]
    assert tool_results(history, "channel_details") == [
        f"#mod-chat (channel_id {MOD_CHAT}): text channel\nDescription: none"
    ]
    assert sent_to(runtime) == [MOD_CHAT]
    # Named, not watched: it is no channel the agent wakes for or reads.
    assert MOD_CHAT not in runtime.engine.last_deps.enabled_channels
    await writer.close()


async def test_a_recipe_opened_on_an_earlier_wake_still_names_its_channel():
    runtime, writer, _ = recipe_runtime(
        [
            ("open_recipe", {"name": "Welcome"}),
            None,
            ("send_channel_message", {"channel_id": MOD_CHAT, "content": "FYI"}),
            None,
        ]
    )

    await runtime.process(batch())
    await runtime.process(batch())

    assert sent_to(runtime) == [MOD_CHAT]
    await writer.close()


async def test_an_edit_withdraws_the_old_texts_channels_until_reopened():
    runtime, writer, api = recipe_runtime(
        [
            ("open_recipe", {"name": "Welcome"}),
            None,
            ("send_channel_message", {"channel_id": MOD_CHAT, "content": "FYI"}),
            None,
        ]
    )
    await runtime.process(batch())
    api.edit(WELCOME_ID, "Use this for posts.\n\nReply.")

    await runtime.process(batch())

    (refused,) = tool_results(
        runtime.engine.agent_runner.history, "send_channel_message"
    )
    assert refused == f"Channel {MOD_CHAT} is not enabled for the proactive bot."
    assert sent_to(runtime) == []
    await writer.close()


async def test_posting_to_a_channel_neither_watched_nor_named_is_refused():
    runtime, writer, _ = recipe_runtime(
        [
            ("open_recipe", {"name": "Welcome"}),
            ("send_channel_message", {"channel_id": ELSEWHERE, "content": "hi"}),
            ("channel_details", {"channel_id": ELSEWHERE}),
            None,
        ],
    )

    await runtime.process(batch())

    history = runtime.engine.agent_runner.history
    assert tool_results(history, "send_channel_message") == [
        f"Channel {ELSEWHERE} is not enabled for the proactive bot."
    ]
    (details,) = tool_results(history, "channel_details")
    assert details.startswith(f"Channel {ELSEWHERE} is not one you can read")
    assert sent_to(runtime) == []
    await writer.close()


async def test_a_recipe_naming_something_other_than_this_servers_text_channel():
    other_guild = "999999999999999990"
    channels = {
        CHANNEL: {"id": CHANNEL, "guild_id": GUILD, "name": "general", "type": 0},
        # Another server's channel, a voice channel, and an id Discord does
        # not know as a channel (a member's, say).
        MOD_CHAT: {"id": MOD_CHAT, "guild_id": other_guild, "type": 0},
        VOICE: {"id": VOICE, "guild_id": GUILD, "type": 2},
    }
    text = f"Use this.\n\nTell <#{MOD_CHAT}>, <#{VOICE}> or <@{ELSEWHERE}>."
    runtime, writer, api = recipe_runtime(
        [
            ("open_recipe", {"name": "Welcome"}),
            *[
                ("send_channel_message", {"channel_id": cid, "content": "FYI"})
                for cid in (MOD_CHAT, VOICE, ELSEWHERE)
            ],
            None,
        ],
        channels=channels,
    )
    api.recipes = {WELCOME_ID: welcome(text=text)}

    await runtime.process(batch())

    results = tool_results(runtime.engine.agent_runner.history, "send_channel_message")
    assert results == [
        f"Channel {cid} is not enabled for the proactive bot."
        for cid in (MOD_CHAT, VOICE, ELSEWHERE)
    ]
    assert sent_to(runtime) == []
    # Each id is asked about once; the 404 is remembered for the next wake.
    before = list(runtime.fetched)
    await runtime.process(batch())
    assert [cid for cid in runtime.fetched[len(before) :] if cid != CHANNEL] == []
    await writer.close()


def test_channel_ids_come_from_previews_and_texts_at_the_listed_revision():
    preview_id = "100000000000000001"
    opened = opened_history(welcome(text=f"Short.\n\nTell <#{MOD_CHAT}>."))
    listed = {"recipes": [preview_of(welcome(text=f"Ping <#{preview_id}>."))]}

    assert recipe_channel_ids(opened, listed) == [preview_id, MOD_CHAT]
    # Edited since: the old text's ids are withdrawn until it is reopened.
    edited = {"recipes": [preview_of(welcome(revision=2, text="Short."))]}
    assert recipe_channel_ids(opened, edited) == []
    # Switched off: nothing.
    assert recipe_channel_ids(opened, {"recipes": []}) == []
    assert recipe_channel_ids(opened, None) == []


def test_the_system_prompt_names_the_tools():
    prompt = " ".join(GUILD_AGENT_SYSTEM_PROMPT.split())
    assert "channel_details tool reads one's description" in prompt
    assert "A channel that an enabled recipe names by its id" in prompt
