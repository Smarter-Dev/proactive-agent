"""Whole-forum support (#248): an enabled forum's posts are each a channel.

The bot gates which posts wake the agent (a post inherits its forum's row
unless it has its own). The worker admits those posts per wake: the agent
can read and answer each post as its own channel, never the forum itself,
and every post shares the forum's watch instructions, saved on the forum.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import types
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from proactive_agent.agent import ToolBudget
from proactive_agent.api import ApplicationAPI
from proactive_agent.beacon import CapabilityBeacon
from proactive_agent.contracts import EnabledChannel, NotificationEnvelope
from proactive_agent.discord import DiscordRESTError
from proactive_agent.environment import WakeActions
from proactive_agent.keys import FORUM_POSTS_CAPABILITY_KEY
from proactive_agent.queue import StreamNotification, WakeBatch
from proactive_agent.runtime import GuildRuntime, forum_post_label
from proactive_agent.types import ActivationResult, ProposedResponse

GUILD = "333333333333333333"
TEXT = "444444444444444444"
FORUM = "555555555555555555"
POST = "666666666666666666"
OTHER_POST = "666666666666666667"
POST_OFF = "666666666666666668"
TEXT_THREAD = "777777777777777777"
TAG_HELP = "888888888888888881"
TAG_PY = "888888888888888882"

CHANNELS = {
    TEXT: {"id": TEXT, "name": "general", "type": 0},
    FORUM: {
        "id": FORUM,
        "name": "help-forum",
        "type": 15,
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
        "applied_tags": [TAG_HELP, TAG_PY],
    },
    OTHER_POST: {
        "id": OTHER_POST,
        "name": "Async help",
        "type": 11,
        "parent_id": FORUM,
    },
    POST_OFF: {"id": POST_OFF, "name": "Leave me be", "type": 11, "parent_id": FORUM},
    TEXT_THREAD: {
        "id": TEXT_THREAD,
        "name": "side chat",
        "type": 11,
        "parent_id": TEXT,
    },
}


def addendum(text: str) -> str:
    expires = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    return json.dumps([{"id": "w1", "text": text, "expires_at": expires}])


def envelope(channel_id: str) -> NotificationEnvelope:
    return NotificationEnvelope(
        schema_version=1,
        notification_id=uuid4(),
        guild_id=GUILD,
        channel_id=channel_id,
        channel_name=CHANNELS[channel_id]["name"],
        kind="mention",
        created_at=datetime.now(UTC),
        body="someone mentioned the bot",
        message_ids=("1",),
        wakes=True,
        passive=False,
        watcher_usage={},
        trace_id=uuid4(),
    )


def wake_batch(*channel_ids: str) -> WakeBatch:
    items = [
        StreamNotification(stream_id=f"{n}-0", envelope=envelope(cid))
        for n, cid in enumerate(channel_ids)
    ]
    return WakeBatch(
        guild_id=GUILD,
        wake_id="wake-1",
        ready_ids=[],
        waking=items,
        pending=[],
        dropped=0,
    )


class FakeEngine:
    """Builds deps as the real engine does and answers in the given channels."""

    agent_model_id = "fake"

    def __init__(self, respond_in=(), midrun=None):
        self.agent_runner = SimpleNamespace(history=[], on_compacted=None)
        self.deps_factory = None
        self.last_deps = None
        self.respond_in = respond_in
        self.midrun = midrun
        self.seen: dict = {}

    async def wake(self, *, enabled_channels, instruction_stores, **_kwargs):
        deps = self.deps_factory(
            enabled_channels=enabled_channels,
            channel_envs={},
            actions=WakeActions(),
            instruction_stores=instruction_stores,
            skim_transcript=None,
            budget=ToolBudget(),
        )
        self.last_deps = deps
        self.seen["at_start"] = dict(enabled_channels)
        if self.midrun is not None:
            await deps.drain_notifications()
        self.seen["enabled"] = dict(deps.enabled_channels)
        self.seen["stores"] = instruction_stores
        if POST in instruction_stores:
            instruction_stores[POST].set_instruction(
                "watch for follow-ups in python posts", ttl_seconds=3600
            )
        return ActivationResult(
            responses=[
                ProposedResponse(reply_to_id=None, content="hi", channel_id=cid)
                for cid in self.respond_in
            ],
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            model_id="fake",
        )


def make_runtime(rows, *, own_rows=None, engine=None, discord_errors=()):
    own_rows = own_rows or {}
    fetched: list[str] = []

    async def channel(channel_id):
        fetched.append(channel_id)
        if channel_id in discord_errors:
            raise DiscordRESTError(503, "discord down")
        return CHANNELS[channel_id]

    api = SimpleNamespace(
        list_enabled_channels=AsyncMock(return_value=tuple(rows)),
        get_channel_settings=AsyncMock(side_effect=lambda _g, cid: own_rows.get(cid)),
        set_watch_addendum=AsyncMock(),
        get_memory=AsyncMock(return_value=None),
    )
    journal = SimpleNamespace(
        completed=AsyncMock(return_value=False), record=AsyncMock()
    )
    runtime = GuildRuntime(
        guild_id=GUILD,
        engine=engine or FakeEngine(),
        history_repository=None,
        history_writer=None,
        api=api,
        discord=SimpleNamespace(
            channel=channel,
            send_message=AsyncMock(return_value={"id": "1"}),
            add_reaction=AsyncMock(),
            send_files=AsyncMock(),
        ),
        redis=None,
        queue=SimpleNamespace(
            drain_midrun=AsyncMock(
                return_value=((engine.midrun,) if engine and engine.midrun else (), 0)
            ),
            claim_failure_notice=AsyncMock(return_value=True),
        ),
        journal=journal,
        bot_user_id="999999999999999999",
        guild_name="guild",
        summarize_web=None,
        image_capabilities=SimpleNamespace(review=None, generate=None),
        media_reader=None,
        author_handler=None,
    )
    runtime._sync_with_store = AsyncMock()
    runtime._load_history = AsyncMock()
    runtime._scrub_runner_history = lambda: None
    runtime._save_history = AsyncMock(return_value=SimpleNamespace(revision=1))
    runtime._record_usage = AsyncMock()
    runtime.fetched = fetched
    return runtime


def forum_row(text="watch python questions", *, is_forum=True):
    return EnabledChannel(
        channel_id=FORUM, watch_addendum=addendum(text), is_forum=is_forum
    )


async def test_a_post_of_an_enabled_forum_is_its_own_channel():
    engine = FakeEngine(respond_in=(POST,))
    runtime = make_runtime([forum_row()], engine=engine)

    await runtime.process(wake_batch(POST))

    # The post is a channel; the forum itself is not one the agent acts in.
    assert list(engine.seen["enabled"]) == [POST]
    assert FORUM not in engine.seen["enabled"]
    runtime.discord.send_message.assert_awaited_once()
    assert runtime.discord.send_message.await_args.args[0] == POST


def test_the_post_label_carries_title_forum_and_tags():
    assert forum_post_label(CHANNELS[POST], CHANNELS[FORUM]) == (
        "How do I sort a dict? (post in forum #help-forum; tags: help, python)"
    )


async def test_the_agent_sees_the_title_and_tags_of_a_new_post():
    engine = FakeEngine()
    runtime = make_runtime([forum_row()], engine=engine)

    await runtime.process(wake_batch(POST))

    assert engine.seen["enabled"][POST] == (
        "How do I sort a dict? (post in forum #help-forum; tags: help, python)"
    )


async def test_two_posts_stay_two_channels():
    engine = FakeEngine(respond_in=(POST, OTHER_POST))
    runtime = make_runtime([forum_row()], engine=engine)

    await runtime.process(wake_batch(POST, OTHER_POST))

    assert set(engine.seen["enabled"]) == {POST, OTHER_POST}
    sent_to = [call.args[0] for call in runtime.discord.send_message.await_args_list]
    assert sent_to == [POST, OTHER_POST]


async def test_a_post_switched_off_on_its_own_is_not_admitted():
    engine = FakeEngine(respond_in=(POST_OFF,))
    runtime = make_runtime(
        [forum_row()],
        own_rows={POST_OFF: {"enabled": False, "watch_addendum": ""}},
        engine=engine,
    )

    result = await runtime.process(wake_batch(POST_OFF))

    # Nothing admitted: no wake at all, so nothing sent.
    assert result.details == {"agent": {"note": "no enabled channels"}}
    runtime.discord.send_message.assert_not_awaited()


async def test_a_text_channel_thread_does_not_inherit():
    engine = FakeEngine(respond_in=(TEXT_THREAD,))
    runtime = make_runtime(
        [EnabledChannel(channel_id=TEXT, watch_addendum="")], engine=engine
    )

    await runtime.process(wake_batch(TEXT_THREAD))

    assert list(engine.seen["enabled"]) == [TEXT]
    runtime.discord.send_message.assert_not_awaited()


async def test_a_disabled_forum_admits_no_posts():
    engine = FakeEngine(respond_in=(POST,))
    runtime = make_runtime(
        [EnabledChannel(channel_id=TEXT, watch_addendum="")], engine=engine
    )

    await runtime.process(wake_batch(POST))

    assert POST not in engine.seen["enabled"]
    runtime.discord.send_message.assert_not_awaited()


async def test_posts_share_the_forums_instructions_saved_on_the_forum():
    engine = FakeEngine()
    runtime = make_runtime([forum_row()], engine=engine)

    await runtime.process(wake_batch(POST))

    assert engine.seen["stores"][POST] is engine.seen["stores"][FORUM]
    runtime.api.set_watch_addendum.assert_awaited_once()
    saved = runtime.api.set_watch_addendum.await_args.kwargs
    assert (saved["channel_id"], saved["enabled"]) == (FORUM, True)
    assert "python posts" in saved["watch_addendum"]


async def test_a_post_that_arrives_mid_wake_can_be_answered_in_that_wake():
    engine = FakeEngine(respond_in=(OTHER_POST,), midrun=envelope(OTHER_POST))
    runtime = make_runtime([forum_row()], engine=engine)

    await runtime.process(wake_batch(POST))

    assert OTHER_POST not in engine.seen["at_start"]
    assert OTHER_POST in engine.seen["enabled"]
    assert runtime.discord.send_message.await_args.args[0] == OTHER_POST


async def test_a_post_whose_forum_discord_will_not_name_is_left_out():
    engine = FakeEngine(respond_in=(POST,))
    runtime = make_runtime([forum_row()], engine=engine, discord_errors={POST})

    await runtime.process(wake_batch(POST))

    runtime.discord.send_message.assert_not_awaited()
    # Not remembered: the next wake asks Discord again.
    assert POST not in runtime.forum_parents


async def test_a_posts_forum_is_looked_up_once():
    runtime = make_runtime([forum_row()])

    await runtime.process(wake_batch(POST))
    await runtime.process(wake_batch(POST))

    # Each wake reads the post (its title and tags may change) and the
    # forum as an enabled row, but the forum as the post's parent only once.
    assert runtime.forum_parents[POST] == FORUM
    assert runtime.fetched.count(POST) == 2
    assert runtime.fetched.count(FORUM) == 2 + 1


async def test_a_failure_notice_goes_to_the_post_never_the_forum():
    runtime = make_runtime([forum_row()])

    channel_id = await runtime.report_failure(wake_batch(POST), "boom")

    assert channel_id == POST
    assert runtime.discord.send_message.await_args.args[0] == POST


async def test_a_failure_notice_with_only_a_forum_on_and_no_post_is_skipped():
    runtime = make_runtime([forum_row()])

    assert await runtime.report_failure(wake_batch(TEXT_THREAD), "boom") is None
    runtime.discord.send_message.assert_not_awaited()


def test_an_opted_out_authors_post_title_is_hidden():
    post = {**CHANNELS[POST], "owner_id": "123456789012345678"}
    blocked = SimpleNamespace(is_blocked=lambda user_id, _message_id=None: True)

    label = forum_post_label(post, CHANNELS[FORUM], blocked)

    assert "How do I sort a dict?" not in label
    assert label.startswith("a post by someone who opted out (post in forum")


async def test_the_runtime_hides_an_opted_out_authors_post_title():
    engine = FakeEngine()
    runtime = make_runtime([forum_row()], engine=engine)
    runtime.blocked_users = SimpleNamespace(
        is_blocked=lambda user_id, _message_id=None: str(user_id) == "42"
    )
    CHANNELS[POST]["owner_id"] = "42"
    try:
        await runtime.process(wake_batch(POST))
    finally:
        del CHANNELS[POST]["owner_id"]

    assert "How do I sort a dict?" not in engine.seen["enabled"][POST]


async def test_a_forum_row_from_an_application_without_the_flag_is_still_a_forum():
    # New worker, older application: the row carries no is_forum, so the
    # worker asks Discord, and the forum is still never a channel itself.
    engine = FakeEngine(respond_in=(POST, FORUM))
    runtime = make_runtime([forum_row(is_forum=False)], engine=engine)

    await runtime.process(wake_batch(POST))

    assert list(engine.seen["enabled"]) == [POST]
    sent_to = [call.args[0] for call in runtime.discord.send_message.await_args_list]
    assert sent_to == [POST]


async def test_the_worker_asks_for_forum_rows_and_reads_the_flag():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json=[
                {"channel_id": TEXT, "watch_addendum": ""},
                {"channel_id": FORUM, "watch_addendum": "", "is_forum": True},
            ],
        )

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="sk_test",
        transport=httpx.MockTransport(handler),
    )
    try:
        rows = await api.list_enabled_channels(GUILD)
    finally:
        await api.close()

    assert seen[0].url.params["forums"] == "true"
    assert [(row.channel_id, row.is_forum) for row in rows] == [
        (TEXT, False),
        (FORUM, True),
    ]


async def test_a_post_discord_refused_once_is_asked_about_again():
    runtime = make_runtime([forum_row()])
    original = runtime.discord.channel
    runtime.discord.channel = AsyncMock(side_effect=DiscordRESTError(403, "no"))
    forums = {FORUM: CHANNELS[FORUM]}

    assert await runtime._forum_post(POST, forums) is None
    assert POST not in runtime.forum_parents

    runtime.discord.channel = original
    assert await runtime._forum_post(POST, forums) is not None


async def test_a_post_discord_says_is_gone_is_remembered():
    runtime = make_runtime([forum_row()])
    runtime.discord.channel = AsyncMock(side_effect=DiscordRESTError(404, "gone"))

    assert await runtime._forum_post(POST, {FORUM: CHANNELS[FORUM]}) is None
    assert runtime.forum_parents[POST] is None


async def test_a_failure_notice_for_a_wake_about_the_forum_is_skipped():
    runtime = make_runtime([forum_row()])

    assert await runtime._failure_notice_channel(wake_batch(FORUM)) is None


async def test_a_failure_notice_skips_a_forum_discord_will_not_type():
    # The forum's row carries no flag and Discord will not say what it is:
    # it is not a destination, and the plain channel is.
    runtime = make_runtime(
        [
            forum_row(is_forum=False),
            EnabledChannel(channel_id=TEXT, watch_addendum=""),
        ],
        discord_errors={FORUM},
    )

    assert await runtime._failure_notice_channel(wake_batch(FORUM)) == TEXT


async def test_a_failure_notice_never_goes_to_a_flagged_forum_even_if_listed_first():
    runtime = make_runtime(
        [forum_row(), EnabledChannel(channel_id=TEXT, watch_addendum="")]
    )

    assert await runtime._failure_notice_channel(wake_batch(TEXT_THREAD)) == TEXT


async def test_the_beacon_keeps_the_forum_posts_capability_set_while_running():
    redis = SimpleNamespace(set=AsyncMock())
    stop = asyncio.Event()
    beacon = CapabilityBeacon(redis, ttl_seconds=30, refresh_seconds=0.01)

    task = asyncio.create_task(beacon.run(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, 1)

    assert redis.set.await_count >= 2
    call = redis.set.await_args
    assert (
        call.args[0]
        == FORUM_POSTS_CAPABILITY_KEY
        == ("proactive:v1:capability:forum-posts")
    )
    assert call.kwargs["ex"] == 30


def _main_runtime_module():
    """The worker as it was before forum support, from git main."""
    try:
        source = subprocess.check_output(
            ["git", "show", "main:src/proactive_agent/runtime.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git main is not available")
    if "FORUM_CHANNEL_TYPES" in source:
        pytest.skip("main already supports forums")
    module = types.ModuleType("proactive_agent._main_runtime")
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module.__name__] = module
    exec(compile(source, "main:runtime.py", "exec"), module.__dict__)  # noqa: S102
    return module


async def test_an_old_worker_never_sees_a_forum_and_answers_no_post():
    # New application, old worker: the old worker does not ask for forum
    # rows, so the application leaves the forum's row out. Even a post wake
    # reaching it (the bot sends none without the capability key) is
    # answered nowhere: not in the post, not in the forum.
    old = _main_runtime_module()
    engine = FakeEngine(respond_in=(POST, FORUM))
    runtime = make_runtime(
        [EnabledChannel(channel_id=TEXT, watch_addendum="")], engine=engine
    )
    runtime._persist_instructions = old.GuildRuntime._persist_instructions.__get__(
        runtime
    )

    await old.GuildRuntime.process(runtime, wake_batch(POST))

    assert FORUM not in engine.seen["enabled"]
    assert POST not in engine.seen["enabled"]
    runtime.discord.send_message.assert_not_awaited()
