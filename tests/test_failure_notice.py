"""A dead-lettered wake has to say so somewhere a human will see it.

The dead-letter stream is durable but unwatched, so before this the only
signal that the agent had given up was the agent going quiet.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from proactive_agent.runtime import (
    FAILURE_NOTICE_ERROR_LIMIT,
    FAILURE_NOTICE_INTERVAL_SECONDS,
    GuildRuntime,
    failure_notice_text,
)

GUILD_ID = "111"


def envelope(channel_id: str) -> SimpleNamespace:
    return SimpleNamespace(channel_id=channel_id)


def batch(
    waking: tuple[str, ...] = (), pending: tuple[str, ...] = ()
) -> SimpleNamespace:
    waking_items = tuple(SimpleNamespace(envelope=envelope(cid)) for cid in waking)
    pending_items = tuple(envelope(cid) for cid in pending)
    return SimpleNamespace(
        guild_id=GUILD_ID,
        wake_id="wake-1",
        waking=list(waking_items),
        notifications=pending_items + tuple(item.envelope for item in waking_items),
    )


def runtime(enabled: tuple[str, ...], *, claimed: bool = True) -> GuildRuntime:
    api = SimpleNamespace(
        list_enabled_channels=AsyncMock(
            return_value=tuple(SimpleNamespace(channel_id=cid) for cid in enabled)
        )
    )
    return GuildRuntime(
        guild_id=GUILD_ID,
        engine=None,
        history_repository=None,
        history_writer=None,
        api=api,
        discord=SimpleNamespace(send_message=AsyncMock()),
        redis=None,
        queue=SimpleNamespace(claim_failure_notice=AsyncMock(return_value=claimed)),
        journal=None,
        bot_user_id="bot",
        guild_name="guild",
        summarize_web=None,
        image_capabilities=None,
        media_reader=None,
        author_handler=None,
    )


async def test_notice_lands_in_the_channel_the_wake_was_for():
    agent = runtime(enabled=("aaa", "bbb"))

    channel_id = await agent.report_failure(batch(waking=("bbb",)), "boom")

    assert channel_id == "bbb"
    agent.discord.send_message.assert_awaited_once()
    posted_channel, content = agent.discord.send_message.await_args.args
    assert posted_channel == "bbb"
    assert "boom" in content


async def test_notice_falls_back_to_an_enabled_channel():
    # The wake's own channel was disabled between the wake and the failure.
    agent = runtime(enabled=("aaa",))

    channel_id = await agent.report_failure(batch(waking=("ccc",)), "boom")

    assert channel_id == "aaa"
    agent.discord.send_message.assert_awaited_once()


async def test_pending_channel_is_preferred_over_an_arbitrary_one():
    agent = runtime(enabled=("aaa", "bbb"))

    channel_id = await agent.report_failure(batch(pending=("bbb",)), "boom")

    assert channel_id == "bbb"


async def test_no_enabled_channel_posts_nothing_and_keeps_the_window_open():
    agent = runtime(enabled=())

    channel_id = await agent.report_failure(batch(waking=("bbb",)), "boom")

    assert channel_id is None
    agent.discord.send_message.assert_not_awaited()
    # Spending the window here would silence the next six hours for nothing.
    agent.queue.claim_failure_notice.assert_not_awaited()


async def test_second_failure_in_the_window_stays_quiet():
    agent = runtime(enabled=("aaa",), claimed=False)

    channel_id = await agent.report_failure(batch(waking=("aaa",)), "boom")

    assert channel_id is None
    agent.discord.send_message.assert_not_awaited()


async def test_window_is_six_hours():
    agent = runtime(enabled=("aaa",))

    await agent.report_failure(batch(waking=("aaa",)), "boom")

    assert agent.queue.claim_failure_notice.await_args.kwargs == {
        "ttl_seconds": FAILURE_NOTICE_INTERVAL_SECONDS
    }
    assert FAILURE_NOTICE_INTERVAL_SECONDS == 6 * 60 * 60


@pytest.mark.parametrize(
    "error, expected",
    [
        ("plain failure", "plain failure"),
        ("fence ``` inside", "fence ''' inside"),
    ],
)
def test_notice_text_neutralises_code_fences(error: str, expected: str):
    assert expected in failure_notice_text(error)


def test_notice_text_stays_inside_the_discord_limit():
    text = failure_notice_text("x" * 10_000)

    assert len(text) < 2000
    assert "x" * FAILURE_NOTICE_ERROR_LIMIT in text
    assert "x" * (FAILURE_NOTICE_ERROR_LIMIT + 1) not in text
