"""Someone who opted back in to the AI assistant (smarter-dev #92).

Opting back in applies to new messages only. The person leaves the list's
``user_ids`` but keeps a ``read_from`` time, and a message they wrote before
it must still reach the model as ``[BLOCKED BY USER]``, together with any
reply marker pointing at it. A message written after it reads normally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import fakeredis.aioredis
import httpx
from purge_fakes import BYSTANDER, CHANNEL, GUILD, TARGET, FakeAPI

from proactive_agent.blocked_users import BlockedUsers, first_snowflake
from proactive_agent.contracts import BlockedUsersList
from proactive_agent.discord import DiscordREST
from proactive_agent.types import BlockedMessage, ChannelMessage

OPTED_IN = datetime(2026, 10, 6, 20, 0, tzinfo=UTC)
OLD = str(first_snowflake(OPTED_IN - timedelta(hours=1)) + 1)
NEW = str(first_snowflake(OPTED_IN + timedelta(minutes=1)) + 1)
REPLY = str(first_snowflake(OPTED_IN + timedelta(minutes=2)) + 1)
OLD_TEXT = "written while opted out"
NEW_TEXT = "written after opting back in"


class ListAPI(FakeAPI):
    async def get_blocked_users(self):
        return BlockedUsersList.model_validate(await super().get_blocked_users())


def _iso(snowflake: str) -> str:
    ms = (int(snowflake) >> 22) + 1_420_070_400_000
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def _messages() -> list[dict]:
    old = {
        "id": OLD,
        "timestamp": _iso(OLD),
        "author": {"id": TARGET, "username": "kai_rs"},
        "content": OLD_TEXT,
    }
    return [
        {
            "id": REPLY,
            "timestamp": _iso(REPLY),
            "author": {"id": BYSTANDER, "username": "nia"},
            "content": "agreed",
            "message_reference": {"message_id": OLD},
            "referenced_message": old,
        },
        {
            "id": NEW,
            "timestamp": _iso(NEW),
            "author": {"id": TARGET, "username": "kai_rs"},
            "content": NEW_TEXT,
        },
        old,
    ]


async def _opted_in_list() -> BlockedUsers:
    api = ListAPI()
    api.blocked = {
        "revision": 3,
        "user_ids": [],
        "read_from": {TARGET: OPTED_IN.isoformat()},
    }
    blocked = BlockedUsers(api, fakeredis.aioredis.FakeRedis())
    assert await blocked.refresh()
    return blocked


async def test_messages_before_the_opt_in_stay_hidden_and_new_ones_read():
    blocked = await _opted_in_list()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(f"/channels/{CHANNEL}/messages"):
            return httpx.Response(200, json=_messages())
        if request.url.path.endswith(f"/guilds/{GUILD}/roles"):
            return httpx.Response(200, json=[])
        raise AssertionError(request.url.path)

    discord = DiscordREST(
        bot_token="t",
        api_base="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
        blocked_users=blocked,
    )
    history = await discord.channel_history(CHANNEL, guild_id=GUILD)
    await discord.close()

    rendered = repr(history)
    assert OLD_TEXT not in rendered and OLD not in rendered
    assert BlockedMessage() in history
    new = [m for m in history if isinstance(m, ChannelMessage) and m.id == NEW]
    assert new and new[0].content == NEW_TEXT
    reply = [m for m in history if isinstance(m, ChannelMessage) and m.id == REPLY]
    assert reply and reply[0].reply_to_id is None


async def test_the_cutoff_needs_the_message_id_and_fails_closed_on_a_bad_one():
    blocked = await _opted_in_list()

    assert not blocked.is_blocked(TARGET)  # a live event is always newer
    assert blocked.is_blocked(TARGET, OLD)
    assert not blocked.is_blocked(TARGET, NEW)
    assert blocked.is_blocked(TARGET, "not-a-snowflake")
    assert not blocked.is_blocked(BYSTANDER, OLD)


async def test_an_older_server_without_read_from_still_parses():
    listing = BlockedUsersList.model_validate({"revision": 1, "user_ids": [TARGET]})
    assert listing.read_from == {}
