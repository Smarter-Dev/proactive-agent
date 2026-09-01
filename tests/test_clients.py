from __future__ import annotations

import json

import httpx

from proactive_agent.api import ApplicationAPI
from proactive_agent.discord import DiscordREST


async def test_application_api_uses_bearer_auth_and_expected_paths():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/auth/validate"):
            return httpx.Response(200, json={"valid": True})
        if request.url.path.endswith("/proactive-settings"):
            return httpx.Response(
                200, json=[{"channel_id": "22", "watch_addendum": "watch builds"}]
            )
        raise AssertionError(request.url)

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="sk_test",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert await api.validate_credentials()
        channels = await api.list_enabled_channels("11")
    finally:
        await api.close()

    assert channels[0].channel_id == "22"
    assert [request.url.path for request in seen] == [
        "/api/auth/validate",
        "/api/guilds/11/proactive-settings",
    ]
    assert all(request.headers["authorization"] == "Bearer sk_test" for request in seen)


async def test_discord_rest_preserves_reply_anchor_and_suppresses_embeds():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "99"})

    discord = DiscordREST(
        bot_token="token",
        api_base="https://discord.test/api/v10",
        transport=httpx.MockTransport(handler),
    )
    try:
        sent = await discord.send_message(
            "22", "hello", reply_to_id="33", suppress_embeds=True
        )
        await discord.add_reaction("22", "33", "👍")
    finally:
        await discord.close()

    payload = json.loads(seen[0].content)
    assert sent["id"] == "99"
    assert payload == {
        "content": "hello",
        "flags": 4,
        "message_reference": {"message_id": "33", "channel_id": "22"},
        "allowed_mentions": {"replied_user": False},
    }
    assert seen[0].headers["authorization"] == "Bot token"
    assert seen[1].method == "PUT"
    assert seen[1].url.path.endswith("/channels/22/messages/33/reactions/👍/@me")
