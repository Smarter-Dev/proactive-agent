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


async def test_privacy_ack_posts_to_the_run_and_reports_unknown_runs():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "unknown" in request.url.path:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json={"accepted": True})

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="sk_test",
        transport=httpx.MockTransport(handler),
    )
    try:
        accepted = await api.post_privacy_ack(
            "run-1",
            component="worker",
            guild_id="333333333333333333",
            outcome="purged",
            stores=["proactive:v1:history"],
            detail="d" * 600,
        )
        unknown = await api.post_privacy_ack(
            "unknown",
            component="worker",
            guild_id="333333333333333333",
            outcome="failed",
            stores=[],
            detail="",
        )
    finally:
        await api.close()

    assert accepted is True
    assert unknown is False
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/api/privacy/purges/run-1/acks"
    body = json.loads(seen[0].content)
    assert body == {
        "component": "worker",
        "guild_id": "333333333333333333",
        "outcome": "purged",
        "stores": ["proactive:v1:history"],
        "detail": "d" * 500,
        "name_hits": {},
        "tombstoned": False,
        "unchecked_names": 0,
        "done_record": "not_written",
    }
