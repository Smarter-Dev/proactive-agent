"""REST-only Discord client used by the extracted proactive agent."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from urllib.parse import quote

import httpx

from proactive_agent.types import ChannelMessage

DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordRESTError(Exception):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


class DiscordREST:
    def __init__(
        self,
        *,
        bot_token: str,
        timeout: float = 15,
        transport: httpx.AsyncBaseTransport | None = None,
        api_base: str = DISCORD_API_BASE,
    ):
        self._client = httpx.AsyncClient(
            base_url=api_base,
            headers={
                "Authorization": f"Bot {bot_token}",
                "User-Agent": "SmarterDev-ProactiveAgent/1.0",
            },
            timeout=timeout,
            transport=transport,
        )
        self._role_names: dict[str, dict[str, str]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    async def current_user(self) -> dict[str, Any]:
        return (await self._request("GET", "/users/@me")).json()

    async def guild(self, guild_id: str) -> dict[str, Any]:
        return (await self._request("GET", f"/guilds/{guild_id}")).json()

    async def channel(self, channel_id: str) -> dict[str, Any]:
        return (await self._request("GET", f"/channels/{channel_id}")).json()

    async def fetch_message(
        self, channel_id: str, message_id: str, *, guild_id: str | None = None
    ) -> ChannelMessage:
        record = (
            await self._request("GET", f"/channels/{channel_id}/messages/{message_id}")
        ).json()
        role_names = await self._roles(guild_id) if guild_id else {}
        return self._message(record, role_names)

    async def channel_history(
        self,
        channel_id: str,
        *,
        guild_id: str,
        limit: int = 60,
        before_id: str | None = None,
    ) -> list[ChannelMessage]:
        remaining = max(1, min(limit, 100))
        params: dict[str, Any] = {"limit": remaining}
        if before_id:
            params["before"] = before_id
        records = (
            await self._request(
                "GET", f"/channels/{channel_id}/messages", params=params
            )
        ).json()
        role_names = await self._roles(guild_id)
        messages = [self._message(record, role_names) for record in records]
        messages.reverse()
        return messages

    async def send_message(
        self,
        channel_id: str,
        content: str,
        *,
        reply_to_id: str | None = None,
        suppress_embeds: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": content}
        if suppress_embeds:
            payload["flags"] = 4
        if reply_to_id:
            payload["message_reference"] = {
                "message_id": reply_to_id,
                "channel_id": channel_id,
            }
            payload["allowed_mentions"] = {"replied_user": False}
        return (
            await self._request(
                "POST", f"/channels/{channel_id}/messages", json=payload
            )
        ).json()

    async def send_files(
        self,
        channel_id: str,
        files: list[tuple[str, bytes, str]],
        *,
        reply_to_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if reply_to_id:
            payload["message_reference"] = {"message_id": reply_to_id}
        multipart = [
            (f"files[{index}]", (filename, data, mime_type))
            for index, (filename, data, mime_type) in enumerate(files)
        ]
        return (
            await self._request(
                "POST",
                f"/channels/{channel_id}/messages",
                data={"payload_json": __import__("json").dumps(payload)},
                files=multipart,
            )
        ).json()

    async def add_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        await self._request(
            "PUT",
            f"/channels/{channel_id}/messages/{message_id}/reactions/"
            f"{quote(emoji, safe='')}/@me",
        )

    async def guild_emojis(self, guild_id: str) -> list[dict[str, Any]]:
        return (await self._request("GET", f"/guilds/{guild_id}/emojis")).json()

    async def _roles(self, guild_id: str | None) -> dict[str, str]:
        if not guild_id:
            return {}
        cached = self._role_names.get(guild_id)
        if cached is not None:
            return cached
        records = (await self._request("GET", f"/guilds/{guild_id}/roles")).json()
        cached = {str(record["id"]): record["name"] for record in records}
        self._role_names[guild_id] = cached
        return cached

    def _message(
        self, record: dict[str, Any], role_names: dict[str, str]
    ) -> ChannelMessage:
        author = record["author"]
        member = record.get("member") or {}
        display = (
            member.get("nick")
            or author.get("global_name")
            or author.get("username")
            or str(author["id"])
        )
        reference = record.get("message_reference") or {}
        reaction_counts = {
            reaction["emoji"].get("name") or str(reaction["emoji"].get("id")): int(
                reaction.get("count", 0)
            )
            for reaction in record.get("reactions", ())
        }
        return ChannelMessage(
            id=str(record["id"]),
            timestamp=datetime.fromisoformat(
                record["timestamp"].replace("Z", "+00:00")
            ),
            author_id=str(author["id"]),
            author_name=author.get("username") or str(author["id"]),
            author_display=display,
            is_bot=bool(author.get("bot", False)),
            content=record.get("content") or "",
            reply_to_id=(
                str(reference["message_id"])
                if reference.get("message_id") is not None
                else None
            ),
            mention_user_ids=tuple(
                str(user["id"]) for user in record.get("mentions", ())
            ),
            mention_everyone=bool(record.get("mention_everyone", False)),
            attachment_count=len(record.get("attachments", ())),
            sticker_count=len(record.get("sticker_items", ())),
            message_type=int(record.get("type", 0)),
            reaction_counts=reaction_counts,
            roles=tuple(
                role_names[role_id]
                for role_id in member.get("roles", ())
                if role_id in role_names
            ),
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        for attempt in range(4):
            try:
                response = await self._client.request(method, path, **kwargs)
            except httpx.HTTPError as error:
                raise DiscordRESTError(None, f"{method} {path}: {error}") from error
            if response.status_code != 429:
                break
            if attempt == 3:
                break
            try:
                delay = float(response.json().get("retry_after", 1))
            except (TypeError, ValueError):
                delay = 1
            await asyncio.sleep(min(60, max(0, delay)))
        if response.status_code >= 400:
            raise DiscordRESTError(
                response.status_code,
                f"{method} {path} -> {response.status_code}: {response.text[:500]}",
            )
        return response
