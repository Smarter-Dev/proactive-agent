"""Authenticated application REST client for all durable worker data."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from proactive_agent.contracts import EnabledChannel, HistorySnapshot


class ApplicationAPIError(Exception):
    def __init__(self, status_code: int | None, message: str):
        super().__init__(message)
        self.status_code = status_code


class ApplicationAPI:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout: float = 15,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "SmarterDev-ProactiveAgent/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=timeout,
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def validate_credentials(self) -> bool:
        response = await self._request("POST", "/auth/validate")
        return bool(response.json().get("valid"))

    async def list_enabled_channels(self, guild_id: str) -> tuple[EnabledChannel, ...]:
        response = await self._request("GET", f"/guilds/{guild_id}/proactive-settings")
        return tuple(EnabledChannel.model_validate(item) for item in response.json())

    async def get_history(self, guild_id: str) -> HistorySnapshot | None:
        response = await self._request(
            "GET",
            f"/guilds/{guild_id}/proactive-agent/history",
            allow_not_found=True,
        )
        if response is None:
            return None
        body = response.json()
        return HistorySnapshot.model_validate(
            {
                "schema_version": body["schema_version"],
                "guild_id": body["guild_id"],
                "revision": body["revision"],
                "checksum": body["checksum"],
                "history": body["history"],
            }
        )

    async def put_history(self, snapshot: HistorySnapshot) -> None:
        await self._request(
            "PUT",
            f"/guilds/{snapshot.guild_id}/proactive-agent/history",
            json={
                "schema_version": snapshot.schema_version,
                "revision": snapshot.revision,
                "checksum": snapshot.checksum,
                "history": snapshot.history,
            },
        )

    async def set_watch_addendum(
        self,
        *,
        guild_id: str,
        channel_id: str,
        enabled: bool,
        watch_addendum: str,
    ) -> dict[str, Any]:
        response = await self._request(
            "PUT",
            f"/guilds/{guild_id}/channels/{channel_id}/proactive-settings",
            json={"enabled": enabled, "watch_addendum": watch_addendum},
        )
        return response.json()

    async def record_usage(
        self,
        *,
        guild_id: str,
        wake_id: str,
        metered_at: datetime,
        passive: bool,
        responses: int,
        entries: list[dict],
    ) -> None:
        await self._request(
            "POST",
            f"/guilds/{guild_id}/channels/guild-wide/proactive-settings/usage",
            json={
                "wake_id": wake_id,
                "metered_at": metered_at.isoformat(),
                "passive": passive,
                "responses": responses,
                "entries": entries,
            },
        )

    async def get_memory(self, guild_id: str) -> dict[str, Any] | None:
        response = await self._request(
            "GET", f"/guilds/{guild_id}/chat-memory", allow_not_found=True
        )
        return None if response is None else response.json()

    async def save_memory_note(
        self, guild_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        response = await self._request(
            "POST", f"/guilds/{guild_id}/chat-memory/notes", json=payload
        )
        return response.json()

    async def reserve_search_preview(self, query: str) -> dict[str, Any]:
        return (
            await self._request("POST", "/search-previews", json={"query": query})
        ).json()

    async def complete_search_preview(
        self, preview_id: str, results: list[dict[str, str]]
    ) -> None:
        await self._request(
            "PUT", f"/search-previews/{preview_id}", json={"results": results}
        )

    async def fail_search_preview(self, preview_id: str) -> None:
        await self._request("POST", f"/search-previews/{preview_id}/failed")

    async def list_handlers(
        self, channel_id: str, *, include_scripts: bool = False
    ) -> list[dict[str, Any]]:
        response = await self._request(
            "GET",
            "/handlers",
            params={
                "channel_id": channel_id,
                "include_scripts": str(include_scripts).lower(),
            },
        )
        return list(response.json())

    async def create_handler(self, payload: dict[str, Any]) -> dict[str, Any]:
        return (await self._request("POST", "/handlers", json=payload)).json()

    async def update_handler(
        self, handler_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return (
            await self._request("PUT", f"/handlers/{handler_id}", json=payload)
        ).json()

    async def delete_handler(self, handler_id: str) -> bool:
        response = await self._request(
            "DELETE", f"/handlers/{handler_id}", allow_not_found=True
        )
        return response is not None

    async def image_quota(self, guild_id: str) -> dict[str, Any]:
        return (
            await self._request(
                "GET", "/image-generations/quota", params={"guild_id": guild_id}
            )
        ).json()

    async def reserve_image(self, guild_id: str) -> dict[str, Any]:
        return (
            await self._request(
                "POST", "/image-generations/reserve", json={"guild_id": guild_id}
            )
        ).json()

    async def release_image(self, guild_id: str) -> None:
        await self._request(
            "POST", "/image-generations/release", json={"guild_id": guild_id}
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        allow_not_found: bool = False,
        **kwargs,
    ) -> httpx.Response | None:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApplicationAPIError(None, f"{method} {path}: {error}") from error
        if allow_not_found and response.status_code == 404:
            return None
        if response.status_code >= 400:
            raise ApplicationAPIError(
                response.status_code,
                f"{method} {path} -> {response.status_code}: {response.text[:500]}",
            )
        return response
