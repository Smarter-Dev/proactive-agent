"""Minimal Kubernetes-compatible health server without a web dependency."""

from __future__ import annotations

import asyncio


class HealthServer:
    def __init__(
        self, redis_client, api=None, *, host: str = "0.0.0.0", port: int = 8080
    ):
        self._redis = redis_client
        self._api = api
        self._host = host
        self._port = port
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, self._host, self._port)

    async def close(self) -> None:
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request_line = await reader.readline()
        path = request_line.decode(errors="replace").split(" ")[1:2]
        healthy = bool(path and path[0] == "/healthz")
        if path and path[0] == "/readyz":
            try:
                redis_ready = bool(await self._redis.ping())
                api_ready = (
                    True
                    if self._api is None
                    else bool(await self._api.validate_credentials())
                )
                healthy = redis_ready and api_ready
            except Exception:
                healthy = False
        status = "200 OK" if healthy else "503 Service Unavailable"
        body = b"ok\n" if healthy else b"unavailable\n"
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Length: {len(body)}\r\n"
            "Content-Type: text/plain\r\nConnection: close\r\n\r\n".encode()
            + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()
