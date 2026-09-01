"""Redis Stream consumer with guild isolation and crash-safe pending batches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import uuid4

from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from proactive_agent.contracts import NotificationEnvelope
from proactive_agent.keys import (
    DEAD_LETTER_STREAM_KEY,
    READY_STREAM_KEY,
    attempts_key,
    batch_dropped_key,
    batch_key,
    lease_key,
    ownership_key,
    pending_dropped_key,
    pending_key,
    wake_stream_key,
)

READY_GROUP = "proactive-agent-workers-v1"
WAKE_GROUP = "proactive-agent-v1"
WAKE_PAYLOAD_FIELD = "payload"

_CLAIM_PENDING_LUA = """
if redis.call('EXISTS', KEYS[2]) == 0 then
  if redis.call('EXISTS', KEYS[1]) == 1 then
    redis.call('RENAME', KEYS[1], KEYS[2])
  end
  local dropped = redis.call('GET', KEYS[3])
  if dropped then
    redis.call('SET', KEYS[4], dropped)
    redis.call('DEL', KEYS[3])
  end
end
local values = redis.call('LRANGE', KEYS[2], 0, -1)
local dropped = redis.call('GET', KEYS[4]) or '0'
table.insert(values, 1, dropped)
return values
"""

_RELEASE_LEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

_RENEW_LEASE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1]
  and redis.call('GET', KEYS[2]) == 'external' then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

_ACQUIRE_LEASE_LUA = """
if redis.call('GET', KEYS[2]) ~= 'external' then
  return 0
end
return redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2]) and 1 or 0
"""

_APPEND_PENDING_LUA = """
local values = redis.call('LRANGE', KEYS[1], 0, -1)
for _, value in ipairs(values) do
  redis.call('RPUSH', KEYS[2], value)
end
if #values > 0 then
  redis.call('DEL', KEYS[1])
end
local dropped = tonumber(redis.call('GET', KEYS[3]) or '0')
if dropped > 0 then
  redis.call('INCRBY', KEYS[4], dropped)
  redis.call('DEL', KEYS[3])
end
table.insert(values, 1, tostring(dropped))
return values
"""


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class ReadyRecord:
    stream_id: str
    guild_id: str


@dataclass(frozen=True)
class StreamNotification:
    stream_id: str
    envelope: NotificationEnvelope


@dataclass
class WakeBatch:
    guild_id: str
    wake_id: str
    ready_ids: list[str]
    waking: list[StreamNotification]
    pending: list[NotificationEnvelope]
    dropped: int

    @property
    def notifications(self) -> tuple[NotificationEnvelope, ...]:
        return (*self.pending, *(item.envelope for item in self.waking))

    @property
    def passive(self) -> bool:
        return bool(self.waking) and all(item.envelope.passive for item in self.waking)


class GuildLease:
    def __init__(
        self,
        redis_client,
        *,
        guild_id: str,
        token: str,
        ttl_seconds: int,
    ):
        self._redis = redis_client
        self.guild_id = guild_id
        self.token = token
        self.ttl_seconds = ttl_seconds

    async def renew(self) -> bool:
        result = await self._redis.eval(
            _RENEW_LEASE_LUA,
            2,
            lease_key(self.guild_id),
            ownership_key(self.guild_id),
            self.token,
            self.ttl_seconds * 1000,
        )
        return bool(result)

    async def release(self) -> None:
        await self._redis.eval(
            _RELEASE_LEASE_LUA,
            1,
            lease_key(self.guild_id),
            self.token,
        )

    async def __aenter__(self) -> GuildLease:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


class RedisWakeQueue:
    """Consume ready signals and guild streams without sharing guild state."""

    def __init__(
        self,
        redis_client,
        *,
        consumer_name: str,
        lease_seconds: int = 180,
        reclaim_idle_seconds: int = 240,
        max_batch_size: int = 20,
    ):
        self._redis = redis_client
        self.consumer_name = consumer_name
        self.lease_seconds = lease_seconds
        self.reclaim_idle_seconds = reclaim_idle_seconds
        self.max_batch_size = max_batch_size
        self._known_guild_groups: set[str] = set()

    async def initialize(self) -> None:
        await self._ensure_group(READY_STREAM_KEY, READY_GROUP)

    async def read_ready(self, *, block_ms: int = 30_000) -> tuple[ReadyRecord, ...]:
        try:
            records = await self._redis.xreadgroup(
                READY_GROUP,
                self.consumer_name,
                {READY_STREAM_KEY: ">"},
                count=self.max_batch_size,
                block=block_ms,
            )
        except RedisTimeoutError:
            return ()
        ready = []
        for _stream, entries in records or ():
            for stream_id, fields in entries:
                guild_id = fields.get(b"guild_id", fields.get("guild_id"))
                ready.append(
                    ReadyRecord(
                        stream_id=_decode(stream_id),
                        guild_id=_decode(guild_id),
                    )
                )
        return tuple(ready)

    async def reclaim_ready(self) -> tuple[ReadyRecord, ...]:
        result = await self._redis.xautoclaim(
            READY_STREAM_KEY,
            READY_GROUP,
            self.consumer_name,
            self.reclaim_idle_seconds * 1000,
            "0-0",
            count=self.max_batch_size,
        )
        entries = result[1] if result else ()
        return tuple(
            ReadyRecord(
                stream_id=_decode(stream_id),
                guild_id=_decode(fields.get(b"guild_id", fields.get("guild_id"))),
            )
            for stream_id, fields in entries
        )

    async def acquire_lease(self, guild_id: str) -> GuildLease | None:
        token = uuid4().hex
        acquired = await self._redis.eval(
            _ACQUIRE_LEASE_LUA,
            2,
            lease_key(guild_id),
            ownership_key(guild_id),
            token,
            self.lease_seconds,
        )
        if not acquired:
            return None
        return GuildLease(
            self._redis,
            guild_id=guild_id,
            token=token,
            ttl_seconds=self.lease_seconds,
        )

    async def externally_owned(self, guild_id: str) -> bool:
        value = await self._redis.get(ownership_key(guild_id))
        return _decode(value) == "external" if value is not None else False

    async def discard_embedded_ready(
        self, guild_id: str, ready_records: tuple[ReadyRecord, ...]
    ) -> None:
        """Drain stale external work after ownership returns to the bot."""
        batch = await self.build_batch(guild_id, ready_records)
        if batch is None:
            await self.acknowledge_ready(ready_records)
            return
        if await self.externally_owned(guild_id):
            return
        await self.acknowledge(batch)

    async def build_batch(
        self,
        guild_id: str,
        ready_records: tuple[ReadyRecord, ...],
    ) -> WakeBatch | None:
        await self._ensure_guild_group(guild_id)
        reclaimed = await self._redis.xautoclaim(
            wake_stream_key(guild_id),
            WAKE_GROUP,
            self.consumer_name,
            self.reclaim_idle_seconds * 1000,
            "0-0",
            count=self.max_batch_size,
        )
        records = await self._redis.xreadgroup(
            WAKE_GROUP,
            self.consumer_name,
            {wake_stream_key(guild_id): ">"},
            count=self.max_batch_size,
        )
        waking = []
        for stream_id, fields in reclaimed[1] if reclaimed else ():
            payload = fields.get(b"payload", fields.get("payload"))
            waking.append(
                StreamNotification(
                    stream_id=_decode(stream_id),
                    envelope=NotificationEnvelope.model_validate_json(_decode(payload)),
                )
            )
        for _stream, entries in records or ():
            for stream_id, fields in entries:
                payload = fields.get(b"payload", fields.get("payload"))
                waking.append(
                    StreamNotification(
                        stream_id=_decode(stream_id),
                        envelope=NotificationEnvelope.model_validate_json(
                            _decode(payload)
                        ),
                    )
                )
        if not waking:
            return None

        wake_id = str(waking[0].envelope.notification_id)
        raw_pending = await self._redis.eval(
            _CLAIM_PENDING_LUA,
            4,
            pending_key(guild_id),
            batch_key(guild_id, wake_id),
            pending_dropped_key(guild_id),
            batch_dropped_key(guild_id, wake_id),
        )
        dropped = int(_decode(raw_pending[0]))
        pending = tuple(
            NotificationEnvelope.model_validate_json(_decode(item))
            for item in raw_pending[1:]
        )
        return WakeBatch(
            guild_id=guild_id,
            wake_id=wake_id,
            ready_ids=[record.stream_id for record in ready_records],
            waking=list(waking),
            pending=list(pending),
            dropped=dropped,
        )

    async def drain_midrun(
        self, batch: WakeBatch
    ) -> tuple[tuple[NotificationEnvelope, ...], int]:
        """Attach notifications that arrived while this guild was running."""
        records = await self._redis.xreadgroup(
            WAKE_GROUP,
            self.consumer_name,
            {wake_stream_key(batch.guild_id): ">"},
            count=self.max_batch_size,
        )
        newly_waking = []
        for _stream, entries in records or ():
            for stream_id, fields in entries:
                payload = fields.get(b"payload", fields.get("payload"))
                newly_waking.append(
                    StreamNotification(
                        stream_id=_decode(stream_id),
                        envelope=NotificationEnvelope.model_validate_json(
                            _decode(payload)
                        ),
                    )
                )
        raw_pending = await self._redis.eval(
            _APPEND_PENDING_LUA,
            4,
            pending_key(batch.guild_id),
            batch_key(batch.guild_id, batch.wake_id),
            pending_dropped_key(batch.guild_id),
            batch_dropped_key(batch.guild_id, batch.wake_id),
        )
        dropped = int(_decode(raw_pending[0]))
        newly_pending = [
            NotificationEnvelope.model_validate_json(_decode(item))
            for item in raw_pending[1:]
        ]
        batch.waking.extend(newly_waking)
        batch.pending.extend(newly_pending)
        batch.dropped += dropped
        return (
            (*newly_pending, *(item.envelope for item in newly_waking)),
            dropped,
        )

    async def acknowledge(self, batch: WakeBatch) -> None:
        async with self._redis.pipeline(transaction=True) as pipeline:
            if batch.waking:
                waking_ids = tuple(item.stream_id for item in batch.waking)
                pipeline.xack(
                    wake_stream_key(batch.guild_id),
                    WAKE_GROUP,
                    *waking_ids,
                )
                pipeline.xdel(wake_stream_key(batch.guild_id), *waking_ids)
            if batch.ready_ids:
                pipeline.xack(
                    READY_STREAM_KEY,
                    READY_GROUP,
                    *batch.ready_ids,
                )
                pipeline.xdel(READY_STREAM_KEY, *batch.ready_ids)
            pipeline.delete(
                batch_key(batch.guild_id, batch.wake_id),
                batch_dropped_key(batch.guild_id, batch.wake_id),
                attempts_key(batch.guild_id, batch.wake_id),
            )
            await pipeline.execute()

    async def record_failure(
        self, batch: WakeBatch, *, error: str, max_attempts: int
    ) -> bool:
        """Record a failed wake; dead-letter and ack it at the retry ceiling.

        Returns true when the batch was moved aside and must not be retried.
        """
        key = attempts_key(batch.guild_id, batch.wake_id)
        attempts = int(await self._redis.incr(key))
        await self._redis.expire(key, 7 * 24 * 60 * 60)
        if attempts < max_attempts:
            return False
        payload = json.dumps(
            {
                "wake_id": batch.wake_id,
                "dropped": batch.dropped,
                "notifications": [
                    item.model_dump(mode="json") for item in batch.notifications
                ],
            },
            separators=(",", ":"),
        )
        await self.dead_letter(
            guild_id=batch.guild_id,
            stream_id=batch.waking[0].stream_id,
            payload=payload,
            error=error[:2000],
            attempts=attempts,
        )
        await self.acknowledge(batch)
        return True

    async def acknowledge_ready(self, records: tuple[ReadyRecord, ...]) -> None:
        if records:
            stream_ids = tuple(record.stream_id for record in records)
            await self._redis.xack(
                READY_STREAM_KEY,
                READY_GROUP,
                *stream_ids,
            )
            await self._redis.xdel(READY_STREAM_KEY, *stream_ids)

    async def dead_letter(
        self,
        *,
        guild_id: str,
        stream_id: str,
        payload: str,
        error: str,
        attempts: int,
    ) -> None:
        await self._redis.xadd(
            DEAD_LETTER_STREAM_KEY,
            {
                "guild_id": guild_id,
                "source_stream_id": stream_id,
                "payload": payload,
                "error": error,
                "attempts": str(attempts),
            },
            maxlen=10_000,
            approximate=True,
        )

    async def _ensure_guild_group(self, guild_id: str) -> None:
        if guild_id in self._known_guild_groups:
            return
        await self._ensure_group(wake_stream_key(guild_id), WAKE_GROUP)
        self._known_guild_groups.add(guild_id)

    async def _ensure_group(self, stream: str, group: str) -> None:
        try:
            await self._redis.xgroup_create(stream, group, id="0", mkstream=True)
        except ResponseError as error:
            if "BUSYGROUP" not in str(error):
                raise
