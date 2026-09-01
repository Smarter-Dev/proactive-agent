"""Generated Python representation of the proactive v1 wire contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

NotificationKind = Literal[
    "mention",
    "reply_to_bot",
    "watcher_summary",
    "new_messages",
    "mode_change",
    "instruction_expired",
    "recovery",
    "reaction",
    "channel_enabled",
]


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    cache_read_tokens: int = Field(0, ge=0)


class NotificationEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    notification_id: UUID
    guild_id: str = Field(pattern=r"^[0-9]{1,20}$")
    channel_id: str = Field(pattern=r"^[0-9]{1,20}$")
    channel_name: str = Field(max_length=100)
    kind: NotificationKind
    created_at: datetime
    body: str
    message_ids: tuple[str, ...]
    wakes: bool
    passive: bool
    watcher_usage: dict[str, TokenUsage]
    trace_id: UUID


class ControlCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    command_id: UUID
    guild_id: str = Field(pattern=r"^[0-9]{1,20}$")
    channel_id: str = Field(pattern=r"^[0-9]{1,20}$")
    mode: Literal["active", "passive"]
    minutes: int = Field(ge=0, le=1440)
    created_at: datetime
    trace_id: UUID


class HistorySnapshot(BaseModel):
    """The Redis/API representation of one guild's model history."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    guild_id: str = Field(pattern=r"^[0-9]{1,20}$")
    revision: int = Field(ge=0)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    history: list[dict]


class EnabledChannel(BaseModel):
    channel_id: str
    watch_addendum: str
