"""Generated Python representation of the proactive v1 wire contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

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
    guild_id: str = Field(pattern=r"^[0-9]{1,22}$")
    revision: int = Field(ge=0)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    history: list[dict]


class EnabledChannel(BaseModel):
    channel_id: str
    watch_addendum: str


SNOWFLAKE_PATTERN = r"^[0-9]{15,22}$"
Snowflake = Annotated[str, StringConstraints(pattern=SNOWFLAKE_PATTERN)]
PurgeName = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class PurgeCommand(BaseModel):
    """One per-user privacy purge run, read from the ``privacy:v1:purge`` stream.

    It carries the target's id and names, so it must never be logged.
    Schema copy: contracts/privacy/v1/purge_command.schema.json.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    request_id: UUID
    run_id: UUID
    user_id: Snowflake
    names: list[PurgeName] = Field(max_length=20)
    guild_ids: list[Snowflake] = Field(min_length=1, max_length=500)
    created_at: AwareDatetime

    def __repr__(self) -> str:
        return f"PurgeCommand(run_id={self.run_id}, guilds={len(self.guild_ids)})"

    __str__ = __repr__


class BlockedUsersList(BaseModel):
    """``GET /api/privacy/blocked-users``: users whose messages never reach
    the model. Purges add to it today; the opt-out list takes it over (#74)."""

    revision: int = Field(ge=0)
    user_ids: list[Snowflake]
    # People who opted back in to the AI assistant (smarter-dev #92): their
    # messages written before the time stay hidden. Absent from older servers.
    read_from: dict[Snowflake, AwareDatetime] = Field(default_factory=dict)
