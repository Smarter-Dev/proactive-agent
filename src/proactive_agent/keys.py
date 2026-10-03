"""Redis key construction shared by worker queue and history components."""

from __future__ import annotations

KEY_PREFIX = "proactive:v1"
READY_GUILDS_KEY = f"{KEY_PREFIX}:guilds-with-wakes"
READY_STREAM_KEY = f"{KEY_PREFIX}:ready"
DEAD_LETTER_STREAM_KEY = f"{KEY_PREFIX}:dead-letter"
LEGACY_HISTORY_PREFIX = "proactive:guild-history"


def guild_tag(guild_id: str) -> str:
    # Up to 22 digits, as the privacy purge contract allows for snowflakes.
    if not guild_id.isdigit() or len(guild_id) > 22:
        raise ValueError("guild_id must be a Discord snowflake")
    return f"{{guild:{guild_id}}}"


def wake_stream_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:wake"


def pending_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:pending"


def pending_dropped_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:pending-dropped"


def ownership_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:owner"


def batch_key(guild_id: str, wake_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:batch:{wake_id}"


def batch_dropped_key(guild_id: str, wake_id: str) -> str:
    return f"{batch_key(guild_id, wake_id)}:dropped"


def lease_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:lease"


def checkpoint_key(guild_id: str, wake_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:checkpoint:{wake_id}"


def attempts_key(guild_id: str, wake_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:attempts:{wake_id}"


def failure_notice_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:failure-notice"


def history_key(guild_id: str) -> str:
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:history"


def legacy_history_key(guild_id: str) -> str:
    return f"{LEGACY_HISTORY_PREFIX}:{guild_id}"


def control_stream_key() -> str:
    return f"{KEY_PREFIX}:control"


def purge_epoch_key(guild_id: str) -> str:
    """Bumped after every privacy purge of the guild's history.

    Runtimes reload history when it moves, and its existence alone forbids
    restoring the pre-split legacy history key for the guild.
    """
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:purge-epoch"


def privacy_lock_key(guild_id: str) -> str:
    """Purge fence for a guild the external worker does not own."""
    return f"{KEY_PREFIX}:{guild_tag(guild_id)}:privacy-lock"


PRIVACY_PURGE_STREAM_KEY = "privacy:v1:purge"


def privacy_enforcing_key(component: str) -> str:
    """Set after each blocked-users fetch; tells the purge job we enforce it."""
    return f"privacy:v1:enforcing:{component}"
