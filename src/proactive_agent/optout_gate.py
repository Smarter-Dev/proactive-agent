"""The opt-out gate at the writers: nothing about an opted-out member is kept.

Members who opt out of the assistant are on the blocked-users list. Model
input is already filtered (``discord.py``, ``runtime._without_blocked``); this
module is the one place the writers share:

- the ``remember`` tool refuses a note that names a blocked user's id;
- every history write blanks blocked authors' transcript lines and scrubs
  their ids from user-prompt and tool-return text;
- the memory block is scrubbed before it is rendered into the prompt.

``blocked_users`` is anything with ``is_blocked(user_id, message_id=None)``;
None blocks nobody (tests/evals).
"""

from __future__ import annotations

import re

from proactive_agent.agent import MEMBER_LINE_PREFIX
from proactive_agent.transcript import BLOCKED_LINE

MENTION_PATTERN = re.compile(r"<@!?([0-9]{15,22})>")
SNOWFLAKE_PATTERN = re.compile(r"(?<![0-9])([0-9]{15,22})(?![0-9])")
_LINE_MESSAGE_ID = re.compile(r"^\[id=([0-9]+)\]")
# The memory fields the web returns as free text (blob and its sections).
_MEMORY_TEXT_FIELDS = ("content", "behavior", "personality")


def _blocked(blocked_users, user_id, message_id=None) -> bool:
    return blocked_users is not None and blocked_users.is_blocked(
        str(user_id), message_id
    )


def scrub_blocked_ids(text: str, blocked_users) -> str:
    """Blocked users' ids out of the text.

    `<@id>`/`<@!id>` become `@[blocked user]`, a bare id `[blocked user]`.
    """
    if blocked_users is None:
        return text

    def mention(match: re.Match) -> str:
        return "@[blocked user]" if _blocked(blocked_users, match[1]) else match[0]

    def bare(match: re.Match) -> str:
        return "[blocked user]" if _blocked(blocked_users, match[1]) else match[0]

    text = MENTION_PATTERN.sub(mention, text)
    return SNOWFLAKE_PATTERN.sub(bare, text)


def names_blocked_user(text: str, blocked_users) -> bool:
    """Whether the text carries a blocked user's id (mention or bare)."""
    return scrub_blocked_ids(text, blocked_users) != text


def blank_blocked_lines(text: str, blocked_users) -> str:
    """A transcript-bearing text with opted-out members removed.

    A line render_transcript_line wrote for a blocked author becomes
    BLOCKED_LINE, and the continuation lines of that message (content holding
    newlines) go with it, up to the next transcript line. Every other
    appearance of a blocked id is scrubbed.
    """
    if blocked_users is None:
        return text
    kept: list[str] = []
    in_blocked_message = False
    for line in text.split("\n"):
        stripped = line.lstrip()
        member = MEMBER_LINE_PREFIX.match(stripped)
        if member is not None:
            message_id = _LINE_MESSAGE_ID.match(stripped)[1]
            in_blocked_message = _blocked(blocked_users, member[3], message_id)
            if in_blocked_message:
                kept.append(BLOCKED_LINE)
                continue
        elif stripped.startswith("[id=") or stripped == BLOCKED_LINE:
            in_blocked_message = False
        elif in_blocked_message:
            continue
        kept.append(scrub_blocked_ids(line, blocked_users))
    return "\n".join(kept)


def _scrub_content(content, blocked_users):
    if isinstance(content, str):
        return blank_blocked_lines(content, blocked_users)
    if isinstance(content, list):
        return [
            blank_blocked_lines(item, blocked_users) if isinstance(item, str) else item
            for item in content
        ]
    return content


def scrub_history(history: list[dict], blocked_users) -> list[dict]:
    """A copy of a serialized pydantic-ai history with opted-out members gone.

    User-prompt and tool-return string content is passed through
    blank_blocked_lines; everything else is kept as is. The input is not
    mutated.
    """
    if blocked_users is None:
        return history
    scrubbed = []
    for message in history:
        parts = message.get("parts")
        if not isinstance(parts, list):
            scrubbed.append(message)
            continue
        new_parts = []
        for part in parts:
            if isinstance(part, dict) and part.get("part_kind") in (
                "user-prompt",
                "tool-return",
            ):
                part = {
                    **part,
                    "content": _scrub_content(part.get("content"), blocked_users),
                }
            new_parts.append(part)
        scrubbed.append({**message, "parts": new_parts})
    return scrubbed


def drop_blocked_lines(text: str, blocked_users) -> str | None:
    """The text without every line that names a blocked user; the other
    lines byte-for-byte. None when nothing (but whitespace) is left."""
    if blocked_users is None:
        return text
    kept = "".join(
        line
        for line in text.splitlines(keepends=True)
        if not names_blocked_user(line, blocked_users)
    )
    return kept if kept.strip() else None


def scrub_memory(memory: dict | None, blocked_users) -> dict | None:
    """A prompt copy of the web's memory bundle without opted-out members.

    Matching the bot: in the blob (content, behavior, personality) every line
    naming a blocked user is dropped, and a field left empty becomes None; a
    note whose content or channel name names one is dropped whole. Stored
    data is never changed.
    """
    if not memory or blocked_users is None:
        return memory
    scrubbed = dict(memory)
    for key in _MEMORY_TEXT_FIELDS:
        if isinstance(scrubbed.get(key), str):
            scrubbed[key] = drop_blocked_lines(scrubbed[key], blocked_users)
    notes = scrubbed.get("notes")
    if notes:
        scrubbed["notes"] = [
            note
            for note in notes
            if not any(
                isinstance(note.get(key), str)
                and names_blocked_user(note[key], blocked_users)
                for key in ("content", "channel_name")
            )
        ]
    return scrubbed
