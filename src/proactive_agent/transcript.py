"""Transcript rendering shared by the eval labeler and the proactive bot.

Stable single-letter speaker tags plus real display names, message ids,
reply markers and a [BOT] prefix for bot-authored lines.
"""

from __future__ import annotations

# The whole model-visible form of a message from a blocked user.
BLOCKED_LINE = "[BLOCKED BY USER]"


def speaker_tags(records: list[dict]) -> dict[str, str]:
    """Stable per-author letter tags (A, B, … AA, AB) by first appearance."""
    tags: dict[str, str] = {}
    for record in records:
        if record.get("blocked"):
            continue
        author_id = record["author_id"]
        if author_id not in tags:
            tags[author_id] = _letter_tag(len(tags))
    return tags


def _letter_tag(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def render_transcript_line(record: dict, tags: dict[str, str]) -> str:
    """The one place a channel message becomes a model-visible line.

    Both the author-id form and the blocked-user form live here, so the
    opt-out work (#74) changes them in one function.
    """
    if record.get("blocked"):
        return BLOCKED_LINE
    bot_marker = "[BOT] " if record["is_bot"] else ""
    reply_marker = (
        f" (reply to id={record['reply_to_id']})" if record["reply_to_id"] else ""
    )
    tag = tags[record["author_id"]]
    return (
        f"[id={record['id']}] {bot_marker}{tag}·{record['author_display']} "
        f"(uid={record['author_id']}){reply_marker}: {record['content']}"
    )
