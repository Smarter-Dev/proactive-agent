"""Discord message splitting kept byte-for-byte compatible with the bot."""

DISCORD_MESSAGE_LIMIT = 2000
SPLIT_TARGET = 1500
SUMMARIZE_THRESHOLD = 3000


def split_for_discord(text: str) -> list[str]:
    stripped = text.strip()
    if len(stripped) <= DISCORD_MESSAGE_LIMIT:
        return [stripped] if stripped else []
    earliest = max(0, len(stripped) - DISCORD_MESSAGE_LIMIT)
    split_at = stripped.rfind("\n", earliest, SPLIT_TARGET + 1)
    if split_at <= 0:
        split_at = stripped.rfind(" ", earliest, SPLIT_TARGET + 1)
    if split_at <= 0:
        split_at = SPLIT_TARGET
    head = stripped[:split_at].rstrip()
    tail = stripped[split_at:].strip()
    if len(tail) > DISCORD_MESSAGE_LIMIT:
        tail = tail[: DISCORD_MESSAGE_LIMIT - 1] + "…"
    return [part for part in (head, tail) if part]
