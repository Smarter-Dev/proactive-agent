"""What starts a wake brief (#278): whole once, then only what changed.

The agent keeps every brief back to its last compaction, so anything resent
unchanged only piles up copies. The first wake of a fresh history, and the
first after each compaction, get the whole head: YOUR CHANNELS (every
channel, forum and post with its id and the watch instructions), the
RECIPES list and YOUR MEMORY, as they are at that wake. Every other wake
gets only the changes: CHANNEL CHANGES (a channel, forum or post the agent
was not told about, a channel or forum switched off), RECIPE LIST CHANGES
(a recipe added, edited or gone), and YOUR MEMORY again only when it
changed. The runner picks the whole head when the history it is about to
send, after any compaction, holds no YOUR CHANNELS block.

What the agent was told is read back from its history: the head of each
brief (the blocks before its notifications, recipe notices first and
memory last), from the newest whole head on. A member's message comes
later in a brief and is never read. A line proves nothing about scope
either way: the runtime checks every id the agent acts on.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart

from proactive_agent.environment import InstructionStore
from proactive_agent.recipes import NOTICES_HEADING, RECIPES_HEADING, listed_recipes

NOTIFICATIONS_HEADING = "NOTIFICATIONS since your last wake"
CHANNEL_LIST_HEADING = "YOUR CHANNELS"
CHANNEL_CHANGES_HEADING = "CHANNEL CHANGES"
RECIPE_CHANGES_HEADING = "RECIPE LIST CHANGES"
MEMORY_HEADING = "YOUR MEMORY"
CHANNEL_LIST_INTRO = (
    f"{CHANNEL_LIST_HEADING}: every channel you may read or act in, with your "
    "watch instructions. You get this list whole after a fresh start or a "
    "compaction; in between, CHANNEL CHANGES tells you what changed, and "
    "list_channels shows the current list any time."
)
CHANNEL_CHANGES_INTRO = f"{CHANNEL_CHANGES_HEADING} since you were last told:"
RECIPE_CHANGES_INTRO = f"{RECIPE_CHANGES_HEADING} since you were last told:"
MEMORY_CHANGED_INTRO = f"{MEMORY_HEADING}, updated (it replaces your earlier copy):"
MEMORY_EMPTY = f"{MEMORY_HEADING}, updated: it is empty now."
# Every line naming a channel ends with its id; the kind starts it.
CHANNEL_LINE = re.compile(
    r"^- (channel|forum|post:|new channel|new forum|new post:|switched off:) "
    r".*\(channel_id=(\d+)\)$"
)
POST_KINDS = frozenset({"post:", "new post:"})
ROW_NAME = re.compile(
    r"^- (?:new )?(channel|forum) (#.*?)(?:, whose posts| \(channel_id=)"
)
RECIPE_CHANGE = re.compile(r"^- (new|edited|gone \(switched off or deleted\)): (.*)$")
MORE_RECIPES = "- …and "
_HEADINGS = (
    NOTICES_HEADING,
    CHANNEL_LIST_HEADING,
    CHANNEL_CHANGES_HEADING,
    RECIPES_HEADING,
    RECIPE_CHANGES_HEADING,
)


@dataclass
class Told:
    """What the agent's history has told it, from the newest whole head."""

    # A whole head is in the history.
    whole: bool = False
    # Channel and forum id -> (kind, "#name").
    rows: dict[str, tuple[str, str]] = field(default_factory=dict)
    posts: set[str] = field(default_factory=set)
    # Each listed recipe as "name: preview".
    recipes: set[str] = field(default_factory=set)
    # YOUR MEMORY's text below its heading.
    memory: str = ""


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def instruction_lines(store: InstructionStore | None) -> list[str]:
    if store is None or not store.entries:
        return ["  - no watch instructions"]
    return [
        f"  - {entry.instruction_id} (expires {entry.expires_at:%H:%M} UTC): "
        f"{_one_line(entry.text)}"
        for entry in store.entries
    ]


def channel_entry(
    kind: str, channel_id: str, label: str, store: InstructionStore | None = None
) -> list[str]:
    """A channel's lines: ``channel``, ``forum`` or ``post``, optionally
    with ``new `` in front. A post's instructions are its forum's, listed
    there; a new post repeats them."""
    label = _one_line(label)
    if kind.endswith("forum"):
        line = (
            f"- {kind} #{label}, whose posts are each a channel of yours, "
            f"sharing these watch instructions (channel_id={channel_id})"
        )
    elif kind.endswith("post"):
        line = f"- {kind}: {label} (channel_id={channel_id})"
    else:
        line = f"- {kind} #{label} (channel_id={channel_id})"
    if kind == "post":
        return [line]
    return [line, *instruction_lines(store)]


@dataclass
class Channels:
    """The channels of this wake: the enabled rows' text channels, the
    enabled forums, the posts admitted so far, and the watch instructions."""

    channels: dict[str, str]
    forums: dict[str, dict]
    posts: dict[str, str]
    stores: dict[str, InstructionStore]

    def lines(self) -> list[str]:
        lines: list[str] = []
        for channel_id, name in self.channels.items():
            lines += channel_entry(
                "channel", channel_id, name, self.stores.get(channel_id)
            )
        for forum_id, forum in self.forums.items():
            name = forum.get("name") or forum_id
            lines += channel_entry("forum", forum_id, name, self.stores.get(forum_id))
        for post_id, label in self.posts.items():
            lines += channel_entry("post", post_id, label)
        return lines or ["- none"]

    def whole(self) -> str:
        """The YOUR CHANNELS block."""
        return "\n".join([CHANNEL_LIST_INTRO, *self.lines()])

    def changes(self, told: Told) -> str:
        """The CHANNEL CHANGES block; empty when nothing changed."""
        lines: list[str] = []
        for channel_id, name in self.channels.items():
            if channel_id not in told.rows:
                lines += channel_entry(
                    "new channel", channel_id, name, self.stores.get(channel_id)
                )
        for forum_id, forum in self.forums.items():
            if forum_id not in told.rows:
                name = forum.get("name") or forum_id
                lines += channel_entry(
                    "new forum", forum_id, name, self.stores.get(forum_id)
                )
        for post_id, label in self.posts.items():
            if post_id not in told.posts:
                lines += channel_entry(
                    "new post", post_id, label, self.stores.get(post_id)
                )
        for channel_id, (kind, name) in told.rows.items():
            if channel_id in self.channels or channel_id in self.forums:
                continue
            its_posts = " or its posts" if kind == "forum" else ""
            lines.append(
                f"- switched off: the {kind} {name}; you no longer watch it"
                f"{its_posts} or act there (channel_id={channel_id})"
            )
        if not lines:
            return ""
        return "\n".join([CHANNEL_CHANGES_INTRO, *lines])


def recipe_changes(
    listing: dict | None, told: Told, scrub: Callable[[str], str] = str
) -> str:
    """The RECIPE LIST CHANGES block: each listed recipe the agent was not
    told about as it is now, and each it was told about that is no longer
    enabled. Empty when nothing changed or the list is unknown."""
    if listing is None:
        return ""
    listed, _more = listed_recipes(listing)
    current = {scrub(item): name for name, item in listed}
    names = [
        " ".join(str(recipe.get("name") or "").split())
        for recipe in listing.get("recipes") or ()
    ]
    lines: list[str] = []
    for item, name in current.items():
        if item in told.recipes:
            continue
        edited = any(old.startswith(f"{name}: ") for old in told.recipes)
        lines.append(f"- {'edited' if edited else 'new'}: {item}")
    for old in sorted(told.recipes):
        if old in current or any(old.startswith(f"{name}: ") for name in names):
            continue
        lines.append(f"- gone (switched off or deleted): {old}")
    if not lines:
        return ""
    return "\n".join([RECIPE_CHANGES_INTRO, *lines])


def memory_body(block: str) -> str:
    """YOUR MEMORY's text below its heading line."""
    return block.split("\n", 1)[1] if "\n" in block else ""


def memory_changes(memory_block: str, told: Told) -> str:
    """YOUR MEMORY again when it differs from the agent's copy."""
    body = memory_body(memory_block)
    if body == told.memory:
        return ""
    return f"{MEMORY_CHANGED_INTRO}\n{body}" if body else MEMORY_EMPTY


def _texts(messages: Iterable[ModelMessage]) -> Iterable[str]:
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart):
                continue
            contents = [part.content] if isinstance(part.content, str) else part.content
            yield from (content for content in contents if isinstance(content, str))


def head_blocks(text: str) -> list[str]:
    """The blocks that open a brief, before its notifications. YOUR MEMORY,
    which may hold blank lines, is the last and runs to the notifications."""
    head = text.split(f"\n\n{NOTIFICATIONS_HEADING}", 1)[0]
    blocks: list[str] = []
    parts = head.split("\n\n")
    for index, block in enumerate(parts):
        if block.startswith(MEMORY_HEADING):
            blocks.append("\n\n".join(parts[index:]))
            break
        if not block.startswith(_HEADINGS):
            break
        blocks.append(block)
    return blocks


def told(messages: Iterable[ModelMessage]) -> Told:
    """What the history told the agent, from its newest whole head on."""
    state = Told()
    for text in _texts(messages):
        for block in head_blocks(text):
            lines = block.split("\n")
            if block.startswith(CHANNEL_LIST_HEADING):
                state = Told(whole=True)
            if block.startswith((CHANNEL_LIST_HEADING, CHANNEL_CHANGES_HEADING)):
                for line in lines[1:]:
                    _tell_channel(state, line)
            elif block.startswith(RECIPES_HEADING):
                state.recipes = {
                    line[2:]
                    for line in lines[1:]
                    if line.startswith("- ") and not line.startswith(MORE_RECIPES)
                }
            elif block.startswith(RECIPE_CHANGES_HEADING):
                for line in lines[1:]:
                    if match := RECIPE_CHANGE.match(line):
                        if match[1] in ("new", "edited"):
                            state.recipes.add(match[2])
                        else:
                            state.recipes.discard(match[2])
            elif block.startswith(MEMORY_HEADING):
                state.memory = memory_body(block)
    return state


def _tell_channel(state: Told, line: str) -> None:
    match = CHANNEL_LINE.match(line)
    if match is None:
        return
    kind, channel_id = match[1], match[2]
    if kind == "switched off:":
        state.rows.pop(channel_id, None)
    elif kind in POST_KINDS:
        state.posts.add(channel_id)
    elif named := ROW_NAME.match(line):
        state.rows[channel_id] = (named[1], named[2])


def whole_head_sent(messages: Iterable[ModelMessage]) -> bool:
    """Whether these messages hold a whole head (a YOUR CHANNELS block)."""
    return told(messages).whole
