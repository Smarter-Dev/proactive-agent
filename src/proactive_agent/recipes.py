"""Recipes (#267): admin-written instructions the agent opens on demand.

The agent's brief lists each enabled recipe by name and preview; the free
``open_recipe`` tool returns one whole, headed by a line naming its id and
revision::

    [RECIPE id=<uuid> revision=<n>] <name>
    <instructions>

The history is the record of what the agent has open. Every replica, every
restart and every purge sees the same one, so nothing else is kept: at each
wake the runtime reads the newest state per recipe out of the history and
compares it with the current list. A recipe edited since (a higher revision)
or no longer listed (switched off or deleted) gets a notice in the brief,
and the notice carries its own marker, so it is said once per change.

Only text the code wrote moves that state, never the admin's instructions,
a member's message or the model's own words (#276):

- an ``open_recipe`` result, by its first line;
- a copy kept in a compaction memory note, read from the section the code
  appends to the note, each copy framed by its length rather than found by
  searching for a delimiter the admin could have typed;
- a notice line, read only from the block that starts a brief, and only for
  a recipe one of the two above already opened.

Compaction is all or nothing for each recipe the agent opened (Zech, #267):
its note either carries the recipe word for word or names it only. The
model never copies a recipe itself; it writes ``KEEP RECIPE: <name>`` and
:func:`apply_recipe_keeps` attaches the text exactly as it was opened,
header included, so the edit notice still works after the fold. A recipe
switched off, deleted or edited since its copy was opened is named only. A
privacy purge keeps no recipe text at all: a recipe can name the person
being forgotten, and cutting them out of a copy would be a partial copy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

OPEN_RECIPE_TOOL = "open_recipe"
# The first line of every memory note (agent.memory_note_pair).
MEMORY_NOTE_PREFIX = "[COMPACTION MEMORY NOTE"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
HEADER = re.compile(rf"^\[RECIPE id=({_UUID}) revision=(\d+)\] (.*)$", re.MULTILINE)
EDITED_MARKER = re.compile(rf"\[RECIPE EDITED id=({_UUID}) revision=(\d+)\]")
GONE_MARKER = re.compile(rf"\[RECIPE GONE id=({_UUID})\]")
NOTICES_HEADING = "RECIPE CHANGES SINCE YOU OPENED THEM:"
NOTICE_LINE = re.compile(
    rf"^- .* \[RECIPE (?:EDITED id=({_UUID}) revision=(\d+)|GONE id=({_UUID}))\]$"
)
# A model writing several keeps writes a list: bullets, numbers, bold and
# quotes around the name are all the same line.
KEEP_LINE = re.compile(
    r"^[ \t]*(?:(?:[-*•]|\d+[.)])[ \t]+)?(?:\*\*|__)?KEEP RECIPE:(?:\*\*|__)?"
    r"[ \t]*(.*?)[ \t]*$",
    re.MULTILINE,
)
_NAME_WRAPPING = "*_\"'`“”‘’ \t"
KEPT_HEADING = "RECIPES KEPT WORD FOR WORD (attached exactly as you opened them):"
KEPT_START = "----- RECIPE {chars} chars -----"
KEPT_START_LINE = re.compile(r"----- RECIPE (\d+) chars -----\n")
KEPT_END = "----- END RECIPE -----"
_FRAMING_LINE = re.compile(r"^----- (?:END )?RECIPE(?: \d+ chars)? -----$")
# Kept text across all recipes in one note, newest opened first; beyond it a
# recipe is named only. 20 kept recipes at the 20,000-character limit would
# put the note alone over the history's compaction limit.
KEPT_CHARS_MAX = 40_000
# The brief's list, which every stored brief repeats.
LISTED_RECIPES_MAX = 50
LISTED_CHARS_MAX = 15_000
# A line of the model's note equal to a line of a recipe it opened, this long
# or longer, is a partial copy and is dropped.
COPIED_LINE_MIN_CHARS = 40
_LIST_MARKER = re.compile(r"^(?:[-*•]|\d+[.)])\s+")
# Where an opened recipe stands in the agent's context: its revision, or
# GONE once the agent was told it no longer exists.
GONE = -1


def name_key(name: str) -> str:
    """How names are matched, as the application matches them."""
    return " ".join(name.split()).casefold()


def format_opened_recipe(recipe: dict) -> str:
    """What open_recipe returns: the header line, then the instructions."""
    name = " ".join(str(recipe.get("name") or "").split())
    return (
        f"[RECIPE id={recipe['id']} revision={int(recipe['revision'])}] {name}\n"
        f"{recipe.get('instructions') or ''}"
    )


def render_recipe_block(listing: dict | None) -> str:
    """The brief's list of enabled recipes; empty when there are none."""
    recipes = list((listing or {}).get("recipes") or ())
    if not recipes:
        return ""
    lines: list[str] = []
    used = 0
    for recipe in recipes[:LISTED_RECIPES_MAX]:
        line = (
            f"- {' '.join(str(recipe.get('name') or '').split())}: "
            f"{recipe.get('preview') or ''}"
        )
        if lines and used + len(line) + 1 > LISTED_CHARS_MAX:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) < len(recipes):
        lines.append(
            f"- …and {len(recipes) - len(lines)} more; ask by name with open_recipe."
        )
    listed = "\n".join(lines)
    return (
        "RECIPES (instructions this server's admins wrote for you; "
        'open_recipe("<name>") reads one whole and is free):\n'
        f"{listed}\n"
        "A preview only says when a recipe applies. Open the recipe before "
        "you follow it."
    )


@dataclass(frozen=True)
class _Copy:
    recipe_id: str
    revision: int
    name: str
    text: str  # the header line and the instructions, as opened


def _copy(text: str) -> _Copy | None:
    match = HEADER.match(text)
    if match is None:
        return None
    return _Copy(match[1].lower(), int(match[2]), match[3], text)


def kept_copies(note: str) -> list[_Copy]:
    """The copies the code attached to a memory note, in order.

    The section starts at the first line that is the kept heading (the
    model's own lines like it were dropped), and each copy is exactly as
    long as its start line says, followed by the end line. Anything else
    ends the section there: the copies read so far stand, none is cut short.
    """
    heading = f"\n{KEPT_HEADING}\n"
    start = note.find(heading)
    if start < 0:
        return []
    position = start + len(heading)
    copies: list[_Copy] = []
    while match := KEPT_START_LINE.match(note, position):
        body_start = match.end()
        body_end = body_start + int(match[1])
        end_line = f"\n{KEPT_END}"
        if note[body_end : body_end + len(end_line)] != end_line:
            break
        after = body_end + len(end_line)
        if after < len(note) and note[after] != "\n":
            break
        copy = _copy(note[body_start:body_end])
        if copy is None:
            break
        copies.append(copy)
        position = after + 1
    return copies


def _notice_markers(brief: str) -> list[tuple[str, int]]:
    """(recipe id, revision or GONE) for each notice the code wrote at the
    head of a brief."""
    if not brief.startswith(f"{NOTICES_HEADING}\n"):
        return []
    markers = []
    for line in brief.split("\n")[1:]:
        match = NOTICE_LINE.match(line)
        if match is None:
            break
        if match[1]:
            markers.append((match[1].lower(), int(match[2])))
        else:
            markers.append((match[3].lower(), GONE))
    return markers


def _prompt_texts(part: UserPromptPart) -> Iterable[str]:
    contents = [part.content] if isinstance(part.content, str) else part.content
    return (content for content in contents if isinstance(content, str))


def _events(messages: Iterable[ModelMessage]) -> Iterable[tuple[str, object]]:
    """("copy", _Copy) and ("marker", (id, revision)) in history order."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                if part.tool_name == OPEN_RECIPE_TOOL and isinstance(part.content, str):
                    copy = _copy(part.content)
                    if copy is not None:
                        yield "copy", copy
            elif isinstance(part, UserPromptPart):
                for text in _prompt_texts(part):
                    if text.startswith(MEMORY_NOTE_PREFIX):
                        for copy in kept_copies(text):
                            yield "copy", copy
                    else:
                        for marker in _notice_markers(text):
                            yield "marker", marker


def opened_recipes(messages: Iterable[ModelMessage]) -> dict[str, tuple[int, str]]:
    """recipe id -> (revision, name) for each recipe open in this context.

    An open_recipe result and a copy a memory note kept open a recipe at
    their revision; a later notice moves one already open to the revision
    it announced, or to GONE. The last one in the history wins.
    """
    state: dict[str, tuple[int, str]] = {}
    for kind, event in _events(messages):
        if kind == "copy":
            state[event.recipe_id] = (event.revision, event.name)
        else:
            recipe_id, revision = event
            if recipe_id in state:
                state[recipe_id] = (revision, state[recipe_id][1])
    return state


def recipe_notices(opened: dict[str, tuple[int, str]], listing: dict | None) -> str:
    """The brief's notice of recipes changed since the agent opened them.

    It goes first in the brief: only there are its markers read back.
    """
    if not opened or listing is None:
        return ""
    current = {
        str(recipe.get("id")).lower(): recipe for recipe in listing.get("recipes") or ()
    }
    lines = []
    for recipe_id, (revision, name) in opened.items():
        recipe = current.get(recipe_id)
        if recipe is None:
            if revision != GONE:
                lines.append(
                    f'- "{name}" is no longer available: an admin switched it off '
                    "or deleted it. Do not follow your copy of it. "
                    f"[RECIPE GONE id={recipe_id}]"
                )
            continue
        new_revision = int(recipe.get("revision") or 0)
        if revision != GONE and new_revision > revision:
            current_name = " ".join(str(recipe.get("name") or name).split())
            lines.append(
                f'- "{current_name}" was edited by an admin since you opened it. '
                f'Your copy is out of date: open_recipe("{current_name}") again '
                "before you follow it. "
                f"[RECIPE EDITED id={recipe_id} revision={new_revision}]"
            )
    if not lines:
        return ""
    return f"{NOTICES_HEADING}\n" + "\n".join(lines)


def _comparable(line: str) -> str:
    return " ".join(_LIST_MARKER.sub("", line.strip()).split()).casefold()


def _is_written_by_code(line: str) -> bool:
    return bool(
        KEEP_LINE.match(line)
        or HEADER.match(line)
        or EDITED_MARKER.search(line)
        or GONE_MARKER.search(line)
        or _FRAMING_LINE.match(line.strip())
        or line.strip() == KEPT_HEADING
    )


def apply_recipe_keeps(
    summary: str, messages: list[ModelMessage], *, allow_keep: bool = True
) -> str:
    """The compaction note with every recipe all or nothing.

    Each ``KEEP RECIPE: <name>`` line is taken out. With ``allow_keep``, a
    recipe the folded history opened (or a note before it kept) is attached
    whole, exactly as it was opened, when that copy is still current: not
    edited, switched off or deleted since. Copies are attached newest opened
    first up to KEPT_CHARS_MAX. Any other name, and every name without
    ``allow_keep`` (a privacy purge), is left as its name only. Header,
    notice and framing lines the model wrote itself are dropped, and so is
    any line it copied from a recipe, so no recipe text reaches the note
    except the attached copies.
    """
    keep_names = [
        name
        for match in KEEP_LINE.finditer(summary)
        if (name := match[1].strip(_NAME_WRAPPING))
    ]
    copies: dict[str, tuple[int, _Copy]] = {}
    copied_lines: set[str] = set()
    for order, (kind, event) in enumerate(_events(messages)):
        if kind != "copy":
            continue
        copies[name_key(event.name)] = (order, event)
        for line in event.text.split("\n")[1:]:
            comparable = _comparable(line)
            if len(comparable) >= COPIED_LINE_MIN_CHARS:
                copied_lines.add(comparable)
    cleaned_lines = [
        line
        for line in summary.split("\n")
        if not (_is_written_by_code(line) or _comparable(line) in copied_lines)
    ]
    note = "\n".join(cleaned_lines).rstrip()

    state = opened_recipes(messages)
    current: list[tuple[int, _Copy]] = []
    name_only: list[str] = []
    seen: set[str] = set()
    for name in keep_names:
        key = name_key(name)
        if key in seen:
            continue
        seen.add(key)
        found = copies.get(key) if allow_keep else None
        if found is not None and state.get(found[1].recipe_id, (None,))[0] == (
            found[1].revision
        ):
            current.append(found)
        else:
            name_only.append(" ".join(name.split()))
    attached: set[str] = set()
    too_long: list[str] = []
    used = 0
    for _order, copy in sorted(current, key=lambda found: found[0], reverse=True):
        if used + len(copy.text) <= KEPT_CHARS_MAX:
            attached.add(copy.text)
            used += len(copy.text)
        else:
            too_long.append(copy.name)
    if name_only:
        note += "\n\nRecipes you can open again with open_recipe: " + ", ".join(
            f'"{name}"' for name in name_only
        )
    if too_long:
        note += (
            "\n\nRecipes too long to keep word for word with the others: "
            + ", ".join(f'"{name}"' for name in reversed(too_long))
            + ". Open them again with open_recipe if you need them."
        )
    if attached:
        note += "\n\n" + KEPT_HEADING
        for _order, copy in current:
            if copy.text in attached:
                note += (
                    f"\n{KEPT_START.format(chars=len(copy.text))}\n"
                    f"{copy.text}\n{KEPT_END}"
                )
    return note
