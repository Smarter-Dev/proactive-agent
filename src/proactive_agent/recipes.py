"""Recipes (#267): admin-written instructions the agent opens on demand.

The agent's brief lists each enabled recipe by name and preview; the free
``open_recipe`` tool returns one whole, headed by a line naming its id and
revision::

    [RECIPE id=<uuid> revision=<n>] <name>
    <instructions>

The history is the record of what the agent has open. Every replica, every
restart and every purge sees the same one, so nothing else is kept: at each
wake the runtime reads the newest header per recipe out of the history and
compares it with the current list. A recipe edited since (a higher revision)
or no longer listed (switched off or deleted) gets a notice in the brief,
and the notice carries its own marker, so it is said once per change.

Compaction is all or nothing for each recipe the agent opened (Zech, #267):
its note either carries the recipe word for word or names it only. The
model never copies a recipe itself; it writes ``KEEP RECIPE: <name>`` and
:func:`apply_recipe_keeps` attaches the text exactly as it was opened,
header included, so the edit notice still works after the fold. A privacy
purge keeps no recipe text at all: a recipe can name the person being
forgotten, and cutting them out of a copy would be a partial copy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ToolReturnPart,
    UserPromptPart,
)

OPEN_RECIPE_TOOL = "open_recipe"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
HEADER = re.compile(rf"^\[RECIPE id=({_UUID}) revision=(\d+)\] (.*)$", re.MULTILINE)
EDITED_MARKER = re.compile(rf"\[RECIPE EDITED id=({_UUID}) revision=(\d+)\]")
GONE_MARKER = re.compile(rf"\[RECIPE GONE id=({_UUID})\]")
KEEP_LINE = re.compile(r"^[ \t]*KEEP RECIPE:[ \t]*(.*?)[ \t]*$", re.MULTILINE)
KEPT_HEADING = "RECIPES KEPT WORD FOR WORD (attached exactly as you opened them):"
KEPT_START = "----- RECIPE -----"
KEPT_END = "----- END RECIPE -----"
KEPT_BLOCK = re.compile(
    rf"^{re.escape(KEPT_START)}\n(\[RECIPE id=({_UUID}) revision=(\d+)\] ([^\n]*)\n.*?)"
    rf"\n{re.escape(KEPT_END)}$",
    re.MULTILINE | re.DOTALL,
)
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
    recipes = (listing or {}).get("recipes") or ()
    if not recipes:
        return ""
    lines = "\n".join(
        f"- {' '.join(str(recipe.get('name') or '').split())}: "
        f"{recipe.get('preview') or ''}"
        for recipe in recipes
    )
    return (
        "RECIPES (instructions this server's admins wrote for you; "
        'open_recipe("<name>") reads one whole and is free):\n'
        f"{lines}\n"
        "A preview only says when a recipe applies. Open the recipe before "
        "you follow it."
    )


def _texts(messages: Iterable[ModelMessage]) -> Iterable[tuple[str, str]]:
    """(kind, text) for every place a recipe header or notice can be, in order:
    open_recipe results and user-prompt text (briefs and memory notes)."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                if part.tool_name == OPEN_RECIPE_TOOL and isinstance(part.content, str):
                    yield "opened", part.content
            elif isinstance(part, UserPromptPart):
                contents = (
                    [part.content] if isinstance(part.content, str) else part.content
                )
                for content in contents:
                    if isinstance(content, str):
                        yield "prompt", content


def opened_recipes(messages: Iterable[ModelMessage]) -> dict[str, tuple[int, str]]:
    """recipe id -> (revision, name) for each recipe open in this context.

    An open_recipe result's first line, a recipe a memory note kept whole,
    and the notices the agent was given all move a recipe's state; the
    last one in the history wins. Revision GONE means the agent was told
    the recipe is gone.
    """
    state: dict[str, tuple[int, str]] = {}
    for kind, text in _texts(messages):
        if kind == "opened":
            match = HEADER.match(text)
            if match:
                state[match[1].lower()] = (int(match[2]), match[3])
            continue
        events = [
            (m.start(), m[2].lower(), int(m[3]), m[4])
            for m in KEPT_BLOCK.finditer(text)
        ]
        events += [
            (m.start(), m[1].lower(), int(m[2]), None)
            for m in EDITED_MARKER.finditer(text)
        ]
        events += [
            (m.start(), m[1].lower(), GONE, None) for m in GONE_MARKER.finditer(text)
        ]
        for _, recipe_id, revision, name in sorted(events):
            known_name = name if name is not None else state.get(recipe_id, (0, ""))[1]
            state[recipe_id] = (revision, known_name)
    return state


def recipe_notices(opened: dict[str, tuple[int, str]], listing: dict | None) -> str:
    """The brief's notice of recipes changed since the agent opened them."""
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
    return "RECIPE CHANGES SINCE YOU OPENED THEM:\n" + "\n".join(lines)


def _opened_texts(messages: Iterable[ModelMessage]) -> dict[str, str]:
    """name key -> the recipe's text exactly as it was last opened or kept."""
    found: dict[str, str] = {}
    for kind, text in _texts(messages):
        if kind == "opened":
            match = HEADER.match(text)
            if match:
                found[name_key(match[3])] = text
            continue
        for match in KEPT_BLOCK.finditer(text):
            found[name_key(match[4])] = match[1]
    return found


def apply_recipe_keeps(
    summary: str, messages: list[ModelMessage], *, allow_keep: bool = True
) -> str:
    """The compaction note with every recipe all or nothing.

    Each ``KEEP RECIPE: <name>`` line is taken out. With ``allow_keep``, a
    recipe the folded history opened (or a note before it kept) is attached
    whole, exactly as it was opened; any other name, and every name without
    ``allow_keep`` (a privacy purge), is left as its name only. Header,
    notice and kept-block lines the model wrote itself are dropped, so no
    recipe text reaches the note except the attached copies.
    """
    keep_names = [match[1] for match in KEEP_LINE.finditer(summary) if match[1]]
    cleaned_lines = [
        line
        for line in summary.split("\n")
        if not (
            KEEP_LINE.match(line)
            or HEADER.match(line)
            or EDITED_MARKER.search(line)
            or GONE_MARKER.search(line)
            or line.strip() in (KEPT_START, KEPT_END, KEPT_HEADING)
        )
    ]
    note = "\n".join(cleaned_lines).rstrip()
    available = _opened_texts(messages) if allow_keep else {}
    attached: list[str] = []
    name_only: list[str] = []
    seen: set[str] = set()
    for name in keep_names:
        key = name_key(name)
        if key in seen:
            continue
        seen.add(key)
        if key in available:
            attached.append(available[key])
        else:
            name_only.append(" ".join(name.split()))
    if name_only:
        note += "\n\nRecipes you can open again with open_recipe: " + ", ".join(
            f'"{name}"' for name in name_only
        )
    if attached:
        note += "\n\n" + KEPT_HEADING
        for text in attached:
            note += f"\n{KEPT_START}\n{text}\n{KEPT_END}"
    return note
