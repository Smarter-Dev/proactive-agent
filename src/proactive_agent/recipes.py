"""Recipes (#267): admin-written instructions the agent opens on demand.

The agent's brief lists each enabled recipe by name and preview; the free
``open_recipe`` tool returns one whole, headed by a line naming its id and
revision::

    [RECIPE id=<uuid> revision=<n>] <name>
    <instructions>

A compaction note copies a recipe it still needs verbatim, header line
included, or only mentions its name (the rule is in ``COMPACTION_PROMPT``).
So the history is the record of what the agent has open: at each wake the
runtime reads the newest header per recipe, from an open_recipe result or a
memory note, and compares it with the current list. A recipe edited since
(a higher revision) or no longer listed (switched off or deleted) gets a
notice that starts the brief, and the notice carries its own marker, so it
is said once per change.
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

from proactive_agent.agent import MEMORY_NOTE_PREFIX

OPEN_RECIPE_TOOL = "open_recipe"
_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
HEADER = re.compile(rf"^\[RECIPE id=({_UUID}) revision=(\d+)\] (.*)$", re.MULTILINE)
NOTICES_HEADING = "RECIPE CHANGES SINCE YOU OPENED THEM:"
NOTICE_LINE = re.compile(
    rf"^- .* \[RECIPE (?:EDITED id=({_UUID}) revision=(\d+)|GONE id=({_UUID}))\]$"
)
# The brief's list, which every stored brief repeats.
LISTED_RECIPES_MAX = 50
LISTED_CHARS_MAX = 15_000
# Where an opened recipe stands in the agent's context: its revision, or
# GONE once the agent was told it no longer exists.
GONE = -1


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


def _prompt_texts(part: UserPromptPart) -> Iterable[str]:
    contents = [part.content] if isinstance(part.content, str) else part.content
    return (content for content in contents if isinstance(content, str))


def opened_recipes(messages: Iterable[ModelMessage]) -> dict[str, tuple[int, str]]:
    """recipe id -> (revision, name) for each recipe open in this context.

    A header opens a recipe at its revision: the first line of an
    open_recipe result, or a line of a memory note that copied one. A
    notice the runtime put at the head of a brief moves a recipe already
    open to the revision it announced, or to GONE. The last one wins.
    Headers anywhere else (a member's message, say) count for nothing.
    """
    state: dict[str, tuple[int, str]] = {}
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                if part.tool_name == OPEN_RECIPE_TOOL and isinstance(part.content, str):
                    if match := HEADER.match(part.content):
                        state[match[1].lower()] = (int(match[2]), match[3])
                continue
            if not isinstance(part, UserPromptPart):
                continue
            for text in _prompt_texts(part):
                if text.startswith(MEMORY_NOTE_PREFIX):
                    for match in HEADER.finditer(text):
                        state[match[1].lower()] = (int(match[2]), match[3])
                elif text.startswith(f"{NOTICES_HEADING}\n"):
                    for line in text.split("\n")[1:]:
                        match = NOTICE_LINE.match(line)
                        if match is None:
                            break
                        recipe_id = (match[1] or match[3]).lower()
                        if recipe_id in state:
                            revision = int(match[2]) if match[1] else GONE
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
