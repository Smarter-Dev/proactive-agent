"""Recipes (#267): the brief's list, open_recipe, edit notices, compaction.

Admins write recipes on smarter-dev; the agent sees each enabled one's name
and preview and opens one whole with the free open_recipe tool. An admin's
edit to a recipe the agent has open reaches it as a notice at its next wake,
and every compaction keeps a recipe word for word or by name only.
"""

from __future__ import annotations

import time
from uuid import uuid4

import fakeredis.aioredis
import httpx
from purge_fakes import CHANNEL, GUILD, TARGET, FakeAPI, batch, make_runtime
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from proactive_agent.agent import (
    COMPACTION_PROMPT,
    KimiAgentRunner,
    build_privacy_purge_prompt,
    memory_note_pair,
    privacy_compaction_summary,
    self_compaction_summary,
)
from proactive_agent.api import ApplicationAPI
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.parity import ProactiveDeps, build_proactive_agent
from proactive_agent.recipes import (
    KEPT_END,
    KEPT_START,
    apply_recipe_keeps,
    format_opened_recipe,
    opened_recipes,
    recipe_notices,
    render_recipe_block,
)

WELCOME_ID = str(uuid4())
RULES_ID = str(uuid4())
WELCOME_TEXT = (
    "Use this when someone new says hello.\n\n"
    "1. Greet them by name.\n2. Point them at #rules.\n  Keep it to two lines."
)


def welcome(revision: int = 1, text: str = WELCOME_TEXT) -> dict:
    return {
        "id": WELCOME_ID,
        "name": "Welcome",
        "instructions": text,
        "revision": revision,
        "updated_at": "2026-10-20T12:00:00Z",
    }


def preview_of(recipe: dict) -> dict:
    return {
        "id": recipe["id"],
        "name": recipe["name"],
        "preview": recipe["instructions"].split("\n\n")[0],
        "revision": recipe["revision"],
        "updated_at": recipe["updated_at"],
    }


class RecipeAPI(FakeAPI):
    """FakeAPI with the guild's recipes, as smarter-dev serves them."""

    def __init__(self, recipes=()):
        super().__init__()
        self.recipes = {recipe["id"]: dict(recipe) for recipe in recipes}
        self.disabled: set[str] = set()
        self.list_calls = 0
        self.fail_list = False
        self.addenda[GUILD] = {CHANNEL: ""}

    async def list_recipes(self, guild_id):
        self.list_calls += 1
        if self.fail_list:
            raise RuntimeError("application down")
        return {
            "preview_chars": 300,
            "recipes": [
                preview_of(recipe)
                for recipe_id, recipe in self.recipes.items()
                if recipe_id not in self.disabled
            ],
        }

    async def open_recipe(self, guild_id, name):
        for recipe_id, recipe in self.recipes.items():
            if recipe["name"].casefold() == " ".join(name.split()).casefold():
                return None if recipe_id in self.disabled else dict(recipe)
        return None

    def edit(self, recipe_id, text):
        recipe = self.recipes[recipe_id]
        recipe["instructions"] = text
        recipe["revision"] += 1


def opened_history(recipe: dict) -> list[ModelMessage]:
    """One past wake that opened ``recipe``."""
    return [
        ModelRequest(parts=[UserPromptPart("NOTIFICATIONS: someone said hello")]),
        ModelResponse(
            parts=[ToolCallPart("open_recipe", {"name": recipe["name"]}, "r1")]
        ),
        ModelRequest(
            parts=[ToolReturnPart("open_recipe", format_opened_recipe(recipe), "r1")]
        ),
        ModelResponse(parts=[TextPart("greeted them")]),
    ]


# -- The prompt rule -----------------------------------------------------------


def test_the_compaction_prompt_carries_the_all_or_nothing_rule():
    assert "Every recipe opened above, or kept whole in an earlier note, is all or" in (
        COMPACTION_PROMPT
    )
    assert "KEEP RECIPE: <the recipe's exact name>" in COMPACTION_PROMPT
    assert "its full text is attached to your note word for word" in COMPACTION_PROMPT
    assert "leave its instructions out entirely and write only its name" in (
        COMPACTION_PROMPT
    )
    assert (
        "Never copy, paraphrase, summarize or partly quote a recipe's instructions"
        in COMPACTION_PROMPT
    )


def test_the_privacy_compaction_keeps_recipe_names_only():
    prompt = build_privacy_purge_prompt(TARGET, ["kai"])
    assert "Recipes: write no KEEP RECIPE line in this note." in prompt
    assert "Name each recipe you\nopened and nothing more" in prompt
    # It still carries the normal rule it overrides.
    assert COMPACTION_PROMPT in prompt


# -- The brief's list ------------------------------------------------------------


def test_the_block_lists_names_and_previews_only():
    block = render_recipe_block(
        {"preview_chars": 300, "recipes": [preview_of(welcome())]}
    )
    assert "- Welcome: Use this when someone new says hello." in block
    assert 'open_recipe("<name>")' in block
    assert "Greet them by name" not in block


def test_no_recipes_no_block():
    assert render_recipe_block(None) == ""
    assert render_recipe_block({"preview_chars": 300, "recipes": []}) == ""


# -- Edit notices ------------------------------------------------------------------


def test_an_opened_recipe_is_read_back_from_the_history():
    assert opened_recipes(opened_history(welcome(revision=3))) == {
        WELCOME_ID: (3, "Welcome")
    }


def test_an_edit_since_opening_is_noticed_once():
    history = opened_history(welcome(revision=1))
    listing = {"recipes": [preview_of(welcome(revision=2))]}

    notice = recipe_notices(opened_recipes(history), listing)

    assert '"Welcome" was edited by an admin since you opened it' in notice
    assert 'open_recipe("Welcome") again' in notice
    # The notice is part of the next brief; once it is in the history, the
    # same edit is not announced again.
    history.append(ModelRequest(parts=[UserPromptPart(f"brief\n\n{notice}")]))
    assert recipe_notices(opened_recipes(history), listing) == ""
    # A further edit is.
    assert "was edited" in recipe_notices(
        opened_recipes(history), {"recipes": [preview_of(welcome(revision=3))]}
    )


def test_reopening_clears_the_notice():
    history = opened_history(welcome(revision=1)) + opened_history(welcome(revision=2))
    listing = {"recipes": [preview_of(welcome(revision=2))]}
    assert recipe_notices(opened_recipes(history), listing) == ""


def test_a_recipe_switched_off_or_deleted_is_noticed_once():
    history = opened_history(welcome())
    notice = recipe_notices(opened_recipes(history), {"recipes": []})
    assert '"Welcome" is no longer available' in notice
    assert "Do not follow your copy" in notice
    history.append(ModelRequest(parts=[UserPromptPart(notice)]))
    assert recipe_notices(opened_recipes(history), {"recipes": []}) == ""


def test_no_notice_without_a_list():
    """A list call that failed says nothing about what changed."""
    assert recipe_notices(opened_recipes(opened_history(welcome())), None) == ""


# -- Compaction: all or nothing ---------------------------------------------------


def test_a_kept_recipe_is_attached_exactly_as_it_was_opened():
    folded = opened_history(welcome())
    note = apply_recipe_keeps(
        "<1:kai> said hello in #general.\nKEEP RECIPE: welcome", folded
    )

    assert "KEEP RECIPE" not in note
    assert note.startswith("<1:kai> said hello in #general.")
    assert f"{KEPT_START}\n{format_opened_recipe(welcome())}\n{KEPT_END}" in note
    # The kept copy still carries the header, so an edit is still noticed.
    assert opened_recipes(memory_note_pair(note)) == {WELCOME_ID: (1, "Welcome")}


def test_a_recipe_not_kept_is_left_out_entirely():
    folded = opened_history(welcome())
    note = apply_recipe_keeps("Greeted someone using the Welcome recipe.", folded)
    assert "Greet them by name" not in note
    assert "[RECIPE" not in note
    assert opened_recipes(memory_note_pair(note)) == {}


def test_a_copy_the_model_wrote_itself_loses_its_header():
    folded = opened_history(welcome())
    summary = (
        f"[RECIPE id={WELCOME_ID} revision=1] Welcome\n"
        f"----- RECIPE -----\n[RECIPE EDITED id={WELCOME_ID} revision=9]\nnotes"
    )
    note = apply_recipe_keeps(summary, folded)
    assert note == "notes"


def test_keeping_an_unknown_name_leaves_the_name_only():
    note = apply_recipe_keeps("notes\nKEEP RECIPE: Rules", opened_history(welcome()))
    assert note == 'notes\n\nRecipes you can open again with open_recipe: "Rules"'


def test_a_recipe_kept_by_an_earlier_note_can_be_kept_again():
    first = apply_recipe_keeps("first\nKEEP RECIPE: Welcome", opened_history(welcome()))
    second = apply_recipe_keeps("second\nKEEP RECIPE: Welcome", memory_note_pair(first))
    kept = f"{KEPT_START}\n{format_opened_recipe(welcome())}\n{KEPT_END}"
    assert kept in second and second.count(KEPT_START) == 1


def test_a_privacy_rewrite_keeps_no_recipe_text():
    note = apply_recipe_keeps(
        "notes\nKEEP RECIPE: Welcome", opened_history(welcome()), allow_keep=False
    )
    assert "Greet them by name" not in note
    assert note == 'notes\n\nRecipes you can open again with open_recipe: "Welcome"'


def summary_model(output: str, seen: list) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart(output)])

    return FunctionModel(respond)


async def test_self_compaction_attaches_a_kept_recipe_word_for_word():
    seen: list = []
    note, _usage = await self_compaction_summary(
        summary_model("Greeted kai.\nKEEP RECIPE: Welcome", seen),
        opened_history(welcome()),
    )
    assert format_opened_recipe(welcome()) in note
    assert brief_of(seen[0]) == COMPACTION_PROMPT


async def test_privacy_compaction_keeps_the_name_only():
    note = await privacy_compaction_summary(
        summary_model("Greeted someone.\nKEEP RECIPE: Welcome", []),
        opened_history(welcome(text=f"Ping <@{TARGET}> first.\n\nThen greet.")),
        user_id=TARGET,
        names=["kai"],
    )
    assert TARGET not in note.text
    assert note.text == (
        'Greeted someone.\n\nRecipes you can open again with open_recipe: "Welcome"'
    )


# -- The runtime, end to end -----------------------------------------------------------


def agent_model(script, seen) -> FunctionModel:
    """Each wake runs the next scripted step: a tool call, then a final note."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append((list(messages), info))
        step = script.pop(0) if script else None
        if step is None:
            return ModelResponse(parts=[TextPart("done")])
        name, args = step
        return ModelResponse(
            parts=[ToolCallPart(name, args, f"call-{uuid4().hex[:6]}")]
        )

    return FunctionModel(respond)


def real_runtime(redis_client, api, script, seen):
    repository = GuildHistoryRepository(redis_client, api)
    writer = DebouncedHistoryWriter(repository, api)
    runtime = make_runtime(redis_client, api, repository, writer)
    runtime.image_capabilities = type("Caps", (), {"review": None, "generate": None})
    runtime.engine = AgentEngine(
        agent_runner=KimiAgentRunner(
            agent=build_proactive_agent(agent_model(script, seen), system_prompt="sys"),
            summarize=None,
        ),
        skim=SkimRunner(summary_model("skim", [])),
        agent_model_id="fake",
        skim_model_id="fake-skim",
        deps_factory=None,
    )
    return runtime, writer


def brief_of(messages: list[ModelMessage]) -> str:
    request = messages[-1]
    return "\n".join(
        part.content for part in request.parts if isinstance(part, UserPromptPart)
    )


def tool_results(messages: list[ModelMessage], tool_name: str) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == tool_name
    ]


async def test_open_then_admin_edit_then_the_next_wake_says_so():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI([welcome()])
    script = [("open_recipe", {"name": "welcome"}), None, None, None]
    seen: list = []
    runtime, writer = real_runtime(redis_client, api, script, seen)

    # Wake 1: the brief lists the recipe; the agent opens it, for free.
    await runtime.process(batch())
    first_brief = brief_of(seen[0][0])
    assert "- Welcome: Use this when someone new says hello." in first_brief
    assert "Greet them by name" not in first_brief
    (opened,) = tool_results(runtime.engine.agent_runner.history, "open_recipe")
    assert opened == format_opened_recipe(welcome())
    assert runtime.engine.last_deps.budget.used == 0

    # An admin edits it; the next wake is told, by name.
    api.edit(WELCOME_ID, WELCOME_TEXT + "\n3. Wave.")
    await runtime.process(batch())
    second_brief = brief_of(seen[2][0])
    assert "RECIPE CHANGES SINCE YOU OPENED THEM:" in second_brief
    assert '"Welcome" was edited by an admin since you opened it' in second_brief
    assert 'open_recipe("Welcome") again' in second_brief

    # Said once: the wake after has no notice.
    await runtime.process(batch())
    assert "RECIPE CHANGES" not in brief_of(seen[3][0])
    await writer.close()


async def test_open_recipe_refuses_a_switched_off_or_unknown_recipe():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI([welcome()])
    api.disabled.add(WELCOME_ID)
    script = [
        ("open_recipe", {"name": "Welcome"}),
        ("open_recipe", {"name": "Nope"}),
        None,
    ]
    seen: list = []
    runtime, writer = real_runtime(redis_client, api, script, seen)

    await runtime.process(batch())

    assert "RECIPES" not in brief_of(seen[0][0])
    results = tool_results(runtime.engine.agent_runner.history, "open_recipe")
    assert results == [
        'No recipe named "Welcome" is available. Only the recipes listed under '
        "RECIPES in your brief can be opened.",
        'No recipe named "Nope" is available. Only the recipes listed under '
        "RECIPES in your brief can be opened.",
    ]
    await writer.close()


async def test_the_list_refreshes_hourly_and_every_wake_while_a_recipe_is_open():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI([welcome()])
    script = [None, ("open_recipe", {"name": "Welcome"}), None, None]
    runtime, writer = real_runtime(redis_client, api, script, [])

    await runtime.process(batch())
    await runtime.process(batch())  # nothing open yet: the hourly cache serves
    assert api.list_calls == 1
    # The second wake opened Welcome; from now on every wake checks for edits.
    await runtime.process(batch())
    await runtime.process(batch())
    assert api.list_calls == 3
    await writer.close()


async def test_a_failed_list_keeps_the_last_one_and_the_wake_goes_on():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI([welcome()])
    seen: list = []
    runtime, writer = real_runtime(redis_client, api, [None, None], seen)
    await runtime.process(batch())
    runtime.recipes_refreshed_at = time.monotonic() - 2 * 3600
    api.fail_list = True

    await runtime.process(batch())

    assert "- Welcome:" in brief_of(seen[1][0])
    await writer.close()


# -- An application without recipes --------------------------------------------------


def old_application() -> ApplicationAPI:
    """smarter-dev from before recipes: every recipe route is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/proactive-recipes" in request.url.path:
            return httpx.Response(404, json={"detail": "Not Found"})
        raise AssertionError(f"unexpected call {request.url}")

    return ApplicationAPI(
        base_url="https://app.test/api",
        api_key="key",
        transport=httpx.MockTransport(handler),
    )


async def test_an_application_without_recipes_lists_none_and_opens_none():
    api = old_application()
    assert await api.list_recipes(GUILD) is None
    assert await api.open_recipe(GUILD, "Welcome") is None
    await api.close()


async def test_a_wake_against_an_application_without_recipes_runs_without_them():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    api = RecipeAPI()
    old = old_application()
    api.list_recipes = old.list_recipes
    api.open_recipe = old.open_recipe
    seen: list = []
    runtime, writer = real_runtime(
        redis_client, api, [("open_recipe", {"name": "Welcome"}), None], seen
    )

    await runtime.process(batch())

    assert "RECIPES" not in brief_of(seen[0][0])
    assert runtime.recipe_listing == {"recipes": []}
    (result,) = tool_results(runtime.engine.agent_runner.history, "open_recipe")
    assert result.startswith('No recipe named "Welcome" is available.')
    await writer.close()
    await old.close()


async def test_the_client_asks_for_a_recipe_by_name():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=welcome())

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="key",
        transport=httpx.MockTransport(handler),
    )
    assert await api.open_recipe(GUILD, "Weekly challenge") == welcome()
    assert requests[0].url.path == f"/api/guilds/{GUILD}/proactive-recipes/by-name"
    assert requests[0].url.params["name"] == "Weekly challenge"
    await api.close()


def test_runtime_deps_carry_open_recipe():
    deps = ProactiveDeps.__dataclass_fields__
    assert "open_recipe" in deps


class Blocks:
    def __init__(self, *user_ids):
        self.user_ids = set(user_ids)

    def is_blocked(self, user_id, message_id=None):
        return str(user_id) in self.user_ids


async def test_a_recipe_reaches_the_model_without_an_opted_out_members_id():
    redis_client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    text = f"Use this when <@{TARGET}> posts a release.\n\nThank <@{TARGET}>."
    api = RecipeAPI([welcome(text=text)])
    seen: list = []
    runtime, writer = real_runtime(
        redis_client, api, [("open_recipe", {"name": "Welcome"}), None], seen
    )
    runtime.blocked_users = Blocks(TARGET)

    await runtime.process(batch())

    assert TARGET not in brief_of(seen[0][0])
    assert "Use this when @[blocked user] posts a release." in brief_of(seen[0][0])
    (result,) = tool_results(runtime.engine.agent_runner.history, "open_recipe")
    assert TARGET not in result
    assert result.endswith("Thank @[blocked user].")
    await writer.close()
