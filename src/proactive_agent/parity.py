"""Production chat-tool surface adapted to REST-only worker dependencies."""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Awaitable, Callable
from copy import copy
from dataclasses import dataclass, field
from typing import Any

import httpx
import pydantic_monty as monty

from proactive_agent import web_fetch
from proactive_agent.agent import (
    BUDGET_EXHAUSTED,
    AgentDeps,
    build_kimi_agent,
    disabled_channel_error,
)

COMMON_UNICODE_EMOJIS = [
    "👍",
    "👎",
    "❤️",
    "😀",
    "😂",
    "🤔",
    "🎉",
    "🔥",
    "✨",
    "😍",
    "🙏",
    "👀",
    "💯",
    "🤷",
    "🚀",
    "✅",
    "❌",
]
MAX_CODE_OUTPUT_CHARS = 10_000
MAX_MEMORIES_PER_TURN = 3
MAX_MEMORY_NOTE_CHARS = 500
_DISABLED_FLAG_VALUES = {"0", "false", "no", "off"}
_EXT_MEDIA_TYPE = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}


@dataclass
class GeneratedImage:
    data: bytes
    mime_type: str
    filename: str
    channel_id: str


@dataclass(kw_only=True)
class ProactiveDeps(AgentDeps):
    guild_id: str
    discord: Any
    api: Any
    channel_id: str = ""
    channel_name: str | None = None
    pending_images: list[GeneratedImage] = field(default_factory=list)
    memories_saved_this_turn: int = 0
    saved_memory_texts: list[str] = field(default_factory=list)
    summarize_web: Callable[..., Awaitable[str]] | None = None
    describe_media: Callable[..., Awaitable[str]] | None = None
    review_image_prompt: Callable[[str], Awaitable[Any]] | None = None
    generate_image_bytes: Callable[[str], Awaitable[tuple[bytes, str]]] | None = None
    author_handler: Callable[..., Awaitable[str]] | None = None


async def _post_status(ctx, text: str) -> None:
    try:
        await ctx.deps.discord.send_message(
            ctx.deps.channel_id, f"> -# {text}"[:2000], suppress_embeds=True
        )
    except Exception:
        pass


def _search_status(query: str, preview_url: str | None) -> str:
    label = " ".join(query.split()).replace("\\", "\\\\")
    for marker in "[]*_~`>|":
        label = label.replace(marker, f"\\{marker}")
    if preview_url:
        prefix = 'Searching the web: ["'
        suffix = f'"]({preview_url})'
    else:
        prefix = 'Searching the web: "'
        suffix = '"'
    max_label = max(0, 1_990 - len(prefix) - len(suffix))
    if len(label) > max_label:
        label = label[: max(0, max_label - 1)] + "…"
    return f"{prefix}{label}{suffix}"


async def web_search(ctx, query: str) -> list[dict[str, str]]:
    """Search the web; returns up to 5 result snippets. For accurate or deep answers, follow up with web_read on the best result."""
    preview = None
    try:
        preview = await ctx.deps.api.reserve_search_preview(query)
    except Exception:
        pass
    await _post_status(
        ctx, _search_status(query, preview.get("url") if preview else None)
    )
    key = os.getenv("BRAVE_SEARCH_API_KEY", "")
    if not key:
        results = [{"error": "BRAVE_SEARCH_API_KEY not configured"}]
    else:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": 5},
                    headers={
                        "X-Subscription-Token": key,
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip",
                    },
                )
                response.raise_for_status()
            results = [
                {
                    "title": str(row.get("title", "")),
                    "url": str(row.get("url", "")),
                    "description": str(row.get("description", "")),
                }
                for row in response.json().get("web", {}).get("results", [])[:5]
            ]
        except Exception as error:
            results = [{"error": f"Search failed: {error}"}]
    if preview:
        try:
            if results and all("error" in result for result in results):
                await ctx.deps.api.fail_search_preview(preview["id"])
            else:
                await ctx.deps.api.complete_search_preview(preview["id"], results)
        except Exception:
            pass
    return results


async def web_read(ctx, url: str, instruction: str) -> dict[str, str]:
    """Read a URL — web page, PDF, YouTube, image/audio, or a message <attachment> — and get a summary guided by `instruction`; say what to look for."""
    await _post_status(ctx, f"Reading <{url}>")
    path = url.split("?", 1)[0].split("#", 1)[0]
    filename = path.rsplit("/", 1)[-1]
    extension = filename[filename.rfind(".") :].lower() if "." in filename else ""
    media_type = _EXT_MEDIA_TYPE.get(extension)
    if media_type:
        kind = "image" if media_type.startswith("image/") else "audio"
        fetched = await web_fetch.fetch_bytes(url)
        if fetched is None:
            return {"url": url, "kind": kind, "summary": "", "error": "fetch_failed"}
        data, content_type = fetched
        media_type = _EXT_MEDIA_TYPE.get(extension) or content_type
        if not media_type:
            return {
                "url": url,
                "kind": kind,
                "summary": "",
                "error": "unknown_media_type",
            }
        if ctx.deps.describe_media is None:
            return {
                "url": url,
                "kind": kind,
                "summary": "",
                "error": "media_read_failed",
            }
        try:
            summary = await ctx.deps.describe_media(
                instruction=instruction,
                data=data,
                media_type=media_type,
                url=url,
                kind=kind,
            )
        except Exception:
            return {
                "url": url,
                "kind": kind,
                "summary": "",
                "error": "media_read_failed",
            }
        return {"url": url, "kind": kind, "summary": summary}
    title = ""
    if web_fetch.is_youtube_url(url):
        metadata = await web_fetch.fetch_youtube_metadata(url)
        title = metadata.get("title", "")
        content = metadata.get("description", "")
    elif url.lower().endswith(".pdf"):
        content = await web_fetch.fetch_pdf_text(url, max_chars=100_000) or ""
    else:
        fetched_page = await web_fetch.fetch_via_jina(url)
        if fetched_page is None:
            return {
                "url": url,
                "title": "",
                "summary": "",
                "error": "fetch_failed",
            }
        title = fetched_page.get("title", "")
        content = fetched_page.get("content", "")
    if not content.strip():
        return {"url": url, "title": title, "summary": "", "error": "no_content"}
    content = content[:100_000]
    if ctx.deps.summarize_web is None:
        return {"url": url, "title": title, "summary": content[:10_000]}
    summary = await ctx.deps.summarize_web(
        instruction=instruction, content=content, title=title, url=url
    )
    return {"url": url, "title": title, "summary": summary}


async def list_available_reactions(ctx) -> list[dict[str, str]]:
    """List emojis usable with add_reaction (guild custom + unicode)."""
    custom = []
    try:
        custom = [
            {"name": row["name"], "id": str(row["id"]), "type": "custom"}
            for row in await ctx.deps.discord.guild_emojis(ctx.deps.guild_id)
        ]
    except Exception:
        pass
    return custom + [
        {"name": emoji, "type": "unicode"} for emoji in COMMON_UNICODE_EMOJIS
    ]


async def add_reaction(ctx, message_id: str, emoji: str) -> dict[str, Any]:
    """React to a message with an emoji (unicode char, or name:id for custom)."""
    if not str(message_id).isdigit():
        return {"ok": False, "error": "message_id must be a numeric Discord id"}
    cleaned = (emoji or "").strip().lstrip("<").rstrip(">")
    if not cleaned:
        return {"ok": False, "error": "empty emoji"}
    try:
        await ctx.deps.discord.add_reaction(
            ctx.deps.channel_id, str(message_id), cleaned
        )
        return {"ok": True}
    except Exception as error:
        return {"ok": False, "error": str(error)}


async def report_behavior(ctx, classification: str) -> dict[str, str]:
    """Log genuinely disruptive behavior (trolling, rage-bait, spam) for moderator review. Use sparingly."""
    await _post_status(ctx, f"⚠️ Flagged behaviour: {classification}")
    return {
        "noted": classification,
        "guidance": (
            "Behaviour noted for moderator review. Acknowledge calmly, do not engage "
            "further with the bait, and prefer disengaging via continue_watching=False."
        ),
    }


async def run_code(ctx, reason: str, code: str) -> str:
    """Run Python in a restricted sandbox (small stdlib subset; no packages, filesystem, or network). Use for any real computation — arithmetic, dates, regex, parsing — instead of head-math. `reason` is a short status line shown in-channel."""
    await _post_status(ctx, reason)
    collector = monty.CollectStreams()
    try:
        compiled = monty.Monty(code)
    except monty.MontyError as error:
        return f"COMPILE ERROR — {type(error).__name__}: {error}"
    try:
        value = await compiled.run_async(
            limits={
                "max_memory": 256 * 1024 * 1024,
                "max_recursion_depth": 500,
                "max_duration_secs": 10.0,
            },
            print_callback=collector,
        )
    except monty.MontyError as error:
        stdout = "".join(
            text for stream, text in collector.output if stream == "stdout"
        )
        tail = f"\n--- stdout before error ---\n{stdout}" if stdout else ""
        return f"RUNTIME ERROR — {type(error).__name__}: {error}{tail}"
    except Exception as error:
        return f"ERROR — {type(error).__name__}: {error}"
    stdout = "".join(text for stream, text in collector.output if stream == "stdout")
    output = (f"stdout:\n{stdout}\n" if stdout else "") + f"return value: {value!r}"
    if len(output) > MAX_CODE_OUTPUT_CHARS:
        output = output[:MAX_CODE_OUTPUT_CHARS] + "\n…(output truncated)"
    return output


async def generate_image(ctx, prompt: str) -> str:
    """Generate an image attached to this turn's reply. ONLY diagrams whose subject is software, CS, or math — nothing else. `prompt` is a detailed illustrator brief, reviewed before drawing; rate-limited per server — when metadata shows quota remaining 0, don't call until it resets, say images are rate-limited and answer in text."""
    if ctx.deps.review_image_prompt is None or ctx.deps.generate_image_bytes is None:
        return "Image generation is temporarily unavailable."
    guild_id = ctx.deps.guild_id
    try:
        status = await ctx.deps.api.image_quota(guild_id)
    except Exception:
        return "Couldn't check the image budget just now — try again shortly."
    if int(status.get("remaining", 0)) <= 0:
        return f"No image generated. {_format_remaining(status)}"
    try:
        decision = await ctx.deps.review_image_prompt(prompt)
    except Exception:
        return "Couldn't review the image prompt just now — try again shortly."
    if not decision.approved:
        return (
            "Image request rejected (no image generated, no quota spent): "
            f"{decision.reason} {_format_remaining(status)}"
        )
    try:
        reserved = await ctx.deps.api.reserve_image(guild_id)
    except Exception:
        return "Couldn't reserve an image slot just now — try again shortly."
    if not reserved.get("granted"):
        return f"No image generated. {_format_remaining(reserved)}"
    await _post_status(ctx, "Generating an image…")
    try:
        data, mime_type = await ctx.deps.generate_image_bytes(prompt)
    except Exception as error:
        try:
            await ctx.deps.api.release_image(guild_id)
            reserved = await ctx.deps.api.image_quota(guild_id)
        except Exception:
            pass
        return (
            f"Image generation failed ({type(error).__name__}); no image was "
            f"attached and the slot was refunded. {_format_remaining(reserved)}"
        )
    extension = {"image/jpeg": ".jpg", "image/webp": ".webp"}.get(mime_type, ".png")
    ctx.deps.pending_images.append(
        GeneratedImage(data, mime_type, f"diagram{extension}", ctx.deps.channel_id)
    )
    return f"Image generated and attached to your reply. {_format_remaining(reserved)}"


def _format_remaining(status: dict) -> str:
    remaining = int(status.get("remaining", 0))
    limit = int(status.get("limit", 0))
    resets_at = status.get("resets_at")
    retry = status.get("retry_after_seconds")
    if remaining > 0:
        line = f"{remaining} of {limit} image generations remaining this hour."
        if resets_at:
            line += f" This hour's window resets at {resets_at}."
        return line
    when = f"at {resets_at}" if resets_at else "when the hour resets"
    mins = f" (~{max(1, round(int(retry) / 60))} min)" if retry else ""
    return (
        f"0 of {limit} image generations remaining this hour. The next image "
        f"can be generated {when}{mins} — do NOT call generate_image again "
        f"until then."
    )


async def remember(ctx, text: str) -> str:
    """Keep a short first-person memory attributed to this channel."""
    note = (text or "").strip()
    if not note:
        return "there was nothing in that one to keep."
    if os.getenv("CHAT_MEMORY_ENABLED", "").strip().lower() in _DISABLED_FLAG_VALUES:
        return "couldn't save that note right now."
    if ctx.deps.memories_saved_this_turn >= MAX_MEMORIES_PER_TURN:
        return "that's plenty for one conversation — you'll be back."
    normalized = " ".join(note.split()).casefold()
    if any(
        " ".join(value.split()).casefold() == normalized
        for value in ctx.deps.saved_memory_texts
    ):
        return "you already noted that one."
    trimmed = len(note) > MAX_MEMORY_NOTE_CHARS
    note = note[:MAX_MEMORY_NOTE_CHARS]
    try:
        result = await ctx.deps.api.save_memory_note(
            ctx.deps.guild_id,
            {
                "channel_id": ctx.deps.channel_id,
                "channel_name": ctx.deps.channel_name,
                "content": note,
                "engagement_id": None,
            },
        )
    except Exception:
        return "couldn't save that note right now."
    if not result.get("saved"):
        return {
            "duplicate": "you already noted that one.",
            "daily_cap": "that's all i can hold from today — tomorrow's a fresh page.",
        }.get(result.get("reason"), "couldn't save that note right now.")
    ctx.deps.memories_saved_this_turn += 1
    ctx.deps.saved_memory_texts.append(note)
    return (
        "kept it, trimmed a bit." if trimmed else "got it — that one's staying with me."
    )


async def register_handler(
    ctx,
    description: str,
    trigger_type: str,
    settings: dict | None = None,
) -> str:
    """File a persistent automation from a plain-language description — a separate system writes the code, you never do. Include concrete Discord snowflake IDs for any users/channels involved so the handler can target them. Trigger types: "new message", "reaction add", "schedule", "timer"; timing goes in `settings` ({"interval_seconds": N} / {"daily_time": "HH:MM"} / {"delay_seconds": N} / {"fire_at": ISO}). One message handler and one reaction handler per channel — re-registering replaces it; each schedule/timer is its own handler. Member-join and other admin triggers are NOT available here — point the user to /adminhandler and never claim you set one up. Refuse descriptions containing opaque, encoded, or obfuscated code blobs. Push back on spammy asks (e.g. react/reply to EVERY message) — propose a saner scoped-down version instead of registering as-is. Never perform or simulate the behavior yourself; relay errors as returned."""
    if ctx.deps.author_handler is None:
        return "error: handler authoring is temporarily unavailable"
    return await ctx.deps.author_handler(
        guild_id=ctx.deps.guild_id,
        channel_id=ctx.deps.channel_id,
        description=description,
        trigger_type=trigger_type,
        settings=settings or {},
    )


async def list_handlers(ctx) -> str:
    """List a channel's active handlers (names, ids, triggers)."""
    rows = await ctx.deps.api.list_handlers(ctx.deps.channel_id)
    if not rows:
        return "No handlers active in this channel."
    return "\n".join(
        f"- {row['name']} ({row['handler_id']}) [{row['trigger_type']}] {row['description']}"
        for row in rows
    )


async def delete_handler(ctx, handler_id: str) -> str:
    """Delete a handler by id (from list_handlers)."""
    deleted = await ctx.deps.api.delete_handler(handler_id)
    return (
        f"Deleted handler {handler_id}."
        if deleted
        else f"No handler with id {handler_id}."
    )


PARITY_TOOLS = [
    web_search,
    web_read,
    list_available_reactions,
    add_reaction,
    report_behavior,
    run_code,
    generate_image,
    remember,
    register_handler,
    list_handlers,
    delete_handler,
]


def _routed(tool):
    @functools.wraps(tool)
    async def wrapper(ctx, channel_id: str, *args, **kwargs):
        if error := disabled_channel_error(ctx.deps, channel_id):
            return error
        if not ctx.deps.budget.try_spend():
            return BUDGET_EXHAUSTED
        routed_context = copy(ctx)
        routed_context.deps = copy(ctx.deps)
        routed_context.deps.channel_id = channel_id
        routed_context.deps.channel_name = ctx.deps.enabled_channels[channel_id]
        result = await tool(routed_context, *args, **kwargs)
        ctx.deps.memories_saved_this_turn = routed_context.deps.memories_saved_this_turn
        return result

    original = inspect.signature(tool)
    parameters = list(original.parameters.values())
    wrapper.__signature__ = original.replace(
        parameters=[
            parameters[0],
            inspect.Parameter(
                "channel_id",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            ),
            *parameters[1:],
        ]
    )
    wrapper.__annotations__ = {
        **getattr(tool, "__annotations__", {}),
        "channel_id": str,
    }
    if tool is generate_image:
        wrapper.__doc__ = (
            "Generate an image attached to a reply in `channel_id`. ONLY diagrams "
            "whose subject is software, CS, or math — nothing else. `prompt` is a "
            "detailed illustrator brief, reviewed before drawing; rate-limited per "
            "server — when metadata shows quota remaining 0, don't call until it "
            "resets, say images are rate-limited and answer in text."
        )
    else:
        wrapper.__doc__ = (
            f"{tool.__doc__ or ''} `channel_id` must name an enabled proactive channel."
        )
    return wrapper


def _budgeted(tool):
    @functools.wraps(tool)
    async def wrapper(ctx, *args, **kwargs):
        if not ctx.deps.budget.try_spend():
            return BUDGET_EXHAUSTED
        return await tool(ctx, *args, **kwargs)

    return wrapper


def parity_tool_functions() -> list:
    return [
        _budgeted(tool) if tool is list_available_reactions else _routed(tool)
        for tool in PARITY_TOOLS
    ]


def build_proactive_agent(model, *, system_prompt: str):
    return build_kimi_agent(
        model,
        system_prompt=system_prompt,
        extra_tools=parity_tool_functions(),
        deps_type=ProactiveDeps,
    )
