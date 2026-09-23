"""Worker failure notes are durable history, not new wake notifications."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

from proactive_agent.runtime import GuildRuntime


def runtime():
    runner = SimpleNamespace(history=[])
    writer = SimpleNamespace(save=AsyncMock(return_value=SimpleNamespace(revision=1)))
    queue = SimpleNamespace(publish=AsyncMock())
    agent = GuildRuntime(
        guild_id="111",
        engine=SimpleNamespace(agent_runner=runner),
        history_repository=SimpleNamespace(
            load=AsyncMock(return_value=SimpleNamespace(history=[], revision=0))
        ),
        history_writer=writer,
        api=None,
        discord=None,
        redis=None,
        queue=queue,
        journal=None,
        bot_user_id="bot",
        guild_name="guild",
        summarize_web=None,
        image_capabilities=None,
        media_reader=None,
        author_handler=None,
    )
    return agent, writer, queue


async def test_recovered_unavailable_note_is_saved_once_without_new_wake():
    agent, writer, queue = runtime()
    batch = SimpleNamespace(wake_id="wake-1")

    await agent.record_unavailable_retries(batch, retries=2, recovered=True)
    await agent.record_unavailable_retries(batch, retries=2, recovered=True)

    writer.save.assert_awaited_once()
    queue.publish.assert_not_awaited()
    history = agent.engine.agent_runner.history
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[1], ModelResponse)
    note = history[0].parts[0]
    assert isinstance(note, UserPromptPart)
    assert "not a Discord user message" in note.content
    assert "retried 2 time(s)" in note.content
    assert "eventually completed" in note.content
    assert agent.history_revision == 1
    assert ModelMessagesTypeAdapter.validate_python(
        writer.save.await_args.kwargs["history"]
    )


async def test_exhausted_unavailable_note_says_wake_may_have_been_missed():
    agent, writer, queue = runtime()

    await agent.record_unavailable_retries(
        SimpleNamespace(wake_id="wake-2"), retries=3, recovered=False
    )

    note = agent.engine.agent_runner.history[0].parts[0].content
    assert "retried 3 time(s)" in note
    assert "may have missed an action" in note
    assert writer.save.await_args.kwargs["previous_revision"] == 0
    queue.publish.assert_not_awaited()
