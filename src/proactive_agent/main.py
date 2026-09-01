"""Process entry point for the standalone proactive-agent worker."""

from __future__ import annotations

import asyncio
import logging
import signal
import socket

import redis.asyncio as redis

from proactive_agent.agent import (
    KimiAgentRunner,
    build_guild_agent_system_prompt,
    self_compaction_summary,
)
from proactive_agent.api import ApplicationAPI
from proactive_agent.capabilities import (
    HandlerAuthor,
    ImageCapabilities,
    MediaReader,
    WebSummarizer,
)
from proactive_agent.config import Settings
from proactive_agent.discord import DiscordREST
from proactive_agent.engine import AgentEngine, SkimRunner
from proactive_agent.health import HealthServer
from proactive_agent.history import DebouncedHistoryWriter, GuildHistoryRepository
from proactive_agent.models import build_model
from proactive_agent.parity import build_proactive_agent
from proactive_agent.queue import RedisWakeQueue
from proactive_agent.runtime import ActionJournal, GuildRuntime, GuildRuntimeRegistry
from proactive_agent.worker import ProactiveWorker


async def run() -> None:
    settings = Settings()
    settings.require_runtime_secrets()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    redis_client = redis.from_url(settings.redis_url, decode_responses=False)
    await redis_client.ping()
    api = ApplicationAPI(
        base_url=settings.api_base_url,
        api_key=settings.proactive_agent_api_key,
    )
    discord = DiscordREST(bot_token=settings.discord_bot_token)
    history_repository = GuildHistoryRepository(redis_client, api)
    history_writer = DebouncedHistoryWriter(
        history_repository,
        api,
        debounce_seconds=settings.proactive_history_debounce_seconds,
    )
    queue = RedisWakeQueue(
        redis_client,
        consumer_name=f"{socket.gethostname()}-{id(asyncio.current_task())}",
        lease_seconds=settings.proactive_lease_seconds,
        reclaim_idle_seconds=settings.proactive_reclaim_idle_seconds,
    )
    me = await discord.current_user()

    async def build_runtime(guild_id: str) -> GuildRuntime:
        guild = await discord.guild(guild_id)
        agent_model = build_model(settings.proactive_agent_model)
        skim_model = build_model(settings.proactive_skim_model)
        system_prompt = build_guild_agent_system_prompt(
            bot_display_name=me.get("username") or "the bot",
            bot_user_id=str(me["id"]),
            guild_name=guild.get("name") or guild_id,
        )
        agent = build_proactive_agent(agent_model, system_prompt=system_prompt)
        summarize_web = WebSummarizer(
            build_model(settings.proactive_web_summarizer_model),
            fallback_model=build_model(
                settings.proactive_web_summarizer_fallback_model
            ),
        )
        image_capabilities = ImageCapabilities(
            reviewer_model=build_model(settings.proactive_image_reviewer_model),
        )
        media_reader = MediaReader(build_model(settings.proactive_media_reader_model))
        author_handler = HandlerAuthor(
            api=api,
            model=build_model(settings.proactive_handler_model),
            discord=discord,
        )

        async def summarize(messages) -> str:
            summary, _usage = await self_compaction_summary(agent_model, messages)
            return summary

        runner = KimiAgentRunner(agent=agent, summarize=summarize)
        engine = AgentEngine(
            agent_runner=runner,
            skim=SkimRunner(skim_model),
            agent_model_id=settings.proactive_agent_model,
            skim_model_id=settings.proactive_skim_model,
            deps_factory=lambda **kwargs: None,
        )
        return GuildRuntime(
            guild_id=guild_id,
            engine=engine,
            history_repository=history_repository,
            history_writer=history_writer,
            api=api,
            discord=discord,
            redis=redis_client,
            queue=queue,
            journal=ActionJournal(redis_client),
            bot_user_id=str(me["id"]),
            guild_name=guild.get("name") or guild_id,
            summarize_web=summarize_web,
            image_capabilities=image_capabilities,
            media_reader=media_reader,
            author_handler=author_handler,
        )

    runtimes = GuildRuntimeRegistry(build_runtime)
    worker = ProactiveWorker(
        queue,
        runtimes,
        concurrency=settings.proactive_worker_concurrency,
        max_attempts=settings.proactive_max_attempts,
    )
    health = HealthServer(redis_client, api, port=settings.proactive_health_port)
    await health.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        await worker.run(stop)
    finally:
        await health.close()
        await history_writer.close(
            timeout=settings.proactive_history_flush_timeout_seconds
        )
        await discord.close()
        await api.close()
        await redis_client.aclose()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
