# Proactive Agent

Standalone worker for the Smarter Dev proactive Discord agent.

The Discord bot remains responsible for Gateway events, message batching, and
watcher classification. This service consumes guild-specific Redis wake
streams, maintains one isolated logical agent per guild, uses the application
REST API for durable data, and uses Discord REST for Discord operations.

The directory is an independently installable project and is intended to be
split into its own repository. It must not import the parent `smarter_dev`
package.

## Local development

```bash
uv sync --group dev
uv run pytest
uv run proactive-agent
```

Required runtime variables are documented in `.env.example`.

## Production deployment

The service deploys as its own `smarter-dev-proactive-agent` Deployment in the
existing `smarter-dev` DigitalOcean Kubernetes namespace. It shares Redis and
the Discord bot identity with the bot, reaches the application through the
internal `API_BASE_URL`, and receives no database credentials. No Kubernetes
Service or ingress is required because the worker only makes outbound calls;
the kubelet probes its health port directly.

The repository CI workflow tests every change. A successful push to `main`
builds `zzmmrmn/smarter-dev-proactive-agent:<commit>`, pushes it to Docker Hub,
applies `deploy/kubernetes.yaml`, and waits for rollout completion. Configure
the same repository Actions secrets used by the bot deployment:

- `DIGITALOCEAN_ACCESS_TOKEN`
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

Before the first deployment, add this key to the existing
`smarter-dev-secrets` Kubernetes Secret:

- `proactive-agent-api-key`: a distinct Skrift key with `bot-api` permission

The manifest continues to use the existing `redis-url`, `discord-bot-token`,
provider-key, and `brave-search-api-key` entries. Production model calls use
those direct provider credentials; LiteLLM remains a local-development option.

Deploy the additive application migration and REST endpoints from the bot
repository before deploying this worker. Keep the bot in `embedded` mode until
the worker is ready; a running worker cannot acquire a guild unless the bot has
marked that guild `external` in Redis.

## Runtime contract

- The Discord bot is the sole Gateway consumer and owns watcher cadence.
- Waking envelopes go to `proactive:v1:{guild:<id>}:wake`; non-waking context
  waits in the guild's bounded pending list until the next wake.
- Workers use consumer groups and a guild lease, so replicas process different
  guilds concurrently while one guild remains serialized.
- The bot writes a per-guild Redis ownership fence at startup and on every
  notification. Workers can acquire or renew a lease only while that guild is
  marked `external`; rollback to `embedded` drains stale external wakes without
  executing their side effects.
- Discord actions use the same bot token through Discord REST. All durable
  application data is read or written through the authenticated application
  API; the worker has no database credentials.
- Agent history is cached immediately in Redis and flushed to Postgres through
  the API after the configured debounce. Shutdown attempts a final flush.

## Rollout from the bot repository

1. Deploy the history API migration and this worker with the bot still in
   `embedded` mode.
2. Put a test guild in `PROACTIVE_AGENT_SHADOW_GUILD_IDS`. The bot continues to
   own side effects while publishing the exact Redis envelopes for inspection.
3. Move that guild to `PROACTIVE_AGENT_EXTERNAL_GUILD_IDS`. This disables its
   embedded consumer and makes the worker the only side-effect owner.
4. Compare response/reaction counts, queue lag, dead letters, history revisions,
   and Discord-visible behavior before expanding the comma-separated guild list.
5. After every guild is external, set `PROACTIVE_AGENT_EXECUTION_MODE=external`.

`PROACTIVE_AGENT_EMBEDDED_GUILD_IDS` provides a per-guild rollback override
after the global mode is external. The embedded, shadow, and external guild
lists are mutually exclusive; the bot refuses an ambiguous configuration.

The bot and worker must share Redis, `DISCORD_BOT_TOKEN`, and an API key with
the existing `bot-api` permission. Rolling back is the reverse flag change;
pending Redis wakes and persisted guild history remain available.

For an immediate per-guild rollback, remove the guild from
`PROACTIVE_AGENT_EXTERNAL_GUILD_IDS`, add it to
`PROACTIVE_AGENT_EMBEDDED_GUILD_IDS`, and redeploy the bot. Do not place a guild
in more than one list. The bot refreshes the Redis ownership fence before its
embedded consumer can resume side effects.

## Repository split

This directory has no imports from the parent application and can become the
root of its own repository with history preserved:

```bash
git subtree split --prefix=services/proactive-agent -b proactive-agent-main
git push <new-repository> proactive-agent-main:main
```

The canonical notification and control schemas must be released in lockstep
with the producer. Versioned models reject unknown fields to prevent silent
contract drift.
