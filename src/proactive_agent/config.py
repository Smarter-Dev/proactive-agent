"""Worker configuration loaded entirely from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    redis_url: str = "redis://localhost:6379/0"
    api_base_url: str = "http://localhost:8000/api"
    proactive_agent_api_key: str = ""
    discord_bot_token: str = ""
    proactive_agent_model: str = "gemini-3.7-flash"
    proactive_skim_model: str = "z-ai/glm-5.3-flash"
    proactive_web_summarizer_model: str = "openai/gpt-5.6-luna"
    proactive_web_summarizer_fallback_model: str = "gemini-3.5-flash-lite"
    proactive_handler_model: str = "gemini-3-flash-preview"
    proactive_image_reviewer_model: str = "gemini-3.1-flash-lite"
    proactive_media_reader_model: str = "gemini-3.1-flash-lite"
    proactive_worker_concurrency: int = Field(8, ge=1, le=128)
    proactive_history_debounce_seconds: float = Field(5, ge=0, le=300)
    proactive_history_flush_timeout_seconds: float = Field(10, gt=0, le=300)
    proactive_lease_seconds: int = Field(180, ge=30, le=3600)
    proactive_reclaim_idle_seconds: int = Field(240, ge=30, le=86400)
    proactive_max_attempts: int = Field(5, ge=1, le=100)
    proactive_health_port: int = Field(8080, ge=1, le=65535)
    log_level: str = "INFO"

    def require_runtime_secrets(self) -> None:
        missing = [
            name
            for name, value in (
                ("PROACTIVE_AGENT_API_KEY", self.proactive_agent_api_key),
                ("DISCORD_BOT_TOKEN", self.discord_bot_token),
            )
            if not value
        ]
        if missing:
            raise ValueError(f"missing required settings: {', '.join(missing)}")
