"""Credential lookup and the v1 encoding shared byte for byte with smarter-dev #482."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from proactive_agent import credentials
from proactive_agent.credentials import (
    CREDENTIALS,
    decode_credential,
    encode_credential,
    get_credential,
)

HUNDRED = "".join(chr(ord("a") + i % 26) for i in range(100))
VECTORS = {
    "hello": "v1.-PEJJ30",
    HUNDRED: (
        "v1.8fYGL3ffnLgeWOKWu7SXGM0HxWDKe99rYkhhqXx6esSxwJ1exa0o3_tIAHdkoi1LK_ib1ff"
        "ado-_J6v18XOqbewvuyfC5Qq_UrICBwjGMhZEUxpOvbVm85RMj_Rruyn6ttJxJg"
    ),
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for spec in CREDENTIALS.values():
        for name in (spec.opaque, *spec.legacy):
            if name:
                monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("value", VECTORS)
def test_vectors(value) -> None:
    assert encode_credential(value) == VECTORS[value]
    assert decode_credential(VECTORS[value]) == value


def test_vector_input_is_100_characters() -> None:
    assert len(HUNDRED) == 100


def test_keystream_spans_blocks() -> None:
    # 100 bytes need four sha256 blocks; a round trip of unicode crosses them too.
    text = "ключ-🔑-" * 20
    assert decode_credential(encode_credential(text)) == text


@pytest.mark.parametrize("bad", ["plain", "v2.abc", "v1.!!!!"])
def test_decode_rejects_bad_values(bad) -> None:
    with pytest.raises(ValueError):
        decode_credential(bad)


def test_opaque_wins_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("KMSVP8", encode_credential("from-opaque"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-legacy")
    assert get_credential("anthropic-api-key") == "from-opaque"


def test_legacy_order(monkeypatch) -> None:
    monkeypatch.setenv("CLAUDE_API_KEY", "second")
    assert get_credential("anthropic-api-key") == "second"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "first")
    assert get_credential("anthropic-api-key") == "first"


def test_missing_gives_default(monkeypatch) -> None:
    assert get_credential("jina-api-key") == ""
    assert get_credential("jina-api-key", "dflt") == "dflt"


def test_invalid_opaque_falls_back_and_logs_no_value(monkeypatch, caplog) -> None:
    monkeypatch.setenv("IRBCKR", "not-encoded-secret-value")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-value")
    with caplog.at_level(logging.DEBUG, logger="proactive_agent.credentials"):
        assert get_credential("openai-api-key") == "legacy-value"
    assert "IRBCKR" in caplog.text
    assert "not-encoded-secret-value" not in caplog.text
    assert "legacy-value" not in caplog.text


def test_source_is_logged_without_value(monkeypatch, caplog) -> None:
    monkeypatch.setenv("AO6HUE", encode_credential("brave-secret"))
    with caplog.at_level(logging.DEBUG, logger="proactive_agent.credentials"):
        assert get_credential("brave-search-api-key") == "brave-secret"
    assert "AO6HUE" in caplog.text
    assert "brave-secret" not in caplog.text


def test_settings_read_opaque_vars(monkeypatch) -> None:
    from proactive_agent.config import Settings

    monkeypatch.setenv("FUC9HA", encode_credential("agent-key"))
    monkeypatch.setenv("FHI1OT", encode_credential("bot-token"))
    settings = Settings(_env_file=None)
    assert settings.proactive_agent_api_key == "agent-key"
    assert settings.discord_bot_token == "bot-token"
    settings.require_runtime_secrets()


def test_provider_builders_use_decoded_keys(monkeypatch) -> None:
    from proactive_agent.models import anthropic_api_key

    monkeypatch.setenv("KMSVP8", encode_credential("sk-decoded"))
    assert anthropic_api_key() == "sk-decoded"


def test_no_legacy_env_reads_outside_credentials() -> None:
    """Non-test code must go through get_credential, not read the old names."""
    names = {n for spec in CREDENTIALS.values() for n in spec.legacy}
    src = Path(credentials.__file__).parent
    pattern = re.compile("|".join(sorted(map(re.escape, names), key=len, reverse=True)))
    offenders = [
        f"{path.name}:{n}"
        for path in src.rglob("*.py")
        if path.name != "credentials.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []
