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

# The plain names phase 1 still read; nothing may read them now.
LEGACY_NAMES = (
    "ANTHROPIC_API_KEY",
    "CLAUDE_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPEN_ROUTER_API_KEY",
    "OPEN_ROUTER",
    "BRAVE_SEARCH_API_KEY",
    "JINA_API_KEY",
    "DISCORD_BOT_TOKEN",
    "PROACTIVE_AGENT_API_KEY",
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
        for name in (spec.opaque, *spec.legacy, *LEGACY_NAMES):
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


@pytest.mark.parametrize(
    "bad",
    ["plain", "v2.abc", "v1.!!!!", "v1.ab+c", "v1.ab/c", "v1.-PEJJ30=", "v1.-PEJJ30=="],
)
def test_decode_rejects_bad_values(bad) -> None:
    with pytest.raises(ValueError):
        decode_credential(bad)


def test_legacy_names_are_not_read(monkeypatch) -> None:
    for name in LEGACY_NAMES:
        monkeypatch.setenv(name, "plain-secret")
    for logical, spec in CREDENTIALS.items():
        if spec.opaque:
            assert get_credential(logical) == ""


def test_missing_gives_default(monkeypatch) -> None:
    assert get_credential("jina-api-key") == ""
    assert get_credential("jina-api-key", "dflt") == "dflt"


def test_plain_value_under_opaque_name_is_refused(monkeypatch, caplog) -> None:
    monkeypatch.setenv("IRBCKR", "not-encoded-secret-value")
    with caplog.at_level(logging.DEBUG, logger="proactive_agent.credentials"):
        assert get_credential("openai-api-key") == ""
    assert "IRBCKR" in caplog.text
    assert "not-encoded-secret-value" not in caplog.text


def test_dev_flag_accepts_plain_value_under_opaque_name(monkeypatch) -> None:
    monkeypatch.setenv("PROACTIVE_ALLOW_PLAIN_CREDENTIALS", "1")
    monkeypatch.setenv("IRBCKR", "plain-dev-value")
    assert get_credential("openai-api-key") == "plain-dev-value"
    # A malformed v1 value is still an error, not a plain key.
    monkeypatch.setenv("IRBCKR", "v1.!!!!")
    assert get_credential("openai-api-key") == ""


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
    pattern = re.compile(
        "|".join(sorted(map(re.escape, LEGACY_NAMES), key=len, reverse=True))
    )
    src = Path(credentials.__file__).parent
    offenders = [
        f"{path.name}:{n}"
        for path in src.rglob("*.py")
        if path.name != "credentials.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert offenders == []


def test_manifest_has_no_legacy_credential_env() -> None:
    manifest = Path(credentials.__file__).parents[2] / "deploy" / "kubernetes.yaml"
    names = set(re.findall(r"- name: (\w+)", manifest.read_text()))
    assert names.isdisjoint(LEGACY_NAMES)
    for spec in CREDENTIALS.values():
        if spec.opaque:
            assert spec.opaque in names
    assert "PROACTIVE_ALLOW_PLAIN_CREDENTIALS" not in manifest.read_text()
    assert "optional: true" not in manifest.read_text()


def test_decoded_value_is_stripped(monkeypatch) -> None:
    monkeypatch.setenv("KMSVP8", encode_credential("sk-key\n"))
    assert get_credential("anthropic-api-key") == "sk-key"


def test_empty_decoded_value_gives_default(monkeypatch, caplog) -> None:
    monkeypatch.setenv("KMSVP8", encode_credential(""))
    with caplog.at_level(logging.WARNING, logger="proactive_agent.credentials"):
        assert get_credential("anthropic-api-key") == ""
    assert "KMSVP8" in caplog.text


def test_settings_do_not_fill_credentials_from_plain_names(
    monkeypatch, tmp_path
) -> None:
    from proactive_agent.config import Settings

    env = tmp_path / ".env"
    env.write_text("PROACTIVE_AGENT_API_KEY=plain\nDISCORD_BOT_TOKEN=plain\n")
    monkeypatch.setenv("PROACTIVE_AGENT_API_KEY", "plain")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "plain")
    settings = Settings(_env_file=env)
    assert settings.proactive_agent_api_key == ""
    assert settings.discord_bot_token == ""
    with pytest.raises(ValueError, match="missing required settings"):
        settings.require_runtime_secrets()


def test_no_settings_field_maps_to_a_legacy_name() -> None:
    from proactive_agent.config import Settings

    fields = {name.upper() for name in Settings.model_fields}
    assert fields.isdisjoint(LEGACY_NAMES)


def test_builders_refuse_an_empty_key(monkeypatch) -> None:
    from proactive_agent import models
    from proactive_agent.credentials import MissingCredentialError

    # The SDKs would read these plain names themselves given an empty key.
    for name in LEGACY_NAMES:
        monkeypatch.setenv(name, "plain")
    monkeypatch.delenv("LITELLM_ENDPOINT", raising=False)
    for model_id in ("claude-opus-5-5", "gemini-3-flash-preview", "glm-5.3-flash"):
        with pytest.raises(MissingCredentialError):
            models.build_model(model_id)
