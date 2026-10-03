"""PurgeCommand and its JSON Schema copy must agree."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from proactive_agent.contracts import PurgeCommand

SCHEMA = json.loads(
    (
        Path(__file__).parents[1]
        / "contracts"
        / "privacy"
        / "v1"
        / "purge_command.schema.json"
    ).read_text()
)

GOLDEN = {
    "schema_version": 1,
    "request_id": "6f1c1f0e-8a52-4f55-9a5e-3c7f5d8a1b20",
    "run_id": "0b6c0f6e-1d4a-4f2e-9a1c-2f6d8b7e9c31",
    "user_id": "111111111111111111",
    "names": ["kai", "Kai the Rustacean"],
    "guild_ids": ["333333333333333333", "444444444444444444"],
    "created_at": "2026-10-03T12:00:00Z",
}


def schema_accepts(payload: dict) -> bool:
    return Draft202012Validator(SCHEMA).is_valid(payload)


def model_accepts(payload: dict) -> bool:
    try:
        PurgeCommand.model_validate_json(json.dumps(payload))
    except ValidationError:
        return False
    return True


def variant(**changes) -> dict:
    payload = copy.deepcopy(GOLDEN)
    for key, value in changes.items():
        if value is ...:
            payload.pop(key)
        else:
            payload[key] = value
    return payload


def test_golden_payload_is_accepted_by_both():
    assert schema_accepts(GOLDEN)
    command = PurgeCommand.model_validate_json(json.dumps(GOLDEN))
    assert command.user_id == "111111111111111111"
    assert command.guild_ids == ["333333333333333333", "444444444444444444"]


@pytest.mark.parametrize(
    "payload",
    [
        variant(extra="field"),
        variant(schema_version=2),
        variant(schema_version=0),
        variant(user_id="1234"),
        variant(user_id="12345678901234567890123"),
        variant(user_id="11111111111111111a"),
        variant(names=["x"] * 21),
        variant(names=[""]),
        variant(names=["n" * 101]),
        variant(guild_ids=[]),
        variant(guild_ids=["333333333333333333"] * 501),
        variant(guild_ids=["abc"]),
        variant(names=...),
        variant(guild_ids=...),
        variant(run_id=...),
    ],
)
def test_both_reject_the_same_payloads(payload):
    assert not schema_accepts(payload)
    assert not model_accepts(payload)


def test_both_accept_empty_names_and_limits():
    for payload in (
        variant(names=[]),
        variant(names=["n" * 100] * 20),
        variant(guild_ids=["1" * 15] * 500),
        variant(user_id="9" * 22),
    ):
        assert schema_accepts(payload)
        assert model_accepts(payload)


def test_model_rejects_a_naive_timestamp():
    assert not model_accepts(variant(created_at="2026-10-03T12:00:00"))


def test_repr_never_carries_the_target():
    command = PurgeCommand.model_validate(GOLDEN)
    text = f"{command!r} {command}"
    assert "111111111111111111" not in text
    assert "kai" not in text.lower()
