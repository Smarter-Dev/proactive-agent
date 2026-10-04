"""Worker acks against the shared Ack v1 schema (byte-identical copy)."""

from __future__ import annotations

import hashlib

import httpx
import pytest
from jsonschema import ValidationError
from purge_fakes import ACK_SCHEMA_PATH, ACK_VALIDATOR, GUILD
from test_purge import Replica, command, honest_model, submit

from proactive_agent.api import ApplicationAPI

# sha256 of smarter-dev's contracts/privacy/v1/purge_ack.schema.json
# (build/privacy-agent-purge 70d895b2). Update only by copying that file.
SMARTER_DEV_SHA256 = "00879077dd732d6babdf8658d7b15269b962049576622d7ca9abd67d7b10d0c3"


def test_schema_is_the_byte_identical_smarter_dev_copy():
    digest = hashlib.sha256(ACK_SCHEMA_PATH.read_bytes()).hexdigest()
    assert digest == SMARTER_DEV_SHA256


async def test_the_api_client_sends_a_schema_valid_body():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"accepted": True})

    api = ApplicationAPI(
        base_url="https://app.test/api",
        api_key="k",
        transport=httpx.MockTransport(handler),
    )
    try:
        await api.post_privacy_ack(
            "run-1",
            component="worker",
            guild_id=GUILD,
            outcome="purged",
            stores=["proactive:v1:history"],
            detail="x" * 600,
            name_hits={"history": 1, "watch": 0},
            tombstoned=False,
            unchecked_names=2,
            done_record="written",
        )
    finally:
        await api.close()

    ACK_VALIDATOR.validate(bodies[0])
    assert set(bodies[0]) == {
        "component",
        "guild_id",
        "outcome",
        "stores",
        "detail",
        "name_hits",
        "tombstoned",
        "unchecked_names",
        "done_record",
    }


def test_the_schema_rejects_what_the_worker_must_never_send():
    with pytest.raises(ValidationError):
        ACK_VALIDATOR.validate(
            {"component": "worker", "guild_id": GUILD, "outcome": "ok"}
        )
    with pytest.raises(ValidationError):
        ACK_VALIDATOR.validate(
            {
                "component": "worker",
                "guild_id": GUILD,
                "outcome": "purged",
                "name_hits": {"History": 1},
            }
        )


async def test_every_ack_the_fake_records_was_schema_validated(redis_client, world):
    api = world
    replica = Replica(redis_client, api, honest_model([]), name="a")
    await replica.consumer.initialize()
    await submit(redis_client, command())

    await replica.consumer.poll_once()

    assert len(api.validated_acks) == len(api.acks) == 1
    await replica.writer.close(timeout=1)


@pytest.fixture
def redis_client():
    import fakeredis.aioredis

    return fakeredis.aioredis.FakeRedis(decode_responses=False)
