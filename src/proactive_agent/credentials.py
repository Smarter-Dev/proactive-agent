"""API credentials, read from opaque encoded env vars first, legacy names after.

Phase 1 of smarter-dev's key obfuscation (#482 web side, #483 here): each key
lives under a non-obvious env name and its value is encoded so a scanner does
not see a key shape. The encoding is shared byte for byte with #482:

* ``MASK`` is the 24 bytes below.
* keystream = ``sha256(MASK + i.to_bytes(4, "big"))`` for i = 0, 1, ...
  concatenated.
* encoded = ``"v1." + base64url_nopad(utf8(value) XOR keystream)``.

This is obfuscation against scanners, not encryption. Values are never logged;
only which source answered.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MASK = bytes.fromhex("f57d64f1e6341340da83b07d00003b1cb3478f4083d21eda")
PREFIX = "v1."


@dataclass(frozen=True)
class CredentialSpec:
    opaque: str  # uppercase of the smarter-dev-secrets key's opaque name; "" if none
    legacy: tuple[str, ...]


# logical name (the smarter-dev-secrets key) -> where it may be found.
CREDENTIALS: dict[str, CredentialSpec] = {
    "anthropic-api-key": CredentialSpec(
        "KMSVP8", ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
    ),
    "openai-api-key": CredentialSpec("IRBCKR", ("OPENAI_API_KEY",)),
    "gemini-api-key": CredentialSpec("IN8OID", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "openrouter-api-key": CredentialSpec(
        "RCGW3A", ("OPENROUTER_API_KEY", "OPEN_ROUTER_API_KEY", "OPEN_ROUTER")
    ),
    "brave-search-api-key": CredentialSpec("AO6HUE", ("BRAVE_SEARCH_API_KEY",)),
    "jina-api-key": CredentialSpec("IDNDWL", ("JINA_API_KEY",)),
    "discord-bot-token": CredentialSpec("FHI1OT", ("DISCORD_BOT_TOKEN",)),
    "proactive-agent-api-key": CredentialSpec("FUC9HA", ("PROACTIVE_AGENT_API_KEY",)),
    # An optional proxy key for a LiteLLM endpoint. Not in the production
    # secret and not part of #482's table, so it has no opaque name yet.
    "litellm-api-key": CredentialSpec("", ("LITELLM_API_KEY",)),
}


def _keystream(length: int) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < length:
        out += hashlib.sha256(MASK + i.to_bytes(4, "big")).digest()
        i += 1
    return bytes(out[:length])


def _xor(data: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, _keystream(len(data)), strict=True))


def encode_credential(value: str) -> str:
    raw = _xor(value.encode("utf-8"))
    return PREFIX + base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_credential(encoded: str) -> str:
    if not encoded.startswith(PREFIX):
        raise ValueError("not a v1 encoded credential")
    body = encoded[len(PREFIX) :]
    if "+" in body or "/" in body or "=" in body:
        raise ValueError("v1 body is unpadded base64url: no '+', '/' or '='")
    try:
        raw = base64.b64decode(
            body + "=" * (-len(body) % 4), altchars=b"-_", validate=True
        )
        return _xor(raw).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("malformed v1 encoded credential") from exc


def get_credential(logical: str, default: str = "") -> str:
    """The credential's value: opaque env var (decoded), then legacy names.

    Returns ``default`` (the empty string) when no source holds it. Logs the
    source's env var name, never the value.
    """
    spec = CREDENTIALS[logical]
    if spec.opaque:
        encoded = os.environ.get(spec.opaque, "").strip()
        if encoded:
            try:
                value = decode_credential(encoded)
            except ValueError:
                logger.warning(
                    "credential %s: %s is not a valid v1 value; trying legacy names",
                    logical,
                    spec.opaque,
                )
            else:
                # A trailing newline in the stored value would break auth and
                # httpx echoes a bad header value into its error text.
                value = value.strip()
                if value:
                    logger.debug("credential %s: read from %s", logical, spec.opaque)
                    return value
                logger.warning(
                    "credential %s: %s decodes to an empty value; trying legacy names",
                    logical,
                    spec.opaque,
                )
    for name in spec.legacy:
        value = os.environ.get(name, "").strip()
        if value:
            logger.debug("credential %s: read from legacy %s", logical, name)
            return value
    return default
