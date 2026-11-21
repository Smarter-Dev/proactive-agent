"""API credentials, read from opaque encoded env vars only.

Phase 2 of smarter-dev's key obfuscation (#482/#487 web side, #483/#488 here): the
plain legacy names (ANTHROPIC_API_KEY and friends) are no longer read at all.
Each key
lives under a non-obvious env name and its value is encoded so a scanner does
not see a key shape. The encoding is shared byte for byte with #482:

* ``MASK`` is the 24 bytes below.
* keystream = ``sha256(MASK + i.to_bytes(4, "big"))`` for i = 0, 1, ...
  concatenated.
* encoded = ``"v1." + base64url_nopad(utf8(value) XOR keystream)``.

This is obfuscation against scanners, not encryption. Values are never logged;
only which source answered.

Local dev and evals: set ``PROACTIVE_ALLOW_PLAIN_CREDENTIALS=1`` to accept a
plain (non-``v1.``) value under an opaque name. Production never sets it, so a
plain value there is refused.
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


DEV_PLAIN_FLAG = "PROACTIVE_ALLOW_PLAIN_CREDENTIALS"


@dataclass(frozen=True)
class CredentialSpec:
    opaque: str  # uppercase of the smarter-dev-secrets key's opaque name
    # Only for a key with no opaque name; it is never a production secret.
    legacy: tuple[str, ...] = ()


# logical name (the smarter-dev-secrets key) -> where it may be found.
CREDENTIALS: dict[str, CredentialSpec] = {
    "anthropic-api-key": CredentialSpec("KMSVP8"),
    "openai-api-key": CredentialSpec("IRBCKR"),
    "gemini-api-key": CredentialSpec("IN8OID"),
    "openrouter-api-key": CredentialSpec("RCGW3A"),
    "brave-search-api-key": CredentialSpec("AO6HUE"),
    "jina-api-key": CredentialSpec("IDNDWL"),
    "discord-bot-token": CredentialSpec("FHI1OT"),
    "proactive-agent-api-key": CredentialSpec("FUC9HA"),
    # An optional proxy key for a LiteLLM endpoint (local development only).
    # Not in the production secret and not part of #482's table, so it has no
    # opaque name and keeps its plain one.
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


def _dev_plain_allowed() -> bool:
    return os.environ.get(DEV_PLAIN_FLAG, "").strip().lower() in {"1", "true", "yes"}


def get_credential(logical: str, default: str = "") -> str:
    """The credential's value from its opaque env var (decoded).

    Returns ``default`` (the empty string) when it is unset or invalid. Logs the
    env var name, never the value.
    """
    spec = CREDENTIALS[logical]
    if spec.opaque:
        encoded = os.environ.get(spec.opaque, "").strip()
        if encoded:
            try:
                value = decode_credential(encoded)
            except ValueError:
                if encoded.startswith(PREFIX) or not _dev_plain_allowed():
                    logger.warning(
                        "credential %s: %s is not a valid v1 value; ignoring it",
                        logical,
                        spec.opaque,
                    )
                    return default
                value = encoded
            # A trailing newline in the stored value would break auth and
            # httpx echoes a bad header value into its error text.
            value = value.strip()
            if value:
                logger.debug("credential %s: read from %s", logical, spec.opaque)
                return value
            logger.warning(
                "credential %s: %s decodes to an empty value", logical, spec.opaque
            )
        return default
    for name in spec.legacy:
        value = os.environ.get(name, "").strip()
        if value:
            logger.debug("credential %s: read from %s", logical, name)
            return value
    return default
