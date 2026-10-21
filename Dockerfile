FROM python:3.13-slim AS runtime

# Refresh the OS store independently of the base image cache (eMudhra distrust).
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.4 /uv /uvx /usr/local/bin/
RUN python -I -c 'import ssl, hashlib; certs = ssl.create_default_context().get_ca_certs(binary_form=True); assert len(certs) > 50; assert "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b" not in {hashlib.sha256(c).hexdigest() for c in certs}'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# Keep the merged pip floor; use patched uv to download its replacement safely.
RUN uv pip install --system --upgrade "pip>=26.2.2"
RUN uv sync --frozen --no-dev
RUN .venv/bin/python -I -c 'import ssl, certifi, hashlib; ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT); ctx.load_verify_locations(certifi.where()); certs = ctx.get_ca_certs(binary_form=True); assert len(certs) > 50; assert "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b" not in {hashlib.sha256(c).hexdigest() for c in certs}'

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/app/.venv/bin/proactive-agent"]
