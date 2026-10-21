FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# Upgrade pip first, then install uv. pip >= 26.2.2, uv >= 0.13.4 and
# certifi >= 2026.10.19 drop the emSign roots whose eMudhra intermediate keys
# were compromised (GHSA-6q48-6c7w-9gw2, Oct 2026); the base image ships an
# older pip.
RUN pip install --no-cache-dir --upgrade "pip>=26.2.2" \
    && pip install --no-cache-dir "uv>=0.13.4" \
    && uv sync --frozen --no-dev

USER 65532:65532
EXPOSE 8080
ENTRYPOINT ["/app/.venv/bin/proactive-agent"]
