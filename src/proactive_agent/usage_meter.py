"""Every provider request a model run makes, for the usage ledger.

A ``UsageMeter`` wraps the model a run uses and keeps each response's own
token counts as it comes back. A run that raises after the provider answered
has still been billed for those responses, and the meter has them; a run's
``result.usage`` would be gone with the error. Each request's counts are also
what the ledger prices a long-context tier on.

``current_meter`` is the meter of the wake in flight: the capabilities the
agent's tools call (web summaries, media reads, image reviews, handler
authoring) run their own agents, and ``metered`` wraps those agents' models
with it, so their spend is billed with the wake.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel

_TOKENS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def request_dict(usage: Any) -> dict[str, int]:
    """One request's (or run's) counts; cache reads and writes are a subset
    of input."""
    return {name: int(getattr(usage, name, 0) or 0) for name in _TOKENS}


@dataclass
class UsageMeter:
    """The requests of the runs whose models it wrapped, by model id."""

    requests: dict[str, list[dict[str, int]]] = field(default_factory=dict)

    def wrap(self, model: Model | str, *, model_id: str | None = None) -> Model:
        """*model*, recording every response into this meter under
        *model_id* (by default the wire id it sends). A ``FallbackModel``'s
        models are each wrapped, so a fallback's requests stay its own."""
        # Metering never stops a run: anything that is not a model it can
        # wrap (no model set, a stand-in, a name it cannot build) goes through
        # as it is, unmetered.
        if isinstance(model, str):
            try:
                model = infer_model(model)
            except Exception:  # noqa: BLE001
                return model
        if not isinstance(model, Model):
            return model
        if isinstance(model, MeteredModel) and model.meter is self:
            return model
        if isinstance(model, FallbackModel) and model_id is None:
            fallback = copy.copy(model)
            fallback.__dict__.pop("_enter_lock", None)
            fallback.models = [self.wrap(inner) for inner in model.models]
            return fallback
        return MeteredModel(model, self, model_id or model.model_name)

    def add(self, model_id: str, tokens: dict[str, int]) -> None:
        self.requests.setdefault(model_id, []).append(tokens)

    def usage_by_model(self) -> dict[str, dict]:
        """Usage dicts by model, as ``agent.usage_dict`` gives them: totals
        and each request's counts."""
        return {
            model_id: {
                **{
                    name: sum(request[name] for request in requests) for name in _TOKENS
                },
                "requests": [dict(request) for request in requests],
            }
            for model_id, requests in self.requests.items()
            if requests
        }

    def merge(self, other: UsageMeter) -> None:
        for model_id, requests in other.requests.items():
            self.requests.setdefault(model_id, []).extend(requests)

    def clear(self) -> None:
        self.requests.clear()

    def __bool__(self) -> bool:
        return any(
            any(request.values())
            for requests in self.requests.values()
            for request in requests
        )


class MeteredModel(WrapperModel):
    """A model whose every response's usage lands in a ``UsageMeter``."""

    def __init__(self, wrapped: Model, meter: UsageMeter, model_id: str):
        super().__init__(wrapped)
        self.meter = meter
        self.metered_id = model_id

    def _record(self, response: ModelResponse | None) -> None:
        if response is None:
            return
        tokens = request_dict(response.usage)
        if any(tokens.values()):
            self.meter.add(self.metered_id, tokens)

    async def request(self, messages, model_settings, model_request_parameters):
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        self._record(response)
        return response

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ) -> AsyncIterator[Any]:
        stream = None
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
        finally:
            response = None
            if stream is not None:
                with suppress(Exception):
                    response = stream.get()
            self._record(response)


current_meter: ContextVar[UsageMeter | None] = ContextVar(
    "proactive_current_meter", default=None
)


def metered(model: Model | str) -> Model | str | None:
    """*model* wrapped with the wake's meter, or None (the agent's own model,
    unwrapped) outside a wake: pass it as ``agent.run(..., model=...)``."""
    meter = current_meter.get()
    return meter.wrap(model) if meter is not None else None


@asynccontextmanager
async def wake_meter() -> AsyncIterator[UsageMeter]:
    """A meter that every capability run inside the block bills to."""
    meter = UsageMeter()
    token = current_meter.set(meter)
    try:
        yield meter
    finally:
        current_meter.reset(token)
