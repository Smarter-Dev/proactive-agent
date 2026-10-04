"""Describe an exception without any of its text.

A provider error can quote the prompt, and the prompt holds members' messages
from every channel the agent read. Logs and Discord notices therefore carry an
exception's types and stack frames, never its message.
"""

from __future__ import annotations

import traceback

_CAUSE = "\nThe above exception was the direct cause of the following exception:\n"
_CONTEXT = "\nDuring handling of the above exception, another exception occurred:\n"


def exception_type_name(error: BaseException) -> str:
    """``module.Qualname`` of an exception, bare for builtins."""
    cls = type(error)
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def exception_trace(error: BaseException) -> str:
    """Types and frames of ``error`` and every exception chained to it.

    Built from the exception objects (file, line and function of each frame),
    never from their messages; no source lines either.
    """
    chain: list[tuple[BaseException, str]] = []
    seen: set[int] = set()
    current: BaseException | None = error
    separator = ""
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append((current, separator))
        if current.__cause__ is not None:
            current, separator = current.__cause__, _CAUSE
        elif current.__context__ is not None and not current.__suppress_context__:
            current, separator = current.__context__, _CONTEXT
        else:
            current = None
    lines: list[str] = []
    for index in range(len(chain) - 1, -1, -1):
        exception, _ = chain[index]
        if exception.__traceback__ is not None:
            lines.append("Traceback (most recent call last):")
            lines.extend(
                f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}'
                for frame in traceback.extract_tb(exception.__traceback__)
            )
        lines.append(exception_type_name(exception))
        if index > 0:
            lines.append(chain[index][1])
    return "\n".join(lines)
