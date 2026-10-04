from __future__ import annotations

from proactive_agent.errors import exception_trace


def test_the_trace_keeps_types_and_frames_and_no_message():
    try:
        try:
            raise ValueError("what someone said")
        except ValueError as cause:
            raise RuntimeError("body: what someone said") from cause
    except RuntimeError as error:
        trace = exception_trace(error)

    assert "what someone said" not in trace
    assert (
        trace.index("ValueError")
        < trace.index("direct cause")
        < trace.index("RuntimeError")
    )
    assert ", in test_the_trace_keeps_types_and_frames_and_no_message" in trace
