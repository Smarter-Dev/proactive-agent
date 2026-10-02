"""Run the real Monty sandbox through run_code and the handler lint.

The tool-registration test never executed either path, so a pydantic-monty
release that changed the ``Monty(code)`` constructor broke both in production
without failing CI.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from proactive_agent.handler_lint import compiles, lint_script
from proactive_agent.parity import run_code


def _ctx():
    discord = SimpleNamespace(send_message=AsyncMock())
    return SimpleNamespace(deps=SimpleNamespace(discord=discord, channel_id="1"))


async def test_run_code_returns_stdout_and_value():
    ctx = _ctx()
    result = await run_code(ctx, "multiply", "print(6 * 7)\n6 * 7")
    assert result == "stdout:\n42\n\nreturn value: 42"
    ctx.deps.discord.send_message.assert_awaited_once()


async def test_run_code_reports_syntax_errors():
    result = await run_code(_ctx(), "broken", "def f(:\n    pass")
    assert result.startswith("COMPILE ERROR — ")


async def test_run_code_reports_runtime_errors_with_prior_stdout():
    result = await run_code(_ctx(), "divide", "print('before')\n1 / 0")
    assert result.startswith("RUNTIME ERROR — ")
    assert "ZeroDivisionError" in result
    assert "before" in result


async def test_run_code_has_no_filesystem_or_network():
    for code in (
        "open('/etc/hostname').read()",
        "import socket\nsocket.create_connection(('example.com', 80))",
        "import os\nos.listdir('/')",
    ):
        result = await run_code(_ctx(), "probe", code)
        assert result.startswith(("COMPILE ERROR", "RUNTIME ERROR")), (code, result)


async def test_run_code_enforces_the_time_limit():
    result = await run_code(_ctx(), "spin", "while True:\n    pass")
    assert result.startswith("RUNTIME ERROR — ")


def test_lint_accepts_a_valid_handler_script():
    assert compiles("x = context\nx") is None
    assert lint_script("x = 1\nx") is None


def test_lint_rejects_a_syntax_error():
    reason = lint_script("def f(:\n    pass\nf()")
    assert reason is not None
    assert reason == compiles("def f(:\n    pass\nf()")
