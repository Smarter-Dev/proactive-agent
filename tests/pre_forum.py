"""The worker as it was before forum support (#248), for mixed-version tests.

Frozen copies of ``queue.py``, ``worker.py`` and ``runtime.py`` at
proactive-agent ``8d84396c``, the last commit before forum support. They
are kept as fixtures, not read from git: CI checks out one commit, where
``main`` is already the new code and the old commit is not fetched.
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

PRE_FORUM_COMMIT = "8d84396c2fd3bd879cc7eb462e75ea63ce931591"
FIXTURES = Path(__file__).with_name("fixtures") / "pre_forum"
# Each copy's git blob id at PRE_FORUM_COMMIT (git rev-parse <commit>:<path>).
BLOB_IDS = {
    "queue": "3839d1f017df23b1c3f145716aabe08c1614f8c5",
    "worker": "4ddfea81ecd0dbd20b4280a29d54eec655a6ad73",
    "runtime": "8a4153ff5d4930ebf8366e4975d055e3adeb75fe",
}


def source(name: str) -> bytes:
    return (FIXTURES / f"{name}.py.txt").read_bytes()


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324


def load(name: str) -> types.ModuleType:
    """The pre-forum module ``name`` (queue, worker or runtime)."""
    module = types.ModuleType(f"proactive_agent._pre_forum_{name}")
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module.__name__] = module
    path = f"{PRE_FORUM_COMMIT[:8]}:src/proactive_agent/{name}.py"
    exec(compile(source(name), path, "exec"), module.__dict__)  # noqa: S102
    return module
