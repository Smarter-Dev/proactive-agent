"""The shared privacy:v1 fold plausibility vectors and the common-word list."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proactive_agent.agent import fold_plausibility_problem

ROOT = Path(__file__).parents[1]
VECTORS = json.loads(
    (ROOT / "contracts/privacy/v1/fold_plausibility_vectors.json").read_text("utf-8")
)


@pytest.mark.parametrize("case", VECTORS, ids=[c["name"] for c in VECTORS])
def test_fold_plausibility_vectors(case):
    result = fold_plausibility_problem(
        case["input_texts"], case["user_id"], case["names"], case["output"]
    )
    if case["accept"]:
        assert result is None, str(result)
    else:
        assert result is not None
        assert result.category == case["category"], str(result)


def test_packaged_common_words_match_the_contract_copy():
    contract = (ROOT / "contracts/privacy/v1/common_words.txt").read_bytes()
    packaged = (ROOT / "src/proactive_agent/privacy_data/common_words.txt").read_bytes()
    assert packaged == contract
