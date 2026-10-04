"""The shared privacy:v1 name matcher."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import pytest

from proactive_agent.agent import (
    checked_names,
    name_hits,
    unchecked_name_list,
    unchecked_names,
)


@pytest.mark.parametrize(
    ("text", "name", "hit"),
    [
        (unicodedata.normalize("NFD", "ask José today"), "José", True),
        ("ask José today", unicodedata.normalize("NFD", "José"), True),
        ("今天王来了吗", "王", True),
        ("ping 🦀 please", "🦀", True),
        ("thanks alice_dev", "alice", True),
        ("alice2 says hi", "alice", True),
        ("2alice says hi", "alice", True),
        ("malice everywhere", "alice", False),
        ("alicea is new", "alice", False),
        ("ask Kai The Rustacean now", "kai the rustacean", True),
        ("ask kai the  rustacean now", "kai the rustacean", False),
        ("ALICE shouted", "alice", True),
        ("ALICE shouted", "  Alice  ", True),
    ],
)
def test_name_matcher_vectors(text, name, hit):
    assert bool(name_hits(text, [name])) is hit


def test_short_ascii_names_are_unchecked_but_others_are_not():
    names = ["k", "", "  ", "王", "kai", "é"]
    assert checked_names(names) == ["王", "kai", "é"]
    assert unchecked_names(names) == 1
    assert name_hits("k is here", ["k"]) == []


VECTORS = json.loads(
    (
        Path(__file__).parents[1]
        / "contracts"
        / "privacy"
        / "v1"
        / "name_matcher_vectors.json"
    ).read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case", VECTORS, ids=[f"{i}:{c['text'][:24]}" for i, c in enumerate(VECTORS)]
)
def test_shared_privacy_v1_vectors(case):
    """The vectors smarter-dev publishes (byte-identical copy) all agree."""
    assert bool(name_hits(case["text"], case["names"])) is case["hit"]
    assert unchecked_name_list(case["names"]) == case["unchecked"]
