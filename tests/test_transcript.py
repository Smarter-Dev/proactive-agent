from __future__ import annotations

from proactive_agent.transcript import render_transcript_line, speaker_tags


def record(**changes) -> dict:
    base = {
        "id": "10",
        "author_id": "222222222222222222",
        "author_display": "nia",
        "is_bot": False,
        "content": "ship it",
        "reply_to_id": None,
    }
    return {**base, **changes}


def test_line_carries_the_author_id():
    line = record()
    assert (
        render_transcript_line(line, speaker_tags([line]))
        == "[id=10] A·nia (uid=222222222222222222): ship it"
    )


def test_reply_and_bot_markers_keep_their_places():
    line = record(is_bot=True, reply_to_id="9")
    assert (
        render_transcript_line(line, speaker_tags([line]))
        == "[id=10] [BOT] A·nia (uid=222222222222222222) (reply to id=9): ship it"
    )
