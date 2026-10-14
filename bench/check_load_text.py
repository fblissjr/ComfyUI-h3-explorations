#!/usr/bin/env python3
"""The generated text of a load (`bench/load_text.py`) against the kept loads' own texts, and what it refuses.

1. **The kept texts, made again.** From each kept load's reference record (what was drawn), the plan's words for the
   subject and the video, the voice table and the load's place-and-action sentence, the generator makes lead_fp_1730's
   text and second_hfp_0484c's word for word, but in the two places ruled on 2026-10-14: the closer view is "of the head
   and shoulders", and the summary takes the place-and-action sentence as written.
2. **What a reference draws, the text says.** The mouth drawn, the eye in contrast or filled, a part left out, and a
   plain cut each change the `<Subject 2>` sentence and nothing else; the singing line follows the subject's `sings` and
   the voice table.
3. **Refused**: a plan with no words for the subject or no film line, a face-and-hair subject with no hair words, an
   empty sentence, and an iris form it has no words for.

    <python> bench/check_load_text.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import load_text as lt  # noqa: E402

KEPT_1730 = "subject_definitions:\n<Subject 1> is the girl whose appearance comes from <Picture 1>, with brown eyes, dark eyebrows and fair skin, and who follows the movements of the head and the facial expressions of the figure in <Video 1>.\n<Subject 2> is the movements of the head and the facial expressions performed by the figure in <Video 1>: a closer view of the head, shoulders and arms of a plain grey body model of a person on a black ground, posed as the person moves, with its eyes, eyebrows and nose drawn on its face as thin white outlines; the eyes are outlined with the iris inside them, and they open, close and look where the person's eyes do.\n\nsummary:\n[reference generation] <Subject 1> holds a black telephone receiver to her ear in a room, with the face of <Picture 1>, and performs <Subject 2>.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): partially_preserved - only the girl's face is retained, wherever it shows: her eyes, eyebrows, nose, mouth and skin; the hair, the clothing and the background of <Picture 1> are not used.\n<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head and the facial expressions are transferred to <Subject 1>; the grey body model's look, the drawn outlines themselves and its black ground are not used.\n\ndetailed_description:\nThe target video is live-action vintage film from the early 1980s, soft and a little grainy, in warm amber tungsten light.\n[Shot 1] <Subject 1>, the girl with brown eyes, dark eyebrows and fair skin, is in a room and holds a black telephone receiver to her ear. Her face is the face of <Picture 1>. <Subject 1> performs <Subject 2>: she follows the movements of the head and the facial expressions of the figure. <Subject 1> (S1) sings, her mouth opening wide on the sung phrases.\n\noverall_soundscape: Natural ambient atmosphere continues throughout the shot.\n\nnon_diegetic_music: A song plays throughout.\n"
KEPT_0484C = "subject_definitions:\n<Subject 1> is the woman whose appearance comes from <Picture 1>, with long, loose blonde hair parted in the middle, fair skin and blue-grey eyes, and who follows the movements of the head and the facial expressions of the figure in <Video 1>.\n<Subject 2> is the movements of the head and the facial expressions performed by the figure in <Video 1>: a closer view of the head, shoulders and hands of a plain grey body model of a person on a black ground, posed as the person moves, with its eyes, eyebrows and nose drawn on its face as thin white outlines; the eyes are outlined with the iris inside them, and they open, close and look where the person's eyes do.\n\nsummary:\n[reference generation] <Subject 1> sits at a kitchen table and cracks eggs into a bowl, with the face and the hair of <Picture 1>, and performs <Subject 2>.\n\nretention_analysis:\n<Subject 1> (appears in [Shot 1]): partially_preserved - only the woman's face and her long, loose blonde hair are retained, wherever they show; the clothing, the accessories and the background of <Picture 1> are not used.\n<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head and the facial expressions are transferred to <Subject 1>; the grey body model's look, the drawn outlines themselves and its black ground are not used.\n\ndetailed_description:\nThe target video is live-action vintage film from the early 1980s, soft and a little grainy, in warm amber tungsten light.\n[Shot 1] <Subject 1>, the woman with long, loose blonde hair parted in the middle, fair skin and blue-grey eyes, sits at a kitchen table and cracks eggs into a bowl in front of her. Her face is the face of <Picture 1>. <Subject 1> performs <Subject 2>: she follows the movements of the head and the facial expressions of the figure.\n\noverall_soundscape: Natural ambient atmosphere continues throughout the shot.\n\nnon_diegetic_music: A song plays throughout.\n"
FILM = "The target video is live-action vintage film from the early 1980s, soft and a little grainy, in warm amber tungsten light."
PLAN = {"text": {"film": FILM}, "subjects": {
    "lead": {"region": "face", "text": {"who": "the girl", "looks": "brown eyes, dark eyebrows and fair skin", "her": "her", "she": "she", "sings": True}},
    "second": {"region": "face and hair", "text": {"who": "the woman", "looks": "long, loose blonde hair parted in the middle, fair skin and blue-grey eyes",
                                                   "her": "her", "she": "she", "hair": "her long, loose blonde hair"}}}}
RECORD_10_13 = {"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": None}}          # a reference of 10/13: the whole ring, no mouth
VOICED = [[579, 603], [1549, 1968]]


def as_ruled(kept: str, view: str, summary_was: str, summary_now: str) -> str:
    """A kept text with the two places ruled on 2026-10-14: the closer view, and the summary's sentence as written."""
    out = kept.replace(f"a closer view of the head, shoulders and {view} of", "a closer view of the head and shoulders of")
    return out.replace(f"<Subject 1> {summary_was}, with", f"<Subject 1> {summary_now}, with")


def the_kept_texts_made_again() -> str:
    lead = lt.text(PLAN, "lead", RECORD_10_13, "is in a room and holds a black telephone receiver to her ear", VOICED, 1730, 1754)
    want = as_ruled(KEPT_1730, "arms", "holds a black telephone receiver to her ear in a room", "is in a room and holds a black telephone receiver to her ear")
    # THE CASE THAT GOES RED when the generator departs from the kept loads anywhere but the two ruled places
    assert lead == want, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want.splitlines(), lead.splitlines()) if a != b)
    second = lt.text(PLAN, "second", RECORD_10_13, "sits at a kitchen table and cracks eggs into a bowl in front of her", VOICED, 484, 515)
    want = as_ruled(KEPT_0484C, "hands", "sits at a kitchen table and cracks eggs into a bowl", "sits at a kitchen table and cracks eggs into a bowl in front of her")
    assert second == want, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want.splitlines(), second.splitlines()) if a != b)
    return "lead_fp_1730's and second_hfp_0484c's texts made again from their records, word for word but the closer view and the summary's sentence, as ruled"


def what_a_reference_draws_the_text_says() -> str:
    base = lt.text(PLAN, "lead", RECORD_10_13, "stands", VOICED, 1730, 1754)
    def s2(record):
        return next(line for line in lt.text(PLAN, "lead", record, "stands", VOICED, 1730, 1754).splitlines() if line.startswith("<Subject 2> is"))
    drawn = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}})
    assert "with its eyes, eyebrows, nose and mouth drawn on its face" in drawn and "the white of the eye is white and the iris black" in drawn \
        and "the mouth is outlined, and it opens and closes where the person's mouth does" in drawn, drawn
    left = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": False}}})
    assert "eyes, eyebrows and nose drawn" in left and "mouth" not in left, left
    assert "a dark iris inside them" in s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "filled"}})
    no_brows = s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth", "brows"], "iris": "between the lids"}})
    assert "with its eyes and nose drawn" in no_brows, no_brows
    plain = s2({"form": "plain", "drawn": {"parts_left_out": []}})
    assert plain.endswith("posed as the person moves.") and "outline" not in plain, plain
    others = lambda t: [line for line in t.splitlines() if not line.startswith("<Subject 2> is")]       # noqa: E731
    assert others(lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}, "stands", VOICED, 1730, 1754)) == others(base)
    assert "(S1) sings" in base and "(S1) sings" not in lt.text(PLAN, "lead", RECORD_10_13, "stands", VOICED, 1000, 1100), "the singing line ignores the voice table"
    return "the mouth drawn or not, the eye's form, a part left out and a plain cut change the <Subject 2> sentence and nothing else; singing follows the voice table"


def what_it_refuses() -> str:
    import copy
    for change, words in ((lambda p: p["subjects"]["lead"]["text"].pop("looks"), "lacks ['looks']"), (lambda p: p.pop("text"), "text.film"),
                          (lambda p: p["subjects"]["second"]["text"].pop("hair"), "`text.hair`")):
        plan = copy.deepcopy(PLAN)
        change(plan)
        who = "second" if "hair" in words else "lead"
        try:
            lt.text(plan, who, RECORD_10_13, "stands", VOICED, 1, 2)
        except lt.Refused as e:
            assert words in str(e), (words, e)
        else:
            raise AssertionError(f"not refused: {words}")
    for record, sentence, words in ((RECORD_10_13, "  ", "is empty"), ({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "glowing"}}, "x", "no words for it")):
        try:
            lt.text(PLAN, "lead", record, sentence, VOICED, 1, 2)
        except lt.Refused as e:
            assert words in str(e), (words, e)
        else:
            raise AssertionError(f"not refused: {words}")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "plan.json").write_text(json.dumps(PLAN))
        (d / "spans.json").write_text(json.dumps({"voiced_spans_inclusive": VOICED}))
        (d / "job.json").write_text(json.dumps({"job": {}, "loads": [{"subject": "lead", "first": 1730, "frames": 25, "motion_file": str(d / "ref.mp4")}]}))
        try:
            lt.for_job(d / "job.json", "stands", d / "plan.json", d / "spans.json")
        except lt.Refused as e:
            assert "is made after it" in str(e), e
        else:
            raise AssertionError("a text was made before its reference")
        (d / "ref.json").write_text(json.dumps(RECORD_10_13))
        words, made_from = lt.for_job(d / "job.json", "stands", d / "plan.json", d / "spans.json")
        assert words.startswith("subject_definitions:") and made_from["drawn"]["parts_left_out"] == ["mouth"], made_from
    return "no words for the subject or no film line, no hair words for a face-and-hair subject, an empty sentence, an unknown iris form and a reference not yet made are refused"


case("the kept texts made again", the_kept_texts_made_again)
case("what a reference draws, the text says", what_a_reference_draws_the_text_says)
case("what it refuses", what_it_refuses)
sys.exit(finish())
