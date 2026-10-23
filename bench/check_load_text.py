#!/usr/bin/env python3
"""The generated text of a load (`bench/load_text.py`) against the owner's text and the kept loads' own texts, and what it refuses.

0. **The owner's text of 2026-10-22, made again.** From second_hfp_0415l's reference record, the plan's words (with
   `picture_hair` and `hair_words`) and the voice table, copied into `bench/fixtures/load_text_second_0415l.json`, the generator
   makes the text the owner wrote by hand for that load, byte for byte but for the changes the owner approved since, each kept in
   the fixture with its old and new words (`approved_changes`, 2026-10-22: the eyes each with a dark iris, and where a face is drawn
   the eyes' look transferred and followed, and the mouth's opening only where the drawing opens a mouth, which 0415's does not;
   2026-10-23, the owner's word: her shortened hair gone, and the face and the picture's hair in the summary and the retention line,
   the face and the hair in [Shot 1]). The picture's hair goes in the definition and the shot's
   in [Shot 1], each in its own line; a subject without either is refused in the tool's own words; no
   text carries the plan's `looks`; the music is "N/A".
1. **The kept texts, made again.** From each kept load's reference record (what was drawn), the plan's words for the
   subject and the video and the voice table (no place-and-action sentence since 2026-10-15), the generator makes lead_fp_1730's
   text and second_hfp_0484c's word for word, but in the places ruled since: the closer view is "of the head and shoulders"
   (2026-10-14), the summary and [Shot 1] are in draft_7's form with no sentence of the place (2026-10-15), no "not used"
   clause (2026-10-21), and the hair words, the eye movements and the music of 2026-10-22 (no mouth words: their records draw no mouth),
   and the face-and-hair wording of 2026-10-23.
2. **What a reference draws, the text says.** The mouth drawn, the eye in contrast or filled, a part left out, and a
   plain cut each change the `<Subject 2>` sentence and nothing else, but an open mouth, which also names the mouth in the retention
   line and [Shot 1]; the singing line follows the subject's `sings` and the voice table.
3. **The mouth's words where the drawing opens one** (the owner's go of 2026-10-22, after second_hfp_0278l): a reference that opens
   no mouth on any frame names no mouth in any line; one that does names it in `<Subject 2>`'s sentence, its retention line and [Shot 1].
4. **Refused**: a plan with no words for the subject or no film line, a face-and-hair subject with no hair words, an
   place-and-action sentence (NO_SCENE), an iris form it has no words for, and a plan that still carries `text.hair_change`
   (retired on the owner's word of 2026-10-23).

    <python> bench/check_load_text.py
"""
from __future__ import annotations

import json
import re
import subprocess
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
    "lead": {"region": "face", "text": {"who": "the girl", "looks": "brown eyes, dark eyebrows and fair skin", "picture_hair": "LEAD PICTURE HAIR",
                                        "hair_words": "LEAD HAIR",
                                        "her": "her", "she": "she", "sings": True}},
    "second": {"region": "face and hair", "text": {"who": "the woman", "looks": "long, loose blonde hair parted in the middle, fair skin and blue-grey eyes",
                                                   "picture_hair": "SECOND PICTURE HAIR", "hair_words": "SECOND HAIR", "her": "her", "she": "she", "hair": "her long, loose blonde hair"}}}}
RECORD_10_13 = {"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": None}}          # a reference of 10/13: the whole ring, no mouth
VOICED = [[579, 603], [1549, 1968]]


def as_ruled(kept: str, view: str, summary_was: str, who_looks: str, detail_was: str, looks: str, who_hair: str, picture_hair: str,
             has: str = "the face") -> str:
    """A kept text with what was ruled since: the closer view (2026-10-14), no place-and-action sentence (2026-10-15, NO_SCENE):
    the summary and [Shot 1] say who the subject is and what of the still it has (draft_7's form), no "not used" clause
    (the owner, 2026-10-21): each retention line ends at what is kept or changed; and the owner's text of 2026-10-22: the
    subject named by the picture's hair in the definition (`picture_hair`) and by the shot's in [Shot 1] (`who_hair`) in place of her
    looks, the music "N/A", and the face alone retained as "the girl's face from <Picture 1>"; and the owner's approval of the same day:
    where a face is drawn, the eyes' look transferred and followed, and the mouth's opening only where the drawing opens a mouth (the
    owner's go of 2026-10-22, after second_hfp_0278l); and the owner's word of 2026-10-23: a face-and-hair subject has "the face and the
    <picture_hair> of <Picture 1>" in the summary and the retention line, and [Shot 1] `has` "the face and the hair of <Picture 1>" (the
    name already says the hair). The kept loads' references of 10/13 drew no mouth (`RECORD_10_13`, as their own `<Subject 2>`
    sentences say), so their texts name none."""
    out = re.sub(r"; [^;\n]* of <Picture 1> are not used\.\n", ".\n", kept)
    out = out.replace("; the grey body model's look, the drawn outlines themselves and its black ground are not used.", ".")
    out = out.replace(f"a closer view of the head, shoulders and {view} of", "a closer view of the head and shoulders of")
    out = out.replace(f"<Subject 1> {summary_was}, with", "<Subject 1>, with").replace(", and performs <Subject 2>.\n", ", performs <Subject 2>.\n")
    out = out.replace("<Subject 1>, with the face and the hair of <Picture 1>, performs", f"<Subject 1>, with the face and the {picture_hair} of <Picture 1>, performs")
    out = out.replace(f"[Shot 1] <Subject 1>, {who_looks}, {detail_was}. Her face is the face of <Picture 1>.",
                      f"[Shot 1] <Subject 1>, {who_hair}, has {has} of <Picture 1>.")
    who = who_hair.split(" with ")[0]
    out = out.replace(f"<Subject 1> is {who} whose appearance comes from <Picture 1>, with {looks}, and who",
                      f"<Subject 1> is {who} with {picture_hair} whose appearance comes from <Picture 1>, and who")
    out = out.replace("attribute_transfer - the movements of the head and the facial expressions are transferred to <Subject 1>.",
                      "attribute_transfer - the movements of the head, the facial expressions and where the eyes look are transferred to <Subject 1>.")
    out = re.sub(r"performs <Subject 2>: (\w+) follows the movements of the head and the facial expressions of the figure\.",
                 lambda m: (f"performs <Subject 2>: {m[1]} follows the movements of the head and the facial expressions of the figure, "
                            f"{dict(she='her', he='his').get(m[1], 'their')} eyes looking where its eyes look."), out)
    out = re.sub(r"partially_preserved - only the (\w+)'s face is retained, wherever it shows: \w+ eyes, eyebrows, nose, mouth and skin\.\n",
                 r"partially_preserved - the \1's face from <Picture 1> is retained.\n", out)
    out = re.sub(r"partially_preserved - only the (\w+)'s face and [^\n]* are retained, wherever they show\.\n",
                 lambda m: f"partially_preserved - the face and the {picture_hair} of <Picture 1> are retained.\n", out)
    return out.replace("non_diegetic_music: A song plays throughout.\n", "non_diegetic_music: N/A\n")


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "load_text_second_0415l.json"


def the_owners_text_made_again() -> str:
    """The owner's word of 2026-10-22 on second_hfp_0415l's text, written by hand from the generator's 0415j: "use that prompt from
    now on. It's simpler and more flexible"; "just saying woman with blonde hair, or girl with X hair is usually enough"."""
    fx = json.loads(FIXTURE.read_text())
    got = lt.text(fx["plan"], fx["subject"], fx["record"], "", fx["voiced_spans_inclusive"], fx["first"], fx["last"])
    want = fx["owner_text"]
    # THE OWNER'S APPROVED CHANGES to that text since (the fixture's `approved_changes`; 2026-10-22, the closed-lip line's eye words), each
    # found once in the owner's text as written: the rest of it stays byte for byte
    for change in fx.get("approved_changes") or []:
        assert want.count(change["old"]) == 1, change["what"]
        want = want.replace(change["old"], change["new"])
    # THE CASE THAT GOES RED when the generator departs from the owner's text by one byte
    assert got == want, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want.splitlines(), got.splitlines()) if a != b) or (got, want)
    # NO LOOKS WORDS: the plan's `looks` is read by no line, on any form (face, face and hair, a changed hair, hair alone, no video)
    faceless = {"form": "drawn", "drawn": {"parts_left_out": ["eyes", "brows", "nose", "mouth"], "iris": "contrast", "mouth": {"drawn": False}}}
    made = [got, lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1730, 1754), lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515),
            lt.text(PLAN, "second", faceless, "", VOICED, 484, 515, parts=["hair"]), lt.text(PLAN, "second", None, "", VOICED, 484, 515, parts=["hair"])]
    looks = [fx["plan"]["subjects"]["second"]["text"]["looks"], PLAN["subjects"]["lead"]["text"]["looks"], PLAN["subjects"]["second"]["text"]["looks"]]
    for words in made:
        # THE CASE THAT GOES RED when any text carries the plan's looks, whole or by its eye, eyebrow or skin words
        assert not any(x in words for x in looks + ["blue eyes", "fair skin", "brown eyes", "dark eyebrows", "blue-grey eyes"]), words
        assert words.endswith("\n\nnon_diegetic_music: N/A\n") and "A song plays" not in words, words[-80:]
    # WHERE THE EYES LOOK on a load whose reference draws a face (the owner's approval of 2026-10-22), and no mouth where the drawing opens
    # none (0415's closed-lip line opens none, the 10/13 records leave the mouth out; the owner's go of 2026-10-22); the head's movements
    # alone on her hair alone
    for words in made[:3]:
        assert "<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head, the facial expressions and where the eyes look are transferred to <Subject 1>.\n" in words, words
        assert "<Subject 1> performs <Subject 2>: she follows the movements of the head and the facial expressions of the figure, her eyes looking " \
               "where its eyes look." in words and "mouth" not in words.split("(S1) sings")[0], words
    assert "where the eyes look" not in made[3] and "attribute_transfer - the movements of the head are transferred to <Subject 1>.\n" in made[3], made[3]
    # A FACE LOAD WHOSE PICTURE'S HAIR AND SHOT'S HAIR DIFFER (the owner, 2026-10-22: the hair named with <Picture 1> is the picture's,
    # [Shot 1] names the hair the shot ends up with): each in its own line, and only there
    lead = made[1].splitlines()
    definition = next(line for line in lead if line.startswith("<Subject 1> is"))
    shot = next(line for line in lead if line.startswith("[Shot 1]"))
    # THE CASE THAT GOES RED when one hair stands in for the other, or either is named in the other's line
    assert definition.startswith("<Subject 1> is the girl with LEAD PICTURE HAIR whose appearance comes from <Picture 1>, and who follows"), definition
    assert shot.startswith("[Shot 1] <Subject 1>, the girl with LEAD HAIR, has the face of <Picture 1>."), shot
    assert [line for line in lead if "LEAD PICTURE HAIR" in line] == [definition] and [line for line in lead if "LEAD HAIR" in line] == [shot], lead
    # THE FACE ALONE RETAINED in the face-and-hair form's words (the owner, 2026-10-22): "the girl's face from <Picture 1> is retained."
    retained = [line for line in lead if line.startswith("<Subject 1> (appears in [Shot 1]):")]
    # THE CASE THAT GOES RED when a face load's retention line lists her features, or says "only" or "wherever it shows"
    assert retained == ["<Subject 1> (appears in [Shot 1]): partially_preserved - the girl's face from <Picture 1> is retained."], retained
    # A SUBJECT WITHOUT `picture_hair` OR `hair_words` IS REFUSED, by name, in the tool's own words and with no traceback; never one
    # filled from the other, nor from `looks`
    import copy
    for key, bad in [(k, b) for k in ("picture_hair", "hair_words") for b in (None, "", "   ", ["blonde hair"])]:
        plan = copy.deepcopy(fx["plan"])
        if bad is None:
            plan["subjects"]["second"]["text"].pop(key)
        else:
            plan["subjects"]["second"]["text"][key] = bad
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "plan.json").write_text(json.dumps(plan))
            (d / "spans.json").write_text(json.dumps({"voiced_spans_inclusive": fx["voiced_spans_inclusive"]}))
            (d / "ref.json").write_text(json.dumps(fx["record"]))
            (d / "job.json").write_text(json.dumps({"job": {}, "loads": [{"subject": "second", "first": 415, "frames": 23, "motion_file": str(d / "ref.mp4")}]}))
            run = subprocess.run([sys.executable, str(Path(lt.__file__)), "show", str(d / "job.json"), "--plan", str(d / "plan.json"),
                                  "--voice-spans", str(d / "spans.json")], capture_output=True, text=True)
        # THE CASE THAT GOES RED when a subject with no hair words gets a text, or is refused by an error on the way
        assert run.returncode == 5 and run.stdout.startswith(f"REFUSED: the plan's `text.{key}` for 'second' is not written") \
            and "Traceback" not in run.stdout + run.stderr, (key, bad, run.returncode, run.stdout, run.stderr)
    return ("second_hfp_0415l's text made again byte for byte from its inputs; no text carries the plan's looks; the music is N/A; a face load "
            "transfers where the eyes look (no mouth where the drawing opens none), a hair load the head's movements alone; the picture's hair in the definition and the shot's in [Shot 1], "
            "each in its own line; a subject without picture_hair or hair_words refused in the tool's words")


def the_kept_texts_made_again() -> str:
    lead = lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1730, 1754)
    want = as_ruled(KEPT_1730, "arms", "holds a black telephone receiver to her ear in a room", "the girl with brown eyes, dark eyebrows and fair skin",
                    "is in a room and holds a black telephone receiver to her ear", "brown eyes, dark eyebrows and fair skin", "the girl with LEAD HAIR",
                    "LEAD PICTURE HAIR")
    # THE CASE THAT GOES RED when the generator departs from the kept loads anywhere but the two ruled places
    assert lead == want, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want.splitlines(), lead.splitlines()) if a != b)
    second = lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515)
    want = as_ruled(KEPT_0484C, "hands", "sits at a kitchen table and cracks eggs into a bowl",
                    "the woman with long, loose blonde hair parted in the middle, fair skin and blue-grey eyes",
                    "sits at a kitchen table and cracks eggs into a bowl in front of her",
                    "long, loose blonde hair parted in the middle, fair skin and blue-grey eyes", "the woman with SECOND HAIR",
                    "SECOND PICTURE HAIR", "the face and the hair")
    assert second == want, "\n".join(f"- {a}\n+ {b}" for a, b in zip(want.splitlines(), second.splitlines()) if a != b)
    return ("lead_fp_1730's and second_hfp_0484c's texts made again from their records, word for word but the places ruled since (the closer "
            "view; no place-and-action sentence, draft_7's form; no \"not used\" clause; hair words, eye movements, N/A; the face-and-hair "
            "wording of 2026-10-23)")


def what_a_reference_draws_the_text_says() -> str:
    base = lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1730, 1754)
    def s2(record):
        return next(line for line in lt.text(PLAN, "lead", record, "", VOICED, 1730, 1754).splitlines() if line.startswith("<Subject 2> is"))
    drawn = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}})
    assert "with its eyes, eyebrows, nose and mouth drawn on its face" in drawn and "the white of the eye is white and the iris black" in drawn \
        and "the mouth is outlined, and it opens and closes where the person's mouth does" in drawn, drawn
    from_map = s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "contrast", "mouth": {"drawn": True, "drawn_as": "the class map", "inside": "the class map"}}})
    assert "nose and mouth drawn" in from_map and "the mouth is outlined, dark inside where it is open, and it opens and closes" in from_map, from_map
    assert "dark inside" not in drawn, drawn
    want_unified = ("<Subject 2> is the movements of the head and the facial expressions performed by the figure in <Video 1>: "
                    "a closer view of the head and shoulders of a plain grey body model of a person on a black ground, posed as the person moves, "
                    "with its face drawn as thin white outlines: the eyes, each with a dark iris, open, close and look where the person's eyes do.")
    closed_line_all = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast",
                                                     "mouth": {"drawn": True, "inside": "the class map", "closed_lip_line": {"why": "test", "closed_frames": [], "open_frames": [[1730, 1754]]}}}})
    closed_line_some = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast",
                                                      "mouth": {"drawn": True, "inside": "the class map", "frames_with_no_mouth": 10,
                                                                "closed_lip_line": {"why": "test", "closed_frames": [[1730, 1740]], "open_frames": [[1741, 1754]]}}}})
    closed_line_none = s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "contrast",
                                                      "mouth": {"drawn": True, "inside": "the class map",
                                                                "closed_lip_line": {"why": "test", "closed_frames": [[1730, 1754]], "open_frames": []}}}})
    assert closed_line_all == closed_line_some == closed_line_none == want_unified, (closed_line_all, closed_line_some, closed_line_none)
    # THE MOUTH WHERE THE RECORD'S FRAMES DRAW ONE (1010j and 5114e, 2026-10-15: the form "auto" with no frame drawing a mouth): none on
    # a load whose every frame has none, "on some frames" where some do, as before where all do (1730-1754 is 25 frames)
    def mouthed(bare):
        return s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True, "frames_with_no_mouth": bare}}})
    # THE CASE THAT GOES RED when the text promises a mouth on a load whose reference draws none, or on every frame where it draws some
    assert "mouth" not in mouthed(25) and "with its eyes, eyebrows and nose drawn" in mouthed(25), mouthed(25)
    assert "on some frames the mouth is outlined too, and there it opens and closes" in mouthed(10) and "nose and mouth drawn" not in mouthed(10), mouthed(10)
    assert mouthed(0) == drawn, mouthed(0)
    # frames with no line of the face (a hair-alone stretch) have no mouth either, and do not make a mouth on every face frame partial
    bare = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "frames_left_bare": list(range(1745, 1755)),
                                          "mouth": {"drawn": True, "frames_with_no_mouth": 10}}})
    assert bare == drawn, bare
    # a mouth on some frames is said where every other part is left out too
    alone = s2({"form": "drawn", "drawn": {"parts_left_out": ["eyes", "brows", "nose"], "mouth": {"drawn": True, "frames_with_no_mouth": 10}}})
    assert alone.endswith("posed as the person moves; on some frames the mouth is outlined too, and there it opens and closes where the person's mouth does."), alone
    # HER EYES SHUT on a person's word (2026-10-15): the sentence says how a shut eye is drawn, and says nothing where none is named
    for form, words in (("lids", "where the person's eyes are shut, each eye is drawn as a single closed line"), ("none", "where the person's eyes are shut, no eye is drawn")):
        shut = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}, "eyes_shut": {"form": form}}})
        # THE CASE THAT GOES RED when the text says the eyes are outlined open on frames the reference draws them shut
        assert words in shut, shut
    shut_lowered = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}, "eyes_shut": {"form": "lowered"}}})
    shut_lowered_cl = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast",
                                                     "mouth": {"drawn": True, "closed_lip_line": {"why": "test", "closed_frames": [[1730, 1754]], "open_frames": []}},
                                                     "eyes_shut": {"form": "lowered"}}})
    assert shut_lowered == drawn and shut_lowered_cl == want_unified, (shut_lowered, shut_lowered_cl)
    assert "shut" not in drawn, drawn
    # THE PLAN'S VOICE (the owner's word of 2026-10-15 on second_hfp_5675b): in place of the voice table's guess where it names one
    import copy
    rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}
    def detail(voice, first=1730, last=1754):
        plan = copy.deepcopy(PLAN)
        if voice:
            plan["subjects"]["lead"]["voice"] = voice
        return next(line for line in lt.text(plan, "lead", rec, "", VOICED, first, last).splitlines() if line.startswith("[Shot 1]"))
    table = detail(None)
    talks = detail([{"first": 1700, "last": 1760, "voice": "talks, not heard", "by": "THE OWNER"}])
    quiet = detail([{"first": 1700, "last": 1760, "voice": "silent", "by": "THE OWNER"}])
    sings = detail([{"first": 1, "last": 20, "voice": "sings", "by": "THE OWNER"}], 1, 20)      # where the table hears no song
    # THE CASE THAT GOES RED when the plan's voice does not stand in place of the voice table's, or its words are not said
    # "talks, not heard" and "silent" write nothing of talking or singing (the owner, 2026-10-15: "just dont mention anything about the mom
    # talking or singing"): the table's sung line goes, and nothing comes in its place
    assert "(S1) sings" in table and talks == quiet == table.replace(" <Subject 1> (S1) sings, her mouth opening wide on the sung phrases.", ""), (table, talks)
    assert "(S1) sings" in sings and "sings" not in detail(None, 1, 20), sings
    assert detail([{"first": 1, "last": 20, "voice": "silent", "by": "X"}]) == table, "a voice on frames outside the load changed it"
    for bad, words in (([{"first": 1700, "last": 1760, "voice": "hums", "by": "X"}], "one of"), ([{"first": 1700, "last": 1760, "voice": "silent", "by": " "}], "no `by`"),
                       ([{"first": 1700, "last": 1740, "voice": "silent", "by": "X"}, {"first": 1741, "last": 1760, "voice": "sings", "by": "X"}], "One load is one voice"),
                       ([{"last": 1760, "voice": "silent", "by": "X"}], "is not {")):
        try:
            detail(bad)
            raise AssertionError(f"taken: {bad}")
        except lt.Refused as e:
            assert words in str(e), e
    left = s2({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": False}}})
    assert "eyes, eyebrows and nose drawn" in left and "mouth" not in left, left
    assert "a dark iris inside them" in s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "filled"}})
    no_brows = s2({"form": "drawn", "drawn": {"parts_left_out": ["mouth", "brows"], "iris": "between the lids"}})
    assert "with its eyes and nose drawn" in no_brows, no_brows
    plain = s2({"form": "plain", "drawn": {"parts_left_out": []}})
    assert plain.endswith("posed as the person moves.") and "outline" not in plain, plain
    others = lambda t: [line for line in t.splitlines() if not line.startswith("<Subject 2> is")]       # noqa: E731
    shut_mouth = lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": False}}}, "", VOICED, 1730, 1754)
    assert others(shut_mouth) == others(base), (shut_mouth, base)
    # AN OPEN MOUTH ALSO NAMES THE MOUTH in the retention line and [Shot 1] (the owner's go of 2026-10-22), and changes nothing else
    mouthed_text = lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}, "", VOICED, 1730, 1754)
    changed = [b for a, b in zip(shut_mouth.splitlines(), mouthed_text.splitlines()) if a != b]
    assert len(shut_mouth.splitlines()) == len(mouthed_text.splitlines()) and len(changed) == 3 and all("mouth" in line for line in changed) \
        and [line[:12] for line in changed] == ["<Subject 2> ", "<Subject 2> ", "[Shot 1] <Su"], changed
    assert "(S1) sings" in base and "(S1) sings" not in lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1000, 1100), "the singing line ignores the voice table"
    return ("the eye's form, a part left out and a plain cut change the <Subject 2> sentence and nothing else; an open mouth names the mouth there, "
            "in the retention line and in [Shot 1]; singing follows the voice table")


def a_load_of_the_hair_alone() -> str:
    """second_hp_5461b (2026-10-15): a load whose part is her hair alone says so, and nothing of her face or its expressions."""
    faceless = {"form": "drawn", "drawn": {"parts_left_out": ["eyes", "brows", "nose", "mouth"], "iris": "contrast", "mouth": {"drawn": False}}}
    hair = lt.text(PLAN, "second", faceless, "", VOICED, 484, 515, parts=["hair"])
    # THE CASE THAT GOES RED when a hair load's text says her face is the still's, keeps her face, or asks for facial expressions
    for wrong in ("face is the face of <Picture 1>", "face and her long", "facial expressions", "with the face", "blue-grey eyes", "eye movements"):
        assert wrong not in hair, (wrong, hair)
    for right in ("<Subject 1>, with the hair of <Picture 1>, performs <Subject 2>", "only the woman's long, loose blonde hair is retained, wherever it shows.\n",
                  "<Subject 1> is the woman with SECOND PICTURE HAIR whose appearance comes from <Picture 1>, and who follows the movements of the head of the figure in <Video 1>",
                  "[Shot 1] <Subject 1>, the woman, has the hair of <Picture 1>.", "attribute_transfer - the movements of the head are transferred"):
        assert right in hair, (right, hair)
    # the plan's voice is not read on a load of the hair alone: no face, no voice (the voice key's cold reader, 2026-10-15)
    import copy
    voiced_plan = copy.deepcopy(PLAN)
    voiced_plan["subjects"]["second"]["voice"] = [{"first": 400, "last": 600, "voice": "talks, not heard", "by": "X"}]
    assert lt.text(voiced_plan, "second", faceless, "", VOICED, 484, 515, parts=["hair"]) == hair, "a hair load read the voice"
    singing = lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1730, 1754)
    assert "(S1) sings" in singing and "(S1) sings" not in lt.text({**PLAN, "subjects": {**PLAN["subjects"], "lead": {**PLAN["subjects"]["lead"],
        "text": {**PLAN["subjects"]["lead"]["text"], "hair": "her hair"}}}}, "lead", faceless, "", VOICED, 1730, 1754, parts=["hair"])
    assert hair.splitlines()[2].endswith("posed as the person moves."), hair.splitlines()[2]
    # her face and hair (1010e's part list, the plan's two parts) read as the face-and-hair form, word for word
    # THE CASE THAT GOES RED when a face-and-hair load is refused for naming both its parts
    assert lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515, parts=["hair", "face_and_neck"]) == \
        lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515)
    # a hair load whose reference draws a face is refused: the text would speak of eyes the load must not draw
    try:
        lt.text(PLAN, "second", {"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "contrast"}}, "", VOICED, 484, 515, parts=["hair"])
        raise AssertionError("a hair load's text was written on a reference that draws a face")
    except lt.Refused as e:
        assert "draws a face" in str(e), str(e)
    # NO REFERENCE VIDEO (motion none, 5461's test): no <Video 1>, no <Subject 2>, nothing of following movements; a hair load only
    none = lt.text(PLAN, "second", None, "", VOICED, 484, 515, parts=["hair"])
    # THE CASE THAT GOES RED when a no-reference text names a video, a second subject or movements to follow
    for wrong in ("<Video", "<Subject 2>", "follows", "performs", "face is the face", "blue-grey eyes"):
        assert wrong not in none, (wrong, none)
    for right in ("<Subject 1> is the woman whose hair comes from <Picture 1>.", "<Subject 1> has the hair of <Picture 1>.",
                  "[Shot 1] <Subject 1>, the woman, has the hair of <Picture 1>.",
                  "only the woman's long, loose blonde hair is retained"):
        assert right in none, (right, none)
    try:
        lt.text(PLAN, "second", None, "", VOICED, 484, 515)
        raise AssertionError("a face-and-hair load's text was written with no reference video")
    except lt.Refused as e:
        assert "motion none" in str(e), str(e)
    # for_job on a job file of such a load: no record is read (motion none), the text has no video
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "plan.json").write_text(json.dumps(PLAN))
        (d / "voice.json").write_text(json.dumps({"voiced_spans_inclusive": VOICED}))
        (d / "job.json").write_text(json.dumps({"job": {"part": {"parts": ["hair"]}}, "loads": [{"subject": "second", "first": 484, "frames": 32,
                                                                                                 "motion": "none", "motion_file": "/nowhere/ref.mp4"}]}))
        words, made = lt.for_job(d / "job.json", "", d / "plan.json", d / "voice.json")
        # THE CASE THAT GOES RED when a motion-none load reads a reference record, or names a video
        assert "<Video" not in words and made["reference_record"].startswith("none"), (words, made)
    # the face alone of a face-and-hair subject reads as a face load; a part with no text form is refused
    face = lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515, parts=["face_and_neck"])
    assert "<Subject 1>, with the face of <Picture 1>, performs <Subject 2>." in face and ", has the face of <Picture 1>." in face, face
    try:
        lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515, parts=["hands"])
        raise AssertionError("a part with no text form was written")
    except lt.Refused as e:
        assert "no text form" in str(e), str(e)
    return ("a load of the hair alone names her hair alone and nothing of her face, its expressions or singing; the face alone reads as a face load; "
            "a part with no text form is refused")


def what_it_refuses() -> str:
    import copy
    for change, words in ((lambda p: p["subjects"]["lead"]["text"].pop("who"), "lacks ['who']"), (lambda p: p.pop("text"), "text.film"),
                          (lambda p: p["subjects"]["lead"]["text"].pop("hair_words"), "`text.hair_words` for 'lead' is not written"),
                          (lambda p: p["subjects"]["lead"]["text"].pop("picture_hair"), "`text.picture_hair` for 'lead' is not written"),
                          (lambda p: p["subjects"]["second"]["text"].pop("hair"), "`text.hair`")):
        plan = copy.deepcopy(PLAN)
        change(plan)
        who = "second" if "`text.hair`" in words else "lead"
        try:
            lt.text(plan, who, RECORD_10_13, "", VOICED, 1, 2)
        except lt.Refused as e:
            assert words in str(e), (words, e)
        else:
            raise AssertionError(f"not refused: {words}")
    # THE CASE THAT GOES RED when a generated text takes a place-and-action sentence (NO_SCENE, 2026-10-15)
    for record, sentence, words in ((RECORD_10_13, "stands at a window", "takes no place-and-action sentence"),
                                    ({"form": "drawn", "drawn": {"parts_left_out": [], "iris": "glowing"}}, "", "no words for it")):
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
            lt.for_job(d / "job.json", "", d / "plan.json", d / "spans.json")
        except lt.Refused as e:
            assert "is made after it" in str(e), e
        else:
            raise AssertionError("a text was made before its reference")
        (d / "ref.json").write_text(json.dumps(RECORD_10_13))
        words, made_from = lt.for_job(d / "job.json", "", d / "plan.json", d / "spans.json")
        assert words.startswith("subject_definitions:") and made_from["drawn"]["parts_left_out"] == ["mouth"], made_from
    return "no words for the subject or no film line, no picture_hair or hair_words, no hair words for a face-and-hair subject, a place-and-action sentence, an unknown iris form and a reference not yet made are refused"


def a_hair_change_is_refused() -> str:
    """The owner's word of 2026-10-23: the second subject's text carries no trace of shortening her hair. The plan's `text.hair_change`
    (2026-10-15 to 2026-10-22: "...; her hair's length is changed to short", the changed hair in the summary and [Shot 1]) is retired: a
    plan that still carries it is refused in the tool's own words, on every load of its subject (face and hair, the face alone, her hair
    alone, no reference video), and a history note under another key is not the field."""
    import copy
    rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}
    faceless = {"form": "drawn", "drawn": {"parts_left_out": ["eyes", "brows", "nose", "mouth"], "iris": "contrast", "mouth": {"drawn": False}}}
    for change in ({"kept": "BLONDE", "what": "LENGTH", "to": "SHORT", "by": "the owner"}, {}, "SHORT", None):
        plan = copy.deepcopy(PLAN)
        plan["subjects"]["second"]["text"]["hair_change"] = change
        plan["subjects"]["lead"]["text"]["hair_change"] = change
        for who, record, frames, parts in (("second", rec, (484, 515), None), ("second", rec, (484, 515), ["face_and_neck"]),
                                           ("second", faceless, (484, 515), ["hair"]), ("second", None, (484, 515), ["hair"]),
                                           ("lead", rec, (1730, 1754), None)):
            try:
                got = lt.text(plan, who, record, "", VOICED, *frames, parts=parts)
            except lt.Refused as e:
                # THE CASE THAT GOES RED when a plan carrying the retired field is taken, or refused by other words than the tool's
                assert str(e) == f"the plan's `text.hair_change` for {who!r} " + lt.HAIR_CHANGE_RETIRED, e
            else:
                raise AssertionError(f"a plan carrying text.hair_change {change!r} was taken for {who} {parts}: {got}")
    # in the tool's own words from the command line, with no traceback
    fx = json.loads(FIXTURE.read_text())
    plan = copy.deepcopy(fx["plan"])
    plan["subjects"]["second"]["text"]["hair_change"] = {"kept": "blonde", "what": "length", "to": "short, ending at her shoulders", "by": "the owner"}
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "plan.json").write_text(json.dumps(plan))
        (d / "spans.json").write_text(json.dumps({"voiced_spans_inclusive": fx["voiced_spans_inclusive"]}))
        (d / "ref.json").write_text(json.dumps(fx["record"]))
        (d / "job.json").write_text(json.dumps({"job": {}, "loads": [{"subject": "second", "first": 415, "frames": 23, "motion_file": str(d / "ref.mp4")}]}))
        def show():
            return subprocess.run([sys.executable, str(Path(lt.__file__)), "show", str(d / "job.json"), "--plan", str(d / "plan.json"),
                                   "--voice-spans", str(d / "spans.json")], capture_output=True, text=True)
        run = show()
        assert run.returncode == 5 and run.stdout == f"REFUSED: the plan's `text.hair_change` for 'second' {lt.HAIR_CHANGE_RETIRED}\n" \
            and "Traceback" not in run.stdout + run.stderr, (run.returncode, run.stdout, run.stderr)
        # A HISTORY NOTE IS NOT THE FIELD (data/fun/plan.json's `hair_change_removed_by`): taken, and the text carries no hair change
        plan["subjects"]["second"]["text"]["hair_change_removed_by"] = repr(plan["subjects"]["second"]["text"].pop("hair_change"))
        (d / "plan.json").write_text(json.dumps(plan))
        run = show()
        assert run.returncode == 0 and run.stdout.startswith("subject_definitions:") and "Traceback" not in run.stderr, (run.returncode, run.stdout, run.stderr)
        # THE CASE THAT GOES RED when any trace of the hair change is written: its words, its shortened hair, the old summary's form
        for wrong in ("changed", "length", "short", "ending at her shoulders", "with the face of <Picture 1> and"):
            assert wrong not in run.stdout, (wrong, run.stdout)
    # nothing of the mechanism is left in the tool to write it
    import inspect
    source = inspect.getsource(lt.text)
    assert "is changed to" not in source and "kept_now" not in source and source.count("hair_change") == 2, source.count("hair_change")
    return ("a plan carrying text.hair_change refused in the tool's own words on every load of its subject, from the command line with no "
            "traceback; a history note under another key is taken and its text carries no hair change; no hair-change words left in the tool")


def her_clothing_named() -> str:
    """The owner's decision of 2026-10-16 (26f208c0): her clothing is redrawn with her and named in the text; per shot (the lead's ruling,
    1f143f59: one apron line made H3 draw straps on 1010-1038, where she wears a dark top). The plan's entry that holds the load's frames is
    one sentence of the detailed line; a load no entry holds, the old single set of words, an entry with no `by`, and a subject whose clothing
    is redrawn and not named are refused."""
    import copy
    plan = copy.deepcopy(PLAN)
    second = plan["subjects"]["second"]
    second["clothing"] = [{"first": 400, "last": 600, "words": "A WHITE APRON OVER A FLORAL BLOUSE", "by": "a person, from frames 450 and 500"},
                          {"first": 601, "last": 700, "words": "A DARK SLEEVELESS TOP", "by": "a person, from frame 650"}]
    got = lt.text(plan, "second", RECORD_10_13, "", VOICED, 484, 515)
    shot = next(line for line in got.splitlines() if line.startswith("[Shot 1]"))
    # THE CASE THAT GOES RED when the plan's clothing for these frames is not written into the text
    assert "has the face and the hair of <Picture 1>. <Subject 1> wears A WHITE APRON OVER A FLORAL BLOUSE." in shot, shot
    assert got.replace(" <Subject 1> wears A WHITE APRON OVER A FLORAL BLOUSE.", "") == lt.text(PLAN, "second", RECORD_10_13, "", VOICED, 484, 515), \
        "naming the clothing changed another line"
    hair = lt.text(plan, "second", None, "", VOICED, 484, 515, parts=["hair"])
    assert "<Subject 1> wears A WHITE APRON OVER A FLORAL BLOUSE." in hair, hair
    # another shot takes its own words, never the first entry's
    other = lt.text(plan, "second", RECORD_10_13, "", VOICED, 620, 650)
    assert "<Subject 1> wears A DARK SLEEVELESS TOP." in other and "APRON" not in other, other
    for frames, words in (((710, 740), "names no words for frames 710-740"), ((590, 610), "names no words for frames 590-610")):
        try:
            lt.text(plan, "second", RECORD_10_13, "", VOICED, *frames)
        except lt.Refused as e:
            # THE CASE THAT GOES RED when a load no entry holds falls back to another shot's words, or to none
            assert words in str(e) and "never another shot's" in str(e), e
        else:
            raise AssertionError(f"frames {frames} took words no entry of theirs gives")
    entry = {"first": 400, "last": 600, "words": "A WHITE APRON", "by": "a person, from frame 450"}
    # THE WHOLE LIST, checked whenever it is read (the clothing read): each refusal by its own words, never by an error on the way
    for bad, words in (({**second, "clothing": "A WHITE APRON"}, "one set of words for every shot"),
                       ({**second, "clothing_by": "a person"}, "the old form"),
                       ({**second, "clothing": [{"first": 400, "last": 600, "words": "A WHITE APRON"}]}, "lacks by"),
                       ({**second, "clothing": [{**entry, "by": "  "}]}, "lacks by"),
                       ({**second, "clothing": ["A WHITE APRON"]}, "is not a {first, last, words, by}"),
                       ({**second, "clothing": [{**entry, "first": "400"}]}, "not a whole frame number"),
                       ({**second, "clothing": [{**entry, "last": True}]}, "not a whole frame number"),
                       ({**second, "clothing": [{**entry, "first": 600, "last": 400}]}, "runs backwards"),
                       ({**second, "clothing": [{**entry, "words": ""}]}, "empty words"),
                       ({**second, "clothing": [entry, {**entry, "first": 650, "last": 640}]}, "runs backwards"),
                       # an overlap anywhere, the load's frames in it or not
                       ({**second, "clothing": [entry, {**entry, "first": 1000, "last": 1100}, {**entry, "first": 1050, "last": 1200}]}, "overlap on 1050-1100"),
                       # two entries that both hold the load
                       ({**second, "clothing": [entry, {**entry, "first": 450, "last": 520, "words": "A DARK TOP"}]}, "overlap on 450-520")):
        try:
            lt.text({**plan, "subjects": {**plan["subjects"], "second": bad}}, "second", RECORD_10_13, "", VOICED, 484, 515)
        except Exception as e:  # noqa: BLE001 -- the refusal is the tool's, by its own words, or the case fails
            # THE CASE THAT GOES RED when a malformed list is read, or refused only by the error it happens to raise
            assert isinstance(e, lt.Refused) and words in str(e), (words, type(e).__name__, e)
        else:
            raise AssertionError(f"refused nothing: {bad.get('clothing')} {bad.get('clothing_by', '')}")
    # nothing to name, said by a look: `words: null` with its `by` names no clothing on those frames
    bare = lt.text({**plan, "subjects": {**plan["subjects"], "second": {**second, "clothing": [{**entry, "words": None, "by": "a person, from 450: her clothes are out of the frame"}]}}},
                   "second", RECORD_10_13, "", VOICED, 484, 515)
    assert "wears" not in bare, bare
    second.pop("clothing")
    second["look"] = {"stay_removed_by": "the owner chose to redraw it"}
    try:
        lt.text(plan, "second", RECORD_10_13, "", VOICED, 484, 515)
    except lt.Refused as e:
        # THE CASE THAT GOES RED when a subject whose clothing is redrawn is given a text that does not name it
        assert lt.NO_CLOTHING in str(e) and "frames 484-515" in str(e), e
    else:
        raise AssertionError("a subject whose clothing is redrawn got a text with no clothing")
    assert "wears" not in lt.text(PLAN, "lead", RECORD_10_13, "", VOICED, 1730, 1754), "a subject with no clothing named got a clothing sentence"
    return ("each shot's own words, one sentence of the detailed line; a load no entry holds refused, never another shot's words; the single "
            "set of words and an entry with no by refused; clothing redrawn and not named refused; none named, no sentence")

def no_not_used_clause() -> str:
    """The owner, 2026-10-21: no "not used" clause; each retention line ends at what is kept or changed. The test-only switch that wrote
    the old form for that day's A/B (`--say-not-used`) is retired since 2026-10-22 on the owner's rule that negatives never help."""
    import copy
    rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}
    plan = copy.deepcopy(PLAN)
    plan["subjects"]["third"] = {"region": "face", "text": {"who": "the man", "picture_hair": "HIS PICTURE HAIR", "hair_words": "HIS HAIR", "her": "his", "she": "he",
                                                            "not_used": "THE CAP, the hair, the clothing and the background"}}
    now = lt.text(plan, "second", rec, "", VOICED, 484, 515)
    # THE CASE THAT GOES RED when a "not used" clause is written, or a shortened sentence does not end cleanly
    assert "not used" not in now, now
    assert ("partially_preserved - the face and the SECOND PICTURE HAIR of <Picture 1> are retained.\n"
            "<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head, the facial expressions, where the eyes look and "
            "how far the mouth opens are transferred to <Subject 1>.\n\n") in now, now
    # the plan's `text.not_used` (third's) is read by no line
    third = lt.text(plan, "third", RECORD_10_13, "", VOICED, 1730, 1754)
    assert "not used" not in third and "CAP" not in third and "the man's face from <Picture 1> is retained.\n<Subject 2>" in third, third
    hair_now = lt.text(plan, "second", None, "", VOICED, 484, 515, parts=["hair"])
    assert "not used" not in hair_now and "only the woman's long, loose blonde hair is retained, wherever it shows.\n\ndetailed_description:" in hair_now, hair_now
    # the switch is gone from the tool: refused by its own parser, with no traceback; nothing recorded of it in what a text is made from
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "plan.json").write_text(json.dumps(PLAN))
        (d / "spans.json").write_text(json.dumps({"voiced_spans_inclusive": VOICED}))
        (d / "job.json").write_text(json.dumps({"job": {}, "loads": [{"subject": "lead", "first": 1730, "frames": 25, "motion_file": str(d / "ref.mp4")}]}))
        (d / "ref.json").write_text(json.dumps(RECORD_10_13))
        words, made_from = lt.for_job(d / "job.json", "", d / "plan.json", d / "spans.json")
        assert "not used" not in words and "not_used_said" not in made_from, made_from
        run = subprocess.run([sys.executable, str(Path(lt.__file__)), "show", str(d / "job.json"), "--plan", str(d / "plan.json"), "--voice-spans",
                              str(d / "spans.json"), "--say-not-used", "the A/B"], capture_output=True, text=True)
        # THE CASE THAT GOES RED when the retired switch is taken
        assert run.returncode == 2 and "unrecognized arguments: --say-not-used" in run.stderr and "Traceback" not in run.stderr, (run.returncode, run.stderr)
    return ("no \"not used\" clause on a face-and-hair, a face and a hair-alone load, each line ending cleanly; the plan's not_used read by no line; "
            "the retired --say-not-used refused by the tool's parser")

def the_mouth_clause_by_the_open_mouth() -> str:
    """The owner's approval of 2026-10-22 (relayed by the lead), in these words: under the closed-lip line the face is drawn and each eye
    has a dark iris, for every subject; and for the lead's outlined open mouth, the lips outlined where the mouth is open (a closed frame
    draws nothing). Every other form, and a record without the field, names no mouth."""
    def s2(open_mouth=None, **mouth):
        cl = {"why": "test", "closed_frames": [], "open_frames": [[1730, 1754]], **({"open_mouth": open_mouth} if open_mouth else {})}
        rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True, "inside": "the class map", "closed_lip_line": cl, **mouth}}}
        return next(line for line in lt.text(PLAN, "lead", rec, "", VOICED, 1730, 1754).splitlines() if line.startswith("<Subject 2> is"))
    lead = ("<Subject 2> is the movements of the head and the facial expressions performed by the figure in <Video 1>: a closer view of the head "
            "and shoulders of a plain grey body model of a person on a black ground, posed as the person moves, with its face drawn as thin white "
            "outlines: the eyes, each with a dark iris, open, close and look where the person's eyes do; where the mouth is open, the lips are "
            "outlined and it is dark inside, and the mouth opens and closes where the person's mouth does.")
    unified = lead.split("; where the mouth is open")[0] + "."
    # THE CASE THAT GOES RED when the lead's outlined form is not the owner's approved sentence, byte for byte
    assert s2("outlined") == lead, s2("outlined")
    # THE CASE THAT GOES RED when another form, or a subject without the field (the second's and the third's references, every one
    # before 2026-10-22), gets a mouth clause
    for form in (None, "inner outline", "capped", "outlined over the lips"):
        assert s2(form) == unified, (form, s2(form))
    # where the reference draws the mouth on some frames, "on some frames"; on none, no clause
    assert s2("outlined", frames_with_no_mouth=10) == unified[:-1] + ("; on some frames, where the mouth is open, the lips are outlined and it is "
                                                                     "dark inside, and there the mouth opens and closes where the person's mouth does."), \
        s2("outlined", frames_with_no_mouth=10)
    assert s2("outlined", frames_with_no_mouth=25) == unified, s2("outlined", frames_with_no_mouth=25)
    none = next(line for line in lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": ["mouth"], "iris": "contrast", "mouth": {
        "drawn": False, "closed_lip_line": {"why": "silent window", "closed_frames": [], "open_mouth": "outlined"}}}}, "", VOICED, 1730, 1754).splitlines()
        if line.startswith("<Subject 2> is"))
    assert none == unified, none
    # the eyes' shut clause first, then the mouth's; with no eye drawn, the words before for the face, unchanged
    rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "eyes_shut": {"form": "lids"},
                                      "mouth": {"drawn": True, "closed_lip_line": {"why": "t", "open_mouth": "outlined"}}}}
    lids = next(line for line in lt.text(PLAN, "lead", rec, "", VOICED, 1730, 1754).splitlines() if line.startswith("<Subject 2> is"))
    assert lids.endswith("each eye is drawn as a single closed line; where the mouth is open, the lips are outlined and it is dark inside, and the "
                         "mouth opens and closes where the person's mouth does."), lids
    rec = {"form": "drawn", "drawn": {"parts_left_out": ["eyes"], "iris": "contrast", "mouth": {"drawn": True, "closed_lip_line": {"why": "t"}}}}
    eyeless = next(line for line in lt.text(PLAN, "lead", rec, "", VOICED, 1730, 1754).splitlines() if line.startswith("<Subject 2> is"))
    assert eyeless.endswith("posed as the person moves, with facial features drawn on its face as thin white outlines."), eyeless
    # THE CASE THAT GOES RED when the retention line or [Shot 1]'s sentence is not the owner's approved words, with the plan's pronouns
    full = lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True, "closed_lip_line": {
        "why": "t", "open_mouth": "outlined"}}}}, "", [], 1730, 1754).splitlines()
    assert "<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head, the facial expressions, where the eyes look and how far " \
           "the mouth opens are transferred to <Subject 1>." in full, full
    shot = next(line for line in full if line.startswith("[Shot 1]"))
    assert shot.endswith("<Subject 1> performs <Subject 2>: she follows the movements of the head and the facial expressions of the figure, her eyes "
                         "looking where its eyes look and her mouth opening and closing where its mouth does."), shot
    plan = json.loads(json.dumps(PLAN)); plan["subjects"]["lead"]["text"].update(she="he", her="his")
    his = next(line for line in lt.text(plan, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}},
                                        "", [], 1730, 1754).splitlines() if line.startswith("[Shot 1]"))
    assert his.endswith(": he follows the movements of the head and the facial expressions of the figure, his eyes looking where its eyes look and his "
                        "mouth opening and closing where its mouth does."), his
    # a plain cut (no face drawn) and a hair load keep the lines before
    plain = lt.text(PLAN, "lead", {"form": "plain", "drawn": {"parts_left_out": []}}, "", [], 1730, 1754)
    assert "and eye movements are transferred" in plain and "follows the movements of the head and the facial expressions of the figure.\n" in plain, plain
    import draw_face_points
    assert set(lt.MOUTH_CLAUSE) <= set(draw_face_points.OPEN_MOUTH_FORMS[1:]), (lt.MOUTH_CLAUSE, draw_face_points.OPEN_MOUTH_FORMS)
    fx = json.loads(FIXTURE.read_text())
    assert "open_mouth" not in json.dumps(fx["record"]) and "the lips are outlined" not in fx["owner_text"]
    # THE CASE THAT GOES RED when the outlined form names the mouth on a load whose closed-lip line opens it on no frame (0278l, 2026-10-22)
    assert s2("outlined") != unified and next(line for line in lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast",
        "mouth": {"drawn": True, "inside": "the class map", "closed_lip_line": {"why": "t", "closed_frames": [[1730, 1754]], "open_frames": [],
                                                                                "open_mouth": "outlined"}}}}, "", VOICED, 1730, 1754).splitlines()
        if line.startswith("<Subject 2> is")) == unified
    return ("the lead's outlined form makes the owner's approved sentence byte for byte; every subject's eyes are drawn each with a dark iris; "
            "no other form and no field says anything of the mouth; on some frames and on none as the drawing; no eye drawn, the words before")


def the_mouth_words_where_the_drawing_opens_one() -> str:
    """The owner's go of 2026-10-22, after second_hfp_0278l: its drawing opened no mouth (closed-lip line, `open_frames` []) and its text
    said her mouth opens and closes; she spoke, lifted her head and looked at the camera, where 0278f, the same drawing with no mouth words,
    kept her lips closed on all 16 frames (docs/masked_v2v/PROTOCOL.md, case study 3). The mouth is named only where the drawing opens one,
    in every line that would name it: `<Subject 2>`'s sentence, its retention line and [Shot 1]."""
    def made(open_frames=None, voiced=(), **mouth):
        cl = {"why": "t", "closed_frames": [[1730, 1754]], "open_mouth": "outlined", **({} if open_frames is None else {"open_frames": open_frames})}
        rec = {"form": "drawn", "frames": [1730, 1754], "drawn": {"parts_left_out": [], "iris": "contrast",
                                                                  "mouth": {"drawn": True, "inside": "the class map", "closed_lip_line": cl, **mouth}}}
        return lt.text(PLAN, "lead", rec, "", list(voiced), 1730, 1754)
    def lines(words):
        out = words.splitlines()
        return (next(x for x in out if x.startswith("<Subject 2> (appears")), next(x for x in out if x.startswith("[Shot 1]")),
                next(x for x in out if x.startswith("<Subject 2> is")))
    shut, opened, some = made([]), made([[1741, 1754]]), made([[1741, 1754]], frames_with_no_mouth=10)
    # THE CASE THAT GOES RED when a load whose drawing opens no mouth names one anywhere (the gate removed), or one that opens it names
    # it in no line
    assert "mouth" not in shut, shut
    assert lines(shut)[:2] == ("<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head, the facial expressions and where "
                               "the eyes look are transferred to <Subject 1>.",
                               "[Shot 1] <Subject 1>, the girl with LEAD HAIR, has the face of <Picture 1>. <Subject 1> performs <Subject 2>: she follows "
                               "the movements of the head and the facial expressions of the figure, her eyes looking where its eyes look."), lines(shut)
    assert all("mouth" in x for x in lines(opened)) and "how far the mouth opens" in lines(opened)[0] \
        and "her mouth opening and closing where its mouth does." in lines(opened)[1] and lt.MOUTH_CLAUSE["outlined"][0] in lines(opened)[2], lines(opened)
    # where the drawing opens it on some frames only, the retention and [Shot 1] lines as where it opens it on all, the sentence "on some frames"
    assert lines(some)[:2] == lines(opened)[:2] and lt.MOUTH_CLAUSE["outlined"][1] in lines(some)[2], lines(some)
    # a record with no `open_frames` (before the closed-lip line counted them): the mouth as drawn on the load's frames, never on none
    assert all("mouth" in x for x in lines(made())) and "mouth" not in made(frames_with_no_mouth=25), made(frames_with_no_mouth=25)
    assert "mouth" not in lt.text(PLAN, "lead", {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": False}}},
                                  "", [], 1730, 1754)
    # the singing line is the voice's, not the drawing's: it stays on a singer's load whose drawing opens no mouth
    assert "(S1) sings, her mouth opening wide" in made([], VOICED) and lines(made([], VOICED))[0] == lines(shut)[0]
    return ("a drawing that opens no mouth (open_frames []) names no mouth in the <Subject 2> sentence, its retention line or [Shot 1]; one that "
            "opens it names it in all three, \"on some frames\" in the sentence where it opens on some; the singing line is the voice's")


def the_mouth_words_where_the_drawing_draws_one_closed() -> str:
    """The owner, 2026-10-23, on second_hfp_5114i: the closed-lip line called every frame closed, nothing was drawn at the mouth and the
    text named none, so nothing told H3 what her mouth does, and it took the still's smile ("If we dont have a drawing for mouth
    expressions or facial expressions like that we probably do need one"). Under the plan's `reference.closed_mouth` "outlined" the drawing
    draws the closed mouth (the record's `closed_lip_line.closed_mouth.frames_drawn_closed`), and the text names the mouth as it does where
    one is drawn open (`TRANSFERRED_DRAWN`, `FOLLOWS_DRAWN`); `MOUTH_CLAUSE`, the open mouth's words, stays tied to an open frame. Without
    the word, or where it drew no frame closed, every text is as before."""
    def made(closed_mouth=None, open_frames=(), open_mouth="inner outline", subject="second", voiced=()):
        cl = {"why": "t", "closed_frames": [[5114, 5129]], "open_frames": [list(x) for x in open_frames], "open_mouth": open_mouth,
              **({} if closed_mouth is None else {"closed_mouth": closed_mouth})}
        rec = {"form": "drawn", "frames": [5114, 5129], "drawn": {"parts_left_out": ["mouth"], "iris": "contrast", "mouth": {
            "form": "auto", "drawn": True, "drawn_as": "the class map", "inside": "the class map", "closed_lip_line": cl}}}
        return lt.text(PLAN, subject, rec, "", list(voiced), 5114, 5129)

    def lines(words):
        out = words.splitlines()
        return (next(x for x in out if x.startswith("<Subject 2> (appears")), next(x for x in out if x.startswith("[Shot 1]")),
                next(x for x in out if x.startswith("<Subject 2> is")))
    drew = {"form": "outlined", "frames_drawn_closed": [[5114, 5129]]}
    before, none_drawn, closed = made(), made({"form": "outlined", "frames_drawn_closed": []}), made(drew)
    # without the word, or with it and no frame drawn closed (a silent window, no lips on any frame): the text of 5114i, no mouth named
    assert "mouth" not in before and none_drawn == before, (before, none_drawn)
    assert made({"form": "sculpted"}) == before
    # THE CASE THAT GOES RED when a drawn closed mouth leaves the mouth unnamed (5114i's text), in the retention line or [Shot 1]
    assert lines(closed)[:2] == ("<Subject 2> (appears in [Shot 1]): attribute_transfer - the movements of the head, the facial expressions, where "
                                 "the eyes look and how far the mouth opens are transferred to <Subject 1>.",
                                 "[Shot 1] <Subject 1>, the woman with SECOND HAIR, has the face and the hair of <Picture 1>. <Subject 1> performs <Subject 2>: she "
                                 "follows the movements of the head and the facial expressions of the figure, her eyes looking where its eyes look and "
                                 "her mouth opening and closing where its mouth does."), lines(closed)
    # the same words, byte for byte, as a load of that subject whose drawing opens the mouth on some frame (0014g's form)
    assert closed == made(open_frames=[[5120, 5124]]), (closed, made(open_frames=[[5120, 5124]]))
    # THE CASE THAT GOES RED when the open mouth's clause is said of a load drawn closed on every frame: the sentence is the one before
    assert lines(closed)[2] == lines(before)[2], lines(closed)[2]
    lead_closed, lead_open = made(drew, open_mouth="outlined"), made(drew, open_frames=[[5120, 5124]], open_mouth="outlined")
    assert lt.MOUTH_CLAUSE["outlined"][0] not in lead_closed and "how far the mouth opens" in lead_closed, lead_closed
    assert lt.MOUTH_CLAUSE["outlined"][0] in lines(lead_open)[2] and lead_open == made(open_frames=[[5120, 5124]], open_mouth="outlined"), lead_open
    # the readers: a record without the key, or a plain one, draws no closed mouth
    assert lt.mouth_drawn_closed({"mouth": {"closed_lip_line": {"closed_mouth": drew}}}) and not lt.mouth_drawn_closed({"mouth": {"closed_lip_line": {}}})
    assert not lt.mouth_drawn_closed(None) and not lt.mouth_drawn_closed({"mouth": {"closed_lip_line": True}}) and not lt.mouth_drawn({}), "a record with no key"
    # a hair load and a plain cut keep their lines whatever the record says of a closed mouth
    plain = lt.text(PLAN, "second", {"form": "plain", "drawn": {"parts_left_out": [], "mouth": {"closed_lip_line": {"closed_mouth": drew}}}}, "", [], 5114, 5129)
    assert "how far the mouth opens" not in plain and "mouth opening" not in plain, plain
    return ("a load drawn closed under the plan's closed_mouth names the mouth in its retention line and [Shot 1], byte for byte as a load "
            "drawn open does; the open mouth's clause only where a mouth is open; without the word, or with no frame drawn closed, 5114i's text")


def her_face_and_hair_with_no_change() -> str:
    """The owner's word of 2026-10-23: a face-and-hair subject (the second's, by her plan's region, never by her name) has the face and
    the picture's hair, in the same words, in the summary and the retention line ("with the face and the blonde hair of <Picture 1>",
    "the face and the blonde hair of <Picture 1> are retained."), and [Shot 1] "has the face and the hair of <Picture 1>" (her name
    there already says the hair); the definition as it was. Until then "with the face and the hair of <Picture 1>", "the woman's face
    and blonde hair from <Picture 1> are retained." and "has the face of <Picture 1>". A face-only subject's lines are unchanged."""
    import copy
    plan = copy.deepcopy(PLAN)
    plan["subjects"]["second"]["text"].update(hair="her blonde hair", picture_hair="blonde hair", hair_words="blonde hair")
    rec = {"form": "drawn", "drawn": {"parts_left_out": [], "iris": "contrast", "mouth": {"drawn": True}}}
    lines = lt.text(plan, "second", rec, "", VOICED, 484, 515).splitlines()
    # THE CASE THAT GOES RED when any of the four lines is not the owner's words of 2026-10-23, byte for byte
    assert "<Subject 1> is the woman with blonde hair whose appearance comes from <Picture 1>, and who follows the movements of the head and the " \
           "facial expressions of the figure in <Video 1>." in lines, lines
    assert "[reference generation] <Subject 1>, with the face and the blonde hair of <Picture 1>, performs <Subject 2>." in lines, lines
    assert "<Subject 1> (appears in [Shot 1]): partially_preserved - the face and the blonde hair of <Picture 1> are retained." in lines, lines
    assert any(line.startswith("[Shot 1] <Subject 1>, the woman with blonde hair, has the face and the hair of <Picture 1>. ") for line in lines), lines
    # the words are the plan's: the picture's hair in the definition, the summary and the retention line, the shot's in [Shot 1]'s name
    plan["subjects"]["second"]["text"].update(who="the girl", picture_hair="PICTURE RED HAIR", hair_words="SHOT RED HAIR")
    girl = lt.text(plan, "second", rec, "", VOICED, 484, 515).splitlines()
    # THE CASE THAT GOES RED when one hair stands in for the other, or a word of the hair is the tool's
    assert [line for line in girl if "PICTURE RED HAIR" in line] == [
        "<Subject 1> is the girl with PICTURE RED HAIR whose appearance comes from <Picture 1>, and who follows the movements of the head and the "
        "facial expressions of the figure in <Video 1>.",
        "[reference generation] <Subject 1>, with the face and the PICTURE RED HAIR of <Picture 1>, performs <Subject 2>.",
        "<Subject 1> (appears in [Shot 1]): partially_preserved - the face and the PICTURE RED HAIR of <Picture 1> are retained."], girl
    assert [line for line in girl if "SHOT RED HAIR" in line] == [next(line for line in girl if line.startswith("[Shot 1]"))] and \
        next(line for line in girl if line.startswith("[Shot 1]")).startswith("[Shot 1] <Subject 1>, the girl with SHOT RED HAIR, has the face and the hair of <Picture 1>. "), girl
    assert not any(w in " ".join(girl) for w in ("blonde", "changed", "length", "short")), girl
    # a face-only subject, and the face alone of a face-and-hair subject: the face-only lines, unchanged
    for who, parts, frames in (("lead", None, (1730, 1754)), ("second", ["face_and_neck"], (484, 515))):
        face = lt.text(PLAN, who, rec, "", VOICED, *frames, parts=parts)
        noun = PLAN["subjects"][who]["text"]["who"].split(" ", 1)[1]
        # THE CASE THAT GOES RED when a face-only load takes the face-and-hair words
        assert f"partially_preserved - the {noun}'s face from <Picture 1> is retained.\n" in face and "[reference generation] <Subject 1>, with the face of " \
               "<Picture 1>, performs <Subject 2>." in face and ", has the face of <Picture 1>." in face and "the face and the" not in face, face
    # the 0415 fixture carries no hair change: its plan is the plan's as it stands
    fx = json.loads(FIXTURE.read_text())
    assert "hair_change" not in fx["plan"]["subjects"]["second"]["text"], "the 0415 fixture carries the retired hair change"
    return ("a face-and-hair subject: \"with the face and the blonde hair of <Picture 1>\", \"the face and the blonde hair of <Picture 1> are "
            "retained.\", \"has the face and the hair of <Picture 1>\", the hair the plan's; a face-only load as before")


case("the owner's text of 2026-10-22 made again", the_owners_text_made_again)
case("a plan carrying the retired hair change is refused", a_hair_change_is_refused)
case("her face and hair with no change", her_face_and_hair_with_no_change)
case("the kept texts made again", the_kept_texts_made_again)
case("what a reference draws, the text says", what_a_reference_draws_the_text_says)
case("the mouth's clause by the open-mouth form", the_mouth_clause_by_the_open_mouth)
case("the mouth's words where the drawing opens one", the_mouth_words_where_the_drawing_opens_one)
case("the mouth's words where the drawing draws one closed", the_mouth_words_where_the_drawing_draws_one_closed)
case("a load of the hair alone", a_load_of_the_hair_alone)
case("what it refuses", what_it_refuses)
case("her clothing named", her_clothing_named)
case("no not used clause; the test switch retired", no_not_used_clause)
sys.exit(finish())
