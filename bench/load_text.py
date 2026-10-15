#!/usr/bin/env python3
"""A load's text, generated from its reference's record, the plan and the voice table, around the one sentence a person writes.

    <python> bench/load_text.py show JOB.json --place-and-action "is in a room and holds a black telephone receiver to her ear"

(`bench/shot_to_load.py text JOB.json --place-and-action "..." --writer "..." --by SESSION` writes it beside the job file.)

**What it is for.** The owner's word of 2026-10-14: every window's text is consistent with the others, and says exactly
what the reference video shows and the least else. Until then a load's text was copied by hand from load to load, and
its `<Subject 2>` sentence named what an earlier reference drew: once the mouth is drawn by default or the eye is drawn
in contrast, a copied sentence describes a reference that is not there. So the fixed lines are made here, from the
records, and a person writes only the shot's place and action, once.

**Where each line comes from.**

- `<Subject 1>`'s definition, and the retention line's words for them: the plan's subject, `text`:
  `who` ("the girl"), `looks` ("brown eyes, dark eyebrows and fair skin"), `her` and `she` (the pronouns), optionally
  `hair` (for a face-and-hair subject: "her long, loose blonde hair"), `not_used` (what of the still is not used; by the
  plan's region when left out) and `sings` (true for the subject whose voice the song is).
- `<Subject 2>`'s sentence: the reference's record. "a closer view of the head and shoulders" (the lead's ruling of
  2026-10-14: the fewest words, nothing typed by sight); the parts drawn (`drawn.parts_left_out`: eyes, eyebrows, nose,
  and the mouth when drawn); the eyes' clause by the iris form (`drawn.iris`); the mouth's clause when it is drawn
  (`drawn.mouth.drawn`), dark inside where it is open when its inside is the class map's (`drawn.mouth.inside`). A plain cut
  (no drawing) has no outline clause.
- The summary and `[Shot 1]`: the person's place-and-action sentence, as written, in both. With the face, or with the
  face and the hair, of <Picture 1>, by the plan's region.
- The film line: the plan's `text.film`, the video's, written once.
- The singing sentence: the subject's `sings`, on a load whose frames meet a voiced span of the video's voice table.
- `<Subject 2>`'s retention line, the soundscape and the music: fixed, the kept loads' words.

**The form is the kept loads'**, word for word but where a record or the plan says otherwise: lead_fp_1730 and
lead_fp_1755 (10/13; the same text), second_hfp_0484c (10/14, face and hair). Two places differ from them, by ruling:
the summary takes the place-and-action sentence as written (1730's put the place last), and the closer view is "of the
head and shoulders" (theirs named arms or hands, typed by sight). `bench/check_load_text.py` holds the generator to
those texts.

**A load of the hair alone** (`--parts hair`, the job's `part.parts`): its words are the hair's, not the plan's region's. The
summary and the retention line name the hair alone, the face is among what of the still is not used, the detailed line says her
hair (not her face) is the still's, and nothing speaks of facial expressions or singing: her face is not replaced (mrarc's
second_hp_5461b, 2026-10-15: the text of a hair load said her face was the still's, and its plan look read a face drawn on her
crown). Nor does it carry the plan's `looks` ("blue-grey eyes"): eye words invite a face where none is drawn, and <Picture 1>
carries who she is (the lead's ruling of 2026-10-15); every face form keeps them, word for word. A part that changes inside one load (`--parts-on`, 1010's 1025-1031 hair alone) cannot be said: the window has one text
and the model reads no frame numbers in it, so such a load's text is its own part's.

**What it cannot see.** The picture: whether the sentence a person writes is what the shot shows; whether a drawn part
reads on the reference (`bench/head_reference.py`'s sheet). It reads records and words.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The drawing's part names (`draw_face_points.PARTS`) as the text names them, in the order it names them.
PART_WORDS = (("eyes", "eyes"), ("brows", "eyebrows"), ("nose", "nose"), ("mouth", "mouth"))
#: The eyes' clause by the iris form (`draw_face_points.IRIS_FORMS`).
EYES = {"whole": "the eyes are outlined with the iris inside them",
        "between the lids": "the eyes are outlined with the iris inside them",
        "filled": "the eyes are outlined with a dark iris inside them",
        "contrast": "inside each eye's outline the white of the eye is white and the iris black"}
#: What of the still is not used, by the plan's region, where the subject's `text.not_used` does not say.
NOT_USED = {"face": "the hair, the clothing and the background", "face and hair": "the clothing, the accessories and the background",
            "hair": "the face, the clothing, the accessories and the background"}
#: What a load renders, by its own part (`job.part.parts`) where it carries one: the hair alone, or the face alone.
RENDERS = {frozenset({"hair"}): "hair", frozenset({"face_and_neck"}): "face", frozenset({"hair", "face_and_neck"}): "face and hair"}
SUBJECT_2_KEPT = ("<Subject 2> (appears in [Shot 1]): attribute_transfer - {moves} are transferred to "
                  "<Subject 1>; the grey body model's look, the drawn outlines themselves and its black ground are not used.")
#: What <Subject 1> follows of the figure: the head and the face, or the head alone where only the hair is replaced.
MOVES = {"face": "the movements of the head and the facial expressions", "face and hair": "the movements of the head and the facial expressions",
         "hair": "the movements of the head"}
SOUND = "overall_soundscape: Natural ambient atmosphere continues throughout the shot.\n\nnon_diegetic_music: A song plays throughout.\n"


class Refused(Exception):
    pass


def joined(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def subject_2(drawn: dict | None, moves: str = MOVES["face"]) -> str:
    """`<Subject 2>`'s sentence from the reference record's `drawn` block; None `drawn` is a plain cut."""
    head = (f"<Subject 2> is {moves} performed by the figure in <Video 1>: a closer view of the head "
            "and shoulders of a plain grey body model of a person on a black ground, posed as the person moves")
    if drawn is None:
        return head + "."
    left = set(drawn.get("parts_left_out") or [])
    mouth = (drawn.get("mouth") or {}).get("drawn")
    if mouth is not None:
        left = left - {"mouth"} if mouth else left | {"mouth"}
    parts = [w for k, w in PART_WORDS if k not in left]
    if not parts:
        return head + "."
    clauses = []
    if "eyes" in parts:
        iris = drawn.get("iris") or "whole"
        if iris not in EYES:
            raise Refused(f"the reference's record says the iris was drawn {iris!r}: no words for it ({', '.join(EYES)})")
        clauses.append(EYES[iris] + ", and they open, close and look where the person's eyes do")
    if "mouth" in parts:
        clauses.append("the mouth is outlined" + (", dark inside where it is open" if (drawn.get("mouth") or {}).get("inside") == "the class map" else "")
                       + ", and it opens and closes where the person's mouth does")
    return head + f", with its {joined(parts)} drawn on its face as thin white outlines" + "".join(f"; {c}" for c in clauses) + "."


def text(plan: dict, subject: str, record: dict, place_and_action: str, voiced: list[list[int]], first: int, last: int,
         parts: list[str] | None = None) -> str:
    """The whole text of one load; `parts`, the load's own part where it carries one (`job.part.parts`)."""
    p = plan["subjects"][subject]
    t = p.get("text") or {}
    lacking = [k for k in ("who", "looks", "her", "she") if not str(t.get(k) or "").strip()]
    film = str((plan.get("text") or {}).get("film") or "").strip()
    if lacking or not film:
        raise Refused(f"the plan's `text` for {subject!r} lacks {lacking or ''}{' and ' if lacking and not film else ''}"
                      f"{'' if film else 'the video`s `text.film` line'}: written once in the plan, read on every load")
    region = p["region"]
    if region not in NOT_USED:
        raise Refused(f"the plan's region for {subject!r} is {region!r}: this text is for a face or a face-and-hair load")
    if parts:
        if frozenset(parts) not in RENDERS:
            raise Refused(f"the load's part is {'+'.join(parts)}: no text form for it ({', '.join('+'.join(sorted(k)) for k in RENDERS)})")
        region = RENDERS[frozenset(parts)]
    if region in ("face and hair", "hair") and not str(t.get("hair") or "").strip():
        raise Refused(f"{subject!r}'s load keeps her hair and the plan's `text.hair` is not written: the retention line names the hair kept")
    sentence = " ".join(str(place_and_action or "").split()).rstrip(".")
    if not sentence:
        raise Refused("the place-and-action sentence is a person's, and it is empty")
    who, looks, her, she = (" ".join(str(t[k]).split()) for k in ("who", "looks", "her", "she"))
    noun = who.split(" ", 1)[1] if who.split(" ", 1)[0] in ("the", "a", "an") else who
    with_what = {"face and hair": "the face and the hair", "hair": "the hair"}.get(region, "the face")
    hair = " ".join(str(t.get("hair") or "").split())
    not_used = " ".join(str((t.get("not_used") if region != "hair" else None) or NOT_USED[region]).split())
    kept = (f"only the {noun}'s face and {hair} are retained, wherever they show" if region == "face and hair" else
            f"only the {noun}'s {hair.split(' ', 1)[1] if hair.split(' ', 1)[0] in ('her', 'his', 'their') else hair} is retained, wherever it shows"
            if region == "hair" else
            f"only the {noun}'s face is retained, wherever it shows: {her} eyes, eyebrows, nose, mouth and skin")
    drawn = None if record.get("form") == "plain" else (record.get("drawn") or {})
    if region == "hair" and drawn is not None and not {"eyes", "brows", "nose"} <= set(drawn.get("parts_left_out") or []):
        raise Refused("the load is her hair alone and its reference draws a face: make the reference again with --no-face-on over its frames "
                      "(bench/shot_to_load.py reference does), then the text")
    sings = bool(t.get("sings")) and region != "hair" and any(a <= last and first <= b for a, b in voiced)
    moves = MOVES[region]
    Her = her[0].upper() + her[1:]
    same = f"{Her} hair is the hair of <Picture 1>." if region == "hair" else f"{Her} face is the face of <Picture 1>."
    with_looks = "" if region == "hair" else f", with {looks},"
    return (f"subject_definitions:\n<Subject 1> is {who} whose appearance comes from <Picture 1>{with_looks} and who follows {moves} "
            f"of the figure in <Video 1>.\n{subject_2(drawn, moves)}\n\n"
            f"summary:\n[reference generation] <Subject 1> {sentence}, with {with_what} of <Picture 1>, and performs <Subject 2>.\n\n"
            f"retention_analysis:\n<Subject 1> (appears in [Shot 1]): partially_preserved - {kept}; {not_used} of <Picture 1> are not used.\n"
            f"{SUBJECT_2_KEPT.format(moves=moves)}\n\n"
            f"detailed_description:\n{film}\n[Shot 1] <Subject 1>, {who}{'' if region == 'hair' else f' with {looks}'}, {sentence}. {same} "
            f"<Subject 1> performs <Subject 2>: {she} follows {moves} of the figure."
            + (f" <Subject 1> (S1) sings, {her} mouth opening wide on the sung phrases." if sings else "") + f"\n\n{SOUND}")


def for_job(job_file: Path, place_and_action: str, plan_file: Path, voice_spans: Path) -> tuple[str, dict]:
    """The text of a job file's load, and what it was made from."""
    d = json.loads(Path(job_file).read_text())
    load = d["loads"][0]
    record_path = Path(load["motion_file"]).with_suffix(".json")
    if not record_path.is_file():
        raise Refused(f"the reference's record {record_path} is not there: the text says what the reference shows, and is made after it")
    record = json.loads(record_path.read_text())
    plan = json.loads(Path(plan_file).read_text())
    voiced = json.loads(Path(voice_spans).read_text())["voiced_spans_inclusive"]
    first, last = load["first"], load["first"] + load["frames"] - 1
    parts = ((d.get("job") or {}).get("part") or {}).get("parts")
    words = text(plan, load["subject"], record, place_and_action, voiced, first, last, parts)
    return words, {"reference_record": str(record_path), "plan": str(plan_file), "voice_spans": str(voice_spans),
                   "drawn": {k: (record.get("drawn") or {}).get(k) for k in ("parts_left_out", "iris", "mouth")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    s = sub.add_parser("show", help="print a load's text, made from its records and the sentence given; writes nothing")
    s.add_argument("job_file")
    s.add_argument("--place-and-action", required=True)
    s.add_argument("--plan", required=True)
    s.add_argument("--voice-spans", required=True)
    v = ap.parse_args(argv)
    try:
        words, _ = for_job(Path(v.job_file), v.place_and_action, Path(v.plan), Path(v.voice_spans))
    except Refused as e:
        print(f"REFUSED: {e}")
        return 5
    print(words, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
