#!/usr/bin/env python3
"""A load's text, generated from its reference's record, the plan and the voice table. No place-and-action sentence since 2026-10-15
(`NO_SCENE`: to H3 a masked load is not an edit, and a text never describes the frozen picture).

    <python> bench/load_text.py show JOB.json --plan PLAN.json --voice-spans SPANS.json

(`bench/shot_to_load.py text JOB.json --plan PLAN.json --by SESSION` writes it beside the job file.)

**What it is for.** The owner's word of 2026-10-14: every window's text is consistent with the others, and says exactly
what the reference video shows and the least else. Until then a load's text was copied by hand from load to load, and
its `<Subject 2>` sentence named what an earlier reference drew: once the mouth is drawn by default or the eye is drawn
in contrast, a copied sentence describes a reference that is not there. So the fixed lines are made here, from the
records. Until 2026-10-15 a person wrote the shot's place and action, once; no longer (`NO_SCENE`).

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
- The summary and `[Shot 1]`: who the subject is and what of <Picture 1> it has (draft_7's form, 2026-10-15): "<Subject 1>,
  with the face of <Picture 1>, performs <Subject 2>." and "<Subject 1>, <who> with <looks>, has the face of <Picture 1>.",
  with the face and the hair, or the hair alone, by the plan's region and its `text.hair_change`. No place-and-action sentence.
- The film line: the plan's `text.film`, the video's, written once.
- The singing sentence: the plan's `voice` where it names one on the load ("sings" alone writes a line), else the subject's
  `sings`, on a load whose frames meet a voiced span of the video's voice table.
- Her clothing: the plan's `clothing` entry that holds the load's frames (per shot, 1f143f59), one sentence in the detailed line,
  "<Subject 1> wears ...". A subject
  whose clothing is redrawn with her (the plan's `look.stay_removed_by`, the owner's decision of 2026-10-16, 26f208c0) and not
  named is refused (`clothing_of`, `NO_CLOTHING`); the job step refuses the same, before anything is made.
- `<Subject 2>`'s retention line, the soundscape and the music: fixed, the kept loads' words.

**The form is the kept loads'**, word for word but where a record or the plan says otherwise: lead_fp_1730 and
lead_fp_1755 (10/13; the same text), second_hfp_0484c (10/14, face and hair). Places differ from them, by ruling: no
place-and-action sentence, the summary and [Shot 1] in draft_7's form (2026-10-15), and the closer view is "of the head and
shoulders" (theirs named arms or hands, typed by sight). `bench/check_load_text.py` holds the generator to
those texts.

**A load of the hair alone** (`--parts hair`, the job's `part.parts`): its words are the hair's, not the plan's region's. The
summary and the retention line name the hair alone, the face is among what of the still is not used, the detailed line says her
hair (not her face) is the still's, and nothing speaks of facial expressions or singing: her face is not replaced (mrarc's
second_hp_5461b, 2026-10-15: the text of a hair load said her face was the still's, and its plan look read a face drawn on her
crown). Nor does it carry the plan's `looks` ("blue-grey eyes"): eye words invite a face where none is drawn, and <Picture 1>
carries who she is (the lead's ruling of 2026-10-15); every face form keeps them, word for word. A part that changes inside one load (`--parts-on`, 1010's 1025-1031 hair alone) cannot be said: the window has one text
and the model reads no frame numbers in it, so such a load's text is its own part's.

**A load with no reference video** (`motion: none`, the lead's ruling of 2026-10-15, one test on second 5461-5497: her hair alone,
her face hidden, the body model guessing her head upright while she is bowed): no `<Video 1>`, no `<Subject 2>`, nothing of
following movements; only her hair from <Picture 1>, on the source's own picture. Refused on any load but one of the hair alone.

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


#: The refusal of a subject whose clothing is redrawn with her and not named (`clothing_of`).
NO_CLOTHING = ("the owner's decision of 2026-10-16 (26f208c0): her clothing is redrawn with her and named in the text; give the plan's "
               "`clothing`")


def clothing_of(p: dict, subject: str, first: int, last: int) -> str | None:
    """The plan's words for a subject's clothing ON THESE FRAMES, or None where the plan names none for the subject. Clothing is per shot
    (the lead's ruling of 2026-10-16, decisions.md 1f143f59: one apron line for the mom made H3 draw straps and a bib on 1010-1038, where she
    wears a dark sleeveless top): the plan's `clothing` is a list of {first, last, words, by}, each from a look at that shot's own frames, and
    a load takes the one entry that holds all of its frames (`words: null`, with a `by` that says why, names nothing there). Refused, the
    whole list checked whenever it is read: the old single set of words and a leftover `clothing_by`; an entry that is not a dict, lacks
    any of the four, has a first or last that is not a whole number, runs backwards or has empty words; two entries that overlap anywhere;
    a subject that names clothing for some shots and none for these frames (never another shot's words);
    and a subject whose clothing is redrawn with her (the plan's look says her stay keep-out was removed, `look.stay_removed_by`, 26f208c0)
    and named for none of these frames. The words are the plan's, never the tool's.

    THE KEY IS `look.stay_removed_by`, the one record of the owner's choice (the lead's ruling of 2026-10-16). It stands in for the test
    the rule means, "a region over her clothing", which no record says before a capture is made."""
    if "clothing_by" in p:
        raise Refused(f"the plan's `clothing_by` for {subject!r} is the old form, one `by` for every shot: each entry of `clothing` carries its own `by`")
    given = p.get("clothing")
    if given is not None and not isinstance(given, list):
        raise Refused(f"the plan's `clothing` for {subject!r} is one set of words for every shot; clothing is per shot (1f143f59): give a list "
                      "of {first, last, words, by}, each from a look at that shot's own frames")
    rows = given or []
    # THE WHOLE LIST, whenever it is read (the clothing read, the lead's ruling of 2026-10-16), not only the entry for the load in hand
    for r in rows:
        if not isinstance(r, dict):
            raise Refused(f"an entry of the plan's `clothing` for {subject!r} is not a {{first, last, words, by}}: {r!r}")
        lacking = [k for k in ("first", "last", "words", "by") if k not in r] + [k for k in ("by",) if k in r and not str(r[k] or "").strip()]
        if lacking:
            raise Refused(f"an entry of the plan's `clothing` for {subject!r} lacks {', '.join(lacking)}: {r}")
        if any(isinstance(r[k], bool) or not isinstance(r[k], int) for k in ("first", "last")):
            raise Refused(f"an entry of the plan's `clothing` for {subject!r} has a first or last that is not a whole frame number: {r}")
        if r["first"] > r["last"]:
            raise Refused(f"an entry of the plan's `clothing` for {subject!r} runs backwards, {r['first']} after {r['last']}: {r}")
        # nothing to name is said as `words: null`, with a `by` that says why from a look at those frames; never an empty string
        if r["words"] is not None and (not isinstance(r["words"], str) or not r["words"].strip()):
            raise Refused(f"an entry of the plan's `clothing` for {subject!r} has empty words: say `words: null` with a `by` that says why: {r}")
    ordered = sorted(rows, key=lambda r: r["first"])
    for x, y in zip(ordered, ordered[1:]):
        if y["first"] <= x["last"]:
            raise Refused(f"two entries of the plan's `clothing` for {subject!r} overlap on {y['first']}-{min(x['last'], y['last'])}: {x} and {y}")
    hit = [r for r in rows if r["first"] <= int(first) and int(last) <= r["last"]]
    if hit:
        return None if hit[0]["words"] is None else " ".join(str(hit[0]["words"]).split())
    if rows:
        raise Refused(f"the plan's `clothing` for {subject!r} names no words for frames {first}-{last} (it names "
                      + ", ".join(f"{r['first']}-{r['last']}" for r in rows) + "): words for this shot come only from a look at its own frames, "
                      "never another shot's")
    if (p.get("look") or {}).get("stay_removed_by"):
        raise Refused(f"{subject!r}, frames {first}-{last}: " + NO_CLOTHING)
    return None


def joined(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def subject_2(drawn: dict | None, moves: str = MOVES["face"], frames: int | None = None) -> str:
    """`<Subject 2>`'s sentence from the reference record's `drawn` block; None `drawn` is a plain cut. The mouth is said only where the
    record's frames draw one (the 5114e and 1010j fault, 2026-10-15: the form was "auto" while no frame drew a mouth): none where it
    draws none (`mouth.frames_with_no_mouth` of the `frames` it holds), and "on some frames" where it draws one on part of them."""
    head = (f"<Subject 2> is {moves} performed by the figure in <Video 1>: a closer view of the head "
            "and shoulders of a plain grey body model of a person on a black ground, posed as the person moves")
    if drawn is None:
        return head + "."
    left = set(drawn.get("parts_left_out") or [])
    mouth = (drawn.get("mouth") or {}).get("drawn")
    bare_frames = (drawn.get("mouth") or {}).get("frames_with_no_mouth")
    faceless = len(drawn.get("frames_left_bare") or [])           # frames with no line of the face (--no-face-on): no mouth there either
    if isinstance(bare_frames, int) and frames:
        bare_frames, frames = bare_frames - faceless, frames - faceless   # the mouth, counted on the frames that draw a face
    some = False
    if mouth and frames and isinstance(bare_frames, int):
        if bare_frames >= frames:
            mouth = False                                   # the form drew a mouth on no frame of the load
        elif bare_frames > 0:
            some = True
    if mouth is not None:
        left = left - {"mouth"} if mouth and not some else left | {"mouth"}
    parts = [w for k, w in PART_WORDS if k not in left]
    if not parts and not some:
        return head + "."
    clauses = []
    if "eyes" in parts:
        iris = drawn.get("iris") or "whole"
        if iris not in EYES:
            raise Refused(f"the reference's record says the iris was drawn {iris!r}: no words for it ({', '.join(EYES)})")
        clauses.append(EYES[iris] + ", and they open, close and look where the person's eyes do"
                       + {"lids": "; where the person's eyes are shut, each eye is drawn as a single closed line",
                          "none": "; where the person's eyes are shut, no eye is drawn"}.get((drawn.get("eyes_shut") or {}).get("form"), ""))
    inside = ", dark inside where it is open" if (drawn.get("mouth") or {}).get("inside") == "the class map" else ""
    if "mouth" in parts:
        clauses.append("the mouth is outlined" + inside + ", and it opens and closes where the person's mouth does")
    elif some:
        clauses.append("on some frames the mouth is outlined too" + inside + ", and there it opens and closes where the person's mouth does")
    drawn_parts = f", with its {joined(parts)} drawn on its face as thin white outlines" if parts else ""
    return head + drawn_parts + "".join(f"; {c}" for c in clauses) + "."


#: A SUBJECT'S VOICE ON A SPAN OF FRAMES, from the plan (`subjects.<label>.voice`: [{"first", "last", "voice", "by"}]; the owner's word
#: of 2026-10-15 on second_hfp_5675b, "Subject 2 shouldnt be singing ... she's mouthing like she's talking but its not audible", and on
#: 1306, where the father does sing the lead singer's words). Where a span overlaps a load it stands in place of the voice table's
#: guess: "sings" (the sung line, whatever the table hears), "talks, not heard" or "silent" (no sung line, and the words below).
#: Only "sings" writes a line (the sung line); "talks, not heard" and "silent" write nothing of talking or singing (the owner, 2026-10-15:
#: "just dont mention anything about the mom talking or singing"): the plan holds who sings as data, and the text says only that.
VOICES = {"sings": "", "talks, not heard": "", "silent": ""}
#: A text says who is generated in each hole and from which reference, never the frozen picture and never as an edit (AGENTS.md,
#: "Settled about H3", made core 2026-10-15; docs/wiki/masked_v2v.md, "The frame every masked text is written in"): so no
#: place-and-action sentence (where she stands or sits, what she does, the room) is written into a generated text.
NO_SCENE = ("a generated text takes no place-and-action sentence: to H3 a masked load is not an edit, and a text says who is generated "
            "in the hole and from which reference, never the frozen picture (where a person stands or sits, what she wears, the room); "
            "AGENTS.md, \"Settled about H3\". Write the text --by-hand if a sentence is truly wanted")


def voice_of(p: dict, first: int, last: int) -> dict | None:
    """The plan's voice for a subject over a load's frames (`VOICES`), or None where the plan names none there. Refused where two voices
    overlap the load (one load is one voice: cut it where the voice changes), or a span says a voice there are no words for, or no who."""
    for v in p.get("voice") or []:
        if not isinstance(v, dict) or not all(isinstance(v.get(k), int) for k in ("first", "last")) or v["first"] > v["last"]:
            raise Refused(f"the plan's voice entry {v!r} is not {{\"first\": F, \"last\": L, \"voice\": ..., \"by\": ...}} with F <= L")
    spans = [v for v in p.get("voice") or [] if int(v["first"]) <= last and first <= int(v["last"])]
    for v in spans:
        if v.get("voice") not in VOICES:
            raise Refused(f"the plan's voice on {v['first']}-{v['last']} is {v.get('voice')!r}: one of {', '.join(VOICES)}")
        if not str(v.get("by") or "").strip():
            raise Refused(f"the plan's voice on {v['first']}-{v['last']} says no `by`: who said so, and when")
    if len({v["voice"] for v in spans}) > 1:
        raise Refused(f"the plan names {len({v['voice'] for v in spans})} voices over {first}-{last}: "
                      + "; ".join(f"{v['voice']} on {v['first']}-{v['last']}" for v in spans) + ". One load is one voice: cut it where it changes")
    return spans[0] if spans else None


def text(plan: dict, subject: str, record: dict | None, place_and_action: str, voiced: list[list[int]], first: int, last: int,
         parts: list[str] | None = None) -> str:
    """The whole text of one load; `parts`, the load's own part where it carries one (`job.part.parts`); `record` None for a load with no
    reference video (`motion: none`)."""
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
    if str(place_and_action or "").strip():
        raise Refused(NO_SCENE)
    who, looks, her, she = (" ".join(str(t[k]).split()) for k in ("who", "looks", "her", "she"))
    # HER CLOTHING, NAMED IN THE TEXT where the plan names it (the owner's decision of 2026-10-16, 26f208c0: redrawn with her, named so
    # H3 draws the same clothes): one sentence in the detailed line, the plan's words
    clothing = clothing_of(p, subject, first, last)
    wears = f" <Subject 1> wears {clothing}." if clothing else ""
    noun = who.split(" ", 1)[1] if who.split(" ", 1)[0] in ("the", "a", "an") else who
    with_what = {"face and hair": "the face and the hair", "hair": "the hair"}.get(region, "the face")
    hair = " ".join(str(t.get("hair") or "").split())
    not_used = " ".join(str((t.get("not_used") if region != "hair" else None) or NOT_USED[region]).split())
    # HER HAIR CHANGED FROM THE STILL'S (the plan's `text.hair_change`, the owner's word of 2026-10-15 on the second subject: her still has
    # long blonde hair; keep its colour, change only its length): the release's guide (vendor_guides/ref_en.md, section 4.1) has
    # `partially_preserved` as "some defined characteristics are changed or only partially retained", so the retention line says which
    # of the still's hair is kept and what is changed, and the summary takes the face from <Picture 1> and the plan's hair; "retained"
    # alone would tell H3 the still's hair is the plan's. The parts are the plan's words, never parsed out of a sentence: `kept` (what of
    # the still's hair stays, or none), `what` (the characteristic changed) and `to` (what it becomes), with `by`. Every other line of the
    # form is as it was
    change = t.get("hair_change") if region in ("face and hair", "hair") else None
    if change is not None and (not isinstance(change, dict) or not str(change.get("what") or "").strip() or not str(change.get("to") or "").strip()
                               or not str(change.get("by") or "").strip()):
        raise Refused(f"the plan's `text.hair_change` for {subject!r} names `what` is changed, `to` what, and `by` whom; got {change!r}")
    bare = hair.split(" ", 1)[1] if hair.split(" ", 1)[0] in ("her", "his", "their") else hair
    if change is not None:
        kept_of = " ".join(str(change.get("kept") or "").split())
        changed_to = f"{her} hair's {' '.join(str(change['what']).split())} is changed to " + " ".join(str(change["to"]).split())
        if region == "hair":                            # her hair alone: what of the still's hair is kept, and what is changed
            with_what = bare
            kept_now = (f"{kept_of} of the {noun}'s hair from <Picture 1> is retained; " if kept_of else "") + changed_to
        else:
            with_what = f"the face of <Picture 1> and {bare}"
            kept_now = (f"the {noun}'s face" + (f" and {kept_of} of {her} hair" if kept_of else "") + f" from <Picture 1> "
                        + ("are" if kept_of else "is") + f" retained; " + changed_to)
    kept = (kept_now if change is not None else
            f"only the {noun}'s face and {hair} are retained, wherever they show" if region == "face and hair" else
            f"only the {noun}'s {hair.split(' ', 1)[1] if hair.split(' ', 1)[0] in ('her', 'his', 'their') else hair} is retained, wherever it shows"
            if region == "hair" else
            f"only the {noun}'s face is retained, wherever it shows: {her} eyes, eyebrows, nose, mouth and skin")
    if record is None:
        if region != "hair":
            raise Refused("a load with no reference video (motion none) is for her hair alone, the lead's ruling of 2026-10-15")
        return (f"subject_definitions:\n<Subject 1> is {who} whose hair comes from <Picture 1>.\n\n"
                f"summary:\n[reference generation] <Subject 1> has {bare if change is not None else 'the hair of <Picture 1>'}.\n\n"
                f"retention_analysis:\n<Subject 1> (appears in [Shot 1]): partially_preserved - {kept}; {not_used} of <Picture 1> are not used.\n\n"
                f"detailed_description:\n{film}\n[Shot 1] <Subject 1>, {who}, has {bare if change is not None else 'the hair of <Picture 1>'}.{wears}"
                f"\n\n{SOUND}")
    drawn = None if record.get("form") == "plain" else (record.get("drawn") or {})
    if region == "hair" and drawn is not None and not {"eyes", "brows", "nose"} <= set(drawn.get("parts_left_out") or []):
        raise Refused("the load is her hair alone and its reference draws a face: make the reference again with --no-face-on over its frames "
                      "(bench/shot_to_load.py reference does), then the text")
    sings = bool(t.get("sings")) and region != "hair" and any(a <= last and first <= b for a, b in voiced)
    said = voice_of(p, first, last) if region != "hair" else None
    if said is not None:
        sings = said["voice"] == "sings"
    moves = MOVES[region]
    Her = her[0].upper() + her[1:]
    has = f"has {bare if change is not None else 'the hair of <Picture 1>'}" if region == "hair" else "has the face of <Picture 1>"
    with_looks = "" if region == "hair" else f", with {looks},"
    return (f"subject_definitions:\n<Subject 1> is {who} whose appearance comes from <Picture 1>{with_looks} and who follows {moves} "
            f"of the figure in <Video 1>.\n{subject_2(drawn, moves, (record['frames'][1] - record['frames'][0] + 1) if record.get('frames') else last - first + 1)}\n\n"
            f"summary:\n[reference generation] <Subject 1>, with {with_what}{'' if change is not None else ' of <Picture 1>'}, performs <Subject 2>.\n\n"
            f"retention_analysis:\n<Subject 1> (appears in [Shot 1]): partially_preserved - {kept}; {not_used} of <Picture 1> are not used.\n"
            f"{SUBJECT_2_KEPT.format(moves=moves)}\n\n"
            f"detailed_description:\n{film}\n[Shot 1] <Subject 1>, {who}{'' if region == 'hair' else f' with {bare}' if change is not None else f' with {looks}'}, {has}.{wears} "
            f"<Subject 1> performs <Subject 2>: {she} follows {moves} of the figure."
            + (f" <Subject 1> (S1) sings, {her} mouth opening wide on the sung phrases." if sings else "")
            + (VOICES[said["voice"]].format(her=her) if said is not None else "") + f"\n\n{SOUND}")


def for_job(job_file: Path, place_and_action: str, plan_file: Path, voice_spans: Path) -> tuple[str, dict]:
    # `place_and_action`: none since 2026-10-15 (`NO_SCENE`); an empty one is taken, a sentence refused
    """The text of a job file's load, and what it was made from."""
    d = json.loads(Path(job_file).read_text())
    load = d["loads"][0]
    if load.get("motion") == "none":
        record_path, record = None, None                # no reference video wired (masked_job_graph.motion_of): the text names none
    else:
        record_path = Path(load["motion_file"]).with_suffix(".json")
        if not record_path.is_file():
            raise Refused(f"the reference's record {record_path} is not there: the text says what the reference shows, and is made after it")
        record = json.loads(record_path.read_text())
    plan = json.loads(Path(plan_file).read_text())
    voiced = json.loads(Path(voice_spans).read_text())["voiced_spans_inclusive"]
    first, last = load["first"], load["first"] + load["frames"] - 1
    parts = ((d.get("job") or {}).get("part") or {}).get("parts")
    words = text(plan, load["subject"], record, place_and_action, voiced, first, last, parts)
    return words, {"reference_record": str(record_path) if record_path else "none: the load takes no reference video (motion none)",
                   "plan": str(plan_file), "voice_spans": str(voice_spans),
                   "drawn": {k: ((record or {}).get("drawn") or {}).get(k) for k in ("parts_left_out", "iris", "mouth")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    s = sub.add_parser("show", help="print a load's text, made from its records, the plan and the voice table; writes nothing")
    s.add_argument("job_file")
    s.add_argument("--place-and-action", default="", help="none: refused if given (NO_SCENE)")
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
