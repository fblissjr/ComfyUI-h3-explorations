#!/usr/bin/env python3
"""The delivery table, generated from the render ledger by one stated rule: which render stands on each shot, and why.

    <python> bench/delivery_table.py --ledger LEDGER.json --shots SHOTS.json --plan PLAN.json \\
        [--prefer "FIRST-LAST:SUBJECT=LOAD:the reason"]... [--hold "FIRST-LAST:SUBJECT=LOAD:the reason"]... \\
        [--span FIRST-LAST] [--assemble-args "..."] [--also-capture FOLDER]... \\
        [--stack-work SCRATCH_FOLDER] --table OUT.txt --reading OUT.md [--json OUT.json]

**What it buys.** The table `bench/assemble_delivery.py` builds a clip from was typed by hand at the end of a day,
from memory of which render was the good one. This writes it from files: the ledger (`bench/render_ledger.py`:
every render, how it was made, the owner's lines on it), the shots file and the plan. Beside the table it writes
a reading that says, for every shot and subject, which render stands and why in one line, and every render it
passed over with why. Under each render it names, the reading prints the owner's typed words on that render,
whole, with when and where the line is (2026-10-13: the owner asked that their words are read with their clicks,
and this is the page a lead reads before a build). The words decide nothing here: the rule reads choices. It
assembles nothing.

**The rule for "stands", in this order. Each step is a fact from the files, never a judgement of a picture.**

0. `--prefer`: a render the lead names for a shot and a subject, with a reason that is printed on its line. It
   goes first among the renders that MAY stand; it cannot make one stand that step 3 rules out, or one the
   owner's last choice says is not the one.
   `--hold` is its mirror: a render the lead names as NOT to be laid on a shot for a subject, with a reason that
   is printed where the render is passed over, in the table's header and in the reading's first paragraph. It is
   A SESSION'S WORD, said to be one each time it is printed, and it writes no verdict anywhere: the owner's own
   choices are read from the review page and nothing here adds to them. The owner's word is over it: a hold on
   a render whose last choice by the owner is a keep is refused. A shot left with nothing standing by a hold says
   so on its `source` line, since "no render in the frozen form yet" would not be true of it. (Added 2026-10-13:
   a render read by a blind reader as not one face across its shot was the only render of that shot, and stood.)
1. The owner's keep (`KEEP`: the last choice the owner made on that render is one of those words; a line said
   of a joined clip is not a choice on a render) on a render whose file agrees with its build record.
2. Else the newest render whose file agrees with its build record, whose last choice by the owner does not say
   it is not the one (`NOT_THE_ONE`; `not sure` bars nothing), and whose form is the frozen one for that
   subject: the plan's region for the subject, and a reference whose own records say it is the body model cut
   to the head with the face points drawn. The plain body model cut to the head (no points, no marks) is the
   form's stated exception for a face under the drawing line; no field this tool reads says why a load took
   it, so such a render stands here and its line says the reason was not read.
   A render whose JOB named its part as some of the plan's parts for the subject (the hair alone where no face
   shows: the owner's answer of 2026-10-13; the ledger's `part_named`) is taken as the plan's region, and its
   line says which parts and on whose word. Its capture's part holds no face class and reads `a part`; held to
   the region's word alone, the table laid the source exactly where the owner had asked for the new hair.
3. Never: a render with no build record, one whose queue record names no file, one whose file disagrees with
   its record, a whole-person render of any subject, a render of two subjects in one pass (not laid by this
   tool). The owner's rule of 2026-10-13: a piece whose making cannot be said is rendered again.
4. Nothing left: a `source` line for the shot saying no render in the frozen form exists yet, and the reading
   names the newest render that was passed over, which is not laid.

**A subject the shots file says is NOT in a shot gets a `source` line too**, saying so and who said it
(2026-10-13). The assembler does not read the shots file: where a capture's shot table counts people in a shot
and nothing is laid or tracked under a label, it fails the build until a row or a `source` line answers for that
label. The first clip built from a generated table failed sixteen times on exactly that, every one a subject
the shots file had answered `no` for.

On one shot several renders can stand on different frames. They are taken in the order above, and each later
one is laid only on frames no earlier one took. Frames of a shot that no render takes are a `source` line when
they are the whole shot (the assembler takes a `source` line only over a whole shot) and a comment in the table
otherwise.

**A plan's `restore`** (a subject's `restore: [{"classes": [...], "first": F, "last": L, "exception": WHY}]` with
`restore_by`; the owner's rule of 2026-10-14, keep accessories, made an EXCEPTION ONLY on 2026-10-16: a span with no
`exception` in words is refused, since nothing puts the original back inside a person's region, and each span's reason
is in the header): `restore=<subject>.<Class>+...` on that subject's rows on those source
frames ONLY, a row cut at a span's edges, so the assembler gives those classes of their own class map back to the
source there (sunglasses inside a face region). Named frames, never every row: the class map calls the lead's eye
makeup Eyeglass on shots without glasses (0294d at 394, 1802b at 1843) and a telephone receiver too (1575b at 1653),
and a restore there would paste the original back into a laid piece. Generated here, never typed per row; the
header says who asked. A class the part model has not is refused. A span with `"touching": "head"` writes
`restore=<subject>.<Class>@head`, whose pieces are given back only where they touch the subject's own head: her
earrings, which the part model labels Apparel with her collar (the lead's ruling of 2026-10-15), on every row of hers.
A thing worn on the head (Eyeglass, Apparel `@head`) is placed by the assembler on the piece's own face (the owner,
2026-10-23; `bench/assemble_delivery.py`, `WORN_ON_THE_HEAD`); a span with `"placed": "as filmed"` and `placed_by`
(whose word and why) keeps the film's place there, written `:filmed` on the token and said in the header. Any other
`placed`, or one with no `placed_by`, is refused.

**`restore=`** follows the practice of the table of 2026-10-12, as its header states it: a subject's row gives
back every OTHER subject the shots file has in that shot (`yes` or `unsure`): `restore=<other>:whole` on a face
row, which has no business inside anybody else, and `restore=<other>` on any other row. The assembler needs a
capture that holds each named subject's track to act on one.

**`run=<capture folder>:<run name>`** on a row ties the piece to the run that made it (the lead's ruling of
2026-10-13; `bench/assemble_delivery.py` has what the tie buys). It is written from records and never found by a
name: the capture is the one the piece's build record names, the run is the one in that capture's manifest named
as the record's load, and it must be of the row's subject. A row that cannot be tied has no `run=` and the
reading says which lack it was.

**The names** are the plan's labels. The plan's `aliases` are printed in the table's header as the `--alias`
flags a capture with an older name needs.

**The commands are printed whole, in order, so a second run types nothing** (`assembly`; in the reading, the
JSON and on the terminal), each as one line a shell takes as it stands, run from the repo's root: the ledger's
own (kept in the ledger's JSON by `bench/render_ledger.py`; an older ledger does not have it, and that is said),
this tool's own, the assembler's, and the stacking tool's build and check. The assembler's: the table, the span,
every capture the standing pieces' build records name, any handed in with `--also-capture`, an `--alias` only
for an older name a listed capture uses, and `--assemble-args` (`--source`, `--out`, `--size`: what only the
caller knows). The stack's is made from that `--source` and `--out` and is whole only with `--stack-work`.

**What each `restore=` has to go on** is printed beside the commands as three counts of its row's frames: those on
which a listed capture holds that subject's track (the restore can give pixels back); those a listed capture
covers with an EMPTY track (the restore gives nothing back, and the assembler raises no flag: by its rule nobody
is there to give back); and those no listed capture covers at all (the assembler's `restore_has_no_mask`). Until
2026-10-13 one count stood for the last two and the text said the assembler would flag them; it flags only the
third (`data/2026-10-13_fun_delivery_tables/read_fun_0000_1919_generated_THROWAWAY_2026-10-13.md`).

**What it cannot see.** Whether a render is good. Whether the render the rule lets stand is the one the owner
would pick: step 2 lets the newest frozen render stand where the owner has made no choice on it or was not sure,
and the reading prints their last choice beside it. Why a load took the plain cut. Whether a restore's subject is
on screen where no capture tracks them. Whether the shots file is right that a subject is not in a shot. Whether
the mask a tied run carried is the one its render was given (the load's own chain holds that). A render made
after the ledger was: the reading says which ledger it read.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER_SCHEMA = "h3_render_ledger/2"
#: The owner's choices that are a keep, as the review page's buttons word them: `keep this one` is the button the
#: sessions' items carry, `good to go` the page's own positive button where an item names no options
#: (`data/<video>/poc/index.html`, the default options; the lead's ruling of 2026-10-13 that it is a keep).
KEEP = ("keep this one", "good to go")
#: The owner's choices that say a render is NOT the one: a render whose last choice is one of these does not stand
#: by step 2 or by `--prefer` (the lead's ruling, 2026-10-13). The words are the buttons the review page has shown
#: under a render (`data/<video>/review/items.json`, each item's `options`, as they stood that day); `not sure` is
#: not among them and bars nothing. A button worded otherwise bars nothing until it is added here.
NOT_THE_ONE = ("wrong", "wrong, and I say where", "not needed", "worse, and I say where")
#: The page's word for a choice withdrawn: the owner then has no choice on the render.
TAKEN_BACK = "taken back"
#: The buttons a render's item is given on the review page (`bench/load_chain.py page`): a keep, a not-the-one, and
#: the one that is neither. Kept here, beside the words this tool reads, so the page shows what the table reads.
NOT_SURE = "not sure"
RENDER_ITEM_BUTTONS = (KEEP[0], NOT_THE_ONE[1], NOT_SURE)
#: The ledger's states of a making (`render_ledger.MAKING`), by what each means here.
AGREES = "recorded, the file agrees"
NO_SOURCE_YET = "no render in the frozen form yet"
#: The source line's words for a subject a person's look says has nothing of the plan's parts on the shot
#: (`shot_coverage.nothing_to_make`); its own state in the coverage table, and not a gap in `bench/piece_gaps.py`.
NOTHING_TO_MAKE = "present, nothing to make under this plan, by a person's look"
#: A shot whose only renders that may stand are held back by the lead (`--hold`): a render exists, and is not laid.
HELD_BACK = "a render that may stand is HELD BACK by the lead (--hold), a session's word and not the owner's, and nothing else may stand"
NOT_IN_THE_SHOT = "the shots file says this subject is not in the shot"
#: With `--laid` (the owner's order of 2026-10-16): a shot and subject the list of what is laid names nothing for.
NOT_LAID = "nothing is laid here in the list of what is laid"


def _shot_coverage():
    import importlib.util
    spec = importlib.util.spec_from_file_location("shot_coverage", HERE / "shot_coverage.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def last_choice(render: dict) -> dict | None:
    """The owner's last CHOICE on a render (a button, not typed words), or None. A line said of a joined clip is
    of that clip and is not a choice on the render, but for one: a KEEP on a joined item of the review page, the whole
    clip, counts as a keep on each piece it lays (the lead's ruling, 2026-10-14: the owner judges the final cut whole;
    the ledger marks it `the_whole_clip`). A "wrong" or "not sure" there names no piece and counts on none."""
    choices = [o for o in render.get("owner") or [] if o.get("what") == "choice"
               and (not o.get("on_the_joined_clip") or (o.get("the_whole_clip") and o.get("said") in KEEP))]
    last = max(choices, key=lambda o: str(o.get("at"))) if choices else None
    return None if last is None or last.get("said") == TAKEN_BACK else last


def owners_words(render: dict) -> list[dict]:
    """The owner's typed words ON A RENDER, oldest first, as the ledger holds them: when, the frame they named, the
    words whole and where the line is. Words said of a joined clip are of that clip and are not here."""
    said = [o for o in render.get("owner") or [] if o.get("what") == "words" and not o.get("on_the_joined_clip") and str(o.get("said") or "").strip()]
    return [{"at": o.get("at"), "frame": o.get("frame"), "said": str(o["said"]), "from": o.get("from"), "row": o.get("row")}
            for o in sorted(said, key=lambda o: str(o.get("at")))]


def words_lines(words: list[dict]) -> list[str]:
    """The reading's lines for a render's words: each whole, on one line (a line break of the owner's is ` / `, as the
    ledger's reading writes it), with its time, the frame it named and where the line is."""
    frame = lambda w: "" if w.get("frame") in (None, "", "None") else f" at frame {w['frame']}"          # noqa: E731
    return [f"  - the owner's words on it, {str(w.get('at'))[:16]}{frame(w)}: {' / '.join(part.strip() for part in w['said'].splitlines())}"
            + (f"  [{w['from']}" + (f", row {w['row']}" if w.get("row") else "") + "]" if w.get("from") else "") for w in words]


def never_stands(render: dict) -> str | None:
    """Why a render never stands (step 3), or None."""
    making = render["making"]
    if making.startswith("NO BUILD RECORD"):
        return "no build record: its making cannot be said"
    if "NAMES NO FILE" in making:
        return "its queue record names no file and no outcome: its making cannot be said from the record"
    if making != AGREES:
        said = (render["records"][-1]["disagrees"] if render.get("records") else None) or ["(the ledger gives no line)"]
        return "its file disagrees with its build record: " + "; ".join(said)
    if (render["form"].get("region")) == "whole":
        return "a whole-person render"
    if len(render.get("subjects") or []) > 1:
        return "a render of two subjects in one pass: not laid by this tool"
    return None


def form_of(render: dict, region_wanted: str, plan_parts: list[str] | None = None) -> tuple[str | None, str]:
    """(`frozen`, `frozen, by its exception`, or None, with the words): is a render in the frozen form for its subject.

    A render whose job NAMED its part as some of the plan's parts for the subject (the ledger's `part_named`: the
    hair alone where no face shows, the owner's answer of 2026-10-13) is taken as the plan's region, and its row
    says which parts and on whose word. Its capture's part reads `a part`, since it holds no face class; without
    this the table laid the source exactly where the owner asked for the new hair. The parts must be some of the
    plan's for the subject and not all of them; a part nobody named, and a named part the plan does not have, are
    held to the plan's region as before."""
    region, has = render["form"].get("region"), (render["form"].get("reference") or {}).get("has")
    named = render["form"].get("part_named") or {}
    subset = bool(named.get("parts")) and region != "whole" and plan_parts is not None and set(named["parts"]) < set(plan_parts)
    said = "" if not subset else (f"a part of the plan's region ({region_wanted}) named for this load by its job: {' + '.join(named['parts'])} alone, "
                                  f"on {named.get('by') or 'a word the job does not give'}; ")
    if region != region_wanted and not subset:
        return None, f"it replaces {region or 'what no file says'}, where the plan's region for this subject is {region_wanted}"
    # HER HAIR ALONE WITH NO REFERENCE VIDEO, on a person's word (`shot_to_load.py job --motion none`, the lead's ruling of 2026-10-15
    # on second_hp_5461c): the source's own picture carries her movement, and only a load of the hair alone is ruled so
    none = render["form"].get("reference_none")
    if not has and none and subset and sorted(named["parts"]) == ["hair"]:
        return "frozen, with no reference video", (said + f"no reference video, on a person's word ({' '.join(str(none.get('why') or '').split())}): "
                                                  "a load of her hair alone made with shot_to_load's --motion none")
    if not has or not has.get("cut_to_the_head"):
        return None, "its reference is not the body model cut to the head: " + (render["form"]["reference"].get("says") or "none")
    if has.get("lid_marks"):
        return None, "its reference has marks drawn on the lids, the form before the frozen one"
    if has.get("face_points"):
        return "frozen", said + "the frozen form: the plan's region, the body model cut to the head with the face points drawn"
    return "frozen, by its exception", (said + "the plain body model cut to the head, the form's exception for a face under the drawing line; WHY this load took "
                                        "it is in no field read here (the load's note or its capture's looks)")


def runs(frames: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def parse_restore_extra(text: str) -> dict:
    """`SUBJECT.CLASS[+CLASS]:FIRST-LAST:the reason` (`--restore-extra`): a restore span for this table alone, a trial."""
    try:
        what, span, why = text.split(":", 2)
        subject, classes = what.split(".", 1)
        first, last = (int(v) for v in span.split("-"))
    except ValueError:
        raise SystemExit(f"--restore-extra {text!r} is not SUBJECT.CLASS[+CLASS]:FIRST-LAST:the reason") from None
    if not why.strip() or first > last or not classes:
        raise SystemExit(f"--restore-extra {text!r}: the classes, a span and the reason")
    return {"subject": subject, "classes": classes.split("+"), "first": first, "last": last, "why": " ".join(why.split())}


def with_trial_restores(plan: dict, extras: list[dict]) -> dict:
    """The plan with `--restore-extra` spans added for this table alone: A TRIAL for the owner (the lead's ask of 2026-10-15, her own
    hair given back on her face rows of 604-1009, an A/B of the cut). The plan file is never written; each span says it is a trial."""
    out = json.loads(json.dumps(plan))
    for x in extras:
        if x["subject"] not in out["subjects"]:
            raise SystemExit(f"--restore-extra names {x['subject']!r}, which is not a subject of the plan ({', '.join(out['subjects'])})")
        out["subjects"][x["subject"]].setdefault("restore", []).append({"classes": x["classes"], "first": x["first"], "last": x["last"], "trial": x["why"]})
    return out


def parse_hold(text: str) -> dict:
    """`FIRST-LAST:SUBJECT=LOAD:the reason`: on that shot, for that subject, the load NOT to be laid, and why."""
    return parse_prefer(text, "--hold")


def parse_prefer(text: str, flag: str = "--prefer") -> dict:
    """`FIRST-LAST:SUBJECT=LOAD:the reason`: on that shot, for that subject, the load to go first, and why."""
    try:
        where, rest = text.split("=", 1)
        span, subject = where.split(":", 1)
        load, reason = rest.split(":", 1)
        first, last = (int(v) for v in span.split("-"))
        if not (subject and load and reason.strip()):
            raise ValueError
    except ValueError:
        raise SystemExit(f"{flag} {text!r} is not `FIRST-LAST:SUBJECT=LOAD:the reason` (a shot of the shots file, a subject, a load's name, "
                         "and the reason in words: it is printed)") from None
    return {"first": first, "last": last, "subject": subject, "load": load, "reason": reason.strip()}


def decide(ledger: dict, shots: dict, plan: dict, prefers: list[dict] | None = None, span: tuple[int, int] | None = None,
           holds: list[dict] | None = None, laid: list[dict] | None = None) -> dict:
    """The table as data: `rows` (pieces laid), `sources` (whole shots left as the source), `gaps` (frames inside a
    shot that has a piece and that no piece takes) and `reading` (one entry a shot and a subject).

    `laid` (the lines of `bench/laid.py`'s list as `laid.read` returns them, bytes checked; this function trusts its
    caller on that, each line with its `line` number) is the ONLY rule when given (the owner's
    order of 2026-10-16): a render stands on a shot for a subject only where a line names it, in the list's order, and
    nothing else may stand (no `--prefer`, no `--hold`, no keep or newest-render fallback). The refusals still hold:
    a line naming a render that never stands, one the owner's last choice says is not the one, or a file that is not
    the ledger's file of that render stops the table."""
    looks_say = _shot_coverage().nothing_to_make          # the rule has one home: the coverage tool's

    def own_restores(label: str) -> list[tuple[int, int, str]]:
        """The plan's `restore` spans for a subject, as (first, last, the row's `restore=` token): those classes of the
        subject's own class map given back to the source on the pieces of theirs on those source frames only, at assembly;
        generated here, never typed. ONLY AS A MARKED EXCEPTION (the owner, 2026-10-16, "The simplification wins. Make sure
        this never happens again", docs/wiki/decisions.md): H3's picture is kept whole inside a person's region, so a span
        that gives back her own classes is refused unless it carries `exception` with its reason in words. A trial
        (`--restore-extra`) carries its reason as `trial` and is said as one. What it replaces: the owner's keep-accessories
        rule of 2026-10-14, under which every such span was laid. NAMED GAP: a `restore=<subject>.<Class>` typed straight into
        a table's .txt reaches the assembler with no exception (it reads the token alone); the sidecar's tie of the table
        JSON's rows to the .txt makes such an edit visible."""
        spans = list(plan["subjects"][label].get("restore") or [])
        if not spans:
            return []
        bare = [x for x in spans if not x.get("trial") and not str(x.get("exception") or "").strip()]
        if bare:
            x = bare[0]
            raise SystemExit(f"the plan's `restore` for {label!r} gives back {label!r}'s own {'+'.join(x['classes'])} on {x['first']}-{x['last']} with no "
                             "`exception`: nothing puts the original back inside a person's region (the owner, 2026-10-16, docs/wiki/decisions.md), "
                             "so a span of her own classes stands only as an exception, with its reason in words (`exception`) and gone when its "
                             "shot is made again. Mark it, or take it out of the plan")
        import capture_masked_run as cap
        names = cap._pack("sapiens2_parts").CLASS_NAMES
        bad = sorted({c for x in spans for c in x["classes"] if c not in names or c == "Background"})
        if bad:
            raise SystemExit(f"the plan's `restore` for {label!r} names {bad}, which are not classes of the part model: one of {', '.join(names[1:])}")
        where = sorted({str(x["touching"]) for x in spans if x.get("touching") not in (None, "head")})
        if where:
            raise SystemExit(f"the plan's `restore` for {label!r} says `touching` {where}: the one form is `head` (the class's pieces that touch "
                             "the subject's own Hair or Face_Neck, the assembler's `@head`)")
        # A THING WORN ON THE HEAD IS PLACED ON THE PIECE'S OWN FACE at assembly (the owner, 2026-10-23); `"placed": "as filmed"` keeps the
        # film's place for one span, on a person's word with its reason (`placed_by`), as the token's `:filmed`
        odd = [x for x in spans if x.get("placed") is not None and (x["placed"] != "as filmed" or not str(x.get("placed_by") or "").strip())]
        if odd:
            x = odd[0]
            raise SystemExit(f"the plan's `restore` for {label!r} says `placed` {x['placed']!r} on {x['first']}-{x['last']}"
                             + ("" if x["placed"] == "as filmed" else ": the one word is \"as filmed\"")
                             + ", which keeps a thing worn on the head where the film has it (the assembler otherwise places it on the piece's own "
                             "face, bench/assemble_delivery.py `WORN_ON_THE_HEAD`), and it takes `placed_by` saying whose word and why")
        return [(int(x["first"]), int(x["last"]), f"{label}.{'+'.join(x['classes'])}" + ("@head" if x.get("touching") == "head" else "")
                 + (":filmed" if x.get("placed") == "as filmed" else "")) for x in spans]

    def cut_at(a: int, b: int, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, list[str]]]:
        """A run of a row cut at the edges of the restore spans, each piece with the tokens of the spans it lies in."""
        edges = sorted({a, b + 1} | {e for f, l, _ in spans for e in (f, l + 1) if a < e <= b})
        return [(x, y - 1, [tok for f, l, tok in spans if f <= x and y - 1 <= l]) for x, y in zip(edges, edges[1:])]
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise SystemExit(f"the ledger's schema is {ledger.get('schema')!r}, not {LEDGER_SCHEMA!r}: make it again with bench/render_ledger.py")
    prefers, holds = list(prefers or []), list(holds or [])
    if laid is not None and (prefers or holds):
        raise SystemExit("--laid is the only rule for what is laid: it takes no --prefer and no --hold (the owner's order of 2026-10-16)")
    if span is not None:
        # a table of part of the video: the shots inside the span, whole. A `source` line covers a whole shot, so a
        # span that cuts a shot is refused and the shot named
        cut = [s for s in shots["shots"] if s["first"] <= span[1] and s["last"] >= span[0] and not (span[0] <= s["first"] and s["last"] <= span[1])]
        if cut:
            raise SystemExit(f"--span {span[0]}-{span[1]} cuts the shot {cut[0]['first']}-{cut[0]['last']} of the shots file: a span ends on shots' edges")
        shots = {**shots, "shots": [s for s in shots["shots"] if span[0] <= s["first"] and s["last"] <= span[1]]}
    renders = list(ledger["renders"].values())
    rows, sources, gaps, reading, absent = [], [], [], [], []
    for flag, named_by in (("--prefer", prefers), ("--hold", holds)):
        for p in named_by:
            if not any(s["first"] == p["first"] and s["last"] == p["last"] for s in shots["shots"]):
                raise SystemExit(f"{flag} names frames {p['first']}-{p['last']}, which is not a shot of the shots file")
            if not any(r["name"] == p["load"] for r in renders):
                raise SystemExit(f"{flag} names the load {p['load']!r}, which the ledger has no render of")
    if laid is not None and span is not None:
        laid = [x for x in laid if span[0] <= x["first"] and x["last"] <= span[1]]   # a table of part of the video reads its own lines
    for x in laid or []:
        if not any(s["first"] == x["first"] and s["last"] == x["last"] for s in shots["shots"]):
            raise SystemExit(f"the list of what is laid, line {x['line']}, names frames {x['first']}-{x['last']}, which is not a shot of the shots file")
        shot = next(s for s in shots["shots"] if s["first"] == x["first"] and s["last"] == x["last"])
        if x["subject"] not in shots["subjects"] or shot["who"][x["subject"]]["present"] == "no":
            raise SystemExit(f"the list of what is laid, line {x['line']}, names {x['subject']!r} on {x['first']}-{x['last']}, and the shots file "
                             "has no such subject there (or says they are not in the shot): the line would be skipped, so it stops the table")
        named = [v for v in renders if v["name"] == x["load"]]
        if not named:
            raise SystemExit(f"the list of what is laid, line {x['line']}, names the load {x['load']!r}, which the ledger has no render of")
        r = next((v for v in named if v.get("file") == x["file"]), named[0])   # a line names one render: its load AND its file
        if r.get("file") != x["file"]:
            raise SystemExit(f"the list of what is laid, line {x['line']}, names the file {x['file']} for {x['load']}; the ledger's file of it is {r.get('file')}")
    for h in holds:
        if any((p["first"], p["last"], p["subject"], p["load"]) == (h["first"], h["last"], h["subject"], h["load"]) for p in prefers):
            raise SystemExit(f"--hold and --prefer both name {h['load']} for {h['first']}-{h['last']}, {h['subject']}: a render is put first or held back, not both")
    for s in shots["shots"]:
        present = {label: s["who"][label]["present"] for label in shots["subjects"]}
        for label in shots["subjects"]:
            if present[label] == "no":
                # a line all the same: the assembler does not read the shots file, and asks for one wherever people are
                # detected in a shot and nothing answers for a label. Who said so, never what they said of it
                by = " ".join(str(s["who"][label].get("by") or "it does not say who said so").replace("#", " ").split())
                absent.append({"first": s["first"], "last": s["last"], "subject": label, "why": f"{NOT_IN_THE_SHOT} ({by})"})
                continue
            wanted = plan["subjects"][label]["region"]
            on = [r for r in renders if label in (r.get("subjects") or []) and r.get("span") and r["span"][0] <= s["last"] and r["span"][1] >= s["first"]]
            preferred = [p for p in prefers if (p["first"], p["last"], p["subject"]) == (s["first"], s["last"], label)]
            held_here = [h for h in holds if (h["first"], h["last"], h["subject"]) == (s["first"], s["last"], label)]
            laid_here = [x for x in laid or [] if (x["first"], x["last"], x["subject"]) == (s["first"], s["last"], label)]
            may, passed, held_from_standing = [], [], []
            for r in on:
                never = never_stands(r)
                choice = last_choice(r)
                kept = bool(choice and choice["said"] in KEEP)
                barred = (f"the owner's last choice on it was \"{choice['said']}\" ({str(choice['at'])[:16]}, row {choice.get('row')}): not the one"
                          if choice and choice["said"] in NOT_THE_ONE else None)
                if laid is not None:
                    line = next((x for x in laid_here if (x["load"], x["file"]) == (r["name"], r["file"])), None)
                    if line is None:
                        passed.append({"render": r["name"], "file": r["file"], "written": r["written"], "why": "not in the list of what is laid"
                                       + (f"; the owner's last choice on it: \"{choice['said']}\" ({str(choice['at'])[:16]})" if choice else "")})
                    elif never:
                        raise SystemExit(f"the list of what is laid, line {line['line']}, names {r['name']} for {s['first']}-{s['last']}, {label}, and it never stands: {never}")
                    elif barred:
                        raise SystemExit(f"the list of what is laid, line {line['line']}, names {r['name']} for {s['first']}-{s['last']}, {label}, and {barred}")
                    else:
                        may.append((0, line["line"], r, f"laid, line {line['line']} ({line['owner']}): {line['by']}", choice))
                    continue
                frozen, form_words = form_of(r, wanted, plan["subjects"][label].get("parts"))
                named = next((p for p in preferred if p["load"] == r["name"]), None)
                held = next((h for h in held_here if h["load"] == r["name"]), None)
                held_words = "" if held is None else f"HELD BACK by the lead (--hold), a session's word and not the owner's: {held['reason']}"
                if held and kept:
                    raise SystemExit(f"--hold names {r['name']} for {s['first']}-{s['last']}, {label}, and the owner's last choice on it was "
                                     f"\"{choice['said']}\" ({str(choice['at'])[:16]}, row {choice.get('row')}): a keep. The owner's word is over the "
                                     "lead's hold: it is held when the owner changes it")
                if held and not never:
                    if frozen and not barred:
                        held_from_standing.append(r["name"])
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"], "held": held["reason"],
                                   "why": held_words + (". Without the hold: " + ("it may stand (step 2), " + form_words if frozen and not barred else
                                                                                   barred if barred else "it is not in the frozen form: " + form_words))})
                    continue
                if never:
                    if named:
                        raise SystemExit(f"--prefer names {r['name']} for {s['first']}-{s['last']}, {label}, and it never stands: {never}")
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"], "why": "never stands: " + never
                                   + (f"; also {held_words}" if held else ""), **({"held": held["reason"]} if held else {})})
                elif named and barred:
                    raise SystemExit(f"--prefer names {r['name']} for {s['first']}-{s['last']}, {label}, and {barred}. The owner's choice is "
                                     "over the lead's preference: it stands when the owner changes it")
                elif named:
                    may.append((0, r["written"], r, f"step 0: named by --prefer: {named['reason']}", choice))
                elif kept:
                    may.append((1, choice["at"], r, f"step 1: the owner's keep of {str(choice['at'])[:16]} (row {choice.get('row')}); its file agrees "
                                                     f"with its build record", choice))
                elif frozen and barred:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": f"its file agrees with its build record and it is in the frozen form, and {barred}"})
                elif frozen:
                    may.append((2, r["written"], r, f"step 2: {form_words}; its file agrees with its build record", choice))
                else:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": "its file agrees with its build record, and it has no keep and is not in the frozen form: " + form_words})
            for p in preferred:
                if not any(m[2]["name"] == p["load"] for m in may):
                    raise SystemExit(f"--prefer names {p['load']} for {s['first']}-{s['last']}, {label}, and no render of that load covers those frames for that subject")
            for h in held_here:
                if not any(r["name"] == h["load"] for r in on):
                    raise SystemExit(f"--hold names {h['load']} for {s['first']}-{s['last']}, {label}, and no render of that load covers those frames for that subject")
            for x in laid_here:
                if not any((m[2]["name"], m[2]["file"]) == (x["load"], x["file"]) for m in may):
                    raise SystemExit(f"the list of what is laid, line {x['line']}, names {x['load']} for {s['first']}-{s['last']}, {label}, and no render of that load covers those frames for that subject")
            if laid is not None:
                may.sort(key=lambda m: m[1])                                   # the list's own order
            else:
                may.sort(key=lambda m: str(m[1]), reverse=True)                # the newest first,
                may.sort(key=lambda m: m[0])                                   # within the rule's steps in their order
            free = set(range(s["first"], s["last"] + 1))
            stands = []
            meant: dict[int, str] = {}      # frames a laid line's own `frames` leave out of its render on this shot, with why
            for step, _when, r, why, choice in may:
                spans_of = set(range(r["span"][0], r["span"][1] + 1))
                line = next((x for x in laid_here if (x["load"], x["file"]) == (r["name"], r["file"])), None) if laid is not None else None
                if line is not None and line.get("frames") and line["frames"] != [[s["first"], s["last"]]] \
                        and line["frames"] != [(s["first"], s["last"])]:
                    # THE LIST'S OWN FRAMES ARE LAID (2026-10-21, W3e): until then `frames` were read nowhere that lays, and a line
                    # that said 5889-5917 laid its render on the whole 5889-5934. Honored or refused, never ignored
                    allowed = set()
                    if not isinstance(line["frames"], list):
                        raise SystemExit(f"the list of what is laid, line {line['line']}, lays {r['name']} on {line['frames']!r}: `frames` is a list of "
                                         "[first, last] spans")
                    for f in line["frames"]:
                        if not (isinstance(f, (list, tuple)) and len(f) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in f)) \
                                or not (s["first"] <= f[0] <= f[1] <= s["last"]):
                            raise SystemExit(f"the list of what is laid, line {line['line']}, lays {r['name']} on {f}, which is not a span inside "
                                             f"the shot {s['first']}-{s['last']}")
                        allowed |= set(range(int(f[0]), int(f[1]) + 1))
                    if not allowed & spans_of:
                        raise SystemExit(f"the list of what is laid, line {line['line']}, lays {r['name']} on {line['frames']} and its render "
                                         f"covers none of those frames ({r['span'][0]}-{r['span'][1]})")
                    said = " ".join(f"the list of what is laid, line {line['line']}, lays {r['name']} on "
                                    f"{', '.join(frames(int(a_), int(b_)) for a_, b_ in line['frames'])} only: {line.get('by') or 'no reason given'}"
                                    .replace("#", " ").split())
                    for n in (free & spans_of) - allowed:
                        meant.setdefault(n, said)
                    spans_of &= allowed
                take = sorted(free & spans_of)
                if not take:
                    passed.append({"render": r["name"], "file": r["file"], "written": r["written"],
                                   "why": f"may stand ({why.split(':')[0]}) and every frame of it on this shot is taken by a render before it in the rule's order",
                                   # an owner's keep set aside by a render before it (the lead's --prefer) is said where the owner reads it
                                   "the_owners_last_choice": None if not choice else {"said": choice["said"], "at": choice["at"], "row": choice.get("row")}})
                    continue
                free -= set(take)
                others = [o for o in shots["subjects"] if o != label and present[o] != "no"]
                region = r["form"].get("region")
                restores = [f"{o}:whole" if region == "face" else o for o in others]
                for a0, b0 in runs(take):
                    for a, b, kept in cut_at(a0, b0, own_restores(label)):
                        rows.append({"first": a, "last": b, "file": r["file"], "name": r["name"], "piece_first": r["span"][0], "subject": label,
                                     "region": region, "restore": restores + kept})
                took = set(take)
                stands.append({"render": r["name"], "file": r["file"], "written": r["written"], "frames": runs(take), "why": why, "step": step,
                               # step 0 is the lead's word, not the owner's: its reason whole, for every page that shows the piece
                               "the_leads_prefer": next((p["reason"] for p in preferred if p["load"] == r["name"]), None) if step == 0 and laid is None else None,
                               "laid_line": next(({k: x[k] for k in ("line", "owner", "by", "sha256")} for x in laid_here if (x["load"], x["file"]) == (r["name"], r["file"])), None),
                               # the frames of these that the piece's own load left as the original (the ledger's read of its capture)
                               # None where the ledger does not say (no capture record of it): not known is not none
                               "leaves_as_the_original": None if r.get("leaves_as_the_original") is None else runs(sorted(
                                   x for a_, b_ in r["leaves_as_the_original"].get("source_frames") or [] for x in range(a_, b_ + 1) if x in took)),
                               "the_owners_last_choice": None if not choice else {"said": choice["said"], "at": choice["at"], "row": choice.get("row")}})
            left = runs(sorted(free))
            # the owner's typed words on each render named here, beside their choice: they asked that their words are read
            # with their clicks (2026-10-13), and this reading is what a lead reads before a build
            by_name = {r["name"]: r for r in on}
            for named_here in stands + passed:
                named_here["the_owners_words"] = owners_words(by_name[named_here["render"]])
            # the owner's last choice on every render passed over, a keep on one that can never stand included: a page that
            # shows the shot says the keep is not laid, and why (2026-10-14, a reader: eight keeps went unsaid)
            for gone in passed:
                if "the_owners_last_choice" not in gone:
                    c = last_choice(by_name[gone["render"]])
                    gone["the_owners_last_choice"] = None if not c else {"said": c["said"], "at": c["at"], "row": c.get("row")}
            entry = {"first": s["first"], "last": s["last"], "subject": label, "present": present[label], "plan_region": wanted,
                     "stands": stands, "passed_over": sorted(passed, key=lambda x: x["written"], reverse=True), "left_as_the_source": left}
            nothing = looks_say(s["who"][label], plan["subjects"][label], s["first"], s["last"]) if not stands else None
            if nothing:
                # a person's lines say none of the plan's parts is seen on the shot: not a gap, no job, no stopped load
                entry["nothing_to_make"] = nothing
                why = f"{NOTHING_TO_MAKE}: " + "; ".join(nothing)
                sources.append({"first": s["first"], "last": s["last"], "subject": label, "why": why})
                entry["source_line"] = why
            elif not stands:
                newest = entry["passed_over"][0]["render"] if entry["passed_over"] else None
                if held_from_standing:
                    why = ("" if present[label] == "yes" else "the shots file is unsure this subject is in the shot; ") \
                        + f"{HELD_BACK} ({', '.join(held_from_standing)})"
                elif laid is not None:
                    why = ("" if present[label] == "yes" else "the shots file is unsure this subject is in the shot; ") + NOT_LAID \
                        + (f" (the newest render passed over: {newest})" if newest else " (no render of this subject covers these frames)")
                else:
                    why = (NO_SOURCE_YET if present[label] == "yes" else f"the shots file is unsure this subject is in the shot; {NO_SOURCE_YET}") \
                        + (f" (the newest render passed over: {newest})" if newest else " (no render of this subject covers these frames)")
                sources.append({"first": s["first"], "last": s["last"], "subject": label, "why": why})
                entry["source_line"] = why
            elif left:
                # a run is said by why it is left: a laid line's own frames (`meant`), or no render that may stand
                here: list[dict] = []
                for a, b in left:
                    for n in range(a, b + 1):
                        w = meant.get(n)
                        if here and here[-1]["meant"] == w and here[-1]["last"] + 1 == n:
                            here[-1]["last"] = n
                        else:
                            here.append({"first": n, "last": n, "subject": label, "meant": w})
                for g in here:
                    w = g.pop("meant")
                    g["why"] = w or f"inside the shot {s['first']}-{s['last']}, which has a piece: no render that may stand covers these frames"
                    if w:
                        g["laid_frames"] = True
                gaps += here
                entry["left_why"] = [{"first": g["first"], "last": g["last"], "why": g["why"], "laid_frames": bool(g.get("laid_frames"))} for g in here]
            reading.append(entry)
    # rows of one render on touching runs with the same restores are one row; face rows after the others on shared frames
    rows.sort(key=lambda r: (r["subject"], r["file"], r["first"]))
    merged: list[dict] = []
    for r in rows:
        if merged and (merged[-1]["file"], merged[-1]["subject"], merged[-1]["restore"]) == (r["file"], r["subject"], r["restore"]) and merged[-1]["last"] + 1 == r["first"]:
            merged[-1]["last"] = r["last"]
        else:
            merged.append(dict(r))
    merged.sort(key=lambda r: (r["first"], r["region"] == "face", r["subject"]))
    return {"rows": merged, "sources": sources, "not_in_the_shot": absent, "gaps": gaps, "reading": reading,
            "held": [{"first": h["first"], "last": h["last"], "subject": h["subject"], "load": h["load"], "reason": h["reason"]} for h in holds]}


def frames(a, b) -> str:
    return f"{a}-{b}"


def tracked_in(capture: Path, labels: set[str], first: int, last: int) -> dict | None:
    """Of the source frames `first`..`last`, those on which a capture holds a non-empty track of a subject known by
    any of `labels` (`on`) and those it covers with an empty one (`empty`); None when the capture has no such subject
    or no manifest. As the assembler reads it (`Captures.mask`): a frame is covered when it is inside the capture's
    frames and the subject's `covered` says so (a file with no `covered` covers every frame of the capture)."""
    import numpy as np
    manifest = capture / "manifest.json"
    if not manifest.is_file():
        return None
    m = json.loads(manifest.read_text())
    entry = next((s for s in m.get("subjects", []) if s["label"] in labels), None)
    if entry is None or not entry.get("sightings"):
        return None
    saved = capture / "subjects" / entry["label"] / f"masks__{entry['sightings'][0]['by']}.npz"
    if not saved.is_file():
        return None
    z = np.load(saved)
    on = z["track"].reshape(m["frames"], -1).any(axis=1)
    covered = z["covered"].astype(bool) if "covered" in z.files else np.ones(m["frames"], bool)
    inside = lambda picked: {m["first_frame"] + int(f) for f in np.nonzero(picked)[0] if first <= m["first_frame"] + int(f) <= last}
    return {"on": inside(on & covered), "empty": inside(~on & covered)}


def run_of(capture: Path, load: str | None, labels: set[str]) -> tuple[str | None, str]:
    """(the name of the run of a capture that made a load's render, why not when None). From records: the run in the
    capture's manifest named as the load, of this subject, naming no other render. Never found by a file's name."""
    manifest = capture / "manifest.json"
    if not load:
        return None, "its build record names no load"
    if not manifest.is_file():
        return None, "the capture its build record names has no manifest"
    run = next((r for r in json.loads(manifest.read_text()).get("runs") or [] if r.get("name") == load), None)
    if run is None:
        return None, f"the capture its build record names holds no run named {load}"
    if run.get("subject") not in labels:
        return None, f"the run {load} of its capture is of {run.get('subject')}, not of this subject"
    return load, ""


def assembly(out: dict, ledger: dict, plan: dict, repo: Path, table: str, span: tuple[int, int], args: str = "", also: tuple[str, ...] = (),
             python: str = "python", own: str = "", stack_work: str = "", shots_file: str = "") -> dict:
    """The commands for this table, whole, the run each row is tied to, and what each `restore=` has to go on.

    The captures are the standing pieces' own (each piece's build record names its capture), in the table's row order, then any handed in
    (`also`: a preview's capture of the span, for its shot tables). Each row is given its `run` here, in `out`, where
    its capture's manifest holds the run its build record's load names (`run_of`), and the lack is kept where not.
    `--alias` is printed for a name of the plan's `aliases` that a listed capture labels a subject by, and for no
    other. `args` is what only the caller knows (`--source`, `--out`, `--size`); without a `--source` and an `--out`
    in it the command is said to be not whole. `own` is this tool's own command, `python` the interpreter the
    commands are printed with, `stack_work` the stacking tool's scratch folder. Every command is one line a shell
    takes as it stands. For each `restore=` on a row: three counts of the row's frames (`tracked_in`). Figures."""
    rel = lambda p: str(Path(p).relative_to(repo)) if Path(p).is_absolute() and Path(p).is_relative_to(repo) else str(p)
    aliases = {k: v for k, v in (plan.get("aliases") or {}).items() if not k.startswith("_")}
    names = lambda label: {label} | {old for old, new in aliases.items() if new == label}
    real = lambda c: Path(c) if Path(c).is_absolute() else repo / c
    captures: list[str] = []
    ties = []
    for r in out["rows"]:
        records = ledger["renders"][r["file"]].get("records") or []
        capture = records[-1].get("capture") if records else None
        if capture and rel(capture) not in captures:
            captures.append(rel(capture))
        if not capture:
            run, why = None, "its build record names no capture" if records else "the ledger has no build record of it"
        elif any(c.isspace() for c in rel(capture)):
            run, why = None, "its capture's path holds a space, which a table's row cannot"
        else:
            run, why = run_of(real(capture), (records[-1].get("load") or {}).get("name"), names(r["subject"]))
        r.pop("run", None)
        if run:
            r["run"] = f"{rel(capture)}:{run}"
        ties.append({"row": f"{frames(r['first'], r['last'])} {r['name']} ({r['subject']})", "run": r.get("run"), "why_not": why})
    captures += [rel(c) for c in also if rel(c) not in captures]
    used = set()
    for c in captures:
        manifest = real(c) / "manifest.json"
        if manifest.is_file():
            used |= {s["label"] for s in json.loads(manifest.read_text()).get("subjects", [])} & set(aliases)
    restores = []
    for r in out["rows"]:
        for x in r["restore"]:
            other = x.split(":")[0]
            on, empty, where = set(), set(), []
            for c in captures:
                got = tracked_in(real(c), names(other), r["first"], r["last"])
                if got:
                    on |= got["on"]
                    empty |= got["empty"]
                    where += [c] if got["on"] else []
            count = r["last"] - r["first"] + 1
            restores.append({"row": f"{frames(r['first'], r['last'])} {r['name']} ({r['subject']})", "restore": x, "frames": count,
                             "frames_with_a_track": len(on), "frames_covered_with_an_empty_track": len(empty - on),
                             "frames_no_capture_covers": count - len(on | empty), "in_the_captures": where})
    given = shlex.split(args)
    value = lambda flag: given[given.index(flag) + 1] if flag in given and given.index(flag) + 1 < len(given) else None
    source, clip = value("--source"), value("--out")
    whole = bool(source and clip)
    q = shlex.quote
    line = lambda words: " ".join(q(str(w)) for w in words)
    build = [python, "bench/assemble_delivery.py", "--table", table, "--span", f"{span[0]}-{span[1]}", *given]
    if shots_file:
        # the record this table's `source` lines were written a shot of: the assembler cuts where it cuts
        build += ["--shots", shots_file]
    for c in captures:
        build += ["--capture", c]
    for old in sorted(used):
        build += ["--alias", f"{old}={aliases[old]}"]
    stack = None
    if whole:
        stacked = str(Path(clip).with_name(Path(clip).stem + "_STACKED.mp4"))
        work = stack_work or "A_SCRATCH_FOLDER"
        # a clip of a span begins on the span's first frame, and the stack is told so: its original row, its panels and the number it burns are the source's
        begins = ["--first", str(span[0])] if span and span[0] else []
        # the per-frame gap check runs on the table first and the stack is handed its JSON (the stack view's reader, 2026-10-16: a view built
        # without it has no gap chips and no red bands); its files beside the table, named as bench/frame_coverage.py names them
        table_json = str(Path(table).with_suffix(".json"))
        coverage = Path(table).with_name("frame_coverage_" + Path(table).stem.removeprefix("t_"))
        cov_json, cov_md = str(coverage) + ".json", str(coverage) + ".md"
        stack = {"coverage": line([python, "bench/frame_coverage.py", "--table", table_json, "--json", cov_json, "--md", cov_md]),
                 "build": line([python, "bench/stack_delivery.py", "build", "--source", source, "--table", table, "--joined", clip, "--coverage", cov_json,
                                "--out", stacked, "--work", work, *begins, "--one-a-second"]),
                 "check": line([python, "bench/stack_delivery.py", "check", "--source", source, "--joined", clip, "--out", stacked, *begins]),
                 "whole": bool(stack_work), "out": stacked, "coverage_json": cov_json}
    return {"command": line(build), "whole": whole, "captures": captures, "captures_handed_in": [rel(c) for c in also],
            "aliases_used_by_a_capture": sorted(used), "runs": ties, "restores": restores, "stack": stack,
            "ledger_command": (ledger.get("command") or {}).get("line"), "own_command": own or None}


def table_text(out: dict, share: str, made: str, plan: dict, name: str) -> str:
    """The table as the assembler reads it (`assemble_delivery.read_table`, `read_intents`)."""
    aliases = {k: v for k, v in (plan.get("aliases") or {}).items() if not k.startswith("_")}
    lines = [f"# {name}: GENERATED by bench/delivery_table.py, never typed. {made}",
             "# The names are the plan's labels. " + ("A capture that labels a subject by an older name is read with: "
                                                      + " ".join(f"--alias {old}={new}" for old, new in aliases.items()) + " (the plan's aliases)." if aliases else "The plan has no aliases."),
             "# WHICH RENDER STANDS is the rule in that tool's docstring (the lead's --prefer; the owner's keep on a render whose file agrees with its build",
             "# record; else the newest such render in the frozen form; never one with no record, a disagreeing file or a whole person). The reading beside",
             "# this file says why each piece stands and what was passed over.",
             *[f"# HELD BACK by the lead (--hold), a session's word and not the owner's: {h['load']} on {frames(h['first'], h['last'])}, {h['subject']}: "
               + " ".join(h["reason"].replace("#", " ").split()) for h in out.get("held", [])],
             "# restore= is the practice of the table of 2026-10-12: every other subject the shots file has in the shot is given back, `:whole` on a face row."]
    lines += [f"# restore={label}.{'+'.join(x['classes'])}{'@head' if x.get('touching') == 'head' else ''}{':filmed' if x.get('placed') == 'as filmed' else ''}"
              f" on {label}'s rows on {frames(x['first'], x['last'])} only: "
              + (f"A TRIAL (--restore-extra), not the plan's: {' '.join(str(x['trial']).replace('#', ' ').split())}" if x.get("trial") else
                 f"from the plan, by {' '.join(str(p['restore_by']).replace('#', ' ').split())}; "
                 f"AN EXCEPTION: {' '.join(str(x.get('exception')).replace('#', ' ').split())}")
              + (f"; PLACED AS FILMED, not on the piece's own face: {' '.join(str(x.get('placed_by')).replace('#', ' ').split())}"
                 if x.get("placed") == "as filmed" else "")
              for label, p in plan["subjects"].items() for x in p.get("restore") or []]
    for r in out["rows"]:
        lines.append(f"{frames(r['first'], r['last'])} {Path(share) / r['file']} {r['piece_first']} subject={r['subject']}"
                     + (f" run={r['run']}" if r.get("run") else "") + "".join(f" restore={x}" for x in r["restore"]))
    if out["gaps"]:
        lines.append("# Left as the source INSIDE a shot that has a piece (a `source` line covers a whole shot, so these are said here):")
        lines += [f"#   {frames(g['first'], g['last'])} {g['subject']}: "
                  + (g["why"] if g.get("laid_frames") else "no render that may stand covers these frames") for g in out["gaps"]]
    for s in out["sources"]:
        lines.append(f"source {frames(s['first'], s['last'])} subject={s['subject']} {s['why']}")
    if out.get("not_in_the_shot"):
        lines.append("# A subject the shots file says is NOT in a shot: said here, since the assembler does not read the shots file.")
        lines += [f"source {frames(s['first'], s['last'])} subject={s['subject']} {s['why']}" for s in out["not_in_the_shot"]]
    return "\n".join(lines) + "\n"


def reading_text(out: dict, made: str, table: str, built: dict | None = None) -> str:
    """The reading: the assembler's command, then a section a shot and a subject, what stands and why, what was
    passed over and why."""
    lines = ["# Which render stands on each shot, and why: the reading beside a generated delivery table", "", made, "",
             f"The table: `{table}`. Nothing was assembled. The rule is in `bench/delivery_table.py`'s docstring; every `why` below is one of its steps, "
             "a fact from the files, and none says a render is good.", ""]
    if built is not None:
        stack = built["stack"]
        lines += ["## The commands, in order, each one line a shell takes as it stands (run from the repo's root)", "",
                  "1. The ledger this table read:", "", "```", built["ledger_command"] or "# this ledger does not keep its command (made by bench/render_ledger.py before 2026-10-13)", "```", "",
                  "2. This table:", "", "```", built["own_command"] or "# not handed to this function", "```", "",
                  "3. The clip:", "", "```", built["command"], "```", ""]
        if not built["whole"]:
            lines += ["NOT WHOLE: no `--source` and `--out` were handed to this tool (`--assemble-args`), so the clip's command lacks them and there is no stack command.", ""]
        if stack:
            lines += ["4. The per-frame gap check on this table, then the clip stacked for the eye with its gaps drawn, and the stack read back:", "",
                      "```", stack["coverage"], stack["build"], stack["check"], "```", ""]
            if not stack["whole"]:
                lines += ["NOT WHOLE: no `--stack-work` was handed to this tool, so the stack's scratch folder is a word to replace.", ""]
        lines += ["The captures are the standing pieces' own, from each piece's build record"
                  + (f", then those handed in: {', '.join(built['captures_handed_in'])}" if built["captures_handed_in"] else "; none was handed in")
                  + ". " + ("`--alias` is given for the older names a listed capture uses: " + ", ".join(built["aliases_used_by_a_capture"]) + "."
                            if built["aliases_used_by_a_capture"] else "No listed capture labels a subject by an older name, so there is no `--alias`."), ""]
        loose = [t for t in built["runs"] if not t["run"]]
        lines += [f"The run each row is tied to (`run=` on the row, from the piece's build record): {len(built['runs']) - len(loose)} of {len(built['runs'])} rows are tied."
                  + (" NOT TIED, and read by the assembler as a subject with no run:" if loose else ""), ""]
        lines += [f"- `{t['row']}`: {t['why_not']}" for t in loose] + ([""] if loose else [])
        lines += ["What each `restore=` has to go on, as three counts of its row's frames: a listed capture holds that subject's track (pixels can be given "
                  "back); a listed capture covers the frame with an EMPTY track (nothing is given back and the assembler raises no flag); no listed capture "
                  "covers the frame (the assembler flags it, `restore_has_no_mask`). Figures, not failures.", ""]
        lines += [f"- `{x['row']}` `restore={x['restore']}`: of the row's {x['frames']} frames, a track on {x['frames_with_a_track']}, covered with an empty "
                  f"track on {x['frames_covered_with_an_empty_track']}, covered by no capture on {x['frames_no_capture_covers']}"
                  + (f" (the track in {', '.join(x['in_the_captures'])})" if x["in_the_captures"] else ": NO listed capture holds that subject's track on these frames")
                  for x in built["restores"]]
        lines += ["" if built["restores"] else "No row has a `restore=`.", ""]
    for e in out["reading"]:
        lines += [f"## {frames(e['first'], e['last'])}, {e['subject']} (the plan: {e['plan_region']}"
                  + ("" if e["present"] == "yes" else f"; the shots file says present: {e['present']}") + ")", ""]
        for st in e["stands"]:
            choice = st["the_owners_last_choice"]
            lines.append(f"- STANDS on {', '.join(frames(a, b) for a, b in st['frames'])}: **`{st['render']}`** (written {st['written']}). {st['why']}. "
                         + (f"The owner's last choice on it: \"{choice['said']}\", {str(choice['at'])[:16]}." if choice else "The owner has made no choice on it."))
            lines += words_lines(st.get("the_owners_words") or [])
        if not e["stands"]:
            lines.append(f"- NOTHING STANDS. The table's line: `source {frames(e['first'], e['last'])} subject={e['subject']} {e['source_line']}`")
        elif e["left_as_the_source"]:
            said = e.get("left_why") or [{"first": a, "last": b, "laid_frames": False} for a, b in e["left_as_the_source"]]
            for g in said:
                lines.append(f"- left as the source inside this shot: {frames(g['first'], g['last'])} "
                             + (f"({g['why']})" if g.get("laid_frames") else "(no render that may stand covers them)"))
        for p in e["passed_over"]:
            lines.append(f"- passed over: `{p['render']}` (written {p['written']}): {p['why']}")
            lines += words_lines(p.get("the_owners_words") or [])
        lines.append("")
    if out.get("not_in_the_shot"):
        lines += ["## Subjects the shots file says are not in a shot", "",
                  "Each has a `source` line in the table saying so, since the assembler does not read the shots file. Nothing was looked for on these.", ""]
        lines += [f"- `source {frames(n['first'], n['last'])} subject={n['subject']}`: {n['why']}" for n in out["not_in_the_shot"]] + [""]
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ledger", required=True, help="the render ledger's JSON (bench/render_ledger.py --json)")
    p.add_argument("--shots", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--prefer", action="append", default=[], metavar="FIRST-LAST:SUBJECT=LOAD:reason",
                   help="on that shot, for that subject, the load to go first among those that may stand; the reason is printed; repeatable")
    p.add_argument("--hold", action="append", default=[], metavar="FIRST-LAST:SUBJECT=LOAD:reason",
                   help="a render NOT to be laid on that shot for that subject, on the lead's word, with a reason that is printed; a session's word, "
                        "refused against an owner's keep; repeatable")
    p.add_argument("--laid", metavar="LAID.jsonl",
                   help="the list of what is laid (bench/laid.py): the ONLY rule when given; every file's sha256 is checked against it first; "
                        "takes no --prefer or --hold")
    p.add_argument("--span", metavar="FIRST-LAST", help="the table of these source frames alone, ending on shots' edges; default: the whole shots file")
    p.add_argument("--assemble-args", default="", metavar='"--source S --out O --size source"',
                   help="what only the caller knows of the build, printed into the assembler's command as given")
    p.add_argument("--also-capture", action="append", default=[], metavar="FOLDER",
                   help="a capture to hand the assembler beside the standing pieces' own (a preview's capture of the span); repeatable")
    p.add_argument("--stack-work", default="", metavar="FOLDER", help="the stacking tool's scratch folder (never the share), for its printed command")
    p.add_argument("--restore-extra", action="append", default=[], metavar="SUBJECT.CLASS[+CLASS]:FIRST-LAST:reason",
                   help="a restore span for this table alone, A TRIAL that the plan does not hold (the plan file is not written); repeatable")
    p.add_argument("--table", required=True, help="where the table is written")
    p.add_argument("--reading", required=True, help="where the reading is written")
    p.add_argument("--json", help="the same as data")
    a = p.parse_args()
    coverage = _shot_coverage()
    shots = coverage.load_shots(a.shots)
    plan = with_trial_restores(coverage.load_plan(a.plan, shots["subjects"]), [parse_restore_extra(x) for x in a.restore_extra])
    ledger = json.load(open(a.ledger))
    span = tuple(int(v) for v in a.span.split("-")) if a.span else tuple(shots["span"])
    laid_lines = None
    if a.laid:
        import laid as laid_list
        laid_lines = laid_list.read(a.laid, ledger["share"])            # a file whose bytes are not the line's sha256 stops here
    out = decide(ledger, shots, plan, [parse_prefer(x) for x in a.prefer], span if a.span else None, [parse_hold(x) for x in a.hold], laid_lines)
    commit = subprocess.run(["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    made = (f"Made {now} at commit {commit} from the ledger `{a.ledger}` ({str(ledger.get('made') or 'which does not say when it was made').split(' from')[0]}), "
            f"`{a.shots}`" + (f" (its shots inside {a.span})" if a.span else "") + f" and `{a.plan}`" + ("; with " + "; ".join(f"--prefer {x!r}" for x in a.prefer) if a.prefer else "; no --prefer")
            + ("; and HELD BACK by the lead, a session's word and not the owner's: " + "; ".join(f"--hold {x!r}" for x in a.hold) if a.hold else "; no --hold")
            + (f"; and ONLY the list of what is laid, `{a.laid}` (sha256 {laid_list.digest(a.laid)[:16]}, {len(laid_lines)} lines, every file's bytes checked)" if a.laid else "")
            + ("; and A TRIAL the plan does not hold: " + "; ".join(f"--restore-extra {x!r}" for x in a.restore_extra) if a.restore_extra else "") + ".")
    own = " ".join(shlex.quote(w) for w in [sys.executable, "bench/delivery_table.py", *sys.argv[1:]])
    built = assembly(out, ledger, plan, HERE.parent, a.table, span, a.assemble_args, tuple(a.also_capture), python=sys.executable, own=own,
                     stack_work=a.stack_work, shots_file=a.shots)
    Path(a.table).write_text(table_text(out, ledger["share"], made, plan, Path(a.table).stem))
    Path(a.reading).write_text(reading_text(out, made, a.table, built))
    if a.json:
        Path(a.json).write_text(json.dumps({"made": made, **out, "assembly": built}, indent=1) + "\n")
    print("the commands, in order (each one line, run from the repo's root):")
    print("  the ledger:  " + (built["ledger_command"] or "(this ledger does not keep its command)"))
    print("  this table:  " + own)
    print("  the clip" + ("" if built["whole"] else " (NOT WHOLE: no --source and --out in --assemble-args)") + ":  " + built["command"])
    if built["stack"]:
        print("  the stack" + ("" if built["stack"]["whole"] else " (NOT WHOLE: no --stack-work)") + ":  " + built["stack"]["build"] + "\n  its check:  " + built["stack"]["check"])
    laid = sum(r["last"] - r["first"] + 1 for r in out["rows"])
    tied = sum(bool(t["run"]) for t in built["runs"])
    print(f"{len(out['rows'])} piece row(s) over {laid} frame-rows ({tied} tied to their capture's run), {len(out['sources'])} `source` line(s) for a subject "
          f"with no render that may stand, {len(out['not_in_the_shot'])} for a subject the shots file says is not in the shot, {len(out['gaps'])} stretch(es) "
          f"left as the source inside a shot that has a piece")
    print("wrote", a.table, "and", a.reading, *(["and", a.json] if a.json else []))
    # LOUD, and by itself: the frames no piece covers, the frames a piece leaves as the original, and a subject in a shot with
    # none of their own (the owner's ask of 2026-10-13: whenever a session does the thing it does to find out what to build next)
    import piece_gaps
    print()
    for line in piece_gaps.loud(out, plan):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
