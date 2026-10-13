#!/usr/bin/env python3
"""One plain page, a row a shot: who is in it and on whose word, each tracker's numbered picture of the shot, and the owner's words on it.

    <python> bench/who_page.py --shots data/<stem>/shots.json --preview JOB.json=TABLES_DIR [--preview ...] \\
        [--answers data/<stem>/poc/answers.jsonl] [--span FIRST-LAST | --shot FIRST ...] [--notes PREVIEWS_RUN.jsonl] \\
        [--source COPY --every-cut-of FIRST-LAST] [--cast data/<stem>/cast/cast.jsonl] --out <folder>

**What it buys.** The owner asked (2026-10-12) to see, in a super simple page, who is labelled in each shot, with
the masks and identifiers over the input video, so a subject who changes clothes or a shot where nobody was
replaced is seen before a render and not after. The facts already exist in two places that nobody can look at
together: the shots record (`data/<stem>/shots.json`: for each shot and subject, present / absent / unsure,
with who said so) and every preview's shot table, whose picture shows the shot's shown frame with every
detection outlined and numbered (`shot_table.py`; one band a shot, top to bottom). This page lays them side by
side, writes nothing else, and is static HTML with no script, so it opens from the share as a file.

**What a row holds.** The shot's frames; for each subject a line `yes` / `no` / `unsure`, then how and by whom
(the record's own words); for each preview whose job spans the shot, the band of each subject's table that is
that shot, cut from the table's picture and labelled with the tracker it belongs to and the frame shown; and
the owner's typed words on any numbered picture of those frames (the page's `answers.jsonl`, rows `b20` and
`who`, unreworded). A shot no preview covers says so.

**Who a numbered picture asks about.** The subjects of the job that made it, and no other: a preview tracks the
subjects its job names, and a number clicked for anybody else has no tracker to seed and no table of theirs to
take a box from (`cast_file.from_tile` refuses it). Until 2026-10-13 every tile asked about every subject of
the shots record, and most of a video's previews name one. A subject of the record the job does not track is
named under the tile in words, with no button.

**A span, and a folder of its own.** `--span FIRST-LAST` shows only the record's shots that have a frame inside
it, so a page can be made for a stretch of a video whose record is one file. The page says where it is answered
from by its `--out` folder's name (`/<folder>/index.html` on the page server, which serves the folder that holds
it), and it writes `page.json` beside itself: its span, its previews and when it was built. A page is NEVER
WRITTEN OVER ANOTHER: an `--out` that holds a page of another span, or one that does not say its span (a page
made before 2026-10-13), is refused by name, since the owner's answers were given on the page as it stood. The
same span is built again in place. `--shot FIRST`, repeatable, names the record's shots one by one in place of a
span: the questions still open on shots far apart are put to the owner on one page, in one batch (2026-10-13). Such
a page says which shots it is of, and one of other shots is not written over it.

**A session's reading, as a guess.** `--notes` takes the file a session wrote when it looked at each preview's
tiles (one line a preview: `job_file`, `first`, and per subject `picked_number`, `picked_looks_like`,
`right_person`, `why`, and from the batch after 2026-10-13 `thinks_it_is`: the number, on the subject's own
tile, of the person the reader thinks IS the subject, or `none of the outlined people`; a reader who says a
pick is wrong is asked to write it, since `bench/picks_against_answers.py` holds it against the owner's
click and cannot read it out of the free text of `why`), as it is. It is shown under the tile of the preview it names, in a fold that is closed,
headed as a session's guess and not the owner's, with the file and the time it was recorded: read after the
click, not before it. It presses no button and colors none. A line that names no preview given is printed when
the page is built; a tile with no line says so.

**A shot shown cut by cut, with boxes in color.** `--every-cut-of FIRST-LAST` names a shot of the record
that the tracker cut into several of its own (2026-10-13: the record's one shot 5935-6394 was twenty-five to the
tracker, the page's one picture of it was a one-frame cut, and the owner could not tell whether a number was the
person's they meant). The page then shows one picture for each of that tracker's cuts as long as the shortest
load (`bench/masked_job.py::shortest_load`: a shorter cut cannot be a load of its own), each on the frame the
tracker numbered, with its own buttons; the shorter cuts are counted and named, not shown. With `--source` each
is three pictures: the frame as the tracker saw it (`fitted_frame`), the same frame with EACH PERSON'S BOX in a
color of their own and their number in that color on a dark plate at the box's middle (`boxes_picture`, from
the table's boxes; the number buttons carry the same colors), and the tracker's own picture. A box is a box:
where two cross, the picture cannot say whose pixels are whose, and the page says so. A fill of each person's
own pixels needs the tracker's masks, which no preview saves; that is node work and is not here.

**What already stands is printed, so nothing is asked twice as if it were new.** With `--cast`, above a
subject's buttons on a numbered picture: every answer that stands in the cast file for that subject on a frame of
the shot, with its frame, what it says (a person of this picture by number where the box is this picture's own,
else the box; or `none of the outlined people`) and who gave it and when; or that nothing stands. The owner's
rule (2026-10-13): what they already answered is read before they are asked, and only the missing fact is asked.
It presses no button.

**What it does not do.** Outside a shot shown cut by cut it draws nothing: the numbers and outlines are the
table's own, made by the tracker when the preview ran, so a later correction is not on them. The numbered
picture is one frame a shot, the table's shown frame. A shot longer than `LONG_SHOT_FRAMES` (or `--long-shot-frames`) also shows the source's
own frame at its first and last quarter, with the two frame buttons and no numbers, so a subject who walks
in after the shown frame can be said to be on THAT frame (the owner's ask, 2026-10-12: late entrances were
not labelled); `bench/page_answers_to_cast.py --shots` takes such answers into the shots record's `present`, and
nothing of them into the cast file. Parts in
color and the pose model's points are the layered view's job (`docs/wiki/next_steps.md`), not this page's.
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PRESENT_WORD = {"yes": "yes", "no": "no", "unsure": "UNSURE"}
#: A shot longer than this shows its first- and last-quarter frames as well as the shown one. Reasoned, a
#: starting value: four seconds at 24 frames a second; a person can walk into a shot and settle inside a
#: second, so a shot this long has room for an entrance the one shown frame misses, and a shorter one shows
#: little that its middle frame does not. Change it when the owner says the extra frames of a shot were noise.
LONG_SHOT_FRAMES = 96


#: A color for each person on a picture of boxes, by their number. Reasoned, starting values: twelve hues that
#: stand apart from one another on a dark plate and over skin and cloth. With more people than colors they come
#: round again, and the legend then shows two numbers in one color.
BOX_COLOURS = ((255, 221, 0), (0, 229, 255), (255, 64, 255), (118, 255, 3), (64, 128, 255), (255, 255, 255),
               (255, 128, 171), (29, 233, 182), (179, 136, 255), (255, 171, 64), (128, 216, 255), (204, 255, 144))
BOXES_SAY = ("A box is a box: the smallest upright rectangle round the tracker's outline of a person. Where two boxes cross, "
             "this picture cannot say whose pixels are whose; the tracker's own picture carries the outlines.")


def colour_of(person: int) -> tuple[int, int, int]:
    return BOX_COLOURS[(int(person) - 1) % len(BOX_COLOURS)]


def stood(standing: dict | None, subject: str, shot: dict, t: dict) -> str:
    """What already stands in the cast file for a subject on the frames of a shot, as a line above that subject's buttons.

    The owner's rule on being asked again (2026-10-13): before a page asks them anything, what they already answered
    is read, and only the missing fact is asked. A box that is one of this picture's own (the same frame, the same
    box) is said by its number here; any other is said by its frame and box. Nothing is pre-chosen: no button is
    pressed by it."""
    if standing is None:
        return ""
    mine = sorted((line for (sub, frame), line in standing.items() if sub == subject and shot["first"] <= frame <= shot["last"]), key=lambda line: line["frame"])
    if not mine:
        return f'<div class="stood none">nothing stands in the cast file for {html.escape(subject)} on a frame of this shot</div>'
    said = []
    for line in mine:
        if line["answer"] != "here":
            what = line["answer"]
        else:
            number = next((p["person"] for p in t.get("boxes", []) if line["frame"] == t["shown"] and p["box"] and [float(v) for v in p["box"]] == line["box"]), None)
            what = f"person {number} on this picture" if number is not None else f"a box {[int(v) if float(v).is_integer() else v for v in line['box']]} (left, top, width, height)"
        said.append(f"frame {line['frame']}: <b>{html.escape(what)}</b> ({html.escape(line['by'])})")
    return (f'<div class="stood">ALREADY ANSWERED for {html.escape(subject)} in this shot, from the cast file: ' + "; ".join(said)
            + '. Click only if that is wrong or this frame says something it does not.</div>')


def quarter_frames(shot: dict, long_frames: int = LONG_SHOT_FRAMES) -> list[int]:
    """The first- and last-quarter frames of a shot longer than `long_frames`; nothing for a shorter one."""
    n = shot["last"] - shot["first"] + 1
    return [shot["first"] + n // 4, shot["first"] + (3 * n) // 4] if n > long_frames else []


def previews(specs: list[str]) -> list[dict]:
    out = []
    for spec in specs:
        job_path, _, folder = spec.partition("=")
        if not folder:
            raise SystemExit(f"--preview {spec!r}: write JOB.json=TABLES_DIR")
        job = json.load(open(job_path))["job"]
        tables = {}
        for s in job["subjects"]:
            p = Path(folder) / f"{job['name']}_{s['label']}_shots_00001_shots.json"
            if not p.is_file():
                raise SystemExit(f"{spec}: no table {p}")
            tables[s["label"]] = json.loads(p.read_text())
        out.append({"name": job["name"], "first": int(job["first"]), "last": int(job["first"]) + int(job["frames"]) - 1,
                    "folder": Path(folder), "tables": tables, "subjects": list(tables)})
    return out


def band(table: dict, picture: Path, k: int) -> Image.Image:
    """The k-th band of a shot table's picture (one band a shot, top to bottom, all of one height)."""
    sheet = Image.open(picture)
    n = len(table["shots"])
    h = sheet.size[1] // n
    if h * n != sheet.size[1]:
        raise SystemExit(f"{picture}: {sheet.size[1]} rows do not divide into {n} bands")
    return sheet.crop((0, k * h, sheet.size[0], (k + 1) * h))


def tiles_for(shot: dict, pv: dict, out: Path, nobody_too: bool = False) -> list[dict]:
    """Each subject's band of this preview's table whose shot overlaps the record's shot, cut and saved. A band with
    nobody on it is left out (said in words, not shown black) unless `nobody_too`, which a shot shown cut by cut
    asks for: a long cut the tracker outlined nobody on is still a cut the owner is asked about."""
    found = []
    for label, table in pv["tables"].items():
        inside = [(k, ts) for k, ts in enumerate(table["shots"])
                  if not (pv["first"] + int(ts["last_frame"]) < shot["first"] or pv["first"] + int(ts["first_frame"]) > shot["last"])]
        for k, ts in [x for x in inside if x[1].get("people") or nobody_too]:
            lo, hi = pv["first"] + int(ts["first_frame"]), pv["first"] + int(ts["last_frame"])
            picture = pv["folder"] / f"{pv['name']}_{label}_shots_00001_shots.png"
            name = f"{shot['first']:04d}_{pv['name']}_{label}_shot{int(ts['shot'])}.jpg"
            (out / "tiles").mkdir(parents=True, exist_ok=True)
            band(table, picture, k).save(out / "tiles" / name, quality=88)
            found.append({"file": f"tiles/{name}", "label": label, "preview": pv["name"], "preview_first": pv["first"], "shot": int(ts["shot"]),
                          "shown": pv["first"] + int(ts["shown_frame"]), "table_frame": int(ts["shown_frame"]),
                          "table": picture.name.replace(".png", ".json"), "frames": [lo, hi],
                          "people": [int(p["person"]) for p in ts.get("people", [])], "subjects": pv["subjects"],
                          "boxes": [{"person": int(p["person"]), "box": p.get("box")} for p in ts.get("people", [])], "size": table.get("size"),
                          "partial": lo < shot["first"] or hi > shot["last"]})
    return found


def owner_words(answers: list[dict], first: int, last: int) -> list[dict]:
    out = []
    for d in answers:
        if d.get("who") != "owner" or d.get("row") not in ("b20", "who"):
            continue
        try:
            f = int(str(d.get("source_frame", "")))
        except ValueError:
            continue
        if first <= f <= last and str(d.get("words", "")).strip():
            out.append({"frame": f, "at": str(d.get("at", ""))[:16], "words": str(d["words"]).strip()})
    return out


def fitted_frame(source: str, frame: int, size: list[int], out: Path) -> str | None:
    """The source's own frame as a preview's tracker saw it: cut to the table's proportions about the middle and
    scaled to the table's size, which is what the job's video loader does with its width and height
    (`_lib.frames.FIT`, the one place that crop is written). A box of the table then lies on this picture pixel
    for pixel."""
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    w, h = int(size[0]), int(size[1])
    name = f"{frame:04d}_as_tracked_{w}x{h}.jpg"
    from _lib.frames import FIT        # the loader's crop and scale, in its one place
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-vf", f"select=eq(n\\,{frame})," + FIT.format(w=w, h=h),
                        "-frames:v", "1", str(out / "tiles" / name)], capture_output=True, text=True)
    return f"tiles/{name}" if r.returncode == 0 and (out / "tiles" / name).is_file() else None


def boxes_picture(fitted: str, t: dict, out: Path) -> str:
    """The fitted frame with each person's box in a color of their own and their number, in that color on a dark
    plate. The larger box is drawn first, so a small one inside it is not drawn over.

    WHERE A NUMBER GOES. Every point of the picture belongs to the SMALLEST box that covers it, as a map gives a
    town to its county and not to its country. A person's number is put on their own ground, as near their box's
    middle as a whole plate fits (else where most of a plate fits); a person with no ground of their own, every
    point of their box inside a smaller one, has their number at their box's top left corner. Until 2026-10-13
    the number sat at the box's middle whatever lay there, and a box that is nearly the whole frame put its
    number on somebody else (a reader's look at the first page of 5935-6394). A plate that would still lie on
    another is moved down by its own height."""
    import numpy as np
    im = Image.open(out / fitted).convert("RGB")
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=max(18, im.size[1] // 24))
    except TypeError:      # an older Pillow has one size
        font = ImageFont.load_default()
    people = sorted((p for p in t["boxes"] if p["box"]), key=lambda p: -(p["box"][2] * p["box"][3]))
    for p in people:
        x, y, w, h = p["box"]
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=colour_of(p["person"]), width=max(2, im.size[1] // 250))
    ground = np.zeros((im.size[1], im.size[0]), np.int32)        # whose ground each point is: the smallest box over it
    for p in people:                                              # largest first, so a smaller box takes its ground back
        x, y, w, h = (int(v) for v in p["box"])
        ground[max(y, 0):y + h, max(x, 0):x + w] = p["person"]
    plates = []
    for p in sorted(people, key=lambda p: p["person"]):
        x, y, w, h = p["box"]
        text = str(p["person"])
        tb = draw.textbbox((0, 0), text, font=font)
        pw, ph = int(tb[2] - tb[0] + 14), int(tb[3] - tb[1] + 12)
        px, py = min(max(x + w / 2 - pw / 2, 0), im.size[0] - pw), min(max(y + h / 2 - ph / 2, 0), im.size[1] - ph)
        own = np.pad((ground == p["person"]).astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        if own[-1, -1] == 0:
            px, py = min(max(x, 0), im.size[0] - pw), min(max(y, 0), im.size[1] - ph)
        elif pw < im.size[0] and ph < im.size[1]:
            held = own[ph:, pw:] - own[:-ph, pw:] - own[ph:, :-pw] + own[:-ph, :-pw]      # of a plate with its top left at each point, how much is own ground
            ys, xs = np.nonzero(held == held.max())
            nearest = int(np.argmin((xs - px) ** 2 + (ys - py) ** 2))
            px, py = float(xs[nearest]), float(ys[nearest])
        while any(abs(px - qx) < pw and abs(py - qy) < ph for qx, qy in plates) and py + 2 * ph <= im.size[1]:
            py += ph
        plates.append((px, py))
        draw.rectangle((px, py, px + pw, py + ph), fill=(16, 16, 16), outline=colour_of(p["person"]), width=2)
        draw.text((px + 7 - tb[0], py + 6 - tb[1]), text, fill=colour_of(p["person"]), font=font)
    name = f"{t['shown']:04d}_{t['preview']}_{t['label']}_shot{t['shot']}_boxes.jpg"
    im.save(out / "tiles" / name, quality=90)
    return f"tiles/{name}"


def plain_frame(source: str, frame: int, out: Path) -> str | None:
    """The source's own frame, cut with ffmpeg at the page's width, for a shot with no numbered picture."""
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    name = f"{frame:04d}_plain.jpg"
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source, "-vf", f"select=eq(n\\,{frame}),scale=768:-2",
                        "-frames:v", "1", str(out / "tiles" / name)], capture_output=True, text=True)
    return f"tiles/{name}" if r.returncode == 0 and (out / "tiles" / name).is_file() else None


CSS = """:root{color-scheme:dark}body{font:17px/1.45 system-ui,sans-serif;margin:16px;background:#14161a;color:#e6e6e6}
table{border-collapse:collapse;width:100%}td,th{border-top:1px solid #333;padding:10px;vertical-align:top;text-align:left}
th{background:#1e2127;position:sticky;top:0}.f{font-family:ui-monospace,monospace;white-space:nowrap;font-size:18px}
.no{color:#8a8f99}.un{color:#ff8a80;font-weight:600}a{color:#8ab4f8}
.tiles{display:flex;flex-wrap:wrap;gap:12px}.tile img{width:768px;max-width:100%;display:block;border:1px solid #333}
.tile{font-size:14px;color:#aab}.ask{margin:4px 0}.lab{display:inline-block;min-width:70px;color:#ddd}
button{font:inherit;font-size:14px;margin:2px;padding:4px 10px;border:1px solid #555;border-radius:4px;background:#23272e;color:#eee;cursor:pointer}
button.on{background:#3b6cf0;border-color:#3b6cf0}.st{font-size:13px;color:#9c9}textarea{width:100%;min-height:60px;background:#1b1e24;color:#eee;border:1px solid #555;font:inherit}
.guess{margin-top:8px;font-size:14px;color:#aab}.guess summary{cursor:pointer;color:#c9a227}
.three{display:flex;flex-wrap:wrap;gap:8px}.three img{width:640px}.chip{display:inline-block;padding:0 7px;border:2px solid;border-radius:3px;background:#101010;font-weight:700}
.stood{margin-top:6px;font-size:14px;color:#9ccc65}.stood.none{color:#9aa3b2}
.say{margin-top:8px}#me{background:#1b1e24;color:#eee;border:1px solid #555;padding:4px}.w{font-size:16px;color:#ddd;white-space:pre-line;margin-bottom:8px}.how{font-size:14px;color:#9aa3b2}"""

SCRIPT = """<script>
(function(){
  var V = document.title; var me = document.getElementById('me');
  try { me.value = localStorage.getItem('whoMe') || 'owner'; } catch (e) {}
  me.addEventListener('input', function(){ try { localStorage.setItem('whoMe', me.value); } catch (e) {} });
  function post(line, st, done){
    line.who = me.value.trim() || 'not said';
    st.textContent = 'saving';
    fetch('/api/answer', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(line)})
      .then(function(r){ return r.json().then(function(j){ if(!r.ok) throw new Error(j.error||r.status); return j; }); })
      .then(function(j){ st.textContent = 'saved ' + j.saved.at + ' (' + j.lines + ' lines)'; if (done) done(); })
      .catch(function(e){ st.textContent = 'NOT saved: ' + e.message + ' (open this page through the local page server, port 8765, not from the share)'; });
  }
  document.querySelectorAll('.tile button').forEach(function(b){
    b.addEventListener('click', function(){
      var t = b.closest('.tile'); var sub = b.dataset.sub, chosen = b.dataset.chosen;
      var line = {row:'who', what:'choice', subject: sub, chosen: chosen, person: /^\\d+$/.test(chosen) ? parseInt(chosen,10) : null,
                  source_frame: parseInt(t.dataset.frame,10), table: t.dataset.table, table_frame: parseInt(t.dataset.tf,10),
                  shot: parseInt(t.dataset.shot,10), frames:[parseInt(t.dataset.lo,10), parseInt(t.dataset.hi,10)], video: V};
      post(line, t.querySelector('.st'), function(){
        t.querySelectorAll('button[data-sub="'+sub+'"]').forEach(function(x){ x.classList.remove('on'); }); b.classList.add('on'); });
    });
  });
  document.querySelectorAll('.say .send').forEach(function(b){
    b.addEventListener('click', function(){
      var d = b.closest('.say'); var ta = d.querySelector('textarea'); if (!ta.value.trim()) return;
      post({row:'who', what:'words', words: ta.value, source_frame: parseInt(d.dataset.lo,10),
            frames:[parseInt(d.dataset.lo,10), parseInt(d.dataset.hi,10)], video: V}, d.querySelector('.st'), function(){ ta.value=''; });
    });
  });
})();
</script>"""


def plain_tile(shots: dict, s: dict, frame: int, file: str, why: str) -> str:
    """A source frame with the two frame buttons a subject: nothing outlined, so no numbers to choose from."""
    ask = "".join(
        f'<div class="ask"><span class="lab">{html.escape(sub)} is</span>'
        + f'<button data-sub="{html.escape(sub)}" data-chosen="not outlined">on this frame, not outlined</button>'
        + f'<button data-sub="{html.escape(sub)}" data-chosen="not on frame">not on this frame</button></div>'
        for sub in shots["subjects"])
    return (f'<div class="tile" data-table="plain" data-frame="{frame}" data-tf="0" data-shot="0" '
            f'data-lo="{s["first"]}" data-hi="{s["last"]}"><img src="{file}" alt="">'
            f'<div>frame {frame}, the source itself: {why}, so nothing is outlined</div>{ask}<div class="st"></div></div>')


def notes_of(path: str | None) -> dict[tuple[str, int], dict] | None:
    """A session's notes on the previews' tiles, by (the preview's job name, its first frame); None when none are given."""
    if not path:
        return None
    out = {}
    for n, text in enumerate(Path(path).read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if "job_file" not in row or "first" not in row:
            raise SystemExit(f"{path} line {n}: a note names its preview by `job_file` and `first`")
        out[(Path(row["job_file"]).stem, int(row["first"]))] = {**row, "from": Path(path).name, "line": n}
    return out


def guess(notes: dict | None, t: dict, found: list[dict] = ()) -> str:
    """A session's reading of one tile's preview, in a closed fold under it: a guess, labelled, that answers nothing.

    A NUMBER IN A NOTE IS OF THAT SUBJECT'S OWN TRACKER'S PICTURE, and the page shows one picture a shot. Each tracker's
    table shows a frame of its own choosing and numbers the people on THAT frame, so where the subject's tracker
    showed another frame than the picture above, its number names somebody else there, and the note says so in
    capitals (2026-10-13: three of the first page's forty-two folds gave such a number with nothing said). `found`
    is every tracker's tile of the shot, the one shown among them."""
    if notes is None:
        return ""
    row = notes.get((t["preview"], t["preview_first"]))
    if row is None:
        return '<div class="how">no session\'s reading of this preview is in the notes this page was built with</div>'

    def whose_picture(label: str) -> str:
        own = sorted(x["shown"] for x in found if x["preview"] == t["preview"] and x["label"] == label)
        if t["shown"] in own or (not found and label == t["label"]):
            return ""
        where = f"frame {', '.join(str(f) for f in own)}" if own else "a frame not among this shot's pictures"
        return (f' <b>THAT NUMBER IS NOT OF THE PICTURE ABOVE</b>: it is of {html.escape(label)}\'s own tracker\'s picture ({where}), '
                f'and the picture above is frame {t["shown"]}, where the people are numbered afresh.')
    said = "".join(
        f'<div><b>{html.escape(str(s.get("label", "")))}</b>: the tracker picked person {html.escape(str(s.get("picked_number", "not said")))}'
        f' ({html.escape(str(s.get("picked_looks_like", "")))}).{whose_picture(str(s.get("label", "")))} Is that the right person, by this reading: '
        f'<b>{html.escape(str(s.get("right_person", "not said")))}</b>.'
        + (f' The reader thinks the subject is: <b>{html.escape(str(s["thinks_it_is"]))}</b> (on that tracker\'s own picture).' if s.get("thinks_it_is") is not None else "")
        + f' {html.escape(str(s.get("why", "")))}</div>' for s in row.get("subjects", []))
    return (f'<details class="guess"><summary>a session\'s guess, NOT the owner\'s: open it after answering</summary>'
            f'<div class="how">A session looked at this preview\'s tiles and wrote this ({html.escape(row["from"])} line {row["line"]}, recorded '
            f'{html.escape(str(row.get("recorded_at", "at a time it does not give")))}). It is a reading of a picture, not an answer, and nothing '
            f'here is stored.</div>{said}</details>')


def span_of(text: str | None, shots: dict) -> list[int]:
    """The frames a page is of: `--span FIRST-LAST`, or the record's own span when none is given."""
    if not text:
        return [int(shots["span"][0]), int(shots["span"][1])]
    first, _, last = text.partition("-")
    if not (first.isdigit() and last.isdigit()) or int(first) > int(last):
        raise SystemExit(f"--span {text!r}: write FIRST-LAST, two source frames, as in 1920-6394")
    return [int(first), int(last)]


def not_over_another(out: Path, span: list[int], only: list[int] | None = None) -> None:
    """Refuse an `--out` that holds another page: one of another span or of other named shots, or one that does not say its span."""
    if not (out / "index.html").is_file():
        return
    said = json.loads((out / "page.json").read_text()).get("span") if (out / "page.json").is_file() else None
    if said is None:
        raise SystemExit(f"{out} holds a page that does not say its span (no page.json: it was made before a page said so). A page is not "
                         "written over another, since its answers were given on it as it stood: give this one an --out of its own")
    if said != span:
        raise SystemExit(f"{out} holds the page of frames {said[0]}-{said[1]} and this one is of {span[0]}-{span[1]}. A page is not written "
                         "over another: give this one an --out of its own")
    named = json.loads((out / "page.json").read_text()).get("only_shots")
    if named != only:
        raise SystemExit(f"{out} holds the page of {'the shots beginning on ' + str(named) if named else 'every shot of its span'} and this one is of "
                         f"{'the shots beginning on ' + str(only) if only else 'every shot of its span'}. A page is not written over another: give this one "
                         "an --out of its own")


def page(shots: dict, previews_: list[dict], answers: list[dict], out: Path, plan: dict | None = None, source: str | None = None,
         long_frames: int = LONG_SHOT_FRAMES, span: list[int] | None = None, notes: dict | None = None,
         every_cut: list[list[int]] | None = None, least_cut: int = 1, standing: dict | None = None, only: list[int] | None = None) -> str:
    """`every_cut` names record shots (first, last) shown cut by cut: one picture for each of the tracker's own cuts
    of the shot that is `least_cut` frames or longer, where every other shot has one picture."""
    rows = []
    span = span or [int(shots["span"][0]), int(shots["span"][1])]
    for s in shots["shots"]:
        if s["last"] < span[0] or s["first"] > span[1] or (only and s["first"] not in only):
            continue
        who = []
        for label in shots["subjects"]:
            w = s["who"][label]
            cls = {"yes": "", "no": "no", "unsure": "un"}[w["present"]]
            who.append(f'<div class="{cls}"><b>{html.escape(label)}</b>: {PRESENT_WORD[w["present"]]}'
                       f'<div class="how">{html.escape(w.get("how", ""))} <i>({html.escape(w.get("by", ""))})</i></div></div>')
        found, covered = [], False
        by_cut = [s["first"], s["last"]] in (every_cut or [])
        for pv in previews_:
            if pv["last"] < s["first"] or pv["first"] > s["last"]:
                continue
            covered = True
            found.extend(tiles_for(s, pv, out, nobody_too=by_cut))
        every, found = found, [x for x in found if x["people"]]
        # ONE picture a shot: every tracker of a preview drew its own picture of the same shot with its own numbering,
        # and asking the same question under each confused the owner (2026-10-12). The picture with the most people
        # outlined is shown; ties go to the first preview and the first tracker listed.
        tiles = []
        if not found and source:          # nobody outlined, or no preview at all: the frame itself
            mid = (s["first"] + s["last"]) // 2
            f = plain_frame(source, mid, out)
            if f:
                tiles.append(plain_tile(shots, s, mid, f, "the trackers found nobody on this shot" if covered else "no preview covers this shot"))
        if found:
            mid = (s["first"] + s["last"]) // 2
            t = max(found, key=lambda x: (len(x["people"]), -abs(x["shown"] - mid)))   # most people; then nearest the middle
            shown, said = [t], ""
            if by_cut:
                # that tracker's cuts of the shot, each as long as a load can be: a shorter cut cannot be a load of its
                # own, so who is on it decides nothing by itself. A long cut with nobody outlined is asked about plain.
                length = lambda x: x["frames"][1] - x["frames"][0] + 1                                               # noqa: E731
                cuts = sorted((x for x in every if (x["preview"], x["label"]) == (t["preview"], t["label"])), key=lambda x: x["frames"][0])
                long = [x for x in cuts if length(x) >= least_cut]
                short = [x for x in cuts if length(x) < least_cut]
                shown = [x for x in long if x["people"]] or [t]
                said = (f'<div class="how"><b>This shot is shown cut by cut.</b> {html.escape(t["label"])}\'s tracker ({html.escape(t["preview"])}) cut it into '
                        f'{len(cuts)}; the {len(long)} of {least_cut} frames or more (the shortest a load can be) are shown, each on the frame the tracker '
                        f'numbered, holding {sum(length(x) for x in long)} of the shot\'s {s["last"] - s["first"] + 1} frames. '
                        f'{len(short)} shorter cut(s) are not shown: ' + (", ".join(f"{x['frames'][0]}-{x['frames'][1]}" for x in short) or "none") + '.</div>')
                if source:
                    for x in (x for x in long if not x["people"]):
                        f = plain_frame(source, x["shown"], out)
                        if f:
                            tiles.append(plain_tile(shots, s, x["shown"], f, f"the tracker outlined nobody on its cut {x['frames'][0]}-{x['frames'][1]}"))
            others = [x for x in found if x not in shown and not (by_cut and (x["preview"], x["label"]) == (t["preview"], t["label"]))]
        for t in (shown if found else []):
            part = " (the table's shot runs %d-%d)" % tuple(t["frames"]) if t["partial"] else ""
            # only the subjects this tile's job tracks, in the record's order (then any the record does not name)
            asked = [sub for sub in shots["subjects"] if sub in t["subjects"]] + [sub for sub in t["subjects"] if sub not in shots["subjects"]]
            # a picture of boxes for a shot shown cut by cut: the frame as the tracker saw it, plain, and again with each person's box
            fitted = fitted_frame(source, t["shown"], t["size"], out) if by_cut and source and t.get("size") else None
            boxed = boxes_picture(fitted, t, out) if fitted else None
            inked = lambda n: (' style="color:rgb(%d,%d,%d);border-color:rgb(%d,%d,%d)"' % (colour_of(n) * 2)) if boxed else ""       # noqa: E731
            cut = f', its cut {t["frames"][0]}-{t["frames"][1]}' if by_cut else ""
            pictures = f'<img src="{t["file"]}" alt="">'
            if boxed:
                legend = " ".join(f'<span class="chip" style="color:rgb(%d,%d,%d);border-color:rgb(%d,%d,%d)">{p["person"]}</span>' % (colour_of(p["person"]) * 2)
                                  for p in sorted(t["boxes"], key=lambda p: p["person"]) if p["box"])
                pictures = (f'<div class="three"><div><img src="{fitted}" alt=""><div>frame {t["shown"]}, the source itself</div></div>'
                            f'<div><img src="{boxed}" alt=""><div>the same frame, each person\'s box in a color of their own: {legend}</div></div>'
                            f'<div><img src="{t["file"]}" alt=""><div>the tracker\'s own picture of it, with its outlines</div></div></div>'
                            f'<div class="how">{BOXES_SAY}</div>')
            elif by_cut:
                pictures += '<div class="how">no picture of boxes: ' + ("no --source was given" if not source else "the frame could not be cut from the source") + '</div>'
            ask = "".join(
                stood(standing, sub, s, t) + f'<div class="ask"><span class="lab">{html.escape(sub)} is</span>'
                + "".join(f'<button data-sub="{html.escape(sub)}" data-chosen="{n}"{inked(n)}>{n}</button>' for n in t["people"])
                + f'<button data-sub="{html.escape(sub)}" data-chosen="not outlined">on this frame, not outlined</button>'
                + f'<button data-sub="{html.escape(sub)}" data-chosen="not on frame">not on this frame</button></div>'
                for sub in asked)
            unasked = [sub for sub in shots["subjects"] if sub not in t["subjects"]]
            if unasked:
                ask += (f'<div class="how">not asked here: {html.escape(", ".join(unasked))}. This preview\'s job ({html.escape(t["preview"])}) '
                        f'tracks {html.escape(", ".join(t["subjects"]))} and nobody else, so a number for another subject has no tracker to go to.</div>')
            also = (f'<div class="how">{len(others)} other picture(s) of this shot exist (' + "; ".join(
                f"{html.escape(x['label'])}'s tracker, frame {x['shown']}, people {x['people'] or 'none'}" for x in others)
                + '); not shown, one question a shot</div>') if others and t is shown[0] else ""
            also = (said if t is shown[0] else "") + also
            tiles.append(f'<div class="tile" data-table="{html.escape(t["table"])}" data-frame="{t["shown"]}" data-tf="{t["table_frame"]}" '
                         f'data-shot="{t["shot"]}" data-lo="{s["first"]}" data-hi="{s["last"]}">{pictures}'
                         f'<div>frame {t["shown"]} as {html.escape(t["label"])}\'s tracker numbered it ({html.escape(t["preview"])} shot {t["shot"]}{cut}), '
                         f'people {t["people"] or "none"}{part}</div>{ask}<div class="st"></div>{also}{guess(notes, t, found)}</div>')
        if source:      # a long shot: its first and last quarter as well, so an entrance after the shown frame has a frame to be said on
            for q, which in zip(quarter_frames(s, long_frames), ("first", "last")):
                f = plain_frame(source, q, out)
                if f:
                    tiles.append(plain_tile(shots, s, q, f, f"the {which} quarter of a shot of {s['last'] - s['first'] + 1} frames, shown because one frame can miss an entrance"))
        words = owner_words(answers, s["first"], s["last"])
        wtxt = "".join(f'<div class="w">frame {w["frame"]} ({w["at"]}): {html.escape(w["words"])}</div>' for w in words)
        note = f'<div class="how">{html.escape(s["note"])}</div>' if s.get("note") else ""
        box = (f'<div class="say" data-lo="{s["first"]}" data-hi="{s["last"]}"><textarea placeholder="your words on {s["first"]}-{s["last"]}"></textarea>'
               f'<button class="send">save</button><span class="st"></span></div>')
        rows.append(f'<tr><td class="f">{s["first"]}-{s["last"]}<br><span class="how">{s["last"] - s["first"] + 1} f</span></td>'
                    f'<td>{"".join(who)}{note}</td><td><div class="tiles">{"".join(tiles) or ("<span class=no>the trackers found nobody on this shot</span>" if covered else "<span class=no>no preview covers this shot</span>")}</div></td>'
                    f'<td>{wtxt or "<span class=no>none yet</span>"}{box}</td></tr>')
    head = (f'<p>{html.escape(shots["video"])}: '
            + (f'the {len(only)} shot(s) named for this page, beginning on {", ".join(map(str, only))}; every other shot of the record is left off'
               if only else f'the shots of frames {span[0]}-{span[1]}')
            + (f' (the record holds {shots["span"][0]}-{shots["span"][1]})' if span != [int(shots["span"][0]), int(shots["span"][1])] else "")
            + (f' of {shots["source_frames"]} frames' if shots.get("source_frames") else "")
            + f'. Record: {html.escape(shots.get("written_by", ""))}</p>'
            f'<p>A numbered picture is the tracker\'s own at the time of its preview: outlines and numbers as it drew them, '
            f'one frame a shot; a shot longer than {long_frames} frames also shows its first- and last-quarter frames plain. Previews: {", ".join(html.escape(p["name"]) + " (" + str(p["first"]) + "-" + str(p["last"]) + ")" for p in previews_)}.</p>')
    replaced = ""
    if plan:
        replaced = ('<p><b>What is replaced, per subject (the plan, not the outline):</b> '
                    + "; ".join(f'{html.escape(k)}: {html.escape(v["region"])}' for k, v in plan["subjects"].items())
                    + '. The outline on a picture is the tracker\'s whole-person mask, which says who the person is and nothing about what is replaced.</p>')
    me = ('<p>Answering as <input id="me" value="owner" size="10"> (a click or a save is one line appended to the page\'s '
          'answers file, through the local page\'s server). <b>To answer, open this page as '
          f'<code>http://&lt;the page server&gt;:8765/{html.escape(out.name)}/index.html</code></b> (the server serves files, not folders, so the '
          'file name is needed); opened from the share as a file it only shows. An answer is about THAT FRAME, never about the whole shot: for each '
          'subject, which number they are, <b>on this frame, not outlined</b>, or <b>not on this frame</b>. Whether they are in the shot at '
          'all is a word in the box, and the tracker is left to judge the other frames.</p>')
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(shots["video"])}</title><style>{CSS}</style>'
            f'<h1>Who is in each shot</h1>{head}{replaced}{me}<table><tr><th>frames</th><th>who, and on whose word</th>'
            f'<th>the trackers\' numbered pictures</th><th>the owner\'s words on these frames</th></tr>{"".join(rows)}</table>{SCRIPT}')


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--shots", required=True)
    ap.add_argument("--preview", action="append", default=[], metavar="JOB.json=TABLES_DIR")
    ap.add_argument("--answers")
    ap.add_argument("--plan", help="the video's plan file; its per-subject region is printed in the header so the outline is not read as the region")
    ap.add_argument("--source", help="the source copy the frame numbers are of; with it, a shot the trackers found nobody on shows its own middle frame")
    ap.add_argument("--long-shot-frames", type=int, default=LONG_SHOT_FRAMES,
                    help=f"a shot longer than this also shows its first- and last-quarter source frames (needs --source); {LONG_SHOT_FRAMES} by default, a starting value")
    ap.add_argument("--span", metavar="FIRST-LAST", help="only the record's shots with a frame inside these source frames; the record's own span when left out")
    ap.add_argument("--shot", action="append", default=[], type=int, metavar="FIRST",
                    help="only this shot of the record, by its first frame; repeatable, in place of --span")
    ap.add_argument("--notes", metavar="PREVIEWS_RUN.jsonl", help="a session's notes on the previews' tiles, one line a preview; shown under each tile in a "
                    "closed fold, headed as a session's guess and not the owner's")
    ap.add_argument("--every-cut-of", action="append", default=[], metavar="FIRST-LAST",
                    help="a shot of the record to show cut by cut: one picture for each of the tracker's own cuts of it as long as the shortest load "
                    "(bench/masked_job.py::shortest_load), and with --source each frame plain and again with every person's box in a color of their own; repeatable")
    ap.add_argument("--cast", metavar="CAST.jsonl", help="the video's cast file; with it, what already stands for a subject on a frame of a shot is printed above "
                    "that subject's buttons, so nothing answered is asked again as if it were not")
    ap.add_argument("--out", required=True, help="the page's own folder; its name is the page's address on the page server, and a page of another span there is refused")
    a = ap.parse_args()
    shots = json.load(open(a.shots))
    only = sorted(set(a.shot)) or None
    if only:
        by_first = {s["first"]: s for s in shots["shots"]}
        if a.span or any(f not in by_first for f in only):
            raise SystemExit(f"--shot names the record's shots by their first frames, in place of --span: {[f for f in only if f not in by_first] or '--span was given too'}")
        span = [only[0], by_first[only[-1]]["last"]]
    else:
        span = span_of(a.span, shots)
    not_over_another(Path(a.out), span, only)
    answers = [json.loads(t) for t in Path(a.answers).read_text().splitlines() if t.strip()] if a.answers else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.load(open(a.plan)) if a.plan else None
    made = previews(a.preview)
    notes = notes_of(a.notes)
    every_cut, least_cut = [], 1
    for text in a.every_cut_of:
        try:
            pair = [int(v) for v in text.split("-")]
        except ValueError:
            pair = []
        if len(pair) != 2 or not any([s["first"], s["last"]] == pair for s in shots["shots"]):
            raise SystemExit(f"--every-cut-of {text!r} is not a shot of the record (its shots are FIRST-LAST, whole)")
        every_cut.append(pair)
    if every_cut:
        import masked_job      # only here: the page needs nothing of the builder otherwise
        least_cut = masked_job.shortest_load()[1]
    standing = None
    if a.cast:
        import cast_file
        standing = cast_file.current(cast_file.read(a.cast))
    text = page(shots, made, answers, out, plan, a.source, a.long_shot_frames, span, notes, every_cut, least_cut, standing, only)
    for name, first in sorted(set(notes or {}) - {(p["name"], p["first"]) for p in made}):
        print(f"a note names a preview this page was not given, and is shown nowhere: {name} (first frame {first}), {notes[(name, first)]['from']} "
              f"line {notes[(name, first)]['line']}")
    (out / "index.html").write_text(text)
    (out / "page.json").write_text(json.dumps({
        "what": "what bench/who_page.py built this page from; a page of another span is not written over it", "span": span, "only_shots": only,
        "built": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "shots": str(a.shots),
        "notes": a.notes, "previews": [{"job": p["name"], "frames": [p["first"], p["last"]], "tables": str(p["folder"])} for p in made]}, indent=1) + "\n")
    tiles = sorted((out / "tiles").glob("*.jpg")) if (out / "tiles").is_dir() else []
    rows = text.count("<tr>") - 1
    print(f"written: {out / 'index.html'}; frames {span[0]}-{span[1]}; {rows} shot(s) of the record's {len(shots['shots'])}; {len(tiles)} tile(s) under "
          f"{out / 'tiles'}; answered from /{out.name}/index.html on the page server")
    return 0


if __name__ == "__main__":
    sys.exit(main())
