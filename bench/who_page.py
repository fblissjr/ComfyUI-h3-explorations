#!/usr/bin/env python3
"""One plain page, a row a shot: who is in it and on whose word, each tracker's numbered picture of the shot, and the owner's words on it.

    <python> bench/who_page.py --shots data/<stem>/shots.json --preview JOB.json=TABLES_DIR [--preview ...] \\
        [--answers data/<stem>/poc/answers.jsonl] --out <folder>

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

**What it does not do.** It draws nothing: the numbers and outlines are the table's own, made by the tracker
when the preview ran, so a later correction is not on them. The numbered picture is one frame a shot, the
table's shown frame. A shot longer than `LONG_SHOT_FRAMES` (or `--long-shot-frames`) also shows the source's
own frame at its first and last quarter, with the two frame buttons and no numbers, so a subject who walks
in after the shown frame can be said to be on THAT frame (the owner's ask, 2026-10-12: late entrances were
not labelled); `bench/page_answers_to_cast.py --shots` takes such answers into the shots record's `present`, and
nothing of them into the cast file. Parts in
colour and the pose model's points are the layered view's job (`docs/wiki/next_steps.md`), not this page's.
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

PRESENT_WORD = {"yes": "yes", "no": "no", "unsure": "UNSURE"}
#: A shot longer than this shows its first- and last-quarter frames as well as the shown one. Reasoned, a
#: starting value: four seconds at 24 frames a second; a person can walk into a shot and settle inside a
#: second, so a shot this long has room for an entrance the one shown frame misses, and a shorter one shows
#: little that its middle frame does not. Change it when the owner says the extra frames of a shot were noise.
LONG_SHOT_FRAMES = 96


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


def tiles_for(shot: dict, pv: dict, out: Path) -> list[dict]:
    """Each subject's band of this preview's table whose shot overlaps the record's shot, cut and saved."""
    found = []
    for label, table in pv["tables"].items():
        inside = [(k, ts) for k, ts in enumerate(table["shots"])
                  if not (pv["first"] + int(ts["last_frame"]) < shot["first"] or pv["first"] + int(ts["first_frame"]) > shot["last"])]
        for k, ts in [x for x in inside if x[1].get("people")]:    # a band with nobody on it is said in words, not shown black
            lo, hi = pv["first"] + int(ts["first_frame"]), pv["first"] + int(ts["last_frame"])
            picture = pv["folder"] / f"{pv['name']}_{label}_shots_00001_shots.png"
            name = f"{shot['first']:04d}_{pv['name']}_{label}_shot{int(ts['shot'])}.jpg"
            (out / "tiles").mkdir(parents=True, exist_ok=True)
            band(table, picture, k).save(out / "tiles" / name, quality=88)
            found.append({"file": f"tiles/{name}", "label": label, "preview": pv["name"], "shot": int(ts["shot"]),
                          "shown": pv["first"] + int(ts["shown_frame"]), "table_frame": int(ts["shown_frame"]),
                          "table": picture.name.replace(".png", ".json"), "frames": [lo, hi],
                          "people": [int(p["person"]) for p in ts.get("people", [])], "subjects": pv["subjects"],
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


def page(shots: dict, previews_: list[dict], answers: list[dict], out: Path, plan: dict | None = None, source: str | None = None,
         long_frames: int = LONG_SHOT_FRAMES) -> str:
    rows = []
    for s in shots["shots"]:
        who = []
        for label in shots["subjects"]:
            w = s["who"][label]
            cls = {"yes": "", "no": "no", "unsure": "un"}[w["present"]]
            who.append(f'<div class="{cls}"><b>{html.escape(label)}</b>: {PRESENT_WORD[w["present"]]}'
                       f'<div class="how">{html.escape(w.get("how", ""))} <i>({html.escape(w.get("by", ""))})</i></div></div>')
        found, covered = [], False
        for pv in previews_:
            if pv["last"] < s["first"] or pv["first"] > s["last"]:
                continue
            covered = True
            found.extend(tiles_for(s, pv, out))
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
            others = [x for x in found if x is not t]
            part = " (the table's shot runs %d-%d)" % tuple(t["frames"]) if t["partial"] else ""
            # only the subjects this tile's job tracks, in the record's order (then any the record does not name)
            asked = [sub for sub in shots["subjects"] if sub in t["subjects"]] + [sub for sub in t["subjects"] if sub not in shots["subjects"]]
            ask = "".join(
                f'<div class="ask"><span class="lab">{html.escape(sub)} is</span>'
                + "".join(f'<button data-sub="{html.escape(sub)}" data-chosen="{n}">{n}</button>' for n in t["people"])
                + f'<button data-sub="{html.escape(sub)}" data-chosen="not outlined">on this frame, not outlined</button>'
                + f'<button data-sub="{html.escape(sub)}" data-chosen="not on frame">not on this frame</button></div>'
                for sub in asked)
            unasked = [sub for sub in shots["subjects"] if sub not in t["subjects"]]
            if unasked:
                ask += (f'<div class="how">not asked here: {html.escape(", ".join(unasked))}. This preview\'s job ({html.escape(t["preview"])}) '
                        f'tracks {html.escape(", ".join(t["subjects"]))} and nobody else, so a number for another subject has no tracker to go to.</div>')
            also = (f'<div class="how">{len(others)} other picture(s) of this shot exist (' + "; ".join(
                f"{html.escape(x['label'])}'s tracker, frame {x['shown']}, people {x['people'] or 'none'}" for x in others)
                + '); not shown, one question a shot</div>') if others else ""
            tiles.append(f'<div class="tile" data-table="{html.escape(t["table"])}" data-frame="{t["shown"]}" data-tf="{t["table_frame"]}" '
                         f'data-shot="{t["shot"]}" data-lo="{s["first"]}" data-hi="{s["last"]}"><img src="{t["file"]}" alt="">'
                         f'<div>frame {t["shown"]} as {html.escape(t["label"])}\'s tracker numbered it ({html.escape(t["preview"])} shot {t["shot"]}), '
                         f'people {t["people"] or "none"}{part}</div>{ask}<div class="st"></div>{also}</div>')
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
    head = (f'<p>{html.escape(shots["video"])}: shots {shots["span"][0]}-{shots["span"][1]}'
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
          '<code>http://&lt;the page server&gt;:8765/who/index.html</code></b> (the server serves files, not folders, so the '
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shots = json.load(open(a.shots))
    answers = [json.loads(t) for t in Path(a.answers).read_text().splitlines() if t.strip()] if a.answers else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.load(open(a.plan)) if a.plan else None
    text = page(shots, previews(a.preview), answers, out, plan, a.source, a.long_shot_frames)
    (out / "index.html").write_text(text)
    tiles = sorted((out / "tiles").glob("*.jpg")) if (out / "tiles").is_dir() else []
    print(f"written: {out / 'index.html'}; {len(shots['shots'])} shot(s); {len(tiles)} tile(s) under {out / 'tiles'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
