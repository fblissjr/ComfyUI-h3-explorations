#!/usr/bin/env python3
"""Write a self-contained scoring page into a blind batch directory.

An instrument, not a check. `bench/blind_batch.py` produces the batch --
neutral `clip_NN.mp4` singles, `pair_NN.mp4` stacks, a MANIFEST that carries
row indices and nothing else -- and seals the arm labels in
`internal/blind_keys/<session>.json`. Scoring used to be a markdown table typed
by hand, which is fine for one session and loses a row the moment the judge
scrolls. This writes `score.html` beside the clips instead: every item in
MANIFEST order, one video each, its questions under it, and the answers held in
`localStorage` so a reload loses nothing.

## Pairs are the primary view

The page opens on the pairs tab, and a pair's primary input is a wide free-text
field -- what differs between the two halves, and which way. That is what a
judge can actually report; a number per clip rates a sample against a
remembered standard, and it was hard to give. Beside the text: quick tags,
clicked per half, and one coarse verdict. **No numeric scale on a pair.**

Singles are secondary and exist for what a stack cannot carry: the audio (the
stacker maps video only) and any defect in one clip on its own. One note, the
same tags, and a reject flag.

## What it may read

**MANIFEST.json, the rubric files, and the brief file. Nothing else.** Not the
sealed key, and not the JSONL the batch came from -- `run_graph_arms` rows
carry a `label` field, so opening the JSONL for so much as a seed would put arm
names into the page. Everything the page needs (item names and their order) is
in the MANIFEST.

## The rubric

`--rubric` names a JSON file holding an ordered list of questions, each
`{"id", "label", "type", "help"}`. Types:

    text    free text; `"primary": true` makes it the wide field at the top
    tags    multi-select over `options`; `"sides": true` asks it once per half
            of a pair and stores `{"Clip 1": [...], "Clip 2": [...]}`
    choice  exactly one of `options`
    flag    on or off
    scale   an integer `min`..`max`, one button per value

`scale` stays supported for a session that wants one; the default rubric,
`bench/rubrics/default.json`, has none. `bench/rubrics/scales.json` is the 1-5
form if you want it back.

An item counts as scored when every *required* question is answered and at
least one answer exists at all. A question is required when its type is `scale`
or `choice`, or when it carries `"required": true`; the default clip rubric
requires nothing, so without the second half of that rule an untouched clip
would count as scored and the progress counter would say nothing.
The pair's text field is deliberately **not** required: a blank one is a pair
the judge had nothing to say about, which is an answer, and
`bench/score_session.py` refuses on unanswered *required* questions only.

`--pair-rubric` overrides the pair questions; the default is `PAIR_RUBRIC`
below.

## Exporting

The page never writes to disk on its own. "Export scores" offers a download of
`scores_<session>.json` and also prints it into a textarea, because a page
opened as `file://` cannot always start a download. That file is what
`bench/score_session.py` joins against the key.

    python bench/blind_score_app.py --batch <share>/Video/blind/session1 \\
        --brief-file internal/session1_brief.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SUMMARY = (__doc__ or "").split("\n")[0]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DEFAULT_RUBRIC = HERE / "rubrics" / "default.json"

SLOTS = ["Clip 1", "Clip 2"]

# A pair is a presentation of two different samples, so the only things it can
# be asked are what differs and which half the judge would keep. No scale.
PAIR_RUBRIC = [
    {"id": "differs", "label": "What differs, and which way", "type": "text",
     "primary": True,
     "help": "The field that matters. Plain sentences: what you notice between "
             "the two halves, and which way it goes."},
    {"id": "tags", "label": "Quick tags", "type": "tags", "sides": True,
     "options": ["good", "not good", "off", "same", "can't tell"],
     "help": "Click any that apply, per half. Nothing here is required."},
    {"id": "verdict", "label": "Verdict", "type": "choice",
     "options": ["Clip 1 better", "Clip 2 better", "same", "can't tell"],
     "help": "Coarse on purpose. 'same' and \"can't tell\" are real answers and "
             "the tally counts them apart."},
]

_VALID_TYPES = {"scale", "flag", "text", "choice", "tags"}


def validate_rubric(doc, where: str) -> list[dict]:
    """Fail on a malformed rubric here rather than silently in the page."""
    if not isinstance(doc, list) or not doc:
        raise ValueError(f"{where}: rubric must be a non-empty list of questions")
    seen = set()
    for q in doc:
        if not isinstance(q, dict) or "id" not in q or "type" not in q:
            raise ValueError(f"{where}: every question needs an id and a type")
        if q["id"] in seen:
            raise ValueError(f"{where}: duplicate question id {q['id']!r}")
        seen.add(q["id"])
        if q["type"] not in _VALID_TYPES:
            raise ValueError(f"{where}: unknown question type {q['type']!r}")
        if q["type"] == "scale":
            lo, hi = q.get("min"), q.get("max")
            if not isinstance(lo, int) or not isinstance(hi, int) or hi <= lo:
                raise ValueError(f"{where}: scale {q['id']!r} needs integer min < max")
        if q["type"] in ("choice", "tags") and not q.get("options"):
            raise ValueError(f"{where}: {q['type']} {q['id']!r} needs options")
    return doc


def load_rubric(path: Path | str | None = None) -> list[dict]:
    """The singles rubric, or any rubric file, validated."""
    path = Path(path) if path else DEFAULT_RUBRIC
    return validate_rubric(json.loads(path.read_text()), str(path))


def _js_safe(payload: dict) -> str:
    """JSON that cannot end the script element it is embedded in."""
    return (json.dumps(payload, ensure_ascii=False)
            .replace("</", "<\\/").replace("<!--", "<\\!--"))


def write_score_app(batch: Path | str, rubric: list[dict] | None = None,
                    brief: str | None = None, out: Path | str | None = None,
                    pair_rubric: list[dict] | None = None) -> Path:
    """Render `score.html` into a blind batch directory and return its path.

    `batch` is the directory `blind_batch.py` wrote. `rubric` is the loaded
    singles rubric (default: `bench/rubrics/default.json`); `pair_rubric`
    defaults to `PAIR_RUBRIC`. `brief` is the session's brief text, shown
    collapsed at the top of the page.
    """
    batch = Path(batch)
    manifest_path = batch / "MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} not found; is that a blind batch?")
    manifest = json.loads(manifest_path.read_text())
    session = manifest.get("session")
    if not session:
        raise ValueError(f"{manifest_path} has no session name")
    clips = [c["clip"] for c in manifest.get("clips", [])]
    pairs = [p["pair"] for p in manifest.get("pairs", [])]
    if not clips and not pairs:
        raise ValueError(f"{manifest_path} lists no clips and no pairs")

    payload = {
        "session": session,
        "clips": clips,
        "pairs": pairs,
        "rubric": rubric if rubric is not None else load_rubric(),
        "pair_rubric": pair_rubric if pair_rubric is not None else PAIR_RUBRIC,
        "slots": SLOTS,
        "brief": brief or "",
    }
    html = (_TEMPLATE
            .replace("__SESSION_TITLE__", session.replace("&", "&amp;").replace("<", "&lt;"))
            .replace("__APPDATA__", _js_safe(payload)))
    out = Path(out) if out else batch / "score.html"
    out.write_text(html, encoding="utf-8")
    return out


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind scoring: __SESSION_TITLE__</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #f7f7f5; --fg: #1b1b1a; --muted: #5d5d58; --line: #d6d6d0;
  --card: #ffffff; --sunk: #efefea; --accent: #1f5f8b; --accent-fg: #ffffff;
  --warn: #8b2f1f; --ok: #1f6b3a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141513; --fg: #e9e9e4; --muted: #a0a099; --line: #34352f;
    --card: #1d1e1b; --sunk: #121310; --accent: #6fb3dd; --accent-fg: #10201a;
    --warn: #e08a78; --ok: #7fce9d;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
}
header { padding: 18px 24px 0; max-width: 1500px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--muted); margin: 0 0 12px; }
details { border: 1px solid var(--line); border-radius: 6px; background: var(--card);
          padding: 8px 12px; margin-bottom: 10px; }
details summary { cursor: pointer; font-weight: 600; }
details pre { white-space: pre-wrap; font: inherit; margin: 10px 0 2px; }
details ul { margin: 10px 0 2px; padding-left: 20px; }
kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 5px;
      background: var(--bg); font: 13px/1.4 ui-monospace, Menlo, Consolas, monospace; }
#bar { position: sticky; top: 0; z-index: 5; background: var(--bg);
       border-bottom: 1px solid var(--line); }
#bar .inner { max-width: 1500px; margin: 0 auto; padding: 10px 24px;
              display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
#bar .grow { flex: 1; }
#tabs { display: flex; gap: 6px; }
#progress { font-variant-numeric: tabular-nums; color: var(--muted); }
button { font: inherit; color: var(--fg); background: var(--card);
         border: 1px solid var(--line); border-radius: 6px; padding: 5px 12px; cursor: pointer; }
button:hover { border-color: var(--accent); }
button.on { background: var(--accent); color: var(--accent-fg); border-color: var(--accent); }
main { max-width: 1500px; margin: 0 auto; padding: 16px 24px 80px; }
section[hidden] { display: none; }
.item { border: 1px solid var(--line); border-left: 4px solid transparent;
        border-radius: 8px; background: var(--card); padding: 14px 16px; margin: 0 0 22px; }
.item.current { border-left-color: var(--accent); }
.item.done .name::after { content: " scored"; color: var(--ok); font-weight: 600; font-size: 13px; }
.head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
.name { font-weight: 600; }
.pos { color: var(--muted); font-size: 13px; }
video { display: block; width: auto; max-width: 100%; max-height: 70vh;
        background: #000; border-radius: 6px; }
/* A stack is twice as tall as a single, so it needs a tighter cap to leave the
   text field above the fold on a 1440-tall screen. */
.item[data-kind="pairs"] video { max-height: 62vh; }
.qs { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }
.q { flex: 0 1 auto; min-width: 220px; padding: 6px 8px; border-radius: 6px;
     border: 1px solid transparent; }
.q.active { border-color: var(--accent); }
.q.primary { flex: 1 1 100%; }
.q .label { font-weight: 600; margin-bottom: 2px; }
.q .help { color: var(--muted); font-size: 13px; margin-bottom: 6px; max-width: 62ch; }
.q .vals { display: flex; gap: 6px; flex-wrap: wrap; }
.side { display: flex; gap: 8px; align-items: center; margin-bottom: 6px; }
.side .who { min-width: 62px; color: var(--muted); font-size: 13px; }
.side.activeside .who { color: var(--accent); font-weight: 600; }
textarea { width: 100%; font: inherit; color: var(--fg); background: var(--sunk);
           border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; }
.q.primary textarea { min-height: 84px; }
#io { max-width: 1500px; margin: 0 auto; padding: 0 24px 60px; }
#io textarea { min-height: 140px; background: var(--sunk);
               font: 13px/1.45 ui-monospace, Menlo, Consolas, monospace; }
#io h2 { font-size: 16px; margin: 22px 0 6px; }
#msg { color: var(--warn); min-height: 20px; }
</style>
</head>
<body>
<header>
  <h1>Blind scoring: <span id="sessname"></span></h1>
  <p class="sub">Nothing on this page says which arm a clip came from. Start on the
  pairs: say what differs between the two halves and which way it goes. The singles
  are there for the audio, which the stacks do not carry, and for anything wrong with
  one clip on its own. Answers are held in this browser; export when you are done.</p>
  <details id="briefbox"><summary>The brief</summary><pre id="brief"></pre></details>
  <details><summary>Keyboard</summary>
    <ul>
      <li><kbd>j</kbd> / <kbd>k</kbd> (or down / up) next and previous item</li>
      <li><kbd>h</kbd> / <kbd>l</kbd> (or left / right) previous and next question</li>
      <li><kbd>n</kbd> jump into the item's main text field, <kbd>Esc</kbd> to leave it</li>
      <li><kbd>1</kbd>..<kbd>9</kbd> pick the n-th option of the active question
          (choice, tags, or scale)</li>
      <li><kbd>s</kbd> switch which half the tag keys apply to</li>
      <li><kbd>0</kbd> clear the active answer, <kbd>f</kbd> toggle the reject flag</li>
      <li><kbd>Space</kbd> play or pause the current video</li>
    </ul>
  </details>
</header>
<div id="bar">
  <div class="inner">
    <nav id="tabs"></nav>
    <span id="progress"></span>
    <span class="grow"></span>
    <button id="btn-export">Export scores</button>
    <button id="btn-clear">Clear this session</button>
  </div>
</div>
<main>
  <section id="sec-pairs" hidden></section>
  <section id="sec-clips" hidden></section>
</main>
<div id="io">
  <div id="msg"></div>
  <h2>Exported scores</h2>
  <p class="sub">A download is offered too, but some browsers block downloads from a
  local file, so the same JSON is here to copy.</p>
  <textarea id="out" spellcheck="false" placeholder="press Export scores"></textarea>
  <h2>Import</h2>
  <p class="sub">Paste a previously exported file to restore it into this browser.</p>
  <textarea id="in" spellcheck="false" placeholder="paste scores_&lt;session&gt;.json here"></textarea>
  <p><button id="btn-import">Import</button></p>
</div>
<script type="application/json" id="appdata">__APPDATA__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("appdata").textContent);
  var SLOTS = DATA.slots;
  var LS_KEY = "h3blind:" + DATA.session;
  var state = { clips: {}, pairs: {} };

  try {
    var raw = localStorage.getItem(LS_KEY);
    if (raw) {
      var prev = JSON.parse(raw);
      if (prev && typeof prev === "object") {
        state.clips = prev.clips || {};
        state.pairs = prev.pairs || {};
      }
    }
  } catch (e) { /* private mode, or corrupt: start empty */ }

  var items = [];
  DATA.pairs.forEach(function (n) { items.push({ kind: "pairs", name: n, qs: DATA.pair_rubric }); });
  DATA.clips.forEach(function (n) { items.push({ kind: "clips", name: n, qs: DATA.rubric }); });

  var cur = 0, curq = 0, side = 0, tab = DATA.pairs.length ? "pairs" : "clips";

  function answers(it) {
    var bucket = state[it.kind];
    if (!bucket[it.name]) { bucket[it.name] = {}; }
    return bucket[it.name];
  }
  function isRequired(q) { return q.required === true || q.type === "scale" || q.type === "choice"; }
  function empty(v) {
    return v === undefined || v === null || v === "" ||
           (Array.isArray(v) && !v.length) ||
           (v && typeof v === "object" && !Array.isArray(v) && !Object.keys(v).length);
  }
  function done(it) {
    // Every required question answered, AND something answered at all: the
    // default clip rubric requires nothing, and without the second half an
    // untouched clip counts as scored and the counter says nothing.
    var a = answers(it);
    var any = false;
    var ok = it.qs.every(function (q) {
      var got = !empty(a[q.id]) && a[q.id] !== false;
      if (got) { any = true; }
      return !isRequired(q) || got;
    });
    return ok && any;
  }
  function tagsOf(a, q, slot) {
    var v = a[q.id];
    if (!q.sides) { return Array.isArray(v) ? v : []; }
    return (v && !Array.isArray(v) && Array.isArray(v[slot])) ? v[slot] : [];
  }
  function toggleTag(a, q, slot, opt) {
    var have = tagsOf(a, q, slot).slice();
    var at = have.indexOf(opt);
    if (at >= 0) { have.splice(at, 1); } else { have.push(opt); }
    if (!q.sides) { a[q.id] = have; return; }
    var obj = (a[q.id] && !Array.isArray(a[q.id])) ? a[q.id] : {};
    if (have.length) { obj[slot] = have; } else { delete obj[slot]; }
    a[q.id] = obj;
  }

  function persist() {
    try { localStorage.setItem(LS_KEY, JSON.stringify(state)); }
    catch (e) { msg("could not save to this browser's storage: " + e.message); }
  }
  function save() { persist(); paint(); }
  function msg(t) { document.getElementById("msg").textContent = t || ""; }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text !== undefined) { n.textContent = text; }
    return n;
  }
  function node(it) {
    return document.querySelector('.item[data-kind="' + it.kind + '"][data-name="' + it.name + '"]');
  }

  function optButtons(it, q, qi, slot) {
    var vals = el("div", "vals");
    var opts;
    if (q.type === "scale") {
      opts = [];
      for (var v = q.min; v <= q.max; v++) { opts.push({ v: v, t: String(v) }); }
    } else if (q.type === "flag") {
      opts = [{ v: true, t: q.label || "flag" }];
    } else {
      opts = q.options.map(function (o) { return { v: o, t: o }; });
    }
    opts.forEach(function (o) {
      var b = el("button", "", o.t);
      b.dataset.val = JSON.stringify(o.v);
      if (slot) { b.dataset.slot = slot; }
      b.onclick = function () {
        cur = items.indexOf(it); curq = qi;
        if (slot) { side = SLOTS.indexOf(slot); }
        var a = answers(it);
        if (q.type === "tags") { toggleTag(a, q, slot || null, o.v); }
        else if (q.type === "flag") { a[q.id] = a[q.id] ? false : true; }
        else { a[q.id] = (a[q.id] === o.v) ? null : o.v; }
        save();
      };
      vals.appendChild(b);
    });
    return vals;
  }

  function buildQuestion(it, q, qi) {
    var wrap = el("div", "q" + (q.primary ? " primary" : "") + (q.type === "text" ? " text" : ""));
    wrap.dataset.qi = String(qi);
    wrap.appendChild(el("div", "label", q.label || q.id));
    if (q.help) { wrap.appendChild(el("div", "help", q.help)); }
    var a = answers(it);
    if (q.type === "text") {
      var ta = el("textarea");
      ta.rows = q.primary ? 4 : 2;
      ta.value = a[q.id] || "";
      ta.oninput = function () { answers(it)[q.id] = ta.value; persist(); };
      ta.onchange = function () { answers(it)[q.id] = ta.value; save(); };
      ta.onfocus = function () { cur = items.indexOf(it); curq = qi; paint(); };
      wrap.appendChild(ta);
    } else if (q.type === "tags" && q.sides) {
      SLOTS.forEach(function (slot) {
        var row = el("div", "side");
        row.dataset.slot = slot;
        row.appendChild(el("span", "who", slot));
        row.appendChild(optButtons(it, q, qi, slot));
        wrap.appendChild(row);
      });
    } else {
      wrap.appendChild(optButtons(it, q, qi, null));
    }
    return wrap;
  }

  function buildItem(it, idx, total) {
    var art = el("article", "item");
    art.dataset.kind = it.kind;
    art.dataset.name = it.name;
    var head = el("div", "head");
    head.appendChild(el("span", "name", it.name));
    head.appendChild(el("span", "pos", (idx + 1) + " of " + total));
    art.appendChild(head);
    var v = document.createElement("video");
    v.controls = true;
    v.preload = "metadata";
    v.src = it.name;
    v.onplay = function () { cur = items.indexOf(it); paint(); };
    art.appendChild(v);
    var qs = el("div", "qs");
    it.qs.forEach(function (q, qi) { qs.appendChild(buildQuestion(it, q, qi)); });
    art.appendChild(qs);
    return art;
  }

  function build() {
    document.getElementById("sessname").textContent = DATA.session;
    if (DATA.brief) { document.getElementById("brief").textContent = DATA.brief; }
    else { document.getElementById("briefbox").hidden = true; }

    var secs = { clips: document.getElementById("sec-clips"), pairs: document.getElementById("sec-pairs") };
    var counts = { clips: DATA.clips.length, pairs: DATA.pairs.length };
    items.forEach(function (it) {
      var within = (it.kind === "clips") ? DATA.clips.indexOf(it.name) : DATA.pairs.indexOf(it.name);
      secs[it.kind].appendChild(buildItem(it, within, counts[it.kind]));
    });

    var tabs = document.getElementById("tabs");
    [["pairs", "Pairs (" + counts.pairs + ")"], ["clips", "Clips (" + counts.clips + ")"]].forEach(function (t) {
      if (!counts[t[0]]) { return; }
      var b = el("button", "tab", t[1]);
      b.dataset.tab = t[0];
      b.onclick = function () { showTab(t[0]); };
      tabs.appendChild(b);
    });
    showTab(tab);
  }

  function showTab(which) {
    tab = which;
    document.getElementById("sec-clips").hidden = (which !== "clips");
    document.getElementById("sec-pairs").hidden = (which !== "pairs");
    if (!items[cur] || items[cur].kind !== which) {
      var first = -1;
      items.forEach(function (it, i) { if (first < 0 && it.kind === which) { first = i; } });
      if (first >= 0) { cur = first; curq = 0; }
    }
    paint();
  }

  function paint() {
    var nDone = 0;
    items.forEach(function (it, i) {
      var art = node(it);
      var isDone = done(it);
      if (isDone) { nDone++; }
      art.classList.toggle("done", isDone);
      art.classList.toggle("current", i === cur);
      var a = answers(it);
      var qnodes = art.querySelectorAll(".q");
      it.qs.forEach(function (q, qi) {
        var qn = qnodes[qi];
        var isActive = (i === cur && qi === curq);
        qn.classList.toggle("active", isActive);
        qn.querySelectorAll(".side").forEach(function (row) {
          row.classList.toggle("activeside", isActive && row.dataset.slot === SLOTS[side]);
        });
        qn.querySelectorAll(".vals button").forEach(function (b) {
          var val = JSON.parse(b.dataset.val);
          var on;
          if (q.type === "tags") { on = tagsOf(a, q, b.dataset.slot || null).indexOf(val) >= 0; }
          else if (q.type === "flag") { on = a[q.id] === true; }
          else { on = a[q.id] === val; }
          b.classList.toggle("on", on);
        });
        var ta = qn.querySelector("textarea");
        if (ta && document.activeElement !== ta) { ta.value = a[q.id] || ""; }
      });
    });
    var inTab = items.filter(function (it) { return it.kind === tab; });
    document.getElementById("progress").textContent =
      inTab.filter(done).length + " of " + inTab.length + " scored in this tab, " +
      nDone + " of " + items.length + " overall";
    document.querySelectorAll("nav button.tab").forEach(function (b) {
      b.classList.toggle("on", b.dataset.tab === tab);
    });
  }

  function move(d) {
    var inTab = [];
    items.forEach(function (it, i) { if (it.kind === tab) { inTab.push(i); } });
    if (!inTab.length) { return; }
    var at = inTab.indexOf(cur);
    at = (at < 0) ? 0 : Math.min(inTab.length - 1, Math.max(0, at + d));
    cur = inTab[at];
    curq = 0;
    paint();
    node(items[cur]).scrollIntoView({ block: "start", behavior: "smooth" });
  }

  function pick(n) {
    var it = items[cur], q = it.qs[curq], a = answers(it);
    if (!q) { return; }
    if (q.type === "scale") {
      var v = q.min + n - 1;
      if (v >= q.min && v <= q.max) { a[q.id] = v; save(); }
    } else if (q.type === "choice") {
      if (n <= q.options.length) { a[q.id] = q.options[n - 1]; save(); }
    } else if (q.type === "tags") {
      if (n <= q.options.length) { toggleTag(a, q, q.sides ? SLOTS[side] : null, q.options[n - 1]); save(); }
    } else if (q.type === "flag") {
      a[q.id] = (n === 1);
      save();
    }
  }

  document.addEventListener("keydown", function (e) {
    var t = e.target;
    if (t && (t.tagName === "TEXTAREA" || t.tagName === "INPUT")) {
      if (e.key === "Escape") { t.blur(); paint(); }
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) { return; }
    var it = items[cur];
    if (!it) { return; }
    if (e.key === "j" || e.key === "ArrowDown") { move(1); }
    else if (e.key === "k" || e.key === "ArrowUp") { move(-1); }
    else if (e.key === "l" || e.key === "ArrowRight") { curq = Math.min(it.qs.length - 1, curq + 1); paint(); }
    else if (e.key === "h" || e.key === "ArrowLeft") { curq = Math.max(0, curq - 1); paint(); }
    else if (e.key >= "1" && e.key <= "9") { pick(Number(e.key)); }
    else if (e.key === "0") { delete answers(it)[it.qs[curq].id]; save(); }
    else if (e.key === "s") { side = (side + 1) % SLOTS.length; paint(); }
    else if (e.key === "f") {
      var flag = null;
      it.qs.forEach(function (q) { if (!flag && q.type === "flag") { flag = q; } });
      if (flag) { var a = answers(it); a[flag.id] = !a[flag.id]; save(); }
    } else if (e.key === "n") {
      var ta = node(it).querySelector("textarea");
      if (ta) { ta.focus(); }
    } else if (e.key === " ") {
      var v = node(it).querySelector("video");
      if (v) { if (v.paused) { v.play(); } else { v.pause(); } }
    } else { return; }
    e.preventDefault();
  });

  function exportDoc() {
    var out = { clips: {}, pairs: {} };
    items.forEach(function (it) {
      var a = answers(it), kept = {};
      it.qs.forEach(function (q) {
        var v = a[q.id];
        if (empty(v) || v === false) { return; }
        kept[q.id] = v;
      });
      if (Object.keys(kept).length) { out[it.kind][it.name] = kept; }
    });
    return {
      session: DATA.session,
      scored_at: new Date().toISOString().replace(/\.\d+Z$/, "Z"),
      rubric: DATA.rubric,
      pair_rubric: DATA.pair_rubric,
      slots: SLOTS,
      clips: out.clips,
      pairs: out.pairs
    };
  }

  document.getElementById("btn-export").onclick = function () {
    var txt = JSON.stringify(exportDoc(), null, 1);
    document.getElementById("out").value = txt;
    var name = "scores_" + DATA.session + ".json";
    try {
      var blob = new Blob([txt], { type: "application/json" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url; a.download = name;
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 4000);
      msg("offered " + name + " as a download; the same JSON is below if the browser refused it.");
    } catch (err) {
      msg("this browser refused the download (" + err.message + "); copy the JSON below into " + name + ".");
    }
  };

  document.getElementById("btn-import").onclick = function () {
    var txt = document.getElementById("in").value.trim();
    if (!txt) { msg("nothing pasted."); return; }
    var doc;
    try { doc = JSON.parse(txt); } catch (err) { msg("not JSON: " + err.message); return; }
    if (doc.session && doc.session !== DATA.session) {
      if (!confirm("That file is for session " + doc.session + ", this batch is " +
                   DATA.session + ". Import anyway?")) { return; }
    }
    var known = { clips: DATA.clips, pairs: DATA.pairs };
    var unknown = [];
    ["clips", "pairs"].forEach(function (k) {
      Object.keys(doc[k] || {}).forEach(function (n) {
        if (known[k].indexOf(n) < 0) { unknown.push(n); return; }
        state[k][n] = doc[k][n];
      });
    });
    save();
    msg("imported" + (unknown.length ? "; ignored items not in this batch: " + unknown.join(", ") : "."));
  };

  document.getElementById("btn-clear").onclick = function () {
    if (!confirm("Clear every answer for " + DATA.session + " in this browser?")) { return; }
    state = { clips: {}, pairs: {} };
    save();
    msg("cleared.");
  };

  build();
  paint();
})();
</script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=_SUMMARY)
    ap.add_argument("--batch", required=True,
                    help="the blind batch directory blind_batch.py wrote (holds MANIFEST.json)")
    ap.add_argument("--rubric", default=None,
                    help=f"singles rubric JSON; default {DEFAULT_RUBRIC.relative_to(REPO)}")
    ap.add_argument("--pair-rubric", default=None,
                    help="pair rubric JSON; default is this file's PAIR_RUBRIC")
    ap.add_argument("--brief-file", default=None,
                    help="text file holding the brief every clip was asked to render")
    ap.add_argument("--out", default=None, help="default <batch>/score.html")
    args = ap.parse_args()

    try:
        brief = Path(args.brief_file).read_text() if args.brief_file else None
        out = write_score_app(args.batch, load_rubric(args.rubric), brief, args.out,
                              load_rubric(args.pair_rubric) if args.pair_rubric else None)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"refuse: {exc}")
    print(f"wrote {out.name} into {out.parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
