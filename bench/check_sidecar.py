#!/usr/bin/env python3
"""`bench/sidecar.py` on made files: each case is a way a video's sidecar could say less than is known, or be trusted
for a video it was not written for.

1. **Write and read back**: the video's sha256, the command as one quoted line, the commit and the files not
   committed, every `H3_` environment key it ran with, the tool's sha256, each input's sha256 and size, and a missing
   input said as missing.
2. **Refusals**: a video rewritten after its sidecar, a video with no sidecar, and a JSON of another kind beside the
   video (never written over, never read as a sidecar).
3. **A load's windows** (read through `bench/what_h3_got.py`): one entry a window with its source frames, the text
   whole with its sha256, the graph whole with its sha256 and the queue's prompt_id, the context and its noise, and
   the seed; a window running past the load's last frame says so.
4. **The bytes a reference was read from**: the queue's digest held to the file now; a remade file whose original was
   set aside under `<folder>_SUPERSEDED_*` points at that copy; no digest kept is said as not known.
5. **No window recorded, no record, no graph**: each is said in the entry, never left out; a render with no job
   record takes its graph from the PNG VHS saved beside it and says so.
6. **A delivery table's pieces**: tied to their job folders through the ledger the table names, and the list of what
   is laid named as an input.
7. **Beside a tool's own record**: the sidecar is `<video file name>.json`; a `<stem>.json` record beside the video is
   neither read as the sidecar nor written over.
8. **Every input the load was handed**, whole, each held to the queue's digest, with the reader's failures (digests on).
9. **The text when the reader refuses**: no `input_dir` read with the folder's default, said; a load the reader refuses
   keeps its graph's text, said.
10. **Through the assembler** (`write_sidecar` with a table JSON and its ledger): the laid rows only, each tied to its
   .txt row; a .txt the JSON does not hold refused as the pieces' source; one bad piece a gap.
11. **What the ledger says**: `making` and `disagrees` carried; the job graph held to the PNG's; a named ledger not found
   said as that.
12. **Paths resolved**, and a tool with no file says why it has no sha256.
13. **A check appended**: a check of a built video goes under `checks`; the build's command, inputs, extra and pieces stay
   whole through two checks; a check against another table is said; a video with no sidecar gets one saying so.
14. **The old name, and a piece that stops**: an old-name sidecar is named and not read; a piece whose reading stops
   (SystemExit) is its own gap.

Needs ffmpeg and Pillow. No model, no card, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_sidecar.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import case, finish  # noqa: E402

import sidecar as sc  # noqa: E402
import what_h3_got as w  # noqa: E402

TEXT = "the text, whole\n<Subject 1> sings, her mouth opening wide on the sung phrases.\n"


def clip(path: Path, frames: int = 12, pattern: str = "testsrc2") -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"{pattern}=size=64x48:rate=24", "-frames:v", str(frames),
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def vhs(video: str, at: float = 0.0, cap: int = 12) -> dict:
    return {"class_type": w.VIDEO_LOADER, "inputs": {"video": video, "custom_width": 64, "custom_height": 48, "frame_load_cap": cap,
                                                     "start_time": at, "force_rate": 0.0, "format": "AnimateDiff"}}


#: A face-points drawing's record in the shape the real ones have (Video/fun/references/*/<subject>_head_reference_*.json,
#: read 2026-10-16): the mouth's `drawn` and its gate's `frames_drawn`, `eyes_shut` with its spans, the points file under `inputs`.
DRAWN = {"line_width_px": 1.5, "a_dot_at_every_point": False, "parts_left_out": [], "frames_left_bare": [101, 102], "iris": "contrast",
         "eye_rings": {"drawn": "contrast", "frames_a_ring_is_left_out_on": {"l_iris": [103], "r_iris": []}},
         "mouth": {"form": "auto", "drawn": True, "gate": {"frames_drawn": 10, "frames": 12}, "frames_with_no_mouth": 2},
         "eyes_shut": {"form": "lids", "frames": [[104, 106]], "on_whose_word": [{"first": 104, "last": 106, "why": "a person's word"}]}}


def job(root: Path, name: str = "load_a", said: list[str] | None = None, digests: bool = True, input_dir: bool = True) -> Path:
    """A job folder with its graph and queue record: a source from frame 100, a reference video, a still."""
    files, jf = root / "files", root / f"{name}_job"
    for d in (files / "refs" / "ref_100", files / "input", jf):
        d.mkdir(parents=True, exist_ok=True)
    src, ref = clip(files / "source.mp4", 200), clip(files / "refs" / "ref_100" / "ref.mp4")
    (files / "refs" / "ref_100" / "ref.json").write_text(json.dumps({"what": "the body model cut to the head", "box": [4, 4, 60, 44],
                                                                      "scale": [1.1, 1.1], "room_set_by": "HEAD_ROOM", "drawn": DRAWN,
                                                                      "inputs": {"points": str(files / "points.json"), "their_first_row_is_source_frame": 100}}))
    from PIL import Image
    Image.new("RGB", (20, 30), (200, 10, 10)).save(files / "input" / "still.png")
    g = {"28": vhs(str(src), 100 / 24, 12), "200": vhs(str(ref)),
         "15": {"class_type": "LoadImage", "inputs": {"image": "still.png"}},
         "50": {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], "size_policy": "max"}},
         "104": {"class_type": w.SOURCE, "inputs": {"frames": ["28", 0], "motion_video": ["200", 0], "replace": "the wired parts"}},
         "74": {"class_type": w.SONG, "inputs": {"source": ["104", 0], "references": ["50", 0], "prompt": TEXT, "seed": 730451892,
                                                  "window_frames": 8, "context_frames": 4, "context_noise": 0.8}}}
    (jf / f"{name}.render.json").write_text(json.dumps(g))
    rec = {"queued_at": "2026-10-16T10:00:00+00:00", "status": "success", "prompt_id": "pid-1", "commit": "abc1234",
           **({"input_dir": str(files / "input")} if input_dir else {}), "files": [f"Video/v/{name}/{name}_00001.mp4"],
           "said": {"75": said if said is not None else ["[1] window from 0.000s: 8 frames", "[2] window from 0.167s: 8 frames"]}}
    if digests:
        rec["digests"] = {str(ref): sc.sha256_of(ref), str(files / "input" / "still.png"): sc.sha256_of(files / "input" / "still.png"),
                          str(src): sc.sha256_of(src)}
    (jf / f"{name}.render.queued.json").write_text(json.dumps(rec))
    return jf


def write_and_read() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "clip.mp4")
        os.environ["H3_INPUT_DIR"], os.environ["H3_SIDECAR_TEST"] = "/in", "x"
        side = sc.write(v, ["python", "bench/assemble_delivery.py", "--note", "two words"], {"the source": v, "gone": Path(tmp) / "nope.mp4"},
                        {"k": 1}, made_by="mrtest")
        del os.environ["H3_SIDECAR_TEST"]
        assert side == Path(tmp) / "clip.mp4.json" == sc.sidecar_path(v), side
        s = sc.read(v)
        assert s["schema"] == sc.SCHEMA and s["video_sha256"] == hashlib.sha256(v.read_bytes()).hexdigest() and s["made_by"] == "mrtest", s
        assert s["command"] == "python bench/assemble_delivery.py --note 'two words'", s["command"]
        assert s["git"]["commit"] and isinstance(s["git"]["dirty"], list), s["git"]
        assert s["environment"]["H3_INPUT_DIR"] == "/in" and s["environment"]["H3_SIDECAR_TEST"] == "x" \
            and all(k.startswith("H3_") for k in s["environment"]), s["environment"]
        assert s["tool"]["path"] == str(sc.REPO / "bench" / "assemble_delivery.py") and s["tool"]["sha256"], s["tool"]
        rows = {x["role"]: x for x in s["inputs"]}
        assert rows["the source"]["sha256"] == s["video_sha256"] and rows["the source"]["size"] == v.stat().st_size, rows
        assert rows["gone"]["sha256"] is None and "not a file" in rows["gone"]["missing"], rows["gone"]
    return "sha256, the command as one quoted line, the H3_ keys, the commit and dirty files, the tool, each input, a missing one said"


def refusals() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "clip.mp4")
        try:
            sc.read(v)
        except SystemExit as e:
            assert "has no sidecar" in str(e), e
        else:
            raise AssertionError("a video with no sidecar was read")
        sc.write(v, ["x"], {})
        clip(v, 6)                                                    # rewritten after its sidecar
        try:
            sc.read(v)
        except SystemExit as e:
            assert "changed after its sidecar" in str(e), e
        else:
            raise AssertionError("a rewritten video's sidecar was trusted")
        other = clip(Path(tmp) / "ref.mp4")
        (Path(tmp) / "ref.mp4.json").write_text(json.dumps({"what": "not a sidecar"}))
        for call in (lambda: sc.write(other, ["x"], {}), lambda: sc.read(other)):
            try:
                call()
            except SystemExit as e:
                assert "not a sidecar" in str(e) or "schema" in str(e), e
            else:
                raise AssertionError("a JSON of another kind was written over or read as a sidecar")
        assert json.loads((Path(tmp) / "ref.mp4.json").read_text()) == {"what": "not a sidecar"}
    return "no sidecar, a rewritten video, and another kind of JSON at the sidecar's name are each refused"


def beside_a_tools_record() -> str:
    """The sidecar is `<video name>.json` (the lead, mrtony, 2026-10-16): a tool's own `<stem>.json` record beside the video
    (a reference's, a zoomed picture's) is neither read as the sidecar nor written over."""
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "ref.mp4")
        record = {"what": "the body model cut to the head", "box": [1, 2, 3, 4]}
        (Path(tmp) / "ref.json").write_text(json.dumps(record))
        try:
            sc.read(v)
        except SystemExit as e:
            assert "has no sidecar" in str(e), e
        else:
            raise AssertionError("a tool's record was read as the video's sidecar")
        # THE CASE THAT GOES RED under the old name (`<stem>.json`): the write refuses, or writes over the record
        side = sc.write(v, ["x"], {})
        assert side.name == "ref.mp4.json" and json.loads((Path(tmp) / "ref.json").read_text()) == record, side
        assert sc.read(v)["video_sha256"] == sc.sha256_of(v)
    return "a tool's own <stem>.json beside the video is neither read as the sidecar nor written over"


def a_loads_windows() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        jf = job(Path(tmp))
        r = sc.load_record(jf, "load_a", output=Path(tmp))
        ws = r["windows"]
        assert [(x["index"], x["first"], x["last"], x["last_in_the_load"]) for x in ws] == [(1, 100, 107, 107), (2, 104, 111, 111)], ws
        raw = (jf / "load_a.render.json").read_text()
        for x in ws:
            assert x["text"] == TEXT and x["text_sha256"] == hashlib.sha256(TEXT.encode()).hexdigest(), x["text"]
            assert x["graph"] == json.loads(raw) and x["graph_sha256"] == hashlib.sha256(raw.encode()).hexdigest() and x["prompt_id"] == "pid-1", x
            assert (x["context_frames"], x["context_noise"], x["window_frames"]) == (4, 0.8, 8), x
        assert r["seeds"] == {"74": {"seed": 730451892}} and r["commit"]["commit"] == "abc1234", (r["seeds"], r["commit"])
        (ref,) = r["references"]
        assert ref["state"] == "same bytes as queued" and ref["chain"][0]["box"] == [4, 4, 60, 44] and ref["chain"][0]["room_set_by"] == "HEAD_ROOM", ref
        # THE CASE THAT GOES RED when the drawing is read by field names the records do not use (the mouth's `drawn`, `eyes_shut`)
        assert ref["chain"][0]["drawn"] == DRAWN, ref["chain"][0]["drawn"]
        assert ref["chain"][0]["points"] == {"file": str(Path(tmp) / "files" / "points.json"), "first_row_is_source_frame": 100}, ref["chain"][0]
        assert ref["loader"]["frame_load_cap"] == 12 and "fit" in ref and isinstance(ref["encoder_frames"], list), ref
        (st,) = r["stills"]
        assert st["state"] == "same bytes as queued" and st["append"]["size_policy"] == "max", st
        past = job(Path(tmp) / "p", "load_p", said=["[1] window from 0.000s: 16 frames"])
        (x,) = sc.load_record(past, "load_p", output=Path(tmp))["windows"]
        assert x["last"] == 115 and x["last_in_the_load"] == 111 and "4 frame(s) past" in x["past_the_load"], x
    return "two windows with their frames, the text and the graph whole, prompt_id, context and noise, the seed; a window past the load said"


def the_bytes_read() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        jf = job(Path(tmp))
        ref = Path(tmp) / "files" / "refs" / "ref_100" / "ref.mp4"
        aside = Path(tmp) / "files" / "refs" / "ref_100_SUPERSEDED_x"
        aside.mkdir()
        os.replace(ref, aside / "ref.mp4")
        clip(ref, 6)                                                  # remade under its name after the render read it
        (r,) = sc.load_record(jf, "load_a", output=Path(tmp))["references"]
        assert r["state"].startswith("CHANGED since queued") and r["read_at"] == str(aside / "ref.mp4"), r
        bare = job(Path(tmp) / "b", "load_b", digests=False)
        (r,) = sc.load_record(bare, "load_b", output=Path(tmp))["references"]
        assert r["state"].startswith("no digest kept") and r["read_at"] is None, r
    return "a remade reference points at its set-aside original by the queue's digest; no digest is said as not known"


def gaps_said() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        jf = job(Path(tmp), said=[])
        (x,) = sc.load_record(jf, "load_a", output=Path(tmp))["windows"]
        assert x["index"] is None and "no window is recorded" in x["why"] and x["text"] == TEXT and x["graph"], x
        share = Path(tmp) / "share"
        piece = share / "Video" / "v" / "p" / "p_00001.mp4"
        piece.parent.mkdir(parents=True)
        clip(piece)
        p = sc.piece_of({"file": "Video/v/p/p_00001.mp4", "name": "p"}, {"renders": {}}, share)
        assert "no queue record" in p["windows"][0]["why"] and "no graph is recoverable" in p["load"]["graph"]["why"], p
        assert "no p_00001_zoomed.json" in p["zoomed"]["why"], p["zoomed"]
        from PIL import Image, PngImagePlugin
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", json.dumps({"74": {"class_type": "x", "inputs": {"seed": 1}}}))
        Image.new("RGB", (4, 4)).save(piece.with_suffix(".png"), pnginfo=info)
        p = sc.piece_of({"file": "Video/v/p/p_00001.mp4", "name": "p"}, {"renders": {}}, share)
        g = p["load"]["graph"]
        assert g["prompt"]["74"]["inputs"]["seed"] == 1 and "PNG VHS saved" in g["from_is"] and "queued by API" in g["workflow_why"], g
        assert p["windows"][0]["graph"] == g["prompt"], p["windows"][0]
        # a job folder whose graph file is gone: the graph from the PNG beside the render, said so
        gone = job(Path(tmp) / "g", "load_g")
        (gone / "load_g.render.json").unlink()
        r = sc.load_record(gone, "load_g", render=piece, output=Path(tmp))
        assert "PNG VHS saved" in r["graph"]["from_is"] and r["windows"][0]["graph"] == g["prompt"] and "does not read it" in r["why"], r
    return "no window, no record and no graph each said; a render with no record takes its graph from VHS's PNG and says so"


def a_tables_pieces() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        jf = job(root)
        share = root / "share"
        piece = share / "Video" / "v" / "load_a" / "load_a_00001.mp4"
        piece.parent.mkdir(parents=True)
        clip(piece)
        ledger = root / "ledger.json"
        ledger.write_text(json.dumps({"share": str(share), "renders": {"Video/v/load_a/load_a_00001.mp4": {
            "records": [{"queued_file": str(jf / "load_a.render.queued.json"), "load": {"name": "load_a"}}]}}}))
        laid = root / "laid.jsonl"
        laid.write_text("")
        table = root / "t.json"
        table.write_text(json.dumps({"made": f"Made now from the ledger `{ledger}` (x); and ONLY the list of what is laid, `{laid}` (sha256 0)",
                                     "rows": [{"first": 100, "last": 107, "file": "Video/v/load_a/load_a_00001.mp4", "name": "load_a", "subject": "a"}]}))
        pieces, named = sc.pieces_of_table(table)
        assert named == {"the ledger the table was made from": str(ledger), "the list of what is laid": str(laid)}, named
        # what it returns goes through write whole, as extra and as inputs, with a Path a caller adds beside
        side = sc.write(piece, ["x"], {**named, "a path": piece}, {**named, "a path": piece}, pieces=pieces)
        back = sc.read(piece)
        assert back["extra"]["the list of what is laid"] == str(laid) and back["extra"]["a path"] == str(piece) and side.is_file(), back["extra"]
        (p,) = pieces
        assert p["sha256"] == sc.sha256_of(piece) and [x["index"] for x in p["windows"]] == [1, 2] and p["windows"][0]["text"] == TEXT, p
    return "a laid piece tied to its job folder by the ledger the table names; the list of what is laid named"


def every_input_whole() -> str:
    """Finding 1 of the blind read: every input the load was handed, each with its bytes against the queue's digest, and
    what_h3_got's own failures with digests on: a source rewritten after the queue is said, never left out."""
    with tempfile.TemporaryDirectory() as tmp:
        jf = job(Path(tmp))
        src = Path(tmp) / "files" / "source.mp4"
        clip(src, 200, "testsrc")                                    # rewritten under its name after the queue
        r = sc.load_record(jf, "load_a", output=Path(tmp))
        rows = {x["role"]: x for x in r.get("inputs") or []}
        assert {"the source", "<Video 1>", "<Picture 1>"} <= set(rows), sorted(rows)
        assert rows["the source"]["state"].startswith("CHANGED since queued") and rows["the source"]["sha256_now"] == sc.sha256_of(src), rows["the source"]
        assert any("sha256 is not the one recorded at queue time" in f and "the source" in f for f in r["read_failures"]), r["read_failures"]
        assert rows["<Video 1>"]["state"] == "same bytes as queued", rows["<Video 1>"]
    return "every input the load was handed, each against the queue's digest; a source rewritten after the queue in the reader's failures"


def the_text_when_the_reader_refuses() -> str:
    """Finding 2: a queue record with no input_dir is read with the input folder's default, said; and where the reader
    refuses the load, the text is the graph's song node's, said so."""
    with tempfile.TemporaryDirectory() as tmp:
        was = os.environ.pop("H3_INPUT_DIR", None)
        try:
            jf = job(Path(tmp), input_dir=False)
            r = sc.load_record(jf, "load_a", output=Path(tmp))
            assert r["windows"][0]["text"] == TEXT and "why" not in r, (r.get("why"), r["windows"][0].get("text_why"))
            assert r["input_dir"]["from"].startswith("not recorded at the queue: the stock folder"), r["input_dir"]
            # THE CASE THAT GOES RED when a still not found in a folder nobody recorded is written as MISSING
            assert r["stills"][0]["state"].startswith("folder not known"), r["stills"]
            assert next(x for x in r["inputs"] if x["role"] == "<Picture 1>")["state"].startswith("folder not known"), r["inputs"]
            os.environ["H3_INPUT_DIR"] = str(Path(tmp) / "files" / "input")
            r = sc.load_record(jf, "load_a", output=Path(tmp))
            assert r["input_dir"] == {"path": str(Path(tmp) / "files" / "input"), "from": "not recorded at the queue: H3_INPUT_DIR at assembly"}, r["input_dir"]
            assert r["stills"][0]["state"] == "same bytes as queued", r["stills"]
        finally:
            os.environ.pop("H3_INPUT_DIR", None)
            if was is not None:
                os.environ["H3_INPUT_DIR"] = was
        # a graph the reader refuses (it appends a reference video): the text from the graph it holds
        odd = job(Path(tmp) / "o", "load_o")
        g = json.loads((odd / "load_o.render.json").read_text())
        g["60"] = {"class_type": "MiniMaxH3AppendRefVideo", "inputs": {"video": ["200", 0]}}
        (odd / "load_o.render.json").write_text(json.dumps(g))
        r = sc.load_record(odd, "load_o", output=Path(tmp))
        (x,) = r["windows"]
        assert "does not read it" in r["why"] and x["text"] == TEXT and x["text_sha256"] == hashlib.sha256(TEXT.encode()).hexdigest(), (r.get("why"), x)
        assert x["text_from"].startswith("the graph's song node"), x
    return "no input_dir read with the folder's default, said; a load the reader refuses keeps the text its graph holds, said"


def _laid(root: Path, rows: list[tuple[int, int, str]], ledger: dict | None = None, ledger_named: bool = True) -> tuple[Path, Path, Path]:
    """A share with a piece per row, a job folder for load_a, a ledger of them, and a table: its .txt and its JSON."""
    share = root / "share"
    jf = job(root)
    recs = {}
    for _a, _b, name in rows:
        piece = share / "Video" / "v" / name / f"{name}_00001.mp4"
        piece.parent.mkdir(parents=True, exist_ok=True)
        clip(piece)
        recs[f"Video/v/{name}/{name}_00001.mp4"] = {"making": "this run's", "records": [{"queued_file": str(jf / "load_a.render.queued.json"),
                                                                                        "load": {"name": "load_a"}, "making": "this run's", "disagrees": []}]}
    ledger_file = root / "ledger.json"
    ledger_file.write_text(json.dumps(ledger if ledger is not None else {"share": str(share), "renders": recs}))
    txt = root / "t.txt"
    txt.write_text("".join(f"{a}-{b} {share / 'Video' / 'v' / n / f'{n}_00001.mp4'} {a}\n" for a, b, n in rows))
    named = f"Made now from the ledger `{ledger_file if ledger_named else root / 'gone.json'}` (x)"
    (root / "t.json").write_text(json.dumps({"made": named, "rows": [{"first": a, "last": b, "file": f"Video/v/{n}/{n}_00001.mp4", "name": n, "subject": "a"}
                                                                      for a, b, n in rows]}))
    return share, txt, root / "t.json"


def through_the_assembler() -> str:
    """Findings 3, 4 and 9 and the test gap, through `assemble_delivery.write_sidecar` with a table JSON and its ledger:
    the pieces are the rows the clip lays (a span leaves the others out), each tied to a row of the .txt the assembler
    read; a .txt the JSON does not match is REFUSED as the pieces' source and said; one bad piece is a gap, the rest written."""
    import argparse
    import assemble_delivery as ad
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        share, txt, _ = _laid(root, [(100, 107, "load_a"), (300, 307, "load_b")])
        out = clip(root / "out.mp4")
        args = argparse.Namespace(table=str(txt), source=str(out), out=str(out), note=None)
        rows = ad.read_table(str(txt))
        segs = ad.segments(rows, (100, 110))
        side = json.loads(Path(ad.write_sidecar(args, rows, {"span": [100, 110], "verdict": "passes"}, segs)).read_text())
        # THE CASES THAT GO RED: a piece outside the span listed as the clip's; the pieces read from a JSON with no tie to the .txt
        assert [p["name"] for p in side["pieces"]] == ["load_a"], [p.get("name") for p in side["pieces"]]
        assert side["pieces"][0]["windows"][0]["text"] == TEXT and side["pieces"][0]["txt_row"] == [100, 107, rows[0]["piece"]], side["pieces"][0]
        assert [k for k in (x["role"] for x in side["inputs"]) if k.startswith("piece ")] == ["piece 1: 100-107"], side["inputs"]
        assert side["extra"]["table_rows_not_laid"] == [[300, 307, rows[1]["piece"]]], side["extra"]
        # a row of the .txt the JSON does not hold: the pieces are not taken from the JSON, and why is said
        txt.write_text(txt.read_text() + f"108-110 {share / 'Video' / 'v' / 'patch' / 'patch_00001.mp4'} 108\n")
        rows = ad.read_table(str(txt))
        side = json.loads(Path(ad.write_sidecar(args, rows, {"span": [100, 110], "verdict": "passes"}, ad.segments(rows, (100, 110)))).read_text())
        assert side.get("pieces") is None and side["extra"]["pieces_why"].startswith("REFUSED") and "108-110" in side["extra"]["pieces_why"], side["extra"]
        # one bad piece: its queue record does not read; it is a gap, and the sidecar is written with the others
        root2 = root / "two"
        root2.mkdir()
        share2, txt2, _ = _laid(root2, [(100, 107, "load_a"), (108, 115, "load_c")])
        bad = root2 / "load_c_job" / "load_c.render.queued.json"
        bad.parent.mkdir()
        bad.write_text("{not json")
        led = json.loads((root2 / "ledger.json").read_text())
        led["renders"]["Video/v/load_c/load_c_00001.mp4"]["records"][0].update(queued_file=str(bad), load={"name": "load_c"})
        (root2 / "ledger.json").write_text(json.dumps(led))
        out2 = clip(root2 / "out.mp4")
        rows2 = ad.read_table(str(txt2))
        args2 = argparse.Namespace(table=str(txt2), source=str(out2), out=str(out2), note=None)
        side = json.loads(Path(ad.write_sidecar(args2, rows2, {"span": [100, 115]}, ad.segments(rows2, (100, 115)))).read_text())
        assert [p.get("name") for p in side["pieces"]] == ["load_a", "load_c"] and side["pieces"][0]["windows"][0]["text"] == TEXT, side["pieces"]
        assert "why" in side["pieces"][1] and "load_c" in side["pieces"][1]["why"], side["pieces"][1]
    return "the clip's laid pieces only, each tied to its .txt row; a .txt the JSON does not hold refused as the pieces' source and said; a bad piece a gap"


def what_the_ledger_says() -> str:
    """Findings 5 and 6: the ledger's own word on the record (`making`, `disagrees`) is carried; the job graph is held to
    the PNG VHS saved beside the render, ComfyUI's `is_changed` set aside; a ledger named and not found says so."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        share, _, table = _laid(root, [(100, 107, "load_a")])
        led = json.loads((root / "ledger.json").read_text())
        rec = led["renders"]["Video/v/load_a/load_a_00001.mp4"]
        rec["making"], rec["records"][0]["making"] = "not this run's", "written before the record was queued: it is not that run's"
        rec["records"][0]["disagrees"] = ["the file's time"]
        (root / "ledger.json").write_text(json.dumps(led))
        piece = share / "Video" / "v" / "load_a" / "load_a_00001.mp4"
        from PIL import Image, PngImagePlugin
        g = json.loads((root / "load_a_job" / "load_a.render.json").read_text())
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", json.dumps({k: {**v, "is_changed": ["x"]} for k, v in g.items()}))
        Image.new("RGB", (4, 4)).save(piece.with_suffix(".png"), pnginfo=info)
        (p,) = sc.pieces_of_table(table)[0]
        assert p["load"]["ledger_says"] == {"render": "not this run's", "record": "written before the record was queued: it is not that run's",
                                            "disagrees": ["the file's time"]}, p["load"].get("ledger_says")
        assert p["load"]["png"]["agrees"] is True, p["load"].get("png")
        g["74"]["inputs"]["seed"] = 1
        info = PngImagePlugin.PngInfo()
        info.add_text("prompt", json.dumps(g))
        Image.new("RGB", (4, 4)).save(piece.with_suffix(".png"), pnginfo=info)
        (p,) = sc.pieces_of_table(table)[0]
        assert p["load"]["png"]["agrees"] is False and p["load"]["png"]["nodes_that_differ"] == ["74"], p["load"]["png"]
        # the job folder's graph gone: the graph is the PNG's, and it is not held to itself
        (root / "load_a_job" / "load_a.render.json").rename(root / "load_a_job" / "gone.json")
        (p,) = sc.pieces_of_table(table)[0]
        # THE CASE THAT GOES RED when the PNG's graph is held to the same PNG and said to agree
        assert p["load"]["png"]["agrees"] is None and "the PNG's own" in p["load"]["png"]["why"], p["load"]["png"]
        (root / "load_a_job" / "gone.json").rename(root / "load_a_job" / "load_a.render.json")
        # a ledger the table names and that is not where it says: the reason is that, never "no ledger was named"
        root2 = root / "two"
        root2.mkdir()
        _, _, table2 = _laid(root2, [(100, 107, "load_a")], ledger_named=False)
        (p,) = sc.pieces_of_table(table2)[0]
        assert p["load"]["why"].startswith("the ledger the table names is not found") and str(root2 / "gone.json") in p["load"]["why"], p["load"]
    return "the ledger's making and disagrees carried; the job graph held to the PNG's, is_changed aside; a named ledger not found said as that"


def paths_resolved() -> str:
    """Finding 16: inputs and the tool written as absolute paths; a tool that is not a file says why it has no sha256."""
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "clip.mp4")
        here = os.getcwd()
        os.chdir(tmp)
        try:
            sc.write(v, ["python", "bench/nowhere.py"], {"the source": "clip.mp4"})
        finally:
            os.chdir(here)
        s = sc.read(v)
        assert s["inputs"][0]["path"] == str(Path(tmp).resolve() / "clip.mp4") and s["inputs"][0]["sha256"], s["inputs"]
        assert s["tool"]["sha256"] is None and s["tool"]["path"] == str(Path(tmp).resolve() / "bench" / "nowhere.py") and "not a file" in s["tool"]["why"], s["tool"]
    return "inputs and the tool resolved; a tool with no file says why it has no sha256"


def a_check_appended() -> str:
    """Finding 12 and the second read's A: a check of a built video (the assembler's --check-only) is appended under `checks`
    (its command, time, verdict and its record's sha256); the build's command, inputs, extra and pieces stay whole, however
    many checks follow, and a check against another table is said as that. A video with no sidecar gets one that says
    how it was made is not known."""
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "clip.mp4")
        table, other = Path(tmp) / "t.txt", Path(tmp) / "other.txt"
        table.write_text("x")
        other.write_text("y")
        record = Path(tmp) / "clip.mp4.check.json"
        record.write_text("{}")
        sc.write(v, ["python", "bench/assemble_delivery.py", "--out", "clip.mp4"], {"the table": table}, {"verdict": "passes"}, made_by="the build",
                 pieces=[{"name": "p"}])
        built = sc.read(v)
        sc.add_check(v, ["python", "bench/assemble_delivery.py", "--check-only"], "FAILS", "FAILS (a frame dropped)", record, table, made_by="the check")
        record.write_text('{"again": 1}')
        sc.add_check(v, ["python", "bench/assemble_delivery.py", "--check-only", "--table", str(other)], "passes", "passes", record, other)
        s = sc.read(v)
        # THE CASE THAT GOES RED when a check replaces the build's account of how the video was made
        assert {k: s[k] for k in ("command", "inputs", "extra", "pieces", "made_by")} == {k: built[k] for k in ("command", "inputs", "extra", "pieces", "made_by")}, s
        assert [(c["verdict"], c["made_by"], c["table_is_the_builds"]) for c in s["checks"]] == [("FAILS", "the check", True), ("passes", s["checks"][1]["made_by"], False)], s["checks"]
        assert s["checks"][1]["check_record"]["sha256"] == sc.sha256_of(record) and "--check-only" in s["checks"][0]["command"], s["checks"]
        bare = clip(Path(tmp) / "bare.mp4")
        sc.add_check(bare, ["python", "bench/assemble_delivery.py", "--check-only"], "passes", "passes", record, table)
        b = sc.read(bare)
        assert b["made_how"].startswith("not known") and len(b["checks"]) == 1 and b["checks"][0]["table_is_the_builds"] is None, b
    return "a check appended under checks, the build's record whole through two checks; another table said; a video with no sidecar said as not known"


def the_old_name_and_a_piece_that_stops() -> str:
    """The second read's D and its could-not-verify: a sidecar under the old name is named and not read; a piece whose reading
    stops (SystemExit, which `except Exception` does not catch) is a gap of its own, never the table's refusal."""
    with tempfile.TemporaryDirectory() as tmp:
        v = clip(Path(tmp) / "clip.mp4")
        (Path(tmp) / "clip.json").write_text(json.dumps({"schema": sc.SCHEMA, "video_sha256": sc.sha256_of(v)}))
        try:
            sc.read(v)
        except SystemExit as e:
            # THE CASE THAT GOES RED when an old-name sidecar is not said
            assert "has no sidecar" in str(e) and f"an old-name sidecar: {Path(tmp) / 'clip.json'}, not read" in str(e), e
        else:
            raise AssertionError("an old-name sidecar was read")
        root = Path(tmp) / "t"
        root.mkdir()
        _, _, table = _laid(root, [(100, 107, "load_a"), (108, 115, "load_c")])
        real = sc.piece_of

        def stops(row, *a, **k):
            if row.get("name") == "load_c":
                raise SystemExit("a reader stopped")
            return real(row, *a, **k)
        sc.piece_of = stops
        try:
            pieces, _ = sc.pieces_of_table(table)
        except SystemExit as e:
            raise AssertionError(f"a piece's stop escaped the per-piece guard: {e}") from None
        finally:
            sc.piece_of = real
        # THE CASE THAT GOES RED when a piece's SystemExit escapes the per-piece guard
        assert pieces[0]["windows"][0]["text"] == TEXT and "a reader stopped" in pieces[1]["why"], pieces[1]
    return "an old-name sidecar named and not read; a piece whose reading stops is its own gap"


case("write and read back", write_and_read)
case("refusals", refusals)
case("a load's windows, whole", a_loads_windows)
case("the bytes a reference was read from", the_bytes_read)
case("every gap said", gaps_said)
case("a table's pieces", a_tables_pieces)
case("beside a tool's own record", beside_a_tools_record)
case("every input the load was handed, whole", every_input_whole)
case("the text when the reader refuses", the_text_when_the_reader_refuses)
case("through the assembler", through_the_assembler)
case("what the ledger says", what_the_ledger_says)
case("paths resolved", paths_resolved)
case("a check appended", a_check_appended)
case("the old name, and a piece that stops", the_old_name_and_a_piece_that_stops)

if __name__ == "__main__":
    sys.exit(finish())
