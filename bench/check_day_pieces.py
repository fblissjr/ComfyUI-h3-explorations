#!/usr/bin/env python3
"""`bench/day_pieces.py` makes the folder of a day's standing pieces, each with the graph it was queued with; each case is a way the folder could hand over the wrong graph, hide a missing input, copy a video, or drop a piece.

  the_graph_is_the_one_queued           `graph_as_queued.json` is the job folder's queued graph byte for byte, `queue_this.json` is that graph as
                                        the server's /prompt body, the text is the load's own, and the line that queues it names the host
  a_missing_input_is_printed_not_refused   a file the graph names that is not there reads MISSING in `inputs.txt`, the run still exits 0, and one
                                        that is there says its size and when it was written
  an_input_made_again_says_so           an input written after the graph was queued is marked so, and one written before is not
  a_model_is_a_name                     a model the graph names is found or not found under the server's model folders, said as that
  no_video_is_copied                    the folder holds no video; the page links the render and the stacked clip by a relative path that
                                        resolves from the folder to the file on the share
  a_piece_with_no_queued_graph_is_still_a_row   it has its row and its folder, which says no graph is beside its record
  the_rows_are_in_frame_order           with the day each render was written and the owner's last choice where they made one
  the_joined_clip_heads_the_page        with --clip, --stacked and --clip-table the page's first links are to them

    <python> bench/check_day_pieces.py

The standard library only; no model, no card. The tool reads job folders by paths relative to the repo, so the made
job folder is under the repo's untracked `data/` and is removed at the end.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TOOL = HERE / "day_pieces.py"


def main() -> int:
    fails = 0

    def case(name, ok, why: object = ""):
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + name + ("" if ok else f": {str(why)[:900]}"))
        fails += not ok

    jobs = Path(tempfile.mkdtemp(prefix="check_day_pieces_job_", dir=REPO / "data"))
    try:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            share, inputs, models, out = root / "share", root / "input", root / "models", root / "share" / "Video" / "v" / "day"
            for p in (share / "Video/v/s/one/one", share / "Video/v/review", share / "Video/other/two/two", inputs / "stills", models / "unet", share / "Video/v/masks"):
                p.mkdir(parents=True)
            for f in ("Video/v/s/one/one/one_00001.mp4", "Video/v/review/one_stacked.mp4", "Video/other/two/two/two_00001.mp4"):
                (share / f).write_bytes(b"a video")
            (inputs / "stills/head.png").write_bytes(b"png")
            (models / "unet/the_model.safetensors").write_bytes(b"weights")
            old, new = share / "Video/v/masks/old.mkv", share / "Video/v/masks/new.mkv"
            old.write_bytes(b"old mask")
            new.write_bytes(b"a mask made again")
            long_ago = time.time() - 10 * 86400
            os.utime(old, (long_ago, long_ago))
            # queued five days ago, with an offset as the queue records write it: between the two masks' times
            queued_at = datetime.fromtimestamp(time.time() - 5 * 86400, timezone(timedelta(hours=-5))).isoformat(timespec="seconds")
            os.utime(inputs / "stills/head.png", (long_ago, long_ago))
            graph = {"1": {"class_type": "Loader", "inputs": {"video": str(old), "image": "stills/head.png"}},
                     "2": {"class_type": "Loader", "inputs": {"video": str(new), "audio": "/nowhere/gone.wav"}},
                     "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "the_model.safetensors"}},
                     "4": {"class_type": "VAELoader", "inputs": {"vae_name": "absent_vae.safetensors"}},
                     "5": {"class_type": "Song", "inputs": {"filename_prefix": "Video/v/s/one/one/one", "seed": 7, "prompt": "words with a / in them are not a file"}}}
            raw = json.dumps(graph, indent=3).encode() + b"\n\n"   # its own bytes, to be copied as they are
            (jobs / "one.render.json").write_bytes(raw)
            (jobs / "one.render.txt").write_text("THE TEXT OF one\n")
            rel = jobs.relative_to(REPO)
            ledger = {"share": str(share), "renders": {
                "Video/v/s/one/one/one_00001.mp4": {"written": "2026-01-05T15:30:00Z", "records": [{"built_file": f"{rel}/one.render.built.json", "queued_file": f"{rel}/one.render.queued.json",
                                                                                                  "commit": "abc1234", "queued_at": queued_at}],
                                                    "owner": [{"at": "2026-01-05T16:00:00Z", "what": "choice", "said": "not sure"}, {"at": "2026-01-05T16:01:00Z", "what": "words", "said": "hm"},
                                                              {"at": "2026-01-05T16:02:00Z", "what": "choice", "said": "keep this one"}]},
                "Video/other/two/two/two_00001.mp4": {"written": "2026-01-04T09:00:00Z", "records": [], "owner": []}}}
            table = {"rows": [{"first": 300, "last": 320, "file": "Video/v/s/one/one/one_00001.mp4", "name": "one", "subject": "a", "region": "face"},
                              {"first": 40, "last": 90, "file": "Video/other/two/two/two_00001.mp4", "name": "two", "subject": "b", "region": "face and hair"}]}
            items = {"items": [{"id": "one", "path": str(share / "Video/v/review/one_stacked.mp4")}]}
            for name, obj in (("ledger.json", ledger), ("table.json", table), ("items.json", items)):
                (root / name).write_text(json.dumps(obj))
            clip = share / "Video/v/day/joined.mp4"
            out.mkdir(parents=True)
            clip.write_bytes(b"the joined clip")
            (out / "joined_STACKED.mp4").write_bytes(b"stacked")
            (out / "joined.txt").write_text("the table")
            cmd = [sys.executable, str(TOOL), "--table", str(root / "table.json"), "--ledger", str(root / "ledger.json"), "--items", str(root / "items.json"),
                   "--inputs", str(inputs), "--models", str(models), "--out", str(out), "--host", "server:1"]
            r = subprocess.run(cmd + ["--clip", str(clip), "--stacked", str(out / "joined_STACKED.mp4"), "--clip-table", str(out / "joined.txt")], capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            one, two = out / "0300_one", out / "0040_two"
            page = (out / "index.html").read_text()
            case("the_graph_is_the_one_queued", (one / "graph_as_queued.json").read_bytes() == raw and json.loads((one / "queue_this.json").read_text()) == {"prompt": graph}
                 and (one / "text.txt").read_text() == "THE TEXT OF one\n" and "--data-binary @queue_this.json http://server:1/prompt" in (one / "how_to_queue.txt").read_text()
                 and "next-numbered file" in (one / "how_to_queue.txt").read_text(), sorted(p.name for p in one.iterdir()))
            lines = (one / "inputs.txt").read_text().splitlines()
            pick = lambda word: next((line for line in lines if word in line), "")                                     # noqa: E731
            case("a_missing_input_is_printed_not_refused", pick("gone.wav").startswith("MISSING") and "/nowhere/gone.wav" in pick("gone.wav")
                 and pick("stills/head.png").startswith("THERE    3 bytes, written ") and f"(as {inputs / 'stills/head.png'})" in pick("stills/head.png")
                 and "2 MISSING" in r.stdout and not any("words with a" in line for line in lines), lines)
            case("an_input_made_again_says_so", "WRITTEN AFTER THE GRAPH WAS QUEUED" in pick("new.mkv") and "WRITTEN AFTER" not in pick("old.mkv")
                 and pick("old.mkv").startswith("THERE") and "WRITTEN AFTER" not in pick("head.png"), [pick("new.mkv"), pick("old.mkv")])
            case("a_model_is_a_name", pick("the_model.safetensors").startswith("THERE    a model, found under the server's model folders as") and "unet/the_model.safetensors" in pick("the_model")
                 and pick("absent_vae").startswith("MISSING  a model, NOT FOUND under the server's model folders") and pick("filename_prefix").startswith("WRITES"), lines)
            videos = [p for p in out.rglob("*") if p.suffix in (".mp4", ".mkv", ".mov") and p.parent != out]
            hrefs = re.findall(r'href="([^"]+)"', page)
            resolve = lambda h: (out / h).resolve()                                                                   # noqa: E731
            case("no_video_is_copied", not videos and "../s/one/one/one_00001.mp4" in hrefs and "../review/one_stacked.mp4" in hrefs and "../../other/two/two/two_00001.mp4" in hrefs
                 and all(resolve(h).exists() for h in hrefs) and not any(h.startswith("/") for h in hrefs), (videos, hrefs))
            case("a_piece_with_no_queued_graph_is_still_a_row", two.is_dir() and (two / "NO_QUEUED_GRAPH.txt").is_file() and not (two / "graph_as_queued.json").exists()
                 and "<b>two</b>" in page and "NO QUEUED GRAPH beside its record" in page and "stacked: not there" in page, sorted(p.name for p in two.iterdir()) if two.is_dir() else "no folder")
            case("the_rows_are_in_frame_order", page.index("<b>two</b>") < page.index("<b>one</b>") and "<td>2026-01-05</td>" in page and "<td>2026-01-04</td>" in page
                 and "keep this one (2026-01-05T16:02:00Z)" in page and "not sure" not in page and "no choice of the owner&#x27;s on it" in page and "built at commit abc1234" in page, page[-900:])
            first = page.index("THE JOINED CLIP")
            case("the_joined_clip_heads_the_page", first < page.index("<table>") and 'href="joined.mp4"' in page and 'href="joined_STACKED.mp4"' in page and 'href="joined.txt"' in page, page[:900])
            bare = subprocess.run(cmd, capture_output=True, text=True)
            assert bare.returncode == 0 and "THE JOINED CLIP" not in (out / "index.html").read_text(), "with no --clip the page still heads with one"
    finally:
        shutil.rmtree(jobs)
    print(f"\n{'all' if not fails else fails} {'of ' if fails else ''}8 case(s) {'FAILED' if fails else 'ok'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
