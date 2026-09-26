#!/usr/bin/env python3
"""Give draft renders their real decode, from the latents the drafts saved.

A graph built with `draft_decode` (`workflows/build_workflows.py::build_api`)
decodes its video with `h3_config.DRAFT_VAE` and saves the sampled latent in
two halves. This finds each draft row's two files in the server's `/history`
and runs `workflows/h3_decode_saved_latent_api.json` on them through
`bench/run_graph_arms.py`, one row out per draft, all under one label, so
the result is a JSONL that `bench/blind_batch.py` can blind like any other.

    python bench/decode_draft_keepers.py --drafts <draft rows>.jsonl \\
        --out <keeper rows>.jsonl [--label full] [--rows 0,3,4]

`--rows` picks keepers by their index among the non-warmup draft rows;
without it every draft is decoded, which is what open_experiments #31 needs.
Each output row's patches name the latent files it decoded, and this prints
the draft row -> latent -> seed mapping, so the pairing survives blinding.

`/history` resets when the server restarts: run this before restarting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "workflows" / "h3_decode_saved_latent_api.json"
#: node ids `build_api(draft_decode=True)` gives the two SaveLatent nodes. Inherited.
VIDEO_SAVE, AUDIO_SAVE = "102", "103"
#: node ids of the two LoadLatent nodes in the keeper graph. Inherited.
VIDEO_LOAD, AUDIO_LOAD = "104", "105"


def saved_latent(history: dict, node: str) -> str | None:
    outs = (history.get("outputs") or {}).get(node) or {}
    files = outs.get("latents") or []
    if not files:
        return None
    f = files[0]
    sub = f.get("subfolder") or ""
    return f"{sub + '/' if sub else ''}{f['filename']} [{f.get('type', 'output')}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drafts", type=Path, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="full")
    ap.add_argument("--rows", default=None, help="comma-separated indices of non-warmup draft rows")
    ap.add_argument("--host", default="127.0.0.1:8188")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.drafts.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("warmup")]
    pick = range(len(rows)) if args.rows is None else [int(x) for x in args.rows.split(",")]
    plan = []
    for i in pick:
        r = rows[i]
        with urllib.request.urlopen(f"http://{args.host}/history/{r['prompt_id']}") as resp:
            hist = json.load(resp).get(r["prompt_id"]) or {}
        video, audio = saved_latent(hist, VIDEO_SAVE), saved_latent(hist, AUDIO_SAVE)
        if not video or not audio:
            raise SystemExit(f"draft row {i} (prompt {r['prompt_id']}) has no saved latent in /history; "
                             "was it rendered by a draft_decode graph on this server session?")
        plan.append((i, r.get("seed"), video, audio))

    for i, seed, video, audio in plan:
        print(f"draft row {i}  seed {seed}  {video}  {audio}", flush=True)
        cmd = [sys.executable, str(REPO / "bench" / "run_graph_arms.py"),
               "--arm", f"{args.label}={GRAPH}",
               "--set", f"{args.label}:{VIDEO_LOAD}.latent={json.dumps(video)}",
               "--set", f"{args.label}:{AUDIO_LOAD}.latent={json.dumps(audio)}",
               "--runs", "1", "--host", args.host, "--out", args.out]
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
