#!/usr/bin/env python3
"""Does Sol's error at ONE block reach the output? Measured, one block at a time.

## The question this closes

`bench/rank_dense_blocks.py` ranks blocks by the error `dense_blocks` removes
LOCALLY -- Sol's error at that block minus sage's. It cannot see propagation:
an error at block 0 travels through 49 more blocks and may be amplified or
washed out, while block 49's lands on the output head directly.
`docs/SOLATTN.md` names that gap as the one thing that could overturn the local
ranking. This measures it.

## Why this needs neither an exact kernel nor a clean render

**The arms differ by one thing and share everything else, so the kernel
cancels.** Every arm runs sage at all 50 blocks; the only difference is that
ONE block is handed to Sol instead, via `dense_blocks` naming the other 49. So
what the output delta measures is Sol-at-block-N against sage-at-block-N,
propagated through whatever follows. Turning sage off would answer a different
and much more expensive question.

**It is one differing forward pass, not a rendered comparison.** The schedule
is 4 steps on euler at a fixed seed, and Sol's sigma window is set to contain
only the FINAL step (start_percent 0.66 -> sigma 0.8608, which sits between the
third step's 0.9231 and the last step's 0.8). Steps 0-2 are therefore
bit-identical across arms by construction, and the saved latent differs only by
what happened at one block on one step. CLAUDE.md's different-sample rule is
about clips drawn from diverged trajectories; nothing here diverges, because
nothing after the measured forward re-samples.

The measured step sits at sigma 0.8, which is where the 4-evaluation PDD
partition's final block begins -- the coarsest, most error-sensitive stretch of
the schedule, and the regime the PDD arms actually run.

## What it produces

Per block: `rel_l2` of the arm's output latent against the Sol-off baseline.
**That single number is already the benefit ranking** -- it folds together how
large Sol's error is at that block and how much of it survives to the output,
which is exactly the quantity `dense_blocks` should be chosen on. A block with
a high value is one worth keeping dense.

## What it does not establish

* One seed, one prompt, one canvas, one step. It measures the network's
  transfer characteristic, not a distribution over content.
* `rel_l2` on the packed latent mixes the video and audio segments. The split
  is reported separately where the layout is recoverable.
* Sol at one block is not Sol at five: the routing is per-block and
  independent, but the errors do not have to add linearly once they interact
  through the residual stream. Read this as a ranking, not as a budget.

    python bench/probe_block_propagation.py --blocks 0,1,2,8,16,24,32,40,48,49
    python bench/probe_block_propagation.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))

BASE_GRAPH = REPO / "workflows" / "h3_text_to_video_api.json"
OUT = REPO / "bench" / "results" / "2026-08-29_block_propagation.json"

# The Sol node's sigma window for this probe, chosen so exactly ONE sampling
# step falls inside it. Derived by walking `comfy.samplers.calculate_sigmas` at
# shift 12 and 4 steps -- sigmas 1.0, 0.973, 0.9231, 0.8 -- and picking a start
# between the third and fourth. Not a shipped value and must never be copied
# into a graph: it exists to isolate one forward.
PROBE_START_PERCENT = 0.66
PROBE_END_PERCENT = 1.0
PROBE_STEPS = 4
N_BLOCKS = 50

# The owner's media directories are shared; probe output goes in its own
# subfolder rather than at the root.
PREFIX = "latents/h3_block_propagation/arm"


def all_but(block: int) -> str:
    """`dense_blocks` naming every block EXCEPT `block`.

    Spelled as explicit ranges rather than a comma list of 49 indices, so the
    node's own log line ("keeping blocks [...] dense of 50") stays readable.
    """
    if block == 0:
        return f"1-{N_BLOCKS - 1}"
    if block == N_BLOCKS - 1:
        return f"0-{N_BLOCKS - 2}"
    return f"0-{block - 1},{block + 1}-{N_BLOCKS - 1}"


def build_arm(base: dict, block: int | None, seed: int) -> tuple[dict, str]:
    """One arm: Sol active at `block` only, or bypassed entirely when None."""
    g = json.loads(json.dumps(base))

    # Find the nodes by class rather than by id, so a regenerated base graph
    # that renumbers does not silently probe the wrong thing.
    def one(*classes):
        """The single node matching any of `classes`.

        Takes alternatives because the Sol node id changed on
        2026-08-30 and this looked for the vendored name alone until
        2026-08-31 -- against a regenerated graph it found zero and
        exited, which at least failed loudly rather than silently.
        """
        hits = [k for k, v in g.items()
                if v.get("class_type") in classes]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly one of {classes}, found {len(hits)}")
        return hits[0]

    sol, sched, sampler, noise = (one("MiniMaxH3SolAttn", "SolAttnMiniMax"), one("BasicScheduler"),
                                 one("KSamplerSelect"), one("RandomNoise"))

    g[sched]["inputs"]["steps"] = PROBE_STEPS
    g[sampler]["inputs"]["sampler_name"] = "euler"   # deterministic
    g[noise]["inputs"]["noise_seed"] = seed

    if block is None:
        # Baseline: sage everywhere. The Sol node is REMOVED and its consumer
        # repointed at its source, rather than left in with a wide dense_blocks
        # -- an installed override that declines every call is not the same code
        # path as no override, and this probe cannot afford that ambiguity.
        src = g[sol]["inputs"]["model"]
        for v in g.values():
            for key, val in v.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and val[0] == sol:
                    v["inputs"][key] = src
        del g[sol]
    else:
        s = g[sol]["inputs"]
        s["start_percent"] = PROBE_START_PERCENT
        s["end_percent"] = PROBE_END_PERCENT
        s["dense_blocks"] = all_but(block)

    # Drop both decodes and the muxer; save the latent instead. The probe reads
    # the model's output, and a VAE pass is minutes of wall time that only adds
    # another approximation between the thing measured and the number.
    sca = one("SamplerCustomAdvanced")
    for cls in ("VAEDecode", "VAEDecodeAudio", "VHS_VideoCombine"):
        for k in [k for k, v in g.items() if v.get("class_type") == cls]:
            del g[k]
    tag = "baseline" if block is None else f"b{block:02d}"
    # H3's sampler output is a packed NestedTensor and `SaveLatent` cannot
    # serialize it ("'NestedTensor' object has no attribute 'contiguous'").
    # `LTXVSeparateAVLatent` unbinds it and declares MiniMax H3 in its own
    # description, so the two streams are saved -- and compared -- apart, which
    # is better than the mixed number this probe would otherwise report.
    g["900"] = {"class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": [sca, 0]}}
    g["901"] = {"class_type": "SaveLatent",
                "inputs": {"samples": ["900", 0],
                           "filename_prefix": f"{PREFIX}_{tag}_video"}}
    g["902"] = {"class_type": "SaveLatent",
                "inputs": {"samples": ["900", 1],
                           "filename_prefix": f"{PREFIX}_{tag}_audio"}}
    return g, tag


def submit(graph: dict, host: str, timeout: float,
           cache_watch: str | None = None) -> tuple[dict, dict]:
    """Queue one graph and wait. Returns (latents, meta).

    `meta["server_ms"]` is the SERVER's own execution span, from the
    `execution_start` and `execution_success` timestamps ComfyUI records in
    history -- not this loop's wall clock, which is quantised to the poll
    interval below. That distinction is not academic: the first version of
    `time_dense_blocks.py` reported arm differences of exactly 3.0 s, which was
    the poll interval rather than the effect.
    """
    client = str(uuid.uuid4())
    body = json.dumps({"prompt": graph, "client_id": client}).encode()
    req = urllib.request.Request(f"http://{host}/prompt", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            pid = json.load(r)["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"submit rejected: {e.read().decode()[:600]}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3.0)
        try:
            with urllib.request.urlopen(f"http://{host}/history/{pid}",
                                        timeout=30) as r:
                hist = json.load(r)
        except Exception:
            continue
        if pid not in hist:
            continue
        status = hist[pid].get("status", {})
        # A failed render appears in history like any other; read the status
        # rather than the presence of the key. This is the defect
        # `bench/run_capture.py` records having shipped.
        if not status.get("completed", False) or status.get("status_str") != "success":
            raise SystemExit(f"render did not succeed: {json.dumps(status)[:600]}")
        outs = hist[pid].get("outputs", {})
        found = {}
        for nid, node in outs.items():
            for f in node.get("latents", []) or []:
                stream = "video" if nid == "901" else "audio" if nid == "902" else nid
                found[stream] = (f["filename"], f.get("subfolder", ""))
        if set(found) != {"video", "audio"}:
            raise SystemExit(f"expected a video and an audio latent, got "
                             f"{sorted(found)}: {json.dumps(outs)[:400]}")
        stamps = {m[0]: m[1].get("timestamp") for m in status.get("messages", [])
                  if isinstance(m, (list, tuple)) and len(m) == 2
                  and isinstance(m[1], dict)}
        server_ms = None
        if stamps.get("execution_start") and stamps.get("execution_success"):
            server_ms = stamps["execution_success"] - stamps["execution_start"]
        # `execution_cached` naming the node in `cache_watch` -- the SAMPLER --
        # means ComfyUI served a previous identical run and nothing was
        # computed. Watching one named node rather than "is anything cached",
        # because the four loaders are cached on every render after the first
        # and a flag that fires every time is a flag nobody can act on.
        #
        # This is how the first timing run produced a 3.0 s "render" and a
        # per-block cost off by more than an order of magnitude: the warm-up
        # had the same inputs as the first timed arm, so the sampler was served
        # from cache while SaveLatent still re-ran and wrote a file.
        cached = False
        if cache_watch is not None:
            for m in status.get("messages", []):
                if (isinstance(m, (list, tuple)) and len(m) == 2
                        and m[0] == "execution_cached"
                        and cache_watch in (m[1] or {}).get("nodes", [])):
                    cached = True
        return found, {"prompt_id": pid, "server_ms": server_ms, "cached": cached}
    raise SystemExit(f"timed out after {timeout}s waiting for {pid}")


def load_latent(path: Path):
    from safetensors.torch import load_file
    d = load_file(str(path))
    key = "latent_tensor" if "latent_tensor" in d else next(iter(d))
    return d[key].float()


def rel_l2(a, b) -> float:
    import torch
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default="0,1,2,8,16,24,32,40,48,49")
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--seed", type=int, default=730451892)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--output-dir", default="/mnt/hub/ai/img/output",
                    help="where the server writes; latents land under it")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--score-existing", action="store_true",
                    help="render nothing; score the latents already on disk. "
                         "How a run interrupted partway is salvaged -- each arm "
                         "is an independent render and its latent persists, so "
                         "the blocks that completed keep their numbers.")
    args = ap.parse_args()

    blocks = [int(b) for b in args.blocks.split(",") if b.strip() != ""]
    bad = [b for b in blocks if not 0 <= b < N_BLOCKS]
    if bad:
        raise SystemExit(f"blocks out of range for a {N_BLOCKS}-block DiT: {bad}")

    base = json.loads(BASE_GRAPH.read_text())
    outdir = Path(args.output_dir)

    arms, timings = {}, {}
    if args.score_existing:
        latents = outdir / "latents" / "h3_block_propagation"
        for block in [None] + blocks:
            tag = "baseline" if block is None else f"b{block:02d}"
            paths = {}
            for stream in ("video", "audio"):
                # An arm may have been rendered more than once; take the newest,
                # which is the one the most recent run wrote.
                hits = sorted(latents.glob(f"arm_{tag}_{stream}_*.latent"))
                if hits:
                    paths[stream] = hits[-1]
            if set(paths) == {"video", "audio"}:
                arms[tag] = paths
                timings[tag] = float("nan")
            elif block is None:
                raise SystemExit(f"no baseline latents under {latents}")
            else:
                print(f"  b{block:02d}      not on disk, skipping")
        blocks = [b for b in blocks if f"b{b:02d}" in arms]
        print(f"  scoring {len(blocks)} arm(s) already on disk, rendering none")

    for block in ([] if args.score_existing else [None] + blocks):
        graph, tag = build_arm(base, block, args.seed)
        label = "baseline (sage only)" if block is None else f"Sol at block {block} only"
        print(f"  {tag:9s} {label} ...", end="", flush=True)
        t0 = time.time()
        got, _meta = submit(graph, args.host, args.timeout)
        dt = time.time() - t0
        timings[tag] = dt
        paths = {}
        for stream, (fname, sub) in got.items():
            p = outdir / sub / fname
            if not p.exists():
                raise SystemExit(f"\nserver reported {p} but it is not readable here")
            paths[stream] = p
        arms[tag] = paths
        print(f" {dt:6.1f}s")

    print("\n  loading latents ...")
    base_t = {k: load_latent(v) for k, v in arms["baseline"].items()}
    rows = []
    for block in blocks:
        row = {"block": block, "seconds": timings[f"b{block:02d}"]}
        for stream, path in arms[f"b{block:02d}"].items():
            t = load_latent(path)
            if t.shape != base_t[stream].shape:
                raise SystemExit(f"block {block} {stream}: shape {tuple(t.shape)} "
                                 f"against baseline {tuple(base_t[stream].shape)}")
            row[f"rel_l2_{stream}"] = rel_l2(t, base_t[stream])
        row["rel_l2"] = row["rel_l2_video"]
        if timings[f"b{block:02d}"] == timings[f"b{block:02d}"]:  # not NaN
            pass
        else:
            row.pop("seconds")
        rows.append(row)

    # A probe whose arms all agree with the baseline measured nothing -- most
    # likely the Sol node never engaged (window, min_tokens, or a dense_blocks
    # spec that swallowed the block). Say so rather than printing zeros as a
    # result.
    live = [r for r in rows
            if max(r["rel_l2_video"], r["rel_l2_audio"]) > 1e-6]
    print(f"\n  Sol at one block only, effect on the output latent "
          f"(baseline = sage at all {N_BLOCKS})\n")
    print("    block     video      audio")
    for r in sorted(rows, key=lambda r: -r["rel_l2"]):
        print(f"      {r['block']:2d}     {r['rel_l2_video']:.6f}   "
              f"{r['rel_l2_audio']:.6f}")
    if not live:
        raise SystemExit(
            "\n  EVERY arm is identical to the baseline. Sol never engaged: check "
            "the sigma window against the schedule, min_tokens against the packed "
            "length, and that dense_blocks left the target block out.")
    print(f"\n    {len(live)} of {len(rows)} arm(s) moved the output")

    if args.write:
        OUT.write_text(json.dumps({
            "measured": "2026-08-29",
            "produced_by": "bench/probe_block_propagation.py",
            "what": "effect on the output latent of letting Sol run at exactly one "
                    "DiT block, against a sage-everywhere baseline. Folds together "
                    "Sol's local error at that block and how much of it survives to "
                    "the output, which is the quantity dense_blocks should be "
                    "chosen on.",
            "base_graph": BASE_GRAPH.name,
            "steps": PROBE_STEPS,
            "sampler": "euler",
            "seed": args.seed,
            "sol_window": {"start_percent": PROBE_START_PERCENT,
                           "end_percent": PROBE_END_PERCENT,
                           "note": "contains only the final step, sigma 0.8 at "
                                   "shift 12 -- steps 0-2 are identical across arms"},
            "baseline": "Sol node REMOVED, sage at all 50 blocks",
            "rows": rows,
            "caveats": [
                "One seed, one prompt, one canvas, one step.",
                "video and audio are reported apart; `rel_l2` mirrors the video "
                "column so a reader who ignores the split gets the larger stream.",
                "Sol at one block is not Sol at five; the per-block errors need "
                "not add linearly. Read the ranking, not a budget.",
                "The measured step is the last of four, sigma 0.8 -- where the "
                "4-evaluation PDD partition's final block begins.",
            ],
        }, indent=1) + "\n")
        print(f"\n  wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
