#!/usr/bin/env python3
"""Emit `2026-08-18_attention_defaults.json` from the raw arm results.

The data file beside this script is the record the docs link to. It is
GENERATED rather than typed, for the reason `CLAUDE.md` gives about numbers in
prose: a hand-copied figure is a second copy that drifts silently, and the
whole point of this run was to stop reasoning from numbers nobody could trace
back to a command.

Input is the runner's `results.jsonl` (one JSON object per arm, written by the
harness as each render finished). Substrate is read live from `nvidia-smi` and
the installed packages, not remembered -- `docs/hardware.md` records that a
board power limit changes render times, is set outside the repo, and is
invisible in a workflow JSON, so a timing record that does not carry it cannot
be compared against later.

    python bench/results/make_attention_defaults_json.py <results.jsonl>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "2026-08-18_attention_defaults.json"

# What each arm varied. `sol` is whether a SolAttnMiniMax node was REACHABLE
# from an output node, which is the property that decides whether it runs --
# not whether the node was present. `docs/evidence.md` records a capture whose
# provenance field asked the wrong one and reported the situation backwards.
ARMS = {
    "A_sol_fp16":      dict(sage="fp16 (most accurate)", sol=True,
                            note="the shipped configuration before this run"),
    "B_nosol_fp16":    dict(sage="fp16 (most accurate)", sol=False,
                            note="shipped sage, Sol not reachable"),
    "C_sol_auto":      dict(sage="auto", sol=True,
                            note="auto resolves to fp8_cuda++ on sm89"),
    "D_nosol_auto":    dict(sage="auto", sol=False,
                            note="the 2026-08-10 configuration, before either change"),
    "E_ref2va_nolora": dict(sage="fp16 (most accurate)", sol=True,
                            note="ref2va checkpoint direct, no ref LoRA; "
                                 "isolates the LoRA's 320 attached patches"),
    "F_userlive":      dict(sage="BYPASSED", sol=True,
                            note="the owner's live UI graph: sage node bypassed, "
                                 "Sol min_tokens 11776, morton on"),
}


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def substrate():
    q = ("name,driver_version,power.limit,power.default_limit,"
         "memory.total,pstate")
    raw = sh("nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader")
    parts = [p.strip() for p in raw.split(",")] if "," in raw else []
    out: dict[str, object] = {"raw": raw}
    if len(parts) >= 6:
        out.update(gpu=parts[0], driver=parts[1], power_limit=parts[2],
                   power_default_limit=parts[3], memory_total=parts[4])
    out["power_limit_is_stock"] = out.get("power_limit") == out.get("power_default_limit")
    # In-process, metadata first: `comfy_kitchen` has no `__version__`
    # attribute (its version lives in dist metadata), which is how the
    # 2026-08-18 workload-grid rows recorded "?" for exactly the packages a
    # substrate row exists to pin.
    for mod in ("torch", "comfy_kitchen", "sageattention"):
        try:
            import importlib.metadata as _md
            out[mod] = _md.version(mod)
        except Exception:
            try:
                out[mod] = getattr(__import__(mod), "__version__", "?")
            except Exception as exc:
                out[mod] = f"unavailable: {exc}"
    # Anchored to THIS repo, not the caller's cwd: invoked from the ComfyUI
    # checkout the bare form records that repo's commit and dirty state as
    # this one's.
    repo = str(Path(__file__).resolve().parents[2])
    out["git_commit"] = sh("git", "-C", repo, "rev-parse", "HEAD")
    out["git_dirty"] = bool(sh("git", "-C", repo, "status", "--porcelain"))
    return out


def main():
    src = Path(sys.argv[1])
    rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("status") == "success" and r.get("sampler_s")]

    arms, by = [], {}
    for r in rows:
        meta = ARMS.get(r["arm"], {})
        by[r["arm"]] = r["sampler_s"]
        arms.append({
            "id": r["arm"],
            "sage_mode": meta.get("sage"),
            "sol_reachable": meta.get("sol"),
            "sampler_seconds": r["sampler_s"],
            "seconds_per_step": round(r["sampler_s"] / 16, 2),
            "sol_composed_in_log": r.get("sol_composed"),
            "note": meta.get("note"),
            "prompt_id": r.get("prompt_id"),
        })

    d = {}
    def ratio(name, a, b, why):
        if a in by and b in by:
            d[name] = {"value": round(by[a] / by[b], 3),
                       "from": f"{a}={by[a]}s / {b}={by[b]}s", "means": why}
    ratio("sol_worth_at_fp16", "B_nosol_fp16", "A_sol_fp16",
          "what an unreachable Sol node costs, at the shipped sage mode")
    ratio("sol_worth_at_auto", "D_nosol_auto", "C_sol_auto",
          "the same, at sage auto")
    ratio("fp16_cost_with_sol", "A_sol_fp16", "C_sol_auto",
          "what fp16 costs in the SHIPPED config, where Sol takes 11 of 16 steps")
    ratio("fp16_cost_dense", "B_nosol_fp16", "D_nosol_auto",
          "what fp16 costs when sage runs all 16 steps")
    ratio("ref_lora_cost", "A_sol_fp16", "E_ref2va_nolora",
          "fl2va + ref LoRA against the ref2va checkpoint direct")
    ratio("worst_over_best", "B_nosol_fp16", "C_sol_auto",
          "the full spread across the 2x2")

    # The step decomposition is the load-bearing part: Sol's own kernel cost is
    # identical across sage modes, so the whole fp16 difference lives in the
    # steps sage actually runs. Derived from the fully-dense arm's rate.
    steps = {}
    for mode, sol_arm, dense_arm in [("fp16", "A_sol_fp16", "B_nosol_fp16"),
                                     ("auto", "C_sol_auto", "D_nosol_auto")]:
        if sol_arm in by and dense_arm in by:
            dense_rate = by[dense_arm] / 16
            sparse_rate = (by[sol_arm] - 5 * dense_rate) / 11
            steps[mode] = {
                "dense_step_seconds": round(dense_rate, 1),
                "sparse_step_seconds": round(sparse_rate, 1),
                "dense_over_sparse": round(dense_rate / sparse_rate, 2),
            }
    d["per_step"] = {
        "by_sage_mode": steps,
        "window": "start_percent 0.2 / end_percent 0.9 with scheduler `simple` "
                  "at 16 steps gives Sol 11 sparse and 5 dense",
        "reading": "the sparse-step cost is the same in both sage modes -- Sol's "
                   "kernel does not depend on the sage mode -- so the entire fp16 "
                   "difference is carried by the 5 dense steps",
    }

    payload = {
        "measurement": "sage kernel mode x Sol-Attn reachability, end to end",
        "date": "2026-08-18",
        "why": ("`docs/evidence.md` withdrew every fp8-vs-fp16 accuracy ratio on "
                "2026-08-16, leaving the shipped `fp16 (most accurate)` default "
                "resting only on a perceptual verdict taken 2026-08-13 at 124 "
                "frames with Sol ABSENT -- one day before Sol-Attn landed. This "
                "run measures the cost side in the configuration actually shipped."),
        "method": ("One variable per arm, same graph, same seed, same sequence. "
                   "Sampler seconds are read from the progress line, so model "
                   "load and VAE decode are excluded. One run per arm."),
        "held_fixed": {
            "graph": "workflows/h3_probe_capture_ref3_api.json, length set to 260",
            "canvas": "1024x768", "frames": 260, "latent_frames": 77,
            "packed_sequence_length": 75118,
            "steps": 16, "sampler": "er_sde", "scheduler": "simple",
            "sigma_shift": [12.0, 3.0], "seed": 730451892,
            "references": 3, "ref_image_size": "max",
            "model": "minimax_h3_fl2va_pruned_int8_convrot + ref LoRA @1.0 "
                     "(except arm E)",
        },
        "substrate": substrate(),
        "arms": arms,
        "derived": d,
        "caveats": [
            "One run per arm. This bench's run-to-run spread was measured at "
            "0.1-0.12% on a previous occasion, but that was a different harness.",
            "Ratios were measured at packed sequence 75,118. Sol's advantage grows "
            "with sequence length, so the Sol ratios understate at 362 frames.",
            "The board power limit was 330 W against a stock 450 W for this whole "
            "run. Every arm shares it, so the RATIOS hold; the absolute seconds "
            "are not comparable to any number recorded before 2026-08-17 15:39.",
            "Sampler seconds only. Total wall clock adds model load, text "
            "encoding and VAE decode, which no arm varied.",
            "This measures COST. It says nothing about output quality; the paired "
            "clips rendered at the same seed are the evidence for that question.",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent.parent)}  ({len(arms)} arms)")
    for k, v in d.items():
        if isinstance(v, dict) and "value" in v:
            print(f"  {k:22} {v['value']}x   {v['from']}")


if __name__ == "__main__":
    sys.exit(main())
