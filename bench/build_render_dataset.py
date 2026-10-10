#!/usr/bin/env python3
"""The 2026-09-26 renders as tables for DuckDB: renders, measures, models, findings.

Written 2026-09-27 for the owner ("put it into a json as well in a structured
schema so I can analyze it as data ... into a duckdb database"), with the
schema agreed with fastdude. Writes JSONL tables into `--out`, joined on
`render_id` = "<record stem>:<label>:<n>" (n counts that label's rows in that
record from 1):

- `renders.jsonl`, one row per run row. It holds the configuration resolved
  by `bench/render_inventory.py` (the graph as rendered, patches applied), the
  prompt (bank id and sha), seed and length, the code version (the row's own
  `substrate.git_commit` and the CHANGELOG version at that commit), timings,
  cache position (from the substrate record, joined on `prompt_id`), validity
  flags and the output basenames.
- `measures.jsonl`, long format: one row per (render, tool, metric) from every
  per-clip record in `data/results/2026-09-2[67]_*.json`. A clip measured
  twice by the same tool is kept once, with every source record listed.
- `models.jsonl`: every model file a render used or we built.
- `findings.jsonl`: the concatenation of `findings_*.jsonl` in `--out`,
  validated against the agreed fields and tag vocabulary.

    python bench/build_render_dataset.py --out data/results/2026-09-27_render_dataset
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))
from render_inventory import apply_patches, describe, graph_at, graph_prefixes, outputs  # noqa: E402
from _paths import comfy_output  # noqa: E402
sys.path.insert(0, str(REPO / "workflows"))
from prompts import identify as prompt_id_of, sha256 as prompt_sha  # noqa: E402

RESULTS = REPO / "data" / "results"
ROW_GLOB = "2026-09-2[67]_*.jsonl"
MEASURE_GLOB = "2026-09-2[67]_*.json"
FINDING_FIELDS = ["finding_id", "author", "date", "kind", "text", "render_ids", "scope", "model_family",
                  "prediction_id", "verdict", "record", "supersedes", "basis", "tags"]
KINDS = {"owner_read", "measured", "prediction", "verdict", "retraction"}
VERDICTS = {"held", "falsified", "not_supported", "open", None}
TAGS = {"grade", "haze", "blacks", "saturation", "warmth", "detail", "motion", "adherence", "clone", "audio",
        "speed", "weights", "int8", "encode", "method", "bug", "prompt", "vae", "schedule", "attention"}
FAMILIES = {"base", "pdd", "flashgen", "fasth3", "hybrid", "route", "turbo", "decode_only", "all"}
#: The follow-up batch's first launch, before the 0.154.8 fix: its PDD and
#: FlashGen rows ran on a model still wrapped by earlier LoRA branches
#: (`2026-09-26_followup_contamination.md`). The rerun began 21:27.
CONTAMINATED = {"record": "2026-09-26_followup.jsonl", "before": "2026-09-26T21:27",
                "arms": ("pdd8", "flashgen")}
#: The swap runner and fastdude's transplant runner overlapped from about
#: 01:25 on 2026-09-27 (the swap record's timing note).
INTERLEAVED = {"2026-09-26_fasth3_swap.jsonl": "2026-09-27T01:25",
               "2026-09-26_flashgen_transplant.jsonl": "2026-09-27T01:25"}

_version_cache: dict = {}


def changelog_version(commit: str | None):
    if not commit:
        return None
    if commit not in _version_cache:
        txt = subprocess.run(["git", "show", f"{commit}:CHANGELOG.md"], cwd=REPO,
                             capture_output=True, text=True).stdout
        m = re.search(r"^## (\d+\.\d+\.\d+)", txt, re.M)
        _version_cache[commit] = m.group(1) if m else None
    return _version_cache[commit]


def family(d: dict) -> str:
    unet = " ".join(d.get("unet", []))
    loras = " ".join(l.get("file", "") for l in d.get("loras", []))
    if not unet:
        return "decode_only"
    if "adaln" in unet or "temb" in unet:
        return "hybrid"
    if "pdd" in loras and "flashgen" in loras:
        return "route"
    if "turbo" in loras:
        return "turbo"
    if "fasth3" in unet:
        return "fasth3"
    if "pdd" in loras:
        return "pdd"
    if "flashgen" in loras:
        return "flashgen"
    return "base"


def renders(root: Path, substrate: dict) -> list[dict]:
    out = []
    for f in sorted(RESULTS.glob(ROW_GLOB)):
        seen: dict = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            label = r["label"]
            seen[label] = seen.get(label, 0) + 1
            rid = f"{f.stem}:{label}:{seen[label]}"
            g, note = graph_at(r["graph"], r.get("graph_sha256", ""))
            patched = apply_patches(g, r.get("patches", [])) if g else {}
            d = describe(patched) if g else {}
            pat = {p.get("field"): p.get("value") for p in r.get("patches", [])}
            ptext = pat.get("MiniMaxH3Conditioning.prompt")
            if ptext is None and patched:
                ptext = next((n["inputs"].get("prompt") for n in patched.values() if isinstance(n, dict)
                              and n.get("class_type") == "MiniMaxH3Conditioning"), None)
            length = pat.get("MiniMaxH3Resolution.length")
            scene, _, arm = label.partition("__")
            sub = r.get("substrate") or {}
            commit = sub.get("git_commit")
            ts = r.get("ts") or ""
            contaminated = (f.name == CONTAMINATED["record"] and ts < CONTAMINATED["before"]
                            and any(arm.startswith(a) or label.startswith("warmup_" + a) for a in CONTAMINATED["arms"]))
            interleaved = f.name in INTERLEAVED and ts >= INTERLEAVED[f.name]
            sp = substrate.get(r.get("prompt_id"))
            fam = family(d)
            out.append({
                "render_id": rid, "record": f.name, "ts": ts, "label": label,
                "scene": scene if arm else None, "arm": arm or label,
                "seed": r.get("seed"), "length": length if isinstance(length, int) else None,
                "prompt_bank_id": prompt_id_of(ptext) if isinstance(ptext, str) else None,
                "prompt_sha256": prompt_sha(ptext) if isinstance(ptext, str) else None,
                "model_family": fam, "graph": Path(r["graph"]).name, "graph_source": note,
                "unet": (d.get("unet") or [None])[0], "loras": d.get("loras", []), "vae": d.get("vae", []),
                "sampler": d.get("sampler", []), "schedule": d.get("schedule", []), "shift": d.get("shift", []),
                "attention": d.get("attention", []), "x0_observer": d.get("x0_observer"),
                "saves_latents": d.get("saves_latents"),
                "git_commit": commit, "git_dirty": sub.get("git_dirty"),
                "code_version": changelog_version(commit),
                "gpu": sub.get("gpu"), "torch": sub.get("torch"), "comfy_kitchen": sub.get("comfy_kitchen"),
                "warmup": bool(r.get("warmup")), "error": r.get("error"),
                "total_s": r.get("total_s"), "sampler_s": r.get("sampler_s"), "decode_s": r.get("decode_s"),
                "wall_s": r.get("wall_s"),
                "per_node_s": [{"node_id": str(k), "class_type": (patched.get(str(k)) or {}).get("class_type"),
                                "seconds": v} for k, v in (r.get("per_node_s") or {}).items()],
                "suspect_cache_hit": r.get("suspect_cache_hit"),
                "session_position": sp and sp.get("position_in_session"),
                "cache_hit_fraction": sp and sp.get("cache_hit_fraction"),
                "contaminated": contaminated, "interleaved": interleaved,
                "valid_look": not contaminated and not r.get("error"),
                "valid_timing": not (contaminated or interleaved or r.get("warmup") or r.get("error")
                                     or r.get("suspect_cache_hit")),
                "prompt_id": r.get("prompt_id"),
                "outputs": outputs(root, label, ts, graph_prefixes(patched)),
            })
    # A file two rows both matched (a label rendered twice whose second render
    # left no files of its own) goes to the row that finished nearest to it.
    import datetime as _dt
    owners: dict = {}
    for r in out:
        t = _dt.datetime.fromisoformat(r["ts"]).timestamp() if r["ts"] else None
        for kind in ("video", "latents"):
            folder = root / ("Video" if kind == "video" else "latents")
            for name in r["outputs"][kind]:
                gap = abs((folder / name).stat().st_mtime - t) if t else float("inf")
                if name not in owners or gap < owners[name][0]:
                    owners[name] = (gap, r["render_id"])
    for r in out:
        for kind in ("video", "latents"):
            r["outputs"][kind] = [n for n in r["outputs"][kind] if owners[n][1] == r["render_id"]]
    return out


def measures(rows: list[dict]) -> list[dict]:
    by_clip = {}
    for r in rows:
        for c in r["outputs"]["video"]:
            by_clip.setdefault(c, r["render_id"])
    agg: dict = {}

    def add(clip, tool, record, metric, value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        key = (clip, tool, metric)
        if key not in agg:
            agg[key] = {"render_id": by_clip.get(clip), "clip": clip, "tool": tool, "metric": metric,
                        "value": float(value), "records": [record]}
        elif record not in agg[key]["records"]:
            agg[key]["records"].append(record)

    for f in sorted(RESULTS.glob(MEASURE_GLOB)):
        try:
            d = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        tool = Path(str(d.get("tool") or d.get("measured_by") or "")).name
        rws, clips, scenes = d.get("rows"), d.get("clips"), d.get("scenes")
        if isinstance(rws, list) and rws and isinstance(rws[0], dict) and "clip" in rws[0]:
            for r in rws:
                if r.get("part", "all") != "all":
                    continue
                for k, v in r.items():
                    add(r["clip"], tool, f.name, k, v)
        elif isinstance(clips, dict):
            for c, v in clips.items():
                for k, x in (v.get("median") or {}).items():
                    add(c, tool, f.name, k, x)
                for k in ("frames", "pairs"):
                    add(c, tool, f.name, k, v.get(k))
        elif isinstance(scenes, dict) and "arms" in d:
            for per in scenes.values():
                for m in per.values():
                    for k, v in m.items():
                        add(m.get("clip", ""), tool, f.name, k, v)
    return list(agg.values())


def models(rows: list[dict]) -> list[dict]:
    mdir = REPO.parents[1] / "models"
    used = {}
    for r in rows:
        if r["unet"]:
            used.setdefault(("diffusion_model", r["unet"]), 0)
            used[("diffusion_model", r["unet"])] += 1
        for l in r["loras"]:
            used.setdefault(("lora", l.get("file")), 0)
            used[("lora", l.get("file"))] += 1
        for v in r["vae"]:
            used.setdefault(("vae", v), 0)
            used[("vae", v)] += 1
    built = [("diffusion_model", "fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a05.safetensors"),
             ("diffusion_model", "fastvideo_fasth3_8step_v2_pruned_int8_convrot_temb_a075.safetensors"),
             ("lora", "h3/minimax_h3_fl2va_fasth3temb_pdd_8step_comfy.safetensors")]
    for k in built:
        used.setdefault(k, 0)
    info = json.loads((HERE / "render_dataset_models.json").read_text())
    # Files the run rows do not name (the text encoders: no row records one)
    # are listed from their description when it carries a `kind`.
    for base, meta in info.items():
        if isinstance(meta, dict) and meta.get("kind"):
            used.setdefault((meta["kind"], base + ".safetensors"), 0)
    out = []
    for (kind, name), n in sorted(used.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        sub = {"diffusion_model": "diffusion_models", "lora": "loras", "vae": "vae",
               "text_encoder": "text_encoders"}[kind]
        meta = info.get(Path(name).name.removesuffix(".safetensors"), {})
        # A file moved since it rendered (2026-09-27, research files into
        # h3_research/ and loras/h3/research/) is found at its new name.
        p = mdir / sub / meta.get("renamed_to", name)
        meta = {k: v for k, v in meta.items() if k != "kind"}
        # No run row records its encoder, so a text encoder has no count.
        out.append({"file": name, "kind": kind, "renders": None if kind == "text_encoder" else n,
                    "size_bytes": p.stat().st_size if p.exists() else None, **meta})
    return out


def findings(out: Path) -> list[dict]:
    rows = []
    for f in sorted(out.glob("findings_*.jsonl")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            bad = [k for k in FINDING_FIELDS if k not in r]
            bad += [f"kind={r.get('kind')}"] if r.get("kind") not in KINDS else []
            bad += [f"verdict={r.get('verdict')}"] if r.get("verdict") not in VERDICTS else []
            bad += [f"tag={t}" for t in r.get("tags", []) if t not in TAGS]
            bad += [f"model_family={r.get('model_family')}"] if r.get("model_family") not in FAMILIES else []
            if bad:
                raise SystemExit(f"{f.name}:{i}: {bad}")
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = comfy_output()
    sub = json.loads((RESULTS / "2026-09-27_followup_substrate.json").read_text())
    substrate = {r["prompt_id"]: r for r in sub["session_rows"] if r.get("prompt_id")}
    rs = renders(root, substrate)
    ms = measures(rs)
    tables = {"renders": rs, "measures": ms, "models": models(rs), "findings": findings(args.out)}
    for name, rows in tables.items():
        (args.out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    unmatched = sum(1 for m in ms if m["render_id"] is None)
    print(json.dumps({k: len(v) for k, v in tables.items()} | {"measures_without_render": unmatched}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
