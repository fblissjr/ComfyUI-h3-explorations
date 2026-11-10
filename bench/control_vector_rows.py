#!/usr/bin/env python3
"""Cache the text rows H3 sees for a set of real prompts, once, for CPU-side study.

What it does: encodes each prompt through the pack's 32B encoder and both
refiner families (the build's own helpers and its default recipe), and writes
every token's encoder row and post-refiner rows with its token id, its vision
tag and whether the `scene_only` mask covers it. Sources are the pair files
under `prompt_bank/control_vectors/` (both sides of every pair) and the
`prompt_bank/` entries of the modes named by `--bank-modes`.

Why: the measurements the control-vector plan asks for (a held-out pair score,
pooling compared three ways, what real prompts vary along, a word-presence
dictionary, the J-Lens sample) all read the same rows. This makes them once.

What it cannot see: reference pictures. A prompt is encoded as text alone, so
bank modes that need media are left out by default.

Output: one `.safetensors` shard per source under `--out`, plus `index.json`
naming every prompt, its source, its side and its row range, with the recipe,
the code commit and the date. Nothing is written under the repo's tracked
folders; `--out` belongs under the untracked `data/`.

    <comfy-venv-python> bench/control_vector_rows.py --out data/control_vectors/rows --limit 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from bench._lib import bootstrap  # noqa: E402

bootstrap()

import folder_paths  # noqa: E402
import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import control_vector_nodes as cv_nodes  # noqa: E402
import h3_spans  # noqa: E402
from bench.control_vector import helpers as _h  # noqa: E402
from workflows import h3_config  # noqa: E402

FORMAT = "h3-text-rows/1"


def _sources(pairs_dir: Path, bank_modes: tuple[str, ...]) -> list[dict]:
    """Every prompt to encode, as `{source, name, side, text}` in a fixed order."""
    out: list[dict] = []
    for toml_path in sorted(pairs_dir.glob("*.toml")):
        doc = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        for i, pair in enumerate(doc.get("pair", [])):
            for side in ("a", "b"):
                out.append({"source": f"pairs:{toml_path.stem}", "name": f"{toml_path.stem}/pair{i}",
                            "side": side, "text": pair[side]})
    if bank_modes:
        bank = json.loads((REPO_ROOT / "prompt_bank" / "bank.json").read_text(encoding="utf-8"))
        for entry in bank.get("prompts", []):
            if entry.get("mode") not in bank_modes:
                continue
            path = REPO_ROOT / "prompt_bank" / f"{entry['id']}.txt"
            if not path.is_file():
                raise SystemExit(f"bank entry {entry['id']!r} has no file at prompt_bank/{entry['id']}.txt")
            out.append({"source": f"bank:{entry['mode']}", "name": entry["id"], "side": "",
                        "text": path.read_text(encoding="utf-8").strip()})
    return out


def _runner(family: str, device: str, no_lora: bool):
    dit_name = h3_config.MODELS["unet_ref2va" if family == "ref2va" else "unet_fl2va"]
    dit_path = Path(folder_paths.get_full_path_or_raise("diffusion_models", dit_name))
    lora_name = h3_config.PIKA_REF2VA_SVD995_LORA if family == "ref2va" else h3_config.PIKA_FL2VA_SVD995_LORA
    # A LoRA that does not resolve is a refusal here: the recipe is part of what the rows mean.
    loras = [] if no_lora else _h._parse_lora_specs([f"{lora_name}:1.0"])
    run = cv_nodes.build_refiner_runner(dit_path, device=device, loras=loras)
    return run, {"checkpoint": dit_name, "loras": [] if no_lora else [f"{lora_name}:1.0"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder, under the untracked data/")
    ap.add_argument("--pairs-dir", default=str(REPO_ROOT / "prompt_bank" / "control_vectors"))
    ap.add_argument("--bank-modes", default="t2va",
                    help="comma-separated prompt_bank modes to add as text-only prompts; empty for none")
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N prompts (a smoke run)")
    ap.add_argument("--device", default="default", help="encoder device")
    ap.add_argument("--refiner-device", default="cpu")
    ap.add_argument("--no-lora", action="store_true", help="refine without the Pika LoRA")
    ap.add_argument("--resume", action="store_true",
                    help="go on in --out from its index.json, skipping sources already written")
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = tuple(m.strip() for m in args.bank_modes.split(",") if m.strip())
    items = _sources(Path(args.pairs_dir), modes)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no prompts to encode")

    encoder_name = h3_config.MODELS["clip"]
    clip = _h._load_clip(encoder_name, device=args.device)
    runners = {}
    recipe = {"encoder": encoder_name}
    for family in ("fl2va", "ref2va"):
        runners[family], recipe[family] = _runner(family, args.refiner_device, args.no_lora)

    try:
        commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"

    index = {"format": FORMAT, "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "commit": commit, "recipe": recipe, "bank_modes": list(modes), "shards": {}, "prompts": []}
    by_source: dict[str, list[dict]] = {}
    for it in items:
        by_source.setdefault(it["source"], []).append(it)

    index_path = out_dir / "index.json"
    if args.resume and index_path.is_file():
        # Named explicitly: a run handed the same folder and told to go on. A recipe or code change refuses.
        prior = json.loads(index_path.read_text(encoding="utf-8"))
        if (prior.get("format"), prior.get("recipe"), prior.get("commit")) != (FORMAT, recipe, commit):
            raise SystemExit("--resume: the folder's index was made with another format, recipe or commit")
        index["prompts"], index["shards"], index["built"] = prior["prompts"], prior["shards"], prior["built"]

    t0 = time.perf_counter()
    done = len(index["prompts"])
    for source, group in by_source.items():
        if source in index["shards"] and (out_dir / index["shards"][source]).is_file():
            continue
        enc, pr = [], {"fl2va": [], "ref2va": []}
        ids, tags_all, scene, owner = [], [], [], []
        for it in group:
            enc_rows, tags, token_ids, _decode = cv_nodes.encode_prompt_for_vector(clip, it["text"])
            span_masks = getattr(tags, "_h3_span_masks", None)
            if span_masks is None:
                raise SystemExit(f"{it['name']}: the encode returned no span masks")
            scene_mask = h3_spans.inference_mask(span_masks, "scene_only")
            n = int(enc_rows.shape[0])
            if not (len(token_ids) == n == int(tags.numel()) == int(scene_mask.numel())):
                raise SystemExit(f"{it['name']}: rows, token ids, tags and mask disagree in length")
            for family, run in runners.items():
                pr[family].append(run(enc_rows).detach().to(device="cpu", dtype=torch.float16))
            enc.append(enc_rows.to(dtype=torch.float16))
            ids.append(torch.tensor(token_ids, dtype=torch.int32))
            tags_all.append(tags.to(dtype=torch.uint8))
            scene.append(scene_mask.to(device="cpu", dtype=torch.uint8))
            owner.append(torch.full((n,), len(index["prompts"]), dtype=torch.int32))
            index["prompts"].append({"name": it["name"], "source": source, "side": it["side"], "rows": n,
                                     "sha256": hashlib.sha256(it["text"].encode("utf-8")).hexdigest()})
            done += 1
        shard = f"{source.replace(':', '__')}.safetensors"
        save_file({"enc": torch.cat(enc), "pr_fl2va": torch.cat(pr["fl2va"]), "pr_ref2va": torch.cat(pr["ref2va"]),
                   "token_ids": torch.cat(ids), "tags": torch.cat(tags_all), "scene_only": torch.cat(scene),
                   "prompt": torch.cat(owner)}, str(out_dir / shard), metadata={"format": FORMAT, "source": source})
        index["shards"][source] = shard
        (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"{source}: {len(group)} prompt(s); {done}/{len(items)} done in {time.perf_counter() - t0:.0f}s",
              flush=True)
    print(f"wrote {len(index['shards'])} shard(s) and index.json under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
