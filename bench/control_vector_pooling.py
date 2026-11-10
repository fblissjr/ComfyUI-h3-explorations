#!/usr/bin/env python3
"""Compare ways of pooling a pair's rows into a direction, from the row cache, with no encoding.

What it does: for every pair file in a cache written by `bench/control_vector_rows.py`, aligns each pair's
two token sequences and pools the post-refiner rows four ways:

    diff_span        the changed rows of B less the changed rows of A (the build's default)
    from_first_diff  every row from the first change to the end, B less A
    scene_text       every scene row, B less A
    unchanged        the rows whose tokens did not change, B less A, row by aligned row

Only rows the `scene_only` mask covers are pooled. For each way it prints, over the pair files:

    held-out   build the direction from all pairs but one, take its cosine with the pair left out, average
    less shared  the same, after projecting the other concepts' mean direction out of the pairs
    lean       the cosine between a concept's direction and the mean direction of the other concepts
    shared     the mean cosine between the directions of different concepts
    to built   the cosine with the vector on disk for that concept (a check that the cache matches the build)
    norm       the length of the pooled direction
    to span    the cosine with the diff_span direction

What it cannot see: what a render does. A high held-out score says the pairs agree with each other, which a
component shared by every pair also produces; read it beside the shared column. CPU only.

    <comfy-venv-python> bench/control_vector_pooling.py --rows data/control_vectors/rows --family fl2va
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import statistics
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
WAYS = ("diff_span", "from_first_diff", "scene_text", "unchanged")


def _pair_directions(rows_a, rows_b, ids_a, ids_b, scene_a, scene_b) -> dict[str, torch.Tensor] | None:
    sm = difflib.SequenceMatcher(None, ids_a, ids_b, autojunk=False)
    ops = sm.get_opcodes()
    changed_a = torch.zeros(len(ids_a), dtype=torch.bool)
    changed_b = torch.zeros(len(ids_b), dtype=torch.bool)
    same_a, same_b = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            same_a.extend(range(i1, i2))
            same_b.extend(range(j1, j2))
        else:
            changed_a[i1:i2] = True
            changed_b[j1:j2] = True
    if not changed_a.any() or not changed_b.any():
        return None  # an insertion or deletion on one side only has no span to pool on the other
    first_a = int(changed_a.nonzero()[0])
    first_b = int(changed_b.nonzero()[0])
    after_a = torch.arange(len(ids_a)) >= first_a
    after_b = torch.arange(len(ids_b)) >= first_b

    def mean(rows, mask):
        return rows[mask].mean(0) if bool(mask.any()) else None

    out: dict[str, torch.Tensor] = {}
    pools = {
        "diff_span": (changed_a & scene_a, changed_b & scene_b),
        "from_first_diff": (after_a & scene_a, after_b & scene_b),
        "scene_text": (scene_a, scene_b),
    }
    for way, (ma, mb) in pools.items():
        a, b = mean(rows_a, ma), mean(rows_b, mb)
        if a is None or b is None:
            return None
        out[way] = b - a
    ia, ib = torch.tensor(same_a), torch.tensor(same_b)
    keep = scene_a[ia] & scene_b[ib]
    if not bool(keep.any()):
        return None
    out["unchanged"] = (rows_b[ib[keep]] - rows_a[ia[keep]]).mean(0)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--family", default="fl2va", choices=("fl2va", "ref2va"))
    ap.add_argument("--vectors", default=str(REPO_ROOT.parent.parent / "models" / "control_vectors"))
    ap.add_argument("--json", default=None, help="also write every per-concept figure here")
    args = ap.parse_args()

    rows_dir = Path(args.rows)
    index = json.loads((rows_dir / "index.json").read_text(encoding="utf-8"))
    by_source: dict[str, list[tuple[int, dict]]] = {}
    for i, p in enumerate(index["prompts"]):
        by_source.setdefault(p["source"], []).append((i, p))

    per_concept: dict[str, dict] = {}
    for source, shard in index["shards"].items():
        if not source.startswith("pairs:"):
            continue
        name = source.split(":", 1)[1]
        with safe_open(str(rows_dir / shard), "pt") as f:
            rows = f.get_tensor(f"pr_{args.family}").double()
            ids = f.get_tensor("token_ids").tolist()
            scene = f.get_tensor("scene_only").bool() & f.get_tensor("tags").bool()
            owner = f.get_tensor("prompt")
        diffs: dict[str, list[torch.Tensor]] = {w: [] for w in WAYS}
        sides: dict[str, dict[str, torch.Tensor]] = {}
        for i, p in by_source[source]:
            sides.setdefault(p["name"], {})[p["side"]] = (owner == i).nonzero().flatten()
        skipped = 0
        for pair in sides.values():
            ra, rb = pair["a"], pair["b"]
            d = _pair_directions(rows[ra], rows[rb], [ids[k] for k in ra.tolist()], [ids[k] for k in rb.tolist()],
                                 scene[ra], scene[rb])
            if d is None:
                skipped += 1
                continue
            for w in WAYS:
                diffs[w].append(d[w])
        if len(diffs["diff_span"]) < 3:
            continue
        entry: dict = {"pairs": len(diffs["diff_span"]), "skipped": skipped}
        built = sorted(glob.glob(os.path.join(args.vectors, f"{name}__{args.family}__diff_span__lora_*.safetensors")))
        built_vec = None
        if built:
            with safe_open(built[0], "pt") as f:
                built_vec = f.get_tensor("vector.post_refiner").double()
        for w in WAYS:
            m = torch.stack(diffs[w])
            mean = m.mean(0)
            held = [_cos((m.sum(0) - m[k]) / (len(m) - 1), m[k]) for k in range(len(m))]
            entry[w] = {"vector": mean, "diffs": m, "held_out": statistics.mean(held), "norm": float(mean.norm()),
                        "to_built": None if built_vec is None else _cos(mean, built_vec)}
        for w in WAYS:
            entry[w]["to_span"] = _cos(entry[w]["vector"], entry["diff_span"]["vector"])
        per_concept[name] = entry

    names = sorted(per_concept)
    # The same held-out score with the part every other concept shares taken out first: for each concept,
    # the mean direction of the other concepts is projected out of its pairs before they are compared.
    for w in WAYS:
        unit = {n: torch.nn.functional.normalize(per_concept[n][w]["vector"], dim=0) for n in names}
        for n in names:
            others = torch.nn.functional.normalize(torch.stack([unit[o] for o in names if o != n]).mean(0), dim=0)
            m = per_concept[n][w]["diffs"]
            m = m - (m @ others)[:, None] * others[None, :]
            held = [_cos((m.sum(0) - m[k]) / (len(m) - 1), m[k]) for k in range(len(m))]
            per_concept[n][w]["held_out_less_shared"] = statistics.mean(held)
            per_concept[n][w]["lean_to_others"] = float(unit[n] @ others)
    print(f"rows built {index['built']} at {index['commit']}; family {args.family}; {len(names)} pair file(s), "
          f"{sum(per_concept[n]['pairs'] for n in names)} pair(s) pooled, "
          f"{sum(per_concept[n]['skipped'] for n in names)} skipped")
    print(f"{'way':16s} {'held-out':>9s} {'less shared':>12s} {'shared':>8s} {'lean':>6s} {'to built':>9s} {'norm':>8s} {'to span':>8s}"
          "   (medians over pair files; shared is a mean over concept pairs; lean is a concept's cosine with the others' mean)")
    for w in WAYS:
        v = torch.nn.functional.normalize(torch.stack([per_concept[n][w]["vector"] for n in names]), dim=1)
        g = v @ v.T
        iu = torch.triu_indices(len(names), len(names), 1)
        shared = float(g[iu[0], iu[1]].mean())
        med = lambda key: statistics.median(per_concept[n][w][key] for n in names if per_concept[n][w][key] is not None)
        print(f"{w:16s} {med('held_out'):9.3f} {med('held_out_less_shared'):12.3f} {shared:8.3f} {med('lean_to_others'):6.3f} "
              f"{med('to_built'):9.3f} {med('norm'):8.2f} {med('to_span'):8.3f}")
    if args.json:
        slim = {n: {"pairs": e["pairs"], "skipped": e["skipped"],
                    **{w: {k: v for k, v in e[w].items() if k not in ("vector", "diffs")} for w in WAYS}}
                for n, e in per_concept.items()}
        Path(args.json).write_text(json.dumps(slim, indent=1), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
