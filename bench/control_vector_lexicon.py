#!/usr/bin/env python3
"""Label each built control vector two ways and print them side by side: by J-Lens, and by a word table
made from cached rows.

What it does: reads the row cache of `bench/control_vector_rows.py`, builds for every token the mean
post-refiner row it has across real prompts (scene rows only) less the mean of all scene rows, and scores a
vector against that table by cosine. For each vector on disk it prints the top tokens from that table beside
the top words `j_lens.decode_vector` gives.

The table used for a concept leaves out that concept's own pair file. A vector is made from its file's rows,
so a table that held them would name the steered clause's words for a reason that says nothing about the
method.

What it cannot see: whether a label is right. It prints both labels; a person reads them. A token seen in
fewer prompts than `--min-prompts` is left out, so a rare word that would be the best label can be missing.
CPU only.

    <comfy-venv-python> bench/control_vector_lexicon.py --rows data/control_vectors/rows --family fl2va
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

import j_lens  # noqa: E402


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def load_rows_reader(rows_dir, family: str, tokenizer_dir, min_prompts: int = 3):
    """The rows table as a reader: `read(vec, concept, top)` gives the top tokens toward and away from `vec`.

    `concept` names the pair file to leave out of the table (`pairs:<concept>` in the cache), so a vector is
    never labelled from the rows it was made of. Returns `(read, info)`; `info` says what the table was built
    from, and `info["sources"]` is the set of sources in the cache. Shared with
    `bench/control_vector_readout_score.py`, so the two tools cannot drift apart.
    """
    rows_dir = Path(rows_dir)
    index = json.loads((rows_dir / "index.json").read_text(encoding="utf-8"))
    tok = _tokenizer(str(tokenizer_dir))
    key = f"pr_{family}"

    # Per source: for each token id, the sum of its scene rows, how many rows, and in how many prompts.
    per_source: dict[str, dict] = {}
    for source, shard in index["shards"].items():
        with safe_open(str(rows_dir / shard), "pt") as f:
            rows = f.get_tensor(key).float()
            ids = f.get_tensor("token_ids").long()
            keep = f.get_tensor("scene_only").bool() & f.get_tensor("tags").bool()
            owner = f.get_tensor("prompt").long()
        rows, ids, owner = rows[keep], ids[keep], owner[keep]
        uniq, inv = torch.unique(ids, return_inverse=True)
        sums = torch.zeros(len(uniq), rows.shape[1]).index_add_(0, inv, rows)
        counts = torch.zeros(len(uniq)).index_add_(0, inv, torch.ones(len(ids)))
        prompts = torch.zeros(len(uniq))
        seen = torch.unique(torch.stack([inv, owner], dim=1), dim=0)[:, 0]
        prompts.index_add_(0, seen, torch.ones(len(seen)))
        per_source[source] = {"ids": uniq, "sums": sums, "counts": counts, "prompts": prompts,
                              "total": rows.sum(0), "n": float(len(rows))}
    all_ids = torch.unique(torch.cat([s["ids"] for s in per_source.values()]))
    pos = {int(t): i for i, t in enumerate(all_ids.tolist())}
    width = next(iter(per_source.values()))["sums"].shape[1]

    def table(leave_out: str | None):
        sums = torch.zeros(len(all_ids), width)
        counts = torch.zeros(len(all_ids))
        prompts = torch.zeros(len(all_ids))
        total, n = torch.zeros(width), 0.0
        for source, s in per_source.items():
            if source == leave_out:
                continue
            at = torch.tensor([pos[int(t)] for t in s["ids"].tolist()])
            sums.index_add_(0, at, s["sums"])
            counts.index_add_(0, at, s["counts"])
            prompts.index_add_(0, at, s["prompts"])
            total += s["total"]
            n += s["n"]
        ok = prompts >= min_prompts
        centered = sums[ok] / counts[ok, None] - total / n
        return all_ids[ok], torch.nn.functional.normalize(centered, dim=1)

    def piece(token_id: int) -> str:
        return tok.decode([token_id]).strip() or repr(tok.decode([token_id]))

    def read(vec, concept: str, top: int):
        ids, tab = table(f"pairs:{concept}")
        score = tab @ torch.nn.functional.normalize(vec, dim=0)
        up = [piece(int(ids[i])) for i in torch.topk(score, top).indices.tolist()]
        down = [piece(int(ids[i])) for i in torch.topk(-score, top).indices.tolist()]
        return up, down

    info = {"sources": set(per_source), "n_sources": len(index["shards"]), "built": index["built"],
            "commit": index["commit"], "n_tokens": len(all_ids), "min_prompts": min_prompts}
    return read, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="a folder written by bench/control_vector_rows.py")
    ap.add_argument("--vectors", default=str(REPO_ROOT.parent.parent / "models" / "control_vectors"))
    ap.add_argument("--family", default="fl2va", choices=("fl2va", "ref2va"))
    ap.add_argument("--tokenizer", required=True, help="folder of the Qwen3-VL tokenizer files")
    ap.add_argument("--min-prompts", type=int, default=3, help="a token must occur in at least this many prompts")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated concept names; default all")
    args = ap.parse_args()

    read, info = load_rows_reader(args.rows, args.family, args.tokenizer, args.min_prompts)

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    files = sorted(glob.glob(os.path.join(args.vectors, f"*__{args.family}__diff_span__lora_*.safetensors")))
    print(f"rows: {info['n_sources']} source(s), built {info['built']} at {info['commit']}; "
          f"{info['n_tokens']} distinct scene tokens; family {args.family}; a token needs {args.min_prompts}+ prompts")
    for path in files:
        name = os.path.basename(path).split("__")[0]
        if only and name not in only:
            continue
        with safe_open(path, "pt") as f:
            vec = f.get_tensor("vector.post_refiner").float()
        up, down = read(vec, name, args.top)
        try:
            jl = j_lens.decode_vector(vec, space="post_refiner", family=args.family, top_k=args.top)
            jl_up = [w["word"] for w in jl["positive_words"]]
            jl_down = [w["word"] for w in jl["negative_words"]]
        except Exception as exc:  # the bundle is optional; say so and go on
            jl_up, jl_down = [f"(j_lens unavailable: {exc})"], []
        held = "its own pair file left out" if f"pairs:{name}" in info["sources"] else "no pair file of this name in the rows"
        print(f"\n{name}  ({held})")
        print(f"  rows table  +: {', '.join(up)}")
        print(f"  rows table  -: {', '.join(down)}")
        print(f"  j_lens      +: {', '.join(jl_up)}")
        print(f"  j_lens      -: {', '.join(jl_down)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
