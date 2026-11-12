#!/usr/bin/env python3
"""Score control-vector readouts blind: can a judge who sees only a readout pick the vector's concept?

A readout is the short word list a reader gives for a control vector: `j_lens.decode_vector`'s words, or the
rows table of `bench/control_vector_lexicon.py`. This tool turns the built vectors into judge packets and
counts what the judges answered. The judging format and the controls come from WorkspaceBench; the option
design is the third one tried, and `bench/results/2026-11-12_control_vector_readout_blind_score.md` says why
the first two were dropped.

    <comfy-venv-python> bench/control_vector_readout_score.py packets --out data/control_vectors/<run> \\
        --rows data/control_vectors/<rows cache> --tokenizer <comfy>/comfy/text_encoders/qwen25_tokenizer
    <comfy-venv-python> bench/control_vector_readout_score.py score data/control_vectors/<run>

**The judging is not done here.** `packets` writes `packets/packet_<n>.json` and two instruction files. A
person hands each packet and its instructions to a judge (any capable model, one packet per judge) and saves
each judge's answers as `answers/<pass>/packet_<n>.json`. `score` reads every pass it finds. Give a judge the
packet and the instructions only, never the run folder: `key.json` beside them holds the answers.

How a run is built:

- **Groups.** The vectors are split into groups of `GROUP_SIZE` near concepts by a greedy rule on the cosine
  of `vector.post_refiner`: the closest unassigned pair seeds a group, and the unassigned vector with the
  highest mean cosine to the group joins until it is full. What is left over at the end is listed and not
  scored. A vector whose name extends another's (`film_16mm_1974_neutral`) is a variant of that concept and
  is left out.
- **Options.** Every member of a group is judged against the same `quality` lines, one per member, in an
  order drawn per item. Each line is the target for exactly one member, so a list says nothing about which
  member is asked for; a judge with no readout lands on one hit per group whatever it does consistently.
- **Items**, per vector and reader:
    `real`     the vector's own readout
    `twin`     the readout of a random direction of the same norm; the right answer is "cannot tell"
    `shuffle`  the readout of the nearest vector outside the group
  and one `options_only` item per vector, the list with no readout, in a packet of its own.
- **Packets.** Each readout packet holds one item per vector, so the variants of one vector's readout are
  never judged side by side.

What `score` prints, per pass and reader: how many `real` items went to the target, to "cannot tell" and to
another option; how many `twin` and `shuffle` items went to the target; how many target picks carried a quote
found in the readout; the same by group; and, with two passes or more, how many items got the same pick in
each pair of passes. It prints counts and draws no conclusion.

What it cannot see:

- whether a render shows the concept: it scores naming, not effect;
- whether a miss is the reader's or the vector's: two vectors that overlap read alike, and a faithful reader
  will be unsure between them, so a count describes the vector and the reader together;
- the judge: which model it was, whether it kept to one item at a time (a group's list comes back once per
  member in a packet), and whether it was given only the packet;
- a reader it was not handed: `--rows` adds the rows table, and without it only J-Lens is read.

CPU only. `packets` needs the J-Lens bundle and the built vectors; `score` needs neither.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import subprocess
import sys
import tomllib
from collections import Counter
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
COMFY_ROOT = REPO_ROOT.parent.parent

FORMAT = "h3-readout-score/1"
GROUP_SIZE = 5  # reasoned, not measured: WorkspaceBench's five options, and small enough for tight groups
LETTERS = "ABCDE"
CANNOT_TELL = "F"
VARIANTS = ("real", "twin", "shuffle")
DEFAULT_TOP = 8  # words per pole, the lexicon tool's default
DEFAULT_SEED = 20261112

JUDGE_INSTRUCTIONS = """# Judge instructions (readout items)

Each item is a readout that an interpretability tool produced from one steering direction of a
video-and-audio generation model. `plus_words` are words the direction pushes toward; `minus_words` are words
it pushes away from. Words can be fragments of words.

Each item has five options, A to E. Each option describes a concept, usually as "X against Y": the direction
pushes toward X and away from Y.

For each item, decide which option the readout describes.

- Answer `F` (cannot tell) when the words do not single out one option, or fit two options about equally.
- Judge each item only from its own words and its own options. Several items may show the same options;
  that says nothing about any one item's answer.
- Give `quote`: the words, copied exactly from that item's `plus_words` or `minus_words`, that single out your
  pick. An empty list for `F`.

Write one JSON array, one object per item, in the order given:
`[{"id": "...", "pick": "B", "quote": ["word", "word"]}, ...]`
"""

JUDGE_INSTRUCTIONS_OPTIONS_ONLY = """# Judge instructions (options-only items)

Each item has five options, A to E, and nothing else. One of the five is the answer to a question you are not
shown. For each item, pick the option you think is most likely to be the answer. Always pick A to E.

Write one JSON array, one object per item, in the order given:
`[{"id": "...", "pick": "B", "quote": []}, ...]`
"""


# ---- building a run (pure: no model, no file) -------------------------------------------------------------


def variants_of_another(names):
    """Names that extend another name: variants of that concept, left out of a run."""
    return sorted(n for n in names if any(n != m and n.startswith(m) for m in names))


def group_near(names, cos, size: int = GROUP_SIZE):
    """Greedy groups of `size` near concepts. Returns `(groups, leftover)`; `cos(a, b)` is the similarity."""
    free, groups = set(names), []
    while len(free) >= size:
        ordered = sorted(free)
        a, b = max(((x, y) for i, x in enumerate(ordered) for y in ordered[i + 1:]), key=lambda p: cos(*p))
        group = [a, b]
        free -= {a, b}
        while len(group) < size:
            nxt = max(sorted(free), key=lambda n: sum(cos(n, m) for m in group) / len(group))
            group.append(nxt)
            free.remove(nxt)
        groups.append(group)
    return groups, sorted(free)


def item_id(seed: int, name: str, reader: str, variant: str) -> str:
    return hashlib.sha256(f"{seed}:{name}:{reader}:{variant}".encode()).hexdigest()[:10]


def build_run(names, quality, cos, readouts, twins, readers, seed: int = DEFAULT_SEED):
    """Packets and key for a run.

    `readouts[(name, reader)]` and `twins[(name, reader)]` are `(plus_words, minus_words)`. Returns
    `(packets, key_items, groups, left_out, leftover)`; `packets` is a list of item lists, the last one the
    options-only packet.
    """
    left_out = variants_of_another(names)
    kept = [n for n in sorted(names) if n not in left_out]
    groups, leftover = group_near(kept, cos)
    group_of = {n: gi for gi, g in enumerate(groups) for n in g}
    slots = [(r, v) for r in readers for v in VARIANTS]
    packets = [[] for _ in range(len(slots) + 1)]
    key_items = {}
    for vi, name in enumerate(n for n in kept if n in group_of):
        group = groups[group_of[name]]
        other = max((n for n in kept if n not in group), key=lambda n: cos(n, name))
        for si, (reader, variant) in enumerate(slots + [("none", "options_only")]):
            order = list(group)
            random.Random(f"{seed}:{name}:{reader}:{variant}").shuffle(order)
            options = {LETTERS[i]: quality[n] for i, n in enumerate(order)}
            iid = item_id(seed, name, reader, variant)
            key_items[iid] = {
                "vector": name, "group": group_of[name], "gold": LETTERS[order.index(name)],
                "option_vectors": dict(zip(LETTERS, order)), "shuffle_source": other,
                "shuffle_source_cos": round(cos(other, name), 3), "reader": reader, "variant": variant}
            if variant == "options_only":
                packets[-1].append({"id": iid, "options": options})
                continue
            plus, minus = {"real": readouts[(name, reader)], "twin": twins[(name, reader)],
                           "shuffle": readouts[(other, reader)]}[variant]
            packets[(vi + si) % len(slots)].append(
                {"id": iid, "plus_words": list(plus), "minus_words": list(minus), "options": options})
    for pi, packet in enumerate(packets):
        random.Random(f"{seed}:packet:{pi}").shuffle(packet)
    return packets, key_items, groups, left_out, leftover


# ---- scoring (pure) ---------------------------------------------------------------------------------------


def judged(key_items, items, answers):
    """One pass's answers as `{(vector, reader, variant): outcome}`. Raises on an unknown or repeated id."""
    seen, out = set(), {}
    for a in answers:
        iid = a.get("id")
        if iid not in key_items:
            raise ValueError(f"answer for an id that is not in the key: {iid!r}")
        if iid in seen:
            raise ValueError(f"id answered twice in one pass: {iid!r}")
        seen.add(iid)
        k, it = key_items[iid], items[iid]
        words = set(it.get("plus_words", [])) | set(it.get("minus_words", []))
        pick = str(a.get("pick", "")).strip().upper()[:1]
        quote = a.get("quote") or []
        out[(k["vector"], k["reader"], k["variant"])] = {
            "pick": pick, "hit": pick == k["gold"], "cannot_tell": pick == CANNOT_TELL,
            "quote_found": bool(quote) and all(q in words for q in quote)}
    return out


def count_pass(key_items, outcome):
    """Counts for one pass. Only vectors whose three variants were all answered are counted for a reader."""
    readers = sorted({k["reader"] for k in key_items.values() if k["reader"] != "none"})
    vectors = sorted({k["vector"] for k in key_items.values()})
    group_of = {k["vector"]: k["group"] for k in key_items.values()}
    result = {"answered": len(outcome), "expected": len(key_items), "readers": {}}
    floor = [outcome[(v, "none", "options_only")] for v in vectors if (v, "none", "options_only") in outcome]
    result["options_only"] = {"n": len(floor), "target": sum(x["hit"] for x in floor)}
    for reader in readers:
        full = [v for v in vectors if all((v, reader, x) in outcome for x in VARIANTS)]
        real = {v: outcome[(v, reader, "real")] for v in full}
        twin = {v: outcome[(v, reader, "twin")] for v in full}
        shuffle = {v: outcome[(v, reader, "shuffle")] for v in full}
        by_group = Counter(group_of[v] for v in full if real[v]["hit"])
        result["readers"][reader] = {
            "n": len(full),
            "real_target": sum(x["hit"] for x in real.values()),
            "real_cannot_tell": sum(x["cannot_tell"] for x in real.values()),
            "real_other_option": sum(not x["hit"] and not x["cannot_tell"] for x in real.values()),
            "real_target_quote_found": sum(x["hit"] and x["quote_found"] for x in real.values()),
            "real_target_and_twin_not": sum(real[v]["hit"] and not twin[v]["hit"] for v in full),
            "twin_target": sum(x["hit"] for x in twin.values()),
            "twin_cannot_tell": sum(x["cannot_tell"] for x in twin.values()),
            "shuffle_target": sum(x["hit"] for x in shuffle.values()),
            "real_target_by_group": {str(g): by_group.get(g, 0) for g in sorted(set(group_of.values()))},
            "real_not_target": sorted(v for v in full if not real[v]["hit"])}
    return result


def compare_passes(key_items, first, second):
    """How two passes agree, per reader and variant: same pick, and for `real` items the target in both."""
    out = {}
    slots = sorted({(k["reader"], k["variant"]) for k in key_items.values()})
    for reader, variant in slots:
        both = [k for k in first if k[1] == reader and k[2] == variant and k in second]
        row = {"n": len(both), "same_pick": sum(first[k]["pick"] == second[k]["pick"] for k in both)}
        if variant == "real":
            row["target_in_both"] = sum(first[k]["hit"] and second[k]["hit"] for k in both)
            row["target_in_one"] = sum(first[k]["hit"] != second[k]["hit"] for k in both)
            row["target_in_neither"] = sum(not first[k]["hit"] and not second[k]["hit"] for k in both)
        out[f"{reader}/{variant}"] = row
    return out


# ---- files and models -------------------------------------------------------------------------------------


def _commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _load_vectors(vectors_dir: Path, family: str):
    from safetensors import safe_open

    found, missing_pair_file = {}, []
    for path in sorted(glob.glob(str(vectors_dir / f"*__{family}__diff_span__lora_*.safetensors"))):
        name = os.path.basename(path).split("__")[0]
        pair = next((p for p in (REPO_ROOT / "prompt_bank" / "control_vectors" / f"{name}.toml",
                                 REPO_ROOT / "data" / "control_vectors" / f"{name}.toml") if p.is_file()), None)
        if pair is None:
            missing_pair_file.append(name)
            continue
        head = tomllib.loads(pair.read_text(encoding="utf-8"))["vector"]
        with safe_open(path, "pt") as f:
            vec = f.get_tensor("vector.post_refiner").float()
        found[name] = {"vec": vec, "quality": " ".join(str(head["quality"]).split()),
                       "file": os.path.basename(path), "pair_file": str(pair.relative_to(REPO_ROOT))}
    return found, missing_pair_file


def cmd_packets(args) -> int:
    out = Path(args.out)
    if (out / "key.json").exists():
        print(f"{out} already holds a run (key.json). Choose a new folder; a run is not rebuilt in place, "
              "because answers already given would then be scored against other items.")
        return 2

    import torch

    sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
    sys.path.insert(0, str(BENCH_DIR))
    sys.path.insert(0, str(REPO_ROOT))
    import j_lens

    vecs, missing_pair_file = _load_vectors(Path(args.vectors), args.family)
    if not vecs:
        print(f"no *__{args.family}__diff_span__lora_*.safetensors with a pair file under {args.vectors}")
        return 2
    names = sorted(vecs)
    unit = {n: torch.nn.functional.normalize(vecs[n]["vec"], dim=0) for n in names}

    def cos(a, b):
        return float(unit[a] @ unit[b])

    def jlens_read(vec, concept):  # J-Lens has nothing to leave out; `concept` is the rows table's
        r = j_lens.decode_vector(vec, space="post_refiner", family=args.family, top_k=args.top)
        return [w["word"] for w in r["positive_words"]], [w["word"] for w in r["negative_words"]]

    readers = {"jlens": jlens_read}
    rows_info = None
    if args.rows:
        if not args.tokenizer:
            print("--rows needs --tokenizer, the folder of the Qwen3-VL tokenizer files")
            return 2
        import control_vector_lexicon

        rows_read, rows_info = control_vector_lexicon.load_rows_reader(args.rows, args.family, args.tokenizer)
        readers["rows"] = lambda vec, concept: rows_read(vec, concept, args.top)

    readouts, twins = {}, {}
    for name in names:
        g = torch.Generator().manual_seed(int(hashlib.sha256(f"{args.seed}:{name}".encode()).hexdigest()[:8], 16))
        twin = torch.randn(vecs[name]["vec"].shape, generator=g)
        twin = twin / twin.norm() * vecs[name]["vec"].norm()
        for reader, read in readers.items():
            readouts[(name, reader)] = read(vecs[name]["vec"], name)
            twins[(name, reader)] = read(twin, name)

    quality = {n: vecs[n]["quality"] for n in names}
    packets, key_items, groups, left_out, leftover = build_run(
        names, quality, cos, readouts, twins, list(readers), seed=args.seed)

    group_rows = []
    for gi, g in enumerate(groups):
        pairs = [cos(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
        group_rows.append({"group": gi, "members": g, "mean_cos": round(sum(pairs) / len(pairs), 3),
                           "min_cos": round(min(pairs), 3)})
    bundle = Path(j_lens.default_bundle_path())
    meta = {
        "format": FORMAT, "commit": _commit(), "family": args.family, "seed": args.seed,
        "top_words_per_pole": args.top, "readers": list(readers), "group_size": GROUP_SIZE,
        "vectors_scored": sum(len(g) for g in groups), "groups": group_rows,
        "left_out_as_variants": left_out, "left_over_ungrouped": leftover,
        "skipped_no_pair_file": missing_pair_file,
        "j_lens_bundle": {"file": bundle.name, "bytes": bundle.stat().st_size, "mtime": int(bundle.stat().st_mtime)},
        "rows_cache": None if rows_info is None else {
            "folder": str(Path(args.rows)), "built": rows_info["built"], "commit": rows_info["commit"],
            "min_prompts": rows_info["min_prompts"]},
        "pair_file_in_rows_cache": None if rows_info is None else sorted(
            n for n in names if f"pairs:{n}" in rows_info["sources"]),
        "vector_files": {n: vecs[n]["file"] for n in names}, "pair_files": {n: vecs[n]["pair_file"] for n in names},
        "readouts": {f"{n}/{r}": {"plus": p, "minus": m} for (n, r), (p, m) in sorted(readouts.items())},
    }
    (out / "packets").mkdir(parents=True, exist_ok=True)
    for pi, packet in enumerate(packets):
        (out / "packets" / f"packet_{pi}.json").write_text(json.dumps(packet, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "judge_instructions.md").write_text(JUDGE_INSTRUCTIONS, encoding="utf-8")
    (out / "judge_instructions_options_only.md").write_text(JUDGE_INSTRUCTIONS_OPTIONS_ONLY, encoding="utf-8")
    (out / "key.json").write_text(json.dumps({"meta": meta, "items": key_items}, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"{meta['vectors_scored']} vectors in {len(groups)} groups of {GROUP_SIZE}, readers {', '.join(readers)}, "
          f"{len(key_items)} items in {len(packets)} packets under {out / 'packets'}")
    for row in group_rows:
        print(f"  group {row['group']}: mean cosine {row['mean_cos']:.2f}, lowest {row['min_cos']:.2f}: {', '.join(row['members'])}")
    for label, listed in (("left out as a variant of another vector", left_out), ("left over, not grouped", leftover),
                          ("skipped, no pair file", missing_pair_file)):
        if listed:
            print(f"  {label}: {', '.join(listed)}")
    print(f"  the last packet (packet_{len(packets) - 1}.json) is options-only and takes judge_instructions_options_only.md")
    print("  hand a judge one packet and its instructions, never this folder: key.json holds the answers")
    return 0


def load_run(run: Path):
    key = json.loads((run / "key.json").read_text(encoding="utf-8"))
    items = {}
    for p in sorted(glob.glob(str(run / "packets" / "packet_*.json"))):
        for it in json.loads(Path(p).read_text(encoding="utf-8")):
            items[it["id"]] = it
    passes = {}
    for d in sorted(p for p in (run / "answers").glob("*") if p.is_dir()) if (run / "answers").is_dir() else []:
        answers = []
        for p in sorted(d.glob("*.json")):
            answers.extend(json.loads(p.read_text(encoding="utf-8")))
        passes[d.name] = answers
    return key, items, passes


def cmd_score(args) -> int:
    run = Path(args.run)
    key, items, passes = load_run(run)
    meta, key_items = key["meta"], key["items"]
    if not passes:
        print(f"no answers under {run / 'answers'}/<pass>/. Save each judge's answers there, one file per packet.")
        return 2
    bundle = meta.get("j_lens_bundle") or {}
    print(f"run {run}: {meta.get('vectors_scored', '?')} vectors, family {meta.get('family', '?')}, "
          f"readers {', '.join(meta.get('readers', []))}, built at {meta.get('commit', '?')}")
    if bundle:
        print(f"  J-Lens bundle {bundle.get('file')}, {bundle.get('bytes')} bytes, mtime {bundle.get('mtime')}")
    if meta.get("rows_cache"):
        print(f"  rows cache {meta['rows_cache']['folder']}, built {meta['rows_cache']['built']} at {meta['rows_cache']['commit']}")
    print("  it scores naming, not what a render shows; a count describes the vector and the reader together; "
          "the judge is not seen from here")
    outcomes, results = {}, {"passes": {}, "between_passes": {}}
    for name, answers in passes.items():
        outcomes[name] = judged(key_items, items, answers)
        counted = count_pass(key_items, outcomes[name])
        results["passes"][name] = counted
        print(f"\npass {name}: {counted['answered']} of {counted['expected']} items answered")
        oo = counted["options_only"]
        if oo["n"]:
            print(f"  options only, no readout: target for {oo['target']} of {oo['n']} "
                  f"(one per group is what a consistent guesser gets: {len(meta.get('groups', []))})")
        for reader, c in counted["readers"].items():
            n = c["n"]
            print(f"  {reader} (n = {n})")
            print(f"    real:    target {c['real_target']}, cannot tell {c['real_cannot_tell']}, another option {c['real_other_option']}; "
                  f"target with a quote found in the readout {c['real_target_quote_found']}")
            print(f"    twin:    target {c['twin_target']}, cannot tell {c['twin_cannot_tell']}")
            print(f"    shuffle: target {c['shuffle_target']}")
            print(f"    real target and twin not: {c['real_target_and_twin_not']}")
            print(f"    real target by group: " + ", ".join(f"g{g} {k}" for g, k in c["real_target_by_group"].items()))
            print(f"    real, not the target: {', '.join(c['real_not_target']) or 'none'}")
    names = list(outcomes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            cmp = compare_passes(key_items, outcomes[a], outcomes[b])
            results["between_passes"][f"{a} vs {b}"] = cmp
            print(f"\npasses {a} and {b}, the same items judged twice:")
            for slot, row in cmp.items():
                line = f"  {slot}: same pick for {row['same_pick']} of {row['n']}"
                if "target_in_both" in row:
                    line += (f"; target in both {row['target_in_both']}, in one {row['target_in_one']}, "
                             f"in neither {row['target_in_neither']}")
                print(line)
    (run / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packets", help="read the built vectors and write judge packets and the key")
    p.add_argument("--out", required=True, help="a new folder for the run; untracked data/control_vectors/<run> is the place")
    p.add_argument("--vectors", default=str(COMFY_ROOT / "models" / "control_vectors"))
    p.add_argument("--family", default="fl2va", choices=("fl2va", "ref2va"))
    p.add_argument("--rows", default=None, help="a folder written by bench/control_vector_rows.py; adds the rows-table reader")
    p.add_argument("--tokenizer", default=None, help="folder of the Qwen3-VL tokenizer files; needed with --rows")
    p.add_argument("--top", type=int, default=DEFAULT_TOP, help="words per pole in a readout")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="draws the option orders, the twins and the packet orders")
    p.set_defaults(func=cmd_packets)
    s = sub.add_parser("score", help="count the judges' answers for every pass under <run>/answers/")
    s.add_argument("run", help="a folder written by `packets`")
    s.set_defaults(func=cmd_score)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
