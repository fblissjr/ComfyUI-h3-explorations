#!/usr/bin/env python3
"""Check that `bench/control_vector_readout_score.py` builds blind judge packets that leak nothing and counts
answers as it says.

The tool scores control-vector readouts by what a blind judge picks. Its first two option designs were
dropped for reasons a check could have caught: one was too easy to rank readers, and the other let a judge
with no readout find the target from the option list alone
(`bench/results/2026-11-12_control_vector_readout_blind_score.md`). These cases hold the third design to what
it promises, on a made-up set of vectors with planted groups, with no model and no bundle.

Claims, i.e. what breaks if a case is deleted:
  groups are the near ones   planted clusters come back as the groups; what does not fill a group is listed
                             and is in no packet; a name that extends another's is left out as a variant
  a list says nothing        every member of a group gets the same option lines, each line is the target for
                             exactly one member, and a guesser that always picks the same line of a list
                             scores one per group through the tool's own counting
  items are what they say    a `real` item carries the vector's readout, a `twin` item the twin's, a `shuffle`
                             item the readout of the nearest vector outside the group
  packets are blind          an item holds an id, the words and the options and nothing else; no id or option
                             order follows the vector's name; each readout packet has one item per vector
  same seed, same run        the same seed gives the same packets; another seed changes an option order
  counts are the answers     hits, "cannot tell", other options, twin and shuffle hits and quote checks come
                             out as planted; a quote word that is not in the readout is not counted as found;
                             a vector with an unanswered variant is not counted for that reader
  bad answers are refused    an id that is not in the key, and an id answered twice in a pass, raise
  two passes are compared    the same-pick and target-in-both counts equal what was planted
  the files round-trip       `score` on a run folder gives the counts `count_pass` gives, writes
                             `results.json`, and says so and returns 2 when there are no answers; `packets`
                             refuses a folder that already holds a run
  constants agree            the letters cover a group, and the instructions name the "cannot tell" letter

Not covered: the readers themselves (J-Lens and the rows table need the bundle and the row cache), the judges,
and whether a count means anything about a render.

No GPU, no model.

    <comfy-venv-python> bench/check_control_vector_readout_score.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

import control_vector_readout_score as rs  # noqa: E402

FAILED: list[str] = []


def case(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILED.append(name)


# A made-up world: three planted clusters of five, two strays, and one variant name.
CLUSTERS = {"red": ["red_a", "red_b", "red_c", "red_d", "red_e"],
            "sea": ["sea_a", "sea_b", "sea_c", "sea_d", "sea_e"],
            "hum": ["hum_a", "hum_b", "hum_c", "hum_d", "hum_e"]}
STRAYS = ["stray_one", "stray_two"]
VARIANT = "red_a_neutral"
AXIS = {"red": 0, "sea": 1, "hum": 2}


def _vector(name: str) -> list[float]:
    v = [0.0] * 8
    if name in STRAYS:
        v[3 + STRAYS.index(name)] = 1.0
        v[0] = 0.05 * (1 + STRAYS.index(name))
    else:
        cluster = name.split("_")[0]
        v[AXIS[cluster]] = 1.0
        v[5 + (ord(name.split("_")[1][0]) - ord("a")) % 3] = 0.2 + 0.03 * (ord(name.split("_")[1][0]) - ord("a"))
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


NAMES = sorted([n for members in CLUSTERS.values() for n in members] + STRAYS + [VARIANT])
VEC = {n: _vector(n) for n in NAMES}
READERS = ["r1", "r2"]
QUALITY = {n: f"the concept of {n} against plainness" for n in NAMES}
READOUTS = {(n, r): ([f"{r}+{n}+{i}" for i in range(3)], [f"{r}-{n}-{i}" for i in range(3)]) for n in NAMES for r in READERS}
TWINS = {(n, r): ([f"{r}+twin+{n}+{i}" for i in range(3)], [f"{r}-twin+{n}+{i}" for i in range(3)]) for n in NAMES for r in READERS}


def cos(a: str, b: str) -> float:
    return sum(x * y for x, y in zip(VEC[a], VEC[b]))


def build(seed: int = 7):
    return rs.build_run(NAMES, QUALITY, cos, READOUTS, TWINS, READERS, seed=seed)


def answers_for(key_items, items, plan):
    """Answers from a plan `{(vector, reader, variant): "hit" | "tell" | "other" | "hit_badquote"}`."""
    out = []
    for iid, k in key_items.items():
        what = plan.get((k["vector"], k["reader"], k["variant"]))
        if what is None:
            continue
        words = items[iid].get("plus_words", [])
        if what in ("hit", "hit_badquote"):
            quote = [words[0]] if what == "hit" and words else (["not-in-the-readout"] if words else [])
            out.append({"id": iid, "pick": k["gold"], "quote": quote})
        elif what == "tell":
            out.append({"id": iid, "pick": rs.CANNOT_TELL, "quote": []})
        else:
            other = next(letter for letter in rs.LETTERS if letter != k["gold"])
            out.append({"id": iid, "pick": other, "quote": [words[0]] if words else []})
    return out


def main() -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    packets, key_items, groups, left_out, leftover = build()
    items = {it["id"]: it for packet in packets for it in packet}
    scored = sorted({k["vector"] for k in key_items.values()})

    print("groups are the near ones")
    case("the planted clusters are the groups", sorted(map(sorted, groups)) == sorted(map(sorted, CLUSTERS.values())),
         f"{groups}")
    case("what does not fill a group is listed", leftover == sorted(STRAYS), f"{leftover}")
    case("a name that extends another's is left out", left_out == [VARIANT], f"{left_out}")
    case("neither is in any packet", not (set(scored) & set(STRAYS + [VARIANT])), f"{scored}")

    print("a list says nothing")
    same_lines, once_each = True, True
    for group in groups:
        for reader, variant in [(r, v) for r in READERS for v in rs.VARIANTS] + [("none", "options_only")]:
            mine = [(iid, k) for iid, k in key_items.items()
                    if k["vector"] in group and k["reader"] == reader and k["variant"] == variant]
            lines = {tuple(sorted(items[iid]["options"].values())) for iid, _ in mine}
            same_lines &= len(lines) == 1 and len(mine) == len(group)
            targets = sorted(items[iid]["options"][k["gold"]] for iid, k in mine)
            once_each &= targets == sorted(QUALITY[n] for n in group)
    case("every member of a group gets the same option lines", same_lines)
    case("each line is the target for exactly one member", once_each)
    guesser = [{"id": iid, "pick": min(items[iid]["options"], key=lambda letter: items[iid]["options"][letter]), "quote": []}
               for iid, k in key_items.items() if k["variant"] == "options_only"]
    floor = rs.count_pass(key_items, rs.judged(key_items, items, guesser))["options_only"]
    case("a guesser that always picks the same line scores one per group",
         floor["target"] == len(groups) and floor["n"] == len(scored), f"{floor}")

    print("items are what they say")
    right_words, right_shuffle = True, True
    for iid, k in key_items.items():
        if k["variant"] == "options_only":
            continue
        group = next(g for g in groups if k["vector"] in g)
        outside = [n for n in NAMES if n not in group and n != VARIANT]
        nearest = max(outside, key=lambda n: cos(n, k["vector"]))
        right_shuffle &= k["shuffle_source"] == nearest and nearest not in group
        want = {"real": READOUTS[(k["vector"], k["reader"])], "twin": TWINS[(k["vector"], k["reader"])],
                "shuffle": READOUTS[(nearest, k["reader"])]}[k["variant"]]
        right_words &= (items[iid]["plus_words"], items[iid]["minus_words"]) == (want[0], want[1])
    case("real, twin and shuffle items carry the readout they name", right_words)
    case("the shuffle readout is the nearest vector outside the group", right_shuffle)

    print("packets are blind")
    case("an item holds an id, the words and the options only",
         all(set(it) in ({"id", "plus_words", "minus_words", "options"}, {"id", "options"}) for it in items.values()))
    case("the options-only packet has no words", all(set(it) == {"id", "options"} for it in packets[-1]) and len(packets[-1]) == len(scored))
    case("no id carries a vector's name", not any(n in iid for iid in items for n in NAMES))
    case("each readout packet has one item per vector",
         all(sorted(key_items[it["id"]]["vector"] for it in packet) == scored for packet in packets[:-1]))
    golds = [k["gold"] for k in key_items.values()]
    case("the target is not always the same letter", len(set(golds)) == len(rs.LETTERS), f"{sorted(set(golds))}")

    print("same seed, same run")
    again = build()
    other = build(seed=8)
    case("the same seed gives the same packets and key", again[0] == packets and again[1] == key_items)
    orders = lambda run: [tuple(k["option_vectors"].values()) for _, k in sorted(run[1].items(), key=lambda kv: (kv[1]["vector"], kv[1]["reader"], kv[1]["variant"]))]
    case("another seed changes an option order", orders(other) != orders((packets, key_items)))

    print("counts are the answers")
    plan = {}
    for i, v in enumerate(scored):
        plan[(v, "r1", "real")] = "hit" if i < 8 else "hit_badquote" if i == 8 else "tell" if i < 12 else "other"
        plan[(v, "r1", "twin")] = "hit" if i == 0 else "tell"
        plan[(v, "r1", "shuffle")] = "hit" if i in (1, 2) else "other"
        plan[(v, "r2", "real")] = "hit" if i < 3 else "tell"
        plan[(v, "r2", "twin")] = "tell"
        if i != 4:  # one vector's shuffle item is left unanswered for r2
            plan[(v, "r2", "shuffle")] = "tell"
    first = rs.judged(key_items, items, answers_for(key_items, items, plan))
    counted = rs.count_pass(key_items, first)
    r1, r2 = counted["readers"]["r1"], counted["readers"]["r2"]
    case("real items: target, cannot tell, another option",
         (r1["real_target"], r1["real_cannot_tell"], r1["real_other_option"]) == (9, 3, len(scored) - 12), f"{r1}")
    case("a quote word that is not in the readout is not counted as found", r1["real_target_quote_found"] == 8, f"{r1}")
    case("twin and shuffle hits", (r1["twin_target"], r1["twin_cannot_tell"], r1["shuffle_target"]) == (1, len(scored) - 1, 2), f"{r1}")
    case("a real hit whose twin also hit is set apart", r1["real_target_and_twin_not"] == 8, f"{r1}")
    case("the vectors that did not go to the target are named", r1["real_not_target"] == sorted(scored[9:]), f"{r1['real_not_target']}")
    by_group = {str(gi): sum(1 for v in g if scored.index(v) < 9) for gi, g in enumerate(groups)}
    case("target counts by group add up", r1["real_target_by_group"] == by_group, f"{r1['real_target_by_group']} vs {by_group}")
    case("a vector with an unanswered variant is not counted for that reader",
         r2["n"] == len(scored) - 1 and r2["real_target"] == 3 and counted["answered"] == counted["expected"] - len(scored) - 1,
         f"{r2['n']} {counted['answered']} {counted['expected']}")

    print("bad answers are refused")
    some_id = next(iter(key_items))
    for label, bad in (("an id that is not in the key", [{"id": "nope", "pick": "A", "quote": []}]),
                       ("an id answered twice in a pass", [{"id": some_id, "pick": "A", "quote": []}] * 2)):
        try:
            rs.judged(key_items, items, bad)
            case(label + " raises", False, "it was accepted")
        except ValueError:
            case(label + " raises", True)

    print("two passes are compared")
    plan2 = dict(plan)
    flipped = [scored[0], scored[9]]  # r1 real: one hit becomes "cannot tell", one "cannot tell" becomes a hit
    plan2[(flipped[0], "r1", "real")] = "tell"
    plan2[(flipped[1], "r1", "real")] = "hit"
    second = rs.judged(key_items, items, answers_for(key_items, items, plan2))
    cmp = rs.compare_passes(key_items, first, second)["r1/real"]
    case("same pick, and the target in both, in one and in neither",
         (cmp["same_pick"], cmp["target_in_both"], cmp["target_in_one"], cmp["target_in_neither"]) == (len(scored) - 2, 8, 2, len(scored) - 10),
         f"{cmp}")
    case("a slot no pass changed agrees on every item", rs.compare_passes(key_items, first, second)["r1/twin"]["same_pick"] == len(scored))

    print("the files round-trip")
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        (run / "packets").mkdir(parents=True)
        for pi, packet in enumerate(packets):
            (run / "packets" / f"packet_{pi}.json").write_text(json.dumps(packet), encoding="utf-8")
        meta = {"format": rs.FORMAT, "readers": READERS, "vectors_scored": len(scored), "family": "made-up",
                "groups": [{"group": gi, "members": g} for gi, g in enumerate(groups)]}
        (run / "key.json").write_text(json.dumps({"meta": meta, "items": key_items}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as said:
            code = rs.main(["score", str(run)])
        case("score says there are no answers and returns 2", code == 2 and "no answers" in said.getvalue(), f"{code}")
        for name, the_plan in (("one", plan), ("two", plan2)):
            (run / "answers" / name).mkdir(parents=True)
            (run / "answers" / name / "packet_all.json").write_text(json.dumps(answers_for(key_items, items, the_plan)), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as said:
            code = rs.main(["score", str(run)])
        written = json.loads((run / "results.json").read_text(encoding="utf-8"))
        case("score on a run folder gives count_pass's counts", code == 0 and written["passes"]["one"] == json.loads(json.dumps(counted)), f"{code}")
        case("and compares the two passes it found", written["between_passes"]["one vs two"]["r1/real"] == cmp)
        case("what it prints says what it cannot see", "naming" in said.getvalue() and "judge is not seen" in said.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as said:
            code = rs.main(["packets", "--out", str(run)])
        case("packets refuses a folder that already holds a run", code == 2 and "already holds a run" in said.getvalue(), f"{code}")

    print("constants agree")
    case("the letters cover a group", len(rs.LETTERS) == rs.GROUP_SIZE and rs.CANNOT_TELL not in rs.LETTERS)
    case("the instructions name the cannot-tell letter and the last option letter",
         f"`{rs.CANNOT_TELL}` (cannot tell)" in rs.JUDGE_INSTRUCTIONS and f"A to {rs.LETTERS[-1]}" in rs.JUDGE_INSTRUCTIONS
         and f"A to {rs.LETTERS[-1]}" in rs.JUDGE_INSTRUCTIONS_OPTIONS_ONLY)

    if FAILED:
        print(f"\nFAIL: {len(FAILED)} case(s): {'; '.join(FAILED)}")
        return 1
    print("\nok: bench/control_vector_readout_score.py builds blind packets and counts answers as it says")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
