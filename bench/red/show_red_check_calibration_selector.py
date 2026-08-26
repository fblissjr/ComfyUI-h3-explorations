"""Show `bench/check_calibration_selector.py` red, one arm at a time.

Every arm of that check grades a SUBPROCESS -- the selector's CLI -- so the
mutation has to reach a file on disk rather than an attribute on a module. Each
case writes a mutated copy of `select_v2_calibration_rows.py` into a scratch
tree that mirrors `bench/` by symlink (the selector resolves its pool and its
sibling imports from its own directory), points the check's `SELECTOR` at it,
and runs the one arm that mutation should turn red.

The baseline is the unmutated selector run through one arm, so every
expectation is derived from it rather than authored per case: a mutation that
never reached the selector leaves the arm green, and green equals the baseline,
which is exactly what MUTATION already refuses.

Each mutation asserts its own source expression is present before replacing it.
The 2026-08-16 instance behind that: a mutation whose source text had moved
applied nothing and the check "passed" having tested nothing.

The near-miss is the half worth reading. A comment added to the selector must
NOT move any verdict -- a check that fires on a reformatted file is a check
whose reds get ignored.
"""
import tempfile
from pathlib import Path

from harness import Harness, MUTATION, NEAR_MISS, main, subject

C = subject("bench/check_calibration_selector.py")
REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "bench"
SELECTOR_SRC = (BENCH / "select_v2_calibration_rows.py").read_text()

# (label, kind, arm name, source expression, replacement)
CASES = [
    ("M1 selection order stops being seeded", MUTATION, "determinism",
     'order = sorted(pool, key=lambda r: hashlib.sha256(f"{args.seed}:{r[\'id\']}".encode()).hexdigest())',
     'order = list({id(r): r for r in pool}.values()); random.Random().shuffle(order)'),
    ("M2 a role floor moves", MUTATION, "pre_change_revision",
     '"video-reference": 6,',
     '"video-reference": 3,'),
    ("M3 the fill stops chasing the stratum furthest behind", MUTATION,
     "stratum_token_shares",
     'name = min(open_strata, key=lambda s: achieved[s] / max(1, targets[s]["tokens"]))',
     'name = open_strata[0]'),
    ("M4 the vision cap stops skipping", MUTATION, "vision_cap",
     '''            if (args.max_vision_tokens_per_row
                    and est["visual_tokens_est"] > args.max_vision_tokens_per_row):
                return "over_vision_cap"''',
     '            pass'),
    ("M5 assignment ignores the corrected family map", MUTATION,
     "family_disjointness",
     '''        if family_map:
            return family_map.get(row["id"]) or row.get("media_component") or row["id"]
        return row.get("media_component") or row["id"]''',
     '        return row.get("media_component") or row["id"]'),
    ("M6 a caveatless component map is accepted", MUTATION, "refusals",
     '''    if not caveat:
        raise SystemExit(''',
     '''    if False:
        raise SystemExit('''),
    ("G1 a comment added to the selector", NEAR_MISS, "determinism",
     'from __future__ import annotations',
     '# a comment that changes nothing\nfrom __future__ import annotations'),
]

ARMS = dict(C.ARMS)


def mutate(source_expr: str | None, replacement: str | None) -> Path:
    """A scratch mirror of bench/ holding one mutated selector."""
    root = Path(tempfile.mkdtemp(prefix="red_ccs_"))
    mirror = root / "bench"
    mirror.mkdir()
    for sibling in BENCH.glob("*.py"):
        if sibling.name != "select_v2_calibration_rows.py":
            (mirror / sibling.name).symlink_to(sibling)
    (mirror / "results").symlink_to(BENCH / "results")
    text = SELECTOR_SRC
    if source_expr is not None:
        if source_expr not in text:
            raise AssertionError(
                f"mutation source expression is absent from the selector, so "
                f"nothing would have been mutated: {source_expr[:80]!r}")
        text = text.replace(source_expr, replacement, 1)
    target = mirror / "select_v2_calibration_rows.py"
    target.write_text(text)
    return target


def run_arm(arm: str, source_expr: str | None, replacement: str | None):
    selector = mutate(source_expr, replacement)
    original = C.SELECTOR
    C.SELECTOR = selector
    try:
        with tempfile.TemporaryDirectory() as tmp:
            _, problems = ARMS[arm](SOURCE, Path(tmp))
        return problems
    finally:
        C.SELECTOR = original


SOURCE = C.source_dir()


def build():
    h = Harness(subject="bench/check_calibration_selector.py")
    if SOURCE is None:
        print("  SKIP  fixture absent: the released tokenizer directory")
        print("        every arm prices rows with it; nothing can run")
        raise SystemExit(2)
    h.fixture(C.POOL, "the accepted calibration pool every arm selects from")
    h.baseline(lambda: run_arm("determinism", None, None))
    for label, kind, arm, source_expr, replacement in CASES:
        h.case(label, kind,
               lambda a=arm, s=source_expr, r=replacement: run_arm(a, s, r))
    return h


if __name__ == "__main__":
    main(build)
