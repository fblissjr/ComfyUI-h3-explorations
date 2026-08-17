"""Show `bench/check_graph_discovery.py` red, and confirm it stays green on the
cases that would make it useless if it fired.

Exercises `enumeration_sites()` -- the COLLECTOR -- on synthetic sources written
to a scratch dir, not just the reporting. The node_ids harness only feeds a
comparator, which would pass a collector that returned its own baseline; that
hole is why this one is written the other way round.

The NEAR_MISS cases are the valuable half. A checker that flags the same glob
inside a comment or a docstring produces false reds, which `CLAUDE.md` rates
worse than no check at all.

Ported onto `harness.py` 2026-08-17. The baseline is now the correct
`graph_paths()` form rather than each case being judged absolutely, so every
expectation is derived. Case filenames are index-based; the previous version
used `abs(hash(label))`, which changes per interpreter run.
"""
import tempfile
from pathlib import Path

from harness import Harness, MUTATION, NEAR_MISS, main, subject

C = subject("bench/check_graph_discovery.py")
TMP = Path(tempfile.mkdtemp(prefix="red_cgd_"))

BASELINE_SRC = 'from h3_config import graph_paths\nfor p in graph_paths(WORKFLOWS): pass\n'

CASES = [
    ("M1 bare glob over workflows", MUTATION,
     'WORKFLOWS = Path("workflows")\nfor p in WORKFLOWS.glob("*.json"): pass\n'),
    ("M2 rglob on a workflows-named var", MUTATION,
     'workflows_dir = Path("w")\nfor p in workflows_dir.rglob("*.json"): pass\n'),
    ("M3 iterdir on a graph dir", MUTATION,
     'WORKFLOWS = Path("w")\nfor p in WORKFLOWS.iterdir(): pass\n'),
    ("M4 non-literal pattern (unjudgeable statically)", MUTATION,
     'WORKFLOWS = Path("w")\npat = "*.json"\nfor p in WORKFLOWS.glob(pat): pass\n'),
    ("G1 the SAME glob inside a comment (the regex trap)", NEAR_MISS,
     '# never write WORKFLOWS.glob("*.json") here\nx = 1\n'),
    ("G2 the same glob inside a docstring", NEAR_MISS,
     '"""Do not use WORKFLOWS.glob(\'*.json\')."""\nx = 1\n'),
    ("G3 process enumeration, not graphs", NEAR_MISS,
     'import pathlib\nfor p in pathlib.Path("/proc").iterdir(): pass\n'),
    ("G4 globbing python files, not graphs", NEAR_MISS,
     'REPO = Path(".")\nfor p in REPO.rglob("*.py"): pass\n'),
]


def audit(src: str, name: str):
    f = TMP / f"check_{name}.py"
    f.write_text(src)
    return C.audit([f])


def build():
    print(f"  scratch: {TMP}")
    h = Harness(subject="bench/check_graph_discovery.py")
    h.baseline(lambda: audit(BASELINE_SRC, "baseline"))
    for i, (label, kind, src) in enumerate(CASES):
        h.case(label, kind, lambda s=src, n=i: audit(s, str(n)))
    return h


if __name__ == "__main__":
    main(build)
