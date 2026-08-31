"""Show every branch of `bench/check_node_ids.py` red, for the right reason.

Feeds `compare()` mutated mappings rather than editing the schema it guards, so
there is no way to leave a mutation behind in a load-bearing file.

The rule under test lives in `docs/comfy_notes.md`: never rename a node's
`node_id=`, and never insert an input or output anywhere but the end. Saved
graphs store `type` as that string and match `widgets_values` by index, so a
rename or an insertion breaks every graph built before it -- including the
owner's graphs outside this repo, which no check can see. That is why this
subject is tier 1.

Ported onto `harness.py` 2026-08-17. Before that it computed a `want_red`
mismatch, printed `<-- WRONG`, and exited 0 regardless.
"""
import copy
import json

from harness import Harness, MUTATION, NEAR_MISS, REPO, main, subject

C = subject("bench/check_node_ids.py")
MANIFEST = json.loads((REPO / "bench" / "node_id_manifest.json").read_text())


def compare(mapping):
    return C.compare(mapping, MANIFEST)


def unmutated():
    return compare(copy.deepcopy(MANIFEST))


def mutate(fn):
    """Return a case callable that applies fn to a fresh copy, then compares."""

    def run():
        m = copy.deepcopy(MANIFEST)
        fn(m)
        return compare(m)

    return run


def build():
    h = Harness(subject="bench/check_node_ids.py")
    h.fixture(REPO / "bench" / "node_id_manifest.json",
              "the committed baseline the schema cannot regenerate; without it "
              "the check has nothing independent to diff against")
    h.baseline(unmutated)

    h.case("M1 node_id renamed", MUTATION,
           mutate(lambda m: m["MiniMaxH3Preflight"].__setitem__("node_id", "MiniMaxH3Preflite")))
    h.case("M2 inputs reordered", MUTATION,
           mutate(lambda m: m["MiniMaxH3SageAttention"]["inputs"].reverse()))
    h.case("M3 input INSERTED in the middle (the 2026-08-10 head_chunks bug)", MUTATION,
           mutate(lambda m: m["MiniMaxH3SageAttention"]["inputs"].insert(1, "new_knob")))
    h.case("M4 outputs reordered (links are integer slots)", MUTATION,
           mutate(lambda m: m["MiniMaxH3Resolution"]["outputs"].reverse()))
    h.case("M5 a registered node disappears", MUTATION,
           mutate(lambda m: m.pop("MiniMaxH3Preflight")))
    h.case("M6 input APPENDED at the end (permitted, must still be recorded)", MUTATION,
           mutate(lambda m: m["MiniMaxH3Resolution"]["inputs"].append("tail_knob")))
    h.case("M7 a new node appears (permitted, must still be recorded)", MUTATION,
           mutate(lambda m: m.__setitem__("MiniMaxH3BrandNew",
                                          {"node_id": "X", "inputs": [], "outputs": []})))

    h.case("restored: unmutated again", NEAR_MISS, unmutated)
    return h


if __name__ == "__main__":
    main(build)
