"""Show `bench/check_doc_inventory.py` red, and green on what must not trip it.

Feeds `audit()` synthetic index text and synthetic filename lists rather than
touching `docs/checks.md`, so a mutation cannot be left behind in a load-bearing
file. `audit()` is pure for exactly this reason -- a collector that read the real
file could not be shown red without editing it.

The NEAR_MISS cases are the half worth having. This check's whole risk is firing
on something that looks like an index row and is not, or on a `.py` mentioned in
prose. A false red here is worse than no check, so the greens are the argument
that it is safe to gate on.
"""
from harness import Harness, MUTATION, NEAR_MISS, main, subject

C = subject("bench/check_doc_inventory.py")

# A minimal well-formed index: two rows, both naming files that really exist.
GOOD = """## The index

| check | defends | needs | claims block | shown red |
|---|---|---|---|---|
| `check_doc_links.py` | pointers resolve | - | yes | yes |
| `check_node_ids.py` | the positional contract | CUDA | yes | yes |

## Next section
"""

ON_DISK = ["bench/check_doc_links.py", "bench/check_node_ids.py"]


def build():
    h = Harness(subject="bench/check_doc_inventory.py")
    h.baseline(lambda: C.audit(GOOD, ON_DISK))

    # --- mutations: each must move the verdict off the baseline ---

    h.case("M1 a check on disk with no row", MUTATION,
           lambda: C.audit(GOOD, ON_DISK + ["bench/check_brand_new.py"]))

    h.case("M2 a row naming a file that does not exist", MUTATION,
           lambda: C.audit(
               GOOD.replace("`check_node_ids.py`", "`check_deleted_yesterday.py`"),
               ["bench/check_doc_links.py"]))

    h.case("M3 the index section is missing entirely", MUTATION,
           lambda: C.audit("# a doc with no index\n\nprose only.\n", ON_DISK))

    h.case("M4 the index heading is present but the table is empty", MUTATION,
           lambda: C.audit("## The index\n\nnothing here yet.\n\n## Next\n", ON_DISK))

    h.case("M5 a row whose subject is not a .py at all", MUTATION,
           lambda: C.audit(
               GOOD.replace("| `check_node_ids.py` |", "| the positional contract |"),
               ON_DISK))

    # --- retirement: "absent" is a third state, graded in both directions ---
    #
    # The escaped instance these exist for: `docs/checks.md` retired
    # `check_mono_ref_audio.py` on 2026-08-29 by striking its subject through
    # and keeping the row for its reasoning. The subject pattern had only ever
    # met live rows, so the leading `~~` did not match and the row was reported
    # as naming no subject at all -- red on a correct state, which stood long
    # enough that two sessions learned to skip this check.
    #
    # **`audit()` is only half pure, and these cases had to be built around
    # it.** It takes the on-disk check list as an argument but resolves each
    # row's subject against the REAL filesystem, so a struck-through row naming
    # a file that exists here cannot be made to look gone by passing a shorter
    # list. The first version of G5 did exactly that and failed. So the gone
    # case uses a name that is genuinely absent, and the back-on-disk case
    # keeps its subject out of the check list so only the intended direction
    # can fire.
    RETIRED_GONE = GOOD.replace("| `check_node_ids.py` |",
                                "| ~~`check_retired_yesterday.py`~~ |")
    RETIRED_BACK = GOOD.replace("| `check_doc_links.py` |",
                                "| ~~`check_doc_links.py`~~ |")

    h.case("M6 a RETIRED row whose file is back on disk", MUTATION,
           lambda: C.audit(RETIRED_BACK, ["bench/check_node_ids.py"]))

    # --- near misses: each must leave the verdict where the baseline put it ---

    h.case("G5 a RETIRED row whose file is correctly gone", NEAR_MISS,
           lambda: C.audit(RETIRED_GONE, ["bench/check_doc_links.py"]))

    h.case("G1 a .py named in prose outside the index", NEAR_MISS,
           lambda: C.audit(
               GOOD + "\nSee `check_totally_fictional.py` for the reasoning.\n",
               ON_DISK))

    h.case("G2 a pipe-led line that is not a table row", NEAR_MISS,
           lambda: C.audit(
               GOOD.replace("## Next section",
                            "| this is prose that happens to start with a pipe\n\n## Next section"),
               ON_DISK))

    h.case("G3 a non-check row, which is listed on purpose", NEAR_MISS,
           lambda: C.audit(
               GOOD.replace("## Next section",
                            "| `preflight_graph.py` | grades, does not gate | - | yes | yes |\n\n## Next section"),
               ON_DISK))

    h.case("G4 a second table after the index is not read", NEAR_MISS,
           lambda: C.audit(
               GOOD + "\n| `check_imaginary.py` | in another table | - | no | no |\n",
               ON_DISK))

    h.case("restored: the good index again", NEAR_MISS,
           lambda: C.audit(GOOD, ON_DISK))
    return h


if __name__ == "__main__":
    main(build)
