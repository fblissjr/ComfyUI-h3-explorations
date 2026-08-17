"""Show the substrate assertions in `bench/check_capture_manifest.py` red.

Scoped to the substrate fields added 2026-08-17 -- `weight_quantization`,
`vae_quantization`, `gpu_power_limit_watts` -- not to the whole checker. The
older invariants predate the spine and are not calibrated here; the index says so.

**Two rules are pinned here.** The three-state rule, and that
`weight_quantization` is a projection of the `unet` filename which must agree with
it -- a second home for one fact is only honest if something checks the two match.

**The three-state rule is what the null cases exist to pin.** A missing key must
fail and a present `null` must pass, and those are one keystroke apart in a
manifest. An assertion that conflated them would look identical on a green run
and would quietly restore the hole the fields were added to close, so the
`null`-passes cases carry as much weight as the absence-fails ones.

Writes synthetic manifests to a scratch dir and calls the real `check_manifest`,
so nothing touches a capture on disk. The checker raises rather than returning,
so each case converts an `AssertionError` into an error list -- the spine reads a
non-empty list as red.
"""
import json
import tempfile
from pathlib import Path

from harness import Harness, MUTATION, NEAR_MISS, main, subject

C = subject("bench/check_capture_manifest.py")
TMP = Path(tempfile.mkdtemp(prefix="red_ccm_"))

# A real manifest is the baseline. Copying the one on disk would couple this
# harness to a capture store outside the repo, so the fixture is built here and
# only the fields under test are varied.
REAL = None
for cand in Path.home().glob("Storage/h3_captures/*/manifest.json"):
    REAL = cand
    break


def load_baseline() -> dict:
    """The known-good manifest, read once."""
    assert REAL is not None
    return json.loads(REAL.read_text())


def link_siblings() -> None:
    """Symlink the capture's tensors beside the scratch manifest.

    The checker verifies that every tensor the manifest lists exists and matches
    its checksum, so a scratch dir holding only `manifest.json` fails on the first
    tensor -- which made the baseline red and every mutation read as "never
    reached the subject". Symlinks rather than copies: the captures are large, and
    nothing here should write into a real capture directory.
    """
    assert REAL is not None
    for src in REAL.parent.iterdir():
        if src.name == "manifest.json":
            continue
        dst = TMP / src.name
        if not dst.exists():
            dst.symlink_to(src)


def verdict(mutate=None):
    """Write a manifest, run the real checker, return errors as a list."""
    def run():
        d = load_baseline()
        if mutate:
            mutate(d)
        link_siblings()
        p = TMP / "manifest.json"
        p.write_text(json.dumps(d))
        try:
            C.check_manifest(p)
        except AssertionError as exc:
            return [str(exc)]
        return []
    return run


def _block(d: dict, block: str) -> dict:
    return d["provenance"] if block == "provenance" else d["workload"]["models"]


def drop(block: str, key: str):
    def m(d):
        _block(d, block).pop(key)
    return m


def setnull(block: str, key: str):
    def m(d):
        _block(d, block)[key] = None
    return m


def build():
    h = Harness(subject="bench/check_capture_manifest.py")
    h.fixture(REAL or "/nonexistent",
              "needs one real capture manifest to mutate. Captures live outside "
              "the repo, so this is a documented run rather than a guard anyone "
              "can execute in a clone.")
    h.baseline(verdict())

    # Absence must fail -- the key is not recorded, so nothing knows the substrate.
    h.case("M1 weight_quantization key removed", MUTATION,
           verdict(drop("models", "weight_quantization")))
    h.case("M2 weight_quantization disagrees with the unet filename", MUTATION,
           verdict(lambda d: d["workload"]["models"].__setitem__(
               "weight_quantization", "fp8_scaled")))
    h.case("M3 gpu_power_limit_watts key removed", MUTATION,
           verdict(drop("provenance", "gpu_power_limit_watts")))

    # null must PASS -- confirmed absent is a recorded answer, not a gap. If any
    # of these goes red the three-state rule has collapsed back into two.
    h.case("G1 weight_quantization present but null", NEAR_MISS,
           verdict(setnull("models", "weight_quantization")))
    h.case("M4 vae_quantization disagrees with the video_vae filename", MUTATION,
           verdict(lambda d: d["workload"]["models"].__setitem__(
               "vae_quantization", "w4a8_mixed")))
    h.case("G2 vae_quantization removed -- deliberately NOT required", NEAR_MISS,
           verdict(drop("models", "vae_quantization")))
    h.case("G3 gpu_power_limit_watts present but null", NEAR_MISS,
           verdict(setnull("provenance", "gpu_power_limit_watts")))

    h.case("restored: the unmutated manifest", NEAR_MISS, verdict())
    return h


if __name__ == "__main__":
    # Scratch is left in place, as the sibling harnesses do -- a failing case is
    # easier to read with the manifest that produced it still on disk.
    print(f"  scratch: {TMP}")
    main(build)
