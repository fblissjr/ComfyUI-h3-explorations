#!/usr/bin/env python3
"""Is a render bit-identical across a full model unload?

## Why this exists

`bench/results/2026-08-28_audio_seed_spread.json` measured audio energy across
four seeds and found a 3.19 dB spread, which is what withdrew the `u4` half of
the carry ablation (`97d4c1a`). **That control straddled a model unload.**
Another session's render OOMed at 18:48:58 on 2026-08-28 and ComfyUI logged
"Got an OOM, unloading all loaded models"; seeds 1-2 rendered before it and
seeds 3-4 after, and seed 4 is the outlier that produced most of the spread.

So the withdrawal rests on a control that may itself be confounded. This settles
it by re-rendering seed 111222333, whose pre-unload value is 0.016280, and
asking whether the pipeline reproduces it.

    reproduces exactly  -> the pipeline is bit-identical across a reload, the
                           seed spread is real, and the withdrawal stands.
    does not            -> the spread is confounded, the withdrawal of the u4
                           result is ITSELF withdrawn, and every arm rendered
                           across an unload this session becomes suspect --
                           including another lane's duration-control arms.

Determinism across ordinary submissions is already established three times over
(identical md5 on separate posts of the same graph, hours and unrelated jobs
apart). A full unload is the case that has NOT been tested.

## Idempotent on purpose

Run it any number of times. If the render is already on disk it grades and
exits; only a missing render causes a post. That is deliberate: this was
originally chained behind another script, and a chained process does not
outlive the session that started it. Posting directly and grading later
survives the session ending, which is the failure this shape avoids.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import run_audio_carry_arms as R  # noqa: E402

SEED, TAG, ARM = 111222333, "_reload", "u4_off"

#: Measured before the unload, in the same run that produced the seed spread.
#: Not recomputed here: the point is to compare against what was recorded.
PRE_UNLOAD_RMS = 0.016280


def existing():
    d = R.OUT / R.SUB
    hits = sorted(d.glob(f"{ARM}{TAG}_a_*.flac"))
    return hits[0] if hits else None


def main() -> int:
    path = existing()
    if path is None:
        print(f"no {ARM}{TAG} render on disk; posting it")
        try:
            g = R.build(ARM, seed=SEED, tag=TAG)
            pid = R.post("prompt", {"prompt": g,
                                    "client_id": f"reload-{SEED}"})["prompt_id"]
        except Exception as e:                       # noqa: BLE001
            print(f"could not post: {e}")
            return 1
        print(f"posted {pid}. It renders behind whatever else is queued.")
        print("Re-run this script once it lands and it will grade and record.")
        return 0

    x = R.audio(path)
    rms = float(x.std())
    delta_db = float(20.0 * np.log10(rms / PRE_UNLOAD_RMS))
    # Exact equality is the question, so the tolerance is a float32 PCM epsilon
    # rather than a judgement call about what counts as close.
    invariant = abs(rms - PRE_UNLOAD_RMS) < 1e-6

    print(f"file            {path.name}")
    print(f"post-unload rms {rms:.6f}")
    print(f"pre-unload  rms {PRE_UNLOAD_RMS:.6f}")
    print(f"delta           {delta_db:+.4f} dB")
    print()
    if invariant:
        print("RELOAD-INVARIANT. The seed spread in "
              "2026-08-28_audio_seed_spread.json is real, and the withdrawal "
              "of the u4 carry result stands.")
    else:
        print("NOT INVARIANT: a full model unload changes the render. The seed "
              "spread is CONFOUNDED, the withdrawal of the u4 carry result is "
              "itself withdrawn, and every arm rendered across an unload this "
              "session needs re-checking -- other lanes' included.")

    out = REPO / "bench/results/2026-08-28_reload_invariance.json"
    out.write_text(json.dumps({
        "date": "2026-08-28",
        "script": "bench/check_reload_invariance.py",
        "question": "is a render bit-identical across a full model unload?",
        "arm": ARM, "seed": SEED,
        "pre_unload_rms": PRE_UNLOAD_RMS,
        "post_unload_rms": rms,
        "delta_db": delta_db,
        "reload_invariant": invariant,
        "consequence": (
            "the seed spread in bench/results/2026-08-28_audio_seed_spread.json "
            "is real and the u4 withdrawal stands" if invariant else
            "the seed spread is confounded and the u4 withdrawal is itself "
            "withdrawn; arms rendered across an unload need re-checking"),
        "do_not_rely_on": [
            "ONE re-render of ONE seed. It answers whether an unload changed "
            "THIS render, not whether unloads are harmless in general.",
            "Audio energy only. A render could differ in ways rms does not "
            "see; a byte comparison against the original file would be "
            "stronger and is not done here because the original was not kept "
            "under a stable name.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
