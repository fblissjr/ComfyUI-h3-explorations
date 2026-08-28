#!/usr/bin/env python3
"""How much does audio ENERGY move between seeds, with nothing else changed?

## The gap this closes

`bench/run_audio_carry_arms.py` measures its ablation against a noise floor
built from `block_start` -- a ~1e-6 numerical perturbation. That floor came out
at 0.19 dB (`u4`) and 0.03 dB (`opt4`), and it bounds exactly one thing:
numerical noise. **It does not bound what a LARGE change unrelated to the
transform would do.** The ablation is a large change, so "it cleared the 1e-6
floor" is a weaker statement than it looks.

The reference distribution that actually answers it is seed variation: two
completely different samples of the same prompt, partition and canvas. If audio
energy swings a decibel between seeds anyway, then `u4`'s +1.07 dB is inside the
sampling distribution and says nothing. If it is stable to a fraction of a dB,
the ablation moved something real.

Rendered at `mode=off` so the probe is not installed at all -- this is the stock
path, varying only the seed.

## What it cannot settle

`opt4`'s +6.39 dB is far outside any plausible seed spread and this run is not
really about that arm; it is about whether `u4`'s smaller effect survives. And
a seed spread measured at ONE partition does not transfer to another, which is
why the partition is named in the output rather than assumed.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
import run_audio_carry_arms as R  # noqa: E402

#: Arbitrary but FIXED, so the run is reproducible. The first is the seed every
#: other arm in this lane uses; the rest are its neighbours in no special sense.
SEEDS = [730451892, 111222333, 424242424, 987654321]

ARM = "u4_off"


def main() -> int:
    print(f"seed spread at {ARM}: {len(SEEDS)} seeds, everything else fixed")
    print(f"length={R.LENGTH} canvas={R.CANVAS}\n")
    rows, waves = {}, {}
    for s in SEEDS:
        tag = f"_s{s}"
        res = R.run(ARM, seed=s, tag=tag)
        if res.get("error"):
            print(f"  ERROR seed {s}: {res['error'][:300]}")
            continue
        x = R.audio(res["audio"][0])
        waves[s] = x
        rows[str(s)] = {"rms": float(x.std())}
        print(f"  seed {s}: rms {x.std():.6f}", flush=True)

    if len(waves) < 2:
        print("\nnot enough successful renders to bound anything")
        return 1
    r = np.array([w.std() for w in waves.values()])
    db = 20.0 * np.log10(r / r.mean())
    spread = float(db.max() - db.min())
    sd = float(db.std(ddof=1))
    print(f"\n{'seed':<14}{'rms':>12}{'dB vs mean':>13}")
    print("-" * 39)
    for (s, w), d in zip(waves.items(), db):
        print(f"{s:<14}{w.std():>12.6f}{d:>13.2f}")
    print(f"\nfull range {spread:.2f} dB, sd {sd:.2f} dB across {len(r)} seeds")

    verdict = []
    for part, eff in (("u4", 1.07), ("opt4", 6.39)):
        verdict.append(
            f"{part} ablation {eff:+.2f} dB is "
            + ("INSIDE the seed spread -- not distinguishable from drawing a "
               "different sample" if abs(eff) <= spread else
               f"{abs(eff) / spread:.1f}x the full seed range")
            + (f" (spread measured at u4 only)" if part != "u4" else ""))
    print()
    for v in verdict:
        print(f"  {v}")

    out = REPO / "bench/results/2026-08-28_audio_seed_spread.json"
    out.write_text(json.dumps({
        "date": "2026-08-28",
        "script": "bench/measure_audio_seed_spread.py",
        "arm": ARM, "seeds": SEEDS,
        "length": R.LENGTH, "canvas": R.CANVAS,
        "grading": R.CROSS_RUN_NOTE,
        "per_seed": rows,
        "range_db": spread, "sd_db": sd,
        "verdict": verdict,
        "do_not_rely_on": [
            "Measured at u4 only. A seed spread does not transfer across "
            "partitions, and opt4 is graded against it only as a rough scale.",
            "Four seeds is a range, not a distribution. It bounds the obvious "
            "case; it does not give a confidence interval.",
            "Audio energy only. Two clips can share an RMS and sound nothing "
            "alike, and no clip here was judged by a person.",
        ],
    }, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
