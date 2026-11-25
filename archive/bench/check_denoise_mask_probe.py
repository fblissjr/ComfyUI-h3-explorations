#!/usr/bin/env python3
"""`denoise_mask_probe.py` observes the mask and changes nothing.

CPU only, no server. Cases:
  returns_input      the probe returns the very tensor it was given, so a
                     sampler running with it takes the same steps as without
  chains_previous    a function already installed is called first and its
                     result is what the probe passes on
  per_stream_stats   a packed AV mask (video frozen at 0, audio open at 1),
                     packed by core's own `pack_latents`, is reported per
                     stream with the right zero and one fractions
  node_installs      the node clones the model and installs the probe
                     through the patcher's own setter, keeping what was there
  telemetry_parses   the log line becomes a `mask_probe` event in
                     `pipeline_telemetry.py`

    <comfy-venv-python> bench/check_denoise_mask_probe.py
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parents[1]))
sys.path.insert(1, str(REPO))

import torch  # noqa: E402

import comfy.utils  # noqa: E402
import denoise_mask_probe as dmp  # noqa: E402
import pipeline_telemetry as pt  # noqa: E402

results = []


def case(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:<18} {detail}")


def main() -> int:
    video = torch.zeros(1, 24, 3, 4, 6)
    audio = torch.ones(1, 32, 2, 5)
    packed, shapes = comfy.utils.pack_latents([video, audio])
    inner = types.SimpleNamespace(latent_shapes=shapes)
    extra = {"model": types.SimpleNamespace(inner_model=inner), "sigmas": None}

    lines = []

    class Grab(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    dmp.logger.addHandler(Grab())
    dmp.logger.setLevel(logging.INFO)

    probe = dmp.make_probe()
    out = probe(torch.tensor([0.5]), packed, extra)
    case("returns_input", out is packed, "same object back")

    calls = []

    def previous(sigma, mask, extra_options):
        calls.append(1)
        return mask * 0.5

    out2 = dmp.make_probe(previous)(torch.tensor([0.5]), packed, extra)
    case("chains_previous", calls == [1] and torch.equal(out2, packed * 0.5),
         "previous called once, its result passed on")

    line = lines[0] if lines else ""
    ok = ("video min=0 max=0 zero=1.0000 one=0.0000" in line
          and "audio min=1 max=1 zero=0.0000 one=1.0000" in line)
    case("per_stream_stats", ok, line[:110])

    class Patcher:
        def __init__(self):
            self.model_options = {"denoise_mask_function": previous}

        def clone(self):
            c = Patcher()
            c.model_options = dict(self.model_options)
            return c

        def set_model_denoise_mask_function(self, fn):
            self.model_options["denoise_mask_function"] = fn

    src = Patcher()
    res = dmp.MiniMaxH3DenoiseMaskProbe.execute(src)
    m = res.result[0] if hasattr(res, "result") else res[0]
    installed = m.model_options["denoise_mask_function"]
    calls.clear()
    installed(torch.tensor([0.5]), packed, extra)
    case("node_installs", m is not src and installed is not previous
         and src.model_options["denoise_mask_function"] is previous and calls == [1],
         "a clone carries the probe; the source is untouched; the old function still runs")

    match = next((pat.search(line) for kind, pat in pt._LOAD_PATTERNS if kind == "mask_probe"), None)
    case("telemetry_parses", bool(match), "matched by pipeline_telemetry._LOAD_PATTERNS")

    print(f"\n{sum(results)} of {len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
