#!/usr/bin/env python3
"""What does promoting the H3 video VAE's encoder to fp32 actually change?

Run it with the ComfyUI venv python (`docs/comfy_notes.md`). Loads the real
video VAE; CPU is fine and no server is needed.

**Why this exists rather than a pair of clips.** Encoder precision is a
numerical knob, and `CLAUDE.md` records that a rendered clip cannot A/B one:
the trajectory diverges completely from any perturbation, on any sampler, so
the changed arm's output is a DIFFERENT SAMPLE rather than a degraded version
of the same one. `workflows/h3_probe_ref_vae_encoder_{fp16,fp32}_api.json`
price the knob and prove it runs; they cannot say which is better and are
labelled so. This file is the comparison that is controlled by construction --
same input tensor, same module, one dtype changed, measured at the call.

**It does not say fp32 is better either.** Nothing here has a reference to be
right against: the fp16 checkpoint is the only artifact on disk, so a delta is
a delta. What it answers is whether the node does anything at all, and how big
"anything" is next to a coarser dtype.

Three arms, and the two that are not the measurement are what make it readable:

  determinism   fp16 against fp16, re-encoded. Must be BIT-IDENTICAL. If it is
                not, every delta below is contaminated by nondeterminism and
                none of them mean anything.
  fp32          the measurement.
  bf16          the direction control. bf16 has fewer mantissa bits than fp16,
                so its delta from the fp16 baseline must be LARGER than fp32's.
                If it is not, this script is not measuring precision -- it is
                measuring cast noise, and the ordering would be arbitrary.

Exit 0 when both controls hold, 1 when either does not, whatever the deltas
said.

    <comfy-venv>/bin/python bench/grade_vae_encoder_precision.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO.parents[1]))         # ComfyUI root, first
sys.path.insert(1, str(_REPO))

# One fixed pixel block, so every arm encodes the same bytes. Small: this is a
# precision measurement, not a throughput one.
SEED = 0
SIZE = (128, 128)


def _dtype(module):
    for p in module.parameters():
        return p.dtype
    return None


def _fresh_vae(vae_name: str):
    import comfy.sd
    import comfy.utils
    import folder_paths
    path = folder_paths.get_full_path("vae", vae_name)
    if path is None:
        return None, None
    # Reloaded from disk per arm rather than deep-copied. The node casts the
    # module in place and the loader caches it, so sharing one object between
    # arms would let arm N-1's cast leak into arm N -- the baseline sharing
    # mutable state with the thing it measures, which CLAUDE.md rules out.
    return comfy.sd.VAE(sd=comfy.utils.load_torch_file(path)), path


def _encode(vae_name: str, encoder: str | None, pixels):
    from vae_precision import MiniMaxH3VAEPrecision
    vae, _ = _fresh_vae(vae_name)
    if vae is None:
        return None, None
    if encoder is not None:
        out = MiniMaxH3VAEPrecision.execute(vae, encoder=encoder,
                                            decoder="unchanged")
        vae = out[0] if isinstance(out, (tuple, list)) else out.result[0]
    enc_dtype = _dtype(vae.first_stage_model.encoder)
    return vae.encode(pixels), enc_dtype


def _delta(a, b):
    a, b = a.detach(), b.detach()
    d = (a.float() - b.float()).abs()
    ref = a.float().abs().mean()
    return {"max": float(d.max()), "mean": float(d.mean()),
            "relative_mean": float(d.mean() / ref) if ref > 0 else None,
            "bit_identical": bool(__import__("torch").equal(a.float(), b.float()))}


def main() -> int:
    try:
        import torch
        import comfy.sd  # noqa: F401
        import folder_paths  # noqa: F401
    except ImportError as exc:
        print(f"ComfyUI is not importable from here ({exc}); run this with the "
              f"ComfyUI venv python (see docs/comfy_notes.md)")
        return 2

    sys.path.insert(0, str(_REPO / "workflows"))
    import h3_config
    vae_name = h3_config.MODELS["video_vae"]

    torch.manual_seed(SEED)
    pixels = torch.rand(1, SIZE[1], SIZE[0], 3)

    base, base_dtype = _encode(vae_name, None, pixels)
    if base is None:
        print(f"{vae_name} is not on disk; nothing to grade")
        return 2
    print(f"video VAE: {vae_name}")
    print(f"baseline encoder dtype: {base_dtype}   latent {tuple(base.shape)}\n")

    rows = {}
    for label, enc in (("determinism", None), ("fp32", "fp32"), ("bf16", "bf16")):
        lat, dt = _encode(vae_name, enc, pixels)
        rows[label] = {"encoder_dtype": str(dt), **_delta(base, lat)}
        r = rows[label]
        rel = "n/a" if r["relative_mean"] is None else f"{r['relative_mean']:.2e}"
        print(f"  {label:<12} encoder {str(dt):<15} max {r['max']:.6f}  "
              f"mean {r['mean']:.8f}  rel {rel}  "
              f"{'identical' if r['bit_identical'] else ''}")

    det_ok = rows["determinism"]["bit_identical"]
    order_ok = rows["bf16"]["mean"] > rows["fp32"]["mean"]
    print("\n--- controls ---")
    print(f"  determinism  fp16 re-encoded is bit-identical: "
          f"{'yes' if det_ok else 'NO -- every delta above is contaminated'}")
    print(f"  direction    bf16 delta exceeds fp32 delta: "
          f"{'yes' if order_ok else 'NO -- this is not measuring precision'}")

    record = {
        "question": "does promoting the H3 video VAE encoder change the latent?",
        "vae": vae_name, "input": {"seed": SEED, "size": list(SIZE)},
        "baseline_encoder_dtype": str(base_dtype),
        "arms": rows,
        "controls": {"determinism_holds": det_ok, "direction_holds": order_ok},
        "reading": ("a delta, not a verdict: the fp16 checkpoint is the only "
                    "artifact on disk, so neither arm has a reference to be "
                    "right against"),
    }
    out = _REPO / "bench" / "results" / "2026-08-21_vae_encoder_precision.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0 if (det_ok and order_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
