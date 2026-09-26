#!/usr/bin/env python3
"""`MiniMaxH3VAEPrecision` refuses to cast a quantized half, and only that.

Since 0.151.0 the shipped video VAE (`h3_config.MODELS["video_vae"]`) is the
INT8 ConvRot build, whose decoder holds core's QuantizedTensor weights. They
report their logical dtype, so nothing in the node could tell they were
quantized, and `decoder="fp32"` would have cast them untested.

Cases, on CPU (ComfyUI's `--cpu`), loading the real files:
  refuses_quantized   the INT8 file with decoder="fp32" raises ValueError
  allows_plain_half   the INT8 file with encoder="fp32" runs: its encoder is
                      not quantized, so the refusal must not over-reach
  control_fp16        the fp16 file with decoder="fp32" runs: the refusal
                      keys on quantization, not on the file

    <comfy-venv-python> bench/check_vae_precision_quantized.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]


def main() -> int:
    sys.path.insert(0, str(COMFY))
    sys.path.insert(1, str(REPO))
    sys.path.insert(2, str(REPO / "workflows"))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import comfy.sd
    import comfy.utils
    import folder_paths
    from h3_config import MODELS, VIDEO_VAE_FP16
    from vae_precision import MiniMaxH3VAEPrecision, _is_quantized

    def load(name):
        path = folder_paths.get_full_path("vae", name)
        if path is None:
            return None
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(path))

    results = []

    def case(name, ok, detail):
        results.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'}  {name:<18} {detail}")

    shipped = MODELS["video_vae"]
    int8 = load(shipped)
    if int8 is None or not _is_quantized(int8.first_stage_model.decoder):
        print(f"SKIP: {shipped} is not on disk or its decoder is not quantized")
        return 2
    try:
        MiniMaxH3VAEPrecision.execute(int8, encoder="unchanged", decoder="fp32")
        case("refuses_quantized", False, "decoder=fp32 on the INT8 decoder ran")
    except ValueError as exc:
        case("refuses_quantized", True, str(exc)[:80])

    int8 = load(shipped)
    try:
        MiniMaxH3VAEPrecision.execute(int8, encoder="fp32", decoder="unchanged")
        case("allows_plain_half", True, "encoder=fp32 on the unquantized encoder ran")
    except ValueError as exc:
        case("allows_plain_half", False, str(exc)[:80])

    fp16 = load(VIDEO_VAE_FP16)
    if fp16 is None:
        print(f"  skip  control_fp16       {VIDEO_VAE_FP16} is not on disk")
    else:
        try:
            MiniMaxH3VAEPrecision.execute(fp16, encoder="unchanged", decoder="fp32")
            case("control_fp16", True, "decoder=fp32 on the fp16 file ran")
        except ValueError as exc:
            case("control_fp16", False, str(exc)[:80])

    print(f"\n{sum(results)} of {len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
