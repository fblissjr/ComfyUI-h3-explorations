#!/usr/bin/env python3
"""Did core's move of the text encoder's RoPE onto comfy-kitchen change its numbers?

ComfyUI core `6cff1e97` (PR 16326, pulled into this install on 2026-09-15)
replaced `comfy/text_encoders/llama.py::apply_rope`, a torch implementation,
with a call to `comfy_kitchen.apply_rope_split_half`; `9b572343` (PR 16351)
then taught the new function 3-d inputs. H3's Qwen3-VL encoder is `Llama2_`
from that file, so every conditioning tensor since then has gone through the
new code. No render record spans the change
(`docs/sol_upstream.md`, section "comfy-kitchen and core, 2026-09-19").

This compares the two implementations at the one seam that changed: the same
bf16 q and k, the same rope tables, old function against new. It needs no
weights and no server. It does not encode a prompt; if the two are
bit-identical here, a full encode cannot differ through this seam, and if
they are not, the size of the difference says whether a full encode is worth
running.

  old  the pre-6cff1e97 body, copied below verbatim: fp32 multiply-adds
       against fp32 tables, one rounding to the input dtype at the end
  new  `comfy.text_encoders.llama.apply_rope` as installed: builds the 2x2
       rotation matrix (`rope_matrix`) and calls kitchen

The tables come from the installed `precompute_freqs_cis` with the Qwen3-VL
32B config (`Qwen3VL_32BConfig`), over position ids that include a vision
span, so the interleaved M-RoPE axes are exercised. q and k are laid out as
the encoder's attention lays them out: `[B, T, H, D]` transposed to
`[B, H, T, D]`, a non-contiguous view.

Both are also graded against an fp64 reference, so a non-identical result
says which side is closer. A mutation control (the old function with the sin
term's sign flipped) must fail the identity check, or the check proves
nothing.

On CPU kitchen dispatches its eager path; the renders use the CUDA kernel.
Run it on the GPU to test what the renders run (`--device cuda`). CPU is only
a smoke test of the script.

    python bench/probe_encoder_rope_kitchen.py --device cuda \\
        --out bench/results/2026-09-19_encoder_rope_kitchen.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402


def old_apply_rope(xq, xk, freqs_cis):
    """Verbatim from `comfy/text_encoders/llama.py` at `6cff1e97^`."""
    org_dtype = xq.dtype
    cos = freqs_cis[0]
    sin = freqs_cis[1]
    nsin = freqs_cis[2]

    q_embed = (xq * cos)
    q_split = q_embed.shape[-1] // 2
    q_embed[..., : q_split].addcmul_(xq[..., q_split :], nsin)
    q_embed[..., q_split :].addcmul_(xq[..., : q_split], sin)

    k_embed = (xk * cos)
    k_split = k_embed.shape[-1] // 2
    k_embed[..., : k_split].addcmul_(xk[..., k_split :], nsin)
    k_embed[..., k_split :].addcmul_(xk[..., : k_split], sin)

    return q_embed.to(org_dtype), k_embed.to(org_dtype)


def mutated_apply_rope(xq, xk, freqs_cis):
    """The control: sin's sign flipped. Must not match."""
    cos, sin, nsin = freqs_cis
    return old_apply_rope(xq, xk, (cos, nsin, sin))


def reference_apply_rope(xq, xk, freqs_cis):
    """fp64, no in-place ops: the value both implementations approximate."""
    cos, sin, nsin = (t.double() for t in freqs_cis)

    def rot(x):
        x = x.double()
        half = x.shape[-1] // 2
        lo, hi = x[..., :half], x[..., half:]
        out = x * cos
        return torch.cat((out[..., :half] + hi * nsin, out[..., half:] + lo * sin), dim=-1)

    return rot(xq), rot(xk)


def position_ids(n_text_before, grid, n_text_after, device):
    """(3, T) M-RoPE ids: text, one vision span (t, h, w), text.

    The vision span follows Qwen2-VL's rule: each axis carries its own grid
    coordinate offset by the running position, and text after it resumes at
    the span's largest id plus one. The exact convention matters less here
    than that the three axes differ inside the span, which is what exercises
    the interleaved sections.
    """
    t, h, w = grid
    before = torch.arange(n_text_before).expand(3, -1)
    base = n_text_before
    tt, hh, ww = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
    vision = torch.stack((tt.flatten(), hh.flatten(), ww.flatten())) + base
    resume = int(vision.max()) + 1
    after = torch.arange(resume, resume + n_text_after).expand(3, -1)
    return torch.cat((before, vision, after), dim=1).to(device)


def grade(name, got, ref, base=None):
    row = {"arm": name}
    for label, g, r in (("q", got[0], ref[0]), ("k", got[1], ref[1])):
        err = (g.double() - r).abs()
        row[f"{label}_max_abs_vs_fp64"] = float(err.max())
        row[f"{label}_rel_l2_vs_fp64"] = float(err.norm() / r.norm())
        if base is not None:
            b = base[0] if label == "q" else base[1]
            row[f"{label}_bit_identical_to_old"] = bool(torch.equal(g, b))
            row[f"{label}_elements_differing_from_old"] = int((g != b).sum())
            row[f"{label}_max_abs_vs_old"] = float((g.float() - b.float()).abs().max())
    return row


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, nargs="+", default=[1.0, 32.0],
                    help="input magnitudes to try; the larger one stresses rounding")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    saved = sys.argv
    if args.device == "cpu":
        # core's model_management picks a CUDA device at import unless told
        # --cpu; parse that one flag so a CPU smoke test never opens a context
        import comfy.options
        comfy.options.enable_args_parsing()
        sys.argv = [saved[0], "--cpu"]
    import comfy_kitchen
    from comfy.text_encoders import llama
    sys.argv = saved

    cfg = llama.Qwen3VL_32BConfig()
    device = torch.device(args.device)
    pos = position_ids(96, (2, 16, 16), 160, device)
    tables = llama.precompute_freqs_cis(cfg.head_dim, pos, cfg.rope_theta, cfg.rope_scale,
                                        cfg.rope_dims, device=device,
                                        interleaved_mrope=getattr(cfg, "interleaved_mrope", False))
    n = pos.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    rows = []
    for scale in args.scale:
        xq = (torch.randn(1, n, cfg.num_attention_heads, cfg.head_dim, generator=gen) * scale)
        xk = (torch.randn(1, n, cfg.num_key_value_heads, cfg.head_dim, generator=gen) * scale)
        xq = xq.to(device, torch.bfloat16).transpose(1, 2)
        xk = xk.to(device, torch.bfloat16).transpose(1, 2)

        ref = reference_apply_rope(xq, xk, tables)
        old = old_apply_rope(xq, xk, tables)
        new = llama.apply_rope(xq.clone(), xk.clone(), tables)
        bad = mutated_apply_rope(xq, xk, tables)
        for name, got in (("old", old), ("new", new), ("control_mutated", bad)):
            row = grade(name, got, ref, base=None if name == "old" else old)
            row["scale"] = scale
            rows.append(row)

    new_rows = [r for r in rows if r["arm"] == "new"]
    ctrl_rows = [r for r in rows if r["arm"] == "control_mutated"]
    identical = all(r["q_bit_identical_to_old"] and r["k_bit_identical_to_old"] for r in new_rows)
    control_fired = all(not (r["q_bit_identical_to_old"] and r["k_bit_identical_to_old"]) for r in ctrl_rows)

    record = {
        "question": "does kitchen's apply_rope_split_half (core 6cff1e97 / PR 16326) reproduce core's "
                    "old torch apply_rope on the Qwen3-VL encoder's q and k",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": torch.__version__,
        "comfy_kitchen": getattr(comfy_kitchen, "__version__", None),
        "python": platform.python_version(),
        "config": "Qwen3VL_32BConfig",
        "heads_q_kv_dim": [cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim],
        "positions": {"text_before": 96, "vision_grid_thw": [2, 16, 16], "text_after": 160, "total": n},
        "seed": args.seed,
        "rows": rows,
        "new_bit_identical_to_old": identical,
        "control_fired": control_fired,
    }
    text = json.dumps(record, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    if not control_fired:
        raise SystemExit("control did not fire: the identity check cannot go red, so it proves nothing")


if __name__ == "__main__":
    main()
