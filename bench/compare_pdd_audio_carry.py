"""Is core's audio carry an error source under PDD? CPU, fp64, no model.

Core carries H3's audio on the video schedule (comfy/ldm/minimax/model.py,
MiniMaxH3Model.forward: carry = sigma_a / sigma_v before _forward, the velocity
rescaled after it with sigma_a at the call's sigma; comfy/model_base.py,
MiniMaxH3.process_latent_out divides by audio_scale at the end). The vendor and
sglang step audio on its own shift-3 schedule. docs/research/pdd/audio_under_pdd.md
section 1 argued that evaluating the transform at a PDD block's START sigma
while the fused head returns a block MEAN costs error that grows with width.

This drives both integrators with the SAME velocity at every block start, over
several partitions, and compares the final audio. Then a negative control
moves core's B coefficient off the block start, which must show an error.

Record: bench/results/2026-09-25_upstream_pdd_comparison.md.
Scope: Euler at eta 0, evaluation at the scheduled sigmas, unmasked audio.
"""
import os

import torch
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
SV, SA = 12.0, 3.0; s = SV/SA
def shifted(sh, b): return sh*b/(1+(sh-1)*b)
def tss(sig, f, t):  # core time_shift_sigma
    b = sig/(f + sig*(1-f)); return t*b/(1+(t-1)*b)
def run(knots, N=32, D=4096):
    g = torch.linspace(1,0,N+1)
    sv = shifted(SV, g[knots]); sa = shifted(SA, g[knots])
    x0 = torch.randn(D); eps = torch.randn(D)
    # vendor: audio on its own schedule
    xa = eps.clone()                    # sigma_a = 1 at start
    # comfy: carried variable y = (sigma_v/sigma_a) x_a ; at sigma 1 carry = 1
    y = eps.clone()
    worst = 0.0
    for i in range(len(knots)-1):
        # an arbitrary "network" velocity, a function of the clean-audio input only (same for both)
        xa_seen_vendor = xa
        carry = tss(sv[i], SV, SA)/sv[i]
        xa_seen_comfy = y*carry
        worst = max(worst, float((xa_seen_vendor-xa_seen_comfy).abs().max()))
        v = torch.tanh(xa_seen_vendor*0.7) + 0.3*torch.randn(D)   # same v fed to both
        xa = xa + (sa[i+1]-sa[i])*v
        sa_i = tss(sv[i], SV, SA)
        out = (1-s)*(y*carry) + (1+(s-1)*sa_i)*v
        y = y + (sv[i+1]-sv[i])*out
    final_comfy = y/s     # at sigma 0 the carry sigma_v/sigma_a -> 1/s ... y = s*x0-form, x_a = y/s
    return worst, float((final_comfy-xa).abs().max()), float(xa.abs().max())
for name,kn in [('u8',list(range(0,33,4))),('u4',list(range(0,33,8))),('tail6',[0,8,16,20,24,28,32]),('opt4',[0,28,30,31,32]),('u2',[0,16,32]),('u1',[0,32])]:
    w, e, mag = run(kn)
    print(f'{name:6s} max|input diff| {w:.2e}  max|final audio diff| {e:.2e}  (|x_a| max {mag:.2f})')

# ---- identities behind the result, and a JSON record
import json
from pathlib import Path
b = torch.linspace(0.001, 1, 1000); v = shifted(SV, b); a = shifted(SA, b); c = v / a
h = 1e-7
dadv = (tss(v + h, SV, SA) - tss(v - h, SV, SA)) / (2 * h)
rec = {
    "claim": "comfy's single-schedule audio carry + one Euler step per block == two-schedule audio Euler (vendor/sglang/T8 dual-clock) for any partition",
    "identities": {
        "carry c = sigma_v/sigma_a is affine in sigma_v: max|c - (s + (1-s) sigma_v)|": float((c - (s + (1 - s) * v)).abs().max()),
        "c * dsigma_a/dsigma_v == 1 + (s-1) sigma_a (core's B), max err (finite diff)": float((c * dadv - (1 + (s - 1) * a)).abs()[5:-5].max()),
        "so y = c*x_a = s(1-sigma_v) x0 + sigma_v eps is a straight flow path in sigma_v": True,
    },
    "end_to_end": {},
    "scope": "euler/eta0, model evaluated at the scheduled sigmas, unmasked audio; final unscale is comfy/model_base.py::MiniMaxH3.process_latent_out (divides audio by audio_scale)",
}
for name, kn in [('u8', list(range(0, 33, 4))), ('u4', list(range(0, 33, 8))), ('tail6', [0, 8, 16, 20, 24, 28, 32]),
                 ('opt4', [0, 28, 30, 31, 32]), ('u2', [0, 16, 32]), ('u1', [0, 32]), ('u32', list(range(33)))]:
    w, e, mag = run(kn)
    rec["end_to_end"][name] = {"max_input_diff": w, "max_final_audio_diff": e, "max_abs_x_a": mag}

print(json.dumps(rec["identities"], indent=1))

# ---- negative control: B evaluated away from the block start

torch.set_default_dtype(torch.float64); torch.manual_seed(0)
SV, SA = 12.0, 3.0; s = SV / SA
def shifted(sh, b): return sh*b/(1+(sh-1)*b)
def tss(sig, f, t):
    b = sig/(f + sig*(1-f)); return t*b/(1+(t-1)*b)
def run_control(knots, mode, D=4096):
    g = torch.linspace(1, 0, 33); sv = shifted(SV, g[knots]); sa = shifted(SA, g[knots])
    eps = torch.randn(D); xa = eps.clone(); y = eps.clone()
    for i in range(len(knots)-1):
        carry = tss(sv[i], SV, SA)/sv[i]
        v = torch.tanh(xa*0.7) + 0.3*torch.randn(D)
        xa = xa + (sa[i+1]-sa[i])*v
        sa_i = tss(sv[i], SV, SA); sa_n = tss(sv[i+1], SV, SA)
        B = {"core": 1+(s-1)*sa_i, "B_at_block_end": 1+(s-1)*sa_n, "B_mean": 1+(s-1)*(sa_i+sa_n)/2}[mode]
        y = y + (sv[i+1]-sv[i])*((1-s)*(y*carry) + B*v)
    return float((y/s - xa).abs().max())
out = {m: {n: run_control(k, m) for n, k in [("u8", list(range(0,33,4))), ("u4", list(range(0,33,8)))]} for m in ("core", "B_at_block_end", "B_mean")}
rec["negative_control_final_audio_diff"] = out
print(json.dumps(out, indent=1))

OUT = Path(os.environ.get("OUT", Path(__file__).resolve().parents[1] / "bench/results/2026-09-25_pdd_audio_carry.json"))
OUT.write_text(json.dumps(rec, indent=1))
