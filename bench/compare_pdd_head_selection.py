"""Record: bench/results/2026-09-25_upstream_pdd_comparison.md. CUDA_VISIBLE_DEVICES="" with the ComfyUI venv.

Core's PDD head selector against ours, on every schedule our shipped PDD graphs run
plus synthetic cases. CPU only; reimplemented from source, nothing imported from
coderef/ or ComfyUI.

core:  comfy/ldm/minimax/model.py::FinalLayer.forward (n > 1 branch): argmin of
       |sample_sigmas - sigma_v|, sigma_next = next entry, both mapped through
       time_shift_sigma(s, shift_v, 1.0) to base-grid indices, start clamped to n-1,
       stop >= start+1. One block for BOTH streams, from the video sigma.
ours:  pdd_lora.py::_StepTracker.observe (schedule_knots over sample_sigmas, deduped)
       + _StepTracker.update/_pick (cdist of the stream's own t_emb row against the
       boundary embeddings at the knots, nearest wins, clamped to nfe-1). Video and
       audio chosen independently, each from its own t_emb row.
t_emb: pruned checkpoint, lerp over adaln_t_table [1025, 8] exactly as
       MiniMaxH3Model._forward and pdd_lora.boundary_embeddings build it.
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
import scipy.stats
import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "workflows"))
import h3_config  # noqa: E402
CKPT = REPO.parents[1] / "models/diffusion_models" / h3_config.MODELS["unet_fl2va"]
OUT = Path(os.environ.get("OUT", REPO / "bench/results/2026-09-25_pdd_head_selection.json"))
N, SV, SA = 32, 12.0, 3.0
BOUNDARY_TOLERANCE = 1e-2                      # pdd_lora.BOUNDARY_TOLERANCE

with safe_open(CKPT, "pt") as f:
    TABLE = f.get_tensor("adaln_t_table").float()


def tss(sig, f, t):                            # core time_shift_sigma, in the input's dtype
    base = sig / (f + sig * (1.0 - f))
    return t * base / (1.0 + (t - 1.0) * base)


def temb(t):                                   # core pruned t_emb for one t (float32)
    tv = torch.tensor([t], dtype=torch.float32)
    pos = tv.clamp(0.0, 1.0) * (TABLE.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=TABLE.shape[0] - 2)
    return torch.lerp(TABLE[i0], TABLE[i0 + 1], (pos - i0).unsqueeze(1))[0]


def pdd_time_grid(shift):
    s = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
    return 1.0 - shift * s / (1 + (shift - 1) * s)


def boundary_emb(shift):                       # pdd_lora.boundary_embeddings, pruned branch
    out = []
    for t in pdd_time_grid(shift).tolist():
        pos = min(max(float(t), 0.0), 1.0) * (TABLE.shape[0] - 1)
        i0 = min(int(pos), TABLE.shape[0] - 2)
        out.append(torch.lerp(TABLE[i0].float(), TABLE[i0 + 1].float(), pos - i0))
    return torch.stack(out)


GRID_EMB = {"video": boundary_emb(SV), "audio": boundary_emb(SA)}


def schedule_knots(ss):                        # pdd_math.schedule_knots
    s = torch.as_tensor(ss).detach().to("cpu", torch.float64).flatten()
    base = s.clamp(0, 1) / (SV + s.clamp(0, 1) * (1.0 - SV))
    idx = torch.round((1.0 - base) * N)
    out = []
    for k in idx.to(torch.int64).clamp(0, N).tolist():
        if not out or k > out[-1]:
            out.append(k)
    return out


def core_pick(ss32, sigma_v):
    i = int((ss32 - sigma_v).abs().argmin())
    nxt = ss32[min(i + 1, ss32.shape[0] - 1)]
    start, stop = (round(float(1.0 - tss(s, SV, 1.0)) * N) for s in (sigma_v, nxt))
    start = min(start, N - 1); stop = max(stop, start + 1)
    return (start, stop)


def ours_pick(knots, t_row, stream):
    table = GRID_EMB[stream][knots]
    e = temb(t_row).reshape(1, -1)
    d = torch.cdist(e, table)[0]
    j = int(d.argmin()); nfe = len(knots) - 1
    j = min(j, nfe - 1)
    return (knots[j], knots[j + 1]), float(d[int(d.argmin())])


def model_calls(ss32, sampler):
    """sigma_v values the sampler evaluates the model at (comfy/k_diffusion/sampling.py)."""
    s = ss32.tolist(); calls = []
    for i in range(len(s) - 1):
        calls.append(("step", i, torch.tensor(s[i], dtype=torch.float32)))
        if sampler == "heun" and s[i + 1] != 0:
            calls.append(("heun corrector", i, torch.tensor(s[i + 1], dtype=torch.float32)))
        if sampler == "dpm_2" and s[i + 1] != 0:
            mid = math.exp((math.log(s[i]) + math.log(s[i + 1])) / 2)
            calls.append(("dpm_2 midpoint", i, torch.tensor(mid, dtype=torch.float32)))
        if sampler == "cfg_dup":
            calls.append(("second call same sigma", i, torch.tensor(s[i], dtype=torch.float32)))
    return calls


def run(name, sigmas, sampler="euler", audio_mask=None, where="synthetic"):
    ss32 = torch.as_tensor(sigmas, dtype=torch.float32)
    knots = schedule_knots(ss32)
    rows, diffs, warn = [], [], False
    for kind, i, sv in model_calls(ss32, sampler):
        sigma_v = (sv * 1000.0 / 1000.0).float().clamp(min=1e-6)        # timestep = sigma*1000, /1000
        t_v = float(1.0 - sigma_v)
        sig_a = tss(sigma_v, SV, SA)
        t_a = float(1.0 - sig_a)
        if audio_mask is not None and audio_mask < 1.0 - 1e-3:
            t_a_row = float(torch.tensor(1.0 - audio_mask * float(sig_a), dtype=torch.float32).clamp(max=max(t_a, 1.0)))
        else:
            t_a_row = t_a
        c = core_pick(ss32, sigma_v)
        ov, dv = ours_pick(knots, t_v, "video")
        oa, da = ours_pick(knots, t_a_row, "audio")
        if max(dv, da) > BOUNDARY_TOLERANCE:
            warn = True
        row = {"call": kind, "step": i, "sigma_v": round(float(sigma_v), 6), "core_both": c,
               "ours_video": ov, "ours_audio": oa, "emb_dist_v": round(dv, 5), "emb_dist_a": round(da, 5)}
        rows.append(row)
        if c != ov or c != oa:
            diffs.append(row)
    widths_ours = [b - a for a, b in zip(knots, knots[1:])]
    return {"name": name, "where": where, "sampler": sampler, "audio_mask": audio_mask,
            "sigmas": [round(float(x), 6) for x in ss32], "ours_knots": knots, "ours_widths": widths_ours,
            "n_calls": len(rows), "n_disagree": len(diffs), "ours_boundary_warning_would_fire": warn,
            "disagreements": diffs, "calls": rows}


# ---------------- schedule builders (reimplemented from core)
MS_SIGMAS = torch.tensor([SV * t / (1 + (SV - 1) * t) for t in (torch.arange(1, 1001, 1) / 1000).tolist()], dtype=torch.float32)


def simple(steps):                             # comfy/samplers.py::simple_scheduler
    ss = len(MS_SIGMAS) / steps
    return [float(MS_SIGMAS[-(1 + int(x * ss))]) for x in range(steps)] + [0.0]


def beta(steps, alpha=0.6, b=0.6):             # comfy/samplers.py::beta_scheduler
    total = len(MS_SIGMAS) - 1
    ts = 1 - np.linspace(0, 1, steps, endpoint=False)
    ts = np.rint(scipy.stats.beta.ppf(ts, alpha, b) * total)
    sigs, last = [], -1
    for t in ts:
        if t != last:
            sigs.append(float(MS_SIGMAS[int(t)]))
        last = t
    return sigs + [0.0]


def kl_optimal(n):                             # comfy/samplers.py::kl_optimal_scheduler
    smin, smax = float(MS_SIGMAS[0]), float(MS_SIGMAS[-1])
    adj = torch.arange(n, dtype=torch.float).div_(n - 1)
    s = adj.new_zeros(n + 1)
    s[:-1] = (adj * math.atan(smin) + (1 - adj) * math.atan(smax)).tan_()
    return s.tolist()


def basic_scheduler(fn, steps, denoise=1.0):   # comfy_extras/nodes_custom_sampler.py::BasicScheduler
    total = steps if denoise >= 1.0 else int(steps / denoise)
    return fn(total)[-(steps + 1):]


def pdd_emit(steps):                           # pdd_lora.emit_sigmas (uniform)
    s = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return (SV * s / (1 + (SV - 1) * s)).to(torch.float32).tolist()


def split_sigmas_denoise(sig, denoise):        # comfy_extras SplitSigmasDenoise low_sigmas
    steps = max(len(sig) - 1, 0); total = round(steps * denoise)
    return sig[-(total + 1):]


MANUAL6 = [1.0, 0.972973, 0.923077, 0.878049, 0.8, 0.631579, 0.0]   # h3_config.PDD_MANUAL_SIGMAS

cases = []
# ---- shipped: every schedule the PDD graphs under h3_config.graph_paths(..., include_bench=True)
#      ran on 2026-09-25, read from each graph's MiniMaxH3PDDLoRA, sampler and ManualSigmas nodes.
#      Re-derive before trusting this list; the record says how.
cases.append(run("PDD node SIGMAS steps=8, euler (20 graphs incl. candidates/probes)", pdd_emit(8), where="shipped"))
cases.append(run("PDD node SIGMAS steps=4, euler (10 graphs)", pdd_emit(4), where="shipped"))
cases.append(run("ManualSigmas tail6 [8,8,4,4,4,4], euler (h3_text_to_video_pdd_manual_sigmas)", MANUAL6, where="shipped"))
cases.append(run("SIGMAS steps=8 + FreezeAudio audio_mask 0.0 (3 candidate graphs)", pdd_emit(8), audio_mask=0.0, where="shipped"))
cases.append(run("SIGMAS steps=8 + AudioFreezeSong audio_mask 0.25 (3 song graphs)", pdd_emit(8), audio_mask=0.25, where="shipped"))
# ---- denoise < 1 (no shipped PDD graph uses it)
cases.append(run("SIGMAS(8) -> SplitSigmasDenoise(0.5) low", split_sigmas_denoise(pdd_emit(8), 0.5)))
_e = pdd_emit(8); _tot = round((len(_e) - 1) * 0.5)
cases.append(run("SIGMAS(8) -> SplitSigmasDenoise(0.5) high (first pass of a two-stage graph)", _e[:-_tot]))
cases.append(run("BasicScheduler(simple, 4, denoise=0.5)", basic_scheduler(simple, 4, 0.5)))
cases.append(run("BasicScheduler(simple, 8, denoise=0.5)", basic_scheduler(simple, 8, 0.5)))
cases.append(run("BasicScheduler(simple, 8, denoise=0.3)", basic_scheduler(simple, 8, 0.3)))
cases.append(run("BasicScheduler(simple, 4, denoise=0.75)", basic_scheduler(simple, 4, 0.75)))
# ---- other schedules / samplers
cases.append(run("simple 5 steps", simple(5)))
cases.append(run("simple 6 steps", simple(6)))
cases.append(run("simple 16 steps", simple(16)))
cases.append(run("beta 8 steps", beta(8)))
cases.append(run("kl_optimal 8 steps", kl_optimal(8)))
cases.append(run("simple 64 steps (finer than the grid)", simple(64)))
cases.append(run("SIGMAS(8), heun", pdd_emit(8), sampler="heun"))
cases.append(run("SIGMAS(8), dpm_2", pdd_emit(8), sampler="dpm_2"))
cases.append(run("SIGMAS(8), duplicate call per step (CFG-style)", pdd_emit(8), sampler="cfg_dup"))

OUT.write_text(json.dumps(cases, indent=1))
for c in cases:
    print(f"\n[{c['where']}] {c['name']}  sampler={c['sampler']} audio_mask={c['audio_mask']}")
    print(f"   sigmas {c['sigmas'][:12]}{' ...' if len(c['sigmas']) > 12 else ''}")
    print(f"   ours knots {c['ours_knots']} widths {c['ours_widths']}  calls {c['n_calls']}  disagree {c['n_disagree']}  ours-warns {c['ours_boundary_warning_would_fire']}")
    for d in c["disagreements"][:8]:
        print(f"     {d['call']:22s} step {d['step']:2d} sigma {d['sigma_v']:.5f}  core {d['core_both']}  ours v {d['ours_video']} a {d['ours_audio']}  dist v {d['emb_dist_v']} a {d['emb_dist_a']}")
    if len(c["disagreements"]) > 8:
        print(f"     ... {len(c['disagreements']) - 8} more")
