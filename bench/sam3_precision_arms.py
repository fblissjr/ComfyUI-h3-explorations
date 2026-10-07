#!/usr/bin/env python3
"""Compare SAM 3.1 at float16, bfloat16 and float32 through core's own nodes, on the same frames, with controls.

Question it answers: does the compute precision core runs SAM 3.1 in change what the model segments and tracks, and where
do two precisions part? It drives core's own `SAM3_Detect` and `SAM3_VideoTrack` (`comfy_extras/nodes_sam3.py`), the calls
`subject_track.py::_sam_callables` makes, on frames decoded from a clip, and compares masks, per-person presence logits and
layer outputs between arms. Every arm loads the same weights file through core's public loader
(`comfy.sd.load_state_dict_guess_config`); only `model_options["dtype"]` and the optional input nudge change. No launch flag
is involved, so an arm is what a loader node could ship. Dynamic VRAM is set up the way `main.py` does it, because that is
the path the server loads on; `arm --cpu` is a stub that skips it.

The arms the comparison reads (tags are free text; these names carry meaning in `compare` and `render`):

    H16_a, H16_b   float16 twice, the run-to-run floor
    F32            true float32: TF32 off for matmul and cudnn, both flags logged, and a self-test that proves it
    F32_n1         float32, every input pixel moved by +1 or -1 level of 8 (fixed seed): the sensitivity floor
    B16            bfloat16, the yardstick (Meta's own code computes in bfloat16 autocast over float32 weights)
    F32_n8         float32, input moved by +-8 levels: a positive control the harness must be able to see
    C16            the other weights file (the ComfyUI-org fp16 repack) as core runs it by default

How to read a result: within the two floors (H16_a vs H16_b, F32 vs F32_n1) precision is not a lever; above the floors but no
larger than B16's difference from F32 it is a perturbation the reference tolerates; above B16's, or one-sided, name the layer
(`compare` prints the per-layer drift). If F32_n8 does not show clearly above F32_n1 the harness could not see a difference.

Subcommands (every path is an argument; bulk arrays go to --work-dir, small json summaries to --json):

    frames   decode stretches of a clip to uint8 .npy      frames --clip C --stretch calm=400:150 --work-dir W
    arm      run one arm on stretches                     arm --ckpt P --tag F32 --dtype fp32 --work-dir W --json J [--make-seeds]
    floor    detector alone, raw class logits             floor --ckpt P --tag floor_fp16 --dtype fp16 --stretch calm --work-dir W
    floor-compare  compare two floor runs, write epsilon  floor-compare --work-dir W --tags floor_fp16 floor_fp32 --md M --json J
    predict  write the marginal-decision prediction       predict --stage a|b --work-dir W --epsilon-json J ...
    compare  analyse the arms of one stretch              compare --work-dir W --stretch calm --md M --json J --csv C
    render   stacked video + chart of the arms            render --work-dir W --stretch calm --out-dir O --arms H16_a=Label ...

`arm` and `floor` need ComfyUI core importable (this file sits in bench/ of the pack, two levels under the ComfyUI root) and a
card; `--cpu` runs them on the CPU for a plumbing check. `meta_sam3/` and `coderef/` are references this work reads: nothing
shipped imports them or this tool.

Frames are the clip's own frames, no resize here; core's nodes resize to 1008 x 1008 themselves (bilinear in `SAM3_Detect`,
bicubic in the tracker's `_prep_frame`).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent

#: inherited: the pack's defaults, `subject_track.py` SUBJECT_PHRASE "person" and MAX_PEOPLE 16 through `counted()`.
PHRASE = "person:16"
#: inherited: `subject_track.py` DETECTION_THRESHOLD.
THRESHOLD = 0.5
#: reasoned: 0.01 percent of the 1008 x 1008 mask grid; a person below it does not count as having a mask.
ALIVE_MIN = 100
#: reasoned: one raw-detector sample every two seconds at 25 frames per second.
DETECT_EVERY = 50
#: reasoned: any fixed seed; the nudge for frame i is seeded NUDGE_SEED + i so every arm that nudges sees the same noise.
NUDGE_SEED = 1234
#: measured (`bench/convert_sam3_checkpoint.py` output): the one tensor core's loader refuses outside the server; neither
#: model nor text features use it, so leaving it out loses nothing the model computes.
TEXT_PROJECTION_KEY = "detector.backbone.language_backbone.encoder.text_projection"
#: inherited: the largest finite float16 value.
FP16_MAX = 65504.0
SIDE = 1008          # inherited: the trunk's image side, `comfy/ldm/sam3`
LABELS = {"H16_a": "float16 (run 1)", "H16_b": "float16 (run 2)", "F32": "true float32", "F32_n1": "float32, input nudged +-1/255",
          "B16": "bf16", "F32_n8": "float32, input nudged +-8/255", "C16": "other weights file (fp16 repack), core default"}
#: (name, first, second): each row of the pairwise table; both arms must have been run for the row to appear.
PAIRS = [("run-to-run floor", "H16_a", "H16_b"), ("sensitivity floor (1/255 nudge)", "F32", "F32_n1"),
         ("float16 vs float32", "H16_a", "F32"), ("bf16 vs float32", "B16", "F32"), ("float16 vs bf16", "H16_a", "B16"),
         ("positive control (8/255 nudge)", "F32", "F32_n8"), ("other weights file vs ours, both fp16", "C16", "H16_a")]

_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


# ---- helpers shared by the analysis commands (numpy / torch only)

def popcount(a: np.ndarray) -> np.ndarray:
    return _LUT[a].sum(axis=(-1, -2))


def areas(P: np.ndarray) -> np.ndarray:
    return np.stack([popcount(P[i]) for i in range(P.shape[0])])


def iou_series(Pa: np.ndarray, Pb: np.ndarray):
    """Per frame, per person IoU of two packed mask stacks [N,K,H,W/8]; both below ALIVE_MIN counts as agreement (1.0)."""
    n, k = Pa.shape[:2]
    out = np.ones((n, k)); aa = np.zeros((n, k), int); ab = np.zeros((n, k), int)
    for i in range(n):
        inter = popcount(Pa[i] & Pb[i]); union = popcount(Pa[i] | Pb[i])
        aa[i] = popcount(Pa[i]); ab[i] = popcount(Pb[i])
        for j in range(k):
            if aa[i, j] >= ALIVE_MIN or ab[i, j] >= ALIVE_MIN:
                out[i, j] = inter[j] / max(union[j], 1)
    return out, aa, ab


def rel_l2(a, b) -> float:
    a = a.float(); b = b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def load_arm(work: Path, tag: str, stretch: str):
    import torch
    f = work / f"arm_{tag}_{stretch}.pt"
    if not f.exists():
        return None
    return torch.load(f, weights_only=False)[stretch]      # this tool's own file: a dict of numpy arrays and small tensors


def tf32_selftest(device):
    """fp32 conv and matmul against a float64 CPU reference: relative error near 1e-6 is true fp32, near 1e-3 is TF32."""
    import torch
    if device.type != "cuda":
        return {}
    g = torch.Generator().manual_seed(7)
    # measured: a 256-channel 3x3 conv (the neck's shape) shows cuDNN's TF32 at about 3e-4; a 64-channel one does not, so the shape matters
    x = torch.randn(1, 256, 72, 72, generator=g); w = torch.randn(256, 256, 3, 3, generator=g) * 0.05
    a = torch.randn(512, 1024, generator=g); b = torch.randn(1024, 512, generator=g)
    ref_c = torch.nn.functional.conv2d(x.double(), w.double(), padding=1); ref_m = a.double() @ b.double()
    got_c = torch.nn.functional.conv2d(x.to(device), w.to(device), padding=1).double().cpu(); got_m = (a.to(device) @ b.to(device)).double().cpu()
    rel = lambda g_, r_: float((g_ - r_).abs().max() / r_.abs().max())
    return dict(conv_rel_err=rel(got_c, ref_c), matmul_rel_err=rel(got_m, ref_m))


# ---- frames

def cmd_frames(a):
    import av
    work = Path(a.work_dir); work.mkdir(parents=True, exist_ok=True)
    for spec in a.stretch:
        name, rng = spec.split("=", 1); start, count = (int(v) for v in rng.split(":"))
        c = av.open(a.clip); s = c.streams.video[0]
        s.thread_type = "AUTO"; s.codec_context.thread_count = a.threads
        fps = float(s.average_rate); t0 = s.start_time or 0
        c.seek(int(max(0, start / fps - 2.0) / s.time_base), stream=s, backward=True)
        frames, first = [], None
        for fr in c.decode(s):
            idx = round(float((fr.pts - t0) * s.time_base) * fps)
            if idx < start:
                continue
            first = idx if first is None else first
            frames.append(fr.to_ndarray(format="rgb24"))
            if len(frames) == count:
                break
        arr = np.stack(frames)
        np.save(work / f"frames_{name}.npy", arr)
        print(f"{name}: asked frame {start}, first decoded {first}, got {len(frames)} frames {arr.shape} at {fps} fps")
        if first != start:
            raise SystemExit("the first decoded frame is not the one asked for")


# ---- core, as the server loads it

def setup_core(cpu: bool, threads: int):
    """Import core the way the server's flags and main.py's dynamic-VRAM setup would; returns (torch, comfy modules, aimdo ok)."""
    sys.argv = [sys.argv[0]] + (["--cpu"] if cpu else ["--supports-fp8-compute", "--mmap-torch-files"])
    sys.path.insert(0, str(COMFY))
    sys.dont_write_bytecode = True
    import comfy.options
    comfy.options.enable_args_parsing()
    from comfy.cli_args import args
    import torch
    if threads:
        torch.set_num_threads(threads)
    ca = None
    if not cpu:
        import comfy_aimdo.control as ca
        headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
        try:
            ca.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
        except TypeError:
            try:
                ca.init(simple_vram_headroom=headroom)
            except TypeError:
                ca.init()
    import comfy.model_management as mm
    import comfy.utils, comfy.sd, comfy.memory_management, comfy.model_patcher
    ok = False
    if ca is not None:
        try:
            ok = ca.init_devices((d.index, int(args.vram_headroom * 1024 ** 3)) for d in mm.get_all_torch_devices())
        except TypeError:
            ok = ca.init_devices(d.index for d in mm.get_all_torch_devices())
        if ok:
            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
    return torch, mm, comfy, ok


def tf32_snapshot(torch):
    return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_conv_fp32_precision=getattr(torch.backends.cudnn.conv, "fp32_precision", None),
                fp16_accumulation=getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None))


def load_model(torch, comfy, ckpt: str, dtype_name, keep_text_projection: bool):
    """Core's public loader with model_options["dtype"]; the text encoder keeps core's default in every arm."""
    from safetensors import safe_open
    mo = {"dtype": {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]} if dtype_name else {}

    def load(drop):
        sd, md = comfy.utils.load_torch_file(ckpt, return_metadata=True)
        if drop:
            sd.pop(TEXT_PROJECTION_KEY, None)
        return comfy.sd.load_state_dict_guess_config(sd, output_vae=False, output_clip=True, output_model=True, model_options=mo, metadata=md)

    with safe_open(ckpt, "pt") as f:
        has = TEXT_PROJECTION_KEY in f.keys()
    note = "not in file" if not has else "accepted"
    try:
        model, clip, _, _ = load(not keep_text_projection)
        if has and not keep_text_projection:
            note = "dropped by flag"
    except RuntimeError as e:
        note = "refused: " + str(e).splitlines()[1].strip()[:160]
        model, clip, _, _ = load(True)
    return model, clip, note


def nudged(u8: np.ndarray, levels: int) -> np.ndarray:
    if levels == 0:
        return u8
    out = np.empty_like(u8)
    for i in range(u8.shape[0]):
        sign = np.random.default_rng(NUDGE_SEED + i).integers(0, 2, size=u8.shape[1:], dtype=np.int8) * 2 - 1
        out[i] = np.clip(u8[i].astype(np.int16) + sign * levels, 0, 255).astype(np.uint8)
    return out


def cmd_arm(a):
    torch, mm, comfy, ok = setup_core(a.cpu, a.threads)
    import comfy_extras.nodes_sam3 as ns
    work = Path(a.work_dir)
    before = tf32_snapshot(torch)
    selftest_default = tf32_selftest(mm.get_torch_device())          # the flags as PyTorch ships them: shows the control can see TF32
    if a.tf32 == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    info = dict(tag=a.tag, ckpt=Path(a.ckpt).name, dtype_option=a.dtype, nudge=a.nudge, tf32_mode=a.tf32, tf32_before=before,
                tf32_after_set=tf32_snapshot(torch), tf32_selftest_default=selftest_default, tf32_selftest=tf32_selftest(mm.get_torch_device()), aimdo=bool(ok),
                patcher=comfy.model_patcher.CoreModelPatcher.__name__, cpu=a.cpu)
    t0 = time.time()
    model, clip, info["text_projection_load"] = load_model(torch, comfy, a.ckpt, a.dtype, keep_text_projection=True)
    mm.load_model_gpu(model)
    device = mm.get_torch_device(); dtype = model.model.get_dtype(); dm = model.model.diffusion_model; cuda = device.type == "cuda"
    info.update(get_dtype=str(dtype), manual_cast=str(model.model.manual_cast_dtype), load_s=round(time.time() - t0, 1))
    print(f"[{a.tag}] {info}", flush=True)
    sync = (lambda: torch.cuda.synchronize()) if cuda else (lambda: None)
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(PHRASE))
    emb, tmask, _ = ns._extract_text_prompts(cond, device, dtype)[0]
    text_feat = cond[0][0].detach().float().cpu()

    calls = collections.defaultdict(list)

    def sketch(o):
        if isinstance(o, dict):
            o = next(v for v in o.values() if torch.is_tensor(v))
        t = (o[0] if isinstance(o, (tuple, list)) else o).detach().float()
        flat = t.reshape(-1); k = max(1, flat.numel() // 8192)
        return dict(norm=float(t.norm()), amax=float(t.abs().max()), sketch=flat[::k][:8192].half().cpu(), shape=tuple(t.shape))

    def dec_pick(o):
        return dict(iou=o[1].detach().float().cpu(), obj=o[3].detach().float().cpu())

    def hook_for(name, fn):
        def h(_mod, _inp, out):
            try:
                calls[name].append(fn(out))
            except Exception as e:      # a layer whose output shape differs from what the sketch expects must not stop the run
                calls[name + "_err"].append(repr(e))
        return h

    tr = dm.tracker; hooks = []
    for nm, mod, fn in (("dec_prop", getattr(tr, "sam_mask_decoder", None), dec_pick), ("dec_inter", getattr(tr, "interactive_sam_mask_decoder", None), dec_pick),
                        ("ptr_prop", getattr(tr, "obj_ptr_proj", None), sketch), ("ptr_inter", getattr(tr, "interactive_obj_ptr_proj", None), sketch),
                        ("mem_attn", getattr(getattr(tr, "transformer", None), "encoder", None), sketch), ("mem_enc", getattr(tr, "maskmem_backbone", None), sketch)):
        if mod is not None:
            hooks.append(mod.register_forward_hook(hook_for(nm, fn)))
    trunk = dm.detector.backbone["vision_backbone"].trunk
    blk_max = torch.zeros(len(trunk.blocks), device=device)

    def blk_hook(i):
        def h(_mod, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            blk_max[i] = torch.maximum(blk_max[i], o.detach().abs().max().float())
        return h
    for i, b in enumerate(trunk.blocks):
        hooks.append(b.register_forward_hook(blk_hook(i)))

    pack = lambda t: np.packbits(t.cpu().numpy().astype(bool), axis=-1)
    summary = dict(info=info, stretches={})
    for name in a.stretches:
        frames_np = np.load(work / f"frames_{name}.npy", mmap_mode="r")
        n = frames_np.shape[0] if not a.smoke else 2
        clean = np.ascontiguousarray(frames_np[:n]); H, W = clean.shape[1], clean.shape[2]
        R = dict(name=name, n=n, H=H, W=W, info=info)
        sp = work / f"seeds_{name}.npz"
        if a.make_seeds or a.seeds_only:
            fr0 = torch.from_numpy(clean[0:1]).float().div_(255.0)
            if a.smoke:         # plumbing stub: two rectangles; the node refines every person with extra trunk passes
                masks0 = torch.zeros(2, H, W); masks0[0, 200:700, 300:700] = 1; masks0[1, 300:800, 1000:1400] = 1
                R["seed_scores"] = [1.0, 1.0]
            else:
                sync(); t = time.perf_counter()
                out = ns.SAM3_Detect.execute(model, fr0, conditioning=cond, threshold=THRESHOLD, individual_masks=True)
                sync(); R["seed_detect_s"] = time.perf_counter() - t
                masks0, boxes0 = getattr(out, "args", out)[:2]
                R["seed_scores"] = [float(b.get("score", 0)) for b in (boxes0[0] if boxes0 else [])]
            np.savez(sp, masks=pack(masks0 > 0.5), scores=np.array(R["seed_scores"]))
            print(f"[{a.tag}] {name}: seeds made on the clean first frame: {masks0.shape[0]} people", flush=True)
            if a.seeds_only:
                continue
        z = np.load(sp); seeds = torch.from_numpy(np.unpackbits(z["masks"], axis=-1)[..., :W]).float(); K = seeds.shape[0]
        R["K"] = K
        frames = torch.from_numpy(nudged(clean, a.nudge)).float().div_(255.0)        # [N,H,W,3] 0..1 like a ComfyUI IMAGE
        print(f"[{a.tag}] {name}: {n} source frames {H}x{W}, nudge {a.nudge}, {K} people seeded", flush=True)
        if K == 0:
            continue
        det, t_raw, t_node = [], [], []
        for fi in ([0] if a.smoke else sorted(set(list(range(0, n, DETECT_EVERY)) + [n - 1]))):
            img = comfy.utils.common_upscale(frames[fi:fi + 1][..., :3].movedim(-1, 1), SIDE, SIDE, "bilinear", crop="disabled").to(device=device, dtype=dtype)
            with torch.no_grad():
                sync(); t = time.perf_counter()
                res = dm(img, text_embeddings=emb, text_mask=tmask, boxes=None, threshold=THRESHOLD, orig_size=None)
                sync(); t_raw.append(time.perf_counter() - t)
            sc = res["scores"][0].detach().float(); pr = res["presence"].float().flatten()[0] if res.get("presence") is not None else None
            det.append(dict(frame=fi, presence=float(pr) if pr is not None else None, scores=sc.cpu()))
        for fi in ([] if a.smoke else sorted({0, n // 2, n - 1})):
            sync(); t = time.perf_counter()
            ns.SAM3_Detect.execute(model, frames[fi:fi + 1], conditioning=cond, threshold=THRESHOLD, individual_masks=True)
            sync(); t_node.append(time.perf_counter() - t)
        R.update(detect=det, detect_raw_s=t_raw, detect_node_s=t_node, detect_peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None)
        for k in list(calls):
            calls[k].clear()
        blk_max.zero_()
        if cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        R["tf32_at_track"] = tf32_snapshot(torch)
        sync(); t = time.perf_counter()
        out = ns.SAM3_VideoTrack.execute(frames, model, initial_mask=seeds, conditioning=None, detection_threshold=THRESHOLD, max_objects=K, detect_interval=1)
        sync(); R["track_s"] = time.perf_counter() - t
        R["track_peak_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
        data = getattr(out, "args", out)[0]
        R["packed_masks"] = data["packed_masks"].cpu().numpy() if data["packed_masks"] is not None else None   # [N,K,Hm,Wm/8]
        R["calls"] = dict(calls); R["trunk_block_absmax"] = blk_max.cpu().tolist()
        al = (areas(R["packed_masks"]) >= ALIVE_MIN).sum(axis=1).tolist() if R["packed_masks"] is not None else []
        summary["stretches"][name] = dict(n=n, K=K, track_s=R["track_s"], ms_per_frame=R["track_s"] / n * 1000, track_peak_gib=R["track_peak_gib"],
                                          detect_raw_s=float(np.mean(t_raw)), detect_node_s=float(np.mean(t_node)) if t_node else None,
                                          detect_peak_gib=R["detect_peak_gib"], people_with_mask_per_frame=al)
        print(f"[{a.tag}] {name}: tracked {n} frames x {K} people in {R['track_s']:.1f}s ({R['track_s'] / n * 1000:.0f} ms/frame), "
              f"peak {R['track_peak_gib']} GiB; tf32 at track {R['tf32_at_track']}", flush=True)
        torch.save({name: R, "__text_feat__": text_feat}, work / f"arm_{a.tag}_{name}{'_smoke' if a.smoke else ''}.pt")
        del frames, out, data
        if cuda:
            torch.cuda.empty_cache()
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1, default=str))


# ---- the detector alone

def cmd_floor(a):
    torch, mm, comfy, _ = setup_core(a.cpu, a.threads)
    import comfy_extras.nodes_sam3 as ns
    work = Path(a.work_dir)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    flags = tf32_snapshot(torch)
    print(f"[{a.tag}] TF32 self-test: {tf32_selftest(mm.get_torch_device())}", flush=True)
    model, clip, _ = load_model(torch, comfy, a.ckpt, a.dtype, keep_text_projection=False)
    mm.load_model_gpu(model)
    device = mm.get_torch_device(); dtype = model.model.get_dtype(); dm = model.model.diffusion_model
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(PHRASE))
    emb, tmask, _ = ns._extract_text_prompts(cond, device, dtype)[0]
    fr_np = np.load(work / f"frames_{a.stretch}.npy", mmap_mode="r")
    idx = np.unique(np.linspace(0, fr_np.shape[0] - 1, a.nframes).round().astype(int))
    S, P, B, T = [], [], [], []
    for i in idx:
        fr = torch.from_numpy(np.ascontiguousarray(fr_np[i])).float().div_(255.0)[None]
        img = comfy.utils.common_upscale(fr[..., :3].movedim(-1, 1), SIDE, SIDE, "bilinear", crop="disabled").to(device=device, dtype=dtype)
        with torch.no_grad():
            t0 = time.perf_counter(); r = dm(img, text_embeddings=emb, text_mask=tmask, boxes=None, threshold=THRESHOLD, orig_size=None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            T.append(time.perf_counter() - t0)
        S.append(r["scores"][0].detach().float().cpu().numpy()); P.append(float(r["presence"].float().flatten()[0])); B.append(r["boxes"][0].detach().float().cpu().numpy())
    np.savez(work / f"floor_{a.tag}.npz", frames=idx, scores=np.stack(S), presence=np.array(P), boxes=np.stack(B), secs=np.array(T), flags=str(flags), dtype=str(dtype))
    print(f"[{a.tag}] {a.stretch}: {len(idx)} frames, dtype {dtype}, mean {np.mean(T[1:] or T):.3f}s per raw detect (first call excluded)", flush=True)


def box_iou(x, y) -> float:
    ix = max(0.0, min(x[2], y[2]) - max(x[0], y[0])); iy = max(0.0, min(x[3], y[3]) - max(x[1], y[1]))
    inter = ix * iy; union = (x[2] - x[0]) * (x[3] - x[1]) + (y[2] - y[0]) * (y[3] - y[1]) - inter
    return inter / max(union, 1e-9)


def cmd_floor_compare(a):
    """The per-query difference is dominated by swaps between near-duplicate queries (two queries trade scores), so the set-level
    differences are what predict.py uses: the sorted top-30 logits per frame and the detections left unmatched by box IoU."""
    work = Path(a.work_dir)
    D = {t: np.load(work / f"floor_{t}.npz") for t in a.tags}
    first, ref_tag = a.tags[0], a.tags[1]
    sig = lambda z: 1 / (1 + np.exp(-z))
    out = ["# Detector alone, no refinement pass: raw query class logits on the same frames, true float32 as the reference\n",
           f"Frames: {len(D[first]['frames'])} spread over the stretch (indices {D[first]['frames'].tolist()}), 200 queries each. TF32 flags: "
           + "; ".join(f"{t}: {D[t]['flags']}" for t in a.tags), "",
           "| comparison | per-query max abs d | per-query 99th pct | per-query median | sorted top-30 logits, max abs d | presence logit max abs d | queries crossing prob 0.5 | detections above 0.5 (ref / other) | unmatched by box IoU 0.5 (ref / other) | matched boxes mean 1-IoU |", "|---|---|---|---|---|---|---|---|---|---|"]
    doc = {}; ref = D[ref_tag]
    for t in a.tags:
        if t == ref_tag:
            continue
        d = np.abs(D[t]["scores"] - ref["scores"]); sa, sb = sig(D[t]["scores"]), sig(ref["scores"])
        s_other = -np.sort(-D[t]["scores"], axis=1); s_ref = -np.sort(-ref["scores"], axis=1); set_d = float(np.abs(s_other[:, :30] - s_ref[:, :30]).max())
        n_ref = n_oth = um_ref = um_oth = 0; one_minus = []
        for f in range(sa.shape[0]):
            ko = np.where(sa[f] > 0.5)[0]; kr = np.where(sb[f] > 0.5)[0]; Bo, Br = D[t]["boxes"][f], ref["boxes"][f]
            n_ref += len(kr); n_oth += len(ko)
            um_ref += sum(1 for j in kr if max([box_iou(Br[j], Bo[i]) for i in ko] or [0.0]) < 0.5)
            um_oth += sum(1 for i in ko if max([box_iou(Bo[i], Br[j]) for j in kr] or [0.0]) < 0.5)
            one_minus += [1 - m for j in kr if (m := max([box_iou(Br[j], Bo[i]) for i in ko] or [0.0])) >= 0.5]
        out.append(f"| {t} vs {ref_tag} | {float(d.max()):.4f} | {np.quantile(d, .99):.4f} | {np.median(d):.5f} | {set_d:.4f} | {np.abs(D[t]['presence'] - ref['presence']).max():.4f} | "
                   f"{int(((sa > 0.5) != (sb > 0.5)).sum())} of {sa.size} | {n_ref} / {n_oth} | {um_ref} / {um_oth} | {np.mean(one_minus):.5f} |")
        if t == first:
            doc = dict(eps=set_d, eps_set=set_d, eps_query_max=float(d.max()), eps_query_p99=float(np.quantile(d, .99)), eps_query_median=float(np.median(d)),
                       kept_ref=n_ref, kept_other=n_oth, unmatched_ref=um_ref, unmatched_other=um_oth, pair=f"{first} vs {ref_tag}")
    out.append(f"\nDetections above probability 0.5 per frame (reference): {(sig(ref['scores']) > 0.5).sum(axis=1).tolist()}. Many are near-duplicate queries on one person (core applies no NMS; the node keeps the top 16 by score).")
    out.append("Seconds per raw detector call (first call excluded): " + "; ".join(f"{t}: {np.mean(D[t]['secs'][1:]):.3f}" for t in a.tags))
    Path(a.md).write_text("\n".join(out) + "\n"); print("\n".join(out))
    Path(a.json).write_text(json.dumps(doc))


# ---- the written prediction

def cmd_predict(a):
    """eps is the largest one-step difference at the set level (sorted top-30 class logits, float16 vs float32, detector alone). A marginal
    presence-gate decision is a person-frame whose presence logit lies within eps of 0. The per-query variants (99th percentile, maximum, which
    includes swaps between near-duplicate queries) are printed beside it as the wider bounds."""
    import torch
    work = Path(a.work_dir); doc = json.loads(Path(a.epsilon_json).read_text())
    eps = doc["eps_set"]; wide = [("typical (per-query median)", doc["eps_query_median"]), ("set-level maximum (primary)", eps), ("per-query 99th pct", doc["eps_query_p99"]), ("per-query maximum", doc["eps_query_max"])]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def gate(tag, stretch):
        R = load_arm(work, tag, stretch); K = R["K"]
        return torch.cat([c["obj"].reshape(-1)[:K] for c in R["calls"]["dec_prop"]]), K, R["n"]
    if a.stage == "a":
        g, K, n = gate(a.calm_tag, a.calm_stretch); pf = a.hard_people * (a.hard_frames - 1)
        lines = [f"# Prediction A, written {now}, before any tracked arm of the hard stretch", "",
                 f"One-step differences of the detector alone, float16 vs true float32 ({doc['pair']}): " + "; ".join(f"{k} {v:.4f}" for k, v in wide) + ".",
                 f"Calm stretch float16 run `{a.calm_tag}`: {g.numel()} person-frames. Hard stretch: {a.hard_people} people x {a.hard_frames - 1} propagated frames = {pf} person-frames.", "",
                 "| eps | share of calm person-frames with abs presence logit < eps | predicted marginal gate decisions on the hard stretch |", "|---|---|---|"]
        for k, v in wide:
            rho = float((g.abs() < v).float().mean()); lines.append(f"| {k} {v:.4f} | {rho:.5f} | {rho * pf:.1f} |")
        lines += ["", f"PRIMARY PREDICTION (set-level eps {eps:.4f}): {float((g.abs() < eps).float().mean()) * pf:.1f} marginal presence-gate decisions on the hard stretch."]
    else:
        g, K, n = gate(a.hard_tag, a.hard_stretch)
        lines = [f"# Prediction B, written {now}, after the run `{a.hard_tag}` of the hard stretch and before its other arms", "",
                 "| eps | person-frames of that run with abs presence logit < eps (of " + str(g.numel()) + ") |", "|---|---|"]
        for k, v in wide:
            lines.append(f"| {k} {v:.4f} | {int((g.abs() < v).sum())} |")
        lines += ["", f"PRIMARY PREDICTION (set-level eps {eps:.4f}): {int((g.abs() < eps).sum())} marginal presence-gate decisions on the hard stretch."]
    Path(a.md).write_text("\n".join(lines) + "\n"); print("\n".join(lines))


# ---- the comparison

def cmd_compare(a):
    import torch
    work = Path(a.work_dir); stretch = a.stretch
    arms = {t: R for t in LABELS if (R := load_arm(work, t, stretch)) is not None}
    pairs = [p for p in PAIRS if p[1] in arms and p[2] in arms]
    if a.first:      # only the first N frames: the masks, and the propagation-decoder calls of frames 1..N-1
        for R in arms.values():
            R["packed_masks"] = R["packed_masks"][:a.first]; R["n"] = a.first
            for k in ("dec_prop", "ptr_prop", "mem_attn"):
                R["calls"][k] = R["calls"][k][:a.first - 1]
    out = []
    P = lambda *x: (print(*x), out.append(" ".join(str(v) for v in x)))
    R0 = arms[next(iter(arms))]; n, K = R0["n"], R0["K"]
    P(f"# Stretch `{stretch}`: {n} frames" + (f" (the first {a.first} of the stretch only)" if a.first else "") + f", {K} people seeded on frame 0\n")
    P("The frames are the clip's own, no resize here; core's nodes resize to 1008x1008 themselves. The seeds are core's SAM3_Detect on the clean first frame, the same masks for every arm.\n")
    P("How to read: within the two floors (float16 run-to-run, float32 input nudge) precision is not a lever; above the floors but no larger than bf16's difference from float32, it is a perturbation the reference tolerates; above bf16's, or one-sided, name the layer. The 8/255 nudge is the positive control.\n")
    P(f"A person has a mask when it holds at least {ALIVE_MIN} pixels (of 1008x1008). IoU is per person per frame at 1008x1008; both below the threshold counts as agreement.\n")
    P("## Arms, settings and cost\n")
    P("| arm | what | get_dtype | TF32 matmul / cudnn at tracking | TF32 self-test conv / matmul rel. error | text_projection key | detect raw s | detect node s | track ms/frame | track peak GiB | detect peak GiB |\n|---|---|---|---|---|---|---|---|---|---|---|")
    nan = float("nan")
    for t, R in arms.items():
        i = R["info"]; tf = R["tf32_at_track"]; st = i.get("tf32_selftest") or {}
        P(f"| {t} | {LABELS[t]} | {i['get_dtype']} | {tf['matmul_allow_tf32']} / {tf['cudnn_allow_tf32']} | {st.get('conv_rel_err', nan):.1e} / {st.get('matmul_rel_err', nan):.1e} (flags as shipped: {(i.get('tf32_selftest_default') or {}).get('conv_rel_err', nan):.1e} / {(i.get('tf32_selftest_default') or {}).get('matmul_rel_err', nan):.1e}) | {i['text_projection_load'][:24]} | "
          f"{np.mean(R['detect_raw_s'][1:] or R['detect_raw_s']):.3f} | {np.mean(R['detect_node_s']) if R['detect_node_s'] else nan:.2f} | {R['track_s'] / n * 1000:.0f} | {R['track_peak_gib'] or nan:.2f} | {R['detect_peak_gib'] or nan:.2f} |")
    Pk = {t: R["packed_masks"] for t, R in arms.items()}
    ar = {t: areas(Pk[t]) for t in arms}; alive = {t: (ar[t] >= ALIVE_MIN).sum(axis=1) for t in arms}
    P("\n## People with a mask per frame (every 25th frame and the last)\n")
    cols = list(range(0, n, 25)) + ([n - 1] if (n - 1) % 25 else [])
    P("| frame | " + " | ".join(arms) + " |\n|---|" + "---|" * len(arms))
    for f in cols:
        P(f"| {f} | " + " | ".join(str(int(alive[t][f])) for t in arms) + " |")
    P("\nFirst frame with nobody having a mask: " + "; ".join(f"{t}: {int(np.argmax(alive[t] == 0)) if (alive[t] == 0).any() else 'never'}" for t in arms))
    P("\n## Pairwise agreement of the masks\n")
    P("| pair | arms | bit-identical masks | mean IoU | median | p5 | person-frames IoU<0.9 | IoU<0.5 | alive-state differs | first frame any IoU<0.5 | people departing | area ratio first/second | frames first has more / fewer people |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    IOU, summary = {}, {}
    for name, x, y in pairs:
        iou, aa, ab = iou_series(Pk[x], Pk[y]); IOU[(x, y)] = iou
        sel = (aa >= ALIVE_MIN) | (ab >= ALIVE_MIN); v = iou[sel] if sel.any() else np.array([1.0]); both = (aa >= ALIVE_MIN) & (ab >= ALIVE_MIN)
        ratio = float(np.mean(aa[both] / np.maximum(ab[both], 1))) if both.any() else nan
        low = np.where((iou < 0.5).any(axis=1))[0]
        summary[name] = dict(arms=[x, y], bit_identical=bool(np.array_equal(Pk[x], Pk[y])), deficit=1 - float(v.mean()), alive_flips=int(((aa >= ALIVE_MIN) != (ab >= ALIVE_MIN)).sum()),
                             lt90=int((iou[sel] < 0.9).sum()), lt50=int((iou[sel] < 0.5).sum()), first_lt50=int(low[0]) if len(low) else None, departing=int((iou < 0.5).any(axis=0).sum()), area_ratio=ratio,
                             more=int((alive[x] > alive[y]).sum()), fewer=int((alive[x] < alive[y]).sum()))
        s = summary[name]
        P(f"| {name} | {x} vs {y} | {s['bit_identical']} | {v.mean():.5f} | {np.median(v):.5f} | {np.quantile(v, .05):.4f} | {s['lt90']} / {int(sel.sum())} | {s['lt50']} | {s['alive_flips']} | {s['first_lt50'] if s['first_lt50'] is not None else 'never'} | {s['departing']} of {K} | {ratio:.4f} | {s['more']} / {s['fewer']} |")
    if ("H16_a", "F32") in IOU:
        iou = IOU[("H16_a", "F32")]
        P("\n## Per person, float16 vs true float32\n")
        P("| person | mean IoU | first frame IoU<0.5 | last frame with a mask, float16 | last frame with a mask, float32 | mean area float16 / float32 (px) |\n|---|---|---|---|---|---|")
        last = lambda q: (int(np.where(q >= ALIVE_MIN)[0].max()) if (q >= ALIVE_MIN).any() else "never")
        for j in range(K):
            a16, a32 = ar["H16_a"][:, j], ar["F32"][:, j]; dep = np.where(iou[:, j] < 0.5)[0]
            P(f"| {j} | {iou[:, j].mean():.4f} | {int(dep[0]) if len(dep) else 'never'} | {last(a16)} | {last(a32)} | "
              f"{a16[a16 >= ALIVE_MIN].mean() if (a16 >= ALIVE_MIN).any() else 0:.0f} / {a32[a32 >= ALIVE_MIN].mean() if (a32 >= ALIVE_MIN).any() else 0:.0f} |")
    P("\n## Inside the tracker: per-person presence logit (the gate: mask replaced when below 0), predicted IoU, pointer and memory-attention outputs\n")
    P("| pair | calls | max abs d presence logit | median abs d | gate sign flips (person-frames) | smallest abs presence logit (first arm) | max abs d predicted IoU | obj-pointer rel L2 median / max | memory-attention rel L2 first third / last third |\n|---|---|---|---|---|---|---|---|---|")
    for name, x, y in pairs:
        Cx, Cy = arms[x]["calls"], arms[y]["calls"]; m = min(len(Cx.get("dec_prop", [])), len(Cy.get("dec_prop", [])))
        if m == 0:
            P(f"| {name} ({x} vs {y}) | 0 | - | - | - | - | - | - | - |"); continue
        ox = torch.stack([Cx["dec_prop"][i]["obj"].reshape(-1)[:K] for i in range(m)]); oy = torch.stack([Cy["dec_prop"][i]["obj"].reshape(-1)[:K] for i in range(m)])
        ix = torch.stack([Cx["dec_prop"][i]["iou"].reshape(-1, Cx["dec_prop"][i]["iou"].shape[-1])[:K] for i in range(m)])
        iy = torch.stack([Cy["dec_prop"][i]["iou"].reshape(-1, Cy["dec_prop"][i]["iou"].shape[-1])[:K] for i in range(m)])
        d = (ox - oy).abs(); flips = int(((ox > 0) != (oy > 0)).sum())
        pr = [rel_l2(Cy["ptr_prop"][i]["sketch"], Cx["ptr_prop"][i]["sketch"]) for i in range(min(len(Cx["ptr_prop"]), len(Cy["ptr_prop"])))]
        ma = [rel_l2(Cy["mem_attn"][i]["sketch"], Cx["mem_attn"][i]["sketch"]) for i in range(min(len(Cx["mem_attn"]), len(Cy["mem_attn"])))]
        th = max(1, len(ma) // 3)
        summary[name].update(gate_flips=flips, gate_d_max=float(d.max()), gate_d_median=float(d.median()), ptr_rel_l2_median=float(np.median(pr)), mem_attn_rel_l2_last_third=float(np.mean(ma[-th:])))
        P(f"| {name} ({x} vs {y}) | {m} | {float(d.max()):.3e} | {float(d.median()):.3e} | {flips} of {ox.numel()} | {float(ox.abs().min()):.3f} | {float((ix - iy).abs().max()):.3e} | "
          f"{np.median(pr):.2e} / {max(pr):.2e} | {np.mean(ma[:th]):.2e} / {np.mean(ma[-th:]):.2e} |")
    allobj = torch.cat([c["obj"].reshape(-1)[:K] for c in arms[next(iter(arms))]["calls"]["dec_prop"]])
    P("\nPresence logits of the first arm over all person-frames: quantiles 1/5/25/50/75/95/99 percent = " + ", ".join(f"{float(allobj.quantile(q)):.2f}" for q in (.01, .05, .25, .5, .75, .95, .99))
      + f"; share within |logit| < 0.5 of the gate {float((allobj.abs() < 0.5).float().mean()):.3f}")
    P("\n## fp16 headroom: largest |activation| leaving any trunk block during the track (fp16 maximum 65504)\n")
    P("| arm | max over blocks | block | headroom factor |\n|---|---|---|---|")
    for t, R in arms.items():
        mx = np.array(R["trunk_block_absmax"]); P(f"| {t} | {mx.max():.1f} | {int(mx.argmax())} | {FP16_MAX / max(mx.max(), 1e-9):.0f}x |")
    P("\n## Reading (by the rule above)\n")
    g = lambda k: summary.get(k, {}).get("deficit")
    verdict = None
    if g("run-to-run floor") is not None and g("sensitivity floor (1/255 nudge)") is not None and g("float16 vs float32") is not None:
        fl = [g("run-to-run floor"), g("sensitivity floor (1/255 nudge)")]; fp = g("float16 vs float32"); bf = g("bf16 vs float32"); pc = g("positive control (8/255 nudge)")
        P(f"1 - mean IoU: run-to-run floor {fl[0]:.5f}; sensitivity floor {fl[1]:.5f}; float16 vs float32 {fp:.5f}; bf16 vs float32 " + (f"{bf:.5f}" if bf is not None else "n/a") + "; positive control " + (f"{pc:.5f}" if pc is not None else "n/a") + ".")
        if pc is not None and pc <= fl[1] + 1e-4:
            P("POSITIVE CONTROL NOT SEEN: the 8/255 nudge is no clearer than the 1/255 nudge; this stretch cannot show a difference.")
        if fp <= max(fl) + 1e-9:
            verdict = "within the floors"; P("float16 vs float32 is within the floors: precision is not a lever on this stretch.")
        elif bf is not None and fp <= bf + 1e-9:
            verdict = "above the floors, no larger than bf16"; P("float16 vs float32 is above the floors and no larger than bf16's difference from float32: a perturbation the reference tolerates.")
        else:
            verdict = "above bf16 or bf16 not run"; P("float16 vs float32 is above the floors and larger than bf16's (or bf16 was not run): see the layer and one-sidedness columns to name where.")
    Path(a.md).write_text("\n".join(out) + "\n")
    Path(a.json).write_text(json.dumps(dict(stretch=stretch, frames=n, people=K, pairs=summary, verdict=verdict), indent=1, default=str))
    if a.csv:
        with open(a.csv, "w") as f:
            f.write("frame," + ",".join(f"alive_{t}" for t in arms) + "," + ",".join(f"minIoU_{x}_vs_{y}" for _, x, y in pairs) + "," + ",".join(f"meanIoU_{x}_vs_{y}" for _, x, y in pairs) + "\n")
            for i in range(n):
                f.write(f"{i}," + ",".join(str(int(alive[t][i])) for t in arms) + "," + ",".join(f"{IOU[(x, y)][i].min():.4f}" for _, x, y in pairs) + "," + ",".join(f"{IOU[(x, y)][i].mean():.4f}" for _, x, y in pairs) + "\n")


# ---- the stacked picture

PAL = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
       (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195)]


def cmd_render(a):
    import av
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    work = Path(a.work_dir); out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True); s = a.stretch
    tags = [p.split("=", 1)[0] for p in a.arms]; labels = [p.split("=", 1)[1] for p in a.arms]
    A = {t: R for t in tags if (R := load_arm(work, t, s)) is not None}
    tg = [t for t in tags if t in A]; lb = [labels[tags.index(t)] for t in tg]
    if not tg:
        raise SystemExit(f"no arm files for stretch {s} in {work}")
    n, K = A[tg[0]]["n"], A[tg[0]]["K"]
    RW, RH = a.width, a.width * 9 // 16
    frames = np.load(work / f"frames_{s}.npy", mmap_mode="r"); Pk = {t: A[t]["packed_masks"] for t in tg}

    def labelmap(Pf):
        bits = np.unpackbits(Pf, axis=-1).astype(bool); L = np.zeros(bits.shape[1:], np.uint8)
        for k in reversed(range(K)):
            L[bits[k]] = k + 1
        return L

    def text(img, msg, org, scale=0.6, color=(255, 255, 255), thick=1):
        cv2.putText(img, msg, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, msg, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def row_image(small, Ls, label):
        img = small.copy(); col = np.zeros_like(img)
        for k in range(K):
            col[Ls == k + 1] = PAL[k % len(PAL)]
        m = Ls > 0; img[m] = (0.5 * img[m] + 0.5 * col[m]).astype(np.uint8)
        for k in range(K):
            ys, xs = np.nonzero(Ls == k + 1)
            if len(ys) >= 30:
                text(img, str(k), (int(xs.mean()) - 6, int(ys.mean()) + 5), 0.6)
        cv2.rectangle(img, (0, 0), (RW, 26), (0, 0, 0), -1); text(img, label, (6, 19), 0.55)
        return img

    path = out_dir / f"{a.prefix}_{s}_stacked_{len(tg)}arms.mp4"
    n_video = 0 if a.chart_only else n
    c = st = None
    if not a.chart_only:
        c = av.open(str(path), "w"); st = c.add_stream("h264", rate=25); st.width = RW; st.height = RH * (len(tg) + 1); st.pix_fmt = "yuv420p"
        st.options = {"crf": "20", "preset": "medium"}
    dis = []
    for i in range(n_video):
        small = cv2.resize(np.ascontiguousarray(frames[i]), (RW, RH), interpolation=cv2.INTER_AREA)
        Ls = [cv2.resize(labelmap(Pk[t][i]), (RW, RH), interpolation=cv2.INTER_NEAREST) for t in tg]
        rows = [row_image(small, Ls[k], lb[k]) for k in range(len(tg))]
        nd = a.disagree_first or len(tg)             # the red row compares rows 1..nd only; later rows (a nudged control) are shown for scale
        diff = np.zeros((RH, RW), bool)
        for k in range(1, nd):
            diff |= Ls[k] != Ls[0]
        gray = (cv2.cvtColor(small, cv2.COLOR_RGB2GRAY) * 0.35).astype(np.uint8); dr = np.stack([gray, gray, gray], -1); dr[diff] = (255, 40, 40)
        cv2.rectangle(dr, (0, 0), (RW, 26), (0, 0, 0), -1)
        union = np.zeros((RH, RW), bool)
        for k in range(nd):
            union |= Ls[k] > 0
        dis.append(int(diff.sum()))
        text(dr, f"Red = pixels where rows 1 to {nd} disagree about who is there", (6, 19), 0.5)
        text(dr, f"{s}  frame {i + 1}/{n}   disagreement {100.0 * diff.sum() / max(1, union.sum()):.1f}% of person pixels", (6, RH - 8), 0.5)
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.concatenate(rows + [dr], axis=0), format="rgb24")):
            c.mux(pkt)
    if c is not None:
        for pkt in st.encode(None):
            c.mux(pkt)
        c.close()
        print("wrote", path, "; frames with any disagreement:", sum(1 for d in dis if d), "of", n)
    alive = {t: (areas(Pk[t]) >= ALIVE_MIN).sum(axis=1) for t in tg}
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True); cols = ["#444444", "#1b7fcc", "#d95f02", "#7570b3", "#1b9e77", "#e6ab02"]
    for k, t in enumerate(tg):
        ax[0].plot(alive[t] + 0.04 * k, label=lb[k], color=cols[k % len(cols)], lw=1.5)
    ax[0].set_ylabel("people with a mask"); ax[0].legend(fontsize=7)
    for k, t in enumerate(tg[1:], start=1):
        iou, aa, ab = iou_series(Pk[tg[0]], Pk[t])
        iou = np.where((aa >= ALIVE_MIN) | (ab >= ALIVE_MIN), iou, np.nan)        # undefined where nobody has a mask in either arm
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax[1].plot(np.nanmin(iou, axis=1), label=lb[k], color=cols[k % len(cols)], lw=1.2); ax[2].plot(np.nanmean(iou, axis=1), label=lb[k], color=cols[k % len(cols)], lw=1.2)
    ax[1].set_ylabel("worst person's IoU\nagainst row 1"); ax[2].set_ylabel("mean IoU over people\nagainst row 1"); ax[2].set_xlabel(f"frame of the {s} stretch (25 per second); IoU is left blank where nobody has a mask in either row")
    for q in ax:
        q.grid(alpha=0.25)
    ax[1].set_ylim(-0.02, 1.02); ax[2].set_ylim(-0.02, 1.02)
    fig.suptitle(f"{s}: {K} people seeded on frame 0, the same seeds in every row", fontsize=10); fig.tight_layout()
    fig.savefig(out_dir / f"{a.prefix}_{s}_data.png", dpi=110); plt.close(fig)


# ---- command line

def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("frames"); p.add_argument("--clip", required=True); p.add_argument("--stretch", action="append", required=True, help="NAME=START_FRAME:COUNT")
    p.add_argument("--work-dir", required=True); p.add_argument("--threads", type=int, default=2); p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("arm"); p.add_argument("--ckpt", required=True); p.add_argument("--tag", required=True); p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--nudge", type=int, default=0, help="levels of 8 each pixel moves by, random sign"); p.add_argument("--tf32", choices=("off", "on"), default="off")
    p.add_argument("--make-seeds", action="store_true"); p.add_argument("--seeds-only", action="store_true"); p.add_argument("--stretches", nargs="+", required=True)
    p.add_argument("--work-dir", required=True); p.add_argument("--json"); p.add_argument("--smoke", action="store_true"); p.add_argument("--cpu", action="store_true")
    p.add_argument("--threads", type=int, default=0); p.set_defaults(fn=cmd_arm)
    p = sub.add_parser("floor"); p.add_argument("--ckpt", required=True); p.add_argument("--tag", required=True); p.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), required=True)
    p.add_argument("--stretch", required=True); p.add_argument("--nframes", type=int, default=20); p.add_argument("--work-dir", required=True)
    p.add_argument("--cpu", action="store_true"); p.add_argument("--threads", type=int, default=0); p.set_defaults(fn=cmd_floor)
    p = sub.add_parser("floor-compare"); p.add_argument("--work-dir", required=True); p.add_argument("--tags", nargs="+", required=True, help="first the float16 run, second the float32 reference")
    p.add_argument("--md", required=True); p.add_argument("--json", required=True); p.set_defaults(fn=cmd_floor_compare)
    p = sub.add_parser("predict"); p.add_argument("--stage", choices=("a", "b"), required=True); p.add_argument("--work-dir", required=True); p.add_argument("--epsilon-json", required=True)
    p.add_argument("--md", required=True); p.add_argument("--calm-tag", default="H16_a"); p.add_argument("--calm-stretch"); p.add_argument("--hard-tag", default="H16_a"); p.add_argument("--hard-stretch")
    p.add_argument("--hard-people", type=int, default=0); p.add_argument("--hard-frames", type=int, default=0); p.set_defaults(fn=cmd_predict)
    p = sub.add_parser("compare"); p.add_argument("--work-dir", required=True); p.add_argument("--stretch", required=True); p.add_argument("--md", required=True)
    p.add_argument("--json", required=True); p.add_argument("--csv"); p.add_argument("--first", type=int, default=0, help="analyse only the first N frames")
    p.set_defaults(fn=cmd_compare)
    p = sub.add_parser("render"); p.add_argument("--work-dir", required=True); p.add_argument("--stretch", required=True); p.add_argument("--out-dir", required=True)
    p.add_argument("--arms", nargs="+", required=True, help="TAG=Label, first is the reference row"); p.add_argument("--prefix", default="sam3_precision"); p.add_argument("--width", type=int, default=640)
    p.add_argument("--disagree-first", type=int, default=0, help="the red row compares only the first N rows (0 = all)")
    p.add_argument("--chart-only", action="store_true", help="redraw the chart without re-encoding the video")
    p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
