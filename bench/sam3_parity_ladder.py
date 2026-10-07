#!/usr/bin/env python3
"""Is core's SAM 3.1 the model Meta released? The same weights and the same input through both, stage by stage.

Core's SAM 3.1 (`comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`, `comfy/text_encoders/sam3_clip.py`) is a re-implementation.
This tool runs it beside Meta's own inference code, the copy under `meta_sam3/`, one stage at a time, so "differs" is a number
and names the stage. `meta_sam3/` is imported here AS A REFERENCE to compare against; nothing shipped imports it and no node
depends on this file. Each rung feeds both sides an equal input, so a difference found below does not leak into the rung above:

    tokens     the token ids for a list of phrases. CPU, no model.
    files      every tensor of two weights files (the ComfyUI-org repack against ours cast to float16). CPU, reads only.
    text       the text encoder's features, core as shipped and with its activation switched in memory to Meta's;
               then what that does to one detect.
    trunk      the image each side hands its trunk, and the trunk's features on an equal tensor.
    detector   the detector's raw outputs (presence, 200 query scores, boxes, masks) on Meta's image tensor and equal text.
    tracker    by outcome, not by stage: core's track node and Meta's tracker class started from the SAME masks. The masks
               come from one detect by core, on the frame as its node gets it or in the trained range (`--seeds-from`):
               which people are seeded, and in what order, differs between the two.
    there-and-back  the tracker again, with what the first run lacked: the window played forward and then in reverse, so a
               subject must end on its own seed; the range corrected the way a loader would (clamped, then mapped, at the
               first layer); every arm repeated with the input moved one level of 255, as its floor; a shape test on
               every mask, so a track that slid onto the frame's border is not counted as a subject held.
    render     print the record's tables from the json the rungs wrote.

What it cannot say. `tracker` counts seeded people who still have a non-empty mask; it does not say a mask is still on the
right person, and each arm is one run. Measured 2026-10-07 on a dense window: beyond the first group of sixteen the count is
not steady, the same arm ending with every person or with a dozen fewer on frames decoded by two routes a couple of
brightness levels apart. Read a difference between arms beyond sixteen as noise until it is repeated under such a change. With several people Meta's tracker-only path applies neither of the two rules its full
pipeline applies between people, and core applies both, so only the one-person arm compares like with like. Core runs in
float32 with TF32 off in `text`, `trunk` and `detector`, Meta under the bf16 autocast its own code enters; a difference the
size of float32 against bf16 is read as this tool's floor, which it does not yet measure by itself. Meta's side is its
list-of-images loading route; its other routes are not compared. Compute precision is a separate question and a separate
tool's: `bench/sam3_precision_arms.py`.

Every path is an argument. A clip is named in the json by its file name and never by what it shows; `--describing` takes a
phrase whose text is not written anywhere.

    <python> bench/sam3_parity_ladder.py tokens --json J
    <python> bench/sam3_parity_ladder.py files --theirs A.safetensors --ours B.safetensors --json J
    <python> bench/sam3_parity_ladder.py text --clip C --second S --width W --json J [--describing "..."]
    <python> bench/sam3_parity_ladder.py trunk --clip C --second S --width W --json J
    <python> bench/sam3_parity_ladder.py detector --clip C --second S --width W --json J
    <python> bench/sam3_parity_ladder.py tracker --clip C --second S --seconds T --width W --rate R --seeds-from node|trained --json J
    <python> bench/sam3_parity_ladder.py there-and-back --clip C --second S --seconds T --width W --rate R --seeds-from node|corrected --json J
    <python> bench/sam3_parity_ladder.py render --json J

Run from ComfyUI's environment, outside the server, with the card free for every rung but `tokens` and `files`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parent.parent
sys.path.insert(0, str(REPO / "workflows"))

#: measured (`bench/convert_sam3_checkpoint.py` output): the one tensor core's loader refuses outside the server; neither the
#: model nor the text features use it.
TEXT_PROJECTION_KEY = "detector.backbone.language_backbone.encoder.text_projection"
SIDE = 1008          # inherited: the trunk's image side, `comfy/ldm/sam3`
SEED_SIDE = 1152     # inherited: the side Meta's tracker takes a seed mask at, `meta_sam3/sam3/model/video_tracking_multiplex.py`
#: reasoned: one-word phrases the lane ships, an article, and describing phrases of growing length; no clip is involved.
TEXT_PHRASES = ["person", "head", "a person", "person wearing a hat", "the tallest person", "man in a red shirt", "hair"]
_LONG = " ".join(["a person standing next to another person in a very large room with tall windows"] * 3)
#: reasoned: case, articles, punctuation, counts as core's tokenizer parses them, non-ASCII, whitespace, over-length, empty.
TOKEN_PHRASES = ["person", "a person", "Person", "PERSON", "head", "hair", "the tallest person", "person wearing a hat",
                 "man in a red shirt", "person's hand", "t-shirt", "dog/cat", "  two   spaces  ", "person.", "person!", "(person)",
                 "café table", "naïve façade", "ｆｕｌｌｗｉｄｔｈ person", "personne âgée", "人", "person 🙂",
                 "person\tholding\na cup", "3 people", "person:2", "eye:2, window panels:4", "hair, head", _LONG, ""]
#: reasoned: the arms that separate "alone in a group", "a group of sixteen" and "a second group" (core's tracker groups by 16).
SEED_ARMS = (("alone", 1), ("with one neighbour", 2), ("17", 17), ("32", 32))


# ---- shared

def put(path: str, key: str, value) -> None:
    f = Path(path)
    data = json.loads(f.read_text()) if f.exists() else {}
    data[key] = value
    f.write_text(json.dumps(data, indent=1, ensure_ascii=False))


def stats(a, b) -> dict:
    import torch.nn.functional as F
    a, b = a.float().flatten().cpu(), b.float().flatten().cpu()
    if a.numel() != b.numel():
        return {"sizes_differ": [a.numel(), b.numel()]}
    return {"relative_l2": round(float((a - b).norm() / b.norm().clamp(min=1e-12)), 5), "cosine": round(float(F.cosine_similarity(a, b, dim=0)), 5),
            "max_abs_diff": round(float((a - b).abs().max()), 4)}


def boot(float32: bool, cpu: bool = False):
    """Import core outside the server. `float32` asks core for float32 model and text encoder and switches TF32 off."""
    sys.argv = sys.argv[:1]
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(COMFY))
    import comfy.cli_args
    comfy.cli_args.args.cpu = cpu
    if float32:
        comfy.cli_args.args.fp32_unet = True
        comfy.cli_args.args.fp32_text_enc = True
    import torch
    if float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch


def tf32_selftest(torch) -> dict:
    """A float32 convolution and matmul on the card against float64 on the CPU: a relative error near 1e-6 is true float32,
    near 1e-3 is TF32. The convolution has the neck's shape, 256 channels and 3x3, because a narrower one does not show
    cuDNN's TF32. After `bench/sam3_precision_arms.py::tf32_selftest`, the independent session's."""
    if not torch.cuda.is_available():
        return {}
    g = torch.Generator().manual_seed(7)
    x, w = torch.randn(1, 256, 72, 72, generator=g), torch.randn(256, 256, 3, 3, generator=g) * 0.05
    a, b = torch.randn(512, 1024, generator=g), torch.randn(1024, 512, generator=g)
    conv = torch.nn.functional.conv2d
    rel = lambda got, ref: float((got.double().cpu() - ref).abs().max() / ref.abs().max())   # noqa: E731
    return {"conv_relative_error": rel(conv(x.cuda(), w.cuda(), padding=1), conv(x.double(), w.double(), padding=1)),
            "matmul_relative_error": rel(a.cuda() @ b.cuda(), a.double() @ b.double())}


def environment(torch, selftest: bool = False) -> dict:
    head = subprocess.run(["git", "-C", str(COMFY), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"torch": torch.__version__, "core_commit": head,
           "tf32": {"matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn": torch.backends.cudnn.allow_tf32}}
    if selftest:
        out["tf32_selftest"] = tf32_selftest(torch)
    return out


def meta_package():
    """The copy under `meta_sam3/`, imported as a package of its own so its relative imports resolve. A reference only."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)


def load_core(ckpt: str):
    import comfy.sd
    import comfy.utils
    import folder_paths
    sd, md = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("checkpoints", ckpt), return_metadata=True)
    sd.pop(TEXT_PROJECTION_KEY, None)
    model, clip, _, _ = comfy.sd.load_state_dict_guess_config(sd, output_vae=False, output_clip=True, output_model=True, metadata=md)
    return model, clip


def load_meta(ckpt: str):
    """Meta's 3.1 predictor built by its own builder, given the same weights file core loaded."""
    import comfy.utils
    import folder_paths
    meta_package()
    import _h3pack.meta_sam3.sam3.model_builder as mb
    p = mb.build_sam3_multiplex_video_predictor(checkpoint_path=None, use_fa3=False, compile=False, warm_up=False,
                                                async_loading_frames=False, device="cuda")
    missing, unexpected = p.model.load_state_dict(comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("checkpoints", ckpt)), strict=False)
    p.model.eval()
    return p, {"missing": len(missing), "unexpected": len(unexpected)}


def one_frame(clip: str, second: float, width: int):
    """The first frame at or after `second`, RGB uint8; shrunk to `width` (Lanczos) when `width` is given."""
    import av
    import numpy as np
    from PIL import Image
    c = av.open(clip)
    s = c.streams.video[0]
    t0 = s.start_time or 0
    c.seek(int(max(0.0, second - 2.0) / s.time_base) + t0, stream=s, backward=True)
    for fr in c.decode(s):
        if float((fr.pts - t0) * s.time_base) >= second - 1e-6:
            arr = fr.to_ndarray(format="rgb24")
            if width and width < arr.shape[1]:
                h = int(round(arr.shape[0] * width / arr.shape[1] / 2) * 2)
                arr = np.asarray(Image.fromarray(arr).resize((width, h), Image.LANCZOS))
            return arr, {"clip": Path(clip).name, "second": second, "decoded_at": round(float((fr.pts - t0) * s.time_base), 4), "size": [int(arr.shape[1]), int(arr.shape[0])]}
    raise SystemExit(f"no frame at or after {second} s")


def text_activation(clip):
    """Core's text MLPs and a switch between the activation it ships and exact GELU. In memory only."""
    import comfy.clip_model
    import torch.nn.functional as F
    mlps = [m for m in clip.cond_stage_model.modules() if isinstance(m, comfy.clip_model.CLIPMLP)]
    quick = comfy.clip_model.ACTIVATIONS["quick_gelu"]

    def switch(exact: bool):
        for m in mlps:
            m.activation = F.gelu if exact else quick
    return mlps, quick, switch


# ---- tokens

def cmd_tokens(a):
    boot(float32=False, cpu=True)
    from comfy.text_encoders.sam3_clip import SAM3TokenizerWrapper, _parse_prompts
    meta_package()
    from _h3pack.meta_sam3.sam3.model.tokenizer_ve import SimpleTokenizer
    meta = SimpleTokenizer(bpe_path=str(REPO / "meta_sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"))
    core = SAM3TokenizerWrapper()

    def core_ids(text):
        """What core hands its text encoder, per prompt part: [(part, ids)]."""
        out = core.tokenize_with_weights(text)
        if "sam3_per_prompt" in out:
            return [(p[0], [t[0] for t in (b["l"][0] if isinstance(b, dict) else b[0])]) for p, (b, _n) in zip(_parse_prompts(text), out["sam3_per_prompt"])]
        return [(text, [t[0] for t in out["sam3_clip"][0]])]

    rows = []
    for text in TOKEN_PHRASES:
        for part, ids in core_ids(text):
            m = meta([part], context_length=len(ids))[0].tolist()      # Meta's default length is CLIP's; core's is the model's
            rows.append({"typed": text[:60], "part": part[:60], "same": ids == m, "length": len(ids), "core_tokens": sum(1 for x in ids if x),
                         "meta_tokens": sum(1 for x in m if x), **({} if ids == m else {"core": ids, "meta": m})})
            print(("same   " if ids == m else "DIFFER ") + repr(part[:60]))
    out = {"parts": len(rows), "differ": sum(not r["same"] for r in rows), "meta_used_ftfy": "ftfy" in sys.modules, "rows": rows}
    print(f"{out['parts']} parts, {out['differ']} differ")
    put(a.json, "tokens", out)


# ---- files

def cmd_files(a):
    import torch
    from safetensors import safe_open
    out = {"theirs": Path(a.theirs).name, "ours": Path(a.ours).name, "shape_differs": [], "differs": [], "equal": 0}
    with safe_open(a.theirs, "pt") as x, safe_open(a.ours, "pt") as y:
        kx, ky = set(x.keys()), set(y.keys())
        out["only_theirs"], out["only_ours"] = sorted(kx - ky), sorted(ky - kx)
        out["theirs_dtypes"] = sorted({str(x.get_slice(k).get_dtype()) for k in kx})
        for k in sorted(kx & ky):
            t, o = x.get_tensor(k), y.get_tensor(k)
            if t.shape != o.shape:
                out["shape_differs"].append(k)
                continue
            o16 = o.to(torch.float16) if o.is_floating_point() else o
            d = float((t.float() - o16.float()).abs().max()) if t.numel() else 0.0
            if d == 0.0:
                out["equal"] += 1
            else:
                out["differs"].append([k, d])
    out["differs_count"] = len(out["differs"])
    out["differs"] = sorted(out["differs"], key=lambda r: -r[1])[:12]
    print(json.dumps({k: v for k, v in out.items() if k != "differs"}))
    put(a.json, "files", out)


# ---- text

def cmd_text(a):
    torch = boot(float32=True)
    import comfy.model_management as mm
    from comfy_extras.nodes_sam3 import SAM3_Detect
    core, clip = load_core(a.ckpt)
    sam = core.model.diffusion_model
    p, _ = load_meta(a.ckpt)
    text_model = next(m for n, m in p.model.named_modules() if n.endswith("language_backbone"))
    mlps, quick, switch = text_activation(clip)
    R = {"environment": environment(torch), "core_text_mlps": len(mlps), "core_ships_quick_gelu": all(m.activation is quick for m in mlps), "phrases": {}}

    def core_text(phrase):
        emb = clip.encode_from_tokens_scheduled(clip.tokenize(phrase))[0][0]                       # [1, 32, 1024]
        mm.load_model_gpu(core)
        resizer = sam.detector.backbone["language_backbone"]["resizer"]
        return emb, resizer(emb.to(device=mm.get_torch_device(), dtype=core.model.get_dtype()))

    def meta_text(phrase):
        kept = []
        h = text_model.encoder.register_forward_hook(lambda m, i, o: kept.append(o))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mask, resized, _ = text_model([phrase], device=torch.device("cuda"))
        finally:
            h.remove()
        tokens = kept[0][1] if isinstance(kept[0], (tuple, list)) else kept[0]
        return tokens, resized.transpose(0, 1), int((~mask)[0].sum())                               # [1, 32, 1024], [1, 32, 256], real tokens

    with torch.inference_mode():
        for phrase in TEXT_PHRASES:
            mt, ms, n = meta_text(phrase)
            row: dict = {"real_tokens": n}
            for label, exact in (("as shipped", False), ("exact GELU", True)):
                switch(exact)
                ct, cs = core_text(phrase)
                row[label] = {"encoder": stats(ct[:, :n], mt[:, :n]), "after_resizer": stats(cs[:, :n], ms[:, :n])}
            switch(False)
            R["phrases"][phrase] = row
            print(phrase, json.dumps(row), flush=True)

        # what the activation does to one detect, the frame handed over in the range the model was trained on
        arr, R["frame"] = one_frame(a.clip, a.second, a.width)
        frame = torch.from_numpy(arr.astype("float32") / 255.0)[None] * 2.0 - 1.0
        R["detect"] = {}
        asked = [("person", "person:16"), ("head", "head:16")] + ([("a describing phrase (text not recorded)", a.describing + ":16")] if a.describing else [])
        for name, phrase in asked:
            row, masks = {}, {}
            for label, exact in (("as shipped", False), ("exact GELU", True)):
                switch(exact)
                cond = clip.encode_from_tokens_scheduled(clip.tokenize(phrase))
                seen = []
                h = sam.register_forward_hook(lambda m, i, o: seen.append(o) if isinstance(o, dict) else None)
                try:
                    out = SAM3_Detect.execute(core, frame, conditioning=cond, threshold=0.5, individual_masks=True)
                finally:
                    h.remove()
                m = getattr(out, "args", out)[0]
                s = seen[0]["scores"][0].float().sigmoid()
                masks[label] = m.to(torch.float32).cpu() > 0.5
                row[label] = {"detections": int(m.shape[0]), "presence": round(float(seen[0]["presence"].float().flatten()[0].sigmoid()), 4),
                              "top_scores": [round(float(x), 3) for x in s.sort(descending=True)[0][:4]]}
            x, y = masks["as shipped"], masks["exact GELU"]
            if x.shape[0] and y.shape[0]:
                xf, yf = x.flatten(1).float(), y.flatten(1).float()
                inter = xf @ yf.T
                iou = inter / (xf.sum(1)[:, None] + yf.sum(1)[None] - inter).clamp(min=1)
                row["lowest_best_iou_between_the_two"] = round(float(iou.max(dim=1)[0].min()), 3)
            switch(False)
            R["detect"][name] = row
            print("detect", name, json.dumps(row), flush=True)
    put(a.json, "text", R)


# ---- trunk

def cmd_trunk(a):
    torch = boot(float32=True)
    import numpy as np
    from PIL import Image

    import comfy.model_management as mm
    import comfy.utils
    core, _ = load_core(a.ckpt)
    mm.load_model_gpu(core)
    ctrunk = core.model.diffusion_model.detector.backbone["vision_backbone"].trunk
    p, weights = load_meta(a.ckpt)
    mtrunk = next(m for n, m in p.model.named_modules() if n.endswith("vision_backbone.trunk"))
    arr, where = one_frame(a.clip, a.second, a.width)
    img = torch.from_numpy(arr.astype(np.float32) / 255.0)[None]                                    # a ComfyUI IMAGE, 0..1
    # what core's detect node hands its model (`comfy_extras/nodes_sam3.py`: a bilinear resize to the side, nothing else)
    x_core = comfy.utils.common_upscale(img.movedim(-1, 1), SIDE, SIDE, "bilinear", crop="disabled")
    # what Meta's list-of-images route hands its model (`meta_sam3/sam3/model/io_utils.py`: a PIL resize to the side, then
    # (x / 255 - 0.5) / 0.5); its other loading routes are not built here
    x_meta = (torch.from_numpy(np.asarray(Image.fromarray(arr).resize((SIDE, SIDE))).astype(np.float32) / 255.0).permute(2, 0, 1)[None] - 0.5) / 0.5
    x_mapped = (x_core - 0.5) / 0.5
    rng = lambda t: [round(float(t.min()), 3), round(float(t.max()), 3)]   # noqa: E731
    R = {"environment": environment(torch, selftest=True), "frame": where, "weights_into_meta": weights, "core_dtype": str(core.model.get_dtype()),
         "input": {"core_range": rng(x_core), "meta_range": rng(x_meta), "core_as_fed_vs_meta": stats(x_core, x_meta),
                   "core_mapped_to_metas_range_vs_meta": stats(x_mapped, x_meta)}}
    x_core, x_meta, x_mapped = x_core.cuda(), x_meta.cuda(), x_mapped.cuda()
    last = lambda t: t[-1] if isinstance(t, (list, tuple)) else t   # noqa: E731
    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_meta, f_meta_unmapped = last(mtrunk(x_meta)), last(mtrunk(x_core))
        f_core = last(ctrunk(x_meta))
        # the two floors the rows below are read against: the same network at bf16, and the same frame moved one level
        floors = {}
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                floors["FLOOR_core_float32_vs_core_under_bf16_same_tensor"] = stats(last(ctrunk(x_meta)), f_core)
        except Exception as e:      # core's trunk under autocast is not a path core itself takes
            floors["FLOOR_core_float32_vs_core_under_bf16_same_tensor"] = {"error": f"{type(e).__name__}: {e}"}
        moved = nudged(arr[None], 1)[0]
        x_moved = (torch.from_numpy(np.asarray(Image.fromarray(moved).resize((SIDE, SIDE))).astype(np.float32) / 255.0).permute(2, 0, 1)[None] - 0.5) / 0.5
        floors["FLOOR_core_float32_same_frame_moved_one_level"] = stats(last(ctrunk(x_moved.cuda())), f_core)
        R["trunk"] = {"shape": list(f_meta.shape), **floors,
                      "equal_tensor_core_vs_meta": stats(f_core, f_meta),
                      "core_as_its_node_feeds_vs_meta": stats(last(ctrunk(x_core)), f_meta),
                      "core_resize_in_metas_range_vs_meta": stats(last(ctrunk(x_mapped)), f_meta),
                      "meta_given_the_unmapped_image_vs_meta": stats(f_meta_unmapped, f_meta),
                      "core_as_fed_vs_meta_given_the_same_unmapped_image": stats(last(ctrunk(x_core)), f_meta_unmapped)}
    cw, mw = ctrunk.patch_embed.proj.weight.detach().float(), mtrunk.patch_embed.proj.weight.detach().float()
    R["patch_embedding"] = {"max_abs_diff_of_weights": float((cw - mw.to(cw.device)).abs().max()),
                            "core_has_bias": ctrunk.patch_embed.proj.bias is not None, "meta_has_bias": mtrunk.patch_embed.proj.bias is not None}
    print(json.dumps(R, indent=1))
    put(a.json, "trunk" if a.width else "trunk, the file's own size", R)


# ---- detector

#: inherited: Meta's 3.1 builder (`meta_sam3/sam3/model_builder.py`, the multiplex builder's detection values): a detection is
#: kept above this joint score, and overlapping detections are removed at this overlap on the smaller mask.
META_KEEP_ABOVE = 0.4
META_NMS_IOM = 0.1
#: read: `comfy_extras/nodes_sam3.py::SAM3_Detect`'s default threshold, on the class score alone.
NODE_KEEP_ABOVE = 0.5


def cmd_detector(a):
    torch = boot(float32=True)
    import torch.nn.functional as F
    from PIL import Image

    import comfy.model_management as mm
    core, clip = load_core(a.ckpt)
    sam = core.model.diffusion_model
    p, _ = load_meta(a.ckpt)
    from _h3pack.meta_sam3.sam3.model.sam3_multiplex_detector_utils import nms_masks
    R = {"environment": environment(torch, selftest=True), "phrase": a.phrase, "core_dtype": str(core.model.get_dtype()),
         "rules": {"node": f"class score over {NODE_KEEP_ABOVE}", "meta": f"class times presence over {META_KEEP_ABOVE}, then overlap removal at {META_NMS_IOM} on the smaller mask"}}
    mtrunk = next(m for n, m in p.model.named_modules() if n.endswith("vision_backbone.trunk"))
    detector = p.model.detector
    _, _, switch = text_activation(clip)
    mm.load_model_gpu(core)
    device, dtype = mm.get_torch_device(), core.model.get_dtype()

    def joint(cls, pres):
        """Meta's `pred_logits` are the joint score, inverse_sigmoid(sigmoid(class) * sigmoid(presence)) clamped to +-10
        (`meta_sam3/sam3/model/sam3_image.py`); core returns the two apart (`comfy/ldm/sam3/detector.py`)."""
        q = (cls.float().sigmoid() * pres.float().sigmoid()).clamp(1e-6, 1 - 1e-6)
        return torch.log(q / (1 - q)).clamp(-10, 10)

    def metas_rule(joint_logits, mask_logits) -> int:
        """How many detections Meta's own rule keeps of these queries: its `nms_masks` on the joint score."""
        keep = nms_masks(joint_logits.sigmoid().float(), (mask_logits > 0).float(), META_KEEP_ABOVE, META_NMS_IOM, nms_use_iom=True)
        return int(keep.sum())

    def meta_side(arr):
        """One one-frame request to Meta's predictor, keeping what its loader handed the trunk and what the detector returned."""
        kept, fed = [], []
        stock = detector.forward_grounding

        def grounding(*args, **kwargs):
            out = stock(*args, **kwargs)
            kept.append({k: v.detach().float().cpu() for k, v in out.items() if torch.is_tensor(v) and k in ("pred_logits", "presence_logit_dec", "pred_masks", "pred_boxes_xyxy")})
            return out

        detector.forward_grounding = grounding
        pre = mtrunk.register_forward_pre_hook(lambda m, i: fed.append(i[0].detach().float().clone()))
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sid = p.handle_request({"type": "start_session", "resource_path": [Image.fromarray(arr)], "offload_video_to_cpu": True})["session_id"]
                p.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": a.phrase})
                p.handle_request({"type": "close_session", "session_id": sid})
        finally:
            pre.remove()
            detector.forward_grounding = stock
        return kept[0], fed[0][:1]

    def core_side(X, exact: bool):
        switch(exact)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(a.phrase))
        emb = cond[0][0].to(device=device, dtype=dtype)
        mask = cond[0][1].get("attention_mask")
        mask = mask.to(device) if mask is not None else torch.ones(emb.shape[:2], dtype=torch.int64, device=device)
        out = sam.detector(X.to(device=device, dtype=dtype), text_embeddings=emb, text_mask=mask, threshold=0.5, orig_size=None)
        switch(False)
        return {k: v.detach().float().cpu() for k, v in out.items() if torch.is_tensor(v)}

    def counts(C, M):
        """The same 200 queries under each rule, on each side's own outputs."""
        cj, mj = joint(C["scores"][0], C["presence"].flatten()[0]), M["pred_logits"].flatten()
        mm_ = M["pred_masks"].reshape(mj.numel(), *M["pred_masks"].shape[-2:])
        return {"presence_logit": {"core": round(float(C["presence"].flatten()[0]), 4), "meta": round(float(M["presence_logit_dec"].flatten()[0]), 4)},
                "core_outputs": {"node_rule": int((C["scores"][0].sigmoid() > NODE_KEEP_ABOVE).sum()), "metas_rule": metas_rule(cj, C["masks"][0])},
                "meta_outputs": {"metas_rule": metas_rule(mj, mm_)},
                "queries_over_half_joint": {"core": int((cj.sigmoid() > 0.5).sum()), "meta": int((mj.sigmoid() > 0.5).sum())}}

    with torch.inference_mode():
        arr, R["frame"] = one_frame(a.clip, a.second, a.width)
        M, X = meta_side(arr)
        R["meta_trunk_input_range"] = [round(float(X.min()), 3), round(float(X.max()), 3)]
        mj = M["pred_logits"].flatten()
        top = mj.argsort(descending=True)[:10]
        for label, exact in (("exact GELU", True), ("as shipped", False)):
            C = core_side(X, exact)
            cj = joint(C["scores"][0], C["presence"].flatten()[0])
            cm = C["masks"][0][top]
            mk = M["pred_masks"].reshape(mj.numel(), *M["pred_masks"].shape[-2:])[top]
            if cm.shape[-2:] != mk.shape[-2:]:
                mk = F.interpolate(mk[None], size=cm.shape[-2:], mode="bilinear", align_corners=False)[0]
            inter = ((cm > 0) & (mk > 0)).flatten(1).sum(1).float()
            union = ((cm > 0) | (mk > 0)).flatten(1).sum(1).float().clamp(min=1)
            R[label] = {**counts(C, M), "joint_score_logits_all_queries": stats(cj, mj), "queries": int(mj.numel()),
                        "top10_by_meta": {"meta": [round(float(x), 3) for x in mj[top].sigmoid()], "core": [round(float(x), 3) for x in cj[top].sigmoid()],
                                          "mask_iou": [round(float(x), 3) for x in inter / union]},
                        "boxes_top10_max_abs_diff": round(float((C["boxes"][0][top] - M["pred_boxes_xyxy"].reshape(-1, 4)[top]).abs().max()), 5)}
            print(label, json.dumps(R[label]), flush=True)
        # the floor for the row above: core against itself with the frame moved one level, on its own equal-input path
        moved, _ = meta_side(nudged(arr[None], 1)[0])[1], None
        Cn, C0 = core_side(moved, True), core_side(X, True)
        R["FLOOR_core_same_frame_moved_one_level"] = {
            "presence_logit": [round(float(C0["presence"].flatten()[0]), 4), round(float(Cn["presence"].flatten()[0]), 4)],
            "joint_score_logits_all_queries": stats(joint(Cn["scores"][0], Cn["presence"].flatten()[0]), joint(C0["scores"][0], C0["presence"].flatten()[0]))}
        print("floor", json.dumps(R["FLOOR_core_same_frame_moved_one_level"]), flush=True)
        R["more_frames"] = {}
        for second in a.more_seconds or []:
            arr2, where = one_frame(a.clip, second, a.width)
            M2, X2 = meta_side(arr2)
            R["more_frames"][str(second)] = {"decoded_at": where["decoded_at"], **counts(core_side(X2, True), M2)}
            print(second, json.dumps(R["more_frames"][str(second)]), flush=True)
    put(a.json, "detector, " + R["frame"]["clip"], R)


# ---- tracker, by outcome

class _TrackerBackbone:
    """The detector's backbone as Meta's tracker asks for it. The 3.1 tracker is built without a backbone. Lent the detector's
    as it is, `forward_image` fails: asked for the detector's own output, that backbone puts tensors at the top level of what
    it returns, and the tracker walks every key as a set of feature maps. The tracker reads only the other two sets."""

    def __init__(self, inner):
        self.inner = inner

    def forward_image(self, img_batch, need_sam3_out=False, need_interactive_out=False, need_propagation_out=False):
        return self.inner.forward_image(img_batch, need_sam3_out=False, need_interactive_out=need_interactive_out, need_propagation_out=need_propagation_out)


def cmd_tracker(a):
    torch = boot(float32=False)         # core as it computes by default; this rung compares outcomes, not tensors
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    import comfy.ldm.sam3.tracker as T
    import server
    from comfy.ldm.sam3.tracker import unpack_masks
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack
    # the loader node the masked graphs use, so the frames are the lane's; it reads the server's instance at import
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    frames = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3].contiguous()
    n, H, W = (int(v) for v in frames.shape[:3])
    core, clip = load_core(a.ckpt)
    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H]},
         "core_dtype": str(core.model.get_dtype()), "arms": {},
         "caution": "with several people Meta's tracker-only path applies neither of its two between-people rules; core applies both"}

    seed_frame = frames[:1] * 2 - 1 if a.seeds_from == "trained" else frames[:1]
    R["seeds_from"] = "one detect by core on the first frame, " + ("in the trained range" if a.seeds_from == "trained" else "as its node gets it")
    with torch.inference_mode():
        det = SAM3_Detect.execute(core, seed_frame, conditioning=clip.encode_from_tokens_scheduled(clip.tokenize("person:64")), threshold=0.5, individual_masks=True)
    masks = getattr(det, "args", det)[0].float().cpu()
    area = masks.flatten(1).sum(1)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cx, cy = (masks * xs).flatten(1).sum(1) / area.clamp(min=1), (masks * ys).flatten(1).sum(1) / area.clamp(min=1)
    subject = int(area.argmax())
    near = ((cx - cx[subject]) ** 2 + (cy - cy[subject]) ** 2).argsort().tolist()       # the largest person first, then by distance from them
    R["detections_on_the_seed_frame"] = int(masks.shape[0])
    seeds = {name: (masks[near[:k]] if k <= 2 else masks[:k]) for name, k in SEED_ARMS if masks.shape[0] >= k}

    def summary(on, scores=None):
        out = {"seeded": int(on.shape[0]), "with_a_mask_at": {str(f): int(on[:, f].sum()) for f in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})},
               "last_frame_each_is_on": [int(r.nonzero().max()) if r.any() else None for r in on]}
        if scores is not None:
            out["object_score_first_frames"] = [round(float(x), 2) for x in scores[:8]]
        return out

    def say(key, value):
        R["arms"][key] = value
        print(key, json.dumps(value), flush=True)

    stock_step, logged = T.SAM31Tracker.track_step, []

    def step(self, *args, **kwargs):
        out = stock_step(self, *args, **kwargs)
        v = out.get("object_score_logits")
        logged.append(None if v is None else float(v.detach().float().flatten()[0]))
        return out

    def core_run(video, seed):
        logged.clear()
        T.SAM31Tracker.track_step = step
        try:
            out = SAM3_VideoTrack.execute(video, core, initial_mask=seed, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
        finally:
            T.SAM31Tracker.track_step = stock_step
        packed = getattr(out, "args", out)[0]["packed_masks"]
        on = torch.zeros((seed.shape[0], n), dtype=torch.bool)
        if packed is not None:
            for k in range(packed.shape[1]):
                on[k] = unpack_masks(packed[:, k]).flatten(1).any(1)
        return on, [x for x in logged if x is not None]

    with torch.inference_mode():
        for label, video in (("core, frames as its node gets them", frames), ("core, frames in the trained range", frames * 2 - 1)):
            for name, seed in seeds.items():
                on, sc = core_run(video, seed)
                say(f"{label} | {name}", summary(on, sc if name == "alone" else None))

    torch.backends.cuda.matmul.allow_tf32 = True     # upstream's predictor sets both; the copy leaves them to its caller
    torch.backends.cudnn.allow_tf32 = True
    p, _ = load_meta(a.ckpt)
    from _h3pack.meta_sam3.sam3.model.io_utils import load_resource_as_video_frames
    tracker = p.model.tracker.model
    tracker.backbone = _TrackerBackbone(p.model.detector.backbone)
    pil = [Image.fromarray((f.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)) for f in frames]

    def meta_run(seed):
        k = seed.shape[0]
        at = F.interpolate(seed[:, None], size=(SEED_SIDE, SEED_SIDE), mode="bilinear", align_corners=False)[:, 0] > 0.5
        on, scores = torch.zeros((k, n), dtype=torch.bool), []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            images, vh, vw = load_resource_as_video_frames(pil, image_size=SIDE, offload_video_to_cpu=False)
            st = tracker.init_state(video_height=vh, video_width=vw, num_frames=len(images))
            st["images"] = images
            tracker.add_new_masks(st, frame_idx=0, obj_ids=list(range(k)), masks=at, add_mask_to_memory=True)
            tracker.propagate_in_video_preflight(st, run_mem_encoder=True)
            for frame_idx, obj_ids, _low, video_res, obj_scores in tracker.propagate_in_video(
                    st, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, tqdm_disable=True, run_mem_encoder=True):
                ids = [int(i) for i in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)]
                alive = (video_res[:, 0] > 0).flatten(1).any(1).cpu()
                for j, i in enumerate(ids):
                    if 0 <= i < k:
                        on[i, frame_idx] = bool(alive[j])
                if 0 in ids:
                    scores.append(float(obj_scores.float().flatten()[ids.index(0)]))
        return on, scores

    for name, seed in seeds.items():
        try:
            on, sc = meta_run(seed)
            say(f"Meta's tracker | {name}", summary(on, sc if name == "alone" else None))
        except Exception as e:      # the call sequence is a reading of Meta's class, not its documented API: say where it stops
            say(f"Meta's tracker | {name}", {"error": f"{type(e).__name__}: {e}", "where": traceback.format_exc().strip().splitlines()[-3:]})
        torch.cuda.empty_cache()
    put(a.json, "tracker, seeds " + a.seeds_from, R)


# ---- there and back: the tracker with a floor, a shape test and the range corrected as a loader would

#: reasoned, on the 252 grid the masks are compared on (a quarter of the model's side): a mask is a PLAUSIBLE subject when it
#: is not a fragment (at least this many cells, about 500 of the model's pixels), not mostly inside a band along the frame's
#: border (the band is about 25 of the model's pixels) and not a sparse scatter (it fills a tenth of its own box). After the
#: independent session's test of 2026-10-07, which found a track that slides onto a strip along the bottom edge and stays "on".
SMALL = 252
PLAUSIBLE_MIN_CELLS = 32
BORDER_CELLS = 6
BORDER_SHARE_MOST = 0.5
BOX_FILL_LEAST = 0.1
#: inherited: `bench/sam3_precision_arms.py::NUDGE_SEED`, so the nudge is the independent session's, frame for frame.
NUDGE_SEED = 1234
THERE_AND_BACK_SEEDS = (("alone", 1), ("with one neighbour", 2), ("16", 16), ("32", 32))
#: inherited: core's own line for "the same object" between detections (`comfy/ldm/sam3/tracker.py`, its mask overlap removal),
#: on the larger of intersection over union and intersection over the smaller mask.
SAME_SUBJECT = 0.5


def nudged(u8, levels: int):
    """Every value moved by `levels` of 255 at random sign, seeded per frame (`bench/sam3_precision_arms.py::nudged`)."""
    import numpy as np
    if levels == 0:
        return u8
    out = np.empty_like(u8)
    for i in range(u8.shape[0]):
        sign = np.random.default_rng(NUDGE_SEED + i).integers(0, 2, size=u8.shape[1:], dtype=np.int8) * 2 - 1
        out[i] = np.clip(u8[i].astype(np.int16) + sign * levels, 0, 255).astype(np.uint8)
    return out


def plausible(m):
    """[..., SMALL, SMALL] bool masks to [...] bool: a mask that passes the shape test above. Shape only, not identity."""
    import torch
    area = m.flatten(-2).sum(-1)
    band = torch.zeros((SMALL, SMALL), dtype=torch.bool)
    band[:BORDER_CELLS] = band[-BORDER_CELLS:] = True
    band[:, :BORDER_CELLS] = band[:, -BORDER_CELLS:] = True
    in_band = (m & band).flatten(-2).sum(-1)
    rows, cols = m.any(-1), m.any(-2)
    height = SMALL - rows.float().argmax(-1) - rows.flip(-1).float().argmax(-1)
    width = SMALL - cols.float().argmax(-1) - cols.flip(-1).float().argmax(-1)
    return (area >= PLAUSIBLE_MIN_CELLS) & (in_band <= BORDER_SHARE_MOST * area) & (area >= BOX_FILL_LEAST * height * width)


def cmd_there_and_back(a):
    torch = boot(float32=False)         # core as it computes by default; outcomes are compared, not tensors
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image

    import comfy.ldm.sam3.tracker as T
    import server
    from comfy.ldm.sam3.tracker import unpack_masks
    from comfy_extras.nodes_sam3 import SAM3_Detect, SAM3_VideoTrack
    server.PromptServer.instance = types.SimpleNamespace(prompt_queue=None, routes=None)
    sys.path.insert(0, str(COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"))
    from videohelpersuite.load_video_nodes import LoadVideoFFmpegPath
    loaded = LoadVideoFFmpegPath().load_video(video=a.clip, force_rate=float(a.rate), custom_width=a.width, custom_height=0,
                                              frame_load_cap=int(round(a.seconds * a.rate)), start_time=float(a.second), format="AnimateDiff")[0][..., :3]
    u8 = (loaded.numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    del loaded
    n, H, W = (int(v) for v in u8.shape[:3])
    core, clip = load_core(a.ckpt)
    patch_embed = core.model.diffusion_model.detector.backbone["vision_backbone"].trunk.patch_embed
    p, _ = load_meta(a.ckpt)
    from _h3pack.meta_sam3.sam3.model.io_utils import load_resource_as_video_frames
    tracker = p.model.tracker.model
    tracker.backbone = _TrackerBackbone(p.model.detector.backbone)

    def there_and_back(x):
        """The window played forward and then in reverse, so its last frame is its first."""
        return np.concatenate([x, x[::-1][1:]], axis=0)

    def as_float(x):
        return torch.from_numpy(x.astype(np.float32) / 255.0)

    class corrected:
        """The range corrected as a loader would: at the patch embedding, clamped to 0..1 and then mapped, so the trunk
        sees exactly -1..1 whatever the node's own resize overshot. In memory, on the module, for the length of the block."""

        def __enter__(self):
            stock = patch_embed.forward
            patch_embed.forward = lambda x: stock(x.clamp(0.0, 1.0) * 2.0 - 1.0)

        def __exit__(self, *exc):
            del patch_embed.__dict__["forward"]

    class as_it_is:
        def __enter__(self):
            pass

        def __exit__(self, *exc):
            pass

    R = {"environment": environment(torch), "frames": {"clip": Path(a.clip).name, "second": a.second, "count": n, "rate": a.rate, "size": [W, H],
                                                        "played": 2 * n - 1},
         "core_dtype": str(core.model.get_dtype()), "seeds_from": "one detect by core on the first frame, " + ("corrected" if a.seeds_from == "corrected" else "as its node gets it"),
         "shape_test": {"grid": SMALL, "least_cells": PLAUSIBLE_MIN_CELLS, "border_cells": BORDER_CELLS, "border_share_most": BORDER_SHARE_MOST, "box_fill_least": BOX_FILL_LEAST},
         "one_run_per_arm": True, "sets": {},
         "caution": "with several subjects Meta's tracker-only path applies neither of its two between-subject rules; core applies both"}

    cond = clip.encode_from_tokens_scheduled(clip.tokenize("person:64"))
    with torch.inference_mode(), (corrected() if a.seeds_from == "corrected" else as_it_is()):
        det = SAM3_Detect.execute(core, as_float(u8[:1]), conditioning=cond, threshold=0.5, individual_masks=True)
    masks = getattr(det, "args", det)[0].float().cpu()
    area = masks.flatten(1).sum(1)
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    cx, cy = (masks * xs).flatten(1).sum(1) / area.clamp(min=1), (masks * ys).flatten(1).sum(1) / area.clamp(min=1)
    subject = int(area.argmax())
    near = ((cx - cx[subject]) ** 2 + (cy - cy[subject]) ** 2).argsort().tolist()       # the largest subject first, then by distance from it
    R["detections_on_the_seed_frame"] = int(masks.shape[0])
    # Core's detect node removes no overlapping detections, so two of its masks can be one subject. Seeded both, core's
    # rules between objects would suppress one by design while Meta's tracker-only path keeps both, which would read as
    # "core keeps fewer". So the seeds are counted for such pairs, and `--distinct-seeds` leaves the second of each out.
    flat = (masks > 0.5).flatten(1).float()
    inter = flat @ flat.T
    sizes = flat.sum(1)
    same = torch.maximum(inter / (sizes[:, None] + sizes[None] - inter).clamp(min=1), inter / torch.minimum(sizes[:, None], sizes[None]).clamp(min=1))
    same.fill_diagonal_(0)
    R["seeds_that_are_a_second_mask_of_an_earlier_one"] = {str(k): int((same[:k, :k].triu(1) >= SAME_SUBJECT).any(0).sum())
                                                            for _, k in THERE_AND_BACK_SEEDS if k > 2 and masks.shape[0] >= k}
    order = list(range(int(masks.shape[0])))
    if a.distinct_seeds:
        order = []
        for i in [subject] + [i for i in range(int(masks.shape[0])) if i != subject]:     # the subject is never the one left out
            if not any(float(same[i, j]) >= SAME_SUBJECT for j in order):
                order.append(i)
        order = sorted(order)
        near = [i for i in near if i in order]
    R["distinct_seeds"] = bool(a.distinct_seeds)
    seed_sets = {name: (masks[near[:k]] if k <= 2 else masks[order[:k]]) for name, k in THERE_AND_BACK_SEEDS if len(order) >= k}

    def small(x):
        """[N, h, w] bool or float masks to [N, SMALL, SMALL] bool: a cell is on if any pixel under it is."""
        return F.adaptive_max_pool2d(x.float()[:, None], (SMALL, SMALL))[:, 0] > 0.5

    stock_step, logged = T.SAM31Tracker.track_step, []

    def step(self, *args, **kwargs):
        out = stock_step(self, *args, **kwargs)
        v = out.get("object_score_logits")
        logged.append(None if v is None else float(v.detach().float().flatten()[0]))
        return out

    def core_run(video, seed, how):
        logged.clear()
        T.SAM31Tracker.track_step = step
        try:
            with torch.inference_mode(), how():
                out = SAM3_VideoTrack.execute(video, core, initial_mask=seed, conditioning=None, detection_threshold=0.5, max_objects=0, detect_interval=1)
        finally:
            T.SAM31Tracker.track_step = stock_step
        packed = getattr(out, "args", out)[0]["packed_masks"]
        got = torch.zeros((seed.shape[0], int(video.shape[0]), SMALL, SMALL), dtype=torch.bool)
        if packed is not None:
            for k in range(min(int(packed.shape[1]), seed.shape[0])):
                got[k] = small(unpack_masks(packed[:, k]).cpu())
        return got, [x for x in logged if x is not None]

    def meta_run(frames_u8, seed):
        k = seed.shape[0]
        at = F.interpolate(seed[:, None], size=(SEED_SIDE, SEED_SIDE), mode="bilinear", align_corners=False)[:, 0] > 0.5
        got, scores = torch.zeros((k, frames_u8.shape[0], SMALL, SMALL), dtype=torch.bool), []
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True    # upstream's predictor sets both
        try:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                images, vh, vw = load_resource_as_video_frames([Image.fromarray(f) for f in frames_u8], image_size=SIDE, offload_video_to_cpu=False)
                st = tracker.init_state(video_height=vh, video_width=vw, num_frames=len(images))
                st["images"] = images
                tracker.add_new_masks(st, frame_idx=0, obj_ids=list(range(k)), masks=at, add_mask_to_memory=True)
                tracker.propagate_in_video_preflight(st, run_mem_encoder=True)
                for frame_idx, obj_ids, _low, video_res, obj_scores in tracker.propagate_in_video(
                        st, start_frame_idx=0, max_frame_num_to_track=None, reverse=False, tqdm_disable=True, run_mem_encoder=True):
                    ids = [int(i) for i in (obj_ids.tolist() if hasattr(obj_ids, "tolist") else obj_ids)]
                    cells = small((video_res[:, 0] > 0).cpu())
                    for j, i in enumerate(ids):
                        if 0 <= i < k:
                            got[i, frame_idx] = cells[j]
                    if 0 in ids:
                        scores.append(float(obj_scores.float().flatten()[ids.index(0)]))
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        return got, scores

    def iou(x, y):
        """[K, F, S, S] against the same: per subject, the mean over frames either has a mask on of intersection over union."""
        inter, union = (x & y).flatten(2).sum(2).float(), (x | y).flatten(2).sum(2).float()
        out = []
        for k in range(x.shape[0]):
            on = union[k] > 0
            out.append(float((inter[k][on] / union[k][on]).mean()) if bool(on.any()) else None)
        return out

    def agreement(x, y):
        v = [t for t in iou(x, y) if t is not None]
        return {"subjects_at_a_half_or_more": sum(t >= 0.5 for t in v), "median": round(float(np.median(v)), 3) if v else None, "of": int(x.shape[0])}

    def describe(got, seed_small, scores):
        ok = plausible(got)                                    # [K, F]
        on = got.flatten(2).any(2)
        last = got[:, -1]                                      # back on the first frame
        inter = (last[:, None] & seed_small[None]).flatten(2).sum(2).float()
        union = (last[:, None] | seed_small[None]).flatten(2).sum(2).float().clamp(min=1)
        back = inter / union                                   # [K, K]: each subject's last mask against every seed
        own = back.diagonal()
        home = (own >= 0.5) & (back.argmax(1) == torch.arange(back.shape[0]))
        out = {"with_a_mask": {"end_of_forward": int(on[:, n - 1].sum()), "back_at_the_start": int(on[:, -1].sum())},
               "plausible": {"end_of_forward": int(ok[:, n - 1].sum()), "back_at_the_start": int(ok[:, -1].sum())},
               "back_on_its_own_seed": int(home.sum()), "back_on_another_seed": int(((back.argmax(1) != torch.arange(back.shape[0])) & (back.max(1)[0] >= 0.5)).sum()),
               "subject_frames": {"with_a_mask": int(on.sum()), "plausible": int(ok.sum()), "of": int(on.numel())}}
        if scores:
            out["object_score_first_frames"] = [round(float(v), 2) for v in scores[:8]]
        return out

    ARMS = (("core, as its node gets the frames", "core", as_it_is, False), ("core, mapped before the node", "core", as_it_is, True),
            ("core, corrected at the first layer", "core", corrected, False), ("Meta's tracker", "meta", None, False))
    for name, seed in seed_sets.items():
        if a.only and name != a.only:
            continue
        seed_small = small(seed)
        kept, rows = {}, {}
        for arm, which, how, mapped in ARMS:
            for level in (0, 1):
                frames_u8 = there_and_back(nudged(u8, level))
                key = arm + (", nudged" if level else "")
                try:
                    if which == "meta":
                        got, scores = meta_run(frames_u8, seed)
                    else:
                        video = as_float(frames_u8)
                        got, scores = core_run(video * 2 - 1 if mapped else video, seed, how)
                    kept[key] = got
                    rows[key] = describe(got, seed_small, scores if name == "alone" else None)
                except Exception as e:      # Meta's call sequence is a reading of its class, not its documented API: say where it stops
                    rows[key] = {"error": f"{type(e).__name__}: {e}", "where": traceback.format_exc().strip().splitlines()[-3:]}
                torch.cuda.empty_cache()
        for arm, _which, _how, _mapped in ARMS:
            if arm in kept and arm + ", nudged" in kept:
                rows[arm]["against_itself_nudged"] = agreement(kept[arm], kept[arm + ", nudged"])
            for key in (arm, arm + ", nudged"):
                if key in kept and "Meta's tracker" in kept and key != "Meta's tracker":
                    rows[key]["against_metas_tracker"] = agreement(kept[key], kept["Meta's tracker"])
        R["sets"][name] = rows
        for key, row in rows.items():
            print(f"{name} | {key}", json.dumps(row), flush=True)
    put(a.json, "there and back, seeds " + a.seeds_from + (", distinct" if a.distinct_seeds else ""), R)


# ---- render

def cmd_render(a):
    D = json.loads(Path(a.json).read_text())
    if "tokens" in D:
        t = D["tokens"]
        print(f"### Tokens\n\n{t['parts']} prompt parts from {len(TOKEN_PHRASES)} typed phrases; {t['differ']} differ.\n")
    if "files" in D:
        f = D["files"]
        print(f"### The two weights files\n\n`{f['theirs']}` against `{f['ours']}` cast to float16: {f['equal']} tensors equal, {f['differs_count']} differ, "
              f"{len(f['shape_differs'])} differ in shape; only in theirs {len(f['only_theirs'])}, only in ours {f['only_ours']}.\n")
    if "text" in D:
        t = D["text"]
        print(f"### Text encoder\n\nCore's {t['core_text_mlps']} text MLPs ship quick_gelu: {t['core_ships_quick_gelu']}. Relative L2 of core's features against Meta's, real tokens only.\n")
        print("| phrase | tokens | encoder, as shipped | encoder, exact GELU | after the resizer, as shipped | after the resizer, exact GELU |\n|---|---|---|---|---|---|")
        for ph, r in t["phrases"].items():
            s, e = r["as shipped"], r["exact GELU"]
            print(f"| `{ph}` | {r['real_tokens']} | {s['encoder']['relative_l2']} | {e['encoder']['relative_l2']} | {s['after_resizer']['relative_l2']} | {e['after_resizer']['relative_l2']} |")
        print(f"\nOne detect on `{t['frame']['clip']}` at {t['frame']['second']} s, the frame in the trained range.\n")
        print("| phrase | detections, as shipped | exact GELU | presence, as shipped | exact GELU | lowest best IoU between the two sets |\n|---|---|---|---|---|---|")
        for ph, r in t["detect"].items():
            print(f"| {ph} | {r['as shipped']['detections']} | {r['exact GELU']['detections']} | {r['as shipped']['presence']} | {r['exact GELU']['presence']} | {r.get('lowest_best_iou_between_the_two', '')} |")
        print()
    for key in sorted(k for k in D if k.startswith("trunk")):
        t = D[key]
        print(f"### Image range and trunk{'' if key == 'trunk' else ', at the file' + chr(39) + 's own size'}\n\n`{t['frame']['clip']}` at {t['frame']['second']} s, {t['frame']['size'][0]}x{t['frame']['size'][1]}. "
              f"Core's node hands its trunk {t['input']['core_range']}; Meta's loader hands its trunk {t['input']['meta_range']}. "
              f"Patch embedding: weights differ by {t['patch_embedding']['max_abs_diff_of_weights']}, bias in core {t['patch_embedding']['core_has_bias']}, in Meta {t['patch_embedding']['meta_has_bias']}.\n")
        print("| trunk features, last level | relative L2 | cosine |\n|---|---|---|")
        for k, v in t["trunk"].items():
            if isinstance(v, dict) and "relative_l2" in v:
                print(f"| {k.replace('_', ' ')} | {v['relative_l2']} | {v['cosine']} |")
            elif isinstance(v, dict):
                print(f"| {k.replace('_', ' ')} | {v.get('error', '')} | |")
        self_test = t["environment"].get("tf32_selftest")
        if self_test:
            print(f"\nTF32 flags {t['environment']['tf32']}; self-test relative error against float64: convolution {self_test['conv_relative_error']:.1e}, matmul {self_test['matmul_relative_error']:.1e}.")
        print()
    for key in sorted(k for k in D if k.startswith("detector")):
        d = D[key]
        if "rules" not in d:      # the first run's shape (2026-10-07), kept as it was recorded
            print(f"### Detector, on Meta's image tensor: the first run\n\n`{d['frame']['clip']}` at {d['frame']['second']} s, phrase `{d['phrase']}`, {d['exact GELU']['queries']} queries. "
                  "\"Over 0.5\" counts queries over 0.5 on the joint score, before any overlap removal: it is not what either side returns.\n")
            print("| core's text | presence logit core / Meta | joint score logits: relative L2, cosine | queries over 0.5 core / Meta | top ten by Meta: Meta | the same queries: core | their mask IoU |\n|---|---|---|---|---|---|---|")
            for label in ("exact GELU", "as shipped"):
                r = d[label]
                j = r["joint_score_logits_all_queries"]
                print(f"| {label} | {r['presence_logit']['core']} / {r['presence_logit']['meta']} | {j['relative_l2']}, {j['cosine']} | {r['kept_over_half']['core']} / {r['kept_over_half']['meta']} | "
                      f"{r['top10_by_meta']['meta']} | {r['top10_by_meta']['core']} | {r['top10_by_meta']['mask_iou']} |")
            print()
            continue
        print(f"### Detector, on Meta's image tensor\n\n`{d['frame']['clip']}` at {d['frame']['second']} s, phrase `{d['phrase']}`, {d['exact GELU']['queries']} queries. "
              f"Rules: the node's is {d['rules']['node']}; Meta's is {d['rules']['meta']}.\n")
        print("| core's text | presence logit core / Meta | joint score logits: relative L2, cosine | queries over 0.5 on the joint score, core / Meta | top ten by Meta: Meta | the same queries: core | their mask IoU |\n|---|---|---|---|---|---|---|")
        for label in ("exact GELU", "as shipped"):
            r = d[label]
            j = r["joint_score_logits_all_queries"]
            print(f"| {label} | {r['presence_logit']['core']} / {r['presence_logit']['meta']} | {j['relative_l2']}, {j['cosine']} | {r['queries_over_half_joint']['core']} / {r['queries_over_half_joint']['meta']} | "
                  f"{r['top10_by_meta']['meta']} | {r['top10_by_meta']['core']} | {r['top10_by_meta']['mask_iou']} |")
        fl = d.get("FLOOR_core_same_frame_moved_one_level")
        if fl:
            j = fl["joint_score_logits_all_queries"]
            print(f"\nThe floor for that row: core against itself with the frame moved one level of 255: presence logit {fl['presence_logit'][0]} and {fl['presence_logit'][1]}; "
                  f"joint score logits relative L2 {j['relative_l2']}, cosine {j['cosine']}.\n")
        print("What each rule keeps of the same queries (core's text with exact GELU):\n")
        print("| frame, s | presence logit core / Meta | core's outputs, the node's rule | core's outputs, Meta's rule | Meta's outputs, Meta's rule |\n|---|---|---|---|---|")
        rows = [(d["frame"]["second"], d["exact GELU"])] + list(d.get("more_frames", {}).items())
        for second, r in rows:
            print(f"| {second} | {r['presence_logit']['core']} / {r['presence_logit']['meta']} | {r['core_outputs']['node_rule']} | {r['core_outputs']['metas_rule']} | {r['meta_outputs']['metas_rule']} |")
        self_test = d["environment"].get("tf32_selftest")
        if self_test:
            print(f"\nTF32 flags {d['environment']['tf32']}; self-test relative error against float64: convolution {self_test['conv_relative_error']:.1e}, matmul {self_test['matmul_relative_error']:.1e}.")
        print()
    for key in sorted(k for k in D if k.startswith("tracker")):
        t = D[key]
        f = t["frames"]
        print(f"### Tracker, from the same masks: seeds from {t['seeds_from']}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second, {f['size'][0]}x{f['size'][1]}; "
              f"{t['detections_on_the_seed_frame']} detections on the seed frame. Counts of seeded people with a non-empty mask. {t['caution'].capitalize()}.\n")
        cols = None
        for key, v in t["arms"].items():
            arm, seeded = key.split(" | ")
            if "error" in v:
                print(f"| {arm} | {seeded} | {v['error']} |")
                continue
            if cols is None:
                cols = list(v["with_a_mask_at"])
                print("| arm | seeded | with a mask at frames " + ", ".join(cols) + " | the last frame each is on | the lone person's object score, first frames |\n|---|---|---|---|---|")
            lasts = v["last_frame_each_is_on"]
            print(f"| {arm} | {seeded} | {list(v['with_a_mask_at'].values())} | {lasts if len(lasts) <= 2 else sorted(x for x in lasts if x is not None)} | {v.get('object_score_first_frames', '')} |")
        print()
    for key in sorted(k for k in D if k.startswith("there and back")):
        t = D[key]
        f = t["frames"]
        print(f"### There and back: seeds from {t['seeds_from']}\n\n`{f['clip']}` from {f['second']} s, {f['count']} frames at {f['rate']} a second played forward and then in reverse "
              f"({f['played']} frames, the last being the first), {f['size'][0]}x{f['size'][1]}; {t['detections_on_the_seed_frame']} detections on the seed frame. One run per arm. "
              f"A subject is PLAUSIBLE when its mask passes the shape test ({t['shape_test']}); shape, not identity. {t['caution'].capitalize()}. "
              f"Seeds that are a second mask of an earlier seed, among the first 16 and 32: {t.get('seeds_that_are_a_second_mask_of_an_earlier_one', 'not counted in this run')}"
              f"{'; such seeds were left out' if t.get('distinct_seeds') else ''}.\n")
        for name, rows in t["sets"].items():
            print(f"Seeded: {name}.\n")
            print("| arm | with a mask, end of forward | plausible, end of forward | plausible, back at the start | back on its own seed | on another's seed | plausible subject-frames | against itself nudged: at a half or more, median | against Meta's tracker: at a half or more, median |\n|---|---|---|---|---|---|---|---|---|")
            for arm, r in rows.items():
                if "error" in r:
                    print(f"| {arm} | {r['error']} | | | | | | | |")
                    continue
                n1, m1 = r.get("against_itself_nudged"), r.get("against_metas_tracker")
                cell = lambda v: "" if not v else f"{v['subjects_at_a_half_or_more']} of {v['of']}, {v['median']}"   # noqa: E731
                print(f"| {arm} | {r['with_a_mask']['end_of_forward']} | {r['plausible']['end_of_forward']} | {r['plausible']['back_at_the_start']} | {r['back_on_its_own_seed']} | "
                      f"{r['back_on_another_seed']} | {r['subject_frames']['plausible']} of {r['subject_frames']['of']} | {cell(n1)} | {cell(m1)} |")
            print()
    if "earlier_run_other_decode" in D:
        e = D["earlier_run_other_decode"]
        px = e["frames_against_the_tools"]["mean_abs_pixel_difference_in_levels_of_255"]
        print(f"### The same core arms on frames decoded another way\n\n{e['what'].capitalize()}. Seeds: {e['seeds']}. "
              f"The two sets of frames are the same frames in time and differ by {min(px)} to {max(px)} levels of 255 on average.\n")
        print("| | core, frames as its node gets them | core, frames in the trained range |\n|---|---|---|")
        for k in ("32_seeded_with_a_mask_at_the_last_frame", "17_seeded_last_frame_the_seventeenth_is_on", "alone_frames_on"):
            print(f"| {k.replace('_', ' ')} | " + " | ".join(str(v) for v in e[k].values()) + " |")
        print()


def main():
    #: the lane's segmenter, one copy: `workflows/h3_config.py::SEGMENTER`
    from h3_config import SEGMENTER
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def cmd(name, fn, frame=False):
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--json", required=True)
        if frame:
            s.add_argument("--clip", required=True)
            s.add_argument("--second", type=float, required=True)
            s.add_argument("--width", type=int, default=0, help="shrink the frame to this width first; 0 keeps the file's size")
            s.add_argument("--ckpt", default=SEGMENTER, help="a file name under models/checkpoints")
        return s

    cmd("tokens", cmd_tokens)
    s = cmd("files", cmd_files)
    s.add_argument("--theirs", required=True)
    s.add_argument("--ours", required=True)
    cmd("text", cmd_text, frame=True).add_argument("--describing", default="", help="a phrase asked of the frame; its text is not written to the json")
    cmd("trunk", cmd_trunk, frame=True)
    s = cmd("detector", cmd_detector, frame=True)
    s.add_argument("--phrase", default="person")
    s.add_argument("--more-seconds", type=float, nargs="*", help="further frames of the clip: what each detection rule keeps there")
    s = cmd("tracker", cmd_tracker, frame=True)
    s.add_argument("--seconds", type=float, required=True)
    s.add_argument("--rate", type=float, required=True, help="the loader's force_rate")
    s.add_argument("--seeds-from", choices=("node", "trained"), required=True,
                   help="the seed masks come from one detect on the first frame as core's node gets it, or in the trained range")
    s = cmd("there-and-back", cmd_there_and_back, frame=True)
    s.add_argument("--seconds", type=float, required=True)
    s.add_argument("--rate", type=float, required=True, help="the loader's force_rate")
    s.add_argument("--seeds-from", choices=("node", "corrected"), required=True,
                   help="the seed masks come from one detect on the first frame as core's node gets it, or with the range corrected at the first layer")
    s.add_argument("--distinct-seeds", action="store_true", help="leave out a seed that is a second mask of an earlier one (overlap of a half or more)")
    s.add_argument("--only", default="", help="run only this seed set, by its name (alone, with one neighbour, 16, 32)")
    cmd("render", cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
