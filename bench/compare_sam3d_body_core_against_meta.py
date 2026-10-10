#!/usr/bin/env python3
"""ComfyUI core's SAM 3D Body predict against Meta's own inference code, on the same image, boxes and camera.

Core's port (`comfy/ldm/sam3d_body`, `comfy_extras/nodes_sam3d_body.py`) is a
re-implementation, so the question is whether it computes what Meta's
released code computes from the same weights. The existing checks hold the
key mapping and the ViT-H backbone's forward; this runs the whole prediction
on both sides and reports how far apart the two bodies are.

Two processes:

  this one   core's `SAM3DBody_Loader` and `SAM3DBody_Predict`, called in
             process with the DINOv3 file, on the CPU in float32.
  a child    Meta's `SAM3DBodyEstimator.process_one_image` from
             `coderef/sam-3d-body`, with Meta's original `model.ckpt` and rig
             file, run by the Python `--meta-python` names. That Python is
             NOT this venv: Meta's code needs packages the ComfyUI venv does
             not have (see `META_NEEDS`), and nothing is installed into the
             server's venv for a one-off.

**Meta's code is run, in the child, and nothing from it enters this
process.** `AGENTS.md` says not to import Python from `coderef/`; a check
may run a coderef file as a one-off numeric reference, and that is all this
is. Nothing shipped depends on it.

The child changes three things, none of them arithmetic, each because the
reference has to run on a CPU with no network:

  - `torch.hub.load("facebookresearch/dinov3", ...)` reads the local
    `coderef/dinov3` checkout instead of GitHub;
  - `.cuda()` and a move to `"cuda"` are no-ops (Meta's inference hard-codes
    both);
  - `MOMENTUM_ENABLED` is set in the environment, which is Meta's own switch
    for loading the rig from the TorchScript file.

Arms, each run on both sides with the same boxes:

  camera `default`   no intrinsics given: both sides fall back to a focal
                     length of the image diagonal.
  camera `fov`       the same vertical field of view on both sides
                     (`--fov`), which is what core's `fov` input and Meta's
                     `cam_int` argument are for.

and on Meta's side twice: as released (the backbone in the precision the
config names) and in float32, which is the like-for-like for core on a CPU.
The difference between Meta's two is printed as a floor to read the
core-against-Meta rows against.

**The control.** Core samples the crop with `grid_sample` at pixel centres and
floors it; Meta samples it with OpenCV on the 8-bit image. `core_opencv_crop`
is core's model run again with the one function that does that
(`warp_affine_batched`) replaced, in this process only, by OpenCV's call as
Meta makes it. If core's distance to Meta falls to the floor under it, the
crop is what separates the two; if it does not, something else does.

Whether the hand decoder's result was USED is read the same way on both
sides, from what `run_inference` returns: the final hand pose equals the hand
decoder's output for that hand exactly when the four tests passed, because
the code copies it in with `torch.where`. Neither code returns the flag. (A
first version compared a full run with a body-only run; that says "used"
always, because a full run re-runs the body decoder with wrist and elbow
prompts and the hand pose changes either way.) The side of each hand's crop
in source pixels is recorded beside it: it is the quantity the second of the
four tests compares with a constant.

**What this does not say.** It is core on the CPU in float32. The server
runs core in half precision on the card through a different model patcher,
and `bench/results/2026-10-05_sam3d_body_vith.md` is the record of how far
those two were apart on one frame. No detector, no mask conditioning, no
track, no smoothing, no ViT-H.

The image is whatever the caller names. With the owner's media the outputs go
under `internal/` and are never committed; the JSON carries the `--label`,
not a path.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/compare_sam3d_body_core_against_meta.py \\
        --image coderef/sam-3d-body/notebook/images/dancing.jpg --label dancing \\
        --boxes whole --boxes-from-mask coderef/sam-3d-body/notebook/images/dancing_mask.png \\
        --meta-python <a python with META_NEEDS> --meta-weights <dir with model.ckpt> \\
        --out-dir <dir>

The Python for Meta's side, built once, outside the repo (as it was on
2026-10-10; `META_NEEDS` is the list the tool checks before it starts):

    uv venv --python 3.12 <dir>
    uv pip install --python <dir>/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python <dir>/bin/python numpy opencv-python-headless roma pytorch-lightning yacs einops \\
        timm omegaconf braceexpand pillow termcolor ftfy regex scikit-learn submitit torchmetrics

    <python> bench/compare_sam3d_body_core_against_meta.py gather <out.json> <a run's JSON>...
    <python> bench/compare_sam3d_body_core_against_meta.py render <the JSON files>   # the record's tables
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import REPO, bootstrap, needs, server_memory_mode  # noqa: E402

DINOV3_FILE = "sam_3d_body_dinov3.safetensors"
META_REPO = REPO / "coderef" / "sam-3d-body"
DINOV3_REPO = REPO / "coderef" / "dinov3"
# read from the imports of Meta's inference path on 2026-10-10 (inherited, not chosen)
META_NEEDS = ("torch", "torchvision", "numpy", "cv2", "roma", "pytorch_lightning", "yacs", "omegaconf",
              "einops", "timm", "braceexpand",
              # coderef/dinov3's hubconf imports its whole package: its requirements.txt
              "termcolor", "ftfy", "regex", "sklearn", "submitit", "torchmetrics")

# what both sides return per person, under the names Meta's estimator uses
FIELDS = ("pred_vertices", "pred_keypoints_3d", "pred_keypoints_2d", "pred_cam_t", "focal_length",
          "global_rot", "body_pose_params", "hand_pose_params", "shape_params", "scale_params", "expr_params")
HAND_SPLIT = 54  # MHRHead.num_hand_comps, left then right (inherited: mhr_head.py)
# index ranges of Meta's 70 keypoints (inherited: coderef/sam-3d-body/sam_3d_body/metadata/mhr70.py)
GROUPS = {"head": list(range(0, 5)), "body": list(range(5, 21)) + list(range(63, 70)),
          "right_hand": list(range(21, 42)), "left_hand": list(range(42, 63))}
WRIST = {"right_hand": 41, "left_hand": 62}

CHILD = r'''
import json, os, sys
os.environ.setdefault("MOMENTUM_ENABLED", "0")
import numpy as np, torch
job = json.load(open(sys.argv[1]))
sys.path.insert(0, job["meta_repo"])
torch.set_num_threads(job["threads"])

_hub_load = torch.hub.load
def _local_hub(repo, name, *a, source="github", **k):
    if repo == "facebookresearch/dinov3":
        return _hub_load(job["dinov3_repo"], name, *a, source="local", **k)
    return _hub_load(repo, name, *a, source=source, **k)
torch.hub.load = _local_hub
torch.Tensor.cuda = lambda self, *a, **k: self
torch.cuda.empty_cache = lambda: None

import sam_3d_body.utils.dist as dist
_to = dist.recursive_to
def _cpu_to(x, target):
    return _to(x, "cpu" if target == "cuda" else target)
import sam_3d_body.sam_3d_body_estimator as est_mod
import sam_3d_body.models.meta_arch.sam3d_body as arch
est_mod.recursive_to = _cpu_to
arch.recursive_to = _cpu_to

_captured = {}
_run_inference = arch.SAM3DBody.run_inference
def _capturing(self, *a, **k):
    _captured["out"] = _run_inference(self, *a, **k)
    return _captured["out"]
arch.SAM3DBody.run_inference = _capturing

def hand_facts():
    # run_inference's own return for a full run: the final pose, the two hand batches, the two hand outputs
    pose, batch_l, batch_r, out_l, out_r = _captured["out"]
    final = pose["mhr"]["hand"].float()
    used = torch.stack([(final[:, :54] == out_l["mhr_hand"]["hand"][:, :54].float()).all(dim=1),
                        (final[:, 54:] == out_r["mhr_hand"]["hand"][:, 54:].float()).all(dim=1)], dim=1)
    side = torch.stack([batch_l["bbox_scale"].flatten(0, 1)[:, 0], batch_r["bbox_scale"].flatten(0, 1)[:, 0]], dim=1)
    return used.float().cpu().numpy(), side.float().cpu().numpy()

from sam_3d_body.utils.config import get_config
from sam_3d_body.utils.checkpoint import load_state_dict

def load(full_precision):
    # Meta's build_models.load_sam_3d_body, with the one config switch this arm is about
    cfg = get_config(os.path.join(job["weights"], "model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = os.path.join(job["weights"], "assets", "mhr_model.pt")
    if full_precision:
        cfg.TRAIN.USE_FP16 = False
    cfg.freeze()
    model = arch.SAM3DBody(cfg)
    ckpt = torch.load(os.path.join(job["weights"], "model.ckpt"), map_location="cpu", weights_only=True)
    load_state_dict(model, ckpt.get("state_dict", ckpt), strict=False)
    return model.to("cpu").eval(), cfg

import cv2
img = cv2.cvtColor(cv2.imread(job["image"]), cv2.COLOR_BGR2RGB)
boxes = np.asarray(job["boxes"], dtype=np.float32)
out = {}
for precision in job["precisions"]:
    model, cfg = load(precision == "float32")
    estimator = est_mod.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)
    out[f"{precision}/backbone_dtype"] = np.array(str(model.backbone_dtype))
    for camera, cam in job["cameras"].items():
        for kind in ("full", "body"):
            cam_int = None if cam is None else torch.tensor(cam, dtype=torch.float32)
            people = estimator.process_one_image(img.copy(), bboxes=boxes.copy(), cam_int=cam_int,
                                                 inference_type=kind)
            for i, p in enumerate(people):
                for key in job["fields"]:
                    out[f"{precision}/{camera}/{kind}/{i}/{key}"] = np.asarray(p[key], dtype=np.float32)
            if kind == "full":
                used, side = hand_facts()
                for i in range(len(people)):
                    out[f"{precision}/{camera}/full/{i}/hand_decoder_used"] = used[i]
                    out[f"{precision}/{camera}/full/{i}/hand_crop_side_px"] = side[i]
    del model, estimator
np.savez(job["out"], **out)
print("meta child wrote", len(out), "arrays")
'''


def _first(node_output):
    """The first value of an `io.NodeOutput`."""
    return node_output.args[0] if hasattr(node_output, "args") else node_output[0]


def _intrinsics(height: int, width: int, fov_degrees: float):
    """Core's `cam_int_from_fov`, written out for the child: the vertical focal on both axes."""
    focal = height / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    return [[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]]


def _box_of_mask(path: Path):
    from PIL import Image
    mask = np.asarray(Image.open(path).convert("L")) > 127
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def opencv_warp(src_t, mats, output_size):
    """`warp_affine_batched`'s signature, computed as Meta's `TopdownAffine` does: 8-bit, cv2.INTER_LINEAR."""
    import cv2
    h_out, w_out = int(output_size[0]), int(output_size[1])
    out = []
    for image, mat in zip(src_t, mats):
        pixels = image.permute(1, 2, 0).cpu().numpy()
        pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
        warped = cv2.warpAffine(pixels, mat.cpu().numpy().astype(np.float64), (w_out, h_out),
                                flags=cv2.INTER_LINEAR)
        if warped.ndim == 2:
            warped = warped[..., None]
        out.append(torch.from_numpy(warped).permute(2, 0, 1).float())
    return torch.stack(out).to(src_t.device)


def run_core(image, boxes_xyxy, cameras, nodes, patcher):
    """core's predict for every camera, full and body-only: {camera/kind/person/field: array}."""
    boxes = [{"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1]} for b in boxes_xyxy]
    out, captured = {}, {}
    inner = patcher.model
    original = inner.run_inference

    def capturing(*a, **k):
        captured["out"] = original(*a, **k)
        return captured["out"]

    inner.run_inference = capturing
    try:
        for camera, fov in cameras.items():
            for kind, hands in (("full", True), ("body", False)):
                pose = _first(nodes.SAM3DBody_Predict.execute(
                    patcher, image, bboxes=boxes, run_hand_refinement=hands, fov=float(fov or 0.0)))
                for i, person in enumerate(pose["frames"][0]):
                    for key in FIELDS:
                        out[f"{camera}/{kind}/{i}/{key}"] = np.asarray(person[key], dtype=np.float32)
                if kind == "full":
                    # the same reading as the child's `hand_facts`, on core's own return
                    final_pose, batch_l, batch_r, out_l, out_r = captured["out"]
                    final = final_pose["mhr"]["hand"].float()
                    used = torch.stack(
                        [(final[:, :HAND_SPLIT] == out_l["mhr_hand"]["hand"][:, :HAND_SPLIT].float()).all(dim=1),
                         (final[:, HAND_SPLIT:] == out_r["mhr_hand"]["hand"][:, HAND_SPLIT:].float()).all(dim=1)],
                        dim=1).float().cpu().numpy()
                    side = torch.stack([batch_l["bbox_scale"].flatten(0, 1)[:, 0],
                                        batch_r["bbox_scale"].flatten(0, 1)[:, 0]], dim=1).float().cpu().numpy()
                    for i in range(len(boxes)):
                        out[f"{camera}/full/{i}/hand_decoder_used"] = used[i]
                        out[f"{camera}/full/{i}/hand_crop_side_px"] = side[i]
    finally:
        del inner.run_inference
    return out


def _stats(a, b):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean())}


def difference(a: dict, b: dict, prefix_a: str, prefix_b: str) -> dict:
    """How one person's prediction differs between two runs, in the units each field is in."""
    get_a = lambda key: a[f"{prefix_a}/{key}"]  # noqa: E731
    get_b = lambda key: b[f"{prefix_b}/{key}"]  # noqa: E731
    out = {}
    va, vb = get_a("pred_vertices"), get_b("pred_vertices")
    raw = np.linalg.norm(va - vb, axis=-1)
    centred = np.linalg.norm((va - va.mean(0)) - (vb - vb.mean(0)), axis=-1)
    out["vertices_m"] = {"mean": float(raw.mean()), "max": float(raw.max()),
                         "mean_after_centring": float(centred.mean())}
    j = np.linalg.norm(get_a("pred_keypoints_3d") - get_b("pred_keypoints_3d"), axis=-1)
    out["keypoints_3d_m"] = {"mean": float(j.mean()), "max": float(j.max())}
    k = np.linalg.norm(get_a("pred_keypoints_2d")[..., :2] - get_b("pred_keypoints_2d")[..., :2], axis=-1)
    out["keypoints_2d_px"] = {"mean": float(k.mean()), "max": float(k.max())}
    out["groups"] = {}
    for name, idx in GROUPS.items():
        row = {"keypoints_3d_m_mean": float(j[idx].mean()), "keypoints_2d_px_mean": float(k[idx].mean()),
               "keypoints_2d_px_max": float(k[idx].max())}
        if name in WRIST:
            # the hand's own shape: each side's keypoints about its wrist
            ra = get_a("pred_keypoints_3d")[idx] - get_a("pred_keypoints_3d")[WRIST[name]]
            rb = get_b("pred_keypoints_3d")[idx] - get_b("pred_keypoints_3d")[WRIST[name]]
            row["about_the_wrist_m_mean"] = float(np.linalg.norm(ra - rb, axis=-1).mean())
        out["groups"][name] = row
    out["camera_translation_m"] = {"a": get_a("pred_cam_t").reshape(-1).tolist(),
                                   "b": get_b("pred_cam_t").reshape(-1).tolist()}
    out["focal_length_px"] = {"a": float(np.asarray(get_a("focal_length")).reshape(-1)[0]),
                              "b": float(np.asarray(get_b("focal_length")).reshape(-1)[0])}
    for key in ("global_rot", "body_pose_params", "hand_pose_params", "shape_params", "scale_params"):
        out[key] = _stats(get_a(key), get_b(key))
    return out


def hand_refinement(run: dict, base: str, person: int) -> dict:
    """`base` is everything before the kind, e.g. "default" or "float32/default"."""
    used = run[f"{base}/full/{person}/hand_decoder_used"]
    side = run[f"{base}/full/{person}/hand_crop_side_px"]
    return {"left_hand_decoder_used": bool(used[0] > 0.5), "right_hand_decoder_used": bool(used[1] > 0.5),
            "left_hand_crop_side_px": float(side[0]), "right_hand_crop_side_px": float(side[1]),
            "expression_all_zero": bool(np.all(run[f"{base}/full/{person}/expr_params"] == 0))}


ROWS = (("core against Meta", "core_against_meta_float32"),
        ("core against Meta, body decoder only", "body_only_core_against_meta_float32"),
        ("core with OpenCV's crop against Meta (the control)", "control_core_opencv_crop_against_meta_float32"),
        ("Meta as released against Meta in float32 (the floor)", "floor_meta_released_against_meta_float32"))


def render(paths) -> str:
    """The record's tables, printed from the JSON files this tool wrote."""
    lines = ["| image | person | box, model pixels per source pixel | camera | comparison | vertices, mean (mm) "
             "| after centring (mm) | 2D keypoints, mean (px) | head (px) | body (px) | right hand (px) "
             "| left hand (px) | hand shape about the wrist, right, left (mm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    hands = ["| image | person | camera | side | left hand: decoder used, crop side (px) "
             "| right hand: decoder used, crop side (px) | expression all zero |", "|---|---|---|---|---|---|---|"]
    yes = lambda flag: "yes" if flag else "no"  # noqa: E731
    records = []
    for path in paths:
        loaded = json.loads(Path(path).read_text())
        records += loaded["records"] if "records" in loaded else [loaded]
    for record in records:
        for row in record["people"]:
            x1, y1, x2, y2 = record["boxes_xyxy"][row["person"]]
            w, h = (x2 - x1) * 1.25, (y2 - y1) * 1.25  # GetBBoxCenterScale's padding (inherited)
            side = max(w, w / 0.75) if w > h * 0.75 else h  # TopdownAffine: 3:4, then the square input
            whole = [x1, y1, x2, y2] == [0.0, 0.0, *map(float, record["image_size_wh"])]
            box = f"{'whole frame' if whole else 'on the person'}, {512 / side:.2f}"
            for camera, c in row["cameras"].items():
                for label, key in ROWS:
                    d, g = c[key], c[key]["groups"]
                    lines.append(
                        f"| {record['image']} | {row['person']} | {box} | {camera} | {label} "
                        f"| {d['vertices_m']['mean'] * 1000:.2f} | {d['vertices_m']['mean_after_centring'] * 1000:.2f} "
                        f"| {d['keypoints_2d_px']['mean']:.2f} | {g['head']['keypoints_2d_px_mean']:.2f} "
                        f"| {g['body']['keypoints_2d_px_mean']:.2f} | {g['right_hand']['keypoints_2d_px_mean']:.2f} "
                        f"| {g['left_hand']['keypoints_2d_px_mean']:.2f} "
                        f"| {g['right_hand']['about_the_wrist_m_mean'] * 1000:.2f}, "
                        f"{g['left_hand']['about_the_wrist_m_mean'] * 1000:.2f} |")
                for label, key in (("core", "core"), ("core with OpenCV's crop", "core_opencv_crop"),
                                   ("Meta float32", "meta_float32"), ("Meta as released", "meta_released")):
                    h_ = c["hands"][key]
                    hands.append(
                        f"| {record['image']} | {row['person']} | {camera} | {label} "
                        f"| {yes(h_['left_hand_decoder_used'])}, {h_['left_hand_crop_side_px']:.0f} "
                        f"| {yes(h_['right_hand_decoder_used'])}, {h_['right_hand_crop_side_px']:.0f} "
                        f"| {yes(h_['expression_all_zero'])} |")
    return "\n".join(lines) + "\n\n" + "\n".join(hands) + "\n"


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "render":
        print(render(sys.argv[2:]))
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "gather":
        # several runs' JSON into the one file a record names: gather <out.json> <in.json>...
        runs = [json.loads(Path(q).read_text()) for q in sys.argv[3:]]
        Path(sys.argv[2]).write_text(json.dumps(
            {"script": "bench/compare_sam3d_body_core_against_meta.py", "records": runs}, indent=1) + "\n")
        return 0
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--label", required=True, help="what the record calls this image; never a path")
    ap.add_argument("--boxes", action="append", default=[],
                    help="`whole`, or x1,y1,x2,y2 in pixels; repeat for more people")
    ap.add_argument("--boxes-from-mask", action="append", default=[], type=Path,
                    help="a mask image; its bounding box is one more person")
    ap.add_argument("--fov", type=float, default=55.0,
                    help="vertical field of view for the `fov` arm, degrees (reasoned: an ordinary lens; "
                         "the arm tests that both sides take the same intrinsics, not this value)")
    ap.add_argument("--meta-python", required=True, type=Path)
    ap.add_argument("--meta-weights", required=True, type=Path,
                    help="Meta's release folder: model.ckpt, model_config.yaml, assets/mhr_model.pt")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    needs(f"the image {args.image.name}", args.image.is_file())
    needs("coderef/sam-3d-body", (META_REPO / "sam_3d_body").is_dir())
    needs("coderef/dinov3", (DINOV3_REPO / "hubconf.py").is_file())
    for name in ("model.ckpt", "model_config.yaml", "assets/mhr_model.pt"):
        needs(f"Meta's {name}", (args.meta_weights / name).is_file())
    probe = subprocess.run([str(args.meta_python), "-c", "import " + ", ".join(META_NEEDS)],
                           capture_output=True, text=True)
    needs(f"a --meta-python with {', '.join(META_NEEDS)}", probe.returncode == 0)

    from PIL import Image
    pixels = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = pixels.shape[:2]
    boxes = []
    for spec in args.boxes:
        boxes.append([0.0, 0.0, float(width), float(height)] if spec == "whole"
                     else [float(v) for v in spec.split(",")])
    boxes += [_box_of_mask(p) for p in args.boxes_from_mask]
    if not boxes:
        ap.error("give at least one --boxes or --boxes-from-mask")
    cameras = {"default": None, "fov": args.fov}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    # Meta, in the child
    meta_npz = args.out_dir / f"meta_{args.label}.npz"
    with tempfile.TemporaryDirectory() as tmp:
        job = {"meta_repo": str(META_REPO), "dinov3_repo": str(DINOV3_REPO), "weights": str(args.meta_weights),
               "image": str(args.image.resolve()), "boxes": boxes, "fields": list(FIELDS),
               "precisions": ["released", "float32"], "threads": args.threads, "out": str(meta_npz),
               "cameras": {"default": None, "fov": _intrinsics(height, width, args.fov)}}
        job_path = Path(tmp) / "job.json"
        job_path.write_text(json.dumps(job))
        child_path = Path(tmp) / "meta_child.py"
        child_path.write_text(CHILD)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MOMENTUM_ENABLED="0")
        done = subprocess.run([str(args.meta_python), str(child_path), str(job_path)], env=env,
                              capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stdout[-2000:])
            print(done.stderr[-4000:])
            raise SystemExit("Meta's side did not run; nothing is compared")
        print(done.stdout.strip().splitlines()[-1])
    meta = dict(np.load(meta_npz))

    # core, here
    bootstrap()
    sys.path.append(str(REPO))
    dynamic_vram = server_memory_mode()
    import comfy.model_management as mm
    import comfy_extras.nodes_sam3d_body as nodes
    import folder_paths
    needs(f"models/detection/{DINOV3_FILE}", DINOV3_FILE in folder_paths.get_filename_list("detection"))
    image = torch.from_numpy(pixels).float().div(255)[None]
    with torch.inference_mode():
        patcher = _first(nodes.SAM3DBody_Loader.execute(DINOV3_FILE))
        core = run_core(image, boxes, cameras, nodes, patcher)
        core_dtype, core_device = str(patcher.model.backbone_dtype), str(mm.get_torch_device())
        # the control: the same model, the crop sampled as Meta samples it
        import comfy.ldm.sam3d_body.model.model as core_model
        import comfy.ldm.sam3d_body.utils as core_utils
        kept = (core_utils.warp_affine_batched, core_model.warp_affine_batched)
        core_utils.warp_affine_batched = core_model.warp_affine_batched = opencv_warp
        try:
            control = run_core(image, boxes, cameras, nodes, patcher)
        finally:
            core_utils.warp_affine_batched, core_model.warp_affine_batched = kept
    np.savez(args.out_dir / f"core_{args.label}.npz", **core)
    np.savez(args.out_dir / f"core_opencv_crop_{args.label}.npz", **control)

    record = {
        "script": "bench/compare_sam3d_body_core_against_meta.py",
        "image": args.label, "image_size_wh": [width, height], "boxes_xyxy": boxes,
        "fov_arm_degrees": args.fov,
        "core": {"file": DINOV3_FILE, "backbone_dtype": core_dtype, "device": core_device,
                 "dynamic_vram": dynamic_vram, "torch": torch.__version__},
        "meta": {"released_backbone_dtype": str(meta["released/backbone_dtype"]),
                 "float32_backbone_dtype": str(meta["float32/backbone_dtype"]), "device": "cpu"},
        "people": [],
    }
    for person in range(len(boxes)):
        row = {"person": person, "cameras": {}}
        for camera in cameras:
            row["cameras"][camera] = {
                "core_against_meta_float32": difference(
                    core, meta, f"{camera}/full/{person}", f"float32/{camera}/full/{person}"),
                "core_against_meta_released": difference(
                    core, meta, f"{camera}/full/{person}", f"released/{camera}/full/{person}"),
                "floor_meta_released_against_meta_float32": difference(
                    meta, meta, f"released/{camera}/full/{person}", f"float32/{camera}/full/{person}"),
                "body_only_core_against_meta_float32": difference(
                    core, meta, f"{camera}/body/{person}", f"float32/{camera}/body/{person}"),
                "control_core_opencv_crop_against_meta_float32": difference(
                    control, meta, f"{camera}/full/{person}", f"float32/{camera}/full/{person}"),
                "control_body_only_core_opencv_crop_against_meta_float32": difference(
                    control, meta, f"{camera}/body/{person}", f"float32/{camera}/body/{person}"),
                "hands": {"core": hand_refinement(core, camera, person),
                          "core_opencv_crop": hand_refinement(control, camera, person),
                          "meta_float32": hand_refinement(meta, f"float32/{camera}", person),
                          "meta_released": hand_refinement(meta, f"released/{camera}", person)},
            }
        record["people"].append(row)
    json_path = args.out_dir / f"sam3d_body_core_against_meta_{args.label}.json"
    json_path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {json_path.name} in the out dir")
    for row in record["people"]:
        for camera, c in row["cameras"].items():
            d, f = c["core_against_meta_float32"], c["floor_meta_released_against_meta_float32"]
            k = c["control_core_opencv_crop_against_meta_float32"]
            print(f"person {row['person']} camera {camera}: vertices mean {d['vertices_m']['mean']:.5f} m, "
                  f"with OpenCV's crop {k['vertices_m']['mean']:.5f}, floor {f['vertices_m']['mean']:.5f}; "
                  f"2D keypoints mean {d['keypoints_2d_px']['mean']:.3f} px, with OpenCV's crop "
                  f"{k['keypoints_2d_px']['mean']:.3f}, floor {f['keypoints_2d_px']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
