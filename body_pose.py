"""A body mesh from frames and boxes, with the crop sampled the way Meta's SAM 3D Body code samples it.

Three nodes, so that no ComfyUI SAM node is wired into a graph of ours:

  `MiniMaxH3BodyModelLoader`  either release of SAM 3D Body from `models/detection/`.
  `MiniMaxH3BodyPose`         frames and per-frame boxes in (`MiniMaxH3SubjectBoxes`' output, unchanged); the pose
                              data out, and a table of what a preflight reads.
  `MiniMaxH3BodyMeshVideo`    the pose data drawn as frames on black, for a motion video.

**What is ours and what is called.** ComfyUI's MODEL code is called as a library and never as a node: the model
class (`comfy/ldm/sam3d_body/model/model.py::SAM3DBody`, and `sam3d_body_vith.py`'s subclass of it for the ViT-H
release) and, for drawing, `comfy_extras/sam3d_body/rasterizer.py::render_pose_data_torch`. Both were measured on
2026-10-10: the model computes what Meta's inference code computes once both are given the same crop
(`bench/results/2026-10-10_sam3d_body_core_against_meta.md`), and the rasteriser puts the mesh where a direct
projection by the predicted camera puts it (`bench/check_body_pose.py`). Nothing is imported from `coderef/`.

**What differs from ComfyUI's own predict node, and why.** That record found one difference between ComfyUI's port
and Meta's code: how the crop is sampled. ComfyUI warps in float with `grid_sample` at pixel centres and floors;
Meta's estimator warps the 8-bit image with OpenCV. Here the crop is Meta's, transcribed (each function below names
the file and line it follows), for the body crop and for the two hand crops, which the model makes inside
`run_inference`: the loader builds a subclass that overrides the one method that makes them. So the model is given
the pixels Meta's code would give it.

**What the table adds.** Neither Meta's code nor ComfyUI's node says whether a hand was refined. This node reads
it from what `run_inference` returns: the final hand pose equals the hand decoder's output for that hand exactly
when the model's four tests passed, because the model copies it in with `torch.where`. The table carries that per
hand, with the side of the hand's crop, which is the quantity the second of those tests compares with
`HAND_BOX_THRESHOLD_PX`.

OpenCV is needed when a crop is made, not when the pack loads: `_cv2` raises with the package's name.

Not here: mask conditioning (Meta's default is off), a field of view estimated from the picture, smoothing over
time, and a face (the model's own head sets expression and jaw to zero).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np
import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.storage
import comfy.utils
import folder_paths
from comfy.ldm.sam3d_body.model.model import SAM3DBody
from comfy_api.latest import io

from .sam3d_body_vith import VITH_MARKER_KEY, SAM3DBodyModel, SAM3DBodyViTH, loader_view

logger = logging.getLogger(__name__)

# The pose data's wire type is ComfyUI's, so a consumer of one reads the other. Only the name is shared.
MHRPoseData = io.Custom("MHR_POSE_DATA")

TABLE_SCHEMA = "h3_body_pose_table/1"
#: The body crop's padding. Inherited: `GetBBoxCenterScale`'s default, coderef/sam-3d-body/sam_3d_body/data/transforms/common.py:110.
BODY_PADDING = 1.25
#: The hand crop's padding. Inherited: `transform_hand`, coderef/sam-3d-body/sam_3d_body/sam_3d_body_estimator.py:57.
HAND_PADDING = 0.9
#: The width over height a box is first widened to. Inherited: `TopdownAffine`'s default, common.py:229.
PRIOR_ASPECT = 0.75
#: The side a hand's crop must exceed, in source pixels, for its decoder's result to be used. Inherited:
#: coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1317, and ComfyUI's copy inside
#: `SAM3DBody.run_inference`. The model does the test; this copy is only written into the table, and
#: `bench/check_body_pose.py` fails when ComfyUI's source no longer holds the same figure.
HAND_BOX_THRESHOLD_PX = 64
#: Meta's fourth argument to `run_inference`. Inherited: sam_3d_body_estimator.py:36.
WRIST_ANGLE_THRESHOLD = 1.4
#: `MHRHead.num_hand_comps`: the hand pose is the left hand's components, then the right's. Inherited.
HAND_SPLIT = 54
N_KEYPOINTS_2D = 70
#: The widget's range for a field of view, degrees. Reasoned: a lens, not a fisheye.
FOV_MIN, FOV_MAX, FOV_DEFAULT = 5.0, 120.0, 55.0
#: Crops per forward pass. Inherited: ComfyUI's own default for its predict node.
BATCH_DEFAULT, BATCH_MAX = 64, 512
#: The precision the model is loaded and run in. Measured: see `load_model`.
MODEL_DTYPE = torch.float32
#: ComfyUI's estimate of the memory a forward needs was calibrated in half precision (`SAM3DBody.memory_used_forward`).
#: Reasoned: twice the bytes a value.
MEMORY_FACTOR = 2
#: The 70 keypoints by part of the body, for a report a person reads. Inherited: the order of Meta's `mhr70`
#: (coderef/sam-3d-body/sam_3d_body/metadata/mhr70.py).
PARTS = {"head": tuple(range(0, 5)), "shoulders and elbows": (5, 6, 7, 8, 63, 64, 65, 66, 67, 68, 69),
         "hips": (9, 10), "knees and ankles": (11, 12, 13, 14), "feet": tuple(range(15, 21)),
         "right hand": tuple(range(21, 42)), "left hand": tuple(range(42, 63))}
CAMERAS = ("image diagonal", "field of view")
STYLES = ("mesh", "silhouette")
SIZES = ("the source's", "width and height")


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # the pack still loads without it; this node says what it needs
        raise RuntimeError(
            "MiniMax H3 Body Pose samples its crops with OpenCV, as Meta's SAM 3D Body code does, and OpenCV "
            "is not installed in this Python. Install the package `opencv-python-headless`.") from exc
    return cv2


# ----------------------------------------------------------------------------- Meta's crop, transcribed

def center_scale(box_xyxy: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    """A box as a centre and a padded size. `bbox_xyxy2cs`, coderef/sam-3d-body/sam_3d_body/data/transforms/bbox_utils.py:45."""
    x1, y1, x2, y2 = (np.float32(v) for v in box_xyxy)
    center = np.array([x1 + x2, y1 + y2], dtype=np.float32) * np.float32(0.5)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * padding
    return center, scale


def fix_aspect(scale: np.ndarray, aspect: float) -> np.ndarray:
    """The size widened on one side to a width over height. `fix_aspect_ratio`, bbox_utils.py:231."""
    w, h = np.hsplit(scale[None, :], [1])  # arrays, as Meta's are: a scalar would change the float type
    return np.where(w > h * aspect, np.hstack([w, w / aspect]), np.hstack([h * aspect, h]))[0]


def crop_scale(scale: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """`TopdownAffine`'s two widenings: to the prior aspect, then to the model's input. common.py:264."""
    w, h = out_wh
    return fix_aspect(fix_aspect(scale, PRIOR_ASPECT), w / h)


def warp_matrix(center: np.ndarray, scale: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    """The affine from the box to the model's input, as three points. `get_warp_matrix` with no rotation, bbox_utils.py:308."""
    cv2 = _cv2()
    dst_w, dst_h = out_wh
    src_w = scale[0]
    src_dir = np.array([0.0, src_w * -0.5])
    dst_dir = np.array([0.0, dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _third_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _third_point(dst[0, :], dst[1, :])
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`_get_3rd_point`, bbox_utils.py:381: `b` turned a quarter about itself from `a`."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def crop(pixels: np.ndarray, box_xyxy: np.ndarray, padding: float, out_wh: tuple[int, int]):
    """One box of an 8-bit image as the model's input: (the crop, its centre, its size, its affine).

    `pixels` is [H, W, 3] uint8. The crop is 8-bit: Meta warps the 8-bit image (`cv2.warpAffine`, linear,
    common.py:305) and divides by 255 afterwards (`ToTensor`).
    """
    cv2 = _cv2()
    center, scale = center_scale(box_xyxy, padding)
    scale = crop_scale(scale, out_wh)
    mat = warp_matrix(center, scale, out_wh)
    out = cv2.warpAffine(np.ascontiguousarray(pixels), mat, (int(out_wh[0]), int(out_wh[1])), flags=cv2.INTER_LINEAR)
    return out, center, scale, mat


def intrinsics(height: int, width: int, camera: str, fov_degrees: float) -> torch.Tensor:
    """The camera both crops are read through, [1, 3, 3].

    "image diagonal" is Meta's fallback when no field of view is estimated
    (coderef/sam-3d-body/sam_3d_body/data/utils/prepare_batch.py:66); "field of view" is a vertical angle on both
    axes, the form Meta's own estimate is put in (coderef/sam-3d-body/tools/build_fov_estimator.py:44).
    """
    if camera == CAMERAS[0]:
        focal = (height ** 2 + width ** 2) ** 0.5
    elif camera == CAMERAS[1]:
        focal = height / (2.0 * math.tan(math.radians(float(fov_degrees)) / 2.0))
    else:
        raise ValueError(f"camera must be one of {CAMERAS}; got {camera!r}")
    return torch.tensor([[[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]], dtype=torch.float32)


def _batch(crops, centers, scales, mats, boxes, cam_int: torch.Tensor, out_wh, device) -> dict:
    """The dict the model's forward reads, for N crops of one kind. The keys are `prepare_batch`'s (prepare_batch.py:46)."""
    n = len(crops)
    w, h = int(out_wh[0]), int(out_wh[1])
    img = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).float().div(255.0)
    f32 = lambda rows: torch.from_numpy(np.stack(rows).astype(np.float32))  # noqa: E731
    batch = {
        "img": img.unsqueeze(0),
        "img_size": torch.tensor([w, h], dtype=torch.float32).expand(n, 2).contiguous().unsqueeze(0),
        "bbox_center": f32(centers).unsqueeze(0),
        "bbox_scale": f32(scales).unsqueeze(0),
        "bbox": f32(boxes).unsqueeze(0),
        "affine_trans": f32(mats).unsqueeze(0),
        "mask": torch.zeros((1, n, 1, h, w), dtype=torch.float32),
        "mask_score": torch.zeros((1, n), dtype=torch.float32),
        "person_valid": torch.ones((1, n), dtype=torch.float32),
        "cam_int": cam_int.to(torch.float32),
    }
    return {k: v.to(device) for k, v in batch.items()}


# ----------------------------------------------------------------------------- the model, with Meta's hand crops

class _MetaHandCrops:
    """Replaces the one method of ComfyUI's model that crops the hands, so they are sampled as Meta samples them.

    Meta mirrors the image for the left hand and crops both with `transform_hand`
    (coderef/sam-3d-body/sam_3d_body/models/meta_arch/sam3d_body.py:1238 and :1279). The caller, ComfyUI's
    `run_inference`, has already mirrored the left boxes' x, as Meta's does.
    """

    def _prepare_hand_batches_gpu(self, img, left_xyxy, right_xyxy, cam_int, is_multi_image):
        left = left_xyxy.detach().float().cpu().numpy()
        right = right_xyxy.detach().float().cpu().numpy()
        n = int(left.shape[0])
        frames = list(img) if is_multi_image else [img] * n
        out_wh = (int(self.image_size[1]), int(self.image_size[0]))
        device = comfy.model_management.get_torch_device()
        built = []
        for boxes, mirrored in ((left, True), (right, False)):
            rows = []
            for frame, box in zip(frames, boxes):
                pixels = frame.cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
                if mirrored:
                    pixels = pixels[:, ::-1]
                rows.append(crop(pixels, box, HAND_PADDING, out_wh))
            crops, centers, scales, mats = zip(*rows)
            built.append(_batch(crops, centers, scales, mats, boxes, cam_int, out_wh, device))
        return built[0], built[1]


class BodyModelDINOv3(_MetaHandCrops, SAM3DBody):
    """ComfyUI's SAM 3D Body model (the DINOv3 release) with Meta's hand crops."""


class BodyModelViTH(_MetaHandCrops, SAM3DBodyViTH):
    """This pack's ViT-H subclass of it with Meta's hand crops."""


def load_model(path: str):
    """Either release in a model patcher, in float32: the model class is chosen by what the file holds.

    Float32 whatever the file stores and whatever the device would pick. measured 2026-10-10 on the card
    (`bench/results/2026-10-10_sam3d_body_core_against_meta.md`, "On the card"): in the half precision ComfyUI's
    loader picks there, the keypoints are outside Meta's own precision floor on four boxes of six; in float32 they
    are inside it on all six, as on the CPU. Meta's code keeps the decoder in float32.
    """
    sd = loader_view(comfy.utils.load_torch_file(path, safe_load=True))
    release = "vith" if VITH_MARKER_KEY in sd else "dinov3"
    load_device = comfy.model_management.get_torch_device()
    torch_dtype = MODEL_DTYPE
    manual_cast_dtype = comfy.model_management.unet_manual_cast(torch_dtype, load_device)
    operations = comfy.ops.pick_operations(torch_dtype, manual_cast_dtype, load_device=load_device,
                                           disable_fast_fp8=True)
    model = (BodyModelViTH if release == "vith" else BodyModelDINOv3)(dtype=torch_dtype, operations=operations)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"this file is not a SAM 3D Body checkpoint in the layout bench/convert_sam3d_body_checkpoint.py "
            f"writes: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}")
    model.backbone_dtype = torch_dtype
    logger.info("[h3] MiniMaxH3BodyModelLoader: the %s release, %d tensors, backbone dtype %s", release, len(sd), torch_dtype)
    return comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=load_device,
        offload_device=comfy.model_management.unet_offload_device(),
        size=comfy.model_management.module_size(model),
        fast_disk=comfy.storage.state_dict_fast_disk(sd),
    )


# ----------------------------------------------------------------------------- prediction

def frame_box_rows(boxes, frames: int) -> tuple[list[list[list[float]]], bool]:
    """`boxes` as a list for each frame of [x1, y1, x2, y2], and whether one frame's list was used for every frame.

    `boxes` is the form `MiniMaxH3SubjectBoxes` writes: a list for each frame of dicts with `x`, `y`, `width` and
    `height`, an empty list where nobody is. One frame's list is used for every frame when only one is given, which
    is right for a still camera and a person who does not move and wrong for a one-frame mask wired by mistake, so
    the caller is told.
    """
    if isinstance(boxes, dict):
        boxes = [[boxes]]
    elif boxes and isinstance(boxes[0], dict):
        boxes = [list(boxes)]
    boxes = list(boxes or [])
    repeated = len(boxes) == 1 and frames > 1
    if repeated:
        boxes = boxes * frames
    if len(boxes) != frames:
        raise ValueError(f"boxes holds {len(boxes)} frame(s) and the frames input holds {frames}: wire the boxes "
                         "made from the same frames")
    return [[[float(b["x"]), float(b["y"]), float(b["x"]) + float(b["width"]), float(b["y"]) + float(b["height"])]
             for b in frame] for frame in boxes], repeated


def unusable(box_xyxy, height: int, width: int) -> str:
    """Why a box cannot be a person's crop, or "" when it can: not a number, no area, or wholly off the frame.

    The model crops whatever it is given and returns a body for it, so a box of one pixel, of no size or outside
    the picture would be drawn as a person (found by mrcorn's cold read, 2026-10-10). Such a box is dropped and
    reported, not raised: one bad frame should not stop a clip's pass.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return "not a number"
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return "no area"
    if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
        return "off the frame"
    return ""


def box_source(box_xyxy, height: int, width: int) -> str:
    """`whole frame` for a box that is the frame, `given` for any other: which one a preflight reads as a fact.

    A whole-frame box is one person's crop only when one person is in the frame.
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    return "whole frame" if (x1 <= 0 and y1 <= 0 and x2 >= width and y2 >= height) else "given"


def _crop_bbox(center, scale) -> list[float]:
    return [float(center[0] - scale[0] / 2), float(center[1] - scale[1] / 2),
            float(center[0] + scale[0] / 2), float(center[1] + scale[1] / 2)]


def predict(patcher, images: torch.Tensor, boxes, *, hands: bool = True, camera: str = CAMERAS[0],
            fov_degrees: float = FOV_DEFAULT, batch_size: int = BATCH_DEFAULT):
    """The pose data for a clip, and per frame per person what the table adds.

    `images` is [N, H, W, 3] in 0..1. Returns `(pose_data, extras, notes)`: `pose_data` in the layout ComfyUI's
    own predict node writes; `extras[frame][i]` for the i-th body of a frame, with which box of that frame's list
    it came from (`person`), the crop's box and each hand's box, crop side and whether its decoder was used;
    `notes` with the boxes dropped as unusable, whether one box list served every frame, and the seconds taken.
    """
    began = time.perf_counter()
    inner = patcher.model
    frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    rows, repeated = frame_box_rows(boxes, frames)
    cam_int = intrinsics(height, width, camera, fov_degrees)
    out_wh = (int(inner.image_size[1]), int(inner.image_size[0]))
    device = comfy.model_management.get_torch_device()
    refused = [{"frame": f, "person": k, "box": [float(v) for v in box], "why": why}
               for f, row in enumerate(rows) for k, box in enumerate(row) if (why := unusable(box, height, width))]
    dropped = {(r["frame"], r["person"]) for r in refused}
    todo = [(f, k) for f, row in enumerate(rows) for k in range(len(row)) if (f, k) not in dropped]
    pose_frames: list[list[dict]] = [[] for _ in range(frames)]
    extras: list[list[dict]] = [[] for _ in range(frames)]
    if todo:
        step = max(1, int(batch_size))
        comfy.model_management.load_models_gpu(
            [patcher], memory_required=MEMORY_FACTOR * inner.memory_used_forward(min(step, len(todo)), bool(hands)))
        pbar = comfy.utils.ProgressBar(len(todo))
        pixels: dict[int, torch.Tensor] = {}
        for start in range(0, len(todo), step):
            chunk = todo[start:start + step]
            for f, _ in chunk:
                if f not in pixels:
                    # rounded, not truncated: Meta reads an 8-bit file, and a frame that was resized sits between
                    # levels, where truncating reads half a level dark on average (mrcorn's cold read)
                    pixels[f] = (images[f] * 255.0).round().clamp(0.0, 255.0).to(dtype=torch.uint8, device="cpu")
            made = [crop(pixels[f].numpy(), np.asarray(rows[f][k], dtype=np.float32), BODY_PADDING, out_wh) for f, k in chunk]
            crops, centers, scales, mats = zip(*made)
            batch = _batch(crops, centers, scales, mats, [rows[f][k] for f, k in chunk], cam_int, out_wh, device)
            result = inner.run_inference([pixels[f] for f, _ in chunk], batch,
                                         inference_type="full" if hands else "body",
                                         thresh_wrist_angle=WRIST_ANGLE_THRESHOLD)
            used = hand_boxes = None
            if hands:
                pose, batch_l, batch_r, out_l, out_r = result
                final = pose["mhr"]["hand"].float()
                used = torch.stack(
                    [(final[:, :HAND_SPLIT] == out_l["mhr_hand"]["hand"][:, :HAND_SPLIT].float()).all(dim=1),
                     (final[:, HAND_SPLIT:] == out_r["mhr_hand"]["hand"][:, HAND_SPLIT:].float()).all(dim=1)],
                    dim=1).cpu().numpy()
                hand_boxes = []
                for side in (batch_l, batch_r):
                    c = side["bbox_center"].flatten(0, 1).float()
                    s = side["bbox_scale"].flatten(0, 1).float()
                    hand_boxes.append(torch.cat([c - s * 0.5, c + s * 0.5], dim=1).cpu().numpy())
            else:
                pose = result
            out = {key: value.float().cpu().numpy() for key, value in pose["mhr"].items()
                   if torch.is_tensor(value)}
            for i, (f, k) in enumerate(chunk):
                person = {
                    "bbox": np.asarray(rows[f][k], dtype=np.float32),
                    "focal_length": out["focal_length"][i],
                    "pred_keypoints_3d": out["pred_keypoints_3d"][i],
                    "pred_keypoints_2d": out["pred_keypoints_2d"][i],
                    "pred_vertices": out["pred_vertices"][i],
                    "pred_cam_t": out["pred_cam_t"][i],
                    "pred_pose_raw": out["pred_pose_raw"][i],
                    "global_rot": out["global_rot"][i],
                    "body_pose_params": out["body_pose"][i],
                    "hand_pose_params": out["hand"][i],
                    "scale_params": out["scale"][i],
                    "shape_params": out["shape"][i],
                    "expr_params": out["face"][i],
                    "mask": None,
                    "pred_joint_coords": out["pred_joint_coords"][i],
                    "pred_global_rots": out["joint_global_rots"][i],
                    "mhr_model_params": out["mhr_model_params"][i],
                }
                extra = {"person": k, "crop_bbox": _crop_bbox(centers[i], scales[i]), "left_hand": None, "right_hand": None}
                if hands:
                    person["lhand_bbox"], person["rhand_bbox"] = hand_boxes[0][i], hand_boxes[1][i]
                    for name, side in (("left_hand", 0), ("right_hand", 1)):
                        box = [float(v) for v in hand_boxes[side][i]]
                        extra[name] = {"bbox": box, "crop_side_px": float(box[2] - box[0]),
                                       "decoder_used": bool(used[i, side])}
                pose_frames[f].append(person)
                extras[f].append(extra)
            pbar.update(len(chunk))
            for f in [f for f in pixels if f < chunk[-1][0]]:
                del pixels[f]
    pose_data = {"frames": pose_frames, "faces": inner.head_pose.faces_np(), "image_size": (height, width)}
    notes = {"refused": refused, "one_box_list_for_every_frame": repeated, "crops": len(todo),
             "seconds": time.perf_counter() - began}
    return pose_data, extras, notes


def outside_parts(points: np.ndarray, height: int, width: int) -> dict[str, list[str]]:
    """{part of the body: the sides of the frame its keypoints are past}, for the parts with any keypoint outside."""
    out: dict[str, list[str]] = {}
    for part, index in PARTS.items():
        p = points[list(index)]
        sides = [side for side, past in (("left", p[:, 0] < 0), ("right", p[:, 0] >= width),
                                         ("above", p[:, 1] < 0), ("below", p[:, 1] >= height)) if bool(past.any())]
        if sides:
            out[part] = sides
    return out


def pose_table(pose_data: dict, extras, notes: dict | None = None, *, camera: str, fov_degrees: float, hands: bool,
               subject: str = "", first_source_frame: int = 0) -> dict:
    """What a preflight reads, per person per frame (`TABLE_SCHEMA`). `bench/capture_masked_run.py` is the reader.

    `first_source_frame` is the caller's word for where these frames sit in the source; nothing here can check it
    against what a loader actually read.
    """
    notes = notes or {}
    height, width = pose_data["image_size"]
    frames = []
    for f, people in enumerate(pose_data["frames"]):
        entries = []
        for k, person in enumerate(people):
            points = np.asarray(person["pred_keypoints_2d"], dtype=np.float64)[:N_KEYPOINTS_2D, :2]
            outside = (points[:, 0] < 0) | (points[:, 0] >= width) | (points[:, 1] < 0) | (points[:, 1] >= height)
            extra = extras[f][k]
            in_camera = (np.asarray(person["pred_keypoints_3d"], dtype=np.float64)[:N_KEYPOINTS_2D]
                         + np.asarray(person["pred_cam_t"], dtype=np.float64).reshape(1, 3)) if "pred_keypoints_3d" in person else None
            entries.append({
                "person": int(extra.get("person", k)), "subject": subject,
                "bbox": [float(v) for v in person["bbox"]],
                "crop_bbox": [round(v, 2) for v in extra["crop_bbox"]],
                "box_source": box_source(person["bbox"], height, width),
                "left_hand": _hand_entry(extra["left_hand"]), "right_hand": _hand_entry(extra["right_hand"]),
                "keypoints_2d": np.round(points, 1).tolist(),
                # camera space, metres under the camera this table names: the 2D points are these, projected
                "keypoints_3d": None if in_camera is None else np.round(in_camera, 3).tolist(),
                "keypoints_outside_frame": int(outside.sum()),
                "outside_parts": outside_parts(points, height, width),
                "focal_length_px": float(np.asarray(person["focal_length"]).reshape(-1)[0]),
            })
        frames.append({"frame": f, "source_frame": f + int(first_source_frame), "people": entries})
    return {
        "schema": TABLE_SCHEMA,
        "image_size": {"width": int(width), "height": int(height)},
        "hand_box_threshold_px": HAND_BOX_THRESHOLD_PX,
        "camera": {"mode": camera, "fov_degrees": float(fov_degrees) if camera == CAMERAS[1] else None},
        "hand_refinement": bool(hands),
        "first_source_frame": int(first_source_frame),
        # a box that could not be a person's crop: no body was predicted for it (`unusable`)
        "boxes_refused": [dict(r, source_frame=r["frame"] + int(first_source_frame)) for r in notes.get("refused", [])],
        "one_box_list_for_every_frame": bool(notes.get("one_box_list_for_every_frame", False)),
        "frames": frames,
    }


def _hand_entry(hand: dict | None) -> dict | None:
    if hand is None:
        return None
    return {"bbox": [round(v, 2) for v in hand["bbox"]], "crop_side_px": round(hand["crop_side_px"], 2),
            "decoder_used": hand["decoder_used"]}


def table_report(table: dict, seconds: float | None = None) -> str:
    """A few lines a person reads on the node: how many bodies, what was dropped, what the hands' decoders did,
    which parts of the body lie outside the frame, and the time."""
    people = [p for frame in table["frames"] for p in frame["people"]]
    with_body = sum(1 for frame in table["frames"] if frame["people"])
    lines = [f"{len(people)} bodies on {with_body} of {len(table['frames'])} frames"]
    if table.get("one_box_list_for_every_frame"):
        lines.append(f"one box list was used for all {len(table['frames'])} frames: wire boxes made from these frames "
                     "unless the person does not move")
    refused = table.get("boxes_refused") or []
    if refused:
        why = sorted({r["why"] for r in refused})
        frames = sorted({r["frame"] for r in refused})
        lines.append(f"{len(refused)} box(es) dropped, no body predicted ({', '.join(why)}), on frame(s) "
                     + ", ".join(str(f) for f in frames[:16]) + (f" and {len(frames) - 16} more" if len(frames) > 16 else ""))
    whole = sum(1 for p in people if p["box_source"] == "whole frame")
    if whole:
        lines.append(f"{whole} of the bodies are from a box that is the whole frame, which is one person's crop only "
                     "when one person is in it")
    if table["hand_refinement"] and people:
        for name in ("left_hand", "right_hand"):
            used = sum(1 for p in people if p[name]["decoder_used"])
            small = sum(1 for p in people if p[name]["crop_side_px"] <= table["hand_box_threshold_px"])
            lines.append(f"{name.replace('_', ' ')}: refined on {used} of {len(people)}"
                         + (f"; too small to refine (crop at or under {table['hand_box_threshold_px']} px) on {small}" if small else "")
                         + ("" if used == len(people) else "; where it was not, the fingers are the body decoder's"))
    parts: dict[str, list] = {}
    for p in people:
        for part, sides in (p.get("outside_parts") or {}).items():
            entry = parts.setdefault(part, [0, set()])
            entry[0] += 1
            entry[1].update(sides)
    if parts:
        said = "; ".join(f"{part} on {count} of {len(people)} ({', '.join(sorted(sides))})" for part, (count, sides) in parts.items())
        lines.append(f"keypoints outside the frame: {said}. A body the frame cuts off is completed past its edge; a "
                     "part outside on every body is that, a part outside on a few is worth a look")
    if seconds is not None:
        lines.append(f"{seconds:.0f} s")
    return "\n".join(lines)


def write_table(table: dict, prefix: str) -> str:
    """`<output folder>/<prefix>_pose_table.json`, replacing one of the same name. Returns the path written."""
    path = os.path.join(folder_paths.get_output_directory(), f"{prefix}_pose_table.json")
    root = os.path.realpath(folder_paths.get_output_directory())
    if os.path.commonpath([root, os.path.realpath(path)]) != root:
        raise ValueError(f"table_prefix must stay inside the output folder; got {prefix!r}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(table, handle, indent=1)
        handle.write("\n")
    return path


# ----------------------------------------------------------------------------- drawing

def render_size(pose_data: dict, size: str, width: int, height: int) -> tuple[int, int]:
    native_h, native_w = pose_data["image_size"]
    if size == SIZES[0]:
        return int(native_w), int(native_h)
    if size == SIZES[1]:
        return int(width), int(height)
    raise ValueError(f"size must be one of {SIZES}; got {size!r}")


def scaled_for(pose_data: dict, width: int, height: int) -> dict:
    """The pose data as the rasteriser needs it at another size: each focal length scaled, the image centred.

    The rasteriser takes the principal point as the image's centre, so one scale on the focal length fits the
    source inside the new size with its centre on the new centre: bars, not a stretch, when the shapes differ.
    """
    native_h, native_w = pose_data["image_size"]
    if (int(height), int(width)) == (int(native_h), int(native_w)):
        return pose_data
    ratio = min(width / native_w, height / native_h)
    frames = [[dict(p, focal_length=np.asarray(p["focal_length"], dtype=np.float32) * ratio) for p in people]
              for people in pose_data["frames"]]
    return dict(pose_data, frames=frames, image_size=(int(height), int(width)))


def render(pose_data: dict, *, style: str = STYLES[0], size: str = SIZES[0], width: int = 1024, height: int = 768) -> torch.Tensor:
    """The mesh of every frame on black, [N, H, W, 3]; a frame with nobody is black."""
    from comfy_extras.sam3d_body.rasterizer import render_pose_data_torch  # the rasteriser, called as a function

    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}; got {style!r}")
    w, h = render_size(pose_data, size, width, height)
    scaled = scaled_for(pose_data, w, h)
    out_device = comfy.model_management.intermediate_device()
    out_dtype = comfy.model_management.intermediate_dtype()
    count = len(scaled["frames"])
    if count == 0:     # `predict` cannot make a pose with no frames; an IMAGE with none would break what reads it
        return torch.zeros(1, h, w, 3, dtype=out_dtype, device=out_device)
    cache: dict = {}
    pbar = comfy.utils.ProgressBar(count)
    frames = []
    for f in range(count):
        image = render_pose_data_torch(scaled, frame_idx=f, W=w, H=h, background=None,
                                       composite="silhouette" if style == "silhouette" else "mesh_only",
                                       person_brightness_falloff=1.0, cache=cache)
        frames.append(image.to(device=out_device, dtype=out_dtype))
        pbar.update(1)
    return torch.stack(frames, dim=0)


# ----------------------------------------------------------------------------- nodes

class MiniMaxH3BodyModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3BodyModelLoader",
            display_name="MiniMax H3 Body Model Loader (SAM 3D Body)",
            category="model/latent/minimax",
            description=("Loads SAM 3D Body from models/detection/ for MiniMax H3 Body Pose: either release, told "
                         "apart by what the file holds. It runs in full precision on purpose: in half precision "
                         "the body was measured outside the band Meta's own code keeps to. The hands are cropped "
                         "the way Meta's own code crops them."),
            inputs=[
                io.Combo.Input("model_file", options=folder_paths.get_filename_list("detection"),
                               tooltip=("A SAM 3D Body file under models/detection/, as "
                                        "bench/convert_sam3d_body_checkpoint.py or its ViT-H twin writes it.")),
            ],
            outputs=[SAM3DBodyModel.Output(display_name="body_model")],
        )

    @classmethod
    def execute(cls, model_file) -> io.NodeOutput:
        return io.NodeOutput(load_model(folder_paths.get_full_path_or_raise("detection", model_file)))


class MiniMaxH3BodyPose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3BodyPose",
            display_name="MiniMax H3 Body Pose (a body mesh from frames and boxes)",
            category="model/latent/minimax",
            description=("Predicts a posed body for each box on each frame, with the crop sampled as Meta's SAM 3D "
                         "Body code samples it, and writes a table of what a preflight reads: each person's box, "
                         "each hand's crop and whether it was refined, and the keypoints. The mesh has no mouth "
                         "and no expression: the model sets them to zero."),
            inputs=[
                SAM3DBodyModel.Input("body_model", tooltip="From MiniMax H3 Body Model Loader."),
                io.Image.Input("frames", tooltip="The frames to read bodies from."),
                io.BoundingBox.Input(
                    "boxes", force_input=True,
                    tooltip=("For each frame, the boxes of the people to predict: MiniMax H3 Subject Boxes' output, "
                             "one of those nodes for each person. A frame with an empty list gets no body. A box "
                             "that is the whole frame is one person's crop only when one person is in it, and "
                             "the table says which rows came from one.")),
                io.Boolean.Input("hand_refinement", default=True,
                                 tooltip=("Runs the hand decoder on a crop of each hand and uses its result where "
                                          "the model's own tests accept it. The table says, per hand, whether they "
                                          "did. Costs two more crops a person.")),
                io.Combo.Input("camera", options=list(CAMERAS), default=CAMERAS[0],
                               tooltip=("How the focal length is set. `image diagonal` assumes a focal length "
                                        "equal to the frame's diagonal. `field of view` uses the angle below.")),
                io.Float.Input("fov_degrees", default=FOV_DEFAULT, min=FOV_MIN, max=FOV_MAX, step=0.1,
                               tooltip=("The vertical field of view, in degrees, used when camera is `field of "
                                        "view`. It moves the body's depth and size, not where it sits in the frame.")),
                io.String.Input("subject", default="",
                                tooltip="A label for the person these boxes follow, written on every row of the table."),
                io.Int.Input("first_source_frame", default=0, min=0, max=10_000_000,
                             tooltip=("The source's frame number of the first frame given here, so the table's rows "
                                      "carry the source's own frame numbers. It is a label you supply: nothing "
                                      "checks it against where the loader actually started.")),
                io.String.Input("table_prefix", default="",
                                tooltip=("Where the table is also written as a file: this prefix under the output "
                                         "folder, with `_pose_table.json` added. Empty writes no file; the table "
                                         "is on the second output either way.")),
                io.Int.Input("batch_size", default=BATCH_DEFAULT, min=1, max=BATCH_MAX, advanced=True,
                             tooltip="How many crops go through the model at once. Lower it if memory runs out."),
            ],
            outputs=[
                MHRPoseData.Output(display_name="pose_data"),
                io.String.Output(display_name="pose_table",
                                 tooltip="The table as JSON, one entry for each person on each frame."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, body_model, frames, boxes, hand_refinement=True, camera=CAMERAS[0], fov_degrees=FOV_DEFAULT,
                subject="", first_source_frame=0, table_prefix="", batch_size=BATCH_DEFAULT) -> io.NodeOutput:
        pose_data, extras, notes = predict(body_model, frames, boxes, hands=bool(hand_refinement), camera=camera,
                                           fov_degrees=fov_degrees, batch_size=batch_size)
        table = pose_table(pose_data, extras, notes, camera=camera, fov_degrees=fov_degrees,
                           hands=bool(hand_refinement), subject=str(subject).strip(),
                           first_source_frame=int(first_source_frame))
        report = table_report(table, notes["seconds"])
        prefix = str(table_prefix).strip()
        if prefix:
            report += f"\ntable written to {os.path.relpath(write_table(table, prefix), folder_paths.get_output_directory())}"
        logger.info("[h3] MiniMaxH3BodyPose: %d crop(s), %s", notes["crops"], report.replace("\n", "; "))
        return io.NodeOutput(pose_data, json.dumps(table), report)


class MiniMaxH3BodyMeshVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3BodyMeshVideo",
            display_name="MiniMax H3 Body Mesh Video (the pose drawn as frames)",
            category="model/latent/minimax",
            description=("Draws MiniMax H3 Body Pose's bodies as frames on black, for a motion video: a movement "
                         "signal with none of the original person's look. A frame with nobody on it is black."),
            inputs=[
                MHRPoseData.Input("pose_data", tooltip="From MiniMax H3 Body Pose."),
                io.Combo.Input("style", options=list(STYLES), default=STYLES[0],
                               tooltip="`mesh` is a lit grey body; `silhouette` is its outline filled white."),
                io.Combo.Input("size", options=list(SIZES), default=SIZES[0],
                               tooltip=("`the source's` draws at the size of the frames the pose was read from. "
                                        "`width and height` draws at the two numbers below, the body fitted "
                                        "inside with bars where the shapes differ. A loader that crops to a "
                                        "canvas fits the other way, so give this node frames already at the "
                                        "canvas or the mesh will sit slightly small against the picture.")),
                io.Int.Input("width", default=1024, min=64, max=8192, step=2,
                             tooltip="The frames' width when size is `width and height`."),
                io.Int.Input("height", default=768, min=64, max=8192, step=2,
                             tooltip="The frames' height when size is `width and height`."),
            ],
            outputs=[io.Image.Output(display_name="frames")],
        )

    @classmethod
    def execute(cls, pose_data, style=STYLES[0], size=SIZES[0], width=1024, height=768) -> io.NodeOutput:
        return io.NodeOutput(render(pose_data, style=style, size=size, width=int(width), height=int(height)))
