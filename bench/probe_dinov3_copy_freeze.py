#!/usr/bin/env python3
"""M1 (`docs/wiki/next_steps.md`): CPU-safe DINOv3 `<Picture 1>` copy & temporal pose-freeze triage probe.

Runs offline on already-rendered videos (`0` ComfyUI renders) using the local
`dino_v3_L_naf_fp32.safetensors` weights from `$H3_MODEL_STORE` via ComfyUI's
native `comfy.clip_vision.load(..., crop=False)` (no downloads, no external
symlinks in `models/clip_vision/`, never used for `who_is_who` cast identity,
and emitting advisory triage notes rather than pass/fail verdicts).

Usage:
    <comfy-venv-python> bench/probe_dinov3_copy_freeze.py --video CLIP.mp4 [--ref PIC1.png]
    <comfy-venv-python> bench/probe_dinov3_copy_freeze.py --dir DIR [--ref PIC1.png] [--json OUT.json]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

BENCH_DIR = Path(__file__).resolve().parent
REPO = BENCH_DIR.parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from _lib import bootstrap  # noqa: E402

DEFAULT_REL_WEIGHTS = Path("Comfy-Org_Pixal3D") / "clip_vision" / "dino_v3_L_naf_fp32.safetensors"


def resolve_dinov3_weights_path(explicit: str | Path | None = None) -> Path:
    """Resolve `dino_v3_L_naf_fp32.safetensors` under `$H3_MODEL_STORE` without modifying `models/`."""
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found at --weights {p}")
        return p
    store = (os.environ.get("H3_MODEL_STORE") or "").strip()
    if not store:
        raise RuntimeError(
            "H3_MODEL_STORE is not set (source repo h3.env or pass --weights); refusing to download"
        )
    p = Path(store).expanduser() / DEFAULT_REL_WEIGHTS
    if not p.is_file():
        raise FileNotFoundError(
            f"DINOv3 weights not found under $H3_MODEL_STORE ({DEFAULT_REL_WEIGHTS})"
        )
    return p


def load_dinov3_clip_vision(
    weights_path: Path | None = None,
    *,
    device: str = "cpu",
):
    """Bootstrap ComfyUI (keeping CPU-only when `device == 'cpu'`) and load DINOv3 ClipVision."""
    use_cpu = device == "cpu"
    bootstrap(cpu=use_cpu)
    import comfy.clip_vision as cv  # noqa: E402

    ckpt = resolve_dinov3_weights_path(weights_path)
    model = cv.load(str(ckpt))
    if model is None:
        raise RuntimeError(f"comfy.clip_vision.load failed to recognize config for {ckpt.name}")
    return model


def _load_image_tensor(path: Path) -> torch.Tensor:
    """Load an image file into `[1, H, W, 3]` float32 `[0, 1]` ComfyUI tensor."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def sample_video_frames(
    video_path: Path,
    *,
    stride: int = 12,
    max_frames: int = 24,
) -> tuple[list[int], torch.Tensor]:
    """Decode sampled RGB frames (`[N, H, W, 3]` float32 `[0, 1]`) keeping first and last frame."""
    import cv2  # noqa: E402

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        indices = list(range(0, total, max(1, int(stride))))
        if (total - 1) not in indices:
            indices.append(total - 1)
        if max_frames > 0 and len(indices) > max_frames:
            lin = np.linspace(0, len(indices) - 1, max_frames, dtype=int)
            indices = [indices[int(i)] for i in lin]
    else:
        indices = []

    picked_idx: list[int] = []
    frames: list[np.ndarray] = []
    if indices:
        target_set = set(indices)
        idx = 0
        max_need = max(indices)
        while idx <= max_need:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in target_set:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
                picked_idx.append(idx)
            idx += 1
    else:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % max(1, int(stride)) == 0:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
                picked_idx.append(idx)
                if max_frames > 0 and len(frames) >= max_frames:
                    break
            idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"decoded 0 frames from {video_path}")
    return picked_idx, torch.from_numpy(np.stack(frames, axis=0))


def encode_image_embeddings(
    clip_vision: Any,
    images: torch.Tensor,
    *,
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode `[N, H, W, 3]` tensor via direct DINOv3 `clip_vision.model(pixel_values=...)` call.

    Note: `comfy/clip_vision.py::ClipVisionModel.encode_image` unconditionally calls
    `out[1].to(...)`, which crashes on `DINOv3ViTModel` returning `(seq, None, pooled, None)`.
    Following `comfy_extras/nodes_triposplat.py:116-123` avoids patching ComfyUI core.

    Returns `(cls_vecs [N, D], patch_vecs [N, P, D] | None)` already L2-normalized.
    """
    import comfy.clip_model  # noqa: E402
    import comfy.model_management  # noqa: E402

    comfy.model_management.load_model_gpu(clip_vision.patcher)
    n_prefix = int(getattr(getattr(clip_vision, "model", None), "num_prefix_tokens", 5))
    cls_chunks: list[torch.Tensor] = []
    patch_chunks: list[torch.Tensor] = []
    has_patches = True
    with torch.no_grad():
        for start in range(0, images.shape[0], max(1, batch_size)):
            chunk = images[start : start + batch_size]
            pixel_values = comfy.clip_model.clip_preprocess(
                chunk.to(clip_vision.load_device),
                size=clip_vision.image_size,
                mean=clip_vision.image_mean,
                std=clip_vision.image_std,
                crop=False,
            ).float()
            out = clip_vision.model(pixel_values=pixel_values)
            lhs = out[0].float().cpu()
            embeds = out[2]
            vec = embeds.float().cpu() if embeds is not None else lhs[:, 0, :]
            cls_chunks.append(F.normalize(vec, dim=-1))
            if lhs.ndim == 3 and lhs.shape[1] > n_prefix:
                patches = lhs[:, n_prefix:, :]
                patch_chunks.append(F.normalize(patches, dim=-1))
            else:
                has_patches = False
    cls_all = torch.cat(cls_chunks, dim=0)
    patch_all = torch.cat(patch_chunks, dim=0) if (has_patches and patch_chunks) else None
    return cls_all, patch_all


def score_clip_embeddings(
    frame_vecs: torch.Tensor,
    *,
    frame_patches: torch.Tensor | None = None,
    ref_vec: torch.Tensor | None = None,
    ref_patches: torch.Tensor | None = None,
    copy_warn_threshold: float = 0.94,
    patch_copy_warn_threshold: float = 0.88,
    freeze_warn_threshold: float = 0.985,
) -> dict[str, Any]:
    """Compute pure cosine copy & temporal pose-freeze diagnostics + advisory triage notes."""
    fv = F.normalize(frame_vecs.float(), dim=-1)
    n_frames = int(fv.shape[0])
    out: dict[str, Any] = {
        "sampled_frames": n_frames,
        "ref_cos_f0": None,
        "ref_cos_mean": None,
        "ref_cos_max": None,
        "ref_patch_cos_mean": None,
        "ref_patch_cos_max": None,
        "temporal_self_cos_mean": None,
        "temporal_self_cos_min": None,
        "temporal_self_cos_p90": None,
        "triage_notes": [],
    }
    notes: list[str] = []

    if n_frames >= 2:
        step_cos = (fv[1:] * fv[:-1]).sum(dim=-1)
        out["temporal_self_cos_mean"] = round(float(step_cos.mean().item()), 5)
        out["temporal_self_cos_min"] = round(float(step_cos.min().item()), 5)
        out["temporal_self_cos_p90"] = round(float(torch.quantile(step_cos, 0.90).item()), 5)
        if out["temporal_self_cos_mean"] >= freeze_warn_threshold:
            notes.append(
                f"high temporal self-similarity (mean={out['temporal_self_cos_mean']:.4f}) — inspect for pose freeze"
            )

    if ref_vec is not None:
        rv = F.normalize(ref_vec.float().reshape(1, -1), dim=-1)
        rcos = (fv @ rv.T).squeeze(-1)
        out["ref_cos_f0"] = round(float(rcos[0].item()), 5)
        out["ref_cos_mean"] = round(float(rcos.mean().item()), 5)
        out["ref_cos_max"] = round(float(rcos.max().item()), 5)
        if out["ref_cos_mean"] >= copy_warn_threshold:
            notes.append(
                f"high <Picture 1> CLS similarity (mean={out['ref_cos_mean']:.4f}) — inspect for layout/pose copy"
            )

    if frame_patches is not None and ref_patches is not None:
        fp = F.normalize(frame_patches.float(), dim=-1)
        rp = F.normalize(ref_patches.float(), dim=-1)
        if rp.ndim == 2:
            rp = rp.unsqueeze(0)
        if fp.shape[-2:] == rp.shape[-2:]:
            pcos = (fp * rp).sum(dim=-1).mean(dim=-1)  # aligned spatial patch cosine per frame
            out["ref_patch_cos_mean"] = round(float(pcos.mean().item()), 5)
            out["ref_patch_cos_max"] = round(float(pcos.max().item()), 5)
            if out["ref_patch_cos_mean"] >= patch_copy_warn_threshold:
                notes.append(
                    f"high spatial patch lock vs <Picture 1> (mean={out['ref_patch_cos_mean']:.4f}) — inspect for static trace"
                )

    out["triage_notes"] = notes
    return out


def probe_videos(
    videos: Sequence[Path],
    *,
    ref_path: Path | None = None,
    weights_path: Path | None = None,
    device: str = "cpu",
    stride: int = 12,
    max_frames: int = 24,
) -> list[dict[str, Any]]:
    clip_vision = load_dinov3_clip_vision(weights_path, device=device)
    ref_vec: torch.Tensor | None = None
    ref_patches: torch.Tensor | None = None
    if ref_path is not None:
        r_cls, r_pat = encode_image_embeddings(clip_vision, _load_image_tensor(ref_path))
        ref_vec = r_cls[0]
        ref_patches = r_pat[0] if r_pat is not None else None

    rows: list[dict[str, Any]] = []
    for vp in videos:
        indices, frames = sample_video_frames(vp, stride=stride, max_frames=max_frames)
        f_cls, f_pat = encode_image_embeddings(clip_vision, frames)
        metrics = score_clip_embeddings(
            f_cls,
            frame_patches=f_pat,
            ref_vec=ref_vec,
            ref_patches=ref_patches,
        )
        rows.append(
            {
                "video": vp.name,
                "frame_indices": indices,
                **metrics,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", nargs="+", type=Path, help="one or more .mp4 files to probe (sharing one DINOv3 load)")
    ap.add_argument("--dir", type=Path, help="directory of .mp4 renders to batch-probe (skips *_hud / *.1fps.* helpers)")
    ap.add_argument("--ref", type=Path, help="optional <Picture 1> reference image for copy similarity")
    ap.add_argument("--weights", type=Path, help="optional explicit dino_v3_L_naf_fp32.safetensors path")
    ap.add_argument("--device", choices=("cpu", "auto"), default="cpu", help="inference device (default cpu)")
    ap.add_argument("--stride", type=int, default=12, help="video frame sampling stride (default 12)")
    ap.add_argument("--max-frames", type=int, default=24, help="max sampled frames per video (default 24)")
    ap.add_argument("--json", type=Path, help="optional output JSON summary path")
    args = ap.parse_args(argv)

    targets: list[Path] = []
    if args.video:
        targets.extend(args.video)
    if args.dir:
        for p in sorted(args.dir.glob("*.mp4")):
            if ".1fps." in p.name or p.name.endswith("-audio.mp4"):
                continue
            targets.append(p)
        if not targets:
            for p in sorted(args.dir.glob("*.mp4")):
                if ".1fps." not in p.name:
                    targets.append(p)
    if not targets:
        raise SystemExit("pass --video CLIP.mp4 or --dir FOLDER")

    rows = probe_videos(
        targets,
        ref_path=args.ref,
        weights_path=args.weights,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
    )

    print("| Clip | Frames | Ref CLS (f0 / mean / max) | Ref Patch Mean | Temporal Self-Cos (mean / min) | Triage Notes |")
    print("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for r in rows:
        if r["ref_cos_mean"] is not None:
            ref_col = f"{r['ref_cos_f0']:.3f} / {r['ref_cos_mean']:.3f} / {r['ref_cos_max']:.3f}"
        else:
            ref_col = "-"
        pat_col = f"{r['ref_patch_cos_mean']:.3f}" if r["ref_patch_cos_mean"] is not None else "-"
        if r["temporal_self_cos_mean"] is not None:
            tmp_col = f"{r['temporal_self_cos_mean']:.4f} / {r['temporal_self_cos_min']:.4f}"
        else:
            tmp_col = "-"
        notes = "; ".join(r["triage_notes"]) or "none"
        print(f"| `{r['video']}` | {r['sampled_frames']} | {ref_col} | {pat_col} | {tmp_col} | {notes} |")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"ref": args.ref.name if args.ref else None, "results": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
