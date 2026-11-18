"""Media decoding, mask rasterization, cutout cropping, VLM reticles, and A|B metrics."""

from __future__ import annotations

import base64
import io
import tempfile
from collections import deque
from pathlib import Path
from typing import Any

from director import compiler
from director.store import sanitize_filename


def decode_video_frame_rgb(raw_bytes: bytes, filename: str, frame_index: int = 0) -> tuple[Any, int, int]:
    """Decode one video frame as ``(rgb_ndarray, resolved_index, total_frames)``.

    Negative ``frame_index`` indexes from the end (``-1`` = last frame);
    out-of-range indices clamp to ``[0, total_frames - 1]``.
    """
    ext = Path(filename).suffix.lower() or ".mp4"
    target_idx = int(frame_index)
    with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(tmp.name)
            try:
                reported_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if reported_total > 0:
                    seek_idx = max(
                        0,
                        min(reported_total - 1, (reported_total + target_idx) if target_idx < 0 else target_idx),
                    )
                    if seek_idx > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, float(seek_idx))
                    ok, frame_bgr = cap.read()
                    if ok and frame_bgr is not None:
                        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), seek_idx, reported_total
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0.0)

                count = 0
                tail_buf: deque[Any] | None = deque(maxlen=max(1, -target_idx)) if target_idx < 0 else None
                last_bgr = None
                picked_bgr = None
                while True:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                    last_bgr = frame_bgr
                    if tail_buf is not None:
                        tail_buf.append(frame_bgr)
                    elif count == target_idx:
                        picked_bgr = frame_bgr
                        count += 1
                        break
                    count += 1
                if target_idx < 0 and tail_buf:
                    resolved = max(0, count + target_idx)
                    idx_in_buf = max(0, len(tail_buf) - (count - resolved))
                    chosen = tail_buf[idx_in_buf]
                    return cv2.cvtColor(chosen, cv2.COLOR_BGR2RGB), resolved, count
                if picked_bgr is not None:
                    return cv2.cvtColor(picked_bgr, cv2.COLOR_BGR2RGB), target_idx, max(reported_total, count)
                if last_bgr is not None and count > 0:
                    return cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB), count - 1, count
            finally:
                cap.release()
        except Exception:
            pass

        try:
            import av  # type: ignore

            tail_av: deque[Any] | None = deque(maxlen=max(1, -target_idx)) if target_idx < 0 else None
            last_rgb = None
            picked_rgb = None
            count = 0
            with av.open(tmp.name) as container:
                for fr in container.decode(video=0):
                    rgb = fr.to_ndarray(format="rgb24")
                    last_rgb = rgb
                    if tail_av is not None:
                        tail_av.append(rgb)
                    elif count == target_idx:
                        picked_rgb = rgb
                        count += 1
                        break
                    count += 1
            if target_idx < 0 and tail_av:
                resolved = max(0, count + target_idx)
                idx_in_buf = max(0, len(tail_av) - (count - resolved))
                return tail_av[idx_in_buf], resolved, count
            if picked_rgb is not None:
                return picked_rgb, target_idx, count
            if last_rgb is not None and count > 0:
                return last_rgb, count - 1, count
        except Exception as exc:
            raise ValueError(f"Unable to decode frame {target_idx} from {filename!r}: {exc}") from exc

    raise ValueError(f"Video {filename!r} contains no readable video frames")


def video_bytes_has_audio(raw_bytes: bytes, filename: str = "clip.mp4") -> bool:
    """Return False when a video container provably has zero audio streams (True on probe error)."""
    ext = Path(filename).suffix.lower() or ".mp4"
    try:
        import av  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(raw_bytes)
            tmp.flush()
            with av.open(tmp.name) as container:
                return bool(container.streams.audio)
    except Exception:
        return True


def extract_video_frame_png(raw_bytes: bytes, filename: str, frame_index: int = -1) -> tuple[bytes, int, int, int, int]:
    """Extract frame ``frame_index`` as lossless PNG ``(png_bytes, resolved_idx, total_frames, w, h)``."""
    from PIL import Image

    ext = Path(filename).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        with Image.open(io.BytesIO(raw_bytes)) as img:
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="PNG")
            return buf.getvalue(), 0, 1, int(rgb.width), int(rgb.height)

    frame_rgb, resolved_idx, total_frames = decode_video_frame_rgb(raw_bytes, filename, frame_index=frame_index)
    img = Image.fromarray(frame_rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), resolved_idx, total_frames, int(img.width), int(img.height)


def decode_frame0_or_image(raw_bytes: bytes, filename: str) -> tuple[bytes, int, int, str, str]:
    """Return ``(png_bytes, width, height, target_filename, source_kind)``."""
    from PIL import Image

    ext = Path(filename).suffix.lower()
    if ext in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        png_bytes, _idx, _total, w, h = extract_video_frame_png(raw_bytes, filename, frame_index=0)
        out_name = sanitize_filename(f"{Path(filename).stem}_frame0.png", default_ext=".png")
        return png_bytes, w, h, out_name, "video_frame0"

    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img.load()
            w, h = int(img.width), int(img.height)
            out_name = sanitize_filename(filename, default_ext=".png")
            if Path(out_name).suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                out_name = f"{Path(out_name).stem}.png"
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                return buf.getvalue(), w, h, out_name, "image"
            return raw_bytes, w, h, out_name, "image"
    except Exception as exc:
        raise ValueError(f"Invalid image upload {filename!r}: {exc}") from exc


def fallback_point_ellipse_mask(
    width: int,
    height: int,
    *,
    positive_points: list[dict[str, int]],
) -> Any:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    pts = positive_points or [{"x": width // 2, "y": height // 2}]
    rx = max(8.0, float(width) * 0.16)
    ry = max(8.0, float(height) * 0.22)
    mask = np.zeros((height, width), dtype=np.float32)
    for pt in pts:
        cx = float(pt["x"])
        cy = float(pt["y"])
        dist = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        soft = np.clip((1.12 - dist) / 0.24, 0.0, 1.0)
        mask = np.maximum(mask, soft)
    return mask


def segment_grabcut_mask(
    rgb_u8: Any,
    *,
    positive_points: list[dict[str, int]],
    negative_points: list[dict[str, int]],
) -> tuple[Any, float]:
    import numpy as np

    h, w = int(rgb_u8.shape[0]), int(rgb_u8.shape[1])
    pts_pos = positive_points or [{"x": w // 2, "y": h // 2}]
    short_side = max(8, min(w, h))

    mask_f32: Any | None = None
    try:
        import cv2  # type: ignore

        gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(1, min(4, short_side // 32))
        gc_mask[:border, :] = cv2.GC_BGD
        gc_mask[-border:, :] = cv2.GC_BGD
        gc_mask[:, :border] = cv2.GC_BGD
        gc_mask[:, -border:] = cv2.GC_BGD

        r_pr = max(12, int(round(short_side * 0.24)))
        r_fg = max(4, int(round(short_side * 0.035)))
        r_neg = max(4, int(round(short_side * 0.04)))

        for pt in pts_pos:
            cv2.circle(gc_mask, (int(pt["x"]), int(pt["y"])), r_pr, int(cv2.GC_PR_FGD), -1)
        for pt in pts_pos:
            cv2.circle(gc_mask, (int(pt["x"]), int(pt["y"])), r_fg, int(cv2.GC_FGD), -1)
        for pt in negative_points:
            cv2.circle(gc_mask, (int(pt["x"]), int(pt["y"])), r_neg, int(cv2.GC_BGD), -1)

        has_fg = np.any((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD))
        has_bg = np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD))
        if not has_bg:
            gc_mask[0, :] = cv2.GC_BGD
        if has_fg and np.any((gc_mask == cv2.GC_BGD) | (gc_mask == cv2.GC_PR_BGD)):
            bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            cv2.grabCut(bgr_u8, gc_mask, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)
            fg_bin = ((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)).astype(np.float32)
            cov_raw = float(fg_bin.mean())
            if 0.005 <= cov_raw <= 0.95:
                mask_f32 = np.clip(cv2.GaussianBlur(fg_bin, (0, 0), sigmaX=1.2), 0.0, 1.0)
    except Exception:
        mask_f32 = None

    if mask_f32 is None:
        mask_f32 = fallback_point_ellipse_mask(w, h, positive_points=pts_pos)
    coverage = float((mask_f32 > 0.5).mean())
    return mask_f32, coverage


def encode_mask_rgb_png(mask_f32: Any) -> tuple[bytes, Any]:
    import numpy as np
    from PIL import Image

    mask_u8 = np.clip(np.round(mask_f32 * 255.0), 0, 255).astype(np.uint8)
    mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(mask_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue(), mask_u8


def encode_rgba_cutout_png(
    rgb_u8: Any,
    mask_u8: Any,
    *,
    pad_px: int = 4,
) -> tuple[bytes, bytes, int, int, dict[str, float]]:
    """Crop RGBA cutout and aligned binary mask PNG tightly to the denoised foreground bounding box."""
    import numpy as np
    from PIL import Image

    h, w = int(mask_u8.shape[0]), int(mask_u8.shape[1])
    fg_u8 = (mask_u8 > 12).astype(np.uint8)
    clean_fg = fg_u8
    try:
        import cv2  # type: ignore

        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(fg_u8, connectivity=8)
        if n_lbl > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_area = int(areas.max()) if len(areas) else 0
            min_area = max(16, int(round(max_area * 0.005)))
            kept = np.zeros_like(fg_u8)
            for idx in range(1, n_lbl):
                if int(stats[idx, cv2.CC_STAT_AREA]) >= min_area:
                    kept[labels == idx] = 1
            if np.any(kept):
                clean_fg = kept
    except Exception:
        pass

    clean_mask = np.where(clean_fg > 0, mask_u8, 0).astype(np.uint8)
    ys, xs = np.where(clean_fg > 0)
    if len(ys) > 0 and len(xs) > 0:
        x0 = max(0, int(xs.min()) - pad_px)
        y0 = max(0, int(ys.min()) - pad_px)
        x1 = min(w, int(xs.max()) + 1 + pad_px)
        y1 = min(h, int(ys.max()) + 1 + pad_px)
    else:
        x0, y0, x1, y1 = 0, 0, w, h

    rgba_crop = np.dstack([rgb_u8[y0:y1, x0:x1], clean_mask[y0:y1, x0:x1]])
    mask_crop_u8 = clean_mask[y0:y1, x0:x1]
    mask_crop_rgb = np.stack([mask_crop_u8, mask_crop_u8, mask_crop_u8], axis=-1)

    cut_buf = io.BytesIO()
    Image.fromarray(rgba_crop, mode="RGBA").save(cut_buf, format="PNG")
    mask_buf = io.BytesIO()
    Image.fromarray(mask_crop_rgb, mode="RGB").save(mask_buf, format="PNG")

    cw = max(1, int(x1 - x0))
    ch = max(1, int(y1 - y0))
    bbox_norm = {
        "x0": round(x0 / max(1, w), 4),
        "y0": round(y0 / max(1, h), 4),
        "x1": round(x1 / max(1, w), 4),
        "y1": round(y1 / max(1, h), 4),
        "cx": round(((x0 + x1) * 0.5) / max(1, w), 4),
        "cy": round(((y0 + y1) * 0.5) / max(1, h), 4),
        "w": round(cw / max(1, w), 4),
        "h": round(ch / max(1, h), 4),
    }
    return cut_buf.getvalue(), mask_buf.getvalue(), cw, ch, bbox_norm


def decode_media_frames(raw_bytes: bytes, filename: str) -> Any:
    import numpy as np

    ext = Path(filename).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp"):
        from PIL import Image

        with Image.open(io.BytesIO(raw_bytes)) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return arr[None, ...]

    with tempfile.NamedTemporaryFile(suffix=ext or ".mp4") as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        frames: list[Any] = []
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(tmp.name)
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            cap.release()
        except Exception:
            frames = []
        if not frames:
            import av  # type: ignore

            with av.open(tmp.name) as container:
                for fr in container.decode(video=0):
                    frames.append(fr.to_ndarray(format="rgb24").astype(np.float32) / 255.0)
        if not frames:
            raise ValueError(f"No frames decoded from {filename!r}")
        return np.stack(frames, axis=0)


def compute_ab_metrics_sync(
    b1: bytes,
    edited_filename: str,
    b2: bytes,
    base_filename: str,
    events_str: str,
    director_mode: str,
    feather: float,
) -> dict[str, Any]:
    import numpy as np

    spatial_director = compiler.import_pack_module("spatial_director")

    f_edit = decode_media_frames(b1, edited_filename)
    f_base = decode_media_frames(b2, base_filename)
    t = min(f_edit.shape[0], f_base.shape[0])
    h = min(f_edit.shape[1], f_base.shape[1])
    w = min(f_edit.shape[2], f_base.shape[2])
    if t <= 0 or h <= 0 or w <= 0:
        return {"ok": False, "available": False, "error": "empty video dimensions"}
    diff = np.abs(f_edit[:t, :h, :w] - f_base[:t, :h, :w]).mean(axis=3)  # [T, H, W]

    if events_str.strip():
        rules_mod = compiler.import_pack_module("h3_rules")
        mask_len = rules_mod.snap_length(max(5, t))
        node_out = spatial_director.MiniMaxH3SpatialDirector.execute(
            mode=director_mode,
            width=w,
            height=h,
            length=mask_len,
            feather=feather,
            invert_mask=False,
            events_json=events_str,
        )
        mask = node_out[0].detach().cpu().numpy()[:t, :h, :w]
    else:
        mask = np.ones((t, h, w), dtype=np.float32)

    in_w = mask
    out_w = np.clip(1.0 - mask, 0.0, 1.0)
    in_mae = float((diff * in_w).sum() / max(1e-6, float(in_w.sum())))
    out_mae = float((diff * out_w).sum() / max(1e-6, float(out_w.sum())))
    ratio = float(in_mae / max(1e-6, out_mae))
    return {
        "ok": True,
        "available": True,
        "in_mask_mae": round(in_mae, 6),
        "out_mask_mae": round(out_mae, 6),
        "focus_ratio": round(ratio, 3),
        "frames_compared": int(t),
    }


def prepare_vlm_reticle_images(
    raw_img: bytes,
    *,
    has_reticle: bool,
    x_n: float,
    y_n: float,
    rx_n: float,
    ry_n: float,
) -> list[dict[str, Any]]:
    """Build full-scene (with blue/white reticle) + zoomed 512px ROI crop blocks for local VLM grounding."""
    from PIL import Image, ImageDraw, ImageOps

    blocks: list[dict[str, Any]] = []
    with Image.open(io.BytesIO(raw_img)) as im:
        full_im = ImageOps.exif_transpose(im).convert("RGB")
        w0, h0 = max(1, full_im.width), max(1, full_im.height)

        scene_im = full_im.copy()
        scene_im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        if has_reticle:
            tw, th = scene_im.width, scene_im.height
            short_t = float(min(tw, th))
            cx_t = x_n * float(tw)
            cy_t = y_n * float(th)
            rx_t = max(8.0, rx_n * short_t)
            ry_t = max(8.0, ry_n * short_t)
            draw = ImageDraw.Draw(scene_im)
            draw.ellipse([cx_t - rx_t - 1, cy_t - ry_t - 1, cx_t + rx_t + 1, cy_t + ry_t + 1], outline=(255, 255, 255), width=4)
            draw.ellipse([cx_t - rx_t, cy_t - ry_t, cx_t + rx_t, cy_t + ry_t], outline=(37, 99, 235), width=2)

        buf_scene = io.BytesIO()
        scene_im.save(buf_scene, format="JPEG", quality=85)
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(buf_scene.getvalue()).decode("ascii"),
                },
            }
        )

        if has_reticle:
            short_0 = float(min(w0, h0))
            half_w = max(48, int(round(rx_n * short_0 * 1.65)))
            half_h = max(48, int(round(ry_n * short_0 * 1.65)))
            cx0 = int(round(x_n * float(w0)))
            cy0 = int(round(y_n * float(h0)))
            x0, y0 = max(0, cx0 - half_w), max(0, cy0 - half_h)
            x1, y1 = min(w0, cx0 + half_w), min(h0, cy0 + half_h)
            if (x1 - x0) >= 24 and (y1 - y0) >= 24:
                roi_im = full_im.crop((x0, y0, x1, y1))
                roi_im.thumbnail((512, 512), Image.Resampling.LANCZOS)
                buf_roi = io.BytesIO()
                roi_im.save(buf_roi, format="JPEG", quality=85)
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(buf_roi.getvalue()).decode("ascii"),
                        },
                    }
                )
    return blocks
