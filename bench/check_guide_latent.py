#!/usr/bin/env python3
"""Check `MiniMaxH3GuideLatent` (`guide_latent.py`) and the shipped guide-latent graphs.

Claims, i.e. what breaks if a case is deleted:
  resolve_guide_frames   `17k+5` temporal snapping, `<5 -> 1` fallback, negative
                         end-relative indexing, `single_frame=True` collapse to
                         1 frame at frame 0, and clear `ValueError` refusals on
                         out-of-range `frame_idx` or clips that overflow the
                         target video
  mask_to_rgb            `[H, W]`, `[1, H, W]`, `[T, H, W]`, and `[T, 1, H, W]`
                         masks convert to clamped `[target_frames, H, W, 3]` RGB
                         batches, and short multi-frame masks refuse with
                         `ValueError`
  execute video + still  `MiniMaxH3GuideLatent.execute` appends guide, optional
                         mask, and optional audio keyframes to `minimax_keyframes`,
                         replaces the AV latent with `(latent_t=1, audio_t=2)`
                         when `single_frame=True`, and produces a conditioning
                         dict that `comfy.ldm.minimax.model.PackedLayout` accepts
                         with `img_update=False` (`VISUAL_COND_TIMESTEP = 0.999`)
  refusal contracts      missing `guide`/`audio`, `mask` without `guide`, `audio`
                         with `single_frame=True`, missing `vae`/`audio_vae`, and
                         non-AV latents all raise `ValueError` naming the node
  shipped guide graphs   `workflows/h3_guide_video_edit_api.json` and
                         `workflows/image/h3_probe_pika_image_guide_edit_api.json`
                         wire `MiniMaxH3GuideLatent` -> `MiniMaxH3Preflight` on
                         `unet_ref2va`; the video graph chains a guide task LoRA
                         after Pika's, the one-frame graph loads Pika's alone

No GPU, no server, no model weights on disk.

    <comfy-venv-python> bench/check_guide_latent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workflows"))

from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)

import torch  # noqa: E402
from comfy.ldm.minimax.model import PackedLayout  # noqa: E402
from comfy_extras.nodes_minimax_h3 import _av_latent, _empty_av_latent  # noqa: E402

import h3_config as cfg  # noqa: E402
from guide_latent import (  # noqa: E402
    MiniMaxH3GuideLatent,
    mask_to_rgb,
    resolve_guide_frames,
)
from preflight import MiniMaxH3Preflight  # noqa: E402


class _DummyVideoVAE:
    """Deterministic stub matching `VAE.encode([T, H, W, 3]) -> [1, 24, latent_t, H//16, W//16]`."""

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        t, h, w, c = pixels.shape
        assert c == 3, f"expected 3 channels, got {c}"
        if t <= 1:
            lt = 1
        elif t <= 5:
            lt = 2
        else:
            lt = ((t - 5) // 17) * 5 + 2
        out = torch.full((1, 24, lt, h // 16, w // 16), float(pixels.mean().item()))
        return out


class _DummyAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform_blc: torch.Tensor) -> torch.Tensor:
        b, samples, _c = waveform_blc.shape
        t40 = max(1, round((samples / self.audio_sample_rate) * 40))
        return torch.ones((b, 32, 2, t40), dtype=torch.float32)


def _dummy_positive() -> list:
    return [[
        torch.zeros((1, 16, 5120)),
        {
            "minimax_token_tags": torch.ones(16, dtype=torch.long),
            "minimax_refs": [{
                "kind": "image",
                "latent": torch.zeros((1, 24, 1, 12, 20)),
                "latent_t": 1,
                "latent_h": 12,
                "latent_w": 20,
            }],
        },
    ]]


def _assert_refuses(fn, substr: str) -> None:
    try:
        fn()
    except ValueError as exc:
        msg = str(exc)
        assert substr in msg, f"expected {substr!r} in ValueError message, got {msg!r}"
        return
    raise AssertionError(f"expected ValueError containing {substr!r}, but call succeeded")


def check_resolve_guide_frames() -> None:
    assert resolve_guide_frames(1, 124, 0) == (0, 1)
    assert resolve_guide_frames(4, 124, 10) == (10, 1)
    assert resolve_guide_frames(5, 124, 0) == (0, 5)
    assert resolve_guide_frames(21, 124, 0) == (0, 5)
    assert resolve_guide_frames(22, 124, 0) == (0, 22)
    assert resolve_guide_frames(124, 124, 0) == (0, 124)
    assert resolve_guide_frames(130, 124, 0) == (0, 124)
    assert resolve_guide_frames(1, 124, -1) == (123, 1)
    assert resolve_guide_frames(10, 124, 0, single_frame=True) == (0, 1)
    assert resolve_guide_frames(1, 124, -1, single_frame=True) == (0, 1)

    _assert_refuses(lambda: resolve_guide_frames(0, 124, 0), "empty")
    _assert_refuses(lambda: resolve_guide_frames(1, 124, 124), "outside the video's 124 frames")
    _assert_refuses(lambda: resolve_guide_frames(22, 124, 110), "does not fit in the video's 124 frames")
    _assert_refuses(lambda: resolve_guide_frames(1, 124, 1, single_frame=True), "single_frame=True has 1 frame")


def check_mask_to_rgb() -> None:
    m2 = torch.tensor([[0.0, 0.5], [1.0, 2.0]])
    rgb = mask_to_rgb(m2, 5)
    assert rgb.shape == (5, 2, 2, 3)
    assert float(rgb.max().item()) == 1.0 and float(rgb.min().item()) == 0.0

    m4 = torch.ones((3, 1, 4, 4))
    rgb4 = mask_to_rgb(m4, 2)
    assert rgb4.shape == (2, 4, 4, 3)

    _assert_refuses(lambda: mask_to_rgb(torch.ones((2, 4, 4)), 5), "fewer than the 5 guide frames")
    _assert_refuses(lambda: mask_to_rgb(torch.zeros((0, 4, 4)), 1), "empty")


def check_execute_video_and_single_frame() -> None:
    vae = _DummyVideoVAE()
    audio_vae = _DummyAudioVAE()
    pos = _dummy_positive()
    latent_124, frame_count = _empty_av_latent(320, 192, 124)
    assert frame_count == 124

    # 1. Video mode with guide + mask + audio
    guide_vid = torch.full((22, 192, 320, 3), 0.25)
    mask_vid = torch.ones((22, 192, 320))
    audio = {"waveform": torch.zeros((1, 2, 32000 * 2)), "sample_rate": 32000}
    out_vid = MiniMaxH3GuideLatent.execute(
        positive=pos,
        latent=latent_124,
        frame_idx=0,
        single_frame=False,
        vae=vae,
        guide=guide_vid,
        mask=mask_vid,
        audio=audio,
        audio_vae=audio_vae,
    )
    cond_vid, lat_vid = out_vid.args
    assert lat_vid is latent_124
    refs_vid = cond_vid[0][1]["minimax_refs"]
    assert len(refs_vid) == 1
    kfs_vid = cond_vid[0][1]["minimax_keyframes"]
    assert len(kfs_vid) == 2
    assert kfs_vid[0]["resolved_frame_index"] == 0
    assert kfs_vid[0]["latent"].shape == (1, 24, 7, 12, 20)
    assert "audio_latent" in kfs_vid[0]
    assert kfs_vid[1]["resolved_frame_index"] == 0
    assert kfs_vid[1]["latent"].shape == (1, 24, 7, 12, 20)

    v_vid, a_vid = lat_vid["samples"].tensors
    layout_vid = PackedLayout(
        16,
        v_vid.shape[2],
        v_vid.shape[3],
        v_vid.shape[4],
        a_vid.shape[-1],
        keyframes=kfs_vid,
        refs=refs_vid,
    )
    cond_rows_vid = sum(b - a for a, b, k in layout_vid.segments if k == "cond")
    ref_rows_vid = sum(b - a for a, b, k in layout_vid.segments if k == "ref_img")
    assert cond_rows_vid == 2 * 7 * (12 // 2) * (20 // 2)
    assert ref_rows_vid == (12 // 2) * (20 // 2)
    pf_vid = MiniMaxH3Preflight.execute(conditioning=cond_vid, samples=lat_vid)
    assert "keyframes" in pf_vid.args[3] and "Ref 1 (image)" in pf_vid.args[3]

    # 2. Single-frame mode (collapses 124-frame latent to t=1, audio_t=2)
    guide_img = torch.full((1, 256, 256, 3), 0.75)
    out_sf = MiniMaxH3GuideLatent.execute(
        positive=pos,
        latent=latent_124,
        frame_idx=0,
        single_frame=True,
        vae=vae,
        guide=guide_img,
    )
    cond_sf, lat_sf = out_sf.args
    v_sf, a_sf = lat_sf["samples"].tensors
    assert v_sf.shape == (1, 24, 1, 12, 20)
    assert a_sf.shape == (1, 32, 2, 2)
    refs_sf = cond_sf[0][1]["minimax_refs"]
    assert len(refs_sf) == 1
    kfs_sf = cond_sf[0][1]["minimax_keyframes"]
    assert len(kfs_sf) == 1
    assert kfs_sf[0]["resolved_frame_index"] == 0
    assert kfs_sf[0]["latent"].shape == (1, 24, 1, 12, 20)

    layout_sf = PackedLayout(
        16,
        v_sf.shape[2],
        v_sf.shape[3],
        v_sf.shape[4],
        a_sf.shape[-1],
        keyframes=kfs_sf,
        refs=refs_sf,
    )
    cond_rows_sf = sum(b - a for a, b, k in layout_sf.segments if k == "cond")
    ref_rows_sf = sum(b - a for a, b, k in layout_sf.segments if k == "ref_img")
    assert cond_rows_sf == (12 // 2) * (20 // 2)
    assert ref_rows_sf == (12 // 2) * (20 // 2)
    pf_sf = MiniMaxH3Preflight.execute(conditioning=cond_sf, samples=lat_sf)
    assert "1 frame" in pf_sf.args[3] and "keyframes" in pf_sf.args[3] and "Ref 1 (image)" in pf_sf.args[3]


def check_refusals() -> None:
    vae = _DummyVideoVAE()
    pos = _dummy_positive()
    lat = _av_latent(320, 192, 2, 8)
    guide = torch.zeros((1, 192, 320, 3))
    mask = torch.ones((1, 192, 320))
    audio = {"waveform": torch.zeros((1, 2, 16000)), "sample_rate": 32000}

    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(positive=pos, latent=lat),
        "needs a `guide` image/video or `audio`",
    )
    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(positive=pos, latent=lat, mask=mask),
        "`mask` requires a `guide`",
    )
    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(
            positive=pos, latent=lat, single_frame=True, vae=vae, guide=guide, audio=audio
        ),
        "`audio` cannot be anchored when `single_frame=True`",
    )
    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(positive=pos, latent=lat, guide=guide),
        "needs the `vae` input",
    )
    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(positive=pos, latent=lat, audio=audio),
        "needs the `audio_vae` input",
    )
    _assert_refuses(
        lambda: MiniMaxH3GuideLatent.execute(
            positive=pos, latent={"samples": torch.zeros((1, 4, 8, 8))}, vae=vae, guide=guide
        ),
        "expects a MiniMax H3 AV latent",
    )


def check_shipped_guide_graphs() -> None:
    vid_path = REPO / "workflows" / "h3_guide_video_edit_api.json"
    img_path = REPO / "workflows" / "image" / "h3_probe_pika_image_guide_edit_api.json"
    assert vid_path.is_file(), f"missing {vid_path}"
    assert img_path.is_file(), f"missing {img_path}"

    # The one-frame graph loads no guide task LoRA since 2026-11-11 (owner):
    # the guide node is wired, the edit LoRA is not.
    for path, want_sf, want_distill_lora, want_task_lora in (
        (vid_path, False, cfg.PIKA_REF2VA_SVD995_LORA, True),
        (img_path, True, cfg.PIKA_REF2VA_SVD995_LORA, False),
    ):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["1"]["inputs"]["unet_name"] == cfg.MODELS["unet_ref2va"], (
            f"{path.name}: expected unet_ref2va"
        )
        assert doc["18"]["class_type"] == cfg.LORA_BRANCH_NODE
        assert doc["18"]["inputs"]["lora_name"] == want_distill_lora
        if want_task_lora:
            assert doc["72"]["class_type"] == cfg.LORA_BRANCH_NODE
            assert doc["72"]["inputs"]["lora_name"] in cfg.GUIDE_TASK_LORAS
        else:
            loaded = sorted(
                n["inputs"]["lora_name"] for n in doc.values()
                if isinstance(n, dict) and "lora_name" in n.get("inputs", {})
            )
            assert loaded == [want_distill_lora], (
                f"{path.name}: expected Pika's LoRA alone, found {loaded}"
            )
        assert doc["70"]["class_type"] == cfg.GUIDE_LATENT_NODE
        assert doc["70"]["inputs"]["single_frame"] is want_sf
        assert doc["26"]["class_type"] == "MiniMaxH3Preflight"
        assert doc["26"]["inputs"]["conditioning"] == ["70", 0]
        assert doc["26"]["inputs"]["samples"] == ["70", 1]


def main() -> int:
    cases = (
        ("resolve_guide_frames grid snapping and bounds", check_resolve_guide_frames),
        ("mask_to_rgb shape normalization and clamping", check_mask_to_rgb),
        ("MiniMaxH3GuideLatent.execute video + single-frame + PackedLayout", check_execute_video_and_single_frame),
        ("MiniMaxH3GuideLatent refusal contracts", check_refusals),
        ("shipped guide-latent workflows (video + single-frame)", check_shipped_guide_graphs),
    )
    failures = []
    print("MiniMaxH3GuideLatent contract and shipped workflows")
    for label, fn in cases:
        try:
            fn()
            print(f"  ok    {label}")
        except Exception as exc:
            failures.append(label)
            print(f"  FAIL  {label}: {exc}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
