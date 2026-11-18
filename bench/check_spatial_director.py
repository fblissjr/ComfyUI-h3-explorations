#!/usr/bin/env python3
"""Offline checks for MiniMaxH3SpatialDirector, MiniMaxH3DirectorHUD, and workflows."""

from __future__ import annotations

import importlib
import json
import math
import re
import sys
import types
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from _lib import bootstrap, case, finish, refused  # noqa: E402

bootstrap(cpu=True)

if str(_REPO) in sys.path:
    sys.path.remove(str(_REPO))

import nodes as _comfy_nodes  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

_PKG = "_h3pack"
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [str(_REPO)]
    sys.modules[_PKG] = _pkg
sd_mod = importlib.import_module(f"{_PKG}.spatial_director")
sc_mod = importlib.import_module(f"{_PKG}.sound_cues")
st_mod = importlib.import_module(f"{_PKG}.sprite_trajectory")
hr_mod = importlib.import_module(f"{_PKG}.h3_rules")


def _nodes_by_class(graph: dict[str, dict], class_type: str) -> list[tuple[str, dict]]:
    return [
        (nid, node)
        for nid, node in graph.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]


def _check_portal_path_default() -> None:
    out = sd_mod.MiniMaxH3SpatialDirector.execute()
    mask, cues_text, kfs_text, plan_json = out.result
    assert tuple(mask.shape) == (141, 768, 1344), f"unexpected mask shape {mask.shape}"
    assert mask.dtype == torch.float32
    assert torch.isfinite(mask).all()
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0

    # Frame 0 portal at (0.18, 0.52), frame 70 at (0.50, 0.50), frame 140 at (0.82, 0.52).
    assert float(mask[0, int(0.52 * 768), int(0.18 * 1344)]) > 0.95
    assert float(mask[0, int(0.52 * 768), int(0.82 * 1344)]) < 0.05
    assert float(mask[70, int(0.50 * 768), int(0.50 * 1344)]) > 0.95
    assert float(mask[140, int(0.52 * 768), int(0.82 * 1344)]) > 0.95
    assert float(mask[140, int(0.52 * 768), int(0.18 * 1344)]) < 0.05

    cues = sc_mod.parse_sound_cues(cues_text, 141, snap_to_3f=True)
    assert len(cues) >= 1 and all(c.resolved_frame % 3 == 0 for c in cues), f"bad cues: {cues}"
    spec = st_mod.parse_sprite_trajectory(kfs_text, 141)
    assert len(spec.waypoints) == 4 and len(spec.hops) == 1

    plan = json.loads(plan_json)
    assert plan["mode"] == "portal_path"
    assert plan["length"] == 141 and len(plan["portal_frames"]) == 141


def _check_portal_shapes_and_short_length() -> None:
    ellipse_events = json.dumps(
        {
            "portal": [
                {"frame": 0, "x": 0.30, "y": 0.50, "rx": 0.25, "ry": 0.12, "shape": "ellipse", "label": "E"},
                {"frame": 89, "x": 0.70, "y": 0.50, "rx": 0.25, "ry": 0.12, "shape": "ellipse", "label": "E"},
            ]
        }
    )
    e_mask = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=320, height=192, length=90, feather=0.05, events_json=ellipse_events
    ).result[0]
    assert tuple(e_mask.shape) == (90, 192, 320)
    # Horizontal rx=0.25 reaches x=0.30+0.20=0.50, while vertical ry=0.12 stays outside y=0.50+0.20=0.70.
    assert float(e_mask[0, int(0.50 * 192), int(0.45 * 320)]) > 0.85
    assert float(e_mask[0, int(0.75 * 192), int(0.30 * 320)]) < 0.05

    split_events = json.dumps(
        {
            "portal": [
                {"frame": 0, "x": 0.35, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "vertical_split", "label": "SPLIT"},
                {"frame": 89, "x": 0.65, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "vertical_split", "label": "SPLIT"},
            ]
        }
    )
    s_mask = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=320, height=192, length=90, feather=0.05, events_json=split_events
    ).result[0]
    assert float(s_mask[0, 96, int(0.20 * 320)]) < 0.05
    assert float(s_mask[0, 96, int(0.80 * 320)]) > 0.95
    assert float(s_mask[89, 96, int(0.50 * 320)]) < 0.05
    assert float(s_mask[89, 96, int(0.90 * 320)]) > 0.95


def _check_point_targets_both_lengths() -> None:
    mask, cues_text, kfs_text, plan_json = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="point_targets", width=640, height=384, length=141
    ).result
    assert tuple(mask.shape) == (141, 384, 640)
    # Before Target A (frame 24), the mask should be inactive everywhere.
    assert float(mask[0].max()) < 1e-5
    # Frame 40 hits TARGET A (0.24, 0.54) and spares TARGET B (0.76, 0.54).
    assert float(mask[40, int(0.54 * 384), int(0.24 * 640)]) > 0.90
    assert float(mask[40, int(0.54 * 384), int(0.76 * 640)]) < 0.05
    # Frame 100 hits TARGET B (0.76, 0.54) and spares TARGET A (0.24, 0.54).
    assert float(mask[100, int(0.54 * 384), int(0.76 * 640)]) > 0.90
    assert float(mask[100, int(0.54 * 384), int(0.24 * 640)]) < 0.05

    cues = sc_mod.parse_sound_cues(cues_text, 141, snap_to_3f=True)
    assert [(c.resolved_frame, c.preset) for c in cues] == [(24, "slam"), (78, "hop")]
    assert all(c.resolved_frame % 3 == 0 for c in cues)

    spec = st_mod.parse_sprite_trajectory(kfs_text, 141)
    assert len(spec.waypoints) == 4 and len(spec.hops) == 1
    plan = json.loads(plan_json)
    assert plan["mode"] == "point_targets" and len(plan["targets"]) == 2

    # Shorter 17n+5=90 length with frame-bounded targets and waypoints.
    short_events = json.dumps(
        {
            "targets": [
                {"id": 1, "label": "SHORT A", "start": 12, "end": 42, "ramp": 3, "x": 0.30, "y": 0.50, "radius": 0.18, "cue": "slam", "gain": 1.1},
                {"id": 2, "label": "SHORT B", "start": 48, "end": 81, "ramp": 3, "x": 0.72, "y": 0.50, "radius": 0.18, "cue": "click", "gain": 1.0},
            ],
            "dpad": {
                "waypoints": [
                    {"frame": 0, "x": 0.25, "y": 0.75, "face": "right", "scale": 1.0},
                    {"frame": 89, "x": 0.75, "y": 0.75, "face": "right", "scale": 1.0},
                ],
                "hops": [{"start": 30, "end": 42, "lift": 0.2}],
            },
        }
    )
    m90, c90, k90, _ = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="point_targets", width=320, height=192, length=90, events_json=short_events
    ).result
    assert tuple(m90.shape) == (90, 192, 320)
    assert [(c.resolved_frame, c.preset) for c in sc_mod.parse_sound_cues(c90, 90, snap_to_3f=True)] == [
        (12, "slam"),
        (48, "click"),
    ]
    assert len(st_mod.parse_sprite_trajectory(k90, 90).waypoints) == 2


def _check_invert_mask() -> None:
    events = json.dumps(
        {
            "portal": [
                {"frame": 0, "x": 0.30, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle"},
                {"frame": 89, "x": 0.70, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle"},
            ]
        }
    )
    base = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=320, height=192, length=90, invert_mask=False, events_json=events
    ).result[0]
    inv = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=320, height=192, length=90, invert_mask=True, events_json=events
    ).result[0]
    assert torch.allclose(inv, 1.0 - base, atol=1e-5)


def _check_refusals() -> None:
    refused(lambda: sd_mod.MiniMaxH3SpatialDirector.execute(length=91), "17n + 5")
    refused(lambda: sd_mod.MiniMaxH3SpatialDirector.execute(events_json="{bad"), "invalid")
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="point_targets",
            length=90,
            events_json=json.dumps(
                {"targets": [{"start": 9, "end": 30, "x": 0.5, "y": 0.5, "radius": 0.2, "cue": "laser_zapper"}]}
            ),
        ),
        "unknown",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="portal_path",
            length=90,
            events_json=json.dumps({"portal": [{"frame": 0, "x": 0.5, "y": 0.5, "rx": 0.0, "ry": 0.2, "shape": "circle"}]}),
        ),
        "outside",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="portal_path",
            length=90,
            events_json=json.dumps({"portal": []}),
        ),
        "at least one keyframe",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="point_targets",
            length=90,
            events_json=json.dumps({"targets": []}),
        ),
        "at least one entry",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="point_targets",
            length=90,
            events_json=json.dumps({"targets": [{"start": 20, "end": 20, "x": 0.5, "y": 0.5, "radius": 0.2}]}),
        ),
        "exceed",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="point_targets",
            width=320,
            height=192,
            length=90,
            events_json=json.dumps({"targets": [{"start": 10, "end": 30, "x": 2.0, "y": 2.0, "radius": 0.05}]}),
        ),
        "outside",
    )
    refused(
        lambda: sd_mod.MiniMaxH3SpatialDirector.execute(
            mode="portal_path",
            width=320,
            height=192,
            length=90,
            events_json=json.dumps({"portal": [{"frame": 0, "x": 0.99, "y": 0.99, "rx": 0.01, "ry": 0.01, "shape": "hex"}]}),
        ),
        "vertical_split",
    )


def _check_hud_layouts() -> None:
    plan = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="point_targets", width=320, height=192, length=141
    ).result[3]
    imgs = torch.full((141, 193, 321, 3), 0.25)
    base = torch.full((141, 193, 321, 3), 0.75)

    ret_only = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=plan, hud_layout="canvas_reticle_only"
    ).result[0]
    assert tuple(ret_only.shape) == (141, 192, 320, 3)
    assert ret_only.shape[1] % 2 == 0 and ret_only.shape[2] % 2 == 0
    assert float(ret_only.min()) >= 0.0 and float(ret_only.max()) <= 1.0

    card_no_base = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=plan, hud_layout="director_card"
    ).result[0]
    card_with_base = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=plan, hud_layout="director_card", baseline_images=base
    ).result[0]
    assert card_no_base.shape[0] == 141
    assert card_no_base.shape[1] % 2 == 0 and card_no_base.shape[2] % 2 == 0
    assert card_no_base.shape[1] > 192
    assert float((card_with_base - card_no_base).abs().max()) > 0.25

    split_no_base = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=plan, hud_layout="split_ab_card"
    ).result[0]
    split_with_base = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=plan, hud_layout="split_ab_card", baseline_images=base
    ).result[0]
    assert split_with_base.shape[0] == 141
    assert split_with_base.shape[1] % 2 == 0 and split_with_base.shape[2] % 2 == 0
    assert split_with_base.shape[2] > 640
    assert float((split_with_base - split_no_base).abs().max()) > 0.25


def _assert_dual_vhs(graph: dict[str, dict], hud_id: str) -> None:
    vhs_nodes = _nodes_by_class(graph, "VHS_VideoCombine")
    assert len(vhs_nodes) == 2, f"expected dual VHS_VideoCombine ('13' + '162'), got {len(vhs_nodes)}"
    assert "13" in graph and "162" in graph
    hud_vhs = graph["13"]["inputs"]
    clean_vhs = graph["162"]["inputs"]
    assert hud_vhs["images"] == [hud_id, 0]
    assert clean_vhs["images"] == ["11", 0]
    assert clean_vhs["audio"] == hud_vhs["audio"]
    assert clean_vhs["filename_prefix"] == f"{hud_vhs['filename_prefix']}_clean"
    for k, v in hud_vhs.items():
        if k not in ("images", "filename_prefix"):
            assert clean_vhs[k] == v, f"VHS param {k!r} mismatch between '13' and '162'"


def _check_weight_halo_and_portal_window() -> None:
    w, h, n_frames, feather = 320, 192, 39, 0.08
    short_edge = float(min(w, h))
    feather_px = feather * short_edge
    px = np.arange(w, dtype=np.float32)[None, :] + 0.5
    py = np.arange(h, dtype=np.float32)[:, None] + 0.5

    # 1. Bitwise identity on default portal_path (no weight/halo/portal_start/portal_end/portal_ramp)
    p_def_json = json.dumps(
        {
            "portal": [
                {"frame": 0, "x": 0.25, "y": 0.50, "rx": 0.20, "ry": 0.20, "shape": "circle"},
                {"frame": 38, "x": 0.75, "y": 0.50, "rx": 0.24, "ry": 0.24, "shape": "circle"},
            ]
        }
    )
    out_def = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=w, height=h, length=n_frames, feather=feather, events_json=p_def_json
    ).result
    mask_def = out_def[0]
    plan_def = json.loads(out_def[3])
    assert plan_def["portal_start"] == 0 and plan_def["portal_end"] == n_frames - 1
    portals, targets, _, _, win = sd_mod.parse_director_events(
        p_def_json, mode="portal_path", length=n_frames, return_portal_window=True
    )
    assert win == (0, n_frames - 1, 0)
    ref_arr = np.zeros((n_frames, h, w), dtype=np.float32)
    for f in range(n_frames):
        x, y, rx, _ry, _shape, _lbl, wt, hr, hw = sd_mod._eval_portal_frame(portals, f)
        assert wt == 1.0 and hr == 0.0 and hw == 0.30
        rad_px = max(2.0, float(rx) * short_edge)
        dist = np.sqrt((px - float(x) * w) ** 2 + (py - float(y) * h) ** 2) - rad_px
        ref_arr[f] = sd_mod._soft_edge_from_signed_dist(dist, feather_px)
    assert torch.equal(mask_def, torch.from_numpy(ref_arr)), "default portal_path mask must stay bitwise identical"

    # 2. Bitwise identity on default point_targets
    t_def_json = json.dumps(
        {
            "targets": [
                {"id": 1, "start": 6, "end": 30, "ramp": 4, "x": 0.5, "y": 0.5, "radius": 0.18, "cue": "click"}
            ]
        }
    )
    mask_t_def = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="point_targets", width=w, height=h, length=n_frames, feather=feather, events_json=t_def_json
    ).result[0]
    _, t_evs, _, _ = sd_mod.parse_director_events(t_def_json, mode="point_targets", length=n_frames)
    t_obj = t_evs[0]
    assert t_obj.weight == 1.0 and t_obj.halo_radius == 0.0 and t_obj.halo_weight == 0.30
    rad_px = max(2.0, float(t_obj.radius) * short_edge)
    s_disc = sd_mod._soft_edge_from_signed_dist(
        np.sqrt((px - float(t_obj.x) * w) ** 2 + (py - float(t_obj.y) * h) ** 2) - rad_px,
        feather_px,
    )
    ref_t_arr = np.zeros((n_frames, h, w), dtype=np.float32)
    for f in range(n_frames):
        tw = sd_mod._target_temporal_weight(t_obj, f)
        if tw > 1e-5:
            ref_t_arr[f] = s_disc * np.float32(tw)
    assert torch.equal(mask_t_def, torch.from_numpy(ref_t_arr)), "default point_targets mask must stay bitwise identical"

    # 3. Portal keyframe cosine-eased weight, outer cosine halo shelf, bounded portal_start/end/ramp, and HUD dashed halo
    bounded_json = json.dumps(
        {
            "portal_start": 6,
            "portal_end": 30,
            "portal_ramp": 4,
            "portal": [
                {
                    "frame": 6,
                    "x": 0.50,
                    "y": 0.50,
                    "rx": 0.18,
                    "ry": 0.18,
                    "shape": "circle",
                    "weight": 0.60,
                    "halo_radius": 0.20,
                    "halo_weight": 0.50,
                },
                {
                    "frame": 30,
                    "x": 0.50,
                    "y": 0.50,
                    "rx": 0.18,
                    "ry": 0.18,
                    "shape": "circle",
                    "weight": 0.90,
                    "halo_radius": 0.20,
                    "halo_weight": 0.50,
                },
            ],
        }
    )
    out_b = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=w, height=h, length=n_frames, feather=feather, events_json=bounded_json
    ).result
    mask_b = out_b[0].numpy()
    plan_b = json.loads(out_b[3])
    assert plan_b["portal_start"] == 6 and plan_b["portal_end"] == 30
    assert plan_b["portals"][0]["weight"] == 0.6 and plan_b["portals"][0]["halo_radius"] == 0.2
    assert plan_b["portals"][0]["halo_weight"] == 0.5
    assert mask_b[:6].max() == 0.0 and mask_b[31:].max() == 0.0
    # Mid-plateau frame 18 has u = 0.5 -> eased weight = 0.75
    assert abs(float(mask_b[18, h // 2, w // 2]) - 0.75) < 1e-4
    # Fade-in frame 6 matches target temporal ramp formula
    expected_f6 = 0.60 * sd_mod._cosine_ease(1.0 / 5.0)
    assert abs(float(mask_b[6, h // 2, w // 2]) - expected_f6) < 1e-4
    # Outer halo check at frame 18: core radius = 0.18*192 = 34.56px, halo_px = 0.20*192 = 38.4px
    # At dx = 55px from center: dist_2d = 20.44px (> half_feather 7.68px so core==0, inside halo <= 38.4px)
    cx = 0.5 * w
    cy = 0.5 * h
    dx_test = 55
    col = int(cx) + dx_test
    row = int(cy)
    dist_pt = math.hypot((col + 0.5) - cx, (row + 0.5) - cy) - (0.18 * short_edge)
    halo_px = 0.20 * short_edge
    expected_halo = (0.75 * 0.50) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, dist_pt / halo_px))))
    assert abs(float(mask_b[18, row, col]) - expected_halo) < 1e-4
    # Beyond core + halo (dx = 80px -> dist ~ 45.4px > 38.4px) is exact 0.0
    assert float(mask_b[18, row, int(cx) + 80]) == 0.0

    # HUD draws dashed outer halo circle visibly when halo_radius > 0
    no_halo_json = json.dumps(
        {
            "portal_start": 6,
            "portal_end": 30,
            "portal_ramp": 4,
            "portal": [
                {"frame": 6, "x": 0.50, "y": 0.50, "rx": 0.18, "ry": 0.18, "shape": "circle", "weight": 0.60},
                {"frame": 30, "x": 0.50, "y": 0.50, "rx": 0.18, "ry": 0.18, "shape": "circle", "weight": 0.90},
            ],
        }
    )
    plan_nh = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=w, height=h, length=n_frames, feather=feather, events_json=no_halo_json
    ).result[3]
    frames = torch.full((n_frames, h, w, 3), 0.35, dtype=torch.float32)
    hud_halo = sd_mod.MiniMaxH3DirectorHUD.execute(
        frames, director_plan=out_b[3], hud_layout="canvas_reticle_only"
    ).result[0]
    hud_no_halo = sd_mod.MiniMaxH3DirectorHUD.execute(
        frames, director_plan=plan_nh, hud_layout="canvas_reticle_only"
    ).result[0]
    assert float((hud_halo[18] - hud_no_halo[18]).abs().max()) > 0.15
    # Closed frame (f=2) differs from active frame (f=18)
    assert float((hud_halo[18] - hud_halo[2]).abs().max()) > 0.15

    # 4. WorldTargetEvent weight + halo
    t_halo_json = json.dumps(
        {
            "targets": [
                {
                    "id": 1,
                    "start": 8,
                    "end": 30,
                    "ramp": 0,
                    "x": 0.5,
                    "y": 0.5,
                    "radius": 0.15,
                    "cue": "click",
                    "weight": 0.50,
                    "halo_radius": 0.18,
                    "halo_weight": 0.40,
                }
            ]
        }
    )
    out_th = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="point_targets", width=w, height=h, length=n_frames, feather=feather, events_json=t_halo_json
    ).result
    mask_th = out_th[0].numpy()
    plan_th = json.loads(out_th[3])
    assert plan_th["targets"][0]["weight"] == 0.5
    assert plan_th["targets"][0]["halo_radius"] == 0.18
    assert plan_th["targets"][0]["halo_weight"] == 0.4
    assert abs(float(mask_th[15, h // 2, w // 2]) - 0.50) < 1e-4
    assert float(mask_th[15, row, int(cx) + 46]) > 0.02

    # 5. Refusals for invalid halo/weight/portal window values
    refused(
        lambda: sd_mod.parse_director_events(
            json.dumps({"portal": [{"frame": 0, "weight": 0.01}]}), mode="portal_path", length=n_frames
        ),
        "weight",
    )
    refused(
        lambda: sd_mod.parse_director_events(
            json.dumps({"portal": [{"frame": 0, "halo_radius": 1.5}]}), mode="portal_path", length=n_frames
        ),
        "halo_radius",
    )
    refused(
        lambda: sd_mod.parse_director_events(
            json.dumps({"portal_start": 20, "portal_end": 10, "portal": [{"frame": 0}]}),
            mode="portal_path",
            length=n_frames,
        ),
        "exceed",
    )
    refused(
        lambda: sd_mod.parse_director_events(
            json.dumps({"portal_ramp": -1, "portal": [{"frame": 0}]}),
            mode="portal_path",
            length=n_frames,
        ),
        "portal_ramp",
    )


def _check_js_sound_presets() -> None:
    js_path = _REPO / "web" / "h3_spatial_director.js"
    js_text = js_path.read_text(encoding="utf-8")
    m = re.search(r"const SOUND_PRESETS\s*=\s*\[(.*?)\];", js_text, re.S)
    assert m is not None, "SOUND_PRESETS declaration not found in web/h3_spatial_director.js"
    js_presets = re.findall(r'"([^"]+)"', m.group(1))
    expected = sorted(sc_mod.SOUND_CUE_PRESETS) + ["none"]
    assert set(js_presets) == set(expected) and len(js_presets) == len(expected), (
        f"SOUND_PRESETS {js_presets} != expected {expected}"
    )
    cues_used = re.findall(r'\bcue\s*:\s*"([^"]+)"', js_text)
    assert cues_used, "expected cue defaults in web/h3_spatial_director.js"
    for cue in cues_used:
        assert cue in sc_mod.SOUND_CUE_PRESETS, f"stale/unknown cue {cue!r} in web/h3_spatial_director.js"


def _check_same_prompt_builder() -> None:
    wf_dir = str(_REPO / "workflows")
    if wf_dir not in sys.path:
        sys.path.insert(0, wf_dir)
    import build_workflows as bw  # noqa: E402

    for task, extra in (
        ("t2v", {}),
        ("r2v", {"ref_images": bw.SPATIAL_PORTAL_REF_IMAGES, "ref_image_count": 1}),
        ("i2v", {"first_frame": True}),
    ):
        g = bw.build_api(
            task,
            length=141,
            prompt=bw.SPATIAL_PORTAL_T2V_PROMPT
            if task == "t2v"
            else bw.SPATIAL_PORTAL_REF2V_PROMPT
            if task == "r2v"
            else bw.SPATIAL_PORTAL_I2V_PROMPT,
            lora=(bw.PIKA_REF2VA_SVD995_LORA if task == "r2v" else bw.PIKA_FL2VA_SVD995_LORA, bw.PIKA_STRENGTH),
            lora_branch=True,
            steps=bw.PIKA_STEPS,
            sampler_name=bw.PIKA_SAMPLER,
            manual_sigmas=bw.PIKA_MANUAL_SIGMAS,
            spatial_director=True,
            control_vector=dict(vector="film_16mm_1974", strength=1.1, edit_pass="edited_only"),
            edit_guidance=dict(same_prompt=True, spatial_mask=True, streams="independent", scale=2.0, audio_scale=1.0),
            **extra,
        )
        assert "130" not in g, f"{task}: same_prompt=True must omit MiniMaxH3PromptEdit ('130')"
        assert g["133"]["class_type"] == "PrimitiveStringMultiline"
        assert g["5"]["inputs"]["prompt"] == ["133", 0]
        assert g["131"]["inputs"]["prompt"] == ["133", 0]
        assert g["115"]["class_type"] == "MiniMaxH3Steer"
        assert g["115"]["inputs"]["edit_pass"] == "edited_only"
        assert g["9"]["class_type"] == "MiniMaxH3EditGuider"
        assert g["9"]["inputs"]["model"] == ["115", 0]
        assert g["9"]["inputs"]["edited"] == ["131", 0]
        assert g["9"]["inputs"]["mask"] == ["160", 0]


def _check_shipped_workflows() -> None:
    portal_path = _REPO / "workflows" / "h3_spatial_portal_pika_api.json"
    targets_path = _REPO / "workflows" / "h3_world_targets_genie_pika_api.json"
    r2v_portal_path = _REPO / "workflows" / "h3_r2v_spatial_portal_pika_api.json"
    i2v_portal_path = _REPO / "workflows" / "h3_i2v_spatial_portal_pika_api.json"
    assert portal_path.exists() and targets_path.exists()
    assert r2v_portal_path.exists() and i2v_portal_path.exists()

    portal_graph = json.loads(portal_path.read_text(encoding="utf-8"))
    sd_nodes = _nodes_by_class(portal_graph, "MiniMaxH3SpatialDirector")
    hud_nodes = _nodes_by_class(portal_graph, "MiniMaxH3DirectorHUD")
    eg_nodes = _nodes_by_class(portal_graph, "MiniMaxH3EditGuider")
    assert len(sd_nodes) == 1 and len(hud_nodes) == 1 and len(eg_nodes) == 1
    sd_id, sd_node = sd_nodes[0]
    hud_id, hud_node = hud_nodes[0]
    _, eg_node = eg_nodes[0]
    assert sd_node["inputs"]["mode"] == "portal_path"
    assert eg_node["inputs"]["streams"] == "independent"
    assert eg_node["inputs"]["mask"] == [sd_id, 0]
    assert hud_node["inputs"]["director_plan"] == [sd_id, 3]
    _assert_dual_vhs(portal_graph, hud_id)

    targets_graph = json.loads(targets_path.read_text(encoding="utf-8"))
    sd_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3SpatialDirector")
    hud_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3DirectorHUD")
    eg_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3EditGuider")
    sc_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3SoundCues")
    st_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3AppendSpriteTrajectory")
    assert (
        len(sd_nodes_t) == 1
        and len(hud_nodes_t) == 1
        and len(eg_nodes_t) == 1
        and len(sc_nodes_t) == 1
        and len(st_nodes_t) == 1
    )
    sd_tid, sd_tnode = sd_nodes_t[0]
    hud_tid, hud_tnode = hud_nodes_t[0]
    assert sd_tnode["inputs"]["mode"] == "point_targets"
    assert eg_nodes_t[0][1]["inputs"]["streams"] == "video_only"
    assert eg_nodes_t[0][1]["inputs"]["mask"] == [sd_tid, 0]
    assert sc_nodes_t[0][1]["inputs"]["cues"] == [sd_tid, 1]
    assert st_nodes_t[0][1]["inputs"]["keyframes"] == [sd_tid, 2]
    assert hud_tnode["inputs"]["director_plan"] == [sd_tid, 3]
    _assert_dual_vhs(targets_graph, hud_tid)

    r2v_graph = json.loads(r2v_portal_path.read_text(encoding="utf-8"))
    sd_r = _nodes_by_class(r2v_graph, "MiniMaxH3SpatialDirector")
    hud_r = _nodes_by_class(r2v_graph, "MiniMaxH3DirectorHUD")
    eg_r = _nodes_by_class(r2v_graph, "MiniMaxH3EditGuider")
    enc_r = _nodes_by_class(r2v_graph, "MiniMaxH3EncodeReferences")
    pr_r = _nodes_by_class(r2v_graph, "MiniMaxH3PromptOnReferences")
    app_r = _nodes_by_class(r2v_graph, "MiniMaxH3AppendRefImage")
    assert len(sd_r) == 1 and len(hud_r) == 1 and len(eg_r) == 1
    assert len(enc_r) == 1 and len(pr_r) == 2 and len(app_r) == 1
    assert "51" not in r2v_graph and "70" not in r2v_graph
    assert r2v_graph["15"]["inputs"]["image"] == "ref_signal/tortoise_pic1.png"
    assert sd_r[0][1]["inputs"]["mode"] == "portal_path"
    assert eg_r[0][1]["inputs"]["streams"] == "independent"
    assert eg_r[0][1]["inputs"]["mask"] == [sd_r[0][0], 0]
    assert hud_r[0][1]["inputs"]["director_plan"] == [sd_r[0][0], 3]
    _assert_dual_vhs(r2v_graph, hud_r[0][0])

    i2v_graph = json.loads(i2v_portal_path.read_text(encoding="utf-8"))
    sd_i = _nodes_by_class(i2v_graph, "MiniMaxH3SpatialDirector")
    hud_i = _nodes_by_class(i2v_graph, "MiniMaxH3DirectorHUD")
    eg_i = _nodes_by_class(i2v_graph, "MiniMaxH3EditGuider")
    res_i = _nodes_by_class(i2v_graph, "MiniMaxH3Resolution")
    cond_i = _nodes_by_class(i2v_graph, "MiniMaxH3Conditioning")
    assert len(sd_i) == 1 and len(hud_i) == 1 and len(eg_i) == 1
    assert len(res_i) == 1 and len(cond_i) == 2
    assert i2v_graph["27"]["inputs"]["shape"] == "same as the picture"
    assert i2v_graph["27"]["inputs"]["picture"] == ["15", 0]
    assert i2v_graph["27"]["inputs"]["length"] == 141
    for cid in ("5", "131"):
        ci = i2v_graph[cid]["inputs"]
        assert ci["width"] == ["27", 0] and ci["height"] == ["27", 1] and ci["length"] == ["27", 2]
        assert ci["canvas"] == "explicit" and ci["first_frame"] == ["15", 0]
    assert eg_i[0][1]["inputs"]["mask"] == [sd_i[0][0], 0]
    assert hud_i[0][1]["inputs"]["director_plan"] == [sd_i[0][0], 3]
    _assert_dual_vhs(i2v_graph, hud_i[0][0])


def _check_combined_and_aux_masks() -> None:
    w, h, n_frames, feather = 320, 192, 39, 0.04
    comb_json = json.dumps(
        {
            "portal": [{"frame": 0, "x": 0.25, "y": 0.50, "rx": 0.14, "ry": 0.14, "shape": "circle"}],
            "targets": [
                {"id": 1, "start": 6, "end": 30, "ramp": 2, "x": 0.78, "y": 0.50, "radius": 0.14, "cue": "impact", "gain": 1.1}
            ],
            "portal_cues": [{"frame": 6, "preset": "click", "gain": 0.6}],
        }
    )
    out_comb = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="combined", width=w, height=h, length=n_frames, feather=feather, events_json=comb_json
    ).result
    m_comb = out_comb[0]
    cues_comb = out_comb[1]
    plan_comb = json.loads(out_comb[3])
    assert plan_comb["mode"] == "combined"
    assert len(plan_comb["cursor_frames"]) == n_frames
    # Check both portal (left) and active target (right) at frame 12
    assert float(m_comb[12, h // 2, int(w * 0.25)]) > 0.90
    assert float(m_comb[12, h // 2, int(w * 0.78)]) > 0.90
    # Frame 6 collision between target cue (gain 1.1) and portal_cue (gain 0.6) deduplicates to max gain (1.1)
    assert cues_comb.count("6:") == 1 and "impact" in cues_comb
    assert "pan=+0.56" in cues_comb, f"expected stereo pan=+0.56 on off-center target x=0.78, got {cues_comb!r}"
    parsed_comb = sc_mod.parse_sound_cues(cues_comb, n_frames, snap_to_3f=True)
    assert abs(parsed_comb[0].pan - 0.56) <= 0.02

    # Optional D-pad sprite envelope trajectory mask union (`include_sprite_envelope=True`)
    env_json = json.dumps(
        {
            "portal": [{"frame": 0, "x": 0.22, "y": 0.50, "rx": 0.14, "ry": 0.14, "shape": "circle"}],
            "targets": [],
            "dpad": {
                "waypoints": [
                    {"frame": 0, "x": 0.80, "y": 0.70, "face": "left", "scale": 1.0},
                    {"frame": n_frames - 1, "x": 0.80, "y": 0.30, "face": "left", "scale": 1.0},
                ],
                "hops": [],
            },
        }
    )
    m_no_env = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=w, height=h, length=n_frames, feather=0.04, events_json=env_json, include_sprite_envelope=False
    ).result[0]
    out_with_env = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="portal_path", width=w, height=h, length=n_frames, feather=0.04, events_json=env_json, include_sprite_envelope=True, sprite_envelope_radius=0.18
    ).result
    m_with_env = out_with_env[0]
    plan_env = json.loads(out_with_env[3])
    assert plan_env["has_sprite_envelope"] is True and abs(plan_env["sprite_envelope_radius"] - 0.18) < 1e-4
    assert float(m_no_env[0, int(0.70 * h), int(0.80 * w)]) < 0.05
    assert float(m_with_env[0, int(0.70 * h), int(0.80 * w)]) > 0.90
    assert float(m_with_env[n_frames - 1, int(0.30 * h), int(0.80 * w)]) > 0.90

    # Optional include_mask (union max) + exclude_mask (0.0 velocity protection even under invert_mask)
    inc = torch.zeros((1, h, w), dtype=torch.float32)
    inc[:, 10:30, 10:30] = 0.85
    exc = torch.zeros((n_frames, h, w), dtype=torch.float32)
    exc[:, h // 2 - 12 : h // 2 + 12, int(w * 0.25) - 12 : int(w * 0.25) + 12] = 1.0

    out_aux = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="combined",
        width=w,
        height=h,
        length=n_frames,
        feather=feather,
        invert_mask=False,
        events_json=comb_json,
        include_mask=inc,
        exclude_mask=exc,
    ).result
    m_aux = out_aux[0]
    plan_aux = json.loads(out_aux[3])
    assert plan_aux["has_include_mask"] is True and plan_aux["has_exclude_mask"] is True
    assert abs(float(m_aux[0, 20, 20]) - 0.85) < 1e-4, "include_mask should union via max"
    assert float(m_aux[12, h // 2, int(w * 0.25)]) == 0.0, "exclude_mask must zero out portal core"

    out_inv_exc = sd_mod.MiniMaxH3SpatialDirector.execute(
        mode="combined",
        width=w,
        height=h,
        length=n_frames,
        feather=feather,
        invert_mask=True,
        events_json=comb_json,
        exclude_mask=exc,
    ).result
    m_inv_exc = out_inv_exc[0]
    assert float(m_inv_exc[12, h // 2, int(w * 0.25)]) == 0.0, "exclude_mask must protect subject even when invert_mask=True"

    imgs = torch.full((n_frames, h, w, 3), 0.30)
    hud_comb = sd_mod.MiniMaxH3DirectorHUD.execute(
        imgs, director_plan=out_aux[3], hud_layout="director_card"
    ).result[0]
    assert hud_comb.shape == (n_frames, h + 144, w, 3)


def main() -> int:
    print("checking MiniMaxH3SpatialDirector, MiniMaxH3DirectorHUD, and spatial workflows")
    case("portal_path default (141 frames) compiles masks, cues, sprite kfs, and plan", _check_portal_path_default)
    case("portal shapes (ellipse, vertical_split) work at length 90", _check_portal_shapes_and_short_length)
    case("point_targets compiles 2D+time targets, 3f cues, and D-pad kfs at 141 and 90", _check_point_targets_both_lengths)
    case("invert_mask equals 1.0 - mask elementwise", _check_invert_mask)
    case("weight, concentric outer halo, bounded portal_start/end/ramp, and HUD dashed ring", _check_weight_halo_and_portal_window)
    case("combined mode (union + deduped 3f cues + HUD) and include_mask / exclude_mask", _check_combined_and_aux_masks)
    case("web/h3_spatial_director.js SOUND_PRESETS and defaults match sound_cues.py", _check_js_sound_presets)
    case("edit_guidance(same_prompt=True) + MiniMaxH3Steer(edit_pass='edited_only') builder", _check_same_prompt_builder)
    case("refuses off-grid lengths, malformed JSON, unknown cues/shapes, and invalid spans", _check_refusals)
    case("MiniMaxH3DirectorHUD renders all layouts, even-clamps odd dims, and draws baseline deltas", _check_hud_layouts)
    case("shipped t2v, world-targets, r2v, and i2v spatial workflows + dual VHS are wired properly", _check_shipped_workflows)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
