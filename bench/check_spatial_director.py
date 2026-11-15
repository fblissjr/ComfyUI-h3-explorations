#!/usr/bin/env python3
"""Offline checks for MiniMaxH3SpatialDirector, MiniMaxH3DirectorHUD, and workflows."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from _lib import bootstrap, case, finish, refused  # noqa: E402

bootstrap(cpu=True)

import torch  # noqa: E402

_PKG = f"custom_nodes.{_REPO.name}"
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


def _check_shipped_workflows() -> None:
    portal_path = _REPO / "workflows" / "h3_spatial_portal_pika_api.json"
    targets_path = _REPO / "workflows" / "h3_world_targets_genie_pika_api.json"
    assert portal_path.exists() and targets_path.exists()

    portal_graph = json.loads(portal_path.read_text(encoding="utf-8"))
    sd_nodes = _nodes_by_class(portal_graph, "MiniMaxH3SpatialDirector")
    hud_nodes = _nodes_by_class(portal_graph, "MiniMaxH3DirectorHUD")
    eg_nodes = _nodes_by_class(portal_graph, "MiniMaxH3EditGuider")
    vhs_nodes = _nodes_by_class(portal_graph, "VHS_VideoCombine")
    assert len(sd_nodes) == 1 and len(hud_nodes) == 1 and len(eg_nodes) == 1 and len(vhs_nodes) == 1
    sd_id, sd_node = sd_nodes[0]
    hud_id, hud_node = hud_nodes[0]
    _, eg_node = eg_nodes[0]
    _, vhs_node = vhs_nodes[0]
    assert sd_node["inputs"]["mode"] == "portal_path"
    assert eg_node["inputs"]["streams"] == "independent"
    assert eg_node["inputs"]["mask"] == [sd_id, 0]
    assert hud_node["inputs"]["director_plan"] == [sd_id, 3]
    assert vhs_node["inputs"]["images"] == [hud_id, 0]

    targets_graph = json.loads(targets_path.read_text(encoding="utf-8"))
    sd_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3SpatialDirector")
    hud_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3DirectorHUD")
    eg_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3EditGuider")
    sc_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3SoundCues")
    st_nodes_t = _nodes_by_class(targets_graph, "MiniMaxH3AppendSpriteTrajectory")
    vhs_nodes_t = _nodes_by_class(targets_graph, "VHS_VideoCombine")
    assert (
        len(sd_nodes_t) == 1
        and len(hud_nodes_t) == 1
        and len(eg_nodes_t) == 1
        and len(sc_nodes_t) == 1
        and len(st_nodes_t) == 1
        and len(vhs_nodes_t) == 1
    )
    sd_tid, sd_tnode = sd_nodes_t[0]
    hud_tid, hud_tnode = hud_nodes_t[0]
    assert sd_tnode["inputs"]["mode"] == "point_targets"
    assert eg_nodes_t[0][1]["inputs"]["streams"] == "video_only"
    assert eg_nodes_t[0][1]["inputs"]["mask"] == [sd_tid, 0]
    assert sc_nodes_t[0][1]["inputs"]["cues"] == [sd_tid, 1]
    assert st_nodes_t[0][1]["inputs"]["keyframes"] == [sd_tid, 2]
    assert hud_tnode["inputs"]["director_plan"] == [sd_tid, 3]
    assert vhs_nodes_t[0][1]["inputs"]["images"] == [hud_tid, 0]


def main() -> int:
    print("checking MiniMaxH3SpatialDirector, MiniMaxH3DirectorHUD, and spatial workflows")
    case("portal_path default (141 frames) compiles masks, cues, sprite kfs, and plan", _check_portal_path_default)
    case("portal shapes (ellipse, vertical_split) work at length 90", _check_portal_shapes_and_short_length)
    case("point_targets compiles 2D+time targets, 3f cues, and D-pad kfs at 141 and 90", _check_point_targets_both_lengths)
    case("invert_mask equals 1.0 - mask elementwise", _check_invert_mask)
    case("refuses off-grid lengths, malformed JSON, unknown cues/shapes, and invalid spans", _check_refusals)
    case("MiniMaxH3DirectorHUD renders all layouts, even-clamps odd dims, and draws baseline deltas", _check_hud_layouts)
    case("shipped spatial portal and genie world-targets workflows are wired properly", _check_shipped_workflows)
    return finish()


if __name__ == "__main__":
    raise SystemExit(main())
