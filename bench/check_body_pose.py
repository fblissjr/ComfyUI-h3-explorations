#!/usr/bin/env python3
"""Hold `body_pose.py` to Meta's SAM 3D Body code: the crop bit for bit, the bodies within Meta's own precision, and no ComfyUI SAM node.

`body_pose.py` replaces ComfyUI's three SAM 3D Body nodes in our graphs. Its
claim is that the model is given the pixels Meta's code would give it, and
that everything a preflight reads from it is true. A wrong crop still draws a
body, a wrong table still parses: each is held here.

The reference is `bench/fixtures/sam3d_body_meta_reference.json`, written by
`bench/compare_sam3d_body_core_against_meta.py fixture` from Meta's own code
run on two public samples (`bench/results/2026-10-10_sam3d_body_core_against_meta.md`):
per box, the hash of Meta's body crop, Meta's 2D keypoints, which hands' decoders
Meta used, and how far Meta's two precisions are from each other (the floor).
Nothing of Meta's is imported or run here.

Without weights, on the CPU:

  crop_is_metas_bit_for_bit      our body crop of every box hashes to Meta's.
                                 The control is ComfyUI's own crop of the same
                                 box, which must NOT: if it ever does, this case
                                 has stopped telling the two apart.
  boxes_are_read_as_written      the boxes input in each form the boxes node and
                                 a person can give it; a box that is the whole
                                 frame is named so; a frame count that does not
                                 match is refused.
  camera_is_one_of_two           the focal length under each camera choice.
  table_says_what_was_predicted  the table from a pose built by hand: every
                                 field a preflight reads, source frame numbers,
                                 keypoints outside the frame counted.
  threshold_is_the_models        the hand constant written into the table is the
                                 one in ComfyUI's model source, and in Meta's
                                 when the checkout is there.
  drawing_scales_with_the_size   a draw at another size scales the focal length
                                 and nothing else.
  no_comfy_sam_node_is_called    the module's source names no `SAM3DBody_` node
                                 and does not import ComfyUI's node file.
  opencv_missing_is_said         without OpenCV the crop raises with the
                                 package's name.

With the DINOv3 file on disk (skipped without it, exit 2; about two minutes
of CPU):

  bodies_are_metas_within_floor  our keypoints against Meta's on every box, the
                                 mean distance under that box's floor.
  hands_decide_as_metas          per hand, the decoder used exactly where Meta
                                 used it, and the crop side Meta had.
  nobody_is_nobody               a frame with an empty box list has no person,
                                 no table row, and draws black.
  mesh_is_where_the_camera_puts_it  every vertex projected by the predicted camera
                                 lands on the drawn silhouette, and the two
                                 bounding boxes agree to a pixel.

The second sample is one frame of a video, decoded here with OpenCV. If this
machine's decoder gives other pixels than the fixture's hash, that image's
cases are skipped and say so: the comparison would be of two different
pictures.

    CUDA_VISIBLE_DEVICES="" <comfy venv python> bench/check_body_pose.py

**`--card`** runs the four weights cases on the card instead, in the precision
the server's loader picks there, with ComfyUI's dynamic memory layer set up
as `main.py` sets it (`_lib.server_memory_mode`). The cases and the bounds are
the same: the claim is that the server's path is also inside Meta's own floor.
It is a second process holding memory beside the server, so ask whoever holds
the card first (`AGENTS.md`, "The server process is the resource"); a sweep
never passes it.

    <comfy venv python> bench/check_body_pose.py --card
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _lib import COMFY, REPO, bootstrap, card_visible, case, finish, in_sweep, needs, server_memory_mode, skip  # noqa: E402

ON_CARD = "--card" in sys.argv[1:]
if ON_CARD:
    needs("to be run by hand: --card holds memory on the card beside the server", not in_sweep())
    needs("a CUDA device for --card", card_visible())
bootstrap(cpu=not ON_CARD)
if ON_CARD:
    server_memory_mode()

FIXTURE = HERE / "fixtures" / "sam3d_body_meta_reference.json"
SAMPLES = {"dancing": REPO / "coderef" / "sam-3d-body" / "notebook" / "images" / "dancing.jpg",
           "office_frame60": REPO / "coderef" / "sam-audio" / "examples" / "assets" / "office.mp4"}
OFFICE_FRAME = 60
DINOV3_FILE = "sam_3d_body_dinov3.safetensors"
# The share of projected vertices that must sit on the drawn silhouette. measured 2026-10-10: 1.0 on four boxes of
# the two samples; the bound leaves room for one vertex on an edge.
ON_SILHOUETTE = 0.999

_state: dict = {}


def _load(name: str):
    """A root module of the pack as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location(f"_h3pack.{name}", REPO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_h3pack.{name}"] = module
    spec.loader.exec_module(module)
    return module


_load("sam3d_body_vith")
bp = _load("body_pose")


def reference() -> dict:
    if "ref" not in _state:
        _state["ref"] = json.loads(FIXTURE.read_text())
    return _state["ref"]


def sample(label: str) -> tuple[np.ndarray | None, str]:
    """(the sample's pixels as [H, W, 3] uint8, "") or (None, why it cannot be compared here)."""
    if label not in _state:
        path = SAMPLES[label]
        if not path.is_file():
            return None, f"needs {path.relative_to(REPO)}"
        if path.suffix == ".mp4":
            import cv2
            capture = cv2.VideoCapture(str(path))
            capture.set(cv2.CAP_PROP_POS_FRAMES, OFFICE_FRAME)
            ok, frame = capture.read()
            capture.release()
            assert ok, f"frame {OFFICE_FRAME} of {path.name} did not decode"
            pixels = np.ascontiguousarray(frame[:, :, ::-1])
        else:
            from PIL import Image
            pixels = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
        _state[label] = pixels
    pixels = _state[label]
    want = next(im for im in reference()["images"] if im["label"] == label)["pixels_sha256"]
    if hashlib.sha256(pixels.tobytes()).hexdigest() != want:
        return None, f"{label} decodes to other pixels here than the fixture's; nothing to compare"
    return pixels, ""


def pixels_of(label: str) -> np.ndarray:
    pixels, why = sample(label)
    if pixels is None:
        skip(why)
    return pixels


def _sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- without weights

def crop_is_metas_bit_for_bit():
    from comfy.ldm.sam3d_body.utils import prepare_batch as comfy_prepare
    graded, skipped = 0, []
    for image in reference()["images"]:
        pixels, why = sample(image["label"])
        if pixels is None:  # one image that cannot be compared does not skip the other
            skipped.append(why)
            continue
        for k, person in enumerate(image["people"]):
            box = np.asarray(person["box_xyxy"], dtype=np.float32)
            ours, _, _, _ = bp.crop(pixels, box, bp.BODY_PADDING, (512, 512))
            assert _sha(ours) == person["meta_body_crop_sha256"], (
                f"{image['label']} box {k}: our crop is not Meta's")
            theirs = comfy_prepare(torch.from_numpy(pixels.copy()), torch.tensor([person["box_xyxy"]]),
                                   input_size=(512, 512))["img"][0, 0]
            theirs = (theirs * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
            assert _sha(theirs) != person["meta_body_crop_sha256"], (
                f"{image['label']} box {k}: ComfyUI's own crop now equals Meta's too, so this case no longer "
                "tells the two samplings apart; read why before trusting it")
            graded += 1
    if not graded:
        skip("needs the public samples under coderef/")
    return f"{graded} boxes" + (f"; not graded: {', '.join(skipped)}" if skipped else "")


def boxes_are_read_as_written():
    one = {"x": 10, "y": 20, "width": 30, "height": 40}
    rows = bp.frame_box_rows([[one], [], [one, one]], 3)
    assert rows == [[[10.0, 20.0, 40.0, 60.0]], [], [[10.0, 20.0, 40.0, 60.0]] * 2], rows
    rows = bp.frame_box_rows([[one]], 3)
    assert len(rows) == 3 and all(r == [[10.0, 20.0, 40.0, 60.0]] for r in rows), "one frame's boxes serve every frame"
    assert bp.frame_box_rows([one, one], 2) == [[[10.0, 20.0, 40.0, 60.0]] * 2] * 2, "a flat list is one frame's people"
    assert bp.box_source([0, 0, 200, 100], 100, 200) == "whole frame"
    assert bp.box_source([0, 0, 199, 100], 100, 200) == "given" and bp.box_source([10, 20, 40, 60], 100, 200) == "given"
    for bad, frames in (([[one], [one]], 3), (None, 2)):
        try:
            bp.frame_box_rows(bad, frames)
        except ValueError as exc:
            assert "frame(s)" in str(exc) and str(frames) in str(exc), exc
        else:
            raise AssertionError(f"boxes for the wrong number of frames must be refused: {bad!r} for {frames}")


def camera_is_one_of_two():
    k = bp.intrinsics(300, 400, "image diagonal", 55.0)[0]
    assert abs(float(k[0, 0]) - 500.0) < 1e-4 and float(k[0, 0]) == float(k[1, 1]), k
    assert (float(k[0, 2]), float(k[1, 2])) == (200.0, 150.0), "the principal point is the image's centre"
    k = bp.intrinsics(300, 400, "field of view", 90.0)[0]
    assert abs(float(k[0, 0]) - 150.0) < 1e-3, "a 90 degree vertical angle puts the focal length at half the height"
    try:
        bp.intrinsics(300, 400, "auto", 55.0)
    except ValueError:
        return
    raise AssertionError("a camera that is not one of the two must be refused")


def _hand_built_pose():
    points = np.zeros((70, 2), dtype=np.float32) + 50.0
    points[0] = [-3.0, 10.0]          # left of the frame
    points[1] = [200.0, 10.0]         # at the width: outside
    points[2] = [199.96, 99.96]       # inside, and rounds to the edge
    person = {"bbox": np.array([10, 20, 60, 90], dtype=np.float32), "pred_keypoints_2d": points,
              "focal_length": np.float32(223.6)}
    pose = {"frames": [[person], []], "image_size": (100, 200)}
    extras = [[{"crop_bbox": [0.0, 5.0, 100.0, 105.0],
                "left_hand": {"bbox": [1.0, 2.0, 41.0, 42.0], "crop_side_px": 40.0, "decoder_used": False},
                "right_hand": {"bbox": [1.0, 2.0, 91.0, 92.0], "crop_side_px": 90.0, "decoder_used": True}}], []]
    return pose, extras


def table_says_what_was_predicted():
    pose, extras = _hand_built_pose()
    table = bp.pose_table(pose, extras, camera="field of view", fov_degrees=40.0, hands=True,
                          subject="lead", first_source_frame=604)
    assert table["schema"] == bp.TABLE_SCHEMA and table["image_size"] == {"width": 200, "height": 100}
    assert table["camera"] == {"mode": "field of view", "fov_degrees": 40.0}
    assert table["hand_box_threshold_px"] == bp.HAND_BOX_THRESHOLD_PX and table["first_source_frame"] == 604
    first, second = table["frames"]
    assert (first["frame"], first["source_frame"]) == (0, 604) and (second["frame"], second["source_frame"]) == (1, 605)
    assert second["people"] == [], "a frame with nobody has no row"
    row = first["people"][0]
    assert set(row) == {"person", "subject", "bbox", "crop_bbox", "box_source", "left_hand", "right_hand",
                        "keypoints_2d", "keypoints_outside_frame", "focal_length_px"}, sorted(row)
    assert row["subject"] == "lead" and row["box_source"] == "given" and row["bbox"] == [10.0, 20.0, 60.0, 90.0]
    assert row["keypoints_outside_frame"] == 2, f"two of the three planted points are outside; got {row['keypoints_outside_frame']}"
    assert len(row["keypoints_2d"]) == 70 and row["keypoints_2d"][2] == [200.0, 100.0], "one decimal"
    assert row["left_hand"] == {"bbox": [1.0, 2.0, 41.0, 42.0], "crop_side_px": 40.0, "decoder_used": False}
    assert row["right_hand"]["decoder_used"] is True
    json.loads(json.dumps(table))
    report = bp.table_report(table)
    assert "1 bodies on 1 of 2 frames" in report and "left hand: refined on 0 of 1" in report, report
    assert "crop at or under 64 px on 1" in report and "2 keypoints placed outside" in report, report
    pose["frames"][0][0]["bbox"] = np.array([0, 0, 200, 100], dtype=np.float32)
    off = bp.pose_table(pose, [[{"crop_bbox": [0, 0, 1, 1], "left_hand": None, "right_hand": None}], []],
                        camera="image diagonal", fov_degrees=40.0, hands=False)
    assert off["camera"]["fov_degrees"] is None and off["frames"][0]["people"][0]["left_hand"] is None
    assert off["frames"][0]["people"][0]["box_source"] == "whole frame", "a box that is the frame is named so"
    assert "the whole frame" in bp.table_report(off)


def threshold_is_the_models():
    line = f"hand_box_size_thresh = {bp.HAND_BOX_THRESHOLD_PX}"
    comfy_source = (COMFY / "comfy" / "ldm" / "sam3d_body" / "model" / "model.py").read_text()
    assert line in comfy_source, f"ComfyUI's model no longer holds `{line}`; the table would name a constant the model does not use"
    meta = REPO / "coderef" / "sam-3d-body" / "sam_3d_body" / "models" / "meta_arch" / "sam3d_body.py"
    if meta.is_file():
        assert line in meta.read_text(), f"Meta's code no longer holds `{line}`"
        return "ComfyUI's and Meta's"
    return "ComfyUI's; Meta's checkout is absent"


def drawing_scales_with_the_size():
    pose, _ = _hand_built_pose()
    pose["frames"][0][0]["pred_cam_t"] = np.zeros(3, dtype=np.float32)
    assert bp.scaled_for(pose, 200, 100) is pose, "the source's own size changes nothing"
    half = bp.scaled_for(pose, 100, 50)
    assert half["image_size"] == (50, 100) and abs(float(half["frames"][0][0]["focal_length"]) - 111.8) < 1e-3
    bars = bp.scaled_for(pose, 400, 100)   # twice as wide, the same height: the height limits it
    assert abs(float(bars["frames"][0][0]["focal_length"]) - 223.6) < 1e-3, "a wider canvas adds bars, not zoom"
    assert float(pose["frames"][0][0]["focal_length"]) == np.float32(223.6), "the input is not edited"
    assert bp.render_size(pose, "the source's", 7, 9) == (200, 100) and bp.render_size(pose, "width and height", 7, 9) == (7, 9)


def no_comfy_sam_node_is_called():
    source = (REPO / "body_pose.py").read_text()
    code = "\n".join(line for line in source.split("\n") if not line.lstrip().startswith("#"))
    body = code.split('"""', 2)[2]          # past the module docstring, which may name what it replaces
    assert "SAM3DBody_" not in body, "body_pose.py names a ComfyUI SAM 3D Body node outside its docstring"
    assert "nodes_sam3d_body" not in body, "body_pose.py reaches into ComfyUI's SAM 3D Body node file"
    assert "coderef" not in "\n".join(line for line in body.split("\n") if "import" in line), "an import from coderef"
    for name in ("MiniMaxH3BodyModelLoader", "MiniMaxH3BodyPose", "MiniMaxH3BodyMeshVideo"):
        assert f'node_id="{name}"' in body, f"{name} is gone"


def opencv_missing_is_said():
    kept = sys.modules.get("cv2")
    sys.modules["cv2"] = None  # an import of it now raises ImportError
    try:
        try:
            bp.crop(np.zeros((8, 8, 3), dtype=np.uint8), np.array([0, 0, 8, 8], dtype=np.float32), 1.25, (4, 4))
        except RuntimeError as exc:
            assert "opencv-python-headless" in str(exc), exc
        else:
            raise AssertionError("a crop without OpenCV must raise")
    finally:
        if kept is None:
            sys.modules.pop("cv2", None)
        else:
            sys.modules["cv2"] = kept


# ----------------------------------------------------------------------------- with the weights

def patcher():
    if "patcher" not in _state:
        named = os.environ.get("H3_SAM3D_DINOV3_FILE")
        path = Path(named).expanduser() if named else COMFY / "models" / "detection" / DINOV3_FILE
        if not path.is_file():
            skip(f"needs models/detection/{DINOV3_FILE} under the ComfyUI root or H3_SAM3D_DINOV3_FILE "
                 "(bench/convert_sam3d_body_checkpoint.py writes it)")
        _state["patcher"] = bp.load_model(str(path))
    return _state["patcher"]


def predicted(label: str):
    """(image, pose, extras, table) for one sample under the default camera, predicted once."""
    key = f"predicted:{label}"
    if key not in _state:
        model = patcher()
        image = next(im for im in reference()["images"] if im["label"] == label)
        pixels = pixels_of(label)
        frames = torch.from_numpy(pixels.copy()).float().div(255)[None]
        boxes = [[{"x": b[0], "y": b[1], "width": b[2] - b[0], "height": b[3] - b[1]}
                  for b in (p["box_xyxy"] for p in image["people"])]]
        pose, extras = bp.predict(model, frames, boxes, hands=True, camera="image diagonal")
        table = bp.pose_table(pose, extras, camera="image diagonal", fov_degrees=55.0, hands=True)
        _state[key] = (image, pose, extras, table)
    return _state[key]


def bodies_are_metas_within_floor():
    notes = []
    for label in SAMPLES:
        image, pose, _, _ = predicted(label)
        for k, person in enumerate(image["people"]):
            want = person["cameras"]["default"]
            got = np.asarray(pose["frames"][0][k]["pred_keypoints_2d"], dtype=np.float64)[:, :2]
            mean = float(np.linalg.norm(got - np.asarray(want["keypoints_2d"]), axis=-1).mean())
            floor = want["floor_keypoints_2d_px"]["mean"]
            assert mean <= floor, (f"{label} box {k}: our keypoints are {mean:.3f} px from Meta's on average, over "
                                   f"the {floor:.3f} px Meta's own two precisions differ by")
            focal = float(np.asarray(pose["frames"][0][k]["focal_length"]).reshape(-1)[0])
            assert abs(focal - want["focal_length"]) < 1e-2, f"{label} box {k}: focal length {focal} against {want['focal_length']}"
            notes.append(f"{mean:.3f}/{floor:.3f}")
    return "mean px over floor, per box: " + ", ".join(notes)


def hands_decide_as_metas():
    used = refused = 0
    for label in SAMPLES:
        image, _, _, table = predicted(label)
        for k, person in enumerate(image["people"]):
            want = person["cameras"]["default"]
            row = table["frames"][0]["people"][k]
            for side, name in enumerate(("left_hand", "right_hand")):
                assert row[name]["decoder_used"] == want["hand_decoder_used"][side], (
                    f"{label} box {k} {name}: decoder used {row[name]['decoder_used']}, Meta's code {want['hand_decoder_used'][side]}")
                assert abs(row[name]["crop_side_px"] - want["hand_crop_side_px"][side]) < 0.5, (
                    f"{label} box {k} {name}: crop side {row[name]['crop_side_px']} against Meta's {want['hand_crop_side_px'][side]}")
                if row[name]["crop_side_px"] <= bp.HAND_BOX_THRESHOLD_PX:
                    assert not row[name]["decoder_used"], "a hand crop under the constant was reported as refined"
                used += row[name]["decoder_used"]
                refused += not row[name]["decoder_used"]
        sources = [row["box_source"] for row in table["frames"][0]["people"]]
        assert sources == ["whole frame"] + ["given"] * (len(sources) - 1), f"{label}: {sources}"
    assert used and refused, "the samples must hold hands of both kinds, or the reading is not exercised"
    return f"{used} hands refined, {refused} not, each as Meta's code decided"


def nobody_is_nobody():
    pixels = pixels_of("office_frame60")
    frames = torch.from_numpy(np.stack([pixels, pixels])).float().div(255)
    box = next(im for im in reference()["images"] if im["label"] == "office_frame60")["people"][2]["box_xyxy"]
    boxes = [[], [{"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]}]]
    pose, extras = bp.predict(patcher(), frames, boxes, hands=False)
    assert [len(f) for f in pose["frames"]] == [0, 1], [len(f) for f in pose["frames"]]
    table = bp.pose_table(pose, extras, camera="image diagonal", fov_degrees=55.0, hands=False)
    assert table["frames"][0]["people"] == [] and len(table["frames"][1]["people"]) == 1
    drawn = bp.render(pose, style="silhouette", size="width and height", width=320, height=180)
    assert tuple(drawn.shape) == (2, 180, 320, 3), tuple(drawn.shape)
    assert float(drawn[0].abs().max()) == 0.0, "the frame with no box must draw black"
    assert float(drawn[1].max()) > 0.5, "the frame with a box must draw a body"


def mesh_is_where_the_camera_puts_it():
    notes = []
    for label in SAMPLES:
        _, pose, _, _ = predicted(label)
        height, width = pose["image_size"]
        for k in range(1, len(pose["frames"][0])):     # the boxes on one person; the whole-frame box is left out
            one = dict(pose, frames=[[pose["frames"][0][k]]])
            silhouette = (bp.render(one, style="silhouette")[0, ..., 0] > 0.5).cpu().numpy()
            person = one["frames"][0][0]
            v = np.asarray(person["pred_vertices"], dtype=np.float64) + np.asarray(person["pred_cam_t"], dtype=np.float64)
            focal = float(np.asarray(person["focal_length"]).reshape(-1)[0])
            x, y = focal * v[:, 0] / v[:, 2] + width / 2, focal * v[:, 1] / v[:, 2] + height / 2
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            xi = np.clip(np.rint(x[inside] - 0.5).astype(int), 0, width - 1)
            yi = np.clip(np.rint(y[inside] - 0.5).astype(int), 0, height - 1)
            pad = np.pad(silhouette, 1)
            near = np.zeros_like(silhouette)
            for dy in range(3):
                for dx in range(3):
                    near |= pad[dy:dy + height, dx:dx + width]
            on = float(near[yi, xi].mean())
            assert on >= ON_SILHOUETTE, f"{label} box {k}: only {on:.4f} of the projected vertices are on the drawn silhouette"
            cols, rows = np.nonzero(silhouette.any(0))[0], np.nonzero(silhouette.any(1))[0]
            drawn = [cols.min(), rows.min(), cols.max(), rows.max()]
            projected = [x[inside].min(), y[inside].min(), x[inside].max(), y[inside].max()]
            worst = max(abs(float(a) - float(b)) for a, b in zip(drawn, projected))
            assert worst <= 1.5, f"{label} box {k}: the silhouette's box {drawn} and the projection's {projected}"
            notes.append(f"{on:.4f}")
    return "share of vertices on the silhouette, per box: " + ", ".join(notes)


def where_it_ran():
    import comfy.model_management as mm
    return f"device {mm.get_torch_device()}, backbone {patcher().model.backbone_dtype}"


def main() -> int:
    print(f"bench/check_body_pose.py -- {'the card' if ON_CARD else 'CPU'}\n")
    torch.set_grad_enabled(False)
    without = (crop_is_metas_bit_for_bit, boxes_are_read_as_written, camera_is_one_of_two,
               table_says_what_was_predicted, threshold_is_the_models, drawing_scales_with_the_size,
               no_comfy_sam_node_is_called, opencv_missing_is_said)
    with_weights = (bodies_are_metas_within_floor, hands_decide_as_metas, nobody_is_nobody,
                    mesh_is_where_the_camera_puts_it)
    for fn in ((where_it_ran,) + with_weights) if ON_CARD else (without + with_weights):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
