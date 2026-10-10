"""A tracked subject's mask as one box a frame, for a node that takes boxes and not masks.

`MiniMaxH3SubjectTrack` hands back a mask. A body-pose node takes boxes: a list for each frame, each box a dict
of `x`, `y`, `width` and `height`, and an empty list for a frame with nobody in it. This pack's own is
`MiniMaxH3BodyPose` (`body_pose.py::frame_box_rows`), which is the one our graphs wire since 2026-10-10; the
form is also what ComfyUI's SAM 3D Body prediction reads
(`comfy_extras/nodes_sam3d_body.py::_per_frame_bboxes_from_detections`). Nothing on a stock server makes one
from the other. `MiniMaxH3SubjectBoxes` does: the box of the mask on each frame, widened by `margin`, and
no box on a frame the mask is empty on.

Why it matters, from a read of Meta's code and a comparison of core's port with it on Meta's own sample
images (`docs/wiki/meta_perception_models.md`, 2026-10-10; session mrfrog). Given no box, that prediction takes
the whole frame as one person's crop, which with two people in the frame is not defined. Core's port matches
Meta's code once both are given the same crop; as shipped the two differ by how the crop is sampled, least when
the person's crop is about the model's own input size and more when a large crop is shrunk (a whole-frame box)
or a small person is enlarged. So a box on the person is the right input, and it does not close the gap for a
person who is small in the frame. And with a box, a frame the subject is not in gets no body at all, where the
whole-frame fallback draws one from whatever is there.

No torch model and no ComfyUI state: `frame_boxes` is the whole of it, and `bench/check_video_mask.py` holds it.
"""
from __future__ import annotations

import torch
from comfy_api.latest import io

from .sapiens2_parts import mask_boxes

#: The widest `margin` the widget takes. Reasoned: the Masked Source's own widest margin.
MARGIN_MAX = 512


def frame_boxes(mask: torch.Tensor, margin: int = 0) -> list[list[dict]]:
    """For each frame of a [N, H, W] mask, a list holding its box as `{x, y, width, height}`, or an empty list.

    The box is the mask's own (`sapiens2_parts.mask_boxes`: the far side exclusive), widened by `margin` pixels
    on every side and held inside the frame. A frame whose mask is empty gets no box, which a consumer reads
    as nobody there.
    """
    if mask.ndim == 4 and int(mask.shape[-1]) == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must be [N, H, W]; got {tuple(mask.shape)}")
    height, width = int(mask.shape[1]), int(mask.shape[2])
    margin = int(margin)
    out: list[list[dict]] = []
    for x0, y0, x1, y1 in mask_boxes(mask).tolist():
        if x0 < 0:
            out.append([])
            continue
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(width, x1 + margin), min(height, y1 + margin)
        out.append([{"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}])
    return out


class MiniMaxH3SubjectBoxes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SubjectBoxes",
            display_name="MiniMax H3 Subject Boxes (a tracked mask as one box a frame)",
            category="model/latent/minimax",
            description=("Turns a subject's mask into the per-frame boxes a body-pose node takes: one box on "
                         "each frame the subject is in, none on a frame they are not. Wire the Subject Track's "
                         "mask in and the boxes into MiniMax H3 Body Pose, one of these per person."),
            inputs=[
                io.Mask.Input("mask", tooltip=("One mask per frame, 1 on the subject: the Subject Track's `mask`. "
                                               "A frame where it is empty gets no box.")),
                io.Int.Input("margin", default=0, min=0, max=MARGIN_MAX,
                             tooltip=("How far the box is widened past the mask on every side, in pixels of the "
                                      "frame. Raise it when a hand or a raised arm is cut off by a mask that is "
                                      "tight; a body-pose node pads its own crop as well.")),
            ],
            outputs=[
                io.BoundingBox.Output(display_name="boxes",
                                      tooltip="For each frame, a list with the subject's box, or an empty list."),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, mask, margin=0) -> io.NodeOutput:
        boxes = frame_boxes(mask, margin)
        have = [i for i, b in enumerate(boxes) if b]
        text = f"{len(have)} of {len(boxes)} frames have a box"
        if have:
            widths = sorted(boxes[i][0]["width"] for i in have)
            heights = sorted(boxes[i][0]["height"] for i in have)
            text += (f"; width {widths[0]} to {widths[-1]} px, height {heights[0]} to {heights[-1]} px, "
                     f"margin {int(margin)} px")
        if len(have) < len(boxes):
            gone = [i for i, b in enumerate(boxes) if not b]
            text += f"; no box from frame {gone[0]} to {gone[-1]} on {len(gone)} frame(s): the mask is empty there"
        return io.NodeOutput(boxes, text)
