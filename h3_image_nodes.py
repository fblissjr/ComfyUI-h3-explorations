"""MiniMax-H3-Image: pick the size of the picture by megapixels and shape (`MiniMaxH3ImageSize`).

Core's `MiniMax H3 Image Conditioning` takes a width and a height in pixels.
Left at 0 it uses the first picture's own size, which for most pictures is
below the size the model card says the model works well at; set by hand it
means working out a size of the right area at the right shape with both sides
a multiple of 32. This node does that arithmetic: it takes megapixels and a
shape and gives the width and height to wire into the conditioning node.

The rule is the vendor's (`size_for` in the release's `examples/context_ir.py`,
held as `workflows/h3_config.py::h3_image_size`). The shape is one of the ten
ratios the vendor's prompt expander may choose, or the shape of a picture
wired in, which is what an edit that keeps its picture's framing wants. The
picture itself is not resized here or anywhere before the conditioning node:
the model reads a reference at its own size.
"""

from __future__ import annotations

import sys
from pathlib import Path

_COMFY_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "comfy_api").is_dir()),
    None,
)
if _COMFY_ROOT is not None and str(_COMFY_ROOT) not in sys.path:
    sys.path.append(str(_COMFY_ROOT))

from comfy_api.latest import io

_HERE = Path(__file__).resolve().parent
if str(_HERE / "workflows") not in sys.path:
    sys.path.insert(0, str(_HERE / "workflows"))
from h3_config import (  # noqa: E402
    H3_IMAGE_ASPECT_OF_PICTURE,
    H3_IMAGE_ASPECTS,
    H3_IMAGE_MEGAPIXELS,
    H3_IMAGE_MEGAPIXELS_RANGE,
    h3_image_size,
)


def resolve_size(megapixels: float, shape: str, picture_size: tuple[int, int] | None) -> tuple[int, int]:
    """`(width, height)` for a shape choice; `picture_size` is `(width, height)` of the wired picture, if any."""
    if shape == H3_IMAGE_ASPECT_OF_PICTURE:
        if picture_size is None:
            raise ValueError(
                f"shape is '{H3_IMAGE_ASPECT_OF_PICTURE}' but no picture is wired into this node. "
                "Wire the picture being edited into 'picture', or choose a ratio.")
        x, y = picture_size
    else:
        x, y = (int(v) for v in shape.split(":"))
    return h3_image_size(megapixels, x, y)


class MiniMaxH3ImageSize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        low, high = H3_IMAGE_MEGAPIXELS_RANGE
        return io.Schema(
            node_id="MiniMaxH3ImageSize",
            display_name="MiniMax H3 Image Size",
            category="model/conditioning/minimax",
            description=(
                "Width and height for MiniMax H3 Image from megapixels and a shape. "
                "Wire them into MiniMax H3 Image Conditioning."
            ),
            inputs=[
                io.Float.Input(
                    "megapixels", default=H3_IMAGE_MEGAPIXELS, min=1.0, max=24.0, step=0.5,
                    tooltip=(
                        f"How large the picture is. The model is at its best near {H3_IMAGE_MEGAPIXELS:g} "
                        f"and works well from about {low:g} to {high:g}; below that the picture loses detail "
                        "and can fall apart. Smaller is faster and takes less memory.")),
                io.Combo.Input(
                    "shape", options=[*H3_IMAGE_ASPECTS, H3_IMAGE_ASPECT_OF_PICTURE], default="1:1",
                    tooltip=(
                        "Width to height. Choose a ratio, or 'same as the picture' to keep the shape of "
                        "the picture wired in below, which is what an edit of that picture usually wants.")),
                io.Image.Input(
                    "picture", optional=True,
                    tooltip=(
                        "Only its shape is read, and only when shape is 'same as the picture'. "
                        "The picture is not changed.")),
            ],
            outputs=[
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, megapixels, shape, picture=None) -> io.NodeOutput:
        picture_size = None if picture is None else (int(picture.shape[2]), int(picture.shape[1]))
        width, height = resolve_size(megapixels, shape, picture_size)
        return io.NodeOutput(width, height)
