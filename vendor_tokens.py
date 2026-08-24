"""Compatibility tombstone for the retired H3 tokenizer workaround.

ComfyUI commit ``924743af`` registers MiniMax H3's seven extra special tokens
inside ``MiniMaxH3Tokenizer``. That native owner reaches every H3 consumer,
including ComfyUI's own conditioning nodes, so this custom-node pack no longer
patches or replaces tokenizers.

The deprecated node ID and module-level function remain as inert pass-throughs
so saved graphs and third-party Python imports do not fail merely because the
workaround was retired. New code must rely on a ComfyUI version containing the
native fix.
"""

from __future__ import annotations

from comfy_api.latest import io


class MiniMaxH3VendorTokens(io.ComfyNode):
    """Deprecated no-op retained only for saved-graph compatibility."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VendorTokens",
            display_name="MiniMax H3 Vendor Special Tokens (retired)",
            category="MiniMaxH3/experimental",
            is_deprecated=True,
            description=(
                "RETIRED NO-OP. ComfyUI commit 924743af moved the seven H3 "
                "special tokens into MiniMaxH3Tokenizer, their native owner. "
                "This node remains registered only so existing graphs load; "
                "remove it from new or edited graphs."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Boolean.Input(
                    "strict", default=True,
                    tooltip=(
                        "Legacy ignored input retained so saved widget values "
                        "keep their positions."
                    ),
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, strict=True, **kwargs):
        return f"vendor_tokens/retired/{strict}"

    @classmethod
    def execute(cls, clip, strict=True) -> io.NodeOutput:
        return io.NodeOutput(clip)


def clip_with_vendor_tokens(clip, strict: bool = True):
    """Deprecated compatibility import; returns ``clip`` unchanged."""
    return clip
