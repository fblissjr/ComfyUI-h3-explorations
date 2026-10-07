"""Two of ComfyUI's departures from Meta's SAM 3.1 code, corrected as ComfyUI patches on a loaded model and text encoder.

ComfyUI's SAM 3.1 (`comfy/ldm/sam3`, `comfy_extras/nodes_sam3.py`) is a port. Run beside Meta's code on equal inputs it
computes the same thing from the trunk on, and departs in two places before the model
(`bench/results/2026-10-07_sam3_core_against_meta.md`; `docs/research/masking/2026-10-07_mryolk.md` walks through both):

1. **The image range.** ComfyUI's SAM nodes resize a frame and hand it to the trunk in 0 to 1. Meta's image processor
   and its list-of-images route hand the trunk -1 to 1, and the patch embedding has no bias, so nothing absorbs it.
2. **The text encoder's activation.** ComfyUI's config runs `quick_gelu`; Meta built the encoder with exact GELU.

**One node, and not a loader.** It takes the MODEL and CLIP any checkpoint loader returned (ComfyUI's own, or another
pack's) and returns clones of them carrying two object patches. Nothing is loaded, cached, moved to a device or
accounted for here: ComfyUI's patcher applies the patches when it loads the model for a call and puts the stock
attributes back when it unloads it, so the pair that was wired in stays stock, ComfyUI's SAM nodes take the corrected
pair like any other, and its memory management owns both. Wired twice in a row it is a no-op: what is attached is read
off the patchers themselves (`corrections`), never guessed from a frame's values.

**What it does not correct**, so "corrected" is not read as "the same as Meta's": the resize (ComfyUI's detect node
shrinks without smoothing), the detection rule (ComfyUI keeps on the class score alone, Meta on class times presence,
with overlapping detections removed), the detect node's refinement passes, the hole-fill value, the pointer token, the
rules between tracked objects and the order of threshold and resize on output.

**The range patch, and why it is shaped as it is.** It replaces the trunk's `patch_embed.forward`, the one door every
image passes: the detect node, its refinement crops, the track node and a direct trunk call. It clamps to 0..1 first,
because ComfyUI's own bicubic resize overshoots a little and Meta's routes saturate, then maps to -1..1. It calls the
module's class `forward` on the module it is attached to, never a bound method captured earlier, and it is bound again
for every clone (`CallbacksMP.ON_CLONE`): ComfyUI can give a clone a second instance of the model, and a patch holding
the first instance's layer would then compute with the wrong one.

**A stock model and its corrected clone in one graph** share one model instance, and ComfyUI's
`ModelPatcher.clone_has_same_weights` does not compare object patches. On the plain patcher the two were measured to
switch correctly in both directions (2026-10-07, a CPU probe through ComfyUI's detect node). On the server's dynamic
patcher that is this node's acceptance test, run as a graph on the queue, and the node is not registered until it
passes.

**It reads what is there and never corrects twice.** Neither correction is tied to a ComfyUI version. The activation
is replaced only in MLPs that run the shipped one, so an encoder ComfyUI already builds with exact GELU is left alone and
the report says "already exact". The range patch looks at what arrives: a frame with values far below zero is already
in the trained range (ComfyUI's nodes changed, or something upstream in the graph mapped it), so from then on the patch
passes every frame through untouched, and `corrections` says so to whoever asks after frames have gone through (the
node's own report is written before any have). That test is best effort before the first such frame: a
bright frame that was already mapped has nothing low enough to give it away. `bench/check_sam31_corrections_on_card.py`
reads the stock pair's first layer, so a ComfyUI that starts mapping the range turns that check red.

**Offload and reload.** The patches live on the patcher (`object_patches`), not on the model: ComfyUI puts the stock
attributes back when it unloads the model and applies the patcher's again on every load, the plain class in
`patch_model` and the dynamic class in `partially_load`, so an offload does not lose them. That is read in ComfyUI's
code and is what the on-card check and the on-queue acceptance exercise. The pack's attention nodes attach through
`model_options` instead because the sampler passes those options into the transformer on every call; ComfyUI's SAM
nodes call the model directly and read no options, so for SAM an object patch is the one hook ComfyUI's own code path
honours.
"""
from __future__ import annotations

import logging

import torch.nn.functional as F
from comfy_api.latest import io

logger = logging.getLogger(__name__)

#: read: the path of the trunk's patch embedding under ComfyUI's SAM 3 model (`comfy/ldm/sam3/sam.py`, `ViTDet.patch_embed`;
#: `comfy/ldm/sam3/detector.py` builds `detector.backbone.vision_backbone.trunk`). `bench/check_sam31_corrections.py` reads it
#: off core's classes so a rename there is red here.
PATCH_EMBED = "diffusion_model.detector.backbone.vision_backbone.trunk.patch_embed"
RANGE_PATCH = PATCH_EMBED + ".forward"
#: reasoned: ComfyUI's bicubic resize was measured to undershoot 0 by about a tenth on real frames
#: (`bench/results/2026-10-07_sam3_core_against_meta.md`'s runs) and cannot undershoot a 0..1 picture by a half; a frame
#: mapped to -1..1 has values down to -1. A half is clear of the first and catches a mapped frame with anything darker
#: than a quarter in it.
ALREADY_MAPPED_BELOW = -0.5
#: read: the activation ComfyUI's SAM 3 text encoder is configured with (`comfy/text_encoders/sam3_clip.py`).
SHIPPED_ACTIVATION = "quick_gelu"
NOT_CORRECTED = ("the resize", "the detection rule (class score alone, no overlap removal)", "the detect node's refinement passes",
                 "the hole-fill value", "the pointer token", "the rules between tracked objects", "threshold before resize on output")


class NotSAM3(ValueError):
    """The loaded checkpoint has no SAM 3 trunk or text encoder where this node attaches."""


class CorrectedRange:
    """`forward` for the trunk's patch embedding: the frame clamped to 0..1, mapped to -1..1, then the stock forward."""

    def __init__(self, module, seen: dict | None = None):
        self.module = module
        self.stock = type(module).forward      # the class's, so a forward patched on the instance is never wrapped twice
        # what the patch has seen, shared by every clone's copy of it (a clone gets its own `CorrectedRange`, bound to
        # its own module): `arrives_mapped` is set by the first frame that is clearly in -1..1 already; then nothing is mapped
        self.seen = {"arrives_mapped": False} if seen is None else seen

    @property
    def arrives_mapped(self) -> bool:
        return bool(self.seen["arrives_mapped"])

    def __call__(self, x, *args, **kwargs):
        if not self.arrives_mapped and float(x.amin()) < ALREADY_MAPPED_BELOW:
            self.seen["arrives_mapped"] = True
            logger.warning("[h3] SAM 3.1 Corrections: a frame reached the model already in -1..1 (lowest value %.2f), so the "
                           "image range is left as it arrives from here on. ComfyUI's SAM nodes may map it themselves now, or "
                           "something upstream in the graph does.", float(x.amin()))
        if self.arrives_mapped:
            return self.stock(self.module, x, *args, **kwargs)
        return self.stock(self.module, x.clamp(0.0, 1.0) * 2.0 - 1.0, *args, **kwargs)


def _attach_range(patcher, seen: dict | None = None) -> None:
    import comfy.utils
    try:
        module = comfy.utils.get_attr(patcher.model, PATCH_EMBED)
    except AttributeError as exc:
        raise NotSAM3(f"SAM 3.1 Corrections: this checkpoint's model has no `{PATCH_EMBED}`; it is not a SAM 3 checkpoint.") from exc
    patcher.add_object_patch(RANGE_PATCH, CorrectedRange(module, seen))


def _rebind_range(parent, clone) -> None:
    """`CallbacksMP.ON_CLONE`: bind the range patch to the clone's own model instance, which need not be the parent's."""
    patch = clone.object_patches.get(RANGE_PATCH)
    if isinstance(patch, CorrectedRange):
        _attach_range(clone, patch.seen)


def text_mlps(clip, shipped_only: bool = True) -> list[str]:
    """The names, under the CLIP patcher's model, of the text encoder's MLPs: those BUILT with the shipped activation, or all.

    "Built with" is read under any patch: the encoder is one module shared by every clone, and while a corrected
    clone is loaded its patch is what `mlp.activation` shows. The activation the module was built with is then in the
    patchers' shared backup, which is read first.
    """
    import comfy.clip_model
    shipped = comfy.clip_model.ACTIVATIONS[SHIPPED_ACTIVATION]
    backup = clip.patcher.object_patches_backup
    return [name for name, m in clip.patcher.model.named_modules()
            if isinstance(m, comfy.clip_model.CLIPMLP) and (backup.get(name + ".activation", m.activation) is shipped or not shipped_only)]


def attach(model, clip, image_range: bool = True, text_activation: bool = True) -> None:
    """Attach the chosen corrections to a SAM 3 pair's patchers, in place. Idempotent: a pair that has one keeps it."""
    from comfy.patcher_extension import CallbacksMP
    if image_range and not corrections(model, None)["image_range"]:
        _attach_range(model)
        model.add_callback_with_key(CallbacksMP.ON_CLONE, "h3_sam31_range", _rebind_range)
    if text_activation and clip is not None:
        if not text_mlps(clip, shipped_only=False):
            raise NotSAM3("SAM 3.1 Corrections: this checkpoint's text encoder has none of ComfyUI's CLIP MLPs; it is not "
                          "ComfyUI's SAM 3 text encoder.")
        for name in text_mlps(clip):      # none when ComfyUI already builds the encoder with exact GELU: nothing to do
            clip.patcher.add_object_patch(name + ".activation", F.gelu)


def corrections(model, clip) -> dict:
    """What the patchers of a SAM 3 pair will apply, read off them.

    `image_range`: the range patch is attached. `frames_arrive_mapped`: it is attached and has seen frames that were in
    -1..1 already, so it maps nothing. `text_activation_layers`: MLPs it switches to exact GELU.
    `text_activation_already_exact`: MLPs the encoder already runs with exact GELU, as built.
    """
    out = {"image_range": False, "frames_arrive_mapped": False, "text_activation_layers": 0, "text_activation_already_exact": 0}
    if model is not None:
        patch = model.object_patches.get(RANGE_PATCH)
        out["image_range"] = isinstance(patch, CorrectedRange)
        out["frames_arrive_mapped"] = bool(out["image_range"] and patch.arrives_mapped)
    if clip is not None:
        out["text_activation_layers"] = sum(1 for k, v in clip.patcher.object_patches.items() if k.endswith(".activation") and v is F.gelu)
        out["text_activation_already_exact"] = len(text_mlps(clip, shipped_only=False)) - len(text_mlps(clip))
    return out


def report(model, clip) -> str:
    """What the pair will do, in the words a user reads."""
    got = corrections(model, clip)
    lines = ["SAM 3.1, as loaded by the checkpoint loader wired in:"]
    lines.append("image range: frames arrive already in -1..1, so nothing is mapped (ComfyUI's nodes or the graph map them)" if got["frames_arrive_mapped"]
                 else "image range: corrected at the model's first layer (clamped to 0..1, mapped to -1..1)" if got["image_range"]
                 else "image range: NOT corrected (the model gets 0..1, as ComfyUI's SAM nodes hand it over)")
    lines.append(f"text encoder activation: corrected to exact GELU in {got['text_activation_layers']} layers" if got["text_activation_layers"]
                 else f"text encoder activation: already exact GELU as ComfyUI builds it ({got['text_activation_already_exact']} layers); nothing to correct"
                 if got["text_activation_already_exact"]
                 else f"text encoder activation: NOT corrected ({SHIPPED_ACTIVATION}, ComfyUI's setting)")
    lines.append("left as ComfyUI computes them: " + "; ".join(NOT_CORRECTED))
    return "\n".join(lines)


def corrected(model, clip, image_range: bool = True, text_activation: bool = True):
    """Clones of a SAM 3 pair with the chosen corrections attached; the pair handed in is not touched. Returns (model, clip)."""
    model, clip = model.clone(), clip.clone()
    attach(model, clip, image_range=image_range, text_activation=text_activation)
    return model, clip


class MiniMaxH3SAM31Corrections(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SAM31Corrections",
            display_name="MiniMax H3 SAM 3.1 Corrections",
            category="model/latent/minimax",
            description=(
                "Corrects two things ComfyUI's SAM 3.1 does differently from Meta's code: the range of the image the "
                "model is given, and the text encoder's activation. Wire a SAM 3.1 checkpoint's model and text encoder "
                "in, and use the outputs wherever a SAM 3 node takes them. The report says what is corrected."),
            inputs=[
                io.Model.Input("segmenter", tooltip="The SAM 3.1 checkpoint's model, from any checkpoint loader."),
                io.Clip.Input("segmenter_clip", tooltip="The SAM 3.1 checkpoint's text encoder, from the same loader."),
                io.Boolean.Input("correct_image_range", default=True,
                                 tooltip=("Give the model each frame in the range it was trained on. Off: frames reach "
                                          "it as ComfyUI's SAM nodes hand them over.")),
                io.Boolean.Input("correct_text_activation", default=True,
                                 tooltip=("Run the text encoder with the activation Meta built it with. It changes "
                                          "little for one-word phrases such as `person`. Off: ComfyUI's setting.")),
            ],
            outputs=[io.Model.Output(display_name="segmenter", tooltip="The SAM 3.1 model, for any SAM 3 node."),
                     io.Clip.Output(display_name="segmenter_clip", tooltip="The SAM 3.1 text encoder, for any SAM 3 node."),
                     io.String.Output(display_name="report", tooltip="What is corrected and what is not.")],
        )

    @classmethod
    def execute(cls, segmenter, segmenter_clip, correct_image_range=True, correct_text_activation=True) -> io.NodeOutput:
        model, clip = corrected(segmenter, segmenter_clip, image_range=bool(correct_image_range),
                                text_activation=bool(correct_text_activation))
        text = report(model, clip)
        logger.info("[h3] MiniMaxH3SAM31Corrections: %s", text.replace("\n", "; "))
        return io.NodeOutput(model, clip, text)
