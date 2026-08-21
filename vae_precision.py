"""Split the H3 video VAE's encode precision from its decode precision.

ComfyUI carries ONE dtype per VAE object. `comfy/sd.py` picks it once
(`vae_dtype`), casts the whole model to it, and both the encode and the decode
paths cast their input to the same thing. On this card that resolves to fp16
unless `--fp32-vae` is passed, and that flag moves both halves together.

The released pipeline does not treat the two halves the same. It keeps the
video VAE resident in fp32 *because the same module also encodes keyframes and
references*, and decodes under fp16 autocast. So the vendor configuration is
not expressible through the flag: `--fp32-vae` buys the encode side by paying
for a decode nobody asked for.

The prices are not symmetric either, which is the whole reason this node is
worth having. Read out of the shipped fp16 checkpoint on 2026-08-21, the
decoder is 93% of the file and the encoder is 7% -- 4.51 GiB against 0.34 GiB.
Promoting the encoder to fp32 costs about a third of a gigabyte. Promoting the
decoder costs four and a half, plus the 2-3x decode time that got `--fp32-vae`
reverted on 2026-08-10 (the reasoning is in `<comfy>/start.sh`, and it still
stands for the decoder).

**Nothing here says fp32 encode looks better.** That is unmeasured. This node
exists so the question can be asked at its own price instead of bundled with a
decode regression, and the arms it enables are a reference render with the
encoder at fp16 against the same render with it at fp32.

How the dtypes are kept consistent. `vae_dtype` becomes the ENCODER's dtype,
because that is what `VAE.encode` casts pixels to. The decoder is left alone,
and `first_stage_model.decode` is wrapped so the latent is cast back to
whatever the decoder actually holds. Both boundaries therefore agree no matter
which combination is selected, including the do-nothing one.

The cast happens on the loaded module, which the VAE wrapper shares. That is
deliberate: a graph has one H3 video VAE, and the decode wrapper normalises the
latent at entry, so a second consumer of the same object still decodes
correctly. It is idempotent -- the marker below means re-running does not
re-wrap.
"""

from __future__ import annotations

import copy
import logging

import torch
from comfy_api.latest import io

logger = logging.getLogger(__name__)

_WRAPPED = "_h3_dtype_wrappers"

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
# The video VAE declares working_dtypes [float16, float32] in comfy/sd.py, so
# bf16 is offered for the encoder only on the same terms as any other
# unsupported-but-runnable choice: it is not a dtype the release uses.
CHOICES = ["unchanged", "fp32", "fp16", "bf16"]


def _module_dtype(module):
    for p in module.parameters():
        return p.dtype
    return None


def _wrap_boundaries(first_stage_model):
    """Cast at both module boundaries to whatever that half actually holds.

    BOTH halves are wrapped, not just the one being promoted, and that is the
    point rather than symmetry for its own sake. The wrapper object this node
    returns is a copy; the module underneath is shared. So a graph can wire the
    ORIGINAL wrapper -- still advertising its old `vae_dtype` -- into a plain
    encode or decode alongside ours. Normalising at the module entry means that
    path stays correct instead of hitting a dtype mismatch that depends on
    which node ran first.

    Idempotent: the marker means a second pass does not wrap a wrapper.
    """
    if getattr(first_stage_model, _WRAPPED, False):
        return
    original_decode = first_stage_model.decode
    original_encode = first_stage_model.encode

    def decode(z, *args, **kwargs):
        target = _module_dtype(first_stage_model.decoder)
        if target is not None and torch.is_tensor(z) and z.dtype != target:
            z = z.to(target)
        return original_decode(z, *args, **kwargs)

    def encode(x, *args, **kwargs):
        target = _module_dtype(first_stage_model.encoder)
        if target is not None and torch.is_tensor(x) and x.dtype != target:
            x = x.to(target)
        return original_encode(x, *args, **kwargs)

    first_stage_model.decode = decode
    first_stage_model.encode = encode
    setattr(first_stage_model, _WRAPPED, True)


class MiniMaxH3VAEPrecision(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VAEPrecision",
            display_name="MiniMax H3 VAE Precision (encode/decode split)",
            category="MiniMaxH3/experimental",
            description=(
                "Set the H3 video VAE's encoder and decoder precision "
                "independently. The released pipeline keeps this VAE in fp32 "
                "and decodes under fp16, which ComfyUI's single --fp32-vae "
                "flag cannot express. The encoder is ~7% of the weights, so "
                "fp32 encode is cheap; fp32 decode is not, and is not what "
                "the release does."
            ),
            inputs=[
                io.Vae.Input("vae"),
                io.Combo.Input(
                    "encoder", options=CHOICES, default="fp32",
                    tooltip=(
                        "Precision for the half that encodes references, "
                        "keyframes and any input frames. fp32 matches the "
                        "released pipeline's residency and costs about 0.34 "
                        "GiB more than fp16 on the shipped checkpoint. "
                        "Whether it changes the output is UNMEASURED."
                    ),
                ),
                io.Combo.Input(
                    "decoder", options=CHOICES, default="unchanged",
                    tooltip=(
                        "Precision for the half that turns latents into "
                        "frames. Leave unchanged. fp32 here costs ~4.5 GiB "
                        "and 2-3x decode time, and the release does not do "
                        "it -- it decodes under fp16."
                    ),
                ),
            ],
            outputs=[io.Vae.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, encoder="fp32", decoder="unchanged", **kwargs):
        # The cast lands on a module the loader caches, so a byte-identical
        # resubmission would otherwise be served from the node cache with
        # whatever dtype the PREVIOUS run left behind. Returning the pair
        # makes a changed selection re-execute. Same reasoning as
        # `reference_fit.py`'s guard on its clamp-lifting arm.
        return f"{encoder}/{decoder}"

    @classmethod
    def execute(cls, vae, encoder="fp32", decoder="unchanged") -> io.NodeOutput:
        model = getattr(vae, "first_stage_model", None)
        if model is None or not hasattr(model, "encoder") or not hasattr(model, "decoder"):
            raise ValueError(
                "MiniMaxH3VAEPrecision needs a VAE whose module exposes "
                "`encoder` and `decoder`; got "
                f"{type(model).__name__}. The H3 video VAE does. The H3 "
                "AUDIO VAE does not go through this node -- it is fp32 in "
                "ComfyUI already, which is what the release uses."
            )

        out = copy.copy(vae)
        if encoder != "unchanged":
            model.encoder.to(DTYPES[encoder])
            if hasattr(model, "quant_conv"):
                model.quant_conv.to(DTYPES[encoder])
        if decoder != "unchanged":
            model.decoder.to(DTYPES[decoder])
            if hasattr(model, "post_quant_conv"):
                model.post_quant_conv.to(DTYPES[decoder])

        enc_dtype = _module_dtype(model.encoder)
        dec_dtype = _module_dtype(model.decoder)
        if enc_dtype is not None:
            # VAE.encode casts pixels to vae_dtype, so it has to be the
            # encoder's. The decode side is handled by the wrapper instead.
            out.vae_dtype = enc_dtype
        _wrap_boundaries(model)

        logger.info(
            "[h3] VAE precision: encoder=%s decoder=%s (vae_dtype=%s)",
            enc_dtype, dec_dtype, out.vae_dtype,
        )
        return io.NodeOutput(out)
