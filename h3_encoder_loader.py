"""Core's `CLIPLoader` for a MiniMax H3 encoder, plus the guards it does not have.

This is deliberately a THIN wrapper. It does not adapt a format, replace
preprocessing, or touch a weight -- core loads the checkpoint exactly as
`CLIPLoader` would, and every difference is a check that runs afterwards or a
declaration stamped on the result. That is the whole point: the int8_convrot
artifact's preprocessing path is core's and is better left alone (float
end-to-end, device-side resize, video blocks that resize instead of refusing),
while core's *loading* path checks almost nothing.

The three things it adds, and what each is for:

1. **The load actually has to populate the model.** `comfy/sd.py:431` loads
   text encoders with `strict=False`, and `comfy/sd.py:277-282` logs the
   missing keys at warning and the unexpected ones at DEBUG -- i.e. invisible
   by default. A checkpoint that silently leaves factory-initialised
   parameters behind still returns a working-looking CLIP. This wrapper
   captures core's own `(missing, unexpected)` and refuses on them.

2. **The tokenizer has to realise the release's own special tokens.** Nothing
   in core asserts that the seven H3 markers land on the ids the DiT was
   trained against. Here they are checked against `vendor_config/`, the
   release's own declaration, rather than against an artifact snapshot.

3. **The CLIP has to say what preprocessing it will get.** A core-loaded CLIP
   declares nothing, so `reference_geometry.effective_policy` downgrades
   `image_policy=encoder` to `comfy` and the reference nodes price a reference
   against a ceiling nobody named. This stamps a contract describing what
   core's own path will actually do, read out of core by introspection.

**A trap this makes visible rather than fixes.** Under core, `video_policy`
`release` fits a video reference to the release's 25,165,824-pixel ceiling and
then `comfy/text_encoders/minimax.py::process_video_block` clamps it back to
its own default. So `release` is a promise core does not keep for video, and
the stamped contract is what lets a caller see that instead of assuming.

**The one edge nothing here controls.** `comfy/text_encoders/qwen3vl.py`
overrides `patch_size` and the normalisation at the CALL SITE, as literals, so
the still-image geometry cannot be introspected the way the video path's can.
This module derives it from the video path -- whose values ARE signature
defaults and module constants -- and refuses if that disagrees with what the
release declares. A core change that moved the still call site alone would
leave the contract wrong and nothing would catch it. Said plainly rather than
papered over.
"""

from __future__ import annotations

import contextlib
import inspect
import logging
from pathlib import Path

from comfy_api.latest import io

from .h3_awq_encoder import expected_special_token_ids

logger = logging.getLogger(__name__)

#: What the stamped contract calls itself. `snapshot_contract` uses the
#: snapshot directory's name; there is no directory here, and naming the code
#: path is the honest answer to "where did these bounds come from".
CONTRACT_SOURCE = "comfyui-native"

#: Keys the release and core must agree on before a contract is stamped. Not
#: the full geometry: `image_mean`/`image_std` are compared separately because
#: a list needs a different equality than an int.
_GEOMETRY_INTS = ("patch_size", "temporal_patch_size", "merge_size")


def _signature_defaults(function, names: tuple[str, ...]) -> dict:
    """Signature defaults, refusing rather than guessing when one is gone.

    The same discipline as `reference_fit.qwen_max_pixels`: these values are
    read out of ComfyUI so this module tracks it instead of holding a second
    copy that goes stale silently.
    """
    parameters = inspect.signature(function).parameters
    out = {}
    for name in names:
        parameter = parameters.get(name)
        if parameter is None or not isinstance(parameter.default, int):
            raise RuntimeError(
                f"{function.__module__}.{function.__qualname__} no longer carries "
                f"an integer `{name}` default, so this wrapper cannot say what "
                "preprocessing a core-loaded H3 encoder will get. Its layout "
                "changed; fix this rather than guessing."
            )
        out[name] = int(parameter.default)
    return out


def native_encoder_contract() -> dict:
    """What core's own H3 preprocessing will do, read out of core.

    Shaped like `h3_awq_encoder.snapshot_contract` because
    `reference_geometry.encoder_contract_from_clip` reads them through the same
    keys. Unlike that one this describes a CODE PATH rather than an artifact,
    which is why every value is introspected and none is typed here.
    """
    from comfy.text_encoders import minimax
    from comfy.text_encoders.qwen_vl import process_qwen2vl_images

    from . import vendor_config

    image_pixels = _signature_defaults(
        process_qwen2vl_images, ("min_pixels", "max_pixels"))
    video_pixels = _signature_defaults(
        minimax.process_video_block, ("min_pixels", "max_pixels"))
    video_ints = _signature_defaults(minimax.process_video_block, _GEOMETRY_INTS)

    # `process_video_block` normalises with these module constants, and
    # `qwen3vl.py`'s still call site passes the same values as literals. The
    # constants are the observable; the literals are the uncontrolled edge the
    # module docstring names.
    geometry = {
        **video_ints,
        "image_mean": list(minimax.QWEN_IMAGE_MEAN),
        "image_std": list(minimax.QWEN_IMAGE_STD),
    }

    declared = vendor_config.patch_geometry()
    disagreements = {
        key: (geometry[key], declared[key])
        for key in geometry
        if key in declared and geometry[key] != declared[key]
    }
    if disagreements:
        raise RuntimeError(
            "ComfyUI's H3 patch geometry disagrees with what the release "
            f"declares: {disagreements} (comfy, release). The contract this "
            "would stamp is the one the reference nodes price against, so it "
            "is refused rather than stamped wrong."
        )

    return {
        "source": CONTRACT_SOURCE,
        "image_bounds": (image_pixels["min_pixels"], image_pixels["max_pixels"]),
        "image_geometry": dict(geometry),
        "video_bounds": (video_pixels["min_pixels"], video_pixels["max_pixels"]),
        "video_geometry": dict(geometry),
    }


@contextlib.contextmanager
def _capture_load_report():
    """Record what core's own load reported, without reimplementing it.

    `CLIP.__init__` computes `(missing, unexpected)` and throws them at the
    logger. Wrapping the method that produces them is the only way to see the
    lists themselves, and it means this guard cannot drift from core's notion
    of what counts as missing -- there is no second comparison here to go
    stale. Scoped to one call and restored in `finally`: nothing in this pack
    leaves ComfyUI patched.
    """
    import comfy.sd

    captured: list[tuple] = []
    original = comfy.sd.CLIP.load_sd

    def recording(self, sd, full_model=False):
        result = original(self, sd, full_model=full_model)
        captured.append(result)
        return result

    comfy.sd.CLIP.load_sd = recording
    try:
        yield captured
    finally:
        comfy.sd.CLIP.load_sd = original


def validate_load_report(captured: list, name: str) -> None:
    """Refuse a load that did not exactly populate the H3 model.

    Core reports `unexpected` at DEBUG, so on a normal server the case this
    catches is invisible: a checkpoint whose keys do not match leaves
    factory-initialised parameters behind and still returns a CLIP. That is the
    2026-08-23 escape's shape, caught there by the AWQ adapter's own inventory
    check and by nothing at all on this path.
    """
    if not captured:
        raise RuntimeError(
            "could not observe ComfyUI's load report; the wrapper's guard did "
            "not run, so this load is unverified. `comfy.sd.CLIP.load_sd` no "
            "longer returns the missing/unexpected pair this reads."
        )
    missing, unexpected = [], []
    for report in captured:
        if not (isinstance(report, tuple) and len(report) == 2):
            raise RuntimeError(
                f"ComfyUI's load report changed shape ({type(report)}); this "
                "guard reads a (missing, unexpected) pair and will not guess."
            )
        missing.extend(report[0])
        unexpected.extend(report[1])
    if missing or unexpected:
        raise ValueError(
            f"{name} does not exactly populate ComfyUI's H3 text encoder. "
            f"missing={sorted(missing)[:6]} unexpected={sorted(unexpected)[:6]} "
            f"({len(missing)} missing, {len(unexpected)} unexpected). Core "
            "loads text encoders non-strictly and would have kept the "
            "factory-initialised weights for anything missing."
        )


def validate_tokenizer(clip) -> None:
    """Prove ComfyUI's tokenizer realises the RELEASE's special-token list.

    The declaration is the release's own `tokenizer_config.json` under
    `vendor_config/`, not an artifact snapshot: this loader serves files that
    carry no configs of their own, and the release is the thing the DiT was
    trained against either way. The id arithmetic is shared with
    `h3_awq_encoder` rather than restated.
    """
    from . import vendor_config

    declared = vendor_config.additional_special_tokens()
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    vocab = tokenizer.get_vocab()
    expected = expected_special_token_ids(declared)
    actual = {token: vocab.get(token) for token in declared}
    if actual != expected:
        wrong = {t: (actual[t], expected[t]) for t in expected if actual[t] != expected[t]}
        raise ValueError(
            "ComfyUI's tokenizer disagrees with the released tokenizer_config "
            f"about {len(wrong)} token id(s): {wrong} (got, expected). A marker "
            "on the wrong id is a different token to the DiT."
        )


def require_h3(clip, name: str):
    """Return the H3 transformer, refusing a CLIP core built as something else.

    `comfy/sd.py::detect_te_model` recognises H3 by the presence of particular
    tensors, so a checkpoint missing one is not rejected -- it is detected as a
    DIFFERENT architecture. Measured 2026-08-29: dropping
    `visual.deepstack_merger_list.0.norm.weight` sends the load into
    `comfy/text_encoders/flux.py`, which dies trying to parse a Mistral
    tokenizer with `TypeError: the JSON object must be str, bytes or
    bytearray, not NoneType`. Nothing in that message names the cause.
    """
    model = clip
    for attribute in ("cond_stage_model", "qwen3vl_32b", "transformer"):
        model = getattr(model, attribute, None)
        if model is None:
            raise ValueError(
                f"{name} loaded, but not as a MiniMax H3 text encoder: the CLIP "
                f"has no `{attribute}`. Core detected some other architecture "
                "from the tensors present, which means this file is missing one "
                "of the keys H3 detection keys on."
            )
    return model


def install_native_contract(clip, name: str = "encoder") -> dict:
    """Stamp what core's path will do, where the reference nodes read it."""
    model = require_h3(clip, name)
    contract = native_encoder_contract()
    model._h3_encoder_contract = contract
    model._h3_processor_source = CONTRACT_SOURCE
    model._h3_image_bounds = contract["image_bounds"]
    return contract


def load_guarded_clip(path: str, embedding_directory):
    """Core's own H3 load, then the three guards, then the contract."""
    import comfy.sd

    name = Path(path).name
    with _capture_load_report() as captured:
        try:
            clip = comfy.sd.load_clip(
                ckpt_paths=[path], embedding_directory=embedding_directory,
                clip_type=comfy.sd.CLIPType.MINIMAX,
            )
        except Exception as exc:
            # See `require_h3`: a missing detection key does not fail as a
            # missing key, it fails as a different model. Say which file and
            # which stage, and keep the original for the traceback.
            raise ValueError(
                f"{name} could not be constructed as a MiniMax H3 text encoder. "
                "Core detects H3 by the tensors present, so a checkpoint that "
                "is incomplete or not H3 at all is built as another "
                f"architecture and fails far from the cause: {type(exc).__name__}: {exc}"
            ) from exc
    validate_load_report(captured, name)
    require_h3(clip, name)
    validate_tokenizer(clip)
    contract = install_native_contract(clip, name)

    image_lo, image_hi = contract["image_bounds"]
    video_lo, video_hi = contract["video_bounds"]
    logger.info(
        "[h3-encoder] loaded %s through ComfyUI's own H3 loader; inventory and "
        "the released token list verified. Preprocessing is core's: still "
        "%d..%d px, video block %d..%d px, %s. Reference nodes will price "
        "against this contract instead of falling back to an unnamed ceiling.",
        name, image_lo, image_hi, video_lo, video_hi,
        ", ".join(f"{k}={v!r}" for k, v in sorted(contract["image_geometry"].items())),
    )
    return clip


class MiniMaxH3EncoderLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        import folder_paths

        names = sorted(
            n for n in folder_paths.get_filename_list("text_encoders")
            if n.endswith(".safetensors")
        )
        return io.Schema(
            node_id="MiniMaxH3EncoderLoader",
            display_name="Load MiniMax H3 Encoder (guarded)",
            category="MiniMaxH3/loaders",
            description=(
                "ComfyUI's own H3 text-encoder load, with the checks it does "
                "not perform: the checkpoint must exactly populate the model "
                "(core loads non-strictly and reports unexpected keys at debug "
                "level), the tokenizer must realise the released special-token "
                "ids, and the CLIP is stamped with what core's preprocessing "
                "will actually do so the reference nodes price against a named "
                "ceiling. Preprocessing itself is unchanged -- for a "
                "compressed-tensors W4A16 artifact use the AWQ loader instead."
            ),
            inputs=[io.Combo.Input("encoder_name", options=names)],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, encoder_name):
        import folder_paths

        path = folder_paths.get_full_path_or_raise("text_encoders", encoder_name)
        return io.NodeOutput(
            load_guarded_clip(path, folder_paths.get_folder_paths("embeddings"))
        )
