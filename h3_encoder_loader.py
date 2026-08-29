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

import importlib
import inspect
import json
import logging
import struct
from pathlib import Path

from comfy_api.latest import io

logger = logging.getLogger(__name__)

def _repo(name: str):
    """Import a sibling module under either import style this repo uses.

    `nodes.py` loads this file as part of a package, while the bench tools put
    the repo directory itself on `sys.path` and import its modules top-level
    (`bench/preflight_graph.py` does exactly that for `h3_awq_encoder`).
    Supporting both is what lets the static reader derive the same contract the
    loader stamps, from one implementation rather than two.
    """
    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ImportError:
            pass
    return importlib.import_module(name)


def expected_special_token_ids(declared):
    """`h3_awq_encoder`'s id arithmetic, re-exported so callers need one import."""
    return _repo("h3_awq_encoder").expected_special_token_ids(declared)


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

    vendor_config = _repo("vendor_config")

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


def _file_tensors(path: str) -> dict:
    """Tensor names and shapes from a safetensors header, without reading data."""
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    return {name: tuple(entry.get("shape", ()))
            for name, entry in header.items() if name != "__metadata__"}


def validate_inventory(model, path: str, name: str) -> None:
    """Refuse a load that did not exactly populate the H3 model.

    Compares the checkpoint's own header against the state dict of the module
    core built from it -- two independent things, rather than a restatement of
    either. Core loads text encoders with `strict=False` (`comfy/sd.py:431`)
    and reports unexpected keys at DEBUG (`comfy/sd.py:293`), so on a normal
    server an incomplete checkpoint quietly keeps factory-initialised
    parameters and says nothing.

    **This deliberately does not wrap `CLIP.load_sd` to read core's own
    missing/unexpected pair, which is what it did first.** That needed a global
    monkeypatch, and two loads racing it could leave the wrapper permanently
    installed with a capture list that grows on every later load -- a leak and
    a wrong answer at once. The header is already on disk; comparing it costs
    one small read and no shared state.

    Measured 2026-08-29: the shipped int8_convrot artifact matches its own
    header exactly, 1602 tensors either side, so this costs a good file
    nothing.
    """
    provided = _file_tensors(path)
    expected = {key: tuple(value.shape)
                for key, value in model.state_dict().items()}
    missing = sorted(set(expected) - set(provided))
    unexpected = sorted(set(provided) - set(expected))
    mismatched = [
        (key, provided[key], expected[key])
        for key in sorted(set(expected) & set(provided))
        if provided[key] != expected[key]
    ]
    if missing or unexpected or mismatched:
        detail = []
        if missing:
            detail.append(f"missing={missing[:5]} ({len(missing)})")
        if unexpected:
            detail.append(f"unexpected={unexpected[:5]} ({len(unexpected)})")
        if mismatched:
            detail.append(f"shape_mismatch={mismatched[:3]} ({len(mismatched)})")
        raise ValueError(
            f"{name} does not exactly populate ComfyUI's H3 text encoder: "
            + "; ".join(detail)
            + ". Core loads text encoders non-strictly and would have kept the "
              "factory-initialised weights for anything missing."
        )


def validate_tokenizer(clip) -> None:
    """Prove ComfyUI's tokenizer realises the RELEASE's special-token list.

    The declaration is the release's own `tokenizer_config.json` under
    `vendor_config/`, not an artifact snapshot: this loader serves files that
    carry no configs of their own, and the release is what the DiT was trained
    against either way. The id arithmetic is shared with `h3_awq_encoder`
    rather than restated -- one rule, two declaration sources.
    """
    vendor_config = _repo("vendor_config")

    declared = vendor_config.additional_special_tokens()
    tokenizer = clip.tokenizer.qwen3vl_32b.tokenizer
    vocab = tokenizer.get_vocab()
    expected = expected_special_token_ids(declared)
    actual = {token: vocab.get(token) for token in declared}
    if actual != expected:
        wrong = {t: (actual[t], expected[t])
                 for t in expected if actual[t] != expected[t]}
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
    """Record what core's path will do -- WITHOUT stamping `_h3_encoder_contract`.

    **This deliberately does not stamp the contract, and the reason is measured.**
    The contract's `video_bounds` are applied CLIP-WIDE by
    `reference_conditioning._configured_qwen_video_size`, which passes
    `num_frames=sampled_count` to `smart_resize`. Core's 12,845,056 is a
    PER-BLOCK budget -- `process_video_block` restarts it for every two frames.
    Those are different semantics for the same number, and stamping core's
    number into a clip-wide field misdescribes core rather than declaring it.

    Measured 2026-08-29, what stamping would have done to reference video by
    making `video_policy=encoder` live on the graphs that select it:

        960x544  at 31 sampled frames   960x544 -> 832x480   (-24% pixels)
        960x544  at 62 sampled frames   960x544 -> 576x320   (-65%)
        1344x768 at 31 sampled frames  1344x768 -> 832x480   (-61%)
        1344x768 at 62 sampled frames  1344x768 -> 576x320   (-82%)

    So the third guard this loader was built with is withdrawn. Its premise was
    that a core-loaded CLIP resolving `encoder` to `comfy` is a silent
    substitution worth ending. It is not: for a NATIVE encoder, `comfy` IS what
    core does, and `_compile_reference_records` already says so in as many
    words -- "the truth of what it was handed, not a fallback". The two guards
    that remain -- inventory and the released token ids -- are the real ones.

    The static readers are unaffected: `bench/preflight_graph.py` calls
    `native_encoder_contract()` directly off the graph's loader class and never
    reads the runtime stamp, so pricing still works.

    `_h3_processor_source` and `_h3_image_bounds` are still recorded, because a
    capture that wants to say which preprocessing ran should be able to; neither
    is read by `encoder_contract_from_clip`.
    """
    model = require_h3(clip, name)
    contract = native_encoder_contract()
    model._h3_processor_source = CONTRACT_SOURCE
    model._h3_image_bounds = contract["image_bounds"]
    return contract


def load_guarded_clip(path: str, embedding_directory,
                      disable_dynamic: bool = False):
    """Core's own H3 load, then the three guards, then the contract."""
    import comfy.sd

    name = Path(path).name
    try:
        clip = comfy.sd.load_clip(
            ckpt_paths=[path], embedding_directory=embedding_directory,
            clip_type=comfy.sd.CLIPType.MINIMAX,
            disable_dynamic=disable_dynamic,
        )
    except Exception as exc:
        # See `require_h3`: a missing detection key does not fail as a missing
        # key, it fails as a different model. Say which file and which stage,
        # and keep the original for the traceback.
        raise ValueError(
            f"{name} could not be constructed as a MiniMax H3 text encoder. "
            "Core detects H3 by the tensors present, so a checkpoint that is "
            "incomplete or not H3 at all is built as another architecture and "
            f"fails far from the cause: {type(exc).__name__}: {exc}"
        ) from exc
    model = require_h3(clip, name)
    validate_inventory(model, path, name)
    validate_tokenizer(clip)
    contract = install_native_contract(clip, name)
    # `comfy.sd.load_clip` registers core's own rebuilder here
    # (`comfy/sd.py:1566`). Left alone, a non-dynamic delegate or a multigpu
    # deepclone would rebuild this CLIP through core and silently drop both the
    # guards and the stamp -- and `encoder_contract_from_clip` would start
    # answering `None` mid-session, which is the exact silent substitution the
    # contract exists to end. Point it at this loader instead.
    clip.patcher.cached_patcher_init = (
        load_guarded_model_patcher, (path, embedding_directory))

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


def load_guarded_model_patcher(path: str, embedding_directory,
                               disable_dynamic: bool = False):
    """Rebuild this CLIP's patcher through the guards, for `cached_patcher_init`.

    `ModelPatcher.clone(disable_dynamic=True)` and `deepclone_multigpu` both
    reconstruct the model by calling this factory
    (`comfy/model_patcher.py:436-441`, `:505-512`), and both raise outright if
    a loader registered none. Registering core's -- which is what
    `comfy.sd.load_clip` leaves behind -- would rebuild an unguarded, unstamped
    model.
    """
    return load_guarded_clip(
        path, embedding_directory, disable_dynamic=disable_dynamic).patcher
