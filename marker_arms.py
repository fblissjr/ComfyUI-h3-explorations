"""Bench-only marker-corpus arms: bind one to a CLIP, and read back what bound.

`bench/marker_corpus/compiled.json` declares each arm as a triple -- prompt
bytes, tokenizer identity, model transform. The prompt bytes are the corpus's
and reach a render through the graph's text widget; this module owns the other
two, and `docs/research/marker_arm_binding.md` is the design it implements.

Two rules shape everything here.

**The arm attaches to a CLONE, never to the loaded model.** `CLIP.clone()`
clones the patcher and shares `cond_stage_model` and the tokenizer by
reference, and `h3_awq_encoder` installs `cached_patcher_init` so ComfyUI can
hand the same loaded CLIP to a later prompt. An in-place row assignment or a
mutated tokenizer would therefore be inherited by every later render that
reuses the model. A patch on a cloned patcher, and a freshly built tokenizer,
cannot be.

**Declaring an arm is not applying one.** Everything this module records is
read back off the live CLIP after the fact -- which markers its tokenizer
actually resolves, what its patcher will actually make of the marker rows --
never the arm name it was given. That is the rule
`bench/check_provenance_stamp.py::closure_is_read_not_declared` established for
the Sol closure, and the reason every red control in
`bench/check_marker_arms.py` has the same shape: the declaration and the read
value have to agree.
"""

from __future__ import annotations

import hashlib
import json
import logging

from comfy_api.latest import io

try:
    from .vendor_config import additional_special_tokens
except ImportError:  # loaded by path, not as a package member
    from vendor_config import additional_special_tokens

logger = logging.getLogger(__name__)

# The state-dict key of the H3 token embedding, relative to `cond_stage_model`.
# Read off a loaded model on 2026-08-25 rather than assembled from the module
# names; `bench/check_marker_arms.py` re-reads it from the real artifact when
# one is installed, because a synthetic fixture built to match this constant
# would satisfy it by construction and prove nothing.
EMBED_KEY = "qwen3vl_32b.transformer.model.embed_tokens.weight"

# A fixed string carrying markers from several families. Its only job is to
# make two tokenizers disagree in a recordable way, so it is a probe and not a
# prompt: the corpus is the only author of prompt text and nothing here is
# ever rendered.
TOKENIZER_PROBE = "he said <d>hold</d> then <|cutoff|> and <|lyrics_start|>la"

ARMS = ("release", "legacy_bpe", "mean_init_rows")


def marker_tokens() -> list[str]:
    """The seven H3 markers, from the release's own declaration.

    The release declares twenty special tokens and the seven H3 ones are the
    tail; taking the tail rather than a retyped list means a reordering
    upstream moves this with it instead of silently disagreeing.
    """
    declared = additional_special_tokens()
    if len(declared) != 20:
        raise ValueError(
            f"the release declares {len(declared)} special tokens, not 20; "
            "the seven H3 markers can no longer be taken as the tail"
        )
    return declared[13:]


def marker_ids(tokenizer) -> dict:
    """Which markers this tokenizer resolves, and to what. `None` for absent."""
    vocab = tokenizer.qwen3vl_32b.tokenizer.get_vocab()
    return {token: vocab.get(token) for token in marker_tokens()}


def _digest(value) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _tensor_digest(tensor) -> str:
    import torch

    value = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    h.update(value.numpy().tobytes())
    return h.hexdigest()


def marker_row_span(rows: int | None = None) -> tuple[int, int]:
    """``(first marker row, count)``, from the ids the release assigns.

    Contiguous by construction -- the markers are appended in declared order --
    and asserted so here, because the offset patch below addresses a span and
    a non-contiguous set would silently patch the wrong rows.
    """
    ids = [151644 + index for index in range(13)] + \
          [151669 + index for index in range(7)]
    span = ids[13:]
    if span != list(range(span[0], span[0] + len(span))):
        raise ValueError(f"the H3 marker ids are not contiguous: {span}")
    if rows is not None and span[-1] >= rows:
        raise ValueError(
            f"marker id {span[-1]} is outside an embedding of {rows} rows"
        )
    return span[0], len(span)


def _embedding(clip):
    """The token embedding as the patcher sees it, or ``None`` if absent."""
    state = clip.patcher.model.state_dict()
    return state.get(EMBED_KEY)


def _patched_marker_rows(clip):
    """The marker rows as this CLIP's patcher will actually produce them.

    Derived through ComfyUI's own ``calculate_weight`` from the patcher's patch
    list, not from any payload this module handed it. That is what makes the
    digest evidence rather than a restatement: a patch that failed to attach,
    or attached to a key the model does not have, leaves these rows unmoved.
    """
    import comfy.lora

    weight = _embedding(clip)
    if weight is None:
        return None
    start, count = marker_row_span(int(weight.shape[0]))
    patches = clip.patcher.patches.get(EMBED_KEY)
    if not patches:
        return weight[start:start + count].clone()
    work = weight.clone()
    comfy.lora.calculate_weight(patches, work, EMBED_KEY)
    return work[start:start + count]


def encoder_arm_record(clip) -> dict:
    """What this CLIP's tokenizer and marker rows ACTUALLY are.

    Readable from any CLIP, including one that never passed through the arm
    node: a render that forgot the node still records what its tokenizer and
    rows were, rather than recording nothing. ``declared_arm`` is present only
    when the node set it, and is a label -- never the evidence.
    """
    record = {
        "declared_arm": getattr(clip, "_h3_declared_marker_arm", None),
        "tokenizer": {},
        "marker_rows": {},
    }
    try:
        ids = marker_ids(clip.tokenizer)
        probe = clip.tokenize(TOKENIZER_PROBE)["qwen3vl_32b"][0]
        record["tokenizer"] = {
            "marker_ids": ids,
            "markers_resolved": sum(1 for v in ids.values() if v is not None),
            "probe": TOKENIZER_PROBE,
            "probe_token_count": len(probe),
            "probe_sha256": _digest([t[0] for t in probe]),
        }
    except Exception as exc:  # noqa: BLE001
        # Type only, never the message: a sidecar is shared next to a render
        # and an exception string is the field that can carry a path off the
        # machine. Same rule as `provenance.py::_geometry`.
        record["tokenizer"] = {"error": type(exc).__name__}
    try:
        rows = _patched_marker_rows(clip)
        if rows is None:
            record["marker_rows"] = {"error": "no embedding at " + EMBED_KEY}
        else:
            start, count = marker_row_span()
            record["marker_rows"] = {
                "key": EMBED_KEY,
                "first_id": start,
                "count": count,
                "sha256": _tensor_digest(rows),
                "patch_keys": sorted(clip.patcher.patches),
            }
    except Exception as exc:  # noqa: BLE001
        record["marker_rows"] = {"error": type(exc).__name__}
    return record


def legacy_tokenizer_clip(clip):
    """A CLIP whose tokenizer is what ComfyUI shipped BEFORE the H3 token fix.

    The same reconstruction as
    `bench/audit_h3_marker_tokenization.py::_unpatched_clip`, and it must stay
    the same: two properties are load-bearing.

    A FRESH tokenizer, not a copy -- ``CLIP.clone()`` assigns the same
    tokenizer object to the clone, so mutating it would reach every other
    holder of that CLIP.

    Emptying whichever token-list attribute EXISTS, then verifying by
    vocabulary that the markers are gone. Keying off one constant's name is how
    a reconstruction returns the patched tokenizer while calling it stock: this
    checkout has the module-level name only, and a version looking for the
    class-level name from the retired local branch would find nothing to empty
    and hand back the release tokenizer.
    """
    import comfy.text_encoders.minimax as mmx
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

    targets = [
        (owner, attr, getattr(owner, attr))
        for owner, attr in ((MiniMaxH3Tokenizer, "H3_SPECIAL_TOKENS"),
                            (mmx, "MINIMAX_EXTRA_TOKENS"))
        if getattr(owner, attr, None)
    ]
    if not targets:
        raise ValueError(
            "no known H3 special-token list is present on ComfyUI's MiniMax "
            "tokenizer, so the pre-fix arm cannot be reconstructed. Emptying "
            "nothing would hand back the release tokenizer labelled legacy."
        )
    try:
        for owner, attr, _ in targets:
            setattr(owner, attr, [])
        fresh = MiniMaxH3Tokenizer(
            embedding_directory=getattr(
                getattr(clip.tokenizer, "qwen3vl_32b", None),
                "embedding_directory", None),
        )
    finally:
        for owner, attr, saved in targets:
            setattr(owner, attr, saved)

    still = [t for t, i in marker_ids(fresh).items() if i is not None]
    if still:
        raise ValueError(
            f"the reconstructed legacy arm still declares {still}, so it is "
            "not the pre-fix tokenizer and every comparison against it would "
            "use the wrong control"
        )
    out = clip.clone()
    out.tokenizer = fresh
    return out


def mean_init_rows_clip(clip):
    """A CLIP whose seven marker rows are the mean of its embedding table.

    An offset-keyed ``set`` patch on the CLONE's patcher, so it addresses the
    seven contiguous rows alone and never writes the shared module. Without the
    offset the patch would need a dense table-sized delta, and an in-place
    assignment would be inherited by every later render that reuses the cached
    model.
    """
    import torch

    weight = _embedding(clip)
    if weight is None:
        raise ValueError(f"this CLIP has no embedding at {EMBED_KEY}")
    start, count = marker_row_span(int(weight.shape[0]))

    # Chunked so the float32 accumulation never materialises a second copy of
    # the table. On the real encoder that table is several gigabytes and the
    # arm may be applied while the model is resident on a 24 GB card.
    # On the weight's own device, not the default one. The encoder is often
    # already resident when an arm is applied -- which is the case this comment
    # anticipated and the accumulator did not: a CPU accumulator against a CUDA
    # table raises "found at least two devices". Every fixture this had met was
    # CPU-resident, so nothing exercised it until a real encoder did.
    total = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
    for lo in range(0, weight.shape[0], 8192):
        total += weight[lo:lo + 8192].to(torch.float32).sum(dim=0)
    mean = (total / weight.shape[0]).to(weight.dtype)

    out = clip.clone()
    matched = out.add_patches(
        {(EMBED_KEY, (0, start, count)): ("set", (mean.unsqueeze(0).repeat(count, 1),))},
        1.0, 1.0,
    )
    if not matched:
        raise ValueError(
            f"add_patches matched nothing for {EMBED_KEY}; the transform did "
            "not attach and the arm would render as the release one"
        )
    return out


def apply_arm(clip, arm: str):
    """Bind one arm and label the result. ``release`` binds nothing."""
    if arm not in ARMS:
        raise ValueError(f"unknown marker arm {arm!r}; expected one of {ARMS}")
    if arm == "release":
        out = clip.clone()
    elif arm == "legacy_bpe":
        out = legacy_tokenizer_clip(clip)
    else:
        out = mean_init_rows_clip(clip)
    out._h3_declared_marker_arm = arm
    return out


class MiniMaxH3MarkerArm(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MarkerArm",
            display_name="MiniMax H3 Marker Corpus Arm (bench)",
            category="MiniMaxH3/experimental",
            description=(
                "BENCH ONLY. Binds one marker-corpus arm to this CLIP and "
                "leaves the loaded model untouched: the transform lives on a "
                "clone, so no later render inherits it. 'release' binds "
                "nothing and is what the release_id and stripped arms use -- "
                "their difference is prompt bytes, which the corpus owns. "
                "'legacy_bpe' swaps in a freshly built pre-fix tokenizer. "
                "'mean_init_rows' replaces the seven H3 marker embedding rows "
                "with the table mean, as a patch, never on disk. Wire the CLIP "
                "through MiniMaxH3ProvenanceStamp to record what actually "
                "bound; the arm name here is a label, not evidence."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input("arm", options=list(ARMS), default="release"),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip, arm="release") -> io.NodeOutput:
        out = apply_arm(clip, arm)
        record = encoder_arm_record(out)
        logger.info(
            "[h3] marker arm %r bound: %s markers resolved, rows %s",
            arm, record["tokenizer"].get("markers_resolved"),
            record["marker_rows"].get("sha256", "?")[:12],
        )
        return io.NodeOutput(out)
