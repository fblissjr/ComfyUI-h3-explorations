"""Give the H3 text encoder the special tokens its own release declares.

ComfyUI backs the H3 tokenizer with its bundled `qwen25_tokenizer` directory,
whose config declares thirteen `additional_special_tokens`. The release declares
twenty. The seven it does not have are `<d>`, `</d>`, `<|cutoff|>`,
`<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>` and `<|caption_end|>`,
and without them a prompt containing one is tokenized as ordinary text -- angle
brackets and letters, several BPE pieces, a different embedding.

Everything else about the two tokenizers agrees: the vocabulary is identical at
151,643 entries, the merges are identical, and all 26 `added_tokens_decoder`
entries match on content and id. Ordinary prose is unaffected. Only the markers
change, which is why nothing has ever noticed.

**The rows exist.** The release declares `vocab_size: 151936` and both repacked
encoders on this box carry `model.embed_tokens.weight` at `[151936, 5120]`,
past the 151,669 the vocabulary and added tokens occupy. Adding the seven puts
them at 151669-151675, inside the table.

**What is NOT established, and this node will not pretend otherwise:** whether
those rows carry meaningful trained values in the repacked encoders, and what
the markers are for. The release lists them without documenting them. So this
node makes the markers *reachable*; it does not make them *useful*, and no
prompt in this repo uses one. `docs/research/official_weights_metadata.md`
carries the same caveat.

**Isolation.** `clip.clone()` shares the tokenizer object by reference, so
mutating the one already on a loaded CLIP would contaminate every graph in the
process -- the same silent-contamination class `reference_fit.py` documents for
its global rebind. This node therefore builds a FRESH tokenizer, adds the
tokens to that, and rebinds it on the clone. Verified 2026-08-21: a second
tokenizer constructed the same way is unaffected by the first one's additions.
"""

from __future__ import annotations

import logging

from comfy_api.latest import io

from . import vendor_config

logger = logging.getLogger(__name__)


def _tokenizer_chain(clip):
    """(the SD1Tokenizer, its inner SDTokenizer) for an H3 CLIP, or a reason."""
    tok = getattr(clip, "tokenizer", None)
    inner = getattr(tok, "qwen3vl_32b", None) if tok is not None else None
    hf = getattr(inner, "tokenizer", None) if inner is not None else None
    if hf is None or not hasattr(hf, "add_special_tokens"):
        return None, None, (
            "this node needs the MiniMax H3 text encoder: a CLIP whose "
            "tokenizer exposes `qwen3vl_32b` with a HuggingFace tokenizer "
            f"under it. Got {type(tok).__name__}."
        )
    return inner, hf, None


class MiniMaxH3VendorTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VendorTokens",
            display_name="MiniMax H3 Vendor Special Tokens",
            category="MiniMaxH3/experimental",
            description=(
                "Add the special tokens the H3 release declares and ComfyUI's "
                "bundled tokenizer does not, so a prompt using <d> or the "
                "lyrics/caption markers tokenizes as those markers instead of "
                "as literal text. Wire between the CLIP loader and the "
                "conditioning node. What the markers DO is undocumented "
                "upstream and unmeasured here."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Boolean.Input(
                    "strict", default=True,
                    tooltip=(
                        "On: refuse if any declared token cannot be added, "
                        "rather than silently conditioning on a prompt whose "
                        "markers are half text. Off: warn and continue."
                    ),
                ),
            ],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def fingerprint_inputs(cls, strict=True, **kwargs):
        # The node builds a fresh tokenizer each run, so nothing persists
        # between runs -- but the flag must still re-execute when flipped.
        return f"vendor_tokens/{strict}/{len(vendor_config.additional_special_tokens())}"

    @classmethod
    def execute(cls, clip, strict=True) -> io.NodeOutput:
        _, hf_probe, why = _tokenizer_chain(clip)
        if why:
            raise ValueError(why)

        declared = vendor_config.additional_special_tokens()
        missing = [t for t in declared if t not in hf_probe.get_vocab()]
        if not missing:
            logger.info("[h3] vendor tokens: all %d already present, "
                        "clip passed through unchanged", len(declared))
            return io.NodeOutput(clip)

        n = clip.clone()
        # A FRESH tokenizer, not a copy of the loaded one: `clone()` shares the
        # tokenizer by reference and a shallow copy would still share the
        # HuggingFace object underneath it.
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
        fresh = MiniMaxH3Tokenizer(
            embedding_directory=getattr(
                getattr(clip.tokenizer, "qwen3vl_32b", None),
                "embedding_directory", None),
        )
        inner, hf, why = _tokenizer_chain_of(fresh)
        if why:
            raise ValueError(why)

        added = hf.add_special_tokens({"additional_special_tokens": declared})
        still = [t for t in declared if t not in hf.get_vocab()]
        if still:
            message = (f"could not add {still} to the tokenizer; a prompt "
                       "using them would condition on literal text")
            if strict:
                raise ValueError(message)
            logger.warning("[h3] vendor tokens: %s", message)

        # `inv_vocab` is built at construction and only read by `untokenize`,
        # which is off the render path. Refreshed anyway so the two views of
        # the vocabulary do not disagree for whoever reads it next.
        if hasattr(inner, "inv_vocab"):
            inner.inv_vocab = {v: k for k, v in hf.get_vocab().items()}

        n.tokenizer = fresh
        vocab = hf.get_vocab()
        logger.info(
            "[h3] vendor tokens: added %d of %d declared -> %s",
            added, len(declared),
            {t: vocab.get(t) for t in missing},
        )
        return io.NodeOutput(n)


def _tokenizer_chain_of(tokenizer_obj):
    inner = getattr(tokenizer_obj, "qwen3vl_32b", None)
    hf = getattr(inner, "tokenizer", None) if inner is not None else None
    if hf is None or not hasattr(hf, "add_special_tokens"):
        return None, None, (
            "the freshly built H3 tokenizer does not expose a HuggingFace "
            "tokenizer under `qwen3vl_32b`; ComfyUI's tokenizer layout changed"
        )
    return inner, hf, None
