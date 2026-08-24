#!/usr/bin/env python3
"""The one declared transformation between the raw presentation and `oneshot`.

`canonical/2026-08-24_gate1_seam_acceptance.md` records the rule this
implements. Supplying an all-ones `attention_mask` makes Transformers SDPA fall
back to its math backend, which materialises a `heads x seq x seq` tensor --
9.13 GiB at float32 and 6,189 tokens, growing quadratically. Omitting a mask
that masks nothing leaves the attention causal and costs nothing.

The hazard is that dropping a key on the way into a launcher is exactly the
kind of silent convenience the rejected preflight was full of. So this is not
an optimisation applied somewhere in a launcher: it is **a named transformation
with a record**, and the seam identity proof for later gates runs on its output
rather than on the raw batch.

What it guarantees:

1. the mask exists -- a batch without one is refused, never assumed all-ones;
2. every element is one, checked on the tensor rather than inferred from how
   the batch was built;
3. the raw presentation hash is recorded before anything is dropped;
4. the effective model-input hash is recorded after; and
5. a single zero anywhere refuses the row. A padded or truncated row is not
   eligible for this normalisation and must stop the run rather than quietly
   keep its mask, because a population where the transform silently applies to
   some rows and not others is a population nobody can describe.

Both hashes travel in the record, so a reviewer can tie an effective batch back
to the presentation it came from and see exactly what changed between them.

Importable from either virtualenv: pure `torch`, no ComfyUI and no
`llmcompressor`.
"""

from __future__ import annotations

import hashlib
import json

import torch

EFFECTIVE_SCHEMA = "h3-effective-calibration-input-v1"
TRANSFORM = "omit-all-ones-attention-mask"
OMITTED_KEY = "attention_mask"


class MaskNotNormalisable(RuntimeError):
    """The batch is not eligible for the declared normalisation.

    Raised rather than handled: `active_plan.md` lists a silently altered
    calibration input as a stop condition, and a row that cannot take the
    transform is one.
    """


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    if value.dtype == torch.bfloat16:
        value = value.view(torch.uint16)
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def batch_sha(batch: dict[str, torch.Tensor]) -> str:
    """One hash over a whole batch: sorted keys, each with its tensor hash.

    Keys are part of the digest, so dropping one changes the batch hash even
    when every surviving tensor is untouched. That is the property that makes
    the raw and effective hashes distinguishable at all.
    """
    parts = {key: tensor_sha(value) for key, value in sorted(batch.items())}
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def effective_batch(raw: dict[str, torch.Tensor],
                    row_id: str | None = None) -> tuple[dict[str, torch.Tensor], dict]:
    """Return the dictionary the dataloader consumes, and the record for it.

    The raw batch is not mutated: the caller keeps its presentation object and
    can re-derive the raw hash independently.
    """
    where = f"{row_id}: " if row_id else ""
    if OMITTED_KEY not in raw:
        raise MaskNotNormalisable(
            f"{where}the raw batch has no {OMITTED_KEY}. This transform asserts "
            f"an all-ones mask before omitting it; an absent mask is not the "
            f"same claim and is refused rather than assumed."
        )

    mask = raw[OMITTED_KEY]
    zeros = int((mask != 1).sum())
    raw_hash = batch_sha(raw)
    mask_hash = tensor_sha(mask)
    if zeros:
        raise MaskNotNormalisable(
            f"{where}{OMITTED_KEY} has {zeros} element(s) that are not 1, so it "
            f"masks real positions. This row is not eligible for the "
            f"{TRANSFORM} normalisation and the run must stop rather than "
            f"process it under a different rule from its neighbours."
        )

    effective = {key: value for key, value in raw.items() if key != OMITTED_KEY}
    record = {
        "schema": EFFECTIVE_SCHEMA,
        "transform": TRANSFORM,
        "row_id": row_id,
        "assertion": f"every element of {OMITTED_KEY} equals 1",
        "assertion_holds": True,
        "attention_mask_present": True,
        "attention_mask_elements": int(mask.numel()),
        "attention_mask_non_one_elements": zeros,
        "attention_mask_sha256": mask_hash,
        "raw_keys": sorted(raw),
        "raw_presentation_sha256": raw_hash,
        "omitted_keys": [OMITTED_KEY],
        "effective_keys": sorted(effective),
        "effective_model_input_sha256": batch_sha(effective),
        "rationale": "an all-ones mask sends Transformers SDPA to its math "
                     "backend, which materialises a quadratic attention tensor; "
                     "omitting a mask that masks nothing leaves the attention "
                     "causal",
    }
    return effective, record
