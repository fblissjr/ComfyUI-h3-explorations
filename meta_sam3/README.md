# meta_sam3: Meta's SAM 3.1 inference code, copied

last updated: 2026-10-06

This directory holds Meta's own inference code for the SAM 3.1 multiplex
video predictor, copied from
[facebookresearch/sam3](https://github.com/facebookresearch/sam3) at commit
`2345a4ad109ac29c569da749c91d84f10dc08c40` and edited only where it must be
to run inside ComfyUI. It is here so the pack can use the model the way its
authors built it.

Nothing in the pack imports it yet. No node, graph or check runs this code;
the one check that reads it compares text.

## Licence

Everything in this directory is under the SAM License in
[`LICENSE`](LICENSE), not under this pack's own licence: Meta's files, our
edits to them, and the files we added. The rest of the pack keeps the licence
in the repository's root.

Every Python file of Meta's begins with three lines that name the upstream
commit and this licence and say how the file differs from upstream. Meta's
own copyright line stays below them. A file of ours begins with one line
saying it is not Meta's.

## What is here

| path | whose | what |
|---|---|---|
| `LICENSE` | Meta's | the SAM License, byte for byte as upstream ships it |
| `sam3/` | Meta's, edited | the copied package; `FILES.txt` lists the upstream paths it holds |
| `sam3/_h3_compat.py` | ours | the two `timm` names Meta's code imports, so `timm` is not a dependency |
| `FILES.txt` | generated | the file list a build reads |
| `EDITS.diff` | generated | every difference from upstream beyond the header and the import rule |
| `__init__.py`, `README.md` | ours | |

## How the copy is made

[`bench/build_meta_sam3.py`](../bench/build_meta_sam3.py) makes `sam3/` from a
checkout of upstream at the commit above. For each path in `FILES.txt` it
applies one rule, then the hunks of `EDITS.diff`.

The rule: the header, and each `from sam3.a.b import c` rewritten as the
relative import for the file's depth. The copy never names itself, because
this pack's directory name is not importable and another pack may ship a
top-level `sam3`.

`EDITS.diff` is the declaration of everything else. It is the unified diff
between "upstream with the rule applied" and this tree, written by
`bench/build_meta_sam3.py --declare` and never by hand. To change the copy,
edit the file under `sam3/`, run `--declare`, and commit both.

[`bench/check_meta_sam3_copy.py`](../bench/check_meta_sam3_copy.py)
regenerates the declaration and compares it with `EDITS.diff` byte for byte.
An edit that is not declared fails it, and so does a declared edit that is no
longer in the tree. It also holds `LICENSE` to upstream's bytes, every file to
its header, and the copy to importing nothing outside itself.

To move to a newer upstream commit: change `COMMIT` in the build script,
rebuild, and re-make by hand each edit the build reports as no longer
applying.

## Why each edit is there

`EDITS.diff` has the lines. This is what they are for.

| for | files | what |
|---|---|---|
| no network | `model_builder.py` | upstream's builder downloads the weights from Hugging Face when it is given no path. The download function, its flag and the `huggingface_hub` import are gone. No path means no weights are loaded |
| a builder that says what it supports | `model_builder.py`, `__init__.py` | the builders that are not on the SAM 3.1 path are deleted as whole functions, with the three files only they needed; the package no longer re-exports them |
| packages ComfyUI does not have | `model_builder.py`, `model/tokenizer_ve.py` | `pkg_resources` and `iopath` replaced by a path and `open` |
| | `model/memory.py`, `model/vitdet.py`, `model/sam3_tracker_base.py`, `model/video_tracking_multiplex.py` | `DropPath` and `trunc_normal_` come from `_h3_compat.py`, not `timm` |
| | `model/tokenizer_ve.py` | `ftfy` is optional; without it the prompt is normalised to NFC only. A prompt outside ASCII can then be tokenised differently from upstream |
| | `train/masks_ops.py` | `pycocotools` is optional. The request path calls `mask_iom` from that file, which does not use it. `rle_encode` does, and has one call site in `model/sam3_video_base.py`; where this pack has run the copy that site was not reached |
| | `model/sam3_image.py`, `model/sam3_tracker_base.py` | `BatchedDatapoint` imported from the file that defines it, not through the training collator |
| nothing process-wide at import or build | `model/sam3_multiplex_base.py` | importing the module asked the driver for device 0 and set the tf32 flags. Removed |
| | `model_builder.py`, `model/sam3_multiplex_video_predictor.py`, `model/sam3_multiplex_base.py` | the tf32 flags are no longer set, and the bfloat16 autocast that was entered at build and never left is gone; `.cuda()` is a `device` argument |
| | `model/tokenizer_ve.py` | upstream set `TOKENIZERS_PARALLELISM` in the process environment at import, to quiet another library's warning. The copy's tokenizer does not use that library, and the environment belongs to the server. Removed |
| builds without the card | `model/position_encoding.py`, `model/decoder.py` | two constructors named `"cuda"` |
| a bug upstream | `model/sam3_base_predictor.py` | `start_session` passes a keyword the SAM 3.1 model's `init_state` does not take and raises. It now filters its keywords by the model's signature, the idiom `add_prompt` uses a few lines below |
| what the rule cannot express | `model/sam3_multiplex_tracking.py` | two bare `import sam3.model.<module>` lines and the two lines that use a module by its dotted name |

## What the copy leaves to its caller

The edits take decisions out of Meta's code that a library inside a shared
server should not make. Each one becomes the caller's:

- Weights. The caller loads a state dict into the built model.
- Precision. Meta ran the model inside a bfloat16 autocast. A caller that
  does not enter one runs it in a precision Meta did not.
- The tf32 flags, which are process-wide.
- The device, and when the model is on it.
- The detector's frame batch. Meta's builder sets
  `batched_grounding_batch_size` (`sam3/model_builder.py`) for its own
  hardware, and the copy leaves the builder's value as it is. It is an
  attribute of the built model, and a caller on a smaller card lowers it
  there. Where this pack has run the copy, that is the one setting that was
  not Meta's default. The record of that run is
  [`bench/results/2026-10-06_meta_sam3_card_run.md`](../bench/results/2026-10-06_meta_sam3_card_run.md).
