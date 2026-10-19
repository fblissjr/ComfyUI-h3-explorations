# VOID pass 1 converted from upstream for core's loader (2026-10-05)

lane: masking
verdict: done and then removed with the VOID code; the upstream checkpoint is a pure rename away from the file core loads

Session mrorange. Found while building the clean-plate probe
(`2026-10-05_void_plate_turn.md`).

**The converter and its check were removed the same day with the rest of the
VOID code, at the owner's word** (`2026-10-05_void_plate_turn.md` has the
sentence). `bench/convert_void_checkpoint.py` and
`bench/check_void_conversion.py` were last present at commit df3bd21b. What
follows is the record of what was done; the commands in it no longer exist in
the tree.

## What was wrong

`UNETLoader` on the file linked as `models/diffusion_models/void_pass1.safetensors`
raises "Could not detect model type". That file is `netflix/void-model`'s own,
in diffusers' key layout (`transformer_blocks.N.attn1.to_q...`). Core's
CogVideoX detection looks for `blocks.0.norm1.linear.weight`
(`comfy/model_detection.py`), and core's VOID template
(`utility_void_video_inpainting.json`) names a different file:
`Comfy-Org/void-model`, `diffusion_models/void_pass1.safetensors`, a repack
that is not on this box. `void_pass2.safetensors` is linked the same way and
is in the same state.

corrected: the masking note of 2026-10-04 said the Comfy-Org
repository holds only the T5 encoder, the VAE and the optical-flow file. The
template's model list names both passes there, and the header of pass 1 was
read from it for this record.

## What the repack is

Read from the two headers, upstream's on disk and the repack's by HTTP range
request (about a hundred kilobytes; no weights fetched):

- 1024 tensors in each, all bf16.
- The file sizes differ by 16432 bytes, which is exactly the difference
  between the two header lengths. The data sections are the same size.
- Renaming upstream's keys by the rule in
  `bench/convert_void_checkpoint.py::rename` gives the repack's key set
  exactly, with every shape and dtype equal. The tensors are stored in a
  different order in the repack.

So nothing is fused, split or transposed.

## The conversion

`bench/convert_void_checkpoint.py` writes a new safetensors file whose data
section is the upstream file's, byte for byte and in the same order, under a
header with core's names. No tensor is read and no value changes. The file's
metadata says what it was converted from.

    <comfy venv python> bench/convert_void_checkpoint.py \
        <Storage>/netflix_void-model/void_pass1.safetensors \
        <Storage>/netflix_void-model/void_pass1_comfy_keys.safetensors --verify

The result is 11143026048 bytes, beside the original, and is linked into
`models/diffusion_models/` under its own name. The two links that were
already there point at the upstream files and were left alone.

## How it was checked

| check | result |
|---|---|
| every key, shape and dtype against the repack's header | 1024 of 1024 equal |
| bytes against the repack's by range request: every tensor outside the blocks and every tensor of blocks 0, 20 and 41, the small ones whole and the large ones their first 65536 bytes | 88 tensors, 58 of them whole, 2479872 bytes, no difference |
| the renamed key set against the state-dict keys of core's `CogVideoXTransformer3DModel` built on CPU | equal (`bench/check_void_conversion.py`, which needs no weights) |
| `comfy.model_detection.model_config_from_unet` on CPU | `CogVideoX_Inpaint` for the converted file, nothing for upstream's |
| through a server | loaded and sampled both arms of the plate probe |

The plate probe sampled from the file as first written by the same code under
`internal/`, before the converter moved to `bench/`. The file was then written
again by the tracked converter and checked again as above; the two writings
differ only in one metadata string.

## Not done

- Pass 2 is not converted. The same command should do it; nothing here has
  run it.
- The byte comparison is a sample. The other 39 blocks rest on the header
  match, the equal data-section size and the fact that the copy does not
  parse the data at all.
