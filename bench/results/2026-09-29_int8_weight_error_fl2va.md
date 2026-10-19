# How much weight the shipped int8 file loses, block by block (2026-09-29)

`docs/attention/quant_policy.md` records that no sensitivity measurement of the int8
linears exists here. This is a weight-only half of one:
`bench/measure_int8_weight_error.py` compares each quantized linear of
`minimax_h3_fl2va_pruned_int8_convrot` with the bf16 file it was made from,
`minimax_h3_fl2va_pruned_bf16`, as the relative L2 error of the dequantized
weight (`err_shipped`). Rows: `2026-09-29_int8_weight_error_fl2va.jsonl`, one
per linear of the 50 blocks. CPU only, one checkpoint.

## What it shows

- **The int8 weight error is nearly the same in every block and module.**
  `qkv_proj`, `fc1` and `fc2` sit within a hair of one value each across all 50
  blocks (about 1% relative). Only `attn.out_proj` moves, and only somewhat.
- **`attn.out_proj` is the one that varies:** highest in blocks 0 to 3, lower
  in the middle, and back up a little at block 49. Block 49's mean over its four
  linears is among the higher ones and still within a few percent of the rest.
- **The tail is not special at the weight level.** Blocks 45, 47, 48 and 49
  carry no more weight error than the middle. Block 49's known int8 problem is
  in attention's K channels and activations
  (`docs/attention/block49_quant_error.md`), which this does not measure.
- **The recipe is close to the file's, not the same.** Quantizing the bf16
  weights with the format the file declares gives codes that match the shipped
  codes for most positions but not all (`recipe_codes_equal_fraction` in the
  rows for blocks 0, 25 and 49); `err_shipped` is unaffected, since it reads the
  shipped codes.

## What it does not show

- The output error a layer causes. W8A8 also quantizes activations, and a layer
  with small weight error can still matter (or not) at the output.
- Anything about a render. Nothing here rendered, and on this model small
  effects sit near the measurement floor.

## So what

If a bf16 layer is ever tried, the weights give no reason to start with the
tail; the first candidates by weight error are `attn.out_proj` in blocks 0 to
3. Whether that helps is a render's or an activation-side measurement's to say.
