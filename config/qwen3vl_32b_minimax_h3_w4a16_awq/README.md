# Qwen3-VL-32B MiniMax H3 W4A16 AWQ config snapshot

These are settings-preserving snapshots made on 2026-08-23 from:

the owner-maintained `fbjr_qwen3-vl-32b-W4A16-AWQ-H3` source artifact directory

The model itself remains outside this repository. The generated graphs select
the ComfyUI `models/text_encoders/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors`
symlink, while the loader is deliberately not bound to that filename:
`h3_awq_encoder.py` checks whichever file the user selects against this
snapshot and the complete native H3 tensor inventory before adapting it.

The JSON values are exact. This repository keeps a conventional final newline;
the source `config.json` and `video_preprocessor_config.json` did not, so those
two byte digests differ by that newline only. `sha256.json` records the files
as committed here; the source digests for those two are respectively
`add5e713...16fc` and `7768af27...6d13`.

What consumes each file:

- `config.json`: validates the BF16 Qwen3-VL-32B architecture and symmetric
  group-128 W4A16 compressed-tensors contract. H3 intentionally consumes only
  language layers 0–49.
- `tokenizer_config.json`: validates that ComfyUI's native MiniMax tokenizer
  realizes all declared special tokens and the seven H3 ids 151669–151675.
- `processor_config.json`: constructs the still-image processor used by the
  custom loader, including its bicubic resize, pixel bounds, normalization,
  and patch geometry.
- `video_preprocessor_config.json`: owns video pixel bounds and patch geometry.
  Its JSON content is identical to the official H3 snapshot already in
  `vendor_config/`; it is retained here because this directory records the
  custom encoder artifact as a self-contained contract.
- `recipe.yaml`: records how the artifact was calibrated and quantized; it is
  provenance, not a runtime configuration file.

Native versus local: ComfyUI now supplies the H3 architecture and corrected
tokenizer, and its stock `CLIPLoader` natively loads the separate
`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` representation.
`comfy-kitchen` supplies the W4A16 operator used here. Recognition of this
artifact's compressed-tensors namespace/metadata, the zero-copy repack, H3
truncation, and source-config preprocessing are implemented by this
repository's custom loader; they are not native `CLIPLoader` support.

See `sha256.json` for the copied files and the canonical external model
artifact. The routine check hashes the small committed snapshots; pass
`--verify-model-hash` to `bench/check_h3_awq_encoder.py` for the intentionally
slow full-file integrity pass over the external model.
