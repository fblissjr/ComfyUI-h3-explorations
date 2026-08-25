# qwen3vl_32b_minimax_h3_w4a16_awq_v2

Settings-preserving snapshot written by `bench/convert_h3_awq_candidate.py` on
2026-08-25 from the candidate directory `qwen3-vl-32b-h3-w4a16-awq-v2-nativecal`. The weights themselves
stay outside this repository; the adapter loads `qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors`.

Every file here is a byte-for-byte copy of the candidate's own, so the digests
in `sha256.json` are digests of the candidate's files as well as of these
copies. Nothing here was retyped or reformatted.

What this snapshot declares, as of the date above:

- 64 source decoder layers, of which H3 consumes 50 (350 W4A16 linears).
- symmetric group-128 pack-quantized W4A16, with 117 entries on the quantizer's ignore list; vision tower, mergers, embedding and LM head stay unquantized.
- top-level storage dtype `float32`; the decoder is bfloat16.
- a still-image budget of 65536..16777216 pixels and a video budget of 4096..25165824.

What consumes each file:

- `config.json`: validates the Qwen3-VL-32B architecture and the symmetric group-128 W4A16 compressed-tensors contract, and is the record the loader matches an artifact against.
- `tokenizer_config.json`: validates that ComfyUI's native MiniMax tokenizer realizes all 20 declared special tokens, including the seven H3 ids 151669-151675.
- `preprocessor_config.json`: constructs the still-image processor: pixel bounds, normalization and patch geometry. The release spelling of the same settings, flat rather than nested inside a processor container.
- `video_preprocessor_config.json`: owns video pixel bounds and patch geometry.
- `recipe.yaml`: records how the artifact was calibrated and quantized; provenance, not a runtime configuration file.
- `h3_v2_run_record.json`: the calibration run's own record, copied from the candidate; provenance, read by nothing at load time.

The loader recognizes an artifact by comparing its embedded `config.json`
against every snapshot under `config/`, not by its filename. Editing any file
here therefore changes which artifacts load, and `bench/check_h3_awq_encoder.py`
hashes this directory on every run.
