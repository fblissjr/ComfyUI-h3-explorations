  I need a forensic reconstruction of yesterday’s Qwen3-VL-32B W4A16 AWQ quantization run, followed by a practical design review for special-token post-training on one RTX 4090.

  Do not quantize, train, upload, commit, or rewrite public documentation during this pass. First establish what actually happened and what can be reproduced. You may create the two internal deliverables requested below.

  Workspace:
    <HOME>/ComfyUI/custom_nodes/ComfyUI-h3-explorations

  Read first:
    AGENTS.md
    CLAUDE.md

  Primary material:
    docs/research/awq_quantization_suite/
    docs/research/awq_quantization_suite/qwen3vl_32b_w4a16_awq_quantization_report.md
    docs/research/qwen3-vl-special-tokens-post-training/h3_special_tokens_post_training.md
    docs/research/official_weights_metadata.md
    docs/h3_awq_encoder.md
    internal/prompts/2026-08-23_caption_marker_arms.md
    internal/postmortems/2026-08-23_session_caption-arms-and-five-broken-controls.md
    internal/postmortems/2026-08-23_session_h3_awq_hf_workflows_and_encoder_ab.md

  Deployed checkpoint:
    <HOME>/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors

  It resolves to:
    <HOME>/Storage/fbjr_qwen3-vl-32b-W4A16-AWQ-H3/qwen3vl_32b_minimax_h3_w4a16_awq.safetensors

  Background and scientific intent
  --------------------------------

  MiniMax added seven H3-specific special tokens to the Qwen3-VL-32B tokenizer:

    151669  <d>
    151670  </d>
    151671  <|cutoff|>
    151672  <|lyrics_start|>
    151673  <|lyrics_end|>
    151674  <|caption_start|>
    151675  <|caption_end|>

  The corresponding embedding rows were not trained by Qwen or MiniMax. They remain statistically equivalent to unused padding rows. Nevertheless, H3 consumes the unnormalized output after language layer 50, and the frozen H3 DiT may have
learned to interpret the stable contextual patterns produced by those rows.

  Important correction: caption-tagged prompts work behaviorally. Do not describe the caption pair as inert or as “doing nothing.” The older caption experiment records limitations in isolating the pair’s marginal advantage over quoted prose;
that causal comparison is a separate question from whether caption-tagged prompts work. Treat the owner’s latest observation as controlling.

  The purpose of post-training is exploratory: determine whether assigning more explicit semantics to these seven rows changes or improves H3 behavior. Do not assume this is a repair, and do not assume an embedding row that looks
linguistically better to Qwen will be better for the frozen H3 DiT.

  Hard constraint: this is a solo experiment on one RTX 4090 with 24 GiB VRAM. Avoid proposals requiring multi-GPU training, a continuously co-resident BF16 32B encoder and 33B DiT, or a large new dataset acquisition.

  Known artifact observations to verify independently
  ---------------------------------------------------

  A header-level inspection of the deployed single-file checkpoint found:

    - file size: 20,394,199,288 bytes, approximately 19.0 GiB;
    - 1,954 tensors;
    - source language layers 0–63;
    - 448 packed W4 linears and 448 scale tensors, i.e. seven per layer across 64 layers;
    - 351 BF16 visual tensors;
    - BF16 embedding table [151936, 5120];
    - BF16 LM head [151936, 5120].

  The ComfyUI adapter discards source layers 50–63, the final language norm, and LM head. It retains 350 W4 linears across layers 0–49.

  The seven deployed AWQ embedding rows have exactly the same norm statistics as the official BF16 release:

    marker mean norm: 0.5049785972
    untrained-tail mean norm: 0.5029426813

  Therefore the AWQ process preserved the BF16 embedding rows; it did not train or quantize them.

  A small backward smoke through the currently installed
  comfy_kitchen.gemv_awq_w4a16 failed with:

    RuntimeError: Trying to backward through
    comfy_kitchen.gemv_awq_w4a16.default but no autograd formula was registered.

  Verify this from the installed implementation. The deployed AWQ inference path must not be assumed differentiable.

  Part 1: reconstruct the actual quantization run
  -----------------------------------------------

  You performed or designed this quantization, so use any surviving run transcript, command, local files, Hugging Face cache metadata, model workspace, logs, and exact source code that are available. Do not fill gaps from memory or from the
report’s prose.

  Determine, with evidence:

  1. The exact command line used, including:

     - model path;
     - output path;
     - --num_samples;
     - --max_seq_length;
     - --local_dataset_jsonl;
     - environment variables;
     - llm-compressor, transformers, compressed-tensors, PyTorch and CUDA versions;
     - exact code revision used, if it differed from the preserved script.

  2. The exact calibration population:

     - total requested rows;
     - total successfully processed rows;
     - unique rows after any fallback duplication;
     - count from StellarVoyager/H3-IR;
     - count from the local extracted MP4 metadata;
     - count from oakmindai/minimax_h3_avatar_500;
     - any other source;
     - source dataset revisions or cached commit hashes;
     - source row indexes or stable identifiers;
     - random seed and sampling order;
     - download, image-decode, ffmpeg, or processor failures;
     - whether fallback duplication occurred and which rows were duplicated.

  3. The actual modality distribution:

     - text-only;
     - one-image;
     - multiple-image;
     - first/last image pair;
     - video input;
     - first frame extracted from a generated MP4;
     - real target video/audio, if any.

  4. The actual H3 task/syntax distribution:

     - T2VA;
     - I2VA;
     - FL2VA;
     - Ref2VA;
     - dialogue;
     - lyrics;
     - captions;
     - cutoff;
     - subject/picture bindings;
     - multilingual dialogue.

  For each special token, report:

     - number of samples containing it;
     - total token occurrences;
     - number of unique prompt contexts;
     - whether it tokenized to the intended dedicated ID;
     - whether both members of each paired marker were present and balanced.

  5. The exact preprocessing applied during calibration:

     - actual still-image pixel bounds;
     - resize algorithm;
     - number of images retained from multi-image source rows;
     - whether video preprocessing ran at all;
     - truncation behavior;
     - actual sequence-length distribution after tokenization;
     - whether add_generation_prompt=True changed the H3 conditioning format.

  6. Whether the run calibrated all 64 decoder layers or only the 50 H3-consumed layers.

  7. Whether the run recorded any BF16-versus-AWQ hidden-state measurements. Distinguish a real model forward and numerical comparison from packaging, shape, tokenizer, or processor checks.

  Part 2: reconcile the preserved report with the implementation
  --------------------------------------------------------------

  There are conflicting claims that must be resolved:

    - the report says 512 calibration samples in four 35/25/25/15 task buckets;
    - the visualization says 256 samples from three sources at roughly 53/31/16;
    - the preserved script defaults to 96 samples and computes approximately
      52/31/remainder across H3-IR/local/avatar;
    - the script appears to pass exactly one image per prompt;
    - the report describes text-only T2VA, FL2VA image pairs, video blocks and four task buckets;
    - the report says 350 quantized linears while also saying all 64 layers were quantized;
    - the artifact contains 448 source linears, while the Comfy adapter retains 350;
    - the report says 40 attention heads, while the embedded config says 64;
    - the report’s still-image pixel bounds differ from processor_config.json;
    - test_layer50_drift.py does not appear to load a model or compute drift;
    - validate.py does not appear to compare BF16 and AWQ hidden states;
    - runtime prose refers to Marlin, while the actual Comfy path uses comfy-kitchen.

  Produce a table with these columns:

    Claim
    Claimed by
    What the code does
    What the artifact proves
    What the run evidence proves
    Verdict: verified / contradicted / unknown
    Required correction

  Do not quietly harmonize conflicting numbers. If the actual run cannot be reconstructed, say precisely which facts are unrecoverable.

  Also identify any technically incorrect descriptions of AWQ, such as Hessian/covariance language or “exact numerical alignment,” and replace them with narrowly accurate wording.

  Part 3: build a reproducible calibration manifest if possible
  -------------------------------------------------------------

  If the original sample population can be reconstructed, create:

    internal/gemini/2026-08-24_awq_calibration_manifest.jsonl

  Do not copy media. Each row should contain only reproducibility/provenance data:

    manifest_version
    source_name
    source_revision
    source_row_id_or_index
    source_split
    local_relative_path where appropriate
    prompt_sha256
    sanitized prompt or prompt-relative-path
    image/video/audio hashes
    media dimensions
    modality count
    H3 task family
    special token occurrence counts
    dedicated token IDs actually emitted
    pre-tokenization text length
    post-tokenization sequence length
    selected_for_calibration
    duplicate_of, if applicable
    failure_reason, if applicable

  Do not expose private absolute paths in the manifest. Use repo-relative paths, storage-neutral identifiers, hashes, or explicitly redacted values.

  If exact reconstruction is impossible, do not fabricate the manifest. Instead provide:

    - the largest reconstructible subset;
    - the missing information;
    - a deterministic manifest-producing modification for the next quantization run.

  Part 4: assess whether this data can support special-token post-training
  -----------------------------------------------------------------------
  Separate three uses of the data:

  1. AWQ calibration.
  2. Encoder representation distillation.
  3. H3 diffusion post-training.

  For each source and use, state whether it is suitable and why.

  In particular:

    - prompt/reference pairs may be useful for encoder representation training;
    - generated MP4 frame-zero images paired with their prompts may be useful as H3-shaped contexts;
    - they are not automatically real target video/audio training examples;
    - a frozen-DiT diffusion objective requires target latents, the true H3 joint video/audio target, timestep/noise records, and correct loss weighting;
    - do not treat an inference sampler as a faithful implementation of the training loss;
    - identify licensing or data-rights limitations rather than assuming them away.

  Report token-family coverage and whether the existing material is sufficient for:

    - <d> / </d>;
    - caption start/end;
    - lyrics start/end;
    - cutoff.

  If coverage is poor, propose the smallest synthetic-context supplement appropriate for representation distillation only. Do not call synthetic prompt strings diffusion training data.

  Part 5: design the smallest defensible 4090 experiment
  ------------------------------------------------------

  The goal is to discover what post-training changes, not to claim a universal improvement.

  Design a staged experiment with these requirements:

  Stage A: no-training baselines

  For each chosen token family, compare:

    - release dedicated IDs;
    - legacy BPE tokenization;
    - stripped markers;
    - for captions, the documented quoted-prose route and the marker-tagged route.

  Use the frozen AWQ encoder and frozen H3 pipeline. Keep prompt prose, references, scheduler, sampler, resolution, duration and model weights fixed.

  Use multiple seeds for perceptual or generation-level conclusions. Do not interpret one divergent sample as a quality comparison.

  Stage B: sparse overlay infrastructure

  The trainable object must be a sparse row delta:

    base embedding lookup, frozen
    replace selected IDs with base_row + trainable_delta
    frozen language layers 0–49
    no final RMSNorm
    no LM head

  Required artifact metadata:

    token strings and resolved IDs
    base-row hashes
    full checkpoint hash
    tokenizer/config hashes
    delta tensor and dtype
    training code revision
    objective configuration
    training-data manifest hash

  The loader must have a literal off state that reproduces the original encoder bit-for-bit and must reject provenance mismatches.

  Do not make the full [151936,5120] embedding table trainable. Do not rely on gradient masking plus AdamW.

  Stage C: representation-level pilot

  Evaluate at least these possible objectives:

    1. Legacy-BPE representation transplantation.
    2. Natural-language/gloss transplantation.
    3. Contextual constraints on aligned ordinary-token layer-50 states.
    4. A delta-radius constraint.

  Explain which question each objective answers and why it might harm compatibility with a frozen DiT.

  Because captions already work, do not frame caption training as a repair. Caption may be:

    - a particularly observable evaluation family because OCR provides a concrete output;
    - a high-risk family to alter if the DiT already learned the released row’s fixed code;
    - a useful falsification case for the hypothesis that “more semantic” embeddings help.

  Recommend whether the first pilot should train:

    - dialogue pair only;
    - caption pair only;
    - one four-token dialogue+caption overlay;
    - or all seven rows.

  Optimize for attribution and feasibility, not for maximum parameter count.

  Stage D: H3 task-level evaluation

  Only propose direct diffusion-loss training if you can identify the exact H3 training objective from authoritative code. Otherwise keep the first experiment representation-level and treat H3 generations as evaluation.

  Suggested family-specific measurements:

  Dialogue:
    lip sync
    active-speaker correctness
    ASR/WER
    voice-activity boundaries
    unwanted duplicated speech
    multilingual behavior

  Captions:
    subtitle presence
    OCR exact/normalized match
    correct number of lines
    association with the correct dialogue span
    unwanted extra text
    caption persistence/timing
    multi-seed human review

  Lyrics:
    sung versus spoken behavior
    lyric fidelity
    vocal presence
    boundary behavior
    instrumental controls

  Cutoff:
    incomplete versus naturally completed utterances
    endpoint timing
    completed controls
    exact resolution of <cutoff> versus <|cutoff|>

  Part 6: solve or bound the AWQ autograd prerequisite
  ---------------------------------------------------

  Inspect the installed comfy-kitchen AWQ implementation and the local adapter.

  Determine the most practical way to propagate gradients only to the input embedding delta while all W4 weights remain frozen.

  Evaluate:

  1. Registering a training-only autograd formula for gemv_awq_w4a16:

       forward: existing quantized kernel
       backward for x only: grad_x = grad_output @ W

     Consider chunked dequantization over output rows so a full BF16 weight matrix is not held persistently. No gradients are needed for qweight, scales, zeros or bias.

  2. A training-only eager dequantized F.linear fallback combined with layer/segment activation checkpointing.

  3. Loading the first 50 layers into some other differentiable 4-bit representation.

  For each option report:

    - correctness risk;
    - expected peak VRAM;
     - additional host RAM;
     - checkpoint conversion requirements;
     - forward/backward speed;
     - implementation complexity;
     - whether Qwen attention and other Comfy execution paths remain autograd-compatible.

   Provide a minimal red/green validation design:

     - compare grad_x against a small fully dequantized BF16 reference;
     - compare forward outputs;
     - confirm no weight or scale receives a gradient;
     - confirm only selected embedding rows receive gradients;
     - confirm marker-free prompts are bit-identical with an enabled zero overlay;
     - perform a one-layer gradient smoke before attempting the full 50-layer encoder.

   Do not claim the 4090 training path is feasible until you have a concrete memory budget for:

     resident retained AWQ weights
     one checkpointed layer’s temporary dequantized weights
     activations
     attention workspaces
     optimizer state for the sparse delta
     CUDA overhead

   Joint encoder-plus-DiT residency is not the assumed path. If you discuss a frozen-DiT objective, analyze an alternating strategy:

     1. compute current conditioning;
     2. unload encoder;
     3. load DiT and obtain dL/dC with conditioning treated as a leaf;
     4. unload DiT;
     5. reload/recompute the encoder and backpropagate the saved dL/dC into the row delta.

   State whether that gradient is mathematically valid, what must remain identical between stages, and whether the runtime makes it practical for solo experimentation.

   Required deliverables
   ---------------------

   Create:

     internal/gemini/2026-08-24_awq_calibration_forensics_and_special_token_training.md

   It must contain:

     1. Executive verdict.
     2. Actual-run reconstruction.
     3. Claim reconciliation table.
     4. Calibration manifest status.
     5. Token-family coverage.
     6. Suitability of the data for each type of training.
     7. Recommended first 4090 experiment.
     8. AWQ autograd feasibility and memory budget.
     9. Exact unresolved questions.
     10. A short list of proposed code changes, but no implementation yet.

   If reconstructible, also create:

     internal/gemini/2026-08-24_awq_calibration_manifest.jsonl

   At the end of your response, give me:

     - the five most important facts you verified;
     - the five strongest claims you had to retract or downgrade;
     - the exact information only I can supply;
     - your recommended first experiment in no more than ten steps.

   Evidence rules
   --------------

   Distinguish throughout:

     - artifact inspection;
     - source-code inspection;
     - surviving run evidence;
     - owner observation;
     - inference;
     - unknown.

  Do not cite the generated report as evidence for itself. Do not convert a plausible mechanism into a measured result. Do not use “exact,” “zero drift,” “protected,” “native,” “lossless,” or “validated” unless the corresponding comparison was actually executed.
