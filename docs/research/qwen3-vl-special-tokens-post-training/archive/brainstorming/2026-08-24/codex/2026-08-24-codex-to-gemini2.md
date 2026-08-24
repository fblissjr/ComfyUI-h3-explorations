Short answer: the runtime integration is careful, but Gemini’s assessment of the offline calibration is too reassuring. The checkpoint demonstrably works; the “LOW/NEGLIGIBLE” impact labels and “98–99% identical” prediction are unsupported.

   Claim                                       My verdict
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Single-image-only calibration               True; impact unknown, not established as low
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   Constrained image bounds                    True, but the token math and comparison bounds are wrong
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   Only a three-token assistant mismatch       Misleading; calibration used a different presentation contract
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   No BF16 drift benchmark                     True and important
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   BF16 vision/embeddings preserved            True
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   Fully resident with DiT headroom            False/misleading
  ──────────────────────────────────────────  ────────────────────────────────────────────────────────────────
   Requantization would be 98–99% identical    Unsupported

  ### 1. Single-image calibration matters despite the BF16 ViT

  Gemini’s reasoning stops one module too early. The ViT and DeepStack mergers are BF16, but their outputs enter the quantized language decoder. Every quantized Q/K/V/O and MLP projection processes residual streams affected by those visual
  features.

  Our runtime reference path can now present:

  - multiple independently processed images;
  - multiple vision spans with <Picture N> labels;
  - video sampled at 2 fps;
  - timestamped two-frame video blocks;
  - arbitrary ordered mixtures of image, video and audio labels.

  That behavior is visible in reference_conditioning.py:388 and the block patchifier in h3_awq_encoder.py:416. None of it appeared during AWQ calibration.

  AWQ depends on observed activation magnitudes, not merely whether the same channel indexes exist. Longer sequences, repeated vision spans, temporal patches, MRoPE positions and DeepStack injections can change which channels are outliers and
  by how much. The BF16 ViT prevents quantization error inside the ViT; it does not guarantee that the downstream W4 calibration generalizes.

  This is an unmeasured distribution gap—not evidence that the checkpoint is bad, but not safely classifiable as low impact.

  ### 2. The pixel/token explanation is incorrect

  The calibration did use 200,704–301,056-pixel still images. But Gemini mixes several different configurations:

  - MiniMax’s original still-image source config: 65,536–16,777,216 pixels.
  - Comfy’s native video-block defaults: 3,136–12,845,056.
  - Saved AWQ still-image processor: 200,704–301,056.
  - AWQ video processor: 4,096–25,165,824 across the sampled clip.

  The new AWQ node deliberately uses its saved 200k–301k configuration for still images at runtime, so this is both a calibration choice and a deployed preprocessing choice.

  Also, the language model did not see approximately 784 merged visual tokens. I instantiated the same processor on square, portrait and 16:9 images; it emitted 264–289 <|image_pad|> positions. The larger raw patch grid is reduced by the 2×2
  spatial merge before entering the language sequence.

  “A longer image activates the same dimensions” is also insufficient: AWQ scaling depends on activation magnitude and maxima across samples.

  ### 3. The presentation mismatch is larger than three assistant tokens

  Production H3 conditioning is explicitly not chat-templated. Core documents and implements:

  T2VA: raw prompt
  FL2VA: "<Picture 1>: " vision ["<Picture 2>: " vision] raw prompt
  Ref2VA: ordered labels/vision blocks/timestamps, then raw prompt

  See <HOME>/ComfyUI/comfy/text_encoders/minimax.py:1.

  Calibration instead used the HF chat template:

  <|im_start|>user
  vision block
  prompt
  <|im_end|>
  <|im_start|>assistant

So the mismatch includes:

  - user/chat framing;
  - assistant framing;
  - missing H3 <Picture N>: presentation;
  - one vision block for every row;
  - no raw text-only T2VA;
  - no multiple vision spans;
  - no timestamped video blocks.

  Counting the assistant suffix as a small percentage of sequence length does not establish negligible AWQ impact. A token repeated in every calibration row can influence channel maxima disproportionately.

  ### 4. The missing BF16 comparison is a genuine gap

  Gemini is correct that this is an evaluation gap rather than proof of defective weights. But the preserved “drift” scripts did not even perform the claimed quantized model forward: they load processors and inspect packaging/interface
  shapes.

  Successful ComfyUI renders establish that the adapter and checkpoint operate. They do not establish BF16 fidelity or justify “generation quality was not ruined.”

  ### What the new code did well

  The integration work itself did not take obvious shortcuts:

  - strict artifact/config/tensor validation;
  - correct 64-to-50-layer adaptation;
  - exact packed-weight view and scale-layout conversion;
  - BF16 visual tower and embedding retention;
  - artifact-owned image/video processing;
  - separate Qwen and VAE views for reference video;
  - ordered typed references, timestamps and two-frame blocks.

  The weak link is the calibration population, not the loader or reference compiler.

  Two other Gemini claims should be dropped:

  - The loader does not keep the entire H3 pipeline resident. Our own docs/h3_awq_encoder.md:254 records about 14.97 GB of staged encoder weights and explicitly says this is not a peak-VRAM guarantee or whole-pipeline residency. “Headroom for
    the DiT” is inaccurate; ComfyUI offloads models between stages.

  - There is no basis for predicting that a richer requantization would produce 98–99% identical W4 weights. That requires actually comparing scales and packed nibbles from two runs.

  Gemini’s latest correction is otherwise sound, with two wording changes:

  - The installed AWQ operation is currently missing an autograd formula; the underlying matrix operation is not inherently nondifferentiable.
  - Call Stage A “no-training,” not “zero-compute.” Multi-seed H3 rendering is expensive, and it can characterize H1/H2/H3 behavior but cannot definitively recover MiniMax’s historical tokenizer.

  My bottom line: keep using the checkpoint. There is no evidence requiring an immediate requantization. Before claiming the shortcuts were harmless, run an encoder-only BF16-versus-W4 layer-50 comparison using native H3 presentation across
  text-only, single-image, multi-image, FL2VA, video-reference and special-token cases. That directly tests the corners we actually cut.
