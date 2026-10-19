# Where ComfyUI's H3 path differs from the vendor's

Every known divergence between this ComfyUI install and the MiniMax H3 release, with what each costs a working user,
its native status, and what in this repo handles or watches it. A catalogue, not an authority: each section names the
page that owns its facts, and where they disagree the owner is right. The companion page,
[`custom_node_gaps.md`](custom_node_gaps.md), asks the same question of this pack's own nodes.

**Native status and local handling are separate facts.** "Native" means the installed ComfyUI checkout without this
pack. A node, check, preflight rule or shipped graph here that works around a gap makes this repo's graphs safe
without closing the gap for anyone else, so it is **locally handled**, never "closed". A gap with nothing in the last
column is one nothing is watching.

Sources: **measured** means a record under `bench/results/` or `data/results/`, named where it applies; **read**
means `coderef/sglang` (at `a7ec6b97f7`, the 2026-08-21 reads at `a41da991c8`) and `coderef/diffusers` against
ComfyUI's own files, an inference, not a run. Two kinds of gap are fixed differently: **config** (the release ships a
value in a file that sglang reads through `AutoTokenizer` / `AutoProcessor` from the paths `model_index.json` names,
`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py:70`, and ComfyUI
hardcodes its own; cheap, read the file) and **behavioral** (sglang decided something we did not; copying it is a
design choice).

## The gaps, by what they cost

| # | gap | kind | native status | handling or enforcement here |
|---|---|---|---|---|
| 1 | seven special tokens absent from the tokenizer | config | **fixed** (PR 15808, installed `924743af`) | `bench/audit_h3_marker_tokenization.py` requires and verifies native behavior |
| 2 | reference video frame rate assumed, not enforced | behavioral | open | typed nodes normalize from loader metadata; graphs keep `force_rate=24`, checked by `bench/check_ref_prompt_labels.py`; `bench/check_reference_runtime.py` |
| 3 | reference still floor (`min_pixels`) | config | open | preflight reports it; **no runtime parity** |
| 4 | reference still ceiling (`max_pixels`) | config | open; binds no shipped graph | opt-in `image_policy=release`, **off by default** |
| 5 | reference soundtracks not truncated | behavioral | open, measured | typed compiler's derived cap on every shipped graph (`check_reference_runtime.py`); preflight reports native socket graphs |
| 6 | reference media never upscaled, and never reported | behavioral | open | the conditioner's preview reports what each reference reached; opt-in `video_policy=release` |
| 7 | more than two audio channels silently cut to two (mono is not a gap) | behavioral | open | typed boundary refuses; audit only (`bench/audit_ref_audio_channels.py`) |
| 8 | video VAE encode precision, and posterior mean against sample | behavioral | open | measured only; **nothing enforces a choice** |
| 10-13 | partition gate, AdaLN cache, CUDA graphs, step caching | architectural | differences, not defects | researched or declined |
| 14 | DiT fp32 island collapses to bf16 under the int8 load | behavioral | open, core-side (`comfy/ops.py`) | **nothing**; measured, 5-20x below the int8 error already present |
| 15 | reference-audio encode crashes under VRAM pressure instead of tiling | behavioral | open, core-side (`comfy/sd.py`) | **nothing**; trimming makes it rarer |
| 16 | the generic VAE crop narrows the reference waveform's sample axis | behavioral | open, core-side; [Comfy-Org/ComfyUI#15972](https://github.com/Comfy-Org/ComfyUI/pull/15972) proposes the one-line fix | **nothing**; the shipped trim lands on a non-aligned length, so every soundtracked graph is exposed |
| 17 | the gated residual rounds once in core and twice in the reference | numerical | open, core-side | **nothing**; read, unmeasured |

Gap 9, H3 VAE tiling as a silent runtime branch, is not a gap: `MiniMaxH3VideoVAE` owns a fixed spatial tile and
overlap policy, always uses its H3 temporal chunking, and routes `encode_tiled`/`decode_tiled` back through the same
implementation, so a generic OOM retry does not switch H3 to another algorithm.

**Where the reference path stands overall.** For the stock path the still floor is the most reachable boundary and
the clip-wide video budget a bounded source-and-length divergence; this repo exposes the video choices as policies.
The VAE mean-against-sample question is independent. None blocks the typed path.

## 1. Seven special tokens: fixed natively

Owner: [`docs/research/2026-08-21_official_weights_metadata.md`](../research/2026-08-21_official_weights_metadata.md).

The release declares twenty `additional_special_tokens`; ComfyUI's bundled `qwen25_tokenizer` declared thirteen. The
seven missing were `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`,
`<|caption_end|>`, which hit every dialogue prompt (the guide mandates `<d>` around spoken lines). BPE fused the
fragments with the text either side, so a marker retokenized the sentence before it:

| | release emits | ComfyUI emitted |
|---|---|---|
| `<d>[English]` | `<d>` `[` `English` | `·<` `d` `>[` `English` |
| `window.<\|cutoff\|>` | `window` `.` `<\|cutoff\|>` | `window` `.<` `\|` `c` `utoff` |
| `<\|lyrics_start\|><d>` | two tokens | six fragments, fusing into `><` |

Measured 2026-08-22 across nine prompt shapes
([`../../data/results/2026-08-24_h3_marker_tokenization_native.json`](../../data/results/2026-08-24_h3_marker_tokenization_native.json)):
in an ordinary two-person dialogue prompt, 9 of 91 non-marker tokens came out different.

**Fixed by [Comfy-Org/ComfyUI PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808)** (kijai), merged and in
this checkout as `924743af`: it declares the seven on a `Qwen3VLSDTokenizer` subclass, so every consumer gets them,
core's `MiniMaxH3ReferenceToVideo` included. An older ComfyUI still has the defect. Against the release tokenizer all
nine scenes reproduce its ids exactly and a marker-free prompt is byte-identical to the reconstructed legacy arm;
[`../../bench/audit_h3_marker_tokenization.py`](../../bench/audit_h3_marker_tokenization.py) is the harness and
refuses an install that lacks the fix. No tokenizer patching remains in this pack.

**The seven do not touch the vision tower.** The tower consumes pixel patches; the vision sentinels are hardcoded ids
(`comfy/text_encoders/minimax.py:29-30`); the seven append above every existing id, so none shifts. Over two images,
an odd-frame video and an audio reference, a marker-free prompt tokenizes byte-identically with and without the fix.
The release's `processor/` bundles a tokenizer config and the image and video preprocessor configs; PR 15808 fixes
the text half, and gaps 3 and 4 are the vision half, untouched. The installed-code walk is in
`docs/research/2026-08-21_official_weights_metadata.md`, "How the fixed install realises them".

| what | where |
|---|---|
| the merged fix | [PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808), `comfy/text_encoders/minimax.py` at `924743af` |
| tokenization audit, nine scenes plus reference integrity | [`../../bench/audit_h3_marker_tokenization.py`](../../bench/audit_h3_marker_tokenization.py) |
| encoder-level companion | [`../../bench/grade_h3_marker_tokens.py`](../../bench/grade_h3_marker_tokens.py), [record](../../data/results/2026-08-21_h3_marker_token_states.json) |
| are the embedding rows trained | [`../../bench/audit_h3_token_embeddings.py`](../../bench/audit_h3_token_embeddings.py) |
| the release's declared list, never retyped | [`../../vendor_config.py`](../../vendor_config.py), guarded by [`../../bench/check_vendor_config.py`](../../bench/check_vendor_config.py) |

The audit's controls: a marker-free prompt must tokenize identically across the release, native and legacy
tokenizers, and the native arm must reproduce the **release** tokenizer exactly, not merely differ from legacy.

## 2. Reference video frame rate, and what the Qwen tower is handed

Owner: [`references.md`](references.md), "Frame rate", which has the mechanism, the measured table (a 30 fps source
read as 25% longer) and the workaround.

Both implementations target 24 fps and only sglang enforces it. The 2 fps conditioner subsample steps by `FPS // 2`
and stamps `i / 2.0`, right at 24 and wrong at anything else. Locally handled: `MiniMaxH3AppendRefVideo` requires the
frames' `VHS_VIDEOINFO` and records `loaded_fps`, `MiniMaxH3ReferenceConditioning` normalizes to 24 before Qwen or the
VAE sees the frames, and every shipped reference graph also keeps `force_rate=24`
([`../../bench/check_ref_prompt_labels.py`](../../bench/check_ref_prompt_labels.py) fails the build otherwise). The
native socket node has no metadata input.

**The per-pair vision call is not a gap, tested.** Core presents a reference video to Qwen3-VL as separate two-frame
calls at `grid_thw = [1, H, W]` where LightX2V and sglang pass the whole clip at `[T, H, W]`.
`internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md` section 2 argued structural equivalence without
a tensor test; [`../../bench/grade_video_block_presentation.py`](../../bench/grade_video_block_presentation.py) runs
both through the real `Qwen3VLVisionModel` at the `qwen3vl_32b` vision config and the merged and deepstack outputs
agree to 7.5e-06, inside fp32 reassociation noise. Two controls show it can see a difference (reordering pairs moves it
by 7.0; perturbing one pair by 1e-3 moves it by 6.0e-03), and it exits non-zero if either fails. The mechanism:
`cu_seqlens` splits attention at every `h*w` (`comfy/text_encoders/qwen35.py:660-663`), so attention never crosses a
frame in either presentation. Random weights are the right instrument: the claim is about routing, not anything
learned.

**The Qwen tower gets a second, duration-aware resize the VAE never sees.** The decoded array is shared by Qwen and
the VAE in sglang
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:382`),
but one call later
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:488`
sends the 2 fps view through the release `Qwen3VLVideoProcessor`, while the VAE consumes the shared array directly.
Shared source is not identical tower resolution (found 2026-08-22 by executing the release processor, after a
docstring-based claim to the contrary was refuted by tracing the call graph). Its budget is clip-wide:
`vendor_config/video_preprocessor_config.json` sets `size.longest_edge` to 25,165,824 px across the clip, so more
sampled frames means each is smaller. Against the real processor at 31 sampled frames (362 target frames):

| fed to the processor | `grid_thw` | effective | Qwen rows |
|---|---|---|---|
| 1344x768 | `[16, 42, 74]` | 1184x672 | 12,432 |
| 960x544 (the measured source) | `[16, 34, 60]` | 960x544, untouched | 8,160 |

The pass has a threshold, swept against the real processor (a first derivation was wrong twice). At 1344x768:

| sampled frames | `grid_thw` | effective |
|---|---|---|
| 24 | `[12, 48, 84]` | unchanged |
| 25 | `[13, 48, 84]` | unchanged (13 blocks, but `smart_resize` budgets it as 24) |
| **26** | `[13, 46, 80]` | **1280x736, first resize** |

Sampling is `ceil(N / 12)`, so 26 samples start at 301 target frames, and the first legal length at or above it is 311.
At the un-upscaled 960x544 source the first resize would be at a target outside the legal range, so the pass never
activates on that clip as core feeds it; there the whole divergence is gap 6, the upscale. So:

- **124 to 294 frames**: only the upscale differs between the two paths.
- **311 frames to the ceiling**: the upscale and the duration-aware limit both apply.

**The duration-aware pass is a cost limiter, which couples gaps 2 and 6.** Upscaling to 1344x768 and feeding Qwen per
pair as now gives 84x48/4 = 1,008 tokens a block x 16 = 16,128 rows, 3,696 more than the vendor's 12,432. Both or
neither: that is why `video_policy=release` owns the two stages together
([`reference_sizing.md`](reference_sizing.md), "Video and audio").

## 2b. Token tags under forced CLIP scheduling: fixed in core

Owner: [`docs/research/2026-08-21_conditioning_nodes.md`](../research/2026-08-21_conditioning_nodes.md). Both branches of
`CLIP.encode_from_tokens_scheduled` now merge the encoder's third return value, so `minimax_token_tags` reaches
conditioning either way (core `6bfaacc6`, #16400, 2026-09-20). No shipped graph wires a CLIP hook.
`bench/check_reference_contracts.py`'s case 5c watched the old state and was retired under its own contract
([`../state/decisions.md`](../state/decisions.md), 2026-09-25).

## 3 and 4. The still preprocessor bounds

Owner: [`reference_sizing.md`](reference_sizing.md), "The encoder's pixel bounds", which has the bounds table and the
measured ceiling sweep. Native ComfyUI does not load the released image or video preprocessor configs; it leaves
`min_pixels` / `max_pixels` on the shared Qwen2-VL helper's defaults and applies them to stills and to each two-frame
video block independently. The patch geometry is right on both sides.

**3, the floor, is the higher-priority half.** The release enlarges anything under 65,536 pixels (256x256); core's
floor is twenty times lower. A 200x200 thumbnail lands at 192x192, 36,864 pixels, and Qwen gets fewer patches than
the vendor intends. Thumbnails and cropped faces land here easily, and nothing about it looks extreme. Both towers
agree with each other, so it is a ComfyUI-against-release split. Preflight reports when the release floor would
enlarge a reference; nothing implements the floor at run time.

**4, the ceiling, is real but extreme.** Past 12,845,056 pixels core's helper resizes what the VAE already received,
so the two towers see different resolutions of one picture (a bilinear downscale with slight aspect drift, not
corruption). Measured 2026-08-22 with the real `process_qwen2vl_images`:

| source | mode | both towers get | Qwen sees | divergent |
|---|---|---|---|---|
| 3840x2160 | match | 1344x768 | 1344x768 | no |
| 6144x2048 | max | 6144x2048 | 6144x2048 | no |
| 6656x2048 | max | 6656x2048 | 6432x1984 | **yes** |
| 7168x2048 | max | 7168x2048 | 6688x1888 | **yes** |
| 4096x4096 | max | 2048x2048 | 2048x2048 | no |

Unreachable under `match` (about 1 MP at 1344x768); under `max` at a 2048 short edge it needs 3.0625:1 or wider.
**No shipped graph reaches it**
([`../../data/results/2026-08-21_shipped_reference_bounds.json`](../../data/results/2026-08-21_shipped_reference_bounds.json)).
Locally, `MiniMaxH3ReferenceConditioning.image_policy` selects whose ceiling applies, `comfy` (default) or `release`,
and pre-applies it before the VAE so both towers stay on one resolution; on the shipped encoder the two give the same
geometry at every legal short edge (`data/results/2026-08-29_qwen_view_under_snapshot.json`). Off by default, so the
native node and every graph on `comfy` remain exposed.

For reference **video**, one canvas-sized frame is below either maximum, but the release budget accumulates over the
clip while core restarts it per block: the boundary in section 2.

| what | where |
|---|---|
| where the ceiling resizes, against the real helper | [`../../bench/measure_qwen_bounds_bite.py`](../../bench/measure_qwen_bounds_bite.py), [record](../../data/results/2026-08-21_qwen_bounds_bite.json) |
| does any shipped graph hand Qwen a reference it shrinks | [`../../bench/audit_shipped_reference_bounds.py`](../../bench/audit_shipped_reference_bounds.py), [record](../../data/results/2026-08-21_shipped_reference_bounds.json) |
| the release's bounds, read not retyped | [`../../vendor_config.py`](../../vendor_config.py) `image_pixel_bounds()` / `video_pixel_bounds()` |

The instruments report; neither refuses.

## 5. Reference soundtracks not truncated

Owner: [`references.md`](references.md), "Audio references", which has the per-path table and the rotary-cursor
consequence.

Core encodes the whole waveform at 80 rows per second of excess; sglang cuts every reference soundtrack, a video's and
a standalone one alike, to the generated duration via `ffmpeg -t`, and diffusers does the same. The excess rows are
attended every step, and an over-long soundtrack also pushes the target streams down the timeline (both independent
reviews: `internal/codex/2026-08-21_h3-conditioning-qwen-independent-review.md` section 5.3; the cursor formula in
`internal/gemini/minimax_h3_comfyui_end_to_end_trace_and_gap_analysis.md`). **Measured at the real entry point,
`_encode_ref_audio`, 2026-08-29**, against the default 124-frame target (5.167 s, `audio_t` 207):

| soundtrack | `ref_audio_t` | rows | excess over the target's 414 |
|---|---|---|---|
| 5 s | 200 | 400 | 0 |
| 15 s | 600 | 1,200 | **786** |

Both are whole seconds, which is why this measurement could not see gap 16: an aligned input is exactly the one on
which that crop is a no-op. Locally handled: the typed compiler derives the duration from the aligned frame count and
slices every soundtrack internally, so there is no `TrimAudioDuration` widget to drift. VHS's video loaders also cap
at `frame_load_cap / force_rate`, so the gap bit a video soundtrack only when `frame_load_cap` was 0; it is real for
standalone `ref_audio_*` on any loader. Core remains exposed in a hand-built socket graph; preflight warns there.

## 6. Reference media never upscaled, and never reported

Owner: [`reference_sizing.md`](reference_sizing.md). Where the source has fewer pixels than the canvas, core uses the
source size rounded to 32; all three vendor implementations put it on the canvas rule.

**The clamp is a defensible default; the silence is the defect.** Reference rows are attended every step, so a larger
reference taxes the whole render: upscaling one 1024x1024 still to 2048 costs 6,144 tokens, half in each copy, and a
960x544 reference video at 124 frames is already 18,870 rows. The cost is measured and large; the benefit is not
measured at all, and a rendered pair could not show it (two arms differing in reference size are two samples). That
argues for the current default. What was missing was any report of what a reference finally reached, so a deliberate
downscale and an accidental one looked the same; this pack's conditioner now shows each reference's size and cost on
the node, and core's still says nothing. "H3 handles it fine" is not evidence that nothing was lost: anyone reporting
it has seen their references at one size only. Settling it would take a blind session under
[`../method/eval_comparison.md`](../method/eval_comparison.md) section 3, which nobody has asked for. Output will not
reproduce the vendor's for the same inputs, which matters only if that is the goal.

| what | where |
|---|---|
| opt-in release video policy | `MiniMaxH3ReferenceConditioning.video_policy=release`, controlled by [`../../bench/check_reference_runtime.py`](../../bench/check_reference_runtime.py) |
| shipped default | `video_policy=comfy`: core's no-upscale VAE geometry and per-pair Qwen processor |

## 7. Audio channels: mono is not a gap; more than two is

Owner: [`references.md`](references.md), "Known limitations, collected". Mono works and always did: core's
`comfy.sd.VAE.encode` runs `vae_encode_crop_pixels`, and the H3 audio VAE declares
`output_channels = 2, pad_channel_value = "replicate"` (`comfy/sd.py:1030-1035`, set in `57500fc5`, the commit that
added H3), so a mono waveform encodes to `[1,32,2,40]`, the 80 rows `PackedLayout` allocates, as exact dual-mono;
sglang's `-ac 2` does the same. The same crop takes `pixels[..., :2]` from a wider input, so a 5.1 reference conditions
on its first two channels with no warning. Measured on the real VAE
([`../../data/results/2026-08-29_ref_audio_channels.json`](../../data/results/2026-08-29_ref_audio_channels.json),
[`../../bench/audit_ref_audio_channels.py`](../../bench/audit_ref_audio_channels.py)). The typed `_prepare_audio`
refuses more than two channels; nothing watches core. Preflight reports channel counts on a real graph's media, and
"unreadable" is its own state.

## 8. The VAE encode question

Owner: [`sglang_comparison.md`](sglang_comparison.md). sglang keeps the video VAE fp32-resident
and decodes in fp16 autocast, because the VAE also encodes keyframes. The decode half was settled on 2026-08-10
(2-3x cost, no established benefit; the vendor does not decode in fp32 either). The open half is encode: reference
stills, reference videos and keyframes, computed once per render, whose job is identity fidelity. `--fp32-vae`
cannot express the split; `MiniMaxH3VAEPrecision` can.

[`../../bench/grade_vae_encoder_precision.py`](../../bench/grade_vae_encoder_precision.py) grades the encoder at the
call ([`../../data/results/2026-08-21_vae_encoder_precision.json`](../../data/results/2026-08-21_vae_encoder_precision.json)):
fp16 is bit-identical to itself, fp32 moves the latent, bf16 moves it far further (the first and last are controls).
Whether the delta is visible is open, tangled with the mean-against-sample difference
([`reference_sizing.md`](reference_sizing.md), "The vendor image path"; `comfy/ldm/minimax/vae.py:685`); an encode
instrument has to separate the two. The probe arms `workflows/h3_probe_ref_vae_encoder_fp16_api.json` and
`workflows/h3_probe_ref_vae_encoder_fp32_api.json` render and assert nothing. Whether the shipped default should follow
the graded result is a decision, not a measurement. The shipped video VAE is `h3_config.MODELS["video_vae"]`.

## 10-13. Architectural, not user-facing

Owner: [`sglang_comparison.md`](sglang_comparison.md), which carries the reasoning.

- **No partition concept.** sglang reads `partition` from `model_index.json` and refuses a task the checkpoint does
  not serve; here a plain t2v graph loads `ref2va` and renders. A ref2va loss at plain t2v is *not a t2v model by its
  own release metadata*, not a defect.
- **Exact AdaLN cache.** Buys unpruned accuracy at pruned memory, not speed; downstream of
  `docs/state/open_experiments.md` #22.
- **Breakable CUDA graphs.** Their quality gate requires them off, and H3's attention runs eager under it. Core has a
  per-block equivalent (`comfy/model_prefetch.py`) the H3 DiT loop does not enable; the research page says why it is
  not worth asking for on this box.
- **Step caching.** Priced and declined 2026-08-20: their speedup came from skipping steps at 50 steps on four H200s;
  at 4 steps there is nothing to skip.

sglang's audited `quality="high"` deployment is a 4xH200 server with sequence parallelism, and its own gate requires
quantization, `torch.compile` and layerwise offload all off: "sglang has it" is not on its own an argument.

## 14. The DiT's fp32 island collapses under the int8 load

Owner: [`docs/research/2026-08-29_comfyui_h3_t2va_trace.md`](../research/2026-08-29_comfyui_h3_t2va_trace.md) §1.5, with the vendor side in
[`sglang_comparison.md`](sglang_comparison.md).

Both implementations declare a small fp32 set inside a bf16 DiT. sglang enforces it with a named frozenset (both patch
projections, the time embedder, both output heads and `rope.inv_freq`;
`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:144-159`, dropping `time_embedder.*` on
a pruned checkpoint at `:2060-2066`). Core builds those layers with `dtype=torch.float32` and calls them "the
checkpoint's fp32 island" (`comfy/ldm/minimax/model.py:302`), but `MixedPrecisionOps.Linear.__init__`
(`comfy/ops.py:1300-1303`) discards the caller's dtype, so on `minimax_h3_fl2va_pruned_int8_convrot.safetensors` every
one loads **bf16** (verified by executing the load path). **Measured, small**: 6.1e-4 on `final_layer.video_out`,
8.9e-4 on `video_patch_proj`, 1.7e-3 on `adaln_proj.linear`, against the 8.8e-3 the int8 block linears already carry,
so no rendered difference should be attributed to it. Two things follow: `adaln_proj` is a strict regression (F16 on
disk to bf16 in memory), and a bf16-against-int8 checkpoint A/B moves four things at once, so
`docs/method/evidence.md` #22's 5.6-9.4% first-step velocity delta is a four-variable measurement. Core-side; nothing
here patches core, and no check asserts our fp32 set against the vendor's list.

**Not a gap, the inverse direction: text-encoder precision.** sglang encodes in bf16; core upcasts to fp32 at the
embedding (`comfy/sd1_clip.py:213`) and pins the compute dtype to match (`comfy/sd.py:269-270`), so an int8 encoder
never reaches an int8 GEMM here (`full_precision_mm=True`, `comfy/sd1_clip.py:114`). int8 costs no more time than
bf16 per resident layer and halves the PCIe transfer that dominates, for 0.88% weight error, so the shipped
`int8_convrot` encoder is the right choice over the local bf16 file
([`docs/research/2026-08-29_comfyui_h3_t2va_trace.md`](../research/2026-08-29_comfyui_h3_t2va_trace.md) §2.4 owns the numbers).

## 15. Reference-audio encode crashes instead of falling back under VRAM pressure

Owner: [`docs/research/2026-08-29_comfyui_h3_t2va_trace.md`](../research/2026-08-29_comfyui_h3_t2va_trace.md) §14 item 9. Found by accident
2026-08-29.

**A long reference soundtrack on a busy card is a hard crash, not a slow render**, and it happens exactly when things
are tight: the reference encode runs at graph start, while the DiT and text encoder may be resident. `comfy/sd.py:514`
defaults `extra_1d_channel` to `None`; both peer stereo VAEs set it to 16 (ACE Step at `:884`, LTX Audio at `:966`),
and the H3 audio branch (`:1030-1046`) sets `latent_dim = 2` and leaves it `None`. So `VAE.encode`'s OOM retry hands a
`[B, L, C]` audio tensor to `encode_tiled_`, the 2D image tiler, which indexes `pixel_samples.shape[3]`
(`comfy/sd.py:1166`). Measured: `vae.encode_tiled(torch.randn(1, 8000, 2))` raises `IndexError: tuple index out of
range`; seen live as a 15-second soundtrack that logged "Ran out of memory when regular VAE encoding, retrying with
tiled VAE encoding" and died in the retry. Setting `extra_1d_channel` on the H3 audio branch as the peers do would fix
it. Nothing here handles it; trimming the soundtrack (gap 5) only shrinks the encode.

## 16. The generic VAE crop narrows the reference waveform's sample axis

Owner: this page. Prompted by [Comfy-Org/ComfyUI#15972](https://github.com/Comfy-Org/ComfyUI/pull/15972), which
proposes the fix as `self.crop_input = False` on the H3 audio branch.

**A paired reference soundtrack is shifted up to 12.5 ms earlier against its own frames, and loses a latent step,
whenever its length is not a multiple of 800 samples, and this repo's default trim is such a length.**
`comfy/sd.py:515` defaults `crop_input` to `True` and the H3 audio branch (`:1030-1046`) never clears it, so
`vae_encode_crop_pixels` runs on every reference-audio encode. `_encode_ref_audio` presents `[1, samples, channels]`
(`comfy_extras/nodes_minimax_h3.py:77`), so the cropped axis is the sample axis, narrowed to a multiple of the audio
`downscale_ratio`, 800, taking `(samples % 800) // 2` off the front. The video reference is not shifted (its crop
covers only H and W, already multiples of 32), so the crop is live on audio and a no-op on video. Measured through
`_encode_ref_audio` with `crop_input = False` as the control
([`../../bench/audit_ref_audio_crop.py`](../../bench/audit_ref_audio_crop.py),
[record](../../data/results/2026-08-30_ref_audio_crop.json)):

| case | samples | dropped front | dropped back | `ref_audio_t` shipped | crop disabled |
|---|---|---|---|---|---|
| aligned | 64,000 | 0 | 0 | 80 | 80 |
| the PR's case | 437,333 | 266 | 267 | 546 | **547** |
| shipped trim | 165,333 | 266 | 267 | 206 | **207** |

`_prepare_audio` caps a soundtrack at `round(duration * sample_rate)` with `duration = frame_count / FPS`
(`reference_conditioning.py:563,224`), and the `length` default is 124 (`reference_conditioning.py:960`): 165,333
samples, remainder 533. That is structural: H3's 24 fps grid against 32 kHz never lands on 800. Core-side; the PR's
line is the fix. Rounding the local cap up to 800 would sidestep it at the cost of a second copy of the modulus, and
could not recover the leading samples of a soundtrack already shorter than the target.

## 17. The gated residual rounds once in core and twice in the reference

Core's `comfy/ldm/minimax/model.py::_mod_gate` folds `x + gate * out` into one in-place `addcmul_` on the bf16 stream,
one rounding; the diffusers reference computes `gate * out` and the add as two bf16 ops
(`coderef/diffusers/src/diffusers/models/transformers/transformer_minimax_h3.py`, the block forward), and sglang's
fused kernel rounds the product to bf16 before adding
(`coderef/sglang/python/sglang/kernels/ops/diffusion/modulate/indexed_modulation_triton.py::_indexed_gate_bf16_kernel`)
to match it. Twice per block, fifty blocks, every step; at most one bf16 ulp of the gated update per element. The
scale-shift path matches. Read 2026-09-04, **unmeasured**; our attention mirror replaces `Attention.forward` only and
never touches the residual. To size it: one block forward on a captured hidden state under both formulas, relative L2
of the stream after fifty blocks, at one step.

## Settled: recorded so nobody re-derives them

- **The 2 fps conditioner subsample**, index pad and merged-pair timestamp included: identical to sglang.
- **Patch geometry**: identical; core passes 16 rather than inheriting Qwen2-VL's 14.
- **qkv row permutation as the cause of the fp8/int8 fidelity gap**: refuted twice over. The fp8 file has a scalar
  `weight_scale` with no row to misalign, and the gap is uniform across module kinds.
- **Mono reference audio**: works on native core and always did (gap 7). Do not re-file it.
- **Core `421a1c24`** ([#15988](https://github.com/Comfy-Org/ComfyUI/pull/15988), 2026-09-08) scales each stream's
  velocity by its denoise mask (`comfy/ldm/minimax/model.py`, `MiniMaxH3Model.forward`), so a masked row's prediction
  matches the mask-scaled sigma the outer x0 conversion assumes. It reaches only a latent carrying a noise mask, which
  the freeze and masked nodes write ([`audio_freeze.md`](audio_freeze.md)); a server started before the checkout
  pulled it runs without it, so restart before a masked-latent render counts.
- **The quantized-matmul gating rework** ([#16185](https://github.com/Comfy-Org/ComfyUI/pull/16185), then
  [#16189](https://github.com/Comfy-Org/ComfyUI/pull/16189), 2026-09-08) leaves the H3 text-encoder path where it was:
  `comfy.ops.use_quantized_matmul` lifts the dequantize only inside `CLIP.generate` (`comfy/sd.py`), and the H3
  conditioners encode through `encode_from_tokens_scheduled` (`comfy_extras/nodes_minimax_h3.py`).
- **Adjacent checkpoint format.** Core natively loads the NVFP4-AWQ H3 encoder (H3 namespace, per-layer
  `comfy_quant` records, 350 language linears at `format=nvfp4`) through the stock `CLIPLoader`. A compressed-tensors
  W4A16 file with the Hugging Face `model.language_model.*` namespace is a different representation that core
  misdetects as Qwen3-VL-8B (reproduced 2026-08-23); this pack's adapter for it was deleted 2026-09-13 with the closed
  lane (`docs/state/decisions.md`). The generator refuses any encoder outside
  `workflows/h3_config.py::CORE_LOADED_ENCODERS`.

## What we do that they do not

sglang has a `sol_attn` attention backend and a `kitchen_int8` linear path dispatching the same
`comfy_kitchen.int8_linear` our checkpoints load through, and its consumer-card table's fastest row is an int8 DiT
with a sage-to-Sol hybrid: convergence, not an action item, and their numbers are not ours to import
([`sglang_comparison.md`](sglang_comparison.md), retracted against `803b4fb31c`). The SLA
router has no counterpart there; [`../attention/sol_attn.md`](../attention/sol_attn.md) owns its numbers.

## The contracts underneath

| what | where |
|---|---|
| core's reference-node contracts | [`../../bench/check_reference_contracts.py`](../../bench/check_reference_contracts.py) |
| ordered plan against core plus intended differences | [`../../bench/check_reference_order.py`](../../bench/check_reference_order.py) |
| typed runtime media and compiler contracts | [`../../bench/check_reference_runtime.py`](../../bench/check_reference_runtime.py) |
| label and preflight visibility of typed chains | [`../../bench/check_typed_reference_consumers.py`](../../bench/check_typed_reference_consumers.py) |
| the typed frame-rate and channel handling | [`../../reference_conditioning.py`](../../reference_conditioning.py) |
| the loader every graph wires | [`../../h3_encoder_loader.py`](../../h3_encoder_loader.py), controlled by [`../../bench/check_h3_encoder_loader.py`](../../bench/check_h3_encoder_loader.py) |

All seven contracts in core's `MiniMaxH3ReferenceToVideo` are controlled. The typed surface keeps the four that remain
semantic contracts and deliberately replaces suffix pairing and fixed modality grouping with ownership and list
position.
