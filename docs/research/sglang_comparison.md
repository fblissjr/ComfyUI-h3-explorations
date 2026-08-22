# sglang's H3 serving path against ours

last updated: 2026-08-22

What the vendor-side serving implementation does that this install does not,
what both do where ours may be the weaker version, and what looks like a gap
until you check the record and find it already priced.

**Read from source 2026-08-21** against `coderef/sglang` at commit
`a41da991c8`, alongside ComfyUI's `comfy_extras/nodes_minimax_h3.py`,
`comfy/ldm/minimax/model.py`, `comfy/model_management.py` and this repo's own
bench records. No renders were made for any of it. Every claim below says
which file it came from; nothing here is a measurement on this card unless it
cites a record in `bench/results/`.

**The clone has moved since that read.** `coderef/sglang` is at `a7ec6b97f7`
as of 2026-08-22. The index rows added that day were read at the new commit;
the prose sections below were read at `a41da991c8` and have **not** been
re-read. `bench/check_doc_links.py` confirms every cited line still exists,
which is not the same as confirming it still says the same thing. Re-read
before quoting an older section as current.

**Scope, and what the index is for.** This file owns the *optimization and
runtime* comparison. The reference-conditioning comparison — sizing, patchify,
presentation, packing, condition timestep — is owned by
[`h3_references.md`](../h3_references.md), section "The vendor image path,
stage by stage", and its detail must not be restated here.

That split kept every divergence documented and left none of them listed
together, so answering "what are all the gaps" meant knowing which of three
files to open. **The index below is the map: one line per divergence, pointing
at the file that owns it.** A row carries no numbers and no mechanism — those
belong to the owner and drift the moment they are copied. Adding a divergence
here without giving it an owner is how this table becomes a second, worse copy
of the docs it indexes.

---

## Every known divergence, and who owns it

Two kinds, and the distinction decides how a gap gets fixed. **Config
inheritance**: the release ships a value, sglang reads it through
`AutoTokenizer` / `AutoProcessor` from the paths `model_index.json` names
(`coderef/sglang/python/sglang/multimodal_gen/runtime/loader/component_loaders/component_loader.py:70`,
`:467`; consumed at
`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/stages/text_encoding.py:34`),
and ComfyUI hardcodes its own. These are the cheap ones: the fix is to read the
file. **Behavioural**: sglang decided something we did not, and copying it is a
design choice rather than a correction.

Priority is by what it costs a working user, not by how interesting it is.

| # | divergence | kind | practical impact | status (2026-08-22) | owner |
|---|---|---|---|---|---|
| **P1** | | | | | |
| 1 | Seven H3 special tokens absent from ComfyUI's tokenizer | config | Every dialogue prompt. Markers become literal text and the BPE fragments corrupt neighbouring words | **Fixed upstream by [PR 15808](https://github.com/Comfy-Org/ComfyUI/pull/15808), OPEN not merged.** Verified here against its own diff | [`official_weights_metadata.md`](official_weights_metadata.md) |
| 2 | Reference video frame rate assumed 24 rather than enforced | behavioural | Any non-24fps reference. Both target 24; sglang converts with an ffmpeg `fps=` filter, we assume. Motion timing and `<T.T seconds>` labels stretch, silently | Open. Workaround `force_rate=24`, gated by `bench/check_ref_prompt_labels.py` | [`h3_references.md`](../h3_references.md) |
| 3 | Image preprocessor **floor** (`min_pixels`) | config | Small references. The release enlarges them, we do not, so they are under-tokenized. No warning, and nothing about it looks extreme | Open, unenforced | [`h3_references.md`](../h3_references.md) |
| **P2** | | | | | |
| 4 | Image preprocessor **ceiling** (`max_pixels`) | config | Only very wide references in `max` mode. Past the crossing the VAE and Qwen see different resolutions of the same picture | Open. Zero shipped graphs reach it (`bench/results/2026-08-21_shipped_reference_bounds.json`) | [`h3_references.md`](../h3_references.md) |
| 5 | Reference soundtracks not truncated | behavioural | Wasted rows on any soundtrack longer than the render. Trim it yourself | Open by choice | [`h3_references.md`](../h3_references.md) |
| 6 | Reference video never upscaled | behavioural | Small reference videos condition on less than the vendor intends | Open. Costs ~5x the image fix | [`h3_references.md`](../h3_references.md) |
| 7 | Mono reference audio raises | behavioural | Hard crash rather than a bad render | Gated by `bench/check_mono_ref_audio.py` | [`h3_references.md`](../h3_references.md) |
| **P3** | | | | | |
| 8 | Video VAE **encode** precision, and mean-vs-sample | behavioural | Identity fidelity of every reference and keyframe, once per render | Precision half measured (`bench/results/2026-08-21_vae_encoder_precision.json`); mean-vs-sample untangled and no downstream evidence | this file, "Open after this read" |
| 9 | VAE tiling unrecorded | behavioural | A tiled decode cannot be distinguished from an untiled one afterwards | Open, unenforced | this file |
| **P4 — architectural, not user-facing** | | | | | |
| 10 | No partition concept | behavioural | A t2v graph loads `ref2va` and renders instead of refusing | Open by design; frames the checkpoint-swap arm | this file |
| 11 | Exact AdaLN cache | behavioural | Unpruned accuracy at pruned memory. No speed here | Downstream of [`open_experiments.md`](../open_experiments.md) #22 | this file |
| 12 | Breakable CUDA graphs | behavioural | None on one card | Their own quality gate requires it off | this file |
| 13 | Step caching | behavioural | None at 4 steps | Priced and declined 2026-08-20 | [`roadmap.md`](../roadmap.md) |
| **Settled — recorded so nobody re-derives them** | | | | | |
| 14 | 2 fps conditioner subsample, index pad, merged-pair timestamp | — | — | **Confirmed identical** to sglang | [`h3_references.md`](../h3_references.md) |
| 15 | Patch geometry (`patch_size`, `temporal_patch_size`, `merge_size`, mean/std) | — | — | **Confirmed identical**; ComfyUI passes 16 rather than inheriting 14 | [`h3_references.md`](../h3_references.md) |
| 16 | qkv row-permutation as the fp8/int8 fidelity cause | — | — | **Refuted** on two independent grounds | this file, "Refuted here" |
| 17 | What sglang does with a reference video shorter than its aligned target | — | — | **Read, not verified.** Cheap to close | [`h3_references.md`](../h3_references.md) |

### The processor directory carries two different things

Worth stating because it is a natural confusion and the answer differs for each
half. The release's `processor/` bundles a tokenizer config *and* the image and
video preprocessor configs, because HF's `AutoProcessor` wraps both halves. Row
1 is the text half; rows 3 and 4 are the vision half. **Fixing one does nothing
for the other.**

**The special tokens do not reach the vision tower, measured rather than
argued.** The tower consumes pixel patches and never a vocabulary; the vision
sentinels are hardcoded ids in `comfy/text_encoders/minimax.py`, not lookups;
the seven append *above* the highest existing added token, so no id shifts. The
assertion that closes it is the reference-integrity arm in
`bench/audit_h3_marker_tokenization.py`: over two images, an odd-frame video
and an audio reference, a marker-free prompt tokenizes byte-identically with
and without the fix, vision structure included. Had any vision id moved, that
arm would have failed.

So PR 15808 leaves rows 3 and 4 exactly where they were, and they are the rows
that actually change what Qwen sees.

---

## The filter: most of sglang's speed is four cards

sglang's audited deployment for `quality="high"` is a 4xH200 fl2va server with
`sp_degree=4` and `ulysses_degree=4`, and `validate_quality_deployment`
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:98-186`)
raises unless the resident server matches it exactly — down to asserting the
device is an H200 at compute capability 9.0. That same gate requires
`enable_torch_compile`, `enable_breakable_cuda_graph`,
`is_dit_layerwise_offload_selected` and `quantization` all to be **off**.

Two consequences worth holding on to. Their headline performance is sequence
parallelism we cannot copy on one card. And the knobs they ship but require
off for their quality claim are knobs they do not stand behind for quality
either, so "sglang has it and we don't" is not on its own an argument.

---

## What they do that we do not

### An exact AdaLN cache

`coderef/sglang/python/sglang/multimodal_gen/tools/build_minimax_h3_adaln_cache.py`
precomputes the modulation parameters for every timestep the schedule will
actually visit, and the DiT then drops `adaln_proj` entirely
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1412-1414`).
There is an in-process prepass that builds the same thing by reading all the
`adaln_proj` layers once
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:1186-1265`),
sized by a plan-width knob whose per-task requirement is stated at
`:1206` — t2va, fl2va and ref2va need different numbers of distinct timesteps
per step, which is the same fact our packing already encodes.

**Why this is interesting here.** It is an exact answer to the problem the
Comfy-Org "pruned" checkpoints answer approximately. The pruned file replaces
those weights with a rank-8 SVD of the time curve
([`evidence.md`](../evidence.md) owns that measurement); the cache keeps full
accuracy and pays a cache tensor sized by the step count instead of by the
weight matrix.

**What it does and does not buy on this box.** No speed. On the pruned file we
run, the AdaLN projection is already tiny, so there is no matmul left worth
eliminating. What it buys is *unpruned accuracy at pruned memory*.

**#22 has since reported, and it half-opened the gate**
([`bench/results/2026-08-21_pruning_sensitivity.json`](../../bench/results/2026-08-21_pruning_sensitivity.json)).
The pruning is **not** invisible at the output: the first-step velocity moves
5.6-9.4% against a determinism floor of exactly zero. So "the residual does not
matter" — the outcome that would have killed this outright — did not happen.
What did happen is that the effect is the same size on both checkpoints and
**smaller than the `fp8_scaled`-vs-`int8_convrot` difference this repo already
ships**, which is why it is still not urgent: an exact AdaLN would remove an
error that is not the largest one in the stack.

**What would make it worth building** is evidence that the difference is
*visible*, which no measurement here can supply — the arms are one forward, not
a render. That is a blind session on pruned against unpruned under
[`eval_comparison.md`](../eval_comparison.md) section 3, and it has not been run
or scheduled.


### Breakable CUDA graphs over a packed sequence

`coderef/sglang/python/sglang/multimodal_gen/runtime/breakable_cuda_graph/model_padders/minimax_h3.py`
pads the packed prompt to the model's sequence alignment and rewrites
`cu_seqlens_q`, `max_seqlen_q` and the position ids so one captured graph
serves varying lengths. That padding layer is the part that makes graph
capture possible for a packed-sequence model at all. ComfyUI has no equivalent
for H3. Note again that their own quality gate requires this off.

### Refusals at admission rather than degraded service

Not speed, but the design difference that shows up most often:

- The release partition gate
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/release_metadata.py:60-135`)
  reads `partition` from `model_index.json._minimax_h3` and raises when the
  requested task does not belong to it. A `t2va` request on the `ref2va`
  partition is refused, not served badly.
- `resolved_parallel_decode_mode()`
  (`coderef/sglang/python/sglang/multimodal_gen/configs/models/vaes/minimax_h3_video.py:42-56`)
  refuses `spatial`, `spatial_shard` and `patch` VAE decode as outside the
  released quality contract, before any large component downloads.

ComfyUI has no partition concept: a plain t2v graph loads `ref2va` and
renders. **This is the frame for the checkpoint-swap arm** — a `ref2va` loss at
plain t2v reads as *not a t2v model by its own release metadata*, not as a
defect in the checkpoint.

---

## What we both do, where ours may be the weaker version

### Video VAE precision, and why our existing note does not settle it

sglang keeps the video VAE **fp32-resident and decodes in fp16 autocast**, and
the config says why in as many words: the VAE also encodes keyframes
(`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py:59-62`,
`vae_precision: "fp32"`, `vae_decode_precision: "fp16"`).

Ours runs fp16 throughout. `comfy/model_management.py:1258-1273` returns fp16
on this card unless `--fp32-vae` is passed, and the default launch mode does
not pass it.

`<comfy>/start.sh` already carries a considered decision against `--fp32-vae`,
added and reverted 2026-08-10: measured 2-3x decode cost, no established
benefit, and the reference's own fp32 evidence was an *audio* VAE measurement
that ComfyUI already honours. That reasoning stands for what it addressed.

**What the sglang read adds is that the decision was scoped to decode, and the
vendor does not decode in fp32 either.** The vendor splits the two: fp32 for
residency and encode, fp16 for decode. So the 2-3x decode cost we measured and
rejected was never the vendor's behaviour, and the open half of the question is
the *encode* side — reference images, reference videos and keyframes, whose
whole job is identity fidelity, computed once per render rather than per step.

`--fp32-vae` cannot express that split; it forces both. So the flag is the
wrong instrument for the remaining question, and the start.sh note should not
be read as having closed it. **Enforced by nothing** — no check asserts our VAE
encode precision against the vendor's, and nothing would notice if it changed.

### VAE tiling is silent here

ComfyUI tiles under memory pressure and records nothing about having done so;
sglang treats decode mode as part of the quality contract and refuses the modes
it considers inexact (cited above). We cannot currently distinguish a tiled
decode from an untiled one after the fact.

---

## Refuted here, and worth keeping refuted

**Hypothesis, raised and killed on 2026-08-21: that the fp8-vs-int8 fidelity
gap is a qkv row-permutation defect in the repack.**

The hypothesis had a real source behind it. sglang treats the H3 qkv layout as
a load-time hazard: the release interleaves each head's Q, K and V rows, sglang
wants them concatenated, so it permutes on load
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:662-689`)
and then permutes **row-indexed quantization metadata the same way** via
`_install_qkv_row_reorder`
(`coderef/sglang/python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py:179-209`),
using the row count as the gate so swizzled and per-tensor scales pass through
untouched. Its unit test states the failure mode: get it wrong and the model
"loads and runs, and renders noise"
(`coderef/sglang/python/sglang/multimodal_gen/test/unit/test_minimax_h3_qkv_scale_reorder.py:22-26`).

It dies on two independent grounds, both already in this repo:

1. **The fp8 file has no row-indexed scale to misalign.** Its `weight_scale` is
   a scalar (`bench/analyze_quant_delta.py`, the format description at the top
   of the file). No permutation of output rows can put a per-tensor scalar on
   the wrong row.
2. **The gap is uniform across module kinds, so it is not qkv-specific.**
   [`bench/results/2026-08-21_quant_delta_fl2va.json`](../../bench/results/2026-08-21_quant_delta_fl2va.json):
   `fp8_vs_bf16` reads the same for `attn.qkv_proj` as for `attn.out_proj`,
   `mlp.fc1` and `mlp.fc2`. A layout defect confined to the fused qkv weight
   would single that module out and it does not.

The reordering itself was never taken on faith: `probe_qkv_layout()` in
`bench/analyze_quant_delta.py` refuses to measure unless reordering is the
better reading, and the same record carries its verdict.

So the remaining explanation for the gap is the one the script was built to
measure: a per-tensor scalar scale against a per-output-row scale, which differ
exactly in the per-channel error distribution and nowhere in a whole-tensor
norm. **No new work is owed here.**

---

## Looks like a gap, already priced

**Step caching.** sglang ships a Cache-DiT configuration for `quality="high"`
with a measured SSIM and PSNR against lossless
(`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/constants.py:57-64`),
so it reads as an obvious gap. It is not. [`roadmap.md`](../roadmap.md) records
the owner decision of 2026-08-20 that step caching dies here: the speedup came
from skipping steps on a 16-step schedule, and at 4 steps there is nothing to
skip. Their number is at 50 steps on four H200s. Different regime.

**Quantization, torch.compile, layerwise offload.** All present in sglang and
all required *off* by its own quality gate, so none of them is evidence for
anything. We already run quantized weights by necessity on 24 GB.

---

## What we do that they do not

Recorded so the comparison is not read one-directionally: Sol-Attn, the sage
kernels and the SLA router have no counterpart in sglang's H3 path, which runs
dense FlashAttention. [`SOLATTN.md`](../SOLATTN.md) owns those numbers. Their
speedups and ours are not comparable and must not be put in the same table.

---

## The 768 cap, and what it did to our code

The open release is 768p on the short edge. Establishing what *kind* of limit
that is changed one doc claim and found one defect; the finding itself is owned
by [`h3_resolutions.md`](../h3_resolutions.md) and is not restated here. The
short version for anyone arriving from a performance question: the area cap is
hard upstream, the short edge only warns, and neither is a property of the
weights.

Auditing this repo against it (2026-08-21, source read) came out mostly clean.
`resolution.py` already grades a canvas by whether `adapt_canvas` is a fixed
point on it and labels anything outside the trained family rather than refusing,
which is the right posture given what the cap turns out to be. `reference_fit.py`
and `bench/check_short_edge_override.py` keep the 2048 reference short edge and
the 768 target cap properly separate — the two knobs this repo has confused
before — and `h3_config.py` has no target short-edge constant to confuse.

**One real defect, found and fixed.** `bench/preflight_graph.py` priced a
reference image that is *not* fed through `MiniMaxH3ReferenceFit` as if it were
upscaled to a 2048 short edge. Core does the opposite: it clamps with
`min(1.0, ...)` in both sizing modes (`comfy_extras/nodes_minimax_h3.py:297-301`)
and never enlarges. The over-count is the square of a scale the reference never
gets — a 1024x1024 reference priced at four times its real row count. No shipped
API graph reaches that branch, because they all wire the fit node; the exposure
is exactly the hand-built graph that `CLAUDE.md` promises preflight can price.
Fixed by defaulting the no-fit case to no upscale, and confirmed non-inert
against a graph rewired to bypass the fit node.

---

## Open after this read

- The VAE **encode** precision question, above. **Half of this closed on
  2026-08-21, after the sentence below was written**: the instrument exists.
  `bench/grade_vae_encoder_precision.py` grades the encoder at the call rather
  than at a rendered clip, and
  [`bench/results/2026-08-21_vae_encoder_precision.json`](../../bench/results/2026-08-21_vae_encoder_precision.json)
  records fp16 bit-identical to itself against fp32 moving the latent, with
  bf16 as the far control. What stays open is whether that delta is *visible*,
  and the variable below, which the instrument does not separate.
  `--fp32-vae` remains the wrong flag because it forces both halves. **It now has a second
  variable tangled with it, found 2026-08-21**: sglang *samples* the released
  posterior under a seed pinned at 42 for keyframes and reference video
  (`coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/keyframe_encoding.py:30`,
  `coderef/sglang/python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/minimax_h3/reference_encoding.py:613`) where ComfyUI takes the mean
  (`comfy/ldm/minimax/vae.py:685`). Any encode instrument has to separate
  precision from mean-versus-sample or it measures neither.
  [`h3_references.md`](../h3_references.md) carries it in the vendor image
  path table.
- Whether an AdaLN cache is worth building, which is downstream of
  [`open_experiments.md`](../open_experiments.md) #22 and should not be
  started before it.
- Nothing left on the reference path: video, audio, frame rate, soundtrack
  pairing and condition noise were all re-derived on 2026-08-21 against sglang,
  diffusers and DiffSynth-Studio, and [`h3_references.md`](../h3_references.md)
  carries the results. One item there is marked unverified and stays that way:
  what sglang does when a reference video is shorter than its aligned target.
  The mono-upmix question that sat beside it was closed on 2026-08-21 — no
  upmix happens and the packed layout raises — and `h3_references.md` carries
  it under Known limitations.
