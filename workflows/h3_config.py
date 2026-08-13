"""Single source of truth for the H3 node chain and its settings.

Both `workflows/build_workflows.py` (which emits the graphs you open in
ComfyUI) and `bench/bench_e2e_h3.py` (which produces the numbers) import
from here. Before this file existed they each carried their own copy of the
SolAttn settings, and those copies drifted the moment one was updated -- so
a bench arm named "sol" and the workflow you would actually render were
different configurations, and the measurement described something nobody
ran. That is the same failure as quoting a speedup for a config that was
never rendered; keep it structural rather than remembered.

Nothing here is allowed to have a second copy anywhere in the repo.
"""

# Kijai's extracted fl2va -> ref2va weight difference (Kijai/MiniMax-H3-experimental,
# 2026-08-08). Rank 256 on the attention and MLP projections, rank 8 on the
# adaln projections, plus full-rank `diff`/`diff_b` deltas on every norm and
# bias -- a whole-model delta extraction, not a trained adapter. Coverage
# matches the fl2va checkpoint exactly: verified against both safetensors
# headers, and against comfy.lora.load_lora, which turns its 794 tensors into
# 530 patches with zero unmatched keys. So at strength 1.0 it should
# reconstruct ref2va, up to rank truncation and requantization error.
#
# That "should" is why `h3_image_ref_plus_text_to_video_ref_lora.json` exists
# as a sibling of the shipped ref graph rather than as a claim: upstream's own
# description is "completely experimental, I don't even know if it has a use
# case at this point". Run the two and judge.
#
# The `h3/` prefix is load-bearing: LoRAs are foldered in this install and
# ComfyUI's combo carries the subfolder in the value, so the bare filename is
# rejected by /object_info validation.
REF_LORA = "h3/minimax_h3_ref_lora_rank_256_bf16.safetensors"

# Checkpoint names are the ones ComfyUI actually offers. The bundled
# templates ask for `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, an NVFP4
# text encoder that is not present in this install (and is a
# Blackwell-oriented quant); the int8_convrot build is the one to use.
MODELS = dict(
    unet_fl2va="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    unet_ref2va="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    clip="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    # int8_convrot decoder (Comfy-Org/ComfyUI#15334 merged 2026-08-06, loader
    # branch at comfy/sd.py:945). 2666.2 MiB resident against fp16's 4966.5 --
    # a real 2300 MiB, engagement confirmed by allocation rather than by a log
    # line, since the loader prints `dtype: torch.float16` for both builds and
    # cannot distinguish int8 storage from a dequantized fallback.
    #
    # Swapped in 2026-08-10 after the quality and speed pass this note used to
    # ask for. Paired arms, same seed, so the latents are identical and every
    # pixel difference is the decoder:
    #
    #   decode      12.8s -> 9.9s median (1.29x) at 124f, 1344x768
    #
    # **Re-measured 2026-08-11 after kijai's decode rewrite (ComfyUI 2a68ce33,
    # which added chunked IO and streams finalized chunks to the intermediate
    # device instead of assembling the video on the GPU): 12.8s -> 10.0s,
    # 1.28x.** Unchanged within noise, so the figure above stands and the
    # rewrite did not move the int8-vs-fp16 relationship. Worth having
    # checked rather than assumed: the numbers predated the rewrite and
    # nothing said whether they survived it.
    #
    # Peak VRAM was NOT reproducible in that run: fp16 measured 20263 and
    # 22528 MiB on two runs of the same seed, and int8 measured higher than
    # fp16 overall. Process peak is dominated by ComfyUI's dynamic VRAM
    # reallocating against free memory, not by what the VAE holds, so peak
    # is not a number to compare VAEs on here.
    #   quality     PSNR 53.3 dB, mean|d| 0.23/255, p99 2/255, max 20/255,
    #               and 79.6% of pixels bit-identical. Lossless PNG straight
    #               off VAEDecode, same latent through both VAE files.
    #   temporal    per-frame error is a flat offset rather than a pulse, and
    #               frame-to-frame motion energy int8/fp16 = 0.998 -- int8 is
    #               fractionally *smoother*, where flicker would read above 1.
    #
    # **The first quality pass here was wrong and is worth keeping as a
    # warning.** It compared the two arms after mp4 encode and reported
    # ~40 dB / 1.3-1.7 of 255 as agreement. An h264 round trip on IDENTICAL
    # pixels measures 1.63/255 at 41.1 dB, so that number was the codec's
    # noise floor and not the decoder at all. The tell was visible and
    # missed: three different comparisons all returned 1.64, which is what
    # happens when the thing being varied is smaller than the instrument's
    # resolution. Redone without a codec in the loop, the real decoder
    # difference is 53.3 dB -- roughly 7x below what the video encode adds
    # anyway, so shipping int8 changes the output less than saving it does.
    #
    # Still measured at 124 frames rather than the 250+ this config runs.
    video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)

# `res_multistep` is one model eval per step (true multistep, reuses
# `old_denoised`), so it costs what euler costs. Note that is not true of the
# whole sampler list: `heun`, `dpm_2`, and the `2s`/`3s`/`res_Ns` families are
# 2-6 evals per step, and picking one silently multiplies the ~91% of render
# time the sampler occupies.
#
# **steps 16, measured and judged 2026-08-06.** 20 was the bundled template's
# default and was never questioned until it turned out to be the largest
# untested lever in the config -- steps multiply everything, including the
# ~24% of the step that attention work cannot reach. At 362 frames:
#
#   20 steps  765.4 s      16 steps  669.2 s  (-12.6%)      12 steps  508.5 s
#
# 12 was rejected, and for a reason worth recording because two of the three
# gates we had prepared would have passed it. It is not smeared and it has no
# late-clip artifact -- it simply **stops following the prompt**. The test
# prompt specifies three shots with explicit cut times, and at 12 steps the
# third one (a pull-back into the street at 00:10) never happens. Shot
# structure is a long-range instruction and the trajectory settles into
# something simpler before reaching it. Invisible in stills, invisible to a
# convergence check, only catchable by watching to the end knowing what was
# asked for. Any future step reduction needs that as a third gate.
#
# Scheduler stays `simple`. `beta57` was tested and dropped -- not because it
# sampled worse but because the comparison was never isolating the scheduler:
# Sol-Attn's window is a *percent* band that `percent_to_sigma` resolves off
# the model's sigma curve with no knowledge of the scheduler, so a different
# scheduler puts a different number of steps inside it. beta57 ran 10 dense
# steps at 20 against simple's 6, which is why it measured slower (825.1 s
# against 765.4). It also came from a custom node pack rather than core.
SAMPLING = dict(sampler="res_multistep", scheduler="simple", steps=16, denoise=1.0)

# SolAttn knobs, pinned so neither a graph nor a bench arm inherits whatever
# the node currently defaults to. Pinning is load-bearing and has already
# nearly failed once: SolAttn changed `int8_qk`, `int8_pv` and `morton_curve`
# defaults underneath us, so an arm named "sol" would have meant different
# things before and after that release with no visible change on our side.
#
# Revised 2026-08-06 on a 4090 / 24 GB, where render time is the objective
# and VRAM headroom only counts insofar as it converts to render time. It
# mostly does not here: weight streaming is 0.6% of a 362-frame step and the
# trace shows it already hidden behind compute, and phase swapping is 2.0%,
# most of it unavoidable because the text encoder and DiT are 45.9 GB
# together and can never co-reside. So the ceiling on headroom-to-speed is
# ~2.6%, and knobs that trade launches for headroom are not worth it.
#
#   tau 1.3           Below the onset of the moving-content artifact -- see
#                     the two-phenomena note below. Costs 82.3 s of sampler
#                     against tau 2.0, measured same-seed at 362 frames
#                     (712.1 s against 629.8 s), and worth it.
#   dense_blocks ""   Was 33-35,39-42, the two highest-error regions on the
#                     author's per-block sensitivity profile. Dropped: it
#                     does not fix the artifact tau does, and costs 39.2 s.
#   exact_kv_and_rows Runs the packed conditioning query rows dense, which
#                     is what keeps the generated audio intact. Those rows
#                     are ~250-400 in a ~38k sequence, thin enough to be
#                     exactly what a block-sparse router drops first -- the
#                     same shape as the object-dissolve artifact above.
#   morton off        Worth 1.16x alone but a net loss stacked on int8
#                     (1.34x against 1.39x), and its arm runs at 94% GPU
#                     utilisation where every other arm hits 99%.
#   int8_qk/pv on     Worth 1.16x on top of plain sparsity at 362 frames.
#
# Head chunking is deliberately not in this chain, and as of 2026-08-10 that
# is measured rather than inferred. The 1-vs-4 A/B this note used to ask for
# has been run -- 260 frames, 1344x768, 16 steps, 2 runs per arm, paired
# seeds, peak VRAM polled from /system_stats through each render:
#
#   arm                sampler   peak VRAM   vs base   sampler
#   head1/ffn1          395.6s   16702 MiB        +0    1.000x
#   head4/ffn1          396.5s   13475 MiB     -3227    0.998x
#   head1/ffn2          397.1s   17376 MiB      +674    0.996x
#   head4/ffn2          398.2s   18607 MiB     +1904    0.994x
#
# The launches are not free, and the headroom does not convert. Head chunking
# frees 3227 MiB -- three times the ~1070 MiB previously estimated -- and
# costs 0.2%. Both halves of the question are answered separately, which is
# why peak VRAM is measured alongside time: a single column cannot tell "freed
# nothing" from "freed something that did not convert", and those call for
# opposite next steps. Take head chunking only to fit a render that otherwise
# will not fit.
#
# Two cautions on reading the table.
#
# The timing gaps are all under 1%, but the ordering reproduced exactly in
# both runs -- more chunking is monotonically slower, which is what launch
# overhead should look like. Trust the ordering, not the magnitudes.
#
# The VRAM needs a noise floor. Baseline peaked at 17094 and 16310 across its
# two runs, a 784 MiB spread, while every chunked arm was stable to within
# 6 MiB. So head4's -3227 is solid; ffn2's +674 is INSIDE that spread and is
# not evidence of anything; head4/ffn2's +1904 is outside it. That last one is
# the surprise -- the two knobs are antagonistic, not additive, and adding FFN
# chunking on top of head chunking costs ~5 GB relative to head chunking
# alone. No mechanism established. Probably allocator or fragmentation
# behaviour under dynamic VRAM loading, but that is a guess and n=2.
#
# FFN chunking (MiniMaxChunkFeedForward) is likewise not here: at this length
# the attention peak sets the ceiling, so chunking the FFN moves something
# that is not the maximum. It remains a short-clip feature.
#
# **The sigma window stays .2-.9, and widening it is closed.** `.1-.95` is
# tempting -- 687.4 s against 768.2 at 20 steps, ~10%, and it passed every
# gate there including prompt adherence. It does not survive at 16 steps:
# 568.8 s, but the shot timeline drifts (the scripted 00:10 cut lands nearer
# 12-13 s) and the subject's motion stalls. Not smearing, not the late-clip
# artifact -- a fourth failure mode, structural timing.
#
# Worth keeping as a caution rather than a footnote: **both factors passed
# adherence individually and the combination failed.** 20 steps + wide hit
# the cut on time; 16 steps + narrow hit it on time; 16 + wide did not. A
# knob validated at one setting of another knob is not validated, and the
# ten minutes spent confirming that was the cheapest measurement of the day.
SOL_RECOMMENDED = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=True, sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", int8_pv=True, verbose=False, use_tma=False,
    dense_blocks="",
)

# Kept for the day someone reproduces the artifact and wants the fix back.
# Not in the shipped config -- see the tau/dense_blocks notes above.
SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")

# The settings the 124-frame evaluation in docs/SOLATTN.md ran on. This exists so
# a bench arm can reproduce an old number, and it deliberately differs from
# SOL_RECOMMENDED above -- do not "fix" it to match. Every recorded ratio in
# docs/SOLATTN.md's frontier table was produced with these, so changing them
# silently makes old and new numbers incomparable while both still print.
#
# Keeping the two side by side is the point: before this file existed the
# bench and the workflow builder each had one of these and neither knew the
# other existed, so the difference read as a bug rather than as two things
# doing different jobs.
SOL_BASELINE_124F = dict(
    tau=1.2, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=False, int8_pv=False, sink_conditioning="exact_kv", morton=False,
    morton_curve="3d", verbose=False, use_tma=False, dense_blocks="",
)

# Our own node. `auto` resolves to fp8++ on sm89, so naming it explicitly
# would change nothing; the explicit modes exist for bisecting accuracy.
# token_refiner runs over the text span only (~2k rows against ~42k), so
# patching it is worth well under 1% of attention time.
# head_chunks 1 = off. It trades ~4x the attention launches for headroom that
# converts to wall-clock at the ~2.6% ceiling measured above, so it is for
# fitting a render that otherwise will not fit. Keep the key ordered as the
# node declares its inputs: the UI graph maps widget values positionally.
SAGE_NODE = dict(mode="auto", patch_token_refiner=False, head_chunks=1)

# Flow shifts, on `MiniMaxH3SigmaShift` (display name ModelSamplingMiniMaxH3).
# 12/3 are the base checkpoint's training shifts and the node's own defaults,
# so these values change nothing on their own. The node is in the graph so the
# shift is *visible and switchable*, because the turbo LoRAs were distilled at
# their own shifts and inherit the sampler's, not the base model's:
#
#   FL2VA Turbo 4-step v0.1     544p mixed aspect   12 / 3   4 steps
#   FL2VA Turbo 8-step v1.0     544p                12 / 3   8 or 4 steps
#   FL2VA Turbo 4-step v1.0     768p (1344x768)      6 / 3   4 steps
#
# The 768p one is the trap, and it is the one that matches CANVAS below. Its
# video shift is 6, half the default, so loading that LoRA into a graph that
# leaves this at 12 samples it off a schedule it was never distilled for. A
# graph with no shift node at all gives you no place to notice.
#
# Steps move with the LoRA too: SAMPLING["steps"] = 16 is a base-model number
# and the whole point of these LoRAs is 4 or 8. Changing shift without
# changing steps, or the reverse, is not a partial improvement.
# Source: coderef/Minimax-H3-Turbo README, model specs table.
SIGMA_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The turbo graph. This is the 8-step v1.0, which is the one present locally;
# the other two are listed in the note the graph carries.
#
# Its shift is 12/3, the same as base, so the shift node does not move for
# this LoRA. Only the steps do: 8 instead of 16. That is worth stating plainly
# because "turbo LoRA" and "change the shift" got learned together, and for
# two of the three checkpoints the shift is already right.
#
# Strength 1.0 is the reference's own default (`--lora-scale` defaults to 1.0
# in Minimax-H3-Turbo's inference script). Not swept here.
TURBO_LORA = "h3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 1.0
TURBO_STEPS = 8
TURBO_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The other released turbo LoRA, and the only one whose shift is not 12/3.
# Constants rather than values typed into a graph because the filename, the
# shift and the step count have to move together -- `bench/check_distill_settings.py`
# grades this triple against the vendor's own README and fails if any of the
# three drifts. Distilled at 1344x768, which is `CANVAS`, so unlike the 8-step
# it is already at home on the default canvas.
TURBO_768P_LORA = "h3/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
TURBO_768P_STEPS = 4
TURBO_768P_SHIFT = dict(shift_video=6.0, shift_audio=3.0)

# Where the 8-step v1.0 was actually distilled: 544p, mixed aspect. This is
# *below* H3's own canvas rule (768 short edge), so `adapt_canvas` never
# returns it and the base model is outside its trained family here -- which is
# the whole tension the turbo note describes. The vendor's own ComfyUI graph
# ships 960x544 for t2va, so this is their answer to it, not ours.
# 510 tokens/frame against 1008 at 1344x768, i.e. 0.26x the attention.
TURBO_HOME_CANVAS = dict(width=960, height=544)

# The sampler the vendor ships on both its turbo graphs, against `SAMPLING`'s
# res_multistep which came from core's base template. A distilled model is
# trained so one Euler step from sigma_i lands at sigma_i+1, so a multistep
# integrator corrects a discretization error that is not the dominant error
# here. That is an argument, not a measurement -- which is why it is a probe
# pair (`h3_probe_turbo_euler.json`) and not a change to `SAMPLING`.
TURBO_SAMPLER = "euler"

# Node order is not cosmetic. Sol-Attn composes with the attention patches
# it finds, so it must come after ours; reversed, it overwrites the patch
# and you silently get sage only.
CHAIN = ["Load Diffusion Model", "MiniMax H3 SageAttention", "SolAttnPatch"]

CANVAS = dict(width=1344, height=768)
FPS = 24.0

# Frame counts snap to a 17k+5 grid. ComfyUI's tooltip puts the trained range
# at ~124-362, but that upper bound is wrong: the reference generates 5-15s at
# 24fps and checks the ceiling AFTER the snap, so 362 (15.083s) is refused and
# 345 (14.375s) is the largest legal count. There is no on-grid count at
# exactly 15.0s. See h3_rules.py for the rule and where it comes from.
#
# **Changed 362 -> 345 on 2026-08-10, and this breaks comparability.** Every
# measurement in the notes above -- the ~24% attention ceiling, the 2.6%
# headroom ceiling, tau 1.3 costing 82.3s against tau 2.0, int8_qk/pv at
# 1.16x -- was taken at 362 frames, a length this config no longer produces.
# The ratios should survive the 5% length change, but they were not re-taken;
# treat any of them re-derived at 345 as the number to trust. 362 stays
# reachable by passing `length=` explicitly if an old figure needs
# reproducing.
LENGTH = 124
LONG_LENGTH = 345

# Fixed rather than randomised, and deliberately not 1. Every graph and
# bench arm shares it, which is what makes any two of them comparable: the
# probe graphs below are pairs differing in exactly one setting, and a seed
# that moved between them would put the difference you are looking for
# underneath the difference you are not. Change it if you want a different
# draw, but change it in one place and regenerate everything.
SEED = 730451892

# `adapt_canvas` imposes short edge 768 and a hard area cap of 768*1344 =
# 1,032,192 px, each axis rounded to 32. That cap is why this is a list of
# aspect ratios rather than resolutions -- there is no higher one to pick.
# 21:9, 16:9 and 9:16 all land on the cap and cost the same; a square never
# reaches it, so 1:1 is a third of the attention cost of 16:9 at equal frame
# count. Landscape and portrait of a ratio are exactly equal in cost, since
# packed rows are (h//32)*(w//32).
ASPECTS = {
    "16x9":  (1344, 768),
    "9x16":  (768, 1344),
    "4x3":   (1024, 768),
    "3x4":   (768, 1024),
    "1x1":   (768, 768),
}

# The strength the ref-LoRA graph ships at. 1.0 is the only value with a
# defined meaning -- it is where the extracted delta is supposed to reconstruct
# ref2va. Everything below it is an interpolation the LoRA was never fitted
# for, which is the interesting part but not the default.
#
# Strength 0.0 and bypassing the node are the same thing, not two baselines.
# `LoraLoader.load_lora` short-circuits when both strengths are zero (ComfyUI
# nodes.py:729) and `LoraLoaderModelOnly` always passes strength_clip=0, so
# either route hands back the untouched model and renders true plain fl2va.
#
# What neither gives you is a baseline that took the same path as the 1.0
# arm. Applying the LoRA to a quantized checkpoint is a dequantize / add /
# requantize round trip, and the zero-strength route skips it entirely -- so
# part of any 1.0-against-0.0 difference is that round trip rather than the
# delta. To see the round trip by itself, render 0.01: visually nil, but it
# does not short-circuit, so it pays the full cost.
REF_LORA_STRENGTH = 1.0

# Where a two-stage split cuts the shared schedule, in steps. 2 of 8 is the
# shipped starting point, not a finding. H3's schedule is far more
# front-loaded than the model the split pattern came from: at video shift 12
# and 8 steps, seven of the eight evaluation points sit at sigma >= 0.8 and
# the final interval alone covers the bottom 63% of the range. Krea 2's
# k=2-3 sweet spot was still sigma 0.84 there; here k=3 is 0.9524. So the
# useful boundary is lower here and the sweep starts at 1.
SPLIT_AT = 2
