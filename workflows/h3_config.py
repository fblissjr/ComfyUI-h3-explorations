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
    # branch at comfy/sd.py). 2666.2 MiB resident against fp16's 4966.5 --
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

# `er_sde` / `simple`, the default since 2026-08-15. The owner's call, and a
# default rather than a finding: `res_multistep` / `simple` were core's
# base-template values carried unquestioned, and `er_sde` looked more
# interesting on the clips actually rendered. Treated like every other default
# in this file -- swap it, but know what it costs.
#
#   er_sde     One model eval per step, so it costs what `res_multistep` and
#              `euler` cost. Read in `comfy/k_diffusion/sampling.py`. Note this
#              is NOT true of the whole sampler list: `heun`, `dpm_2` and the
#              `2s`/`3s`/`res_Ns` families are 2-6 evals per step.
#
#              It is stochastic. `s_noise` defaults to 1.0 and each iteration
#              adds fresh noise, the first such default here. `noise_sampler`
#              is seeded from the sampler seed, so two arms at one seed still
#              draw the same noise and an A/B stays paired -- but a knob that
#              perturbs attention numerics will read as more "reseeded" than it
#              did under a deterministic ODE. `SamplerER_SDE` exposes
#              `solver_type="ODE"`, which zeroes the noise and runs the same
#              solver deterministically; the graphs wire plain `KSamplerSelect`
#              and do not expose it yet.
#
#   simple     Kept, and not by inertia. Sol-Attn's window is a percent band
#              that `percent_to_sigma` resolves off the sigma curve with no
#              knowledge of the scheduler, so the scheduler decides how many
#              steps land inside it. At 16 steps and shift_video 12.0: `simple`
#              gives Sol 11 sparse / 5 dense, `beta` gives 9 / 7. `beta` was
#              tried on 2026-08-15 and reverted for that reason -- two fewer
#              sparse steps, no benefit measured against it. `simple` is also
#              the only scheduler reproducing a distilled LoRA's own sigma
#              grid, which matters at 4 steps where the deviation is most of
#              the schedule.
#
#   steps 16   Measured 2026-08-06 at 362 frames: 20 steps 765.4 s, 16 steps
#              669.2 s (-12.6%), 12 steps 508.5 s. 12 was rejected because it
#              stops following the prompt -- the test prompt's third scripted
#              shot at 00:10 never happens. Not smeared, no late-clip artifact,
#              invisible in stills and to a convergence check. Any future step
#              reduction needs prompt adherence as a gate. That judgement was
#              made on `res_multistep` and has not been re-run on `er_sde`.
#
# Every timing recorded in this repo before 2026-08-15 was taken on
# `res_multistep`. The sampler is step-cost-neutral so they should carry, but
# they were not re-taken.
SAMPLING = dict(sampler="er_sde", scheduler="simple", steps=16, denoise=1.0)

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
#
#                     **Reframed 2026-08-14 by the algorithm's author, and
#                     this is a correction to what the line above implies.**
#                     Kijai: "tau 1.0 is where it's the default max quality,
#                     any higher further degrades it, but also speeds it up."
#                     So quality is maximal at 1.0 and falls monotonically
#                     from there -- our 1.3 is NOT the quality choice this
#                     note has been presenting it as. It is a speed-for-
#                     quality trade, taken without knowing there was a trade.
#
#                     Both statements are true and they are different
#                     thresholds. Ours is where the object-dissolve artifact
#                     APPEARS (~1.5); his is where quality PEAKS (1.0).
#                     Between them quality degrades gradually with no
#                     dramatic tell, which is exactly the region we sit in
#                     and exactly the region a stills-based judgement cannot
#                     see. Everything we measured about 1.3 was measured
#                     against 2.0, so it says 1.3 beats a worse setting; no
#                     arm here has ever compared 1.3 against 1.0.
#
#                     UNCHANGED pending measurement, deliberately. Moving it
#                     costs sampler time on every render and the quality
#                     claim is upstream's, not ours -- so it is a bench arm
#                     (`sage+sol[tau=1.0]`, the syntax already exists), not
#                     an edit. Do not "fix" this to 1.0 on the strength of
#                     this paragraph.
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
#
#                     **That is a SPEED result and it is the only axis anyone
#                     has measured.** Kijai, 2026-08-14: "morton may or may
#                     not increase quality, that's something to test." So
#                     `morton=False` is settled on speed and silent on
#                     quality -- reordering video tokens so each 64-token
#                     block is a compact 3D neighbourhood is exactly the kind
#                     of change that would alter WHICH blocks the router
#                     keeps, and nobody here has looked. If it does improve
#                     quality, the 1.16x it costs buys something, and the
#                     current default is trading an unmeasured gain for a
#                     measured one. See docs/open_experiments.md.
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

# The CUDA node's knob set (`SolAttnMiniMax`, comfy_kitchen.sol_attn). A
# SEPARATE dict rather than overrides on the Triton one, because the two nodes
# do not share a vocabulary: `int8_qk`, `int8_pv` and `use_tma` do not exist
# here (the CUDA kernel routes in INT8 unconditionally), and `centroid_tail`,
# `routed_cap_percent` and `reuse_qkv_memory` do not exist there. Merging them
# would let a Triton-only knob be silently dropped on a CUDA arm, which turns
# `sage+sol+int8` into plain `sol` while it still prints as an int8 result.
#
# Values are the node's own defaults EXCEPT the two noted, so this reproduces
# what a user gets from dropping the node in untouched:
#   min_tokens 12288  the node's default, NOT SOL_BASELINE_124F's 4096.
#                     Upstream puts the dense/sparse crossover near 12k and
#                     the win only appears at high token counts; 4096 engages
#                     Sol-Attn in the regime where it costs time. Which of the
#                     two is right here is unmeasured -- that is the point of
#                     having both spellings visible.
#   verbose False     as everywhere else; the verbose arm opts in by name.
#
# NOT wired into any graph, deliberately: the node id is provisional until
# upstream lands global attention timestep scheduling in core. Bench arms are
# code we can rename, saved graphs are not.
# What the graphs wire as of 2026-08-14: SOL_RECOMMENDED's measured choices,
# translated into the CUDA node's vocabulary. This is the shipped config.
#
# Carried over unchanged, each with its evidence in the SOL_RECOMMENDED block
# above: tau 1.3 (below the artifact onset, costs 82.3 s against 2.0 at 362
# frames), exact_kv_and_rows (keeps generated audio intact), morton off (net
# loss stacked on int8), dense_blocks "" (does not fix what tau fixes, costs
# 39.2 s), start/end 0.2/0.9 (never measured, on either backend).
#
# **This migration is not settings-neutral, and one knob makes that
# unavoidable.** The Triton kernel evaluates the pooled tail per row; the CUDA
# node defaults `centroid_tail=True`, one tail per 64-token query block. There
# is no CUDA spelling of "what Triton did" -- `centroid_tail=False` is the
# closest and is a different code path, not the same one. Measured 2026-08-14,
# the two modes differ by cos 0.9988 against the algorithm's own reference,
# which is a real change to what the model computes. True is chosen because it
# is the node's default and where upstream is heading (it is weighing making it
# unconditional), not because it was measured better here.
#
# Two knobs deliberately left at the node's default rather than tuned, to keep
# this a single-variable change:
#   min_tokens 4096   NOT the node's 12288, and it does not matter. Retracted
#                     2026-08-14: this said "very likely wrong, a third of the
#                     node's crossover". That reasoned from a call distribution
#                     that does not exist. H3's DiT has exactly ONE attention
#                     site (`comfy/ldm/minimax/model.py`) at the full packed
#                     length, and frame counts satisfy n %% 17 == 5, so the
#                     shortest clip past 5 frames is 22 frames -> S = 7,194,
#                     already above 4096. Only a 5-frame render falls below.
#                     4096 and 12288 therefore select the same thing -- all of
#                     it -- at every length anyone renders. The small calls that
#                     suggested otherwise were SageChainAssert's own probes.
#   reuse_qkv_memory  False. Verified numerically identical to the normal entry
#                     (cos agreeing to six digits), so it cannot change output,
#                     and upstream reports it drops attention's peak below the
#                     FFN's. Left off only because it is a separate question
#                     from the migration. Cheap win when someone measures it.
SOL_RECOMMENDED_CUDA = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
    reuse_qkv_memory=False, verbose=False, dense_blocks="",
)

SOL_CUDA_DEFAULTS = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=12288,
    sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
    reuse_qkv_memory=False, verbose=False, dense_blocks="",
)

# Our own node. NOT `auto`, which resolves to fp8++ on sm89 -- the FASTEST
# kernel, which is the wrong end of the tradeoff for this project.
#
# Changed 2026-08-13 on two pieces of evidence that point the same way:
#
#   Numeric. Swept against an fp32 reference at H=56 D=128 across
#   S = 4608 / 24576 / 41822 / 78336: fp16-PV holds mean_rtol 0.0362-0.0363
#   while every fp8 variant sits at 0.0969-0.0984. **2.7x more accurate, and
#   flat across a 17x range of S** -- so there is no crossover and no
#   "use fp16 above S=X" rule. One choice covers every canvas, length and
#   reference count.
#
#   **That 2.7x is a SYNTHETIC-INPUT number and the gap is ~1.3x on real
#   activations** (added 2026-08-14). `bench/bench_minimax_attn.py` builds
#   `torch.randn`, and nothing in `bench/` uses captured activations, so every
#   figure above inherits it -- our 0.0969-0.0984 matches the sage fork's
#   synthetic 0.098, not its real-activation 0.026. The fork measures the
#   fp8-to-fp16 gap narrowing from 2.6x to 1.3x on q/k/v captured from an
#   actual H3 forward, and calls every synthetic rtol a pessimistic bound
#   rather than an estimate. Real attention has structure -- concentrated
#   softmax, correlated keys -- that quantization handles far better than iid
#   gaussian noise. The flatness claim comes from the same sweep and is
#   equally unverified on real inputs.
#
#   **The decision does not change**, because its other leg is perceptual and
#   independent: the owner judged fp16 clearer with better motion and less
#   drift on video at the same seed. 1.3x still favours fp16. What weakens is
#   the numeric argument's size, not its direction.
#
#   **`h3_capture.py` ran on 2026-08-15 and the captures exist**, at blocks
#   0/24/49 of a dense 124-frame 1344x768 render, in
#   ~/Storage/h3_captures/2026-08-15_dense_124f_1344x768/. They were made for a
#   different question and **no sage kernel has been graded against them yet**,
#   so this caveat is unchanged -- but the data no longer has to be produced
#   before someone can settle it.
#
#   All three fp8 variants land within 0.0004 of each other,
#   so the PV accumulator is not the lever: quantizing V to fp8 at all is.
#
#   Perceptual. Same seed, same prompt, 124 frames, fp8++ against this:
#   the owner judged fp16 clearer, with better motion and less drift. That is
#   the half no rtol answers, and it agreed with the numbers.
#
# The cost is real and accepted: this is the one mode with no
# `sageattn_consume` entry point, so it holds the float q/k/v for the whole
# call instead of releasing them at quantization. Synthetic says ~1.58x wall
# clock. `mode_releases_qkv` already reads this correctly and disables the
# v-clone, which would be a flat loss on a non-releasing kernel.
#
# `auto` remains available and is what the probe arms bisect against.
#
# token_refiner runs over the text span only (~2k rows against ~42k), so
# patching it is worth well under 1% of attention time.
# head_chunks 1 = off. It trades ~4x the attention launches for headroom that
# converts to wall-clock at the ~2.6% ceiling measured above, so it is for
# fitting a render that otherwise will not fit. Keep the key ordered as the
# node declares its inputs: the UI graph maps widget values positionally.
SAGE_NODE = dict(mode="fp16 (most accurate)", patch_token_refiner=False,
                 head_chunks=1)

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

# A third-party turbo LoRA that is NOT interchangeable with the two above, and
# cannot be loaded by `LoraLoaderModelOnly` at all on our checkpoint.
#
# Measured from the safetensors headers 2026-08-13 (see
# docs/h3_ref2v_distillation.md): the official fl2v LoRAs touch 208 modules,
# all `qkv_proj` / `out_proj` / `fc1` / `fc2`. This one touches 259 -- the same
# 208 plus **51 `adaln_proj.linear`**, at a separate rank 16 against 64 for
# everything else. Those 51 are the 50 per-block `adaln_proj` and
# `final_layer.adaln_proj`, which is exactly where fl2va and ref2va diverge
# most: that last one has a relative delta of 1.92, i.e. is rewritten.
#
# It needs `ComfyUI-MiniMax-H3-Turbo`'s own two nodes rather than the stock
# loader, and the reason is specific to us: our base is PRUNED
# (`..._ref2va_pruned_int8_convrot`), and that pack's node re-injects the
# LoRA's time conditioning at run time from a `silu(t_emb)` grid it ships.
# The stock loader would apply the weights and silently skip that.
#
# Settings are the pack's own, not ours to tune here. README: 4 steps is the
# minimum, 4-8 the useful range, 6-8 noticeably better, past 8 no benefit and
# it starts over-sharpening. Its shipped example uses 6; we take 8 because
# every reference arm carries audio and audio is the axis its README calls
# still-weak at low step counts. Strength is tuned for 1.0. Scheduler stays
# `simple`.
#
# Shift stays at the base 12/3 -- the pack's `generate.py` hardcodes
# SHIFT_VIDEO 12 / SHIFT_AUDIO 3 and its example graph carries no
# ModelSamplingMiniMaxH3 node at all.
#
# `low_vram` merges the LoRA instead of applying it at run time. It is the
# cheaper peak, and it is the WRONG default here: the README says merging is
# softer on quantized bases, and ours is int8 *and* pruned, so we would be
# paying that penalty twice. Off unless something OOMs.
TURBO_PACK_LORA = "h3/minimax_h3_turbo_v4_step600_ema.safetensors"
TURBO_PACK_STEPS = 8
TURBO_PACK_STRENGTH = 1.0
TURBO_PACK_SCHEDULER = "simple"
TURBO_PACK_LOW_VRAM = False

# `CHAIN` was here and is gone as of 2026-08-14. It listed the node order --
# Load Diffusion Model, MiniMax H3 SageAttention, SolAttnMiniMax -- and nothing
# imported it. Node order IS load-bearing (Sol composes with the attention
# patches it finds, so it must come after ours; reversed it overwrites the
# patch and you silently get sage only), which is exactly why a copy of it
# that no code reads is worse than none: when the graphs moved to
# `SolAttnMiniMax` on 2026-08-14 this list stayed on the Triton node id and
# nothing could notice, because there was nothing to notice with.
#
# The order now lives in one place that a reader reaches, docs/SOLATTN.md's
# Ordering section, and in one place a machine checks -- every graph's actual
# wiring, with `SageChainAssert` as the runtime gate that fails the render
# when the chain is not composed as intended. A constant is not a check.
#
# If this is ever wanted back, bring it back with a check that reads it.

# The single-image edit path. Everything in this block exists only for
# `length=1`, and every one of these values is wrong for a video.
#
# **The VAE is not optional and not interchangeable.** It is the same H3 VAE
# with a decoder fine-tuned to reconstruct one image from a single temporal
# latent. Verified from the safetensors rather than from its README, 2026-08-15:
# of 562 tensors, 121 are byte-identical to Comfy-Org's video VAE -- all 116
# encoder tensors, `quant_conv`, and `latents_mean`/`latents_std` -- and the 441
# that differ are 439 decoder tensors plus both `post_quant_conv`. So the
# ENCODER IS FROZEN and the latent space is untouched: a seed produces the same
# latent either way and swapping VAEs is a pure decoder swap. Its metadata
# declares `h3_t1_output_slice: 3`, which is exactly what ComfyUI's own
# `decode()` already does at `z.shape[2] == 1` (`_adaptive_decode(z)[:, :, -1:]`
# of `vae_ratio_t == 4` frames), so the convention matches core and there is no
# off-by-one to chase.
#
# **Never wire it into a video graph.** Its own README: the image-specialised
# decoder materially regresses multi-frame reconstruction and can introduce
# patch-grid ghosting and cross-frame mixing. It is a one-frame decoder.
#
# Source: huggingface.co/Mamad8/MiniMax-H3-Image-VAE (experimental, step 1597).
IMAGE_VAE = "minimax_h3_t1_image_vae_step1597.safetensors"

# 2:3 portrait, and INSIDE the trained family -- `adapt_canvas(2, 3)` returns
# exactly this. Worth stating because the community workflow this path follows
# renders 1024x1536 (1.57 MP), which is 52% over H3's 768*1344 area cap and
# outside the family; it works, and it costs 1.78x the attention per frame for
# a canvas the checkpoint never trained on. Ours is the conservative default,
# not a claim that the bigger one is wrong -- the Resolution node's `custom`
# option reaches it and says which side of the family you are on.
IMAGE_EDIT_CANVAS = dict(width=768, height=1152)

CANVAS = dict(width=1344, height=768)
FPS = 24.0

# Frame counts snap to a 17k+5 grid. ComfyUI's tooltip puts the trained range
# at ~124-362, but that upper bound is wrong: the reference generates 5-15s at
# 24fps and checks the ceiling AFTER the snap, so 362 (15.083s) is refused by
# THE REFERENCE PIPELINE and 345 (14.375s) is the largest count it will emit.
# **362 is trained** -- corrected 2026-08-14; the reference's 15.0 is a round
# number one grid step below the real maximum, not a model boundary. 345 stays
# the default because a graph exported from here then also runs in diffusers,
# which is a portability argument and not a quality one. There is no on-grid count at
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
# nodes.py) and `LoraLoaderModelOnly` always passes strength_clip=0, so
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

# Generated length for the reference-video graphs, and it is lower than
# LONG_LENGTH for a measured reason. At 345 frames the reference arm builds a
# 182,092-token sequence -- 102,816 video, 60,212 references, 16,352 text --
# and it does NOT fit on a 24 GB 4090. Measured 2026-08-13: the render reached
# step 4 of 16 at 123.5 s/it, then Sol-Attn's kernel OOMed and fell back, then
# sage's OOMed and fell back, then ComfyUI's own SDPA OOMed with 21.05 GiB
# allocated against a 23.54 GiB limit. The fallback chain behaved perfectly;
# there was simply no room.
#
# A reference video is the most expensive input in the model: it is truncated
# to the GENERATED frame count, so its cost scales with this number twice over
# -- once for the video rows and once for the reference rows.
#
#   345 frames -> 182,092 tokens   (OOM on 24 GB at 1344x768, refs upscaled)
#   209 frames -> 120,918
#   124 frames ->  82,686
#
# But shortening the render was the WRONG lever, and shipping 124 was a
# mistake worth naming. Because the reference is truncated to the generated
# frame count, cutting the render to fit cuts the reference too -- so the
# 124-frame arms were testing a 5.2-second reference, and a reference arm
# that cannot carry a long reference is not testing the expensive case at
# all. Canvas and reference-image detail are incidental to what these arms
# measure; reference duration is the whole point.
#
# Re-measured 2026-08-13, best case first: the full 345 frames DOES fit once
# the two incidental costs are given up. Same 14.375-second reference, on the
# same card, end to end:
#
#   345f @ 1024x768, refs not upscaled -> SUCCESS, peak 22,735 MiB, 34.3 min
#
# 1,829 MiB of headroom on a 24,564 MiB card, so this is close to the edge
# and not a general-purpose budget: a third reference image or a longer
# soundtrack can still push it over. Preflight is the thing to read before
# widening any of it.
REF_VIDEO_LENGTH = 345

# The canvas and reference-image policy that measurement bought. 1024x768 is
# 4:3 rather than the 1344x768 the rest of the repo defaults to -- a real
# change in what these arms look like, taken deliberately so the reference
# stays full length. `ref_upscale=False` leaves reference images at their
# native size instead of taking them to 2048 on the short edge.
#
# Spread into every video-bearing reference arm so the three numbers have one
# home. Editing them here moves all eight arms together, which is the point.
REF_VIDEO_CANVAS = dict(width=1024, height=768)
REF_VIDEO_BUDGET = dict(length=REF_VIDEO_LENGTH, **REF_VIDEO_CANVAS,
                        ref_upscale=False)
