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
#   morton off        **The 1.16x speed cost below is retracted for the CUDA
#                     backend, measured 2026-08-16.** Isolated properly -- all
#                     50 blocks dense, morton on against morton off, so the
#                     permutation is the only difference -- it came out +0.8 s
#                     of 861. The sparse pair moved 1.2 s of 454 the OTHER way,
#                     i.e. morton-on faster, which it cannot be. Opposite signs,
#                     both at or under this bench's run-to-run spread on one run
#                     per arm. **The permutation is free at 1344x768 / 294
#                     frames on the CUDA kernel**, and neither delta should be
#                     quoted as a cost. The old figure was Triton, 362 frames,
#                     and stacked on int8; it is not wrong for what it measured,
#                     it just does not describe this backend.
#
#                     morton stays OFF anyway, now on a different basis: the
#                     reason is no longer cost, it is that nothing has shown it
#                     changes the output. See docs/morton.md.
#
#                     Do NOT quote peak VRAM from that run. The four arms
#                     spanned 17,326-23,208 MiB with morton saving 3.7 GB in
#                     the sparse arm and costing 2.1 GB in the dense one --
#                     opposite signs, so not a morton effect, and consistent
#                     with the warning above that process peak here tracks the
#                     allocator rather than the arm.
#
#                     **That is a SPEED result and it is the only axis anyone
#                     has measured.** Kijai, 2026-08-14: "morton may or may
#                     not increase quality, that's something to test." So
#                     `morton=False` is settled on speed and silent on
#                     quality -- reordering video tokens so each 64-token
#                     block is a compact 3D neighbourhood is exactly the kind
#                     of change that would alter WHICH blocks the router
#                     keeps, and nobody here has looked. The old form of this
#                     sentence said a quality gain would mean "the 1.16x it
#                     costs buys something" -- caveat decay inside this very
#                     comment, three paragraphs under the retraction. There is
#                     no cost to buy anything with: the permutation is free, so
#                     any quality gain at all would make morton worth turning
#                     on. See docs/morton.md and docs/open_experiments.md.
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
    # `3d`, not `2d_frame`, since 2026-08-16. This changes NOTHING today
    # because morton is off; it changes which curve you get if you turn it on.
    # On captured activations `3d` beats `2d_frame` on per-block centroid
    # fidelity at every depth sampled (0.7915/0.9434 against 0.7665/0.8804 at
    # blocks 24/49), and all three curves measured speed-identical, so the
    # switch was wired to the weakest of them for no reason. `2d_frame` was
    # chosen on the FRAME_PER_TOKEN argument -- (1,4,4,4,4) means a 3D curve
    # groups temporally distant tokens -- which is mechanically correct and
    # which the measurement does not refute; it just does not win. See
    # docs/morton.md.
    #
    # **OPEN, raised 2026-08-16: this pin was selected at `3d`'s best canvas.**
    # Every number above is 1344x768, where `3d` is 97.9% connected. Swept over
    # all 48 legal canvases at the SHIPPED length it is the MOST
    # canvas-variable of the four orderings -- floor 51.5%, well below plain
    # `hilbert`'s 77.1%, worse than plain `hilbert` on 14 of the 48, and its
    # four worst are all in the shipped set (1952x544, 1888x544, 1568x672,
    # 1440x736). Length matters here and only for this curve: `3d` mixes
    # frames, so its floor is 67.2% at 124 frames and 51.5% at 362.
    # So the default was chosen where this curve looks
    # best and would be deployed across a set where it is the least
    # predictable. That is geometry; whether the ACTIVATION advantage is
    # canvas-contingent too is unmeasured, and the two possibilities point
    # opposite ways. Do not read this as "the pin is wrong" -- read it as a
    # pin resting on one canvas. `docs/morton.md` has both readings and names
    # the experiment that separates them. Nothing is at risk while
    # `morton=False`; the exposure is the next person who turns it on.
    morton_curve="3d", centroid_tail=True, routed_cap_percent=0,
    reuse_qkv_memory=False, verbose=False, dense_blocks="",
)

SOL_CUDA_DEFAULTS = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=12288,
    sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", centroid_tail=True, routed_cap_percent=0,
    reuse_qkv_memory=False, verbose=False, dense_blocks="",
)

# Our own node. `auto`, which resolves to fp8_cuda++ on sm89.
#
# **Changed to `auto` on 2026-08-18, reversing the 2026-08-13 flip to fp16.**
# Measured, not argued: `bench/results/2026-08-18_attention_defaults.json` is
# the record -- a 2x2 of sage mode against Sol reachability, one variable per
# arm, same graph and seed, generated by the script beside it. Read the ratios
# there rather than here; a number copied into this comment is a second copy.
#
# The reason is not that `auto` is faster, which was always true and was
# always the accepted cost. It is that the argument for fp16 was measured in
# a configuration this repo no longer ships. The perceptual A/B was taken at
# 124 frames with **Sol-Attn absent** -- it landed the next day -- so sage ran
# every step. Today Sol takes the steps inside its sigma window and sage keeps
# the rest, so fp16 buys accuracy on a minority of steps while being paid for
# on all of them. The arms show the whole fp16 difference living in exactly
# those dense steps, because Sol's own kernel cost does not move with the sage
# mode. That is caveat decay of the kind `docs/evidence.md` exists to catch:
# the verdict was sound for what it measured and was carried somewhere else.
#
# **What would reverse this**: a blind paired judgement at the SHIPPED config
# (Sol on), not dense. The clips for it are rendered -- same seed, one variable
# -- and named in the data file. If fp16 wins there, flip this back and say so.
#
# The 2026-08-13 history is kept below because the withdrawal is the lesson.
#
# It rested on two legs; one was withdrawn, and the other did not transfer.
#
#   Perceptual, and this is now the whole argument. Same seed, same prompt,
#   124 frames, fp8++ against this: the owner judged fp16 clearer, with better
#   motion and less drift. That is the half no rtol answers, and it is the half
#   that has held.
#
#   Numeric -- **WITHDRAWN 2026-08-16 by the owner, as untrusted.** This block
#   used to carry an fp8-vs-fp16 accuracy ratio, the `mean_rtol` sweep behind
#   it, and a smaller ratio reported secondhand from the sage fork's captured
#   activations. All of it is removed rather than caveated, because the
#   provenance could not be defended: the sweep is `torch.randn`, which is not
#   the input distribution H3 has; the real-activation figure was never
#   re-derived here and the script producing it is not committed in the fork;
#   and nothing in `bench/` uses captured activations, so every accuracy number
#   this repo could print inherits the synthetic instrument. See
#   `docs/evidence.md`. **Do not reintroduce a ratio here.**
#
#   **The decision does not change and never depended on the withdrawn leg.**
#
#   **Captures exist, but not the one this comment named.** The dense 124-frame
#   1344x768 render at blocks 0/24/49 is no longer on disk; what is there is the
#   2026-08-17 reference-heavy pair at 362 frames 1024x768. Both were made for a
#   different question and **no sage kernel has been graded against either**. That is the run that would let this repo state an accuracy figure
#   of its own; until it happens there is no number to quote.
#
# fp16's other cost, unchanged and now not paid: it is the one mode with no
# `sageattn_consume` entry point, so it holds the float q/k/v for the whole
# call instead of releasing them at quantization. `mode_releases_qkv` reads
# this correctly and disables the v-clone, which would be a flat loss on a
# non-releasing kernel. Selecting `auto` restores both the release and the
# clone, so this is a memory improvement as well as a wall-clock one.
#
# `fp16 (most accurate)` remains available and is what the probe arms bisect
# against.
#
# token_refiner runs over the text span only (~2k rows against ~42k), so
# patching it is worth well under 1% of attention time.
# head_chunks 1 = off. It trades ~4x the attention launches for headroom that
# converts to wall-clock at the ~2.6% ceiling measured above, so it is for
# fitting a render that otherwise will not fit. Keep the key ordered as the
# node declares its inputs: the UI graph maps widget values positionally.
SAGE_NODE = dict(mode="auto", patch_token_refiner=False, head_chunks=1)

# Step caching, on ComfyUI core's EasyCache node (comfy_extras/
# nodes_easycache.py). Added 2026-08-18. The node thresholds the relative
# change of the model's input between adjacent steps and, under threshold,
# skips the whole transformer forward and reuses a cached residual -- on a
# cached step neither sage nor Sol runs, so this composes with the attention
# chain by bypassing it rather than by negotiating with it.
#
# Why it is worth a probe arm at all: NVLabs' MiniMax-H3 RTX 4090 runtime
# (Sana repo, sol-engine branch, PR #466, read 2026-08-18) attributes 3.18x of
# its 4.44x end-to-end speedup to TeaCache-family step caching and only 1.22x
# to Sol-Attn -- measured at 50 steps against a same-card dense baseline, one
# warm sample, no quality metric. At this repo's 16 steps the ceiling is far
# lower: with the first ~15% and last ~5% of steps forced dense by
# start/end_percent below, at most 12 of 16 forwards are even skippable.
#
# EasyCache handles H3's [video, audio] dual latent: it slices per-stream on
# latent channels, and MiniMaxH3AV declares latent_channels=32 precisely so
# such slices keep both streams whole (comfy/latent_formats.py). That is a
# source read, not an H3 test -- the probe arm is the test.
#
# `verbose=True` because the node's hit/miss log lines in the server log are
# the only record of how many steps a run actually reused; a timing without
# that count is uninterpretable.
#
# Two standing cautions, both from CLAUDE.md rules: the shipped sampler
# `er_sde` re-noises every step, which inflates adjacent-step input deltas
# and can suppress reuse -- a null result on er_sde is a sampler artifact
# until reproduced on a deterministic sampler; and any cache-on/off quality
# judgement is a numeric-perturbation A/B, which must run on a deterministic
# sampler. Keep the key order matching the node's declared inputs: UI graphs
# map widget values positionally.
CACHE_NODE_CLASS = "EasyCache"
CACHE_NODE = dict(reuse_threshold=0.2, start_percent=0.15, end_percent=0.95,
                  verbose=True)

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
# loader, for TWO independent reasons. Both measured 2026-08-16 from the
# safetensors headers; neither is the int8.
#
# 1. Key names, and this one is about the file, not our base. Its keys are
#    bare (`blocks.0.adaln_proj.linear.lora_A.weight`), while
#    `comfy/lora.py:192-196` builds its key map from `model.state_dict()`,
#    where every key carries a `diffusion_model.` prefix. Nothing matches, so
#    zero keys load. The stock loader does not "apply the weights and skip the
#    time conditioning" -- it applies nothing at all.
#
# 2. Our base is curve-form, which is what PRUNED means here. Such a
#    checkpoint ships an `adaln_t_table` and has no `time_embedder` module at
#    all (`comfy/ldm/minimax/model.py:440-452`, consumed at :629-636), and its
#    adaln takes an 8-wide curve coordinate: `blocks.0.adaln_proj.linear.weight`
#    is [96768, 8]. This LoRA's adaln half was trained full-width, lora_A being
#    [16, 2688], so `lora_B @ lora_A` is [96768, 2688] and cannot be added to a
#    [96768, 8] weight at any strength or by any loader. Its attn/mlp half
#    would fit ([64, 5376] against a [21504, 5376] weight); only the 51 adaln
#    modules are impossible. The `fp8_scaled` build carries the same [96768, 8]
#    and the same table, so swapping quantization changes nothing.
#
# Hence the pack shipping its own `silu(t_emb)` grid: on a curve-form base the
# table must be regenerated, not patched. The official `_comfyui_` turbo LoRAs
# load here precisely because they carry no adaln keys at all.
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

# What every single-frame image graph spends, in one place.
#
# **`ref_upscale=False`, and it is the opposite of the video default.** The fit
# node's upscaling takes a reference's short edge to 2048; on this path that is
# the single largest cost.
#
# **Confirmed here 2026-08-16 rather than inherited.** `h3_image_style`, the
# two-reference graphite scene, rendered both ways at the same seed with
# nothing else changed:
#
#   allow_upscale=True    89.1s
#   allow_upscale=False   18.1s     <- ships here
#
# 4.9x the wall clock. The two images were compared side by side and against
# the source reference: same identity, same freckle pattern, same head angle,
# same expression, same hairstyle, and the graphite medium transferred in both
# without the style reference dragging its cottage along. If anything the
# cheaper one has crisper hair strands.
#
# That reproduces the earlier 84s/18s ladder (`open_experiments` #16e) on a
# second occasion, which is why the default moved here where #16e declined to
# move it: that entry rested on ONE subject at one seed and said so. This is
# still a small n -- two subjects, two seeds -- but the renders are seconds, so
# the cost of being wrong is a re-render rather than an afternoon.
#
# **The video graphs keep `ref_upscale=True`.** Nothing here transfers to them:
# a 124-frame render is minutes, identity has to survive motion as well as a
# still, and REF_VIDEO_BUDGET turns it off for an unrelated reason (fitting a
# long reference in 24 GB).
#
# **`steps` stays at 16 on this path, and that was tested, not assumed.** The
# obvious companion optimisation is fewer steps, and on the easy scenes it
# looks free: `h3_image_edit` (one reference, a camera move) renders at 16 in
# 13.0s and at 8 in 4.0s, and the two are near-indistinguishable -- same man,
# same suit, same tie, same three-quarter view. `h3_image_style` at 8 keeps its
# freckling and its medium too.
#
# **`h3_image_multiperson` is where it breaks, and it is the scene that
# matters.** Three references, 16 steps 25.0s against 8 steps 10.0s: at 8 the
# woman's freckling is largely gone and her pendant has disappeared, and
# freckling is precisely the identity marker that scene's `partially_preserved`
# entry names. So the saving is ~15s on the one graph where the detail is the
# whole point.
#
# The lesson is about the test, not the number: measured only on the
# one-reference portrait, 8 steps looks free everywhere. **A check whose input
# already satisfies the expected outcome cannot fail**, and a single portrait
# is that input for step count. One paired render per scene, consistent with
# the expected mechanism (fewer steps, less fine detail) -- not a sweep, and
# not enough to justify a per-scene step count.
#
# **`ref_image_size` stays `max` and is still a no-op**, but for a NEW reason,
# and the old one is now wrong. It used to be a no-op because the fit node had
# already reached 2048 so core's `min(1.0, 2048/short_edge)` was 1.0. With the
# fit node no longer upscaling, a sub-2048 source hits `min(1.0, >1.0)` = 1.0
# and is left at native size. Same outcome, different mechanism -- and for a
# source ABOVE 2048 the two diverge, so do not simplify this away.
IMAGE_EDIT_BUDGET = dict(**IMAGE_EDIT_CANVAS, ref_upscale=False)

# The default canvas, chosen by TIER rather than typed, so "render this
# cheaper" is one edit instead of a hunt through GRAPHS.
#
# **The structural fact that makes this worth having**, measured against
# `adapt_canvas` on 2026-08-16 by enumerating the whole legal family:
#
#   Every ratio at or above 1.75 costs the SAME. 1344x768, 1536x672,
#   1792x576 and 2016x512 are all 1008 tokens/frame, because the area cap
#   binds and simply trades width for height. **Going wider is free and buys
#   no speed.** The only cheap direction is toward square, and it is a smooth
#   32px ramp -- 20 legal landscape canvases between 1:1 and 16:9, each about
#   0.04 apart in ratio. Attention goes as the SQUARE of tokens, so small
#   width steps move the cost a lot.
#
# All four tiers are legal `adapt_canvas` outputs and inside the trained
# family. Three of them hit a common ratio exactly, which 1344x768 does not --
# it is 1.75, and true 16:9 (1376x768, 1.79) is not only absent from the
# family below the cap but costs MORE.
#
#   tier    canvas      ratio           tok/f  attention
#   full    1344x768    1.75            1008   1.00x   <- ships
#   near    1280x768    1.67 exact 5:3   960   0.91x
#   fast    1152x768    1.50 exact 3:2   864   0.73x
#   draft   1024x768    1.33 exact 4:3   768   0.58x
#
# **Which tier to use.** `fast` is the iteration canvas: exact 3:2, 27% off
# attention, and only 0.25 of ratio from what ships, so framing reads the
# same. `near` when a comparison has to stay visually close to the shipped
# canvas. `draft` changes the framing enough that it is for "does the pipeline
# run", not "does this look right".
#
# **The trap, and it is specific to Sol-Attn.** Sol needs roughly 60k tokens
# before it shows anything; below that a null result reads as "this knob does
# nothing". At 243 frames: `full` is 72,576 tokens, `fast` is 62,208 (just
# above), `draft` is 55,296 -- BELOW the floor. So a Sol measurement may use
# `fast` and must not use `draft`. Non-Sol work has no such constraint.
CANVAS_TIER = "full"

CANVAS_TIERS = {
    "full":  dict(width=1344, height=768),
    "near":  dict(width=1280, height=768),
    "fast":  dict(width=1152, height=768),
    "draft": dict(width=1024, height=768),
}

CANVAS = dict(CANVAS_TIERS[CANVAS_TIER])
FPS = 24.0

# Frame counts snap to a 17k+5 grid. 362 is the ceiling -- the longest length
# H3 was trained on -- and `h3_rules.MAX_LENGTH` is where that lives. Read its
# docstring before quoting it: it is an owner decision on thin evidence, not a
# measurement.
#
# **Restored 362 -> LONG_LENGTH on 2026-08-16, reverting the 2026-08-10 change
# to 345.** 345 was never a model boundary; it is the largest count *diffusers*
# will emit, and this repo spent a week presenting that as legality. The
# portability argument it rested on is now a question you ask explicitly
# (`h3_rules.reference_would_emit`) rather than a default that quietly caps the
# render.
#
# **Comparability, both directions.** The measurements above -- the ~24%
# attention ceiling, the 2.6% headroom ceiling, tau 1.3 against tau 2.0,
# int8_qk/pv at 1.16x -- were taken at 362, then the default moved to 345 and
# they were never re-taken. Moving back to 362 restores the length they were
# measured at. Anything measured BETWEEN 2026-08-10 and 2026-08-16 was taken at
# 345 and now sits one grid step below the default; the 5% length change should
# not move a ratio, but it was not re-checked in either direction.
LENGTH = 124
LONG_LENGTH = 362

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

# HOW a reference graph gets its reference capability. Canonical since
# 2026-08-16: the fl2va base plus the extracted delta LoRA above, rather than
# loading the separate ref2va checkpoint.
#
# **This is the one switch.** Flip it to False and regenerate, and every
# reference graph goes back to the ref2va checkpoint with no other edit. That
# is the whole reason it exists as a constant rather than a per-graph choice.
#
# Why the LoRA is the better default here, and it is not about quality:
#
#   It makes reference conditioning a KNOB. The checkpoint path has two
#   states, ref2va or not. The LoRA path has a continuous strength, so
#   "how much reference conditioning" becomes an axis you can sweep in one
#   graph instead of a variable you can only toggle by swapping a 21 GB file.
#
#   It collapses two code paths into one base. Every graph loads fl2va;
#   reference graphs add a patch. Fewer checkpoint permutations to keep
#   consistent, and a bench arm can move the strength without a reload.
#
# **Measured 2026-08-16 by `bench/analyze_ref_lora.py`, and the answer is two
# answers.** Keep COVERAGE and RECONSTRUCTION apart; conflating them is the
# same shape as the phantom-keys bug in provenance.py.
#
#   COVERAGE      verified exactly: 474/474 modules, zero unmatched either
#                 direction, 794 tensors = 264 lora pairs + full-rank .diff on
#                 every norm and .diff_b on every bias. A whole-model delta.
#
#   RECONSTRUCTION splits on the INT8 BOUNDARY, not on rank:
#                 * 64 non-quantized modules -- residual 0.0022, cosine
#                   1.0000. Essentially exact, and this includes all 51 rank-8
#                   adaln projections, which are ALSO the most-rewritten
#                   modules in the model (relative delta 1.86; final_layer's
#                   is 1.92, i.e. replaced rather than adjusted). The parts
#                   that change most are the parts reconstructed best.
#                 * 200 int8 modules -- NOT GRADEABLE from these files. The
#                   LoRA's delta there is ~0.36 quantization steps RMS, below
#                   the half-step needed to resolve it, and the only available
#                   target differences two INDEPENDENTLY quantized checkpoints
#                   so it carries both files' quantization error against a
#                   true relative delta of only 0.033.
#
# A naive read of those 200 says residual 1.05 / cosine 0.023, i.e. "the LoRA
# is wrong". It is not: the split falling exactly on the int8 boundary rather
# than on rank is the tell, since a bad extraction would not be perfect on 64
# modules and orthogonal on 200. **Do not quote that number as a defect.**
#
# **The consequence that bounds the whole approach:** the delta being applied
# is comparable to the quantization step of the checkpoint it is applied to.
# So a higher-rank extraction would be writing detail this int8 base cannot
# store -- the binding constraint is the quantization, not the rank. That is
# the argument against trying to extract a better LoRA.
#
# **Still open:** whether the two produce the same OUTPUT. ComfyUI applies it correctly -- no
# `__metadata__` and no `.alpha` tensors, so
# `comfy/weight_adapter/lora.py:247-250` takes the `alpha = 1.0` branch and
# applies the raw factors with no `alpha/rank` division, which is right for an
# SVD of a real weight difference. But "correctly applied" and "exact on the
# gradeable modules" are still not "renders the same". That needs a paired
# render, and until it exists do not call the two interchangeable.
# NAME MATTERS: this must NOT end in `_LORA`. `bench/check_lora_alpha.py`
# selects LoRA *filenames* with `n.endswith("_LORA")` and then resolves each on
# disk, so a bool called REF_VIA_LORA crashed it with a bare TypeError on
# 2026-08-16. `*_LORA` means "a LoRA filename" in this file; a flag about one
# has to be spelled differently.
#
# REVERSED to False on 2026-08-18, owner decision. The knob-and-one-base
# argument above still holds; it lost to a simpler one: the upcoming
# bandwidth/sparsity experiments want the fewest moving parts in the model
# path, and the LoRA is a moving part -- a dequantize/add/requantize round
# trip at load whose output-equivalence to ref2va was never verified (the
# paired render the block above calls for never ran). Reference graphs load
# the ref2va checkpoint directly. Flip back to True to restore the LoRA path;
# nothing else changed.
REF_LORA_ENABLED = False

# With the switch False every reference graph is on the checkpoint and the
# LoRA slot is free again. The paragraph below described the True state and is
# kept for whoever flips it back: the builder has ONE LoRA slot, so a
# reference graph that already spends it -- the three turbo probes -- cannot
# also carry the ref delta and keeps the checkpoint. Verified after the
# 2026-08-16 regeneration: 18 graphs on the LoRA path, 3 on the checkpoint,
# and the 3 are exactly the turbo ones. So under True, "reference
# conditioning" and "turbo distillation" do not stack -- a builder limitation,
# not a model one; stacking would need a second LoraLoaderModelOnly in the
# chain, which nobody has tried.


def ref_base_and_lora():
    """What a reference graph loads: (unet_name, (lora_name, strength) or None).

    Both builders call this, so the switch above cannot be honoured in one
    graph format and missed in the other -- which is exactly the drift this
    file exists to prevent.
    """
    if REF_LORA_ENABLED:
        return MODELS["unet_fl2va"], (REF_LORA, REF_LORA_STRENGTH)
    return MODELS["unet_ref2va"], None


# Where a two-stage split cuts the shared schedule, in steps. 2 of 8 is the
# shipped starting point, not a finding. H3's schedule is far more
# front-loaded than the model the split pattern came from: at video shift 12
# and 8 steps, seven of the eight evaluation points sit at sigma >= 0.8 and
# the final interval alone covers the bottom 63% of the range. Krea 2's
# k=2-3 sweet spot was still sigma 0.84 there; here k=3 is 0.9524. So the
# useful boundary is lower here and the sweep starts at 1.
SPLIT_AT = 2

# **`REF_VIDEO_LENGTH` was deleted on 2026-08-16. Do not reintroduce it.**
# Owner's reasoning, and it is the right one: a safe length for a reference
# arm is not a constant. It depends on how many references are wired, their
# kinds, their durations, the canvas, and whether they are upscaled -- so a
# single number can only be right for the one configuration it was measured
# on, and wrong-but-silent everywhere else. A test or a bench should render
# the duration that test calls for. The video-bearing arms now take
# `LONG_LENGTH` like everything else.
#
# The measurement that number came from is kept, because it is data and the
# ceiling it describes is real. **At 345 frames, 1024x768, references not
# upscaled, the reference arm peaked at 22,735 MiB of 24,564 and took 34.3
# minutes end to end** (2026-08-13). That is 1,829 MiB of headroom, and it was
# the *best* case -- the same arm at 1344x768 with references upscaled built a
# 182,092-token sequence and OOMed at step 4 of 16, after Sol-Attn, sage and
# ComfyUI's own SDPA each fell back correctly and still found no room.
#
#   345 frames -> 182,092 tokens   (OOM on 24 GB at 1344x768, refs upscaled)
#   209 frames -> 120,918
#   124 frames ->  82,686
#
# **So expect these arms to sit at or over the edge at 362.** That is ~5% more
# tokens against 1,829 MiB, and a reference video is the most expensive input
# in the model -- truncated to the GENERATED frame count, so this length costs
# twice over, once for the video rows and once for the reference rows. Read
# preflight before running one, and if it OOMs, shorten THAT run rather than
# reaching for a new constant.
#
# One thing not to relearn: shortening the render is the wrong lever for
# fitting a reference arm. The reference is truncated to the generated frame
# count, so cutting the render cuts the reference too -- the 124-frame arms
# this repo once shipped were testing a 5.2-second reference, and a reference
# arm that cannot carry a long reference is not testing the expensive case at
# all. Canvas and reference-image detail are the incidental costs to give up
# first; reference duration is the whole point.

# The canvas and reference-image policy that measurement bought. 1024x768 is
# 4:3 rather than the 1344x768 the rest of the repo defaults to -- a real
# change in what these arms look like, taken deliberately so the reference
# stays full length. `ref_upscale=False` leaves reference images at their
# native size instead of taking them to 2048 on the short edge.
#
# Spread into every video-bearing reference arm so the three numbers have one
# home. Editing them here moves all eight arms together, which is the point.
REF_VIDEO_CANVAS = dict(width=1024, height=768)
REF_VIDEO_BUDGET = dict(length=LONG_LENGTH, **REF_VIDEO_CANVAS,
                        ref_upscale=False)


# The three references `h3_probe_capture_ref3` wires, in socket order:
# character, garment, environment. Chosen for the MEASUREMENT rather than the
# picture, from `internal/reference_library.md`, which records every row cost
# below.
#
#   777 rows  0.56 ar  0.78 MP   the only asset that cannot fill a 2048 short
#                                edge from real pixels -- the undersized case
# 2,500 rows  1.00 ar  2.56 MP   legible text and a logo; the library calls it
#                                the probe for whether sparse attention drops
#                                high-frequency detail, which is the failure
#                                mode a routing study is about
# 4,128 rows  1.79 ar  4.23 MP   the size ladder's top end
#
# 5.3x span in DiT rows and aspects 0.56 / 1.00 / 1.79, deliberately, because
# a capture that varies neither cannot say whether reference load or reference
# SHAPE moves the router.
#
# **The environment reference contains two people.** The generated prompt scopes
# it to "setting, palette, and lighting", which steers away from them; widening
# that line makes this asset the wrong choice rather than a merely awkward one.
#
# `product_soccer_jersey` carries real brand marks. Internal capture artifact
# only -- see the flags section of `internal/reference_library.md`.
CAPTURE_REF_IMAGES = (
    "h3_refs/subject_performer_stage_662x1177.png",
    "h3_refs/product_soccer_jersey_1600x1600.png",
    "h3_refs/scene_loft_couch_duo_2752x1536.png",
)


# ---------------------------------------------------------------------------
# Where generated graphs live
# ---------------------------------------------------------------------------

# Graphs are foldered by USE CASE, relative to `workflows/`. Video is the
# primary case and stays at the root; the single-frame image gen/edit path is
# experimental and gets its own folder, so "what does this repo ship for
# video" is answerable by listing a directory.
#
# **This tuple is the discovery list, and it is shared on purpose.** Every
# check in `bench/` that walks the shipped graphs used a bare
# `workflows/*.json`, which is non-recursive -- so the moment the image graphs
# moved down a level, six checks would have gone on passing over a set that no
# longer contained them. That is the failure mode this repo keeps naming:
# correctly-absent and broken look identical from a green run. Adding a
# directory here is what makes every walker see it at once.
#
# `bench/` and `archive/` are deliberately NOT here. The stamped bench graphs
# read another pack's closure internals and are expected to break; the archive
# is history. Neither should be graded against the live schema.
GRAPH_DIRS: tuple[str, ...] = ("", "image")


def graph_paths(workflows, pattern: str = "*.json") -> list:
    """Every shipped graph under `workflows`, in a stable order.

    `workflows` is the repo's `workflows/` directory as a `pathlib.Path`.
    Returns paths, sorted within each directory, root first.
    """
    out = []
    for sub in GRAPH_DIRS:
        out += sorted((workflows / sub if sub else workflows).glob(pattern))
    return out
