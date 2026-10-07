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

# The only import in this file, and it stays that way: `h3_config` must be
# importable with nothing else on `sys.path` -- `bench/check_graph_discovery.py`
# imports it with only `workflows/` there, so a check whose own imports are
# broken is still audited.
from dataclasses import dataclass as _dataclass
from pathlib import Path


#: The ComfyUI-native INT8 ConvRot encoder (Comfy-Org's file), shipped on every
#: graph since 2026-08-27 (late) by owner decision. On the 13-row holdout it
#: sits an order of magnitude closer to the BF16 release at layer 50 than
#: either W4A16 AWQ artifact it replaced
#: (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`).
#:
#: **Against the bf16 pruned file on shipped graphs** (the release truncated
#: to the 50 layers H3 reads by `bench/convert_h3_bf16_encoder.py`): text
#: tokens stay close, while image tokens carry a heavy tail that the int8
#: DECODER adds -- the vision tower and embedding table are bf16 in both
#: files -- for seconds of encode time per prompt
#: (`bench/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`).
#: Whether the DiT responds to that tail is
#: `bench/measure_encoder_quant_dit.py`'s question.
#:
#: **That comparison is about two badly executed artifacts, not about the
#: method** (owner, 2026-09-20): we controlled the calibration, the group size
#: and everything else, did it poorly, and then priorities shifted. It is not
#: evidence that quantising our own encoder is unpromising, and nothing in
#: `bench/results/` establishes that. All four holdout arms also hold the
#: vision tower at BF16, so the record is silent on tower precision. The owner
#: primarily runs the bf16 pruned encoder, so this default is not their
#: working configuration. `docs/wiki/decisions.md`, 2026-09-20 and 2026-09-27.
#:
#: It loads through `MiniMaxH3EncoderLoader`, which is core's own `CLIPLoader`
#: plus checks and a stamped preprocessing contract (`h3_encoder_loader.py`'s
#: docstring lists them). Preprocessing is core's: the
#: still-image bounds are `process_qwen2vl_images`' own defaults, read out of
#: core by `h3_encoder_loader.native_encoder_contract`, never typed here.
#:
#: The W4A16 AWQ artifacts (`..._w4a16_awq.safetensors`, `..._v2-comfy`) and
#: the adapter that loaded them (`h3_awq_encoder.py`, `MiniMaxH3AWQEncoderLoader`)
#: were deleted on 2026-09-13 with the lane they served
#: (`docs/roadmap.md` "Closed lanes", `docs/wiki/decisions.md`). Their records
#: under `bench/results/` and `docs/research/` stand as history.
ENCODER_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

MODELS = dict(
    unet_fl2va="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    unet_ref2va="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    # The fl2va PDD8 backbone bake, built 2026-09-05 by
    # `bench/bake_pdd_checkpoint.py`: the shipped pruned checkpoint with its
    # 200 int8 backbone linears replaced by quantise(W_release + PDD delta at
    # strength 1.0), round-to-nearest, every other key copied byte for byte.
    # The strength-zero control reproduced the shipped codes up to rounding
    # ties (`bench/results/2026-09-05_bake_identity_control.json`), so this is
    # the shipped file with the LoRA folded in and nothing else moved. It
    # loads ONLY with `PDD_FL2VA_STRIPPED_LORA`; the node refuses the full
    # sidecar on it (double apply) and refuses the stripped one on the base.
    # `docs/pdd_artifacts.md` is the inventory; `_s1` in the name is the
    # baked strength.
    unet_fl2va_pdd8_baked=("minimax_h3_fl2va_pruned_int8_convrot"
                           "_pdd8_baked_s1.safetensors"),
    # One fl2va/ref2va hybrid, int8_convrot, fl2va everywhere except the adaln
    # projections: `adaln_all` is built here by `bench/build_hybrid.py` (all
    # 50 blocks plus final_layer). It exists to ask whether an fl2v distill
    # LoRA transfers to reference work better on fl2va's linears than on
    # ref2va's; `docs/roadmap.md`, the regime section. The filename ends
    # `-int8`, not `_int8_convrot`; `substrate.py` tags it. A second hybrid,
    # `unet_hybrid_b30` (the HF release, blocks 30-49), was retired on
    # 2026-10-05: its download was gone from the box and the owner called it
    # long deprecated (`docs/wiki/decisions.md`). `build_hybrid.py` once
    # reproduced it byte for byte as its control.
    # FastVideo's FastH3 8-step V2 (HF FastVideo/FastVideo-FastH3-Comfy): a full
    # distilled T2VA DiT (data-free DMD2, trained WITH VSA-H3), pruned int8
    # convrot with its OWN curve basis and time table (not fl2va's; measured
    # 2026-09-26, bench/results/2026-09-26_fasth3_weights.md), plus a `to_gate_compress` per main block
    # that core's model detection now reads from the keys
    # (`comfy/model_detection.py`, `gate_compress`). It also quantizes the token
    # refiner, which ours keeps bf16. T2VA only: FL2VA and Ref2VA were not
    # distilled. Sampling contract: `FASTH3_*` below.
    unet_fasth3_v2="fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
    unet_hybrid_adaln_all="minimax_h3_hybrid_fl2va_ref2va_adaln_all-int8.safetensors",
    # `unet_vsa`, kijai's experimental FastVideo VSA checkpoint, left with
    # `MiniMaxH3VSAAttention` and its two probe graphs in 0.173.0 (owner,
    # 2026-09-29); `docs/research/vsa/fastvideo_vsa_checkpoint.md` keeps what it was.
    clip=ENCODER_INT8,
    # **The INT8 ConvRot build, by owner decision 2026-09-26** ("yes
    # switch"), after the owner could not tell it from fp16 on a 345-frame
    # clip pair. Measured: its decoder is the only quantized half, and its
    # encoder encodes bit-identically to the fp16 file's
    # (`bench/results/2026-09-26_vae_encoder_int8_file.json`), so the switch
    # moves the decode only. That decode is faster, holds less VRAM, and adds
    # no flicker (`bench/results/2026-09-26_vae_decoders_345f.md`).
    #
    # History: it shipped from 2026-08-10, was removed by owner decision on
    # 2026-08-21 in favour of VIDEO_VAE_FP16 (below), and came back on
    # 2026-09-26 when ComfyUI's own templates had moved to it
    # (`docs/sol_upstream.md`). MiniMaxH3VAEPrecision refuses to cast its
    # quantized decoder.
    video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)

#: The fp16 video VAE: the shipped build until 2026-09-26, and the reference
#: the INT8 build is measured against (`bench/compare_vae_decoders.py`). It is
#: the most faithful build in ComfyUI's format. **Measured 2026-08-21** against
#: the official release's fp32 weights (`MiniMaxAI/MiniMax-H3`,
#: `video_vae/source/model.safetensors`): all 559 shared tensors match in name
#: and shape, median relative delta 2.07e-4 and max 2.48e-4, which is fp16's
#: own rounding floor and no more -- so this file is that fp32 cast down, with
#: `latents_mean` and `latents_std` folded in as tensors. Only the fp32
#: original is more faithful, at twice the size; a bf16 conversion would be
#: less, since fp16 carries three more mantissa bits.
VIDEO_VAE_FP16 = "minimax_h3_video_vae_fp16.safetensors"


#: Encoder files core's own `CLIPLoader` (type `minimax`) loads, and therefore
#: the only files `MiniMaxH3EncoderLoader` (core's load plus guards) can open.
#: The generator refuses any other encoder name, so a graph never names a
#: file its loader cannot open.
CORE_LOADED_ENCODERS = frozenset({
    ENCODER_INT8,
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
})

# `euler` / `simple`. **Owner decision 2026-10-05: no `er_sde` anywhere; every
# graph that carried it runs Euler.** A default rather than a finding, like the
# one it replaces: `er_sde` was the owner's pick on 2026-08-15 over core's
# base-template `res_multistep` because it "looked more interesting on the
# clips actually rendered". What moved it is that every reference
# implementation steps the base with Euler at eta 0 (sglang, vllm-omni and
# diffusers' `MiniMaxH3Scheduler`; vllm-omni's 2026-10-04 `res_multistep` is
# opt-in and leaves Euler the default), and that a stochastic sampler was the
# standing obstacle to every numeric A/B here. Core's base template still
# ships `res_multistep`, so this is upstreams that disagree: an ordinary
# judgement call, not the adopt-upstream rule.
#
#   euler      One model eval per step, the same cost as `er_sde` and
#              `res_multistep`. Read in `comfy/k_diffusion/sampling.py`.
#              Deterministic: no noise after the initial draw, so the base and
#              the distills now share starting noise at one seed, and the
#              step cache's deterministic-sampler caveat (CACHE_NODE below) no
#              longer applies to the shipped default. First order, where
#              `er_sde` was a third-order multistep: at `steps` below the
#              discretization error is larger per step, and **no render here
#              has judged 16 Euler steps on the base** (the references run
#              their Euler at their own, larger, default step count). Reasoned,
#              not measured; the step count is the knob to revisit if base
#              clips soften.
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
#              the schedule. **That sentence was prose and enforced by nothing
#              until 2026-08-23; `bench/check_distill_grid.py` is now its
#              control**, against the grid the vendor publishes in
#              `coderef/Minimax-H3-Turbo/README.md` rather than against a
#              number computed here. Measured there: `simple` is EXACT at 4 and
#              8 steps (it reads the discrete 1,000-entry table, and both
#              divide 1,000), where `beta` is off by 0.10, `normal` by 0.67 and
#              `sgm_uniform` by 0.007. At 16 steps -- this line's own value --
#              `simple` quantizes by ~0.002, which is why the check grades only
#              the graphs that load a distilled LoRA: the base checkpoint was
#              never fitted to a step grid, so the vendor rule does not bind it.
#
#   steps 16   Measured 2026-08-06 at 362 frames: 20 steps 765.4 s, 16 steps
#              669.2 s (-12.6%), 12 steps 508.5 s. 12 was rejected because it
#              stops following the prompt -- the test prompt's third scripted
#              shot at 00:10 never happens. Not smeared, no late-clip artifact,
#              invisible in stills and to a convergence check. Any future step
#              reduction needs prompt adherence as a gate. That judgement was
#              made on `res_multistep` and has not been re-run on `euler`.
#
# Every timing recorded in this repo before 2026-08-15 was taken on
# `res_multistep`, and every base render from then to 2026-10-05 on `er_sde`.
# The three are step-cost-neutral so timings should carry, but they were not
# re-taken; a base clip rendered before the change is not seed-comparable to
# one rendered after it.
SAMPLING = dict(sampler="euler", scheduler="simple", steps=16, denoise=1.0)

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
#
#                     **Moved to 1.0 on 2026-08-20 by owner decision**, in
#                     SOL_RECOMMENDED_CUDA below (this Triton-era dict keeps
#                     1.3 as the record of what was measured). The decision is
#                     the owner's, not a measurement: with the 4-step
#                     distilled LoRAs as the working regime, 1.3 has to earn
#                     its way back by showing no difference from 1.0 across
#                     many seeds judged blind while buying meaningful speed.
#                     The speed half is a `--set SolAttnMiniMax.tau=1.3` bench
#                     patch; the quality half is an 8-seed blind session, not
#                     yet run. Reversal condition, stated: that session finds
#                     the two indistinguishable AND the patch arm's sampler
#                     time is materially lower.
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
# `selection` and `reuse_qkv_memory` do not exist there. Merging them
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
#   min_tokens 4096   The Triton-era value. SUPERSEDED 2026-08-27 -- the CUDA
#                     recipe now takes the node's 12288; see its own note.
#
#                     **Corrected 2026-08-28.** This said H3's DiT "has exactly
#                     ONE attention site... the small calls that once suggested
#                     otherwise were SageChainAssert's own probes." That is a
#                     retracted claim (retracted 2026-08-14) and it was wrong
#                     twice over: the refiner calls are model code, not
#                     instrumentation. `docs/SOLATTN.md` owns this and states
#                     it -- one SOURCE LINE
#                     (`comfy/ldm/minimax/model.py`'s single
#                     `optimized_attention` inside `Attention.forward`) but 52
#                     MODULES, because `RefinerBlock` and `DiTBlock` both
#                     instantiate that same `Attention`: 50 DiT blocks at the
#                     full packed length plus 2 token-refiner blocks on the
#                     text span alone.
#
#                     The min_tokens conclusion is unaffected, and now rests on
#                     the right reason: the refiner calls sit at ~311 rows,
#                     below BOTH 4096 and 12288, so the two thresholds select
#                     identically on them. What separates the two values is the
#                     DiT calls, which is the note below.
#
#                     **Corrected 2026-08-27.** This said the two values "select
#                     the same thing at every length anyone renders", reasoning
#                     from S = 7,194 at 22 frames being "already above 4096".
#                     Above 4096 is not the test -- 7,194 is BELOW 12288, so at
#                     that length the two disagree outright: 4096 runs Sol and
#                     12288 keeps it dense. The claim is true only for the
#                     lengths this repo actually renders, which are 31k-128k
#                     tokens and far above both. Say that, not the stronger
#                     thing.
#
#                     So 4096 is a deliberate choice to engage Sol BELOW the
#                     crossover the node's own default encodes, in a regime
#                     nothing here has measured and nothing here renders.
#   reuse_qkv_memory  False. Verified numerically identical to the normal entry
#                     (cos agreeing to six digits), so it cannot change output,
#                     and upstream reports it drops attention's peak below the
#                     FFN's. Left off only because it is a separate question
#                     from the migration. Cheap win when someone measures it.
#: **RETIRED 2026-09-11: empty, so no step count gets a dense last step.**
#: The owner's rule is to take upstream's default where sglang and ComfyUI
#: agree, and both run Sol through the last step: sglang's `sol_attn` backend
#: has no end cutoff and core's `BlockSparseAttention` defaults `end_percent`
#: to 1.0 (`docs/research/sglang_comparison.md`, the 2026-09-11 defaults
#: section). `SOL_RECOMMENDED_CUDA` carries 1.0 now. The name stays because
#: `sol_for_graph` reads it; the history below is why it held
#: `{4: 0.74, 6: 0.83, 8: 0.87}` from 2026-08-26 until then.
#:
#: `end_percent` per sampler step count, so the FINAL step runs dense.
#:
#: **This one is ours, not the vendor's, and it is a fix for something that
#: broke silently.** Sol's window is a sigma band, so which steps land inside
#: it depends on the step count. At 16 steps the last step sits at sigma 0.447,
#: below the band, and sage takes it dense. At 8 steps the last step is 0.632
#: -- still INSIDE the band -- so it runs sparse. Halving the steps for PDD
#: therefore removed the dense tail without anyone choosing to.
#:
#: That is the worst step to lose. At shift 12 the final step covers sigma
#: 0.632 -> 0, the largest jump in the schedule and where high-frequency detail
#: resolves, and it is also where PDD's fused heads deviate most from the base
#: (0.0146 against 0.0047 at the first step). Two approximations were stacking
#: on the one step that can least afford either.
#:
#: Derived by walking `comfy.samplers.calculate_sigmas` at shift 12 and finding
#: the largest `end_percent` whose sigma still exceeds the final step's, so the
#: last step falls outside the band. Costs one sparse step of six at 8, one of
#: three at 4.
#:
#: NVLabs express the same intent as a COUNT (`SOL_ATTN_FIRST_DENSE_STEPS=10`
#: of 50), which does not survive a step-count change either -- ten dense steps
#: of eight is all of them. Neither recipe covers 8 steps, because their
#: reference config runs 50. This restores what 16 steps gave us.
#:
#: **Keyed on step count ALONE, and the three other candidates were checked
#: rather than assumed:**
#:
#:   shift        cancels. `percent_to_sigma` and `calculate_sigmas` apply the
#:                same shift, so both the band floor and the last step's sigma
#:                move together. Verified 2026-08-26: shift 12 and shift 6 both
#:                want 0.87 at 8 steps, 0.83 at 6, 0.74 at 4. 11 shipped graphs
#:                run shift 6 and need no separate row.
#:   length,      no effect. The window is a SIGMA band and sigma is a position
#:   resolution   on the trajectory; it does not know the sequence length. A
#:                5-second 768x1024 clip and a 15-second 1344x768 one at the
#:                same step count have identical sigmas and identical splits.
#:                (`min_tokens` IS length-dependent -- see its note below.)
#:
#: **The node cannot do this itself, which is why it is baked in here.**
#: `vendor/sol_attn_minimax.py:654` calls `percent_to_sigma` at PATCH time and
#: stores fixed sigma thresholds; at run time it only compares the current
#: sigma against them. At patch time the step count is not knowable -- the
#: scheduler is downstream of the Sol node. So `start_percent`/`end_percent`
#: are static widgets and the generator is the only place that can pick the
#: right one per arm.
#:
#: **The consequence, and it is a real edge:** loading a shipped graph and
#: changing `steps` by hand leaves `end_percent` stale, and nothing at run time
#: will say so. `bench/check_attention_defaults.py` catches it for shipped
#: graphs; a hand-edited one is on the person editing it.
SOL_END_PERCENT_BY_STEPS = {}  # retired 2026-09-11; was {4: 0.74, 6: 0.83, 8: 0.87}

#: ComfyUI core's own Sol node (`comfy_extras/nodes_sparse_attention.py`,
#: Comfy-Org/ComfyUI#16072, merged 2026-09-06), for an arm that renders it
#: against ours. No graph carries it since 2026-09-15, when
#: `h3_probe_t2v_sol_core` retired with its sage-floor A/B; the generator's
#: `sol_impl="core"` still builds one.
SOL_CORE_NODE = "BlockSparseAttention"

#: Its inputs at the node's OWN schema defaults, in API form. **Inherited, not
#: chosen:** read from `BlockSparseAttention.define_schema` at core `1f641fd9`
#: on 2026-09-10, and compared against the live server's /object_info by
#: `build_workflows.py` whenever it validates, so a core release that moves a
#: default fails the build rather than leaving this a quiet copy. Its
#: "sol-attn" method is the adaptive-tau selection, the counterpart of our
#: node's "adaptive tau"; `extra_tokens` is kitchen's `token_aug`, applied to
#: every eligible call.
SOL_CORE_DEFAULTS = {
    "selection": "sol-attn",
    "selection.tau": 1.3,
    "start_percent": 0.2,
    "end_percent": 1.0,
    "dense_blocks": "",
    "min_tokens": 12288,
    "extra_tokens": 256,
    "sink_conditioning": "exact_kv_and_rows",
    "verbose": False,
}

#: Option A candidate: shields the top 3 worst middle blocks (>24% error) plus
#: the terminal block 49 (which feeds final_layer.video_out). Adopted 2026-10-02.
SOL_DENSE_OPTION_A = "39,41,42,49"

#: Option B candidate: shields the entire 23%–26% middle error plateau plus
#: the terminal block 49.
SOL_DENSE_OPTION_B = "39,40,41,42,49"

#: Option C / Full Ridge Shield: shields blocks 38 through 42 plus
#: the terminal block 49. Discovered in Test 9A to eliminate the Block 38 spike
#: under tau=1.3, reducing peak network error to 17.73% and ref_img error to 19.37%.
SOL_DENSE_OPTION_C = "38,39,40,41,42,49"
SOL_DENSE_RIDGE_SHIELD = SOL_DENSE_OPTION_C

#: Historical 2026-09-25 tail default (from unrotated INT8 K-norm outliers):
SOL_DENSE_HISTORICAL_TAIL = "45,48,49"

#: The Sol node's `dense_blocks` default, SOL_DENSE_OPTION_C. Adopted
#: 2026-10-02 on local probe error and kept the same night by owner decision:
#: the output-level panel found the dense_blocks choice does not move the
#: verdict, and C is the list SOL_SINK_DEFAULT was judged with
#: (`bench/results/2026-10-02_sol_dense_blocks_panel.md`). Mirrors
#: `sol_attn_h3.py::SOL_DENSE_TAIL`; `bench/check_attention_defaults.py`
#: holds the two together.
SOL_DENSE_TAIL = SOL_DENSE_OPTION_C

#: The Sol node's `sink_conditioning` default; mirrors
#: `sol_attn_h3.py::SOL_SINK_DEFAULT`, whose comment carries the evidence
#: (owner decision 2026-10-02, measured; `exact_kv_and_rows` before).
SOL_SINK_DEFAULT = "exact_kv_and_all_rows"

SOL_RECOMMENDED_CUDA = dict(
    # Keyed to `MiniMaxH3Sol`'s inputs since the redesign (2026-09-27,
    # `docs/research/2026-09-27_sol_node_redesign.md`). Retired with the old
    # node: `selection` (the SLA lane closed, so tau is the only selection),
    # `pooled_tail` (always on; it is the method), `morton` and
    # `morton_curve` (Morton closed), `token_aug_blocks` (now inside the
    # `token_routing` combo), and the `qk_balance`/`rotate` booleans (one
    # `quantizer` choice). Their history is in git before that date.
    # 1.0 since 2026-08-20, owner decision; see the tau note above for the
    # reversal condition. 1.3 was the value every Sol number before that date
    # was measured at.
    tau=1.0,
    # **0.2 here; 0.0 on every PDD graph since 2026-10-01** (SOL_PDD_OVERRIDES,
    # owner decision, measured). FlashGen graphs keep 0.2: 0.184.1
    # extended 0.0 to them unmeasured and the owner reverted it the same day
    # (SOL_DISTILL_LORA_OVERRIDES). On the t2v PDD8-to-FlashGen finish the owner
    # could not tell 0.0 from 0.2 in five blind pairs, and 0.0 cut the sampler
    # by roughly a fifth (bench/results/2026-10-01_start_percent_panel.md).
    # Still unmeasured for the base graphs, which keep 0.2: the node's tooltip
    # justifies it only as "the paper uses 0.2". Until 2026-10-01 this said it
    # had never been measured at any value, which was true then.
    #
    # Priced 2026-08-27, arithmetic not measurement: it forces the top of the
    # trajectory dense and that costs a FLAT 25% of evaluations at every step
    # count -- 4 of 16, 2 of 8, 1 of 4. Scale-invariant, because it is a fixed
    # fraction of a schedule that is uniform in base sigma. So it is not a
    # low-step problem; it is a constant quarter of Sol's opportunity.
    #
    # At 0.0 the 4-step arm would go from 2 sparse steps to 3. Whether that is
    # free or harmful is open both ways: the first step's input is pure noise,
    # which argues it is the most redundant place to route sparsely, and it also
    # sets global composition, which argues a routing error there propagates
    # into everything after. The speed half is one bench patch; the quality half
    # is a numerical knob and needs `docs/eval_comparison.md` section 3.
    start_percent=0.2,
    # 1.0 since 2026-09-11, adopting upstream: sglang's `sol_attn` backend has
    # no end cutoff and core's `BlockSparseAttention` defaults to 1.0. It was
    # 0.9, which kept the last step dense at 16 steps; SOL_END_PERCENT_BY_STEPS
    # records why that tail existed. Sol now runs through the last step, and
    # the steps before `start_percent` go to the dense backend under Sol:
    # `DENSE_BACKEND_NODE` on the default graphs, sage only on the arms
    # `bench/check_attention_defaults.py::FLOOR_STEMS` names.
    end_percent=1.0,
    # 4096 against the node's own 12288. Both are no-ops **at the lengths this
    # repo renders** -- every DiT call is at the full packed length, 31k-128k
    # tokens, far above either threshold, and every token-refiner call is ~311
    # rows, far below both.
    #
    # **CLOSED 2026-08-27, raised 2026-08-26.** The question was whether the
    # figure or the reasoning was off, given S = 7,194 at 22 frames sits
    # between the two. The reasoning was: being above 4096 does not make the
    # thresholds agree, being above 12288 does. At 22 frames they genuinely
    # disagree and 4096 is the permissive one. Arithmetic, not a measurement,
    # so it needed no run. Nothing is at risk -- we do not render there -- but
    # the unqualified "both select the same thing" is retired.
    # **12288 since 2026-08-27, adopting the node's own default; 4096 before.**
    #
    # The reason is what this gate actually chooses between, which is not what
    # the name suggests. Below the threshold Sol declines and the call falls
    # through to `previous`, which is a good dense kernel, not dense torch
    # (`vendor/sol_attn_minimax.py::make_override`'s `dense()`, read
    # 2026-08-27). When this was written that was SAGE on every graph. Since
    # 2026-09-15 the default graphs install `DENSE_BACKEND_NODE` below
    # (kitchen's `int8_attention`) under Sol, and sage remains only on the arms
    # `bench/check_attention_defaults.py::FLOOR_STEMS` names. So this is Sol
    # against the kitchen kernel, and the crossover against it is unmeasured.
    #
    # The sage argument moved the crossover UP. `docs/SOLATTN.md` puts sage
    # about 2.7x ahead of torch's flash backend on this shape, so a sparse
    # kernel has to clear a good dense one, not a naive one. `SOL_CUDA_DEFAULTS`
    # below already
    # recorded the direction -- upstream puts the crossover near 12k and "4096
    # engages Sol-Attn in the regime where it costs time" -- and the sage
    # baseline only sharpens it.
    #
    # **Changes nothing this repo renders**, which is why it is safe to make on
    # an argument: every DiT call is 31k-128k tokens and every token-refiner
    # call is ~311 rows, so both values select identically. The reachable gap is
    # ~22 frames / S ~ 7,194, where 4096 handed the call to Sol at a length
    # nobody has shown Sol wins. This removes that.
    #
    # Still unmeasured on this box, and this is deference, not evidence. What
    # overturns it: measuring the actual Sol-against-sage crossover here. That
    # measurement would beat both values, including this one.
    min_tokens=12288,
    # `exact_kv_and_all_rows` since 2026-10-02 (SOL_SINK_DEFAULT above):
    # every conditioning query row exact. The roadmap's 2026-09 plan made
    # this the flip "if all-rows loses on no scene"; on the 2026-10-02 panel
    # it lost on none and was the one Sol arm rated fine on dialogue.
    sink_conditioning=SOL_SINK_DEFAULT,
    verbose=True,
    # Empty again by owner decision on 2026-09-02. `0-2,32` shipped from
    # 2026-08-29 until this correction, but it came from an EXPERIMENT rather
    # than a production-default result: `2026-08-29_block_propagation.json`
    # sampled only 11 of 50 blocks, on one base-model trajectory at a specially
    # isolated sigma, with one output-distance proxy and no perceptual A/B. It
    # did not test the PDD head, canonical PDD active sigmas, a complete block
    # sweep, or interactions among blocks. That is enough to motivate a future
    # arm and not enough to hardcode four bypasses into every workflow.
    #
    # Keep the mechanism: an experiment may still set any block list explicitly.
    # Do not repopulate the shared default until all 50 blocks have been measured
    # on the actual target schedule/model and the resulting set survives a
    # set-level, multi-scene validation. The 2026-09-02 production-geometry
    # route capture deliberately runs with this empty so blocks 0-2 and 32 are
    # observable instead of bypassed.
    #
    # **Updated 2026-10-02 (Option C / Full Ridge Shield adoption)**: Under quantizer="rotated"
    # (Hadamard rotation), block 48 error dropped to 7.14% and 45 to 9.55%,
    # while middle blocks 38–43 spike to ~25% error. dense_blocks adopts Option C
    # ("38,39,40,41,42,49") via SOL_DENSE_TAIL, shielding the full middle plateau
    # (38, 39, 40, 41, 42) plus terminal block 49 (feeds final_layer.video_out),
    # dropping peak network error below 18% (17.73%) even under high sparsity tau=1.3.
    # Option B is SOL_DENSE_OPTION_B; Option A is SOL_DENSE_OPTION_A. `docs/wiki/decisions.md`.
    dense_blocks=SOL_DENSE_TAIL,
    # `quantizer` "balanced" = qk_balance on, rotate off, the shipped state
    # until 2026-09-27; "rotated" since (below).
    # qk_balance: on since 2026-09-15, owner decision; off from its introduction that
    # morning. The kernel's own per-head q/k channel rebalancing inside its
    # INT8 quantizers, carried on the owner's kitchen fork (h3-frontier) and
    # graded on captures by bench/grade_channel_balance.py; exact for every
    # attention score, so what it changes is the INT8 error on the blocks
    # whose K-norm is lopsided (docs/h3_block49_quant_error.md). Measured on
    # captures, not judged blind: it lowers Sol's quantization error on block
    # 49 and is neutral on blocks 0 and 32, where its per-head gate stays shut
    # (bench/results/2026-09-15_channel_balance_kernel_b{49,0,32}_s15.json),
    # at no measurable wall time (bench/results/2026-09-15_block49_diner_batch.md).
    # Adopted with the kitchen dense floor (DENSE_BACKEND_NODE below): with
    # sage out of the default chain, Sol's routed steps are the only unrotated
    # INT8 left on those blocks. h3_probe_t2v_ck and h3_probe_t2v_exact_tail
    # carry False as declared deviations (bench/check_attention_defaults.py).
    # rotate: off (2026-09-15, night). Sol's Hadamard rotation of q/k before INT8 (kitchen
    # fork branch h3-sol-rotate): graded on captures at about half of Sol's
    # block-49 quantization term (bench/results/2026-09-15_sol_rotate_*.json);
    # an experiment until a witness render says otherwise.
    # **"rotated" since 2026-09-27** (owner decision, measured): on captures of
    # the shipped PDD8 t2v and ref2va graphs it has the lowest Sol quantization
    # error on all 28 Sol-block cells and costs less than "balanced"; the total
    # error moves little because sparsity dominates
    # (bench/results/2026-09-27_sol_redesign_test2.md). "balanced" before, the
    # history above. Changes every Sol render's output from that date.
    quantizer="rotated",
    # **Token routing OFF everywhere.** One DynamicCombo since the redesign
    # (`sol_attn_h3.py::SOL_ROUTING_CHOICES`); `custom` carries its own list.
    # Comfy-Org/comfy-kitchen #156, released in 0.2.33, kept the same tree as
    # the head this repo graded, so the 2026-09-04 grade transferred without
    # being redone (`bench/results/2026-09-08_kitchen_0233_blk_cnt_rebase.json`).
    #
    # It ships off, and it is per block rather than global, because the grade
    # is per block and mixed: on the captured Base16 cells token routing
    # lowered Sol's error against exact attention on four of the five captured
    # blocks at every captured step, and RAISED it on block 49 at every step.
    # `docs/research/2026-09-04_sol_token_aug_grade.md` owns the numbers. A
    # global switch can only express the configuration that grade says is
    # wrong.
    #
    # **No render has ever been judged with it on.** The whole case for it is
    # offline error on one capture, which is not the standard anything ships
    # on here; `docs/research/2026-09-05_token_aug_plan.md` stage 5 is the
    # blind pair that would change that.
    #
    # If it is ever turned on, 64 is the budget: 64, 128 and 256 measured
    # indistinguishable in accuracy AND in isolated kernel time
    # (`bench/results/2026-09-04_sol_exact_random_1128df6_token_aug_timing.json`),
    # so the wider budgets buy nothing while switching it on at all costs.
    token_routing="off",
)

# The Sol config for a PDD arm. **Since 2026-09-11 it is the base recipe
# unchanged**: its one override, the narrower `end_percent` described below,
# went with the dense last step (see SOL_END_PERCENT_BY_STEPS). The notes
# below are the history of that override. `dense_blocks` is inherited empty;
# any protected-block list is an explicit experiment, not a PDD default.
#
#   end_percent   0.74 at EVERY PDD step count, against the step-count
#                 derivation in SOL_END_PERCENT_BY_STEPS, which gives 0.87 at
#                 8. This one is ARITHMETIC, not a measurement, and it is the
#                 only knob here that is: `percent_to_sigma(0.75)` is 0.8
#                 EXACTLY at shift 12, and 0.8 is index 24 of PDD's 32-point
#                 grid -- where the final block of the 4-evaluation partition
#                 begins. So 0.74 is one widget step onto the safe side of
#                 "run dense over the coarsest schedule's last block": a fixed
#                 point on the SIGMA PATH rather than on the step grid, which
#                 is why one constant serves both PDD step counts where the
#                 step table needs a row each. Strictly more conservative than
#                 the derivation, never less -- at 8 steps it takes the
#                 second-to-last evaluation dense as well as the last, so
#                 sparse coverage goes 5 of 8 to 4 of 8.
#                 **Untested prediction it makes:** at 8 evaluations 0.87
#                 should be WORSE than 0.74 if the mechanism is trajectory-
#                 pinned rather than grid-pinned. Nothing has run that.
#
# **Dropped 2026-08-29, all three for the same reason: no evidence, and no
# effect either.** Keeping them made SOL_PDD_CUDA look like a five-knob
# finding when it was one derivation and one measurement.
#   min_tokens 11776   Inert. Every PDD graph packs 60,972 to 113,032 rows
#                      (`bench/preflight_graph.py` over all of them), so 11776
#                      and the inherited 12288 select identically on every one.
#   morton_curve       Inert while `morton=False`, which both configs are. The
#     "2d_frame"       inherited `3d` at least has a centroid-fidelity
#                      measurement behind it.
#   dense_blocks       See above -- the widening to "0-5,48-49" was extrapolated
#     "0-5,48-49"      from a three-point decay and is withdrawn.
#
# Everything else is shared with SOL_RECOMMENDED_CUDA and reaches here through
# it: selection, tau 1.0, sink_conditioning, morton off,
# centroid_tail, reuse_qkv_memory. Spelled as an override dict rather than a
# second full literal, because a full copy is the second copy this file forbids.
#
# **What none of this establishes.** No protected-block list is validated for
# the PDD head or its canonical active sigmas. The base-model propagation probe
# that temporarily installed `0-2,32` did not establish that result either, so
# both configurations inherit the empty default while the instrumentation lane
# gathers the missing all-block evidence.
# Empty again since 2026-10-05: the owner reversed their 2026-10-01 call
# ("0.0 makes no sense to have on. zero dense doesnt make sense"; "no 0.0
# default for sol anywhere"), so PDD graphs take SOL_RECOMMENDED_CUDA's
# start_percent 0.2 like every other graph. The 2026-10-01 panel
# (bench/results/2026-10-01_start_percent_panel.md) had set 0.0 here: on the
# t2v finish the owner could not tell it from 0.2 in five blind pairs and it
# saved two of eight evaluations. What it never tested was a render that
# depends on a reference cue; the ref2va PDD8 arms of 2026-10-05 ran Sol from
# step zero and lost the subject's video reference
# (bench/results/2026-10-05_masked_v2v_motion_arms.md). Was empty from
# 2026-09-11 (retired end_percent=0.74) to 2026-10-01.
SOL_PDD_OVERRIDES = dict()

SOL_PDD_CUDA = dict(SOL_RECOMMENDED_CUDA, **SOL_PDD_OVERRIDES)

# The distill LoRAs applied at the call (FlashGen) keep SOL_RECOMMENDED_CUDA's
# start_percent 0.2. **Reverted by the owner the same day it was set**: 0.184.1
# gave them 0.0, extended from the PDD measurement and never measured on them (the
# panel behind SOL_PDD_OVERRIDES rendered the PDD8-to-FlashGen finish, whose
# FlashGen pass starts at sigma 0.8, so no FlashGen render from pure noise
# was judged at 0.0); 0.184.3 empties this again. Kept, empty, as the one place a
# FlashGen-only knob goes, apart from SOL_PDD_CUDA. `SOL_DISTILL_LORA_FILES`
# (beside the FlashGen constants) says which files it applies to. PDMD's files
# were in that set until the lane was retired on 2026-10-05.
SOL_DISTILL_LORA_OVERRIDES = dict()  # was start_percent=0.0 in 0.184.1 only

SOL_DISTILL_LORA_CUDA = dict(SOL_RECOMMENDED_CUDA, **SOL_DISTILL_LORA_OVERRIDES)


def sol_for_graph(pdd, steps, distill_lora=False):
    """The Sol config one graph should carry, from what the graph IS.

    The single resolver for both halves of the question, because they were
    two copies before: the generator derived `end_percent` from the step
    count and `bench/check_attention_defaults.py` re-derived the same lookup
    to grade against. A PDD branch written twice would drift the same way.

    `pdd` -- the graph loads a Parallel Decoding Distillation LoRA -- takes
    SOL_PDD_CUDA whole, at every step count, so `steps` is ignored on that
    branch. Everything else takes SOL_RECOMMENDED_CUDA with `end_percent`
    lowered per SOL_END_PERCENT_BY_STEPS. The table is empty since
    2026-09-11; SOL_PDD_OVERRIDES is empty again since 2026-10-05 (it carried
    `start_percent` 0.0 from 2026-10-01), so the PDD branch no longer differs
    from the recommended config; the distill branch below is empty since 0.184.3.

    `distill_lora` -- the model carries a FlashGen LoRA
    (`SOL_DISTILL_LORA_FILES`) and no PDD -- takes SOL_DISTILL_LORA_CUDA
    whole, the same way, since 2026-10-01. PDD wins when both are set.
    """
    if pdd:
        return dict(SOL_PDD_CUDA)
    if distill_lora:
        return dict(SOL_DISTILL_LORA_CUDA)
    end = SOL_END_PERCENT_BY_STEPS.get(steps)
    sol = dict(SOL_RECOMMENDED_CUDA)
    if end is not None:
        sol["end_percent"] = end
    return sol

# **What `MiniMaxH3Sol` gives you untouched**: its `define_schema` defaults,
# for an arm that wants the node's own answer rather than the recipe's. Since
# the redesign (2026-09-27) the node's defaults ARE the recipe
# (SOL_RECOMMENDED_CUDA): `quantizer` defaults to "rotated" in the node, so
# the two no longer disagree on the balance the way `MiniMaxH3SolAttn`'s
# `qk_balance` default did. `bench/check_sol_kernel.py`'s schema case grades
# every key here against what the node declares.
SOL_CUDA_DEFAULTS = dict(
    tau=1.0, quantizer="rotated", dense_blocks=SOL_DENSE_TAIL,
    sink_conditioning=SOL_SINK_DEFAULT, token_routing="off",
    start_percent=0.2, end_percent=1.0, min_tokens=12288, verbose=True,
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
#   different question. **This said no sage kernel had been graded against
#   either and that there was no number to quote. Both were false when written:**
#   `bench/results/2026-08-18_sage_accuracy_on_capture.json` grades sage against
#   the 2026-08-17 capture named here, with a float64 reference, produced by
#   `bench/grade_sage_on_capture.py`. Read the number there rather than this
#   comment.
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
# token_refiner runs over the text span only, which is a few hundred rows
# against a hundred thousand -- this said "~2k rows against ~42k" and both
# halves were wrong; `docs/SOLATTN.md` has the measured breakdown off a shipped
# graph, and this file's own numbers elsewhere say a few hundred. So
# patching it is worth well under 1% of attention time.
# head_chunks 1 = off. It trades ~4x the attention launches for headroom that
# converts to wall-clock at the ~2.6% ceiling measured above, so it is for
# fitting a render that otherwise will not fit. Keep the key ordered as the
# node declares its inputs: the owner's editor-saved graphs map widget values
# positionally.
SAGE_NODE = dict(mode="auto", patch_token_refiner=False, head_chunks=1)

# The dense attention kernel under Sol, on ComfyUI core's `ModelAttentionBackend`
# node (`comfy_extras/nodes_model_advanced.py`). **The default chain since
# 2026-09-15, owner decision, replacing `MiniMaxH3SageAttention`**, which now
# appears only on graphs that declare why (bench/check_attention_defaults.py
# `FLOOR_STEMS`). "comfy kitchen attention" is kitchen's `int8_attention`: it
# rotates q and k before INT8, so no loud channel can own a shared scale, and
# on the block-49 capture it is the most accurate INT8 attention this repo has
# graded (bench/results/2026-09-15_ck_int8_attention_block49.json, measured).
# Wall time against the sage chain: one market render per arm, same prompt and
# seed (bench/results/2026-09-15_block49_community_chain.md, measured once).
# The node rather than `--use-ck-attention`: the flag is server-global and
# reaches every model, the node travels with the graph. It installs an
# attention override, and Sol chains onto it, so this kernel also takes every
# call Sol declines and every step outside Sol's window.
DENSE_BACKEND_NODE = dict(attention="comfy kitchen attention")

#: The dense kernel a video graph gets when its GRAPHS entry names none, and
#: what each choice means in `build_workflows._attention_plan`'s vocabulary.
#: Two chains, both Sol on top, and the owner's standing position (2026-09-17)
#: is that neither has won: they are good in different ways and both move.
#:
#:   "kitchen"  core's Model Attention Backend node (kitchen's rotated int8
#:              attention). Needs nothing else: the kernel rotates q/k itself.
#:   "sage"     the sage node in its ROTATED mode (since 2026-09-19, the owner;
#:              balanced until then), which rotates q/k by a fixed Hadamard
#:              matrix inside its own quantizer: the most accurate INT8 arm on
#:              every captured H3 block for about one percent of the call
#:              (bench/results/2026-09-17_sage_qk_rotate_kernel.json), and the
#:              same matrix kitchen's dense kernel uses, so the two chains now
#:              treat the loud channels the same way. Named explicitly rather
#:              than as `auto`, so a graph and its records say what ran. No
#:              MiniMaxH3ChannelBalance node and no balanced mode on top:
#:              under rotation the balance factor finds nothing to do, and the
#:              weights fold re-rounds q/k for nothing
#:              (bench/results/2026-09-17_channel_balance_vs_sage_balanced_b49_s15.json).
#:
#: `DEFAULT_DENSE_CHAIN` is what the shipped tree is generated on. The other
#: chain's set is one command, written OUTSIDE the shipped tree because every
#: check here reads that tree as one chain:
#:
#:   python workflows/build_workflows.py --chain sage --out <some dir>
DENSE_CHAINS = {
    "kitchen": dict(dense_attn="ck"),
    "sage": dict(dense_attn="sage_sol", sage_mode="fp8++ rotated"),
}
DEFAULT_DENSE_CHAIN = "kitchen"

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
# Two cautions, written while the base sampler was `er_sde` (it is `euler`
# since 2026-10-05, `SAMPLING` above, so neither binds the shipped default
# now): a sampler that re-noises every step inflates adjacent-step input
# deltas and can suppress reuse -- a null result on one is a sampler artifact
# until reproduced on a deterministic sampler; and any cache-on/off quality
# judgement is a numeric-perturbation A/B, which must run on a deterministic
# sampler. Keep the key order matching the node's declared inputs: the
# owner's editor-saved graphs map widget values positionally.
#
# Measured 2026-08-18 (bench/results/2026-08-18_cache_arms.jsonl, all-refs
# workload, 362f 1024x768, Sol on, 450 W stock): on er_sde at this 0.2
# threshold the cache skips NOTHING (change rates 0.20-0.30, all just above
# threshold) and costs nothing -- sampler 1489 s vs control 1491 s, output
# pixel-identical. At 0.3 on er_sde it skips ~4 steps for 1.31x. On
# res_multistep at 0.2 it skips 7 of 16 for **1.74x on the sampler**, the
# largest single-lever ratio measured on this box, stacked on Sol. The
# er_sde null was the predicted sampler artifact. Quality of the 1.74x arm
# vs its seed-matched control -- committed records, not prose:
# bench/results/2026-08-18_cache_rm_quality.json (res_multistep pair) and
# bench/results/2026-08-18_euler_quality.json (euler pair, the better
# behaved of the two) -- ranked by bench/quality_metrics.py, judged by
# nobody yet. Whether to change the shipped sampler or threshold is an
# owner decision gated on watching those pairs.
#
# VERDICT 2026-08-20, owner decision: NOT canonical. Barely tested, and a
# 16-step lever -- the 1.74x is 7 of 16 steps skipped, and the distilled
# 4-step students the owner is moving to have nothing to skip. Stays a probe.
#
# Uncontrolled edge, disclosed: MiniMaxH3ProvenanceStamp records the Sol
# keys and versions and knows nothing about this node, so a stamped render
# with cached steps carries a provenance record indistinguishable from a
# dense one. No shipped graph wires both today; a bench patching the cache
# into a stamped graph would. Enforced by nothing.
CACHE_NODE_CLASS = "EasyCache"
CACHE_NODE = dict(reuse_threshold=0.2, start_percent=0.15, end_percent=0.95,
                  verbose=True)

# Flow shifts, on `MiniMaxH3SigmaShift` (display name ModelSamplingMiniMaxH3).
# 12/3 are the base checkpoint's training shifts and the node's own defaults,
# so these values change nothing on their own. The node is in the graph so the
# shift is *visible and switchable*, because a distill runs at its trainer's
# shift, not the base model's: FastH3 at `FASTH3_SHIFT`, PDD and FlashGen at
# these. The lightx2v turbo LoRAs and their 6/3 rows were retired 2026-09-26
# (docs/wiki/decisions.md); their table is in git.
#
# **2.22 is not a trained shift here, and it will be offered to you.** DiffSynth's H3
# pipeline defaults to flow shift 2.22 and applies it to video and audio alike
# (`coderef/DiffSynth-Studio/diffsynth/diffusion/flow_match.py::set_timesteps_minimax_h3`
# -- a source read, not a build). ComfyUI uses 12/3. The disagreement is only
# the CONSTANT: DiffSynth builds `linspace(1, 0, N+1)[:-1]`, then applies the
# same shift algebra, so both agree on the rule. Nothing in this repo runs at 2.22, and a port that adopted it as
# "the MiniMax default" was reverted on 2026-08-23; it also carries a Gaussian
# center-weighted LOSS weight (`set_training_weight`) with no inference
# analogue at all. If a schedule here ever reads 2.22, it came from that path
# and not from anything we sample.
#
SIGMA_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The sampler for every distilled arm (inherited: the lightx2v turbo vendor
# shipped it on both its graphs, since retired), against `SAMPLING`'s
# res_multistep which came from core's base template. **Since 2026-10-05 the
# base runs Euler too (`SAMPLING`), so this constant and that one agree; it
# stays separate because it is the distills' contract, which a later change
# to the base default must not move.** The history below is as written. A distilled model is
# trained so one Euler step from sigma_i lands at sigma_i+1, so a multistep
# integrator corrects a discretization error that is not the dominant error
# here. That was an argument rather than a measurement, which is why it was a
# probe pair and not a change to the defaults -- until the decision below.
# **Owner decision 2026-08-27: every distilled arm now runs this, and `simple`
# with it.** The note above kept `er_sde` as the default because the Euler
# argument was an argument rather than a measurement, and made euler a probe
# pair instead. The owner's call reverses that: at 4 and 8 evaluations the
# final step covers the largest jump in the schedule, and a sampler that
# re-noises has no step left to recover from it. The two `_euler` probes that
# existed to name the difference were retired the same day, because with the
# defaults moved they no longer named one.
#
# **PDD does not merely prefer this, it requires it.** A fused head IS the
# block's mean velocity and the paper's Algorithm 1 defines an Euler step as
# its consumer; `er_sde` would consume the heads with an update rule they were
# never distilled against. Every PDD graph already carried euler before this
# change.
#
# Applied by `DISTILL_SAMPLING` below rather than by each call site
# remembering, which is how the turbo arms ended up split across two samplers
# in the first place. Named `TURBO_SAMPLER` until 2026-09-26.
DISTILL_SAMPLER = "euler"

#: Sampler and scheduler for any arm carrying a distillation LoRA. The builder
#: applies these whenever one is wired, so a new distilled arm cannot forget
#: and a `sampler_name=` at the call site is only needed to DEVIATE. Same shape
#: as `SOL_END_PERCENT_BY_STEPS`: a value the generator derives from what the
#: graph is, not one a person retypes per graph.
DISTILL_SAMPLING = dict(sampler=DISTILL_SAMPLER, scheduler="simple")

# Parallel Decoding Distillation (alibaba-pai), the acceleration LoRA that is
# not a step distillation. The trajectory stays a 32-point grid; what changes
# is that the final output head is replicated per interval and each sampling
# step decodes a block of four of them as one mean velocity, so 8 transformer
# evaluations cover a 32-step trajectory.
#
# **The block boundaries ARE the plain 8-step shifted schedule.** Not close to
# it -- bit-identical, because `linspace(1, 0, 33)[::4]` is `linspace(1, 0, 9)`
# and the shift is pointwise. So this is the only accelerator here that moves
# nothing but the step count: shift stays at the base 12/3, scheduler stays
# where it is, and `MiniMaxH3SigmaShift` does not budge.
#
# **The published files do not load.** Their keys are diffusers-side with bare
# `lora_down`/`lora_up` suffixes, which no ComfyUI weight adapter matches, so
# all 728 tensors are skipped with a log line and the render comes out as an
# undistilled 8-step pass that looks like the LoRA is bad. These names are the
# CONVERTED files from `bench/convert_pdd_lora.py`, loaded by
# `MiniMaxH3PDDLoRA` -- `LoraLoaderModelOnly` cannot carry them either, because
# two of the three mechanisms are not weight patches.
#
# Strength 1.0 is the vendor's own default and what their published comparison
# clips were rendered at (README, "LoRA weight of 1.0 at both 4 and 8 NFE").
# Unlike the plain LoRA path, 0.0 IS a valid control here: the node falls the
# output heads back to the checkpoint's own rather than merely zeroing a delta.
#: Every node class that loads a LoRA onto the model. ONE list: three copies
#: existed as of 2026-08-26 (two in check_distill_settings.py, one in
#: generate_capture_manifest.py) with nothing red when they diverged, and the
#: manifest's copy was already a class behind -- it recorded `loras: []` for
#: every graph running one of ours.
#:
#: A class-name list is the weaker of the two shapes available. `substrate.py`
#: matches on the VALUE looking like a weight filename instead, which cannot go
#: quietly incomplete when a new loader appears. This list is used where the
#: node's other inputs are needed too, which that approach does not give.
#: `bench/check_distill_settings.py` asserts every node of ours with a
#: `lora_name` input is here, since 2026-09-26, when `MiniMaxH3LoRABranch`
#: was missing and its graph read as a base graph.
LORA_LOADER_CLASSES = ("LoraLoaderModelOnly", "MiniMaxH3TurboLoRA",
                       "MiniMaxH3PDDLoRA", "MiniMaxH3LoRABranch")

PDD_FL2VA_LORA = "h3/minimax_h3_fl2va_pdd_8step_comfy.safetensors"
# The stripped sidecar, cut 2026-09-05 against `MODELS["unet_fl2va_pdd8_baked"]`
# with `bench/convert_pdd_lora.py --omit-backbone --baked <that file>
# --baked-strength 1.0`: heads, adaln bake and the refiner LoRA, no backbone.
# The two are a PAIR and the node enforces it by content (int8 codes and row
# scales of one probe module), so a graph that names one without the other
# fails at execute rather than rendering. Until 2026-09-05 this comment said
# no such constant could exist because no bake did; the bake now does.
PDD_FL2VA_STRIPPED_LORA = "h3/minimax_h3_fl2va_pdd_8step_stripped_comfy.safetensors"
PDD_REF2VA_LORA = "h3/minimax_h3_ref2va_pdd_8step_comfy.safetensors"
# The step counts one converted file serves. The published grid is 32 points,
# so any divisor is a legal arm from the same weights and each lands exactly on
# the plain shifted schedule for its own count -- the node fuses the heads at
# load for whichever is asked. 8 is what the file records; 4 is the other count
# the vendor's README reports rendering at. Both, not one, because the pair IS
# the parallel-decoding claim: one weight set, two step counts, both exact.
PDD_STEPS = 8
PDD_STEPS_FAST = 4
PDD_STRENGTH = 1.0

#: The `[8,8,4,4,4,4]` partition of the 32-point grid, as the sigma vector a
#: `ManualSigmas` node feeds the sampler. Six evaluations.
#:
#: **Why this one.** Under shift 12 the uniform 4-evaluation partition spends
#: its LAST Euler step on 80% of the trajectory, and four evaluations cannot do
#: better: `[8,8,8,8]` is the only partition of 32 into four blocks that starts
#: every block on a multiple of `L_min` and keeps every width within `L_max`.
#: This one puts the coarse blocks at the FRONT, where the trajectory is nearly
#: flat, and keeps the final step at 63.2% -- the same tail the vendor's own
#: 8-evaluation schedule has.
#:
#: **Measured, one seed, 2026-08-28.** Against the uniform 4-evaluation arm on a
#: matched pair (same seed, canvas, prompt, LoRA, Sol, sampler; only the
#: schedule differs) the owner called that one "jaggedy lines and scratchy
#: audio" and this one acceptable in both. `docs/research/pdd/audio_under_pdd.md`
#: has the arithmetic and the caveats -- and note that no partition experiment
#: can say WHY, because every coarseness statistic ranks the arms identically
#: whether computed in video time or through the audio transform.
#:
#: Written out rather than derived so the graph shows the schedule it runs.
#: `bench/grade_pdd_partitions.py::MANUAL["tail6"]` is the same vector.
PDD_MANUAL_SIGMAS = "1.0, 0.972973, 0.923077, 0.878049, 0.8, 0.631579, 0.0"

#: Evaluations `PDD_MANUAL_SIGMAS` runs. Passed as the graph's `steps` so
#: `_sol_for_steps` derives Sol's `end_percent` from the count the sampler
#: actually runs -- the PDD node's own `steps` input goes to 0, since
#: ManualSigmas replaces the schedule it would emit.
PDD_MANUAL_EVALS = 6
PDD_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# ---- Audio-only refinement after a distilled pass ----------------------------
#: The refine pass's schedule: `steps` run at the tail of a `steps / denoise`
#: long `simple` schedule on Euler, with the video frozen and the audio
#: reopened (`audio_refine.py::MiniMaxH3AudioRefineMask`), on the model from
#: BEFORE the distill LoRA. **Inherited:** these are ComfyUI-H3-AudioRefine's
#: own node defaults (`coderef/ComfyUI-H3-AudioRefine/nodes.py`,
#: `H3AudioRefineSampler`: steps 6, euler, simple, audio_denoise 0.5), which its
#: README shows after a 4-step Turbo pass. Nothing here has tuned them; the
#: first pair is the 2026-09-25 C1 session (`docs/wiki/next_steps.md`).
AUDIO_REFINE = dict(steps=6, sampler="euler", scheduler="simple", denoise=0.5,
                    video_mask=0.0, audio_mask=1.0)

#: The frozen-video cache (`frozen_video_cache.py`), at the node's API inputs,
#: on the refine pass's model (`refine_cache`), its one use. **Inherited**:
#: int4 and no refresh are ComfyUI-H3-AudioRefine's node defaults; `verify`
#: is off because it runs every cached step stock as well. The refine pass on
#: the card: `bench/results/2026-09-25_frozen_cache_s1.md` (the sampler
#: halved) and `bench/results/2026-10-06_frozen_cache_stage_split.md`. The
#: masked-window use (`masked_cache`, `halo`) is retired, 2026-10-06:
#: `bench/results/2026-10-06_frozen_cache_masked_window.md` and
#: `archive/frozen_cache_masked/`.
FROZEN_VIDEO_CACHE = dict(precision="int4", refresh=False, refresh_every=2, verify=False)
FROZEN_VIDEO_CACHE_NODE = "MiniMaxH3FrozenVideoCache"

# ---- Masked video-to-video on the song node ----------------------------------
#: The segmenter core's SAM3 nodes take from CheckpointLoaderSimple: Meta's
#: `sam3.1_multiplex.pt` repacked by `bench/convert_sam3_checkpoint.py`, which
#: changes no weight and checks that on readback. From the original rather
#: than a repackage so the conversion is ours (owner, 2026-10-04).
SEGMENTER = "sam3.1_multiplex_fp32.safetensors"
#: `MiniMaxH3SubjectTrack`'s inputs, equal to the node's defaults
#: (`subject_track.py`, which says where each comes from). The node replaced
#: core's tracker chain in the shipped graph on 2026-10-04 (owner: "Lets make
#: that the shipped workflow"): core's tracker spends a cap on every person
#: and splits one subject into several objects across cuts, which had to be
#: selected by typed index (`bench/results/2026-10-04_masked_v2v_band.md`).
#: `corrections` is written at its empty default since 2026-10-06 so that a
#: graph shows the one input a clip with many cuts needs, and a runner can
#: patch it: on the first clip with a crowd the match carried the subject
#: across one cut in six and every other shot took a typed line.
SUBJECT_TRACK = dict(subject_phrase="person", pick="largest", pick_on="automatic", match="automatic",
                     cuts="automatic", detection_threshold=0.5, max_people=16, head_phrase="head",
                     corrections="")
#: `MiniMaxH3MaskedSource`'s inputs, equal to the node's defaults
#: (`video_mask.py`). `grow_pixels` and `composite` are the owner's choice on
#: the band clip, 2026-10-04: the wider margin "preserved identity better"
#: than one DiT token of canvas, which was the first, reasoned value, and
#: "only what changed" is the composite of the render they called the best.
#: One clip and one seed each; the feather is reasoned and unjudged.
#: `replace` and `paint_out` are at the values that worked: head and hair
#: gave an oversized head on a long-haired original, and painting out was
#: judged worse. `reuse_mask` on keeps the finished mask across runs
#: (`mask_store.py`); a kept mask is the tracked one's bytes, so it changes
#: how long a run takes and not what it renders.
#: `motion_reference` and its two settings (2026-10-05) are the shipped render
#: as it was: no motion reference. `subject only` with `motion_vae` off is the
#: masking board's route 1, rendered as an arm before any default moves.
MASKED_SOURCE = dict(grow_pixels=64, feather_pixels=8, replace="whole subject", paint_out=False,
                     part_phrases="hair, head", part_threshold=0.5, part_margin=8,
                     composite="only what changed", change_threshold=0.05, reuse_mask=True,
                     motion_reference="none", motion_short_edge=384, motion_vae=False,
                     start_from="noise", start_top=0.3, start_blur=16, start_knots=1)
#: **Measured** 2026-10-05 (`bench/results/2026-10-05_masked_v2v_motion_arms.md`):
#: with the subject's own frames as an encoder-only video reference, ref2va
#: carries the original's turn at 12 and 16 steps and loses it at 8 with or
#: without its PDD bake, and fl2va never takes it at any count. 12 is the
#: lowest count seen to carry it, one seed on that ladder; the shipped fl2va
#: PDD8 graph stays the default for shots that need no movement from the
#: source.
MASKED_MOTION_STEPS = 12
#: The Masked Source as the ref2va motion graph ships it: the subject's own
#: frames on grey as the encoder-only motion reference (the masking board's
#: route 1, measured above), every other value `MASKED_SOURCE`'s.
MASKED_MOTION_SOURCE = dict(MASKED_SOURCE, motion_reference="subject only")
#: `MiniMaxH3MaskedPrompt` (`masked_prompt.py`), which writes the masked
#: graphs' prompt from these choices and from the Masked Source wired into
#: it. Equal to the node's defaults: a reference still of anybody (`subject`
#: is the user's own words for who it shows), who is the voice on the track, with what the still provides read off the Masked
#: Source's `replace`. **Reasoned**: a shipped graph holds a placeholder
#: still, so its text cannot assume a man or a woman. **Measured** once,
#: 2026-10-06 (`bench/results/2026-10-06_masked_v2v_person_text.md`): on the
#: motion graph "a person" carried the band clip's turn and started it later
#: than "man" at the same seed, so a user sets `subject` to match their
#: still. The sentences and what each has rendered on are in
#: `masked_prompt_text.py`.
MASKED_PROMPT_NODE = "MiniMaxH3MaskedPrompt"
#: The Sapiens2 route to a part of the subject (`sapiens2_parts.py`): the
#: loader's two folders under `models/sapiens2/`, and `MiniMaxH3SubjectParts`
#: at its own defaults, which tick hair and face-and-neck, the region the
#: Masked Source's `head and hair` asks SAM 3 for. No matting model: the
#: Masked Source reads the `parts` mask, not the matte, so loading one would
#: cost a pass per frame for nothing. **Measured** on one shot of one clip
#: (`bench/results/2026-10-05_sapiens2_first_frame.md`): hair found on every
#: frame, where the SAM phrase landed on a neighbour on a few.
SAPIENS2 = dict(segmentation="facebook_sapiens2-seg-1b", matting="none")
SUBJECT_PARTS = dict(hair=True, face_and_neck=True, upper_clothing=False, lower_clothing=False,
                     hands=False, mouth=False, other_classes="", crop_margin=32, subject_margin=8,
                     matte_reach=8, hold_missing=True)
#: The parts graph's Masked Source and prompt: the region is the part node's
#: mask, and since that can be any part, the prompt node is told what the
#: still provides (`masked_prompt_text.resolve_gives` refuses to guess).
MASKED_PARTS_SOURCE = dict(MASKED_SOURCE, replace="the wired parts")
MASKED_PARTS_PROMPT = dict(subject="person", voice="the main voice on the track",
                           picture_gives="the head and hair", add_to_shot="")
MASKED_PROMPT = dict(subject="person", voice="the main voice on the track",
                     picture_gives="what the Masked Source replaces", add_to_shot="")
#: The head and upper body on the ref2va motion graph: the part node's hair,
#: face and neck, upper clothing and hands as the region, the legs kept from
#: the source, the subject's own frames as the motion reference, and the
#: prompt node told the still gives the head and upper body. **Measured**,
#: by the owner's eye on playback: the one arm called solid on a window of a
#: fourth clip at both window lengths, and better at a smaller margin on a
#: fifth where the subject is small
#: (`bench/results/2026-10-06_masked_v2v_body_window_arms.md`). One seed
#: each. The same region on the fast chain was called broken there, which is
#: why no fast graph ticks these parts. `grow_pixels` stays the Masked
#: Source's default: the margin that held on a close subject.
MASKED_UPPER_PARTS = dict(SUBJECT_PARTS, upper_clothing=True, hands=True)
MASKED_UPPER_SOURCE = dict(MASKED_MOTION_SOURCE, replace="the wired parts")
MASKED_UPPER_PROMPT = dict(subject="person", voice="the main voice on the track",
                           picture_gives="the head and upper body", add_to_shot="")

# ---- FastH3 8-step V2 ------------------------------------------------------------
#: **Inherited** from ComfyUI's own template, Comfy-Org/workflow_templates
#: `templates/video_fastvideo_fasth3_t2v.json` (read 2026-09-25): 8 `simple`
#: steps on `res_multistep`, core's `MiniMaxH3SigmaShift` at 10/3 (the card:
#: "its video scheduler shift is 10, not the base model's 12"), the kitchen
#: backend, and core's `BlockSparseAttention` in VSA mode.
FASTH3_STEPS = 8
FASTH3_SAMPLER = "res_multistep"
FASTH3_SCHEDULER = "simple"
FASTH3_SHIFT = dict(shift_video=10.0, shift_audio=3.0)
#: Core's node at the template's widget values. `keep_percent` 10 is the
#: TEMPLATE's; FastVideo's own contract is sparsity 0.8, keep 20, on every step.
#: This constant stays the template arm; `FASTH3_CONTRACT_VSA` below is the
#: contract, and `bench/results/2026-09-26_fasth3_contract_s1.md` compares them.
#: Until 2026-09-26 this said the template "wins here" (`docs/wiki/decisions.md`).
FASTH3_CORE_VSA = {k: v for k, v in dict(SOL_CORE_DEFAULTS, **{
    "selection": "vsa", "selection.keep_percent": 10.0}).items() if k != "selection.tau"}

# ---- FastH3 V2 as FastVideo's own contract runs it (2026-09-26) ----------------
#: **Inherited** from the release's `fastvideo_inference.json`, as vllm-omni
#: validates it (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/fasth3_checkpoint.py`,
#: `FastH3CheckpointSpec.from_metadata`):
#: `dmd_denoising_steps` [999, 874, ..., 125], video shift 10, audio shift 3,
#: guidance 1, `vsa_sparsity` 0.8, VSA tile 64. vllm-omni steps it with Euler at
#: eta 0 and runs VSA on every step. The ComfyUI template above differs on the
#: sampler, the schedule, the kept fraction and the dense warm-up; the owner,
#: 2026-09-26: make FastH3 as good as it can be before calling it worse.
FASTH3_CONTRACT_POSITIONS = (0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0)
#: The positions shifted at the video shift 10 (`s * u / (1 + (s - 1) * u)`);
#: core derives the audio stream's sigmas from these through the model's own
#: 10/3 shifts, so `MiniMaxH3SigmaShift` must stay at `FASTH3_SHIFT`.
FASTH3_CONTRACT_SIGMAS = ", ".join(
    "0.0" if u == 0 else f"{10.0 * u / (1 + 9.0 * u):.6f}".rstrip("0").rstrip(".")
    for u in FASTH3_CONTRACT_POSITIONS)
FASTH3_CONTRACT_SAMPLER = "euler"
#: Sparsity 0.8 is the fraction of cubes DROPPED (vllm-omni `diffusion/data.py`,
#: "the nominal fraction of key blocks dropped"), so core's `keep_percent` is 20.
#: Core's tooltip says FastH3-VSA was "trained at 10", which is the 90% sparsity
#: vllm-omni quotes for the preview-v1 VSA student, not V2. `start_percent` 0:
#: the student was trained sparse on every step, coarse branch included.
FASTH3_CONTRACT_VSA = dict(FASTH3_CORE_VSA, **{"selection.keep_percent": 20.0,
                                                "start_percent": 0.0})

# ---- FlashGen 4-step LoRA ------------------------------------------------------
#: Beidouqixing's `minimax-h3-4step-lora-flashgen` (Apache-2.0): a 4-step
#: data-free distribution-matching (VSD, no GAN) LoRA for T2VA, trained at
#: 1344x768 and 5.2 s on the FL2VA partition. This is kijai's conversion
#: (HF `Kijai/MiniMax-H3-experimental`, 2026-09-22): a dynamic rank resize at
#: sv_fro 0.95 from rank 64, lossy by construction, with the adaln update
#: projected onto the pruned checkpoint's curve basis. So it loads on
#: `MODELS["unet_fl2va"]` and on nothing else. A plain weight LoRA with per-module
#: alphas. Graphs apply it through `MiniMaxH3LoRABranch`, as every LoRA on
#: int8 since 0.154.0 (`docs/h3_quant_policy.md`).
FLASHGEN_LORA = "h3/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors"
#: The publisher's LoRA at its full rank 64, converted here by
#: `bench/convert_flashgen_lora.py` (`bench/results/2026-09-25_flashgen_lora_conversion.json`):
#: exact to the release where kijai's resize keeps about 95% of each delta.
FLASHGEN_R64_LORA = "h3/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors"
#: The same at rank 64 for the pruned Ref2VA checkpoint: `convert_flashgen_lora.py
#: --partition Ref2VA` fits the adaln onto that partition's own time basis
#: (`bench/results/2026-09-26_flashgen_lora_conversion_ref2va.json`). FlashGen was
#: trained on FL2VA for T2VA; on Ref2VA it is an untested transfer.
FLASHGEN_R64_REF2VA_LORA = "h3/minimax_h3_flashgen_4step_v1.0_768p_ref2va_pruned_rank64_comfy.safetensors"
#: The node that applies a LoRA at the call instead of merging it (`lora_branch.py`).
LORA_BRANCH_NODE = "MiniMaxH3LoRABranch"
#: **Inherited:** the publisher's merge scale, `merge_lora_ckpt.py --scale`
#: default 1.0 ("lora_alpha / rank; 1.0 when alpha equals rank"). Both
#: conversions carry alpha so that 1.0 here is that scale.
FLASHGEN_STRENGTH = 1.0
FLASHGEN_STEPS = 4
#: **Inherited:** the file's own `manual_sigmas_shift12` metadata, which is the
#: publisher's `base_schedule` [1.0, 0.7, 0.4, 0.15, 0.0] mapped through shift
#: 12. Read from the header, so a re-download that changed it would show.
FLASHGEN_MANUAL_SIGMAS = "1.0, 0.965517, 0.888889, 0.679245, 0.0"

#: The distill LoRAs applied at the call whose Sol nodes take
#: SOL_DISTILL_LORA_CUDA (`sol_for_graph(..., distill_lora=True)`): every
#: FlashGen file. Owner decision 2026-10-01; see that constant.
SOL_DISTILL_LORA_FILES = frozenset({FLASHGEN_LORA, FLASHGEN_R64_LORA, FLASHGEN_R64_REF2VA_LORA})

#: Route 3 of the distill-routing idea (docs/research/2026-09-26_distill_routing.md;
#: the design is in docs/wiki/next_steps.md, agreed by two sessions and the
#: owner, 2026-09-26): FlashGen's own first two steps, then PDD8 from exactly
#: where FlashGen left the latent. Pass 1 is derived from FLASHGEN_MANUAL_SIGMAS
#: so it cannot drift from it. Pass 2's last two points are PDD8's own knots,
#: **measured**: `comfy.samplers.calculate_sigmas(simple, 8)` at shift 12 prints
#: 1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0
#: (bench/check_pdd_sigmas.py's `comfy_simple`, 2026-09-26). Starting at
#: 0.888889 puts PDD's first step on grid heads 19..23 (pdd_math.schedule_knots
#: [19, 24, 28, 32]); the tracker may warn that the step sits off a boundary.
STEP_SWITCH_PASS1_SIGMAS = ", ".join(FLASHGEN_MANUAL_SIGMAS.split(", ")[:3])
STEP_SWITCH_PASS2_SIGMAS = ", ".join([FLASHGEN_MANUAL_SIGMAS.split(", ")[2], "0.8", "0.631579", "0.0"])

#: PDD8's own schedule, **measured**: `comfy.samplers.calculate_sigmas(simple,
#: 8)` at shift 12, which `bench/check_pdd_sigmas.py` proves bit-identical to
#: what `MiniMaxH3PDDLoRA` emits at 8 steps (2026-09-26).
PDD8_SIGMAS = "1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0"

#: The reverse step switch (owner, 2026-09-26: fix PDD's weaknesses with the
#: distills' complements, "my gut says pdd"): PDD8's own early steps, which
#: follow the teacher's layout, then FlashGen finishing, so its committed final
#: jump replaces PDD's hedging late block (docs/research/2026-09-26_distill_routing.md,
#: `docs/h3_distills.md` P3). Two handoffs, both on PDD8 knots:
#:   h063: PDD steps 1..7 to 0.631579 (knot 28), then FlashGen 0.631579 -> 0,
#:         near FlashGen's trained last step 0.679245 -> 0;
#:   h080: PDD steps 1..6 to 0.8 (knot 24), then FlashGen 0.8 -> 0.679245 -> 0.
#: FlashGen's pass starts off its trained points in both (reasoned; a probe).
_P8 = PDD8_SIGMAS.split(", ")
STEP_SWITCH_REV = {
    "h063": (", ".join(_P8[:8]), "0.631579, 0.0"),
    "h080": (", ".join(_P8[:7]), ", ".join(["0.8", FLASHGEN_MANUAL_SIGMAS.split(", ")[3], "0.0"])),
}
#: PDD8 cut at a handoff, then the undistilled base finishes on Euler
#: (`docs/open_experiments.md` #37). **Reasoned:** pass 2 walks the 32-point
#: grid at shift 12, `12t / (1 + 11t)`, from the handoff down: h063 is t = 4/32
#: (PDD8's last block boundary, 4 base evaluations over the tail PDD8 takes in
#: one) and h080 is t = 8/32, exactly 0.8 (8 evaluations; the handoff of
#: `STEP_SWITCH_REV["h080"]`, so it sits beside the FlashGen finish).
def _base_tail(pass1, k):
    pts = [f"{12 * (i / 32) / (1 + 11 * (i / 32)):.6f}" for i in range(k - 1, 0, -1)]
    return ", ".join([pass1.split(", ")[-1], *pts, "0.0"])


STEP_SWITCH_BASE = {
    "h063": (", ".join(_P8[:8]), _base_tail(", ".join(_P8[:8]), 4)),
    "h080": (", ".join(_P8[:7]), _base_tail(", ".join(_P8[:7]), 8)),
}
# Each handoff is the grid point its tail starts from.
assert all(abs(float(p1.split(", ")[-1]) - 12 * t / (1 + 11 * t)) < 5e-7
           for (p1, _), t in zip(STEP_SWITCH_BASE.values(), (4 / 32, 8 / 32)))
#: PDD8 cut at 0.8, then FastH3's own checkpoint finishes on Euler (a second
#: full checkpoint, so a model swap between the passes). Handoff at base-grid
#: t = 8/32 = 0.25, which is also FastH3's rung 0.25 (FASTH3_CONTRACT_POSITIONS).
#: **The audio decides the shift.** Core derives the audio sigma from the video
#: sigma through the model's own shifts (`time_shift_sigma`): PDD8 leaves it at
#: 0.5 at video sigma 0.8 (12/3). A pass at 10/3 that keeps video sigma 0.8
#: tells the audio 0.545 instead; on the same base-grid point, video sigma
#: 0.769231, both agree at 0.5. So:
#:   s10: FastH3's own 10/3, handoff and tail on its rungs (0.769231, 0.588235).
#:        The audio is exact; the video is told 0.769231 where PDD8 left 0.8.
#:   s12: 12/3 as PDD8 ran, 0.8 and 0.631579. Both streams exact; off FastH3's
#:        trained shift.
#: **Reasoned, not rendered.**
def _fasth3_tail(shift):
    return ", ".join(f"{shift * u / (1 + (shift - 1) * u):.6f}" for u in (0.25, 0.125)) + ", 0.0"


STEP_SWITCH_FASTH3 = {
    "s10": (STEP_SWITCH_REV["h080"][0], _fasth3_tail(10.0), FASTH3_SHIFT),
    "s12": (STEP_SWITCH_REV["h080"][0], _fasth3_tail(12.0), SIGMA_SHIFT),
}
#: Every step-switch graph's (pass 1, pass 2) sigma pair, for the checks.
STEP_SWITCH_PAIRS = (((STEP_SWITCH_PASS1_SIGMAS, STEP_SWITCH_PASS2_SIGMAS),) + tuple(STEP_SWITCH_REV.values())
                     + tuple(STEP_SWITCH_BASE.values()))
#: **Reasoned:** the card names no sampler, and both its deployment targets
#: (vllm-omni, MindIE-SD) step Euler deterministically.
FLASHGEN_SAMPLER = "euler"

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
# wiring, graded by `bench/check_attention_defaults.py`. (`SageChainAssert`,
# the runtime gate that failed a render whose chain was not composed as
# intended, left every generated graph on 2026-09-17.) A constant is not a
# check.
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

# The draft decoder for scouting renders: core's tiny autoencoder for H3, which
# stock `VAELoader` offers from `models/vae_approx/` and loads as a whole VAE
# (`comfy/sd.py`, the `decoder.22.bias` branch with 24 latent channels; its
# frame count follows the same 17k+5 rule as the real decoder). Inherited from
# core (`comfy/latent_formats.py::MiniMaxH3Video.taesd_decoder_name`).
#
# **Retired as a shipped graph 2026-09-26** (owner: "not worth it"; the INT8
# VAE left a draft about 15 s cheaper per discarded seed, and taeh3 ghosts). Kept
# for `bench/compare_vae_decoders.py` and the `draft_decode` switch.
#
# **A draft clip is for choosing which seed to keep, never for judging.** It
# is an approximation of the real decoder, and a graph with `draft_decode`
# saves the sampled latent so a keeper gets the real decode from
# `h3_decode_saved_latent_api.json` without sampling again. What a draft
# costs against the real decode, and whether a keep decision made on it
# survives the real decode, is `docs/open_experiments.md` #31.
DRAFT_VAE = "taeh3.safetensors"

# 2:3 portrait, and INSIDE the trained family -- `adapt_canvas(2, 3)` returns
# exactly this. Worth stating because the community workflow this path follows
# renders 1024x1536 (1.57 MP), which is 52% over H3's 768*1344 area cap and
# outside the family; it works, and it costs 1.78x the TOKENS per frame -- so
# about 3.2x the attention, which goes as their square, a distinction this line
# got wrong until 2026-08-28 while the tier table three blocks down had it
# right -- for
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
#   Going wider is free, and buys no speed. **Not "every ratio at or above
#   1.75 costs the same" -- that was this line until 2026-08-28 and it is
#   false across the band; the four canvases named were selected, not
#   representative.** What is true is that the area cap binds, so widening
#   trades width for height at about the same cost. These four are exactly
#   equal: 1344x768, 1536x672,
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
#: **345 since 2026-08-30, owner instruction, reversing the 2026-08-16
#: restoration of 362 recorded above.** "345 the new long default yes", after
#: "thats how we render - 1334x768 or 1152x768 and 345 frames".
#:
#: The 2026-08-16 note above is kept rather than rewritten, because its
#: argument was sound and is not what changed. It reverted 345 -> 362 on the
#: grounds that 345 was the largest count *diffusers* will emit -- a
#: portability artifact presented as legality. That reasoning still holds:
#: `reference_would_emit()` is where the portability question belongs, and 345
#: is not a model boundary.
#:
#: What changed is that 345 has three justifications the 2026-08-16 decision
#: never weighed, and 362 has none of them:
#:   1. **It is what the owner renders.** That is the whole instruction and it
#:      needs no support from the other two.
#:   2. **It is exact on the audio clock.** H3's audio latent rate is 40 Hz
#:      against 24 fps, so `frames * 40 / 24` must be an integer for the two
#:      streams to land on the same grid. 345 gives 575 exactly; **362 gives
#:      603.33 and does not**. The exact set is `39 + 51k`: 39, 90, 141, 192,
#:      243, 294, 345. Raised by a peer session 2026-08-30 about a different
#:      length and verified here by arithmetic.
#:   3. **362 exceeds the vendor's own duration cap.** The reference pipeline
#:      hard-codes `max_duration = 15.0` and 362 is 15.083s, which
#:      `h3_rules.py` already records and calls a portability note.
#:
#: **What this does NOT establish.** That a non-integer audio latent count is
#: off-distribution. The count rounds and 362 renders; nobody has sourced
#: whether the release ever emits it. So this is "it works is not it was
#: trained for" as a REASON TO PREFER 345, not a demonstrated defect in 362.
#:
#: `h3_rules.MAX_LENGTH` stays 362 deliberately. That constant is the claimed
#: trained CEILING and answers "what is legal"; this one is the default and
#: answers "what do we render". Collapsing them would lose the distinction the
#: 2026-08-16 note spent a week earning.
#:
#: **Comparability, stated because the 2026-08-16 note stated it in the other
#: direction.** Every figure taken at 362 -- and that is most of the Sol work,
#: the attention ceiling, the tau arms -- now sits one grid step above the
#: default. A 5% length change should not move a ratio, and it has not been
#: re-checked in either direction.
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


# **`REF_VIDEO_LENGTH` was deleted on 2026-08-16, REINTRODUCED on 2026-08-22
# (`f9d63c59`), and is defined about a hundred lines below with its own
# justification. 28 shipped graphs render at it.** The prohibition below stood
# here unqualified until 2026-08-28 while the constant it forbids lived in the
# same file, which is the failure this file is most prone to: a decision
# recorded as a rule, reversed, and the rule left standing.
#
# The reasoning is kept because it is still the right question to ask of any
# length constant -- read it as an argument, not as a prohibition: a safe length for a reference
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
# native size instead of taking them to 2048 on the short edge; it is an
# ARM setting here, against the node default (upscale on since 2026-09-13),
# because these arms exist to carry a long reference video on a 24 GB card.
#
# Spread into every video-bearing reference arm so the three numbers have one
# home. Editing them here moves all eight arms together, which is the point.
#: The size pre-filled under `MiniMaxH3AppendRefImage.qwen_view = separate`,
#: the text encoder's own copy of a still. Read from `h3_rules`, never
#: retyped: the NODE's pre-filled value and the generator's must be one number.
#:
#: **Not the default view since 2026-09-13.** The node and every generated
#: graph default to `shared`: one prepared copy feeds both the video VAE and
#: Qwen3-VL, which is what sglang, diffusers and DiffSynth do (owner decision,
#: `docs/wiki/decisions.md`). `separate` at this size was the shipped default
#: from 2026-08-27 to 2026-09-13 on the strength of one render whose dialogue
#: bound to the wrong speaker when two 2048 references sat ahead of the
#: prompt; that observation is recorded in the CHANGELOG under 0.82.0 and was
#: never re-measured. The reference-view ablation
#: (`bench/refview2_arms.json`) is the arm that tests it.
#: --- Paths. One resolver, because counting `..` by hand has cost real time. ---
#:
#: The bug this closes, twice over. `bench/grade_pdd_partitions.py` resolved its
#: output directory as `Path(__file__).resolve().parents[2] / "output"`, which
#: from `bench/` is `custom_nodes/output` -- not an output directory on any
#: install. Seven arms rendered, then every one was thrown away undecoded.
#:
#: The reason it is easy to get wrong is that TWO conventions are in use here
#: and they differ by one:
#:
#:     Path(__file__).resolve().parents[2]   # from the FILE:      custom_nodes
#:     HERE.parents[2]                       # HERE = the DIR:     ComfyUI root
#:
#: Both appear across `bench/`. Neither is wrong; reading one while writing the
#: other is. So do not count levels -- ask for the thing by name.
REPO_ROOT = Path(__file__).resolve().parent.parent
COMFY_ROOT = REPO_ROOT.parent.parent


def output_dir() -> Path:
    """Where the server writes renders.

    `H3_OUTPUT_DIR` wins, because this box starts ComfyUI with
    `--output-directory` pointing at a share and nothing in the HTTP API
    reports it back -- so a script CANNOT derive the real location and must be
    told. The stock path is the fallback, not the assumption.

    Raises rather than returning a path that does not exist. A missing output
    directory is only ever discovered when something tries to read a render
    back, which is after the GPU time has been spent; failing here moves that
    to before it.
    """
    import os
    named = os.environ.get("H3_OUTPUT_DIR")
    out = Path(named) if named else COMFY_ROOT / "output"
    if not out.is_dir():
        raise SystemExit(
            f"output directory {out} does not exist"
            + (" (from H3_OUTPUT_DIR)" if named else
               f" (stock location under {COMFY_ROOT}; this box overrides it "
               f"with --output-directory, see start.sh)")
            + ". Set H3_OUTPUT_DIR and re-run. Checked up front so no render "
              "is queued against a path its results cannot be read from.")
    if named:
        return out
    # **An EXISTING stock directory is not evidence it is the right one, and
    # this is the escaped instance the first version of this function walked
    # straight into.** On this box `<comfy>/output` exists holding ComfyUI's
    # `_output_images_will_be_put_here` placeholder and nothing else, because
    # the server writes to a share via `--output-directory`. So the
    # existence check above passes and the caller gets a real, empty, WRONG
    # directory -- which is worse than a missing one, because it reads as
    # success. A peer session lost time to exactly this: their analysis found
    # an empty `pddref/` and reported no renders rather than a bad path.
    #
    # `check_output_dir_resolution.py`'s own docstring named this case -- "a
    # script can honour it and still fall back to a wrong stock path when it is
    # unset" -- and the guard could not fire, because it only ever raised on
    # absence. So the fallback now has to EARN its use by containing renders.
    media = {".png", ".mp4", ".flac", ".webm", ".webp"}
    for i, entry in enumerate(out.iterdir()):
        if entry.suffix.lower() in media:
            return out
        if i > 512:          # bounded: a real output dir shows one immediately
            break
    raise SystemExit(
        f"output directory {out} exists but holds no renders, so it is almost "
        f"certainly not where this server writes -- the stock path is only a "
        f"fallback, and this box starts ComfyUI with --output-directory (see "
        f"start.sh). Set H3_OUTPUT_DIR to the real location. Raising rather "
        f"than returning it, because an empty-but-present directory reads as "
        f"success and produces 'no results' instead of 'wrong path'.")


def _ref_qwen_short_edge() -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_h3_rules_const", Path(__file__).resolve().parent.parent / "h3_rules.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod.REF_QWEN_SHORT_EDGE)


#: The encoder's own view of a still, as a short edge: the append node's
#: default (`qwen_view = separate` at this size) and what the generator writes
#: on every still's append node since 2026-10-03. Read from `h3_rules.py`,
#: which owns the value and its history; the node cannot import this file.
#:
#: **Pinned to `shared` instead (the generator passes 0) on the instruments:**
#: the `_savelat` and `_x0` twins and the `bench/` graphs, whose renders are
#: compared byte for byte with earlier ones, and the `h3_probe_refview2_*`
#: scenes, whose manifest (`bench/refview2_arms.json`) defines its `parity` arm
#: as "the graph as built" at the 2026-09-13 values. The generator always
#: writes the selection, so none of these inherits the node default.
REF_QWEN_SHORT_EDGE = _ref_qwen_short_edge()

REF_VIDEO_CANVAS = dict(width=1024, height=768)
# Reference-video graphs render at the length of the reference CLIP, not at
# `LONG_LENGTH`. Since 2026-08-22 the shared clip is trimmed to 14.375s ending
# in a 0.3s silence, and 345 is the 17n+5 count that lands exactly there
# (14.375 * 24). Matching them is the point: the untrimmed 19.56s source was
# cut mid-delivery by a 15.083s render, the reference kept talking past the
# end, and the last third of every render drifted --
# `bench/results/2026-08-22_swap_prompt_verdict_362.json` has the per-third
# numbers. 362 frames against this clip would fail the other way, ending 0.7s
# after the reference runs out.
#
# **This is NOT the 2026-08-10 global move to 345 that was reverted on
# 2026-08-16.** That one capped every render at diffusers' emit limit and broke
# comparability with measurements taken at 362. `LONG_LENGTH` is untouched and
# t2v and keyframe graphs still render 362; only the graphs wired to
# this clip move, because only they have a clip to match.
REF_VIDEO_LENGTH = 345

# The VHS loader every generated reference-video graph uses: the ffmpeg one,
# not the cv2 `VHS_LoadVideo`. Owner's choice (2026-09-23), for decode
# accuracy. The two are not interchangeable, and each difference was measured
# rather than assumed, in `bench/results/2026-09-23_vhs_loader_comparison.json`
# (`bench/compare_vhs_loaders.py` re-derives it). At `force_rate` 24 on a 25 or
# 30 fps source they keep different frames: cv2's run late against the 24 fps
# grid and ffmpeg's land on the nearest frame, so renders from before the
# switch are not substrate-comparable with renders after it. cv2's decode is
# also biased dark against an accurate bt709 reference, and ffmpeg's is not.
# Output slot 1 is a MASK here, where cv2's is an INT frame count; no
# generated graph wires slot 1.
REF_VIDEO_LOADER = "VHS_LoadVideoFFmpeg"

REF_VIDEO_BUDGET = dict(length=REF_VIDEO_LENGTH, **REF_VIDEO_CANVAS,
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
#: The two stills the dialogue ref2v arm wires, in the order its prompt names
#: them: <Picture 1> is the man, <Picture 2> is the woman. Socket order IS the
#: label, so swapping these swaps who speaks which lines.
#:
#: Chosen to be far apart -- different age, sex, dress, palette and lighting --
#: so a blended or swapped identity is visible in one frame instead of needing
#: a 100% crop on a face. `1-man.png` is already the repo's reference still.
DIALOGUE_REF_IMAGES = ("1-man.png", "5-woman.png")

#: The reference-view ablation of 2026-09-13 (`bench/refview2_arms.json`):
#: five scenes, each a ref2va bank prompt and the stills it carries, in
#: append order. Owner-chosen stills from the shared input directory. Every
#: scene graph is built at the node defaults (vendor parity: 2048 short
#: edge, upscale on, one copy for both towers) and the manifest's patches
#: are the arms. The old three-arm Gate 6 family
#: (`h3_probe_refview_{a_source,b_qwen2048,c_parity}`) was priced on
#: 2026-08-25, never rendered, and replaced by this.
REFVIEW2_SCENES = (
    # (graph stem suffix, prompt id, stills in <Picture N> order)
    ("stairwell_backstage", "ref2va_stairwell_dialogue_backstage",
     ("20260315_193704.jpeg",)),
    ("stairwell_circus", "ref2va_stairwell_dialogue_circus",
     ("Digital_Circus_Jax-12.jpg", "Digital_Circus_Pomni-108.jpg")),
    ("diner", "ref2va_diner_breakup_refs",
     ("20260315_193704.jpeg", "20260511_123213_37d2ec75.jpeg",
      "20260505_112742_563ba791.jpeg")),
    ("porter", "ref2va_night_porter_refs", ("20260310_153551.png",)),
    ("dancer", "ref2va_studio_dancer_close_refs",
     ("20260504_152156_5b0e8151.jpeg",)),
)

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
# is history. Neither should be graded against the live SCHEMA.
#: **`image` was removed 2026-08-27 when the single-frame path was parked.**
#: The graphs are `archive/workflows/image/` and are no longer generated,
#: discovered, or graded. Anything deriving a single-frame class from this
#: tuple now derives an EMPTY one -- that is the correct answer, and
#: `bench/check_attention_defaults.py` was checked against it rather than
#: left to pass vacuously.
#: **`distill_experiments` added 2026-09-27** (owner: "you and fastdude's
#: modified/new workflows can go into a new subfolder called
#: distill_experiments"). The generator routes a graph there by
#: `build_workflows._is_distill_experiment`: every `_savelat` or `_x0` twin,
#: every `h3_probe_*` graph that runs a distill, and entries marked
#: `distill_experiment=True`. The shipped distill graphs stay at the root.
#: **`daily` added 2026-10-03** (owner, of the ref2va finish graph: "if we're
#: using something regularly ... why keep that in distill experiments?").
#: `DAILY_GRAPHS` says which graphs those are and which probe or shipped graph
#: each was promoted from; a fact about use, so it is declared here and nowhere else. The rest
#: of the by-purpose layout the owner chose the same day is prepared on the
#: branch `workflows-by-purpose` and has not landed.
DAILY_DIR = "daily"
#: The one folder that holds single-frame graphs (`build_workflows._graph_dir`).
#: Named so a reader of `GRAPH_DIRS` can tell that class from every other
#: folder: `bench/check_attention_defaults.py` used to treat any folder but
#: the root and `distill_experiments` as single-frame, and `daily` tripped it.
SINGLE_FRAME_DIR = "image"
DAILY_GRAPHS: dict[str, str] = {
    "h3_ref2v_pdd8_flashgen_finish": "h3_probe_r2v_step_switch_pdd8_flashgen_h080.json",
    "h3_t2v_pdd8_flashgen_finish": "h3_probe_t2v_step_switch_pdd8_flashgen_h080.json",
    "h3_i2v_pdd8_flashgen_finish": "h3_probe_i2v_step_switch_pdd8_flashgen_h080.json",
    # The masked lane (owner, 2026-10-05: "make sure we have a daily section
    # for mask"): the default graph, and the ref2va motion graph for a shot
    # that needs the original's movement (`MASKED_MOTION_STEPS`). Promoted
    # from shipped root graphs, not probes; both stay at the root as well.
    "h3_mask_pdd8": "h3_video_to_video_masked_song_pdd8.json",
    "h3_mask_ref2va_motion": "h3_video_to_video_masked_song_ref2va_motion.json",
    # 2026-10-06: the part of the subject Sapiens2 finds, and the look before
    # a render (tiles, shot table, the region and the prompt, no sampling).
    "h3_mask_parts_pdd8": "h3_video_to_video_masked_parts_song_pdd8.json",
    "h3_mask_review": "h3_video_to_video_masked_review.json",
    # 2026-10-07: the head and upper body on the ref2va motion graph, the
    # recipe the owner's playback verdicts of 2026-10-06 picked.
    "h3_mask_upper_ref2va_motion": "h3_video_to_video_masked_upper_song_ref2va_motion.json",
}
GRAPH_DIRS: tuple[str, ...] = ("", "distill_experiments", DAILY_DIR)

# What `bench/` is exempt from is schema grading, and only that. A bench graph
# naming a model file that no longer exists is not schema drift -- it is the
# same broken graph a shipped one would be, and on 2026-08-21 the three stamped
# graphs carried the deleted video VAE exactly as the shipped ones did. So the
# directory is reachable through `graph_paths(include_bench=True)` rather than
# through a second glob somewhere: `bench/check_graph_discovery.py` forbids a
# check from enumerating graphs itself, and the way to widen coverage is to
# widen this function, never to work around it.
BENCH_GRAPH_DIRS: tuple[str, ...] = ("bench",)

# ---------------------------------------------------------------------------
# Which nodes carry prompt text into the encoder
# ---------------------------------------------------------------------------

#: Every node class that carries prompt text to the encoder, mapped to the input
#: that holds it. **The one copy.** Preflight, the prompt catalogue, the camera
#: check, the guide-conformance and label checks and `workflows/prompts.py`
#: each kept their own list until 2026-10-01, and the lists disagreed: the
#: song node was in only two of them, so its prompts were never graded, catalogued
#: or tied to a bank id, and preflight said "nothing to grade" over a graph
#: that renders a prompt. (The same escape had happened once before, in
#: preflight, when `MiniMaxH3Conditioning` was missing from its list.)
#: `bench/check_prompt_guide_conformance.py` fails when a shipped graph carries
#: a `prompt` string on a class that is not here. Core's two H3 nodes are
#: listed because hand-built and HF-style graphs use them.
PROMPT_INPUTS: dict[str, str] = {
    "MiniMaxH3Conditioning": "prompt",
    "MiniMaxH3ReferenceConditioning": "prompt",
    # The prompt half of the two-node reference path (`reference_encode.py`,
    # 2026-10-03). No shipped graph wires it yet; listed so one that does is
    # graded. Its references are on `MiniMaxH3EncodeReferences`, one link up.
    "MiniMaxH3PromptOnReferences": "prompt",
    "MiniMaxH3ImageToVideo": "prompt",
    "MiniMaxH3ReferenceToVideo": "prompt",
    "MiniMaxH3AudioFreezeSong": "prompt",
}

#: The carriers whose stored text is a template, not what the encoder reads:
#: `__name__` placeholders filled from the Prompt List nodes chained into the
#: node's `lists` input, and optionally one block per timeline label, each
#: opened by a `--- label` line (`prompt_lists.py` and `loop_plan.py` own that
#: grammar). `workflows/prompts.py::carriers` expands them so a grader sees the
#: text the model reads.
PROMPT_TEMPLATE_CARRIERS: tuple[str, ...] = ("MiniMaxH3AudioFreezeSong",)

#: The carriers that take the six-section reference format. A carrier in
#: `REF_FORMAT_WHEN_WIRED` takes it only when its `references` input is wired.
REF_FORMAT_CARRIERS: tuple[str, ...] = ("MiniMaxH3ReferenceToVideo",
                                        "MiniMaxH3ReferenceConditioning")
REF_FORMAT_WHEN_WIRED: tuple[str, ...] = ("MiniMaxH3AudioFreezeSong",)

#: The carriers that decode their own audio, so a graph with one has an audio track
#: for the audio sections to describe even though it carries no `VAEDecodeAudio`.
#: A grader that read only the decoder node would call those two sections
#: optional on every song graph and pass a prompt that has neither.
AUDIO_DECODING_CARRIERS: tuple[str, ...] = ("MiniMaxH3AudioFreezeSong",)


# ---------------------------------------------------------------------------
# Reading a value out of a graph
# ---------------------------------------------------------------------------

# A node input is EITHER a literal (`"steps": 4`) OR a link to another node's
# output (`"steps": ["50", 0]`). Every static reader in `bench/` reads the
# first form and nothing reads the second, so **linking a widget that a check
# reads turns that check red on a graph that is completely fine.**
#
# That is not a hypothetical failure mode, it is a repeat of one. When
# `MiniMaxH3PDDLoRA` started emitting SIGMAS the PDD graphs stopped carrying a
# `BasicScheduler`, three checks' local step readers each returned `None`, and
# all three treat "could not read" as a failure --
# `bench/check_attention_defaults.py::_steps_of` records that it "reported
# eleven correctly-wired graphs as wrong". `graph_schedule` below exists
# because of it. The link form is the same defect one level down: the value is
# present and unambiguous, the reader cannot see where it comes from, and the
# check reports the graph rather than itself.
# `bench/check_distill_settings.py::_literal` had that written down as
# deliberate behaviour until it was pointed here.
#
# So the walk lives here, once, beside the discovery rule it rhymes with.
#
# **What this deliberately does not do.**
#
#   It never executes and never guesses. A slot whose value is computed at run
#   time is reported as computed. `MiniMaxH3Resolution.width` parses a
#   DynamicCombo label ("1344x768  7/4  1008 tok/frame  1.00x") inside
#   `execute`; a static reader that parsed the same string would be a second
#   copy of `resolution._parse` living in a file that cannot import it.
#
#   It never reads a UI graph. It walks API form only, which is the only form
#   shipped; an editor-saved graph is read by `bench/preflight_graph.py`'s own
#   reader.
#
#   It never decides whether an unresolvable value is a failure. It reports a
#   state and a reason; the caller owns the policy, because "no step count" is
#   fatal to `check_distill_settings.py` and merely uninteresting to a reader
#   that was only curious.

#: How many nodes one chain may walk through before it is called malformed.
#: A widget fed by a constant node is one hop and a fan-out through two or
#: three is plausible; this is not a policy about graph style, it is the bound
#: that keeps the walk terminating if the cycle set is ever wrong.
MAX_LINK_HOPS = 16

#: The four outcomes, and they are four rather than three on purpose.
#:
#:   RESOLVED   the value is `GraphValue.value`.
#:   COMPUTED   a node produces it at run time and no static reader will ever
#:              know it. **The graph is fine.** Skip the value, not the graph.
#:   OPAQUE     this resolver cannot see it: a class with no row in
#:              `OUTPUT_SOURCES`. **The graph is probably fine and the
#:              RESOLVER is incomplete**, so the fix is a table row, not a
#:              graph edit.
#:   MALFORMED  the link does not describe a reachable value -- absent node,
#:              slot out of range, cycle, over-deep chain, or an input name the
#:              node does not have. **The graph is broken** and a caller
#:              should go red.
#:
#: COMPUTED and OPAQUE are not merged, because their fixes are opposite and
#: because merging them means answering "no static reader can know this" about
#: a node nobody has described yet. That is the shape of the confident wrong
#: answer a deleted encoder registry once gave: a known key returning the wrong
#: contract reads as authoritative, where "no contract" sends you to look.
RESOLVED = "resolved"
COMPUTED = "computed"
OPAQUE = "opaque"
MALFORMED = "malformed"


@_dataclass(frozen=True)
class GraphValue:
    """One resolved input. `state` is one of the four constants above.

    `value` is meaningful only when `state == RESOLVED`; `ok` is that test.
    `reason` is prose for a check's failure line and is populated for every
    other state. `via` is the node keys walked, source last, so a report can
    name the chain rather than only its ends.
    """
    state: str
    value: object = None
    reason: str = ""
    via: tuple = ()

    @property
    def ok(self) -> bool:
        return self.state == RESOLVED


@_dataclass(frozen=True)
class Passthrough:
    """An output slot that hands one of the node's own inputs straight through.

    `input_name` is the key the slot's value sits under in the node's `inputs`.
    """
    input_name: str


#: `class_type -> one entry per OUTPUT SLOT, in slot order`. `None` means the
#: slot is computed at run time (state COMPUTED); a `Passthrough` means the
#: slot is one of the node's own literals (state RESOLVED, once read). A class
#: absent from this table resolves to OPAQUE, never to a guess.
#:
#: **This is a claim about `execute`, not about a schema, which is why it is a
#: table rather than a derivation.** A schema gives slot names and types; that
#: output 0 of `PrimitiveInt` IS its `value` input is a fact about the body of
#: `execute`, and nothing declares it. What a schema CAN settle is the slot
#: count and the input name, and `bench/check_graph_values.py` grades both --
#: against `bench/node_id_manifest.json` for this pack's nodes and against the
#: class's own `define_schema()` for core's. Add a row only after reading the
#: node's `execute`, and expect that check to disagree with you if you do not.
OUTPUT_SOURCES: dict = {
    # comfy_extras/nodes_primitive.py: each of the five is
    # `def execute(cls, value): return io.NodeOutput(value)` over a single
    # `value` input, so slot 0 is that input and there is no second slot.
    "PrimitiveInt": (Passthrough("value"),),
    "PrimitiveFloat": (Passthrough("value"),),
    "PrimitiveString": (Passthrough("value"),),
    "PrimitiveStringMultiline": (Passthrough("value"),),
    "PrimitiveBoolean": (Passthrough("value"),),
    # `resolution.MiniMaxH3Resolution`: all seven outputs come out of
    # `execute`, which parses the selected DynamicCombo label and then runs the
    # token arithmetic. Not one is a literal sitting on an input, so every slot
    # is COMPUTED -- and the three that shipped graphs actually wire (width,
    # height, length) are why that state has to exist. A reader that returned
    # MALFORMED here would report every reference graph in the tree.
    "MiniMaxH3Resolution": (None,) * 7,
}


def _is_link(value) -> bool:
    """Whether an API-form input value is a link rather than a literal.

    Mirrors `comfy_execution.graph_utils.is_link`, which is the rule the
    executor itself applies: list, length two, `str` id, numeric slot. It is
    mirrored rather than imported because this module must stay importable with
    no ComfyUI on `sys.path` -- `bench/check_graph_discovery.py` imports it that
    way on purpose, so a check with a broken import is still audited. Mirroring
    a rule is a second copy, so it gets what a second copy needs:
    `bench/check_graph_values.py::case_link_rule_matches_core` runs both
    predicates over one battery and goes red the day they disagree.
    """
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str)
            and isinstance(value[1], (int, float)))


def _graph_nodes(graph):
    """`{key: node}` for an API graph, keys as strings, non-node values dropped."""
    return {str(k): v for k, v in graph.items() if isinstance(v, dict)}


def _next_hop(node, source, via):
    """Follow one `Passthrough` off `node`.

    Returns `("value", literal)`, `("link", (key, slot))`, or a `GraphValue` to
    hand straight back to the caller.
    """
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or source.input_name not in inputs:
        return GraphValue(
            MALFORMED,
            reason=f"{node.get('class_type')} has no input "
                   f"{source.input_name!r} to pass through, which is what "
                   f"OUTPUT_SOURCES says it does",
            via=tuple(via))
    raw = inputs[source.input_name]
    if _is_link(raw):
        return ("link", (str(raw[0]), int(raw[1])))
    return ("value", raw)


def _walk(nodes, node_key, slot, max_hops) -> GraphValue:
    """Follow a chain of links to the literal at its head."""
    via: list = []
    seen = set()
    while True:
        if len(via) >= max_hops:
            return GraphValue(
                MALFORMED,
                reason=f"link chain is longer than {max_hops} nodes",
                via=tuple(via))
        if (node_key, slot) in seen:
            return GraphValue(
                MALFORMED,
                reason=f"link cycle: node {node_key} slot {slot} feeds itself",
                via=tuple(via))
        seen.add((node_key, slot))
        node = nodes.get(node_key)
        if not isinstance(node, dict):
            return GraphValue(
                MALFORMED,
                reason=f"link points at node {node_key}, which is not in the "
                       f"graph",
                via=tuple(via))
        via.append(node_key)
        cls = node.get("class_type")
        spec = OUTPUT_SOURCES.get(cls)
        if spec is None:
            return GraphValue(
                OPAQUE,
                reason=f"no OUTPUT_SOURCES row for {cls!r}, so its slot {slot} "
                       f"cannot be read statically",
                via=tuple(via))
        if not isinstance(slot, int) or isinstance(slot, bool) or not 0 <= slot < len(spec):
            return GraphValue(
                MALFORMED,
                reason=f"{cls} declares {len(spec)} output(s); the link wants "
                       f"slot {slot!r}",
                via=tuple(via))
        source = spec[slot]
        if source is None:
            return GraphValue(
                COMPUTED,
                reason=f"{cls} output {slot} is computed at run time",
                via=tuple(via))
        hop = _next_hop(node, source, via)
        if isinstance(hop, GraphValue):
            return hop
        kind, payload = hop
        if kind == "value":
            return GraphValue(RESOLVED, payload, via=tuple(via))
        node_key, slot = payload


def resolve_link(graph, value, *, max_hops: int = MAX_LINK_HOPS) -> GraphValue:
    """Resolve one API-form input value: a literal, or a `[node_id, slot]` link.

    A literal comes straight back as RESOLVED. A link is followed to the node
    it names, its slot looked up in `OUTPUT_SOURCES`, and the walk repeated if
    that slot is itself fed by a link -- so a widget behind two constant nodes
    resolves, and a chain that closes on itself is MALFORMED rather than a
    hang. The four states are documented above `RESOLVED`.

    `resolve_widget` is what a caller usually wants; this is the entry point
    for a value already pulled out of `node["inputs"]`.
    """
    if not _is_link(value):
        return GraphValue(RESOLVED, value)
    return _walk(_graph_nodes(graph), str(value[0]), int(value[1]), max_hops)


def resolve_widget(graph, node, name, *,
                   max_hops: int = MAX_LINK_HOPS) -> GraphValue:
    """The concrete value of `node`'s input `name`: `node["inputs"][name]`,
    literal or link.

    A node with no such input is MALFORMED, not OPAQUE: either the class
    changed under the caller or the caller asked for the wrong name, and both
    are worth a red.
    """
    inputs = node.get("inputs")
    if not isinstance(inputs, dict) or name not in inputs:
        return GraphValue(
            MALFORMED,
            reason=f"{node.get('class_type')} has no input {name!r}")
    return resolve_link(graph, inputs[name], max_hops=max_hops)


#: The node that marks a sampler's latent as an audio-only refinement pass.
AUDIO_REFINE_MASK_NODE = "MiniMaxH3AudioRefineMask"


def refine_scheduler_ids(graph) -> set:
    """Ids of the `BasicScheduler` nodes that schedule an audio-only refine pass.

    A refine pass is a `SamplerCustomAdvanced` whose `latent_image` comes from
    `AUDIO_REFINE_MASK_NODE`, and its schedule is the node on its `sigmas`
    input. Those steps are extra evaluations on the audio after the graph's own
    pass, so every reader of "the graph's step count" must leave them out.
    """
    ids = set()
    for n in graph.values():
        if not isinstance(n, dict) or n.get("class_type") != "SamplerCustomAdvanced":
            continue
        inp = n.get("inputs", {})
        lat, sig = inp.get("latent_image"), inp.get("sigmas")
        if (isinstance(lat, list) and isinstance(sig, list)
                and graph.get(str(lat[0]), {}).get("class_type") == AUDIO_REFINE_MASK_NODE
                and graph.get(str(sig[0]), {}).get("class_type") == "BasicScheduler"):
            ids.add(str(sig[0]))
    return ids


def graph_schedule(graph) -> tuple:
    """`(steps, scheduler)` for one graph, from whichever node owns the schedule.

    **There are two owners now, and which one it is is not a style choice.**
    Until 0.83.0 every graph took its schedule from `BasicScheduler`, so three
    checks each grew their own two-line reader for it. Then the PDD graphs
    stopped carrying one: `MiniMaxH3PDDLoRA` emits `SIGMAS` and the sampler
    reads it, precisely so that a scheduler and a step count are no longer
    settable independently of the grid the heads were fused for. A reader that
    knows only about `BasicScheduler` returns `None` on those graphs, and every
    one of the three treats "could not read" as a failure -- which is what
    happened, in all three at once.

    So the rule lives here rather than three times:

      * `BasicScheduler` present -> its `steps` and its `scheduler`.
      * otherwise, a `MiniMaxH3PDDLoRA` with a non-zero `steps` -> that count,
        and the scheduler is **`simple` by construction**, not by declaration.
        The node emits `1 - pdd_time_grid`, which IS the plain shifted schedule
        for the block count and is bit-identical to `BasicScheduler(simple, N)`
        at every count the shipped graphs run (`bench/check_pdd_sigmas.py`
        grades that against ComfyUI's own `calculate_sigmas`). Reporting
        `simple` here is therefore a fact about the emitted vector and not a
        convenient label -- if that equality ever breaks, that check goes red
        before anything reading this does.
      * a `ManualSigmas` node -> the step count is `len(vector) - 1`, and the
        scheduler is **`manual`**. Added 2026-08-28 with the tail-weighted PDD
        partition, the first graph whose schedule is neither a
        `BasicScheduler` curve nor -- at the time -- a count the PDD node could
        emit. Reading it off the vector is not a convenience: it is the only
        place the count exists on that graph, and three checks went red at
        once when it did not.
        **Corrected 2026-08-31.** This said the node's SIGMAS output "only
        expresses counts that DIVIDE the 32-point grid". True when written on
        2026-08-28 and false since 2026-08-29, when `resolve_emit_steps` gained
        the envelope route -- six IS emittable now, as `[8,8,4,4,4,4]`. The
        graph still wires `ManualSigmas`, which remains correct and is graded
        against `partition_bounds` by `bench/check_distill_grid.py`. This was
        the fourth place carrying that withdrawn claim; the others were the
        node's `steps` tooltip, `docs/h3_pdd.md`'s sweep table, and
        `check_distill_grid` itself, which was RED on a correct graph because
        of it.
      * neither -> `(None, None)`, and the caller decides whether that is a
        failure. It still is for every current caller.

    Reads an API graph (`{id: {"class_type", "inputs"}}`).

    **Every value goes through `resolve_widget`, so a LINKED widget reads as
    the value behind it.** The first version of this read `inputs.get("steps")`
    and accepted only an `int` or a `float`, which is the same blind spot the
    PDD rewiring exposed one level up: wire `steps` from a constant node and
    this returned `None` on a graph that is completely fine, and all three
    callers would have gone red. Nothing in the tree links a `steps` widget
    today; that is exactly when the reader is cheap to fix and free to verify,
    and `bench/check_graph_values.py` holds a synthetic graph that does.

    **The four resolver states collapse to `None` here, deliberately.** A
    caller of this function wants a number or a failure, and the states that
    are not RESOLVED all mean "no number": a computed step count is as
    unusable to `bench/check_distill_grid.py` as a broken link is. A caller
    that needs to tell those apart -- to report "this graph computes its step
    count" rather than "this graph is wrong" -- should call `resolve_widget`
    itself and read `GraphValue.reason`.
    """
    steps = scheduler = None
    pdd_steps = manual_steps = None
    # A refine pass's scheduler is the refine pass's, not the graph's (added
    # 2026-09-25 with `audio_refine.py`): its steps are extra audio-only
    # evaluations, and read as the graph's they made a PDD arm look untiled.
    skip = refine_scheduler_ids(graph)
    for nid, n in list(graph.items()):
        if not isinstance(n, dict) or nid in skip:
            continue
        kind = n.get("class_type")
        if kind == "BasicScheduler":
            got = resolve_widget(graph, n, "scheduler")
            if got.ok and isinstance(got.value, str):
                scheduler = got.value
            got = resolve_widget(graph, n, "steps")
            if got.ok and isinstance(got.value, (int, float)):
                steps = int(got.value)
        elif kind == "MiniMaxH3PDDLoRA":
            got = resolve_widget(graph, n, "steps")
            if got.ok and isinstance(got.value, (int, float)) and int(got.value) > 0:
                pdd_steps = int(got.value)
        elif kind == "ManualSigmas":
            got = resolve_widget(graph, n, "sigmas")
            if got.ok and isinstance(got.value, str):
                pts = [x for x in got.value.split(",") if x.strip()]
                if len(pts) >= 2:
                    manual_steps = len(pts) - 1
    if steps is None and manual_steps is not None:
        # `manual` rather than `simple`: the vector is an explicit non-uniform
        # partition, so calling it `simple` would assert an equality with
        # `calculate_sigmas` that is FALSE here by construction.
        return manual_steps, "manual"
    if steps is None and pdd_steps is not None:
        return pdd_steps, "simple"
    return steps, scheduler


def manual_sigmas(graph):
    """The explicit sigma vector a `ManualSigmas` node states, or None.

    The companion to the `manual` branch above, which returns only the COUNT.
    A caller grading whether that vector is on the LoRA's grid needs the values
    -- `bench/check_distill_grid.py` compares them against
    `pdd_math.partition_bounds`, which is the only honest grading for a
    schedule no scheduler produces.

    Goes through `resolve_widget` for the same reason everything else here
    does: a linked widget must read as the value behind it.
    """
    for n in list(graph.values()):
        if not isinstance(n, dict):
            continue
        if n.get("class_type") != "ManualSigmas":
            continue
        got = resolve_widget(graph, n, "sigmas")
        if got.ok and isinstance(got.value, str):
            try:
                vals = [float(x) for x in got.value.split(",") if x.strip()]
            except ValueError:
                return None
            if len(vals) >= 2:
                return vals
    return None


def graph_paths(workflows, pattern: str = "*.json", include_bench: bool = False) -> list:
    """Every shipped graph under `workflows`, in a stable order.

    `workflows` is the repo's `workflows/` directory as a `pathlib.Path`.
    Returns paths, sorted within each directory, root first.

    `include_bench=True` adds `workflows/bench/`. Pass it when the property
    being checked is true of ANY graph -- a model file that exists, a node id
    that resolves -- and leave it off when the property is about the shipped
    set, which is what schema grading is.
    """
    dirs = GRAPH_DIRS + (BENCH_GRAPH_DIRS if include_bench else ())
    out = []
    for sub in dirs:
        out += sorted((workflows / sub if sub else workflows).glob(pattern))
    return out
