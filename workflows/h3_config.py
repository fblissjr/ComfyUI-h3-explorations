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


#: **NOT SHIPPED. `MODELS["clip"]` is `ENCODER_INT8`.** A W4 artifact kept for
#: the small-host variant and for the arms that measure encoders; no shipped
#: graph loads it. This docstring opened with "the shipped text encoder ... v2
#: since 2026-08-27" until 2026-08-28, which was true for hours and then was
#: not. **The observable is `MODELS["clip"]`; do not restate ship state here.**
#: Named by the name `h3_awq_encoder.py`'s snapshot registry recognizes,
#: replacing
#: `qwen3vl_32b_minimax_h3_w4a16_awq.safetensors`. v2 declares the release's
#: own still-image budget, 65536..16777216 px, where v1's snapshot declared
#: 200704..301056. Under v1 every reference reached the conditioner reduced to
#: roughly 290 merged tokens whatever it was prepared at, which is why
#: `MiniMaxH3AppendRefImage.qwen_short_edge` could not do anything; under v2 a
#: 2048 short edge arrives intact, and so does a 512 one. The knob is
#: bidirectional -- `qwen_view_size` has no `min(1.0, ...)` -- and 2026-08-27
#: showed the DOWNWARD direction is the one that matters here: reference tokens
#: sit in the text segment ahead of the prompt, so two upscaled references left
#: a ~1,000-token prompt at 9.5% of its own segment and a two-speaker scene
#: rendered with the dialogue attributed to the wrong subject. That is a cost
#: as well as a capability --
#: an unclamped Qwen view costs `(w/32)*(h/32)` tokens, the same arithmetic as
#: the DiT reference rows, and both segments sit inside Sol-Attn's exact sink.
#:
#: **Name the artifact here; never symlink one file into another's name.** The
#: loader recognizes an artifact by its embedded `config.json`, but the static
#: readers (`bench/preflight_graph.py`) resolve by filename through
#: `h3_awq_encoder.ARTIFACT_SNAPSHOTS`. A symlink splits the two. Measured
#: 2026-08-27, pointing the v1 name at the v2 file: runtime resolved the v2
#: snapshot and its release bounds, while the static reader resolved the v1
#: bounds and preflight priced a clamp that would not happen. The v1 name is a
#: known key in that registry, so the lookup returned a confident wrong answer
#: rather than the "no contract" an unknown name would have produced.
ENCODER_V2 = "qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors"

#: **NOT SHIPPED either. `MODELS["clip"]` is `ENCODER_INT8`.** This said
#: "shipped again since 2026-08-27" and was overtaken the same day by
#: `4ff3f0b`. Its snapshot resolves to `None` in `h3_awq_encoder.ARTIFACT_SNAPSHOTS`,
#: which is that module's own config -- the 200,704..301,056 px still budget.
#:
#: What sent it back is not a weights result. On the Gate 5 holdout v2's median
#: relative L2 against BF16 is a wash or slightly worse: noup 0.3667 against
#: 0.3594, up2048 0.3326 against 0.3122, t2va 0.0611 against 0.0671
#: (`bench/results/2026-08-25_v2_holdout_layer50.json`; at up2048 the MEANS
#: disagree in sign with the medians, annotated beside the Gate 5 table in
#: `canonical/2026-08-25_v2_launch_record.md`). Everything v2 bought was its
#: snapshot -- it reads as a quant generation and was doing a preprocessing job.
#:
#: And that snapshot is what broke a render. v2 declares the release's own
#: 65,536..16,777,216 px, so 2048-short-edge references stopped clamping and
#: reached the conditioner at 9,408 tokens against a ~1,000-token prompt --
#: the prompt fell to 9.5% of its own segment where v1's bounds had left it
#: near 63%. A two-speaker scene at that default attributed the dialogue to the
#: wrong subject, and the speaker binding lives in exactly those prompt tokens.
#:
#: **The mechanism is priced, not proven.** One render, one seed, and the
#: arithmetic is consistent with it. The arm that would settle it holds the
#: WEIGHTS fixed and varies the snapshot -- v2 weights under v1 bounds via
#: `h3_awq_encoder.install_source_processors(image_bounds=...)`, which exists
#: for exactly that and is not reachable from a graph. Until it runs, going
#: back is the cheap way to stop paying for an unproven mechanism, not a
#: verdict on v2.
ENCODER_V1 = "qwen3vl_32b_minimax_h3_w4a16_awq_v1-comfy.safetensors"

# Checkpoint names are the ones their owning loader actually offers. The text
# encoder is the canonical custom W4A16 AWQ build. Core CLIPLoader also lists it,
# but that is filesystem discovery, not format support: the file uses
# compressed-tensors' full Hugging Face namespace and metadata, while native
# H3 expects Comfy's 50-layer namespace and quant metadata. This repo's
# `MiniMaxH3AWQEncoderLoader` performs that adaptation in memory and dispatches
# the weights through comfy-kitchen. Architecture and tokenizer are native
# ComfyUI; format recognition/repacking and config-driven preprocessing are
# explicitly local handling. The graph must name a concrete file; the custom
# loader itself is not filename-bound and validates any selected file by its
# metadata and complete adapted tensor inventory.
#: The ComfyUI-native INT8 ConvRot encoder, the encoder of record for the
#: Gate 6 reference-view arms and the marker arms since 2026-08-25: on the
#: 13-row holdout it sits about fifteen times closer to the BF16 release at
#: layer 50 than either W4A16 artifact (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`).
#: That comparison was taken against v1 and the v2 candidate as they stood on
#: that date; it is the reason to keep measuring this encoder, not a reason the
#: shipped graphs cannot move. **They load `MODELS["clip"]`, which is THIS
#: file** -- the sentence here used to say v2, contradicting the line below it
#: in the same comment block, and `git blame` shows one commit wrote both.
#:
#: **SHIPPED ON EVERY GRAPH SINCE 2026-08-27 (late), by owner decision**,
#: replacing the W4A16 AWQ builds. The fidelity case is the table above; what
#: changed is the judgement that a stack already carrying pruning, int8
#: quantisation and a third-party PDD LoRA should not also carry the least
#: faithful encoder on the box.
#:
#: **It loads through core's `CLIPLoader`, which stamps no
#: `_h3_encoder_contract`.** So `reference_geometry.encoder_contract_from_clip`
#: returns `None`, `effective_policy` resolves `encoder` to `comfy`, and the
#: AWQ adapter's `preprocess_embed` -- what applied a snapshot's declared still
#: bounds -- is not installed. The operative ceiling becomes core's own
#: `process_qwen2vl_images` defaults, 3,136..12,845,056 px, 43x wider than v1's
#: 200,704..301,056.
#:
#: **That would have re-run 2026-08-27 morning's failure, and
#: `REF_QWEN_SHORT_EDGE` is why it does not.** Measured on the shipped
#: reference pair at `qwen_short_edge` 512: 522 merged tokens under v1's
#: bounds, 592 under core's. The knob was inert under v1 and is load-bearing
#: here -- the case its own note argued when it was kept rather than deleted.
#: **Do not set it back to 0 on this encoder**: unclamped, those two references
#: cost 9,408 tokens inside the prompt's own segment.
#:
#: `bench/preflight_graph.py` reports "no contract" for this file rather than
#: guessing one, which is the designed behaviour for a name its registry does
#: not know.
ENCODER_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

MODELS = dict(
    unet_fl2va="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    unet_ref2va="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    # Two fl2va/ref2va hybrids, both int8_convrot, both fl2va everywhere
    # except the adaln projections named in their filenames. `b30` is the HF
    # release (blocks 30-49, final layer left on fl2va); `adaln_all` is built
    # here by `bench/build_hybrid.py` (all 50 blocks plus final_layer), which
    # first reproduces the HF file byte-for-byte as its control. They exist
    # to ask whether an fl2v distill LoRA transfers to reference work better on
    # fl2va's linears than on ref2va's; `docs/roadmap.md`, the regime section.
    # The filenames end `-int8`, not `_int8_convrot`; `substrate.py` tags them.
    unet_hybrid_b30="minimax_h3_hybrid_fl2va_ref2va_b30-49-int8.safetensors",
    unet_hybrid_adaln_all="minimax_h3_hybrid_fl2va_ref2va_adaln_all-int8.safetensors",
    clip=ENCODER_INT8,
    # The fp16 video VAE, and it is the best build available in ComfyUI's
    # format. **Measured 2026-08-21** against the official release's fp32
    # weights (`MiniMaxAI/MiniMax-H3`, `video_vae/source/model.safetensors`):
    # all 559 shared tensors match in name and shape, median relative delta
    # 2.07e-4 and max 2.48e-4, which is fp16's own rounding floor and no more
    # -- so this file is that fp32 cast down, with `latents_mean` and
    # `latents_std` folded in as tensors. Only the fp32 original is more
    # faithful, at twice the size; a bf16 conversion would be less, since
    # fp16 carries three more mantissa bits.
    #
    # The int8_convrot decoder shipped here from 2026-08-10 and was removed
    # by owner decision on 2026-08-21, file deleted from disk. It decoded
    # 1.29x faster at 124f/1344x768 and cost 53.3 dB against fp16 on
    # lossless pixels; the CHANGELOG holds that measurement. Nothing may name
    # it again -- `bench/check_model_files.py` goes red on any graph or
    # constant pointing at a file the server does not offer.
    video_vae="minimax_h3_video_vae_fp16.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)


#: Encoder files core's own `CLIPLoader` (type `minimax`) loads; everything
#: else named as a graph's encoder goes through the repo's
#: `MiniMaxH3AWQEncoderLoader`, which refuses a file that is not a W4A16
#: compressed-tensors artifact. The generator picks the loader node from
#: this set, so a graph never names a file its loader cannot open.
CORE_LOADED_ENCODERS = frozenset({
    ENCODER_INT8,
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
})

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
SOL_END_PERCENT_BY_STEPS = {4: 0.74, 6: 0.83, 8: 0.87}

SOL_RECOMMENDED_CUDA = dict(
    # "adaptive tau" since the v3 node (2026-08-22). It is the threshold
    # selection every Sol number here was measured under, so it is the
    # continuity choice, not a preference between the two: the node's other
    # option, "top-k (SLA)", is a different selection rule and no arm here has
    # been rendered under it.
    selection="adaptive tau",
    # 1.0 since 2026-08-20, owner decision; see the tau note above for the
    # reversal condition. 1.3 was the value every Sol number before that date
    # was measured at.
    tau=1.0,
    # **`start_percent` has never been measured, at any value, ever.**
    # `docs/SOLATTN.md` says so in its knob table and again in its
    # open-experiments table ("zero measurements, ever"); the node's own tooltip
    # justifies it only as "the paper uses 0.2". It has been 0.2 in every graph
    # this repo has ever shipped.
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
    start_percent=0.2, end_percent=0.9,
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
    # through to `previous` -- and on every graph here that is SAGE, not dense
    # torch (`vendor/sol_attn_minimax.py::make_override`'s `dense()`, read
    # 2026-08-27; the render log says it too, "sage registered as the
    # attention-override fallback" then "chaining onto an existing attention
    # override"). So this is Sol against sage.
    #
    # That moves the crossover UP. `docs/SOLATTN.md` puts sage about 2.7x ahead
    # of torch's flash backend on this shape, so a sparse kernel has to clear a
    # good dense one, not a naive one. `SOL_CUDA_DEFAULTS` above already
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
    morton_curve="3d", centroid_tail=True,
    reuse_qkv_memory=False, verbose=False,
    # "0-1" since 2026-08-26, and it is the vendor's number rather than ours.
    # NVLabs ships `SOL_ATTN_FIRST_DENSE_LAYERS = 2` on BOTH their H3 configs --
    # the 8xGB200 rack one and the single-card RTX 5090 one -- and every sparse
    # config in their tree carries `stage2_dense_layers = "0-1"`. That it is
    # identical on a consumer single card and on a rack is what makes it
    # portable: it is a statement about where this model is numerically
    # fragile, not about hardware. Ulysses degree, VAE sharding and compile
    # settings differ between those two configs; the Sol policy does not.
    #
    # We shipped "" until today, which was the one place our recipe was less
    # conservative than the vendor's tested one. Costs 2 of 50 blocks on the
    # sparse steps. The node logs `keeping blocks [0, 1] dense of 50` when it
    # takes, and warns loudly if it cannot index blocks, so this is verifiable
    # from the log rather than assumed.
    #
    # Their `SOL_ATTN_THRESH_TYPE = "diag"` is deliberately NOT copied: no such
    # input exists on our node or in `comfy_kitchen.sol_attn`, so there is
    # nothing to map it onto.
    dense_blocks="0-1",
)


# The Sol config for a PDD arm, which is NOT SOL_RECOMMENDED_CUDA with a
# derived `end_percent`. **Owner decision 2026-08-29, from rendering rather
# than from a controlled measurement**: on the distilled arms this settings
# group is what has looked best so far. It is a PRIOR in the sense CLAUDE.md
# means -- no matched-seed distribution behind it, no per-knob attribution --
# and it is recorded here as a decision so that a later measurement has
# something to overturn. What overturns it: a blind distribution per knob
# through `docs/eval_comparison.md` section 3.
#
# Four knobs differ from the non-PDD recipe, and each one is here on the
# owner's reading, not on a number this repo holds:
#
#   end_percent  0.74 at EVERY PDD step count, where the derivation in
#                SOL_END_PERCENT_BY_STEPS gives 0.87 at 8. Strictly more
#                conservative, never less: at shift 12 and 8 steps the band
#                floor moves from sigma 0.642 to 0.8083, which takes the
#                second-to-last step (sigma 0.8) dense as well as the last
#                one (0.6316). Sparse coverage goes 5 of 8 to 4 of 8; at 4
#                steps 0.74 is already what the table gives and nothing
#                moves. So this does not undo the dense-tail fix -- it
#                widens the dense tail past it.
#   min_tokens   11776 against 12288. Selects identically on everything this
#                repo renders (DiT calls 31k-128k tokens, token-refiner calls
#                ~311 rows), so it is a preference, not a behaviour change,
#                on the graphs that exist.
#   morton_curve 2d_frame against the pinned 3d. Inert while `morton=False`,
#                which it is here; it decides which curve a person gets on
#                turning morton on. `docs/morton.md` measured 3d ahead on
#                centroid fidelity at one canvas and records that the pin
#                rests on that one canvas.
#   dense_blocks "0-5,48-49" against the vendor's "0-1". Widens the vendor's
#                front band and adds a tail pair. Until 2026-08-29 this was
#                "0,1,2,48,49,-1", where `-1` resolved to 49 on a 50-block DiT
#                (`vendor/sol_attn_minimax.py::parse_blocks`), making it the
#                same five blocks as "0-2,48-49" -- confirmed by calling the
#                function, not by reading it. Costs 8 of 50 blocks on the
#                sparse steps rather than 2, and those eight are SAGE blocks,
#                NOT exact ones: `dense()` hands the call to `previous`, which
#                on every shipped graph is sage. Nothing here reaches exact
#                attention at a block, and the widest reading of "dense" is
#                the trap this knob sets.
#
# Held identical to SOL_RECOMMENDED_CUDA, so a change there still reaches
# here: selection, tau, start_percent, sink_conditioning, morton,
# centroid_tail, reuse_qkv_memory, verbose. Spelled as an override dict over
# that one rather than a second full literal, because a full copy is exactly
# the second copy this file forbids.
#
# **Two of these have a calculable answer and the literals are placeholders.**
# `docs/SOLATTN.md`, "What would replace the eyeballing, knob by knob":
#   end_percent   `percent_to_sigma(0.75)` is 0.8 EXACTLY at shift 12, and 0.8
#                 is index 24 of PDD's 32-point grid -- the start of the final
#                 block at 4 evaluations. So 0.74 is, one widget step out, the
#                 rule "run dense over the coarsest schedule's last block": a
#                 fixed point on the SIGMA PATH rather than on the step grid,
#                 which is why one constant serves both step counts where
#                 SOL_END_PERCENT_BY_STEPS needs a row each. Worth writing as
#                 an expression only after the 0.74-against-0.87 arm at 8
#                 evaluations has run; until then it is a prettier spelling of
#                 the same literal.
#   dense_blocks  measured evidence exists at THIS tau and points elsewhere.
#                 A block named here does NOT run dense attention: `dense()`
#                 hands the call to `previous`, which on every shipped graph is
#                 SAGE. So what the knob buys is Sol's error MINUS sage's, and
#                 `bench/results/2026-08-29_dense_block_ranking.json` does that
#                 subtraction: block 40 removes the most (0.209) and ships
#                 nowhere, block 0 the least (0.093) of any block in the model,
#                 block 49 a sound mid-table 0.160. So `0,1,2` is the weak half
#                 of this list and `48,49` the sound half. 1, 2 and 48 appear
#                 in no capture. What the ranking cannot see is propagation --
#                 error at an early block travels through 49 more -- and that
#                 is the gap between "worst approximated" and "worth keeping
#                 dense".
SOL_PDD_OVERRIDES = dict(
    end_percent=0.74,
    min_tokens=11776,
    morton_curve="2d_frame",
    # "0-5,48-49" since 2026-08-29, was "0,1,2,48,49,-1" (the same five blocks
    # -- `-1` resolved to 49). **Widened at the FRONT on a measurement, and
    # that measurement reversed the ranking this file carried hours earlier.**
    #
    # `bench/results/2026-08-29_block_propagation.json`: letting Sol run at
    # exactly one block, everything else sage, and reading the output latent.
    # Video rel L2 against a sage-everywhere baseline --
    #     block 0   0.0306      block 2   0.0254
    #     block 1   0.0272      block 49  0.0128
    # -- the REVERSE of the local error ranking in
    # `2026-08-29_dense_block_ranking.json`, where 0 is Sol's most accurate
    # block. Propagation is why: block 0's error is carried through 49 more
    # blocks, block 49's lands on the head. **So the vendor's front-loaded
    # `0-1` was right and this repo's local ranking was measuring the wrong
    # thing.**
    #
    # **Blocks 3, 4 and 5 are EXTRAPOLATED, not measured**, and that is the
    # weakest number in this dict. The measured front decays about 9% per
    # block (0.0306 -> 0.0254 over three), which if continued puts block 5
    # near 0.019 -- still ~1.5x block 49. It cannot continue at that rate or
    # it would reach zero long before 49, so the curve flattens somewhere
    # unmeasured and the band edge is a guess inside that flattening. The
    # sweep that settles it was interrupted at 4 of 12 arms to free the
    # server; resume with `--blocks 8,16,24,32,40,45,48`.
    #
    # The cutoff is budget, not data. On a 4-step PDD arm Sol runs 2 of 4
    # steps, so a dense block costs about 1.5 s of a ~200 s render (0.75%);
    # 8 blocks against 5 is roughly +2 s.
    #
    # `48-49` is retained and is the WEAKEST part of this set. The probe cannot
    # see the case for it: its baseline runs sage at 49 too, so it measures
    # Sol-against-sage there and not sage-against-exact, and sage is
    # pathological at 49 (`cos_min` NEGATIVE, -0.04 to -0.11). It also ran on
    # the BASE model, so PDD's fused output head -- the reason to protect the
    # last blocks at all -- was not in the path.
    dense_blocks="0-5,48-49",
)

SOL_PDD_CUDA = dict(SOL_RECOMMENDED_CUDA, **SOL_PDD_OVERRIDES)


def sol_for_graph(pdd, steps):
    """The Sol config one graph should carry, from what the graph IS.

    The single resolver for both halves of the question, because they were
    two copies before: the generator derived `end_percent` from the step
    count and `bench/check_attention_defaults.py` re-derived the same lookup
    to grade against. A PDD branch written twice would drift the same way.

    `pdd` -- the graph loads a Parallel Decoding Distillation LoRA -- takes
    SOL_PDD_CUDA whole, at every step count, so `steps` is ignored on that
    branch. Everything else takes SOL_RECOMMENDED_CUDA with `end_percent`
    lowered per SOL_END_PERCENT_BY_STEPS, which is untouched at counts the
    table does not name (16 and 20 already put their last step below the
    band).
    """
    if pdd:
        return dict(SOL_PDD_CUDA)
    end = SOL_END_PERCENT_BY_STEPS.get(steps)
    sol = dict(SOL_RECOMMENDED_CUDA)
    if end is not None:
        sol["end_percent"] = end
    return sol

# `selection` is the v3 node's DynamicCombo: it picks HOW exact key blocks are
# chosen, and the chosen option brings its own input. "adaptive tau" carries
# `tau` and is what every graph here ships; "top-k (SLA)" carries
# `keep_percent` and is the selection the lightx2v SLA LoRAs were distilled
# against. `SOL_SELECTION_INPUTS` in build_workflows.py owns which key belongs
# to which option, because both graph forms have to agree about it and they
# encode it differently.
#
# `routed_cap_percent` went with the v2 node on 2026-08-22; the v3 node does
# not declare it, and `bench/check_sol_kernel.py`'s schema case fails on a
# pinned knob the node has never heard of.
SOL_CUDA_DEFAULTS = dict(
    selection="adaptive tau", tau=1.3,
    start_percent=0.2, end_percent=0.9, min_tokens=12288,
    sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", centroid_tail=True,
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
# shift is *visible and switchable*, because the turbo LoRAs were distilled at
# their own shifts and inherit the sampler's, not the base model's:
#
#   FL2VA Turbo 4-step v0.1     544p mixed aspect   12 / 3   4 steps
#   FL2VA Turbo 8-step v1.0     544p                12 / 3   8 or 4 steps
#   FL2VA Turbo 4-step v1.0     768p (1344x768)      6 / 3   4 steps  <- v1.1 inherits this row
#   Ref2VA Turbo 4-step v0.1    544p mixed aspect   12 / 3   4 steps
#   FL2VA Turbo 4-step v0.1 SLA 768p (1344x768)      6 / 3   4 steps
#
# The 768p ones are the trap, and they are the ones that match CANVAS below.
# Their video shift is 6, half the default, so loading one into a graph that
# leaves this at 12 samples it off a schedule it was never distilled for. A
# graph with no shift node at all gives you no place to notice.
#
# Steps move with the LoRA too: SAMPLING["steps"] = 16 is a base-model number
# and the whole point of these LoRAs is 4 or 8. Changing shift without
# changing steps, or the reverse, is not a partial improvement.
# Source: coderef/Minimax-H3-Turbo README, model specs table, for the first
# four. The SLA row is from a different vendor repo (lightx2v's
# Minimax-h3-Turbo-SLA card) and its LightX2V inference config; see
# TURBO_SLA_LORA below for how `bench/check_distill_settings.py` grades it.
#
# **2.22 is not one of these, and it will be offered to you.** DiffSynth's H3
# pipeline defaults to flow shift 2.22 and applies it to video and audio alike
# (`coderef/DiffSynth-Studio/diffsynth/diffusion/flow_match.py::set_timesteps_minimax_h3`
# -- a source read, not a build). ComfyUI and every vendor row above use 12/3,
# or 6/3 for the 768p students. The disagreement is only the CONSTANT: DiffSynth
# builds `linspace(1, 0, N+1)[:-1]`, which is the Turbo README's own
# `q_i = (N - i) / N`, then applies the same shift algebra, so all three agree
# on the rule. Nothing in this repo runs at 2.22, and a port that adopted it as
# "the MiniMax default" was reverted on 2026-08-23; it also carries a Gaussian
# center-weighted LOSS weight (`set_training_weight`) with no inference
# analogue at all. If a schedule here ever reads 2.22, it came from that path
# and not from anything we sample.
#
# **v1.1 is the 768p arm.** It is what every 768p graph loads, what every note
# names, and the only 768p file this repo treats as current. v1.0 is
# historical: it is the row the vendor documented and the file earlier runs
# were measured on, and it is not a fallback -- nothing here should offer it,
# reach for it, or describe the shipped graph as loading it.
#
# What v1.1 inherits rather than owns is its schedule. The vendor README
# carries no v1.1 row (checked 2026-08-23), so the 6/3 shift and 4 steps below
# come from v1.0's row on the strength of the filename family. That is
# declared, not assumed: `bench/check_distill_settings.py::UNATTESTED` names
# the row, the vendor case FAILS if a LEGAL row is neither found in a source
# nor declared there, and it fails again if a vendor source later carries the
# row and the declaration is left standing.
#
# The version a note SHOWS is derived from the filename below via
# `turbo_label()`, never typed. They were independent strings until
# 2026-08-23 and drifted the moment this constant moved: sixteen graphs loaded
# v1.1 under help text still saying v1.0.
# `check_distill_settings.py::notes_match_the_lora` is that control.
SIGMA_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The turbo graph. This is the 8-step v1.0; the others are listed in the
# note the graph carries.
#
# Path: the lightx2v releases are foldered by HF repo under `h3/` since
# 2026-08-20 (`lightx2v_Minimax-h3-Turbo/` and `lightx2v_Minimax-h3-Turbo-SLA/`),
# because the SLA release shipped as a separate repo with a filename the
# first one could have collided with. ComfyUI's loader walks the folder
# recursively and follows symlinked directories (`folder_paths.recursive_search`
# passes `followlinks=True`), so the sub-folder and the symlink both resolve.
#
# Its shift is 12/3, the same as base, so the shift node does not move for
# this LoRA. Only the steps do: 8 instead of 16. That is worth stating plainly
# because "turbo LoRA" and "change the shift" got learned together, and for
# two of the three checkpoints the shift is already right.
#
# Strength 1.0 is the reference's own default (`--lora-scale` defaults to 1.0
# in Minimax-H3-Turbo's inference script). Not swept here.
TURBO_LORA = "h3/lightx2v_Minimax-h3-Turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
TURBO_LORA_STRENGTH = 1.0
TURBO_STEPS = 8
TURBO_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The other released turbo LoRA, and one of the two whose shift is not 12/3
# (the SLA one below is the other). Constants rather than values typed into a
# graph because the filename, the shift and the step count have to move
# together -- `bench/check_distill_settings.py` grades this triple against the
# vendor's own README and fails if any of the three drifts. Distilled at
# 1344x768, which is `CANVAS`, so unlike the 8-step it is already at home on
# the default canvas.
# v1.1 since 2026-08-23; see the note above SIGMA_SHIFT for why, and for what
# is inherited rather than attested about its shift.
TURBO_768P_LORA = "h3/lightx2v_Minimax-h3-Turbo/minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors"
# **Training provenance, kept separate from the render recipe below.** This is
# the NFE the v1.0 row documents and the count v1.1 inherits: it is what the
# student was distilled to do, and it is what `check_distill_settings.LEGAL`
# holds. Nothing renders at it today. Keeping it named means the vendor row
# stays gradeable against the vendor while the recipe moves independently -- a
# single `TURBO_768P_STEPS` would have made changing the recipe look like
# rewriting the vendor's row.
TURBO_768P_DISTILLED_STEPS = 4

# **The render recipe: owner-selected, 2026-08-23, and provisional.** Six steps
# at strength 0.75 on the owner's own trials -- "it seems to be working best".
# Not a vendor number and not measured here against a distribution, so it is
# declared as a recipe rather than folded into the row above.
#
# **Six does not divide the 1,000-step training grid** (1000 % 6 == 4), so
# these graphs are NOT exact vendor-grid arms and must not be graded as one.
# `bench/check_distill_grid.py` routes them down a separate owner-recipe path
# that still asserts `simple` is the nearest scheduler at this count, rather
# than loosening the tolerance that makes the vendor-grid claim mean anything.
TURBO_768P_STEPS = 6

# Dedicated, NOT `TURBO_LORA_STRENGTH`. That constant is the vendor's own
# default of 1.0 and is what the 8-step and every other turbo arm loads;
# pointing the 768p arm at it and then changing it would have moved every arm
# at once. `TURBO_OWNER_STRENGTH` is also not this: that belongs to the
# `h3_probe_turbo_768p_owner` graph, which additionally moves the sampler and
# the scheduler and exists to be judged against the vendor recipe.
TURBO_768P_STRENGTH = 0.75

TURBO_768P_SHIFT = dict(shift_video=6.0, shift_audio=3.0)


def turbo_label(lora_path: str) -> str:
    """`minimax_h3_fl2v_turbo_4step_v1.1_768p_...` -> `4-step v1.1 768p`.

    Every note that names a LoRA version derives it from here, so the string a
    graph SHOWS and the file it LOADS come from one place. They were typed
    independently until 2026-08-23 and drifted the moment `TURBO_768P_LORA`
    moved to v1.1: sixteen graphs loaded v1.1 while their own help text said
    v1.0, and nothing looked. `check_distill_settings.py::notes_match_the_lora`
    is the control.

    Returns "" for a filename this cannot parse, so a caller writing a note
    gets an obviously empty label rather than a confident wrong one.
    """
    import re as _re
    name = str(lora_path).rsplit("/", 1)[-1]
    m = _re.search(r"(\d+)step_(v\d+\.\d+)(?:_(\d+p))?", name)
    if not m:
        return ""
    steps, version, res = m.groups()
    return f"{steps}-step {version}" + (f" {res}" if res else "")

# The SLA release (lightx2v/Minimax-h3-Turbo-SLA, 2026-08-20). Same tensor
# keys, rank, alpha and base as the 768p arm -- header read 2026-08-20
# against v1.0, which was the 768p file at the time: 624 tensors, attn+mlp of all 50 blocks plus the refiner, rank
# 128, alpha 128, `base_model: minimax_h3_fl2va_bf16` -- so it loads through
# the stock loader exactly as that one does. What differs is how it was
# trained, and that is the whole point of wiring it.
#
# SLA is "sparse-linear attention": the student was distilled with its
# attention running a top-k block router -- the card says an 85% sparsity
# ratio, and LightX2V's config for it sets `dynamic_sparse_attn` with
# `sparsity_ratio 0.85` and `operator sage2`, whose router keeps the top 15%
# of 64/128-token key blocks per query block by mean-pooled q.(k - mean k)
# score (`coderef/LightX2V/.../attn/utils/sla_util.py::get_block_map`; a
# source read, not a build, and the training code is not in that checkout).
# So this LoRA was fitted to produce attention that survives a fixed-budget
# block cut. Three regimes follow, and nothing on this box measures any of
# them:
#   - under SLA's own router: what it was trained for; no kernel here runs it
#   - under Sol-Attn (how every shipped video graph runs it): a different
#     router -- threshold on pooled scores with a dense fallback, not a fixed
#     top-k -- so the LoRA's sparsity is not Sol's sparsity
#   - under dense attention: every block it learned to do without is back
# The probe graph `h3_probe_turbo_768p_sla.json` is the 768p graph with only
# this file swapped, Sol on per the repo default, so the first render
# answers "does it work at all under Sol" and nothing finer.
#
# Shift 6/3 and 4 steps are the 768p v1.0's row, and that is not a guess:
# LightX2V's SLA config carries `video_flow_shift 6.0 / audio_flow_shift 3.0`,
# and its `infer_steps 5` is their N+1 convention (`h3_step_update:
# training_euler`), the same 5 their 768p v1.0 configs carry for a LoRA the
# README lists at 4. `bench/check_distill_settings.py` grades this triple
# against that config rather than the Turbo README, which has no SLA row.
TURBO_SLA_LORA = "h3/lightx2v_Minimax-h3-Turbo-SLA/minimax_h3_fl2v_turbo_4step_v0.1_768p_sla_comfyui_bf16.safetensors"
TURBO_SLA_STEPS = 4
TURBO_SLA_SHIFT = dict(shift_video=6.0, shift_audio=3.0)

# The owner's working recipe for the 768p students as of 2026-08-20, from
# their own t2v trials: euler, `beta`, 4 steps, strength 0.75. Shipped as
# `h3_probe_turbo_768p_owner.json` so it is a graph with a sha rather than a
# memory of widget values, and so bench arms can patch the LoRA file onto
# it. Three things differ from the vendor row the 768p graph ships
# (er_sde -> euler, simple -> beta, 1.0 -> 0.75), and two of them carry a
# known cost that the note on the graph states:
#
#   - strength below 1.0 at 4 steps under-distills a schedule that only
#     works distilled (docs/h3_ref2v_distillation.md's "below ~0.5 you pay 8
#     steps for a model that needs 16" applies harder at 4);
#   - `beta` halves Sol's sparse steps at 4 steps and shift 6. Arithmetic
#     from the shift-6 sigma grid and the 0.2/0.9 window (sigma 0.96 down to
#     0.40): `simple` puts 3 of 4 steps inside the window, `beta` puts 2 of
#     4, because beta's second sigma is 0.966, just above the ceiling. At 6
#     steps it is 4/6 against 3/6. The 16-step figure in the SAMPLING note
#     above (11/5 against 9/7) does not transfer.
#
# Whether the recipe is better is the owner's preference over a blind
# distribution, not a measurement here; the vendor-recipe arm in the same
# session is what it is judged against. Not the scheduler of SAMPLING and
# not a new default.
TURBO_OWNER_STRENGTH = 0.75
TURBO_OWNER_SCHEDULER = "beta"

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
# in the first place.
TURBO_SAMPLER = "euler"

#: Sampler and scheduler for any arm carrying a distillation LoRA. The builder
#: applies these whenever one is wired, so a new distilled arm cannot forget
#: and a `sampler_name=` at the call site is only needed to DEVIATE. Same shape
#: as `SOL_END_PERCENT_BY_STEPS`: a value the generator derives from what the
#: graph is, not one a person retypes per graph.
DISTILL_SAMPLING = dict(sampler=TURBO_SAMPLER, scheduler="simple")

# A third-party turbo LoRA that is NOT interchangeable with the two above, and
# cannot be loaded by `LoraLoaderModelOnly` at all on our checkpoint.
#
# Measured from the safetensors headers 2026-08-13 (see
# docs/h3_ref2v_distillation.md): the official fl2v LoRAs touch 208 modules,
# all `qkv_proj` / `out_proj` / `fc1` / `fc2`. This one touches 259 -- the same
# 208 plus **51 `adaln_proj.linear`**, at a separate rank 16 against 64 for
# everything else. Those 51 are the 50 per-block `adaln_proj` and
# `final_layer.adaln_proj`. (An earlier version of this comment said that is
# "exactly where fl2va and ref2va diverge most", citing a relative delta of
# 1.92 on `final_layer.adaln_proj`. Withdrawn 2026-08-20: that figure compared
# the curve-form coefficient matrices directly, and the two checkpoints'
# `adaln_t_table` bases carry opposite signs on half their columns, so the
# coefficients are not comparable. At the modulation output the two
# checkpoints differ by a few percent there, the same order as the linears.
# `bench/analyze_checkpoint_delta.py` and `docs/evidence.md` carry it.)
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
# where it is, and `MiniMaxH3SigmaShift` does not budge. Contrast the 768p
# turbo, which needs 6/3 and therefore changes two things at once.
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
LORA_LOADER_CLASSES = ("LoraLoaderModelOnly", "MiniMaxH3TurboLoRA",
                       "MiniMaxH3PDDLoRA")

PDD_FL2VA_LORA = "h3/minimax_h3_fl2va_pdd_8step_comfy.safetensors"
# The ref2v turbo, on disk since 2026-08-18 and NAMED here for the first time
# on 2026-08-26. It exists to be the thing PDD is measured against.
#
# **This is the comparison the PDD release is actually making.** alibaba-pai's
# README is a three-column table -- base, lightx2v Turbo, PDD Acc-8Step -- run
# on test cases taken from the Turbo repo's own examples. Speed over base is
# the premise both distills already share; the claim being made is quality
# against the other distill. Everything in this repo compared PDD against PDD
# until this row existed.
#
# Its row in check_distill_settings: 544p mixed aspect, shift 12/3, 4 steps.
# The shift and the step count MATCH the PDD 4-evaluation arm exactly, which is
# what makes a matched pair possible at all -- and the vendor's own demo does
# not match them, showing 8-step PDD against 4-step turbo.
#
# **Stated confound:** it was distilled at 544p mixed aspect and the paired arm
# renders 1344x768, so the turbo is outside its training canvas there. Run at
# 768p anyway because that is where the vendor compared, and because PDD's own
# training canvas is not stated in its metadata -- so moving to 544p would swap
# a known confound for an unknown one.
TURBO_REF2VA_LORA = "h3/lightx2v_Minimax-h3-Turbo/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
TURBO_REF2VA_STEPS = 4
TURBO_REF2VA_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

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

# Where a two-stage split cuts the shared schedule, in steps. 2 of 8 is the
# shipped starting point, not a finding. H3's schedule is far more
# front-loaded than the model the split pattern came from: at video shift 12
# and 8 steps, seven of the eight evaluation points sit at sigma >= 0.8 and
# the final interval alone covers the bottom 63% of the range. Krea 2's
# k=2-3 sweet spot was still sigma 0.84 there; here k=3 is 0.9524. So the
# useful boundary is lower here and the sweep starts at 1.
SPLIT_AT = 2

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
# native size instead of taking them to 2048 on the short edge.
#
# Spread into every video-bearing reference arm so the three numbers have one
# home. Editing them here moves all eight arms together, which is the point.
#: The encoder-only view every shipped reference graph gives Qwen3-VL, in
#: `MiniMaxH3AppendRefImage.qwen_short_edge`. 512 since 2026-08-27; 0 (one
#: view, whatever the VAE got) before.
#:
#: **SUPERSEDED 2026-08-28: this paragraph describes an encoder the graphs no
#: longer load.** It was true between `72e97c3` (v1 on every graph) and
#: `4ff3f0b` (ship INT8 ConvRot), both 2026-08-27 -- a window of hours. Today
#: `MODELS["clip"]` is `ENCODER_INT8`, which loads through core `CLIPLoader`
#: with no contract stamped, so core's own 3,136..12,845,056 px bounds apply
#: and this knob is **LOAD-BEARING, not inert** -- the `ENCODER_INT8` note near
#: the top of this file is the current statement and explains why 0 would cost
#: two references 9,408 tokens inside the prompt's own segment. Keep the v1
#: paragraph below: it is why the knob was kept when it looked useless, and it
#: is the regime anyone loading a v1 snapshot is still in.
#:
#: **The v1 regime, for when it applies.** Those graphs ran the v1 conditioner,
#: whose still bounds are a 1.5x window
#: (200704..301056), so `smart_resize` lands every non-square reference on the
#: identical view whatever it was prepared at: 512, 1024 and 2048 all reach
#: the encoder as 264 merged tokens at 16:9, and only square moves at all.
#: Measured across four aspects in
#: `bench/results/2026-08-27_qwen_view_under_snapshot.json`. Not "close to
#: inert" -- exactly inert for every aspect these graphs use.
#:
#: It stays rather than coming out because it re-arms. Under v2's bounds the
#: same knob spans 448 to 7,296 merged tokens over that range, so it is a live
#: lever again the moment anyone moves `MODELS["clip"]`. Deleting it would
#: leave nothing here to warn them -- the same reason
#: `bench/check_attention_defaults.py` derives its single-frame exempt class
#: instead of deleting it.
#:
#: **Why it exists, and it is a PRIOR rather than a measurement.** Reference
#: tokens land in the TEXT segment ahead of the prompt, so they compete with it
#: rather than merely costing sequence length. Under v2 on 2026-08-27 two
#: 2048-short-edge references cost 9,408 tokens there against a ~1,000-token
#: prompt, leaving the prompt 9.5% of its own segment; 512 put it back to 63%
#: while the DiT kept all 9,408 of its reference rows. Priced with
#: `bench/preflight_graph.py`, which could not see this until 26a0dbe -- it
#: costed the segment from the DiT rows. 63% is v1's ratio, and v1's ratio was
#: an accident of that snapshot's bounds rather than a number anyone chose.
#:
#: The observation behind it is ONE render: a two-speaker scene at the old
#: default attributed the dialogue to the wrong subject, and the binding lives
#: in the prompt tokens whose share had collapsed. One seed, and the arm that
#: would have moved this number never ran -- the queue was stopped when the
#: graphs went back to v1. That arm would not have settled it anyway: v2-at-0
#: differs from v1 in BOTH weights and bounds, so it could not separate "the
#: proportion was the problem" from "v2's weights are worse and shrinking the
#: view happens to help". The pair that isolates it holds the weights fixed --
#: v2 under v1's bounds against v2 under its own, via
#: `h3_awq_encoder.install_source_processors(image_bounds=...)` -- needs no
#: render, and belongs to the encoder lane.
#: Read from `h3_rules`, never retyped. The NODE's default and this must be
#: the same number -- a graph that omits the input takes the node's, and a
#: generator that disagreed would produce graphs whose behaviour changed
#: depending on whether the key happened to be written. `h3_rules` imports no
#: ComfyUI, so this works from a bare `sys.path` with only `workflows/` on it,
#: which is how every bench script loads this file.
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
# t2v, keyframe and turbo graphs still render 362; only the graphs wired to
# this clip move, because only they have a clip to match.
REF_VIDEO_LENGTH = 345

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
GRAPH_DIRS: tuple[str, ...] = ("",)

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
#   It never invents a UI widget position. UI graphs store widget values
#   positionally and this repo has been bitten by positional reads --
#   `bench/check_pdd_sigmas.py::case_ui_and_api_agree` exists for exactly that.
#   The caller passes the literal it read; this only decides whether a link
#   overrides it. **A linked widget leaves a STALE literal behind in
#   `widgets_values`**, which is why "read the widget and stop" is wrong rather
#   than merely incomplete -- it returns a plausible number instead of nothing.
#   Every shipped UI reference graph carries one, because the conditioner's
#   width/height/length arrive from `MiniMaxH3Resolution` over links while the
#   widgets still hold whatever was last typed there;
#   `bench/check_graph_values.py::case_ui_link_beats_stale_widget` grades that
#   population rather than naming a node here.
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
#:              `OUTPUT_SOURCES`, or a UI widget whose position the caller did
#:              not supply. **The graph is probably fine and the RESOLVER is
#:              incomplete**, so the fix is a table row, not a graph edit.
#:   MALFORMED  the link does not describe a reachable value -- absent node,
#:              slot out of range, cycle, over-deep chain, or an input name the
#:              node does not have. **The graph is broken** and a caller
#:              should go red.
#:
#: COMPUTED and OPAQUE are not merged, because their fixes are opposite and
#: because merging them means answering "no static reader can know this" about
#: a node nobody has described yet. That is the shape of the confident wrong
#: answer `ENCODER_V2`'s note describes: a known key returning the wrong
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

    `input_name` is what an API graph calls it. `ui_widget` is where a UI graph
    keeps it in `widgets_values`, or `None` when that position is not knowable
    from the schema alone -- in which case a UI-form chain through this slot
    reports OPAQUE instead of reading a neighbouring widget.
    """
    input_name: str
    ui_widget: object = None


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
    "PrimitiveInt": (Passthrough("value", 0),),
    "PrimitiveFloat": (Passthrough("value", 0),),
    "PrimitiveString": (Passthrough("value", 0),),
    "PrimitiveStringMultiline": (Passthrough("value", 0),),
    "PrimitiveBoolean": (Passthrough("value", 0),),
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
    """`({key: node}, is_ui)` for either graph form, keys as strings.

    API keys are already strings; UI ids are ints, and a link row names them as
    ints too. Both are normalised to `str` so one walk serves both forms and a
    lookup can never miss by type.
    """
    if isinstance(graph.get("nodes"), list):
        return ({str(n.get("id")): n for n in graph["nodes"]
                 if isinstance(n, dict) and n.get("id") is not None}, True)
    return ({str(k): v for k, v in graph.items() if isinstance(v, dict)}, False)


def _ui_link_table(graph):
    """`{link_id: (source_node_key, source_slot)}` for a UI graph.

    LiteGraph rows are `[id, src, src_slot, dst, dst_slot, type]`, and that is
    the only form in this tree. The object form (`origin_id`/`origin_slot`) is
    read too: reported, not verified -- no graph here carries one and none has
    been produced to test against, so this is a defensive branch, taken because
    silently skipping an unrecognised row would report a perfectly good graph
    as broken, which is the failure this whole section exists to stop.
    """
    table = {}
    for row in graph.get("links") or ():
        if isinstance(row, list) and len(row) >= 3:
            table[row[0]] = (str(row[1]), row[2])
        elif isinstance(row, dict) and row.get("id") is not None:
            table[row["id"]] = (str(row.get("origin_id")), row.get("origin_slot"))
    return table


def _ui_input_entry(node, name):
    """The UI `inputs` entry for input `name`, or None.

    A converted widget carries `{"widget": {"name": ...}}` beside its own
    `name`; a plain socket carries only `name`. Both are matched, because an
    input that used to be a widget and is now a socket is still that input.
    """
    for entry in node.get("inputs") or ():
        if not isinstance(entry, dict):
            continue
        widget = entry.get("widget")
        if isinstance(widget, dict) and widget.get("name") == name:
            return entry
        if entry.get("name") == name:
            return entry
    return None


#: Distinguishes "the caller passed no literal" from "the caller passed None",
#: which a UI widget can legitimately hold.
_UNSET = object()


def _ui_hop(node, name, links, via):
    """`(source_key, slot)` if UI input `name` is linked, else None or a
    `GraphValue` describing why the link cannot be followed."""
    entry = _ui_input_entry(node, name)
    if entry is None or entry.get("link") is None:
        return None
    hop = links.get(entry["link"])
    if hop is None:
        return GraphValue(
            MALFORMED,
            reason=f"node {node.get('id')} input {name!r} records link "
                   f"{entry['link']}, which is not in the link table",
            via=tuple(via))
    return hop


def _next_hop(is_ui, links, node, source, via):
    """Follow one `Passthrough` off `node`.

    Returns `("value", literal)`, `("link", (key, slot))`, or a `GraphValue` to
    hand straight back to the caller.
    """
    if not is_ui:
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

    hop = _ui_hop(node, source.input_name, links, via)
    if isinstance(hop, GraphValue):
        return hop
    if hop is not None:
        return ("link", hop)
    if source.ui_widget is None:
        return GraphValue(
            OPAQUE,
            reason=f"{node.get('type')}.{source.input_name} is an unlinked UI "
                   f"widget and OUTPUT_SOURCES does not record its position",
            via=tuple(via))
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list) or len(widgets) <= source.ui_widget:
        held = len(widgets) if isinstance(widgets, list) else 0
        return GraphValue(
            MALFORMED,
            reason=f"{node.get('type')} holds {held} widget value(s); "
                   f"{source.input_name!r} should be at index "
                   f"{source.ui_widget}",
            via=tuple(via))
    return ("value", widgets[source.ui_widget])


def _walk(nodes, is_ui, links, node_key, slot, max_hops) -> GraphValue:
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
        cls = node.get("class_type") or node.get("type")
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
        hop = _next_hop(is_ui, links, node, source, via)
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
    for a value already pulled out of `node["inputs"]`. A UI graph stores no
    API links, so a link-shaped value in one is read as the literal it is.
    """
    if not _is_link(value):
        return GraphValue(RESOLVED, value)
    nodes, is_ui = _graph_nodes(graph)
    if is_ui:
        return GraphValue(RESOLVED, value)
    return _walk(nodes, False, {}, str(value[0]), int(value[1]), max_hops)


def resolve_widget(graph, node, name, literal=_UNSET, *,
                   max_hops: int = MAX_LINK_HOPS) -> GraphValue:
    """The concrete value of `node`'s input `name`, in either graph form.

    API form: reads `node["inputs"][name]`, literal or link. `literal` is
    ignored -- the graph names its inputs, so there is nothing to fall back to.

    UI form: a linked widget appears in `node["inputs"]` as an entry whose
    `widget.name` is `name`, and **the value it left behind in
    `widgets_values` is still there and is stale**. So the link is checked
    first, and `literal` -- whatever the caller read positionally -- is used
    only when there is no link. Omit `literal` and an unlinked UI widget is
    OPAQUE rather than silently absent.

    An API node with no such input is MALFORMED, not OPAQUE: either the class
    changed under the caller or the caller asked for the wrong name, and both
    are worth a red.
    """
    nodes, is_ui = _graph_nodes(graph)
    if not is_ui:
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or name not in inputs:
            return GraphValue(
                MALFORMED,
                reason=f"{node.get('class_type')} has no input {name!r}")
        return resolve_link(graph, inputs[name], max_hops=max_hops)

    links = _ui_link_table(graph)
    hop = _ui_hop(node, name, links, [])
    if isinstance(hop, GraphValue):
        return hop
    if hop is not None:
        return _walk(nodes, True, links, hop[0], hop[1], max_hops)
    if literal is _UNSET:
        return GraphValue(
            OPAQUE,
            reason=f"{node.get('type')}.{name} is an unlinked UI widget and no "
                   f"literal was supplied; widget positions are the caller's")
    return GraphValue(RESOLVED, literal)


def _widget(widgets, index):
    """`widgets_values[index]`, or `_UNSET` when there is no such widget.

    The positional half of a UI-form read, kept beside its caller rather than
    inside `resolve_widget`: the index is a fact about the node's schema, and
    `bench/check_pdd_sigmas.py::case_ui_and_api_agree` is what grades it by
    comparing this read against the API form's named one.
    """
    if isinstance(widgets, list) and 0 <= index < len(widgets):
        return widgets[index]
    return _UNSET


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
        partition, which is the first graph whose schedule is neither a
        `BasicScheduler` curve nor a count the PDD node can emit: the node's
        SIGMAS output only expresses counts that DIVIDE the 32-point grid, and
        `[8,8,4,4,4,4]` is six evaluations. Reading it off the vector is not a
        convenience -- it is the only place the count exists on that graph, and
        three checks went red at once when it did not.
      * neither -> `(None, None)`, and the caller decides whether that is a
        failure. It still is for every current caller.

    Accepts both graph forms: UI (`{"nodes": [...]}` with `widgets_values`) and
    API (`{id: {"class_type", "inputs"}}`).

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
    nodes = (graph.get("nodes") if isinstance(graph.get("nodes"), list)
             else list(graph.values()))
    steps = scheduler = None
    pdd_steps = manual_steps = None
    for n in nodes:
        if not isinstance(n, dict):
            continue
        kind = n.get("type") or n.get("class_type")
        widgets = n.get("widgets_values")
        if kind == "BasicScheduler":
            # UI widget order is [scheduler, steps, denoise].
            got = resolve_widget(graph, n, "scheduler", _widget(widgets, 0))
            if got.ok and isinstance(got.value, str):
                scheduler = got.value
            got = resolve_widget(graph, n, "steps", _widget(widgets, 1))
            if got.ok and isinstance(got.value, (int, float)):
                steps = int(got.value)
        elif kind == "MiniMaxH3PDDLoRA":
            # Widget order is [name, strength, patch_heads, nfe, steps]; the
            # input is APPENDED, so index 4 is the only place it can be.
            got = resolve_widget(graph, n, "steps", _widget(widgets, 4))
            if got.ok and isinstance(got.value, (int, float)) and int(got.value) > 0:
                pdd_steps = int(got.value)
        elif kind == "ManualSigmas":
            got = resolve_widget(graph, n, "sigmas", _widget(widgets, 0))
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
