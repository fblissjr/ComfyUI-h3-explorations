"""Single source of truth for the H3 node chain and its settings.

`workflows/build_workflows.py` (the graphs) and `bench/bench_e2e_h3.py` (the numbers) both import from here, so a
bench arm named "sol" is the configuration a graph renders. They once carried separate copies of the SolAttn
settings, and the copies drifted. Nothing here may have a second copy anywhere in the repo.

Each constant's comment says what it is for, its provenance (measured, inherited or reasoned, with the record it
rests on), when it took its current value and why, and which graphs or uses take which setting. What currently
ships is read from the graphs and `MODELS`, not from this prose (`docs/method/config_drift.md`).
"""

# These are the only imports, and they stay that way: `bench/check_graph_discovery.py` imports this file with only
# `workflows/` on `sys.path`, so a check whose own imports are broken is still audited.
from dataclasses import dataclass as _dataclass
from pathlib import Path


def _h3_rules():
    """`h3_rules.py`, loaded by its path once: the node pack's rules, which this file reads and never retypes. This file
    cannot import the node package (`bench/check_graph_discovery.py` imports it with nothing else on `sys.path`)."""
    import importlib.util
    import sys as _sys
    got = _sys.modules.get("_h3_rules_const")
    if got is None:
        spec = importlib.util.spec_from_file_location("_h3_rules_const", Path(__file__).resolve().parent.parent / "h3_rules.py")
        got = importlib.util.module_from_spec(spec)
        _sys.modules["_h3_rules_const"] = got
        spec.loader.exec_module(got)
    return got


# ---- Model files -----------------------------------------------------------------------------------------------
#: The text encoder `MODELS["clip"]` names: Comfy-Org's ComfyUI-native INT8 ConvRot build of Qwen3-VL-32B.
#: **Measured, then decided** (owner, 2026-08-27): on the 13-row holdout it sits an order of magnitude closer to the
#: BF16 release at layer 50 than either W4A16 AWQ artifact it replaced
#: (`data/results/2026-08-25_four_encoders_holdout_layer50.json`; all four arms keep the vision tower at BF16, so the
#: record says nothing about tower precision). Against the bf16 file pruned to the 50 layers H3 reads
#: (`bench/convert_h3_bf16_encoder.py`), text tokens stay close and image tokens carry a heavy tail that the int8
#: decoder adds (`data/results/2026-09-27_encoder_int8_vs_bf16_conditioning.json`); whether the DiT responds to it is
#: `bench/measure_encoder_quant_dit.py`'s question.
#: The owner works on the bf16 pruned encoder, so this default is not their working configuration, and the AWQ
#: comparison judged two poorly made artifacts, not the idea of quantizing our own encoder
#: (`docs/state/decisions.md`, 2026-09-20 and 2026-09-27).
#: Loads through `MiniMaxH3EncoderLoader` (core's `CLIPLoader` plus checks); the still-image bounds are core's own,
#: read by `h3_encoder_loader.native_encoder_contract`, never typed here. The AWQ files and their loader were deleted
#: 2026-09-13 with their lane (`docs/state/roadmap.md`, "Closed lanes").
ENCODER_INT8 = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"

MODELS = dict(
    unet_fl2va="minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    unet_ref2va="minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    # The fl2va PDD8 bake (2026-09-05, `bench/bake_pdd_checkpoint.py`): the 200 int8 backbone linears replaced by
    # quantize(W_release + PDD delta at strength 1.0), round to nearest, every other key byte for byte. **Measured**:
    # the strength-zero control reproduces the shipped codes up to rounding ties
    # (`data/results/2026-09-05_bake_identity_control.json`). Loads only with `PDD_FL2VA_STRIPPED_LORA`; the node
    # refuses the full sidecar on it (double apply) and the stripped one on the base. `_s1` is the baked strength;
    # `docs/distills/pdd_artifacts.md` is the inventory.
    unet_fl2va_pdd8_baked=("minimax_h3_fl2va_pruned_int8_convrot"
                           "_pdd8_baked_s1.safetensors"),
    # FastVideo's FastH3 8-step V2 (HF FastVideo/FastVideo-FastH3-Comfy): a distilled T2VA DiT (data-free DMD2,
    # trained with VSA-H3), pruned int8 convrot with its own curve basis and time table, not fl2va's, and its token
    # refiner quantized where ours is bf16 (**measured** 2026-09-26, `bench/results/2026-09-26_fasth3_weights.md`). Each
    # main block carries a `to_gate_compress` that core's model detection reads from the keys. T2VA only: FL2VA and
    # Ref2VA were not distilled. Sampling contract: `FASTH3_*` below.
    unet_fasth3_v2="fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors",
    # The fl2va/ref2va hybrid: fl2va everywhere except the adaln projections (all 50 blocks plus final_layer), built by
    # `bench/build_hybrid.py`, to ask whether an fl2v distill LoRA transfers to reference work better on fl2va's
    # linears than on ref2va's (`docs/state/roadmap.md`, the regime section). Its name ends `-int8`, not
    # `_int8_convrot`; `substrate.py` tags it. The second hybrid, `unet_hybrid_b30`, was retired 2026-10-05
    # (`docs/state/decisions.md`).
    unet_hybrid_adaln_all="minimax_h3_hybrid_fl2va_ref2va_adaln_all-int8.safetensors",
    # `unet_vsa` (kijai's experimental FastVideo VSA checkpoint) left with `MiniMaxH3VSAAttention` in 0.173.0 (owner,
    # 2026-09-29); `docs/research/vsa/2026-08-30_fastvideo_vsa_checkpoint.md` keeps what it was.
    clip=ENCODER_INT8,
    # The INT8 ConvRot video VAE. **Measured, then decided** (owner, 2026-09-26, after not telling it from fp16 on a
    # 345-frame clip pair): only its decoder is quantized, its encoder encodes bit-identically to the fp16 file's
    # (`data/results/2026-09-26_vae_encoder_int8_file.json`), and its decode is faster, holds less VRAM and adds no
    # flicker (`bench/results/2026-09-26_vae_decoders_345f.md`). It replaced `VIDEO_VAE_FP16`, which had replaced it on
    # 2026-08-21; ComfyUI's own templates had moved to it. `MiniMaxH3VAEPrecision` refuses to cast its decoder.
    video_vae="minimax_h3_video_vae_int8_convrot.safetensors",
    audio_vae="minimax_h3_audio_vae_fp32.safetensors",
)

#: The fp16 video VAE: the shipped build until 2026-09-26, and the reference the INT8 build is measured against
#: (`bench/compare_vae_decoders.py`). **Measured** 2026-08-21 against the release's fp32 weights
#: (`MiniMaxAI/MiniMax-H3`, `video_vae/source/model.safetensors`): all 559 shared tensors match in name and shape,
#: median relative delta 2.07e-4 and max 2.48e-4, which is fp16's own rounding floor. Only the fp32 original is more
#: faithful, at twice the size; a bf16 cast would be less, since fp16 carries three more mantissa bits.
VIDEO_VAE_FP16 = "minimax_h3_video_vae_fp16.safetensors"


#: Encoder files core's own `CLIPLoader` (type `minimax`) loads, and so the only files `MiniMaxH3EncoderLoader` can
#: open. The generator refuses any other encoder name, so a graph never names a file its loader cannot open.
CORE_LOADED_ENCODERS = frozenset({
    ENCODER_INT8,
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
})

# ---- Base sampling -----------------------------------------------------------------------------------------------
# The base model's sampler, scheduler and step count. Distilled arms take `DISTILL_SAMPLING` instead.
#
#   euler     **Decided** (owner, 2026-10-05: no `er_sde` anywhere). Every reference implementation steps the base
#             with Euler at eta 0 (sglang, vllm-omni, diffusers' `MiniMaxH3Scheduler`), and a deterministic sampler is
#             what makes a numeric A/B possible here. It replaced `er_sde`, the owner's pick of 2026-08-15 over core's
#             template `res_multistep`; core's base template still ships that, so upstreams disagree and this is a
#             judgment call, not the adopt-upstream rule. **Reasoned, not measured:** Euler is first order where
#             `er_sde` was a third-order multistep, and no render has judged 16 Euler steps on the base; if base clips
#             soften, `steps` is the knob to revisit. A base clip from before 2026-10-05 is not seed-comparable with
#             one after it; the step cost is the same, so timings carry.
#   simple    **Measured.** Sol's window is a sigma band resolved off the curve with no knowledge of the scheduler,
#             so the scheduler decides how many steps land inside it: at 16 steps, shift 12 and the 0.2-0.9 window of
#             the time, `simple` gave Sol 11 sparse / 5 dense and `beta` 9 / 7 (`beta` was tried and reverted
#             2026-08-15). `simple` is also the only scheduler that reproduces a distilled LoRA's published sigma grid
#             exactly at 4 and 8 steps (`beta` is off by 0.10, `normal` by 0.67, `sgm_uniform` by 0.007);
#             `bench/check_distill_grid.py` grades that on every graph that loads one, against the vendor's grid.
#   steps 16  **Measured** 2026-08-06 at 362 frames on `res_multistep`: 20 steps 765.4 s, 16 steps 669.2 s, 12 steps
#             508.5 s. 12 was rejected because it stopped following the prompt (the scripted third shot at 00:10 never
#             happens), which no still and no convergence check showed, so any further cut needs prompt adherence as
#             a gate. Not re-run on `euler`.
SAMPLING = dict(sampler="euler", scheduler="simple", steps=16, denoise=1.0)

# ---- Sol-Attn ----------------------------------------------------------------------------------------------------
# Every Sol knob is pinned, never left to the node's defaults: SolAttn once changed its `int8_qk`, `int8_pv` and
# `morton_curve` defaults underneath us, so an arm named "sol" would have meant two configurations.
#
# SOL_RECOMMENDED is the Triton-era node's (`SolAttnMiniMax`) recipe, measured 2026-08-06 to 2026-08-16 on a 4090 at
# 362 frames, where render time is the objective. It is the record of what was measured; graphs carry
# SOL_RECOMMENDED_CUDA below. **Measured**, per knob:
#   tau 1.3            Below the onset of the moving-object dissolve artifact (~1.5). Costs 82.3 s of sampler
#                      against tau 2.0 (712.1 s against 629.8 s, same seed). The algorithm's author puts peak quality
#                      at 1.0 (kijai, 2026-08-14), so 1.3 is a speed-for-quality trade; no arm compared 1.3 with 1.0.
#                      The CUDA recipe moved to 1.0 on 2026-08-20 (its note).
#   dense_blocks ""    Was 33-35,39-42, the author's highest-error blocks: it did not fix the artifact tau fixes, and
#                      cost 39.2 s. Kept as SOL_ARTIFACT_INSURANCE.
#   exact_kv_and_rows  Runs the packed conditioning query rows (~250-400 in a ~38k sequence) dense, which keeps the
#                      generated audio intact; rows that thin are what a block-sparse router drops first.
#   morton off         Free on speed on the CUDA kernel (2026-08-16: +0.8 s of 861 dense, -1.2 s of 454 sparse, both
#                      inside run-to-run spread) and never tested on quality. Morton is a closed lane
#                      (`docs/attention/morton.md`; `docs/state/roadmap.md`, "Closed lanes").
#   int8_qk/pv on      Worth 1.16x on top of plain sparsity at 362 frames.
#   window .2-.9       Widening is closed. `.1-.95` saved ~10% and passed every gate at 20 steps, but at 16 steps the
#                      scripted 00:10 cut drifted to 12-13 s and the subject's motion stalled. Each factor passed
#                      alone and the two failed together: a knob validated at one setting of another is not validated.
#
# Head chunking and FFN chunking are not in the chain: SAGE_NODE's `head_chunks` has the measurement. At these
# lengths the attention peak sets the ceiling, so chunking the FFN moves something that is not the maximum. Weight
# streaming is 0.6% of a 362-frame step and phase swapping 2.0% (the encoder and DiT are 45.9 GB together and never
# co-reside), so trading kernel launches for headroom buys at most ~2.6% (2026-08-06).
SOL_RECOMMENDED = dict(
    tau=1.3, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=True, sink_conditioning="exact_kv_and_rows", morton=False,
    morton_curve="2d_frame", int8_pv=True, verbose=False, use_tma=False,
    dense_blocks="",
)

# The protected-block list from before 2026-08-06 (the author's highest-error blocks), kept unwired for an arm that
# reproduces the dissolve artifact (`sol_block_probe.py`). Not shown to fix it: see SOL_RECOMMENDED's dense_blocks.
SOL_ARTIFACT_INSURANCE = dict(tau=1.3, dense_blocks="33-35,39-42")

# The settings the 124-frame evaluation in `docs/attention/sol_attn.md` ran on, so a bench arm can reproduce a number
# from its frontier table. **Measured** record. It differs from SOL_RECOMMENDED on purpose; do not "fix" it to match,
# since every ratio in that table was produced with these.
SOL_BASELINE_124F = dict(
    tau=1.2, start_percent=0.2, end_percent=0.9, min_tokens=4096,
    int8_qk=False, int8_pv=False, sink_conditioning="exact_kv", morton=False,
    morton_curve="3d", verbose=False, use_tma=False, dense_blocks="",
)

#: `end_percent` per sampler step count, so the last step ran dense. **Retired 2026-09-11, adopting upstream:**
#: sglang's `sol_attn` backend has no end cutoff and core's `BlockSparseAttention` defaults `end_percent` to 1.0
#: (`docs/h3/sglang_comparison.md`, the 2026-09-11 defaults section), so every step count runs Sol through the last
#: step. From 2026-08-26 it held the values in the trailing comment, **reasoned** by walking `calculate_sigmas` at
#: shift 12 (shift cancels and length does not enter): at 8 steps the last and largest step (sigma 0.632 -> 0) fell
#: inside the band, where at 16 it fell below it. The name stays because `sol_for_graph` reads it.
SOL_END_PERCENT_BY_STEPS = {}  # retired 2026-09-11; was {4: 0.74, 6: 0.83, 8: 0.87}

#: ComfyUI core's own Sol node (`comfy_extras/nodes_sparse_attention.py`, Comfy-Org/ComfyUI#16072, merged
#: 2026-09-06), for an arm that renders it against ours; the generator's `sol_impl="core"` builds one. The FastH3
#: graphs run it in VSA mode (`FASTH3_CORE_VSA`).
SOL_CORE_NODE = "BlockSparseAttention"

#: Its inputs at the node's own schema defaults, in API form. **Inherited:** read from
#: `BlockSparseAttention.define_schema` at core `1f641fd9` on 2026-09-10 and compared against the live server's
#: /object_info whenever `build_workflows.py` validates, so a core release that moves a default fails the build. Its
#: "sol-attn" method is the adaptive-tau selection; `extra_tokens` is kitchen's `token_aug`, on every eligible call.
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

#: Candidate `dense_blocks` lists from the 2026-10-02 per-block error campaign under the rotated quantizer
#: (`bench/results/2026-10-02_sol_campaign_reanalysis.md`), **measured** on local probe error. A: the three worst
#: middle blocks (>24% error) plus block 49, which feeds `final_layer.video_out`.
SOL_DENSE_OPTION_A = "39,41,42,49"

#: B: the whole 23-26% middle error plateau plus block 49.
SOL_DENSE_OPTION_B = "39,40,41,42,49"

#: C, the "full ridge shield": blocks 38-42 plus 49. In Test 9A (tau 1.3) it removed block 38's spike, taking peak
#: network error to 17.73% and ref_img error to 19.37%. The shipped default, as SOL_DENSE_TAIL.
SOL_DENSE_OPTION_C = "38,39,40,41,42,49"
SOL_DENSE_RIDGE_SHIELD = SOL_DENSE_OPTION_C

#: The default of 2026-09-25: the three blocks whose K-norm is lopsided on the release (unrotated INT8 outliers).
SOL_DENSE_HISTORICAL_TAIL = "45,48,49"

#: The Sol node's `dense_blocks` default, Option C. **Measured, then decided:** adopted 2026-10-02 on local probe error
#: (which the re-read showed was mostly blocks leaving the measured set) and kept that night by the owner, because the
#: output-level panel found the choice does not move the verdict and C is the list SOL_SINK_DEFAULT was judged with
#: (`bench/results/2026-10-02_sol_dense_blocks_panel.md`). Mirrors `sol_attn_h3.py::SOL_DENSE_TAIL`;
#: `bench/check_attention_defaults.py` holds the two together.
SOL_DENSE_TAIL = SOL_DENSE_OPTION_C

#: The Sol node's `sink_conditioning` default; mirrors
#: `sol_attn_h3.py::SOL_SINK_DEFAULT`, whose comment carries the evidence
#: (owner decision 2026-10-02, measured; `exact_kv_and_rows` before).
SOL_SINK_DEFAULT = "exact_kv_and_all_rows"

SOL_RECOMMENDED_CUDA = dict(
    # Keyed to `MiniMaxH3Sol`'s inputs since the redesign of 2026-09-27
    # (`docs/research/2026-09-27_sol_node_redesign.md`), which retired `selection`, `pooled_tail`, `morton`,
    # `morton_curve`, `token_aug_blocks` and the `qk_balance`/`rotate` booleans; their history is in git.
    #
    # tau: 1.0 since 2026-08-20, **decided** (owner): with distilled LoRAs as the working regime, 1.3 has to earn its
    # way back by being judged indistinguishable from 1.0 blind across seeds while buying real speed (a
    # `--set` bench arm at tau 1.3). The algorithm's author puts peak quality at 1.0. Every Sol number before
    # 2026-08-20 was measured at 1.3.
    tau=1.0,
    # start_percent: 0.2, the paper's value (the node's tooltip), never measured on the base graphs. **Measured** only
    # on the t2v PDD8-to-FlashGen finish: the owner could not tell 0.0 from 0.2 in five blind pairs and 0.0 cut the
    # sampler by about a fifth (`bench/results/2026-10-01_start_percent_panel.md`). PDD graphs ran 0.0 from 2026-10-01
    # until the owner reversed it on 2026-10-05 (SOL_PDD_OVERRIDES). It holds the top of the trajectory dense: a flat
    # 25% of evaluations at any step count (4 of 16, 2 of 8, 1 of 4), by arithmetic.
    start_percent=0.2,
    # end_percent: 1.0 since 2026-09-11, **inherited** (sglang has no end cutoff; core defaults to 1.0); 0.9 before,
    # which kept the last step dense at 16 steps (SOL_END_PERCENT_BY_STEPS). Steps outside the window go to the dense
    # kernel under Sol: `DENSE_BACKEND_NODE`, or sage on `bench/check_attention_defaults.py::FLOOR_STEMS`.
    end_percent=1.0,
    # min_tokens: 12288 since 2026-08-27, **inherited** (the node's own default; 4096 before). Below it Sol declines
    # and the call goes to the dense kernel under it, so this is where Sol must beat a good dense kernel, and upstream
    # puts that crossover near 12k. Inert at the lengths this repo renders: every DiT call is 31k-128k tokens and every
    # token-refiner call ~311 rows, so both values select alike (they differ only near 22 frames, S ~ 7,194). The
    # crossover against kitchen's kernel on this box is unmeasured; measuring it would beat both values.
    min_tokens=12288,
    # sink_conditioning: SOL_SINK_DEFAULT (every conditioning query row exact) since 2026-10-02, **measured, then
    # decided** (owner): on the 2026-10-02 panel it lost on no scene and was the one Sol arm rated fine on dialogue.
    # The masked lane runs `MASKED_JOB_SOL_SINK` instead.
    sink_conditioning=SOL_SINK_DEFAULT,
    verbose=True,
    # dense_blocks: SOL_DENSE_TAIL (Option C) since 2026-10-02; see that constant. Under the rotated quantizer blocks
    # 38-43 reach ~25% error while 45 and 48 fall to 9.55% and 7.14%. It was empty from 2026-09-02, when the owner
    # withdrew `0-2,32` (shipped from 2026-08-29 on an experiment: 11 of 50 blocks, one base trajectory, one distance
    # proxy, no perceptual A/B). `docs/state/decisions.md`, 2026-10-02.
    dense_blocks=SOL_DENSE_TAIL,
    # quantizer: "rotated" (a Hadamard rotation of q/k before INT8) since 2026-09-27, **measured, then decided**
    # (owner): on captures of the shipped PDD8 t2v and ref2va graphs it has the lowest Sol quantization error on all 28
    # Sol-block cells and costs less than "balanced"; total error moves little because sparsity dominates
    # (`bench/results/2026-09-27_sol_redesign_test2.md`). It changed every Sol render's output from that date.
    # "balanced" (per-head q/k channel rebalancing, on since 2026-09-15; `docs/attention/block49_quant_error.md`) came
    # before it.
    # `h3_probe_t2v_ck` and `h3_probe_t2v_exact_tail` deviate on purpose (`bench/check_attention_defaults.py`).
    quantizer="rotated",
    # token_routing: off everywhere, **measured** offline only. On the captured Base16 cells it lowered Sol's error on
    # four of five captured blocks and raised it on block 49 at every step
    # (`docs/research/2026-09-04_sol_token_aug_grade.md`; the grade holds on kitchen 0.2.33,
    # `data/results/2026-09-08_kitchen_0233_blk_cnt_rebase.json`), and no render has been judged with it on
    # (`docs/research/2026-09-05_token_aug_plan.md`, stage 5). `custom` carries a per-block list
    # (`sol_attn_h3.py::SOL_ROUTING_CHOICES`). If it is turned on, 64 is the budget: 64, 128 and 256 measured alike in
    # accuracy and kernel time (`data/results/2026-09-04_sol_exact_random_1128df6_token_aug_timing.json`).
    token_routing="off",
)

# Overrides on SOL_RECOMMENDED_CUDA for a graph that loads a PDD LoRA, spelled as overrides because a second full
# literal is the copy this file forbids. Empty since 2026-10-05, **decided** (owner: no 0.0 start for Sol anywhere),
# so PDD graphs run the base recipe. From 2026-10-01 it set `start_percent` 0.0 on the t2v finish panel
# (`bench/results/2026-10-01_start_percent_panel.md`); a ref2va PDD8 arm run with Sol from step zero then lost the
# subject's video reference (`bench/results/2026-10-05_masked_v2v_motion_arms.md`). Before that it held `end_percent`
# 0.74, retired 2026-09-11 with SOL_END_PERCENT_BY_STEPS. No protected-block list is validated for the PDD head.
SOL_PDD_OVERRIDES = dict()

SOL_PDD_CUDA = dict(SOL_RECOMMENDED_CUDA, **SOL_PDD_OVERRIDES)

# The same for a distill LoRA applied at the call (FlashGen; `SOL_DISTILL_LORA_FILES` lists the files). Empty: 0.184.1
# gave it `start_percent` 0.0, extended unmeasured from the PDD panel (whose FlashGen pass starts at sigma 0.8), and
# the owner reverted it the same day (0.184.3). Kept as the one place a FlashGen-only knob goes.
SOL_DISTILL_LORA_OVERRIDES = dict()  # was start_percent=0.0 in 0.184.1 only

SOL_DISTILL_LORA_CUDA = dict(SOL_RECOMMENDED_CUDA, **SOL_DISTILL_LORA_OVERRIDES)


def sol_for_graph(pdd, steps, distill_lora=False):
    """The Sol config one graph should carry, from what the graph is.

    The one resolver: the generator and `bench/check_attention_defaults.py` both call it, so the branch is not
    written twice.

    `pdd` (the graph loads a PDD LoRA) takes SOL_PDD_CUDA whole, at every step count. `distill_lora` (a FlashGen
    LoRA from `SOL_DISTILL_LORA_FILES`, and no PDD) takes SOL_DISTILL_LORA_CUDA; PDD wins when both are set.
    Anything else takes SOL_RECOMMENDED_CUDA, with `end_percent` from SOL_END_PERCENT_BY_STEPS when the table has
    the step count. All three are empty now, so every branch returns the recommended recipe; the branches stay as
    the place a per-kind override goes.
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

# `MiniMaxH3Sol`'s own `define_schema` defaults, for an arm that wants the node's answer rather than the recipe's.
# Since the 2026-09-27 redesign they equal SOL_RECOMMENDED_CUDA. `bench/check_sol_kernel.py`'s schema case grades
# every key here against what the node declares.
SOL_CUDA_DEFAULTS = dict(
    tau=1.0, quantizer="rotated", dense_blocks=SOL_DENSE_TAIL,
    sink_conditioning=SOL_SINK_DEFAULT, token_routing="off",
    start_percent=0.2, end_percent=1.0, min_tokens=12288, verbose=True,
)

# ---- Dense attention under Sol -----------------------------------------------------------------------------------
# `MiniMaxH3SageAttention`, on the arms that declare a sage floor (`bench/check_attention_defaults.py::FLOOR_STEMS`;
# the default chain is DENSE_BACKEND_NODE). Keep the keys in the node's input order: editor-saved graphs map widget
# values by position.
#   mode auto      Resolves to fp8_cuda++ on sm89. **Measured, then decided** 2026-08-18, reversing the 2026-08-13 flip
#                  to fp16: `bench/results/2026-08-18_attention_defaults.json` (a 2x2 of sage mode against Sol, one
#                  variable per arm) puts the whole fp16 difference in the dense steps, a minority once Sol runs. The
#                  owner's fp16 verdict (clearer, at 124 frames) was taken with Sol absent. What would reverse it: a
#                  blind pair at the shipped config, Sol on; the clips are named in that file. Sage's accuracy on a
#                  real capture is `data/results/2026-08-18_sage_accuracy_on_capture.json`
#                  (`bench/grade_sage_on_capture.py`); a ratio on `torch.randn` inputs was withdrawn 2026-08-16 and
#                  does not come back (`docs/method/evidence.md`). fp16 also holds q/k/v for the whole call (it has no
#                  `sageattn_consume`); `auto` releases them. `fp16 (most accurate)` stays the arm to bisect against.
#   patch_token_refiner False  The refiner runs on the text span alone, a few hundred rows
#                  (`docs/attention/sol_attn.md` has the breakdown), so patching it is worth well under 1% of attention.
#   head_chunks 1  Off. **Measured** 2026-08-10 (260 frames, 1344x768, 16 steps, 2 runs per arm, paired seeds, peak
#                  VRAM polled from /system_stats):
#
#                    arm          sampler   peak VRAM   vs base   sampler
#                    head1/ffn1   395.6s    16702 MiB      +0     1.000x
#                    head4/ffn1   396.5s    13475 MiB   -3227     0.998x
#                    head1/ffn2   397.1s    17376 MiB    +674     0.996x
#                    head4/ffn2   398.2s    18607 MiB   +1904     0.994x
#
#                  Head chunking frees 3227 MiB for 0.2%: take it only to fit a render that otherwise will not. The
#                  ordering reproduced in both runs; the magnitudes are under 1%. The baseline's VRAM spread 784 MiB
#                  across its runs, so ffn2's +674 is noise and head4/ffn2's +1904 is not: the two knobs are
#                  antagonistic, mechanism unknown (n=2).
SAGE_NODE = dict(mode="auto", patch_token_refiner=False, head_chunks=1)

# The dense kernel under Sol on the default chain: core's `ModelAttentionBackend` node with kitchen's
# `int8_attention`, which rotates q and k before INT8. **Measured, then decided** (owner, 2026-09-15, replacing the sage
# node): the most accurate INT8 attention graded on the block-49 capture
# (`data/results/2026-09-15_ck_int8_attention_block49.json`); wall time against the sage chain measured once
# (`bench/results/2026-09-15_block49_community_chain.md`). A node rather than `--use-ck-attention`, which is
# server-global. Sol chains onto its override, so it also takes every call Sol declines and every step outside Sol's
# window.
DENSE_BACKEND_NODE = dict(attention="comfy kitchen attention")

#: The dense kernel a video graph gets when its GRAPHS entry names none, in `build_workflows._attention_plan`'s
#: vocabulary. Both chains run Sol on top; the owner's position (2026-09-17) is that neither has won.
#:   "kitchen"  core's Model Attention Backend node (kitchen's rotated int8 attention).
#:   "sage"     the sage node in its rotated mode (since 2026-09-19; balanced before), the same Hadamard matrix
#:              kitchen uses: the most accurate INT8 arm on every captured H3 block for about 1% of the call
#:              (`data/results/2026-09-17_sage_qk_rotate_kernel.json`). Named, not `auto`, so records say what ran. No
#:              channel-balance node on top: under rotation it finds nothing to do
#:              (`data/results/2026-09-17_channel_balance_vs_sage_balanced_b49_s15.json`).
#: `DEFAULT_DENSE_CHAIN` is what the shipped tree is generated on. The other chain is one command, written outside the
#: shipped tree because every check reads that tree as one chain:
#:
#:   python workflows/build_workflows.py --chain sage --out <some dir>
DENSE_CHAINS = {
    "kitchen": dict(dense_attn="ck"),
    "sage": dict(dense_attn="sage_sol", sage_mode="fp8++ rotated"),
}
DEFAULT_DENSE_CHAIN = "kitchen"

# ---- Step cache (a probe, not the default) ------------------------------------------------------------------------
# Core's EasyCache node: under a threshold on the input's relative change between steps it skips the transformer
# forward and reuses a cached residual, so neither Sol nor the dense kernel runs on a cached step. Added 2026-08-18
# after NVLabs' 4090 runtime (Sana, sol-engine branch, PR #466) credited step caching with 3.18x of its 4.44x at 50
# steps. **Measured** 2026-08-18 (`data/results/2026-08-18_cache_arms.jsonl`, 362f 1024x768, Sol on): at 0.2 it
# skipped nothing on `er_sde` and 7 of 16 steps on `res_multistep`, 1.74x on the sampler; the quality pairs
# (`data/results/2026-08-18_cache_rm_quality.json`, `data/results/2026-08-18_euler_quality.json`) are judged by nobody.
# **Decided** 2026-08-20 (owner): not canonical, a 16-step lever, and 4-step distills have nothing to skip.
# `verbose=True`: the hit/miss log is the only record of how many steps a run reused. Key order is the node's.
# Uncontrolled edge: `MiniMaxH3ProvenanceStamp` knows nothing of this node, so a cached render's stamp reads as dense.
CACHE_NODE_CLASS = "EasyCache"
CACHE_NODE = dict(reuse_threshold=0.2, start_percent=0.15, end_percent=0.95,
                  verbose=True)

# ---- Flow shift and the distills' sampler ---------------------------------------------------------------------------
# Flow shifts on `MiniMaxH3SigmaShift`. **Inherited**: 12/3 are the base checkpoint's training shifts and the node's
# defaults; the node is there to make the shift visible and switchable, since a distill runs at its trainer's shift
# (FastH3 at `FASTH3_SHIFT`; PDD and FlashGen at these). DiffSynth's pipeline defaults to 2.22 for both streams (a
# source read); nothing here runs at it, and a port that adopted it was reverted 2026-08-23.
SIGMA_SHIFT = dict(shift_video=12.0, shift_audio=3.0)

# The sampler every distilled arm runs. **Decided** (owner, 2026-08-27; inherited from the lightx2v turbo vendor's
# graphs): at 4 and 8 evaluations the final step is the largest jump, and a re-noising sampler has no step left to
# recover. PDD requires it: a fused head is the block's mean velocity, and the paper's Algorithm 1 consumes it with an
# Euler step. Separate from `SAMPLING` (Euler too since 2026-10-05) because it is the distills' contract, which a
# change to the base must not move. Named `TURBO_SAMPLER` until 2026-09-26.
DISTILL_SAMPLER = "euler"

#: Sampler and scheduler for any arm carrying a distillation LoRA. The builder applies these whenever one is wired, so
#: a new distilled arm cannot forget them, and a `sampler_name=` at the call site is only needed to deviate: a value
#: the generator derives from what the graph is, not one a person retypes per graph.
DISTILL_SAMPLING = dict(sampler=DISTILL_SAMPLER, scheduler="simple")

# ---- LoRAs and Parallel Decoding Distillation (PDD) ---------------------------------------------------------------
#: Every node class that loads a LoRA onto the model, in one list (three copies existed until 2026-08-26, and they had
#: diverged). Used where a check needs the node's other inputs; `substrate.py` matches the weight filename instead,
#: which cannot go quietly incomplete. `bench/check_distill_settings.py` asserts every node of ours with a `lora_name`
#: input is here (since 2026-09-26, when `MiniMaxH3LoRABranch` was missing and its graph read as a base graph).
LORA_LOADER_CLASSES = ("LoraLoaderModelOnly", "MiniMaxH3TurboLoRA",
                       "MiniMaxH3PDDLoRA", "MiniMaxH3LoRABranch")

# PDD (alibaba-pai): the final output head is replicated per interval and each step decodes a block of four as one
# mean velocity, so 8 evaluations cover the 32-point trajectory. Its block boundaries are the plain 8-step shifted
# schedule, bit-identical, so shift and scheduler stay at the base's. The published files do not load in ComfyUI
# (bare `lora_down`/`lora_up` keys: all 728 tensors are skipped, and the render looks like a bad LoRA). These are the
# files `bench/convert_pdd_lora.py` writes, loaded by `MiniMaxH3PDDLoRA`, since two of the three mechanisms are not
# weight patches. Strength 1.0 is **inherited**: the vendor's default and what their comparison clips used. 0.0 is a
# valid control: the node falls back to the checkpoint's own heads.
PDD_FL2VA_LORA = "h3/minimax_h3_fl2va_pdd_8step_comfy.safetensors"
# The stripped sidecar (2026-09-05, `bench/convert_pdd_lora.py --omit-backbone --baked <file> --baked-strength 1.0`):
# heads, the adaln bake and the refiner LoRA, no backbone. It pairs with `MODELS["unet_fl2va_pdd8_baked"]`, and the
# node checks the pair by content, so a graph naming one without the other fails at execute.
PDD_FL2VA_STRIPPED_LORA = "h3/minimax_h3_fl2va_pdd_8step_stripped_comfy.safetensors"
PDD_REF2VA_LORA = "h3/minimax_h3_ref2va_pdd_8step_comfy.safetensors"
# The step counts one converted file serves: any divisor of the 32-point grid lands exactly on the plain shifted
# schedule, and the node fuses the heads for the count asked. **Inherited**: 8 is what the file records, and 4 the
# other count the vendor's README renders at; one weight set at two exact counts is the parallel-decoding claim.
PDD_STEPS = 8
PDD_STEPS_FAST = 4
PDD_STRENGTH = 1.0

#: The `[8,8,4,4,4,4]` partition of the 32-point grid as `ManualSigmas` input: six evaluations, the coarse blocks at
#: the nearly flat front, and a final step at 63.2%, the tail of the vendor's 8-evaluation schedule. The uniform
#: 4-evaluation partition's last step spans 80% of the trajectory, and no other four-block partition is legal.
#: **Measured** on one seed, 2026-08-28: against the uniform 4-evaluation arm, otherwise matched, the owner found
#: jagged lines and scratchy audio there and neither here (`docs/research/pdd/2026-08-28_audio_under_pdd.md`, with
#: the caveats). Written out so the graph shows its schedule; `bench/grade_pdd_partitions.py::MANUAL["tail6"]` is the
#: same vector.
PDD_MANUAL_SIGMAS = "1.0, 0.972973, 0.923077, 0.878049, 0.8, 0.631579, 0.0"

#: Evaluations `PDD_MANUAL_SIGMAS` runs, passed as the graph's `steps` so every reader of the step count sees what the
#: sampler runs; the PDD node's own `steps` goes to 0, since `ManualSigmas` replaces the schedule it would emit.
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
#: first pair is the 2026-09-25 C1 session (`docs/state/next_steps.md`).
AUDIO_REFINE = dict(steps=6, sampler="euler", scheduler="simple", denoise=0.5,
                    video_mask=0.0, audio_mask=1.0)

#: The frozen-video cache (`frozen_video_cache.py`), at the node's API inputs:
#: on the refine pass's model (`refine_cache`), and on the song node's model in
#: the masked probe graph (`masked_cache`). **Inherited**: int4 and no refresh
#: are ComfyUI-H3-AudioRefine's node defaults; `verify` is off because it runs
#: every cached step stock as well. `halo` is **reasoned**, not measured: the
#: node's own default, a width of no tokens, which recomputes no kept row; on
#: the refine pass, where the video is frozen whole, it has nothing to widen.
#: A width for a masked window is `verify`'s ring figure's to set. The refine
#: pass on the card: `bench/results/2026-10-06_frozen_cache_stage_split.md`;
#: a masked window: `bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`.
FROZEN_VIDEO_CACHE = dict(precision="int4", refresh=False, refresh_every=2, verify=False, halo=0)
FROZEN_VIDEO_CACHE_NODE = "MiniMaxH3FrozenVideoCache"

# ---- Masked video-to-video on the song node ----------------------------------
#: The segmenter core's SAM3 nodes take from CheckpointLoaderSimple: Meta's
#: `sam3.1_multiplex.pt` repacked by `bench/convert_sam3_checkpoint.py`, which
#: changes no weight and checks that on readback. From the original rather
#: than a repackage so the conversion is ours (owner, 2026-10-04).
SEGMENTER = "sam3.1_multiplex_fp32.safetensors"
#: `MiniMaxH3SubjectTrack`'s inputs, equal to the node's defaults (`subject_track.py` gives each one's source). It
#: replaced core's tracker chain in the shipped graph on 2026-10-04 (owner): core's tracker spends a cap on every
#: person and splits one subject into several objects across cuts (`bench/results/2026-10-04_masked_v2v_band.md`).
#:   pick         `most central` since 2026-10-07, **measured, then decided**: on the lane's crowd clip `largest`
#:                picked a figure at the frame's edge on all three test windows
#:                (`bench/results/2026-10-07_subject_track_under_nudge.md`); an arm before that date used `largest`.
#:   corrections  written at its empty default since 2026-10-06, so a graph shows the input a many-cut clip needs: on
#:                the first crowd clip the match carried the subject across one cut in six.
#:   track_head   `True` since 2026-10-19, **measured** on one shot (the kitchen lead, 751-785; owner, C198/C199,
#:                `docs/state/decisions.md`); it replaced `False` and the capture's un-tracked `sam31_head_tube`.
SUBJECT_TRACK = dict(subject_phrase="person", pick="most central", pick_on="automatic", match="automatic",
                     cuts="automatic", detection_threshold=0.5, max_people=16, head_phrase="head",
                     corrections="", track_head=True)
#: `MiniMaxH3MaskedSource`'s inputs, equal to the node's defaults (`video_mask.py`).
#:   grow_pixels 64, composite "only what changed"  the owner's choice on the band clip, 2026-10-04, one clip and one
#:                seed each: the wider margin held identity better than one DiT token, the first, reasoned value. A
#:                masked job's loads use `MASKED_LOAD_COMPOSITE` instead. The feather is reasoned and unjudged.
#:   replace, paint_out  the values that worked: head and hair gave an oversized head on a long-haired original, and
#:                painting out was judged worse.
#:   motion_reference "none"  the shipped render as it was (2026-10-05); `subject only` is MASKED_MOTION_SOURCE.
#:   grow_by      a fixed margin. The subject-sized margin is **reasoned**: rendered (2026-10-08 and 2026-10-09,
#:                `bench/results/2026-10-09_masked_one_window_why.md`) but not judged against the fixed one as a pair.
#: `reuse_mask` was removed 2026-10-10: nothing an earlier run made is reused by a key.
MASKED_SOURCE = dict(grow_pixels=64, grow_by="a fixed margin", feather_pixels=8, replace="whole subject", paint_out=False,
                     part_phrases="hair, head", part_threshold=0.5, part_margin=8,
                     composite="only what changed", change_threshold=0.05,
                     motion_reference="none", motion_short_edge=384, motion_vae=False,
                     start_from="noise", start_top=0.3, start_blur=16, start_knots=1)
#: Steps on the ref2va motion graph. **Measured** 2026-10-05 (`bench/results/2026-10-05_masked_v2v_motion_arms.md`):
#: with the subject's own frames as an encoder-only video reference, ref2va carries the original's turn at 12 and 16
#: steps and loses it at 8 with or without its PDD bake, and fl2va never takes it. **Judged** 2026-10-07 (owner, one
#: pair on one masked window and seed): 12 and 16 look the same, an artifact near five seconds included
#: (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`, "Twelve against sixteen steps"). The fl2va PDD8
#: graph stays the default for shots that need no movement from the source.
MASKED_MOTION_STEPS = 12
#: The Masked Source as the ref2va motion graph ships it: the subject's own frames on gray as the encoder-only motion
#: reference (the masking board's route 1, measured above), every other value `MASKED_SOURCE`'s.
MASKED_MOTION_SOURCE = dict(MASKED_SOURCE, motion_reference="subject only")
#: `MiniMaxH3MaskedPrompt` (`masked_prompt.py`), which writes the masked graphs' prompt from these choices and from
#: the Masked Source wired into it. Equal to the node's defaults: a reference still of anybody (`subject` is the
#: user's own words for who it shows), who is the voice on the track, and what the still provides, read off the
#: Masked Source's `replace`. **Reasoned**: a shipped graph holds a placeholder still, so its text cannot assume a
#: man or a woman. **Measured** once, 2026-10-06 (`bench/results/2026-10-06_masked_v2v_person_text.md`): on the
#: motion graph "a person" carried the band clip's turn and started it later than "man" at the same seed, so a user
#: sets `subject` to match their still. The sentences and what each has rendered on are in `masked_prompt_text.py`.
MASKED_PROMPT_NODE = "MiniMaxH3MaskedPrompt"
#: The Sapiens2 route to a part of the subject (`sapiens2_parts.py`): the loader's two folders under
#: `models/sapiens2/`, and `MiniMaxH3SubjectParts` at its own defaults, which tick hair and face-and-neck, the region
#: the Masked Source's `head and hair` asks SAM 3 for. No matting model: the Masked Source reads the `parts` mask,
#: not the matte, so loading one would cost a pass per frame for nothing. **Measured** on one shot of one clip
#: (`bench/results/2026-10-05_sapiens2_first_frame.md`): hair found on every frame, where the SAM phrase landed on a
#: neighbor on a few.
SAPIENS2 = dict(segmentation="facebook_sapiens2-seg-1b", matting="none")
SUBJECT_PARTS = dict(hair=True, face_and_neck=True, upper_clothing=False, lower_clothing=False,
                     hands=False, mouth=False, other_classes="", crop_margin=32, subject_margin=8,
                     matte_reach=8, hold_missing=True)
#: The Sapiens2 pose model, the one that names 308 points a frame: its folder
#: name in the owner's model store. `bench/pose_along_shot.py` is told where
#: that store is; nothing here knows a path. **Inherited**: the folder the
#: first run loaded (`bench/results/2026-10-11_sapiens2_pose_first_frames.md`).
POSE_MODEL = "facebook_sapiens2-pose-1b"
#: How far the figures read off those points (`bench/_lib/face_points.py`)
#: are trusted. Every line is a STARTING VALUE from one or two clips of one
#: video: a tool prints the clip's own figures beside it and says when they
#: sit elsewhere, and none of these is a rule for another video.
#: `eyes_shut_under`, `eyes_open_over`: the two ends of the eye opening that
#: a person looking at every frame agreed with. Between them the figure is not
#: readable: smiling, blurred eyes and open ones read alike. **Measured** by
#: eye on one close-up, a face far larger than most in a shot (same record).
#: `lip_score_at_least`: under it the lip gap jumped or wandered on a steady
#: mouth. **Measured** on that close-up. It does not make the lip gap an
#: opening: a shut mouth scored near one and read a third open.
#: `lid_gap_px_to_draw`: under it the lash lines are too close on the canvas
#: for a mark to be drawn from them. **Reasoned**: between the median of the
#: one shot whose marks carried and those of shots whose lids are under a
#: pixel apart (the tables under `data/2026-10-11_sapiens2_pose_consecutive/`).
#: `eye_ring_score_at_least`: under it an iris or a pupil ring is not drawn
#: (`bench/draw_face_points.py`), taken at the median over the ring's points.
#: **Reasoned**, from one shot (the second subject's short kitchen shot,
#: source frames 1039 to 1060): the pupil points scored 0.20 to 0.28 a frame
#: and lay hundreds of pixels off her face, the iris points scored 0.93 to
#: 0.96 and lay on her eyes, and the line is put between. Meta's own demo
#: drops a point under 0.3 (`coderef/sapiens2`, `pose_render_utils.py`,
#: `kpt_thr`), which sits at the top of the bad group on that shot and was
#: not taken.
POSE_READING = dict(eyes_shut_under=0.020, eyes_open_over=0.045, lip_score_at_least=0.8, lid_gap_px_to_draw=1.6,
                    eye_ring_score_at_least=0.5)
#: The eye marks `bench/draw_lid_marks.py` draws on a body-model video: each eye's outline is grown about its own
#: center by `face_px` over the shot's median face height, so a frontal face's marks come out one size whatever the
#: face's. **Measured** on the one shot whose marks carried
#: (`bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md`); a starting value. A rule by the marks' own
#: width was tried first and failed on a face in profile, whose far eye is narrow.
LID_MARKS = dict(face_px=318.0, fill=(255, 255, 255), edge=(0, 0, 0))
#: The face lines `bench/draw_face_points.py` draws on a body-model video, at the face's true size: a line is the
#: shot's median face height over `face_over_width` wide and never under `least_px`, with a black edge of `edge_px` on
#: each side. **Reasoned**, a starting value; no render had been made with these lines when they were set. The width
#: scales with the face (the lead's brief for a first drawing); `least_px` is where a line stops leaving any white
#: the tool can read back; the colors are the lid marks'. `bench/check_draw_face_points.py` is its check.
FACE_POINTS = dict(face_over_width=60.0, least_px=1.0, edge_px=0.5, line=(255, 255, 255), edge=(0, 0, 0))
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

# ---- A masked job: several subjects, a preview first -------------------------
# What `workflows/masked_job_graph.py` writes into a job's graphs where the
# value is the job's and not the shipped masked row's. A job is one clip with
# any number of subjects, run as a no-sampling preview, then one load per
# subject per shot (the masking board, `guide-order-of-operations-masked-job`).
# `bench/check_masked_job.py` holds each dict that says it equals a node's
# defaults to that node's schema.
#: The loader that reads a file by its path. A job's source is the every-frame
#: 24 fps copy of the clip, which lives outside ComfyUI's input folder, and
#: its saved masks live under `data/`. **Inherited** from
#: `bench/patch_render_window.py` (`LOADER`), 2026-10-08.
MASKED_JOB_LOADER = "VHS_LoadVideoFFmpegPath"
#: The loader's rate for a job: the file's own. The copy is already at the
#: model's rate with no frame dropped, and a forced rate drops or repeats
#: frames of a file that is not. **Measured** on one clip, 2026-10-08
#: (`docs/h3/references.md`, the note dated 2026-10-10 under the loader's
#: rate table).
MASKED_JOB_FORCE_RATE = 0.0
#: How far before a frame, in frames, a loader is asked to start so that the
#: frame is the first it yields. **Inherited** from
#: `bench/patch_render_window.py` (`SEEK_BACK`), which gives it as reasoned (a
#: time between two frames cannot land one off) and measured once on
#: 2026-10-08. `masked_job_graph.seek` is the one place it is used.
MASKED_JOB_SEEK_BACK = 0.4
#: `MiniMaxH3SAM31Corrections`, every correction on, equal to the node's
#: defaults, between the SAM loader and everything that takes the segmenter.
#: The image range and the text activation: **measured** on one stretch of one
#: clip, a first read (`bench/results/2026-10-10_sam31_corrections_on_one_stretch.md`).
#: `memory_selection`: **measured** on three `fun` scenes and a group dance with
#: SAM 3.1, 2026-10-17, no arm worse with it on
#: (`data/2026-10-17_sam_memory_selection/`), and on at the owner's word that day
#: (`docs/state/decisions.md`). A preview job turns it off only with a person's
#: reason (`sam_memory_selection` in the job block, `masked_job_graph.check_job`).
MASKED_JOB_SAM_CORRECTIONS = dict(correct_image_range=True, correct_text_activation=True, memory_selection=True)
#: The part node in a job's preview: `SUBJECT_PARTS` with the node's own fill
#: of a doubted frame off. **Reasoned**: the capture offers that fill itself
#: and grades it against the class map (`bench/capture_masked_run.py`,
#: `grade_holds`), and a part filled before the capture reads it cannot be
#: graded. The ticks are a subject's own when the job names them.
MASKED_JOB_PARTS = dict(SUBJECT_PARTS, hold_missing=False)
#: The body model a job's pose pass loads, from `models/detection/`: the
#: release the pose check runs against Meta's reference
#: (`bench/check_body_pose.py`).
MASKED_JOB_BODY_MODEL = "sam_3d_body_dinov3.safetensors"
#: `MiniMaxH3SubjectBoxes` in a job. `margin` is **inherited** from the
#: session builder every render of 2026-10-10 came from; nothing has judged it
#: against the node's own default. `smallest_mask` equals the node's default
#: (`subject_boxes.SMALLEST_MASK_PX`, which has its provenance).
MASKED_JOB_BOXES = dict(margin=16, smallest_mask=2048)
#: `MiniMaxH3BodyPose` in a job, equal to the node's defaults. `subject`,
#: `first_source_frame` and `table_prefix` come from the job file, and the
#: graph wires the subject's own track to `subject_mask`, so that a box
#: holding two people poses the one tracked. Wired, the node gives the model
#: its mask prompt (**inherited**: Meta's estimator and ComfyUI's own node
#: both do when a track exists) and shows it the person alone (**measured**
#: on one shot, and ours: `body_pose.py`, the module's note on
#: `subject_mask`).
MASKED_JOB_BODY_POSE = dict(hand_refinement=True, camera="image diagonal", fov_degrees=55.0, batch_size=64)
#: The mesh a job's preview draws for each subject: the plain lit body, at the
#: canvas's width and height, which the graph adds. `marked` is committed and
#: had not been rendered when this was written (`docs/masked_v2v/state_signals.md`,
#: "How a signal reaches the model").
MASKED_JOB_MESH = dict(style="mesh", size="width and height")
#: How a job's preview saves a mask or a class map: lossless, no chroma step, so a class index survives as a gray
#: level. **Inherited** from the session builder of 2026-10-10; `bench/capture_masked_run.py files` reads exactly these
#: files.
MASKED_JOB_LOSSLESS = dict(format="video/ffv1-mkv", level="3", coder="1", context="1", gop_size=1,
                           slices="16", slicecrc="1", pix_fmt="bgra", loop_count=0, save_metadata=False,
                           pingpong=False, save_output=True)
#: How a job's preview saves a picture meant for a person to look at (the
#: mesh). **Inherited** from the same builder.
MASKED_JOB_PICTURE = dict(format="video/h264-mp4", pix_fmt="yuv420p", crf=14, loop_count=0,
                          save_metadata=False, trim_to_audio=False, pingpong=False, save_output=True)
#: The shipped graph a job's loads start from, under `workflows/`: the generator's own output for the masked
#: upper-body row on ref2va. A load takes its model, sampler, steps, shift, attention and reference still from
#: that file as it stands, so a default moved in the generator moves a job's loads at the next rebuild.
#: **Measured** 2026-10-10: the graph every render of that day and the 2026-10-08 final were patched from
#: differs from this file in seven inputs and no node, and each of the seven is a value below or a value of
#: the job file.
MASKED_JOB_BASE_GRAPH = "h3_video_to_video_masked_upper_song_ref2va_motion_api.json"
#: The frames a later window keeps from the one before, in a job. **Inherited** from the 2026-10-08 final and
#: every render of 2026-10-10; the shipped row keeps fewer. What the kept frames do to a window is argued in
#: `docs/masked_v2v/window_context.md`.
MASKED_JOB_CONTEXT = 90
#: How clean a load that continues another is shown the frames it keeps (the song node's `context_noise`), set by
#: default only when a load has a `continue_from`. **Measured** on one stretch of one clip, one seed
#: (`docs/masked_v2v/window_context.md`, "Results, dated"; the records are `data/2026-10-10_fun_pose/kitchen_cn_*`):
#: at this level a continuation followed its motion video through a head turn that the same continuation with clean
#: kept frames did not make; a lower level had not been read. A load of several chained windows takes its own value
#: for its inner windows on a person's word (`shot_to_load job --context-noise`; the lead's ruling, 2026-10-15). A
#: starting value for another clip, which a load's own `context_noise` replaces.
MASKED_JOB_CONTEXT_NOISE = 0.8
#: Sol-Attn's sink in this lane. **Decided** by the owner, 2026-10-10, for the masked lane; the node's own
#: default is another decision of theirs, on generated dialogue audio (2026-10-02), and is not moved by this.
#: `bench/results/2026-10-10_kept_frames_against_the_mesh.md` has the two sinks side by side on two shots.
MASKED_JOB_SOL_SINK = "exact_kv_and_rows"
#: The Masked Source's choices a load sets by name. A body mesh saved by the preview is a video the builder
#: wires. A motion video goes in at the canvas's short edge, which the
#: graph works out: the vendor rule for a reference video (`docs/h3/references.md`, "Video references"),
#: adopted and not shown to be a gain (the node's own default, `video_mask.MOTION_SHORT_EDGE`, is smaller). A
#: load's own `motion_edge` replaces it.
MASKED_JOB_MOTION = dict(none="none", wired="a video I wire")
#: A band left open under a head-and-hair part, for a still whose hair falls past the original's jaw
#: (`bench/masked_job.py::hair_band`; a load asks for it with `hair_band`, a share of the part's height, and there is
#: no default). The fault it answers: below the part every cell is a keep-out's to give back to the source, and the
#: clothing mask's pieces change frame to frame, so the region's lower edge moved and cut the drawn hair where it moved
#: (the long kitchen shot, source frames 693 to 703, on playback).
#:   cuts "clothes"   the keep-out the band is taken out of, by its label in a load's `others`. **Reasoned** (the
#:                    lead's design, 2026-10-12).
#:   wider "margin"   the band reaches past the part's box on each side by the load's own `margin` in px (a share of
#:                    the width is still allowed, as a number). **Measured** 2026-10-14, replacing 0.1 of the box's
#:                    width: at 0.1, on second_hfp_0484b, the clothes keep-out handed back a column of margin cells at
#:                    each shoulder where the new long hair falls, and the owner saw her hair come and go there (frame
#:                    488). At the margin's width no clothes cell beside her head was handed back on 484b (516 before)
#:                    and 362 on 0604_b (7436 before), and nothing new was drawn over a held thing
#:                    (internal/claude/2026-10-14_mrdrill_hair_read.md).
#:   smooth_frames 4  the box is the median of the boxes this many frames either side, so the band's edge does not
#:                    move with one frame's mask. **Reasoned** (2026-10-12).
#:   most 2.0         the deepest band a load may ask for, in part heights. **Reasoned** (2026-10-12).
#:   starts "top"     the band begins at the part box's top, so the cut keep-out is emptied beside the head as well as
#:                    under it ("bottom", under the part only, was the 2026-10-12 build). **Measured** 2026-10-14: on
#:                    second_hf_0604_b (604-948) the clothes keep-out's track ran a strip down the original woman's
#:                    hair edge, and its cells were handed back beside her head on 90 frames
#:                    (internal/claude/2026-10-14_mrtwo_scripts/hair_0604/findings.md).
#: No reasoned value here was measured on a render; the first face-and-hair load that carries a band is read by a
#: session that did not build it.
MASKED_JOB_HAIR_BAND = dict(cuts="clothes", wider="margin", smooth_frames=4, most=2.0, starts="top")
#: The strangers keep-out (`bench/strangers.py`): every person the part model (`SAPIENS2["segmentation"]`) finds on
#: the whole frame, less each of the preview's subjects' tracks grown by `grow_px`; a load whose plan names it has its
#: part held to its own subject's track grown the same (`bench/shot_to_load.py`). The grow is there because the part
#: model reaches past the tracker's edge on the subject's own body.
#:   grow_px 8      **Measured** as a starting value, 2026-10-14, on two close groups (the lead's 3167-3213 and
#:                  3240-3276; internal/claude/2026-10-14_mrdrill_stranger_read.md), against 0, 4 and 16: at 8 the
#:                  region took back 812 of 7546 cells on 3167 and 293 of 4464 on 3240, with fewer frames holding an
#:                  enclosed hole than at 0. Faces about 109 px tall on 3167; a face of another size may want another.
#:   hold_frames 4  how far an empty frame of a track is held to the nearest frame with one (`strangers.held_track`).
#:                  **Reasoned**: one latent step of the video model. A tracker that drops its subject for a step is
#:                  holding the same person; a track empty for longer is a subject who left, and holding them there
#:                  would keep a newcomer to that spot out of the strangers file. The records list every frame held
#:                  and every frame left empty.
MASKED_STRANGERS = dict(grow_px=8, hold_frames=4)
#: A look's `hairline` room (`bench/look_rooms.py`): for a subject whose hair is kept and whose region is the face,
#: his hair over his face and a band of his hair at the hairline, regenerated, so the region's edge sits in his curls
#: and holds still. `hold_frames`: a pixel is in the room on a frame when the class map has it so on any frame this
#: many either side. `band_px`: how far into his hair from Face_Neck the band reaches, in px of the canvas.
#: **Measured** both, 2026-10-14, on third_fp_1222b's masks (the region re-made as the node makes one, zoomed 2x as
#: the remake would be; internal/claude/2026-10-14_mrpre/figures/hairline_ab_1222b_zoomed2x.json): his hair switching
#: between regenerated and kept fell from 263 cells on 24 frames (his part alone) to 46 on 14, with 4 notch cells
#: against 37, at the cost of the region reaching 42-54 px above his hairline instead of 16-26; a 24 px band gave 33
#: on 13. Ruled by the lead that day (arm A, 16 px). One shot, one face of about 58 px: a starting value for others.
LOOK_HAIRLINE = dict(hold_frames=4, band_px=16)
#: What the text encoder is shown of a load's still: the video model's own copy, not a smaller one of its
#: own. **Decided** by the owner, 2026-10-08, for the reference stills of this lane (the largest size, enlarged
#: to it, one copy for both models); the 2026-10-08 final and every render of 2026-10-10 ran so. The shipped
#: row gives the encoder its own copy (`REF_QWEN_SHORT_EDGE`), and is not moved by this.
MASKED_JOB_STILL_VIEW = "shared"
#: The song node's encode for a job's render. **Inherited** from the session builder of 2026-10-10, which
#: took it from `bench/patch_render_window.py` (`PATCH_CRF`): a pass is encoded again at delivery.
MASKED_JOB_CRF = 12
#: The capture's word for a region's edge against the Masked Source's (`bench/capture_masked_run.py --plan`,
#: `edge=`): a loads file is written in the capture's words.
MASKED_JOB_EDGE = dict(cells="latent cells", tokens="whole tokens")
#: What a load's job file is given in its `defaults` when a tool writes the file (`bench/shot_to_load.py`), so the
#: file stands alone and no load types them. **Inherited**, every one, from the job files of the loads of
#: 2026-10-12 and 2026-10-13, where each was typed into every file with this value; none was measured for this
#: list, and a load's own key replaces any of them. The seed a job file is given is `SEED`.
MASKED_LOAD_DEFAULTS = dict(carried="held", motion="mesh", feather=8, edge="cells", motion_edge=768)
#: The margin of a part load, by what is replaced (the plan's region for the subject). **Inherited** the same
#: way: the value on every face load and on every face-and-hair load of those two days. There is none for
#: `whole`: that tool makes no whole-person load.
MASKED_LOAD_MARGIN = {"face": 16, "face and hair": 64}
#: The composite every load is built with (`workflows/masked_job_graph.py`), in place of the Masked Source's own
#: default ("only what changed"). **Measured, then decided** (owner, 2026-10-16, with the instruction that it never
#: happen again): a code trace (`internal/gemini/2026-10-16_holes_trace.md`) found "only what changed" restored 19% of
#: the mom's regenerated pixels to the source on second_hfp_0014d, as holes stepping every 4 frames; the one-knob
#: trial was second_hfp_0014f. A load may still name another, and its build record says so.
MASKED_LOAD_COMPOSITE = "whole region"
#: The part model's parts (`sapiens2_parts.PARTS`) that the largest-piece rule is never applied to: a load whose part
#: holds any of their classes on any frame is refused, at the job (`bench/shot_to_load.py`) and at the capture
#: (`bench/capture_masked_run.py refuse_pieces_on_a_head`, on the class map). **Measured, then decided** (owner,
#: 2026-10-16, `docs/state/decisions.md`, e6993b5e): the rule dropped pieces of the mom's face and hair on 146 of 154
#: frames of second_hfp_0014d (`internal/gemini/2026-10-16_holes_trace.md`) and of the lead's face on 180 of 362
#: frames of lead_fp_0613c, up to 9333 px on frame 772. Open: the case the rule was set for (scraps of her part on an
#: arm, grown by the margin, enclosing a cell of her own shoulder, frames 3262-3265, 2026-10-14) is to be met by
#: closing a hole the region encloses, never by dropping a piece; not built.
NO_PIECES_ON = ("hair", "face_and_neck")
#: The canvas a job is made on when a tool writes its file. **Inherited**: every job file of this lane since
#: 2026-10-10 names it; `docs/h3/geometry.md` lists it (its 4:3 row).
MASKED_JOB_CANVAS = (1024, 768)

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
#: Core's node at the template's widget values. `keep_percent` 10 is the template's; FastVideo's own contract is
#: sparsity 0.8, keep 20, on every step (`FASTH3_CONTRACT_VSA` below). This stays the template arm;
#: `bench/results/2026-09-26_fasth3_contract_s1.md` compares the two.
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
#: int8 since 0.154.0 (`docs/attention/quant_policy.md`).
FLASHGEN_LORA = "h3/minimax_h3_4step_lora_flashgen_v1.0_768p_fl2va_pruned_avg_rank_13_bf16.safetensors"
#: The publisher's LoRA at its full rank 64, converted here by
#: `bench/convert_flashgen_lora.py` (`data/results/2026-09-25_flashgen_lora_conversion.json`):
#: exact to the release where kijai's resize keeps about 95% of each delta.
FLASHGEN_R64_LORA = "h3/minimax_h3_flashgen_4step_v1.0_768p_fl2va_pruned_rank64_comfy.safetensors"
#: The same at rank 64 for the pruned Ref2VA checkpoint: `convert_flashgen_lora.py
#: --partition Ref2VA` fits the adaln onto that partition's own time basis
#: (`data/results/2026-09-26_flashgen_lora_conversion_ref2va.json`). FlashGen was
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

#: The distill LoRAs applied at the call whose Sol nodes take SOL_DISTILL_LORA_CUDA
#: (`sol_for_graph(..., distill_lora=True)`): every FlashGen file, since 2026-10-01 (owner).
SOL_DISTILL_LORA_FILES = frozenset({FLASHGEN_LORA, FLASHGEN_R64_LORA, FLASHGEN_R64_REF2VA_LORA})

# ---- Step switches: one distill's pass hands off to another --------------------------------------------------------
#: Route 3 of the distill-routing idea (`docs/research/2026-09-26_distill_routing.md`; the design is in
#: `docs/state/next_steps.md`, agreed with the owner 2026-09-26): FlashGen's own first two steps, then PDD8 from exactly
#: where FlashGen left the latent. Pass 1 is derived from FLASHGEN_MANUAL_SIGMAS so it cannot drift from it. Pass 2's
#: last two points are PDD8's own knots, **measured**: `comfy.samplers.calculate_sigmas(simple, 8)` at shift 12 prints
#: 1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0 (`bench/check_pdd_sigmas.py`'s
#: `comfy_simple`, 2026-09-26). Starting at 0.888889 puts PDD's first step on grid heads 19..23
#: (`pdd_math.schedule_knots` [19, 24, 28, 32]); the tracker may warn that the step sits off a boundary.
STEP_SWITCH_PASS1_SIGMAS = ", ".join(FLASHGEN_MANUAL_SIGMAS.split(", ")[:3])
STEP_SWITCH_PASS2_SIGMAS = ", ".join([FLASHGEN_MANUAL_SIGMAS.split(", ")[2], "0.8", "0.631579", "0.0"])

#: PDD8's own schedule, **measured**: `comfy.samplers.calculate_sigmas(simple,
#: 8)` at shift 12, which `bench/check_pdd_sigmas.py` proves bit-identical to
#: what `MiniMaxH3PDDLoRA` emits at 8 steps (2026-09-26).
PDD8_SIGMAS = "1.0, 0.988235, 0.972973, 0.952381, 0.923077, 0.878049, 0.8, 0.631579, 0.0"

#: The reverse step switch (owner, 2026-09-26: answer PDD's weaknesses with the other distills' strengths, PDD first):
#: PDD8's own early steps, which follow the teacher's layout, then FlashGen finishing, so its committed final jump
#: replaces PDD's hedging late block (`docs/research/2026-09-26_distill_routing.md`; `docs/distills/which_distill.md`
#: P3). Two handoffs, both on PDD8 knots:
#:   h063: PDD steps 1..7 to 0.631579 (knot 28), then FlashGen 0.631579 -> 0, near FlashGen's trained last step
#:         0.679245 -> 0;
#:   h080: PDD steps 1..6 to 0.8 (knot 24), then FlashGen 0.8 -> 0.679245 -> 0.
#: FlashGen's pass starts off its trained points in both (**reasoned**; a probe).
_P8 = PDD8_SIGMAS.split(", ")
STEP_SWITCH_REV = {
    "h063": (", ".join(_P8[:8]), "0.631579, 0.0"),
    "h080": (", ".join(_P8[:7]), ", ".join(["0.8", FLASHGEN_MANUAL_SIGMAS.split(", ")[3], "0.0"])),
}
#: PDD8 cut at a handoff, then the undistilled base finishes on Euler
#: (`docs/state/open_experiments.md` #37). **Reasoned:** pass 2 walks the 32-point
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

# Node order is load-bearing: Sol composes with the attention patch it finds, so it comes after the dense kernel. It
# is graded on every graph's wiring by `bench/check_attention_defaults.py` and explained in
# `docs/attention/sol_wiring.md`, "The order". A `CHAIN` constant that no code read was removed 2026-08-14.

# ---- Single-frame image path (parked 2026-08-27) and the draft decoder -----------------------------------------------
# The image path's values exist only for `length=1`, and every one is wrong for a video.
#: The image VAE (huggingface.co/Mamad8/MiniMax-H3-Image-VAE, experimental, step 1597): H3's VAE with a decoder tuned
#: to reconstruct one image from one temporal latent. **Measured** 2026-08-15 from the safetensors: of 562 tensors,
#: all 116 encoder tensors, `quant_conv` and the latent mean and std are byte-identical to the video VAE's, so the
#: latent space is untouched and swapping is a pure decoder swap; its `h3_t1_output_slice: 3` is what core's
#: `decode()` already does at one latent frame. Never wire it into a video graph: its README says the decoder
#: regresses multi-frame reconstruction and can ghost.
IMAGE_VAE = "minimax_h3_t1_image_vae_step1597.safetensors"

# Core's tiny autoencoder for H3, from `models/vae_approx/`, **inherited** from core
# (`comfy/latent_formats.py::MiniMaxH3Video.taesd_decoder_name`). Retired as a shipped graph 2026-09-26 (owner: not
# worth it; the INT8 VAE left a draft about 15 s cheaper per discarded seed, and taeh3 ghosts); kept for
# `bench/compare_vae_decoders.py` and the `draft_decode` switch. A draft is for choosing a seed, never for judging: a
# `draft_decode` graph saves the latent, so a keeper gets the real decode from `h3_decode_saved_latent_api.json`
# without sampling again (`docs/state/open_experiments.md` #31).
DRAFT_VAE = "taeh3.safetensors"

# 2:3 portrait, inside the trained family (`adapt_canvas(2, 3)` returns it). **Reasoned**: the community workflow this
# path follows renders 1024x1536, 52% over H3's 768*1344 area cap, which costs 1.78x the tokens per frame (about 3.2x
# the attention) on a canvas the checkpoint never trained on. The Resolution node's `custom` option reaches it.
IMAGE_EDIT_CANVAS = dict(width=768, height=1152)

# What every single-frame graph spends.
#   ref_upscale False      **Measured** 2026-08-16, the opposite of the video default: `h3_image_style` at one seed took
#                          89.1 s with upscaling and 18.1 s without, with the same identity, medium and detail (the
#                          earlier 84 s / 18 s ladder, `open_experiments` #16e, reproduced on a second subject). A small
#                          n, but a render is seconds. The video graphs keep upscaling on.
#   steps 16               **Measured**, one pair per scene: 8 steps looks free on `h3_image_edit` (13.0 s against
#                          4.0 s) and `h3_image_style`, but on `h3_image_multiperson` (25.0 s against 10.0 s) it loses
#                          the woman's freckling and pendant, the identity marks that scene grades. Measured on a single
#                          portrait only, a step cut would have looked free everywhere.
#   ref_image_size `max`   A no-op: with no upscaling, core's `min(1.0, 2048/short_edge)` leaves a source under 2048 at
#                          native size. For a source above 2048 it is not, so do not simplify it away.
IMAGE_EDIT_BUDGET = dict(**IMAGE_EDIT_CANVAS, ref_upscale=False)

# ---- Canvas, frame rate, length and seed ---------------------------------------------------------------------------
# The default canvas, chosen by tier so "render this cheaper" is one edit. **Measured** against `adapt_canvas` on
# 2026-08-16 by enumerating the legal family: the area cap binds, so going wider trades height for width and buys no
# speed (1344x768, 1536x672, 1792x576 and 2016x512 are all 1008 tokens/frame); the only cheaper direction is toward
# square, a 32 px ramp. Attention goes as the square of tokens. All four tiers are legal and inside the trained
# family; true 16:9 (1376x768) is outside it below the cap and costs more.
#
#   tier    canvas      ratio           tok/f  attention
#   full    1344x768    1.75            1008   1.00x
#   near    1280x768    1.67 exact 5:3   960   0.91x
#   fast    1152x768    1.50 exact 3:2   864   0.73x
#   draft   1024x768    1.33 exact 4:3   768   0.58x
#
# `fast` is the iteration canvas (27% off attention, framing reads the same); `near` when a comparison must stay close
# to `full`; `draft` only for "does the pipeline run". A Sol measurement may use `fast` and must not use `draft`: Sol
# needs roughly 60k tokens to show anything, and at 243 frames `draft` is 55,296 (`fast` 62,208, `full` 72,576).
CANVAS_TIER = "full"

CANVAS_TIERS = {
    "full":  dict(width=1344, height=768),
    "near":  dict(width=1280, height=768),
    "fast":  dict(width=1152, height=768),
    "draft": dict(width=1024, height=768),
}

CANVAS = dict(CANVAS_TIERS[CANVAS_TIER])
FPS = 24.0

# The generator's default `length`, on the 17k+5 frame grid. The ceiling, the longest length H3 was trained on, is
# `h3_rules.MAX_LENGTH`, read below and never retyped (AGENTS.md, "Settled about H3").
LENGTH = 124
#: The model's ceiling on a length, `h3_rules.MAX_LENGTH`, read here for the tools that cannot import the node pack.
MAX_LENGTH = _h3_rules().MAX_LENGTH
#: What a long render is: the ceiling.
LONG_LENGTH = MAX_LENGTH
#: The longest window a run of several windows may take: the longest length on both clocks under the ceiling
#: (`h3_rules.chain_lengths`, the seam rule; worked out, never typed). A graph whose `length` is the song node's
#: `window_frames` takes this; a window that holds its whole track may go to `LONG_LENGTH`.
LONG_WINDOW = max(_h3_rules().chain_lengths())

# One fixed seed shared by every graph and bench arm, so two probe graphs that differ in one setting differ in
# nothing else. Change it here and regenerate everything.
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


# ---- Paths -------------------------------------------------------------------------------------------------------
#: One resolver, because counting `..` by hand has cost real time: `bench/grade_pdd_partitions.py` once resolved
#: `Path(__file__).resolve().parents[2] / "output"`, which from `bench/` is `custom_nodes/output`, and seven arms were
#: rendered and thrown away undecoded. Two conventions differ by one level (`parents[2]` of a file is
#: `custom_nodes`; of a directory, the ComfyUI root), so ask for the thing by name.
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
    # An existing stock directory is not evidence it is the right one: on this box `<comfy>/output` holds only
    # ComfyUI's placeholder, since the server writes to a share, and an empty right-looking directory reads as "no
    # renders" rather than "wrong path" (a peer session lost time to exactly that). So the fallback must hold a render.
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


# ---- Reference inputs --------------------------------------------------------------------------------------------
def _ref_qwen_short_edge() -> int:
    return int(_h3_rules().REF_QWEN_SHORT_EDGE)


#: The encoder's own view of a still, as a short edge: the append node's default (`qwen_view = separate` at this
#: size) since 2026-10-03, and what the generator writes on every still's append node. Read from `h3_rules.py`, which
#: owns the value and its history (`shared` was the default from 2026-09-13 to 2026-10-03), so the node and the
#: generator hold one number.
#: **Pinned to `shared` instead (the generator passes 0)** on the instruments, whose renders are compared byte for
#: byte with earlier ones: the `_savelat` and `_x0` twins, the `bench/` graphs, and the `h3_probe_refview2_*` scenes,
#: whose manifest (`bench/refview2_arms.json`) defines its `parity` arm as the graph as built at the 2026-09-13 values.
#: The generator always writes the selection, so none of these inherits the node default.
REF_QWEN_SHORT_EDGE = _ref_qwen_short_edge()

# The canvas for graphs that carry a reference video: 4:3 rather than 1344x768, taken so the reference stays full
# length on 24 GB. **Measured** 2026-08-13: at 345 frames, 1024x768, references not upscaled, a reference arm peaked at
# 22,735 MiB of 24,564 (34.3 min end to end); at 1344x768 with upscaled references it ran out of memory at step 4 of
# 16 after every attention fallback had fired. A reference video is truncated to the generated frame count, so it
# costs twice (video rows and reference rows), and shortening the render shortens the reference: give up canvas and
# still detail first, never reference duration. Read preflight before a long reference arm.
#
#   345 frames -> 182,092 tokens   (1344x768, references upscaled: out of memory on 24 GB)
#   209 frames -> 120,918
#   124 frames ->  82,686
REF_VIDEO_CANVAS = dict(width=1024, height=768)
# Reference-video graphs render at the length of the reference clip, not `LONG_LENGTH`: since 2026-08-22 the shared
# clip is trimmed to end in a 0.3 s silence, and its frame count lands exactly there. **Measured**: the untrimmed
# 19.56 s source was cut mid-delivery by a 15.083 s render, and the last third of every render drifted
# (`data/results/2026-08-22_swap_prompt_verdict_362.json`). Only the graphs wired to this clip take it; it is no
# ceiling.
#: The shared reference clip's own length in seconds. **Measured** (2026-08-22, the trimmed clip).
REF_CLIP_SECONDS = 14.375
#: Its frame count at `FPS`, on the 17n+5 grid: the clip's, worked out from its measured length.
REF_VIDEO_LENGTH = int(round(REF_CLIP_SECONDS * FPS))

# The VHS loader every generated reference-video graph uses: the ffmpeg one, not the cv2 `VHS_LoadVideo`. **Measured,
# then decided** (owner, 2026-09-23, for decode accuracy; `data/results/2026-09-23_vhs_loader_comparison.json`,
# re-derived by `bench/compare_vhs_loaders.py`): at `force_rate` 24 on a 25 or 30 fps source cv2's frames run late
# against the 24 fps grid where ffmpeg's land on the nearest frame, so renders from before the switch are not
# substrate-comparable with renders after; cv2's decode is also biased dark against a bt709 reference. Output slot 1
# is a MASK here, where cv2's is an INT frame count; no generated graph wires slot 1.
REF_VIDEO_LOADER = "VHS_LoadVideoFFmpeg"

#: What every video-bearing reference arm spreads in, so the three numbers have one home: the reference clip's length,
#: the 4:3 canvas above, and `ref_upscale=False` (stills stay at native size rather than going to 2048 on the short
#: edge; the node's default is upscaling on since 2026-09-13). Set for these arms, because they exist to carry a long
#: reference video on a 24 GB card.
REF_VIDEO_BUDGET = dict(length=REF_VIDEO_LENGTH, **REF_VIDEO_CANVAS,
                        ref_upscale=False)


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

#: The three references `h3_probe_capture_ref3` wires, in socket order (character, garment, environment), chosen for
#: the measurement rather than the picture from `internal/reference_library.md`, which records each row cost:
#:
#:     777 rows  0.56 ar  0.78 MP   the one asset that cannot fill a 2048 short edge from real pixels
#:   2,500 rows  1.00 ar  2.56 MP   legible text and a logo: the probe for whether sparse attention drops detail
#:   4,128 rows  1.79 ar  4.23 MP   the size ladder's top end
#:
#: A 5.3x span in DiT rows and three aspects, so a capture can tell reference load from reference shape. The
#: environment still holds two people; the prompt scopes it to "setting, palette, and lighting", and widening that
#: line makes it the wrong asset. `product_soccer_jersey` carries real brand marks: an internal capture artifact only.
CAPTURE_REF_IMAGES = (
    "h3_refs/subject_performer_stage_662x1177.png",
    "h3_refs/product_soccer_jersey_1600x1600.png",
    "h3_refs/scene_loft_couch_duo_2752x1536.png",
)


# ---- Where generated graphs live -------------------------------------------------------------------------------
# Graphs are foldered by use, relative to `workflows/`. `GRAPH_DIRS` is the discovery list every walker shares
# (`graph_paths`): a bare `workflows/*.json` glob is not recursive, so a folder added anywhere else would leave checks
# passing over a set that no longer holds it. `bench/` and `archive/` are not in it: bench graphs read another pack's
# internals and are expected to break, and the archive is history.
#: `image` left 2026-08-27 when the single-frame path was parked; its graphs are `archive/workflows/image/`.
#: `distill_experiments` (2026-09-27, owner) holds every `_savelat` or `_x0` twin, every `h3_probe_*` graph that runs a
#: distill, and entries marked `distill_experiment=True` (`build_workflows._is_distill_experiment`); the shipped
#: distill graphs stay at the root. `daily` (2026-10-03, owner: graphs in regular use) holds `DAILY_GRAPHS`, which
#: says which graphs and where each was promoted from.
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
    # The masked lane (owner, 2026-10-05): the default graph, and the ref2va motion graph for a shot that needs the
    # original's movement (`MASKED_MOTION_STEPS`). Promoted from shipped root graphs, which stay at the root too.
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

# Bench graphs are exempt from schema grading only. A bench graph naming a missing model file is as broken as a
# shipped one (on 2026-08-21 the stamped bench graphs carried the deleted video VAE too), so the folder is reached
# through `graph_paths(include_bench=True)`, never a second glob: `bench/check_graph_discovery.py` forbids a check from
# enumerating graphs itself.
BENCH_GRAPH_DIRS: tuple[str, ...] = ("bench",)

# ---- Which nodes carry prompt text into the encoder ----------------------------------------------------------------

#: Every node class that carries prompt text to the encoder, mapped to the input that holds it; the one copy. Until
#: 2026-10-01 six tools kept their own lists, and the song node was missing from four of them, so its prompts went
#: ungraded. `bench/check_prompt_guide_conformance.py` fails when a shipped graph carries a `prompt` string on a class
#: that is not here. Core's two H3 nodes are listed because hand-built and HF-style graphs use them.
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


# ---- Reading a value out of a graph --------------------------------------------------------------------------------

# A node input is a literal (`"steps": 4`) or a link to another node's output (`"steps": ["50", 0]`). A static reader
# that sees only literals reports a correctly linked graph as broken: when the PDD graphs dropped `BasicScheduler`,
# three checks went red at once (`bench/check_attention_defaults.py::_steps_of`). So the walk lives here, once.
#
# It never executes or guesses: a value computed at run time (`MiniMaxH3Resolution.width` parses a DynamicCombo label
# inside `execute`) is reported as computed. It reads API form only; an editor-saved graph is
# `bench/preflight_graph.py`'s to read. It never decides whether an unresolvable value is a failure: it reports a
# state and a reason, and the caller owns the policy.

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
#: COMPUTED and OPAQUE stay apart because their fixes are opposite, and because merging them would answer "no static
#: reader can know this" about a node nobody has described yet: a confident wrong answer, where "no row" sends you to
#: look.
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

    Three owners, read in this order, and which one a graph has is not a style choice:

      * `BasicScheduler` -> its `steps` and its `scheduler`.
      * otherwise a `ManualSigmas` node -> `len(vector) - 1` and **`manual`**: the vector is an explicit partition
        (the tail-weighted PDD schedule, 2026-08-28), and the count exists nowhere else on that graph.
        `bench/check_distill_grid.py` grades the vector against `pdd_math.partition_bounds`.
      * otherwise a `MiniMaxH3PDDLoRA` with a non-zero `steps` -> that count, and **`simple` by construction**: the
        node emits `1 - pdd_time_grid`, which is bit-identical to `BasicScheduler(simple, N)` at every count the
        shipped graphs run (`bench/check_pdd_sigmas.py` grades that against core's `calculate_sigmas`). The node
        emits SIGMAS so that scheduler and step count cannot be set apart from the grid its heads were fused for.
      * none -> `(None, None)`, and the caller decides whether that is a failure. It is for every current caller.

    A refine pass's `BasicScheduler` (`refine_scheduler_ids`) is skipped: its steps are extra audio-only evaluations.
    Reads an API graph (`{id: {"class_type", "inputs"}}`).

    Every value goes through `resolve_widget`, so a linked widget reads as the value behind it
    (`bench/check_graph_values.py` holds a synthetic graph that links `steps`). The resolver's states collapse to
    `None` here: a caller that must tell "this graph computes its step count" from "this graph is wrong" calls
    `resolve_widget` itself and reads `GraphValue.reason`.
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
