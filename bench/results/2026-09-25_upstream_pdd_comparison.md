# PDD across upstream implementations: head fusion, head selection, audio carry, 2026-09-25

The numeric half of the 2026-09-25 PDD survey. The comparison and what it
means for decisions live in
`docs/research/pdd/2026-09-25_upstream_pdd_comparison.md`. Every
implementation below is reimplemented from its source in a probe under
`bench/`, and none is imported. All three probes are CPU-only; run them with
`CUDA_VISIBLE_DEVICES=""` in the ComfyUI venv.

## 1. Fused-head fidelity — `bench/compare_pdd_head_fusion.py`

The reference is an fp64 fusion of alibaba-pai's raw FL2VA head stack: the
dt-weighted block mean over the shifted 32-interval grid, per stream, at video
shift 12 and audio shift 3. Two metrics:

- `rel_corr`: the error against the distilled correction, ‖W − W_ref‖ /
  ‖W_ref − W_base‖;
- `rel_out`: the same on the head output for a fixed random input.

Each cell is the worst block of the partition. Full output:
`2026-09-25_pdd_head_fusion.json`.

| implementation | u8 video / audio rel_corr | u4 video / audio rel_corr |
|---|---|---|
| ours (fp64 fuse, fp32 master) | 4.79e-6 / 6.78e-6 | 6.54e-6 / 7.84e-6 |
| vendor as it runs (fp32 heads and plan) | 7.69e-6 / 1.14e-5 | 1.28e-5 / 1.67e-5 |
| UtilsCollection (fp64 then fp32) | same as ours | same as ours |
| core `_pdd_head` on exact fp32 deltas (formula control) | 4.79e-6 / 6.78e-6 | 6.55e-6 / 7.85e-6 |
| core on Kijai's **current** upstream file (HF `f94b1bcc94`) | 4.79e-6 / 6.78e-6 | 6.55e-6 / 7.85e-6 |
| core on T8's bf16 deltas (T8's native-core route) | 6.12e-3 / 3.88e-3 | 5.68e-3 / 3.44e-3 |
| sglang as shipped (bf16 plan, bf16 storage) | 0.325 / 0.406 | 0.370 / 0.389 |
| T8 fallback (bf16 plan, einsum and hidden) | 0.325 / 0.406 (rel_out 0.62 / 0.75) | not reachable (8-step only) |
| **core on the local Kijai copy in this install** | **228 / 331** | **282 / 381** |

Reading it:

- Ours, the vendor, UtilsCollection and core with correct deltas all sit at
  the fp32 floor, so none of them has a precision to gain.
- sglang's bf16 plan weights no longer sum to one, which makes a systematic
  scale error, and storing in bf16 adds to it. The probe splits the two:
  "plan-rounding only" and "storage only".
- The local Kijai copy decodes under merged core to a head roughly doubled.
  That file is the pre-re-upload one: its bank rows are absolute heads, and
  merged core adds each row to head 0. Local sha256 `1bee03dfc1…` (FL2VA) and
  `c0efad0058…` (Ref2VA pruned) do not match upstream's current LFS oids,
  `71347e2725…` and `6f18e1c2ec…` (HF API, 2026-09-25). No shipped graph names
  any `Acc-8Step` file.

**Inputs.**
- The vendor stack is under
  `coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs/`. That HF repository's newest
  commit is 2026-08-27, a README change.
- Ours is `h3_config.PDD_FL2VA_LORA`.
- Kijai's current file is range-fetched: its safetensors header plus the byte
  ranges holding the `final_layer` tensors. They go in a directory named by
  `H3_KIJAI_PDD_UPSTREAM_DIR`, as `header.json`, `part1.bin` and `part2.bin`.
  The probe's `P2OFF` is the data offset where the second range begins.

## 2. Head selection — `bench/compare_pdd_head_selection.py`

The probe compares two selectors:
- core's: argmin against `sample_sigmas`, the next sigma, inverse shift to
  base-grid indices, and one block for both streams;
- ours: each stream's own `t_emb` row against the boundary embeddings, from
  `adaln_t_table` on `h3_config.MODELS["unet_fl2va"]`.

The cases cover every schedule the PDD graphs ran on 2026-09-25, plus
synthetic ones. Full output: `2026-09-25_pdd_head_selection.json`.

- **Shipped, no audio mask:** uniform 8, uniform 4, and the ManualSigmas
  tail6 graph. The selectors agree on every call, and our off-schedule
  warning never fires.
- **Shipped, with an audio mask:**
  - The three song graphs run `audio_mask` 0.25 through
    `MiniMaxH3AudioFreezeSong`. They disagree on 7 of 8 calls.
  - The masked audio row sits at `1 − 0.25·σa` for the whole render, so ours
    picks the final audio block every step. Our off-schedule warning would
    fire, once per render (`pdd_lora.py` guards it with `warned`). That is
    inferred from the probe; no retained server log covers a song-graph run.
  - Core would give the audio the video's block, spans the row never visits.
  - Video agrees.
  - The three candidate freeze graphs run `audio_mask` 0.0. Their heads also
    differ there, but the mask multiplies the audio velocity by zero, so it
    has no effect.
- **Synthetic cases:**
  - SplitSigmasDenoise halves, BasicScheduler at `denoise` 0.3, 0.5 and
    0.75, simple 5, 6 and 16, beta 8, heun, and a duplicated call per step:
    all agree.
  - kl_optimal 8 disagrees on one step.
  - dpm_2's midpoint calls make core build odd spans.
  - At simple 64, our audio runs one interval ahead of video on four steps.
    Core's shared, shift-invariant index is the more robust of the two there.

**Re-derive the shipped list before trusting it.** Walk
`h3_config.graph_paths(..., include_bench=True)`. For each graph with a
`MiniMaxH3PDDLoRA`, read its `steps`, its sampler, any `ManualSigmas`, and
any freeze node's `audio_mask`.

## 3. The audio carry is exact under Euler — `bench/compare_pdd_audio_carry.py`

Both integrators are driven with the same velocity at every block start:
- core's single-schedule carry (its `MiniMaxH3Model.forward` transform and
  the final unscale in `MiniMaxH3.process_latent_out`);
- the vendor's two-schedule audio Euler.

Full output: `2026-09-25_pdd_audio_carry.json`.

| partition | u8 | u4 | tail6 | opt4 | u2 | u1 | u32 |
|---|---|---|---|---|---|---|---|
| max final audio difference | 1.1e-15 | 8.9e-16 | 1.3e-15 | 8.9e-16 | 8.9e-16 | 4.4e-16 | 2.7e-15 |

The negative control moves core's B coefficient to the block end, or to the
block mean. That gives 0.36 / 0.62 and 0.16 / 0.28 at u8 / u4, so the harness
does see a transform error when there is one.

**Why it holds.**
- With shifts 12 and 3, the carry factor σv/σa equals `s + (1 − s)·σv`
  with s = 4 (checked to 1.3e-15).
- The carried audio is therefore `s(1 − σv)·x0 + σv·ε`, a straight flow
  path in σv.
- Core's velocity transform, evaluated at the block start, is the chain rule
  along that path. So one Euler step lands where the vendor's audio step
  lands, at any block width.

**Scope.** Euler at eta 0, the model evaluated at the scheduled sigmas,
unmasked audio. Core multiplies the velocity by `audio_denoise_mask` before
the transform, and that case is not simulated.

**What this bounds.** `bench/results/2026-08-28_audio_carry_ablation.json`
re-applied the transform at the block's mean sigma and measured audio energy
rise at every partition. That observation stands. It is now a departure from
the vendor's integrator that adds energy, not the correction of an error:
evaluating B at the block mean is exactly what the negative control shows to
be off the exact path. `bench/measure_pdd_audio_carry.py` compares against a
block mean of the transform, not against the vendor's integrator, so its
"applied − exact" gap is not an error of core's.
