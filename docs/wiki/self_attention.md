# Self-Attention in H3: How the 5-Way Joint Attention Works, What Probing Showed, and What to Explore Next

Last updated: 2026-11-27.

MiniMax H3 is not just a video model — it is a **5-way joint self-attention machine**. Inside every one of H3's
**50 DiT blocks (`56` heads each)**, five kinds of tokens are packed into one sequence (`comfy/ldm/minimax/model.py::PackedLayout`)
and attend to each other simultaneously with no cross-modal mask:

```text
[ text (Qwen3-VL) | keyframes (cond / cond_audio) | refs (<Picture N>, <Video N>, voice) | video patches (t, y, x) | stereo audio (L & R) ]
```

So far, we have built **block-band and stream-level self-attention routers across four corners** of that `5 × 5`
matrix — Camera vs. Mic (`QueryRouter`), Split-World / Portal / Multi-Shot (`SpatialQueryRouter`), Melody vs.
Vocal Timbre (`AudioRefRouter`, already using `+log(2)` pre-softmax boosting), and zero-SAM `video -> text` /
`video -> audio` self-masks (`SelfMaskRouter`). What our `2026-11-27` probing just opened up is going **inside
blocks to individual heads** (where `1` head out of `56` carries a signal that averaging buries) and extending
head-level / calibrated `±log(m)` controls to the remaining untouched corners (`Q_vid -> K_ref_img`,
`Q_vid(t) -> K_vid(t')`, and the `Qwen3-VL` ↔ `DiT` bridge).

This page gives the plain-English mental model for H3's joint self-attention, summarizes what our `2026-11-27`
self-mask probes showed, credits the outside LLM interpretability work (`SauersML`) on the other side of the
`Qwen3` coin, and maps both immediate mask follow-ups and broader non-masking self-attention explorations.

Every measurement for the self-mask campaign lives in the dated record
[`2026-11-27_self_mask_harvest_step_and_heads.md`](../../bench/results/2026-11-27_self_mask_harvest_step_and_heads.md);
nothing is restated as prose-only claims here.

**Quick links:** [sensor & router code (`steering/live_policy.py`)](../../steering/live_policy.py) ·
[spatial router (`steering/spatial_query_router.py`)](../../steering/spatial_query_router.py) ·
[query router firewall (`steering/hooks.py`)](../../steering/hooks.py) ·
[read-only probe (`bench/probe_h3_sensors.py`)](../../bench/probe_h3_sensors.py) ·
[full node reference (`self_steering.md`)](self_steering.md) ·
scripts and arrays: `data/self_mask_probe_2026-11-27/` ·
clips and sheets: `$H3_OUTPUT_DIR/Video/self_mask_late_harvest_2026-11-27/`

---

## 1. Plain-English Mental Model (How Zero-SAM Masking Works)

Two axes get mixed up easily if you come from LLMs or from video editing:

- **Outer loop — Denoising steps (`0..5` on 6-step Pika):** The model takes 6 passes from pure noise (`Step 0`)
  to a clean video (`Step 5`).
- **Inner stack — DiT layers/blocks (`0..49`, `56` heads per block):** Inside *each* denoising step, all tokens
  pass through 50 transformer blocks, just like layers in an LLM.

### Whose Attention This Is (Not Qwen's)

The attention we read belongs to **H3's DiT**, never Qwen3-VL:

- **Qwen3-VL runs once, before sampling starts.** It reads the raw prompt (with no chat template, plus vision
  tokens for any wired reference image/video) and hands the DiT one hidden state per token
  (core `comfy/text_encoders/minimax.py`; see [`steering.md`](steering.md) section 2). Its own attention is never
  read.
- **The DiT packs `[text | video | audio]` into one sequence**
  (`comfy/ldm/minimax/model.py::PackedLayout`, `MiniMaxH3Model._embed_and_pack`).
- **Every block runs one bidirectional self-attention over that whole sequence**
  (`comfy/ldm/minimax/model.py::Attention.forward`). Every `32×32` video patch (per 4-frame latent chunk) has a
  **Query** (`Q_vid`), and every prompt token has a **Key** (`K_txt`). Because all rows mix in every block, by
  blocks `35..48` a word row is no longer the raw vector Qwen wrote — it has already been rewritten by picture
  and sound context.
- **Qwen's tokenizer is borrowed once as a lookup table** to find which text-token indices spell the target
  phrase (e.g., `"woman"`). That is the only reason `MiniMaxH3SelfMaskRouter` takes a `clip` wire.

### The Two-Phase Loop (`7` Calls Instead of `6`, Zero Second Stage)

1. **Phase 1 — Peek on Step 0 (pure noise, unmodified model):**
   - `_make_harvest_attn_override` + `_Step0HarvestState.probe_block` (`steering/live_policy.py`) eavesdrop on
     `Q_vid` and `K_txt[target_phrase]` at selected blocks (shipped default: blocks `35 + 40`, all 56 heads
     averaged; experimental single-head option: **block `45`, head `43`**).
   - It computes the raw scaled dot product `Q_vid @ K_txt[noun]^T / sqrt(d)` — **pre-softmax scores, not
     post-softmax attention weights** (materializing full softmax over ~50k video+audio tokens on a 24 GB GPU is
     prohibitive).
   - `synthesize_4d_attention_mask` blurs that `[T, H, W]` score grid lightly in space and time, z-scores each
     latent frame against its own median and spread, runs a sigmoid around `threshold_z`, and zeroes out
     everything below `0.10`.
   - *(Speaker modes do the exact same read against `K_aud` in each frame's time window instead of `K_txt`.)*
2. **Phase 2 — Steer in the same forward generation:**
   - With `replay_step0=True`, we discard the throwaway unrouted Step-0 output and **replay Step 0 with routing
     turned on**, then run Steps `1..5` (`7` DiT calls instead of `6`, `1.17×` cost, no VAE round-trip).
   - `SpatialVisualQRRuntime` (`steering/spatial_query_router.py`) splits attention on early/mid isolated blocks
     (e.g., `early_0_24_shield`): video patches **outside** the mask attend to **Base Prompt (World A)**, while
     patches **inside** the mask attend to **Region Prompt (World B)** running as a parallel shadow text stream.
     On late blocks (`25..49`), both sides share the full video canvas again so lighting and edges merge.
   - For small targets (`< ~9%` of canvas, where `1×` routing washes out), leaving `region_conditioning` unwired
     hands `policy.harvested_mask_frames` straight to `2× MiniMaxH3EditGuider`.

### Our Shipped Self-Attention Levers in H3 (All Block/Stream-Level So Far)

| Mechanism | File | Corner of the `5 × 5` Matrix | What It Does |
| :--- | :--- | :--- | :--- |
| **Read-only sensor** (`MiniMaxH3SelfMaskRouter`) | [`steering/live_policy.py`](../../steering/live_policy.py) | `Q_vid -> K_txt`, `Q_vid -> K_aud` | Peeks at raw logits (`B35+40` or `B45 H43`) to synthesize a zero-SAM 4D mask |
| **Camera vs. Mic + Firewall** (`MiniMaxH3QueryRouter`) | [`steering/query_router.py`](../../steering/query_router.py), [`steering/hooks.py`](../../steering/hooks.py) | `Q_vid` vs `Q_aud -> K_txt`, `Q_aud -> K_vid` | Routes video and stereo audio to separate prompts/ears/time slices; attenuates `Q_aud -> K_vid` via pre-softmax bias (`_build_aud_attn_bias`) |
| **Split-World / Portal / Cuts** (`MiniMaxH3SpatialQueryRouter`) | [`steering/spatial_query_router.py`](../../steering/spatial_query_router.py) | `Q_vid_base` vs `Q_vid_region`, `Q_aud` (`binaural_split`, `follow_mask`) | Splits Base vs. Region video queries onto separate prompts on isolated blocks, merges on late blocks; supports 1-frame multi-shot cuts and visual-firewalled binaural ears |
| **Melody vs. Timbre Ref Router** (`MiniMaxH3AudioRefRouter`) | [`audio_ref_router.py`](../../audio_ref_router.py) | `Q_aud -> K_cond_audio`, `Q_aud -> K_ref_audio` | Early blocks follow melody scaffold; late blocks firewall melody (`-1e4`) and **boost** `<Audio 1>` vocal timbre via `+log(2)` pre-softmax side-SDPA |

*(By contrast, `MiniMaxH3Steer` shifts the text residual stream before attention, and `MiniMaxH3EditGuider`
blends outer velocity predictions `orig + scale * mask * (edit - orig)` rather than internal attention weights.)*

Nothing in H3 asks the model to introspect or report its own attention; we read and intervene from outside.

---

## 2. The Best Current Recipes (Moving Subject on 6-Step Pika)

### Small `< 9%` Moving Subject — Hold the Base Take Still (`2× EditGuider`, Verified `2026-11-27`)

When the subject is small (`~4%` of frame on the walk clip) and you want to **keep the base courtyard, framing,
and stride intact**, harvest `block 45, head 43` on Step 0 with `region_conditioning=None` and `dilation_passes=1`
(zero reharvest needed), then hand the mask straight to `2× MiniMaxH3EditGuider` (`edit_guidance.py::_effective_raw_mask`):

```python
routed, _pol = attach_self_mask_router(
    model, region_conditioning=None, mask_source="text_phrase", target_phrase="woman, crimson, coat",
    threshold_z=1.80, dilation_passes=1, replay_step0=False,
    harvest_blocks=[45], harvest_heads=[43], reharvest="off",
)
# -> MiniMaxH3EditGuider(routed, original=base_cond, edited=edit_cond, scale=1.95, first_step=1, last_step=3, streams="video_only", mask=None)
```

On the walk clip (`k2`, `87.5 s`, record sections 12–14 & 16), this transforms her head-to-toe across all `141f`
(median `0.000` coat-colored, `0/141` frames above `0.15`, `5.12×` in/bg MAE) while **holding the base layout,
framing, all `11` upper arches, chimneys, and walk pace** (`upper_courtyard_mae = 6.98` vs. `22.05` for `1×` `f1`
and `28.65` for mask-off `EditGuider`) with `reharvest="off"`. Owner playback accepted `k2` (footsteps automatically
re-sync to the taller chrome figure's longer strides, `10` steps at `13.7f` vs. `12` at `11.0f`, record section 14).
On the two-person clip (`p3_k2_paired_woman`, record sections 15 & 16), the same `B45 H43` mask keeps all `11` upper
arches intact (`upper_courtyard_mae = 6.24` vs. `18.36` mask-off, which collapses the upper floor to `6` wide arches),
while the paired prompt keeps the man yellow (`0.577`) — though because `B45 H43` masks both walkers, the man's coat
length and face still re-draw (section 15). It is **close, not pixel-identical**: small fixtures (lantern left of
center door → small barred window), fine stone texture, and figure height (`+44..64 px` taller chrome body) shift.
`dilation_passes=0` (`k1`, `5.58×` contrast) is tighter around the boots but leaves her hair/head partly human where
the taller head pokes above the undilated mask.

### `1×` Single-Pass Routing (Generates a New Take)

For `1×` routing without `EditGuider`, read `block 45, head 43` on Step 0 and once more during the first routed pass:

```python
attach_self_mask_router(
    model, region_conditioning=..., mask_source="text_phrase", target_phrase="woman, crimson, coat",
    replay_step0=True, harvest_blocks=[45], harvest_heads=[43], reharvest="first_routed_step",
)
```

On the walk clip (`f1`) this transforms the whole figure for the whole walk, where the shipped `B35 + B40` all-head
reading only caught head and shoulders for the first half (record, section 8), **but produces a different take of
the shot** (larger figure, altered courtyard details; record, section 11). Both `harvest_heads` and `reharvest`
live on `attach_self_mask_router` only and are off by default; the node UI and defaults stay unchanged until the
head index is checked beyond wide walking shots (section 5).

---

## 3. What Probing Showed (`2026-11-27`)

All of these findings come from one class of shot: one or two small figures crossing a locked-off wide frame.

### Depth vs. step: when you read matters more than how you clean the map

At Step 0 on the shipped blocks (`35 + 40`, all 56 heads averaged), the map for the target words does not follow
a moving subject (`covered = 0.344` early, `0.054` late). No word's map does. Two or three steps later, the same
words follow her on nearly every frame (record, section 4).

Our working hypothesis (not directly measured): **layer depth and denoising step trade against each other.** A
late block (`45`) on Step 0 already holds rough layout structure that middle blocks (`35 + 40`) only hold two
steps later.

### A later harvest gives a better mask and a weaker edit

Moving the harvest to Step 2 or 3 via `start_percent` puts the mask right on her for the whole walk, but starting
routing that late no longer changes her appearance (`d2` / `d3`, record section 5).

### One specialist head already knows at Step 0 — averaging buried it

At **block `45`, head `43` (of `56`)**, the moving subject is already cleanly marked at Step 0 (`covered = 0.952`
early / `0.925` late). Chosen on one clip, it then ranked **first of all 56 heads** across all 8 probed blocks on
three held-out clips: another seed, a man in a yellow raincoat walking right-to-left, and a child in a green
jacket in a gallery (record, section 6).

Averaging all 56 heads at block `45` yields almost no mask (`covered = 0.009`), which is why the earlier
calibration excluded block `45`. The signal was there the whole time inside one head, buried by the mean.

### Routing changes the take, and a second reading tracks the new trajectory

In routed renders, the walker ends farther along than in the plain clip — and close-up inspection showed that
**every routed render is a different take of the scene altogether**, not the base clip with only the subject
swapped (she is larger in frame and courtyard details differ; record, section 11). A mask frozen from the
unrouted first pass trails her late in the clip and she reverts.

Reading `block 45, head 43` **once more while the first routed pass runs** (`reharvest="first_routed_step"`,
zero extra steps) locks the mask onto the trajectory the routed render is actually producing (record, section 8).

### Two-stage head split on two people: generic "movers" head vs. word-binding heads

`B45 H43` turns out to be a **generic foreground-subject / mover detector**, not a word-specific detector:

- On a single walker, `B45 H43` finds her almost as well against **all text tokens** (`AUC 0.789`) as against her
  own words (`0.818`).
- With **two people** crossing (woman in crimson coat L→R, man in yellow raincoat R→L), `B45 H43` marks **both**
  people whichever person's words you query (`0.827 / 0.744` for her words; `0.833 / 0.855` for his).
- In the routed renders, **the region prompt — not the target phrase — picks who changes** (swapping direction in
  the region prompt swapped who changed; swapping the target words did not, though mismatched target words left
  bits of original clothing behind on close-ups; record, sections 9 & 11).
- **Separate word-disambiguating heads exist 1–3 blocks deeper** (`B46 H13` `0.914`, `B48 H3` `0.897`, `B48 H29`
  `0.885`, `B47 H17` `0.873` on `her - him`). They are poor at finding people on their own (`0.58..0.62`), and
  they were selected on that one clip only. Intersecting `B45 H43` ("is this a person?") with `B46 H13` ("her
  minus his") cleanly isolated her (`woman 0.549`, `man 0.000`) — a strong lead, not yet a validated result.

### Fixes that did not work

- **Subtracting a control map at Step 0 on `B35 + B40`:** made the mask worse (`covered 0.344 -> 0.139`); only
  helped slightly from Step 2 onward.
- **Normalizing each frame against the whole clip** instead of against itself: no measurable change.
- **Subtracting all-text mean on `B45 H43` in the two-person render (`h3`):** narrowed the mask, weakened the
  edit on her (`still red 0.010 -> 0.175`), and did not spare him more (`still yellow 0.486 -> 0.420`). Not
  landed.

### Which words point where

Once the maps settle (Step 2+), action and attribute words (`walks`, `footsteps`, `lone`, `AUC ~0.80`) track the
subject better than the subject noun (`woman`, `0.71`), while setting words (`arches`, `limestone`, `stones`,
`AUC 0.13..0.21`) actively point away from her (record, section 4).

---

## 4. Attribution & Relationship to Outside Work (`SauersML`)

Several of our probing habits and next-step experiments are directly adapted from open interpretability work by
**[@Sauers_](https://x.com/Sauers_) (`SauersML`)**, whose repos are kept locally for reference under
`coderef/sauers/`:

### `attention_introspection` ([github.com/SauersML/attention_introspection](https://github.com/SauersML/attention_introspection))

- **What Sauers studied:** Metacognitive **attention control** (`"Focus on [P2]"` / `"Completely ignore [P2]"`)
  and **attention introspection** (injecting `w += log(multiplier)` pre-softmax bias and asking whether Qwen3-32B
  notices) in a 64-layer causal text LLM. Mechanistically (`mechinterp/RESULT.md`), he found a two-stage circuit:
  **6 copier heads (layers 20–26)** copy the target label into answer queries (`query swap = 100%`, `key swap =
  0%`), feeding **reading heads (layers 24–36)** that shift attention up or down on the same heads.
- **Why our measurements differed from his:**
  - Sauers measures **post-softmax attention weights** (so every token competes in the softmax denominator against
    all other tokens) and compares **ratios / % changes against a neutral bare-mention baseline** (`"Note that one
    of the paragraphs is labelled [P2]."` — needed because merely naming a paragraph already raises attention
    `+10%`), rotated across all four paragraphs (`4×4` specificity check).
  - Our live H3 harvester reads **raw pre-softmax `Q_vid @ K_txt^T` logits** for a single phrase without rotating
    across competing subjects — which is why `B45 H43` firing generically on both people went unnoticed until we
    added a second subject.
- **What carried over cleanly to H3:**
  1. Expect a **tiny handful of specialist heads** to carry an effect that averaging hides (`B45 H43` out of 56).
  2. Validate selected heads against **random head sets of the same size** and held-out inputs.
  3. **What did *not* carry over directly:** simple control subtraction on Step-0 raw logits (`B35 + B40`) hurt,
     and H3's bidirectional `Q_vid <-> K_vid` cross-talk + `< 9%` small-mask washout mean soft pre-softmax bias
     cannot simply replace hard KV isolation for bold World A / World B swaps.
- **Shared instrument:** H3's Sight-to-Sound Firewall (`steering/hooks.py::_build_aud_attn_bias`) is the same
  additive pre-softmax bias (`w = w + bias`) Sauers uses for attention injection.

### Other `SauersML` Repos Worth Exploring Later

| Repo | Core Idea | Potential H3 Exploration |
| :--- | :--- | :--- |
| **[`SauersML/vlens`](https://github.com/SauersML/vlens)** | Feeds intermediate hidden-state layers of `Qwen3-VL` directly into an image DiT (Diffusion Lens) to visualize what the encoder holds before its final layer | Probe what H3 draws when conditioned on earlier `Qwen3-VL` cut layers (or steer at an intermediate `Qwen3-VL` layer instead of only the final cut) |
| **[`SauersML/weight-lens`](https://github.com/SauersML/weight-lens)** | Reads what a LoRA fine-tune pushes into residual directions from `ΔW = BA` alone on CPU (~18 s, no inference) | Inspect text-side / projector LoRA directions, or pair `ΔW = BA` SVD with our J-Lens unembedding probes on H3 adapters |
| **[`SauersML/body`](https://github.com/SauersML/body)** | Plain PyTorch residual-stream probes & steering of spatial/body-part maps and ownership (`"mine"` vs. place code) in Qwen3 | Methodology reference for separating orthogonal axes (e.g. spatial location vs. identity/attribute) in H3 text/DiT residual streams |
| **[`SauersML/neural-pairing`](https://github.com/SauersML/neural-pairing)** | Cross-attends two different models' hidden states mid-stack without exchanging tokens | Inspiration for latent-level cross-branch coupling (e.g. coupling two H3 worlds or a small auxiliary model mid-DiT without text round-trips) |

---

## 5. What to Try Next & Open Questions

### 5A. Immediate Self-Mask & Small-Subject Follow-Ups

1. **Completed (`2026-11-27`, record sections 12–16) — Hold the base layout on `< 9%` subjects + mask-off control (`B45 H43` SelfMask → `2× MiniMaxH3EditGuider`):**
   - Verified on both `c0_walk_base` (`k2`, owner playback accepted) and two-person `p3` (`p3_k2_paired_woman`,
     `p3_k2_single_chrome`), plus both **mask-off control twins** (`walk_mask_off`, `p3_paired_mask_off`, section 16):
     - Dropping the `B45 H43` mask collapsed the courtyard's `11` narrow upper arches into `5–6` wide arches even on
       the verbatim paired prompt (`upper_courtyard_mae`: `28.65` vs. `6.98` walk, `18.36` vs. `6.24` paired `p3`),
       proving the single-head zero-SAM mask is doing unmistakable architectural preservation work.
     - Because `B45 H43` masks both walkers on `p3`, omitting the man in `edited` (`p3_k2_single_chrome`) erases him
       completely, and keeping him in `edited` (`p3_k2_paired_woman`) still re-draws his face and coat hem (section 15).
2. **Next — Sweep 1A alone first: Rotation + text-only softmax on the `"who is who"` heads (`B46 H13`, `B48 H3`, `B48 H29`, `B47 H17`):**
   - Start directly from `data/self_mask_probe_2026-11-27/run_reharvest_two_subjects.py` and
     `analyze_two_people_heads.py` across new held-out multi-subject clips (pick heads on one set, score on
     another) before writing a broader probe library.
   - Compare **raw `target - other` logits** vs. **text-only softmax** across prompt tokens (and check whether
     pulling attention log-sum-exp `LSE` is needed to avoid dropping the video-token denominator).
   - Direct payoff from section 15: if word-binding heads generalize and zero the man out of the mask, `p3_k2_paired`
     can leave the untouched walker's original face and coat intact too.
3. **Wider shot survey before changing any node default:**
   - Test `B45 H43` (and sweep all `56` heads) on **still objects** (`MiniMax-H3-Image` `4 MP` bread on a table),
     **close-ups**, **moving cameras**, and **`ref2va`** — using offline `SAM 3.1` truth for static/close-up
     scoring (since background-median subtraction requires a moving subject on a locked camera).
   - If promoted to the node, prefer exposing raw optional `harvest_heads` / `reharvest` rather than baking a
     checkpoint-specific head index into a preset default.
4. **Non-direction region-prompt cues:**
   - Test a region prompt cue other than walking direction (color, wardrobe, size) to see how far "the region
     prompt picks who changes" extends when two subjects share a mask.

### 5B. Beyond Masking — Self-Attention Explorations on the DiT Side of the Coin

Because H3 runs `5`-way joint self-attention across all `50 × 56` heads, Sauers' head-level toolkit on `Qwen3`
maps cleanly onto non-masking corners of H3's attention matrix (run one sweep at a time with held-out truth, and
build **one** actuator in Phase 2 — whichever corner's heads beat random sets):

1. **Reference-Image Identity vs. Pose / Copy-Freeze Head Surgery (`Q_vid -> K_ref_img`):**
   - `MiniMaxH3ReleaseReferences` currently drops all heads across all blocks at once. Sweep `Q_vid -> K_ref_img`
     head-by-head to separate **face/identity heads** from **pose/background copy-freeze heads**, then attenuate
     only copy-freeze heads (or apply `-log(m)` on background ref tokens) while holding identity through the full
     clip — or blend two references (`<Picture 1>` face + `<Picture 2>` lighting/wardrobe).
2. **Selective Audio ↔ Video Sync Heads & Realized-Gain Calibration (`Q_aud <-> K_vid`):**
   - Calibrate our existing `QueryRouter` firewall (`aud_to_vid_gain`) by **realized attention mass ratio** (as in
     `coderef/sauers/attention_introspection/attn_intro.py:213-215`), not only nominal gain.
   - Sweep `Q_aud <-> K_vid` head-by-head to separate **lip-sync / foley-timing heads** from **visual-ambience /
     genre-bleed heads** — unlocking a selective firewall (swap music/atmosphere without breaking lip-sync),
     positive boosts (`+log(m)`) to tighten soft lip-sync or beat reactions, and L/R attention-driven stereo.
3. **Calibrated `±log(m)` Word & Clause Volume Knobs + Temporal Ramps (`Q_vid -> K_txt`):**
   - Extend the additive pre-softmax bias pattern (`steering/hooks.py::_build_aud_attn_bias`) to specific prompt
     words/clauses — turning `"rain"`, `"16mm grain"`, or `"slow motion"` smoothly up or down on the **exact same
     seed** (avoiding Qwen's pink-elephant effect on `"no fog"`), or ramping `m(t)` over time inside one shot.
   - Reading-side attention gain (distinct from the push-side multivector routing retired on `2026-11-24`, but
     confirm owner green-light before building); unlikely to replace hard KV isolation (`SpatialVisualQRRuntime`)
     for bold World A / World B swaps due to bidirectional `Q_vid <-> K_vid` cross-talk.
4. **Temporal Memory Heads vs. Local Motion Heads (`Q_vid(t) -> K_vid(t')`, later — needs fresh hooked pass):**
   - Saved probe arrays store video-to-word maps only, so measuring `t ± 1` vs. `t = 0` requires a dedicated
     in-hook pass. Separate short-range motion heads from long-range identity anchor heads to curb drift on
     `192f–345f` clips or enable same-character shot-reverse-shot cuts.
5. **Crossing the `Qwen3-VL` ↔ `DiT` Bridge (later — note layer-50 truncation):**
   - H3's `Qwen3-VL-32B` checkpoint is **truncated at layer 50** (`comfy/text_encoders/minimax.py`), so any
     Sauers-style `w += log(m)` boost inside `Qwen3-VL` must target layers `<= 50` (e.g., `21..49`, not `21..53`).
     Test whether Qwen-side boosts onto a subject clause or `<Picture 1>` vision tokens survive `TokenRefiner`
     into the rendered video, plus `vlens`-style earlier cut layers and `body`-style camera-vs-subject motion axes.
6. **50-Block Functional Head Atlas (Readable vs. Load-Bearing):**
   - Pair head sweeps with Sauers-style **causal head zeroing / patching vs. random head sets**
     (`coderef/sauers/attention_introspection/mechinterp/heads.py`) and **Query vs. Key swaps**
     (`coderef/sauers/attention_introspection/mechinterp/qk_all.py`) to classify `(block, head)` pairs into
     load-bearing levers vs. readout gauges across motion, identity, AV sync, and lexical binding.

Any node-code change is built in a worktree outside `custom_nodes` (root `AGENTS.md`, operative rule 3).

---

## 6. What This Does Not Show Yet

- **One kind of shot, one seed per clip, one render per arm.** No still subjects, no close-ups, no single-frame
  `MiniMax-H3-Image`, no `ref2va`. The two-person finding rests on one prompt.
- **The subject truth mask uses background subtraction** (difference from the per-pixel temporal median), which
  requires a locked-off camera and a moving subject, and includes cast shadows.
- **Raw pre-softmax scores, not full attention weights.** No full-sequence softmax over video + audio + text keys
  was computed.
- **The edits were judged on stills.** `compare_harvest_step_and_head_grid.mp4` and `two_people_grid.mp4` (under
  `$H3_OUTPUT_DIR/Video/self_mask_late_harvest_2026-11-27/`) and `k1`/`k2` (under
  `$H3_OUTPUT_DIR/Video/head_atlas_2026-11-27/`) still need human playback & audio review.
- **`1×` routing produces a new take; `2× EditGuider` (`k2`) held layout, not every pixel or the soundtrack.**
  Every `1×` routed walk arm changed framing and courtyard details; `B45 H43 -> 2× EditGuider` (`k2`, sections 12
  & 13) held the courtyard layout and stride on `c0_walk_base`, while small background fixtures (lantern left of
  center door → barred window), fine stone texture, and the soundtrack still shifted. Tested on one prompt/seed
  only, and the chrome figure is a high-contrast test pattern, not a tuned look.
- **In a mask strip, the green overlay is the thresholded mask, not the raw attention map.**

---

## 7. Run It Again

With the ComfyUI venv's python, from the repo root, after `set -a; . ./h3.env; set +a`:

- `data/self_mask_probe_2026-11-27/run_control_probe.py --repo . --out <folder>`: one plain render with a map per
  word, per step.
- `data/self_mask_probe_2026-11-27/run_late_harvest.py --repo . --out <folder> --clips <folder>`: per-head
  readings and the three late-harvest renders.
- `data/self_mask_probe_2026-11-27/run_head43.py --repo . --out <folder> --clips <folder>`: the three held-out
  clips and the first single-head render.
- `data/self_mask_probe_2026-11-27/run_reharvest_two_subjects.py --repo . --part a|b ...`: the second-reading
  renders, the two-person probe, and the two-person arms.
- `data/head_atlas_2026-11-27/run_phase0_edit_guider.py --clips <folder> --base-clip <c0> --f1-clip <f1>`:
  Phase 0 `B45 H43` SelfMask → `2× MiniMaxH3EditGuider` (`k1` `d0`, `k2` `d1`) renders, courtyard/ground metrics,
  and close-up sheets.
- `data/head_atlas_2026-11-27/run_phase0b_two_people_edit_guider.py --clips <folder> --base-clip <p3> --h2-clip <h2>`:
  Two-person `p3` through `k2` (`p3_k2_paired_woman`, `p3_k2_single_chrome`) and `two_people.py` scoring.
- `data/head_atlas_2026-11-27/run_mask_off_twins.py --clips <folder> --walk-base <c0> --walk-k2 <k2> --p3-base <p3> --p3-k2 <p3_k2>`:
  Mask-off `2× EditGuider` control twins (`walk_mask_off`, `p3_paired_mask_off`) and masked-vs-unmasked comparison sheets.

Each frees the card through `:8188` first and refuses when its queue is busy. The `analyze_*.py` files beside
them turn the arrays into the record's tables on the CPU.
