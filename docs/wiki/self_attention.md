# Self-Attention in H3: How It Works, What Probing Showed, and What to Try Next

Last updated: 2026-11-27.

H3's video model looks at the prompt, the picture, and the sound inside **one joint self-attention**. Our
self-mask ([`self_steering.md`](self_steering.md)) reads a tiny slice of that attention to find where a named
subject is — with **no SAM, no bounding boxes, and no extra models** — and then routes attention inside the same
generation to steer it.

This page gives the plain-English mental model, explains whose attention is being read, summarizes what probing
and close-up inspection showed, credits the outside interpretability work we borrowed ideas from, and lists the
next experiments and open questions. Every measurement lives in the dated record
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

### Our Three Self-Attention Levers in H3

| Mechanism | File | What It Does |
| :--- | :--- | :--- |
| **Read-only sensor** (`MiniMaxH3SelfMaskRouter`) | [`steering/live_policy.py`](../../steering/live_policy.py) | Peeks at raw `Q_vid @ K_txt` (or `Q_vid @ K_aud`) logits to synthesize a 4D mask |
| **Hard KV isolation** (`MiniMaxH3SpatialQueryRouter`) | [`steering/spatial_query_router.py`](../../steering/spatial_query_router.py) | Splits Base vs. Region queries onto separate prompt streams on isolated blocks, then merges on late blocks |
| **Additive pre-softmax bias** (`MiniMaxH3QueryRouter`) | [`steering/hooks.py`](../../steering/hooks.py) | Adds a negative pre-softmax bias (`_build_aud_attn_bias`) to block sight-to-sound leakage — the exact same mathematical lever used for attention injection in LLM studies |

Nothing in H3 asks the model to introspect or report its own attention; we read and intervene from outside.

---

## 2. The Best Current Recipe (Moving Subject on 6-Step Pika)

For a small moving subject crossing a wide shot, read **one specialist head at a late block** on Step 0, and read
it **one more time** during the first routed pass:

```python
attach_self_mask_router(
    model, region_conditioning=..., mask_source="text_phrase", target_phrase="woman, crimson, coat",
    replay_step0=True, harvest_blocks=[45], harvest_heads=[43], reharvest="first_routed_step",
)
```

On the walk clip this transforms the whole figure for the whole walk, where the shipped `B35 + B40` all-head
reading only caught head and shoulders for the first half (record, section 8). Both options live on
`attach_self_mask_router` only and are off by default; the node UI and defaults stay unchanged until the head
index is checked beyond wide walking shots (section 5).

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

Ordered from highest signal / cheapest to verify to broader architectural questions:

1. **Rotation check + text-only softmax on the `"who is who"` heads (`B46 H13`, `B48 H3`, `B48 H29`, `B47 H17`):**
   - Run Sauers' rotation design across `5–10` new multi-subject clips (two people, person + dog, car + cyclist):
     score every subject phrase against every competing subject phrase from the *same* Step-0 forward pass (zero
     extra renders).
   - Compare **raw `target - other` logits** vs. **text-only softmax** across prompt tokens (and check whether
     pulling attention log-sum-exp `LSE` is needed to avoid the missing-video-denominator trap).
   - Goal: establish whether a fixed small set of layer `46–48` heads reliably separates subjects by word across
     scenes, and test `B45 H43` ∩ `word-binding head` in live renders (with dilation or `2× EditGuider` handoff
     if the intersection covers `< 9%` of canvas).
2. **Wider shot survey before changing any node default:**
   - Test `B45 H43` (and sweep all `56` heads) on **still objects** (e.g., `MiniMax-H3-Image` `4 MP` bread on a
     table), **close-ups**, **moving cameras**, and **`ref2va`**.
   - Key open question: is `B45 H43` a *motion/mover* head or a general *foreground-subject* head?
3. **Causal head ablation & patching (`B45 H43` and `B46–48` vs. random head sets):**
   - Zero out (or patch between paired prompts) just `B45 H43` or the top word-binding heads, compared against
     random sets of the same size (Sauers `mechinterp/heads.py` pattern).
   - Key open question: are these heads **causal steering levers** inside the DiT, or clean **readout gauges**
     that only summarize state computed upstream?
4. **Query vs. Key swap between paired prompts:**
   - Swap `Q_vid` vs. `K_txt` across two prompts that swap subject attributes (Sauers `mechinterp/qk_all.py`
     pattern).
   - Key open question: does subject-word binding at blocks `45–48` live on the ~50 text tokens (cheap to
     intervene on) or the ~50k video tokens?
5. **Soft pre-softmax bias (`w += log(m)`) for in-prompt emphasis / suppression:**
   - Test adding a gentle `±log(m)` pre-softmax bias on specific `(video_patch, text_token)` or `(video_patch,
     ref_img_token)` pairs for words/refs *already in the prompt* (e.g., attenuating `"fog"` on a face, boosting
     `<Picture 1>` on a region) without hard-splitting KV.
   - Expected limit: unlikely to replace `SpatialVisualQRRuntime` (`early_0_24_shield`) for bold opposite-world
     swaps because bidirectional `Q_vid <-> K_vid` cross-talk still dominates.
6. **Holding the base take still (open question from section 11):**
   - Because every `1×` routed walk arm produced a different take of the scene, test **`SelfMask (B45 H43) -> 2×
     MiniMaxH3EditGuider`** (or `fl2va` / `lock_take` anchoring) against `1×` routing to see when a zero-SAM
     mask can edit a subject *without* changing the framing and background details.
7. **Non-direction region-prompt cues:**
   - Test a region prompt cue other than walking direction (color, wardrobe, size) to see how far "the region
     prompt picks who changes" extends when two subjects share a mask.

Any node-code change is built in a worktree outside `custom_nodes` (root `AGENTS.md`, operative rule 3).

---

## 6. What This Does Not Show Yet

- **One kind of shot, one seed per clip, one render per arm.** No still subjects, no close-ups, no single-frame
  `MiniMax-H3-Image`, no `ref2va`. The two-person finding rests on one prompt.
- **The subject truth mask uses background subtraction** (difference from the per-pixel temporal median), which
  requires a locked-off camera and a moving subject, and includes cast shadows.
- **Raw pre-softmax scores, not full attention weights.** No full-sequence softmax over video + audio + text keys
  was computed.
- **The edits were judged on stills.** `compare_harvest_step_and_head_grid.mp4` and `two_people_grid.mp4` in the
  clips folder stack the arms with their masks for playback.
- **Nothing here holds the base scene still.** Every `1×` routed clip is a new take of the shot, and the chrome
  figure is a high-contrast test pattern, not a tuned look.
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

Each frees the card through `:8188` first and refuses when its queue is busy. The `analyze_*.py` files beside
them turn the arrays into the record's tables on the CPU.
