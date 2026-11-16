# Bucket 4 Verification — Mass-Conserving `return_lse=True` Subset Attention Balance & Dual-Backend Parity

Date: 2026-11-16
Model: Pika H3 6-step (`MINIMAX_H3_MODEL_PIKA` + `PH3_VIDEO_VAE` + `MINIMAX_H3_ENCODER`)
Resolution / Grid: `1344x768`, `context_frames = 39` (`c = 12` latent steps), `context_noise = 0.6`, `context_profile = "seam_taper"`, `section_anchor = "per_section"`, `anchor_end_percent = 0.33`, `seed = 730451892`
Attention Chains:
- `sage + Sol`: [`MiniMaxH3SageAttention`](../../nodes.py#L229)`(mode="auto", head_chunks=1)` + [`MiniMaxH3Sol`](../../sol_attn_h3.py)`(dense_blocks="49", quantizer="rotated", sink_conditioning="exact_kv_and_all_rows", start_percent=0.2)`
- `kitchen + Sol`: `ModelAttentionBackend("comfy kitchen attention")` + `MiniMaxH3Sol(dense_blocks="49", ...)` + [`sol_observe`](../../sol_observe.py)

Script: [`bench/probe_bucket4_lse_balance.py`](../probe_bucket4_lse_balance.py)
Unit Suites: [`bench/check_lse_balance.py`](../check_lse_balance.py) (`6/6` CPU + CUDA cases), [`bench/check_audio_freeze.py`](../check_audio_freeze.py#L1594) (`check_anchor_lse_balance_and_gain_mode`)
Summary & Outputs: `data/bucket4_probe_2026-11-16/probe_summary.json`, `$H3_OUTPUT_DIR/Video/bucket4_probe_2026-11-16/` (zero media committed to `data/`)

---

## Code Changes Verified in Bucket 4

1. **Mass-conserving `return_lse=True` subset balance engine ([`attention.py`](../../attention.py#L394), [`lse_subset_balance_bhnd`](../../attention.py#L589)):**
   - [`resolve_lse_balance_plan`](../../attention.py#L394): resolves disjoint subset spans $A$, $B$ (or single-subset complement $\bar A$ when `spans_b` is unset) and target `query_spans` from explicit spans, `PackedLayout` segment kinds (`kinds_a`, `kinds_b`, `query_kinds`), or `0`-based `ref_img` indices (`ref_indices_a`, `ref_indices_b`); checks active DiT `blocks` and `[sigma_start, sigma_end]`; enforces `require_layout=True` by default so token-refiner calls are untouched; and refuses overlapping $A \cap B$.
   - [`lse_subset_balance_bhnd`](../../attention.py#L589): computes an exact within-slice mass-conserving convex combination between $O_A$ and $O_B$ via exact flash/sub-span log-sum-exp statistics $\ell_A, \ell_B$ so $P_{\text{new}}(A \cup B) = P(A \cup B)$ and every token outside $A \cup B$ (text, audio, references) retains its exact original attention share. Uses `sage-fork` `sageattn_consume(..., return_lse=True)` whenever [`MiniMaxH3SageAttention`](../../nodes.py#L229) is active on CUDA (preserving `clone_v` early QKV freeing), and pure PyTorch `torch.ops.aten._scaled_dot_product_flash_attention` (`D <= 256` `bf16`/`fp16` CUDA, `fp64`/`fp32` exact fallback on CPU) otherwise.
   - [`install_lse_balance_override`](../../attention.py#L809): installs `h3_kernel = "lse_balance"` idempotently beneath `MiniMaxH3Sol` in either node order by invoking `"h3_sol"` `ON_PREPARE_STATE` callbacks at patch time.
   - **Broader ref/AV surface ([`attach_lse_reference_balance`](../../attention.py#L841), [`MiniMaxH3AudioAttentionGain`](../../audio_freeze.py#L756)):**
     - `MiniMaxH3AudioAttentionGain` appends optional `mode = "kv_scale" | "lse_balance"` ([`audio_freeze.py`](../../audio_freeze.py#L752), [`L827`](../../audio_freeze.py#L827)), which works on **any** joint AV H3 workflow (both frozen audio and standard `t2va`/`fl2va`/`ref2va`) without requiring `MiniMaxH3SageAttention`.
     - [`attach_lse_reference_balance`](../../attention.py#L841) provides the reusable `attention.py` API to balance `<Picture N>` / `<Video N>` `ref_img` slices (`ref_indices_a`, `ref_indices_b`) in `ref2va` workflows (`anchor_weight` and `MiniMaxH3AudioAttentionGain` expose the two UI surfaces currently shipped).
2. **`MiniMaxH3Sol` routing & provenance composition ([`sol_attn_h3.py`](../../sol_attn_h3.py#L767), [`_describe_override`](../../sol_attn_h3.py#L1097)):**
   - Gate in `_decide_and_run` sits immediately before `dense_blocks`: whenever `resolve_lse_balance_plan(options, tokens)` is active on the current `(block, sigma)`, Sol delegates cleanly to the dense balance override and logs its own `route("lse_balance", f"block {block} in lse_balance (w={lse_plan['weight']:.3g})")` label (verified in [`bench/check_sol_node_equivalence.py`](../check_sol_node_equivalence.py) CPU & GPU and [`bench/check_provenance_stamp.py`](../check_provenance_stamp.py)).
3. **`MiniMaxH3AudioFreezeSong` anchor knobs ([`audio_freeze_song.py`](../../audio_freeze_song.py#L687), [`L1080`](../../audio_freeze_song.py#L1080)):**
   - Appended optional `anchor_weight` (`default = 1.0`, `[0.1, 16.0]`, never compared against `0`) and `anchor_blocks` (`default = "30-49"`), attached inside [`attach_section_anchor_release`](../../audio_freeze_song.py#L687) between `spans_a = section_anchor` (`1,008` tokens) and `spans_b = carried context video` (`12,096` tokens for `39` frames) over target generated video queries while `frac_step < anchor_end_percent`.
4. **Unit & kernel parity coverage ([`bench/check_lse_balance.py`](../check_lse_balance.py)):**
   - Verified `6/6` cases across CPU and CUDA: disjoint 2-subset and single-subset complement balance within `< 1e-13` of explicit `float64` softmax re-weighting, bitwise `0.0` preservation on non-target query rows, `< 1e-14` invariance outside $A \cup B$, composition beneath `MiniMaxH3Sol` and [`provenance.py`](../../provenance.py) in either patch order, invalid-span/overlap refusals, and live CUDA `D = 128` FlashAttention-2 + `sage-fork` `return_lse=True` parity.

---

## Section 1 — 2-Window Studio LSE Weight & Dual-Backend Sweep (`243 f = 10.125 s`)

Single `studio` section ([`prompt_bank/t2va_studio_dancer_close.txt`](../../prompt_bank/t2va_studio_dancer_close.txt)), `window_frames = 141`, `context_frames = 39`, `seam_taper` at `context_noise = 0.6`, 120 BPM drum track (`rms_dbfs = -22.06`, `peak_dbfs = -0.10`), alongside Bucket 2 unanchored (`b2_taper_off_ref`) and Bucket 3 unweighted (`b3_2win_rel033`) references. All means reduced in `float64`.

| Arm (`backend`, `w`, `blocks`) | Wall Time (s) | `W1 -> W2` Studio Luma (`% Delta`) | Seam `L1` (`f140 -> f141`) | Seam Jump Ratio | Seam Luma Jump | `W2` Motion `L1` | `W2` Sharpness (`Lap Var`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `b2_taper_off_ref` (`sage`, `off`) | `132.14` | `140.170 -> 126.341` (`-9.87%`) | `3.0608` | `0.6725` | `0.6294` | `6.6940` | `52.73` |
| `b3_2win_rel033` (`sage`, `w=1.0`) | `133.15` | `140.577 -> 128.175` (`-8.82%`) | `2.8223` | `0.5302` | `0.1900` | `7.2808` | `42.62` |
| `b4_2win_w4_b30_49` (`sage`, `w=4.0`, `30-49`) | `148.89` | `140.577 -> 128.572` (**`-8.54%`**) | `2.8768` | `0.5813` | `0.1738` | `7.4203` | `47.39` |
| `b4_2win_w8_b30_49` (`sage`, `w=8.0`, `30-49`) | `148.73` | `140.577 -> 127.609` (`-9.22%`) | **`2.6157`** | **`0.5081`** | **`0.1195`** | **`7.6336`** | **`47.87`** |
| `b4_2win_kitchen_w4` (`kitchen`, `w=4.0`, `30-49`) | `191.01` | `141.159 -> 129.672` (**`-8.14%`**) | `3.2508` | `0.7341` | `0.2151` | `6.1569` | `44.39` |

- **Why `w = 4.0` improves over `w = 1.0`:** At `1344x768`, `section_anchor` occupies `1` latent step (`1,008` tokens) against `12` carried context steps (`12,096` tokens, a `12:1` cardinality ratio). Under unweighted `w = 1.0` (where anchor vs. context average comparable raw logits), the anchor receives only `~7.7%` of the anchor+context slice. Raising `anchor_weight` to `4.0` raises its relative slice allocation to `~25%` across blocks `30–49` through step fraction `0.33`, lifting `W2` studio luma (`-8.82% -> -8.54%`, surpassing even Bucket 3's `1.00` full-step hold at `128.300`) while raising `W2` motion (`7.4203` vs `7.2808`) and sharpness (`47.39` vs `42.62`). At `w = 8.0`, the `f140 -> f141` seam discontinuity drops further (`seam_l1 = 2.6157`, `seam_jump_ratio = 0.5081`, `seam_luma_jump = 0.1195`) with strong motion (`7.6336`), though overall `W2` wall luma dips slightly (`127.609`) relative to `w = 4.0` (`128.572`).
- **Dual-backend independence (`kitchen` vs. `sage`):** `b4_2win_kitchen_w4` (`ModelAttentionBackend("comfy kitchen attention") + MiniMaxH3Sol`, `42.1 s` slower than `sage-fork` `148.89 s`) ran `lse_balance` natively on pure PyTorch FlashAttention-2 with zero `MiniMaxH3SageAttention` or `sage-fork` dependency, recording `-8.14%` `W2 vs W1` luma retention.

### Visual Findings (`bucket4_2win_stacked.mp4` & `bucket4_2win_stacked.1fps.00000-00242.mp4`)

- **Identical Sage Window 1 (`f000..f140`):** Panels 1–3 (`w=1.0`, `w=4.0`, `w=8.0`) share identical `W1` pixels (`140.577`); Panel 4 (`kitchen,w4`) diverges cleanly by backend (`141.159`, brisk opening shoulders and torso twist across `f073..f140`).
- **Mid-`W2` framing & facial sharpness (`f148..f180`):** At `w = 1.0` (left), the dancer turns far into side profile across `f162..f176`, whereas `w = 4.0` keeps a centered 3/4 frontal punch cadence matching the `W1` studio portrait, with visibly firmer jaw/eye contrast (`sharpness = 47.39` vs `42.62`). Panel 4 (`kitchen,w4`) opens the mouth cleanly for `6` frames around `f152..f157` before closing again across `f160..f242`.

---

## Section 2 — Live `sol_observe` Route Accounting (`b4_2win_kitchen_w4`)

Arming [`sol_observe`](../../sol_observe.py) (`raw=off`) on `b4_2win_kitchen_w4` recorded all `600` attention invocations across `2` windows x `6` steps x `50` DiT blocks (`token_refiner` stays excluded by `require_layout=True`):

| Window | Active `lse_balance` Spec | `lse_balance` | `sol` | `outside_range` | `dense_block` | Total Calls | Observed `lse_balance` Blocks |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Window 1 (`f0..140`) | Inactive (no anchor) | `0` | `245` | `49` (Step 0, `0..48`) | `6` (Block `49`, Steps `0..5`) | `300` | `—` |
| Window 2 (`f141..242`) | Steps `0–1`, Blocks `30–49` (`w=4.0`) | **`40`** | `226` | `30` (Step 0, `0..29`) | `4` (Block `49`, Steps `2..5`) | `300` | `[30..49]` (`20` blocks x `2` steps) |
| **Combined** | — | **`40`** | **`471`** | **`79`** | **`10`** | **`600`** | `[30..49]` |

- Because the `lse_balance` gate executes right ahead of `dense_blocks` in [`sol_attn_h3.py`](../../sol_attn_h3.py#L767), all `20` blocks in `[30..49]` on Window 2 Steps `0` and `1` (including Step 0 outside Sol's `start_percent=0.2` and Block `49`) report cleanly under their own `lse_balance` route label rather than conflating into `sol` or `dense_block`.

---

## Section 3 — 4-Window 2-Section Timeline (`447 f = 18.625 s`) & Default `anchor_weight = 1.0`

Same 2-section schedule (`--- studio` -> `--- stage` at `f243`, `4 x 141 f`, `context_frames = 39`, `seam_taper` at `0.6`) comparing Bucket 3 `off` and `w = 1.0` vs. Bucket 4 `w = 4.0`:

| Arm (`section_anchor`, `w`, `blocks`) | Wall Time (s) | `W1` Studio Luma | `W2` Studio Luma (`W2 vs W1`) | `W3` Stage Luma (`B - R`) | `W4` Stage Luma (`B - R`) | `W4 vs W3` Drift | `Seam 1->2` (`L1 / Ratio`) | `Seam 3->4` (`L1 / Luma Jump`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `b3_4win_off` (`off`, `—`) | `276.29` | `141.604` | `130.612` (`-7.76%`) | `29.223` (`+32.99`) | `50.251` (`+35.94`) | **`+71.96%`** | `—` | `—` |
| `b3_4win_per_section` (`w=1.0`, `—`) | `267.94` | `141.604` | `130.959` (`-7.52%`) | `39.312` (`+32.21`) | `45.306` (`+33.52`) | **`+15.25%`** | `3.5435` / `0.8273` | `2.8268` / `0.1983` |
| `b4_4win_per_sec_w4` (`w=4.0`, `30-49`) | `280.70` | `141.604` | `130.853` (`-7.59%`) | `39.363` (`+30.77`) | `53.682` (`+35.38`) | `+36.38%` | **`3.1654` / `0.6185`** | `2.9179` / **`0.1335`** |

### Visual Findings (`bucket4_4win_stacked.mp4` & `bucket4_4win_stacked.1fps.00000-00446.mp4`) & Default Choice

- **Improved intra-section seams under `w = 4.0`:** Both same-room seams (`W1 -> W2` studio: `3.1654 / 0.6185` vs `3.5435 / 0.8273`; `W3 -> W4` stage: `seam_luma_jump = 0.1335` vs `0.1983`, crisp centered spotlight posture and profile pose) tighten under `w = 4.0`, and lips remain closed across `W4` (`f345..f446`, note a brief `f350..356` subtle expression shift).
- **Why `anchor_weight` defaults to `1.0` (not `4.0`) on `MiniMaxH3AudioFreezeSong`:** Across an abrupt bright-studio -> dark-stage `--- label` switch, Window 3 still carries `39` studio context frames locked cleanly at `c - 2, c - 1`, so its first `8` post-trim frames `f243..f250` stay inside the rehearsal studio (`b3_4win_per_section` per-frame luma `f243 = 130.92`, `f250 = 130.34 -> f251 = 12.77`; `b4_4win_per_sec_w4` `f243 = 99.12`, `f250 = 102.90 -> f251 = 13.90`) before cutting to the dark blue spotlight at `f251`. Because `per_section` captures `images[:1]` (`f243`) as the `stage` anchor, overweighting that transitional bright studio frame at `w = 4.0` pulls `W4` stage spotlight highlights brighter (`45.306 -> 53.682`, `+36.38%` vs `+15.25%` at `w = 1.0`). Keeping `anchor_weight = 1.0` as the shipped node default keeps cross-room timeline cuts conservative while making `w = 4.0..8.0` opt-in for single-environment song loops.
