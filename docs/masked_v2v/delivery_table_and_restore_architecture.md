# Delivery Table, `restore=` Spans, and Assembly Entanglement

 Companion to [`docs/masked_v2v/AGENTS.md`](AGENTS.md), [`docs/masked_v2v/PROTOCOL.md`](PROTOCOL.md), [`docs/masked_v2v/chain_trace_and_remakes.md`](chain_trace_and_remakes.md), and [`docs/research/masking/2026-10-26_code_map_and_simplification_plan.md`](../research/masking/2026-10-26_code_map_and_simplification_plan.md).

---

## 1. Root Cause of the `day_2026-10-27` Double-Sunglasses Bug (`lead_fp_3967c` & `lead_fp_2823b`)

When `fun_0000_6394_joined_2026-10-27T1421Z_from_laid.mp4` (`day_2026-10-27`) was assembled from `data/fun/laid.jsonl`, the owner spotted double-stacked sunglasses on `lead` in `3967–4035` (`lead_fp_3967c`).

### 1.1 What happened on `3967–4052` (`lead_fp_3967c`)
1. **In the H3 render (`lead_fp_3967c_00001.mp4`)**:
   - The prompt (`data/2026-10-25_lead_fp_3967c_job/lead_fp_3967c.render.txt`) includes `<Subject 1> wears white sunglasses.`
   - H3 painted clean white sunglasses with dark lenses directly onto the replaced face across `3970–4035`.
   - The per-shot review clip (`lead_fp_3967c_3967_4052_stacked.mp4`, built by [`bench/load_chain.py`](../../bench/load_chain.py) `land` via [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) `video`) displays `lead_fp_3967c_00001.mp4` directly — it **never** reads `data/fun/plan.json`'s `restore` spans or runs [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py).
2. **At whole-video delivery assembly (`bench/delivery_table.py` → `bench/assemble_delivery.py`)**:
   - `data/fun/plan.json` (`subjects.lead.restore`) still held a frame-span entry from `2026-10-14` / `2026-10-23`:
     ```json
     {
       "classes": ["Eyeglass"],
       "first": 3909,
       "last": 4052,
       "exception": "..."
     }
     ```
   - That entry was originally written for `lead_fp_3909b` (`3909–3966`) and the pre-remake shot `3967–4052` before `lead_fp_3967c` rendered its own sunglasses in H3.
   - Because `bench/delivery_table.py::own_restores` matched `plan["subjects"][label]["restore"]` purely by **frame span** (`3909–4052`) rather than by **load name**, it automatically emitted `restore=lead.Eyeglass` onto `lead_fp_3967c`'s delivery row (`3970–4035`).
   - During whole-video assembly, `bench/assemble_delivery.py::worn_of` (`WORN_ON_THE_HEAD = frozenset({"Eyeglass", "Apparel"})`) cut the 1980s source actress's translucent sunglasses (with her original eyes and blue eyeshadow visible through the lenses) out of the source plate and pasted them on top of `lead_fp_3967c_00001.mp4`'s H3-generated sunglasses. Worse, on the 18 frames where Sapiens missed `Eyeglass` on the source plate (`3993–4002`, `4025–4035`), the pasted source glasses vanished and revealed H3's sunglasses underneath, creating a double-glasses flicker.

### 1.2 Join-Test Audit of All Self-Accessory `restore=lead.*` Rows (`3967c`, `2823b`, `3098b`, `3909b`, `0168c`)
Across all 73 rows of `data/fun/delivery_table_laid.txt`, only 6 rows carried self-accessory `restore=lead.<Class>` tokens from `data/fun/plan.json`. Running span join tests with [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) both **with** and **without** `restore=` across `3909–4052` and `2823–3109` established the exact behavior on each clip:

1. **`3970–4035` (`lead_fp_3967c`, `restore=lead.Eyeglass`) — Double Sunglasses WITH `restore=`, Clean WITHOUT `restore=`**:
   - `lead_fp_3967c_00001.mp4` already has clean H3-rendered white sunglasses across `3970–4035`. Removing `restore=lead.Eyeglass` from `lead_fp_3967c` leaves a single crisp pair of white sunglasses on every frame.
2. **`3098–3109` (`lead_fp_3098b`, `restore=lead.Eyeglass` and `restore=lead.Left_Hand+Right_Hand`) — Double Top Rim WITH `restore=`, Clean WITHOUT `restore=`**:
   - In `data/2026-10-23_lead_fp_3098b/manifest.json` (`L98–106`, `L969`), `part_is_made_of` included `"Eyeglass"` (`94.8%–100%` of `Eyeglass` lies inside `matte > 127`), the prompt explicitly said `<Subject 1> wears white sunglasses`, and the 39-frame window included 27 unmasked context frames (`3071–3097`) where she wears the white winged sunglasses.
   - Consequently, `lead_fp_3098b_00001.mp4` **already has the white sunglasses rendered on her face** all the way until she leaves the frame at `3109`.
   - Applying `restore=lead.Eyeglass` (`bench/assemble_delivery.py::worn_of` / `worn_moved`) pasted a face-shifted copy of the source sunglasses on top of `lead_fp_3098b_00001.mp4`, creating a **double top rim** on `3099–3101` and `3104–3108` (an earlier visual check had misread those frames before the A/B join test). Assembled **without** `restore=`, `lead_fp_3098b` has one clean pair of sunglasses aligned with her face through `3109`.
3. **`3060–3062` & `3063–3097` (`lead_fp_2823b`, `restore=lead.Left_Hand+Right_Hand` and `restore=lead.Eyeglass`) — Double Ear-Hook/Edge WITH `restore=`, Clean WITHOUT `restore=`**:
   - In `data/2026-10-15_lead_fp_2823b/manifest.json` (`L703–711`, `L840`), `part_is_made_of` included `"Eyeglass"`, the prompt explicitly said `<Subject 1> sings to the camera in a close-up, then raises a pair of white sunglasses with both hands and puts them on`, and `lead_fp_2823b`'s reconstructed `"only what changed"` matte (`lead_fp_2823b_00001.matte_reconstructed.json`, built `2026-10-17`) already preserves the sunglasses and hands across all 38 frames `3060–3097`.
   - Historical context: the `3060–3109` `restore` entries in `data/fun/plan.json` were added on `2026-10-16` *before* `lead_fp_2823b`'s own matte was reconstructed on `2026-10-17` (when delivery still laid `2823b` by whole `region`) and before `9efd2f2b` (`2026-10-23`) changed `Eyeglass` restore to shift by `(dy, dx)`.
   - With `9efd2f2b`'s face-shifted `restore=lead.Eyeglass` active on `lead_fp_2823b`, `bench/assemble_delivery.py` pasted the source sunglasses on top of `lead_fp_2823b_00001.mp4`'s existing sunglasses, producing a **double ear-hook** on `3068` and a **double top edge** on `3072`. Assembled **without** `restore=`, every frame `3060–3097` has a single clean pair of sunglasses and clean hands.
   - **Earrings (`2823–3097`)**: The `Apparel@head` restore was removed on `2026-10-16` (`docs/wiki/decisions/2026-10-16_to_2026-10-17.md:331-338`). Because `57%–79%` of her dangling earrings hang below the earlobe outside `matte` (`matte == 0`, untouched source video) and `lead_fp_2823b` uses `"composite": "only what changed"`, the earrings remain aligned with her ears and face across `2823–3097` without any `restore=`.
4. **`3909–3966` (`lead_fp_3909b`, `restore=lead.Eyeglass`) — Needed**:
   - In `lead_fp_3909b_00001.mp4`, the prompt did not mention sunglasses and H3 rendered her side profile on the stairs bare-faced; `restore=lead.Eyeglass` places her sunglasses onto her face.
5. **`168–184` (`lead_fp_0168c`, `restore=lead.Apparel@head`) — Needed**:
   - In `lead_fp_0168c_00001.mp4`, H3 redrew the top of her head without her black hat; `restore=lead.Apparel@head` restores the hat shifted to `lead_fp_0168c`'s face height.

### 1.3 Physical Rule: When Accessories Are Rendered by H3 vs. Restored from Source
The join tests across `0168c`, `2823b`, `3098b`, `3909b`, and `3967c` establish three distinct regimes for accessories:

| Regime | Mask Coverage (`matte`) | Prompt / Conditioning | What Happens in H3 Render | Should `restore=` Be Used? | Examples |
|---|---|---|---|---|---|
| **1. Inside mask + in prompt/context** | Accessory class (`Eyeglass`) is wired into `part_is_made_of` (`matte > 127`) | Prompt names the accessory (`wears white sunglasses`) and/or unmasked context frames show it | H3 generates the accessory directly on the new face (or `only what changed` preserves it seamlessly) | **No** — applying `restore=` pastes a second shifted copy from the source plate on top of the render (double rims/hooks) | `lead_fp_3967c` (`3967–4052`), `lead_fp_2823b` (`2823–3097`), `lead_fp_3098b` (`3098–3109`) |
| **2. Inside mask + omitted by H3** | Accessory lies inside the regenerated head/face region (`matte > 127`) | Prompt omitted the accessory (or H3 redrew the crown bare without the hat) | Accessory is missing on the rendered head/face | **Yes (clip-specific exception)** — `restore=` copies the accessory from the source plate shifted by `(dy, dx)` to the rendered face | `lead_fp_3909b` (`3909–3966`, `Eyeglass`), `lead_fp_0168c` (`168–184`, `Apparel@head` hat) |
| **3. Outside mask** | Accessory hangs outside the face/neck mask (`matte == 0` on `57%–79%` of pixels) | Not needed in prompt | Source video pixels outside `matte` remain untouched and attach naturally at the feathered ear/neck boundary | **No** — `Apparel@head` grow margins pull in collar/shoulder patches | `lead_fp_2823b` dangling earrings (`2823–3097`) |

---

## 2. Why `delivery_table.py` and `assemble_delivery.py` Became Entangled

Currently, [`bench/delivery_table.py`](../../bench/delivery_table.py) (`898` lines) and [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) (`3,354` lines) do **four distinct jobs** in one global whole-video pass at the very end of the pipeline:

```mermaid
flowchart TD
    subgraph Current["Current Entangled Whole-Video Pass (6,395 frames)"]
        P["data/fun/plan.json (frame-span restore rules)"] --> DT["bench/delivery_table.py"]
        L["data/fun/laid.jsonl"] --> DT
        DT -->|"emits restore=<other> + restore=<self>.<Class>"| AD["bench/assemble_delivery.py (3,354 lines)"]
        AD --> J1["Job 1: Per-clip self-accessory patching (worn_Moved, Pose shift, Sapiens class map)"]
        AD --> J2["Job 2: Multi-subject pass compositing (Matte.at, keep-out restore=<other>)"]
        AD --> J3["Job 3: Whole-video timeline splicing (6,395 frames + audio mux)"]
        AD --> J4["Job 4: Pixel-diff QA & anomaly flagging (changed_outside_region, two_pieces_change_the_same_pixels)"]
    end
```

### The two completely different meanings of `restore=`
Because [`bench/delivery_table.py`](../../bench/delivery_table.py) emits both kinds of `restore=` onto the same table row, two unrelated operations share one keyword:
1. **Cross-subject protection (`restore=second`, `restore=third:whole`)**:
   - When `lead` is laid on a shot where `second` is also present, `restore=second` prevents `lead`'s rectangular zoom box from overwriting `second`.
   - Note that since `2026-10-20`, `bench/assemble_delivery.py::Matte.at` (`L1372–1474`) already composites each piece using the ComfyUI node's saved per-frame soft `matte` (`*_window_*_region.npz["matte"]`), which has `second` and `strangers` carved out during the load itself.
2. **Self-accessory source give-back (`restore=lead.Eyeglass`, `restore=lead.Apparel@head`, `restore=lead.Left_Hand+Right_Hand`)**:
   - This is a **single-clip post-processing patch** for an older render where H3 failed to draw a hat or sunglasses.
   - Running this inside [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) across the 4.5-minute (`6,395`-frame) joined video has two major architectural flaws:
     - **Review vs. Delivery mismatch**: The owner reviews and clicks `"keep this one"` on `<load>_<first>_<last>_stacked.mp4` (`bench/load_chain.py` `land`), which is built **without** `bench/assemble_delivery.py`. Any self-accessory `restore=` only appears later when the whole-video cut is assembled!
     - **Silent leakage across remakes**: Because `plan.json` keyed `restore` by frame span (`first`, `last`), remaking a shot or an adjacent shot inside that span silently applied the legacy source-paste on top of the new render.

---

## 3. Step 1 (Landed `187dc9fd`): Pinning `plan.json` `restore` Spans to Explicit `loads`

To stop silent leakage immediately without breaking existing kept clips (`lead_fp_0168c`, `lead_fp_2823b`, `lead_fp_3098b`, `lead_fp_3909b`) or requiring re-renders:
1. **`bench/delivery_table.py::own_restores` (`L308–355`) and `decide` (`L523`)**:
   - Supports `"loads": ["<load_name>", ...]` (validated as a non-empty list of non-empty strings by both `bench/delivery_table.py::own_restores` and `bench/shot_coverage.py::load_plan`).
   - When `"loads"` is present on a `plan["subjects"][label]["restore"]` entry, `own_restores(label, r["name"])` emits the `restore=` token **only** when `r["name"] in x["loads"]`.
2. **`bench/shot_to_load.py::load_job` (`L3496–3500`)**:
   - Filters `job["restore"]` by `not r.get("loads") or name in r["loads"]`, so a new remake on those frames never inherits `--over-the-eyes` or `--given-back` flags from an older load's exception.
3. **`data/fun/plan.json` (`subjects.lead.restore`)**:
   - Every active exception in `data/fun/plan.json` is now explicitly pinned to the exact kept load(s) it was approved for:
     - `3909–3966` (`Eyeglass`): `"loads": ["lead_fp_3909b"]` (removed from `3967–4052` / `lead_fp_3967c`).
     - `3063–3109` (`Eyeglass`) and `3060–3109` (`Left_Hand`, `Right_Hand`): `"loads": ["lead_fp_2823b", "lead_fp_3098b"]`.
     - `168–184` (`Apparel` `@head`): `"loads": ["lead_fp_0168c"]`.

---

## 4. Long-Term Architecture: Clip-Level Restore at `land` Time & Decoupled Delivery Splicing

Every operation in `bench/assemble_delivery.py` is **100% frame-local** to the shot being replaced (verified across `bench/assemble_delivery.py`: `Matte.at` at `L1372`, `restore_weight` at `L1340`, `worn_of` at `L706`, and `worn_fill` at `L655`, which looks at most `WORN_FILL_FRAMES = 2` frames left/right within the same capture). Nothing in `restore=` requires the 4.5-minute timeline.

The clean long-term separation splits the four jobs into their natural pipeline stages:

```mermaid
flowchart LR
    subgraph Stage4["Stage 4: Per-Load Land (bench/load_chain.py land)"]
        R["Raw H3 Render (<load>_00001.mp4)"] --> CR["1. Clip-Level Composite / Optional Restore (on 15-90f shot only)"]
        CR --> SC["2. Review Stack (<load>_stacked.mp4 shown on :8765)"]
        CR --> LP["3. Final Shot Piece (<load>_final.mp4)"]
    end
    subgraph Stage5["Stage 5: Whole-Video Splice (bench/rebuild_delivery.py)"]
        LP --> LL["laid.jsonl (sha256-verified shot pieces)"]
        LL --> SP["Pure Frame Splicer (drops finished pieces onto source timeline + muxes audio)"]
    end
```

### 4.1 Why Clip-Level Restore at `land` Time Wins
1. **What you review on `:8765` is byte-for-byte what goes into the joined video**:
   - If a clip uses a clip-level restore or matte composite, it runs **once** at `bench/load_chain.py` `land` time on that 15–90 frame clip before `<load>_<first>_<last>_stacked.mp4` is written.
   - The owner sees the exact final pixels on `:8765` before clicking `"keep this one"`, eliminating any surprise difference between the review clip and the joined delivery cut.
2. **Scoped to the single load (`job.json`), never global**:
   - Any exception is requested on that specific load and baked into that load's landed output artifact. A remake (`second_hfp_4912l`, `lead_fp_3967c`) starts clean with zero risk of inheriting a global frame-span rule.
3. **Fast, simple whole-video splicing**:
   - Once each kept load's landed clip already has its matte and any per-clip restore applied (and Pass 2 loads are rendered on the Pass 1 composite cut per [`AGENTS.md`](AGENTS.md) §1), whole-video assembly reduces to a pure timeline splice that drops the kept shot clips onto the source timeline and muxes the audio track in seconds.
