# Handover: read this before anything else

Rewritten 2026-10-28 at the close of the 2026-10-28 working day (`03:03Z` on 2026-10-29) by the Gemini lead (`gemlead`).
Where this page and a record disagree, the record is right: `git log`, [`decisions.md`](decisions.md), the lane rules in
[`../masked_v2v/AGENTS.md`](../masked_v2v/AGENTS.md), [`../masked_v2v/PROTOCOL.md`](../masked_v2v/PROTOCOL.md),
[`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md),
[`../masked_v2v/verifying_raw_frames_and_keeps.md`](../masked_v2v/verifying_raw_frames_and_keeps.md), and `data/fun/`.

## State when closed

- **Landed on `main` at close of session (`dc07ccee`, `c3b4c64d`, `fbafa9ef`)**:
  - **Branch `gemlead/cranium-track-and-marks` (`dc07ccee`)**: Replaced `CRANIUM_SCORE_FLOOR = 0.30` and `CRANIUM_FACE_OUTLIER = 2.50` in `bench/crop_reference_to_head.py::cranium_box` with 4-corner SAM 3.1 `head_track` containment (`CRANIUM_HEAD_TRACK_GROW = 0.50`) + `grey_head` bounded to `win[k]`, one active sizing median (`face`) used everywhere, and `CRANIUM_FACE_OUTLIER = 2.00` fallback only when no `head_track` is present (`0/110` records on disk). Enabled `head_track` discovery under `--cranium` in `bench/crop_reference_to_head.py::run` and `bench/head_reference.py::run`. Normalised `bench/head_reference.py::marks_against_the_body_model` by `min(frame_fh, span_fh)`. Added `bench/replay_cranium_references.py` writing stamped JSON tables under `data/<video>/`.
  - **Branch `gemlead/coverage-no-check` (`c3b4c64d`)**: Added `--no-check` (`no_check: bool = False`) table-only mode to `bench/frame_coverage.py` (`check_frame_coverage.py` 12/12).
  - **Branch `gemlead/shot-to-load-depth-check` (`fbafa9ef`)**: Added plan-level `depth` recipe check to `bench/check_shot_to_load.py` (`45/45`).
- **Renders kept and laid today (`data/fun/owner_answers.jsonl` & `data/fun/laid.jsonl`)**:
  - `77` laid rows in `data/fun/laid.jsonl` (`sha256 fc5cd74ee2a6603d`, `0` missing or changed bytes), including today's laid pieces: `lead_fp_4647d` (`4647–4725`, row 70), `lead_fp_0516f` (`516–558`, row 71), `lead_fp_0559` (`559–578`, row 72), `lead_fp_0579` (`579–599`, row 73), `lead_fp_3759c` (`3759–3908`, row 74), `lead_fp_2254c` (`2271–2598`, row 75), `lead_fp_3909c` (`3909–3966`, row 76), and `lead_fp_0294e` (`294–414`, row 77).
- **Saved replay & diagnostic scan files under `data/fun/`**:
  - `data/fun/cranium_replay_fun_2026-10-29T0302Z.json`: full 110-record 3-way replay (`pre-filter` vs `landed b7875bf1` vs `current dc07ccee`) with `writer_stamp` and per-record frame lists.
  - `data/fun/cranium_raw_vs_window_scan_fun_2026-10-29T0301Z.json`: raw vs. window-bounded `grey_head` edge comparison across all 110 cranium references.
  - `data/fun/cranium_crown_and_marks_scan_fun_2026-10-29T0301Z.json`: crown top-grow scan (`0.50..0.75`, `0.75..0.85`, `0.85..1.00`, `>1.00` face heights above `hy0`) and 3D `marks_against_the_body_model` scan across all 110 cranium references.

## What not to assume

- **Late-session verification status on `dc07ccee`, `c3b4c64d`, and `fbafa9ef` (landed at `03:02Z`)**:
  - **Verified before landing**: `check_crop_reference_to_head.py` (`9/9`), `check_head_reference.py` (`24/24`), `check_frame_coverage.py` (`12/12`), `check_shot_to_load.py` (`45/45`); 110-record replay on disk (`0/110` smaller than pre-filter by `>= 0.05x`, `0/110` fell to the scale fallback; `lead_5044_5113` goes from `1.111x (e=0)` pre-filter and `3.606x (e=9)` under `b7875bf1` to `2.798x (e=0)` under `dc07ccee`; `lead_0294_0414_b` is `2.674x (e=0)`); raw-vs-window contact sheets read on all 28 records where raw `grey_head` reaches the box edge (all raised arms, hands, shoulders, or upside-down legs except shallow crown clips of `<= 7 px` on `4696, 4705, 4940, 5045, 5060`).
  - **Not yet verified**:
    1. **Blind read completed (`dc07ccee`, `c3b4c64d`, `fbafa9ef`)**: Claude's blind read and `break_checks` runs passed (`3/3 SEEN`, controls green first). The four `gemlead/*` local branches cut during tonight's landing (`gemlead/cranium-track-and-marks`, `gemlead/coverage-no-check`, `gemlead/shot-to-load-depth-check`, `gemlead/handover-and-decisions-2026-10-28`) have been deleted after fast-forwarding `main`.
    2. **No H3 render yet under `dc07ccee`**: The first reference made under `dc07ccee` must have its contact sheet read before `first`.
    3. **Three jobs built and rendered earlier tonight under `b7875bf1` (`lead_fp_4272c`, `lead_fp_1134b`, `lead_fp_0438b`)**: All three rendered around `01:10Z`, landed, and are on the review page awaiting owner verdict (`4272c` rendered on `2.681x` vs `3.141x` under `dc07ccee`; `0438b` rendered on `1.724x` vs `2.169x` under `dc07ccee`; `1134b` rendered on `2.188x` vs `2.174x` under `dc07ccee`). The reference difference under `dc07ccee` matters only if the owner's look on the page says the head is small.
    4. **Measured follow-ups and open doc/tooling items not in `dc07ccee`** (numbers already saved in `data/fun/cranium_crown_and_marks_scan_fun_2026-10-29T0301Z.json` and `data/fun/cranium_raw_vs_window_scan_fun_2026-10-29T0301Z.json`):
       - Asymmetric top-grow above `hy0` (`0.75–0.85 * face` vs `0.50 * face` on the sides) to hold shallow 3D crowns where SAM's head track stops at the brow (`second_4647_4725_g` frame `4696` at `0.85 fh` / `7 px`, `lead_5044_5113` frames `5045, 5060` at `0.76, 0.82 fh`, `second_4912_4946_f` frame `4940` at `0.72 fh`), plus recording `seen & kept` fallback in `sizing_fallback` and updating the two `"no head track read"` strings in `bench/crop_reference_to_head.py`.
       - 3D body-model marks exclusion list (for flipped-mesh frames inside a valid head track where green is painted on legs, e.g. `lead_3476_3523` frame `3489` at `nose_from_body_model_fh = 4.32, eye_roll_diff_deg = 172.6` and `second_1093_1133_h` frame `1095` at `nose_from_body_model_fh = 12.95, eye_roll_diff_deg = 114.8`), replayed before landing since `0.50 fh` also fires on normal frames.
       - Build 3b (`laid.py` overlap refusal, `--trim`, sidecar stamp `laid.jsonl.stamp.json`, readers through `laid.read`, and `rebuild_delivery.py` snapshot) and Build 4 (generated `docs/masked_v2v/fun_status.md` via `bench/shot_coverage.py`, updating `shot_to_load.py` plan-depth record strings, and replacing remaining hardcoded share-path defaults / relative `output` fallbacks with a refusal when `H3_OUTPUT_DIR` is unset or when a tool is asked to create a directory under the repo root outside `data/`).

## Tomorrow, in order

1. **Owner review on the page for `lead_fp_4272c` (`4272–4309`), `lead_fp_1134b` (`1134–1155`), and `lead_fp_0438b` (`438–483`)**:
   - All three are rendered, landed, and on the review page (`01:10Z`). Lay the ones kept; remake `4272c` (`3.141x`) or `0438b` (`2.169x`) under `dc07ccee` only if not kept on the page.
2. **Build & render `lead_fp_4053d` (`4053–4091`, `39f`) as the first reference under `dc07ccee` (`1.958x`), reading its `_sheet.jpg` before `first`, followed by `lead_fp_5044b` (`5044–5113`, `2.798x` with `0` edge frames) and `lead_fp_6105e` (`6105–6394` once drops are written)**.
3. **Follow-up builds**: asymmetric crown top-grow + 3D-marks exclusion replay in `cranium_box`, `laid.py` overlap refusal / `--trim` / stamp lock, generated `docs/masked_v2v/fun_status.md` from `shot_coverage.py`, and refusing unset `H3_OUTPUT_DIR` / repo-root directory writes outside `data/`.

## Where things are

- **Start-of-day reading order** ([`../../AGENTS.md`](../../AGENTS.md)):
  1. [`handover.md`](handover.md) (this file) and [`decisions.md`](decisions.md) (`2026-10-28`).
  2. `data/fun/cranium_replay_fun_2026-10-29T0302Z.json`, `data/fun/cranium_raw_vs_window_scan_fun_2026-10-29T0301Z.json`, and `data/fun/cranium_crown_and_marks_scan_fun_2026-10-29T0301Z.json`.
  3. [`../masked_v2v/delivery_table_and_restore_architecture.md`](../masked_v2v/delivery_table_and_restore_architecture.md), [`../masked_v2v/chain_trace_and_remakes.md`](../masked_v2v/chain_trace_and_remakes.md), and [`../masked_v2v/reference_and_prompt_rules.md`](../masked_v2v/reference_and_prompt_rules.md).
- **Laid ledger & delivery table**: `data/fun/laid.jsonl` (`77` rows, `sha256 fc5cd74ee2a6603d`) and `data/fun/delivery_table_laid.txt`.
