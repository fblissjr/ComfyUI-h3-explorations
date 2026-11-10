# Worked examples

last updated: 2026-11-10

[Back to the start page](../control_vectors.md)

Only examples a session has run are here, each with where its result is. "The plan" is
[`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md),
and the full 66-arm Pika `t2v` / `ref2va` benchmark sweep (`Arms 01..66`) is recorded in
[`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md).

`<python>` below is the ComfyUI venv's python (`docs/comfy_notes.md`). A build needs the card free.

## 1. Build a vector from a pair file in the bank

```bash
<python> bench/control_vectors.py build prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite
```

It writes one file per family into the models folder's `control_vectors/` and prints, for each, the
recommended strength for the node's default mode (`add_norm_matched`). To build the warmth-scrubbed neutral
variant alongside it, pass `prompt_bank/control_vectors/film_16mm_1974_neutral.toml`.

## 2. Build a vector from your own prompt (with optional `--bank-scenes` cancellation)

```bash
<python> bench/control_vectors.py from-prompt \
  --target-prompt-file my_prompt.txt \
  --words "old wording that occurs once in the prompt -> new wording" \
  --bank-scenes 3 \
  --ref-images my_reference.jpg \
  --families ref2va --name my_dial --overwrite
```

The old wording must occur once in `my_prompt.txt` (or use `summary:`, `soundscape:`, or `music:` section
prefixes; pass `--target-prompt "<inline text>"` when supplying literal prompt text instead of a file).
`--bank-scenes 3` appends 3 deterministic mode-matched scenes from `prompt_bank/*.txt` (weighted `1.0`
behind `Pair 0`'s `--target-weight 2.5`) so single-scene nuisance context averages out.

## 3. Verified `Arms 01..66` Pika & `ref2va` recipes (2026-11-05)

From [`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)
(run under `mask = non_vision` unless noted):

- **Single-vector visual look on `ref2va` (`Arms 17..23`):** Apply `film_16mm_1974` (`mode = add_norm_matched`, `strength = 1.0..1.3`) on `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json` (with `start_percent > 0` in `Arm 23` for late-window entry).
- **Stacked dual-vector composition & orthogonal holds (`Arms 08..12, 58`):** Chain two `MiniMaxH3ApplyControlVectors` nodes (`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`), or set `mode = hold` on the second node (`Arm 58`: `cursed_cctv` + `hold(chaos_panic)`).
- **Section-scoped description or audio masks (`Arms 59..60`):** `mask = soundscape_only` (`Arm 59`) or `mask = description_only` (`Arm 60`).
- **Zero-`.toml` CPU word synthesis (`< 2 ms` via J-Lens, `Arms 53, 54, 61..65`):**
  ```bash
  <python> bench/control_vectors.py words "blizzard, snowdrift, freezing, snow" --name blizzard_words --families fl2va
  ```

### 2026-11-10 3-lever span masks (`h3_spans.py`)

Added on 2026-11-10 (`comfyui_nodes_and_controls.md`; same-seed mask comparison against `non_vision` is open):

- **3-lever masks on `ref2va`:** `look_only` (`subject_definitions:` + `summary:`), `description_only` (`detailed_description:`), `scene_only` (`look_only | description_only`, excluding audio sections), and `soundscape_only` (`overall_soundscape:` + non-`"N/A"` `non_diegetic_music:` + `<Audio N>:` retention lines). On text-only `t2va`, `look_only`, `description_only`, and `scene_only` select the same `integrated_multimodal_description:` prose rows.
- **CLI activation & DiT grading under any mask:**
  ```bash
  <python> bench/control_vectors.py grade \
    --vector film_16mm_1974 --prompt t2va_covered_market_long \
    --mask scene_only --pika --sol
  ```

## 4. Compare a vector with an edit on one shot

`bench/edit_guidance_arms.py` takes a shipped graph, a prompt and one edit, and builds a set of renders
that differ in one thing each: the prompt as written, the edited prompt, edit guidance, and with
`--vector-arm` the original prompt under a control vector.

```bash
<python> bench/edit_guidance_arms.py \
  --base workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json \
  --prompt-file my_prompt.txt --ref-image my_reference.jpg \
  --edit "old wording -> new wording" \
  --name my_shot --out data/results/my_round --seeds 1 2 \
  --node-arms --only base rewrite node_s1.5 bank \
  --vector-arm "bank=<a vector file>:1.0"
```

It prints both prompts in full and writes the graphs. Add `--queue` to render them. A third field on
`--vector-arm` sets the mask (e.g. `look_only`, `scene_only`, `description_only`, `soundscape_only`).

## 5. Steer with an edited prompt and no vector (`last_step` fast handoff)

The two generated graphs and the nodes are on [`../edit_guidance.md`](../edit_guidance.md); setting
`last_step = 2` or `3` on 6-step Pika hands late steps to 1-pass `[edited]` (`9–10` DiT passes vs `12`; `8`
passes at `last_step = 1`).

## Not rendered in a recorded benchmark arm

Building from two pictures (`make-media`), `sculpt`, inline J-Lens `hold_words` (vector-on-vector
`mode = "hold"` *was* rendered in `Arm 58`), and the two in-graph build nodes (`MiniMaxH3MakeControlVector`
and `MiniMaxH3MakeMediaControlVector`, whose underlying build helpers are exercised via the CLI and
`bench/check_control_vectors.py`).
