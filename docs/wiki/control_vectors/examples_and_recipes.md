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
  --target-prompt my_prompt.txt \
  --words "old wording that occurs once in the prompt -> new wording" \
  --bank-scenes 3 \
  --ref-images my_reference.jpg \
  --families ref2va --name my_dial --overwrite
```

The old wording must occur once in `my_prompt.txt` (or use `summary:`, `soundscape:`, or `music:` section
prefixes). `--bank-scenes 3` appends 3 deterministic mode-matched scenes from `prompt_bank/*.txt` (weighted
`1.0` behind `Pair 0`'s `--target-weight 2.5`) so single-scene nuisance context averages out.

## 3. Verified `Arms 01..66` Pika & `ref2va` recipes

From [`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md):

- **Single-lever visual look on `ref2va`:** Apply `film_16mm_1974` (`mode = add_norm_matched`, `strength = 1.0..1.3`, `mask = scene_only` or `look_only`) on `workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json` (`PIKA_SOL_CUDA_DEFAULTS`: `0.20..0.95` keeps Step 0 and Step 5 dense).
- **Stacked dual-lever visual + motion/camera:** Chain two `MiniMaxH3ApplyControlVectors` nodes (`workflows/h3_text_to_video_pika_dual_control_vectors_api.json`), e.g. `look_only` on the look vector and `description_only` on the action/motion vector, or use `mode = hold` on the second node to project out an unwanted nuisance direction.
- **Audio-only steering without touching picture look:** Set `mask = soundscape_only` (steers `overall_soundscape:` and non-`"N/A"` `non_diegetic_music:`, leaving visual sections untouched).
- **Zero-`.toml` CPU word synthesis (`< 2 ms` via J-Lens, `Arms 53, 54, 61..65`):**
  ```bash
  <python> bench/control_vectors.py words "blizzard, snowdrift, freezing, snow" --name blizzard_words --families fl2va
  ```
- **CLI activation & DiT grading under any 3-lever mask:**
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
`last_step = 2` or `3` on 6-step Pika hands late steps to 1-pass `[edited]` (`8–9` DiT passes vs `12`).

## Not rendered in a recorded benchmark arm

Building from two pictures (`make-media`), `sculpt`, inline J-Lens `hold_words` (vector-on-vector
`mode = "hold"` *was* rendered in `Arm 58`), and the two in-graph build nodes (`MiniMaxH3MakeControlVector`
and `MiniMaxH3MakeMediaControlVector`, whose underlying build helpers are exercised via the CLI and
`bench/check_control_vectors.py`).
