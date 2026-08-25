# Point handoff for 2026-08-26

**What this is:** the brief for whoever runs technical point on the Qwen3-VL
AWQ v2 lane on 2026-08-26. Not canonical: it says where things are and what
comes next, and defers to `canonical/` for every claim. Written 2026-08-25
evening by the outgoing point session ("v2-lead"); the section at the end is
filled in after the calibration run lands.

## Read first, in this order

1. `CLAUDE.md` at the repo root, then `docs/checks.md` if you are about to
   change behaviour.
2. `canonical/README.md`, then `canonical/active_plan.md` (gates, locked
   decisions, roles, stop conditions).
3. `canonical/2026-08-25_v2_launch_record.md`: what launched, why the first
   launch died, the host budget, the split, the swap, and **the pre-registered
   Gate 5 bar**. Grade the candidate against that bar and nothing else.
4. `canonical/2026-08-25_gate2_arrangement.md` for the storage and kernel
   decisions the run used.

## Authority

The owner approves; point decides the technical calls and labels acceptances
**ACTING-POINT DECISION** in canonical with the reading they rest on. Codex is
not returning; its accepted records stand and are not reopened without a
measured reason. Evidence is labelled MEASURED / SOURCE / INFERENCE / UNKNOWN.
A number goes into prose only where it changes the next action.

## Standing rules that bit today

- **Stop ComfyUI by its port owner before any bridge load** (`ss -ltnp` on
  8188, kill that pid, not the wrapper); restart from the ComfyUI install's
  `start.sh default` afterwards. Owner-approved standing; no need to ask.
- **Host memory bounds the calibration population, not the card**: AWQ costs
  about 430 KB of host per population token on top of resident weights. Price
  a population before launching it; run with the weights on the bridge's disk
  tier (`--host-reserve-gib 114`) and under a cgroup cap
  (`systemd-run --user --scope -p MemoryMax=115G`). A 128 GB swap file exists
  as the safety net; it is not a budget.
- **The shared index**: peers share this checkout. `git add` and `git commit`
  by explicit pathspec only; never `uv.lock`, never gitignored paths. Never
  push.
- **Venvs**: `coderef/llm-compressor/.venv/bin/python` for anything touching
  llm-compressor or the bridge; the ComfyUI install's venv python for anything
  importing `comfy` or building bundles.
- `pgrep -f` matches your own shell if the pattern is in your command line;
  use the `[p]attern` form.
- No machine-local absolute paths in repo content (a hook blocks them), no
  emojis, never the owner's real name.

## Where the working set is

Everything the sprint needs that is not committed lives in
`internal/v2_run_2026-08-25/` (gitignored, durable, on the root NVMe):

- `bundle_v2_Bp2/`: the calibration population that ran (29 rows, 214,187
  tokens, mixed geometry).
- `bundle_v2_holdout_v2b_up2048/`, `bundle_v2_holdout_v2b_noup/`: the rebuilt
  13-row holdout, same rows, both still geometries.
- `bundle_t2va_holdout/`: the 13-row text-only regression population.
- `selections/`: every selector output, including the superseded ones.
- `launchers/`: the launch, probe and relaunch scripts and their logs, for
  the record; they carry the point session's scratchpad paths and are not
  meant to be rerun as they are.
- `run_gate5.sh`: the Gate 5 driver with repo-relative paths (convert,
  acceptance check, six captures, per-row compare, summary).
- `candidate_probe_disk/`: a 2-layer packed candidate in the real v2 format,
  a fixture for the convert path.

Candidate directory:
`coderef/llm-compressor/models/qwen3-vl-32b-h3-w4a16-awq-v2-nativecal`.
Run record: `bench/results/2026-08-25_v2_calibration_run.json`. The deployed
artifact, its symlink and the source checkpoint are untouched.

## Peers on 2026-08-25, by name

Sessions may or may not survive the night; the names and what each owns:

- **mr_sparkles**: the v2 adapter/conversion/config-snapshot path
  (`bench/convert_h3_awq_candidate.py`, `config/` snapshots), the marker-arm
  nodes and the provenance stamp's `encoder_arm` block
  (`docs/research/marker_arm_binding.md`), and the checkpoint/resume module
  for the pilot, built and proven at fixture scale, then parked by owner
  decision (item 6 below).
- **mr_data**: the calibration-set review harness
  (`bench/review_v2_calibration_bundle.py`), the near-duplicate adjudication
  and the corrected family map
  (`bench/results/2026-08-25_pool_component_map_corrected.json`), the marker
  corpus (`bench/marker_corpus/`), the Gate 6 render population
  (stratified by upscale factor, one null-control row), the T2VA bundle.
  **Read the family map's own `caveat` field before relying on its count.**
  Added after this handoff was written: the pool's videos were reviewed and
  turned up two rows that are the same shot list rendered twice at more than
  twice the perceptual threshold, so same-brief-different-render relatedness is
  real in this source and the window cannot reach it. The count is a floor on
  relatedness, not a measurement of it, and the image population is too large
  to inspect the way the nineteen videos were.
- **performance_and_refs**: the reference nodes, `qwen_short_edge`, the three
  ablation graphs `workflows/h3_probe_refview_{a_source,b_qwen2048,c_parity}`
  with `bench/gate6_refview_arms.json`, the conditioning-cache design note
  (`docs/research/conditioning_cache.md`), and a queued occupancy render that
  waits for the card.

If a session is gone, a fresh one takes the role with its files; the records
above are what it needs.

## Card order, and what runs next

1. **Gate 5** (`internal/v2_run_2026-08-25/run_gate5.sh all`, ComfyUI
   stopped first): converts the candidate to
   `qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors` under the ComfyUI
   install's `models/text_encoders/` with snapshot
   `config/qwen3vl_32b_minimax_h3_w4a16_awq_v2/` (commit that directory; it
   cannot exist before the real candidate), runs the acceptance check, then
   captures BF16 / v1 / v2 on all three bundles in one session and grades
   per row. Output `bench/results/2026-08-25_v2_holdout_layer50.json`. The
   comparator's `--w4-path` / `--all-rows` and the summarizer are new today
   and get their first real run here; if the sequence-length guard trips,
   that is an instrument fault, not a verdict.
2. Grade against the bar in the launch record. Accept: v2 into the ablation
   graphs (the encoder combo already names the v2 file). Reject: the
   population or recipe is the suspect; do not rename a partial result.
3. performance_and_refs reruns `bench/preflight_graph.py` on the three arms
   under the resolved v2 contract, patches mr_data's rows in, and prices
   them; then their occupancy render. **Server reserve: do not change
   `start.sh default` on a guess.** The Gate 5 comparator needed
   `--reserve-vram-gib 16` for a 21.7k-token BF16 row, but that is a
   standalone process; the server unloads the encoder before the DiT runs.
   If the occupancy instrument shows the encode step OOM on arm B or C
   (about 18.7k encoder tokens on the W4 encoder), launch those graphs with
   `./start.sh default --reserve-vram <GiB>`; if it does not, the default
   stands.
4. **Gate 6 blind pairs**: `bench/run_graph_arms.py --manifest
   bench/gate6_refview_arms.json` with matched seeds, then
   `bench/blind_batch.py`, the scoring app, `bench/score_session.py`. The
   process is `docs/eval_comparison.md` section 3; the owner judges. The
   arms differ only where a still's short edge is below 2048; the null
   control row must read as identical.
5. **Marker arms on v2**: the corpus (`bench/marker_corpus/compiled.json`)
   rendered through `MiniMaxH3MarkerArm` with the stamp's `encoder_arm`
   block as evidence; the arms are release IDs, legacy BPE, stripped, and
   mean-initialised rows (a patched clone, never written). Training is off
   the table unless the ID arm loses and re-init does not help.
6. **Checkpoint/resume: parked by owner decision at 17:00 on 2026-08-25.**
   The module, design note and check exist (`bench/h3_calibration_checkpoint.py`,
   `bench/check_calibration_checkpoint.py`, proven bit-identical at fixture
   scale on CPU); the pilot integration and the real-weight card proof are
   deliberately not done. Bring it back only for a run longer than tonight's
   (the larger population after the modifier-cache change), at a cadence of
   every 4 layers.

Deferred, named in the plan: a larger calibration population needs the
modifier's parent-argument cache off host memory (a change to
`AWQModifier`), not a bigger box.

## Filled in after the run

- **The run landed at 16:59:54** (exit 0, 3 h 22 min, host peak 82 GiB,
  controls green); candidate 19.01 GiB in 14 files; run record committed
  (acc4fd5). Converted to the single file and its snapshot
  (`config/qwen3vl_32b_minimax_h3_w4a16_awq_v2/`, committed fcda1f7);
  acceptance check green on every case except the fixture case, which
  compares the real candidate against the *smoke* snapshot by design and is
  red on any real candidate.
- **Gate 5: reject against the pre-registered bar.** v2's median relative L2
  against BF16 is above v1's on both geometries (0.333 vs 0.312 at 2048,
  0.367 vs 0.359 at no-upscale), v2 wins only 8 and 6 of 13 rows, its worst
  rows are better than v1's, text is marginally better. Numbers, the
  per-row reading and the suspects are in the launch record's Gate 5
  section; the record is `bench/results/2026-08-25_v2_holdout_layer50.json`
  and the per-row captures are under `internal/v2_run_2026-08-25/gate5_*`.
- **v2 does not replace v1.** It stays on disk and in its snapshot; the
  deployed artifact is unchanged. It can carry the ablation as the encoder
  that accepts the 2048 view, which v1 clamps.
- **Four-encoder table done 18:45 (`bench/results/2026-08-25_four_encoders_holdout_layer50.json`):
  INT8 ConvRot is ~15x closer to BF16 than either W4 artifact on every
  population; NVFP4 between. Owner decision: INT8 is the encoder for the
  ablation and the marker arms. First thing tomorrow: switch the three
  ablation graphs' encoder combo to `qwen3vl_32b_minimax_h3_int8_convrot`
  in `workflows/build_workflows.py` (h3_config's encoder constant), rebuild,
  keep `check_attention_defaults` and `check_model_files` green, rerun
  preflight, then the occupancy render to price INT8's encode time per
  prompt. The W4 lane is the small-host variant only.
- **Prefix-attention instrument landed (4592072):** `bench/measure_dit_prefix_attention.py`
  and its check. On the T2VA capture the DiT reads the prompt late (13% of
  attention at block 49 against a 0.29% share; 0.2% at block 0), through a
  minority of heads, over-reads section keys and under-reads cut
  timestamps. The ref2va version needs the capture path to dump the
  encoder's token tags and vision spans, then one captured render
  (`workflows/h3_probe_capture_ref3_api.json`). Also found: that capture's
  `manifest.json` describes a different render than its tensors and
  `check_capture_manifest.py` passes it; ungated.
- **(superseded) First thing tomorrow, before the ablation:** extend
  `bench/compare_transformers_comfy_layer50.py`'s ComfyUI arm with
  `--clip-path` (core's own CLIP loader, for the ComfyUI-native
  `qwen3vl_32b_minimax_h3_int8_convrot` and `..._nvfp4_awq` encoders), then
  grade all four encoders on the same holdout against BF16. That table
  decides which encoder the ablation and the marker arms run on, and closes
  the open item from `internal/postmortems/2026-08-23_session_h3_awq_hf_workflows_and_encoder_ab.md`.
- **Two confounds, found after the verdict (launch record, Gate 5
  section):** v1 was calibrated with `duo_scaling: false`, v2 with `true`,
  so data and scale rule changed together; and the candidate's `recipe.yaml`
  is empty because the emit path saves after the session closes (the run
  record carries the recipe). The cheapest attribution experiment is a
  rerun on the same 29 rows with `--awq-duo-scaling false` (about 3.5 h,
  same launcher): v2b close to v1 means the data did not matter at this bit
  width; v2b better than v1 means duo scaling hurt. Fix the emit path to
  write the recipe before that run so the artifact carries it.
- **Overfit test: done 18:25** (`bench/results/2026-08-25_v2_calibration_rows_layer50.json`):
  v2 11% better than v1 on its own rows, 7% worse on the holdout; leaning
  overfit, but the floor (0.33 to 0.38 for both, everywhere) is the story.
  Recipe first: AWQ+GPTQ or more bits on the sensitive layers, on a
  token-balanced stratified population with a per-stratum bar.
- **Then, if v2's population is the suspect:** the modifier-cache change is
  what a larger population needs (deferred work in the plan); a recipe
  variant (smoothing `o_proj`, a smaller group) is the cheaper experiment
  and runs at today's population in the same 3.5 hours.
- One text-only row is bad under both W4 artifacts (relative L2 0.74 / 0.89
  where the median is 0.06); find which and why before reading the text
  criterion as clean.
- Instrument notes from today's first real run: the ComfyUI arm needs
  `--reserve-vram-gib 16` for rows above about 20k tokens; a rerun skips
  captured rows (`--all-rows`); text-only rows are handled. The
  `bench/check_no_owner_paths.py` red from the converter's print is fixed
  (a5ff7b3).
