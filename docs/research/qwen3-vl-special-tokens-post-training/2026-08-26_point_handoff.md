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

## Where the lane stands at the end of 2026-08-25

Read the launch record's last three sections
(`canonical/2026-08-25_v2_launch_record.md`: Gate 5 result, overfit test,
four-encoder table) before anything below; they carry the numbers and the
decisions. The short form:

- **v2 (W4A16 AWQ, native-H3 calibration) was rejected** against the
  pre-registered bar: it redistributed the 4-bit error between rows rather
  than reducing it, and it leaned overfit on its own 29 rows. Two confounds
  are recorded (v1 ran `duo_scaling false`, v2 `true`; the emit path shipped
  an empty `recipe.yaml`, since repaired at the source and in the snapshot).
- **INT8 ConvRot is the encoder of record** (owner decision on the
  four-encoder table: about fifteen times closer to BF16 than either W4
  artifact on every population). The three ablation graphs now load it
  through core's `CLIPLoader` (`h3_config.ENCODER_INT8`,
  `CORE_LOADED_ENCODERS` picks the loader node). Its encode cost on the real
  arm B graph: about 60 s for ~19k encoder tokens against 600 s of sampling
  (`bench/results/2026-08-25_refview_b_qwen2048_int8_occupancy.json`,
  compute-bound, not streaming-bound). Arm A's render was stopped before it
  wrote; rerun it for the cheap-view number if wanted.
- **The W4 lane continues only as the small-host variant**, with GPTQ as its
  method check: `bench/h3_gptq_recipe.py`, the pilot's
  `--modifier gptq | awq_gptq`, and `bench/check_h3_gptq_recipe.py` are
  committed. GPTQ alone runs first (AWQ-then-GPTQ has a Hessian seam, read
  the recipe docstring). Nothing has run on the card yet.
- **The DiT reads the prompt late and through few heads**
  (`bench/measure_dit_prefix_attention.py`, T2VA capture): 13% of
  attention on the prefix at block 49 against a 0.29% share, 0.2% at block
  0; section keys over-read, cut timestamps under-read. The ref2va version
  needs the capture path to dump token tags and vision spans, then one
  captured render.
- **The pool cannot weight the schema**: of the seven H3 markers only
  `<d>`/`</d>` occur in H3-IR (536 rows); the other five and every timestamp
  tag occur zero times (`bench/results/2026-08-25_v3_selection_*.json`,
  `markers_observed_in_pool`). A token-balanced selection moves text from
  14% to 24% of tokens but strict schema positions stay near 1%. Weight on
  markers is a corpus question (`bench/marker_corpus/`), not a selector one.

## What runs next, in order (card work; ComfyUI stopped first, restarted after)

1. **GPTQ probe**, ~10 min: `internal/v2_run_2026-08-25/run_gptq_probe.sh`
   (2 layers, 8 rows, disk tier, cgroup cap, emits to scratch). Read the
   record for `hessian_peak_gib_by_device` (decides
   `--gptq-offload-hessians`), `cholesky_fallbacks` (must be empty, else
   raise `--gptq-dampening-frac`), `seconds_in_gptq_quantize` per layer, and
   whether the emitted candidate carries `actorder: static` and the adapter
   accepts it (`probe_awq_recipe_boundary.py` compares `actorder` with the
   deployed artifact and will differ by design).
2. **GPTQ full run on the same 29 rows** (`run_gptq_full.sh`, pass the
   offload flag if the probe says so): the method check against v2, same
   holdout, same bar. Then the Gate 5 pass with `run_gate5.sh` pointed at the
   new candidate name (edit `CAND`/`V2`/snapshot name; a new suffix, never
   `--force` over a snapshot an artifact matches). Accept for the small-host
   variant only if it beats v1 on the bar; INT8 stays the encoder of record
   regardless.
3. **Gate 6 blind pairs on INT8**: `bench/run_graph_arms.py --manifest
   bench/gate6_refview_arms.json` with mr_data's population patched in
   (their selection file under their results; the arms differ only where a
   still's short edge is below 2048; the null-control row must read as
   identical), then `bench/blind_batch.py`, the scoring app,
   `bench/score_session.py`; the owner judges (`docs/eval_comparison.md`
   section 3).
4. **Marker arms on INT8**: the corpus (`bench/marker_corpus/compiled.json`)
   through `MiniMaxH3MarkerArm` with the stamp's `encoder_arm` block as
   evidence; arms release IDs, legacy BPE, stripped, mean-initialised rows.
5. **Prefix attention on ref2va**: capture-side change (dump
   `minimax_token_tags` and `embeds_info` beside the tensors), one captured
   render of `workflows/h3_probe_capture_ref3_api.json`, then the measurement
   with the label-span classes. Also gate the capture manifest against its
   own tensors (`check_capture_manifest.py` passed a manifest describing a
   different render).
6. **v3 population**, only if GPTQ earns it: re-run the two v3 selections
   with `--exclude-selection internal/v2_run_2026-08-25/selections/selection_holdout_v2b.json`
   (they excluded the catalogue series but not the holdout), decide the
   vision cap rather than inherit 6,000, build two bundles (the 26 text-only
   rows need their own bundle and run: the trace fixes the modality
   envelope), review with mr_data's harness, then run.
7. Parked by owner decision: checkpoint/resume (module and check exist;
   integrate only for a run longer than five hours); the `duo_scaling false`
   attribution rerun (superseded by the method change).

## Loose ends found tonight, none blocking

- The generator warns that three `h3_image_probe_format_*` graphs carry
  three widget values for four `MiniMaxH3AppendRefImage` widgets (from the
  `qwen_short_edge` addition); rebuild and check before touching those
  graphs.
- One text-only holdout row is bad under both W4 artifacts (relative L2 0.74
  and 0.89 where the median is 0.06); find which and why before reading the
  text criterion as clean.
- `report["boundary"]` in the pilot still says "no recipe instantiated" on
  modifier-bearing runs; existing records carry it; a separate decision.
- The `server reserve` rule from earlier stands: measured on the ablation
  arms before any `start.sh` default changes; the owner also wants the other
  defaults in that script audited, by measurement.
- Peer sessions ended tonight; fresh ones take the roles with the files
  named above.
