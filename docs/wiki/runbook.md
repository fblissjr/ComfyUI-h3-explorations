# The card and ledger runbook: pointers to what is written, and where sessions improvised

last updated: 2026-10-17 (initial six-section pointer runbook `(a)` through `(f)` for the Gemini lead on the card and the ledger, with dated session improvisations from `internal/claude/2026-10-1*/` and the git history of [`handover.md`](handover.md))

**A router, not a second copy.** Where this page and an owner file or code symbol disagree, the owner is right ([`AGENTS.md`](../../AGENTS.md), "Do not trust prose, including this file's"). Run every script with `<comfy-venv-python>`, never `uv run` or `uv sync` ([`docs/comfy_notes.md`](../comfy_notes.md), "Running things").

---

## (a) ComfyUI

### Written procedure (pointers)

- **Environment keys (`H3_OUTPUT_DIR`, `H3_INPUT_DIR`, `H3_MODEL_STORE`, `H3_CAPTURE_ROOT`)**: [`docs/comfy_notes.md`](../comfy_notes.md) ("Where this machine keeps things: three environment keys" and "Captures are transient"); read by `workflows/h3_config.py::output_dir`, `bench/masked_job.py::input_dir`, [`bench/shot_to_load.py`](../../bench/shot_to_load.py), and [`bench/load_chain.py`](../../bench/load_chain.py).
- **When a restart is needed**: whenever a node-pack module or schema the chain executes has changed ([`AGENTS.md`](../../AGENTS.md), "The shared tree"; [`docs/comfy_notes.md`](../comfy_notes.md), "Changing a graph"), so the server's start time (`ps -o lstart= -p $PID`) is later than the edited file and `/object_info` reflects the current schema.
- **Armed-environment check before any kill or timing**: the server process is the resource, not the GPU ([`AGENTS.md`](../../AGENTS.md), "Operative rules"; [`docs/comfy_notes.md`](../comfy_notes.md), "The three arming keys": `H3_CAPTURE`, `H3_SOL_OBSERVE`, `H3_SOL_PROBE`, `H3_TELEMETRY`). Always resolve the port owner with `ss -ltnp | grep ':8188'` and inspect `tr '\0' '\n' < /proc/$PID/environ | grep '^H3_'` before stopping or restarting; if armed, ask first.
- **Start and stop (by hand only)**: [`bench/restart_comfy.sh`](../../bench/restart_comfy.sh) is **DISABLED since 2026-09-02** (exits 2) because it hung on every run. Follow the manual recipe in [`docs/comfy_notes.md`](../comfy_notes.md) ("The recipe, by hand"):
  1. Find port 8188's PID via `ss -ltnp`, check `/proc/$PID/environ` for `^H3_`, kill `$PID`, confirm port 8188 is free, and verify no `start.sh` or `uv run --active` survivor remains in `ps` ("Between the kill and the launch, confirm the launcher chain is gone too").
  2. Launch detached from the ComfyUI checkout (`setsid nohup ./start.sh > <log_file> 2>&1 < /dev/null &`), never via `uv run`, and poll `curl -s localhost:8188/system_stats` and `GET /internal/logs/raw` ("Traps that have each bitten more than once").
  3. Stop any server you started when done ([`docs/comfy_notes.md`](../comfy_notes.md), "If you started the server, stop it when you are done").
- **Cache states (cold vs. warm) and `/history` as the observable**:
  - ComfyUI's RAM-pressure node cache serves upstream outputs whose input signature is unchanged (`execution_cached` in `/history`), recorded per prompt by [`bench/record_render_substrate.py`](../../bench/record_render_substrate.py) ([`docs/comfy_notes.md`](../comfy_notes.md), "What is cached, and by whom").
  - Calling `POST /free` (which [`bench/load_chain.py`](../../bench/load_chain.py) `land` runs before its standalone pose pass) unloads all models even when `unload_models` is false, so the next queued job starts cold ([`docs/comfy_notes.md`](../comfy_notes.md), "Traps that have each bitten more than once").
  - This pack keeps no cross-run key cache of its own ([`AGENTS.md`](../../AGENTS.md), "Nothing an earlier run made is reused by a key").

### Where sessions did it differently (improvised)

- **Uncommitted node edits in the shared tree holding or racing a server restart**: on 2026-10-10, sessions edited node code (`video_mask.py`, `workflows/h3_config.py`) directly in the shared tree while the card was active, either blocking a needed restart for 22 minutes or forcing hunks to be stashed and restored (`internal/claude/2026-10-10_mrscribe/findings_D1_coordination.md`, rows at lines 43–44 and 56).
- **Triggering `uv` in this checkout and dirtying `uv.lock`**: despite the ban in [`docs/comfy_notes.md`](../comfy_notes.md), sessions invoked `uv` commands that rewrote or created `uv.lock` on 2026-10-12 ([`handover.md`](handover.md) at commit `55f1db74`, "What not to assume") and again on 2026-10-17 (`internal/gemini/2026-10-17_claude_to_gemini.md`, `C2` item 1 and `C3`).
- **Reading `--input-directory` from `/proc/<pid>/cmdline` mid-session**: on 2026-10-16, when the sidecar needed `H3_INPUT_DIR` during a delivery rebuild, the session inspected `/proc/<pid>/cmdline` of the live server and hardcoded `export H3_INPUT_DIR=<input_dir>` into `internal/claude/2026-10-16_mrtoon/rebuild.sh` (lines 10–11).
- **Ad-hoc log scraping inside session wrapper scripts**: `internal/claude/2026-10-15_mrgo_scripts/render_land.sh` and `internal/claude/2026-10-16_mrgan/render_land.sh` piped `curl -s -m 10 http://127.0.0.1:8188/internal/logs/raw` through an inline Python one-liner between `render` and `land` to scrape `dense (` / `h3-sol] sparse` lines from the last 400 log entries.

---

## (b) One load, from preview to land

### Written procedure (pointers)

End-to-end stage table and data flow: [`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md) ("The flow" and stages 4–18) and [`docs/wiki/masked_v2v.md`](masked_v2v.md) ("A shot, made whole: the recipe the owner kept on 2026-10-15" and "Running a job").

1. **Shot preview and who-is-who (`preview`, stages 4–6)**:
   - [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `preview` (calling [`bench/masked_job.py`](../../bench/masked_job.py)) builds `shot_c_<first>` with one SAM 3.1 tracker, Sapiens parts/class maps, pose, and body mesh per subject in `data/<video>/shots.json`.
   - **Gate**: a reader inspects the tracker tiles; who is in each shot is read from `data/<video>/shots.json`, `data/<video>/cast/cast.jsonl`, and `data/<video>/owner_answers.jsonl` ([`AGENTS.md`](../../AGENTS.md), "Where state lives": read those and [`bench/page_answers_ledger.py`](../../bench/page_answers_ledger.py) before ever asking the owner about a picture; [`bench/who_page.py`](../../bench/who_page.py) -> [`bench/page_answers_to_cast.py`](../../bench/page_answers_to_cast.py)). Wrong picks remake the preview under the next letter (`--remake`).
2. **Job, head reference, and text (stages 8–10)**:
   - [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `job`: writes `<subject>_<form>_<first>.json`, printing and recording `THE DEFAULTS THIS LOAD CARRIES` (`bench/shot_to_load.py::defaults_carried`).
   - [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `reference` (calling [`bench/head_reference.py`](../../bench/head_reference.py)): cuts the body mesh and source to the head box on the load's frames and draws the face lines (`bench/shot_to_load.py::REFERENCE_DEFAULT`). Writes to `Video/<video>/references/<subject>_<first>_<last>[_<letter>]/`, never overwriting or moving a folder a render has read (`bench/shot_to_load.py::REFERENCE_SUFFIXES`). **Gate**: fails on `face_on_the_track` outside `bench/head_reference.py::FILL_GAP`; a person reads `<stem>_sheet.jpg`.
   - [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `text` (calling [`bench/load_text.py`](../../bench/load_text.py)): generates the load's text from the reference record, `data/<video>/plan.json`, and the voice spans (`bench/load_text.py::NO_SCENE`, `bench/load_text.py::NO_CLOTHING`).
3. **First capture, node plan, on-plan capture, and frame table (stages 11–12)**:
   - [`bench/load_chain.py`](../../bench/load_chain.py) `first`: builds the initial capture via [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py), runs `verify`, evaluates the preflight gate (`flags.json`), and writes the strip under `<capture>_first`.
   - [`bench/load_chain.py`](../../bench/load_chain.py) `plan`: builds and queues the no-sampling plan graph (`workflows/masked_job_graph.py`). Refuses if `first`'s gate is blocked.
   - [`bench/load_chain.py`](../../bench/load_chain.py) `on-plan`: copies the first capture to `<capture>_first`, rebuilds the capture in place on the node's plan, runs `verify` and the gate, draws the plan strip (`Video/<video>/looks/<date>_<load>/`), rebuilds the hair band on the carved track (`band-again` when applicable), and writes `frame_table.jsonl` ([`bench/frame_table.py`](../../bench/frame_table.py)).
4. **Pre-render gates: second reader's look, full prompt read, and owner's go (stages 13–14)**:
   - **Second reader's look**: a session that did not build the capture inspects the plan strip and records [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) `looked` (`looks.json`, timestamp newer than the capture), or on a clear gate with no forcing rule (`bench/load_chain.py::FORCING_FLAGS`, `bench/load_chain.py::FORCING_FIGURES`) the maker records `load_chain.py no-look`.
   - **Full prompt read**: [`bench/what_h3_got.py`](../../bench/what_h3_got.py) `--stage plan` reads the prompt in full from the queued graph ([`AGENTS.md`](../../AGENTS.md), "A render's prompt is read in FULL before it renders").
   - **Owner's go**: the owner reviews the plan strip / diff or gives a go on the Open List / terminal ([`AGENTS.md`](../../AGENTS.md), "How the owner and the sessions talk").
5. **Render, land, sidecar, and what is laid (stages 15–18)**:
   - [`bench/load_chain.py`](../../bench/load_chain.py) `render --grounds GROUNDS.txt --show SHOW.txt`: verifies the look/no-look timestamp, clear gate, `frame_table.jsonl` digest, byte-equality of the job file with `job.plan.json`, clean node code, and mask frame hashes, then queues the render graph.
   - [`bench/load_chain.py`](../../bench/load_chain.py) `land`: frees the card (`POST /free`), runs the pose pass on the rendered video (`POSE_PASS_FAILED` if incomplete), records cost lines from `/history`, and builds the stacked review clip (`Video/<video>/review_clips/<load>_<first>_<last>_stacked.mp4`) and sidecar (`<video>.mp4.json` via [`bench/sidecar.py`](../../bench/sidecar.py)).
   - [`bench/load_chain.py`](../../bench/load_chain.py) `page`: prepends the item to `data/<video>/review/items.json`.
   - [`bench/laid.py`](../../bench/laid.py) (`seed`, `verify`, `refs`): maintains `data/<video>/laid.jsonl`, the single list of kept/laid pieces with their `sha256` ([`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), stage 18).

### Where sessions did it differently (improvised)

- **Five per-session shell scripts before `load_chain.py`, typing mask frame offsets by hand**: on 2026-10-13 (`internal/claude/2026-10-13_mrpin_scripts/{load_first.sh,load_on_plan.sh,queue_render.sh,load_land.sh,page_item.py}`), sessions passed 13 arguments by hand and typed the load's first frame instead of the preview's first frame, shifting masks 4 and 8 frames off the picture while `verify` and the gate passed ([`bench/load_chain.py`](../../bench/load_chain.py) docstring).
- **Session wrapper scripts chaining `first`, `plan`, and `on-plan` and truncating output with `grep` / `cut`**: `internal/claude/2026-10-13_mrpen_scripts/to_plan.sh`, `internal/claude/2026-10-14_mrcircle_scripts/to_plan.sh`, `internal/claude/2026-10-15_mrtall_scripts/to_plan.sh`, `internal/claude/2026-10-16_mrgan/to_plan.sh`, and `internal/claude/2026-10-16_mrtaco/plans/to_plan.sh` looped over `first plan on-plan` and filtered stdout with `grep -E ... | cut -c1-300`.
- **Renaming (`mv`) reference folders on disk to `*_SUPERSEDED_*` before `_b`/`_c` suffixing landed**: in `internal/claude/2026-10-15_mrarc_scripts/chain_hp5461b.sh` (line 5) and `internal/claude/2026-10-15_mrarc_scripts/chain_remakes4.sh` (lines 12–13), the session ran `mv $R/<span_folder> $R/<span_folder>_SUPERSEDED_...` so `shot_to_load.py reference` would not refuse existing folders, breaking path links for earlier renders until [`handover.md`](handover.md) commit `64a9a92e` and `bench/shot_to_load.py::REFERENCE_SUFFIXES` forbade moving read references.
- **Shifting a test load's span by 1 frame to avoid a reference folder collision**: in `internal/claude/2026-10-15_mrarc_scripts/chain_eyetest.sh` (lines 7–9), two eye-shut test loads on the same span were built on `2945–2981` and `2944–2980` ("one frame earlier than its twin ... so the two references do not share a folder").
- **Assuming `look_carry.py` checked a changed reference video**: on 2026-10-14 (`internal/claude/2026-10-14_mrcircle_night_notes.md` and [`handover.md`](handover.md) at commit `4ebab898`, "What not to assume"), sessions planned to carry looks across remade loads whose reference video had changed, before noting that `look_carry.py` only compared masks.

---

## (c) The review server and the owner's pages

### Written procedure (pointers)

- **Starting, checking, and stopping `data/<video>/poc/serve.py` (port 8765)**: [`docs/comfy_notes.md`](../comfy_notes.md) ("The local review page has its own small server, started and stopped by hand").
  - Launch from the repo root detached (`setsid nohup python3 data/<video>/poc/serve.py > <log_file> 2>&1 < /dev/null &`); needs no ComfyUI and no GPU.
  - Verify with `python3 data/<video>/poc/check_page.py` or `curl` against `http://<host>:8765/fun_share/TO_REVIEW.html` and `PROVENANCE.html` (`data/<video>/poc/fun_share` symlinks to `Video/<video>/` on the share).
  - Stop by port owner (`ss -ltnp | grep ':8765 '`, then `kill <PID>`); if already running when a session arrives, ask before stopping since the owner may be reviewing.
- **Recording the owner's clicks into primary records and verdicts**: `serve.py` appends clicks to `data/<video>/owner_answers.jsonl` (append-only, never edited; `data/<video>/poc/answers.jsonl` is a symlink to it; [`AGENTS.md`](../../AGENTS.md), "Where state lives"; [`bench/page_answers_ledger.py`](../../bench/page_answers_ledger.py) lists any line no record cites). [`bench/page_answers_to_verdicts.py`](../../bench/page_answers_to_verdicts.py) writes `data/<video>/verdicts/verdicts.jsonl` ([`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), stage 25), and [`bench/covered_items.py`](../../bench/covered_items.py) updates `data/<video>/review/covered.json`.
- **Building `Video/<video>/TO_REVIEW.html` and `PROVENANCE.html`**: [`bench/review_pages.py`](../../bench/review_pages.py) (`to-review` and `provenance` subcommands; [`AGENTS.md`](../../AGENTS.md), "One ledger, made new on every landing"; [`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), stage 24).
  - One copy each in `Video/<video>/` on the share, never in a day's folder.
  - Since commit `0f0af6cb` (D19c), both subcommands require `--gaps <frame_coverage.json>` (or `assembly.stack.coverage_json` in the table JSON matching `table_sha256`) and refuse a missing or stale gap check.

### Where sessions did it differently (improvised)

- **`pages.sh` and `rebuild.sh` existed only in untracked session folders**:
  - `internal/claude/2026-10-14_mrtwo_scripts/{pages.sh,rebuild.sh}`
  - `internal/claude/2026-10-15_mrtall_scripts/{pages.sh,rebuild.sh}`
  - `internal/claude/2026-10-16_mrtoon/{pages.sh,rebuild.sh}`
  Each session copied the previous day's script and edited hardcoded dates (`data/2026-10-16_fun_delivery_tables`), session paths (`internal/claude/2026-10-16_mrtoon/start_here.json`), trial table lists (`TB=(...)`), and scratchpad directories (`/tmp/claude-1000/.../scratchpad/stack_work`). After `0f0af6cb` landed `--gaps`, `internal/claude/2026-10-16_mrtoon/pages.sh` as written refuses until `--gaps` is passed (`internal/gemini/2026-10-17_claude_to_gemini.md`, `C4`).
- **Hand-written session Python scripts before `bench/review_pages.py`**: on 2026-10-13 and 2026-10-14, the pages were built by untracked scripts (`internal/claude/2026-10-13_mrtower_scripts/{to_review_page.py,provenance_page.py}` and `internal/claude/2026-10-14_mrtwo_scripts/{to_review_page.py,provenance_page.py}`) with hardcoded server URLs, typed decision lists, and a stopped-loads list in `internal/claude/2026-10-13_mrtower_scripts/stopped_2026-10-13_evening.json`, writing into `Video/fun/day_2026-10-13/` instead of `Video/fun/` ([`bench/review_pages.py`](../../bench/review_pages.py) docstring; [`handover.md`](handover.md) at commit `4ebab898`).
- **Launching `serve.py` attached to an agent's session so it died on handoff**: on 2026-10-16, `mrtoon` started `serve.py` without detaching it from the session's process group, so the server died when `mrtoon` handed off and had to be restarted by `mrtony` (`internal/claude/2026-10-16_mrtony/HANDOFF.md`, "Review server").

---

## (d) Provenance

### Written procedure (pointers)

Rule: [`AGENTS.md`](../../AGENTS.md) ("One ledger, made new on every landing" and "Where state lives") and [`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md) (stages 18–24). After every landing or stopped load, run the chain in order:

1. **Verify what is laid**: [`bench/laid.py`](../../bench/laid.py) `verify --laid data/<video>/laid.jsonl --share $H3_OUTPUT_DIR` (every laid file present and matching its recorded `sha256`).
2. **Render ledger**: [`bench/render_ledger.py`](../../bench/render_ledger.py) walks the share and `data/` records to write `data/<date>_<video>_delivery_tables/ledger_<video>_<stamp>.json` and `.md` ([`docs/wiki/index.md`](index.md), `bench/render_ledger.py` row).
3. **Delivery table**: [`bench/delivery_table.py`](../../bench/delivery_table.py) `--ledger ... --laid data/<video>/laid.jsonl` writes `t_<video>_0000_6394_<stamp>.txt`, `.json`, and `reading_<video>_0000_6394_<stamp>.md`, and prints the exact assembly, gap-check, and stack commands ([`docs/wiki/index.md`](index.md), `bench/delivery_table.py` row).
4. **Shot and per-frame gap checks**: [`bench/shot_coverage.py`](../../bench/shot_coverage.py) (what the video still needs per shot and subject) and [`bench/frame_coverage.py`](../../bench/frame_coverage.py) (per-frame gap check: `original_while_tracked`, `emptied_inside`, `laid_but_barely_changed`, `region_far_below_its_own_size`, `original_tracking_not_known`; [`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), stage 22).
5. **Assemble and stack the delivery cut (when rebuilding the cut)**: [`bench/assemble_delivery.py`](../../bench/assemble_delivery.py) builds the joined clip and `.check.json` (with `laid.py verify` before and after assembly); [`bench/stack_delivery.py`](../../bench/stack_delivery.py) `build` and `check` produce `..._VIEW.mp4` (`_STACKED.mp4`) and `.stack.json` ([`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), stages 21 and 23).
6. **Owner's pages**: [`bench/review_pages.py`](../../bench/review_pages.py) `to-review` and `provenance` with `--table`, `--ledger`, and `--gaps` (section `(c)` above).

### Where sessions did it differently (improvised)

- **Choosing what stands across four competing mechanisms before `laid.jsonl`**: until 2026-10-16 (`f923ef3b`), sessions controlled what the delivery table laid via ad-hoc text files `internal/claude/2026-10-14_mrtwo_scripts/prefers.txt` and `internal/claude/2026-10-15_mrtall_scripts/{prefers.txt,holds.txt}`, plus hardcoded `--prefer` and `--hold` flags inside `rebuild.sh` and newest-table-by-mtime lookups, causing what the owner reviewed to drift from what the cut laid ([`bench/laid.py`](../../bench/laid.py) docstring; `internal/claude/2026-10-15_mrtall_scripts/rebuild.sh` lines 7–11).
- **Scraping commands out of stdout with `grep` + `sed` + `eval` and ignoring stack check failures**: `internal/claude/2026-10-16_mrtoon/rebuild.sh` (lines 24–37) extracted `the clip:`, `the stack:`, and `its check:` from `table_fun_*.out` via `grep`/`sed`, ran them with `eval`, and appended `|| true` to `eval "$K"` so a failing stack check exited 0.
- **Hand-editing `verdicts.jsonl`, `items.json`, and `shots.json` with backup copies in `internal/claude/`**: on 2026-10-13 through 2026-10-16, sessions edited JSON/JSONL state files directly and left dozens of manual backups in session folders (`internal/claude/2026-10-13_mrtower_scripts/verdicts.before_mrtower_*.jsonl`, `internal/claude/2026-10-14_mrtwo_scripts/verdicts.before_mrtwo_*.jsonl`, `internal/claude/2026-10-16_mrtoon/verdicts.before_take.jsonl`).

---

## (e) End of day

### Written procedure (pointers)

- **Handover (`docs/wiki/handover.md`)**: rewritten at the day boundary to roughly 80 lines under **four headings** ([`AGENTS.md`](../../AGENTS.md), "Where state lives"):
  1. `## State when closed`
  2. `## What not to assume`
  3. `## Tomorrow, in order`
  4. `## Where things are`
  Every timestamp comes from `date -u` or a record, never an estimate ([`handover.md`](handover.md), "What not to assume").
- **Top block of `docs/wiki/next_steps.md`**: updated to match [`handover.md`](handover.md)'s order ([`AGENTS.md`](../../AGENTS.md), "Where state lives"; [`docs/wiki/next_steps.md`](next_steps.md)).
- **Daily changelog from commits (`changelog.d/README.md`)**: written once a day from `git log --since=<day_start> --reverse` as a single fragment `changelog.d/day-<YYYY-MM-DD>.md`, built as a standalone unpiped command with [`bench/build_changelog.py`](../../bench/build_changelog.py) `--with changelog.d/day-<YYYY-MM-DD>.md`, verified with `git diff --numstat -- CHANGELOG.md pyproject.toml`, and committed by pathspec ([`changelog.d/README.md`](../../changelog.d/README.md), [`AGENTS.md`](../../AGENTS.md)).
- **Read-only snapshot of `data/<video>/owner_answers.jsonl`**: take a read-only copy of `data/<video>/owner_answers.jsonl` into `data/<video>/owner_answers_snapshots/` at each day's close ([`AGENTS.md`](../../AGENTS.md), "Where state lives").
- **ComfyUI stopped at close**: stop the ComfyUI server on port 8188 by the manual recipe unless an owner-approved overnight job is running, in which case the card session stops ComfyUI as soon as that job finishes ([`docs/comfy_notes.md`](../comfy_notes.md); [`handover.md`](handover.md)).
- **Review server state explicitly recorded**: stop `data/<video>/poc/serve.py` on port 8765 at the close (or leave it up when the owner is still reviewing) and state explicitly in [`handover.md`](handover.md) whether port 8188 and port 8765 are stopped or running, with their PIDs/timestamps ([`docs/comfy_notes.md`](../comfy_notes.md), [`handover.md`](handover.md)).

### Where sessions did it differently (improvised)

- **Keeping `answers.jsonl` only inside `data/fun/poc/` with no close-of-day snapshot**: until 2026-10-17 (`67154307`), the owner's raw answers lived only at `data/fun/poc/answers.jsonl` with no read-only snapshot directory (`data/fun/owner_answers_snapshots/`), and a session on 2026-10-13 even edited `answers.jsonl` with a manual backup in `internal/claude/2026-10-13_mrtower_scripts/answers.before_mrtower_owner_removal_1161.jsonl`.
- **Deferring the daily changelog entry to the next morning**: at the close of 2026-10-16, `mrtony` did not write `changelog.d/day-2026-10-16.md` at closeout and instead placed "write the 2026-10-16 changelog entry from all that day's commits" as item 1 of "Tomorrow, in order" in [`handover.md`](handover.md) (commit `e100cab2`).
- **Letting `handover.md` grow well past the ~80-line target through incremental night amendments**: on 2026-10-15 and 2026-10-16, `handover.md` was amended 15–20 times after the initial close commit (`37a1dd06` through `82edf653` on 2026-10-15; `e9c8df8d` through `483d5b38` on 2026-10-16), reaching 131 lines with multi-paragraph branch review logs inside `Tomorrow, in order`.
- **Estimating timestamps in prose instead of running `date -u`**: on 2026-10-16, three estimated timestamps written by sessions were wrong, prompting the rule in [`handover.md`](handover.md) ("Every time written today came from `date -u` after three estimated ones were wrong").
- **Chaining `build_changelog.py` with `; git commit` before the per-day rule**: on 2026-10-10 (`internal/claude/2026-10-10_mrscribe/findings_D2_accounts.md`, line 54, commit `4f03d521`), a session chained `build_changelog.py` with `;` ahead of `git commit`, so when the changelog build refused, the commit still ran and swept an unrelated edit in `docs/checks.md`.

---

## (f) A render (what is read before it queues, and what changes when a render fails)

### Written procedure: before a render queues (pointers)

1. **Read the prompt in FULL from the graph as queued**: run [`bench/what_h3_got.py`](../../bench/what_h3_got.py) `--job-folder data/<day>_<load>_job --load <load> --stage plan` (and `--stage render` when built) and read the entire prompt text, never a summary, preview, or paraphrase ([`AGENTS.md`](../../AGENTS.md), "A render's prompt is read in FULL before it renders"; [`docs/wiki/masked_v2v.md`](masked_v2v.md), "Reading one window whole"). Verify that every wired `<Picture k>` and `<Video k>` socket and digest matches what the text describes.
2. **Check `THE DEFAULTS THIS LOAD CARRIES`**: read the lines printed by `shot_to_load.py job` and stored in `job.defaults` (`bench/shot_to_load.py::defaults_carried`, `bench/shot_to_load.py::REFERENCE_DEFAULT`): track-held reference (`bench/pose_along_shot.py::HOLD_GROW_PX`, `bench/head_reference.py::FILL_GAP`), eye form (`contrast`), mouth check (`auto`), zoom (`zoomed in where the node's box allows` vs. `--whole-frame`), keep-outs (`plan.json`), `pieces`, `hair_band`, and `restore` spans. Turning any default off requires a written reason ([`AGENTS.md`](../../AGENTS.md), "Settled about H3").
3. **Check window length and seam rules**:
   - **Ceiling and short windows**: `h3_rules.py::MAX_LENGTH` (`362`) is the only length cap; any `17n + 5` grid length up to `MAX_LENGTH` is valid for a single-window load (`workflows/masked_job_graph.py::window_for`, `loop_plan.py::stands_alone`). Because attention scales with the square of sequence length, short windows are encouraged and cheaper: a window takes only the frames its shot or text needs, never the ceiling by habit ([`docs/h3_resolutions.md`](../h3_resolutions.md), "The length axis"; [`AGENTS.md`](../../AGENTS.md), "Settled about H3").
   - **Audio clock alignment (best practice vs. seam rule)**: lengths whole on both the `24` fps video clock and the `40` Hz audio latent clock (`h3_rules.py::on_both_clocks`, every third `17n + 5` step starting at `39`) avoid the 1/3-audio-latent rounding remainder at the last frame. For a single-window load this is a **best practice, not a rule** — never pad or cut a shot to reach `39` or another both-clock length ([`docs/h3_resolutions.md`](../h3_resolutions.md), "Whole on the audio clock: a best practice, not a rule"). For multi-window loads, chained seams are enforced on both clocks from `h3_rules.py::SHORTEST_CHAINED` via `h3_rules.py::chain_lengths` (never typed as a literal number; placed on a quiet frame with `shot_to_load.py job --seam`; [`docs/wiki/masked_v2v.md`](masked_v2v.md), "Rules of the lane").
4. **Check the four timelines together**: text, window frames, mask/region, and audio ([`docs/wiki/masked_v2v.md`](masked_v2v.md), "The rule: data before a render, and the same data after" and "Reading one window whole"), plus the capture's `flags.json`, `frame_table.jsonl`, and second reader's look (`looks.json`).

### Written procedure: when a render fails, what changes and where each setting lives

Before changing any knob or queuing a test render, read the capture and the four timelines on the failing frames first ([`docs/wiki/masked_v2v.md`](masked_v2v.md), "The rule: data before a render, and the same data after"). Every remake gets a new load letter (`shot_to_load.py job --remake`), and carries the earlier load's span words when `--carry-words-from` is given; a remake without it is not yet refused (`D20c`).

| Lever changed on a remake | Where it lives | What written docs say it governs |
|---|---|---|
| **The text** (who is generated from `<Picture 1>`, `<Subject 2>`'s reference description, singing, clothing) | `data/<video>/plan.json` (`subjects.<s>.text`, `voice`, `clothing`), [`bench/load_text.py`](../../bench/load_text.py), `shot_to_load.py text` (`--by-hand WHY`), [`docs/prompting.md`](../prompting.md), [`docs/wiki/masked_v2v.md`](masked_v2v.md) ("The frame every masked text is written in") | Progressive disclosure: never describe the frozen picture (`bench/load_text.py::NO_SCENE`); say `"sings"` only where the subject sings in that shot/window; name per-shot clothing when clothing is redrawn (`bench/load_text.py::NO_CLOTHING`); hair-only loads (`--parts hair`) omit facial/eye/singing words. |
| **Reference video & drawing** (drawn vs. plain cut, mouth form, iris form, crop room) | [`bench/head_reference.py`](../../bench/head_reference.py), [`bench/crop_reference_to_head.py`](../../bench/crop_reference_to_head.py) (`--room`), [`bench/draw_face_points.py`](../../bench/draw_face_points.py), `shot_to_load.py reference` (`--plain WHY`, `--mouth`, `--iris`, `--form-why`), `data/<video>/plan.json` (`reference`) | Default is `bench/shot_to_load.py::REFERENCE_DEFAULT` (`mouth: "auto"`, `iris: "contrast"`). Use `--plain WHY` when a face is too small or turned for drawn lines ([`docs/wiki/masked_v2v.md`](masked_v2v.md), "Rules of the lane"). A hair-only load where the body model guesses the head pose wrong can set `motion: none` ([`bench/load_text.py`](../../bench/load_text.py) docstring). |
| **Span words (`--no-face-on`, `--hold-part`, `--no-mouth-on`, `--mouth-on`, `--eyes-shut-on`, `--parts-on`, `--track-not`)** | [`bench/shot_to_load.py`](../../bench/shot_to_load.py) `job` flags, stored in `<load>.json` and read by [`bench/head_reference.py`](../../bench/head_reference.py) and [`bench/capture_masked_run.py`](../../bench/capture_masked_run.py) | `--no-face-on A-B=WHY` when the face is hidden or turned away (prevents drawing a face on the back of a head; [`docs/wiki/masked_v2v.md`](masked_v2v.md), "A shot, made whole"); `--hold-part A-B=WHY` when the face shows but Sapiens misses it; `--no-mouth-on` / `--mouth-on` when the mouth gate drops or misdraws lips; `--eyes-shut-on` when eyes are closed in the source; `--track-not` when another subject's track overlaps. |
| **Part floor (`--least`) and `pieces`** | `shot_to_load.py job --least PX --least-why WHY`, `data/<video>/plan.json` (`pieces`, `pieces_by`), `workflows/h3_config.py::MASKED_LOAD_DEFAULTS` | Lower `--least` when a small or profile face is emptied by the default pixel floor (`4096` px); `pieces` is off for face/hair loads (`decisions.md`, 2026-10-16). |
| **Keep-outs (`held`, `hands`, `clothes`, `strangers`), `hair_band`, and `restore` give-backs** | `data/<video>/plan.json` (`keep_outs`, `hair_band`, `restore`), `shot_to_load.py job` (`--no-strangers`), [`bench/delivery_table.py`](../../bench/delivery_table.py) | Keep-outs carve non-subject classes out of the region; when `held`/`hands`/`clothes` carve into the subject's own neck, hair, or apron, retire them in `plan.json` and redraw clothing (`decisions.md`, 2026-10-16/17). Another tracked subject inside the margin is given back via `restore=` at assembly ([`docs/wiki/masked_v2v.md`](masked_v2v.md), "A shot, made whole"). |
| **Zoom (`render_region`)** | `shot_to_load.py job` (zoomed by default via `bench/shot_to_load.py::defaults_carried`; `--whole-frame WHY` or `plan.json` `zoom: false` to disable), [`docs/wiki/masked_v2v.md`](masked_v2v.md) ("A small subject, rendered zoomed in") | Zooms the render crop around the replaced part so a small subject gets more DiT tokens at the same canvas cost; disable with `--whole-frame WHY` when the box cannot hold the motion or context. |
| **Margin (`margin`)** | `workflows/h3_config.py::MASKED_LOAD_MARGIN` (by plan `region`), `load.margin` in `<load>.json`, `video_mask.py` | Grows the Sapiens part mask by `margin` pixels before 32-px token snapping ([`docs/wiki/masked_v2v_pipeline.md`](masked_v2v_pipeline.md), "Where a frame's content is decided"). |
| **Context frames, `context_noise`, and `--seam`** | `workflows/h3_config.py::MASKED_JOB_CONTEXT`, `workflows/h3_config.py::MASKED_JOB_CONTEXT_NOISE`, `shot_to_load.py job --seam`, [`docs/wiki/window_context.md`](window_context.md) | Controls how strongly a continuation window is pinned to the previous window's frozen rows vs. following a new turn or bow in the reference (`context_noise` `0.8` loosened a stiff continuation on `604–1009`; [`docs/wiki/window_context.md`](window_context.md), [`docs/wiki/next_steps.md`](next_steps.md) item 1). |
| **Seed (`seed`)** | `workflows/h3_config.py::SEED`, `load.seed` in `<load>.json`, [`docs/wiki/window_context.md`](window_context.md) | Held constant across A/B comparisons ([`docs/eval_comparison.md`](../eval_comparison.md)); varied only when testing whether a defect is a single-seed sample artifact. |

### Documented gaps (where no tracked doc maps a failure mode to a knob)

- **No diagnostic decision tree for a bad render when the capture and text are clean**: when the mask is on the right person, the reference points are on-track, and the generated text passes [`bench/what_h3_got.py`](../../bench/what_h3_got.py), no tracked document specifies **which knob to change first or in what order** (e.g., when to change `margin` vs. `--whole-frame` vs. `seed` vs. `--room` vs. `context_noise` for a wrong head pose, a boundary pop, or a likeness drift such as `1010c` sitting upright facing the camera in [`handover.md`](handover.md)).
- **Gaze and sideways glances**: recorded as an open limit ([`docs/wiki/next_steps.md`](next_steps.md) item 8; `decisions.md`, 2026-10-14 and 2026-10-15). Testing tighter crop room (`--room 0.3`) and painted-out eye sockets is a queued experiment, not a settled procedure.
- **Selecting `--least` (the part pixel floor)**: `4096` px is a starting constant (`workflows/h3_config.py::MASKED_LOAD_DEFAULTS`), and `480` px was copied across loads by sessions from one earlier load (`third_fp_4479b`); no tracked doc gives a formula or rule deriving `--least` from `face_height_px`.

### Where sessions did it differently (improvised)

- **Spending renders on kernel settings and audio files before reading the load's text**: on 2026-10-10 (`docs/research/postmortems/2026-10-10_span_fun-three-people.md` section 1.1; [`docs/wiki/masked_v2v.md`](masked_v2v.md), "Reading one window whole" and "Rules of the lane"), sessions attributed wrong renders to a signal or kernel setting four times and spent three renders before reading the full prompt text, where one sentence fixed the fault.
- **Dropping span words (`--no-mouth-on`, `--no-face-on`) when remaking a load by hand**: on 2026-10-16, when `second_hfp_1010c` was built to replace `second_hfp_1010j`, the session typed the new job command without `1010j`'s `no_mouth_on 1010-1018` word ([`handover.md`](handover.md), commits `791ad782` and `32dc23a6`, prompting D20b `--carry-words-from`).
- **Copying `--least 480` by habit across shots and subjects**: in `internal/claude/2026-10-15_mrarc_scripts/chain_remakes4.sh` (line 10) and `internal/claude/2026-10-16_mrgan/job_kitchen_lead_b.sh` (line 11), sessions passed `--least 480` citing "480 as on third_fp_4479b, a starting value" whenever the 4096 default emptied frames, rather than a per-shot measurement.
- **Mutating capture JSONs and job files with one-off Python scripts in `internal/claude/`**: on 2026-10-13 and 2026-10-14 (`internal/claude/2026-10-13_mrpen_scripts/{override_dropped.py,set_pieces.py}` and `internal/claude/2026-10-14_mrcircle_scripts/{override_dropped.py,set_pieces.py}`), sessions patched `flags.json` overrides and `pieces="largest"` directly via scratch Python scripts before `shot_to_load.py` and `load_chain.py` carried them from `plan.json`.
