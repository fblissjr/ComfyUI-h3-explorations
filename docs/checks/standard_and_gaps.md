# Check Standard Case Studies, Running Notes, Retired Checks & Gaps

> **Extracted from [`docs/checks.md`](../checks.md) on 2026-10-27.**
> Preserves the `break_checks.py` history, metric-ranking, expression-driving, and queue-execution case studies alongside retired checks, uncontrolled requirements, and gaps. For the active check index and `doc-link-absent` ledger, see [`docs/checks.md`](../checks.md).

## Mutation Controls (`break_checks.py`) & Standard Case Studies

Added 2026-10-10, and not a requirement either: when a session does break
something to show a check red, the break is evidence only beside a GREEN run
of the same unbroken copy by the same command, and only when the failing
line names the behaviour that was broken. That day a session reported
fourteen breaks as seen; the copies could not import ComfyUI, so every run
was red whatever was broken, and four of the breaks had never been seen by
any case. `bench/break_checks.py` is the tool that came out of it: it makes
the copy where a check can import what it imports in the tree, runs the
control first, and calls a red with an import error the wrong reason.

Added 2026-10-16, after a night of mutation runs by Claude readers and Gemini
builders, none of which used `break_checks.py`: the same trap came back (a
loop that ran each break with only `PATH` set and the system python, whose
unbroken copy was never run that way), and two more things were learnt. A
copy outside `custom_nodes` cannot import the package, so a hand-made copy
sits under a folder of that name. And a break caught only by one broad case
(every one of eight removed refusals failing the same "a key left uncarried
is refused") shows that something changed, not that each guard has its own
case; a guard counts as covered when the case that fails names it. Why such a
run is slow: each break reruns the whole check, every case building its small
job on disk, one after another, because two breaks in one copy would
overwrite each other's file; running them side by side needs a copy each.

### A metric that ranks two arms is a claim about the metric

**Adopted 2026-08-31, after two wrong conclusions were published in
consecutive commits, in the file whose own opening line warns about scope.**
The instance is worth stating in full because both halves were made by people
who had just written the warning.

One LoRA delta merged into an `int8_convrot` weight, three metrics, three
different blind spots:

| metric | what it prefers | what it cannot see |
|---|---|---|
| stored-weight distance `‖Q(W+d) − (W+d)‖ / ‖W+d‖` | round-to-nearest | that RTN wins **because it barely applies the update** — the target is close to the unmerged weight, so doing nothing scores well |
| realisation `<Q(W+d) − Q(W), d> / <d, d>` | stochastic | that a sub-step delta is realised as sparse **full-step jumps** — the direction is right, the per-weight representation is not |
| noise against the update `‖Q(W+d) − (W+d)‖ / ‖d‖` | neither | (it is the one that shows both failures, and nobody was recording it) |

The first metric said deterministic rounding was better and a lever was
proposed on it. The second refuted that and said the shipped stochastic path
was fine. The third showed the shipped path applies the turbo LoRA and injects
**twelve times its magnitude in noise**. Records:
`bench/results/2026-08-31_merge_realisation_{pdd,turbo}.json`,
`bench/measure_merge_realisation.py`.

**The generalisation is not "use more metrics".** It is that *a ranking is a
property of the metric until you have named what the metric is blind to*, and
the naming has to happen before the ranking is published, not after a peer
refutes it. The cheap discipline: when a measurement orders two arms, write
down in the same sentence what would have to be true for the loser to actually
be better. If that sentence is hard to write, the ranking is not ready.

**A subset preserves the centre and loses the tail, and that was measured
rather than suspected.** Same lane, later the same day: a 100-module half of a
200-module population reproduced the median to three decimals — 1.979 against
1.981 — and got the MINIMUM wrong by 3.2x, 0.877 against 0.274. The maximum
moved the same way on another arm, 24.889 against 27.313. So a subsample
agreeing on the headline is not evidence it agrees anywhere else, and the
statistics people quote from a sample (median, mean) are exactly the ones that
survive subsampling while the ones that bound the claim do not. **The loss was
structured rather than random, which is what makes it predictable**: the two
kinds easiest to sample were `qkv_proj` and `fc2`, and `qkv_proj` is the LEAST
affected kind on both partitions, so the bias had a direction and a cause. A
random subsample of the same size would have missed the tail too, but not in a
way that flattered the result. Ask what makes a module easy to sample before
trusting a subset of them. **A proposal to
replace the running full sweep with stride sampling was declined by the owner
on the day both sessions had already recorded that their small samples were
optimistic structurally rather than randomly.** He was right and the sweep is
the reason any of the tail figures here are real.

**What made it recoverable was a second implementation, not a second reading.**
Both corrections came from a peer session writing the statistic fresh rather
than importing it, and the agreement between implementations — worst-module
0.0199 against 0.01988 on different subsets — is what made either number
trustworthy. That is `CLAUDE.md`'s "re-reading your own work does not meet the
standard", and here it cost two published conclusions to relearn.

### Drive the expression, never restate it

**A check that RESTATES the expression it grades cannot fail when the
expression is wrong.** It is grading its own copy, and the copy agrees with
itself no matter what the code does. This is a specific and non-obvious form of
"a check whose input already satisfies the expected outcome cannot fail",
because the check looks like it is comparing two things.

It has now escaped twice in one lane on one day, which is why it is here rather
than in a commit message:

| the restated thing | what it hid |
|---|---|
| `check_pdd_sigmas::emitted()` recomputed `1 - block_bounds(...)` | dropping the `1.0 -` in `pdd_lora.py` left **every case in that file green** |
| the same file's refusal cases recomputed `num_steps % steps` | they asserted their own arithmetic, so no behaviour of the node could have reddened them |

Both were fixed the same way, and the fix names itself: **lift the expression
into a module-level function and have the check call it.** `emit_sigmas` and
`resolve_emit_steps` exist for no other reason -- neither is used twice in the
node. After the lift, dropping the `1.0 -` reddens three cases and a
non-dividing count reddens one.

The tell, when reading a check: if you can delete the implementation and the
check still passes because it computed the answer itself, it is not a check. Ask
where the number came from, and make it come from the thing under test.

This does not conflict with **prefer a control the check compares against** --
it is the same rule seen from the other end. A control is a second, independent
answer; a restatement is the same answer twice.

### A node tested on the queue has to be made to run, in the state under test

**A test that runs a node through the queue must make the executor actually run
it in the state under test.** The executor serves a node from its cache when its
class and inputs are unchanged, whatever its id, and runs each node once per
prompt, usually before any model of that prompt is loaded. Change an input to
force the run, and have the test report the state it ran in.

It escaped twice in one afternoon (2026-10-07), on one node
(`sam31_corrections.py`):

| the test | why it could not fail |
|---|---|
| the node's first acceptance on a queue: three rounds, an H3 sample between them | the node ran once per prompt before anything was loaded, then came from the cache; its fault needed it to run while its own earlier output was loaded |
| the step added for that fault: a second node of the same class on the same loader output | the same inputs, so the executor handed it the first node's output and never ran it; the step passed on the faulty file |

The second was caught by its control: the step was run against the node as first
served and passed, which said the step did not reach the fault. The fix is in
`bench/sam31_corrections_queue_test.py::rerun_graphs`: the second node differs
in one input, and the probe reports what the shared module showed before the
call, so a pass names the state it passed in.


## Running Sweep & Masking Details

**A sweep and a commit are two steps** (owner, 2026-10-05). The sweep runs in
the background; nothing is called green until its result has been read; and
the commit is a separate foreground command, never chained to the sweep with
`&&`. A chained commit lands on whatever the sweep's last line happened to
exit with, and a foreground sweep holds the session for its whole length.

**`--changed` is the quick form, and it widens itself.** It runs the checks
the working tree's changed files can turn red: a changed check, every check
whose source names a changed file, and the few that read the whole tree
(`bench/run_checks.py::ALWAYS`). When a node file is among the changes (a root
module, anything under `workflows/`, a vendored config, the harness itself;
`bench/run_checks.py::is_node_file`) it runs every check, because the rule above
asks for that and nothing short of reading the diff can say a root module's
change is not a schema change. It prints `TARGETED` or `FULL` and why each
check was chosen. `--since <commit>` adds the files changed since a commit.

It prints one row per check (exit code, seconds, and the line that explains
a result that is not green) and exits 1 only when a check is red that
`bench/checks_baseline.json` does not name. That file lists the checks that
are red for a stated reason, one line each; an entry whose check is no longer
red is printed as STALE and should be deleted. Exit 2 from a check is
"nothing graded" and is never a failure of the sweep. Every check is started
with `CUDA_VISIBLE_DEVICES=` and with `H3_CHECK_SWEEP=1`, which the one check
that would queue a render reads and refuses on. It does not run
`smoke_h3.py`, and it does not run the card cases: the checks that need the
card report 2, and running those unmasked on a free card is still a separate
step.

**No check needs `PYTHONPATH`.** Those that import comfy put the ComfyUI root
on `sys.path` themselves, and the rest never import it. Until 2026-10-05 this
section said exactly two needed it, `check_clone_v_wiring.py` and
`check_correctness.py`; a sweep run without the variable found
`check_reference_encode.py` needed it too. Each now calls
`bench/_lib::bootstrap`.

**`bench/_lib` is the shared helper.** `bootstrap()` puts the ComfyUI root on `sys.path` and core on its CPU path
when no card is visible; `needs(what, present)` exits 2 with "nothing graded:
needs ..." when a card, a dataset or an environment variable is absent; and
`case(name, fn)` with `finish()` print one `ok` / `FAIL` line per case and
return the exit code. A new check uses them. An old check keeps the harness it
has; nobody is asked to port one. Its module docstring is the reference.

**Run `smoke_h3.py` after any change to the generator.** It is the only check
that submits, and the class of bug it catches -- a graph every validator passes
and the server refuses -- is invisible to everything else here. Free the GPU
(`POST /free` with `unload_models`) before the CUDA checks or they OOM and the
failure looks like a regression.

**While a render is on the card, mask it.** Every check and the generator
run with `CUDA_VISIBLE_DEVICES=` (empty) while another process is rendering.
This box always has a card, so "run only the CPU cases" is not a choice the
scripts can make for you: the Sol checks gate on `torch.cuda.is_available()`
and walk into their kernel cases the moment a card is visible, and the
generator's import of `comfy.model_management` opens a CUDA context to size
VRAM before a single graph is built. Both reached a busy card on 2026-09-04,
once each, and the equivalence check's docstring carries the masked
invocation as the example:

```bash
CUDA_VISIBLE_DEVICES= python bench/check_sol_node_equivalence.py   # sink cases only
CUDA_VISIBLE_DEVICES= <comfy venv python> workflows/build_workflows.py
```

A check that needs the card says so in its exit code (2, nothing graded) when
masked; that is the expected reading, not a failure. The same code covers a
dataset, a directory or a server that is not there. The generator takes
ComfyUI's CPU path when no device is visible, and so does a check that calls
`bootstrap()`. These checks use no device and crashed in a masked sweep until
2026-10-05, because they imported core without that switch
(`check_audio_carry_inversion.py`, `check_channel_balance.py`,
`check_dit_prefix_attention.py`, `check_keyframe_canvas.py`,
`check_lowvram_handoff.py`).


## Retired checks

- **`check_void_conversion.py` and `check_void_conditioning.py`, deleted
  2026-10-05 with the VOID code, the day they were written.** The first held
  a checkpoint converter's key rename to core's CogVideoX model; the second
  held a conditioning node to upstream VOID's arithmetic, with core's own
  node as its control. The owner had VOID tested as a clean-plate tool for
  the masked lane, judged every clip, and had the code built for it removed
  (`docs/wiki/decisions.md`, 2026-10-05). Their subjects are gone, so
  nothing replaces them. The record of the test is
  `bench/results/2026-10-05_void_plate_turn.md`.
- **`check_workflow_schema.py`, deleted 2026-09-14 with the UI workflows.** It
  validated the saved UI graphs, positional `widgets_values`, against a live
  `/object_info`. Shipped workflows have been API format only since that day,
  so its subject no longer ships. `workflows/build_workflows.py::validate_api`
  grades every graph by input name against the same live schema before
  anything is written, and the owner's editor-saved graphs outside the repo
  keep `check_node_ids.py`.
- **`check_uncontrolled_claims.py`, built and deleted 2026-09-01.** It asserted
  that no row of the table below claimed "nothing" guards a numbered guide
  section some `bench/check_*.py` already names. Its escaped instance was real
  -- the camera-motion row said "nothing" for four days while this file's own
  check index listed `check_camera_vocabulary.py`. **But correcting that row was
  the fix; the check was not.** Measured across `docs/`: 132 lines claim nothing
  guards something, and exactly one cites a numbered guide section -- the row
  already corrected. So it scanned zero subjects and printed "this run proves
  nothing" on every run. Widening it to the other 131 needs judgement about
  prose, which means false reds, which this file refuses. **A check that cannot
  fail is not a cheap safety net, it is a green light nobody earned.**

## Uncontrolled requirements

**Next sweep: 2026-09-28.** At a sweep every row reading "nothing" is either
given a control, or re-stated as a deliberate decision not to build one with the
reason recorded. Per this file's own no-new-check rule most will correctly stay
"nothing" -- the sweep exists so that staying uncontrolled is a judgement
somebody made rather than a row nobody read. Adopted 2026-08-28;
[`sustainability.md`](../sustainability.md) argues the case.

The standing audit. For each imperative a person could violate in code, the
question is *what assertion goes red if someone ignores this?* `CLAUDE.md` is
swept; a full sweep of `docs/` is not done.

| requirement | enforced by |
|---|---|
| **`h3_trace.py`'s five route outcomes name the path that actually ran** -- `sage`, `sage_chunked`, `fallback_kernel`, `fallback_chunked`, `sol_delegate`, each written at its own exit in `attention.py` | **nothing.** Added 2026-09-08, and it earned the row the same day: the sibling field `module` was hardcoded `"dit"` at the only `record` call site while the patch replaces `Attention.forward` on a class BOTH `DiTBlock` and `RefinerBlock` instantiate, so every refiner call was labelled a DiT one and nothing noticed. That is an instance escaping, which is this file's bar for a new check. The outcomes themselves are still correct-by-construction: nobody has forced each of the five and confirmed where it lands. **The consequence is not cosmetic** -- a dispatch share computed from an unforced outcome field is a plausible number rather than evidence, and the sage fork intends to use exactly that share to decide whether kernel work on this model is worth doing. A control would drive each exit on a fixture and assert the counter, the way `check_sol_observe.py` already does for the OTHER recorder that happens to use the same word |
| **an armed Sol route record joins `/history` by `prompt_id`, labels every DiT block of every forward, and mislabels no non-DiT call** | **`bench/grade_sol_record.py`, by hand, on a record** -- it needs an armed render, so nothing runs it unprompted. First run 2026-09-01 on the canonical PDD graph as shipped: every assertion held (eight forwards, blocks 0-49 once each, routes as configured, counts and CRCs on `sol` rows only, prompt id known to `/history`). The row was "nothing yet" earlier the same day; what changed is that the render happened. The expected refiner and probe row counts are PARAMETERS of the grader because both proved graph- and run-dependent on that first record |
| **what is INSIDE a named model file is what the repo recorded** | **`check_model_contents.py`** since 2026-09-20, against `bench/results/model_contents_baseline.json`. A committed baseline the schema cannot regenerate, like `check_node_ids.py`'s manifest, which is what makes it a control. Nothing watched this before: name resolution was checked and contents were not |
| **no file a graph loads sets its own attention unseen** | **`check_model_contents.py`** since 2026-10-09: the header of every `.safetensors` a constant or a graph names, read for the key core's loaders act on. Core gained that key's Sol-Attn form and its LoRA route in the week of 2026-10-06; nothing here watched a file's header for it before, and a one-off scan was the only evidence that none carries it |
| a graph or constant names a model file that exists **and its loader owns that representation** | **`check_model_files.py`** since 2026-08-21 for discovery and 2026-08-23 for format ownership (since 2026-09-13: membership of `CORE_LOADED_ENCODERS`, the only representation a loader here opens). The deleted video VAE escaped the first gap; core offering but failing the custom W4A16 file escaped the original discovery-only claim. `workflows/bench/` is covered too |
| a constant ending in `_LORA` names a LoRA file | **`check_lora_alpha.py`.** Selects by suffix, resolves on disk, and **went red for real** on 2026-08-16 when a bool named `REF_VIA_LORA` matched. The positive control for this table |
| every check that walks graphs goes through `graph_paths()` | **`check_graph_discovery.py`.** Parses every check with `ast` and fails any that enumerates graph files directly. Exempts `check_ref_prompt_labels.py`, whose subject *is* discovery coverage. Was "nothing, satisfied by convention" until `226c53c` |
| **never rename a node's `node_id=`** | **`check_node_ids.py`, against `bench/node_id_manifest.json`** -- a committed baseline the schema cannot regenerate, which is what makes it a control rather than a tautology. Landed `1fe598b`. Its red harness was removed on 2026-09-11. This row read "nothing, and structurally unenforceable" until 2026-08-17, describing the state before that commit |
| **the seven load-bearing contracts in core's `MiniMaxH3ReferenceToVideo`** | **`check_reference_contracts.py`**, all seven since 2026-08-22 -- five that morning and contracts 4 and 5 the same day, each shown red by a red harness (removed 2026-09-11). This row read "nothing" through two postmortems, and then "five of seven" for part of a day; both are worth leaving visible, because the last two were held open by an assumption about where they could be asserted rather than by any cost |
| **a render's recorded marker arm is what actually bound to its CLIP, not what a node was told** | **`check_marker_arms.py`** since 2026-08-25. The requirement exists because the two arms are invisible downstream: a legacy tokenizer and mean-initialised marker rows change no shape, no timing and no output field, so a mislabelled arm produces a clean render and a confident sidecar. Enforced as agreement between the label and values re-read off the live CLIP, with the governing case being that two arms differing only in transform must not produce identical records |
| **the typed reference runtime owns order, source fps, soundtrack duration, mono normalization, and an atomic release-video policy** | **`check_reference_runtime.py`** since 2026-08-23, with nine mutations in `show_red_reference_runtime.py`; it drives a VHS-shaped lazy AUDIO mapping, pins release Qwen's raw-sample boundary, proves VAE/Qwen views remain separate and that `comfy` and `release` disagree on one input at both stages, proves that preflight reads the same bounds from the loader node that the guarded loader records (mutations M7, M8; the `encoder` policy and its stamped-contract arms left with the AWQ lane on 2026-09-13), and that `qwen_short_edge` gives the encoder its own view while the VAE keeps stage one, priced apart by preflight (mutation M9 feeds the Qwen view to the VAE). **`check_typed_reference_consumers.py`** keeps the chain visible to label discovery and preflight. Two 39-frame GPU smokes are green: all-media logged five presentation labels and 21,283 packed rows; release policy logged 960x544 → 1344x768 VAE preparation, four raw Qwen samples, and 23,892 packed rows. Both ran Sage and Sol sparse execution |
| **base-en Part One matches the actual I2VA/FL2VA/L2VA mode, final shot, and snapped duration** | **`preflight_graph.py`** parses the literal templates from the release guide and resolves graph values; `show_red_preflight_guide_split.py` made wrong mode, duration and final-shot variants red until `bench/red/` was removed in `0a764a33`; **that pointer no longer resolves and the Part One rule now has no control** (noted 2026-09-20). `check_speaker_id_control.py` is the live shape for one. Presence-only checking had accepted the I2VA sentence on an FL2VA prompt |
| **the four defects `MiniMaxH3Conditioning` exists to fix** | **`check_conditioning_behaviour.py`** since 2026-08-22, against core as the reference. The arms that matter are the DIFFER ones: a fix that silently stops happening leaves the node running and the graphs rendering, so it is asserted as a difference from core rather than against a remembered number. Carried as a forward item through **three** postmortems before it was built |
| ~~a prompt's camera motion comes from base_en 4.3's vocabulary~~, and a speaker is identified where they first appear (4.4) | **the motion half is CONTROLLED since 2026-08-28 by `check_camera_vocabulary.py`** (indexed above), and this row went four days without saying so — a table whose job is to list what nothing watches, contradicting the same file's own check index. Corrected 2026-09-01. The speaker-identity half is still uncontrolled and is **not mechanizable**. What the row said when it was written, and why it earned its place:  `preflight_graph.py` grades sections, markers, labels and the Part One template, and graded the shipped market t2v prompt GREEN while it carried `whip pan` (absent from 4.3's table), `tracks left` (conflating `Truck Left` with the separate `Tracking Shot`), `at medium amplitude and moderate speed` (neither value is in the table), and a speaker described a shot after he first appears. Found 2026-08-27 by reading the guide against a prompt that had already rendered badly, not by any gate. **The split is the point: motion vocabulary is decidable and unbuilt** -- a denylist of known-bad terms is cheap, proving every motion phrase in-vocabulary is harder -- **and speaker-identity placement is not mechanizable at all.** Two further items were listed here for part of an hour and are NOT rules: 4.6 constrains only "1-4 English sentences in one continuous paragraph", and it puts physical action sounds in the soundscape by name. Both were readings of the guide's examples, and a checker for either would have graded the market prompt green exactly as preflight did. The rules are restated above `LONG_T2V_PROMPT` in `workflows/build_workflows.py`, each marked STATED RULE or NOT A RULE |
| **the five undocumented markers nest one way only** (`docs/prompting.md` §7; written against `internal/PROMPTING.md` §7 blocks E and F, which was deleted 2026-09-01) | **`bench/preflight_graph.py::marker_rules`** since 2026-08-22, which is a report rather than a gate -- the same standing the rest of that file has, and the right one, because the prompts this rule most needs to reach are hand-built and never enter `graph_paths()`. Written the same hour as the requirement, so this row has never read "nothing". Its structural half is asserted; its judgement half (is this caption the right string) is not, and cannot be |
| anything patching another pack does it at `execute()` time and counts | **nothing.** Written 2026-08-16, same day, by the session that found the hazard |
| **a distilled LoRA is sampled on the sigma grid it was distilled at**, i.e. `simple` and not another scheduler ([`h3_config.py`](../../workflows/h3_config.py)'s `SAMPLING` note) | **`check_distill_grid.py`** since 2026-08-23. This row did not exist before that: `check_distill_settings.py` was reading as coverage for it, and grades the shift and the step count but never where the scheduler puts the steps. The escaped instance was a lightx2v turbo graph on `beta` at 4 steps, 0.10 off the distilled grid at the last step, while every check here passed (that lane is retired). What the check deliberately does not reach: the 16-step base graphs, which were never fitted to a step grid |
| **a split graph's two `MiniMaxH3SigmaShift` nodes are identical** ([`build_workflows.py`](../../workflows/build_workflows.py)`::_plain_model_chain`) | **nothing since 2026-09-26.** `check_distill_grid.py`'s split-arm case graded it from 2026-08-23 until the only split graphs (lightx2v turbo) were retired. No split graph ships; a new one needs the case back |
| **a PDD arm is consumed by `euler`, not a re-noising sampler** ([`docs/h3_pdd.md`](../h3_pdd.md), "Euler, not `er_sde`" -- stated there as a requirement and not a preference, because a fused head IS the block's mean velocity and one Euler step across the block integrates exactly that) | **nothing.** `check_distill_grid.py` grades the SCHEDULER and `check_distill_settings.py` the shift and step count; no check reads `KSamplerSelect.sampler_name` off a graph at all, and the node cannot see the sampler from where it sits. Found 2026-08-28 while auditing what the 0.83.0 SIGMAS rewiring does and does not reach -- it removes the scheduler and the step count as settable mistakes and leaves this one untouched. **No escaped instance:** every shipped graph carries `euler` today, and per this file's own rule that is the reason not to build the check yet rather than a reason to call it covered |
| **a converted PDD file's head bank is the published heads, in the encoding the converter assumes** | **`check_pdd_bank_encoding.py`** since 2026-08-28, and the guard runs inside `convert_pdd_lora.py` on every conversion rather than only in the check. Before it, the verbatim copy was unguarded end to end: no gate compared the bank to anything that was not derived from it |
| **an explicit PDD `steps` tiles the 32-point grid, and the graphs consume the schedule the node emits** | **`check_pdd_sigmas.py`** since 0.83.0, against ComfyUI's own `calculate_sigmas` for the numbers and against the shipped graphs for the wiring. This row could not have existed before the node emitted a schedule: the step count lived on a node downstream of the one that owns the grid, so nothing was in a position to refuse it |
| the ref2va checkpoint stays in `MODELS` | **nothing** found |
| **Sol-Attn is on by default in every shipped video workflow** ([`CLAUDE.md`](../../CLAUDE.md)) | **`check_attention_defaults.py`**, since 2026-08-18. This cell read *one graph, by accident* until then, and the objection it raised against building a real check -- that a hand-edited exception list is the shape that goes stale silently -- is answered rather than ignored: the single-frame class is derived from `GRAPH_DIRS` -- empty since that path was parked on 2026-08-27, and left derived rather than deleted -- and every remaining exemption must be NECESSARY, so an exemption that stops being true fails instead of covering a graph nobody reads. The check also pins the VALUES, not only presence, because a graph wired at the wrong `tau` or the wrong sage mode is a silently different experiment -- which mattered the same day, when the sage mode changed in every graph at once and nothing verified the rewrite took. **What it still does NOT cover**: graphs outside `graph_paths()`. The escaped instance that prompted all this was a hand-built graph under `internal/`, which no static check reaches; `bench/preflight_graph.py` reports on those, and reports rather than refuses |
| conditioning rows stay in the block population (`open_experiments` #18) | **`analyze_routing.py`'s `run_controls`**, which prints that `kcvar` spans every block centroid, conditioning included. Was enforced by nothing; a mutant computing `kcvar` over video blocks only passed every control the script had. Its red harness, removed on 2026-09-11, until 2026-08-17 had no expected-outcome comparison at all and so could not fail. **It needs a capture and exits 2 without one**, so it is a documented run rather than a guard anyone can execute |
| the connected-block figures in `sol_curves.py`, `docs/morton.md` and the node tooltip | **`analyze_canvas_geometry.py`'s `connected_frac`**, shown red by mutating it to 26-neighbour adjacency. It reproduces `docs/morton.md`'s published figures using a different implementation, so agreement is confirmation rather than a tautology. This row read "nothing, and no instrument exists" for one hour on 2026-08-17 -- transcribed from `docs/check_postmortems.md` instead of derived, which is the defect this table exists to find |
| the shared substrate readers have exactly one definition | **nothing, and it is half true.** `git_head` collapsed 2026-08-17: `provenance.py` imports it from `substrate.py` behind an adapter that preserves the stamp's `"not detected"` sentinel, so one implementation now does the reading and no stamp byte moved. Verified against both consumers, which load the file differently -- ComfyUI as a package member (node registers), and `check_provenance_stamp.py` by path under a bare module name (exit 0). That second one is why `provenance.py` carries a try/except ImportError pair: a plain relative import raises there, and the check catches its own import failure and returns 2, so it would have gone from green to silently skipped rather than red. **Still duplicated:** `_comfy_kitchen_version` and `_sage_version` do the same reads as `substrate._package` in a different shape -- logic, not a name, so nothing would flag it. That half is blocked on the schema rather than on effort: those strings are part of the stamp's format, so reshaping them must bump `STAMP_SCHEMA_VERSION` deliberately |
| timings are compared only at equal host GPU power state ([`docs/hardware.md`](../hardware.md)) | **nothing, and the shape differs per artifact.** Captures: `provenance.gpu_power_limit_watts` **exists** in [`docs/capture_manifest_schema.md`](../capture_manifest_schema.md) and the one manifest on disk populates it -- and as of 2026-08-17 it is required and asserted, so a manifest omitting it fails. **This row said `check_capture_manifest.py` inspects it nowhere; that was true when written and was fixed hours later in the same session.** What remains uncontrolled is the other two artifacts. Renders: the stamp has no power, driver or GPU field at all. Bench runs: persist nothing, and are the class whose numbers move most with power state. `bench/hwinfo.py` prints the limit and flags a non-stock one, but it is a report a person chooses to run, not a guard, and no check can see the power state of a run whose number is already written down. **This row asserted "the manifest has no field for host power state" when it was written on 2026-08-17; that was inferred from the `required` list without reading the properties, and a peer session refuted it the same day.** Corrected rather than deleted, because the correction is the point: the field existing and being unasserted is a different defect from the field being absent, and only the second one is fixed by adding a field |

## Gaps

One line each. The narrative behind items marked *(pm)* is in
[`docs/check_postmortems.md`](../check_postmortems.md).

1. **The `claims block` column is the record.** Where it reads `no`, "what
   breaks if this case is deleted" needs reading the assertions and inferring
   backwards.
2. ~~No runner.~~ `bench/run_checks.py` since 2026-10-05 gives the one-report
   run, per check. Still true: most scripts print their own `ok` / `FAIL`
   and there is no case registry, so there is no reliable case count.
3. ~~The `PYTHONPATH` split is invisible until it fails.~~ Closed 2026-10-05:
   every check that imports comfy bootstraps `sys.path`, the last three
   through `bench/_lib::bootstrap`. What is left of it is that most checks
   still carry their own copy of the two lines.
4. **`smoke_h3.py` cannot run unattended** -- it needs a live ComfyUI with
   models loaded, and a silently skipped check reads the same as a passing one.
   `check_distill_grid.py` answers this properly, by exiting 2 rather than 0
   when a control did not run. Worth copying.
5. **Controls whose input could not fail** -- four instances, one defect.
   `sol_curves.verify_adjacency` was the clearest, fixed 2026-08-16. *(pm)*
6. **The uncontrolled-requirement audit is one pass deep.** The table above
   covers `CLAUDE.md` only; `docs/` is unswept. *(pm)*
