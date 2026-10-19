# The Sol node redesign: plan, reasons, and the tests that decide it

**In short (as of 2026-09-27).** An audit of `MiniMaxH3SolAttn` found knobs that did nothing, two that overlapped, a default that disagreed with the shipped graphs, fragile plumbing (installed once, patching core process-wide) and one kernel bug. The owner approved a new node, `MiniMaxH3Sol`, and the deletion of the old one with every piece of code only it reached; that shipped, and Morton, top-k and `tau_profile` went with it (`docs/state/decisions.md`, 2026-09-27; [`../attention/sol_attn.md`](../attention/sol_attn.md)). The plan's reading that `qk_balance` was inert was wrong (test 1), and the default quantizer became `rotated` after test 2.

Opened 2026-09-27. The owner approved the approach the same day ("I'm good
with this approach"). Nothing here is built yet.

This page owns the plan and the reasons for it. The evidence is in
[`bench/results/2026-09-27_sol_node_compound_audit.md`](../../bench/results/2026-09-27_sol_node_compound_audit.md),
cited here as "the audit", by section. The node-against-node comparison is
[`bench/results/2026-09-27_attention_parity.md`](../../bench/results/2026-09-27_attention_parity.md).
Numbers live in those records, not on this page.

## Why redesign at all

The owner asked for an account of `MiniMaxH3SolAttn` as a compound system:
every knob, how the knobs combine, how they differ from core's
`BlockSparseAttention`, and which ones do nothing. The audit found five
problems:

1. **Several knobs do nothing, or nothing any more.**
   - `tau_profile` is wired in no graph.
   - The `2d_frame` and `hilbert` curves are measured worse and already
     deprecated.
   - Top-k selection is used in no graph, and its LoRA arm retired.
   - `pooled_tail` is True everywhere.
   - `dense_blocks` switches every other Sol knob off on its blocks (audit
     §4.1).
2. **Two knobs overlap.** `qk_balance` and `rotate` compose, and rotation
   does most of balance's work (audit §4.4). The shipped `qk_balance=True` is
   probably a no-op behind the dense tail, because its gate is recorded as
   opening only on the blocks we send dense. (Wrong, 2026-09-27: test 1
   found it changes every render, and the gate opens on some heads of every
   block. `../../bench/results/2026-09-27_sol_redesign_test1.md`.)
3. **One default is wrong by the owner's rule.** The node defaults
   `qk_balance` off and 129 of 131 graphs turn it on, which breaks "best
   defaults end to end".
4. **The plumbing is fragile.** The node installs its override once, so any
   attention node placed after it silently takes over. It also patches core
   process-wide (`PackedLayout.__init__`, `model._forward`), even though
   core now publishes the layout itself (audit §4.8, §7). Both shapes caused
   trouble before: the second is the one that got the VSA node parked.
5. **One kernel bug was found in passing** (audit §5.1; below).

## What is certain without a test

Each of these follows from reading the code or counting the shipped graphs,
so no measurement can move it:
- the balance/token-routing kernel bug;
- `dense_blocks` taking precedence over every other knob;
- top-k switching off `tau` and `tau_profile`;
- the zero-use knobs;
- install-once fragility;
- the process-wide patching;
- the default mismatch;
- sage's mode never reaching Sol's own calls (audit §9b);
- the prose that lost to code (audit §5.4);
- removing widgets in place would re-point values in every editor-saved
  graph.

## The design

**Ship a new node, `MiniMaxH3Sol` ("MiniMax H3 Sol-Attn"), appended to the
node list, then delete the old one** (owner, 2026-09-27: "delete").
Editor-saved graphs map widget values by position (`sol_attn_h3.py` (lines 1678-1681 as of 2026-09-27; the file was restructured 2026-09-27)),
so editing `MiniMaxH3SolAttn` in place would corrupt them. A new ID avoids
that. Once the generated graphs are rebuilt on the new node,
`MiniMaxH3SolAttn` and every piece of code only it reaches are deleted.
- **Cost:** an editor-saved graph that still uses the old node fails to load
  until the node is swapped by hand. The owner accepted that.
- **Removal is safe for the rest of the node list:** unlike insertion, it
  moves nothing that follows (`nodes.py`, the extension's node list).

**Inputs.** Each input is a decision, not a mechanism:

| Input | Form | Reason |
|---|---|---|
| `tau` | float | The only selection method left. The SLA lane closed on 2026-09-27, taking top-k, `keep_percent` and `pooled_tail` with it (the tail stays on, as every shipped graph has it). |
| `quantizer` | plain / balanced / rotated / balanced+rotated | Replaces two booleans that interact. The default is chosen by test 2: `rotated` (owner, 2026-09-27). |
| `dense_blocks` | text | Kept. Its default is chosen by test 2 for each dense chain, not carried over from the kitchen chain. |
| `sink_conditioning` | as today, including `exact_kv_and_all_rows` | Kept. Its default flips if pair C of `bench/sol_core_ab_arms.json` says so. Upstream makes every conditioning row dense (audit §9). |
| `token_routing` | off / measured blocks / all / custom (the list as a sub-input) | The list stops being a top-level string. The requirement "all" makes of the quantizer is re-derived after the kernel fix (test 3). |
| `start_percent`, `end_percent`, `min_tokens`, `verbose` | advanced | They mean the same as core's. |

**Retired from the node:**
- `tau_profile`;
- top-k with `keep_percent` and `pooled_tail`: the SLA lane closed
  2026-09-27;
- `morton` and `morton_curve`: Morton closed 2026-09-27 (see "Morton"
  below).

**Plumbing:**
- **Re-install the override on top at every step,** as core does
  (ON_PREPARE_STATE). That closes the silent takeover.
- **Read core's `minimax_h3_layout` and `block_index`** instead of patching
  `PackedLayout` and `model._forward`. The sink needs only the layout.
  Morton's hooks go with Morton. The node then patches nothing in core
  process-wide. That also removes a collision: the LongMedia pack patches
  `PackedLayout` process-wide too (bug #11).
- **Write the dense kernel into provenance.** The node stays chainable, so
  both dense chains keep working. It detects which dense kernel sits under it
  (kitchen int8, a sage mode, bf16) and records that in each render's
  provenance and log. The dense steps are where composition is decided (the
  coin beat in `bench/results/2026-09-15_block49_community_chain.md`), and
  nothing records that kernel today except by inference from the graph.
- **Raise on a kernel exception.** Today the node logs the exception and runs
  the dense kernel instead: a render that succeeds and is not a Sol render,
  which this repo refuses everywhere else.
- **Accept fp16 on the direct entry.** The kernel already takes it, and the
  bf16-only prose is wrong (audit §5.4).
- **Keep the direct entry, with per-call statistics.** Core's chunked entry
  carries statistics from the previous step. That measured further from the
  floor (`data/results/2026-09-10_sol_impl_capture_grade.json`), and the
  chunked entry takes no `qk_balance`. `MiniMaxH3SolChunked` stays the
  memory option; it engages only in the sage chain.

**Kitchen fork (`h3-frontier`):** fix §5.1 and add a test that tracks eager
for qk_balance with token_aug. Merge, rebuild, and record the new build
(`vendor/rebuild_kernel.sh`).

## Implementation stages

The work happens on branch `sol-redesign`, in a worktree outside
`custom_nodes/`, so no ComfyUI server loads half-done code. It merges into
`main` after tests 0 to 3, which run on the unchanged node. Each stage is one
commit on the branch.

1. **Segment bounds from core** (done: `h3_layout.py`). The sink,
   `h3_capture` and `sol_observe` read core's `minimax_h3_layout`, trusted
   only when the call is as long as the layout.
2. **The new node, `MiniMaxH3Sol`**, beside the old one in `sol_attn_h3.py`:
   - inputs as tabled above;
   - block index from `h3_layout.block_index`;
   - re-installs itself on top every step (ON_PREPARE_STATE);
   - raises on a kernel error;
   - takes bf16 and fp16;
   - names its dense fallback in the log, the settings record and
     provenance;
   - `provenance.SOL_CLOSURE_KEYS` matches its closure.
3. **The generator moves to the new node:** `h3_config` (its knob set
   renamed to the new inputs), `build_workflows.py`, and every check that
   reads the Sol widgets. Then rebuild all graphs and run the full check
   sweep.
4. **Delete the old node** and all code only it reaches:
   - Morton: `install_h3_morton`, the curves, `sol_curves.py`,
     `_patch_packed_layout`, `_SPANS`, `bench/check_sol_reorder_equivalence.py`;
   - `tau_profile`, top-k, `pooled_tail`;
   - `_install_block_index`, together with `sol_block` in its consumers.

   The analysis scripts that only reproduce closed-lane records are
   reviewed one by one, and deleted ones are cited by commit.
5. **Prose:** `docs/attention/sol_attn.md`, `docs/attention/morton.md`, the module docstrings,
   `CHANGELOG.md`, and `docs/state/decisions.md`.
6. **Merge to `main`, rebuild, restart ComfyUI, and smoke-render** one graph
   per mode.

## Retiring code (owner, 2026-09-27)

> "if you retire any parts of the node, be sure to do the same to the
> underlying code that it uses as well, since its all git tracked anyway."

For each retired input, the code only it reaches goes too: helpers, hooks,
the checks that pin them, and the prose that describes them. Records that
cite a deleted script keep citing it by the commit where it last existed.
That commit goes in the `docs/state/decisions.md` line for the retirement.

Starting map. `grep -rlw` over the pack, excluding `coderef/`, `internal/`
and `vendor/`. This is a starting point, not the final list. Each item is
re-traced when it is removed.

| Input | Code that reaches it today |
|---|---|
| `tau_profile` | `sol_attn_h3.py` (`parse_tau_profile`, the `block_tau` path), `sol_chunked_h3.py`, `provenance.py`, `bench/check_sol_kernel.py`, `bench/check_sol_observe.py` |
| `morton`, `morton_curve` (all curves) | `install_h3_morton`, `morton_perm` and `_perm_for` in `sol_attn_h3.py`; `sol_curves.py`; `bench/check_sol_reorder_equivalence.py` and the `morton`/`morton_curve` entries in `provenance.py`, `h3_config.py`, `build_workflows.py`, `bench_e2e_h3.py`, `check_sol_observe.py`. The sink moves to core's `minimax_h3_layout` **before** any of it is deleted: today `install_h3_morton` is also what gives the sink its layout. The curve rows below are the same retirement. |
| `2d_frame`, `hilbert` | `sol_curves.py`, `sol_attn_h3.py`, `sol_block_probe.py`, `workflows/h3_config.py`; the analysis scripts `bench/sweep_sol_orderings_on_capture.py`, `analyze_routing.py`, `analyze_capture.py`, `analyze_morton.py`, `analyze_canvas_geometry.py`, `gen_figures.py`, `_live_sol.py`, `bench_e2e_h3.py` |
| top-k, `keep_percent`, `pooled_tail` | the node path in `sol_attn_h3.py`, `build_workflows.py`, `h3_config.py`, `render_inventory.py`, `check_widget_deviations.py`, `check_sol_node_equivalence.py`. The kernel's `topk_ratio` stays: VSA and core use it. The capture instruments that sweep it are reviewed one by one. |

`vendor/sol_attn_minimax.py` is not ours. It is the pristine upstream
reference and stays byte-identical.

## Morton

**Closed 2026-09-27, so the node loses it** (owner: "if morton doesnt exist
anywhere or has no traction and you see no value it can go").

- **Upstream.** No H3 implementation anywhere reorders tokens:
  - core's only Morton code sorts meshes (`comfy_extras/nodes_mesh_postprocess.py`);
  - kitchen has none;
  - NVLabs' GB200 H3 profile lists per-call Morton as deliberately not done;
  - LightX2V refuses it for H3.

  It ships only for Wan and Hunyuan.
- **The one measured advantage** is lower Sol error at equal routed density
  on every captured cell (`bench/results/2026-09-17_sol_orderings.md`). That
  could only ever buy speed on Sol's share of a render. It was never tested
  as such, and the blind panel at the same tau found no defect removed
  (`bench/results/2026-09-18_sol_reorder_panel.md`).
- **The cost** is the node's most intrusive plumbing: process-wide
  `PackedLayout` and `_forward` patching.

Closed, not refuted: the capture result stands as a recorded property of the
`3d` order, and the records keep citing the deleted code by commit.

## Tests, in order

The test material (scenes, seeds, and graphs for t2v, a distill, i2va and
ref2va) comes from fastdude, per the owner. The owner wants i2va and ref2va
included to see whether shapes and activations differ by mode. The card is
free only after fastdude's int8-attention after-runs, and ComfyUI is
restarted on the new kitchen before anything runs.

0. **Determinism control.** Render the same graph twice and check the
   latents are bit-identical. `bench/results/2026-09-18_sol_reorder_panel.md`
   records renders repeating bit for bit, but on the old kitchen; this
   confirms it on `0.2.35+sol.863e953.up.c8c7825`. Every "does this knob do
   anything" answer rests on it.
1. **`qk_balance` on against off**, shipped graph, same seed, latents
   compared, in each mode.
   - **If bit-identical:** it is inert behind the dense tail, so the shipped
     graphs need nothing fork-only from Sol, and `quantizer` matters only for
     arms that change `dense_blocks`.
   - **If not:** some block below 45 opens the gate, and the node default
     becomes on.
   - Bit-identity is a yes/no answer for the prompts run. The gate depends on
     activations, so a second scene is what generalizes it.
2. **One capture session.** Done 2026-09-27: `../../bench/results/2026-09-27_sol_redesign_test2.md`.
   On PDD8, `rotated` is the better quantizer, but the choice barely moves
   total error. Kitchen int8 beats sage on the dense tail on every cell.
   Flipping the default is the owner's call.
   - Blocks 40 and 44 to 49, at steps 4 and 15, in each mode, on the
     current PDD8 graphs. (Blocks 44 to 48 were captured on 2026-09-19, but
     from a base-model render on older code. Those captures gave a
     provisional grade in which `rotated` wins:
     `../../bench/results/2026-09-27_sol_redesign_test1.md`.)
   - Graded offline for Sol under the four quantizers, against kitchen int8
     and sage rotated as the dense kernel.
   - It decides the `quantizer` default, whether the dense tail pays under
     each chain, which blocks belong in `dense_blocks`, and kitchen against
     sage on matched cells, which has never been graded (audit §9b).
   - Captures, not renders, because these are numerical questions
     (`CLAUDE.md`: a rendered clip cannot A/B a numerical change).
   - Grade each attention call on its own captured inputs, as
     `bench/grade_sol_quantizer_on_capture.py` does. A comparison taken at
     the DiT's output instead sits on a floor that the int8 blocks add
     downstream. There, noise across a 100x range of doses reads the same
     (encoderdude, 2026-09-27: `../../data/results/2026-09-27_encoder_quant_dit.json`
     and `..._encoder_quant_refiner.json`). An at-the-output number needs a
     tiny-noise arm beside it.
3. **The kernel fix, then a re-grade of token routing** per quantizer on the
   same captures. This decides the token-routing gate and whether routing
   becomes a default anywhere.
4. **Test renders, last,** only where captures cannot decide: the chain
   choice (the coin beat) and sink pair C. These are ordinary test renders
   for the owner to look at, not blind panels (owner, 2026-09-27: "I dont
   need blind renders just test renders"). Matched seeds, at least two
   scenes.

**Test material** (fastdude, 2026-09-27; checked before use):

| Mode | Graph | Before rows |
|---|---|---|
| t2v base | `workflows/h3_text_to_video_api.json` | `data/results/2026-09-26_distill_run.jsonl`, base arms (most scenes on bank text older than 0.154.6) |
| distill | `workflows/distill_experiments/h3_text_to_video_pdd_savelat_api.json` (PDD8, saves latents); `h3_text_to_video_flashgen_savelat` has the same coverage | `2026-09-26_followup.jsonl`; after on this kitchen: `2026-09-27_int8attn_after.jsonl` |
| i2va | `workflows/h3_first_frame_to_video_api.json` (canvas follows the keyframe; no Resolution node). Distill: `distill_experiments/h3_first_frame_to_video_pdd_api.json`, never rendered | none |
| ref2va | `workflows/h3_image_ref_plus_text_to_video_api.json` | none on this kitchen |

- **Scenes** (seed 730451892, on-length at 345 frames): slapstick piano,
  courtroom verdict, samurai duel, subway chase, and the `t2va_look_*`
  family.
- **Short on-length clips:** noodle bar (107), and `subway_chase_short` and
  the spec ladder (124).
- **Avoid for temporal measures:** scenes that ask for frame-to-frame
  brightness change (kpop, silent film, subway chase short).
- **Warmup only:** swimming lesson.
- **Prompt setup:** prompts are patched with `@bank:<id>`, and
  `MiniMaxH3Resolution.length` is set to the bank's frames, as
  `bench/make_followup_manifest.py` does.

**Capture tooling:**
- Captures are enabled by `H3_CAPTURE="dir=...,blocks=a:b:c,steps=x:y"` in
  the server's environment (`h3_capture.py` docstring), with `pre=1` to add
  the pre-norm qkv.
- The file tag `_ksol`/`_ksage` names the kernel that took the call. Sol's
  seam captures what Sol's kernel receives
  (`bench/results/2026-09-19_sol_seam_capture.md`).
- **Unverified:** whether the dense-tail blocks on the kitchen chain are
  captured at all, since they bypass Sol and go to core's backend. One short
  render checks that before the session.
- A cell at the trained canvas and full length is several GB; budget disk
  before a session.

**Gotchas that shape the runs:**
- **Waiting on a run:** wait on the runner's pid, never a `pgrep -f` match.
  Two runs interleaved on 2026-09-26 that way.
- **Determinism across processes:**
  - FastH3: confirmed.
  - FlashGen: only with the same load order.
  - PDD on the exact branch: unchecked. Test 0 covers it.
- **FastH3's contract graph never calls the dense backend,** so it is
  useless for tests of the backend or the quantizer.
- **Balanced arms keep token routing off** until the fix in bug #1.

What each result can claim:
- Tests 0 and 1 are exact for what they ran.
- Test 2 is exact on the captured cells and inference elsewhere.
- Test 4 is the owner's look at a few seeds, never certainty.

## Bugs and issues found, by reach into `workflows/`

Every entry is from the audit. The workflow census is the one in audit §1:
131 Sol nodes in 119 graphs, read with json over `h3_config.graph_paths`.

| # | Issue | Reach in shipped workflows | Status |
|---|---|---|---|
| 1 | **Kernel:** `qk_balance` + `token_aug` score in different spaces (the token stage's centroid is unbalanced, the keys are balanced) | None today: token routing is off in all 131. Live for the "all blocks" preset, custom routing under the shipped `qk_balance=True`, and every balanced token arm in `2026-09-15_sol_token_aug_x_options_b49_s15.json`. | **fixed** in kitchen fc32da2, installed as `0.2.35+sol.fc32da2.up.c8c7825` (CHANGELOG 0.159.2). The test fails on the old build and passes on the new one |
| 2 | **Kernel:** `sol_attn_chunked` with `rotate` and `topk_ratio` gets a threshold mixing two spaces | None: core never passes `rotate`, and ours never uses the chunked entry | open, fix with #1 |
| 3 | **Kernel (possible):** on the chunked entry, exact K is centred on the last step's mean and the tail on this call's | Core's H3 path only, and the VSA probe graphs that run core's node | unmeasured |
| 4 | **Node:** installs its override once, so an attention node placed after it silently takes over | None: every shipped graph puts Sol last. Any hand-built graph can break it. | fix in the redesign |
| 5 | **Node:** patches `PackedLayout.__init__` and `model._forward` process-wide whenever the sink is on | Every shipped Sol graph | fix in the redesign |
| 6 | **Node:** a kernel exception becomes a logged dense render | Any graph, whenever the kernel fails | fix in the redesign |
| 7 | **Node:** the "all blocks" gate requires `rotate` as well as `qk_balance`, but the record supports needing balance only | None (off everywhere) | re-derive after #1 |
| 8 | **Node:** default `qk_balance=False` against True shipped | 129 graphs deviate from the node default | **resolved**: `quantizer` defaulted to balanced, the shipped state, and has defaulted to rotated since test 2 (owner, 2026-09-27). Test 1 found it is not inert; whether it beats plain is graded on captures (`../../bench/results/2026-09-27_sol_redesign_test1.md`) |
| 9 | **Core:** `block_index` is never cleared, so refiner calls see a stale index | Output-neutral while `min_tokens` exceeds the refiner length | recorded, not ours to fix |
| 11 | **Node:** our sink patches `PackedLayout.__init__` process-wide, and so does the LongMedia pack (`coderef/ComfyUI-MiniMax-H3-LongMedia/motion_context_layout_patch.py`, which only knows how to defer to KJNodes' `._morton_h3` patch) | Any install with both packs: two process-global patches stacked on one constructor | fix in the redesign (the sink reads core's layout) |
| 13 | **Upstream kitchen tests:** 12 binding-validation tests expect errors no binding raises (at v0.2.35 and at upstream main). Their short-buffer cases then run the kernel into an illegal memory access that poisons the process. `test_topk_ties_over_select` also fails. | None in the pipelines: tests only. A kitchen test run must exclude or isolate them | recorded (CHANGELOG 0.159.2); upstream's to fix |
| 12 | **Provenance:** the stamp recorded every Sol setting as "not detected" from 2026-09-19 to 2026-09-27. It read one closure level, and the override has been a wrapper since the capture seam (b3a15bd1). | Every Sol render stamped in that window: output unaffected, the settings record empty | **fixed** on `main`, 0.159.1 (7fe005b9) |
| 10 | **Prose** that lost to code (audit §5.4) | n/a | correct with the redesign |

Anything the tests turn up is added here with its reach.
