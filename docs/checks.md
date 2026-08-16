# What this repo checks, and why

Index of every check in `bench/`. One row per script: what it defends, what it
needs to run, and whether it has earned trust.

There is no test suite and no runner. Each script is standalone, prints its own
`ok` / `FAIL` lines, and returns a non-zero exit code on failure.

**Last full run: 2026-08-16**, same box, ComfyUI restarted, live server, GPU
freed first. All `check_*.py` pass, including the five CUDA ones that had been
owed since 2026-08-14 (`check_correctness`, `check_clone_v_wiring`,
`check_short_edge_override`, `check_sol_kernel`, `check_solattn_correctness`)
and `check_single_frame`. `smoke_h3.py` passes on `h3_probe_sol_on_api.json`
(exit 0, all three composition lines).

That run mattered rather than being routine: the config had moved twice the
same day -- `LONG_LENGTH` 345 -> 362, and 32 graphs onto fl2va + the ref LoRA
-- so every one of those checks had been guarding a configuration that no
longer existed. It also caught a real collision: `check_lora_alpha.py` went red
because a new boolean named `REF_VIA_LORA` matched its `endswith("_LORA")`
filename filter and crashed it with a bare `TypeError`. Renamed to
`REF_LORA_ENABLED`; the check was doing its job.
**Partial run 2026-08-15**, when the single-frame path landed. `check_single_frame`
**does touch CUDA** despite living with the fast checks -- importing the H3
node module pulls in `nodes` -> `comfy.model_management`, which initialises the
device at import -- so free the GPU before it like the others. These pass
(`check_single_frame`, `check_keyframe_canvas`,
`check_schema_defaults`, `check_generator_constants`, `check_reference_fit`,
`check_ref_prompt_labels`, `check_prompt_guide_conformance`,
`check_workflow_schema`), the generator validated 73 graphs against a live
`/object_info`, and `smoke_h3.py` rendered the new image graph. **The CUDA
checks were NOT re-run that day** -- a render was resident and they OOM against
a busy GPU, which is indistinguishable from a regression. They are owed.

There are now **21 `check_*.py`** and **87 graphs** (68 in `workflows/`, 16 in
`workflows/image/`, 3 bench-stamped), all wiring the CUDA
Sol-Attn node.

**Plus one thing in this index that is not a `check_*.py` and is deliberately
excluded from that count**: `bench/preflight_graph.py`, listed last. It grades
and prices rather than gating, it is run by a person before a render rather
than by the suite, and its filename does not match the glob the count is taken
from. Counting it there would make a load-bearing sentence wrong for the fourth
time; leaving it out of the index entirely would hide a tool that has a red
control and defends real ground.

**Partial run 2026-08-16**, when the image graphs moved into
`workflows/image/` and their prompts were rewritten. Pass:
`check_ref_prompt_labels`, `check_prompt_guide_conformance`,
`check_workflow_schema`, `check_single_frame`, `check_distill_settings`,
`check_sol_kernel`, `check_generator_constants`, `check_schema_defaults`,
`check_reference_fit`, `check_retraction_consumers`,
`check_bench_matches_shipped`; the generator validated all 87 against a live
`/object_info`. Later the same day all 20 `check_*.py` were run together and
all pass.

**Third partial run 2026-08-16**, when the Sol-Attn docs were reorganised and
`check_doc_links.py` was added as the 21st. **16 of 21 run, all pass**:
`check_doc_links`, `check_retraction_consumers`, `check_generator_constants`,
`check_schema_defaults`, `check_reference_fit`, `check_keyframe_canvas`,
`check_distill_settings`, `check_lora_alpha`, `check_bench_matches_shipped`,
`check_provenance_stamp`, `check_override_routing`, `check_lowvram_handoff`,
`check_ref_prompt_labels`, `check_prompt_guide_conformance`,
`check_single_frame`, `check_sol_kernel`. **Five were NOT run and are owed**:
the four CUDA ones (`check_correctness`, `check_clone_v_wiring`,
`check_short_edge_override`, `check_solattn_correctness`) because a peer
session held the card, and `check_workflow_schema` because it needs a live
`/object_info`. That change touched no graph, no node and no schema, so the
five cover nothing it could have broken -- but "did not apply" and "did not run"
are different states and this file does not conflate them.

**Ten single-frame renders were submitted** to price `allow_upscale` and
`steps` (`open_experiments` #16e, #16g), which means three of the eight image
graphs -- `h3_image_edit`, `h3_image_style`, `h3_image_multiperson` -- have now
executed end to end rather than merely validating. **`smoke_h3.py` was still
not run**, and the other five image graphs remain unsubmitted, so for those a
green validator is still an unverified graph. `check_correctness.py`, `check_clone_v_wiring.py` and
`check_short_edge_override.py` were re-run too -- owed since the
`mode="fp16 (most accurate)"` flip, and owed again because `comfy_kitchen` is
now built from source rather than the pinned wheel. All pass. Free the GPU
(`POST /free` with `unload_models`) before the CUDA ones or they OOM and look
like regressions.

Three things this run found that no previous run could have:

- **`check_override_routing.py` and `check_lowvram_handoff.py` had been dead
  since 2026-08-13.** `456654d` added `from . import h3_trace` to
  `attention.py`; both checks import `attention` top-level, where a relative
  import raises. They died at import, nobody ran them, and the breakage was
  invisible for a day. `attention.py` now supports both load forms.
- **`smoke_h3.py` was broken in two directions at once** -- see its own
  docstring. One needle could never match; two assertions could never pass on
  a compliant graph.
- **`check_solattn_correctness.py` was grading Triton cross-mode** after the
  oracle was re-vendored, and passed anyway. See the note below.
- **`bench_e2e_h3.py` had been benching a sage config nobody ships**, since
  2026-08-13. See the note below; it is now covered by
  `check_bench_matches_shipped.py`.

The counts in this header have been wrong twice now -- once claiming twelve
checks and 24 UI graphs, once claiming 62 graphs after the total moved to 65. A
header that states a scope it no longer has is the same defect this file exists
to catch, one level up, and it keeps recurring because nothing checks it.

## The standard

From `CLAUDE.md`, and it is the reason the last column exists:

> A check here is not trusted until it has been shown to go red for the right
> reason -- break the thing it guards, watch it fail, put it back.

This is not theoretical. Three checks written on 2026-08-10 passed for the
wrong reason on first writing, and one reported zero failures with the bug
reintroduced. **An empty "shown red" cell is a finding, not a formatting gap.**

## Running them

Most need neither CUDA nor a model and finish in about a second.

```bash
# from the repo root
python bench/check_reference_fit.py

# three of them import comfy without bootstrapping sys.path themselves
PYTHONPATH=/path/to/ComfyUI python bench/check_clone_v_wiring.py
```

**The `PYTHONPATH` requirement is undocumented at the point of failure.**
Exactly three checks need it -- `check_clone_v_wiring.py`,
`check_correctness.py` and `check_short_edge_override.py` -- and they die with
a bare `ModuleNotFoundError: No module named 'comfy_api'` without it. Every
other check runs from anywhere: those that import comfy bootstrap `sys.path`
themselves, and the rest never import it. That inconsistency is a real
papercut and is listed under Gaps.

## The index

| check | defends | needs | claims block | shown red |
|---|---|---|---|---|
| `check_correctness.py` | the patched H3 forward against the stock one; mean relative error 0.0732 on both the eager norm path and the fused RMSNorm+RoPE path | CUDA, `PYTHONPATH` | no | not recorded |
| `check_clone_v_wiring.py` | `clone_v` reaches the forward, and only on modes that earn it | CUDA, `PYTHONPATH` | no | not recorded |
| `check_override_routing.py` | which calls the attention override sends to sage and which it declines, including the fallback when the kernel raises | - | yes | not recorded |
| `check_lowvram_handoff.py` | **more than its name says.** Three cases are KJNodes interop (the `[x]` list hand-off, the fallback with a list input, `minimax_head_chunks` honoured from `transformer_options`); **two are ours regardless of KJNodes** -- the plain tensor path, and that head chunking at 1/2/3/7 groups reassembles bit-identically | - | yes | not recorded |
| `check_schema_defaults.py` | every node's `io.Schema` defaults match its `execute()` signature defaults, for all 7 nodes. ComfyUI does **not** inject a schema default for an input a prompt omits, so the two are independent and a split means the UI and the API path see different values | `PYTHONPATH` self-bootstrapped | yes | **yes**, on the real `length` split, 2026-08-13 |
| `check_ref_prompt_labels.py` | every ref graph's prompt names **exactly** the labels its graph wires. The tokenizer derives `<Picture i>` / `<Video k>` / `<Audio j>` from the wired sockets, not from the prompt, so the two drift silently -- and a video's soundtrack takes `<Audio 1>` ahead of a standalone clip, which is easy to number wrong by hand. **Also owns discovery**: a case added 2026-08-16 asserts no directory under `workflows/` holds graphs the shared walk cannot see. `assert graphs` alone could not catch that -- a PARTIAL walk satisfies it too, demonstrated the same day when a stale `GRAPH_DIRS` made this file and `check_prompt_guide_conformance` both exit 0 while covering 20 ref graphs instead of 28 | - | yes | **yes**, both directions, 2026-08-13; the discovery case and four image-graph mutations 2026-08-16 |
| `check_prompt_guide_conformance.py` | every shipped ref prompt against the **official guide's own tables**, parsed at run time -- the six sections and their order, the `[...]` task-type vocabulary and its ` + ` combining rule, markers never crossing the visual/audio sets, and `<d>` only in `detailed_description`. Exists because `check_ref_prompt_labels.py` compares the generator to itself and so passed clean while all fourteen arms shipped a hardcoded `[reference generation]`. The `keyframe completion` case checks the GRAPH, not the vocabulary: the type is allowed only where the graph wires `MiniMaxH3AddGuide` (added to ComfyUI 2026-08-13, and merged with refs additively by `comfy/model_base.py`, so the mechanism now exists). Before that node it was rejected outright on the grounds that nothing could honour it -- reasoning that expired the day the node landed. Exits 2, not 0, when the guide is absent. Carries **two waivers**, `_STRUCTURE_PROBES`: `h3_probe_prompt_concise` and `h3_image_probe_format_flat` are unstructured on purpose, so their section and prefix cases are skipped **by name and printed on every run** -- markers, dialogue placement and label agreement are still enforced, proven by mutating them. **Two changes on 2026-08-16.** The blanket waiver on the image graph was replaced by a structural rule: `overall_soundscape` and `non_diegetic_music` are not required of a graph with no `VAEDecodeAudio`, read off the graph rather than off a name, so the other four sections are now graded on the image path where a whole-file waiver had been buying silence on them. And the "in order" case now actually checks order -- it compared against a list built by iterating the guide's sections, so it could only ever detect a missing one | the guide in `internal/` | yes | **yes**, six mutations incl. the fail-open guard, plus the waiver shown narrow, 2026-08-13; the narrowed waiver and the order case 2026-08-16 |
| `check_distill_settings.py` | **every** shipped graph, both forms: a turbo graph matches its LoRA's shift and steps, a base graph sits at the base checkpoint's 12/3, and the UI and API forms of each are paired and compared. Shifts *and* recommended step counts graded against the vendor README, not against itself. Exits 2, not 0, when that control is skipped | - | yes | **yes**, eight mutations, 2026-08-13 |
| `check_solattn_correctness.py` | Sol-Attn's **CUDA kernel** against the algorithm's own eager reference, cosine > 0.998, graded in the kernel's own measured `centroid_tail` mode. **Scope narrowed 2026-08-16**: the Triton arms went with the pack. They graded a kernel no graph had wired since 2026-08-14, and worse, the CUDA arm was coupled to them by control flow alone -- the script loaded Triton first and returned 2 on failure, *before* the CUDA arm was reached, so an absent pack silently disabled the only correctness check on the kernel that does run. The tail mode is now measured BEFORE the graded cases so they use the matching oracle. Exits 2 when there is no CUDA or no `sol_attn`, which is the expected state on a machine that has not built the fork | CUDA and a fork build of comfy_kitchen | yes | **yes**, 2026-08-16: the red control was shown red against a copy by pointing it at the same tau, where it reports cos 0.999919 and correctly fails "must differ". Numbers identical before and after the Triton strip, so removing those arms moved no CUDA figure |
| `check_bench_matches_shipped.py` | that `bench_e2e_h3.py`'s `shipped` arm builds the same sage and Sol settings the graphs actually wire, node for node. Exists because `check_generator_constants.py` pins the **generator** and nothing pinned the **bench** -- a gap that cost a day of numbers when the `fp16 (most accurate)` flip landed everywhere except here | - | yes | **yes**, 2026-08-14, by reintroducing the exact historical `mode="auto"` |
| `check_retraction_consumers.py` | that a retracted claim has not reached a file nobody signed off on. Reads the `retraction-consumers` block in `docs/evidence.md` -- a phrase per retracted claim plus the files allowed to contain it -- and fails on a hit anywhere else. **Deliberately an allowlist, not caveat-detection**: "is this mention caveated" is not decidable (`docs/bench_plan.md` legitimately contains `zero DiT calls` inside a RETRACTED bullet), and the same string can be two claims (`attention.py`'s `2.7x` is kernel speed, not the retracted accuracy figure). It answers the one decidable question, which is also the failure that occurred four times on 2026-08-14: all four consumers were files that acquired the claim *after* the retraction. **Does not defend the pairing case** -- `bench_plan.md`'s "one 345-frame video reference", where the reference length looked like the shipped config and hid a 362-frame target, has no matchable token, and a second reader found it | - | yes | **yes**, 2026-08-14, two controls: a new consumer added to `README.md` (red, named the file), and the ledger block deleted (red, refuses to pass with nothing to check). It also went red on its own first run, correctly -- it was scanning itself and the definition block, which is why both are now excluded |
| `check_doc_links.py` | that a doc does not point at a file or line that is gone. Two pointer classes: relative markdown links between docs, and `path:line` code citations. Exists because the 2026-08-16 `sol_engine_reference.md` -> `sol_upstream.md` rename broke `CLAUDE.md:41` **within an hour of a wholesale CLAUDE.md rewrite whose purpose was removing stale references**, and nothing said so -- a peer session caught it by eye. Takes paths on the CLI; with none it **walks the repo** rather than globbing a directory somebody remembered, which is the hole that let `workflows/image/` go ungoverned by two prompt checks. Resolves against the filesystem, not git, so brand-new untracked work is not a false red. `coderef/` citations warn rather than fail -- a machine that has not cloned diffusers is not broken. Skips `CHANGELOG.md` on purpose: a changelog records what was true then, and `CHANGELOG.md:455` correctly names a file that no longer exists. **Cannot tell whether a cited line still says what the citation claims**, which is why `docs/morton.md` pins a commit as well. Its `ambiguous_roots` case found a live one on its first run: `CLAUDE.md` cited a bare `nodes.py` with a line range, meaning ComfyUI's 2595-line file, and it resolved to our 194-line one -- in the paragraph warning that `import nodes` resolves to ours. Citations into ComfyUI's tree now carry a `ComfyUI/` prefix. **This row deliberately does not spell that citation in its own backticked form**, for the same reason `check_retraction_consumers.py` excludes itself: the file describing a bad pointer would otherwise contain one. **Scope widened the same day to `path::symbol`**, after a peer found six of those invisible to the `path:line` grammar. Five were the same ambiguous bare `nodes.py`, written that day. **The sixth was a diffusers path introduced in `6375763` on 2026-08-06 that had resolved nowhere for ten days**, in a doc read and edited repeatedly since -- so one pass caught new work and a stale committed defect together, and only the new work had a second reader looking for it. That gap was unreadable from a green run: the whole-repo pass said 34 citations and green while one doc contributed zero and held two ambiguous refs. **Not covered and correctly absent look identical unless the check prints what it examined**, which is why `parses_the_corpus` reports counts per grammar rather than only passing | - | yes | **yes**, 2026-08-16, eight controls. Five on the original grammar: a line past EOF (`citations_in_range`), the doc rename re-applied (`doc_links_resolve`, 4 files named), a one-file corpus with no references (`parses_the_corpus` -- it refuses to pass on silence), a full path cut to a basename (`no_bare_basenames`), and the `ComfyUI/` prefix removed (`ambiguous_roots`). Three more on the symbol grammar: a symbol absent from the file it names (`symbols_exist`), a bare `nodes.py::` (`ambiguous_roots`), and a symbol ref to a nonexistent file (`citations_resolve`). **Two failures worth more than the controls that passed.** One attempt appeared to prove the check inert and was a wrong grep pattern in the control. The other was real: the symbol pass was written *below* the reporting section, so it classified into lists already printed and the check reported green over a path resolving nowhere -- caught only because the diffusers path was known-bad beforehand |
| `check_provenance_stamp.py` | that `provenance.py` records the knobs that actually ran. Five cases; the load-bearing one is **`closure_is_read_not_declared`** -- two overrides built from the real `make_override`, differing in exactly one CUDA-only knob, must produce DIFFERENT recorded values. A key being present proves it was LISTED; only a value that MOVES proves it was READ, and the old version passed the former while failing the latter. Also pins that every `SOL_CLOSURE_KEYS` name is a real node parameter and that every node parameter is recorded -- the 2026-08-16 bug was wrong in BOTH directions at once, asking for three Triton knobs the CUDA node does not have (recorded as "not detected" forever, indistinguishable from a knob that was off) while omitting three that do run, `centroid_tail` among them. `version_bump_has_no_consumer` records that NOTHING reads `stamp_schema_version`, so the 1 -> 2 bump protects a future reader only, and fails the day a consumer appears without handling it | - | yes | **yes**, 2026-08-16, by reintroducing the exact pre-fix key list: both key cases go red and name the specific keys |
| `check_sol_kernel.py` | that the installed `comfy_kitchen` still carries `sol_attn`, that it is the CUDA backend and not the eager reference alone, and that its signature still accepts the kwargs our node passes. **The first check here covering a call INTO a dependency we do not control**, and it covers exactly one contract. Presence is gated on a graph wiring `SolAttnMiniMax`, because Sol is shipped OFF and absent is the expected state; ungated it exits 2. Also pins `SOL_CUDA_DEFAULTS` against the inputs the node declares -- parsed with `ast` rather than imported, so the check stays free of ComfyUI | - | yes | **yes**, 2026-08-14: `present` with the stock PyPI wheel as the control, `schema` by simulating an upstream rename |
| `check_lora_alpha.py` | that every `*_LORA` in `h3_config.py` resolves on disk, and that none of them hides a scale ComfyUI cannot see. ComfyUI reads `alpha` only from a `"<module>.alpha"` tensor and falls back to 1.0, **never** from the file's `__metadata__`; diffusers' #14408 documents a published H3 turbo LoRA carrying alpha 8 against rank 128 in metadata alone, which such a loader applies **16x too strong**, silently. **The first check here whose subject is a third-party binary** -- `check_prompt_guide_conformance.py` parses someone else's file but grades our prompts with it, `check_sol_kernel.py` grades a dependency's API. The exemption is the hard part: three kijai `_resized_avg_` conversions in this install carry a metadata `alpha` and are correct, because they also declare the scale baked into `lora_B`, so a declared bake outranks a declared alpha. Also pins its own premise by reading `comfy/lora.py`, so it fails loudly rather than quietly the day that loader grows a metadata channel. Both controls are synthetic and run every invocation, because every real file here is currently clean and an all-clean corpus cannot tell a working check from an inert one | - | yes | **yes**, 2026-08-16, five mutations, documented in-file: a dangling `REF_LORA` (red, and the scale case correctly declined to run), the unsafe branch deleted (`control:unsafe` red), the exemption widened by presence (`control:unsafe` red), the exemption removed (`control:baked` red, and three correct files misclassified), and a fake ComfyUI that reads `__metadata__` (`premise` red). A sixth mutation changed no verdict and is recorded as proving nothing |
| `check_single_frame.py` | that `single_frame.py`'s patch to ComfyUI's H3 nodes changes `length=1` and **provably nothing else**: pristine vs patched over the node's entire declared domain must differ at exactly one input. Also that 2-4 still snap to 5 (the region the relaxed clamp exposes), that the floor reaches `INPUT_TYPES`, that a module already supporting single frames is left alone, that an unrecognised module is refused rather than crashed, and that **every loaded copy** of the module is found -- ComfyUI registers `comfy_extras` files under a file-path module name and the directory has no `__init__.py`, so a dotted import makes a second, independent copy and patching one looks exactly like success. Also pins the invariant with the worst failure on this path -- **the one-frame VAE and `length=1` imply each other in every shipped graph, both directions** -- which nothing else covers: the image VAE in a video graph is the ghosting case its own README warns about, and the video VAE in the image graph is 22.04 dB against 37.27. Carries the only **LIVE** case in `bench/`: it asks the running server what floor it reports, which is the sole surface that can tell a patched module from a patched *copy* of one. That case has THREE states, not two -- no server, deliberately disabled (read from the serving process's environment, since `/object_info` cannot tell "off" from "broken"), and wrong -- and the first two exit **2**, never 0 and never 1 | `PYTHONPATH` self-bootstrapped, **and CUDA**: importing the H3 node module initialises the device, so free the GPU first. The LIVE case wants a running ComfyUI and says loudly when it is skipped | yes | **yes**, 2026-08-15, three ways: a mutation breaking the T=1 boundary at `n <= 4` with the internal guard disabled (red on 4 cases, naming lengths 1-4); a CONTROL case that hands `apply()` a deliberately wrong implementation and asserts it refuses leaving the module untouched; and the review's own mutation of a *helper* at `n == 200`, which the first baseline could not see at all -- it held references into the module being patched, so both sides ran the patched helpers. Baseline is now re-executed from core's source and the same mutation fails with "2 changed: [1, 200]" |
| `check_keyframe_canvas.py` | canvas derivation, plus the aspect and duration rules | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_reference_fit.py` | reference image sizing against both upstream rules, and that the stock resize becomes a no-op after our node | `PYTHONPATH` self-bootstrapped | yes | not recorded |
| `check_short_edge_override.py` | the reference short-edge override applies once and never leaks | `PYTHONPATH` | no | **yes**, documented in-file |
| `check_generator_constants.py` | the workflow generator reads upstream constants rather than repeating them | `PYTHONPATH` self-bootstrapped | no | not recorded |
| `check_workflow_schema.py` | saved UI graphs against a live ComfyUI `/object_info`, type-checking widget values positionally. Takes paths from the CLI, so it is the one walker that cannot read `GRAPH_DIRS` -- pass `workflows/*.json workflows/image/*.json`. Exempts `LoadImage` values carrying a subfolder from the combo-membership test (2026-08-16): that combo comes from a non-recursive `os.listdir`, while the node's own `VALIDATE_INPUTS` checks the filesystem, so this file was rejecting graphs the server renders | **live ComfyUI**, or `--object-info` cache | no | not recorded |
| `preflight_graph.py` | **not a gate -- a report, run by hand before you queue.** Grades a prompt against the guide's mechanical rules (ordinals derived from sockets, every defined label cited in `detailed_description`, one retention line per label, markers never crossing the visual/audio sets, no `(Sx)` in `retention_analysis`, `<d>` placement and language tag, cut times inside the clip) and prices the packed sequence statically. Takes paths, never globs, so hand-built graphs are gradeable -- which is why it exists: the two prompt checks only ever saw `workflows/*_api.json`. **Reports, never refuses**, because a tool that blocks you in your own repo gets disabled. Names what it cannot count: a video reference is a floor, not a budget. Imports `wired_labels`, `_audio_sections_optional` and `_STRUCTURE_PROBES` rather than restating them -- the first run without the last two reported FAIL on 7 of 8 image graphs for sections that structurally cannot apply | reference images on disk; no CUDA, no model, no server | no | **yes**, 6 mutations 2026-08-16: wrong ordinal (both directions), `(Sx)` in retention, visual marker on an audio label, audio marker on a visual label, a timestamp on `[Shot 1]`, `<d>` with no language tag. A 7th appeared green and was **my mutation failing to apply** -- the source said `partially_copy` where the patch expected `reference`, so nothing changed and the check "passed" having tested nothing. Caught only by diffing the mutant against the source; recorded because it is the trap this file exists for |
| `smoke_h3.py` | the H3 chain composes and runs, after any node-pack update. **The only thing here that actually POSTs a prompt**, and on 2026-08-13 it was the only reason anyone discovered every API graph was unsubmittable -- `validate_api` was asserting the wrong shape, so no static check could have found it | live ComfyUI, GPU, model | no | n/a, it is a smoke test |

"claims block" means the file carries a `Claims, i.e. what breaks if a case is
deleted:` header enumerating what each case defends. Eight of fifteen do.

### Citations that cannot resolve, for `check_doc_links.py`

The ledger that check reads, kept here rather than in the script for the same
reason `retraction-consumers` lives in `docs/evidence.md`: the enumeration
belongs beside the prose it governs, and nothing gets a second copy.

An entry asserts that someone looked and the target is *deliberately* gone. It
is not a way to silence a broken link.

```doc-link-absent
PATH: _morton.py
WHY: the Triton pack's Wan Morton file, deleted with the pack on 2026-08-16
     (commit 6872dfd). docs/morton.md quotes its docstring as the only stated
     payoff for reordering anywhere in either pack, so the citation has to
     stay. Recover from github.com/kijai/ComfyUI-SolAttn_triton at 842c4ea.

PATH: _morton_h3.py
WHY: same pack, same deletion. Cited for the opposite reason -- it is the H3
     variant and makes NO quality or sparsity claim, which is the whole point
     of the comparison it appears in.
```

### A note on `check_lowvram_handoff.py`

Its name undersells it, and the name is why it looks droppable. KJNodes'
`MiniMaxLowVRAMAttention` does three things, and we already own one of them:

| what their node does | ours? |
|---|---|
| head chunking via `minimax_head_chunks` | yes, already our widget |
| block-level `h` release (the `[x]` hand-off) | no, the only additive piece |
| `sol_take_forward` so Sol-Attn keeps the low-VRAM path | no |

The division is currently clean and deliberate on their side: their
**attention** patch yields to ours (`if attn_key in m.object_patches:
continue`), while their **block** patch is unconditional and unguarded. If we
ever write our own block-level release, both packs would write
`diffusion_model.blocks.{idx}.forward` with no marker convention and
last-node-wins silently -- the collision class `reference_fit.py`'s
`_WRAP_MARKER` exists to prevent. **Decided 2026-08-13: keep the split, do not
reimplement.** The interop cases stay because that boundary has already
produced one real bug (the `clone_v` regression at `head_chunks=4`).

### A note on `check_solattn_correctness.py`: updating an oracle changes the check

Re-vendoring `bench/_sol_attn_reference.py` on 2026-08-14 (`ad9a4a8` ->
`c04ef20`) added `centroid_tail`, defaulting **True**. The Triton kernel has
no such parameter and runs the per-row mode. So the moment the oracle was
updated, every Triton case was grading the kernel against a different
algorithm than the one it implements -- and **all of them still passed**,
because the two modes differ by cos 0.9988 and the bar is 0.998. The bar was
looser than a whole-branch change to the algorithm.

Three things worth keeping from that:

- **Nobody edited a case, and the cases broke.** The defect entered through a
  dependency the check trusts. A check is only as pinned as its oracle, and
  the oracle here is deliberately something we do not control.
- **It passed, which is the bad outcome.** Had it gone red the re-vendor would
  have been examined immediately. Passing is what let it sit.
- The fix was not to tighten the bar but to **measure which mode each kernel
  is on** and grade it against that. The mode is now printed on every run,
  for both kernels, because the source does not document it and reading the
  kernel to decide would be an inference where a measurement was available.

The general form: when an oracle gains an option, every assertion against it
inherits a new case, exactly as CLAUDE.md says an "off"/"absent" state does.

### A note on the bench: an error that flatters nothing gets missed longest

`bench_e2e_h3.py` hardcoded `"mode": "auto"` on its sage node. The
`mode="fp16 (most accurate)"` flip on 2026-08-13 changed `h3_config.py` and
every shipped graph and did not change the bench, so from that day every e2e
arm was compared against a baseline nobody runs.

The instructive part is why it survived. `auto` resolves to `fp8_cuda++`, the
**fastest** kernel. A fast baseline makes every competing arm look *worse*, so
the bug produced conservative numbers. Nothing looked too good to be true,
which is the signal people actually check for. **A bug that overstates gets
caught; one that understates does not.**

`check_generator_constants.py` already enforced "read the shared constant,
don't repeat it" -- for the generator. The bench was never in scope, and the
bench is where the numbers come from.

## What is deliberately not checked

`docs/open_experiments.md` is the other half of this document: seven things
this repo has decided **not** to measure, each with its cost, the decision it
would change, and the actual blocker. Read it before proposing a new check --
several obvious ideas are already there with a reason attached.

A suite of twelve **render** scenes is designed but not run. They are quality
gates for output, not code checks: each carries a pre-registered binary claim
and a predicted per-arm direction, and a human watches and listens. Nothing
executes them, and nothing here can -- judging whether a third shot happened
or two voices stayed distinct is not a `check_*.py`.

> Those live in `internal/`, which is **gitignored and not distributed**. If
> you cloned this repo you do not have them, and that is deliberate: they are
> the owner's working research notes, not shipped content. Everything this
> document describes is in `bench/` and is present in the clone.

Two things there have no code check and cannot get one until the graphs exist:

- **Two-stage split graphs do not exist yet.** When they do, the checks worth
  writing are: both stages read one schedule, stage 2 carries `DisableNoise`,
  the split point is inside the step range, and the finish-stage LoRA's shift
  matches the shared schedule. That last one is `check_distill_settings.py`
  extended, not a new file.
- **ref2v with an fl2v distill LoRA is out of distribution by construction.**
  All three turbo LoRAs are `fl2v`; the vendor lists ref2v distillation as
  future work. It is on the test matrix deliberately as an experiment, at
  varied LoRA strength and as a two-stage split. It must not be validated as
  a supported pairing, so `check_distill_settings.py` deliberately does not
  police which task type a turbo LoRA is loaded into.
  `docs/h3_ref2v_distillation.md` works out why it resists distillation, what
  to expect, and what failure to look for.

## Gaps

Ordered by how much they undermine the standard above.

1. **Ten of fourteen have no record of having been shown red.** Only
   `check_short_edge_override.py`, `check_distill_settings.py`,
   `check_schema_defaults.py` and `check_ref_prompt_labels.py` document their
   own calibration. For the rest, the repo's central trust standard is
   unverifiable from the artifacts. This does not mean they are wrong -- it
   means nobody can tell.

2. **Seven of fourteen have no claims block**, so "what breaks if this case is
   deleted" is not recoverable without reading the assertions and inferring
   backwards. The seven that have one are the model to copy.

3. **No runner.** Every script prints its own ad-hoc `ok` / `FAIL`, with no
   shared harness and no case registry. There is no way to run everything and
   get one report, and no reliable way to count cases.

4. **The `PYTHONPATH` split is invisible until it fails.** Three scripts
   require it and give a bare import error; six do not. Either all of them
   should bootstrap `sys.path`, or none should.

0. **Run `smoke_h3.py` after any change to the generator.** It is the only
   check that submits, and the class of bug it found -- a graph the validators
   pass and the server refuses -- is invisible to everything else here.

5. **`check_workflow_schema.py` and `smoke_h3.py` cannot run unattended.**
   Both need a live ComfyUI, and `smoke_h3.py` needs the models loaded too, so
   both are absent from any headless pass -- and a check that is silently
   skipped reads the same as a check that passed.
   `check_distill_settings.py` is the only one that answers this properly, by
   exiting 2 rather than 0 when one of its controls did not run. That pattern
   is worth copying.

---

## Write the evidence kind inside the claim, not beside it

Not about a check, but about how a claim in this repo stops being true.

On 2026-08-13 an upstream finding arrived explicitly labelled unverified — a
read of somebody's source, not a build and not a measurement. It was written
into `attention.py` as "on inspection, is not", with the label dropped. Nobody
asserted anything false at any point. Each hop repeated the previous hop's
confidence and left the caveat behind, because a trailing "(unverified)" reads
as the sender hedging rather than as part of the claim, and hedges are what
get trimmed when text is copied.

The wording that survives a copy-paste states what kind of evidence it is
*inside* the sentence:

> **Reported, not verified:** the sm89 kernel appears already stride-aware on
> its output … That is a source read from upstream, **not a build and not a
> measurement**.

against the version that does not:

> …which reads as out of reach and, on inspection, is not. The sm89 kernel is
> already stride-aware on its output.

Both are honest when written. Only one is still honest after somebody quotes
half of it. This matters here more than in most repos because measured
numbers, upstream source reads and analytical estimates sit in the same
paragraphs, and six months later they are indistinguishable by tone.

### `SageChainAssert`'s call-time case cannot see sage

Found 2026-08-13 by removing Sol-Attn from a graph and watching the assert
fail for a reason unrelated to what changed.

`_exercise` pushes one tensor through the composed attention and requires a
routing counter to move. The counter it reads is resolved by scanning loaded
modules for a callable named **`sol_attn_stats`** (`assert_chain.py`)
— Sol-Attn's counters. `attention.py` exposes no counter of its own; the only
state it publishes is `reset_fallback_state`.

So on a sage-only graph the probe runs, sage routes it, nothing named
`sol_attn_stats` moves, and the node reports "the composed path was not
taken". Sage is fine. The instrument cannot observe it.

**Confirmed from the log, not only from the source.** The arm that passes
prints `[h3] chain assert, call-time: routed as sparse=1` — `sparse` is
Sol-Attn's counter name. The arm that fails prints the sage patch line
(`50 attention modules patched`) and no `[sol_attn]` lines at all, then
fails. Both halves of the diagnosis are visible in one run.

The inverse is the part that matters for graphs we actually ship: when the
assert passes at call time, **what it confirmed is that Sol-Attn routed the
probe**. It says nothing at call time about sage, which is the node it is
named for. And because Sol-Attn's module is imported process-wide whenever the
pack is installed, `sol_attn_stats` resolves even in graphs that do not use
it — so the check cannot distinguish "Sol is not in this graph" from "the
composed path was not taken".

This is the same check that, per the note at `assert_chain.py`, "ran
registration-only from the day it was written until 2026-08-11, and said so in
a line nobody read, under a final `chain assert ok`". The 2026-08-11 fix
closed the registration-only gap and wired the new case to the wrong module's
counters.

**Consequences, in order:**

1. The sage-only configuration is not merely unmeasured (open experiment 9),
   it is currently **unrunnable** with the shipped assert in the graph.
2. Every "routed as …" line in this repo's logs is a statement about Sol.
3. The fix is **not** a counter of our own, which was the first plan. The
   sage fork already exports `get_last_dispatched_kernel()` and
   `KNOWN_KERNEL_NAMES` as public API, set on every sage call including the
   sm89 fp8++ path. That proves routing *and* identity in one read, so the
   assert can require "landed on fp8_cuda++" rather than "something moved" —
   the claim this node's name has always implied and never made.

   **Two preconditions, both of which would otherwise reproduce today's false
   negative.** The value is `threading.local`, so the probe and the read must
   happen on the same thread: fine while `SageChainAssert` runs as a graph
   node, *not* fine if anyone moves it to an HTTP-side check, where it would
   return `None` and read as "sage did not route". And it is last-dispatch,
   not a count, so it must be read immediately after the probe.

   It also needs a reset to be sound. Without one the check reduces to a
   before/after comparison that is conclusive in one direction only: a change
   proves routing, but an unchanged value does not disprove it, since the
   probe may route to the same kernel a previous call already recorded and the
   thread-local persists across prompts on one worker. That failure mode is a
   **false negative on graphs that route consistently** — the same defect being
   fixed, wearing a better API. `_reset_dispatch_for_test` exists but is
   explicitly not public; upstream is promoting it through their downstream
   symbol process so it acquires a removal checklist. The repair waits for
   that rather than importing an underscore symbol.

**FIXED and verified 2026-08-13**, without needing the reset and without any
   new contract surface. The probe now fires on a **fresh thread**: the
   dispatch value lives on a `threading.local`, so a thread that has never made
   a sage call returns `None` by construction. The thread-locality that was the
   hazard becomes the mechanism.

   Two things the verification itself turned up:

   * **The off-thread probe does traverse the composed forward** — the open
     question when this was designed. Confirmed by the log: at 4608 tokens it
     produced `[sol_attn] sparse (1, 4608, 56, 128)`.
   * **That first attempt still failed, and correctly.** At 4608 the sparse
     patch *takes* the call and runs its own kernel, so sage never runs and the
     new check truthfully said so. The right probe size **inverted** when the
     instrument changed: the old counter check needed a probe large enough for
     the sparse kernel to fire, the new one needs a probe small enough for the
     sparse patch to decline, so the call falls through to sage. That is the
     composition claim this node is named for — *sage handles what the sparse
     patch does not* — and it had never been the thing being tested.

   The probe now reads the gate's own `min_tokens` from `transformer_options`
   and sizes to half of it, so lowering that threshold in a graph cannot
   silently push the probe back above it.

   **One probe was still not enough, and the reason is the same shape again.**
   The sparse gate *falls through* to our patch whenever it declines
   (`take = gate is not None and ...` then `return patched_forward(...)`), so a
   call reaching sage is consistent with two different worlds: composed and
   healthy with the gate declining, or composition dead with the gate never
   engaging. A small probe reports green in both — evidence that cannot
   separate "working as designed" from "the mechanism is absent", which is
   precisely the counter bug it replaced.

   It now fires a **pair**, pinning the gate from both sides:

   | probe | requirement | proves |
   |---|---|---|
   | below `min_tokens` | must reach sage | the fall-through works |
   | above `min_tokens` | must **not** reach sage | the gate is live and taking |

   The second assertion is sound *only* because of the fresh thread. `None`
   normally means "cannot tell"; on a thread that has made exactly one call it
   cannot mean anything else, so `None` after a large probe is positive
   evidence sage did not route it. The mechanism adopted for the baseline
   turned out to license the negative too.

   It also refuses to default a missing `sol_compose`. An absent key *is* the
   dead-composition case, so substituting 4096 would size a probe against a
   gate that is not there and call it green. Present → sparse expected; absent
   → sage-only, and the message says which was verified.

   Verified live, both configurations, and they are now distinguishable:

   ```
   composed:  sage routed a 2048-token probe on fp8_cuda++ and correctly did
              NOT get the 4608-token one, so the sparse gate at 4096 is live
              and sage is taking what it declines
   sage-only: sage routed a 2048-token probe on fp8_cuda++; no sparse patch
              published `sol_compose`, so this graph is sage-only
   ```

### The same defect pointed inward

Within an hour of writing the rule above, the same failure recurred in the
other direction. A number had been flagged — correctly, and by me — as
config-dependent and needing re-derivation per config. Two messages later it
was used as a known input to a solve, and the result pre-registered as a
prediction.

Nothing careless happened in between. **A caveat accepted about someone else's
number does not attach to your own later use of that number**, and no normal
process makes it attach: the caveat is filed as a fact about the old claim,
while the new claim is being built somewhere else. That makes it structural
rather than a lapse in attention, which is why "be more careful" does not fix
it any more than it fixes caveat decay.

The counter that seems to work: **when a caveated number becomes an INPUT,
re-read the caveat as a precondition of the new claim, not as history attached
to the old one.** If the caveat says "re-derive per config", then a solve
using it is blocked until that derivation exists — the same way a missing
argument blocks a call.

Worth pairing with a second habit from the same incident: check that the
quantity you are about to measure is the one that enters the model. That solve
was reformulated from a step count to a wall-clock share, and the instrument
already planned would have returned a precise value for the abandoned
variable — a real measurement of the wrong thing, which is harder to notice
than no measurement at all.
