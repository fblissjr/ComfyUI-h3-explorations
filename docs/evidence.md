# What holds, and what does not

Last updated: 2026-08-20.

`docs/checks.md` indexes what is *checked*. `docs/open_experiments.md` indexes
what is *not measured*. This file is the third case and the one that kept
biting: claims that **were** measured, are written down as numbers, and should
not be relied on — because the conditions they were taken under are not the
conditions we ship.

An entry leaves this file when someone re-measures it, not when someone
re-reads it.

## Relationship to `SOLATTN.md`'s header

Both were written on 2026-08-14, in parallel, by two sessions that did not
know the other was doing it — which is its own entry in the postmortem.

The split, so neither drifts: **SOLATTN.md's header is the Sol-Attn quick
reference**, read by someone about to quote a number off that page. **This
file is the repo-wide ledger** and owns the three things a topic page cannot:
claims that are not about Sol-Attn (the fp8/fp16 accuracy figure is a sage
claim; the smoke's prompt substitution is a harness one), the **what restores
it** column, and the environment section below — the venv and `coderef/` are
not in git and nothing else records what changed there.

If the two disagree, this file is wrong until proven otherwise: SOLATTN.md is
maintained by whoever is running the measurements.

## Why this file exists

On 2026-08-14 two sessions working the same repo found nine separate claims in
this class, and **not one of them was caught by whoever wrote it**. Every one
came from a second reader. The pattern was identical each time: a number
measured under one configuration, carried into another, with the qualifying
sentence living in a different file — or in the same file, one paragraph away
from the table.

That is caveat decay, and `docs/checks.md` already names it. What it did not
have was somewhere for the *surviving* list to live, so each caveat was
attached to whichever page happened to discuss it, and none of them were
attached to the number.

---

## Do not rely on these

| claim | why not | what restores it |
|---|---|---|
| **"the six-section prompt format protects the speech in a reused soundtrack"**, and any reading of the 2026-08-22 swap-prompt arms as a result about prompt format ([`bench/results/2026-08-22_swap_prompt_verdict.json`](../bench/results/2026-08-22_swap_prompt_verdict.json)) | **Never established; the write-up that claimed it was corrected the same day.** Three prompts -- structured, concise, and a community imperative arm -- were rendered on identical references, canvas and length. Four renders landed before the batch was stopped: structured at two seeds, the others at one each. The owner found exactly ONE of the four clean on speech and it was a structured render -- **but that arm's own second seed was not clean**, so the split does not follow the prompt. One clean render of four, with the only within-arm pair disagreeing with itself, is a draw from a distribution: `CLAUDE.md`'s different-sample rule applies to prompts as much as to numbers. The decision to delete the imperative arm was the owner's and stands on its own; the mechanism story does not. **What the batch DID surface is that reused-soundtrack speech degrades often at 1152x768 / 124 frames, across registers**, which is a question and not about prompt format. Five mechanisms are ruled out on inspection rather than suspected -- fps desync (`force_rate=24` is wired against a 25 fps source), soundtrack span expansion (the recorded patches show `frame_load_cap=124`, so `ref_audio_t` 207 against a video span of 206.667 is a tie), grid-snap loss (124 is exactly `17*7+5`), the 2 fps conditioner subsample (correct at 24 fps and confirmed identical to sglang), and gap 8's mean-vs-sample (`comfy/ldm/minimax/audio_vae.py:427-443` has a `mean_proj` and no logvar head, so there is nothing on the audio path for sglang's sampling to differ from) | **ANSWERED 2026-08-22, same day: the length was the substrate, not the prompt.** The shipped graph rendered UNPATCHED -- 362 frames at 1024x768, seed 730451892, the same seed and prompt as the damaged `structured_00001` -- and the owner judged its speech good ([`bench/results/2026-08-22_swap_length_probe.jsonl`](../bench/results/2026-08-22_swap_length_probe.jsonl)). The bad render differs from it by nothing but the bench patch: `length=124`, `1152x768`, `frame_load_cap=124`. **So the four-arm prompt comparison was run on a substrate that damages speech in every arm, and it measured the bench setting.** Two caveats stated rather than buried: length is NOT isolated, since the canvas moved with it (1152x768 -> 1024x768) and only one render exists at the good setting; and **the soundtrack trim shipped that day gets no credit** -- the bad run already carried `frame_load_cap=124`, so VHS had capped the audio to 5.167s correctly and the trim node's absence was a no-op. **Re-run and MEASURED the same day** ([`bench/results/2026-08-22_swap_prompt_verdict_362.json`](../bench/results/2026-08-22_swap_prompt_verdict_362.json)): both shipped graphs unpatched, three matched seeds each. On soundtrack reproduction the structured arm beats concise in **all three pairs** -- margins over control 0.63/0.53/0.43 against 0.23/0.07/-0.11 -- and concise at seed 894 scores BELOW its own control, meaning it resembles a mismatched window of the same recording more than the one it was conditioned on. **Confirmed by ear the same day**: the owner watched the set and reports the concise arm is broken speech **every single time, 3 of 3** -- which is the order the measure had already produced without hearing anything. So the arms are separated by a listener and by an instrument that agree, on matched seeds, on the shipped substrate. **Two findings, and merging them was the first reading's error**: concise is *gibberish*, a prompt problem, broken from the first third on two of three seeds where every structured render starts at 0.59-0.70. Structured's *late* drift is a separate issue and not about the prompt -- it starts strong every time and falls in the last third, which the owner attributes to a 15.083s render cutting a 19.56s utterance mid-delivery. That second one is untested and **cannot be settled by rendering longer**: 362 frames / 15.083s is the longest length H3 was trained on, per `MiniMaxH3Resolution`'s own tooltip and `h3_config.LONG_LENGTH`, and the vendor's reference pipeline stops a grid step earlier at 345. So the fix is on the INPUT side -- a source longer than 15.083s is always truncated, and for soundtrack reuse it should be cut to fit at a phrase boundary before it becomes a reference. Testing the mechanism needs a source whose speech ends inside 15.083s, compared against this one at the same length. **And still confounded**: structured and concise differ in structure AND length AND whether the soundtrack is a retention line or prose, so **the SECTIONS are not established as the cause**. What is established is narrower and still useful: this concise prompt breaks speech reliably on a soundtrack-reuse task. The separating arm -- six sections with the soundtrack demoted to prose -- is what would attribute it |
| **the 2026-08-22 selection run reporting that top-k helps the SLA LoRA and hurts v1.1** (`bench/results/2026-08-22_sol_selection_verdict.json`) | **VOID, same day, by replication.** It reported a sign flip of about 20 points, resting on an SLA adaptive-tau arm at 150.05s. The identical configuration -- same LoRA, same selection, same prompt, same kernel build, same seed -- re-run after a server restart gives 124.5s and 124.4s. **What makes it void rather than noisy: every arm in that run but one is ~20% slower than its counterpart in the replication, INCLUDING the top-k arms, whose density is fixed by construction and cannot vary with model or prompt.** A knob that cannot move, moved. That is the substrate changing, not the knob. Two candidate mechanisms are ruled out rather than suspected: Sol override accumulation, because the node's own `composed with N patched attention forward(s)` is a constant 50 across every render in both sessions; and thermal, because the earliest arms in that run were the slowest and a later arm the fastest, which is backwards for accumulating load. **The mechanism is unknown and is left unknown here.** | nothing restores that run. The question it asked is answered by [`bench/results/2026-08-22_sol_selection_verdict_subway.json`](../bench/results/2026-08-22_sol_selection_verdict_subway.json), n=2 per arm on a freshly started server |
| **any timing measured on a long-lived ComfyUI session** | Generalised from the row above on the day it happened, so treat it as a working rule and not a measured law. **Two independent measures moved together across a restart on identical inputs**: sampler seconds 150.1 -> 124.5, and the 400 Hz audio peak 23.1 dB -> 16.1 dB on the same prompt, LoRA and seed. A session that has been rendering for an hour is not the same substrate as a fresh one, and nothing in the bench path records which you were on. It also retroactively explains a +6.9 dB audio shift blamed the same day on the kernel swap and the tokenizer patch: the 2026-08-20 baseline of 16.2 dB matches the fresh-session 16.1 dB, and both are innocent | restart before timing an arm set. A control would be the same arm timed at the start and end of a long session; nobody has run it |
| **any capture manifest's `attention.sol_attn`, spelled here once as `bypassed_for_capture` so the phrase has a home** | **FIXED 2026-08-17, same day; this row is why the fix exists.** It WAS a constant: `bench/generate_capture_manifest.py` assigned it as a literal and nothing reassigned it, and the script did not mention `SolAttn` anywhere, so it wrote `bypassed_for_capture` whether Sol ran or not. It is now derived by `_sol_attn_state`, which reports `absent`, `orphaned` or `wired`. **Any manifest generated before that date still carries the constant and must not be read as an observation.** Verified 2026-08-17: the graph it was generated from **does** contain a `SolAttnMiniMax` node, orphaned by a hand-edit rather than absent, and the field reports the same string either way. Sibling defect in the same function: `:133` writes `"rank": 256` for every LoRA it finds, correct for the one LoRA this repo ships and silently wrong for any other. Both are the failure `CLAUDE.md` names — a check, or a record, whose input already satisfies the expected outcome. **It was cited as evidence** that a capture ran Sol-free, in `docs/drift_frontier.md`, which is corrected in place. The graph copy stored beside a capture is real evidence for that question; this field never was | make it read the graph — but note the correct read is **reachability from the output node, not node presence**, since the node here is present and orphaned. A naive "is `SolAttnMiniMax` in the graph" fix reports this capture backwards |
| **any capture manifest's `attention.sage_mode`, `models.clip`, and any `models.loras` array that is empty or missing a loader** | **FIXED 2026-08-26, same day. The same function and the same dict literal as the `attention.sol_attn` row above, nine days later.** `sage_mode` was the literal `"fp16 (most accurate)"`, overwritten only when a `MiniMaxH3SageAttention` node happened to be in the graph -- so every graph that runs no sage reported a mode for a kernel that never ran, and the constant did not even match `h3_config.SAGE_NODE["mode"]`, which is `auto`. **Any manifest generated before that date carries the constant and must not be read as an observation.** Separately `models.loras` was built by matching `LoraLoaderModelOnly` alone and by PRESENCE: a graph running a `MiniMaxH3PDDLoRA` or `MiniMaxH3TurboLoRA` recorded `loras: []`, and an active-but-unconsumed loader of any class would have been recorded as one that ran. **An empty array asserting no LoRA was loaded is worse than an absent field, because it reads as a measurement.** **Consumers enumerated 2026-08-26, which is what makes this row done rather than filed:** no shipped doc, check or result quotes either field -- `bench/check_capture_manifest.py` grades presence and type, never the value; `bench/check_provenance_stamp.py` and `substrate.py` derive their own; the 282 manifests under `internal/v2_run_2026-08-25/` are the comparator's `h3-crossstack-layer50-v1` schema and carry neither field (confirmed by the peer session holding that lane). So nothing downstream moves, and the exposure is captures taken before today read later. **A sixth field joined this row hours later and is the reason the class-level fix exists:** `models.clip` matched `CLIPLoader` only, which NO graph this repo ships uses -- every one loads the encoder through `MiniMaxH3AWQEncoderLoader` -- so **every manifest ever generated here emitted `clip: ""`, asserting no text encoder.** Nobody found it by reading; the sentinel emitter found it on its first run, which is what distinguishes converting the class from fixing instances. | both are derived now -- `sage_mode` reports `absent`/`orphaned`/`wired` like its sibling, and LoRAs are collected by reachability across `h3_config.LORA_LOADER_CLASSES`. Manifests are stamped `1.3.0`, so the version distinguishes a derived record from a defaulted one. The class-level fix is the sentinel emitter below |
| **Sol vs sage = 1.611x** | Wrong on **two** axes, not three: an **fp8** sage baseline the graphs do not ship, and sage running 5 of 16 steps inside a Sol arm so the fp16 fix moves **both** arms. **The length objection is WITHDRAWN 2026-08-14** — 362 is trained; only the reference pipeline declines it | rerun with the shipped fp16 baseline; length was never the problem |
| **`centroid_tail` = 2.5% e2e** | Ours, two runs, 0.1% spread, at 362 — **which is fine.** Retained here only because it shares Run 1's fp8 baseline | rerun on the fp16 baseline |
| **Upstream's `centroid_tail` ~5–10% e2e** | Upstream conversation, his box, his settings. Never reproduced here, and our own 2.5% disagrees | reproduce, or drop the figure |
| **CUDA is 1.4x over Triton e2e** | Upstream conversation. Never reproduced here | a paired run on this box |
| **`centroid_tail` is "~1.4x", or "the biggest speed knob in the node"** | **Tripwire row, and the trap has now caught two readers.** The node's own tooltip says "~1.4x faster" and that figure is the **operation**, not end-to-end. Ours measured **2.5% e2e** (row above), upstream reports 5–10%. Against the shipped frontier — sol 1.20x, +int8 1.39x, tau 1.6 1.53x — `centroid_tail` is the **smallest** knob in the node, not the biggest. `docs/SOLATTN.md` recorded itself making this exact conflation on 2026-08-14; an analysis on 2026-08-16 made it again and built an argument on top of it (that a token ordering earns its keep by making the "1.4x knob" safe to leave on — the mechanism is real, the prize is 2.5%). **The tell is the unit**: a knob's op-level speedup and a config's e2e speedup share a denominator with nothing. **Deliberately NOT added to the `retraction-consumers` block** — "1.4x" is correct where it appears (the node tooltip, upstream's CUDA-vs-Triton figure, `SOLATTN.md`'s knob table), so hunting the string would fire on every correct use and train readers to skim the output. Same shape as the "2.7x" note below: one spelling, two claims, and only the pairing is wrong. | nothing to restore — the correct figures already exist. This row is here so the next person who reads the tooltip finds the correction first |
| **`reuse_qkv_memory` saves nothing** | Not a negative result. The VRAM column was reporting torch-active bytes and could not have seen it | fixed instrument (`f1dff99`), rerun |
| **any peak-VRAM figure between 13:08:49 and `f1dff99`** | An external write reverted the device poller on disk; a `git add -A` then committed the reverted state | re-measure |
| **any fp8-vs-fp16 sage accuracy ratio**, spelled here once as **"2.7x more accurate"** so the tripwire below has something to match | **WITHDRAWN 2026-08-16 by the owner, and deleted rather than caveated** — the only entry in this table handled that way. This row is now the sole place the phrase appears in the repo. Two competing ratios had accumulated (a synthetic `torch.randn` sweep run here, and a smaller figure reported secondhand from the sage fork's captured activations) plus the `mean_rtol` values behind both. Ruled out on provenance, not on size: the sweep measures an input distribution H3 does not have; the real-activation figure was never re-derived here and its script is not committed in the fork, so it was an uncommitted ad-hoc run cited across a repo boundary; and nothing in `bench/` uses captured activations, so a rerun today would reproduce the synthetic instrument rather than replace it. **The decision this row said was unaffected has since been REVERSED — `SAGE_NODE` ships `auto` as of 2026-08-18.** The withdrawal was not what reversed it. What did: the surviving perceptual verdict was taken at 124 frames with Sol-Attn **absent**, one day before Sol landed, so it graded a configuration this repo no longer ships — with Sol on, sage runs only the steps outside the sigma window. Cost measured in [`bench/results/2026-08-18_attention_defaults.json`](../bench/results/2026-08-18_attention_defaults.json); reasoning in `workflows/h3_config.py`::`SAGE_NODE`; what would reverse it again is a blind paired judgement at the shipped config, whose clips are rendered and named in that file. **This row is kept, not rewritten, because the sentence it used to carry is the lesson: a verdict can be sound for what it measured and still not transfer.** **`attention.py` and `README.md` keep a "2.7x" that is a *speed* figure against torch's flash backend — a different, correct claim. Two numbers, one spelling; do not sweep it.** | **RESTORED 2026-08-18, by the run this cell asked for.** [`bench/results/2026-08-18_sage_accuracy_on_capture.json`](../bench/results/2026-08-18_sage_accuracy_on_capture.json), produced by `bench/grade_sage_on_capture.py`, grades every sage mode against a **float64** reference on the 2026-08-17 reference-heavy captures — the pair this cell named as unused. Read the ratio there, not here. Three things worth knowing before anyone quotes it: it is **smaller than the withdrawn synthetic figure**, which is the direction the withdrawal predicted and the reason provenance beat size; the advantage **decays with depth while the absolute error grows**, so fp16 helps least where the kernel is worst; and it is **kernel fidelity, not perceptual quality** — the unit trap on this same page. fp32 was not good enough to be the reference and the file says why. The harness refuses to report unless a deliberately wrong scale is caught first |
| **"the LoRA arm carried the reference's studio lighting onto his cheek"** | **Retracted 2026-08-16, same day, by the author.** The OBSERVATION is sound and triple-confirmed (owner blind, me from stills, Gemini blind): one arm lights the man with a hard key shadowing his right cheek, the other softly and evenly. The EXPLANATION was invented. `ref_1` is soft near-frontal studio light with no defined shadow edge anywhere on the face — **there was no shadow to carry**. The premise was never checked before the mechanism was built on it, which is the failure this file exists for, committed hours after writing a doc about it. Gemini's reading is the better one: a hard key is defensible as low golden-hour sun through the window, so the lighting difference is two valid interpretations of the brief rather than one arm failing it | nothing restores it. What survives from that render is parallax coherence and identity equivalence, both recorded in `docs/roadmap.md` |
| **any reference measurement taken before `_ref_prompt()` is fixed** | **RESTORED 2026-08-17:** Step 0 prompt refactor implemented and negative-guarded (`bench/check_ref_prompt_labels.py`). Step 0b paired blind re-renders on seed 42 and seed 137 verified 5.0/5.0 identity retention on both arms and equivalent planar depth between checkpoint and LoRA. LoRA adapter confirmed canonical across all 18 reference workflows. | Restored 2026-08-17 via Step 0 & 0b blind protocol |
| **"AdaLN is replaced, not adjusted, in every block" / `final_layer.adaln_proj` is "essentially rewritten" (rel_delta 1.92) / "the hybrids are a linear dial"** | **Withdrawn 2026-08-20.** Every one of these compared `adaln_proj.linear.weight` between the checkpoints directly. They are curve-form checkpoints: what a block consumes is `adaln_t_table[t] @ W.T + b`, the two files were factorised separately, and their bases agree in sign on columns 0-3 and are flipped on 4-7 (per-column cosine +1, +1, +0.996, +0.996, -0.9997, -0.9997, -0.99, -0.99). The large-norm coefficient columns sit on the flipped basis columns, so a coefficient comparison returns rel-delta ~1.9 with negative cosine for a modulation that is nearly the same. The per-block "delta energy" profile and the hybrid coverage fractions (91.1% down to 81.3%) were the squared norm of that sign flip. Consumers, all corrected in place: `bench/analyze_checkpoint_delta.py`, `docs/roadmap.md` (three places, one of them the 2026-08-16 measurement that already had the right 1.1-4.7% figure under the wrong mechanism), `docs/h3_ref2v_distillation.md`, `workflows/h3_config.py` (two comments), `bench/analyze_ref_lora.py`, two generated-graph notes in `workflows/build_workflows.py`. The 2026-08-18 session log and postmortem are annotated, not rewritten. The 2026-08-18 record carries a `retraction_2026-08-20` key naming the withdrawn fields; its linears, norms, scales and global rows stand and were reproduced exactly by the rewrite | **Replaced the same day** by [`bench/results/2026-08-20_dit_internals.json`](../bench/results/2026-08-20_dit_internals.json): at the modulation output the parents differ 1.4-4.7% per block, 5-9% in the time-varying part (cosine 0.996-0.999), ~12% in the final layer's time-varying part -- the same order as the ~3% int8 linears. Nothing at the weight level singles adaln out |
| **Sol is 0.999919 accurate** | Implementation fidelity, not total error — the harness compares kernel against reference **at the same tau**, so the sparse approximation is on both sides and cancels. Also `T=512`, and the O(T²) reference cannot run at real length | the Sol-vs-dense diagnostic (`44becf0`), once the card frees |
| **text = 38 rows, sequence = 12,264** | `smoke_h3.py` substitutes **both** the prompt (27 words, against the graph's 216) and the length. Not a scaled-down shipped graph — text does not scale with length and was replaced | one preflight on a shipped graph, unmodified |
| **everything derived from that**: audio dominates the sink; text is the whole v2 narrowing; `sink_q` start is 0 on t2v | all smoke-harness statements. On the shipped graph, text extrapolates to ~304 rows / 4 blocks — still an extrapolation from one point | same preflight |
| **the sink's audio framing** | A **t2v** framing. In reference-heavy graphs the sink is overwhelmingly *reference* rows, and a video reference's failure mode is motion drift, not the thinness argument the knob is named for | measurement at reference load; the bench has no `--refs` axis yet |
| **reference-load table: 35.1% / 57.9%** | Wrong on three axes — v1 formula (v2 stops running reference queries dense), 362 frames, 1344x768. The shipped reference arms were 345 at 1024x768, and are 362 since 2026-08-16. **Half redone 2026-08-16**: `docs/SOLATTN.md` now carries a v1-vs-v2 table, with `A` measured from `temporal_shape` and the v1 column reproduced from the doc's own formula as a control. 57.9% becomes **35.6%**, and the 23-point swing becomes **0.5 points**. Still derived — each row's `S` is recovered from the published `exact_kv` percentage rather than counted. **This retraction had existed since 2026-08-16 and three claims in `SOLATTN.md`'s own prose went on asserting the withdrawn numbers regardless** — a retraction in a ledger does not retract its consumers. **Attempted and abandoned the same day, with the reason:** `bench/count_packed_rows.py` makes video and target audio exact and `docs/h3_references.md` gives measured reference costs, but recomputing `S/T` from those reproduces the published `exact_kv` column on only 2 of 5 rows (1.4% vs 1.5%, 29.5% vs 29.6%) and misses badly on the video-reference rows (49.4% vs 35.1% at 345f). **The published `S` values are not reconstructible from anything committed** — the gap is the text estimate at heavy reference load, which needs the encoder. So this row's label cannot become "measured" by arithmetic; it would take a render at each configuration | **nothing, and it no longer matters — see the next paragraph.** The conclusion drawn from this table does not depend on `S` |
| **"with Sol on, sage gets nothing"** | Retracted 2026-08-14. Reasoned from `min_tokens` and forgot the sigma window. Sage runs 5 of 16 steps | — corrected in place |
| **`min_tokens` 4096 is "very likely wrong"** | Retracted, and the retraction's reasoning was itself corrected 2026-08-14: it is one `optimized_attention` call in source but **52 modules** through it. Conclusion survives — the 50 DiT blocks run at the full packed length, above both thresholds, and the 2 refiner blocks run at the text span, below both — so 4096 and 12288 still select the same thing | — corrected in place |
| **"fp16 lands on the steps where precision matters most"** | Inference, not measurement. `start_percent` has never been measured at any length on either backend | measure `start_percent` (Run 2, not started) |
| **any bench progress read from its own stdout mid-run** | The warmup `print` lacks `flush=True`, and `tee` makes stdout block-buffered, so finished lines sit in the buffer. Read ComfyUI's progress lines instead | add `flush=True` |
| **Morton is "worth 1.16x alone", at "94% GPU utilisation"** | Retracted 2026-08-16. Triton, 362 frames, stacked on int8 — correct for what it measured, and not a description of the CUDA backend, where the permutation is free. It had been the standing argument for `morton=False` in `h3_config.py` and in `docs/SOLATTN.md`'s Configuration findings, and the two disagreed for several hours | a CUDA Morton arm at fixed tau, if anyone wants a cost number at all |
| **the CUDA replacement, "0.8 s of 861, or 1.0009x"** | **Do not quote this either**, and it is the more instructive of the two. It is one arm of a two-arm control presented as the isolated number: the dense pair moved +0.8 s and the sparse pair moved −1.2 s, i.e. opposite signs, both at or under the bench's measured run-to-run spread on one run per arm. Morton-on cannot be faster, so the pair is measuring noise. The correct claim is unquantified — **free** | more runs per arm, if the cost is ever worth pinning. Nothing currently depends on it |
| **any A/B of a numerical knob that rests on one rendered clip per arm**, including the 2026-08-13 comparison that chose `fp16 (most accurate)` | **Not controlled evidence about the knob.** The sampling trajectory diverges completely from any perturbation on any sampler, so the changed arm is a *different sample*, not a degraded version of the same one. Sound as a preference between two outputs; never a result about the kernel. Ranked against it, the call-level measurement from `bench/grade_sage_on_capture.py` is the stronger claim, which is the opposite of how the two were weighted at the time | many seeds per arm judged blind in aggregate, or a call-level comparison on captured activations |
| **"the DiT is sensitive to the seven H3 marker rows"**, and the 0.135 relative-L2 figure in [`bench/results/2026-08-27_marker_epsilon.json`](../bench/results/2026-08-27_marker_epsilon.json) read as evidence about markers | **Withdrawn 2026-08-27, the same day, by the control it never had.** Replacing the seven marker embedding rows with the table mean moves the DiT's denoised prediction 0.135 video / 0.148 audio at step 8, at fixed noise, latents and sigma. That number has no null distribution, and a peer session caught it before it was quoted anywhere. The control ([`2026-08-27_marker_epsilon_control.json`](../bench/results/2026-08-27_marker_epsilon_control.json)) applies the SAME substitution to non-marker rows, matched on what the marker arm actually perturbs -- **16 of 408 prefix positions, not seven rows**, because only two of the seven occur in the prompt and each occurs eight times. At step 8: markers 0.135/0.148, `arb_profile` (2 ids, 16 positions) 0.116/0.212, `arb_draw_1` (11 ids) 0.241/0.430, `arb_draw_2` (8 ids) 0.289/0.418. **The markers sit at the bottom of the control range, not above it.** Two things make this a clean negative rather than a weak one. The controls were perturbed about **6% less hard per row** (0.987 against 1.050 mean relative L2 on the INT8 table -- relative L2 divides by the row's own norm and the untrained marker rows have small norms), so the bias ran the markers' way and they still lost. And the effect tracks the number of **distinct ids** rather than positions: the two-id arms cluster, the eight- and eleven-id arms are two to three times higher at the same sixteen positions, because perturbing eleven words damages more of a prompt than perturbing one word eight times. The markers are two ids, which is why they are at the floor. **The measurement stands; only the reading is withdrawn** -- the DiT plainly responds to those rows, and so it does to any comparable sixteen | **nothing at the prediction level restores it, and the natural next arm was cancelled by this row**: `release_id` vs `mean_init_rows` judged blind was going to cost about seven hours of card time and an hour of the owner's judging, motivated by the rows being load-bearing. It would take evidence on an axis this does not measure -- outsized attention mass on marker positions from `bench/measure_dit_prefix_attention.py`, whose `prefix_marker_token` class has never fired because the only capture carries no markers, or a perceptual separation nobody has looked for |
| **the 2026-08-15 ordering arms read as a matched pair** | **They were never paired.** The arms carry a different seed per clip, so the comparison was two unrelated samples before the divergence argument above even applies. Check the seed before trusting any pair at all | re-render the arms on matched seeds |

**362 is the max length, and every "345 is legal / 362 is illegal" claim is
withdrawn. Owner decision, 2026-08-16.** 345 is the largest count the
*reference pipeline* will emit — its `max_duration` is a hard-coded 15.0 s and
362 is 15.083 s — which is a fact about diffusers. This repo turned that into
"362 is illegal / out of distribution" for most of 2026-08-14 and withdrew a
bench run over it; that was an inference from a validator, not a fact about the
checkpoint, and the framing survived the first correction because 345 stayed
the default.

`LONG_LENGTH` is now 362 and all shipped graphs carry it. `duration_in_range()`
is the model's window; `reference_would_emit()` is the separate portability
question.

**What 362 rests on, stated because this table exists for exactly this.** One
upstream statement recorded 2026-08-14 (`6e85e48`) with no artifact attached,
plus LightX2V shipping a 362-frame config. MiniMax's README gives a rounded
"4-15 seconds" and the official checkpoint configs state no frame limit at all,
so the primary source neither confirms nor refutes it. This is a decision taken
on thin evidence, not a measurement — do not quote it as one.

---

## These hold

Kept short on purpose — every row is something a second reader confirmed or an
instrument that has been shown red.

- **The marker ids are the release's own, fixed by construction rather than by
  literal.** No file in the release assigns `<d>`..`<|caption_end|>` an id --
  checked across all 27 tokenizer/vocab files -- which has now generated the
  same false finding twice ("core derived them and may have guessed"). Loading
  the release's own tokenizer directory with the standard loader assigns
  **151669-151675** and tokenizes all seven as single tokens, `len(tokenizer)`
  151676 against a 151643 base plus 26 added. Core's "ids fixed by the released
  tokenizer" is correct. Recorded in
  [`2026-08-27_marker_tokenization_alignment.json`](../bench/results/2026-08-27_marker_tokenization_alignment.json)
  so the third reader finds the answer instead of re-raising it.
- **Two of the seven markers reach anything we render or calibrate.** `<d>` and
  `</d>` appear in shipped graphs, in the v2 calibration bundle and in all three
  holdouts; `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`,
  `<|caption_start|>` and `<|caption_end|>` appear in **zero** graphs and zero
  rows of the pool the selection drew from. So every encoder measurement here --
  Gate 5, the overfit test, the four-encoder table -- covers two of seven.
  `bench/marker_corpus/` is the only surface that reaches all of them.
- **The prefix length is a coordinate origin, not just a length.**
  `comfy/ldm/minimax/model.py::PackedLayout.__init__` sets
  `cursor = float(text_len)` and hands it to `_audio_grid` and `_video_grid` as
  their temporal origin, so any arm that changes token count moves the whole
  target timeline in RoPE t-space. The prefix-end to target-start gap is
  invariant; what moves is the relative distance from early prefix rows,
  graded to zero at the boundary. An arm that changes tokenization is therefore
  never prefix-local, and a "few tokens in a large sequence" argument about one
  is measuring the wrong thing.
- **The CUDA seam works.** Live render, `cuda-int8` in the log, override
  chained, 50 forwards composed. Not a source read.
- **One `optimized_attention` call in source**, `comfy/ldm/minimax/model.py`,
  reached by **52 modules** — 50 DiT blocks plus 2 token-refiner blocks, all
  sharing one `Attention.forward` (`num_layers=50`,
  `token_refiner_num_layers=2`). "One call site" is not "one call", and the
  refiner blocks see only the text span while the 50 see the full sequence.
- **Sage runs 5 of 16 steps under Sol** at the shipped window — verified at two
  layers of the node source and cross-checked against this repo's own
  sigma-window table, which independently gives 11/16 sparse.
- **`reuse_qkv_memory` is numerically identical** to the normal entry, six
  digits. It cannot change output. What it *buys* is unmeasured.
- **Both backends are arithmetically equivalent** at `T=512` fidelity.
- **All 34 API graphs carry `LONG_LENGTH`**, read from their widgets.
- **Every `bench/check_*.py` passes**, and the ones added today were each shown
  red first: `check_sol_kernel` (4 cases), `check_bench_matches_shipped`,
  `vendored`, `node_version`.
- **`int8_convrot` sits about three times closer to the bf16 release than
  `fp8_scaled`.** Measured 2026-08-21 against the release itself
  (`bench/analyze_quant_delta.py`; records
  [`bench/results/2026-08-21_quant_delta_fl2va.json`](../bench/results/2026-08-21_quant_delta_fl2va.json)
  and its ref2va twin): median relative delta per block linear 0.0091 against
  0.0265, on both checkpoints, cosine 0.9999 against 0.9996. fp8's error is
  flat to four digits across all 200 modules while int8's varies by module kind
  and is worst on `attn.out_proj`. **Both files sit at their format's floor**:
  quantizing one release weight in the script itself, scalar-scaled e4m3
  against per-row int8 through the same rotation, reproduces 0.02651 and
  0.00882 where the shipped files read 0.02651 and 0.00882. The gap between
  the formats is the formats, not the calibration. Every tensor neither file quantizes is
  byte-identical between them, so nothing else is in the comparison.
  **Stored weights only**: fp8 carries an `input_scale` on 150 of its 200
  quantized tensors, int8 stores no activation scale at all, and what either
  does with activations at run time is invisible to a weight comparison. The controlled runtime version
  is `docs/open_experiments.md` #22's fixed-input first-step forward.
- **The bf16 release and the Comfy repack order `attn.qkv_proj` rows
  differently.** The release interleaves per head, `[head][q|k|v][head_dim]`;
  the repack concatenates, `[q|k|v][head][head_dim]`, which is what
  `comfy/ldm/minimax/model.py` splits on. Compared as stored, a qkv weight
  reads a relative delta of 1.40 against its own origin with cosine ~0, in
  both quantizations equally; reordered, it reads the same ~0.9% as every
  other int8 linear. `analyze_quant_delta.py` measures both readings and
  refuses to run if the reordering is not the better one.
- **`DeepBeepMeep/MiniMax-H3`'s FL2VA files against `Comfy-Org/MiniMax-H3`'s,
  measured 2026-08-25** by `bench/compare_dit_checkpoints.py` on headers and
  range-fetched samples (records `bench/results/2026-08-25_dit_fl2va_*.json`).
  The int8_convrot files, pruned and unpruned, are Comfy's bytes under a
  different header. The bf16 files are Comfy's bytes except `qkv_proj`, which
  DBM stores in the **release order** above; head 0's k rows sit where
  ComfyUI's split expects head 1's q, so a DBM bf16 file is a WanGP file and
  not ComfyUI-loadable as-is. DBM `pruned_rank8` and Comfy `pruned` are the
  same rank-8 AdaLN factorisation in a sign-flipped basis (basis-change
  residual 3e-7), with the modulation agreeing to 2e-4 relative, which is
  Comfy storing the factors in F16 where DBM keeps F32. DBM `pruned` with no
  suffix is a **rank-64** factorisation on a 1,001-point grid that Comfy does
  not publish; against rank-8 its modulation differs by 1e-5 to 7e-5 across t.
- **The `adaln_all` hybrid applies ref2va's AdaLN linears to fl2va's curve
  table, measured 2026-08-25** with the same tool. `bench/build_hybrid.py`
  copies `blocks.*.adaln_proj.linear.*` and leaves the shared `adaln_t_table`
  on fl2va; the two partitions' tables span nearly the same subspace (residual
  5e-5) but are not the same table, so the hybrid's modulation sits 0.1-0.2%
  from ref2va's own at every block, an order above the rank-8 truncation and
  an order below the 2-4% by which ref2va's AdaLN differs from fl2va's. A
  partial hybrid cannot avoid this (one table serves both parents' blocks); the
  all-blocks build could copy ref2va's table and has not been rebuilt.
- **The H3 Turbo LoRAs across `lightx2v/Minimax-h3-Turbo` and
  `DeepBeepMeep/MiniMax-H3/loras`, measured 2026-08-25** by
  `bench/compare_lora_files.py` (records `bench/results/2026-08-25_lora_*.json`;
  whole sampled tensors per module kind, every alpha scalar read). DBM's five
  `lightx2v_*` files are lightx2v's diffusers files byte for byte plus one
  `.alpha` scalar per module, equal to lightx2v's metadata alpha where one
  exists (128 for v1.0 and v1.1, 8 for 8-step and ref2v). **v0.1 declares no
  alpha anywhere and DBM assigned 16**; its origin is in neither file, and
  against rank 128 it is an 8x smaller scale than v1.0's. lightx2v's
  `comfyui_bf16` variants are exact conversions: q, k, v fused block-diagonally
  into one rank-384 `qkv_proj` LoRA (each band byte-identical to its source),
  `fc1` lora_B halves swapped as the file's `swi_glu_mapping` declares,
  everything else byte-identical, fused alpha scaled with the rank so the
  effective scale is unchanged. The shipped `minimax_h3_turbo_v4_step600_ema`
  is DBM's `larryvrh_v4_step600_ema`, identical on every sampled tensor; DBM's
  `turbo_4step_ema_ckpt850` shares its modules and metadata and differs on
  every tensor, so it is another checkpoint of that recipe, not a re-save.
- **The node is vendored, symlinked and hash-pinned**, so the file ComfyUI
  loads cannot drift from the tracked one, and an unrecorded hash fails rather
  than warns.

---

## Enumerated consumers, checked mechanically

A retraction is done when every **consumer** of the claim is enumerated, not
when the claim is corrected. That is today's most-repeated failure in one
sentence: "sage gets nothing" was retracted and went on producing wrong answers
in three more places, and the 23-point swing was retracted in this file while
still carrying an argument in `docs/bench_plan.md`.

The block below makes that enumeration executable.
`bench/check_retraction_consumers.py` fails when a retracted phrase appears in a
file **not** listed for it. There is no caveat-detection and no judgement: a
mention inside a `RETRACTED:` block and a live use are indistinguishable to a
matcher, so the check does not try. It answers one decidable question — has this
claim reached a file nobody signed off on — which is the failure mode that
actually occurred, four times, all in files that acquired the claim *after* the
retraction.

Adding a file here is a claim that someone read that occurrence and it is
correct in context. Do not add one to silence the check.

```retraction-consumers
ONE LINE PER FIELD. The parser reads only the line beginning `ALLOW:`, so a
wrapped continuation is silently dropped and the row enumerates fewer files
than it appears to. That fails closed -- the missing files show up as
unlisted consumers -- but it wastes a run, which is how it was found on
2026-08-16.

`ALLOW: (none)` means the phrase must appear in NO file: the claim was deleted
repo-wide rather than caveated, so it has no consumers and any occurrence is a
reintroduction. Spelled out rather than left empty, because a bare `ALLOW:`
is treated as a truncated line and fails the parse.

PHRASE: 2.7x more accurate
ALLOW: docs/evidence.md
WHY: the figure was DELETED repo-wide on 2026-08-16, not caveated, so the
     allowlist collapses to this ledger. The phrase is retained here so the
     check keeps a tripwire on it: if it reappears in any file, someone has
     reintroduced a withdrawn number. Still deliberately NOT matching bare
     "2.7x" -- attention.py and README.md use that string for a speed figure
     against torch flash, a different and correct claim.

PHRASE: 23-point swing
ALLOW: docs/evidence.md docs/bench_plan.md docs/SOLATTN.md
WHY: bench_plan's is inside the retraction block that replaced the argument;
     SOLATTN's are its "Do not rely on" table and the reference section.

PHRASE: sage gets nothing
ALLOW: docs/evidence.md docs/SOLATTN.md
WHY: both are retraction statements naming it as withdrawn.

PHRASE: zero DiT calls
ALLOW: docs/bench_plan.md docs/checks.md
WHY: bench_plan's is inside a "RETRACTED 2026-08-14" bullet stating the
     5-of-16 correction. checks.md quotes it as the worked example of why this
     check is an allowlist rather than caveat-detection -- and it was caught
     by the check on the commit that added it, which is the behaviour wanted.

PHRASE: exactly one attention site
ALLOW: docs/SOLATTN.md
WHY: the sentence that contains it says the page claimed this until 2026-08-14
     and corrects it to 52 modules.

PHRASE: 38 text rows
ALLOW: docs/SOLATTN.md
WHY: both name it as a smoke-harness figure rather than a shipped-graph one.

PHRASE: worth 1.16x alone
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md docs/bench_plan.md CHANGELOG.md
WHY: Triton, 362 frames, stacked on int8, retracted 2026-08-16 when a dense
     control showed the CUDA permutation free. morton.md owns the
     retraction; SOLATTN.md's is the "do not rely on" row and the corrected
     Configuration-findings bullet; open_experiments.md and bench_plan.md
     name it as withdrawn in the paragraphs that used to argue from it;
     CHANGELOG.md records the retraction as history and is not rewritten.
     Enumerated the same day the retraction was made -- the earlier three
     same-session retractions were not, and one of them (this one) went on
     being asserted in SOLATTN.md's Configuration findings for hours
     afterwards. That is the open question raised in
     internal/postmortems/2026-08-15_session_morton-sampler-and-refs.md,
     answered by the failure recurring rather than by argument.

PHRASE: essentially rewritten
ALLOW: docs/evidence.md docs/h3_ref2v_distillation.md CHANGELOG.md
WHY: the 1.92 `final_layer.adaln_proj` figure, withdrawn 2026-08-20 as a
     curve-form basis artifact. h3_ref2v_distillation.md keeps it inside the
     parenthesis that withdraws it; CHANGELOG.md records it as history.

PHRASE: replaced rather than adjusted
ALLOW: (none)
WHY: the same figure in its other spelling, carried by h3_config.py and by
     bench/analyze_ref_lora.py until 2026-08-20; h3_config.py now says so
     without the phrase, and analyze_ref_lora.py was deleted 2026-08-21 with
     the rest of the reference LoRA. No consumer is left, so this row is a
     tripwire rather than an enumeration: it went `(none)` on 2026-08-25,
     having warned as a stale allowlist since the cleanup.

PHRASE: 94% GPU utilisation
ALLOW: docs/evidence.md docs/morton.md docs/SOLATTN.md docs/open_experiments.md CHANGELOG.md
WHY: the same measurement and the same retraction. Listed separately because
     the two halves of that claim were quoted independently -- h3_config.py
     carried the utilisation figure as the mechanism story for the 1.16x, so
     either half can reappear without the other.
```

**What this does not defend, and the gap is not small.** The sharpest decay
found on 2026-08-14 has no string to match: `docs/bench_plan.md` read "one
**345-frame** video reference", where 345 is the *reference* length and looks
like the shipped config, concealing that the *target* was 362. The retracted
thing was the **pairing** of two numbers, not any token in the sentence. No
phrase matcher can see that, and a check that appears to cover a class it
silently omits is how the 2026-08-13 validator bug worked — so it is said here
rather than left to be assumed.

**The general form, named 2026-08-16 after it happened again.** This check
defends the *spelling* of a retracted claim, not the claim. A retracted
**measurement** can reappear in any wording containing no listed phrase, and
**derived figures are exactly that shape** — a spread between two withdrawn
numbers, a ratio of them, a rescale, a rounding. The instance: after the
accuracy ratios were withdrawn, `CLAUDE.md` kept a sentence saying all three
fp8 variants land within a stated tolerance of each other — a mean_rtol spread
from the same withdrawn sweep — nine lines below the paragraph announcing there
was no numeric half left. The commit that removed its neighbours left it, and
its author read that paragraph twice the same day. Two properties made it
invisible: it carried no retracted token, and it read as a conclusion rather
than as a number.

So when withdrawing a measurement, grep for the *quantities* it produced and
not only for the sentence it was written in.

**And it cannot be closed by adding a row, which was tried on 2026-08-16 and
reverted the same hour.** The ledger is an allowlist: every phrase needs at
least one file that legitimately contains it, and the `stale_allowlist` case
warns when a listed phrase is absent from its own allowed files. A spelling
that should appear **nowhere** therefore cannot be expressed — listing it emits
a permanent warning on every run, which is the crying-wolf failure this repo
treats as worse than no check. The model fits a claim with a legitimate home
and a forbidden everywhere-else; it does not fit a claim that is simply banned.
Catching those needs a different instrument, and until one exists this
paragraph is the instrument.

---

## Environment, because it is not in git

The venv and ComfyUI were both changed on 2026-08-14 and neither is
version-controlled here. Updated 2026-08-22, where noted:

- `comfy-kitchen` replaced: `0.2.31` → `0.2.31+sol.c04ef20`, **built from
  source for sm_89 only**. It will not work on another architecture, and it
  declares a version the stock wheel also declares.
  **Rebuilt 2026-08-22 to `0.2.31+sol.23d1a66`**, the same branch five commits
  on, for the rewritten routing kernel and `topk_ratio`. Everything above
  still holds; only the tag moved.
- `nanobind` 2.14.0 added.
- `custom_nodes/ComfyUI-SolAttn-cuda/` created; its `sol_attn_minimax.py` is a
  **symlink into this repo's `vendor/`**. Editing through the installed path
  writes into the tracked file.
- `coderef/comfy-kitchen-sol/` cloned with submodules; at `c04ef20` from
  2026-08-14 and at `23d1a66` since 2026-08-22. Its `pyproject.toml` is
  **edited during a build only** — `vendor/rebuild_kernel.sh` applies the
  version-tag patch, substitutes the checkout's own short sha, and reverts on
  every exit path, so the tree is left as found. It was carried as a standing
  working-tree modification until 2026-08-14, which blocked `git pull` outright
  rather than conflicting loudly as intended.

**The model build is software substrate too, and it was missing from this list.**
`workflows/h3_config.py` points at `minimax_h3_fl2va_pruned_int8_convrot` and
`minimax_h3_ref2va_pruned_int8_convrot`, so every number in this repo was measured
on weights that are **pruned, convrot-rotated and int8** — three separate
properties, not one. `fp8_scaled` and `w4a8_mixed` builds of the same models sit
on disk beside them.

**What "pruned" is, and what it costs -- measured 2026-08-20 against the
unpruned `int8_convrot` files** ([`bench/results/2026-08-20_adaln_pruning_residual.json`](../bench/results/2026-08-20_adaln_pruning_residual.json),
`bench/analyze_adaln_pruning.py`). The unpruned DiT carries a timestep MLP and
a full-width AdaLN projection per block (`[96768, 2688]`; the projections are
41% of the DiT's parameters). The pruned file replaces them with the rank-8
SVD of the mean-centred `silu(e(t))` curve: `adaln_t_table` is `U @ Sigma`
(its column norms equal the singular values to four digits, and the fit residual
equals the SVD residual) and the curve's mean is folded into each block's bias.
Every other tensor is byte-identical between the pruned and unpruned file of
the same checkpoint, so the pruned build is the unpruned build with the AdaLN
swapped and nothing else. At the modulation output the swap costs 0.1-0.2%
of the whole output per block and 0.06-0.35% at the timesteps a render
evaluates; on the final layer, which the unpruned file stores in bf16, the
residual is ~0.02%, so most of the per-block figure is int8 error on the
*unpruned* side, and the rank-8 truncation itself is ~0.02%. **fl2va and ref2va
lose the same**: the per-block residual ratio ref2va/fl2va runs 0.97-1.01. The
hypothesis that ref2va compresses worse under its own factorisation, the one
ref2va-specific mechanism this repo had found with a concrete test, is refuted.
**And the sensitivity to it, measured 2026-08-21**
([`bench/results/2026-08-21_pruning_sensitivity.json`](../bench/results/2026-08-21_pruning_sensitivity.json),
`bench/grade_pruning_sensitivity.py`, `docs/open_experiments.md` #22). A
perturbation of that size still moves the network output by **5.6-9.4%**
relative L2 on the first-step velocity, at a fixed input. That is not
invisible, and it is roughly two orders of magnitude larger than the
modulation residual above, so **do not read "the pruning costs 0.1-0.2% of the
modulation" as "the pruning costs 0.1-0.2% of the output"** -- the error
compounds through the stack, peaking around block 36. What keeps it in
proportion is the scale beside it: on the same input the `fp8_scaled` build
differs from `int8_convrot` by 12-21%, so the pruning is smaller than a
quantisation choice this repo already ships.

**fl2va and ref2va are equally sensitive**, which closes the ref2va question
from the other side: ref2va moves 0.86 of what fl2va moves on the reference
input and 0.81 on the plain one, consistently in the opposite direction to the
hypothesis. The determinism floor was exactly zero -- the repeat arm reproduced
its twin bit for bit across a checkpoint reload -- so none of this is noise.
**The entry's pre-registered prediction (under 1%) was wrong**, and is recorded
as wrong in the record's caveats rather than dropped.

Side fact from the same record: the time embedder itself differs between the
two checkpoints (10.6% relative, cosine 0.994), so the fine-tune moved `e(t)`,
not only the projections. Still unmeasured: the int8 error of the linears
against the bf16 release, which needs the bf16 files.

This is not only bookkeeping. `convrot` applies a rotation (`docs/roadmap.md`,
established while grading the ref LoRA), so any claim reasoning about q/k geometry
— centroid similarity, block locality, what a token ordering buys — is reasoning
about a rotated, pruned, quantized space. Whether that reaches Sol's routing or
the orderings is unmeasured and filed as open experiment 19.

**Where a run does and does not record it.** A conforming capture manifest already
says which build ran: `models` requires `unet`, `clip` and `video_vae`, and those
filenames are self-describing. But `weight_quantization` and `vae_quantization`
were in no `required` list and `bench/check_capture_manifest.py` inspected neither -- **fixed 2026-08-17**, both now required and asserted,
so a manifest can omit both and still pass. **A bench run emits no manifest at
all**, so a timing carries no record of the weights underneath it — the same shape
as the power-limit gap below, and enforced by nothing.

Record the node hash and the `comfy-kitchen` tag with any measurement. The
branch rebases and the build declares a stock version number, so nothing else
can tell two builds apart.

**The host is the other half of this, and `docs/hardware.md` owns it.** That
list is software; the machine underneath has its own ungoverned state, and the
one that moves timings is the GPU board power limit. It is set outside the repo,
appears in no workflow, log or capture manifest, survives reboots once a systemd
unit exists for it, and changes s/it without changing a single output byte.
`bench/hwinfo.py` prints it and says so when it is not stock — run it alongside
the node hash and paste both. Nothing enforces this.

The power limit on this box **was changed away from stock on 2026-08-17**, after
every timing on `docs/bench_plan.md` and in `docs/SOLATTN.md` had been recorded.
Those measurements are not retracted — they were correct at stock — but
reproducing one now means checking `hwinfo.py` first and resetting the limit if
the comparison is meant to be like-for-like.
