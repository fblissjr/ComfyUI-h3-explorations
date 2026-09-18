# Two bank prompts were rendered at three times their length, 2026-09-18

Model: MiniMax H3. Found by the owner on 2026-09-18: "the prompt is clearly
intended for a shorter length ... we're using it OOD right now."

## What happened

Every bank prompt declares the length it was written for (`frames` in
`prompt_bank/bank.json`): its action and dialogue fill that many frames. Two
prompts were patched into the full-length text-to-video graph without setting
the length:

| prompt | written for | rendered at |
|---|---|---|
| `t2va_noodle_bar` | 107 frames | 345 |
| `t2va_post_office` | 141 frames | 345 |

Past the end of its script the model has nothing to follow and improvises. The
noodle bar's script (a customer steps up, holds up two fingers, says one line,
lowers his hand; the cook nods) is over a little after four seconds, and the
"morph at four seconds" that every plain-order arm showed, with extra figures
arriving and leaving, starts there.

## Which records it touches

Audit of every committed render row that names a bank prompt and a length
(re-derive with `workflows/prompts.py::entry` against the `rendered` field of
each row of `bench/results/*.jsonl`): all off-length rows are these two prompts, in

- `2026-09-17_sol_options_noodlebar_arms.jsonl`, `_arms_2.jsonl`,
  `2026-09-17_sol_reorder_noodlebar_arms.jsonl` (the noodle bar batch and the
  reorder arms the owner ranked),
- `2026-09-17_sol_start_percent_arms.jsonl`, `_arms_2.jsonl` (post office and
  noodle bar start_percent stacks),
- `2026-09-18_sol_reorder_under_memory_compiler_arms.jsonl`,
- `2026-09-18_sol_reorder_panel_arms.jsonl` (the post office pair only),

and, outside those files, the noodle bar and post office pairs of
`2026-09-15_block49_repro_batch.md` and the capture set
`2026-09-18_noodle_bar_sage_chain_plain`. Every other row is on length.

## What still stands, and what does not

- **Stands: every bit-identity result.** The memory-compiler acceptance, the
  kitchen rebuild check and the "nothing changed between the two days" checks
  compare a render to itself; what the clip shows does not matter to them.
- **Stands as a measurement, not as its reading:** the reorder removes the
  extra figures on the over-length noodle bar; on the capture of that render,
  plain order concentrates Sol's error in the region and frames where the
  improvised figures appear and the reorder flattens it
  (`2026-09-18_noodle_bar_capture.md`). What that is evidence OF has changed:
  it is how Sol behaves while the model improvises past its script, which is
  not how the node is used. It is not evidence of a defect in a typical render.
- **Does not stand:** "every plain-order arm morphs, so plain order has a
  visible defect the reorder fixes" as a statement about normal use; the
  noodle bar and post office observations in the `start_percent` verdicts; the
  post office pair of the reorder panel. The market `start_percent`
  observations are on length and stand.
- **Unaffected:** capture-metric results on the covered market and courtroom
  sets, both rendered at the length their prompts declare.

## What changed so it cannot recur

`bench/run_graph_arms.py` refuses a bank prompt rendered at a length other
than its bank entry declares, names the fix in the message, and takes
`--allow-off-length` when the mismatch is the thing under test. Its `@bank:`
shortcut now hands over the stripped text.

## The test, run the same day

The noodle bar at 107 frames and post office at 141, plain order against the
`3d` reorder, seed 730451892, default chain, memory compiler on
(`2026-09-18_on_length_arms.jsonl`; clips in `Video/on_length/`). All four
rendered without error.

Noodle bar, read from five stills per clip across the take (frames 20 to 100),
by the analyst, NOT yet by the owner: in BOTH orderings there is one customer
in a grey raincoat and one cook in an apron for the whole clip, the customer
raises his hand, the sign reads correctly, and no extra figure arrives. So at
the length the prompt was written for, plain order shows none of what the
345-frame renders showed. The morph was the over-length render. Stills cannot
see flicker or a brief morph between sampled frames; the owner's look at the
two clips is what closes this.

### The owner's look, the same afternoon, which OVERTURNS the stills reading above

"Still someone ghosts in out of nowhere at the 4s mark" in the plain-order clip
at 107 frames; "that does NOT happen" in the `3d` clip. "Better scene now tho
for both, but still ghosting on that first one but not second is a big deal."
Checked afterwards on a crop of the last second of the plain clip: a second
head does appear beside the customer from about frame 88. The five stills
sampled above, at low resolution across the whole take, missed it, which is the
limit that paragraph named.

So the over-length render made the noodle bar much worse (crowds of invented
figures for ten seconds), but it was NOT the whole story: at the length the
prompt was written for, plain order still ghosts a figure in near the end and
the reorder does not. Post office at 141 frames: the owner cannot tell the
orders apart. One scene, one seed, a short clip; a short-clip panel (more seeds
of this scene and three other short prompts, plain against `3d`, blind) was
queued the same day: `Video/short_clip_panel/`.

What this does to the sections above: "Does not stand" still holds for the
345-frame renders as evidence about normal use. But the statement that the
morph "was the over-length render and not the token order" is WRONG as written;
both contributed.

What was planned before the test:


The noodle bar at 107 frames and post office at 141, plain order against the
`3d` reorder, same seed (`Video/on_length/`). If plain order is clean at the
length the prompt was written for, the morph was the over-length render and not
the token order.
