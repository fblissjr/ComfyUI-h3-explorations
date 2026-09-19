# How often the judge ties, from the structured verdict files, 2026-09-19

Tool: `bench/tally_judge_verdicts.py` (CPU, reads files only). Numbers:
`bench/results/2026-09-19_judge_tie_rate.json`, per file and in total.

## What was counted

Every `bench/results/*verdict*.json` in the blind-batch join's shape: scored
pair contests with one of the rubric's four answers ("Clip 1 better", "Clip 2
better", "same", "can't tell"). Six files qualify, from 2026-08-20 to
2026-09-05; every scored pair in them is a same-seed pair of two different
arms, and every file is marked partial. Not counted: the 2026-09-17 and
2026-09-18 panels, which were recorded as prose in their `.md` records
(for example `bench/results/2026-09-18_sol_reorder_panel.md`), not in this
shape.

## What it says, in direction

- **Ties are about as common as decisive verdicts.** "same" is the more
  frequent of the two tie answers, and "can't tell" the rarer. The rate varies
  a lot between sessions (the JSON has each).
- **For the small-panel arithmetic** in
  `docs/research/2026-09-19_evaluation_one_judge.md` section 0.4: at this tie
  rate, the review's five-scene rule sits at the upper end of the
  false-pass probabilities that file computes for a judge who is guessing.
  So a five-scene verdict needs the decoy pair to be read, which is what that
  file already recommends.
- **Slot order.** Clip 1 wins more of the decisive verdicts than clip 2, but
  the split is well within what chance gives at this many decisive pairs
  (`slot_split_two_sided_p` in the JSON). No evidence of a slot bias; not
  enough pairs to rule a small one out.

## What it is not

Not the judge's false-positive rate. These are contests between arms that may
really differ, so a decisive verdict here can be right. The false-positive
rate needs decoys (two takes of one arm), which no session so far has had.
