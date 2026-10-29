# 2026-10-29: two findings from the evening, checked on disk

A dated record, not current guidance: what was true on 2026-10-29. The rule pages carry what follows from it
([`reference_and_prompt_rules.md`](reference_and_prompt_rules.md), [`docs/wiki/decisions.md`](../wiki/decisions.md)).
Found by the render session (`gemrender`) and the 4912 session (`gem4912`), each checked independently by the Claude
advisory session against the files named here. A figure in prose points at the file that holds it.

## 1. `second_hfp_0278n` opened her eyes and her mouth; the source has both shut

The owner marked it wrong at 21:20Z (`data/fun/owner_answers.jsonl`, the row for `second_hfp_0278n`) and guessed an old
prompt. The text is the current form (`data/2026-10-29_second_hfp_0278n_job/second_hfp_0278n.render.txt`, the same words as
the kept `second_hfp_4647i` apart from the clothing sentence and the `<Subject 2>` mouth clause, which `4647i` carries because its
mouth opens on some frames and `0278n` does not), but against `second_hfp_0278m`, the earlier letter whose mouth
stayed shut, two things changed, both by way of the drawing:

- **The eyes.** The reference `second_0278_0293_j` was drawn with `--eyes-shut-form closed` on all 16 frames: the sculpted
  eye painted skin-grey, one lid line, no iris. The text has no sentence for that form (`bench/load_text.py` has none for
  `lowered` or `closed`), so it went on saying "the eyes, each with a dark iris, open, close and look where the person's
  eyes do". With no iris to see, H3 took the still's straight-ahead eyes, which is the owner's own rule of 2026-10-22 ("if
  frame 1 can't see a black iris, it defaults to the ref image eyes"). The body model's head also tilts up to face the
  camera on 287 to 293 where the source stays bowed (`gemrender`'s measurement of the pose table; the source's own points lift by the same amount, so whether "stays
  bowed" is more than a difference of degree is not settled), and the render's eyes open from 288. Across every job on disk, `closed` was used on `0278n` alone; the two kept loads with an
  eye word used `lids` (`second_hfp_0604h`, one line along the upper lash and no iris) and `lowered` (`second_hfp_4602d`, every eye
  with a dark iris).
- **The mouth.** The second subject's plan form draws a closed mouth outlined (`data/fun/plan.json`,
  `.subjects.second.reference.closed_mouth`). `0278m` predates that form and drew nothing at the mouth, so its text used
  the no-mouth wording ("where the eyes look"). `0278n` drew the closed mouth, `bench/load_text.py::mouth_drawn` read that
  as a mouth drawn, and the text got "how far the mouth opens are transferred" and "her mouth opening and closing where
  its mouth does" on a shot where her mouth never opens. `gemrender`'s bug claim is the asymmetry: the retention line and
  `[Shot 1]` are gated on `mouth_drawn` while the `<Subject 2>` definition is gated on `mouth_drawn_open`, so the text
  named a mouth it never described, and the same words had made her speak on this shot once before (`0278l`, 2026-10-22,
  which is why `0278m` was built without them). The render opened it (parted on 278 to 287, wide open on 288 to 293).
  The advisory session's reading, not `gemrender`'s: the same text and drawing kept the mouth shut on `second_hfp_5114k`
  (kept 2026-10-25, the shot the outlined form was built for), so this is a tendency on a close-up, not a template bug
  that fires every time.

**The scan** (the advisory session's, reproduced by its verifier). Every job file under `data/2026-10-*/` with a
`motion_file` was checked, with its text and the reference record beside the motion file, for the mouth-follow wording
on a record where `bench/load_text.py::mouth_drawn_open` is false, and for the `closed` eye form. The mouth pattern is on two
rendered loads, `0278n` and the kept `5114k` (four 10-22 jobs carry the words because their records predate the form,
none kept). The `closed` form is on `0278n` alone. The built backlog is clean: `second_hfp_0014h` (open on four frames, two spans) and `lead_fp_5049b` (one frame) have a
mouth that opens, and the unbuilt lead loads take the lead's `closed_mouth: sculpted`. Nine loads of
10-13 to 10-23 have no reference record on disk (the verifier's list: `lead_fp_0606`, `0606b`, `0751`, `0751d`, `1920`,
`2050`, `4053b`, `third_fp_1222c`, `second_hfp_4647e`), none kept or on the backlog. `gemrender`'s addition: any unbuilt silent `second` shot (its list: `1419–1465`,
`5768–5791`, `1093–1133`) takes the same trap from the subject's `closed_mouth: outlined` unless the shot sets `sculpted`
in the plan, as `278–293` now does.

**What follows.** `second_hfp_0278o` is built on words, no code: nothing drawn at the mouth (`closed_mouth: sculpted` on
the shot in `data/fun/plan.json`, the recipe that kept the mouth shut on `0278m`), the `closed` eye drawing, and a text
written by hand (`text --by-hand`) whose eye sentence says the eyes are shut, on the `lids` clause's model. The advisory
session first recommended `--eyes-shut-form lowered`; `gemrender` checked and refuted it: on `278–293` the downcast rule
had already drawn dark irises in propped-open lids on every frame of `0278m`'s reference, so a `lowered` reference is the
same picture to codec noise (mean difference under one grey level) with the same text, a re-render of the rejected
letter. The sheet is read against the source on 287 to 293 before `first`. In the advisory session's reading, two template changes are judgement calls for a render-free window, not obvious wins,
since `5114k` kept with the same text (`gemrender` reads the gating asymmetry as a bug): the mouth clause keying on a mouth drawn open rather than any mouth drawn, and a sentence of the text's own for
the `closed` and `lowered` forms.

## 2. `bench/frame_table.py` reports restored frames as emptied

Found by `gem4912` on `second_hfp_4912p` (`data/2026-10-29_second_hfp_4912p/frame_table.md`) and re-measured from the
held-part mask (`subjects/second/parts_held__*.mkv`):

| frame | the table says | the held-part mask has |
|---|---|---|
| 4920 | 5,279 px, replaced | 7,606 px |
| 4921 | 0 px, emptied, nothing saved | 320 px |
| 4939 | 4,675 px, replaced | 8,950 px (its `part_grades.json` row already says 8,950) |
| 4943 to 4946 | 0 px, emptied, nothing saved | 972, 1,102, 1,741, 4,549 px |

The cause, read in the code: the table takes a frame's saved pixel count from the masks file written before
`her_hair_in_hole` runs, and uses it for every frame absent from `part_grades.json` and for the "saved part kept, fill
refused" rows (`bench/frame_table.py`, the `else` branch after the grade rows and the `kept_px` line). The grade update
after `her_hair_in_hole` (`bench/capture_masked_run.py`, the loop over `pg["frames"]`) only touches rows that already
exist, and `grade_holds` writes rows for doubted, offered and floor-emptied frames only, so a frame restored from 0 px
never gets one. Every reader and flag that cited an "emptied" frame from a table today read that number; until the fix
lands, an "emptied" row is checked against `parts_held` before it is cited.

The fix, being built by `gem4912` on its own branch: the table counts the held-part mask it already opens and hashes,
and uses that count where a frame has no grade row and wherever a grade row has no `px`; the capture appends a grade
row for a frame restored from 0 px. Its check, as `gem4912` wrote it: `4912p` with the counts above, and only 4922 to 4925 and 4941 to 4942 emptied
(4926's row existed, 12 px under the floor, so the update in place worked there). Added by the advisory session: one
capture where `her_hair_in_hole` changed nothing, shown byte-identical, since that is `gem4912`'s own least-harm
argument. By the repo's rule it lands after a blind read, between loads, and the table is re-run on `4912p` and `4912q`.

Landed 2026-10-29 as `812ec389` and `2ccad408` (the second refuses a held-part mask whose frame count is not the
capture's), after the advisory session's blind read with the byte-identical control on `0278n`; the tables of `4912p`
and `4912q` were re-run and carry the counts above.

## 3. On 4912, a share of her own hair lies outside the hole, inside the margin the model is told to blend into

Found 2026-10-29 in the evening. `gemrender` measured on `4912q`'s window record that part of the subject's own hair
(the part model's Hair class) lies outside the hole; `gem4912` corrected which array is which (the window's `mask` is
the zoomed hole, the regenerated region is wider and holds all of her hair); the advisory session reproduced the
share on the canvas grid from the preview's class video (her Hair on her track, before carves) against
`parts_held`. The script and per-frame rows are in the advisory session's day file (23:27Z); the figures:

| load | her hair outside the hole, first frame / median over the shot | share of the 12 px ring outside the hole that is her own hair, first frame / median |
|---|---|---|
| `second_hfp_4912q` | 0.17 / 0.23 (0.33 to 0.36 on 4916 to 4919) | 0.24 / 0.23 (0.41 to 0.49 on 4916 to 4919) |
| `second_hfp_4479h`, kept | 0.14 / 0.11 | 0.14 / 0.10 |
| `second_hfp_5114k`, kept | 0.04 / 0.05 | 0.05 / 0.07 |

What it means. The hole is cut to the SAM head track; the crown join in `bench/head_tube_parts.py` adds a hair
component only where enough of it lies above the track's top and never below, so hair at the back and sides inside
the head band stays out. That hair sits in the margin band, which regenerates from noise like the hole (the
sampler's noise mask is the grown region's tokens, `audio_freeze_song.py`, the `noise_mask` line), so the model is
not shown it there. It reaches the model two ways, both read by `gemrender` and checked: the margin is grown from
the hole, not from her hair, so the region's edge lands on the outer edge of her hair and the pinned cells just
outside, encoded from the unblanked source, carry it in the VAE's receptive field; and the carve for the others
keeps only the hole as hers (`video_mask.py`, `tokens * (1 - theirs * (1 - own))` with `own` from the hole), so a
cell of her hair that a bystander touches is given back to the source. On 4912 the edge therefore meets her own
dark hair and the bystanders', where on the kept kitchen shots it meets wall and doorway. (An earlier wording here
said the margin was shown to the model as source; it is not.) The hole's size is not the difference: on the canvas 4912's hole is the size of `5114k`'s and after
the zoom covers more latent cells than any kept kitchen shot. The new hair's length and shape differ from hers
(`<Picture 1>` is long and straight; she wears a short crop), and the margin is a fixed pixel count, not measured
against the new hair; that is a plan question for the subject and would widen every `second` load.

Two things follow, independent of why the fill is dark: by the owner's rule a visible subject is always replaced,
so a third of her hair left as filmed on 4916 to 4919 is wrong on its own; and this is the one structural thing
peculiar to 4912's first window after seven renders that changed seed, crown join, conditioning and reference. It is
the next one-variable test: the hole taking all of her hair (`--no-head-track WHY --head-tube` on `first`, or the
join extended to the back and sides, replayed on every kept `second` shot so no kept hole changes), then one render
with nothing else changed. A "hair outside the hole" share per frame belongs in `bench/frame_table.py` so the next
shot shows it before it renders.
