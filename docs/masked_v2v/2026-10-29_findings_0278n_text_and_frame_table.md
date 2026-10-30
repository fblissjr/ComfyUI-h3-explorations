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

## 4. Post-`4912r`/`4912s` root cause: the `4k + 1` causal VAE latent steps, `strangers.mkv` swallowing her visible crown on `4939–4946`, and why splitting a 35-frame shot violates the 39-frame window rule

Landed 2026-10-29 as `16ccd81a` (`head_tube_parts.py` + `frame_table.py`), which joins attached `Hair` within the head
band (`y <= track_y1`) and cut `hair_outside_hole_share` on `4912–4919` from `0.17–0.36` (`4912q`) to `0.014–0.089` (`119–657 px` in `4912r`/`4912t`; shot-wide `hair_outside_hole` dropped from `21.57%` = `34,207 / 158,622 px` on `4912q` to `4.28%` = `6,782 / 158,622 px` on `4912r`). Running
`second_hfp_4912r` with that hole fix alone lifted hair median `(L*, b*)` from `(6.3, +9.0)` (`4912q`) to
`(11.8, +16.0)` (`p90 L* = 40.8, p90 b* = +33.0`, reaching `med L* = 31.0, med b* = +34.0` on frame `4939`), while a
one-time reference image test (`second_hfp_4912s`, prompt text byte-identical) produced `(10.2, +14.0)`.

### 4.1 Why splitting `4912–4946` (`second_hfp_4927b`, `4927–4940`) violates H3's 39-frame window rule

Testing a 14-frame mid-span split (`second_hfp_4927b`, frames `4927–4940`) exposed two hard structural violations of
`docs/masked_v2v/frames_rules.md` and `workflows/masked_job_graph.py`:

1. **In-shot padding stays real and unmasked (`window_pad` + `held_context`)**:
   - The shortest legal dual-clock window is `39` frames (`17n + 5` on video and whole on audio). A 14-frame load
     (`4927–4940`) requires `after = 25` padding frames (`4941–4965`).
   - `workflows/masked_job_graph.py::held_context` (`lines 610–624`) reads `shot_span(job) = [4912, 4946]`. Because
     `4941–4946` lie **inside** the shot, `held_context` holds only the `19` frames across the hard cut at `4946`
     (`held_after = 19`, frames `4947–4965`) and leaves **`4941–4946` (`6` frames) as unheld real source frames** inside
     the 39-frame window (`video_mask.py::pad_context`), putting her unmasked real dark hair on `4943–4946` directly into
     the window (the `lead_fp_2050e` failure mode in `docs/research/masking/2026-10-25_context_frames_and_identity_on_2050.md`).
   - Under the 2026-10-25 rule (*"pad the load to the valid window length masked end to end and throw away what you do
     not need at the lay"*), running masked to `4946` (plus `4` held copies on `4947–4950`) is the full 35-frame shot
     (`4912–4946`) again.
2. **Span-edge failure on `4927` (`head_reference.py`)**:
   - Cutting at `4927` makes an off-track re-emergence frame the span's first frame with no earlier clean frame `4920` to
     hold or interpolate from, so `head_reference.py` exits `1` (`THE FACE IS OFF THE SUBJECT'S TRACK on 1 of 14 frames
     (4927), in runs longer than 4 frames or at the span's edge`).

### 4.2 What the `4k + 1` / 17-frame causal VAE steps actually see (`bench/inspect_mask_steps.py` on `4912r`)

`comfy/ldm/minimax/vae.py` encodes the 39-frame window (`35` shot frames `4912–4946` + `4` held copies of `4946` on
`4947–4950`) in **three independent 17-frame clips**, where each clip is split into `1 + 4 + 4 + 4 + 4` frames (`5`
causal latent steps per clip; the last `3` steps of Clip 3 are dropped, leaving `12` latent steps `0..11`). Comparing
`bench/inspect_mask_steps.py --capture data/2026-10-29_second_hfp_4912r` with `window_record.npz`, `strangers.mkv`, and
the rendered hair `(L*, b*)` on every latent step reveals the exact two mechanisms still pinning dark hair into the
window:

| Clip | Latent step | Window frames (`k`) | Shot frames | Step structure & pinned (`denoise_mask = 0`) dark hair in the zoom box | `4912r` rendered hair `med L* / p90 L* / med b*` |
|---|---|---|---|---|---|
| **Clip 1** (`17f`: `4912–4928`) | **Step 0** (`frame 0` of Clip 1) | `0` (`1f`) | `4912` | Standalone `1`-frame `4k+1` step. `0 px` of `second`'s hair outside the hole; `0 px` of bystander hair in the zoom box. | **`17.3 / 47.5 / +17.0`** (brightest frame of Clip 1) |
| **Clip 1** | **Step 1** | `1..4` (`4f`) | `4913–4916` | Foreground dancer (`strangers.mkv`) enters bottom-right of zoom box (`0 -> 21,959 px` of pinned dark hair). | `7.1–12.9 / 27.8–45.1 / +10..+16` |
| **Clip 1** | **Step 2** | `5..8` (`4f`) | `4917–4920` | Foreground dancer approaches center (`26,281 -> 34,669 px` pinned dark hair). On `4920`, `strangers.mkv` swallows the lower-right back of `second`'s hair (`hair_outside_hole_share = 0.0186` = `93 px` in `4912r/frame_table.jsonl`, down from `~14.7%` on pre-`16ccd81a` `4912q`). | `9.0–12.5 / 26.7–36.9 / +14..+18` |
| **Clip 1** | **Step 3** | `9..12` (`4f`) | `4921–4924` | **`3` `BLUE-ONLY` frames (`4922–4924` have `0 px` mask)!** Activated only because `4921` kept a `316 px` sliver (`37` tokens). `98.8%` of the step is pinned (`denoise_mask = 0`) to the foreground dancer's dark hair (`51,657 px`) filling the center of the zoom box. | `8.2 / 14.5 / +14.0` on the `316 px` sliver |
| **Clip 1** | **Step 4** | `13..16` (`4f`) | `4925–4928` | **`1` `BLUE-ONLY` frame (`4925` has `0 px` mask, pinned dancer hair) + `4926` (`818 px` `loaded_part_px` sliver on dancer's neck from `2,460 px` MoGe-2 depth bleed)** packed into the **same 4-frame causal VAE token** as `4927–4928` (`second` re-emerging)! | `7.5–13.3 / 15.3–37.6 / +13..+19` |
| **Clip 2** (`17f`: `4929–4945`) | **Step 5** (`frame 0` of Clip 2) | `17` (`1f`) | `4929` | Standalone `1`-frame `4k+1` start of Clip 2. Unoccluded `second` (`0 px` of `second`'s hair outside the hole). | **`12.5 / 54.1 / +18.0`** |
| **Clip 2** | **Steps 6–7** | `18..25` (`8f`) | `4930–4937` | Unoccluded `second` (`0 px` of `second`'s hair outside the hole; background man's dark hair at top-left `y < 290`). | `11.4–13.7 / 41.2–51.4 / +15..+19` |
| **Clip 2** | **Step 8** | `26..29` (`4f`) | `4938–4941` | **`strangers.mkv` (the white-hat woman) spills onto `second`'s visible hair crown on `4939` (`hair_outside_hole_share = 0.0514` = `143 px` in `4912r`, down from `~9.5%` on `4912q`; `83 px` = `0.0299` on `4912t`, with `463 px` = `0.0495` on `4938`) and swallows `100%` of her visible dark hair crown on `4940`** (`parts_held` drops to `1,947 px` face-only, `2 px` hair), leaving her real dark hair crown on `4940` pinned (`denoise_mask = 0`) inside Step 8! (`4941` is `0 px` `BLUE-ONLY` behind the white hat.) | **`31.0 / 48.2 / +34.0`** on `4939` (where `2,485 px` of hair was held inside `4938`'s token box); `4940` crown left unmasked |
| **Clip 2** | **Step 9** | `30..33` (`4f`) | `4942–4945` | **Left half of `second`'s own dark hair dome above the white hat brim is pinned (`denoise_mask = 0`) across all 4 frames!** `4942` is `0 px` (`BLUE-ONLY`) even though her dark crown is already emerging above the hat brim (`100%` swallowed by `strangers.mkv`). On `4943–4945`, `strangers.mkv` swallows the left `80–89%` of her crown and `lost_held_from: 4938` (`head[4938] & ~taken[k]`) stamps frame `4938`'s right-shifted head box (`x = 613..705`), leaving the left half of her real dark hair crown (`x = 530..612`) unmasked (`denoise_mask = 0`) right against the regenerated right half! | `5.9–11.8 / 21.6–41.6 / +7..+15` (regenerated right half blends directly into pinned left half of her own dark hair) |
| **Clip 3** (`17f`: `4946` + `16` held copies) | **Step 10** (`frame 0` of Clip 3) & **Step 11** (`4947–4950`) | `34` (`1f`) & `35..38` (`4f`) | `4946` (held across all `17` frames of Clip 3) | **Frame `4946` is `frame 0` of Clip 3 (`Step 10`, standalone `1`-frame `4k+1` step) and is repeated across all `17` frames of Clip 3 (`4946–4950` + `12` VAE tail copies).** Because `strangers.mkv` swallows the left half of `second`'s dark hair dome above the white hat brim on `4946` (and has Swiss-cheese holes inside the right half) while `head[4938] & ~taken[34]` only covers the right crescent, **every single frame of Clip 3 has the left half of `second`'s real dark hair dome pinned in the VAE latent (`denoise_mask = 0`)**. | `8.2 / 33.7 / +7.0` on `4946` |

Visual proof contact sheets saved on disk in `internal/gemini/2026-10-29_4912/`:
- `internal/gemini/2026-10-29_4912/4912r_strangers_swallow_4938_4946.jpg`: frames `4938–4946` showing raw `strangers.mkv` in yellow (the woman in the white hat, NOT `third_track` in magenta) swallowing `second`'s visible hair crown on `4939–4940` and the left half of her dark hair dome above the white hat brim on `4942–4946`.
- `internal/gemini/2026-10-29_4912/4912r_crown_4942_4946_pinned.jpg`: Steps `9` and `10` (`4942–4946`) showing the pinned (`denoise_mask = 0`) left half of `second`'s real dark hair crown right beside the cyan regenerated token region on `4942–4946`.
- `internal/gemini/2026-10-29_4912/4912r_all_steps_pinned.jpg` and `internal/gemini/2026-10-29_4912/4912r_strips.jpg`: all steps `0..11` (`4912–4950`) from `bench/inspect_mask_steps.py`.

### 4.3 Concrete root cause and solution for `second_hfp_4912t`

1. **Root Cause 1 — Commit `426638a6` (`2026-10-27 13:09:44 -0500`, `bench/strangers.py:567-571`: `det |= m & (c == 0)`) bloated `shot_c_4912i_strangers.mkv` with `c == 0` (`Background`) pixels, carving `20%–30%` of `second`'s `26 px` (`64 px` zoomed) hair margin on unoccluded frames (`4912–4918`, `4929–4937`) AND swallowing `second`'s own dark hair crown on `4939–4940` and `4942–4946`**:
   - **Clarification on the owner's `2026-10-23` sheet boxes (`data/fun/owner_sheets/2026-10-23_4912_who_is_who_owner.webp`)**: Those yellow boxes and notes (*"Make sure we change her hair to blonde even tho we cant see her face!! this is 2nd subject / mom here sitting down with her hair sticking out!!"*) are **Group 2 rough visual sketches** ("just a rough here's what my eyes see", per `0121b793` and `cf244cb1` in `docs/wiki/decisions.md`), never raw seed coordinates to be pasted into `cast.jsonl`. They are meant to drive general pipeline and `head_marks` logic (`capture_masked_run.py` / `strangers.py` / `plan.json`).
   - **How `426638a6` broke `4912` in two distinct places**:
     - Across all 17 kept `second` renders on disk (`0415n`..`5889e`), **15 of 17 have `0 px` of `c == 0` `strangers` inside `second`'s `margin_px` hair margin** (`13` have no `strangers` keep-out at all; `0604h`, `4602d`, `4726c`, and `5011e` use pre-`426638a6` `strangers.mkv` files). Only `4647i` (the shot `426638a6` was written for) and `4912m..4912s` (`shot_c_4912i_strangers.mkv`, built from the one-off `data/fun/jobs/shot_c_4912i.json` on `2026-10-27 16:18` right after `426638a6`) use post-`426638a6` `strangers.mkv` files.
     - **Before vs. After `426638a6` across all `4912` `step_carves_summary` records (`manifest.json runs[0].step_carves_summary`)**:
       - **Before `426638a6` (`shot_c_4912h` on `2026-10-25` `4912g` and `2026-10-26` `4912j`, `4912k`)**: `strangers` carved **`681` cells** (`4912g`, `7/12` steps), **`739` cells** (`4912j`, `7/12` steps), and **`1,429` cells** (`4912k`, `10/12` steps). On unoccluded steps, `strangers` carved **`0` cells** on `S0 (4912)` and `S6 (4930–4933)` across every pre-`426638a6` run (`4912c, 4912e, 4912g, 4912j, 4912k`), `34–45` cells on `S1 (4913–4916)`, `26–27` cells on `S5 (4929)`, and `7–12` cells on `S7 (4934–4937)`.
       - **After `426638a6` (`shot_c_4912i` on `2026-10-27` `4912m`, `2026-10-28` `4912n/o` [see `docs/masked_v2v/2026-10-28_session_handover.md` §1.1 item 4], and `2026-10-29` `4912p/q/r/s`)**: `steps_carved` jumped to **`12/12` (`100%` of steps)** and total cells carved by `strangers` more than doubled to **`2,753` (`4912m/n`) -> `2,867` (`4912o/p/q`) -> `2,986` (`4912r/s`)**. On unoccluded steps, `strangers` carved **`92–131` cells** on `S0 (4912)`, **`182–210` cells** on `S1 (4913–4916)`, **`95–99` cells** on `S5 (4929)`, **`178–226` cells** on `S6 (4930–4933)`, and **`186–202` cells** on `S7 (4934–4937)`.
     - **Effect A — Clamping `second`'s `26 px` (`64 px` zoomed) hair margin on unoccluded frames (`4912–4918` and `4929–4937`)**: On `4912–4918` and `4929–4937`, `shot_c_4912i_strangers.mkv` puts **`2,069 to 5,296 px/frame` (`90,337 px` across 28 of 35 frames) of `c == 0` (`Background`) pixels and `0 px` of `c > 0` (`Sapiens2` human) pixels** inside `second`'s `26 px` margin (`100%` of the margin carve on `4912–4914, 4918, 4930–4936` is `c == 0`, coming from SAM 3.1 whole-frame `"person"` detections of the background man in the blue denim shirt behind the table, whom Sapiens2 classified as `c == 0`). Because `strangers_of` only subtracts `second_track` grown by `8 px` while the node grows `parts_held` by `26 px` (`64 px` zoomed), `video_mask.py:2262` (`tokens * (1 - theirs * (1 - own))`) carves **`70 to 104` of her `16x16` margin cells per latent step** off the top and back of her head across Steps `0, 1, 2, 5, 6, 7`—clamping the cyan token boundary right down to `8 px` from her short dark pixie skull so the model has no room to draw longer/fuller blonde hair!
     - **Effect B — Swallowing `second`'s visible hair crown during the white-hat pass-by (`4939–4940` and `4942–4946`)**: Whole-frame Sapiens2 (`shot_c_4912i_strangers_classes.npz`) accurately segmented the white-hat woman (`c > 0`) up to her white hat brim (`y >= 333–346`) and classified `second`'s dark hair dome above the white hat (`y = 311..350, x = 538..705`) as `Background` (`c == 0`). However, `426638a6` unioned unowned SAM detections (`p=3/6` background man's arm + `p=4/7` white-hat spill where `c == 0`) into `strangers.mkv`, swallowing `100%` of her hair crown on `4940` and `4942` and the left `80–89%` of her hair dome on `4943–4946` (`parts_held` on `4944` dropped from `3,161 px` in `4912k` to `1,102 px` in `4912m..4912s`, and on `4945` from `4,565 px` to `1,741 px`).
     - Meanwhile, `her_hair_in_hole` Stage 2 (`bench/capture_masked_run.py:5418-5442`) stamped static `head[4938] & ~taken[k]`. Because `second`'s head moved **left by `~50 px`** between `4938` (`x = 583..712`) and `4944–4946` (`x = 538..690`) and moved **left by `~15 px`** on `4940` (`x = 575..620`), static `head[4938]` missed `x = 538..613` on `4944–4946` and `x = 575..613` on `4940`, leaving **the left half of `second`'s real dark hair dome pinned (`denoise_mask = 0`) in Step 8 (`4940`), Step 9 (`4942–4945`), and all 17 frames of Clip 3 (`frame 0` = `4946` in Step 10 + `4947–4950` in Step 11)**.
2. **Root Cause 2 — Stale `2026-10-26` `head_marks` in `data/fun/plan.json` (`4921` and `4926` marked `hers`) activates Step 3 (`4921–4924`) and contaminates Step 4 (`4925–4928`) with `BLUE-ONLY` pinned dancer dark hair**:
   - Even after the owner annotated `4921` on `2026-10-28` (`bf3e1b5c`: *"her hair a sliver, another woman's dark hair fills the window"*), `data/fun/plan.json` kept `4921` (`316 px` sliver) and `4926` (`815 px` sliver on the dancer's neck where MoGe-2 depth bleed carved `2,460 px` of the dancer) as `hers`, turning Step 3 (`4921–4924`) into an active step with `3` `BLUE-ONLY` frames (`98.8%` pinned dancer hair) and packing `4925–4926` into Step 4 (`4925–4928`) alongside `4927–4928`.
3. **Solution (`second_hfp_4912t`)**:
   - **Clean Steps 3 & 4 (`4921–4926`) in `data/fun/plan.json`**: Mark `4921–4926` as `hidden` in `head_marks` (`--head-hidden 4921-4926`), and extend `4939` `hers` to `4939–4940` `hers`. That makes `4921–4926` `0 px`, turning **Step 3 (`4921–4924`) completely `untouched` (`0` active tokens, `0` `BLUE-ONLY` frames)** and removing the `815 px` dancer-neck sliver on `4926` from Step 4.
   - **Restore `second`'s `26 px` (`64 px` zoomed) hair margin and visible hair crown (`4939–4940`, `4943–4946`) against `c == 0` `strangers` spill in `bench/capture_masked_run.py`**:
     1. In `her_hair_in_hole`: when `whole_classes` (`strangers_classes.npz`) is present, `strangers` in `others` only blocks her head/hair recovery where `whole_classes > 0` (human classes in whole-frame Sapiens2, excluding `c == 0` SAM background spill), and Stage 2 (`hers` on lost/partial-shrink frames) includes horizontal crown room above the occluder so her left-shifted hair dome on `4940` and `4943–4946` is 100% covered by `parts_held` + `margin`.
     2. In `capture_masked_run.py` keep-out carving (`bench/capture_masked_run.py:3030-3052`): strip `c == 0` (`Background`) pixels from `strangers` inside `h_mask` and its `m_px + CELL` (`26 + 16 = 42 px`) dilation ring on frames where `k_f in hers_r or int(h_mask[k_f].sum()) >= PART_LEAST_PX` (`4096 px`; on `4647i`, `second`'s part on the bystander frames `4686–4693` is `675–2,046 px < 4096 px` and unmarked, so the strip leaves `4686–4693` untouched and fires only on `hers`-marked `4697–4698`), restoring `70–104` carved margin cells per step across Steps `0, 1, 2, 5, 6, 7, 10, 11`, and write `subjects/strangers/track__<by>.mkv` inside the capture directory so `masked_job.py` wires the capture's cleaned `strangers` video into the node.
   - **Verified `on-plan` (`step_carves_summary`) and `inspect_mask_steps.py` on `second_hfp_4912t` (`ba86fbe9`)**:
     - `bench/inspect_mask_steps.py --capture data/2026-10-29_second_hfp_4912t --run second_hfp_4912t` reports **`0` `BLUE-ONLY` frames**, **`0` `SPECK-IN-BLUE` frames**, **`0` `EDGE-SHRUNK` frames**, and **`0` `STRANGERS SELF-SWALLOW` frames** (`clean_whole_head_frames = 20 / 20`).
     - In `data/2026-10-29_second_hfp_4912t/manifest.json` (`runs[0].step_carves_summary`), total cells carved by `strangers` across the 12 latent steps dropped from **`2,986` (`12/12` steps)** in `4912r` down to **`1,613` (`10/12` steps, `-1,373` cells)**, returning unoccluded steps (`S0, S1, S5, S6, S7`) to pre-`426638a6` (`shot_c_4912h`) levels while expanding the crown recovery on `S8–S11`:

| Latent step | Shot frames | `4912g` (`shot_c_4912h`, pre-`426638a6`) `strangers` carved (`final`) | `4912r` (`shot_c_4912i`, post-`426638a6`) `strangers` carved (`final`) | `4912t` (`shot_c_4912i` + `ba86fbe9` fix) `strangers` carved (`final`) | What changed in `4912t` vs `4912r` |
|---|---|---|---|---|---|
| **S0** | `4912` (`1f`) | `0` (`678`) | `131` (`605`) | **`0` (`722`)** | `-131` carved (`+117` active margin cells) |
| **S1** | `4913–4916` (`4f`) | `37` (`664`) | `210` (`580`) | **`71` (`716`)** | `-139` carved (`+136` active margin cells) |
| **S2** | `4917–4920` (`4f`) | `105` (`614`) | `325` (`411`) | **`170` (`621`)** | `-155` carved (`+210` active cells) |
| **S3** | `4921–4924` (`4f`) | `0` (`0`) | `357` (`113`) | **`0` (`0`, untouched)** | `-357` carved (`4921` marked `hidden`; `0` active tokens) |
| **S4** | `4925–4928` (`4f`) | `214` (`424`) | `418` (`343`) | **`280` (`344`)** | `-138` carved (`4926` `818 px` dancer-neck sliver dropped) |
| **S5** | `4929` (`1f`) | `26` (`549`) | `99` (`501`) | **`26` (`598`)** | `-73` carved (`+97` active margin cells) |
| **S6** | `4930–4933` (`4f`) | `0` (`599`) | `226` (`499`) | **`15` (`711`)** | `-211` carved (`+212` active margin cells) |
| **S7** | `4934–4937` (`4f`) | `7` (`667`) | `202` (`613`) | **`37` (`762`)** | `-165` carved (`+149` active margin cells) |
| **S8** | `4938–4941` (`4f`) | `270` (`459`) | `314` (`540`) | **`301` (`702`)** | `+162` active cells (`4939`: `12,877 px`, `4940`: `6,768 px` vs `9,071 / 1,947 px`) |
| **S9** | `4942–4945` (`4f`) | `0` (`0`) | `312` (`92`) | **`321` (`257`)** | `+165` active crown cells (`4943–4945`: `3,090 / 6,003 / 7,489 px` vs `972 / 1,102 / 1,741 px`) |
| **S10** | `4946` (`1f`) | `0` (`0`) | `196` (`332`) | **`196` (`480`)** | `+148` active crown cells (`4946`: `7,108 px` vs `4,549 px`) |
| **S11** | `4947–4950` (`4f` held) | `22` (`459`) | `196` (`332`) | **`196` (`480`)** | `+148` active crown cells across all held Clip 3 frames |
| **Total** | `4912–4950` (`12` steps) | **`681`** (`7/12` steps) | **`2,986`** (`12/12` steps) | **`1,613`** (`10/12` steps) | **`-1,373` cells carved by `strangers`** |

   - **Rendered & Landed Verification (`second_hfp_4912t`, `data/2026-10-29_second_hfp_4912t/looks/second_hfp_4912t.reading.md`)**:
     - Across the 35-frame shot (`4912–4946`, `151,970 px` of Sapiens `Hair` in `(second_classes == 4) & parts_held`), `second`'s hair transforms from `4912o`/`4912q`'s near-black (`median L* = 5.9–6.3, mean = 12.6–12.8, p90 = 36.5, b* = +8.0..+9.0`) and `4912r`'s chestnut brown (`median L* = 11.4, mean = 17.8, p90 = 43.1, b* = +16.0`) into **warm golden-blonde / honey-blonde hair** (`p10/median/mean/p90 L* = 8.2 / 25.5 / 30.6 / 58.8`, `a* median = +17.0`, `b* median = +30.0`, `+14.1 L*` and `+14.0 b*` over `4912r`, matching the `26–39 L*, +22..+35 b*` kept kitchen shots):
       - **Pre-occlusion (`4912–4920`, `67,863 px`)**: `median L* = 23.9, mean = 28.8, p90 = 58.4, b* = +28.0` (per-frame median `L* = 20.0–27.5`, `b* = +24.0..+31.0`, vs `4912r` `10.6 / 16.0 / 38.4 / +15.0`).
       - **First occlusion (`4921–4926`)**: `0 px` mask (`Step 3` `4921–4924` `0` active tokens), keeping the passing foreground dancer 100% untouched.
       - **Mid-occlusion (`4927–4940`, `84,107 px`)**: `median L* = 26.7, mean = 32.1, p90 = 58.8, b* = +31.0` (per-frame median `L* = 23.5–37.3` on `4927–4938` and `45.9` on `4939`, vs `4912r` `12.5 / 19.5 / 45.9 / +17.0`).
       - **Second occlusion (`4941–4942`) & crown above white hat (`4943–4946`, `3,090, 6,003, 7,489, 7,108 px`)**: `4941–4942` stay 100% untouched (`0 px` mask), and the full horizontal crown dome above the white hat brim on `4943–4946` renders golden-blonde (`parts_held` `median L* = 38.0, 31.4, 27.5, 23.1`, `b* = +29.0, +27.0, +28.0, +28.0`, vs `4912r` `14.9, 17.6, 17.3, 13.3`).


