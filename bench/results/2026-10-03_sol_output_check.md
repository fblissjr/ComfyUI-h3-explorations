# The output check for sparse attention (2026-10-03)

**Status: closed by the owner the same day.** No measurement predicts the
owner's verdict on the scene that was tested, the owner does not want an
automatic judge built ("no"), the shipped default stays ("leave the shipped
default as is"), and the remaining renders were stopped ("i dont need more
renders on this topic"). What the search found is below, because it says what
a number can and cannot do here, and it corrects a reading of the 2026-10-02
panel.

**Why it ran.** The owner put it ahead of the per-head calibration
(`2026-10-03_tau_sweep.md`): every sparse lever in flight is lossy, nobody is
judging clips, and local error did not predict the 2026-10-02 verdicts
(`2026-10-02_sol_dense_blocks_panel.md`), so a calibrated table had nothing
that could accept it.

**Conditions.** `workflows/distill_experiments/h3_probe_r2v_step_switch_pdd8_flashgen_h080_savelat_api.json`
(PDD8 first stage, six evaluations; FlashGen finisher, two), 1344x768, 345
frames, one reference still, market scene unless said, seeds 730451891 to
730451893. Arms and patches: `bench/sol_output_distance_arms.json` and the
rows in `2026-10-03_sol_output_distance.jsonl` (each row carries its patches,
seed and sampler seconds). No probe, observer or sweep armed. Scored by
`bench/score_output_distance.py` and `bench/score_clip_shots_loudness.py`.
Arms rendered after the manifest was written (the single-stage arms at the
other seeds, the 512 encoder view) are in the rows.

**Layout.** Everything here ran on the shared-view layout: the reference
still goes to the video VAE and to the text encoder at the same 2048 short
edge, so one still contributes its latent rows and about as many encoder
tokens again. The owner made a 512 encoder view the default for reference
stills later the same day; on that layout the conditioning prefix of a
one-reference graph is about half as long, and a timing, a routed share or a
table taken here describes a different shape.

## 1. The render is deterministic, so the judged set can be re-rendered

The twelve clips judged on 2026-10-02 are on the share and their seed was
held; their saved latents had been deleted. Today's shipped render at that
seed is byte-identical (`cmp`) to the judged market clip for the shipped
setting, and today's all-dense render to the judged dense clip, across a
kitchen rebuild and server restarts. A guard render taken after a peer's
conditioning-node edits were loaded matched again, clip and saved latents.

A second guard at the end of the day (row `guard_shipped` in
`2026-10-03_sol_output_distance.jsonl`), on a server that had loaded every
commit through `7c7cc468`, the LoRA branch change `753e62a0` among them: the
clip is byte-identical to the morning's, and the four saved latents (video
and audio, after each sampler) are equal tensor for tensor. Sampler 335.3 s,
first stage 251.0 and finisher 84.3, against 335.2 s, 251.1 and 84.1 in the
morning's row at that seed (`2026-10-03_r2v_finish_e2e.jsonl`).

## 2. Latent distance from the dense render

Relative L2 on the saved latents against the all-dense render at the same
seed (`2026-10-03_sol_output_distance_seed89{1,2,3}.json`). "Coarse" is the
video latent average-pooled over 8x8 latent pixels; "pass1" is the latent
after the first sampler.

Seed 730451892:

| arm | video | coarse | pass1 | audio | sampler s |
|---|--:|--:|--:|--:|--:|
| sparse in the finisher only | 0.211 | 0.062 | 0.000 | 0.093 | 442 |
| stock bf16 attention on every block | 0.586 | 0.334 | 0.093 | 0.300 | 1122 |
| sparse in the first stage only | 0.872 | 0.653 | 0.167 | 0.666 | 373 |
| shipped (sparse in both) | 0.873 | 0.653 | 0.167 | 0.668 | 337 |
| reference still to the encoder only (mrblue's arm) | 0.950 | 0.747 | 0.188 | 0.718 | |
| encoder view of the still at 512 | 0.968 | 0.782 | 0.192 | 0.882 | 276 |
| another seed, same setting (three pairs) | 1.141 | 1.001 | 1.400 | 1.290 | |

The all-dense render's sampler is 479 s.

- **The distance is mostly saturated.** Stock attention against the dense
  INT8 kernel is a change the owner accepts, with about a tenth of Sol's
  per-call error (`2026-10-03_tau_sweep.md`, "Premises"); it already moves
  the final latent half way to an unrelated sample.
- **Nearly all of Sol's distance arises in the first stage.** Sparse in the
  first stage only is the shipped sample to within 0.22; the finisher's Sol
  moves the sample about 0.2 whichever first stage it follows. The same
  holds at the other two seeds (first-stage-only against shipped: 0.211 and
  0.213; against dense 0.788 and 0.846, as shipped).
- **It can certify near-identity.** Sparse in the finisher only is nearer
  the dense sample than stock attention is, at both seeds it was rendered at.

## 3. Cuts and loudness

Cuts by ffmpeg's scene score over `measure_clip_delta.CUT_SCORE`; both
prompts script three shots. Loudness is EBU R128 integrated, against the
dense clip.

Market (`2026-10-03_judged_clips_market.json`, `2026-10-03_market_cuts_by_seed.json`):

| render | 891 | 892 | 893 |
|---|---|---|---|
| dense | 3 cuts | 2 cuts | 3 cuts |
| stock attention | | 2, at dense's times | |
| shipped | 3 | 4 | 3 |

Every sparse-finisher render at seed 892 (the panel's five settings and
today's) has three or four. The owner confirmed by eye that the extra cut
near 9.4 s in those is real: "they hard cut to the ref image. the ref image
guy looks fine".

Backstage (`2026-10-03_judged_clips_backstage.json`): two cuts on every
judged arm. Loudness against dense: the arm with every conditioning row exact
+1.1 LU, the four other sparse arms +3.8 to +4.5, as the panel record found.

## 4. The owner's eye, and what it does to the measurements

Verbatim, 2026-10-03, on market clips unless said:

- shipped at 892, the judged setting: "at the 7s mark (3rd shot) hes walking
  backward with two orange crates, one stuck to his back. the dense doesnt do
  that". On the panel's optC clip: "he puts the coins into the orange crate
  as hes walking back, so it gets entangled"; later, "i meant the
  entanglement between shot 1 and 2 ... shot 2 is where the weird shit
  happens".
- "od_market__dense_00002 is fine. od_market__stock_00001 doesnt have two
  crates but its not as well done as dense and still slightly awkward but way
  better and acceptable. od_market__dense_00001 is fine too.
  encoder_only_00001 is fine too. ship_00003 is fine too.
  od_market__dense_00003 is fine too. the rest that i glanced at from todays
  renders had that double crate weirdness".
- "od_market__finisher_only_00001 is fine. od_market__pdd_only_00001 is
  fine". The first-stage-only clips at 891 and 893: "These are fine".
  `ship_00001`: "no double crate. looks fine."
- Backstage, encoder view at 512: "the characters looked good but the woman
  responded 'nobody' when the man should have. not sure thats related tho".

| setting | 891 | 892 | 893 | sampler s |
|---|---|---|---|--:|
| dense in both stages | fine | fine | fine | 479 |
| sparse first stage, dense finisher | fine | fine | fine | 373 |
| sparse in both (shipped) | fine | double crate | fine | 337 |
| dense first stage, sparse finisher | not looked at | fine | not rendered | 442 |

- **No measurement tracks the verdict.** Clips with one extra cut are fine
  (dense at 891 and 893, shipped at 891 and 893) and one with two is fine
  (the encoder-only arm). A shipped clip the owner passed sits as far from
  its dense twin as the one with the double crate. Loudness tracked the
  audio verdicts on backstage and says nothing on market.
- **The 2026-10-02 panel judged the one bad seed.** It read "on market every
  Sol arm got a scripted action wrong and the dense render did not" as a
  property of sparse attention. At the two neighbouring seeds the shipped
  setting is fine. The panel's own caveat ("could be a lucky seed; the second
  seed decides that") was the right one, with the luck on the other side.
- **At the bad seed the finisher decides it.** The same first-stage latent
  finished dense is fine; finished sparse it has the double crate.
- A hypothesis recorded on the way and left untested: sparse attention drops
  video-to-video links far apart in time, which is where a shot plan is
  settled (ditman's; board finding d2-03). The cut evidence that prompted it
  did not survive section 3.

## 5. The encoder's view of the reference at 512

The owner's ask of the same day. Shipped settings otherwise, seed 730451892.

| scene | sequence length | sampler s | against the 2048 view |
|---|--:|--:|---|
| market | 112,589 | 276 | 119,485 tokens, 337 s; video latent 0.818 from the shipped render |
| backstage | 112,720 | 272 | 333 s (the panel's row); +3.8 LU against the dense clip where shipped is +1.1; cuts at 3.46 and 6.96 s against 4.75 and 9.58 |
| backstage, dense attention in both stages | 112,720 | 434 | 480 s for the dense render at the 2048 view; +4.1 LU against that clip; cuts at 4.29 and 9.25 s |

*Corrected the same day:* the first two token counts were swapped when this
was first committed; the reference line in the server log beside each render
is the authority.

One clip per row, each a different sample from the shipped one. The sparse
backstage clip is loud and has a line spoken by the wrong character (the
owner's ear, above). The owner asked for the third row to rule the view in or
out ("Ok do one more with the 512 so we can rule that out"), and of that clip
said: "dialogue is good here and everything is fine". So the swapped line
does not come with the 512 view alone, and from one clip cannot be pinned on
sparse attention either. **The level shift does come with the 512 view**: it
is as large with dense attention as with sparse
(`2026-10-03_backstage_qview512_clip.json`), and the owner heard that clip
and passed it. The 512 view became the default for reference stills the same
day (`docs/state/decisions.md`).

## Decisions (owner, 2026-10-03)

- No automatic judge for action defects. The owner's glance is the
  acceptance test for a lossy change.
- The shipped sparse default stays. A dense finisher (the second sparse
  node's `dense_blocks` set to every block) is a remedy for a single render
  whose action came out wrong, not a default.
- No more renders on this.

## What this does not show

- One scene carries the verdict table, at three seeds, by one person's
  glance, with the setting known. A rate for the double crate under the
  shipped setting cannot be read from three seeds.
- Sparse in the finisher only was looked at on one seed.
- Nothing here was rendered on the dialogue scene with a dense finisher.
- Not rendered, stopped with the queue: the first stage with its first two
  evaluations dense, tau 0.5, tau 0.25, the fourth single-stage replication.
- The 512 view with sparse attention, which is what a generated graph runs,
  has one dialogue clip, and it is the one with the swapped line.
