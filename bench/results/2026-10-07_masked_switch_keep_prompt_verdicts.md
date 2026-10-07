# A region switched on partway, a kept patch inside the region, and three prompt arms: the owner's verdicts and what was measured after (2026-10-07)

lane: masked
verdict: owner, on stacked renders of one window each, one seed unless said: a region turned on partway through a window shows the new subject late and the same on and off the frame grouping (measured after: 12 frames late turned on at frame 153, 5 frames late at 162, the region regenerated as the original in between); a patch of the source kept inside the region keeps the held thing and the person becomes the original around it; a shadows sentence and a bare-noun subject change nothing they can see; the prompt rewritten in the vendor guide's form is better than the node's text on two seeds, with the subject facing away at one cut on one of them; the margin flicker they see in the switch arms is measurable at the seam and is the same size in the always-on arm

**What was asked.** Four ideas of the day, each rendered once on the
upper-body recipe and put to the owner as stacked videos: turning the region
on at a chosen moment (the first step of a timed change), keeping what the
original subject holds as its own pixels, two small changes to the prompt
node's text, and that text rewritten in the form of the vendor's guide
(`vendor_guides/ref_en.md`). The owner answered on the masking board with
annotated stills, which are not tracked. This record holds their verdicts and
the measurements made afterwards to check what they saw. **A verdict is the
owner's eye on playback; nothing below replaces it.**

**Conditions, every arm.**
`workflows/h3_video_to_video_masked_upper_song_ref2va_motion_api.json`, one
window of 345 frames, typed arms through `bench/run_graph_arms.py`, the same
seed in every arm of a comparison unless a second seed is named. The Subject
Track's `pick` was `largest` in every arm: these were rendered around the
change of its default (`docs/wiki/decisions.md`, 2026-10-07). The switch and
keep graphs were scratch copies of the shipped graph and are not in the tree;
what each changed is said below. Every figure is in
[`2026-10-07_masked_switch_keep_prompt_verdicts.json`](2026-10-07_masked_switch_keep_prompt_verdicts.json).

## 1. The region turned on partway through a window

`thrill_2160.mkv` from 67.0 s, canvas 1024x768. The subject's mask was
blanked before frame N with core's mask nodes, so the whole frame is source
until N and the usual region regenerates from N on. Two arms: N = 153, a
multiple of 17 and the first frame of one of the video model's latent steps,
and N = 163, neither (by the lane's frame grouping the region turns on at
162). The control is the shipped graph, region on throughout, same server
process and seed.

**The owner:** both switch arms show a delay between the region turning on
and the new subject appearing, and a flicker; on and off the boundary look
alike. In the second shot of the window both switch arms show a
constant flicker at the margin that the always-on arm does not.

**Measured after** (`bench/masked_render_against_source.py landing`, a box on
the subject's torso, each arm against the control, frames 140 to 180):

| region on from | first frame the arm matches the control's subject | late by |
|---|--:|--:|
| 153 | 165 | 12 frames |
| 162 | 167 | 5 frames |

Between the two frames the region is being regenerated and is drawn as the
original subject: the arm stays as far from the control as it was before the
region turned on, then falls within one frame. So the delay is real, it is
not a property of the 17-frame grouping, and it is not one fixed length.
Whether it follows the action or a property of the model is not known; the
same switch at two other moments is the next render.

**Added the same evening: the switch at two other moments.** Region on from
frame 85 and from frame 221, both a multiple of 17 and the first frame of a
latent step like 153; same control, same seed; not shown to the owner yet.
Measured as the difference from the source on the pixels where the control's
subject differs most from it:

| region on from | lands at | late by | what is at the landing frame |
|---|--:|--:|---|
| 85 | 109 | 24 frames | nothing the shot table marks |
| 153 | 165 | 12 frames | nothing the shot table marks |
| 162 | 167 | 5 frames | nothing the shot table marks |
| 221 | 237 | 16 frames | the window's first cut |

In all four the arm leaves the source slowly for the first ten frames or so
(a regenerated subject that still reads as the original) and then changes
within one frame. **The delay is not a length that can be led by**: four
switches land at three moments, and the one that had a cut ahead of it landed
exactly on the cut. That fits "the model holds the subject it has been
shown until something in the clip lets it change", which is a reading: the
two landings with no cut were not looked at for what happens there. What it
supports for a timed change: put the switch on a cut. A change in the middle
of a shot needs something this test does not have.

**The flicker** (`flicker` mode: the residual render minus source, and its
change between frames, in bands read from the render's own overlay). In the
second shot the band just inside the seam changes about three times as
much from frame to frame as the frozen band just outside it, and its mean
brightness steps by about one level a frame; in the first shot that
band is nearly as quiet as the frozen one. **All three arms measure alike
there**: the switch arms' brightness step is a tenth to a fifth above the
always-on arm's, and their single-pixel flicker is within two percent of it.
So the flicker the owner sees is at the seam and is measurable, and this
measure does not show the switch arms worse than the control. Either the
measure misses what the eye catches or the control has it too. Not resolved.

## 2. A patch of the source kept inside the region

`lotsofpeopledance_0414_0720.mkv` from 4.0 s, canvas 1344x768, one shot with
no cut (the shot table has one shot), over which the subject's size in the
frame changes a great deal. Three arms on one seed:
stock; the Masked Source's `keep` input fed by a second Subject Track on a
phrase for the held thing (found on frames 0 to 43); `keep` fed by the part
node's `held` output, which asks no model for a name (non-empty on frames 0
to 39 and 274 to 316).

**The owner:** the held thing is there in both kept arms and missing in
stock. In both kept arms the person is the original for as long as the thing
is kept. The no-name arm turns back into the original again late in the
window; the phrase arm loses the subject at the end. Stock holds the new
subject throughout.

**Measured after.**

- *The region was the same in all three arms.* Read from each render's
  overlay, the regenerated region overlaps stock's by 0.99 on average in both
  kept arms; the frames that differ are the ones with a kept patch. So the
  tracking and the mask did not differ; the kept patch is the only
  difference in what the video model was given.
- *The person in the kept arms is regenerated, not frozen source.* A rough
  measure (fixed boxes, the source decoded by ffmpeg and not by the lane's
  loader, so its floor is high): over the first seconds the chest box differs
  from the source by one and a half to four times the frozen background's
  figure in the kept arms, and by eight times in stock. The face box is
  confounded by what the render changes there and is not read.
- *Late in the window the part mask nearly vanishes* (its share of the frame
  falls to well under a hundredth around frame 308, in all three arms alike).

**Reading, not tested.** Sections 1 and 2 have one thing in common: pixels of
the original subject sit against the regenerating region, just before it in
time in the first and inside it in space in the second (a kept patch is whole
16-pixel blocks, so it carries a little of what surrounds a small thing), and
in both the video model draws the original and not the still's subject. The
other candidate is the motion reference, which shows the original subject to
the text encoder in every arm. The late changes in the kept arms are one
render each on one seed, and a rendered clip cannot attribute them
(`AGENTS.md`, "A rendered clip cannot A/B a numerical change").

**What follows.** Keeping source pixels inside the region is the wrong way to
keep a held thing that touches the person. Next: the thing pasted back over
the finished stock render with the `held` mask, which needs no model run and
gives the video model nothing of the original. `keep` stays as an input: it
is the right tool for a thing that does not touch the subject, and for
excluding another person from the region.

## 3. Two small changes to the prompt node's text

Same window as section 1, the shipped graph with one field changed: a
sentence added through `add_to_shot` asking for edge blending, light wrap,
contact shadows and occlusion (the phrases are Patrick Sheedy's, from
https://patricksheedy.com/posts/minimax-h3-character-swap-part-2, tried here
on a masked lane); and the subject left
as the node's default word instead of a typed description.

**The owner:** no real difference from stock in either; the simpler text
stays. Neither is adopted, and a typed description of the subject is not
shown to be needed on this recipe.

## 4. The prompt rewritten in the vendor guide's form

Same window. The node's assembled text replaced by a typed prompt in the
guide's fields: the subject's features stated in the definition, the
retention line and the first shot; no "that person"; no "untouched"; the role
said once without mouth choreography; shot headers without times, by the
house rule (`docs/prompting.md`, 3.1). Stock and rewrite on the graph's seed
and on a second seed.

**The owner:** the rewrite is better than stock on both seeds (movement and
head movement follow the input more closely); the second seed is nearly
perfect; on the first seed the subject faces away from the camera at the cut
to the second shot, which is wrong. An earlier version of the rewrite, with
times in its headers, showed an unwanted turn on that same seed.

**Not yet changed:** `masked_prompt_text.py`. One more render first, the
rewrite on the first seed without two phrases that may invite a back view;
the stock text has similar words and does not turn, so the wording is a
guess and the seed may be the cause.

## Also answered the same day

- **The cache on the upper-body recipe:** the owner cannot tell stock from
  cached on the pair of
  [`2026-10-07_frozen_cache_masked_upper_window.md`](2026-10-07_frozen_cache_masked_upper_window.md).
- **The kept mask on disk stays** (the owner, on the board): its removal,
  asked for on 2026-10-06, is withdrawn while it still saves the SAM pass on
  a second render.

## What this does and does not show

- One window per idea, one seed except section 4's two. Each verdict is on
  those renders.
- The landing frames are a difference in one box against one control; they
  say when, not why.
- The flicker measure cannot rank what the eye ranks; it locates the flicker
  at the seam and sizes it against a frozen floor.
- The keep measurement against the source is rough and says so in the json.
- Not run: the switch at other moments; a kept patch without the motion
  reference; the paste-back; the rewrite without the two phrases.

## Files

- `2026-10-07_masked_switch_keep_prompt_verdicts.json`: the arms, the landing
  series per frame, the flicker bands per span and arm with the calibration,
  the keep arms' region overlap and rough source differences, and the owner's
  verdicts as recorded here.
- `bench/masked_render_against_source.py`: the two measurements; its
  docstring says what each does not show.
