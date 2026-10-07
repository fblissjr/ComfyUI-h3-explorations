# The reference still given to the text encoder alone, in the masked lane: the owner's verdicts (2026-10-07)

lane: masked
verdict: owner, on two stacked renders of one clip, one seed each: with the still's copy in the video model switched off the result is worse in both places; on the upper-body recipe the subject is brighter, lit wrongly for the scene and looks superimposed; on the short whole-subject window only the stock arm carries the still's facial identity, at the encoder's 512 view and at the large shared view alike; the superimposed head that window already shows is there in all three arms, so it is not caused by the still's copy in the video model; sampling is shorter without the copy at the 512 view and not at the large one; the masked graphs keep giving the still to both

**What was asked.** The owner, 2026-10-07: reference stills encoded "solely
with qwen3-vl actually worked really well" in earlier workflows, and whether
that changes anything for this lane. The earlier records are
[`2026-10-03_encoder_only_reference.md`](2026-10-03_encoder_only_reference.md)
and the 2026-09-04 reference pathway verdict, both on ordinary reference
renders and both bounded at one seed. This is the same switch
(`MiniMaxH3AppendRefImage.use_vae` off) in a masked render.

**How to read it.** One clip, two windows, one seed, one render per arm,
judged by the owner on stacked videos, not blind. No session looked at a
picture. Figures are in
[`2026-10-07_masked_still_encoder_only.json`](2026-10-07_masked_still_encoder_only.json).

## What ran

- **The short whole-subject window** (`body141`): the shipped ref2va motion
  graph with the patches of the 2026-10-06 arm that showed the still's
  framing in the region, rendered stock again as the control, then with the
  still to the encoder alone at the graph's own 512 view, then at the shared
  large view, which is the configuration the 2026-10-03 record ran.
- **The upper-body recipe** (`upper`): the shipped upper-body graph with the
  still to the encoder alone at the 512 view. Its control is `baseline_again`
  of
  [`2026-10-07_frozen_cache_masked_upper_window.md`](2026-10-07_frozen_cache_masked_upper_window.md),
  the same window, seed and server process.
- The motion video goes to the encoder alone in every arm, as shipped, so the
  encoder-only arms have no reference rows in the video model at all.

## The owner's verdicts

On the upper-body recipe, stock beside encoder-alone:

> "looks like encoder only is worse - it seems brighter and lighting isnt
> correct with the scene. looks more superimposed"

On the short whole-subject window, stock beside the two encoder-alone arms:

> "has the head superimposed in the middle on all 3 clips, just fyi. but only
> the stock one actually LOOKS like and carries the identity of the original
> ref image. the middle one and one on the right (encoder only and encoder
> large view) both botch the facial identity"

## What it says

- **The still's copy in the video model carries identity and the match to
  the scene's light in this lane.** Without it the face is lost on one
  window and the subject sits wrongly in the light on the other.
- **The view size is not what fails.** The large shared view lost the face
  as the 512 view did.
- **The superimposed head on the whole-subject window is not that copy's
  doing**: it is in the two arms without it. That window's fault stays with
  the explanation in `masked_prompt_text.py` (a still that shows a person
  from the chest up, asked to dress a whole body), which is why the
  upper-body recipe exists.
- **Why this differs from the earlier "could not tell" results** is not
  shown. Those were ordinary reference renders; here the subject has to fit
  a scene that is already fixed. Reasoned, not tested.

## The time

| arm | sampling |
|---|--:|
| `body141`, the still to both | 99.7 s |
| `body141`, encoder alone, 512 view | 79.5 s |
| `body141`, encoder alone, large shared view | 97.5 s |
| `upper`, the still to both (`baseline_again`) | 313.4 s, the sampler's bar 5:12 |
| `upper`, encoder alone, 512 view | the sampler's bar 4:35; stage seconds not read |

The saving at the 512 view is real and is not taken: the picture is worse.

## What this does and does not show

- One clip, one seed per arm, the owner's eye on stacked pairs, not blind.
  It closes the switch as a free speed-up for the masked graphs as they are;
  it does not say the encoder-only route cannot work in a masked render with
  another prompt or another still.
- The upper arm's stage seconds were lost to a server restart before they
  were read; the bar's time is from the server log.
- Nothing here was looked at by a session.

## Files

- `2026-10-07_masked_still_encoder_only.json`: each arm's graph, patches and
  seconds, the control's, and the owner's words as given.
