# Block-49 probe renders, market scene: viewer feedback, verbatim

2026-09-15. Three renders of the shipped text-to-video graph's prompt (the
covered-market scene), seed 730451892, 345 frames at 1344x768, 16 steps, on
the served build (sage fork v0.7.19, kitchen 0.2.34+sol.2aff3c5). Watched as
the original files, in this order:

- clip 1 = `h3_t2v_00019-audio.mp4` (shipped: INT8 attention everywhere)
- clip 2 = `h3_probe_t2v_balanced_00001-audio.mp4` (channel-balance node + sage `fp8++ balanced`)
- clip 3 = `h3_probe_t2v_exact_tail_00001-audio.mp4` (blocks 45, 48, 49 on bf16 attention)

The blind singles made for this comparison were not scored; every note below
is about the originals in the order above.

## Outside viewer, first pass

> general - woman changes identity in all the videos
> vid1. the guy morphs when he turns around
> vid2. is the only one where the man ends up carrying two crates but it kinda morphs in
> vid3. best quality perhaps

## Outside viewer, second pass (watching in the same order, on a phone)

> I don't know which is which order wise on mobile but the one where the
> dude morphs backwards is the worst. The last one seemed the best, no
> morphing in the crate, and the way he shifts the weight and carries the
> crate is more natural, including resting it on the edge of the table
> instead of holding it with one hand when he puts the coins in the can.
> Also maybe less motion blur in that clip too, but it's hard to say

## Owner, who has rendered this scene many times

> actually he moves out of the way in clip 3 of the market scene too. ive
> never seen that in this scene before. in #1 with int8 convrot across the
> board, he always morphs. in clip 2 he goes straight. in 3 with bf16 on
> the sensitive layers, he moves AROUND ppl. and the audio seems better too.
>
> in clip 1 he and the crate morph into something else. ive seen that often
> in this scene in prior renders. the guy and the crate like morph into a
> different direction entirely - it happens often with this scene.
>
> in clip 2 he doesnt... its way better... but clip 3 is even better. and i
> dont see anything WRONG with it - its much better than clip 1. but clip 3
> is way better in subtle ways.
>
> clip 3 is louder audio, sounds crisper... and someone said this and i
> noticed it too for the first time in this prompt/scene: "I've never seen
> him prop the crate against the table before either in that prompt. He
> always used his superhuman strength to one hand it like a beast." - he
> actually sets the crate down on the table and then lifts it up. coupled
> with moving around the people walking toward him its just a better scene.

Interpretation and caveats: `docs/attention/block49_quant_error.md`, section 6.

## Owner, later: what was wrong with clip 2, and what the morph is

> There is one very minor problem with the crates. He says he'll "take two".
> In the worse outputs can't decide if a box is one horizontal segment or
> two, and that's probably why it morphs and acts weird.
>
> In the best output, it decides a single tray or box is just one horizontal
> segment, and they stack into two. It's much more consistent in this
> depiction, however it would be impossible to carry them the way he is if
> the segments are two separate trays stacked together. If he held it in the
> middle like that, the bottom tray would fall.
>
> In the other gens, it seems like the two horizontal slats are meant to be
> one deeper crate vs two shallow ones. Not sure which is more accurate for
> that kind of market though.

## Measured, 2026-09-15 evening: the market pair (levers vs policy) and the audio question

The owner could not see a difference between `h3_probe_t2v_levers_00001`
and `h3_probe_t2v_policy_00001` but heard the policy clip as louder.
`ffmpeg ebur128` integrated loudness / true peak, and `bench/measure_clip_delta.py`
(frame-to-frame motion, %busy at the file's threshold):

| clip | LUFS | peak dBFS | %busy |
|---|---|---|---|
| shipped (h3_t2v_00019) | -17.5 | -3.9 | 44.8 |
| exact_tail | -14.6 | -1.5 | 36.9 |
| levers | -20.4 | -7.2 | 28.8 |
| policy | -15.8 | +0.2 (clipping) | 35.2 |

`bench/compare_clip_pixels.py` levers vs policy: every frame differs, mean
abs diff 29 of 255; different takes, as any numerics change gives.

Read against the diner batch (six clips, three arms, two seeds), where all
six sit within about two LU of each other and no arm is louder
(`2026-09-15_block49_diner_batch.md`): the market loudness differences are
take-to-take variation on one scene, not an effect of the bf16 tail. The
audio thread from the morning closes on that evidence. Nothing measured
here ranks the two clips' video; the pixel and motion tools say "different
takes, comparable motion", which is all they can say.

## Viewers on the Tier 1 pair (clip 1 = levers, clip 2 = policy), 2026-09-15 evening

The owner: "i cannot see the difference between them" at first; another
viewer:

> The right has better small hands. (Much better than left, but still AI
> bad.) Look at the woman's hands near scene end, and at the woman's hand
> on the left of the shot carrying a bag.

> Two has better coins

The owner, after: "i saw this too. look at 8s-9s mark in both clips."

Checked on extracted frames (8.0, 8.5, 9.0 s; 13.8, 14.2 s): in the policy
clip the left tin is open and full of coins through the beat and at 8.5 s
the fingers hold a coin over the right tin; in the levers clip the left
tin's lid stays closed and the hand reaches into an empty tin with no coin
anywhere. The same failure shape as the crate: a small prompt-named object
left undecided by the arm without the bf16 tail. The hands claim could not
be settled from single frames (the takes diverge into different poses by
the end); the coin beat is the clean evidence. So on this scene the free
levers do not fully reach the ceiling; the bf16 tail still buys the last
commitment on small objects.

*Corrected 2026-09-15, same evening.* The paragraph above said the levers
clip leaves the coins undecided. It does not: at 0.2 s spacing (7.6 to
8.8 s, crops on the tins) the levers clip shows one small coin in the
fingers at 7.8 s going into the tin at 8.2 s; the policy clip shows a
stack of several larger coins dropped more slowly, with the left tin open
instead of lidded. Both render the exchange; the policy clip makes it
legible, the levers clip makes it small and fast. The owner's reading
("faster motion and smaller coins") stands; "undecided" was an artefact of
half-second frame sampling. The crate parallel is withdrawn: this is a
legibility difference, not an object left unresolved. What remains true is
that a viewer separated the pair on it, so the bf16 tail still buys
something visible on this scene.

A third viewer on the same pair:

> super close call. Right (clip 2) seems like maybe faces are a tiny bit
> less distorted but theres nothing jumping out at me as substantially
> different quality wise

Three readings of the Tier 1 pair so far: cannot tell them apart (owner,
first look), clip 2 on hands and coins (one viewer, the coins verified as
a legibility difference above), and "super close, maybe faces" (this one).
Against the morning's three-way look, where every viewer separated shipped
from the rest without prompting, this is a much smaller gap: the free
levers close most of it, the bf16 tail is a small remaining edge that
takes a careful look to find.
