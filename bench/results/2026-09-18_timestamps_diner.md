# Cut timestamps on shot headers: one same-seed pair, 2026-09-18

Model: MiniMax H3, default chain (kitchen int8 dense, Sol with `qk_balance`,
plain token order), 345 frames at the trained canvas, seed 730451892. Clips:
`Video/block49_repro/diner_default_s730451892_*` (2026-09-15, prompt with its
two cut times) and `Video/timestamps_test/diner_plain_no_timestamps_*`
(2026-09-18, kitchen `0.2.35+sol.8176242`, a build shown the same day to
reproduce the older clip bit for bit). The prompts differ ONLY by the two
header timestamps (`[Shot 2] At 00:05.200, the shot cuts to` against
`[Shot 2] The shot cuts to`; likewise Shot 3 at 00:09.800). Labeled stack, not
blind: `Video/timestamps_test/stack_diner_with_vs_without_timestamps.mp4`.

## Why

The owner's house rule of 2026-09-18 removes timestamps from every shot header,
against both vendor guides, on the judgement that fixed cut times are "probably
a big cause of issues". Nothing had measured it in either direction.

## Where the cuts fell

`ffmpeg` scene detection (`select='gt(scene,0.3)'`), seconds:

| clip | scripted cuts | detected cuts |
|---|---|---|
| with timestamps | 5.2, 9.8 | 5.0, 9.54 |
| without timestamps | none | 4.75, 9.5 |

Both clips cut twice and in nearly the same places. With the times written,
the model did not hit them exactly either (it cut early both times); without
them it divided the clip into three shots of almost the same lengths by itself.

## The owner's look

"clip 1 [with timestamps] is a better scene. theres no window behind him in
clip 2. but theres nothing really wrong with either of them." The prompt puts
"steady rain streaming down the exterior window beside a vintage diner menu" in
Shot 1 without saying where the window is; the no-timestamp take left it out
of the opening frame.

## What it says

Removing the cut times did no harm to this prompt: same number of cuts, same
pacing to within a quarter of a second, each line still in its own shot, and
the owner found nothing wrong with either clip. It also showed no benefit: the
owner slightly preferred the timestamped take, for a staging reason (a window)
that the prompt leaves open and that looks like the difference between two
takes. So the rule is SAFE on this evidence, and its claimed upside is
unshown. One prompt, one seed, a three-shot scene whose scripted times already
fit its length; a prompt whose written times did NOT fit its length is where
the rule should help, and that has not been rendered.
