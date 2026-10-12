# Where the fun clip stands, a shot and a subject at a time: what is laid, what is left, what is unsure

2026-10-12, session mrf, at the owner's ask ("finish the end to end video; find the gaps first"). Read from
files by `bench/shot_coverage.py`; no clip was watched for this page and nothing here says whether a render
is good. The owner's own verdicts on playback are where the records of 2026-10-10 and 2026-10-11 and the
page's `answers.jsonl` hold them.

## What it buys

On the evening of 2026-10-11 a joined clip was made and called the best so far; its own table says it is
the 10th's table with five rows swapped, and nobody could say per shot and per subject what the owner had
ruled, what existed and what was missing. This page is that list, made by a tool from three files, so it
can be made again after every render and is not typed by hand.

## How it was made

    bench/shot_coverage.py --shots data/fun/shots.json --plan data/fun/plan.json \
        --table data/2026-10-11_fun_joined_b/t_fun_0000_1919_b.txt --renders data/fun/poc/renders.json \
        --json data/fun/coverage_2026-10-12_table_b.json

- `data/fun/shots.json` (untracked, the video's record): the trackers' cuts from three previews and, for
  every shot and subject, whether they are in it, with who said so and from which frame or file. Where an
  owner's answer on a numbered picture exists it is the source (rows `b20` and `who` of
  `data/fun/poc/answers.jsonl`); where two sources disagree the row says `unsure` and quotes both.
- `data/fun/plan.json`: the owner's defaults of 2026-10-11 (the lead, face only; the second subject, face
  and hair; the third subject, face only, his own hair; `docs/wiki/decisions.md`, 2026-10-11) and their
  dated notes on stretches, each with where it was said.
- the delivery table the joined clip of 2026-10-11 was built from, and the page's inventory of every
  render of this video read from each render's saved graph.

The whole printout, every row with its sources, is beside the JSON under `data/fun/`. The tool reads the
graph's kind only (parts or whole): face against face and hair is not in a saved graph's summary.

## The states, in frames

| subject | as planned | partly | not as planned | source, said why | source, no line | presence unsure |
|---|---|---|---|---|---|---|
| the lead | 792 | 377 | 0 | 259 | 106 | 33 |
| the second subject | 0 | 154 | 0 | 616 | 0 | 319 |
| the third subject | 0 | 377 | 84 | 47 | 0 | 406 |

A frame counts under a subject only on the shots that subject is in (or is unsure on). The video is 6395
frames; everything here stops at 1919, where the working copy, the cuts, the answers and every render stop.

## The gaps, by subject, in source frames

**The second subject (face and hair wanted; her whole-person renders were ruled out on 2026-10-11).**
Nothing of her is laid as planned anywhere. The one face-and-hair render (p4, 14-154) covers her first
shot but its last 13 frames (155-167). Left as the original with a line saying why, because only
whole-person renders exist: 278-293, 415-437, 484-515, 604-1009, 1010-1038, 1039-1060, 1093-1133 (small
at the back), 1419-1465 (barely visible; the table's line says she is not there and the owner's answer
says she is). Unsure whether she is there at all: 1134-1188 and 1466-1729 (small, soft, at the back; no
numbered picture answered). The load now on the card (mroh, 2026-10-12: `second_k_0604`, 604-1009, p4's
recipe) is the first of these; 1010-1038 and 1039-1060 wait on it.

**The lead (face only).** As planned on 294-414, 438-483 (zoomed), 604-1009, 1010-1038, 1061-1092,
1093-1133, 1156-1188, 1222-1305. Partly: 1306-1418 (laid to 1402; 1403-1418 bare) and 1466-1729 (laid
1466-1535 and 1575-1729; 1536-1574 bare). Never attempted, no line: 168-185 and 516-603 (the owner named her
on frames 172, 180, 552, 576). Said why: 1134-1155 (her face shows eight frames only), 1419-1465 (her head
hidden behind him), and the five inset shots 1730-1919, where the owner wants her face swapped in the
circles and the face is a few tokens tall, so the zoomed load with a motion video is the route, which is
node code today. Unsure: 1189-1221, where the owner clicked "1 = the lead" on frame 1193 and the earlier
reading says a hand and an object; put to the owner.

**The third subject (face only, his own hair).** Every render of him is a whole-person one: 1222-1305 (not
as planned), 1306-1410 and 1466-1631 (partly, and whole). The owner's words on the joined clip: his face
from the reference still, nothing else, and singing where his mouth moves. No face-only render of him
exists. Unsure whether he is in 604-1009 (nobody has said; those previews tracked two subjects).

**Not about a region, from the owner's rows on the joined clip:** the lead's mouth is shut where the
original's sings on 1063-1306 (text, not mask), and the dad's too.

## What this page does not say

Whether any laid piece is good: the owner's eye, and the records. Whether the cuts are right: nobody has
looked at them as cuts (one shot, 604-1009, was looked at frame by frame by mroh on 2026-10-12 and holds).
Whether the second subject is in the shots marked unsure: a numbered picture, answered, settles each.
