# A tracker's take against a person's answer: what size and likeness could have told the gate (2026-10-10)

lane: masked video to video
verdict: not a rule. On eighteen takes of one clip, eleven of them wrong, neither size reading separates wrong from right; likeness does on this stretch, and for a reason that does not carry to another one.

**Read this first.** One clip, two loads, fifteen shots, three named
subjects, no corrections typed on purpose. "Right" and "wrong" are one
person's reading of the tiles and the source (the lead session), written
to an untracked file; nothing about them was measured. The figures beside
each take are the capture's. This record exists so that the size rule is
not proposed again as news, and so that the likeness line it shows is not
picked up as a rule either.

**What was asked.** The gate (`bench/capture_masked_run.py preflight`)
blocks an absence with people on screen and a take under the tracker's
own line. It has nothing for a take of the wrong person over the line:
such a take reads "clear". On the last five shots of this clip a large
person the job does not name is on screen with a named subject small in
an inset, and five takes of that person read clear. From the sizes alone,
knowing the shot, a reader could tell. Could a rule?

## How

Two no-sampling previews, captured with `files` and read with `rules`.
For every shot a tracker called taken: the likeness and the line from the
shot table, the subject's share of the frame over the shot, the same
figure on the shot its table calls picked, and every detected person's
share on the shot's tile. The rows, with where each column comes from,
are in
[`2026-10-10_take_size_and_likeness_against_known_answers.json`](2026-10-10_take_size_and_likeness_against_known_answers.json).

## What the eighteen takes say

| reading | wrong takes (11) | right takes (7) | separates |
| --- | --- | --- | --- |
| the take's share over the subject's share on its pick shot | 0.03 to 3.4 | 0.21 to 1.81 | no |
| largest on the tile, over five times the next person | fires on 6 | fires on 1 | no |
| likeness | 0.59 to 0.873 | 0.80, 0.873, then 0.88 to 0.92 | on this stretch, with two right takes on the wrong side |

**Share against the pick.** The unnamed person's takes are 1.1 to 1.6
times the subject's pick share, inside the right takes' range; the one
right take of the inset subject is 0.21 of her pick, the smallest of the
right ones. A subject's size changes from shot to shot for ordinary
reasons, and it did here.

**Largest on the tile by a wide margin.** It fires on six wrong takes
and one right one. The right one is a subject in front with a speck
detected behind, which is the commonest right shot there is: on a clip
with a subject in the foreground and somebody small at the back it would
fire on most shots.

**Likeness.** Every wrong take is at 0.873 or under. Five right takes are
at 0.88 or over, all of one subject. The two right takes below are a
subject small in an inset (0.80) and a take the tracker made because
there was one candidate (0.873, exempt from its line). The line of the
subject with the five high takes sat at its floor of 0.80 on both loads
(the `line` column has each take's). The tracker's owner reads the cluster
this way, and it is why the line is not a rule: likeness here is taken
against the other person on the pick frame, so it is high wherever the
scene is the pick's own scene. The split is between that scene and
everything else, not between the right person and a wrong one. On
another load of the same clip the right answer after a change of look
was 0.64.

## What follows

- No size rule is added to the gate. What told the five takes apart was
  a sentence from a person who had seen the shot. The place for that
  sentence is the job (a stretch left out for a subject, or a statement
  that a subject is small from a frame on), not a threshold.
- The gate's per-shot list (`shots`) prints the state, likeness and
  share beside every "clear", and says in its header that clear means no
  rule blocks.
- A line set from takes a person has confirmed, against everybody else
  on the same looks, is the tracker owner's open item; these rows are its
  first sample and are kept as that.
- Seen by nothing today: a take of a person nobody named, over the line,
  in a shot where the joint who-is-who decision has a subject free to
  hand that person to.
