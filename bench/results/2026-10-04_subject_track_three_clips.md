# The subject track on three clips: what a match is compared on (2026-10-04)

Session mrhf, evening. Probes outside the server, loading the module by path
and calling it with core's real SAM 3 detector on the card (mrblue's method,
from their validation of the same node earlier that day); then the landed node inside
the server as mask-only graphs. The probe scripts and outputs are local and
untracked.

## The clips

- band: `thinkaboutthings_compressed.mp4` from 95 s, 810 frames. Six people
  in one sweatshirt, cutaways. The lead is in shots 3 (picked), 5 and 7.
- solo: `idontwannatalk.mp4`, 768 frames. One singer, full-length shots and
  close-ups, and a hanging microphone the detector marks as a person while
  the frame is empty.
- car: `friday_chorus_14s.mp4`, 345 frames. New today. Several young people
  in a car, cutting between group shots and close-ups. The lead is in shots
  3 (picked), 5 and 7, and small and under a dissolve in shot 1; shots 2 and
  4 are close-ups of two other women. Who is who was judged by eye on the
  node's preview tiles.

## The node as landed in 0.186.45, on the car clip

It took the two other women as the lead (shots 2 and 4, at 0.82 and 0.83,
over a line of 0.80) and missed her in shot 1 (0.76). Seen on the tiles.

## Three signatures, each shot's best score

Relative similarity on band and car, plain on solo (nobody else on its pick
frame). `top`: the trunk's features under the top third of the mask, which
is what 0.186.45 compares. `head`: the same features under the head SAM 3
finds for the phrase `head` inside the person's mask. `crop`: the trunk run
again on the head's box and pooled under the head there.

| clip | signature | lowest shot holding the subject | highest shot not holding them |
|---|---|---|---|
| band | top | 0.94 | 0.77 |
| band | head | 0.94 | 0.85 |
| band | crop | 0.88 | 0.73 |
| band | lower of top and head | 0.94 | 0.77 |
| car | top | 0.87 | 0.83 |
| car | head | 0.86 | 0.77 |
| car | crop | 0.79 | 0.82 |
| car | lower of top and head | 0.86 | 0.76 |
| solo | top | 0.73 | (nobody else) |
| solo | head | 0.72 | (nobody else) |
| solo | crop | 0.90 | (nobody else) |

Car's shot 1 is not counted as holding the subject in this table; she
scores 0.76 there on the lower of the two, the same as the woman in shot 2.

Reading:
- No one signature is right on all three. The top third takes wrong people
  on the car clip, the head alone takes a wrong person on the band clip, and
  the crop fails on the car clip.
- The lower of top and head takes exactly the lead's shots on band and car
  with the node's own automatic line.
- The crop is the only one that holds across the solo clip's change of
  framing, so the fall there is a matter of scale and not of region. It was
  not adopted: it fails on the car clip and costs a trunk pass per person.

## The microphone

On the solo clip every detection on which no head was found is the
microphone (six detections on four frames), and every detection of the
singer has one. Before today a score floor kept it out, by 0.06.

## The node after the change (0.186.47)

A match is the lower of top and head; a person with no head is no match;
with nobody else on the pick frame, a shot's best frame showing one person
with a head is taken whatever it scores.

| clip | shots | result | node's time, outside the server |
|---|---|---|---|
| band | 9 | lead taken in 5 and 7, six cutaways left; mean IoU against the hand-picked reference 0.982 | 62 s (49 s before) |
| solo | 7 | singer taken in all; microphone never; mean IoU 0.952, the shortfall being the old reference's microphone | 64 s (56 s before) |
| car | 7 | lead taken in 5 and 7; the two other women left; her shot 1 missed | 40 s in the server |

Inside the server the three reports equal the probes'.

## Limits

Three clips, and the rule was chosen on them: there is no clip it has not
seen. On the car clip the room is small on both sides of the line (0.76
under, 0.86 over, line 0.80), and the lead in shot 1 cannot be told from
another woman by these features at all. SAM 3's trunk says what a thing is,
not who; an identity model is the real answer for faces this alike, and
none is wired here.
