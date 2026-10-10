# Who is who after a cut: two trackers' shot tables read together (2026-10-10)

lane: masked video to video
verdict: one load of one clip, two named subjects, read from the trackers' own shot tables. On that load the likeness the Subject Track uses cannot name a person against a line (a frame of one subject scores over the regain line against the OTHER subject's gallery on 6 of 14 frames) and does say which of the two a person is more like (14 of 14, by as little as 0.023). The shot both trackers left empty was not refused for the reason first reported: one subject fell under an automatic line placed between her own two shots, and what was read as a mislabel was one person returned as three nested detections.

**Read this first.** One load, one clip, two people. Nothing here says the
same holds in a crowd, and the load has no shot with a stranger in it and a
named subject away, which is the case a decision by elimination is weakest
on. The first part needs no model and is exact. The second part is one CPU
run; the tables it reads were made on the card.

## What was asked

A 29-frame shot of a two-subject load was left empty by both
`MiniMaxH3SubjectTrack` nodes: each called its subject absent after the
cut, and both subjects are in the shot. How should a subject be remembered
across shots, when the head and shoulders alone do not carry it?

The load is 447 frames starting at frame 604 of the every-frame 24 fps
copy, three shots: frames 0-405, 406-434 and 435-446 of the load. The two
subjects are called `lead` and `second` here. In the middle shot the lead
is large and in front, with her hat off and her hair thrown about, and the
second subject is small and half hidden behind her. In the last shot the
one person is the second subject.

## What the trackers' own tables say

Read from the two shot tables of the uncorrected load and the two of the
load corrected by hand (which is the known answer).

| shot | tracker | people on its shown frame | best likeness | the line | called | the truth |
|---|---|---|---|---|---|---|
| 406-434 | lead | 2 | 0.667, on the large person in front | 0.80, the floor | absent | that person is the lead |
| 406-434 | second | 4 | 0.799, on "person 3" | 0.851, automatic | absent | the small person behind is the second subject |
| 435-446 | lead | 1 | 0.34 | 0.80 | absent | right: the lead is away |
| 435-446 | second | 1 | 0.904 | 0.851 | taken | right |

Three things in those rows were not what was first reported.

**The automatic line refused her, between two right answers.** The second
subject's line is 0.851 because `subject_track.py::auto_match` puts the
line in the middle of the widest gap between the shots' scores, and her
only two other shots scored 0.799 and 0.904. Both hold her. (The floor,
0.80, would have refused 0.799 as well, by a thousandth.)

**"Person 3" is the same person as "person 4".** On that frame the
detector returned four detections: the lead, and three masks of the second
subject at one place, with the same left edge and top and heights of 515,
269 and 186 pixels, scored 0.52, 0.68 and 0.77 by the detector. Their
areas are within a tenth of each other (0.0244, 0.0232 and 0.0222 of the
frame): the taller ones add a scrap below her. Their three number labels
are drawn on top of each other, so the tile shows one, and the table's
"closest: person 3" is the middle mask of the person a hand correction
took as person 4. A size floor removes none of them.

**What each tracker measures against is one detection.** A likeness after
a cut is taken relative to the other people on the pick frame. For the
second subject's tracker that is one detection, a printed figure on a box
on the counter (52 by 85 pixels); for the lead's it is the second subject.

## The galleries: a line cannot name a person here, a comparison can

No model. Each table carries its subject's gallery, the signatures of
eight frames of the first shot's own track, in the two places a person is
compared (the top third of the mask, and the head). For every gallery
frame: how like the REST of its own gallery it is, and how like the other
subject's, by the node's own gallery rule (`subject_track.gallery_scores`:
per place the best over the gallery, then the lower of the two places).

    <python> bench/who_is_who_across_shots.py galleries --table lead=<shots.json> --table second=<shots.json> --json J

Data: [`2026-10-10_who_is_who_across_shots_galleries.json`](2026-10-10_who_is_who_across_shots_galleries.json);
the tool's `render --json` prints every row.

| by the rule (the lower of the two places) | lead's frames | second subject's frames |
|---|---|---|
| frames with a head, so the rule applies | 6 of 8 | 8 of 8 |
| against its own gallery | 0.906 to 0.936 | 0.921 to 0.970 |
| against the other subject's gallery | 0.844 to 0.895 | 0.826 to 0.896 |
| at or over the regain line (`REGAIN_SAME`, 0.88) against the OTHER's | 3 | 3 |
| own minus other | 0.023 to 0.072 | 0.071 to 0.116 |

So inside one shot, in one light, a frame of either subject is over the
line as the other subject on 6 of 14 frames, and is more like its own
gallery than the other's on 14 of 14. One of those 14 leads is under the
node's own margin (`REGAIN_MARGIN`, 0.03); taken one place at a time, the
smallest lead is 0.014, at the head. The lead's two gallery frames
without a head are the ones where she is entering and leaving the frame;
their top third scores 0.46 against the rest of her own gallery.

What this does not show: anything across the cut. The galleries are all
from the first shot.

## What followed from it

- Detections that are one thing are joined before anyone is numbered
  (`subject_tracks.py::one_each`), with the detections left alone when the
  one that would be kept may be a part.
- The shot table keeps every look of a shot judged after a cut, with every
  person's likeness per place, and the signatures of the people on each
  shot's shown frame, so a shot's people can be compared with another
  tracker's subject (`shot_table.py::LOOKS_ARE`, `SIGNATURES_ARE`).
- The report says when the automatic line stands above its floor and which
  two scores it was put between (`subject_track.py::moved_by_a_gap`).
- Deciding a shot for all subjects together is the next step, and the
  measurement below is what it waits for.

## Across the cut, on the model

Not written yet: the run is `bench/who_is_who_across_shots.py looks`, on
the CPU, and its tables are added here when it ends. Its reading was
written before the run, in the tool's docstring.

## Not done

- A second clip, and a crowd.
- A shot with a stranger in it and a named subject away.
- Sapiens2's features or the body model's shape as a vote: after the step
  above has a number to beat.
