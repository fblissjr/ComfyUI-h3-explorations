# Who is who after a cut: two trackers' shot tables read together (2026-10-10)

lane: masked video to video
verdict: one load of one clip, two named subjects, read from the trackers' own shot tables and one CPU run. Handing the frame's people to all the named subjects at once, by the highest total likeness, gave the known answer on nine looks of nine across the cut, by no margin on the first frame after the cut and by a real one four frames later; asking each subject for its own best person failed, and a gallery read against a line would have put one subject's mask on the other. On that load the likeness the Subject Track uses cannot name a person against a line (a frame of one subject scores over the regain line against the OTHER subject's gallery on 6 of 14 frames) and does say which of the two a person is more like (14 of 14, by as little as 0.023). The shot both trackers left empty was not refused for the reason first reported: one subject fell under an automatic line placed between her own two shots, and what was read as a mislabel was one person returned as three nested detections.

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
- Deciding a shot for all subjects together is built, to ASK and not to
  decide: `subject_tracks.py::hand_out` totals every way of handing a
  shot's people to the subjects over several looks, and
  `bench/who_is_who_across_shots.py corrections` reads the trackers' shot
  tables of one preview and writes one file for the gate, a row per shot,
  with the correction to type filled in. It takes nothing without a person
  (`subject_tracks.py::TOGETHER_LEAD` is unmeasured). The measurement below
  says what form it could not take (each subject's best person) and what
  form held on this load (the sum, over more than one look). It has not
  read a real table yet: the tables of this load were written before a
  person's signatures were kept.

## Across the cut, on the model

One run, on the CPU, SAM 3.1 as ComfyUI ships it, the node's own detector
and signatures (`subject_track.py::_sam_callables`). Five frames of the
middle shot and four of the last; every person on each is compared with
BOTH galleries by the gallery rule, in each place a person is compared.

    <python> bench/who_is_who_across_shots.py looks --cpu --clip <the copy> --first-frame 604 --frames 447 \
             --width 1024 --height 768 --table lead=<shots.json> --table second=<shots.json> \
             --look 406,410,418,426,434,435,439,443,446 --json J

Data: [`2026-10-10_who_is_who_across_shots_looks.json`](2026-10-10_who_is_who_across_shots_looks.json);
`render --json` prints every table.

**Controls.** The detections made here on each table's shown frame are the
table's people to within 2 pixels and 0.035 of the detector's score on
five frames of six. On the sixth, frame 410, this run has three detections
where the card's table has four: the loosest of the three nested masks,
scored 0.52 on the card, is at the detection threshold and is absent on
the CPU. Joined, both are two people. (The card's count after joining is
not measured: a table keeps boxes, not masks.) The top third signed by the
tool is the node's own, to four places, on every person.

**The middle shot, both subjects there** (by the rule, the lower of the two
places; "lead / second" is the person's score against each gallery):

| frame | the large person in front | the small person behind |
|---|---|---|
| 406 | 0.912 / 0.865 | 0.744 / 0.700 |
| 410 | 0.894 / 0.846 | 0.805 / 0.855 |
| 418 | 0.863 / 0.802 | 0.756 / 0.798 |
| 426 | 0.601 / 0.575 (top third 0.838 / 0.771) | 0.708 / 0.712 |
| 434 | 0.948 / 0.891 | no head found (top third 0.545 / 0.511) |

**The last shot, the second subject alone:** 0.846 / 0.939, 0.865 / 0.944,
0.891 / 0.943, 0.887 / 0.936 on frames 435, 439, 443 and 446.

What those rows say:

- **The large, clear person is the best match of BOTH galleries.** By the
  top third, the lead's person scores 0.77 to 0.89 as the second subject;
  the second subject's own person, small and half hidden, scores 0.51 to
  0.86 as herself. A rule that gives each subject its best person hands them both
  the same one.
- **A gallery against a line takes the wrong person.** In the last shot
  the second subject scores 0.891 and 0.887 against the LEAD's gallery,
  over the regain line (0.88), with nobody else on the frame to lead. A
  gallery carried across shots and read against a line would have laid the
  lead's mask on her there. It would also have taken the lead correctly in
  the middle shot (over the line on three looks of five), which is why it
  looks like a fix.
- **Which named subject a person is MORE like is right for the clear
  person and unreliable for the small one.** The large person is more like
  the lead on five looks of five (by 0.026 at the least). The small person
  is more like the second subject on three looks of four that have a head
  (by 0.050, 0.042 and 0.004) and more like the LEAD on the first frame
  after the cut (by 0.044).

**The rule written before the run failed.** The tool's `settle` asks each
subject for its best person and then for a lead over the other subject. On
the middle shot it gave the second subject nobody on four looks of five,
for the reason in the first bullet. By the reading written before the run,
that is not a rule to build.

**All subjects at once, by the sum** (`together`, added to the tool AFTER
the run and so not tested by it): every way of handing the frame's people
to the subjects, one each at most, gets the sum of the scores it uses; the
highest total is the answer and its lead over the next way says how sure.

| frame | the answer | top third | the rule | whole mask | top third and whole mask, summed |
|---|---|---|---|---|---|
| 406 | right | 0.004 | 0.002 | 0.068 | 0.072 |
| 410 | right | 0.098 | 0.098 | 0.066 | 0.164 |
| 418 | right | 0.110 | 0.102 | 0.035 | 0.144 |
| 426 | right | 0.082 | 0.030 | 0.002 | 0.084 |
| 434 | right | 0.023 | no head | 0.032 | 0.055 |
| 435 | right: the lead has nobody | 0.093 | 0.093 | 0.053 | 0.146 |
| 439 | right: the lead has nobody | 0.079 | 0.079 | 0.056 | 0.135 |
| 443 | right: the lead has nobody | 0.052 | 0.052 | 0.053 | 0.105 |
| 446 | right: the lead has nobody | 0.049 | 0.049 | 0.058 | 0.107 |

It gives the known answer on nine looks of nine in every place it can be
made, and on the first frame after the cut by two to four thousandths,
which is no margin. So:

- **One frame is not enough.** Frame 406 is the frame the lead's tracker
  shows for that shot; four frames later the same decision has a lead of
  0.098. A decision for a shot has to add up its looks, following each
  person from look to look, or take the look with the widest lead.
- **The whole mask does not beat the top third, and the two fail on
  different frames** (406 for the top third, 426 for the whole mask).
  Summed, the smallest lead is 0.055, which as a mean of two scores is
  0.028 against 0.004 and 0.002 for either alone. By the reading written
  before the run the whole mask earns a place only if it widens the
  smallest lead: alone it does not; beside the top third it does. One
  shot.
- **The last shot is decided by the count, not by a score.** With one
  person and two subjects, the second subject takes her (by 0.05 to 0.09)
  and the lead is left with nobody. Nothing here says what happens when a
  stranger stands in that frame.

## Not done

- A second clip, and a crowd.
- A shot with a stranger in it and a named subject away, which is the
  case `hand_out` is known to get wrong (it hands the stranger to the
  subject) and the one a second clip has to hold before anything is taken
  without a person. Of the owner's cleared test videos, the crowd clip
  (`lotsofpeopledance_0414_0720.mkv`) has other people on every frame (the
  2026-10-07 records count sixteen detections a look), so any cut of it
  after which a followed subject is off screen is this case. Whether a
  stretch of it has such a cut is one run of the tracker's cut finder and
  has not been looked at; no stretch is named here because none was seen.
- The acceptance on this load: the preview queued again so its tables
  carry signatures, then `corrections` on the two tables.
- Sapiens2's features or the body model's shape as a vote: after the step
  above has a number to beat.
