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

## One scale: every person signed from a crop (run the same evening)

The question, from the owner's follow-up: a signature is the trunk's
features under a mask at the FRAME's scale, so a distant person is a
handful of cells; would signing each person from a crop of their box,
resized to one size, make a distant look and a close one comparable? One
CPU run on the same load: the middle shot's five looks and two of the last
shot's, every person also signed from a square crop around their box (1.2
times its longer side, resized to the trunk's own input size), the top
third and the whole mask, with the galleries made the same way on their
own frames.

    <python> bench/who_is_who_across_shots.py looks --cpu --crops ... --look 406,410,418,426,434,439,446 --json J

Data: [`2026-10-10_who_is_who_across_shots_crops.json`](2026-10-10_who_is_who_across_shots_crops.json).
The reading was written in the tool's docstring before the run: worth a
place only if, on the middle shot, the small subject's lead over the other
subject and the lead of the right hand-out are WIDER than at the frame's
scale on the looks where they were nothing, and no wider lead elsewhere is
lost.

**The small subject, her own gallery less the other subject's** (top third;
positive is right):

| frame | at the frame's scale | from a crop |
|---|---|---|
| 406 | -0.044 | -0.003 |
| 410 | +0.050 | +0.025 |
| 418 | +0.049 | +0.007 |
| 426 | +0.015 | +0.005 |
| 434 | -0.034 | -0.002 |

**The lead of the right hand-out over the wrong one:**

| frame | top third, frame | top third, crop | whole, frame | whole, crop | both summed, frame | both summed, crop |
|---|---|---|---|---|---|---|
| 406 | 0.004 | 0.023 | 0.068 | 0.047 | 0.072 | 0.071 |
| 410 | 0.098 | 0.060 | 0.066 | 0.034 | 0.164 | 0.094 |
| 418 | 0.110 | 0.024 | 0.035 | 0.038 | 0.144 | 0.062 |
| 426 | 0.082 | 0.042 | 0.002 | 0.046 | 0.084 | 0.088 |
| 434 | 0.023 | 0.025 | 0.032 | 0.030 | 0.055 | 0.055 |
| 439 | 0.079 | 0.040 | 0.056 | 0.031 | 0.135 | 0.071 |
| 446 | 0.049 | 0.028 | 0.058 | 0.033 | 0.107 | 0.061 |

**By the reading written first: not worth a place.** The crop does what
was hoped on the two looks where the frame's scale had nothing or the
wrong sign: the small subject's gap goes from clearly wrong to about zero,
and the top third's weakest lead rises from 0.004 to 0.023. It pays for
that everywhere else: the leads that were wide are roughly halved, and the
two views summed, which is what the decision uses, have the same weakest
lead (0.055) and a much lower strongest one. Over the middle shot's five
looks the summed lead is 0.519 at the frame's scale and 0.370 from crops.

What the crop does to the scores explains it: it raises EVERY likeness.
By the top third the small subject scores 0.51 to 0.86 against her own
gallery at the frame's scale and 0.73 to 0.94 from a crop; but against
the OTHER subject's gallery she rises just as far (0.73 to 0.91), and the
lead's person rises against the second's gallery too (0.90 on three looks
of five).
At one scale everybody looks more like everybody: the wrong answers stop
being wrong by much, the right answers stop being right by much, and a
line is further from usable than before. The right hand-out is still on
top on seven looks of seven either way.

So a look is NOT changed to a crop on this evidence, and the cast file's
looks stay at the frame's scale. Two things it leaves open: one load of
soft film; and a crop was tried only for both sides at once (the person
and the gallery), not for a small person against a gallery kept at the
frame's scale.

## The joint decision on real tables (the acceptance, the same evening)

The preview of the same load queued again on the server once it served the
tracker of b0079d98, by the tracked job builder: both trackers on the same
447 frames from frame 604 of the copy, the pack's SAM corrections on, the
second tracker told who the first holds, no correction typed. Its two shot
tables are under the output folder as
`Video/mrsheep/fun_kitchen_who_0604/fun_kitchen_who_0604_lead_shots_00001_shots.json`
and `..._second_shots_00001_shots.json`; the file the tool wrote from them is
`data/2026-10-10_fun_kitchen_who_0604_job/who_is_who.json` (untracked, with
the job's captures). Queued and run by the lead session; read here from the
files.

    <python> bench/who_is_who_across_shots.py corrections --table lead=<...lead_shots...json> --table second=<...second_shots...json> --first-frame 604 --out <the file>

**The trackers alone, on these tables.**

| shot | lead's tracker | second's tracker |
|---|---|---|
| 0-405 | picked on frame 60 | picked on frame 4 |
| 406-434 | absent: 0.641 on its best look, 0.52 to 0.55 on the other three, the floor 0.80 | TAKEN at 0.815, the floor 0.80, two people on its tile |
| 435-446 | absent at 0.316, rightly | taken at 0.884 |

So the second subject's tracker, which called her absent in the morning's
preview of this shot, takes her now, by fifteen thousandths. What moved,
read from the tables: the three nested detections are joined, so her tile
has two people (`duplicates_joined`: one on the shown frame, the kept mask
0.948 of the largest); and her two shots' first-look scores are 0.815 and
0.884 where the morning's were 0.799 and 0.904, so they are no longer far
enough apart for the automatic line to be put between them
(`match_moved_by_gap` is null and the line is the floor). `others` left
nobody out on any frame looked at, so it is not the cause. What moved the
scores themselves is not in a table; the lead session confirmed it from
the graphs: the morning's previews of this load wired no SAM corrections
node and this one wires it, so the model's input changed. Three causes,
then, for one changed call: the scores moved with the model's input, the
join made her tile two people, and the line fell back to the floor because
her two shots came out closer together. And she is taken by fifteen
thousandths, the same thin margin the refusal had. Two previews of one
load, one frame apart in nothing but settings, gave opposite answers from
a line; the joint decision's answer for her was the same both times.

**The tool's three rows.**

| shot | status | looks used | people | lead of the top way | lead's tracker: called, the sum says | second's tracker: called, the sum says |
|---|---|---|---|---|---|---|
| 0-405 | unresolved | 0 of 2 | 3 | none | picked person 2; cannot say | picked person 2; cannot say |
| 406-434 | differs | 4 of 4 | 2 | 0.430 | absent; person 1 on its tile, frame 406 | taken person 2; person 2 |
| 435-446 | agrees | 2 of 2 | 1 | 0.293 | absent; nobody | taken person 1; person 1 |

Three rows of three as said beforehand. The middle shot's row asks about
the lead and only the lead, with no answer filled in (`suggests` false:
"which numbered person is lead? 1, 2, or none"), and what the top way
would type is `shot 2: person 1`, the correction the lead session typed by
hand that afternoon from the same tile. The last shot leaves the lead with
nobody. The first shot is unresolved for the reason given beforehand: the
two pick frames are 56 frames apart, one shows the printed figure and the
other the lead, and no look holds all three.

Three things the file shows that the made-up tables did not:

- **A person's total for one subject, read alone, points at the wrong
  person.** In the middle shot the second subject's total over the four
  looks is 6.861 on the lead's person and 6.176 on her own; the lead's is
  7.276 and 6.161. The hand-out is right because it totals a WAY: lead and
  second on their own people 13.452, swapped 13.022. So the `candidates`
  a weak row offers must never be ordered or marked by their own totals;
  they are written in tile order, and a figure worth showing a person
  beside a candidate would be the best total of any way that gives the
  subject that person, which is not built.
- **The lead of 0.430 is four looks added up**, the first frame after the
  cut among them; on the first measurement that frame alone had a lead of
  a few thousandths.
- **The printed figure is in the first shot's row as a person**: seen on
  one look of two, 4,335 pixels of box, handed to nobody only because the
  row is unresolved. Nothing in the tool knows it is a thing.

**Changed after reading that file: a shot every tracker picked agrees.**
The gate read the file the same hour and blocked on the first shot's
`unresolved` row, which would have blocked every job on its pick shot. A
pick is a person's word and is not this tool's to decide, so a shot in
which every tracker picked its subject now reads `agrees`, with a `note`
saying so and whether anything could be compared; it reads `unresolved`
only when two trackers picked one person on one frame. Run again on the
same two tables the first row is `agrees` and the other two are unchanged.

One load, one clip, an answer known in advance, and tables the tool had not
seen. It says the tool reads real tables and repeats the measurement's
answer on the card with the corrections on; it does not make the form
tested on a second clip.

## A cast file: remembering each named subject across shots, loads and sessions (a scope; nothing is built)

The owner's question, the same day: why not look across cuts, how far back
is looked, and can what is tracked be stored so a subject who does not
appear again until much later is still known.

**What is remembered today, read from the code.** A tracker call cannot
cross a cut, so each shot is seeded afresh. Across a cut a person is
compared with ONE look, the pick frame, however long ago: nothing decays,
there is simply nothing else (`subject_track.py::follow`). Inside a shot a
subject the track let go is looked for against up to
`subject_track.py::GALLERY_MOST` looks of that shot's own track. And one
memory is already on disk: a shot table carries the picked shot's gallery,
and a later run can be handed it (`shot_table.py::gallery_from`, the
node's `subject_from`). That hand-over picks its subject by the gallery
AGAINST A LINE (`subject_track.py::clear_best`), the form the measurement
above found unsafe: on a frame that holds only another named subject there
is no next person to lead, and she scored 0.89 against the lead's gallery.
Not seen to happen; it follows from the code and that figure.

**The file.** One per job, beside its captures: every NAMED subject, and
under each the looks of every shot in which they were confirmed.

- **A look** is a person on one frame: the source frame, the load and the
  shot it came from, the box and its share of the frame, the three
  signatures the tables now carry (top third, head, whole mask), and how it
  was confirmed: by the tracker's own rule with its score and line, or by a
  person, with the correction that was typed. Each look has an id.
- **What goes in.** Only a shot a tracker took by its own rule, or one a
  person settled by typing a correction. A row the joint decision is unsure
  of is NOT a confirmation until it has been typed and the preview run
  again, at which point the table itself says so.
- **Who writes it.** A bench tool, with no model: it reads the trackers'
  shot tables of a preview and adds the confirmed shots' shown-frame people
  and the picked shot's gallery. The tables are the raw material and stay
  as they are.
- **Who reads it.** The joint decision
  (`bench/who_is_who_across_shots.py::settle_tables`): each subject's
  gallery becomes its cast looks, from every earlier shot and load, in
  place of one table's gallery, and the shot is handed out for all named
  subjects together (`subject_tracks.py::hand_out`). Never one subject's
  looks against a line. No node changes for this step.
- **A subject away for many shots.** Nothing decays and nothing should:
  the score is already the best over the looks, so the nearest look in
  size and angle decides. What the tables suggest and do not show: scores
  fall with a change of size (the 2026-10-06 record) and this load's two
  hard cases were a much larger and a much smaller person than their
  galleries held, each with something else changed as well. Whether more
  looks WIDEN the lead is not known: the other subject's best over a
  bigger gallery rises too. That is the first thing to measure, on a load
  with several confirmed shots a subject, leaving one shot out at a time.
- **Getting a wrong look out.** A confirmed mistake would be compared with
  every later shot. So every decision the file feeds names the look that
  gave each person's best score, and removing a look, or everything from
  one shot, is one command and leaves a line saying who removed it and
  why. Nothing is ever edited in place.
- **How it reaches a tracker.** Not at first: the trackers keep deciding
  alone and the cast is used where shots are settled together, which
  writes corrections a person types. Later, and only if that holds on a
  second clip: one tracker for all named subjects that reads the cast,
  which is also what would replace the hand-over above.
- **Things, not only people.** The same file can hold a named thing with
  its looks: for a thing there is no head and no top third, so its
  signature is the whole mask alone. Whether the tracker can follow a bowl
  or a hat in a hand by phrase or by box, and whether a whole-mask
  signature tells one held thing from another, has not been tried here.
- **What it does not fix.** A stranger on the frame while a named subject
  is away; two people who look alike; and the first shot, which a person
  still picks.

**Three additions, from the owner's follow-up** (more than one pick frame;
what raises the odds for a person who is distant, close, side-on or moving).
Measured and guessed are kept apart.

- **Looks that differ on purpose, stored up front. Worth building; cheap;
  no model.** Before any shot is settled the file is seeded with a few
  confirmed looks per named person that differ: large and small in the
  frame, facing and turned, and whatever changes their outline in the clip.
  The tool proposes which to confirm, so a person answers a few questions
  at the start and not one a shot: for each shot, the person the joint
  decision hands to a subject is scored against what the file already
  holds of that subject (the best likeness over its looks, top third and
  whole mask), and the shots are ranked by the LOWEST such score among
  those where the hand-out still leads. MEASURED: nothing; the signatures
  to rank by are in the tables since today. A GUESS that it helps, with
  one reason for care: a low score is also what the wrong person looks
  like, so the ranking finds the looks most worth a person's eye and can
  never confirm one itself.
- **Compare at one scale. Not known; one CPU run would say.** A signature
  is the image trunk's features averaged under a mask at the FRAME's
  scale, so a distant person is a handful of feature cells. MEASURED: a
  subject found again at a quarter of her gallery's size scored lower than
  at a like size (the 2026-10-06 record), and on this load the small,
  half-hidden subject is the one a single frame could not place. NOT
  MEASURED: whether taking each person's signature from a crop of their
  box, resized to one size, makes a distant look and a close one
  comparable. It costs one more trunk pass a person a look, only on the
  frames looked at. The run that decides it before anything is built: the
  middle shot's five looks, both people, both galleries, signed from
  crops, read for whether the small subject's own-against-other lead
  widens on the frames where it was nothing.
- **Looks chosen where the picture is steady, not by a fixed stride. Half
  free; untested.** The tracker already has, for every frame, how much it
  differs from the frame before (`subject_track.py::cut_scores`, the
  number it finds cuts with): a frame in a run of small differences can be
  preferred at no cost. That is movement of the whole frame, not sharpness
  of the person; a sharpness figure under the person's box is cheap and is
  not computed today. MEASURED: only that the first frame after the cut
  was the worst look of the shot for one subject, and that on one other
  frame the head view alone collapsed. Whether either was blur was not
  looked at.

**Lines taken from this clip, not from another** (the owner's standing
rule of the same day: a rule that depends on the picture takes its value
from the clip, measured in the no-sampling preview; a constant measured on
another clip is a starting value). The confirmed shots in a cast file are
what would set them, because they are the only labelled data a job has:

| the line today | what it rests on | what the job's own confirmed looks give |
|---|---|---|
| `subject_tracks.py::TOGETHER_LEAD`, unset | nothing yet | the right hand-out's lead on every confirmed shot, look by look: the smallest is this clip's floor |
| `subject_track.py::REGAIN_SAME` and `REGAIN_MARGIN` | one clip, one shot | each subject's looks against their OWN others and against every OTHER subject's: whether any line separates them on this clip, and by how much. On this load there is a hundredth between the lowest own score and the highest other, so the honest per-clip answer can be "no line" |
| `subject_track.py::MATCH_FLOOR` and `PLAIN_FLOOR` | one clip each | the confirmed person's likeness to the pick against everybody else's on the same looks, which the tables now keep |
| `subject_tracks.py::SAME_THING` and `subject_track.py::A_PART_UNDER` | one frame; reasoned | how far inside each other this clip's detections lie, pair by pair, and how much of the largest the best scored is: the tracker has the masks while it runs and could write the two figures per join |

None of this is built, and none of it can come from a clip's first shot:
there is nothing confirmed yet, so the first shot runs on the starting
values and says so.

Order: the joint decision read on real tables first (done); then the
one-scale run, since it could change what a look IS before any are stored
(done: it does not, see "One scale"); then the
file, its seeding, add and remove, and the decision reading it, all
without a model; then the leave-one-shot-out measurement on a second clip;
only then a tracker that reads it.

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
- (Done the same evening: "The joint decision on real tables".)
- Sapiens2's features or the body model's shape as a vote: after the step
  above has a number to beat.
