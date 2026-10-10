# Who is who: recognising a named person again across cuts, shots, loads and sessions

last updated: 2026-10-10 (first written: a design and its premortems; nothing here is built unless a line names a file or a commit)

Written by hand. A design for the Subject Track of
[`masked_v2v.md`](masked_v2v.md) and the tools around it, and the
premortems of that design. It carries no measurement: a figure is cited by
the record that holds it, and the limits the design proposes are in one
table at the end of Part one, each marked as a guess. Where this page and
the code disagree the code is right. The tracker's owner builds from it and
adds a dated line under a section when a part is built, measured or
dropped; nothing is rewritten.

## The first screen: what this is, in plain words

**The problem.** On 2026-10-10 a shot reached a viewer with nobody
replaced in it. Two trackers each decided alone, against one remembered
frame, that their person was not there. Both were
([`known_before_a_render.md`](known_before_a_render.md), the first row of
that day's table).

**What it is.** A memory of the people named in an input video, kept in
one file for that video and shared by every job on it, and one decision a
shot that is made for all of them together.

- A named person is remembered as several **appearances** (a hat and a
  coat; no hat), each with a few confirmed pictures of them: large and
  small, facing and turned.
- It looks back as far as the video goes. Nothing is forgotten with time.
  Clothes and a hat count only inside the scene they were seen in; the face
  and the build count everywhere, once they have been measured.
- Every shot is settled for all the named people, the bystanders it has
  been shown, and "nobody", in one go, over several frames of the shot and
  never the first frames after a cut.
- The same file can name a thing: a bowl, a hat in a hand.

**What it asks of you.** One yes or no for each named person in each scene,
on a picture with that person outlined, before anything renders. After
that, only the shots it is unsure of, each as the same kind of picture. No
frame number is ever typed. The owner may answer, or the lead session
looking at the same picture; the file keeps who did.

**What it will not do.**

- Take a person without being asked, yet. Everything is asked until a
  second clip, one with a bystander on screen and a named person away, has
  been run. That is the case the built decision is known to get wrong.
- Decide by clothes across a change of scene.
- Tell two people apart who look alike in every way it measures. It asks.
- Know who you mean in the first place. The first picture of each named
  person is yours.

## Part one: the design

The words used below: a **subject** is a person or thing the job names
(`lead`, `second`). A **bystander** is a person on screen the job does not
name. A **look** is one person or thing on one source frame, with what was
measured of them there. A **shot** is the frames between two cuts. A
**scene** is a set of shots in which nobody named changes appearance. The
**cast file** is the input video's memory of all of these.

### 1. The ground it stands on

All of it is one load of one clip with two named subjects:
[`../../bench/results/2026-10-10_who_is_who_across_shots.md`](../../bench/results/2026-10-10_who_is_who_across_shots.md).
The design takes six things from that record and builds on nothing else.

- **A line cannot name a person.** A frame of one subject scored over the
  tracker's "same person" line against the other subject's saved looks,
  inside one shot and across a cut (the record, "The galleries" and the
  last shot's rows). Any rule of the form "this subject's looks, over a
  line" is out. That includes today's hand-over to a later load
  (`subject_track.py::clear_best`, reached through `subject_from`).
- **A comparison can.** Which of the named subjects a person is more like
  was right for the clear person on every look and unreliable for the small
  one (the record, "What those rows say").
- **Each subject's best person fails.** A large, clear person is the best
  match of every gallery.
- **All subjects at once held.** Every way of handing the frame's people to
  the subjects is totalled and the highest total wins
  (`subject_tracks.py::hand_out`). It gave the known answer on every look
  it could be made on, by no margin on the first frame after the cut and by
  a real one a few frames later. The form was written after the run was
  seen, so that run does not test it.
- **Two views summed lead by more than either alone**, and the two failed
  on different frames. One shot, in which nobody changed clothes.
- **The detector's count is not steady.** One person came back as nested
  masks, one of them at the detection threshold, present on the card and
  absent on the CPU (the record, "Controls").

Not measured, so the design does not lean on it: a second clip; a crowd; a
bystander on screen while a named subject is away, which `hand_out` is
known to get wrong; whether more saved looks widen a lead or narrow it;
anything across a change of outfit; anything about a thing; and anything on
a sharp picture, since every figure in that record is from soft film.
`subject_tracks.py::TOGETHER_LEAD` is `None` for those reasons and this
design leaves it so.

### 2. The clues, and how far each reaches

Recognition today rests on one clue: the features of SAM 3's image trunk,
averaged under a mask. That trunk is trained to say what a thing is, not
who (`subject_track.py`, "How far the matching can be trusted"). The table
lists that clue's three views and every other source of evidence in the
owner's model store. **Reach** is the column the owner's constraint adds: a
*stable* clue should hold for one person through the whole clip; a
*scene-bound* clue is true only while an appearance lasts.

| clue | reach | tried here for recognition? | cost on the card | what makes it lie |
|---|---|---|---|---|
| trunk features under the top third of the mask | scene-bound in part: see the worked example below | yes: the matching-tops clip and the kitchen load | paid already on every frame looked at | two people with like hair and shoulders; a person of a few feature cells; the first frames after a cut |
| the same three trunk views from a crop of the person at one scale | as the view | being run (the bench tool's `--crops`; its reading is in the tool's docstring, written first). The result goes here as a dated line, whichever way it falls | one more trunk pass a person a look | not known |
| trunk features under the head | steadier against clothes; moves with a hat, hair and the head's angle | yes, the same two clips | paid already | no head found (the record has such looks for both subjects); a head seen from behind (`subject_track.py`, the same section) |
| trunk features under the whole mask | scene-bound: most of it is clothes | one shot, summed with the top third | paid already (`subject_track.py::follow`, `sign_whole`) | a change of outfit; a shared outfit ([`../research/masking/2026-10-04_mrhf.md`](../research/masking/2026-10-04_mrhf.md), the whole-mask scores of neighbours in one top); a mask that takes in a held thing |
| the face alone: features under the part model's face classes | stable while the face shows | no | the part model on every person of a look (today it runs on the tracked subject only) | a profile, hair over the face, a face a few tokens tall; nothing says yet whether a face shows ([`state_signals.md`](state_signals.md), "Whether the face shows") |
| Sapiens2's own features under a mask, as an embedding | not known | no. Nobody here has read whether it tells one person from another | one more forward a person a look, on crops nobody makes today; free for the tracked subject, whose crop the part model already sees | not known. Trained on people is not the same as trained on identity. Its one advantage known in advance: every person is seen at one scale |
| body build: SAM 3D Body's shape and scale values | stable in principle: build does not change with clothes | no. `body_pose.py::predict` keeps both for every body and `pose_table` does not write them | nothing more where a pose pass runs on that person; a pose pass a person a look otherwise | the model sees one image and estimates them afresh on every frame; a half-hidden person gets a completed body, so their build is partly the model's prior; a close-up; loose clothes; and it returns no confidence ([`meta_perception_models.md`](meta_perception_models.md), "There is no confidence to read") |
| height against the others, and who is nearer, in one camera | a relation, stable while both stand | no (an order on one public frame, no record: `state_signals.md`, "Who is in front of whom") | a pose pass on each of them | sitting, bending; a box that is the whole frame; two people at nearly one distance |
| colour of hair and of clothing under the part classes | scene-bound | no. Lightness alone ranked one subject first among a crowd on every look of one stretch ([`../../bench/results/2026-10-07_subject_likeness_in_a_group.md`](../../bench/results/2026-10-07_subject_likeness_in_a_group.md)), which is a reason to try it and not a result about it | the part model on every person of a look, then arithmetic | coloured light; a change of outfit; a uniform; a hat over the hair; a subject who does not differ from the people around in lightness (the same record's untried case) |
| the count: fewer people than names, so somebody is away | a constraint, not a clue about a person | yes: it decided the load's last shot | none | a bystander; a duplicate detection; a thing taken for a person; a person the detector misses |
| place between two looks of one shot (`subject_tracks.py::same_people`) | one shot | in use, to carry a name between looks | none | two people crossing; a newcomer where somebody stood |

A voice is not on the list. Isolating one by a mask needs an environment of
its own and is the prompt its authors call the weaker one
(`meta_perception_models.md`, "SAM-Audio").

**The worked example: the lead, a hat on and off.** In the record's first
table the lead in the middle shot, her hat off and her hair thrown about,
scores under the floor against the one frame she was picked on, and is
called absent. Against eight looks of her from the shot before she is over
the "same person" line on some looks of that shot and under it on others
(the record, "The middle shot, both subjects there"). So the same person,
one cut apart, is not one appearance to the one clue in use. The design's
answer has three parts: more than one look; looks grouped by appearance;
and a rule that when no appearance explains a person, the tool asks and
does not guess from what is left.

**How a clue is allowed to count.** Nothing here is fitted, so no clue gets
a weight. Each has one of three grades, kept beside it in the cast file:

- `shown`: printed on a row a person is asked to answer. Decides nothing.
- `veto`: may turn a take into an ask. Never the reverse.
- `scored`: part of the total a decision is made on.

Today the top third and the whole mask are `scored` (the record's sum), the
head is `shown`, and every clue marked "no" above starts at `shown`. A clue
moves up a grade only by a record. So a new vote can make the tool ask more
before it is measured, and can never make it take more.

**What to measure first, on the kitchen's middle shot**, where both answers
are known. In this order, by what each costs to get, since nothing says
which is best: the one-scale crop (running); the build values, which fall
out of a pose pass once two fields are written; colour under the part
classes; Sapiens2's features, which need a tap that does not exist; the
face alone, which needs "the face shows" first. The reading for all of
them, written before any is run: a clue earns `veto` if it is right on the
lead's looks in that shot, where the trunk fell; it earns `scored` only if,
added to the sum, it widens the smallest lead on the looks where the
trunk's lead was nothing and loses no wider lead elsewhere. One shot earns
a second clip, not a default. A build that separates a small person from a
large one is the easy case and proves nothing about two adults.

### 3. Scenes and appearances

**A cut is not a scene.** The cut finder gives shots. A scene is what lets
a scene-bound clue vote: inside one, a subject's outfit, hat and hair are
taken as fixed, so the whole mask and the colours are evidence. Across two
scenes they are not.

**Who says where a scene ends**, in order of authority:

1. A person's word, kept in the cast: a scene is a name with ranges, and
   may be several ranges (a video that cuts back and forth). What a
   person says here stands. Nobody types a frame number to say it: the
   ranges are the tool's proposal, accepted or split on the sheet of
   section 5.
2. A proposal from the tool. Two signals, both untried: the plate, as the
   trunk's features averaged over the frame outside every person mask on
   each look (one more vector a look; it exists only on frames that are
   looked at); and time, shots close together being likelier one scene.
   The proposal orders the sheet. It decides nothing by itself.

The proposal will be wrong in two ways. It will split one room in two when
the camera turns round, which costs a few more questions. It will join two
scenes filmed in one place in different outfits, which is the dangerous
way, and section 6's evidence total is what catches it: a person under
the impostor level against the appearance that scene holds counts
against being taken, and is asked about.

**An appearance** belongs to one subject and has a name a person gives or
the tool numbers (`lead/1`). A subject's claim on a person in a shot is
always made from one appearance, never the top third of one and the whole
mask of another.

**A new outfit is an expected event, with its own path.** In a scene where
no appearance of a subject explains anybody on screen, the decision is weak
by construction, and it asks. The row shows the stable clues at whatever
grade they have, and the nearest looks of each appearance. A yes makes a
new appearance from that look. Clothes are never the reason for a take
across a scene boundary.

**One confirmation a subject a scene.** Each named subject is confirmed by
a person once in every scene they are in, before any shot of that scene is
decided, even when the figures look sure. This is the design's main bound
on a poisoned look and on a bystander in a familiar outfit. It is also why
a person's attention grows with the number of scenes and not the number of
shots.

### 4. The cast file

One an input video, at `data/<the video's stem>/cast` in the untracked
`data/`, written first by the analysis workflow of section 5 and shared by
every job, load and session on that clip. A log, one event a line,
appended and never edited; the state a tool reads is the log folded from
its first line, and a **revision** is the hash of the log up to a line.

- **The source it is about** is its first event: the every-frame copy's
  name, frame count, size and checksum (the copy and its per-frame
  checksums are step 0 of the masking board's order of operations). Every
  frame number in the file is that copy's.
- **A scene**: name, ranges, who said so.
- **A subject**: name; person or thing; for a thing, what it was seeded
  with and the subject it belongs to, if any (section 11).
- **A look**: an id made from its content; the subject or bystander it is
  of; the appearance; the source frame; the box and the mask itself,
  run-length coded, because a shot table keeps boxes and no masks and a
  look must be able to be signed again; its share of the frame and, where
  a pose row exists, which way it faces; each clue's value with the
  **fingerprint** of what made it; a small fingerprint of the frame; and
  its **basis**:
  - `typed`: a person said so. What was answered, the sheet and tile it
    was answered on with that tile file's checksum, and who: the owner, or
    a session by name.
  - `rule`: a tracker took the shot by its own rule. Its score and line.
  - `decision`: the together-decision took it. Its id and lead.
- **Not this subject.** When a person names the subjects on a tile,
  everybody else outlined on that tile becomes a bystander's look for that
  scene; if only one subject was answered, a look marked as not that
  subject. A confirmation is of a frame, so one answer gives the looks to
  be like and the looks to be unlike.
- **A removal**: a look, a tile, a shot or a whole sheet; who and why. It
  marks stale every decision that leaned on what was removed and every
  `decision` look those decisions added.
- **A decision**: the shot, the revision it read, its verdict, its lead,
  and the id of the look that gave each person's best score.

Three rules about it:

- **Only `typed` looks may lean a take.** `rule` and `decision` looks are
  kept, because they say where a person has been seen, and they rank what
  to show a person next. A mistake of the tool's is then never compared
  with a later shot as if a person had vouched for it.
- **Clues are compared only between like fingerprints.** A signature is
  the trunk's features on the whole frame squashed to a square, averaged
  under the mask (`subject_track.py::_sam_callables`, `signature`). So the
  fingerprint carries the model file, the precision, the canvas and crop
  the frame was fed at, whether the pack's SAM corrections were on, and
  the signing code's version. What is known of each: the card against the
  CPU, and a track's mask against a detection's, did not move the top
  third on the frames one run compared (the tracker owner's reading of
  `whole_gallery` in
  `../../bench/results/2026-10-10_who_is_who_across_shots_looks.json`);
  another canvas or crop is not measured; the corrections change the
  model's input at its first layer (`sam31_corrections.py::attach`), so on
  and off are taken as not comparable until a run says otherwise. Unlike
  fingerprints: the look is signed again from its stored frame and mask,
  or the row asks.
- **One writer at a time.** A lock file beside the log; an append is the
  only write; a tool that cannot take the lock says who holds it and
  stops. Two answers that give one person on one frame to two subjects
  suspend both looks and raise an ask.

**The files, and who reads which** (2026-10-10, agreed with the builder of
the analysis workflow before either side is built):

| file under `data/<the video's stem>/` | written by | read by |
|---|---|---|
| `cast/log.jsonl` | the cast tool only, by appending | the cast tool only |
| `cast/state.json` | the cast tool, as the fold of the log; made again from the log at any time and never edited | everything else: the analysis workflow, the decision, the gate, the job builder |
| `cast/lock` | whoever is appending | the cast tool |
| `cast/sheets/<n>/` | the cast tool: the tiles, `sheet.json` (what each tile asks and the count of them) and the answers as given | a person; the cast tool |
| `analysis/<run>/` | the analysis workflow: the census (below) and later the trackers' tables | the cast tool, the decision |

`state.json` holds the revision, the source it is about, the scenes, the
subjects in the order they were named, their appearances and looks, the
known bystanders, the clues' grades, the floors with the shots they came
from, the audit, and `questions`: how many tiles were asked up front, how
many shots ask now, and how many answers each of the owner and each
session has given. That last field is where the count of section 9 is
read.

**What the cast is not.** It says who is in the source. Which still
replaces a subject and which parts a pass takes are a job's choices and
live with the job: two jobs on one clip may put different people on the
same subject. And it holds no correction: a correction is in the numbers
of one tracker's own tile on one load, so it is the decision's output for
that load (section 8), made from the cast.

**What the census has to write that a shot table does not carry today:**

- for the whole clip, every frame's cut score, not only the cuts;
- for each shot, several steady looks and not one, each with its frame,
  the plate vector and the count of people;
- for each person on each look: the mask itself, since a look that is
  confirmed must be storable and signable again; the feature cells under
  it; and a name that means the same person on every look of the shot;
- once a run, the fingerprint of what made its signatures.

### 5. Seeding it up front

**The casting pass.** Before any load is planned, over the whole input
video: the cuts, and on each shot one look taken where the
picture is steady and not in the first frames after a cut
(`subject_track.py::PROBE_OFFSET`, and the cut score's own frame-to-frame
difference), with more looks in a long shot. On each look: the people by
phrase, duplicates joined, the heads, every clue at its grade, and the
plate vector. No subject is named, no tracker call is made, nothing
samples. It needs a mode the Subject Track does not have: a table of the
people on each look, with no subject (section 8).

**It is part of the analysis that runs first for an input video.** The
owner's rule of 2026-10-10 (`AGENTS.md`, "A rule that depends on the
picture takes its value from the input video, up front") is that the
models are run over the clip before anything renders so that the clip
sets its own rules, and the owner placed it the same day: one analysis
workflow an input video, run before anything else, writing under
`data/<the video's stem>/`, which every later stage refuses to run
without. It is being scoped with the tracked job builder (the masking
board, `build-multi-person-graph-in-the-generator`). The casting pass is
a part of that workflow and the sheet is its one moment where a person
answers. While the models are loaded the pass also writes down, for the
whole clip, what the tracker, the gate and the job builder each work out
separately today or do not have:

- the cuts and every cut score, with the ones just under the threshold;
- each person's size on each look, so a load can be planned to start
  where a subject is large and the gate's size rules have their figures
  before a tracker runs;
- how many people each look holds, which detections sit close to the
  detection threshold, and what was joined as a duplicate;
- the scene groups it proposes, and the plate vector behind them;
- how alike different people on one frame score, and how alike one
  person's looks are once a sheet is answered: the clip's own two
  distributions, from which every floor in section 6 and most rows of
  the table at the end of this part are set.

**The sheet.** One a scene. A tile is a frame with one person outlined and
a question a glance answers.

- The first tile of a video is the only one that needs more than a yes or a
  no: every person on it is numbered, as the tracker's tiles are, and a
  person says which number each named subject is (`lead: 2`,
  `second: 5`). This replaces the pick rule for a video with more than
  one subject.
- Every later tile shows one person outlined in a subject's colour and
  asks: is this the lead, yes or no.
- Where the tool has no answer worth suggesting, the tile numbers its
  candidates and asks which one, or none (section 6, "An ask row").

**Which few tiles.** Inside a proposed scene the scene-bound clues mean
something, so the people of its looks are grouped by them, and for each
group the sheet shows looks that differ on purpose: the largest, the
smallest, the most turned where a pose row says so, and the one least like
the rest of its group. Up to `CAST_SEED_LOOKS` a group, the first of them
being the confirmation section 3 requires. The grouping is only for
choosing what to show. A low likeness is also what the wrong person looks
like, so nothing on a sheet is confirmed by the tool.

**After the first sheet** the ranking the record scopes picks what to show
next: the shots where the together-decision still leads and the person it
hands to a subject is least like anything the file holds of them.

### 6. The decision for a shot

Made for one shot, for every named subject, every bystander the cast knows
in that scene, and "nobody", together.

**The looks.** The shot's looks from the casting pass and from the
trackers' tables, never one inside `PROBE_OFFSET` of a cut, at least
`DECISION_LOOKS_LEAST` of them, or the row asks. A shot a tracker takes on
its first look has one look in its table today, so the casting pass's
extra looks are what make most shots decidable at all.

**Names between looks.** By place, as today, with the two checks the
tracker's owner proposed, neither built and both arithmetic on what a
table already holds: a name whose signature on a look is more like the
*other* name's previous look than its own has probably swapped; and two
names whose boxes overlapped on a look or between two looks
(`NAMES_BOXES_MET`) mark where place is not to be trusted. Either one, or a
change in who is on screen, splits the shot there and the decision is made
on each part. Parts that disagree about who is who ask, and say the names
may have swapped. Carrying everybody through the shot with the tracker
itself is not the answer: it moves a mask onto a neighbour with no empty
frame between
([`../../bench/results/2026-10-07_subject_regain_looks.md`](../../bench/results/2026-10-07_subject_regain_looks.md)).

**The slots.** Each person goes to one subject, one known bystander, or
"unknown". Each subject takes one person or nobody.

**The score** of a subject on a person: per appearance of that subject in
the scene, per `scored` clue, the best likeness over the appearance's
`typed` looks nearest to the person in size, at most `CAST_LOOKS_COMPARED`
of them so that a subject with more looks has no more chances; the clues
summed; the best appearance kept. Open: one odd look in a gallery is a real
thing (the record, the lead's two frames entering and leaving the frame),
but nothing measured says the nearest in size is the right neighbour, and
the one-scale run may make size stop mattering.

**Two totals, because a sum of likenesses has no zero.**

- *The built total* (`subject_tracks.py::hand_out`): the scores summed.
  Every likeness is positive, so handing a subject anybody adds to it and
  a subject is left with nobody only by the count. It says who is most
  like whom and nothing about whether anybody is. Here it orders the
  candidates on a tile.
- *The evidence total*: each person's score for a taker, less that
  taker's **impostor level** in the scene, which is the highest score any
  look known not to be that taker (another subject's, a bystander's)
  reaches against the same appearance. Over the level counts for a way,
  under it counts against, and "nobody" and "unknown" are worth zero. It
  is summed over the shot's looks and not judged look by look, so one poor
  look does not refuse a person. This is what makes the two slots mean
  something. It needs the levels from the clip, so it is step 5 of section
  12 and means nothing before that.

The level is a line, and the record says no line names a person. It is
not used to name. It says where zero is, from this clip's own frames.
Whether it should be taken at a like size is open: nothing measured says
the level falls at a like size.

**What the evidence total says on the very shot this design exists for.**
By the record's own rows, on the kitchen's middle shot each subject's own
person scores under that subject's impostor level on most looks or on all
of them (the tracker owner's reading of "The galleries" against "The
middle shot, both subjects there"). So the evidence total leaves both
subjects with nobody, the built total says who each is most like, and the
row asks. The same will happen on every shot where a subject's look has
changed across a cut. That is not a second failure beside section 3's new
appearance: it is the same event, seen in the arithmetic. No appearance
explains the person, so a person is asked, and the answer is a new look or
a new appearance that later shots are measured against. It is safe, and
it is most of the count of questions (section 9).

**A take needs all of these, or the row asks.**

1. *The evidence leads.* The top way of the evidence total leads the next
   way by the clip's floor.
2. *The two totals agree.* It hands each subject the person the built
   total does, or nobody where the count or the known bystanders account
   for everyone on screen. Where the evidence says nobody and the built
   total hands that subject a free person, the row asks whether this is
   the subject in a new appearance.
3. *Look by look.* The share of the looks whose own top way is the summed
   top way is at least `DECISION_LOOKS_AGREE`, and no clue graded `veto`
   disagrees.

*2026-10-10, a correction the same day, from the tracker owner's read of
this page. As first written this section had "nobody" as a choice under
the built total, where it can never win, and a separate second test that
refused, look by look, any person under the impostor level "however the
totals fall". The two are now one total with a zero, summed over looks;
and the page says in advance, here and in section 9, that the measured
shot asks under it.*

And never a take when: a person was left out as seldom seen; the count
alone decided and the scene has any bystander or any detection close to
the detection threshold; the subject is not yet confirmed in this scene;
the person is under `SIGN_CELLS_LEAST` feature cells; or the cast's audit
has a look in doubt (Part two, 3).

**Where the floors come from, and why not from a constant.** From the
clip's own typed shots, two ways: leave one shot out and decide it from the
rest, which gives the leads right answers have; and a simulated absence,
the subject's own person removed from a typed look and the look decided
again, which gives the lead and the score a wrong take would have had. The
floors sit above the second set, are written into the cast state with the
shots they came from, and are not used in a scene with fewer than
`CAST_SHOTS_TO_CALIBRATE` typed shots a subject. This is the owner's rule
applied (`AGENTS.md`, the same paragraph), and the reason is the picture
as much as the people. Every figure on record is from soft, grainy film;
the second test clip is sharp, modern video. A margin measured on one says
nothing about the other, in either direction, and a floor measured on a
clip's own frames has that clip's picture in it. The tool says so when a
clip's own figures disagree with a starting value. Two limits, said here so nobody
reads the floors as more than they are: a simulated absence only ever
stands in somebody already in the clip's frames, so it says how wrong the
tool can be among people it has seen and nothing about a newcomer; and
whatever a clip's own floors say, a take stays off until a second clip with
a bystander on screen and a subject away has been run (the condition
beside `TOGETHER_LEAD`).

**An ask row** is what is built today (the correction, in that tracker's
own tile numbers, since b0079d98) shown as a tile, with four things added:
which test failed; the stable clues at any grade; the ids of the nearest
cast looks; and what the answer is worth, as the shot's frames and the
subject's size. Where the evidence total's lead is at or over
`ASK_SUGGEST_LEAD` the tile outlines the suggested person and asks yes or
no. Under it the tile numbers the candidates, in the built total's order,
and suggests nobody, because a yes to a weak suggestion is a guess with a
person's name on it. Until a clip has a floor for the lead, every row is
under it; that is intended.

**Stop or guess: one switch, `unsure`, in the job file; the default is
stop.** The owner has not chosen. With `stop` an ask blocks the gate until
it is answered, as an absence with people on screen does now
(`bench/capture_masked_run.py::flag_shots`). With `guess` the top way of
the built total is taken; the row is marked guessed in the decision file,
in the gate's flags (listed, not blocking) and in the delivery record; and
a guessed shot adds only `decision` looks, which lean nothing. Two rows
stop under either setting: a subject's first shot in a scene, and a row
the tool marks as its own fault.

### 7. The hard cases

- **A subject away for a long time.** Nothing in the file decays. A person
  back in a scene they were confirmed in is decided as any shot. Back in a
  new scene, they are asked about once (section 3).
- **Back, and unlike every stored look.** The evidence total leaves
  them with nobody, the built total still says who they are most like,
  and the row asks. The answer is a new look and, if the person says so,
  a new appearance.
- **Bystanders on screen and a named subject away.** The bystanders the
  cast knows take their own people. One it does not know is "unknown" if
  under every subject's impostor level. One over it is the case this
  design does not solve: Part two, 1.
- **More people than names.** Today this always asks
  (`subject_tracks.py::sure`). Here the known bystanders are matched first
  and only people still unmatched, over a subject's impostor level, force
  an ask.
- **Identical clothes.** The cast's audit compares every subject's looks
  with every other's and every bystander's when a sheet is confirmed. Two
  that are closer than the clip's own floors are a look-alike pair: the
  whole mask is dropped from the score between them, and any shot that
  could hold both asks.
- **One person wholly in front of another.** The hidden one is not
  detected and takes nobody, which is right: there is nothing to replace.
  When they come out, who is on screen has changed and the decision is
  made again from there. Half hidden is the small-subject case (Part two,
  5).
- **A cut the finder misses.** No decision is made at it, and the tracker
  carries a mask into the next shot. It has happened: a jump cut inside a
  cutaway and a cut between two angles of one room
  (`../research/masking/2026-10-04_mrhf.md`;
  [`../../bench/results/2026-10-06_subject_track_defaults.md`](../../bench/results/2026-10-06_subject_track_defaults.md)).
  Signals that exist at preview time: a cut score between
  `subject_track.py::CUT_FLOOR` and the threshold in use; a jump in the
  number of people between two looks; the plate vector changing; a name's
  signature breaking from its own previous look. Any of them splits the
  shot for this decision.
- **A cut the finder invents.** Two decisions where one would do, and a
  few looks lost to the offset. If every person's box continues across
  the frame, the row says the cut may not be one.

### 8. Across loads and sessions

The cast is in source frame numbers and belongs to the input video, and
no job starts without it (section 5), so a second load, a second job and
a second session read the same file. That is the whole of what
`subject_from` was for. Two stages.

**Stage one changes no tracker.** A bench tool reads the trackers' shot
tables of a preview and the cast, decides every shot (section 6) and
writes the corrections each tracker already accepts, one for every shot of
the load, `none` included. A fully corrected load does not depend on the
tracker's own pick (`subject_track.py::follow` applies corrections even
when nothing was picked), so `subject_from` is not wired. "Take" then
means the job builder writes the correction, and "ask" means a person
answers a tile and the tool writes it. Two things this stage needs that do
not exist: the casting pass's table of people with no subject, and a seed
a tool can write for a shot without reading a tile: the source frame and
the mask or box of the subject there, or nobody. Today a correction is two
numbers read off one tracker's tile, it applies on the frame that tile
shows, and the tracker's own pass chooses that frame; so a subject who is
not on it cannot be named in that shot, and a tool that has not seen that
load's tiles cannot write one at all. Typed numbers stay as they are for a
person.

**Stage two is one tracker for every name**, reading the cast and writing
one mask a subject, with the decision made where the detections are. It
replaces `subject_from` and the `others` wiring. It is built only if stage
one holds on the second clip.

### 9. What it costs

**In a person's attention, which is the scarce thing.** Counted in
questions, each a glance at a tile:

- *Once an input video:* one tile to say which numbered person each
  subject is.
- *Once a scene:* a yes or no for each named subject who is in it, and for
  each further tile up to `CAST_SEED_LOOKS`.
- *After that:* one tile for each shot that asks.

For the job of 2026-10-10, three named people in three scenes by the lead
session's count: one naming tile, then at most nine required yes-or-no
answers, fewer since not everybody is in every scene. For the one load
that has been through a measurement, the kitchen's three shots: the middle
shot asks, on one tile showing both subjects, and the trackers alone were
right about the other two (the record's first table). With the tool as
built today the first of those two will likely read as unresolved and not
as agreeing, since the two trackers' pick frames are apart and show
different sets of people, so no look holds everybody; the casting pass's
own looks remove that.

**What to expect, said in advance.** A subject whose look has changed
since their last typed look is asked about, once for each such change
(section 6). So the questions are one for each named subject in each
scene, plus one for each time a subject turns up looking unlike what the
cast holds of them in that scene: a hat off, hair thrown about, far
smaller. On soft film with two looks a subject in a scene, that second
number may be as large as the first. What the rest of
that job would ask is not known and is the first figure step 4 of section
12 reports. While takes are off, every shot where the decision differs
from the trackers is a tile; that is the cost of the present state and
the reason step 6 matters.

`ASKS_BEFORE_RESEED` is a budget: over it the tool stops listing shots and
shows the few more looks whose confirmation would settle the most of them.
A job whose count the owner finds too high has failed this design whatever
its answers (section 10, the last result).

**On the card.** The casting pass, once an input video: on each look a
detection for the phrase, one for heads and one trunk pass. Signing a stored look
again when fingerprints differ. A pose or part pass on every person of a
look, only for a clue that has earned more than `shown`, or while it is
being measured. Everything else reads files. None of this is timed; the
first casting pass is, in a record under `bench/results/` that says the
cache state.

### 10. How it is tested

The clips are the two the owner named for this on 2026-10-10, in their
order of priority, each described by what the test needs from it. "The
picture" is what its quality does to the trunk's features: untried on
both, since nothing has been compared across them.

| clip | what the test needs from it | what the records already show there | the picture |
|---|---|---|---|
| `fun` (captures under `data/2026-10-10_fun_*`) | two then three named people; a subject whose appearance changes between scenes; a small, half-hidden subject; a shot decided by the count; a thing the detector takes for a person | everything in section 1; it has no bystander on screen with a named subject away (the record says so) | soft, grainy film. Every figure on record is from it |
| `thinkaboutthings_compressed` | people in matching tops; many cuts; cutaways to other people while the group is away; a shot seen from behind; a cut the finder misses; coloured light | the top third with the pick frame's others subtracted held the front person and took a wrong person in one cutaway at first; whole-mask scores of neighbours in one top are as high as the same person's in another shot (`../research/masking/2026-10-04_mrhf.md`; [`../../bench/results/2026-10-04_subject_track_three_clips.md`](../../bench/results/2026-10-04_subject_track_three_clips.md)) | sharp, modern |

The second clip that turns takes on must not be chosen for being easy, so
it is the matching-tops clip and the test on it is set up to be hard.

**The named subjects there are two people at the back of the group, not
the person at the front.** They share a top and a build with the people
beside them, are smaller in the frame, are partly hidden and come and go
across many cuts. The bandmates play the bystander who looks like a
subject. The test is run three ways on the same stretch, because "all
named people together" behaves differently when the people not named look
like the ones who are:

- **Two named, the other four confirmed as known bystanders** on the
  sheet. This is the design as written.
- **Two named, the others unknown** to the cast. This is what the
  "unknown" slot and the impostor level are for.
- **All six named.** The most the together-decision can be given, as the
  upper bound.

The truth for every shot is a person's, answered once on tiles for the
whole stretch; that is a cost of the test and is counted. The cutaways are
the bystanders-with-subjects-away case that today's job lacks.

**The results that would falsify a part of the design**, written before
any is run:

1. *The take rule.* On any shot of the second clip, in the first way of
   running it, a wrong answer passes every test for a take. Then the tests or
   the floors change before anything is taken anywhere.
2. *Known bystanders.* The first way makes no fewer wrong top answers than
   the second. Then confirming bystanders buys nothing and comes out of
   the sheet.
3. *The impostor level.* In the second way a bandmate is handed to a named
   subject who is away and passes the impostor level. Then the level is
   not a refusal worth having and every shot with an unknown person asks.
4. *More looks.* With the cast's looks from several shots, the smallest
   lead of the right answers is no wider than with one shot's gallery.
   Then the cast stays as the memory of what was confirmed and stops being
   a reason to expect fewer asks.
5. *The simulated absence.* On the cutaways, real wrong answers have wider
   leads than the simulated ones. Then a clip's own floors are not safe and
   takes stay off.
6. *Sharp against soft.* The procedure holds on the sharp clip and, run
   the same way on the soft one, gives wrong takes or asks on nearly every
   shot. Then floors from a clip's own frames do not absorb the picture,
   and that claim of section 6 is withdrawn.
7. *The seeding choice.* The tiles the tool chooses settle no more shots
   than the same number taken at random. Then they are taken by time.
8. *The scene proposal.* It disagrees with a person about a large share of
   the shots on both clips. Then scenes are a person's word only.
9. *The count of questions.* On a real job it is more than the owner will
   answer. Then the design has failed its first requirement.

**A gap in the tests.** Neither clip has far more people than names with
the detector at its cap, which is premortem 5's worst case and the one
where the count of questions could be largest. Records exist of the
tracker on such a picture
([`../../bench/results/2026-10-07_subject_track_under_nudge.md`](../../bench/results/2026-10-07_subject_track_under_nudge.md)),
and none of this design. Which clip that test may be run on is the
owner's word; none is proposed here.

### 11. Things, as first-class entries

The owner's rule for this lane is that every step works for parts,
clothing, objects and parts of objects, and does not depend on a fine
label existing. In the cast a thing is a subject like any other: named,
seeded, with looks, appearances and scenes, confirmed on the same sheet,
removed the same way, and decided in the same file.

**What carries over from people:** the file and its three rules; the
sheet and its yes or no; one confirmation a scene; the grades of a clue;
the fingerprint; "unknown" and "nobody"; the stop or guess switch.

**What does not:**

- No head and no top third. The only trunk view is the whole mask.
- No build, no height, no face.
- No "more like whom" worth the name between two things of one kind: two
  bowls are one bowl to a model that says what a thing is. A thing is
  known first by a relation and only then by its look.
- No count rule: things do not come one to a name.

**So a thing is decided after the people, in one of three ways:**

- **It belongs to a subject** (a hat in a hand, a cup). It is looked for
  only near that subject. One candidate there is theirs, with the
  subject's own verdict. Several alike ask.
- **It stands in the scene** (a bowl on a counter). Inside one scene it is
  known by its place on the plate and its whole-mask look. In a new scene
  it is asked about, as a person is.
- **It is a part** of a subject or of a thing (a sleeve, a hand, a
  handle): a region riding on its owner's track, cut by the part model's
  classes where they exist and by a typed box where they do not. Its
  identity is its owner's and it needs no entry beyond the owner's.

**No label is needed to seed one.** A phrase, a box typed off a numbered
tile, or the part node's `held` mask, which names nothing
(`sapiens2_parts.py`, "What the subject holds").

**What "within X of a subject" should mean.** X in the subject's own
units, never pixels: the same hand is a few pixels in a wide shot and
hundreds in a close-up. Three meanings, the second as the default:

1. *Held:* inside the subject's own track and near the lips or a hand.
   Built, with its reach in pixels of the source frame
   (`sapiens2_parts.py::HELD_NEAR`, reasoned). Proposed: the reach as a
   share of the subject's box height.
2. *Touching:* a thing detected by itself whose mask, grown by
   `THING_NEAR` of the subject's head height, meets the subject's mask on
   most looks of the shot.
3. *In reach:* the thing's centre within an arm's length of a wrist, in
   the pose table's camera space. It needs a depth for the thing, which
   the body model does not give (`state_signals.md`, "objects are another
   model").

What is known: a held thing was found by a phrase for the first seconds of
one window and by the `held` mask on parts of it
([`../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`](../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md));
a phrase finds a prop and then loses it ([`sam3_prompting.md`](sam3_prompting.md));
and nothing follows a bowl or a dish unless a session adds a tracker by
hand. Not known: whether a whole-mask signature tells one held thing from
another, and everything else in this section.

### 12. The order to build it in

Each step waits on the one before it.

1. The two runs already under way: the one-scale crop, and the built
   together-decision on real tables.
2. More kept at preview, all small: the plate vector a look, the build
   values in the pose table, every person's clues on every look.
3. The cast file with no model: the log, the fold, the audit, remove, the
   sheet, and reading an answer.
4. The casting pass on the job of 2026-10-10. Its first sheet, its time,
   and the count of questions for the whole job.
5. The decision reading the cast, asking only. Accepted when, on the
   kitchen load, it asks on the middle shot with the right people shown
   and leaves the lead with nobody on the last.
6. The matching-tops clip, three ways. Section 10's results are read here.
7. The untried clues, one record each, each moving a grade or staying.
8. Takes, a job at a time, on the owner's word.
9. One tracker for every name; `subject_from` retired. Things, starting
   with one that belongs to a subject.

### The limits this design proposes

This table is where these values are held until code holds them, and the
only place on this page a number of the design's appears. By the owner's
rule (`AGENTS.md`, "A rule that depends on the picture takes its value
from the input video, up front") each row is one of two things: set from
the clip by the analysis of section 5, with no value here at all; or a
starting value, every one of them a guess, with what the clip has to show
for it to be replaced. When a step is built its row moves into the code as
a constant with its provenance beside it, and leaves this table.

| name | for | starting value | what the clip sets it from |
|---|---|---|---|
| the lead's floor (today `subject_tracks.py::TOGETHER_LEAD`, `None`) | the lead a take needs | none: from the clip | above the widest lead a simulated wrong answer reaches on the clip's typed shots |
| the impostor level | where zero is in the evidence total | none: from the clip | per subject and scene, the highest score a look known not to be them reaches |
| `ASK_SUGGEST_LEAD` | the lead under which a tile suggests nobody | none: from the clip | the clip's own floor for the lead |
| `SIGN_CELLS_LEAST` | the feature cells under which a signature is not scored | none: from the clip | the size under which a typed look's lead over the other subjects is inside the spread of one person's own looks; the one-scale run gives its first reading |
| `CAST_LOOKS_COMPARED` | looks of one appearance a person is compared with, nearest in size | `4` | leaving one typed look out at a time: the number past which the smallest right lead stops widening |
| `DECISION_LOOKS_LEAST` | looks a shot needs before it may be taken | `3` | on the clip's typed shots: the number of looks after which the summed answer stops changing |
| `DECISION_LOOKS_AGREE` | share of looks whose own answer is the summed answer | `0.8` | the share the clip's typed shots show when left out and decided from the rest |
| `NAMES_BOXES_MET` | overlap of two names' boxes at which place is doubted | `0.3`, the value `subject_tracks.py::AGREE_AT` uses to call two boxes one person, itself reasoned | the overlaps of different people's boxes on one frame, over the clip's looks |
| `THING_NEAR` | reach of "touching", in the subject's head heights | `0.5` | the gap between a confirmed thing's mask and its subject's, over the typed looks |
| `CAST_SHOTS_TO_CALIBRATE` | typed shots a subject a scene before a clip's own floors are used | `3` | not the picture: how many it takes for the floors to stop moving as shots are added |
| `CAST_SEED_LOOKS` | tiles shown for a group of alike people on a scene's sheet | `4` | not the picture: a person's attention. The job file's to set |
| `ASKS_BEFORE_RESEED` | shots asking in one job before the tool shows more looks to confirm in their place | `12` | not the picture: a person's attention. The job file's to set |

## Part two: the premortems

Each is written as if it were three weeks from now and that part had
failed on a real job. **Signal** is what would show it before a render,
from files that exist after a preview: the shot tables, the casting table,
the cast log and its audit, the decision file, the pose table, the class
maps. **Left** is what the design does not cover. The order is likelihood
times cost, each scored one to three. Both scores are guesses, printed so
they can be argued with.

| # | what failed | likely | cost | the signal before a render | the mitigation, in a line |
|---|---|---|---|---|---|
| 1 | a bystander is taken as a named subject who is away | 3 | 3 | the taken person's score beside the subject's impostor level; the scene has bystanders in the casting table; the count alone decided | "unknown" and "nobody" as slots worth zero; the impostor level; no take on the count in a scene with bystanders |
| 2 | the questions are answered carelessly because there are too many | 3 | 3 | the count of shots asking; answers the rest of the cast contradicts | one question a scene, not a shot; a budget; no suggestion on a weak row; a second ask when the cast disagrees |
| 3 | a wrong look is confirmed and every later shot leans on it | 2 | 3 | the audit: a typed look more like another subject's looks than its own appearance | only typed looks lean; removal marks what leaned; one confirmation a scene bounds the spread |
| 4 | a new outfit; and a bystander in the new scene wearing the old one | 2 | 3 | a claim made from an appearance last seen in another scene | scene-bound clues stop at a scene's edge; the first shot in a scene always asks |
| 5 | a subject so small the signature is noise | 3 | 2 | the mask's feature cells and box height on every look | under `SIGN_CELLS_LEAST` the look is not scored and the row asks |
| 6 | two people swap names inside a shot | 2 | 3 | a name more like the other name's previous look; two boxes that met | the shot is split there and decided on each part |
| 7 | the lead's floor, set from one clip | 2 | 3 | the calibration's own sample, printed with the floor | floors from the clip's own shots; takes off until the second clip |
| 8 | soft film and sharp video do not behave alike | 2 | 3 | each clip's floors and leave-one-out leads, side by side | nothing carried between clips; result 6 of section 10 |
| 9 | the detector's duplicate masks change the count | 2 | 2 | detections near the detection threshold; a count that differs between looks | people seen on most looks; no take on the count beside such a detection |
| 10 | scenes are grouped wrong | 2 | 2 | a subject under the floor against the scene's own appearance | the proposal decides nothing; a person's word; the evidence total |
| 11 | a vote that lies is given weight | 2 | 2 | the clue's grade and the record behind it, in the cast | grades; a clue rises only by a record, and starts where it can only add asks |
| 12 | stored signatures are no longer comparable | 2 | 2 | unlike fingerprints between a look and a table | compared only like with like; signed again, or asked |
| 13 | a session answers for the owner and is wrong | 2 | 2 | the share of typed looks answered by a session | the file keeps who; a page of session-answered tiles for the owner's glance |
| 14 | two people who look alike in every clue | 1 | 3 | the audit's look-alike pairs when a sheet is confirmed | every shot that could hold both asks |
| 15 | a cut the finder misses | 1 | 3 | cut scores just under the threshold; the plate vector changing inside a shot | the shot is split for the decision |
| 16 | the cast is stale against a re-cut source | 1 | 3 | the source's checksum; a look's frame fingerprint | refused on a mismatch, before anything is read |
| 17 | two sessions write the file at once | 2 | 1 | the lock's holder; two answers for one person on one frame | one writer; append only; a conflict suspends both |
| 18 | the guess switch is left on | 1 | 3 | `unsure` printed at the top of every decision file and gate report | default stop; guessed rows listed in the delivery record |
| 19 | a thing is handed to the wrong subject | 2 | 1 | more than one candidate near a subject | several alike ask; a thing takes its subject's verdict |

### 1. A bystander is taken as a named subject who is away

**What happened.** A cutaway showed two people, neither of them named. The
job named two. Two people, two names: the totals handed one to each, the
lead was wide because the two bystanders did not look alike, and both were
replaced. `hand_out` leaves a subject with nobody only when there are
fewer people than subjects, and `sure` asks only when there are more
(`subject_tracks.py`). Equal numbers pass.

**Signal.** In the decision file: the taken person's score against the
subject's impostor level for that scene. In the casting table: whether
that scene has any person no sheet has named. In the row: that the count
decided.

**Mitigation.** Every subject may take "nobody" whatever the count; every
person may be "unknown", each worth zero; a person under the impostor
level counts against any way that takes them; a subject's first shot in a
scene always asks; and takes are off until this exact case has been run
on a second clip.

**Left.** A person never seen before, on screen without the subject, who
scores over the impostor level against a subject already confirmed in
that scene. With takes on, that is a wrong take with every test passed. The impostor level comes from people the job has seen, and says
nothing about one it has not.

### 2. The questions are answered carelessly because there are too many

**What happened.** A job with many scenes and takes still off asked about
dozens of shots. By the third sheet every answer was a yes to whatever the
tile suggested. One of them was wrong, and it was a typed look.

**Signal.** The count of shots asking, against `ASKS_BEFORE_RESEED`. And
the one thing the files can say about an answer: after it is added, the
rest of the cast is asked about that look, and an answer the others
contradict by more than the clip's floor is shown once more, saying why.

**Mitigation.** Questions grow with scenes, not shots; over the budget the
tool offers a few more looks to confirm in place of the list; a tile whose
lead is under `ASK_SUGGEST_LEAD` suggests nobody; there is no "yes to all"
for rows under the floors; tiles are ordered by what the answer is worth.

**Left.** A person who says yes without looking. A typed look is trusted
because a person vouched for it, and nothing in a file can tell a glance
from none. This is why the count of questions is itself a result in
section 10.

### 3. A wrong look is confirmed and every later shot leans on it

**What happened.** On a scene's sheet a yes was given to the wrong person.
Every shot of that scene was then compared with a look of somebody else,
decided confidently, and typed looks from those decisions made it worse.

**Signal.** The audit, run whenever the file changes and read by the
gate: each typed look left out in turn and compared with its own
appearance and with everybody else's. A look more like another subject's
or a bystander's is in doubt, and so is an appearance whose looks fall in
two groups. No take leans on a look in doubt.

**Mitigation.** Only typed looks lean a take, so the tool's own mistakes
do not compound. Every decision names the look that gave each best score,
so a removal marks exactly what leaned on it. One confirmation a scene
keeps a wrong look inside its scene for the scene-bound clues.

**Left.** A wrong look that is consistent with itself: the first tile of
a job answered for the wrong person, and everything after it agreeing.
The file cannot know who was meant.

### 4. A new outfit, and a bystander wearing the old one

**What happened.** The lead changed clothes between scenes. In the new
scene somebody else wore something like her first outfit. The whole mask
and the colours said "the lead, as first seen", and that person was taken.

**Signal.** A claim made from an appearance whose looks are all in another
scene. The decision file has both facts.

**Mitigation.** A scene-bound clue does not vote across a scene boundary;
the first shot of each subject in a scene asks whatever the figures say;
the tile shows the stable clues and the nearest look of every appearance.

**Left.** The same thing inside one scene: a change of outfit that no
scene boundary marks, with a look-alike of the old one in shot. And the
whole mask has never been tried across a change of outfit at all (the
record, section 1).

### 5. A subject so small the signature is noise

**What happened.** This is the fault of 2026-10-10, again: a subject far
from the camera and half hidden, a mask of a few feature cells, and a
likeness that names nobody (the record's rows for the small person behind).

**Signal.** The feature cells under the mask and the box height on every
look, both from the table. Its neighbours in the gate already read size
(`bench/capture_masked_run.py::flag_face_size` and the small-subject
rule).

**Mitigation.** Under `SIGN_CELLS_LEAST` a look's signature is not scored.
The row is decided by who else was accounted for, or it asks. If the
one-scale run shows a crop helps, a small person is signed from a crop.

**Left.** A small subject among small bystanders, in a wide shot of a
crowd. Every such shot asks, and a person may not be able to tell either.
The lane also renders such a subject poorly whoever they are.

### 6. Two people swap names inside a shot

**What happened.** Two people crossed between two looks. Names are carried
by place, each took the other's, and the totals over the shot were a blend
of both people. The lead was thin and the row asked, or it was wide enough
on the looks after the crossing and the wrong person was taken for the
whole shot.

**Signal.** A name whose signature on a look is more like the other
name's previous look than its own. Two names whose boxes overlapped. Both
are arithmetic on a table.

**Mitigation.** The shot is split at either signal and decided on each
part; parts that disagree ask.

**Left.** A crossing between two people the clues cannot tell apart leaves
no signature signal, only the boxes. With the top third and the whole
mask as the scored views, two people in one outfit are that case. The
head view is the one that would still differ, so it is kept on every look
even while it is graded `shown` and scores nothing. And both signals need looks close
enough together to catch the crossing; the tracker's looks are a fixed
stride apart (`subject_track.py::PROBE_STRIDE`).

### 7. The lead's floor, set from one clip

**What happened.** A floor that held on one clip was written into the
constant, and the next clip's right answers and wrong answers both sat on
the other side of it.

**Signal.** A floor is never printed without the shots and looks it came
from.

**Mitigation.** No floor is carried: `TOGETHER_LEAD` stays `None` and each
clip measures its own. A scene with too few typed shots has no floor and
asks.

**Left.** A clip's own floor is set above wrong answers that were
simulated. Real ones may be wider (section 10, result 5).

### 8. Soft film and sharp video do not behave alike

**What happened.** Everything was tuned by eye on soft, grainy film, the
second clip was sharp and passed easily, takes were turned on, and the
next soft job took wrong people; or the other way about.

**Signal.** The leave-one-out leads and the floors of each clip, side by
side, in their own records. A clip whose right answers lead by far less
than another's is a different regime.

**Mitigation.** Floors come from the clip. The second clip is the hard one
by design. And "holds on sharp, fails on soft" is a stated result that
withdraws the claim.

**Left.** Nothing here has been compared across picture qualities. It is
a guess that a clip's own floors absorb the difference.

### 9. The detector's duplicate masks change the count

**What happened.** One person came back as two detections on some looks
and one on others. On the looks with two, there were as many people as
names, and a subject who was away was handed the duplicate.

**Signal.** Detections whose score is close to the detection threshold; a
count that differs between the looks of one shot; what
`subject_tracks.py::one_each` joined and what it left alone; `left_out` in
the answer.

**Mitigation.** Duplicates are joined before anybody is numbered
(7103e264). A person on few of a shot's looks is left out and the row
asks (`subject_tracks.py::sure`). No take on the count while such a
detection is on the shot. Decisions read tables the preview wrote on the
card, never a second run somewhere else.

**Left.** A duplicate that is steady on every look and not nested enough
to be joined is a person to this design. So is a thing the detector takes
for a person, until a sheet says it is a bystander.

### 10. Scenes are grouped wrong

**What happened.** Two scenes filmed in one room in different outfits
were proposed as one, a person said yes to the grouping without seeing
the difference, and the old outfit's looks voted in the new scene.

**Signal.** A confirmed subject under the impostor level against their
own appearance in that scene, on a shot where they are plainly present.

**Mitigation.** The proposal only orders the sheet. The evidence total turns
a wrong join into asks. A scene split in two costs questions and nothing
else.

**Left.** A wrong join where the new outfit is close enough to the old to
pass, which is also the case where the join does little harm.

### 11. A vote that lies is given weight

**What happened.** Build was promoted after it separated two people of
very different size on one shot. On the next job it was read off a
close-up and off a half-hidden person, where it is the model's prior, and
it vetoed right answers; or colour was scored and a scene under coloured
light scattered it.

**Signal.** Each clue's grade and the record that earned it are in the
cast and printed with every decision.

**Mitigation.** Grades. An unmeasured clue can add asks and nothing else.
The reading for a promotion is written before the run (section 2).

**Left.** A clue measured on one clip and promoted for all. The grade is
a clip's setting, and the page says one shot earns a second clip, but
nothing stops a promotion on thin evidence except whoever reads the
record.

### 12. Stored signatures are no longer comparable

**What happened.** A job's first loads were previewed with the pack's SAM
corrections off and its later ones with them on, or at another canvas.
Stored looks and new people were compared anyway, and every likeness fell
a little, evenly, so that nothing looked broken and more shots asked; or
it fell unevenly and the answers moved.

**Signal.** A look's fingerprint beside the table's. It is a string
comparison.

**Mitigation.** Like with like only; otherwise the look is signed again
from its stored frame and mask on the card, or the row asks.

**Left.** A difference the fingerprint does not name. What moves a
signature has been checked for one pair of conditions on one run.

### 13. A session answers for the owner and is wrong

**What happened.** To save the owner's time the lead session answered a
sheet from the tiles. It mistook one person for another in a scene where
the owner would not have.

**Signal.** The share of typed looks whose `who` is a session.

**Mitigation.** The file keeps who answered and against which tile file.
The audit lists session-answered looks apart, and one page a scene shows
them for the owner's glance.

**Left.** A session is as good as its reading of one still. It can say
which outlined person stands where; it does not know the people.

### 14. Two people who look alike in every clue

**What happened.** Two members of a group in matching tops, of one build,
were both named. Every clue put them within the clip's floors of each
other and the totals were decided by noise.

**Signal.** The audit's look-alike pairs, known as soon as a sheet is
confirmed and before any shot is decided.

**Mitigation.** The whole mask is dropped between a look-alike pair and
every shot that could hold both asks.

**Left.** All of it, if a person cannot tell them apart on a tile either.
The design can only say so early.

### 15. A cut the finder misses

**What happened.** A jump cut inside a shot was not found. No decision was
made there, and the tracker carried the lead's mask onto whoever stood in
that place after it.

**Signal.** Cut scores between the floor and the threshold in use; a jump
in the people on screen; the plate vector changing; a name's signature
breaking from its previous look.

**Mitigation.** Each of those splits the shot for this decision. The
tracker's own cut list is not changed by this design.

**Left.** A cut between two angles of one room with the same people in
the same places gives none of the four.

### 16. The cast is stale against a re-cut source

**What happened.** The source was trimmed or made again at another frame
rate, and every frame number in the cast was off by the same amount. The
looks were signed again from the wrong frames.

**Signal.** The source's checksum in the cast's first event; a small
fingerprint of each look's frame.

**Mitigation.** A mismatch is refused before anything is read, and the
tool says which. A look whose frame no longer matches is dropped from
every decision.

**Left.** Nothing known, if both checks are built. The same class of
fault, a first frame typed wrong, has cost this lane before
(`bench/capture_masked_run.py`, the `at=` field), which is why it is here.

### 17. Two sessions write the file at once

**What happened.** Two sessions answered sheets for the same job in the
same minute. One's lines were lost, or the two disagreed about a person.

**Signal.** Who holds the lock; two typed looks for one person on one
frame under two subjects.

**Mitigation.** One writer, append only, a revision in every decision. A
conflict suspends both looks and asks.

**Left.** The lock binds this tool. An editor, `cp` or a script does not
ask it.

### 18. The guess switch is left on

**What happened.** `unsure` was set to guess for a rough first pass and
was still set when the job was rendered for keeping.

**Signal.** The setting is the first line of every decision file and of
the gate's report.

**Mitigation.** The default is stop. Guessed shots are listed in the
delivery record, add nothing a take can lean on, and never cover a
subject's first shot in a scene.

**Left.** A list nobody reads.

### 19. A thing is handed to the wrong subject

**What happened.** Two people each held a cup. The cup that belonged to
the lead in the cast was taken from the second subject's hand on the
shots where they stood close.

**Signal.** More than one candidate within reach of a subject, or one
candidate within reach of two subjects.

**Mitigation.** Several alike ask. A thing takes the verdict of the
subject it belongs to, so it is never surer than they are.

**Left.** The whole section on things is untried, and this is the least
of what may be wrong with it.
