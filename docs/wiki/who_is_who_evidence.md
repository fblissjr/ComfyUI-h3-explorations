# Who is who: which evidence, why, and how it is weighed

last updated: 2026-10-10 (first written, at the owner's ask: a plan; nothing here is built or measured unless a line names a record; the same evening the owner agreed the starting rule, said who records it, and removed the classifier)

Written by hand. A plan that goes with [`who_is_who.md`](who_is_who.md).
That page says how a named person is remembered and how a shot is decided.
This one says what the decision may look at: which features of which
models, why those and not others, how they are weighed, and how cuts are
found. Two of the owner's rules shape it. The first is in `AGENTS.md`, "A
rule that depends on the picture takes its value from the input video, up
front": applied to evidence, a feature counts on a video only for what
that video shows it can do. The second was said for this page on
2026-10-10: simple and elegant over anything else, because the more parts
a thing has, the more of it fails. It carries no measurement; a figure is
cited by its record. Where this page and the code disagree the code is
right. Results are added as dated lines; nothing is rewritten.

*2026-10-10, the owner, as the principle for all of it: this does not
need to be perfect. There will always be edges, and not everything can be
automated. The aim is the best that can be done in a way that is simple,
can be added to, and is driven by the data. So an edge goes to a person
as a question and is not a reason for more machinery, and a step is
judged by whether the common case stays simple and the next idea can be
added without rebuilding it.*

## The first screen: the plan in plain words

**Where we are.** One model decides who is who, and it was not built to:
SAM 3.1's image features, averaged under a person's mask. It became the
only judge because those features were already computed on every frame
looked at, and they worked on the first clips. On soft film with two or
three people it has almost no margin, and the first run with three named
people made a wrong take in most of its shots
([`../../bench/results/2026-10-10_who_is_who_across_shots.md`](../../bench/results/2026-10-10_who_is_who_across_shots.md);
`data/2026-10-10_fun_rest_who_is_where_truth.json`, untracked).

**The simplest thing that could work, and it goes first.** Two votes, and
they must agree.

1. **The likeness we have**, decided for everybody together as it is now
   built.
2. **SAM's own memory, started from a picture where we already know who
   it is.** The owner's idea. SAM's tracker looks back at earlier frames;
   put a confirmed picture in front of a shot, and it looks back at that.

Agree: that is the answer. Disagree, or either cannot tell: a person is
asked, on a picture. No weights, no new model, no new pass over the video.

**Agreed by the owner, 2026-10-10,** with one condition: what the two
votes said, and the answer where they did not agree, is recorded by a
neutral party or by whoever leads the effort, not by the session whose
tool cast a vote (section 2, "Who records it").

**Everything else on this page waits behind that**, in order of how
little it adds: comparing a person part by part, then body build. Each is
tried only if the step before it still gets known shots wrong, and each
has a rule written in advance for when to stop.

**What it will not do.** It will not fix a stranger who resembles a named
person in everything that is measured. Nothing in it fine-tunes a large
model. And nothing is added because it is available: only because it
removed a wrong answer on a shot where the answer was known.

## 1. Why not one judge

No reason was ever written down for SAM 3.1 being the only judge. It was
the cheapest choice and it held on the first clip it met
(`subject_track.py`, "How far the matching can be trusted"). Against it,
from the record of 2026-10-10, in words: a line cannot name a person; a
large clear person is everybody's best match; the margin between two
people in one room is thin; and the same frames scored differently enough
in two previews to flip a call ([`who_is_who.md`](who_is_who.md), section
1 and premortem 12). In the three-subject run, read off its tiles by the
lead session with no corrections typed, trackers took other named
subjects and a person nobody named for their own; the file above has each
shot and its `count`.

A second judge of the same kind would repeat the mistake: another
appearance model sees the same pixels. What helps is a vote that can be
wrong for a different reason.

## 2. The simple plan: two votes that agree, or a question

**Vote one: the likeness, as built.** Every way of handing a shot's
people to the names is totalled over several looks, with "nobody" a valid
answer (`hand_out` (removed 2026-10-10: `decisions.md`, "What the owner cut from the tools around the nodes"); `who_is_who.md`, section 6).

**Vote two: the seeded tracker (the owner's idea, 2026-10-10).**
ComfyUI's tracker looks back at three things: the last few frames
(`comfy/ldm/sam3/tracker.py`, `num_maskmem`), a longer run of small
per-object tokens (`max_obj_ptrs_in_encoder`), and the frame that came
with a mask, which `collect_memory_tokens` always includes and which
never ages out. Today that frame is inside the shot being tracked, and
the person on it is chosen by vote one. The idea is to use a frame where
the person is already known:

- put one confirmed look in front of the shot's frames, at the same
  canvas;
- give its mask as the tracker's seed and track forward;
- drop the first frame's output, and read which detected person the mask
  landed on, and the tracker's own score that anybody was there
  (`subject_track.py::ScoreTap`).

One call for each confirmed look. It compares patch by patch where the
likeness compares one average, and it is the part of SAM that was trained
to find the same object again.

What is not known, and why it is a vote and not the answer: Meta's code
has nothing for a cut and its training video had none
([`../research/masking/2026-10-06_mrsun.md`](../research/masking/2026-10-06_mrsun.md),
"At a cut"); nobody here has measured a track across one
([`../../bench/results/2026-10-06_meta_sam3_card_run.md`](../../bench/results/2026-10-06_meta_sam3_card_run.md),
its verdict line); a look from another shot is "one frame back" only
because we put it there; and this tracker has moved a mask onto a
neighbour with no empty frame between
([`../../bench/results/2026-10-07_subject_regain_looks.md`](../../bench/results/2026-10-07_subject_regain_looks.md)).
To try it takes no node: it is a measurement with ComfyUI's own track
node. To use it, the Subject Track would need its per-shot seed (built by
the tracker's owner, in its sweep when this was written) and a way to
carry a foreign frame in front of a shot.

**The rule.** Both votes name the same person for a name: that is the
answer. They differ, or one of them cannot tell: ask. A vote "cannot
tell" when the likeness has no lead, or the seeded calls from different
confirmed looks land on different people, or the tracker's score says
nobody is there.

*2026-10-10, the first measurement of vote two, the same night
(`data/2026-10-10_fun_seeded_memory/`, untracked: `cases.json` is the
eighteen cases and what a useful vote would do in each, written by the
tracker's session before any run; `reading_by_mrpop.json` is the reading,
by the lead session, which built none of it, and holds every count; the
tracked account is the record's section "A second vote of another kind:
SAM's own memory, seeded on a confirmed frame", ff1568c9). The
cases are the wrong takes and the one miss of the two uncorrected
three-subject previews, with six right takes as controls; each subject
was seeded from one frame a person had confirmed in another shot.*

- *Under "both votes agree, or a person is asked", all but one of the
  wrong calls would have stopped. The one that would not: both votes took
  the lead, small in an inset, for the second subject. Most of the right
  takes pass with no question, and two become a question nobody needed.*
- *The memory alone: it found the lead in her inset in every inset case.
  Where the subject was not in the shot it let go at the join in most
  cases and held a wrong person in two.*
- *Three things only laying the track over the frames showed. The first
  frames after the join are unreliable: they can carry the seed's own
  shape or a wrong person before the mask settles, so a vote is not read
  there. A re-find by the likeness overwrites the track backward, so a
  seeded shot has to have the re-find off for the memory's vote to be
  read at all. And both wrong holds came from one seed, the second
  subject's single confirmed frame, landing on another woman in a shot
  the subject is not in.*
- *What it is worth: one clip, one confirmed look a person, one run each,
  read on six frames a case by eye. No take leans on it. It does mean the
  starting rule now has a second vote that exists, on this clip; and that
  "several confirmed looks that vote" in section 3 has its first reason,
  since the two wrong holds had one look to go on.*

**Who records it** (the owner's condition, 2026-10-10). The session that
builds or runs a vote does not write down its outcome. For each shot, what
each vote said, whether they agreed, and the answer a person gave where
they did not, is recorded by the lead session of the job or by a session
with no part in the tool, and the record says who that was. The same goes
for the known answers a test is scored against: the truth of the first
test bed was read off the tiles by the lead session, not by the tracker's
owner, and that stays the form. In the cast this is the `who` of a typed
look (`who_is_who.md`, section 4); what this adds is that the `who` may
not be the vote's own builder. *My reading of one sentence of the
owner's; if they meant something narrower, this paragraph is what
changes.*

**Why this is the place to start.** It adds one idea and nothing else: no
model that is not already loaded, no pass that does not already run, no
number to tune. And its two votes fail for different reasons. The
likeness is fooled by whoever is large and clear. The tracker is fooled
by whoever stands where it last looked.

## 3. Other tricks with what is already loaded

Each is one idea, needs no new model, and is tried only when section 2
has a result. In the order I would reach for them.

| trick | what it is | what it buys | what could make it worthless |
|---|---|---|---|
| **the round trip** | take the person the seeded tracker found in the new shot, use THAT as the seed, and put the confirmed look after it. If it lands back on the confirmed person, the match holds both ways | a check that costs one more call and needs no second opinion from anywhere else | two people alike to SAM pass it in both directions |
| **several confirmed looks, and they vote** | one seeded call for each look of a person in that scene | a wrong landing from one look is outvoted, and the split is itself the confidence | all looks from one shot are one look |
| **show the confirmed look again** | on a schedule, or when the tracker's own score falls, and not only at a cut that was found | a cut the finder missed stops mattering (section 8) | none known; it costs calls |
| **seed the bystanders too** | a known bystander gets a seeded call of their own | a person the bystander's call claims is not a candidate for a name | a bystander who was never confirmed |
| **the tracker's score as "nobody"** | when every seeded call for a name says the object is not there, the name has nobody | "nobody named is in this shot" from the model itself, not from a line of ours | the score has not been read against a known absence yet (`subject_track.py`, "The tracker's own score") |
| **grow from what a person confirmed** | decide first the shots nearest a confirmed look, in the same scene | the freshest seed is the same outfit a few seconds away | only a look a person confirmed may seed; a guessed shot used as a seed is how one mistake becomes ten (`who_is_who.md`, premortem 3) |

Tried already and set aside: signing every person from a crop at one size.
It raised every likeness, right and wrong together, and widened no lead
that mattered (the record, "One scale: every person signed from a crop").

## 4. Every source of evidence, what it costs, and whether it is used

Three kinds, because votes are independent only when they look at
different things.

| kind | what it looks at | what it is blind to |
|---|---|---|
| **appearance** | what the pixels of a person look like | a change of outfit, a hat, the light; a person of a few pixels |
| **continuity** | where SAM's own memory of a known person lands | two people alike to SAM; a jump it was never trained on |
| **geometry** | how the body is built and how large beside others | whatever the camera hides: a close-up, a person cut by the frame |

Everything named is in the owner's model store unless the row says so.
"Runs already" is whether the pass that makes it runs in today's preview.
Nothing here has been timed.

| feature | kind | runs already? | what it costs a look | used |
|---|---|---|---|---|
| SAM 3.1's trunk features under the mask (the likeness) | appearance | yes, on every frame looked at | nothing more | **yes: vote one** |
| SAM 3.1's tracker seeded from a confirmed look | continuity | the tracker runs; these calls do not | one tracker call a confirmed look, over a few frames | **yes: vote two** |
| the same trunk features, pooled part by part | appearance | the part model runs on the tracked subject only | one part-model forward for every other person of a look | later: section 6 |
| Sapiens2's own backbone features | appearance | no: the forward runs, nothing reads its features | the same forward as the row above, and a tap that does not exist | later, with the row above |
| colour of hair and clothing under the part classes | appearance | no | arithmetic on the part maps | later, inside one scene only |
| body shape and scale from SAM 3D Body | geometry | the pose pass runs on the tracked subject; it keeps these values and does not write them (`body_pose.py::predict`) | two fields; a pose forward for every other person of a look | later: section 6 |
| height beside the others, in one camera | geometry | as above | as above | shown to a person only |
| the DINOv3 backbone inside the body model | appearance | it runs inside the pose pass; nothing reads its features | a tap | only if Sapiens2's features disappoint |
| the tracker's per-object tokens, compared directly | continuity | computed inside the tracker | a tap inside the tracker | no: the seeded tracker uses them as they were trained to be used |

**Left out, and why.**

- *Sapiens2's pointmap and matting models:* a surface and an outline.
  Neither says who.
- *SAM Audio:* it can say who is singing, not who is on screen.
- *A face-recognition or person re-identification model:* not in the
  owner's model store, so a download and the owner's call. Said plainly:
  for a face that shows, a model trained on identity is the most direct
  tool there is, and nothing here replaces it. This page is what can be
  done without one.
- *Fine-tuning SAM 3.1 on each video:* the labels are a handful of
  confirmed looks. It would memorise them, could damage the finding and
  following that work, and breaks every comparison with Meta's reference.

*2026-10-10, the owner, on the crop at one size and on appearance as a
whole: a subject cannot be identified by a face or its features alone,
least of all turned around, or in other clothing or accessories. That is
the "blind to" column of the first table said from the other side, and
it settles what a closer or cleaner look at a person is worth: nothing,
for that case. What is left when a named person is turned away in clothes
the cast has not seen is not an appearance clue at all: the tracker's
memory carrying them from a picture where they are known, who else in
the shot is already accounted for, body build if it earns a place, and a
question. (The crop that was measured was of the whole person, not the
face; it failed for its own reason, in section 3.)*

Two cautions that hold for every row. A different model is not a
different vote: SAM's features and Sapiens2's see the same pixels and
fail together on a small, half-hidden person. And "one scale" is not the
advantage it looked like: for the trunk it flattened the comparison (the
record's section named above), so it is not assumed for Sapiens2 either.

## 5. How the evidence is weighed

**The rule this plan starts with has no weights: agree, or ask**
(section 2). A vote names a person, or names nobody, or says it cannot
tell. Nothing is multiplied by anything.

**What the video decides is who may vote.** Once a sheet is answered
(`who_is_who.md`, section 5) the clip has two piles for every feature:

- the **same** pile: how alike two confirmed looks of one person score,
  from different shots;
- the **different** pile: how alike two different people score. It is
  free and large, since any two people on one frame are certainly
  different and nobody has to say so.

A feature whose two piles stand apart on this video may vote on it. One
whose piles overlap has nothing to say here and is left out, for this
video, without anybody deciding so. That is the owner's rule applied: the
video sets it, and the tool prints which features voted and which were
silent.

**A third vote joins the same way.** When a later feature earns a place
(section 6), the rule stays "all who voted agree, and at least two
voted". A feature that cannot tell abstains; a missing face or a body
cut by the frame is an abstention, not a no.

**Only if agreement asks too often: evidence with a zero.** The next
step up, kept in reserve. Each score is turned into how much likelier it
is to have come from the same pile than the different one; over the
middle counts for an answer, under it counts against, missing counts
zero; one feature a kind, summed across kinds. That is the evidence total
of `who_is_who.md` section 6 with its zero given a rule. It is more
machinery, it needs enough confirmed looks to draw the piles well, and it
can be confidently wrong in a way a plain disagreement cannot. So it
waits for a count of questions that says the simple rule is not enough.

*2026-10-10, the same evening: a first form of it was run on one real
load (reported by the tracker's owner, who built it; its figures are to
be in the record's section on the first run with three subjects). With
the levels taken from that load's own galleries, it was right that two
named subjects were not in five shots, where the built total had handed
them a person nobody named. And on the same five shots it lost the one
subject who was there, small in an inset: under her own level on some,
and beaten by the clear unnamed person on others, where the built total
had her right on all. So the two totals failed in opposite directions on
the same shots, and required to agree they ask on every one. Nothing here
says the evidence total is the better of the two. What it suggests, from
one load: a level has to come from looks at the size a person will be
compared at.*

*2026-10-10, a correction to the line above, from the tool's builder
(the record, "The first run with three subjects"). The built total did
not have the subject who was there right on all five shots: on most, each
from a single look, and the first reading was right for a wrong reason
(a later tracker's looks leave out the people earlier trackers hold, and
the total followed whoever the first tracker did not hold; the tool now
reads only looks nobody was removed from). And the evidence total did not
call her nobody on any of them: it handed her the clear unnamed person
every time. The direction of the line stands: the built total cannot say
nobody, the evidence total prefers a clear bystander to a small subject,
and required to agree, no row of that run agrees with a wrong take. One
thing it adds for this page: a vote read from a look that somebody was
removed from is not a vote. The census of `who_is_who.md` section 5 is
where complete looks come from.*

**Never:** a weight somebody typed.

## 6. What comes later, each only if the step before still fails

**Part to part.** Today a person is one average. The part model labels
every pixel of a person (`sapiens2_parts.py`, its class list), so
features can be pooled under the face, the hair, the upper clothing and
the lower clothing, and compared like with like. Two things it buys: a
missing part becomes no evidence where today a person with no head found
is no match at all; and what survives a change of outfit (the face, hair
as cut) separates from what does not (clothes, a hat) part by part. Its
weak point is the case most wanted: on a person a few tokens tall the
part map is itself unreliable, and the whole person is used. It needs the
part model on every person of a look, which is a new pass.

*2026-10-10, the owner, on segments as features: a white top a person
wears is a feature even though they will not always wear it; segment it,
follow it, and it counts while it is there. The same for the shape of
their hair, and for other parts. That is this paragraph said better, and
it changes two things in it.*

- *A segment is a feature with a lifetime. A person in the cast is then a
  short list of segments for each appearance (this top, this hair, this
  hat), each with what was measured of it: its colour, its outline, the
  features under it. A segment that is not there is no evidence either
  way. One that is there and matches counts; one that is there and
  plainly differs inside one scene counts against. Adding a kind of
  segment adds a feature and rebuilds nothing.*
- *Its first use is free, and is a check, not a search. The part model
  already runs on every frame of a tracked subject, and the capture
  already saves each segment a frame (`bench/capture_masked_run.py`, the
  files a class map adds). So once the two votes have named a person and
  the tracker has followed them, their segments can be compared with the
  segments of that subject's confirmed look in the same scene, with no
  new pass: a dark jacket where the confirmed look has a white top is a
  wrong take, or a change of clothes, and either way a question. Only
  the search over every candidate on a look needs the new pass, and that
  stays where section 7 puts it.*
- *Not known: how often tonight's wrong takes were of a person dressed
  differently from the subject in that scene. The test bed can say, from
  files that exist.*

*2026-10-10, the owner, a second use of the same segments: a segment is
also something to replace. A top can be swapped for a designed one given
as a reference picture, and likewise any other part. So one segment does
two jobs from one record: it is evidence of who a person is while it is
there, and it is a region a job may redraw. Which is which stays where
[`who_is_who.md`](who_is_who.md) section 4 puts it: the cast says what
segments a subject has in the source; a job says which of them it
replaces and with what picture. Two things follow for recognition.
Recognition always reads the source, never a render, since a render may
have had the segment swapped. And replacing a part by its classes is
built and has rendered ([`masked_v2v.md`](masked_v2v.md), the parts
graphs; `workflows/h3_config.py::MASKED_UPPER_PARTS`), while a garment
swapped for a reference picture has not been rendered: the masking
board's card `use-part-menu-garment` has what it still needs.*

*2026-10-10, the owner, on what segments cannot be counted on: a hat on
the street hides the hair, so the hair says nothing there; a shot may
show only a face; hair worn down is a different shape from hair worn up.
Each look shows a different set of segments and none is always there.
So no segment is ever required, a look is described by the segments it
actually shows, and a subject has several looks a scene on purpose
(`who_is_who.md`, section 5). A close-up of a face is compared with the
faces of that subject's confirmed looks and with nothing else; if no
confirmed look has a face to compare, it is a question. The one vote
that does not depend on which segments show is the tracker's memory
from a frame where the person is known, which is one more reason it is
half of the starting rule.*

**Body build.** The one clue that should not change with an outfit. The
body model estimates it afresh on every frame and completes what it
cannot see, so it is read only on frames where most of the body is in the
picture ([`meta_perception_models.md`](meta_perception_models.md)).

*2026-10-10: a small classifier trained for each video was the third
thing in this section, as a last resort. The owner removed it the same
evening, with the words that the more complex a thing is the more of it
fails. It is not part of this plan and is not to be proposed again
without the owner asking; the commit before this line has what it said.*

## 7. The order, and what would stop each step

Every step is scored on shots where the answer is known, and renders
nothing. **The test bed:** the two loads of the three-subject run
(`data/2026-10-10_fun_rest_who_is_where_truth.json`, read by the lead
session off the tiles, untracked) and the kitchen load before them, on
the soft clip; then the matching-tops clip of `who_is_who.md` section 10.

**The score** of an idea is two counts on the test bed: the wrong takes
it would have avoided, and the right calls it would have broken, with
the shots that would ask counted beside them. The reading for every step,
written before any is run: it stays if it avoids more than it breaks on
both loads and asks about fewer shots than there are; otherwise it is set
aside with a dated line saying why. One clip earns a second clip, not a
default.

1. **The seeded tracker, on the known wrong takes.** Stops if the mask
   lands on whoever stands where the confirmed person stood, whoever they
   are: that is place, which the design already has.
2. **Agree or ask**, both votes, over the test bed. This is the first
   number that says whether two votes beat one.
3. **The round trip and several looks**, if step 2 leaves wrong takes
   that both votes agreed on. *And, since the owner's point on segments:
   the free check of the taken person's segments against the confirmed
   look's, scored the same way, since it needs no new pass.*
4. **The piles**, from what the tables already hold: which features
   separate this clip's people at all. No model.
5. **The part pass**, if appearance is still the weak vote. Stops if the
   part maps are wrong on most of the people that matter.
6. **Body build**, if two people alike in appearance are still confused.
   Stops if one person's values scatter over a shot as widely as two
   people's differ.
7. **The second clip.** Sharp video, matching tops, the named people at
   the back.
8. **Evidence with a zero**, only on a count of questions that says the
   simple rule asks too much.

## 8. Cuts: finding them, and making a missed one harmless

**What finds a cut today.** A delta between neighbouring frames, of one
kind: each frame is reduced to a small map of where its edges are, and a
cut is a frame whose map does not match the one before
(`subject_track.py::cut_scores`). A change of light moves no edges, so it
is not taken for a cut. The line is set from the clip's own scores, in
their widest gap (`auto_cuts`), and flat bars at a frame's edges are left
out (`flat_borders`).

**How it has done.** It finds a hard cut between two different pictures.
On record it has missed a jump cut inside a cutaway, a cut between two
angles of one room, and two camera changes inside one scene, and it has
called every frame of a black opening a cut
([`../research/masking/2026-10-04_mrhf.md`](../research/masking/2026-10-04_mrhf.md);
[`../../bench/results/2026-10-06_subject_track_defaults.md`](../../bench/results/2026-10-06_subject_track_defaults.md);
`data/CAPTURE_GAPS.md`, entry 13, untracked). A clip that opens on a
dissolve has no single frame to find.

**Three small changes, simplest first.**

1. **A cut is a spike, not a level.** Today every frame over the line is
   a cut. A real cut is one frame that stands far above the frame before
   and the frame after. A run of high scores is something else: black
   frames, a flash, fast movement. One comparison with the two
   neighbours, and a frame with no edges at all is skipped.
2. **A second delta of a different kind, only for the doubtful ones.** A
   cut inside one room keeps the room's edges, so the edge map barely
   moves. What does move is the people: the tracked mask jumps
   (`subject_tracks.py::unbroken` already measures a mask that has left
   the figure it was on), the tracker's own score falls, the number of
   people changes. These exist already and are read today only after a
   loss. Here they would mark a possible cut.
3. **Look a little further for a dissolve.** A frame compared with the
   one half a second on, where neighbours never differ enough.

**And the better answer: do not need it to be perfect.** A missed cut
costs something today because the tracker carries a mask across it onto
whoever is there. With the confirmed look shown again on a schedule or
when the tracker's score falls (section 3), a missed cut is caught by the
next showing, and an invented cut costs one decision that was not needed.
The finder then has to be good, not perfect.

*2026-10-10: the owner agrees a cut finder does not need to be perfect.
So the order here is the spike rule, then showing the confirmed look
again, and the rest only if cuts are still costing shots after those.*

**If rules are not enough,** a small model trained only to find shot
changes exists as a kind of thing. None is in the owner's model store, so
it is a download and the owner's call, and it is one more model to load.
It is named here so that nobody builds a fourth rule before asking.

**How it is checked.** Against cuts a person has already listed for clips
in the records (the notes and record above list them by frame), counting
cuts missed and cuts invented, before and after each change.

## 9. What could go wrong with this plan

Each as if it were three weeks on. **Signal** is what would show it
before anything is built on it.

- **The two votes were fooled by the same person.** The likeness took the
  large clear neighbour and the seeded mask slid onto them too.
  *Signal:* on the test bed, the wrong takes on which both votes agree.
  *Held by:* the round trip, several looks. *Left:* two people alike to
  SAM are alike to both votes, since both are SAM.
- **The seeded tracker followed place, not person.** *Signal:* where the
  mask lands on shots where the person is known to have moved across the
  frame. *Held by:* step 1's stop rule.
- **Agreement asked about nearly everything.** *Signal:* the count of
  shots that ask on the test bed, before anything is built on the rule.
  *Held by:* that count is part of every step's score. *Left:* if two
  votes rarely agree, the simple rule is safe and useless, and section
  5's reserve is the next thing, with its own risks.
- **A later feature was added because it was there.** *Signal:* a step
  with no count of wrong takes avoided. *Held by:* section 7's reading.
- **The piles were drawn from too little.** A subject confirmed in one
  shot has no "same" pile. *Signal:* the count behind each pile, printed
  with it. *Held by:* a feature with too few behind it does not vote.
- **The spike rule dropped a real cut** that came beside a flash or a
  second cut. *Signal:* cuts missed on the listed clips, before and
  after. *Held by:* a missed cut is made cheap by showing the look again.
- **Everything was measured on one soft clip.** *Signal:* none; it is
  true today. *Held by:* what may vote is set on each video, and step 7
  is a sharp one.

## 10. What enforces each rule today, and what a session has to remember

*2026-10-10, written at the end of the day the two pages were written.*
One row for each rule of [`who_is_who.md`](who_is_who.md) and of this
page. "A tool" means something refuses, blocks or cannot do otherwise,
and a check holds it. "A session" means nothing stops the rule being
broken: it is kept only if whoever runs the job knows it. The second
column is the list the next job most needs.

| the rule | enforced today by a tool | or kept only by a session remembering |
|---|---|---|
| a job is built on its input video's analysis folder | `bench/masked_job.py` refuses without one, or with another file's or a stale one (`clip_folder.py`; `bench/check_clip_folder.py`) | |
| a render is queued only behind a verified capture and a clear gate | `bench/masked_job.py`, `queue` | |
| a shot called absent with people on screen is answered before a render | the gate's `absent_with_people_on_screen` (`bench/capture_masked_run.py::flag_shots`) | |
| a shot is decided for all named subjects together, and a row that does not agree is answered | the gate's `who_is_who_unanswered` (`bench/capture_masked_run.py::flag_who`), when the decision's file is given to it | giving it the file: a preflight run without one says so in a field and does not block |
| nothing is taken without a person | `TOGETHER_LEAD` was `None`, so every differing row asked (removed 2026-10-10: `decisions.md`, "What the owner cut from the tools around the nodes") | not passing `--unsure take` to `bench/who_is_who_across_shots.py`; the switch exists and nothing refuses it |
| a weak row suggests nobody | the corrections file (`--suggest-lead` unset); `bench/check_subject_track.py` | |
| one detection a person; a detection with no head is handed to nobody; only looks nobody was removed from are read; nobody has a value | `subject_tracks.py::one_each` and the corrections tool, each with a check | |
| no look in the first frames after a cut | for the tracker's own looks, `subject_track.py::PROBE_OFFSET` | for any look taken another way |
| a seed a tool can write for a shot | the Subject Track's `seeds` input (`subject_track.py::parse_seeds`) | the step from source frames to the load's frames: nothing does that subtraction yet |
| thresholds come from the clip | the cut line and the match line (`subject_track.py::auto_cuts`, `auto_match`); the levels the corrections tool takes from a load's own tables | every other line: the rule is in `AGENTS.md` and no tool checks a constant against it |
| the census: every shot looked at with no subject and nobody set aside | | not built. `bench/clip_analysis.py` makes the folder and nothing more yet. Until it exists most shots have one complete look |
| the cast file: one writer, never edited, a revision in every decision, a removal that marks what leaned on it | | not built. Today's memory of who is who is typed corrections in a job file, a load at a time |
| one confirmation a subject a scene; scenes and appearances | | not built. A session reads every tile |
| only a look a person confirmed leans a take | | not built; nothing is taken, so nothing can break it yet |
| clues are compared only when made the same way (the SAM corrections on or off, the canvas, the precision) | | a session. Two previews of one load differed enough to flip a call the day this was written |
| two votes must agree, or a person is asked | | not built: the seeded-tracker vote does not exist |
| what the votes said and the answer given is recorded by the lead or a neutral session | | a session. The one instance is the truth file of the first test bed |
| no typed weight; a feature votes on a video only if that video's own piles stand apart | | a session; nothing computes the piles |
| a feature is scored only after a record earns it | | a session reading the record |
| each idea is scored on shots with a known answer before anything is built on it | | a session. The truth file exists; no tool scores against it |
| no classifier for each video; recognition reads the source, never a render | | a session, from [`decisions.md`](decisions.md) and this page |
| a thing inside a region is answered for before a render | for a tracked subject's own parts, the gate's `segment_inside_region`, which warns and does not block | for an object: a session adds a tracker by hand, or nothing knows it is there |
| a cut is a spike, not a level; the confirmed look is shown again | | not built: `subject_track.py::find_cuts` takes every frame over the line |

*2026-10-10, later the same night: one row of the right column moved.
The cast file exists in its least form, `bench/cast_file.py` with
`bench/check_cast_file.py` (35e45d29 to 68347328, the tracker owner's). It
keeps what a person answered about a frame: for each named subject, that
person's box on the frame, or the words "none of the outlined people". It
asks about the people outlined, shows nothing a tracker concluded, stores
a tile whole or not at all under a lock, never rewrites a line, and can
write a load's seeds from the answers. It holds no likeness, no scene and
no appearance, and it decides nothing, which is the owner's simple form
of section 4 of [`who_is_who.md`](who_is_who.md) and not all of it. What
is still a session remembering: nothing calls it yet, so a job's answers
reach it only if somebody runs it; and one confirmation a subject a scene
is still a rule on a page.*

**What the table says.** What is enforced is what was built this day
around the trackers as they are: the folder, the gate, the decision that
asks. Everything that is this design and not yet code is in the right
column, and the two rows there that cost a job most are the census and
the confirmation for each scene: without them a person answers shot by
shot, load by load, and the answers live in a job file.
