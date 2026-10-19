# Who is who: recognizing a named person again across cuts, shots, loads and sessions

**Today who is who is the owner's answer, kept in one file per video, and nothing is taken without a person.** A
tracker's pick is a guess. A session that did not build the preview reads its tiles, the owner clicks on the who
page, and the cast file keeps the answer with its frame and box; seeds from it start each tracker. The plan to ask
less is a second vote of another kind (SAM's own tracker memory, started from a frame a person confirmed): where it
agrees with the tracker's pick, that is the answer; where it differs, a person is asked. Nothing more is built until
that step's count of questions says the simple rule is not enough.

Two of the owner's principles shape it (2026-10-10): simple and elegant over anything else, because the more parts a
thing has the more of it fails; and it does not need to be perfect, so an edge goes to a person as a question and is
not a reason for more machinery. And a feature counts on a video only for what that video shows it can do (root
[`AGENTS.md`](../../AGENTS.md), "A rule that depends on the picture takes its value from the input video, up front").

How SAM finds and follows a person is [`sam31.md`](sam31.md); this page is only who that person is.

## Who is who: a session reads first, the owner answers, and both are kept

A tracker follows whoever it was started on. Started on the largest person of a frame it is guessing, and the batches
of guesses kept against the owner's answers (`data/<video>/picks_against_answers.jsonl`, the table beside it) show how
often the guess was somebody else. So nothing is built on a preview until a person has said who its tracker holds.

| step | what happens | tool |
|---|---|---|
| 1 | a session that did not build the preview reads its tiles and writes one line a preview: the picked person's number, what they look like, whether that is the right person, why, and `thinks_it_is` | `bench/who_page.py`'s docstring has the fields; the line goes in the batch's notes file |
| 2 | the who page shows that reading as a guess, closed, under the tile, and the owner clicks; their typed words are read with their clicks, whole, never a click without them | `bench/who_page.py --notes` |
| 3 | the cast run stores the owner's answers and writes the picks batch in the same run; a shot the owner's own words doubt is marked unsure, with those lines, and counted apart | `bench/page_answers_to_cast.py --picks` (refused without it), `--owner-unsure` |
| 4 | a shot the record is `unsure` of for a subject is planned by nothing: its preview is made to ask, the owner is asked once on its numbered picture, and no load is built on it until answered | `bench/shot_to_load.py preview --ask SUBJECT:why` |
| 5 | each batch is read against the last: counts, never a rate from one batch, with what the table cannot see | `bench/picks_against_answers.py` |
| 6 | a shot the tracker cut into several is shown cut by cut, every person's box in a colour of their own; a preview made again takes its cuts from the earlier preview's table, and each cut that holds a box gets its own seed | `bench/who_page.py --every-cut-of`; `bench/shot_to_load.py preview --seed-cuts-of` |

- **Before the owner is asked anything, what they already answered is read**, and only the missing fact is asked,
  with why the earlier answer does not cover it. The cast file (`data/<video>/cast/cast.jsonl`, each line with its
  frame, its box and the answers lines it came from) is the record, and the who page prints the answer that already
  stands above a subject's buttons. Example, 2026-10-13: the owner asked whether the small figure at the back of the
  hall shot is the second subject; the cast file held their answer of 2026-10-11 saying so on two frames of that shot,
  and the tracker was seeded from it. A session answers that from the record and asks nothing.
- **Without cuts, a shot gets one seed**, the box latest in the cast file, and every other cut is matched against how
  the subject looks on that one frame.
- **Every box in the cast file is the owner's answer**, so every subject of every tile still takes their click. The
  day they say a session's reading may stand as a seed, the page asks only where a session could not tell, where the
  subject is not outlined, or where the owner's words doubt the shot; the cast file would take a session's box marked
  as a session's, replaced by the owner's click. Until then a session's reading seeds nothing.
- **Who records a vote is not the vote's builder** (the owner's condition, 2026-10-10): what each vote said and the
  answer given where they differ is recorded by the job's lead or a session with no part in the tool, and the record
  says who.

## What is built, and what each does

| tool | what it does | what it does not |
|---|---|---|
| `bench/cast_file.py` (`bench/check_cast_file.py`) | keeps what a person answered about a frame: for each named subject, that person's box, or "none of the outlined people"; stores a tile whole or not at all under a lock, never rewrites a line, and writes a load's seeds from the answers (`bench/cast_file.py::seeds`) | holds no likeness, no scene and no appearance, and decides nothing |
| the Subject Track's `seeds` (`subject_track.py::parse_seeds`) | one line a shot, a frame of the LOAD and a box, or `none`; the detection the box names is taken at or over `subject_track.py::SEED_AGREES` (a starting value), else the shot is left empty as "absent (seed refused)" with the nearest box. A seed on the picked shot is the pick | a load whose every shot is seeded still runs the pick rule first |
| a tracker told who earlier trackers hold | lists everybody on a frame and marks the held, in one numbering for every tracker of a job (`shot_table.py::TABLE_VERSION`, `subject_track.py::held_people`) | |
| `bench/who_is_who_across_shots.py corrections` | lays the takes of one load side by side, a row a shot: `answered` when every call is a person's word (a pick, a typed correction, a seed) or nothing with a head was detected, `ask` otherwise, with the question in the tile's own numbers; `--frames` cuts tables of two previews to a load's frames and refuses when they differ | compares nobody with anybody, suggests nothing, takes nothing |
| the gate (`bench/capture_masked_run.py`) | `who_is_who_unanswered` (when given the corrections file), `absent_with_people_on_screen`, `subject_never_found`, `taken_under_its_own_line` | a take of the wrong person over the tracker's line reads "clear" ([`known_before_a_render.md`](known_before_a_render.md)) |
| the shot table (`shot_table.py::LOOKS_ARE`, `SIGNATURES_ARE`) | keeps what a refused shot was judged on; the report says when the automatic line was put in a gap between two shots (`subject_track.py::moved_by_a_gap`) | |

**Removed, and not to be rebuilt without the owner** (2026-10-10, [`docs/state/decisions.md`](../state/decisions.md),
"What the owner cut from the tools around the nodes"; ace202c4, e9de8cc3): the sum over all subjects at once
(`hand_out`), its lead (`TOGETHER_LEAD`), the evidence total with its impostor level, the suggestion floor, and the
switch that took a row without a person (`--unsure take`). They were the tracker's own likeness arranged again, not a
second vote, and one handed a stranger to a subject who was away. Also removed the same evening: a small classifier
trained for each video, not to be proposed again unless the owner asks.

## What the evidence showed

All from one load of one clip with two, then three, named subjects:
[`../../bench/results/2026-10-10_who_is_who_across_shots.md`](../../bench/results/2026-10-10_who_is_who_across_shots.md).
Every figure there is from soft, grainy film.

- **A line cannot name a person.** A frame of one subject scored over the tracker's "same person" line against the
  other subject's saved looks, inside one shot and across a cut. So no rule of the form "this subject's looks, over a
  line" holds, today's hand-over to a later load (`subject_track.py::clear_best`, through `subject_from`) included.
- **A large, clear person is everybody's best match**, so each subject taking its best person fails.
- **Which of the named subjects a person is more like** was right for the clear person on every look and unreliable
  for the small one.
- **The same person one cut apart is not one appearance**: the lead with her hat off scored under the floor against
  the frame she was picked on, and over and under the line against eight looks from the shot before.
- **The detector's count is not steady**: one person came back as nested masks, one at the detection threshold,
  present on the card and absent on the CPU. Duplicates are now joined before anybody is numbered
  (`subject_tracks.py::one_each`).
- **Scores move between runs**: two previews of one load hours apart gave a subject scores a couple of hundredths
  apart, and her tracker's call on a shot flipped. A call that turns on a hundredth is not a call.
- **Signing every person from a crop at one size** raised every likeness, right and wrong together, and widened no
  lead that mattered (the record, "One scale"). Set aside.
- **In the three-subject run** (`data/2026-10-10_fun_rest_who_is_where_truth.json`, read off the tiles by the lead
  session; its `count` has the totals): trackers took a large person nobody named for the lead and the third subject
  on most of five shots, where the lead was small in an inset; pictures on a wall and a hand with an object came back
  as people; a "take the only person left" rule took another named subject; two trackers took each other's subject.
  In another outfit the lead fell under her own level.
- **A likeness score between crops** (`subject_track.signature`) cannot say whether a render looks like its reference
  either ([`state_signals.md`](state_signals.md), "Whether a render looks like the reference").

The owner's reading (2026-10-10): a subject cannot be identified by a face or its features alone, least of all turned
around or in other clothing or accessories. What is left when a named person is turned away in unseen clothes is not
an appearance clue: the tracker's memory from a frame where they are known, who else in the shot is accounted for,
body build if it earns a place, and a question.

## The plan: two votes that agree, or a question

1. **Vote one: the tracker's own take** on its subject, shot by shot, with the people another tracker holds marked
   (`subject_track.py`, "Two nodes, one person").
2. **Vote two: SAM's tracker memory, seeded from a confirmed look** (the owner's idea, 2026-10-10). ComfyUI's tracker
   always looks back at the frame that came with a mask, which never ages out (`comfy/ldm/sam3/tracker.py`,
   `collect_memory_tokens`). Put one confirmed look in front of a shot's frames at the same canvas, seed its mask,
   track forward, drop the first frame's output, and read which detected person the mask landed on and the tracker's
   own score that anybody was there (`subject_track.py::ScoreTap`). One call a confirmed look. It compares patch by
   patch where the likeness compares one average.

**The rule.** Both votes name the same person: that is the answer. They differ, or either cannot tell (the likeness
has no lead, seeded calls from different looks land on different people, or the tracker's score says nobody): ask. No
weights, no new model, no new pass over the video. Agreed by the owner on 2026-10-10. The two votes fail for different
reasons: the likeness is fooled by whoever is large and clear, the tracker by whoever stands where it last looked.

**The first measurement of vote two** (2026-10-10; `data/2026-10-10_fun_seeded_memory/`, untracked: `cases.json` is
the eighteen cases written before the run, `reading_by_mrpop.json` the reading by a session that built none of it;
the record's section "A second vote of another kind: SAM's own memory, seeded on a confirmed frame", ff1568c9):

- under "both agree, or ask", all but one wrong call would have stopped; the one that would not was both votes taking
  the lead, small in an inset, for the second subject. Most right takes pass; two become a question nobody needed;
- the memory alone found the lead in her inset in every inset case, let go at the join where the subject was absent
  in most cases, and held a wrong person in two, both from one seed (the second subject's single confirmed frame);
- the first frames after the join are unreliable, so a vote is not read there; a re-find by the likeness overwrites
  the track backward, so a seeded shot needs the re-find off for the memory's vote to be read.

One clip, one confirmed look a person, one run each, read on six frames a case by eye. No take leans on it. What is
not known: Meta's code has nothing for a cut and its training video had none (a reading of 2026-10-06); nobody has measured
a track across one ([`../../bench/results/2026-10-06_meta_sam3_card_run.md`](../../bench/results/2026-10-06_meta_sam3_card_run.md));
and this tracker has moved a mask onto a neighbour with no empty frame between
([`../../bench/results/2026-10-07_subject_regain_looks.md`](../../bench/results/2026-10-07_subject_regain_looks.md)).
Using it needs a way to carry a foreign frame in front of a shot.

**The order, each scored on shots with a known answer and rendering nothing.** The test bed is the two loads of the
three-subject run and the kitchen load before them, then the matching-tops clip. An idea's score is the wrong takes it
would have avoided, the right calls it would have broken, and the shots that would ask; it stays if it avoids more
than it breaks on both loads and asks about fewer shots than there are. One clip earns a second clip, not a default.

| step | what | stops if |
|---|---|---|
| 1 | the seeded tracker on the known wrong takes | the mask lands on whoever stands where the confirmed person stood |
| 2 | agree or ask, both votes, over the test bed | (the first number that says whether two votes beat one) |
| 3 | the round trip, several confirmed looks that vote, and the free segment check (below) | both votes still agree on wrong takes |
| 4 | the piles: which features separate this clip's people at all, from what the tables hold | |
| 5 | the part pass, if appearance is still the weak vote | the part maps are wrong on most of the people that matter |
| 6 | body build, if two people alike in appearance are still confused | one person's values scatter as widely as two people's differ |
| 7 | the second clip: sharp video, matching tops, the named people at the back | |
| 8 | evidence with a zero, only on a count of questions that says the simple rule asks too much | |

**Other tricks with what is already loaded**, each tried only once step 2 has a result:

| trick | what it buys | what could make it worthless |
|---|---|---|
| the round trip: seed from the person found, put the confirmed look after it, and see if it lands back | a two-way check for one more call | two people alike to SAM pass both ways |
| several confirmed looks that vote | a wrong landing from one look is outvoted, and the split is the confidence | all looks from one shot are one look |
| show the confirmed look again on a schedule or when the tracker's score falls | a missed cut stops mattering | none known; it costs calls |
| seed the bystanders too | a person a bystander's call claims is not a candidate | a bystander never confirmed |
| the tracker's score as "nobody" | "nobody named is here" from the model, not a line of ours | not yet read against a known absence |
| grow from what a person confirmed: decide the nearest shots first | the freshest seed is the same outfit seconds away | only a confirmed look may seed; a guessed one is how one mistake becomes ten |

## The evidence, and whether it is used

Votes are independent only when they look at different things: **appearance** (blind to a change of outfit, a hat,
the light, a person of a few pixels), **continuity** (blind to two people alike to SAM, a jump it was never trained
on) and **geometry** (blind to what the camera hides). Nothing here has been timed.

| feature | kind | runs already? | used | what makes it lie |
|---|---|---|---|---|
| SAM 3.1's trunk features under the mask (the likeness): the top third, the head, the whole mask | appearance | yes, on every frame looked at | vote one | two people with like hair and shoulders; a few feature cells; the first frames after a cut; the whole mask is mostly clothes, so a change or a shared outfit (`../../bench/results/2026-10-04_subject_track_three_clips.md`) |
| SAM 3.1's tracker seeded from a confirmed look | continuity | the tracker runs; these calls do not | vote two | place, when a neighbour steps where the person stood |
| the trunk pooled part by part, under the part classes | appearance | the part model runs on the tracked subject only | later: step 5 | a person a few tokens tall, where the part map is itself unreliable |
| Sapiens2's own backbone features | appearance | the forward runs; nothing reads them | later, with the row above | trained on people is not trained on identity |
| colour of hair and clothing under the part classes | appearance | no | later, inside one scene only | coloured light, a uniform, a hat; lightness alone ranked one subject first in a crowd on one stretch, a reason to try it and not a result ([`../../bench/results/2026-10-07_subject_likeness_in_a_group.md`](../../bench/results/2026-10-07_subject_likeness_in_a_group.md)) |
| body shape and scale from SAM 3D Body | geometry | the pose pass keeps them and does not write them (`body_pose.py::predict`) | later: step 6, read only where most of the body is in the picture | estimated afresh each frame; a half-hidden person's build is partly the prior; no confidence ([`sam3d_body.md`](sam3d_body.md), "There is no confidence to read") |
| height beside the others, and who is nearer | geometry | as above | shown to a person only | sitting, bending; two people at nearly one distance |
| the DINOv3 backbone inside the body model | appearance | it runs; nothing reads it | only if Sapiens2's features disappoint | |
| the count: fewer people than names | a constraint | yes | in the gate | a bystander, a duplicate detection, a thing taken for a person |

**Left out, and why.** Sapiens2's pointmap and matting say a surface and an outline, not who. SAM Audio can say who is
singing, not who is on screen. A face-recognition or re-identification model is not in the owner's store: a download
and the owner's call, and for a face that shows it is the most direct tool there is. Fine-tuning SAM 3.1 on each video
would memorize a handful of looks, could damage finding and following, and breaks every comparison with Meta's
reference. A different model is not a different vote: SAM's features and Sapiens2's see the same pixels.

**Segments as features with a lifetime** (the owner, 2026-10-10). A white top, a hair shape, a hat is a feature while
it is there. A person in the cast is then a short list of segments for each appearance, each with what was measured of
it; a segment that is absent is no evidence either way, one that is there and plainly differs inside one scene counts
against. No segment is ever required, and a close-up of a face is compared only with faces. The first use is free: the
part model already runs on every frame of a tracked subject and the capture saves each segment a frame, so once a
person is named their segments can be checked against the confirmed look's in the same scene. The same segment is
also a region a job may redraw; recognition always reads the source, never a render.

**How the evidence would be weighed.** No weights: a vote names a person, names nobody, or cannot tell. What the video
decides is who may vote: once a sheet is answered the clip has a "same" pile (two confirmed looks of one person) and a
large free "different" pile (any two people on one frame). A feature whose piles stand apart on this video votes on
it; one whose piles overlap is silent, and the tool says which. A third vote joins as "all who voted agree, and at
least two voted". Evidence with a zero (how much likelier a score is from the same pile than the different one) is
kept in reserve: it was run once on one load and failed in the opposite direction from the likeness, preferring a
clear bystander to a small subject who was there.

## Cuts: finding them, and making a missed one harmless

**What finds a cut today**: one minus the correlation of neighbouring frames' edge maps (`subject_track.py::cut_scores`),
the line set in the widest gap of the clip's own scores (`auto_cuts`), flat bars left out (`flat_borders`). It finds a
hard cut between two pictures. On record it missed a jump cut inside a cutaway, a cut between two angles of one room
and two camera changes inside a scene, and called every frame of a black opening a cut
([`../../bench/results/2026-10-06_subject_track_defaults.md`](../../bench/results/2026-10-06_subject_track_defaults.md);
`data/CAPTURE_GAPS.md`, entry 13). A clip that opens on a dissolve has no single frame to find.

**The order agreed with the owner** (a cut finder does not need to be perfect): first, a cut is a spike, not a level
(a frame far above both neighbours; a run of high scores is black frames, a flash or fast movement); then show the
confirmed look again on a schedule, so a missed cut is caught by the next showing; the rest only if cuts still cost
shots: a second delta of another kind for the doubtful ones (the tracked mask jumps, the tracker's score falls, the
count changes), and a look half a second on for a dissolve. A small trained shot-change model is a download and the
owner's call; it is named so nobody builds a fourth rule before asking. Checked against cuts a person has listed,
counting cuts missed and invented. None of this is built: `subject_track.py::find_cuts` takes every frame over the line.

## The design further ahead

Built only as the steps above earn it. Nothing here is code unless the table of what is built says so.

- **The cast as the video's memory**, one file a video under `data/<the video's stem>/cast`, shared by every job: each
  named subject remembered as several **appearances** (a hat and a coat; no hat), each with a few confirmed looks,
  large and small, facing and turned. Nothing is forgotten with time. A **scene** is a set of shots in which nobody
  named changes appearance: a person's word, ordered by the tool's proposal (the plate's features outside every
  person mask, and time). Scene-bound clues (clothes, colours, the whole mask) vote only inside a scene; stable ones
  (face, build) everywhere once measured. A new outfit is an expected event: where no appearance explains anybody on
  screen, the row asks, and a yes makes a new appearance. **One confirmation a subject a scene**, before any shot of it
  is decided: the main bound on a poisoned look, and why a person's attention grows with scenes, not shots.
- **The cast's rules**: only a look a person confirmed may lean a take; clues are compared only between like
  fingerprints (the model file, the precision, the canvas and crop, whether the SAM corrections were on: the
  corrections change the model's input at its first layer, `sam31_corrections.py::attach`, so on and off are not
  comparable until a run says so; the card against the CPU and a track's mask against a detection's did not move the
  top third on the frames one run compared, `data/2026-10-10_who_is_who_across_shots/2026-10-10_who_is_who_across_shots_looks.json`);
  one writer at a time, append only; the source's checksum is its first event, so a re-cut source is refused.
- **The casting pass (seeding it up front)**, part of the analysis that runs first for a video: the cuts and every cut
  score; on each shot several steady looks, never in the first frames after a cut (`subject_track.py::PROBE_OFFSET`),
  with every person's mask, size and clues, nobody held and no subject named. Its sheet, one a scene, is the one moment
  a person answers: the first tile names which numbered person each subject is; later tiles ask yes or no on one
  outlined person; looks shown differ on purpose (largest, smallest, most turned, least like the rest). It also
  writes the clip's own two distributions, from which every floor is set.
- **Things as first-class entries** (a bowl, a hat in a hand), decided after the people: one that belongs to a subject
  is looked for only near them and takes their verdict; one that stands in the scene is known by its place on the
  plate inside a scene; a part rides on its owner's track. A thing is seeded by a phrase, a typed box, or the part
  node's `held` mask, which names nothing (`sapiens2_parts.py`, "What the subject holds"); a phrase finds a small prop
  for the first seconds and then loses it ([`sam31.md`](sam31.md), "a small thing";
  [`../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`](../../bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md)).
  "Near" is in the subject's own units (head heights), never pixels. Before a render, every thing near a subject would
  be found, tied to that subject, and answered for in one word a job (it **stays**, it **goes with** the subject, or
  it is **left** to chance); a thing that stays and touches the subject is given back at assembly, never kept at the
  sampler. Today that knowledge arrives after a render: a bowl and a dish inside a whole-person region were redrawn as
  other things (`data/CAPTURE_GAPS.md`, "Segments, not only people"), and a thing in a hand was caught only on a mask a
  session made by hand (`data/2026-10-10_fun_open_hair_face_0014`, its `whatif` folder;
  `data/2026-10-10_fun_open_hair_face_run_0014`). The gate's `segment_inside_region` warns for a tracked subject's own
  parts; for an object, a session adds a tracker by hand.
- **One tracker for every name**, reading the cast and replacing `subject_from` and the `others` wiring, only if the
  seeded, person-answered form holds on the second clip.

**The second clip** is `thinkaboutthings_compressed`, chosen to be hard: people in matching tops, many cuts, cutaways
to other people while the group is away, a shot seen from behind, a missed cut, coloured light, sharp modern video
([`../../bench/results/2026-10-04_subject_track_three_clips.md`](../../bench/results/2026-10-04_subject_track_three_clips.md)).
The named subjects there are two people at the back of the group, run three ways: the others confirmed as bystanders,
the others unknown, and all six named. Neither clip has far more people than names with the detector at its cap
([`../../bench/results/2026-10-07_subject_track_under_nudge.md`](../../bench/results/2026-10-07_subject_track_under_nudge.md));
which clip tests that is the owner's word.

## Failures still open

What each of these would show before a render, and what is not covered. The design's other failure cases named parts
that were removed and have nothing to act through.

| failure | the signal before a render | what covers it | left |
|---|---|---|---|
| a bystander taken as a named subject who is away | the scene's people no answer has named; the count deciding | every take not a person's word asks | asked only where the gate is given the who-is-who file |
| too many questions, answered carelessly | the count of shots asking | questions by scene, not shot, once the casting pass exists | a person who says yes without looking |
| a wrong look confirmed, and later shots lean on it | a confirmed look more like another subject's than its own | only confirmed looks seed; a removal marks what leaned on it | the first tile answered for the wrong person |
| a subject so small the signature is noise | the mask's feature cells and box height (`bench/capture_masked_run.py::flag_face_size`) | the row asks | a small subject among small bystanders |
| two people swap names inside a shot | a name more like the other's previous look; two boxes that met | the shot split there | the tracker's looks are `subject_track.py::PROBE_STRIDE` apart |
| stored signatures no longer comparable | unlike fingerprints | compare like with like, or sign again, or ask | a difference the fingerprint does not name |
| a session answers for the owner and is wrong | the share of answers by a session | the file keeps who answered | a session does not know the people |
| two people alike in every clue | look-alike pairs when a sheet is answered | every shot that could hold both asks | all of it, if a person cannot tell them apart either |
| a cut the finder misses | cut scores just under the line; the count or the plate changing | showing the confirmed look again | a cut between two angles of one room with the same people in place |
| two sessions write the cast at once | the lock's holder | one writer, append only | an editor or a script ignores the lock |
| a thing handed to the wrong subject | more than one candidate near a subject | several alike ask | the whole of "things" is untried |
