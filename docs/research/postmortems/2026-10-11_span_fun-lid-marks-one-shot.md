---
mode: span
scope: fun-lid-marks-one-shot
date: 2026-10-11
range: 2026-10-11 (commits 90e85a87..2fe62983)
summary: The way of working held (every render through the builder, two looks by a session that did not make the load, and every correction of substance from a reader given files and no conclusions), and the day's one good result was still read further than it went, told to the owner from a whole-shot average, written up and carried to a second shot that changed several things at once, so its failure there cannot be placed; the ideas that slipped were small arms that make a result readable, most of them named as "next" more than once.
evidence_commit: 2fe62983
artifacts:
  - docs/wiki/handover.md
  - docs/wiki/decisions.md
  - docs/wiki/masked_v2v.md
  - docs/wiki/meta_perception_models.md
  - bench/results/2026-10-11_close_render_made_again.md
  - bench/results/2026-10-11_kitchen_shot_head_and_eyes_two_tests.md
  - bench/results/2026-10-11_kitchen_shot_lid_marks_on_the_body_model.md
  - bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md
  - bench/results/2026-10-11_sapiens2_pose_first_frames.md
  - bench/results/2026-10-11_embeddinggemma2_who_is_who_first_crops.md
  - bench/masked_job.py
  - bench/capture_masked_run.py
  - bench/draw_lid_marks.py
  - bench/read_render_against_original.py
  - workflows/h3_config.py
  - docs/research/postmortems/2026-10-10_span_fun-three-people.md
  - commit 255ccc5e
  - commit 04f0789d
  - commit 12e460a1
  - commit 1ee23440
  - commit f411239c
  - commit 46aa8a9b
  - commit 08fa89a2
  - transcript mrhop
  - transcript mrice
  - transcript mrfire
  - transcript mrcolor
  - transcript mrtie
  - transcript mrdoc
---

# Postmortem: 2026-10-11, one shot read as a result, six sessions

> **Written with the session transcripts.** A session that took no part in the day (mrmuse) wrote this the same evening from the transcripts of the six sessions that worked the masked lane, the owner's saved answers on the local page, the day's commits, `docs/wiki/decisions.md`, the records under `bench/results/` and the H3 Open List. The evening's account by the session that held the build is `docs/wiki/handover.md`; this page checks it and does not build on it. Where this page and a transcript row disagree, the row is right: add a dated annotation under the finding and leave the finding as written.

## The short version

- **The way of working held, and it is the day's best news.** Every render went through the builder with a capture of its own and two recorded looks by a session that did not make the load. No graph posted by hand and no commit carrying another session's lines was found. Every correction of substance came from one thing: a reader given the files and not the conclusions, who wrote before reading anyone else (mrice in the morning, mrtie in the evening). Section 4.
- **The day's one good result was read further than it went, the same evening.** Marks of the original's eye openings, drawn on the body-model video, lowered the second subject's lids to the original's level on one shot. That was told to the owner as the marks working and her eyes matching the original's, from a whole-shot average, while a blind reader's split of the same shot (no render opened her eyes where the original's are open) was already in a peer's inbox. The owner's word to keep moving through the scenes came five minutes before the correction. Within the hour the form was written into the wiki, three tools were assigned, and it was tried on a second shot, where it did nothing for her eyes (sections 1.1 and 1.2).
- **The second shot cannot say why.** It changed several things at once, one of them a rule for the size of the marks that had never been rendered. The one-change repeat was named only after the null came back (section 1.3).
- **The ideas that slipped were small arms that make a result readable.** A second seed. The same load with one thing changed. The vendor guide's form of the reference text on the large marks. A zoomed load that keeps its motion video. Two more frames before the eye figure was trusted. A fairer pair already on the share. Most were raised early by the session nearest the data or by the owner, and most were named as "next" more than once (section 2).
- **The owner's rulings on wording arrived as runs of short messages across up to three terminals.** Three things were built on the first part of a run and withdrawn by its end; none was rendered. Where the owner answered on a card, a queue that had sat for most of an hour was answered in about a minute (sections 1.4 and 6).
- **Yesterday's nine, a day later:** two done, one half done, and the rest not built or not touched. Item 1's "pass the text itself, never a description of it" was broken three times, once in yesterday's exact form (section 7).

## What this is built from

**The transcripts.** Six sessions: mrhop (the morning's orientation, the Open List page, a fresh run of the opening frames by hand; first row 13:44Z, last 17:26Z), mrice (an independent second reader, a local page for the owner's answers, the perception-models page; 14:41Z to 18:07Z), mrfire (the build, the card and every render after midday; from 16:12Z, and the evening's handover), mrcolor (the owner's local page, then second reader of every render; from 16:35Z), mrtie (a blind reader who had not lived the day, then three tools; from 20:43Z) and mrdoc (the wiki; from 22:01Z).

**How they were read.** Every row the owner typed (those typed mid-turn included), every message between sessions and every shell command were pulled out by raw line number with the script kept from the previous day's postmortem. I read all of the owner's rows to all six sessions myself, and the whole of the owner's saved lines on the local page (`data/fun/poc/answers.jsonl`, untracked). Four readers then took a slice each: the morning; the build session's day; the second reader's day with the owner's saved lines; and the late sessions together with the traffic between sessions and the previous postmortem's nine forward items. Each was asked for faults, for proposals and warnings that were not acted on, for where everyone went the same way, and for the owner's side. The rows this page leans on hardest were then found again by me in the extracted rows. The four sessions still open were each asked the same seven questions (section 5).

**Not done.** No render was watched and no still was opened for this page; what a render shows is as a session measured it or the owner said it on playback. No text was read whole by me, which the previous day's postmortem did and this one should have: where this page says what a text contains, it is from a reader who opened the text file or from the tracked record. Most tool output was not read. A reader's "not found" means not found in the rows it read. The readers' rows stop at about 22:27Z; what came after is from the commits, the handover and the sessions' messages to me.

**Conventions.** Times are UTC with a Z; local time is five hours earlier, and commit times in `git log` are local. `mrfire L2567` is raw line 2567 of that session's transcript. `ans 228` is line 228 of the owner's saved answers. The people in the clip are the lead, the second subject and the third subject. "The kitchen shot" is source frames 14 to 154, the second subject; "the close-up" is source frames 1575 to 1729, the lead singing. An owner's instruction is given by its substance, its row and its time, never in their words.

## 1. What went wrong, and what would have shown it sooner

Each item says what happened, where it was first visible, and what made the mistake easy. None is answered by reminding sessions of something more often; the last line of each says what it is a flaw in.

### 1.1 A whole-shot figure was told to the owner as "following", three times in one evening

- **19:50Z, a correlation.** The build session told the owner her eyes do not move with the original's, and that this held with or without the mesh. The second reader's own read of the tables made the first a steady offset and found the second not shown on this shot; both were withdrawn at 20:08Z (mrfire L1737, L1890, L1912).
- **21:00Z, a median.** The build session told the owner, marked as its own reading, that her eyes had moved toward the original's in all three overlay renders, on a median over the shot and two stills (mrfire L2567, 21:00:02Z). Seventy-two seconds later the blind reader's first read reached it: no render raised the stretch where the original's eyes are open, and the render with the large marks is lower than the kept render there too (mrfire L2589, 21:01:14Z). At 21:03:19Z the second reader told the owner, from a mean over the shot, that the eye marks work and her eyes match the original's on average (mrcolor L3116). At 21:04Z the build session acknowledged the blind reader's point to it and, in its next message to the owner, said all three readers agree her lids are lower, without that point (mrfire L2642, L2659). The correction reached the build session at 21:14Z as "a level, and a direction followed weakly; not timing" (mrfire L2948) and the owner at 21:18:17Z, twenty seconds after the owner had written that they were leaving (mrcolor L3458).
- **21:25Z, the head.** "Her head still follows" was read from a correlation; the blind reader's pixel figures showed her head sitting a few pixels off. "Follows the turn less" was then read off three stills and told to the owner; the blind reader showed the turn is the same in every render and only the position differs. Withdrawn at 21:41Z (mrcolor L3568, L3672, L3832; mrtie L943, L1137).

*First visible:* in the per-frame column each time. For the eyes, in the split by where the original's eyes are open, which was in the build session's own table from the first minute (its own words in the handover, "What to carry forward").

*What made it easy:* the first figure a table gives is one number over the whole shot, and on a shot where the original's eyes are lowered most of the time, a lower level and following look the same in that number. Each reading went to the owner before the second reader's answer was in, and at 21:00Z to 21:04Z two sessions told the owner in two terminals. The build session's own standing rule is per-frame columns before a headline, so this is not a rule that was missing. It is a flaw in what the reading prints first, and in a result having two tellers. The first half has an answer as of the same night: the tool that reads a render against its original (`bench/read_render_against_original.py`, `940517fe`) prints the eye opening taken apart by where the original's eyes are open and shut, and its docstring says why.

*What it cost:* the owner's word to keep moving through the scenes (mrcolor L3279, 21:09:31Z) and the second reader's plan to apply the same form to the other shots (mrcolor L3297, 21:10:16Z) both came before the correction. The tracked record was written after it and says "as a level".

### 1.2 One shot became a written practice inside the hour that tested it

The order, from the rows, the commits and the remote's own log:

- 22:01Z: the owner asked that the day's form go into the config, the workflows, the nodes and the docs (mrcolor L4190 to L4204).
- 22:05Z: the build session held the builder and the generated graph until the form had held on two shots, and split the rest: three tools to one session, the wiki to another (mrfire L4044).
- 22:06Z: the close-up, the second shot, was queued.
- 22:08Z to 22:09Z: the owner asked for the practice to be written in the wiki now, with a warning that it had worked on one case and may not on others (mrcolor L4300 to L4335).
- 22:14:54Z: the practice section was committed (`255ccc5e`), opened by a "provisional" paragraph, after the blind reader had corrected six of its sentences. It went out in a push at 22:20:27Z (`git reflog show origin/main`).
- 22:20Z to 22:22Z: three sessions, each before seeing another's reading, read the close-up as no change in her eyes (mrfire L4161; mrcolor L4517).
- 22:23:31Z: a dated "did not repeat" note went on top of the section (`04f0789d`), written from one session's own recount of the tables without waiting for the second reader that had been asked for; a corrective commit followed within minutes (`b506a700`). At 22:23:55Z the first tool landed with the size rule as a constant marked as a starting value (`12e460a1`).

*What held:* no node and no generated workflow changed. The section's first paragraph had promised a dated note, so the note had a place. The record of the result had itself said, before any of this, that the form is accepted when it has held on a second shot.

*What made it easy:* the write-up was asked for while its second test was sampling, and, in the build session's own words at the close, nobody said "wait for the second shot, it is sampling now". The sessions did hold the workflow without putting that to the owner; nobody held the wiki. The cost was small because of the provisional paragraph: a note, one corrective commit and about six minutes in which the public page said the practice without the note.

### 1.3 The second shot changed several things at once, so its null cannot be placed

- **What differed from the kitchen load.** The tracked record lists five, none separated: the marks' size against the face; a shut eye drawn as a sliver; face only at a small margin against face and hair at a wide one; a sentence that she sings; another person, still and a longer window (`bench/results/2026-10-11_lead_closeup_lid_marks_did_not_carry.md`). The rows add the vocals stem as the load's audio, a body-model video from a new preview, and overrides entered by the load's maker (mrfire L3806). The load was matched to the 10th's render of the same frames, so against that render it changed fewer things (the documentation session's correction to its own first answer, section 5).
- **The size rule had no render under it.** The second reader proposed sizing the marks by the model's cell and not by the face, for a small street face (mrcolor L3456, 21:18:11Z). Its basis was one step between two renders of one shot, and the face height it quotes is a part-box height where the pose tables give about half that (reader's finding; mrcolor L3423 against L3098). The build session built it as a fixed size on the canvas, which on the close-up's larger face grew the marks less than the kitchen's factor (mrfire L3806). Neither look questioned the factor. The build session says it noticed the arithmetic on the small faces and did not turn it round for a large one, and said nothing because the rule had just replaced a worse one of its own.
- **Raised and not a hold.** The blind reader wrote at 21:39Z that the three agreeing renders are "one result seen three times" and that the next shot, or a second seed, is what would make it two (mrtie L1113). At 22:04Z, two minutes before the queue, it wrote that the load changes several things at once and that this did not ask anyone to hold the render (mrfire L4026).
- **The cheap check came after.** Whether a shut eye's thin mark is visible at all at the size the model is shown the reference was asked in this same form for the kitchen marks at 20:26Z, and acting on it produced the enlarged-marks load, the only one of the three that reached the original's level (mrcolor L2351). It was not asked again for the close-up. The marks tool, landed at 22:24Z, at once reported marks too thin to leave any white (mrfire L4212).

*First visible:* when the rule was written, by asking what render it rested on; and in the list of what differed, which both readers wrote before the render.

*What made it easy:* a shot that differs was chosen on purpose, as the first check that the result repeats (mrcolor L3297), and the owner had asked for other examples. "A different shot" and "a repeat" are two tests, and one render was asked to be both. Every arm also costs a new job, a new capture and two looks (section 7, item 4), so the cheap one-change arm was not cheap.

*Still open:* why it failed. The handover's writer leans to the marks' size and says it would not be surprised to be wrong; the second reader holds only that a rule went from one shot to the next with no render under it; the blind reader's one candidate is that a shut eye's mark is mostly its own black edge.

### 1.4 A ruling was contradicted six minutes later, by the session that then passed the text

- The owner ruled that the text is not to prompt for her eyes, and that it says what the reference is and that she follows it (mrcolor L2151, 20:17:06Z; L2212, 20:20:21Z; L2234).
- The build session's draft still named her eyes, and it asked the second reader, not the owner, whether that was the thing the owner had ruled out (mrfire L2182, 20:23:31Z).
- The second reader answered with a rewording, a relation in place of a behaviour (mrcolor L2351, 20:26:07Z), then passed the three texts at two looks (mrcolor L2390, L2479). Three renders ran.
- The blind reader found the clause by reading the three texts whole before opening any render, and put it to the owner (mrtie L293, 20:52:13Z). The owner ruled again: nothing about her eyes.

*What it cost:* each of the three overlay renders carried two changes against the kept render; one more render was needed to separate them, half an hour later, and it showed the clause was not what lowered her lids (mrfire L3372).

*What made it easy:* the reader who took the look had written the clause. And the owner had agreed to a description of two forms of text; the sentence itself was not shown to them. This is the previous day's item 1 again: a text travelled as a description.

Between 20:11Z and 21:08Z the wording changed every few minutes in one terminal and each change was relayed and built at once. A builder key for a softened start was written and reverted inside three minutes; two loads were built on the first part of a run of messages and never run (mrcolor L2115 to L2157; L2903 to L3121; L3224 to L3255). None of it was rendered, which is the gate and the two looks doing their work.

### 1.5 The first clip the owner saw was the day's worst, and its cause hardened without a test

In the morning the owner asked for a fresh run of the opening frames by hand, to find the blind spots. The first render, a whole-person replacement of the second subject on the kitchen shot, came out badly wrong (the owner's reaction, mrhop L1593, 15:54Z).

- **It differed from the 10th's close render in several things at once:** the node's own short text with no caption of the shot, a whole-person region about half the frame against face and hair, nothing of hers kept out, and the mesh tied to the whole body (mrhop L1690, L1697, L1713).
- **It was visible before the render** in the job file, typed under three minutes after the ask with no parts for her (mrhop L686, 14:59Z); in a row the second reader added to the owner's list two minutes later, that a fairer pair of whole person against face and hair was already on the share (Open List row x66, 15:01Z); in the prompt grader's length warning (mrhop L1018); and in the capture's own flags that the region is mostly not the subject and holds her hands (mrhop L1201). The owner's saved answer asking for face and hair only landed six minutes after the queue; the render was kept as a baseline and not cancelled (mrhop L1474).
- **The session did run the day's process:** a capture, the gate, two looks by another session, and predictions written first. Both readers wrote that the things on the table would be redrawn and so did not count it a finding; neither predicted what came (mrhop L1599).
- **The cause was never separated and its statement grew stronger.** At 15:59:05Z the render could not say which of three differences produced the result; fifty-two seconds later it came from the text and the large region together; by the handoff the short text had done it (mrhop L1713, L1730, L1924). The one render that would have tested the text alone was offered and not taken. A third suspect, the graph's start settings, was read by nobody and is in no handoff (mrice L992).

*What made it easy:* the job file asks for subjects, stills and whole-or-part before any picture exists. And a cost both readers expected read as acceptable.

### 1.6 Twenty-three minutes on "dots" that one still ended

The owner saw white dots on playback, and their first remark tied them to the reference's eye marks (mrcolor L2959, 20:55Z). Three sessions searched her eyes with three threshold tests and found nothing; two loads were built on theories of it, one rendered (mrfire L2659; mrcolor L3116; mrtie L552). The owner said not to dig into it and, in the same message, asked whether it came from the reference or the prompt; each session answered one half (mrcolor L3017; mrfire L2572). The blind reader asked which file the owner had played (mrfire L2813, 21:11Z). The owner sent a still, and it was a warm point on the side of her nose, present in every render since the day before; the owner closed it (mrcolor L3395, L3424 to L3426).

*What made it easy:* a thing seen on playback was described without a frame or a place, and every session looked where the first remark pointed. A threshold for near-white cannot see a warm highlight, and "found nothing" did not say what the test could not see.

### 1.7 Said from a name, a length, a refused write or a memory

- **"The pair is clean."** The clip with the singing sentence and the clip without it were told to the owner as differing only in that sentence, from audio, region and seed. The session had printed the three texts' lengths and not compared the texts; the pair the owner had been pointed at differs by two sentences. The second reader caught it within five minutes by reading the graphs saved in the renders' own picture files (mrfire L1258; mrice L1614).
- **A list write cited before its result was read, twice.** Two writes to the Open List were refused by the store because another session held the row, and were reported to peers and written into a handoff as done, in the same message as the write (mrhop L1963 to L1966, L2255 to L2257). The receiver caught both by reading the row. The store's version guard is what kept anything from being overwritten.
- **A script named as the maker of two masks from its file name; a builder limit read from a docstring.** Both caught by the receiving session opening the script and doing the thing (mrhop L2003).
- **"Blank" and "faceless" for the body model's face,** said to the owner by two sessions, with one render reasoned from it, before any frame of the motion video was opened. It is a modelled neutral face (mrfire L2182; mrcolor L2005).
- **Times typed from memory, by three sessions.** Headings and list rows ran up to half an hour ahead of the clock (mrhop L1810, L2254; mrcolor L2207; mrdoc L448). Each caught it by reading the clock. The third had read the second's note about the same slip that afternoon.
- **After a context cut, an order of events was told the wrong way round.** The second reader told the owner and a peer that its wording about her eyes came before the ruling; the rows have it six minutes after (mrcolor L2794, L2809 against L2212, L2351). Its own later written account has the order right.
- **Marked as answered, with no row that answers.** Five questions the owner queued in one terminal between 19:20Z and 19:27Z are "answered" in that session's notes; a reader scanned every row to the owner in that stretch and found none that answers them. The protocol went into the page instead, and the owner then asked for help telling the clips apart (mrcolor L1219 to L1481, L1773).

*What made it easy, for all of these:* the cheap observable was one command away each time (compare the texts, read the write's result, open the script, open a frame, run `date -u`) and the sentence was written first.

### 1.8 A printed line read past, and flags that travelled as counts

"Emptied by the grade, by design" printed three times in front of the morning's session for one load, and the gate read clear; the second reader decoded the file the load actually reads and found the part mask empty on half the load's frames (mrhop L1164; mrice L696). Three makings of that one load passed `verify` and the gate, and all three were stopped by that reader before anything sampled. The floor that emptied the frames is `PART_LEAST_PX` in `bench/capture_masked_run.py`, a constant reasoned on another clip, on a stretch whose accepted render of the day before used those very frames. It is unchanged. `docs/wiki/known_before_a_render.md` names the constant; that it emptied half of this load is in one private handoff.

After the first capture of the afternoon, flags went round as "gate clear, N iffy". Of the day's captures that led to a render, one has an outcome entered for most of its flags and the rest have none, or only their maker's overrides (the `outcomes.json` beside each capture under `data/2026-10-11_*`).

*What it is a flaw in:* the same as the previous day. A warning list longer than anyone reads is summarised as a count, and a count cannot be answered.

### 1.9 A correction reached the newest place and not always the place it is read from

- The six sentences the blind reader corrected in the wiki section before its commit are still, five of them, in the tracked record that section cites for every figure and caveat (the kitchen lid-marks record; reader's finding from the file as it stood at 22:35Z).
- The same record keeps a mouth column with no note that the mouth figure later failed its trial (`08fa89a2` holds the verdict).
- The eye band committed as `POSE_READING` in `workflows/h3_config.py` was found, three minutes after its commit, not to carry from an original to a render (mrtie L1757); the handover says so and the constant's comment does not.
- The second reader's recorded looks on the overlay loads still describe the eyes clause as passed.

None of these is wrong in the place a careful reader lands first. Each is a second place for a fact, which the owner's standing rule of 2026-10-10 says to avoid.

## 2. Raised, and not acted on

One row each. The last column is what the rest of the day showed, and "cannot tell" is an answer. This is the section to read if only one is read.

| The idea or warning | Raised by | What happened to it | Did the day show it was right |
|---|---|---|---|
| A fairer pair for whole person against face and hair is already on the share; a whole-person region is what redraws the table | mrice, Open List row x66, 15:01Z, and to the owner | No answer. The whole-person load was queued 35 minutes later | Yes. The owner pointed at that face-and-hair render as the close one within the hour and ruled out whole-person renders of her (mrhop L1671, L2139) |
| Two more frames, a blink and a half-open mouth, before the pose model's eye figure is trusted | mrice, mrfire L1086, 17:12Z | Agreed on the list row. Never run. The figure became the yardstick for every render from 19:49Z | Yes. Judged at 21:58Z: right at its two ends and not readable between them; the mouth figure fails (mrtie L1234) |
| "Is that the thing the owner ruled out?" of the eyes clause | mrfire to mrcolor, L2182, 20:23Z | Reworded by the peer; not put to the owner | Yes (section 1.4) |
| A second seed of the picked render: three agreeing renders are one result seen three times | mrtie L1113, 21:39Z; also mrcolor and mrdoc | Written into the record as a caution. Never proposed as a render. It is the previous day's item 5 again | Cannot tell yet. Nothing rendered so far has a second seed |
| This load changes several things at once | mrtie, mrfire L4026, 22:04Z | Said as not a hold; rendered two minutes later | Yes: the null cannot be placed (section 1.3) |
| The vendor guide's own form of the reference text, on the large marks | Found by mrcolor, L2263, 20:21Z. The owner said twice that a reference is used only if it is prompted as the vendor guide says (mrcolor L2100, L3036) | Rendered on the small marks only, where the pose measure and the owner's eye both put it with the good ones, while the owner said of the pack's form at the same marks that her eyes and mouth do not move (ans 240 to 244). Named as the next single change three times (mrcolor L3154, L4107, L4112). When the plain form then fell short on the close-up, the render proposed was larger marks | Cannot tell. Two kinds of evidence on one shot point at it, and the example now in the wiki is the pack's form |
| A zoomed load that keeps its motion video | The owner called the zoomed render the better one on playback twice (mrhop L2201, 17:19Z; ans 183, 19:41Z). mrfire found at 21:13Z that a zoomed load refuses a motion video today | Put back five times in the build session's rows, each time as its own later card; taken off the owner's card at 22:00Z to keep the card to one decision (mrcolor L4150). The build session remembers it as deferred twice | Yes for the lead's street faces: the pose pass measured their lids at about a pixel apart at whole-frame size (mrfire L3442). Cannot tell for the close-up. It never reached the owner as a decision |
| Is a shut eye's thin mark visible at the size the model is shown the reference? | Asked for the kitchen marks by mrcolor, L2351, 20:26Z, and acted on then | Not asked again for the close-up, where a shut eye's mark is a sliver | Likely. The marks tool reported marks too thin to leave any white, after the render. It is the handover's first no-render step |
| Add singing to every prompt and see what happens | The owner, mrhop L2101, 16:37Z | Narrowed to a card of four loads; the owner clicked it at 19:16Z; never run. It waits on a per-video record that was not started | The owner's playback verdict the same day was that the sentence is wanted where she sings (mrhop L2201, L2219). The test of where it holds is still owed |
| Try the perception models on disk; this keeps being put off | The owner, three times (mrhop L1928; mrice L1595, L1862) | A plan of seven trials written before any was run; the owner's go at 19:15Z. The first is judged. The second stopped at a loading error. The owner's assent to a session's question, whether to write a table onto the list, had come back to them within the hour as their ruling that the depth models wait (mrhop L1971; mrice L1626) | Cannot tell for the other six. The first one's verdict changed what the day's measure could be trusted for |
| The form into the config, the workflows and the nodes | The owner, 22:01Z and again 22:26Z (mrcolor L4190, L4626) | Held by the sessions until two shots agree, and recorded in `decisions.md` as decided by the sessions and not objected to. The owner was not asked | The hold was right: the second shot failed twenty minutes later. What is not answered is the second ask, that the code and tooling be left in a good spot; the handover's "Where things are" is the answer given |
| No standing second reader once mrice left; someone has to be named | mrcolor to the owner, L1092, 18:08Z | No answer. The second reader then helped write what it read (section 1.4) | Likely. A reader who had not followed the day arrived at 20:44Z and corrected four readings in ninety minutes |
| A rough page gets kept, with no check and no owner; promoting it should be decided on purpose | mrhop to the owner, L480, 14:44Z | The same session picked that page as the one page two hours later, and the owner agreed in a word. Whether it is tracked, and whether it moves to the share, were asked twice and are unanswered (Open List row t72) | It was kept. It is where every decision of the evening was made, it is untracked, and its "start here" table is typed by hand |
| The storyboard planner and debugger the owner asked for at 14:12Z | The owner, mrhop L226 | Passed through three sessions with no design page. What exists is a review-and-answer page for frames 0 to 603; it plans nothing and reads no per-video record. One app or two is not ruled | Cannot tell. The owner said of the layered view that it gives full visibility (ans 129) |
| Nothing should be planned from the overnight preview's masks: one tracker found nobody | mrice to the owner, L481, 15:05Z; list row x63 | No answer. Neither handoff says it | The owner answered that preview's numbered pictures at 17:46Z and noted subjects not masked |
| The one-change variant of a built load | The owner on the 10th; mrice's list row x65, 15:01Z | Not built (`bench/masked_job.py`, its own "not here yet"). No hand-posted graph was found; every arm was a new job, a capture and two looks | The honest route was taken every time and it was dear. It is part of why the second shot was asked to be two tests at once |
| The floor that empties a small part mask is a constant from another clip | mrhop and mrice, 15:29Z | Worked round with an option; the constant is unchanged | Yes in substance: it emptied half of a load whose render of the day before had been accepted. `AGENTS.md` already says such a value is set from the input video |
| mrwolf's queued readings from the 10th | The previous postmortem, item 9; list row n43 | Neither read nor dropped, a second day | Cannot tell |
| The owner's ten points on the joined clip from the morning | The owner, ans 19 to 46 | No render of the day addressed them; every render was of one kitchen shot or one close-up | They are the delivery's actual list |

Read down the second column. As on the day before, the idea usually came from the session nearest the data or from the owner, and it usually came before the render it would have informed. What differed from the day before: nothing here was overridden in haste. Each was deferred politely, as "the next single change", "its own card" or "not a hold", and a deferral that sounds tidy gets asked about less than a refusal.

## 3. Where everyone went the same way

- **Stay on the kitchen shot.** After "one shot, one seed" was first written between sessions (mrcolor L1828, 19:54Z), seven more renders of the same frames ran and two more were built and not run. The session that wrote the caution most often also proposed or passed each of them, and says each next render answered a question the owner had just asked. The dissent was the owner's: other examples, or the day overindexes (mrcolor L3463, 21:19Z).
- **The pose model's eye figure as the yardstick.** All three readers used it or a relative of it. Dissent: mrice at 17:12Z (section 2), and the blind reader's cautions on the line drawn between the owner's good and not-good clips: drawn after the answers were seen, on one shot, and "not good" was not always about her eyes (mrtie L1263).
- **The 10th's standing sentences held constant in every text, so that one thing changes at a time.** The one render that removed the edit sentences read as the same piece (mrfire L2188). They stayed anyway in every later text, in the picked render's text that the wiki now shows as its example, and in the close-up's, which also holds a sound line calling a sung shot quiet and a sentence true of only part of its window. All three readers flagged the last before the render; it was kept on purpose and is listed as a debt.
- **The next shot should differ.** No dissent before it ran (section 1.3).
- **The size rule for the marks.** No question at either look.
- **Reproduce the 10th's close render first.** All three morning sessions. It was run on the 10th's own two mask files, whose maker nobody found, so it showed that today's builder and node code make the same piece from those masks, and not that today's tools can make the masks.
- **A sentence first, the overlay only if needed** (mrfire and mrcolor, 20:11Z). The owner turned it round in the same minute: a sentence does not scale to other scenes, and a text that prompts for an expression cannot be told from the reference working.
- **A result goes to the owner when its maker has read it.** Nobody proposed waiting for the second reader's answer before the first telling.

## 4. What worked

- **A reader given files and no conclusions, who writes before reading anyone else.** In the morning mrice decoded the file a load actually reads and stopped three makings that all read "gate: clear"; it caught the pair that was not clean, the lip points sitting above the lips, and eight things in the design. In the evening mrtie, who had not lived the day, caught the eyes clause from the inputs, the level that was not following, the head's position against its turn, the mouth figure, six over-said sentences in the wiki before they were committed and three more in the note. The form it used is worth copying: the correction, the replacement sentence, and what the figure covers. Every session asked named this as the thing to copy.
- **A floor, asked for before the first reading.** The blind reader asked for the same measure on the 10th's render of the same seed, so that a small difference could be told from none (mrtie L220, 20:48Z). Every later "beyond the floor" rests on that ask.
- **What a render is expected to show, written before it samples.** It is what made the close-up readable as a null within minutes by three sessions, each before seeing another's reading.
- **A doubt about what the model can see, answered with one more arm before the pair was spent.** The second reader measured the marks in the file, found them inside one cell of the video model, and the enlarged-marks load was added (mrcolor L2351). It is the only arm of the three that reached the original's level.
- **Not building on one shot where it would have been dear.** The builder and the generated graph were held until two shots agree, three minutes after the owner asked for workflows. The null cost a note and not a node change.
- **A click held against the words beside it.** Twice a button and the owner's words disagreed, and the session asked before anything was built on either (mrcolor L972; L4150). The owner's rule that typed words count over a button came out of the first.
- **Everything through the builder, a new job name for every variant, superseded loads left built and unqueued** with a note of the one change. No hand post and no carried line was found, and the builder's refusal beside a peer's uncommitted config file was respected and not overridden (mrtie L1621).
- **The numbered picture, not the table.** The first preview's tables read right; the pictures showed a tracker on the wrong person and on a figure printed on a box, before anything was planned (mrhop L890).
- **Receivers who checked their handoffs** by opening the script and listing the folder the handoff named (mrhop L2003, L2120).
- **One card a decision.** Design rows too long to answer had kept the card idle from 17:27Z to 19:18Z (mrfire L1360). Cut into cards, they were answered in about a minute, and the owner's pass over eleven kitchen clips took six.
- **A cheap negative result with its bar written first.** The likeness trial with a new embedding model did not pass the bar set before it ran; the record says so, and the lane rests.
- **Saying a slip at once.** As on the day before, nearly every withdrawn claim in the rows was withdrawn by the session that made it, usually within minutes of the reader's note.

## 5. What the sessions said when asked, against the rows

The four sessions still open were asked the same seven questions, to answer from memory without re-reading. I sent them before the day had closed; three had answered by the time the owner's word to wait reached them, so those three answers are from about 22:25Z, kept as given, and each session was asked at the close only what it would now answer differently. The fourth, the blind reader, had not answered when this was written. The two morning sessions had closed; their accounts are their notes and handoffs.

| Session | An idea that was right and came late | What it went along with while doubting | Against the rows |
|---|---|---|---|
| mrfire | The zoomed load with its motion video; nobody refused it | The size rule for the marks; it said nothing because the rule had just replaced a worse one of its own | Lines up, and the rows are harder on the first than it is: put back five times and taken off the owner's card once, where it remembers twice. It remembers the wiki write-up as about an hour before the close-up; the commits and rows put them minutes apart. At the close it holds the size explanation more weakly |
| mrcolor | The vendor guide's form on the large marks; it named it as next and let it rest | Staying on the one kitchen shot; each next render was a tidy single change that answered the owner's last question | Lines up (section 2). Its written account of its own slips is accurate on eleven and leaves out four the rows show: the size rule, which it named to me as the day's biggest miss a minute and a half after writing that account; the questions marked answered; the order told wrongly after the context cut; and the zoom taken off the card. At the close it holds only that a rule went from one shot to the next with no render under it |
| mrdoc | A repeat that changes one thing | That the example text still carries sentences the owner ruled out on the 10th; it said so in the page and not to the owner, because the ask reached it through a relay | Lines up. Its "already pushed" is confirmed by the remote's log. At the close it corrected its first answer: the close-up was matched to the 10th's render of the same frames, so the one-change render is the build's own next step; and it added that it wrote the "did not repeat" note on its own recount without waiting |

Asked for the day's biggest miss, the three named one fault from three seats: that the level was said late and softly (the session that read it first), that a rule went to a second shot with no render under it (the session that proposed the rule), and that one shot was read as a result (the session that wrote it up). Asked what they built while suspecting it was not right, each named something it had labelled as a debt at the time: the close-up's text, the second likeness run, the practice section. Asked what the owner could do differently, each answered from its own distance from the owner's terminal (section 6). All three read the handover at their start. Two are unsure whether they read the previous postmortem and one is sure it did not; the rows show one of the three did. What they say changed their conduct came through the rules written out of the 10th into `AGENTS.md` and the memory folder, not through the page itself.

## 6. The owner's side

The owner asked what they themselves can learn from the day. This is from the rows, plainly, with what made things cheap first.

**What made things cheap.**

- **A still with a place on it.** One still ended a search three sessions had run for twenty-three minutes.
- **Answering on the page, one card a decision, and one table for one shot with a pick.** The last form, "start here" for the kitchen shot, is the one the owner went through fastest, and their verdicts on eleven clips sorted the same way as the measure.
- **Saying to try other examples.** It is why the form was tested before it was built into a workflow, and no session had said it.
- **Starting a reader who had not followed the day.** Four over-readings were caught in its first ninety minutes.
- **Asking a session, before it left, whether anything had been left open or vague with it.** That question is what completed the morning's handoff: the list it produced was not in the handoff file (mrhop L2032, L2046).
- **Marking an answer on the session's own paragraph.** Cheap and exact.

**What made things dear.**

- **A ruling typed as a run of short messages, while a session answers and builds between them.** Every ruling on wording from 20:11Z on came this way, in up to three terminals. Three things were built on the first part of a run. A session can hold a relayed wording until it is saved on the page or said to be complete; the owner can say that they are thinking and that it is not yet a go. The one go that cost nothing was a single numbered line answering two cards, saved to the page's file by the session that took it down.
- **A decision that reached the session doing the work second or third hand.** The build session held a render on a go that came as a click, a second click, a question and then a line typed to another session. The documentation session's task reached it in four pieces in eight minutes, none from the owner, and its one question to the owner was overtaken before it was answered.
- **A thing seen on playback, described without a frame or a place.** Section 1.6. The review clips now burn in frame numbers so that this is one screenshot.
- **Sharing one reader's result with another reader who was asked to be blind.** Twice a message meant to help ended a blind read (mrfire L1742; mrcolor L3160). The readers said so each time.
- **An assent answers the question as it was asked.** Agreeing that a session should write a table onto the list had become a ruling within the hour (section 2, the models row). When a session's paragraph is pasted back with a mark, most of that row is the session's words.
- **A verdict on playback moved as better clips appeared.** The same render of the kitchen shot was judged close in the morning, the best at 20:07Z and wrong at 21:55Z, and the owner asked whether they had marked it good before (ans 18, 218, 251, 252). Nothing is wrong with that; a session should record an early "good" as the best so far and say what it was the best of.
- **Asking for the write-up and the workflows while the second test was on the card.** Section 1.2.
- **Saved answers that nobody was told had arrived.** Twice in the morning the owner said in a terminal that they had answered on a page before the session had read it, and once pasted a whole review a second time. Every save lands in a file at once; nothing tells a session that a line has come.
- **The session best placed to build the app was given the fresh run by hand and two handoffs.** The app reached its third session with no design page.

**Said more than once before it held:** not to prompt for her eyes (three times in thirty-five minutes); not to dig into the dots; clips already judged moved out of the way; face and hair only for the second subject; and that the models on disk be tried.

## 7. Yesterday's nine, a day later

The previous postmortem ended on nine things to pick up. Each also has a dated annotation in that page.

| # | Yesterday's item | Today | Row or file |
|---|---|---|---|
| 1 | The builder prints a load's whole text; the text itself travels between sessions | Done in the tool. Texts were read whole before every plan by a session that did not build the load. A text or a sentence still travelled as a description three times: the eyes clause shown to the owner as two "forms", "the singing sentence" that is three things, and three texts' lengths compared in place of the texts | `bench/masked_job.py`, `text_shown`; sections 1.4 and 1.7 |
| 2 | One built load read end to end by a session that did not build it | Done, on the first load and on every load after it. It stopped a load whose mask was empty on half its frames. It did not predict the morning's failed render | mrice L696, L808 |
| 3 | Who is in each shot | Half done. The owner answered the numbered pictures for frames 0 to 603 on the page; the answers are raw lines and are in no cast file. The owner's question whether it matters that a subject's number changes between clips has no answer in the rows | mrcolor L1709; ans 133 |
| 4 | The one-change variant of a built load | Not built. No hand post was found: every arm went the long way round | section 2 |
| 5 | Three renders that each decide something, and first a reading of why the morning's mouth opened | Not done. No second seed of anything all day. The reading is still an open row on the list | Open List rows n41, n42 |
| 6 | A flag is answered or it is not printed | Contradicted once in the morning, then flags travelled as counts | section 1.8 |
| 7 | A check's fixture is cut from a real job file | Not done. The day's new checks hold the tools to a face made of points known by construction; the real-file proof was done by hand and reported in a message. The made fixture did catch a real drawing fault | `bench/_lib/face_points_fixture.py`; mrtie L1684 |
| 8 | The commit helper | Not built. No carried line was found; fewer hands were in the tree at once | reader's search of the rows and shell logs; the commits were not read for this |
| 9 | mrwolf's queued readings | Neither read nor dropped | Open List row n43 |

Of the six sessions, the three that started before 16:30Z read the previous postmortem whole; the three that started after were pointed at the handover and the handoffs and did not, and two of those three were the day's readers.

## 8. Where the handover is right, wrong or short

`docs/wiki/handover.md` was written at 22:29Z by the session that held the build, whose memory had been cut once that evening, and it says so. It is right in outline and candid: its first section says the day's good result was read too far the same evening, by its writer first of all. Each of the following has a dated line in that page.

1. **Who read the close-up.** As first written, three sessions; there were four readings. Its writer corrected it within minutes (`46aa8a9b`).
2. **"Every step is a tracked tool."** Step 7, the pose model on the original and on the render, ran on one-off scripts all day. The page says so further down; the step does not.
3. **"A flag overridden is a person's statement."** True of the overrides. After the afternoon's first capture, no capture that led to a render has an outcome entered for its flags beyond its maker's overrides (section 1.8).
4. **"Each was read first hand in the page's file before anything was queued on it."** True of what was queued. Three things were built on the first part of a run of relayed messages and then withdrawn (section 1.4).
5. **The morning is thinner than the rows.** Not in the page: the morning's failed whole-person render and that its cause was never separated; the floor that emptied half a load's part mask; the warning that the overnight preview's masks should not be planned from; the owner's ten points on the joined clip, which no render of the day addressed.
6. **The zoomed load with a motion video** is the third thing down in tomorrow's order, after two other single changes. The rows say the owner has twice called the zoomed render the better one and has not been asked (section 2).
7. **The vendor guide's form of the reference text** is not in tomorrow's order at all (section 2).
8. **A second seed** is not in tomorrow's order. The page's own "what the writer is not certain about" says three renders agreed on one seed.

## 9. What to pick up next

Small, and each can be marked done or wrong. The owner decides the order; the first three cost no render.

1. **Read a render from the split, and give a result one teller.** The tool that reads a render against its original (`bench/read_render_against_original.py`, landed the same night) already takes the eye opening apart by what the original does and gives the head in pixels. What is left is to use it as the first read of the next render, and for the session that holds the build to tell the owner a result once, after the second reader's answer is in. *Done when* the next result's first telling carries the split. *Wrong premise if* the next over-reading comes from a split that was printed and read past.
2. **Put the record right where the wiki points at it.** Five sentences in the kitchen lid-marks record, the mouth column's note, the comment on `POSE_READING`, and the two recorded looks that passed the eyes clause (section 1.9). A quarter of an hour, by the session that wrote each.
3. **The handover's own first no-render look:** what of the close-up's reference the model is actually shown, at the size the video model gets it.
4. **Then renders that each change one thing on a load that already has a reading,** in an order the owner picks: the close-up with the marks at the kitchen's factor (the handover's choice); a second seed of the picked kitchen render; the vendor guide's form on the large marks. *Each is done when* its expectation was written first and its reading is split as in item 1.
5. **Put the zoomed load that keeps its motion video to the owner as one card,** with what it costs in node code and a restart. It has been its own later card five times.
6. **Run what the owner already clicked:** the singing sentence on four of the lead's loads, and the still cut to head and hair. If the per-video record they wait on is not coming this week, say so on the card and run them on the old one.
7. **When a ruling covers a sentence, show the owner the sentence.** One card with the candidate wording on it, edited there, and nothing built on a relayed wording until it is saved on the page. *Refuted if* the next withdrawn load was built from a saved card.
8. **Something that tells the session holding the page that a line has been saved.** A watch on `data/fun/poc/answers.jsonl` by that session is enough. *Done when* the owner no longer has to say in a terminal that they have answered.
9. **Two old items, decided in one line each:** mrwolf's queued readings (read or drop; two days unread says drop), and the floor `PART_LEAST_PX` (set from this clip, or marked as a starting value with the tool saying when a clip's own sizes disagree, as `AGENTS.md` already asks).

For whoever starts a session after midday: the handover points at the day's postmortem, and the brief pasted to a new session can name it too. Today the two sessions whose whole job was reading had not read the last one.

Two habits found more than any rule again today, and one is new. Hand the real files to a reader who has not been told the answer. Write what a result should be before reading it. And ask of any new rule or size, before the render that leans on it: what render does this rest on?

## 10. Where the sessions agreed, where they split, and where each saw it from its own seat

From the three sessions' answers, their one-line changes at the close, and the two closed sessions' written accounts.

**Unanimous.**

- **What to copy.** All three named the same thing in nearly the same words: a second reader given files and no conclusions, writing before it reads anyone's. All three named the blind reader's work as what helped their own most.
- **Their own memory.** Both sessions whose context had been cut that evening said, unasked, that everything before the cut is their own summary. The third had been in the day for twenty-five minutes and had still typed a time from memory.
- **One shot was read as a result.** Three different sentences for it (section 5), no dissent.
- **Nobody refused the idea each thinks was right.** The zoom, the guide's form and the one-change repeat were each deferred for tidiness or for one change at a time, and each session says so of its own.
- **The owner's decisions cost most where they arrived in pieces or through another session.** Three seats, one answer.

**Split.**

- **Why the second shot failed.** The marks' size (the build session, more weakly at the close); cannot say, only that the rule had no render under it (the second reader); a shut eye's mark being mostly its own black edge (the blind reader's candidate, relayed); cannot say (the documentation session). Not decided, and no render separates them yet.
- **Whether the second shot was a fair test.** Six things changed at once (the documentation session's first answer); three against the render it was matched to, with the one-change render already next (its answer at the close, and the build's record). Both are true of different comparisons.
- **Whether to wait for a second reader before the "did not repeat" note.** One asked for the wait; one wrote the note at once because the page was already public; the blind reader then showed the note leaned on a recount for more than it supports. One corrective commit.
- **Which kitchen render to recommend.** The large marks with the longer text (the second reader, for the head) or the plain text (the build session, by measure). The owner picked the plain text on playback and said the candidates look alike.

**No agreement: each saw it from its own seat.**

- **The idea that came too late.** Three sessions, three ideas, each from its own work: the zoom, the guide's form, the one-change repeat.
- **What the owner could do differently.** Say a go once, in one place (the session that held a render on it). A still or a frame number at once, and one message a ruling (the session whose terminal the rulings were typed into). One line typed to a new session saying what it owns (the session whose task arrived third hand).

**Raised by one, contested by nobody.** That the example text in the wiki still carries the 10th's ruled-out sentences (mrdoc). That the measure's agreement with the owner's eye was drawn after the answers were seen (mrtie, on the page). That a reader who has followed the day all day is worth less than one who has not (mrcolor, of itself, when briefing the blind reader).

*The fourth session's answers, and anything a finding here gets wrong, go below as dated annotations.*

*Annotation 2026-10-11, 22:54Z, the same night: a finding made after the day closed, sent by the build session at the owner's ask.* The bench tools' shared fit of a source to the canvas, `bench/_lib/frames.py::FIT`, was not the fit the graphs' video loader uses when the source's shape differs from the canvas's. FIT scaled to cover and then cut the centre; the loader cuts to the canvas's shape and then scales (`ComfyUI-VideoHelperSuite`, `videohelpersuite/load_video_nodes.py`, the line under its comment about cropping to the new aspect ratio before the scale; read for this note, with the uncommitted correction in the tree). FIT's own comment said it had been read off that loader.

- **Who found it and how** (the build session's account, and its notes): the blind reader, at about 22:35Z, by the test it had set for its new tracked pose tool, which was to give exactly the points the day's one-off script gave. With FIT the points sat about half a pixel off, all up and down; with the one-off's fit, every point was identical. It then read the loader's code and measured two renders' own pixels outside their redrawn region against both fits.
- **What it touched:** every bench tool that reads the source through FIT (`git grep _lib.frames -- bench` lists them); in the capture tool, the source under every strip and stacked clip, and the readings of a render against its source, on every capture of this video made on 2026-10-10 and 2026-10-11. What it did to any flag: nobody has looked.
- **What it did not touch:** the pose tables, the eye marks and the figures in the records of 2026-10-11, which were made by one-off scripts that used the loader's order.
- **Where it was first visible:** at about 20:20Z the build session measured that a centre cut and then a scale registers with the renders and a plain stretch does not, used the right fit in its one-offs from then on, and did not hold it against the tracked FIT. The same pixel comparison with FIT as a third arm would have shown it. For two hours the right fit lived in untracked scripts and the wrong one in the tree.
- **What it adds to this page.** It is section 1.7 again: a constant's comment said "read off the loader", and the comment was believed over one more arm of a comparison that was already being run. And it is section 4 again: it came out because a session writing a tracked tool wrote down a number first (reproduce the one-off's points exactly) and would not accept half a pixel. It is also what the day's one-offs cost: the previous day's item 7, a check's fixture cut from real data, is the same idea.
- **Next:** the correction lands in FIT's one place with a check that holds its text to the loader's line (`bench/check_loader_fit.py`), by the session that found it. Then one look, on one capture of each day, at whether any flag or reading moves with the corrected fit. *Done when* that look is written beside the capture.

*Annotation 2026-10-11, 23:11Z, the same night: the fit correction landed, and the look this page asked for was made.* The correction is `100f9e09`, with `bench/check_loader_fit.py` holding the fit to the loader's own line. The build session then made the kitchen load's capture again on the corrected fit beside the one made earlier that day (`docs/wiki/handover.md`, "After the page was first committed"): the gate's flags are the same, every one; the reading that moves is `changed`, whose floor falls, so a multiple of that floor quoted before that night would read higher if made again. One capture of 2026-10-11 was read both ways; none of 2026-10-10. The "Next" line of the note above is done for the 11th.

*Annotation 2026-10-11, 23:11Z: the fourth session's answers (mrtie, the blind reader), given at its stopping point, set against the rows, and what they change in sections 5 and 10.*

| Asked | Its answer, in substance | Against the rows |
|---|---|---|
| An idea that was right and did not happen | A second seed. Every floor of the day was one render made twice at one seed, and each main statement of the evening rests on one seed. Nobody decided against it; it never became a card, and card time went to new wordings | Lines up (mrtie L1113; section 2) |
| What it went along with while doubting | Every text of the day carried the sentences the owner ruled out on the 10th, and a sound line calling a sung shot quiet. It wrote that into each reading as a note and not as an objection, because it was told an earlier render showed no effect of removing them and it had arrived an hour before | Lines up (mrfire L4026; section 3). The documentation session gave the same doubt and the same reason for not pressing it (section 5) |
| What it ran while suspecting it was not right | The search for the dots: two detectors built after its own first count had found nothing and nobody had said where on the picture they were | Lines up on substance. It remembers about forty minutes; the rows put the whole search at twenty-three and its own part at under fifteen (asked 21:04Z, closed 21:18Z) |
| The day's biggest miss | The bench tools' fit (the note above), the one miss it can speak to first hand. First catchable when other tools wrote the loader's order down correctly as copies of their own and nobody held the shared reader to them | The others had answered before this was found. One such copy is in `bench/cast_file.py` (found by search for this note; a second was not looked for) |
| What to copy | Each reader writing its reading, what it expected and what would count as a failure before opening anyone else's, with a measure that shares nothing with the others'; and an acceptance test set as a number beforehand | The same answer as the other three, with the second half added |
| What the owner could do differently | When something is seen on playback, say in the first line which file, where on the picture and on which clips | The same as the second reader's answer |
| The handover and the previous postmortem | Read the first part of the handover; did not read the postmortem. The handover's line that nothing in the lane is known good, the prompt least, is why it read every text whole against its frames before each render | Lines up (the first 140 lines of the handover; no read of the postmortem) |
| What it saw in its first ten minutes that the others no longer could | That against the kept render each overlay load changed two things together; that the marks were a few pixels tall on the canvas; that "follows" and "works" were being said of a level. And one thing it got wrong from the same distance: it called the enlarged marks one band, from four frames | Lines up (sections 1.1 and 1.4; its own correction seven minutes later, mrtie L437) |

*What this changes.* Section 10's "unanimous" on what to copy is now four of four, and all four name a slip of their own that was caught by reading what was actually written and not what was meant. "The idea that came too late" is four sessions and four ideas, the second seed being the fourth. On the day's biggest miss it is three and one: three named one shot read as a result, and the reader who arrived last named the fault found last. And a new pair: two of the four (the blind reader and the documentation session) doubted that every text still carried the 10th's ruled-out sentences, and both wrote it down as a note and did not put it to the owner; the example text now in the wiki carries them. That question is the owner's, and it is cheap to ask.

*Annotation 2026-10-11, the same night: a ruling made after this page was committed.* At the close the owner ruled that a figure is data given beside its context and not a verdict, and that green and red are for a hard failure (`docs/wiki/decisions.md`, 2026-10-11, the last entries; the scope over existing checks is open). Where this page repeats a record's "passes" or "fails" for the pose model's trial, that is the record's word of the day. The "done when" lines of section 9 are things a reader can tick, and none of them is a gate.

*Annotation 2026-10-11, 23:12Z, the same night: the sessions' close-outs, and one thing found in the last ten minutes.* At the owner's ask each of the four open sessions wrote what it did, what it did not get to, what it forgot (looked for on purpose) and the one thing the next session must not assume. They are kept as received, in private notes. What they add to this page:

- **None of the day's six sessions knew that the licence of the models the lane is built on had already been read.** The Sapiens2 licence was read on 2026-10-04, by a session no longer live, before the models were fetched, and the reading was recorded that day. On 2026-10-11 no session knew of it: the question came up in the last hour, when a session asked where a list of the pose model's point names should be kept, and the licence was opened as if for the first time. The models page says what is on disk and what loads each, and has no column for a model's licence; one such column, with who read it and when, is the view that would have carried the earlier reading forward. It is the pattern of the day before: the thing was known, and nothing put it where the next session looks. The second reader, asked at the close what it would now answer differently, named this as possibly the day's biggest miss; it believed then, as the other open sessions did, that the licence had never been read.
- **Forgotten, by each session's own search, where it bears on the owner:** whether the page's folder stays untracked was never cut onto a card, so the owner never saw it as a decision; the owner's question whether it matters that a subject's number changes between pictures was answered by a line on the page and never to them; bare renders were pointed at twice before a stacked clip existed; a question the documentation session put to the owner (should the pose pass and the marks be nodes) was never passed on; and the description of the reference video in the wiki was written without a frame of it having been opened by its writer, who has since looked at four.
- **Flag outcomes** (section 1.8): the build session confirms it and has not repaired it, on the ground that an outcome entered without looking at the render would be bookkeeping. Owed on the picked render's capture and the close-up's.
- **What each says not to assume:** that a figure on the page's "start here" table is computed (it is typed by hand); that the lid-marks section is a recipe; that a green check means the thing was exercised (every check of the capture tool, the builder and the delivery tool stayed green through the half-row fault, because no fixture is a source of another shape than its canvas); and that a figure near the original means the render is right.
- **One line of this page they correct.** Section 4 says every correction of substance came from a reader given files and no conclusions. The fit was found another way, by a tool held to a number set beforehand, and the licence came up through a question asked for another reason. Both are in the notes above.

*Annotation 2026-10-11, 23:14Z: the times on the four notes above.* As first committed, three of them carried a time typed from the writer's own sense of it, four to eight minutes ahead of the clock. They now carry the time of the commit that added each. It is the slip section 1.7 records of three other sessions that day, made by the session writing it up, and caught the same way: by reading the clock.
