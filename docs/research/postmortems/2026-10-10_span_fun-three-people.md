---
mode: span
scope: fun-three-people
date: 2026-10-10
range: 2026-10-10 (commits 4b1f49dd..dab2f38f)
summary: The day's costly faults were not unknown problems; in each one the owner, a session or a tool's own printed flag had already said the right thing and nothing made anyone answer it, the text a model was given was read last every time a render was wrong, and every real catch came from a second reader on real files or a number written down first, never from a check.
evidence_commit: dab2f38f
artifacts:
  - docs/wiki/decisions.md
  - bench/results/
  - docs/wiki/handover.md
  - masked_prompt_text.py
  - bench/results/2026-10-10_prompt_text_hardcoded_in_code.md
  - data/2026-10-10_fun_job_inputs/jobs/lead_face_street.txt
  - data/2026-10-10_fun_job_inputs/jobs/lead_face_night_phone.txt
  - bench/results/2026-10-10_preprocessing_data_read_by_a_second_session.md
  - bench/cast_file.py
  - bench/results/2026-10-10_who_is_who_across_shots.md
  - AGENTS.md
  - docs/checks.md
  - bench/results/2026-10-10_one_load_per_shot_against_one_long_load.md
  - data/2026-10-10_fun_delivery_tables/t_fun_0000_1919_a_build.txt
  - bench/masked_job.py
  - bench/capture_masked_run.py
  - bench/who_is_who_across_shots.py
  - bench/assemble_delivery.py
  - bench/check_assemble_delivery.py
  - bench/check_capture_masked_run.py
  - bench/check_masked_job.py
  - bench/
  - commit 47f42dec
  - commit 0d20e742
  - commit e2469ab7
  - commit ace202c4
  - commit 35e45d29
  - commit ff1568c9
  - commit 56e35121
  - commit 91af9d3d
  - commit 96f204f6
  - commit f874e895
  - commit 4f03d521
  - commit 768944d1
  - commit 5e56d3fe
  - commit 23b01495
  - commit 68347328
  - commit 4cbfaa10
  - commit 2bd4a3b3
  - commit dab2f38f
  - commit d300c55f
  - commit 7c902d53
  - commit e550c443
  - commit e9de8cc3
  - commit 368fbaa8
  - commit d46ece22
  - transcript mrcorn
  - transcript mrdeer
  - transcript mrfrog
  - transcript mrghost
  - transcript mrpop
  - transcript mrsheep
  - transcript mrwolf
---

# Postmortem: 2026-10-10, three people replaced in one clip, eight sessions

> **Written with the session transcripts.** A session that took no part in the day (mrscribe) wrote this the same evening from the transcripts of the eight sessions that worked the masked lane, the day's commits, `docs/wiki/decisions.md`, the records under `bench/results/` and the masking board. The lead session's own account is `docs/wiki/handover.md`; this page checks it and does not build on it. Where this page and a transcript row disagree, the row is right: add a dated annotation under the finding and leave the finding as written.

## The short version

- **The video is not finished, by the owner's ruling, and the day is to be judged by what the next job is made to build on** (mrpop L11401 21:56Z; L14764 00:16Z). It left a job builder, a capture tool with a gate, a tracker for several named subjects, a delivery assembler, and a set of removals. Whether those get used is the test, and by the end of the day three of the new tools had been run on real job data by any session other than the one that built them (section 6).
- **The faults that cost renders were not unknown problems.** For the short shot that faded back to the source, the head that did not turn, the shot left as the source in a locked file and a load that ran over another subject, the right cause or the right action had been written down beforehand by a session, said by the owner, or printed by a tool as a flag. In each case it was deferred, read as good news, or left in a list nobody had to answer (section 1).
- **When a render was wrong, the text the model was given was read last.** Four times in one day the first explanation was a signal (motion, size, the audio file, an attention setting), renders were spent on it, and the cause was a sentence in the text or a sentence the text lacked. No session said "read the text" before the owner did, and the other sessions knew the texts only from the lead session's descriptions of them (section 1.1).
- **Nothing that caught a real fault was a check.** Every catch before a viewer was one of three things: a second session reading or running the thing on real files without the author's conclusion; a number written down before a build that then did not match; a person reading the tracker's own tiles (section 4).
- **Asked directly, every session named a doubt it had not said aloud.** The reasons they give are alike: the rule had been asked for, the owner's word was explicit, or an objection from the session that had just missed something would read as an excuse. One question, asked by the lead session in the evening (keep, simplify or remove what you added today), is what brought them out (sections 3 and 5).
- **The handover is right in outline and wrong or short in particulars**, listed in section 8 with the rows.

## What this is built from

**The transcripts.** Eight sessions: mrpop (lead; first row 13:47Z, last 00:56Z), mrcorn (delivery assembler, cold reads), mrdeer (capture tool and gate), mrfrog (tracker, who is who, cast file, body pose), mrnemo (node work, latent steps, the cuts), mrsheep (job builder, analysis folder; from 20:09Z), mrghost (design pages, independent reader, prompt audit; from 20:56Z) and mrwolf (the signal lane and the check of the earlier postmortem; 18:02Z to 20:18Z). The handover counts seven; mrwolf is the eighth. A ninth transcript last written that midday belongs to the previous day's job and holds no owner row and no shell command on 2026-10-10; it is not counted.

**How they were read.** Every row the owner typed (including the ones typed mid-turn, which a transcript stores as queued commands and not as user rows), every message between sessions, and every shell command were pulled out by raw line number. I read all of the owner's rows to the lead session myself. Five readers then took a slice each: the lead session's day in three windows, the traffic between the other sessions, and each session's own written account against its own transcript. Each reader was asked for faults, for proposals and warnings that were not acted on, and for points where everyone went the same way. The rows this page leans on hardest were then found again by me in the extracted rows. Two of the day's texts were read whole. Finally all seven sessions still open were asked the same five questions and answered from memory without re-reading (section 5).

**Not done.** No render was watched and no still was opened for this page; what a render shows is as a session or the owner reported it. Most tool output was not read. A reader's "not found" means not found in the rows it read.

**Conventions.** Times are UTC with a Z; local time is five hours earlier, and commit times in `git log` are local. `mrpop L7850` is raw line 7850 of that session's transcript. The people in the clip are the lead, the second subject and the third subject; "the lead session" is mrpop. An owner's instruction is given by its substance, its row and its time.

## 1. What went wrong, and what would have shown it sooner

Each item says what happened, where it was first visible, and what made the mistake easy. None of these is answered by reminding sessions of something more often; where one seems to be, the last line says what it is a flaw in.

### 1.1 The text was read last, four times

- **Morning, an added sentence.** The lead's expression went wrong on the street shots. The first explanation was the motion channel and a new test arm was built for it (mrpop L1534 14:56Z); the cause was one sentence the lead session had added to the text (L1630 14:58Z).
- **Midday, one text over seven shots.** A text written from one shot, holding an action sentence, was used for a load of seven shots (L3019 16:13Z). The subject took that action's pose in a shot where the source does something else. The file was delivered (L4151 17:07Z) before the text was read as the cause (L5218 17:45Z).
- **Evening, the shut mouth.** A face render's mouth stayed shut where the source's is open (first seen L13081 23:06Z). The lead session blamed the audio file (L13220), then the attention sink (L13861), and queued three test renders. The owner was the first to name the text (L13935 23:49Z); no session had, in any of six transcripts over that stretch. The lead session then read the text and took the absence of any sentence about the mouth as clearing it (L13951 23:50Z). Two more arms later it added one caption sentence saying the lead sings, and the mouth opened (mrdeer's reading, L14438 00:04Z).
- **All day, the wording.** At 14:09Z the owner said that from the model's side a new person is being generated and nothing is being replaced (L479). Five minutes later the texts gained sentences about a scene "already lit, framed and cut" and things that are "the scene's own" (raw L688 14:14Z), and a rule written into the tree that hour kept one of them (`47f42dec`). The evening's face texts carried "stays as it is" and the pack's two sound lines, which say the scene is quiet over a song (L11017, written 21:40Z). The other sessions did not have the texts in front of them: they were given each one as a clause in a message from the lead session, such as a text with certain things "named as the scene's own" (mrdeer L3621, L5128; mrcorn L4424), and built on that description. Where a peer did check a text it checked a fact in it (mrdeer, whether a voice is present; mrghost at 23:07Z, whether an object is named) and never the wording. Nobody questioned the wording until the owner did at 23:57Z (L14220). The sound lines came from constants in `masked_prompt_text.py`; the audit of every such sentence is `bench/results/2026-10-10_prompt_text_hardcoded_in_code.md`.

*First visible:* in the text itself, read whole beside the frames of its load. The texts are short, and they were in the job folder before the first of the evening's test renders.

*What made it easy:* the builder's printout of what each model is shown gave a text's length where it gave the still's size and the region in words (the builder session's own answer, section 5). A text had no frames attached to it, so one text could serve four shots without anything saying so. The capture, the gate and the look all read masks; nothing put a load's text in front of the session at the moment it decided what was wrong. And a text travelled between sessions as a summary of itself, so the one session that had read it was the one that wrote it.

*Still open:* the morning's accepted face render opened its mouth with no sentence about singing in its text (L13861). Nobody has said why. Read whole and side by side for this page, the street face text and the evening's close-up text (`data/2026-10-10_fun_job_inputs/jobs/lead_face_street.txt` and `data/2026-10-10_fun_job_inputs/jobs/lead_face_night_phone.txt`, untracked) are the same in every standing sentence: the same edit sentences, the same two sound lines, and neither says the lead sings. They differ in the scene they describe and in one sentence about an object held at the face. So the standing sentences were ruled out by the owner's principle and not by a measured harm, and they do not by themselves explain a shut mouth; one arm that removed the object sentence helped a little (L14360). The caption sentence is one load and one seed, measured by a pixel stand-in and seven stills, and it changed the rest of the face more than any other arm did (L14438).

### 1.2 A whole shot left as the source in a file the owner locked

Source frames 1010 to 1038 are one shot. Both trackers called their subject absent there, so nothing was laid for either subject (L7802 19:14Z).

- It was known to be uncovered five times before the file was built: by mrdeer (L1345 14:51Z), by the lead session (L1869; L2863 16:00Z, which says the shot needs a typed correction and its own preview), by mrcorn (L3584) and again at the gate (L4621 17:20Z).
- Two tools had flagged those frames: the capture's `absent_with_people_on_screen`, at a level that did not block and with no outcome entered, and the delivery record's `piece_changes_nothing`. The delivery session "read them as good news each time" (mrcorn L3909 19:15Z).
- The message that told the owner the file was ready listed the shot as a known fault for the second subject in one line, reported "nothing is laid on frames 1010-1038" as the cut check passing, and said nothing of the lead on that shot (L5873 18:16Z). The owner called the file good and asked for it to be locked (L5921, L6004). A viewer found it within the hour (L7671 19:11Z), and the lead session called it "a miss on my side" (L7825).

*What made it easy:* a zero where the job was to change something read as a pass, and a flag that nobody had to answer. The delivery tool's proofs could only fail something that had been laid. It now fails a file when a named subject is on screen with nothing laid (`0d20e742`).

### 1.3 The first explanation was wrong, and the right one was already on the table

- **The short shot that faded back to the source.** Before the two loads for that shot were queued, mrnemo wrote that a load this short holds a tail of pinned frames showing the original person, named the frames to read and named the fix (mrpop L7850 19:15Z). The reply was not to build it yet (L7878). Both renders faded, one more render went to another idea, and the fix (the open tail, `e2469ab7`) was measured as holding on all three loads at 20:46Z (L9949).
- **The head that did not turn.** At 18:15Z mrdeer said the window boundary and the turn fall on the same frames (L5821) and the owner said a window leans on the frames it keeps from the one before (L5840). The next renders tested where a patch opens, the size of the motion mesh and a sentence. A load with no kept frames made the turn at 19:07Z, and the measure at 19:55Z showed the kept frames decide it at either mesh size (L8662).
- **"She ignores her mesh because she is small."** Said of the second subject in one shot (L9537), repeated from a headline its author then withdrew (L10549), and tested with a zoomed mesh that changed nothing. The one change that fixed it was the reference still (L10565). One shot, one seed.
- **"The street face is too small."** Given to the owner as the cause (L9733 20:40Z), relayed to three sessions as fact, and answered with a gate rule and a zoomed render. Withdrawn by the lead session sixteen minutes later, when the owner read the same unchanged render as swapped (L10083, L10130). It was never tested.

*What made it easy:* a test arm was cheap to add and a one-change control was not, because every graph before 21:11Z came from a session script edited in place (L8857). Adding arms felt like progress; the session that had named the cause had no way to make that outrank the queue.

### 1.4 A pass read as right

The day's rows hold many instances of a check, a proof or a gate being green while the thing was wrong. The ones that reached a file or a queue:

- every proof of the delivery assembler green on a file whose render had changed frames across cuts (mrcorn L1927 17:02Z), on a file with a patch laid under the piece it was to replace (L2672 17:57Z), and on the locked file of section 1.2;
- the re-check written to catch the second of those, which the known-bad file passed (L3004);
- `verify` exit 0 and a clear gate on twelve loads, four of them wrong (section 1.5), and a clear gate on takes of the wrong person, which no rule can see (mrdeer L7460);
- the who-is-who tool's check, whose made-up tables all had complete lists of people, while a later tracker's real table has the earlier trackers' people removed (mrfrog L4802);
- the look's check, whose made-up job file had the same mistake as the code it tested, so that no real job could have recorded a look (mrdeer L7964; mrcorn L7551).

What found each one: a count against an expectation stated beforehand, the author measuring the output file, a viewer, a second session's read of every frame, the lead session comparing two previews of one load, and a second session running two halves together on a copy of a real job file.

*What made it easy:* fixtures written from the code's idea of its input, and reports to the owner that led with "green", "clean" or "measured before I'm telling you it exists" (L5873).

### 1.5 Twelve renders queued on a list answered in under a minute

The gate's list for the rest of the clip arrived with five blocking flags and one plain piece of advice from its author: end the third subject's load before the two subjects cross (L12228 22:42Z). The lead session entered an outcome on all five 39 seconds later (L12241), kept that load whole, and queued twelve renders (L12249 22:43Z). Its own look at six frames a load was made after the queue (L12329 to L12354; L12469 says so). The owner's instruction to verify the data, and then to have someone verify it independently with eyes, arrived in the next three minutes (L12311, L12370).

The independent reader (mrghost) was given the files and the questions and not the lead session's conclusion (L12377). It found two face loads whose part was fragments on hair, wall and another subject, a whole-person region measured against a track that was a head alone, and a track that took another subject's arm (L12694, L12807, L12883). All four were still pending when pulled. The record is `bench/results/2026-10-10_preprocessing_data_read_by_a_second_session.md`.

Two things were already known and did not travel. The lead session had itself noted, half an hour earlier, that the lead's track in that shot was her head only and that a whole-person pass could not live with it (L11747), and did not carry that to the other subject's whole-person load over the same shot. And the fragments were in two warning rules that the gate's author had left out of every list it sent that evening (mrdeer L6630).

*What made it easy:* a builder that would queue the moment a gate read clear, a gate whose warnings outnumbered what anyone could read, and no step between "the data exists" and "the run starts" that belonged to someone else.

### 1.6 One measurement of likeness, arranged three ways

By 19:23Z mrfrog had measured that no likeness threshold separates two of the people in this clip (mrpop L8113). The afternoon then built a sum of likenesses and an "evidence total" on the same measurement, and treated their agreement as support. The sum followed whichever person the first tracker did not hold; a grading of "right on five of five" went to three sessions and was withdrawn within the hour when the lead session ran a second preview of the same load (mrfrog L4904). mrghost had written at 22:06Z that models looking at the same pixels are not independent votes (mrghost L857); nobody answered it. The arithmetic was removed whole that night (`ace202c4`).

The simpler ideas were the owner's, and both came first as questions: keep a person's answer about who is who up front (L10054 20:53Z), and start the tracker from a frame where the person is already known. The first answers to both were heavier designs. The simple forms were built or tested only at night (`bench/cast_file.py`, `35e45d29`; the seeded test in `bench/results/2026-10-10_who_is_who_across_shots.md`, `ff1568c9`).

### 1.7 Eight sessions in one tree

- **Commits that carried another session's uncommitted lines.** The rows show ten: `56e35121`, `91af9d3d`, `96f204f6`, `f874e895`, `4f03d521`, `768944d1`, `5e56d3fe`, `23b01495`, `68347328` and `4cbfaa10`. Nothing was lost in any. Each time the committer had printed the file's status in the same command as the edit or the commit and had not stopped on it. The guard built that afternoon (`2bd4a3b3`) refused correctly twice, and both commits went through because they were chained behind it. The rule in `AGENTS.md` was followed in spirit each time and was not enough; the sessions' own proposal is a commit helper that reads the diff in the same command and refuses when it is not what its author expects (mrdeer L7842).
- **A claim made from a rule and not from its source.** Sessions wrote "not pushed" in their reports all day. At 18:12Z the lead session had itself written into the shared notes that the owner pushes through the day (mrpop L5712), and that evening it wrote "nothing is pushed" into the handover and onto the board. `git reflog show origin/main` holds the pushes. The lead session found and corrected it (`dab2f38f`).
- **A retired session's open work.** When mrwolf retired, every reading it had queued was still unread (mrwolf L1660 20:13Z). Its brief for the builder and its order-of-operations guide were carried on; no later row reads the queued renders.
- **The lead session as the narrow point.** Every render and preview until 21:11Z came from the lead session's own script, which the owner had said at 15:10Z not to build on (L1894, no reply found). The lead session's own words at 20:09Z: the builder was the first priority and nobody was building it while it coordinated (L8847). The tracked builder's first job was queued at 21:11Z.

## 2. Good ideas that came late, were deferred, or were dropped

One row each. The last column is what the rest of the day showed, and "cannot tell" is an answer. Ideas that everyone turned down for a reason that then held up are left out.

| The idea | Raised by | What happened to it | Did the day show it would have helped |
|---|---|---|---|
| The model does not know anything is being replaced; write the least | Owner, mrpop L479, L666 (14:09Z to 14:13Z) | The opposite wording entered the texts five minutes later and stayed all day | Yes: section 1.1 |
| Do not build on an old graph that was patched for one case | Owner, L1894 15:10Z | No reply found; every graph until 21:11Z was that graph patched by a session script | Yes: the script is where the audio length and two inputs were mis-set (L8857) |
| Hold the mask open over the held tail of a short load | mrnemo, L7850 19:15Z | "Do not build ... yet"; built after three renders had faded | Yes: L9949 |
| The window boundary, not the motion signal, explains the head | mrdeer L5821; owner L5840 (18:15Z) | Answered by adding test arms | Yes: L8662 |
| The uncovered shot needs a typed correction and its own preview | Lead session, L2863 16:00Z | Done at 19:13Z, after a viewer had seen it | Yes: section 1.2 |
| End the third subject's load before the subjects cross | mrdeer, L12228 22:42Z | Overridden in 39 seconds; pulled and re-made shorter after the independent read | Yes: L12883 |
| One written statement of who is in each shot, read by every tool | mrdeer L11840 22:15Z; mrsheep L2365 | Approved that evening; not built. The same answer was typed through five forms that night (mrdeer L6630) | Being built as this is written, on the owner's word, with a confidence on each line and shown, not refused |
| A cheap way to change one thing on a built load | Owner, L14545 00:09Z | A card. The lead session had posted five graphs by hand that evening for want of it (L14547) | The five hand posts are its cases |
| One command that shows a wrong render's text, mask and timelines first | Owner, L14616 00:13Z | A card | It is the stretch from L13081 to L13951 |
| Keep a person's answer about who is who; seed the tracker from a known frame | Owner, L10054 and after | Heavier designs first; simple forms at night | The seeded test is one clip and one reader; cannot tell yet |
| Patch on top of the previous render, where every pass from the original needs a merge | Lead session to the owner, L4675 17:21Z | No answer found; never tested | Cannot tell. The other choice is what needed the restore rules, the owner map and the contested-pixel option |
| Feed the audio in as a labelled reference | Owner, L666 14:13Z; again L14166 | Never tried | Cannot tell |
| A whole-person render of the lead | The lead session, in its answer to section 5; no row | Never said, never run. The lead was a face part on every load, as the brief asked | Cannot tell, and it is the cheapest untested alternative for a stiff face |
| One row a check in `docs/checks.md`; a commit guard by the diff | mrdeer, L2033 and L2251 (17:21Z, 17:34Z) | Deferred | Yes: two later carried commits were in that file |
| Two warnings on the frames that later failed (a mouth open with no voice; a part in fragments) | The gate itself (L13716, L6630) | In a list nobody read | Yes for the first: it named the stretch where a test render could show nothing |
| Objects near a tracked subject get their own segment | Owner, L2114 15:17Z; L9834 20:43Z | A section of a design page (`d300c55f`) | Cannot tell |
| Tie a render to the analysis folder it was built from | Owner, L10531 21:12Z; mrnemo | Built, read cold, held for a day | Cannot tell |
| Look at the whole face before accepting the caption render | mrdeer, L14438 00:04Z | Accepted 95 seconds later; no look in motion found | Cannot tell; the handover lists it as uncertain |
| mrwolf's queued readings | mrwolf L1660 20:13Z | Nobody picked them up | Cannot tell |

Read down the second column: the owner and the session nearest the data are the usual source, and the idea usually arrives before the render it would have saved. What the day lacked was not ideas. It lacked any way for a warning from the session nearest the data to outrank the queue.

## 3. Where everyone went the same way

- **A fault is answered with one more rule, flag or refusal.** No row in any transcript objects to this before the owner did (L12513 22:51Z; again L15612 00:41Z). Two sessions say they doubted it from mid-afternoon and kept quiet (section 5). By evening each author marked much of its own day for removal, and the gate's author counted its flags and found that most came from two rules it had been filtering out of its own reports (mrdeer L6630).
- **A clear gate, a green sweep or a passing record is the state of the data.** Wrong each time it mattered (section 1.4). The lead session's own later words to the owner: "replaced, by stills and numbers" had turned out not to mean it reads as replaced on playback (L9733).
- **A wrong render is a signal problem.** Section 1.1 and 1.3. The dissent each time was one line from the session nearest the data, or the owner.
- **The wording of the texts.** No dissent at all until 23:57Z.
- **A person is recognised by likeness against a line, and then by a sum of likenesses.** Dissent: the measurement itself (L8113) and one unanswered line (mrghost L857).
- **The second subject is replaced whole.** Fixed after one render on one seed (L1116 14:37Z). The owner's later verdict that the whole replacement follows the head better (L11294 21:53Z) compares two renders that also differ in whether they were given a motion video (L11091).
- **One load a shot.** It rests on two shots and one seed (`bench/results/2026-10-10_one_load_per_shot_against_one_long_load.md`). The alternative was not weighed again, and the evening's text rule now makes a load that holds a cut unbuildable. The builder session says its own first job was one load over three shots and it never looked at whether that rendered worse (section 5).
- **Every pass from the original, merged at assembly.** Never tested against patching on top of a finished render, which is the one route that had worked two days earlier.
- **Land every removal the same night.** Two sessions doubted it and did not say so once the owner had asked for it (section 5). It landed without a stop, each piece read cold first. What gave way was smaller: the looks one session had asked for at named frames, the one preview that would show the new tracker's drawn label, and the day's own job files, which no longer build.

## 4. What worked

- **A second session that reads the real thing without the author's conclusion.** mrghost's read of the twelve loads (section 1.5). mrcorn's cold reads of other sessions' patches, which ran the code on real shot tables and real job files and found something real in most of them (a missing field read as "nobody was left out", mrcorn L6427; a bench node that would have raised on the landing, mrpop L13578). mrnemo's audit of the prompt fix, which went back to the vendor guide and overturned both the fix's new input and a row of the audit it came from (mrghost L2229). Withholding the conclusion was a choice the lead session made on purpose (L12377), and it is what made the read independent.
- **A number written down before the build.** The delivery session wrote what a record should say before reading it, and that caught the changed frames across cuts, the patch laid under its piece, and a whole-clip file built short (mrcorn L1894, L2546, L7316). mrfrog wrote bounds from a CPU run before the run on the card, which caught half precision putting the body model outside its maker's own floor (mrfrog L1610).
- **A person reading the tracker's tiles before anything rendered.** The first three-subject previews were wrong on most shots; the lead session found that from tables and tiles, and nothing was rendered from them (L11452).
- **Reading what was saved, where a test render was the other choice.** mrnemo decoded a saved window and showed the sampler, not the composite, drew the original back (L8739). mrdeer's per-frame measure found a fade that stills at even spacing had missed (L8773).
- **A small pair before a long pass.** A 141-frame pair showed that a keep at the sampler brings the original back, and the long pass was pulled (L3115).
- **Saying a slip at once.** Almost every carried commit and every withdrawn claim in the rows was reported by the session that made it, usually within a minute.

## 5. What the sessions said when asked, against the rows

All seven open sessions were asked the same five questions after the day closed and answered from what they still held, without re-reading or conferring. Every one of them said first that its morning survives only as its own summary. Their answers agree with the rows on substance and drift on order and time.

| Session | An idea that was right and came late | What it went along with while doubting | Against the rows |
|---|---|---|---|
| mrpop | The one-change variant; texts written per load from that load's frames | The lead as a face part only; it thought a whole-person render might explain the stiff face and did not raise it | Lines up (L14545, L14547). Its "queued having looked at six frames" is kinder to itself than the rows: the look came after the queue (L12469). The unsaid doubt has no row, as it says |
| mrdeer | One statement of who is in a shot | A rule for every fault; nobody read the warnings, itself included | Lines up (L6630). It raised the statement's first form earlier than it remembers (L11840) |
| mrcorn | The same statement; its tool got its own version only after the missed shot | A proof for every fault; "stop adding checks" from the session that had just missed something would read as an excuse | Lines up (L3909). Its written account says the owner's playback found one fade; the rows say mrdeer's reading did (mrdeer L4588) |
| mrfrog | The owner's two ideas for who is who | Landing the tracker change the same night | Lines up (L4904). No row shows the doubt, as it says |
| mrghost | The owner's seeded tracker, which its own first answer buried under a heavier plan | That one model's features are the judge at all | It did say nearly this at the time (mrghost L857) and nobody answered. Its account of the music line has the order reversed: the lead session told the owner first (mrpop L14100, then L14187) |
| mrnemo | Tying a render to its analysis folder | Landing every cut the same night | Lines up: it argued for the next day before the owner's word (mrpop L13131) and not after |
| mrsheep | The one-change variant; "I built refusals all day and no cheap honest route" | That a load must lie inside one shot; it framed that as a consequence to accept, not a choice to test | Lines up (mrsheep L2721); the comparison it did not run is not in the rows either |

Asked for the day's biggest miss, four named the texts, two the data that passed its checks and was wrong, one the shot left as the source, and one said it in a form none of the others did: nobody looked at one load's text, frames, mask and audio as one view, and reading one built graph end to end would have shown most of it (mrnemo). Asked what helped most, every session named a second session reading or running its work on real files. None named a check.

## 6. Deviations from the plan

| Planned | Shipped | Verdict |
|---|---|---|
| Three people replaced across the clip, from its first frame, audio untouched (the owner's brief, L15 14:00Z) | Frames 604 to 1050 as a locked file with one shot of it left as the source; three joined files to 1050; eleven first-pass pieces for 1039 to 1729; a joined file to 1919 built, not placed, not looked at; nothing after 1919 | Scoped down by the owner's own ruling. The table of what is left as the source is `data/2026-10-10_fun_delivery_tables/t_fun_0000_1919_a_build.txt` (untracked) |
| The lead as face and likeness only | Done so on every load | As planned; the alternative was never run (section 2) |
| The second subject whole, with a report of what to keep if the mask is awkward | Whole, fixed after one render; face and hair tried later at the owner's suggestion | Open: the comparison that decided it differs in two things |
| No prompts from templates; start from what worked last time; write the least | Hand-written texts with edit sentences and the pack's own sound lines in every one | The deviation that cost the most (section 1.1) |
| Use the data tooling that exists, before and after a render | A capture tool, a gate, a builder and an assembler built that day | More than planned in tools. Used by a session other than the builder on real job data: `bench/masked_job.py`, `bench/capture_masked_run.py` and `bench/who_is_who_across_shots.py`, each by the lead session. `bench/assemble_delivery.py` and the rest were run by their builders only |
| Tidy up while the whole clip runs, then a postmortem | The removals and a second stretch landed the same night | Changed by the owner's word (L13223 23:10Z); see section 3 |

## 7. Escapes

Each is a check that was green while the thing was wrong, and why it could not see it.

- **`bench/check_assemble_delivery.py`:** every proof could only fail something that had been laid, so a shot with nothing laid passed. Now fails on a named subject with nothing laid and no reason (`0d20e742`).
- **`bench/check_capture_masked_run.py` and the gate:** no rule sees a take of the wrong person scored over the tracker's line. Still open, and said so in the tool. The statement of who is in a shot is the answer on the table.
- **The who-is-who tool's check:** fixtures with complete lists of people. A later tracker's real table does not have them.
- **The look's check and `bench/check_masked_job.py`:** a made-up job file that shared the code's own mistake; the builder's check runs no node.
- **The new prompt check:** green on a newly written assertion; its reader said plainly that the guard against a new false sentence is a person and not the check (mrnemo, mrpop L14967).
- **Single checks at each step of the landing:** two doc pointers to deleted code were found by the full sweep (`7c902d53`), and three more stale sentences by reading (`e550c443`).

The pattern is one thing: a fixture or a proof shaped by the code's idea of its input. The catches in section 4 all came from the real input.

## 8. Where the handover is wrong or short

`docs/wiki/handover.md` was written the same evening from a memory that had been summarised several times, and says so. Each of these has a dated line in that page.

1. **"Frames 0 to 1050 were delivered earlier in the week and that file is locked."** The job began that morning. The locked file is `fun_0604_1050.mp4`, built and locked on 2026-10-10 (L5856, L6004), and its record has read as failing since 19:27Z (L8211). The files covering 0 to 1050 were first placed at 20:34Z the same day (L9638) and none is locked.
2. **"Four commits of the day carried a peer's uncommitted lines."** The rows show the ten of section 1.7.
3. **"Twelve renders were queued after six frames a load had been looked at."** The look came after the queue (L12469).
4. **"Eleven commits" for the first landing.** `e9de8cc3` to `368fbaa8` is thirteen commits by four sessions (L14716).
5. **"Nothing is pushed."** Corrected by the lead session in `dab2f38f`.
6. **"Led seven sessions."** Eight worked the lane; mrwolf's readings were left open when it retired.
7. **"Two of the lead's figures were wrong the first time they were said."** The page names neither. The rows have a count of wrong takes (L11569, corrected L11664) and the "five for five" (L11689, corrected L11878).
8. **The whole-person verdict.** The two renders the owner compared also differ in whether they had a motion video (L11091).
9. **"Three renders before anyone read the text."** True of the lead session reading the text for that fault. The order was the audio file, then the attention setting; the first of the three also changed the text; and mrghost had read the same texts against their frames for another question at 23:07Z (L13132).
10. **"Every swept commit of the day was reported by the session that made it ... before anyone else found it."** True of most. `768944d1` was found by the owner of the carried line, and one half of `4f03d521` was named by the lead session first.

## 9. What to pick up next

Small, and each can be marked done or wrong.

1. **Show the text where the decision is made, and pass the text itself between sessions, never a description of it.** The builder prints a load's whole text beside what each model is shown, as the look's strip already does (`d46ece22`). When the cause of a fault is not obvious, the session also shows the owner the whole text (Owner, 2026-10-10; `AGENTS.md`, "When a render is wrong"). *Done when* `build plan` prints the text and not its length. *Wrong premise if* the next text-caused fault is again found by a render.
2. **Read one built load end to end before building on any of it:** its text, its window's frames, its mask and its audio together. This is the owner's last instruction of the day and the first lines of the handover. *Done when* the first load of the next job has one written reading of the four, by a session that did not build it.
3. **The statement of who is in each shot, with a confidence, shown per window and refusing nothing** (in progress as this is written). *Check after its first real job:* did a confident wrong "in" get noticed at the window it was shown on. *Refuted if* sessions read past it as they read past the warnings.
4. **The one-change variant of a built load.** *Done when* a test arm is made without a hand-posted graph. *Refuted if* hand posts continue after it exists.
5. **Three renders that each decide something already on the list:** the caption sentence on a second seed; the same sentence on a voiced load with a smaller face; and one whole-person render of the lead on the close-up load, the alternative nobody ran. Before any of them, with no render: set the morning's face text and settings beside the evening's and say why the morning's mouth opened.
6. **A flag is answered or it is not printed.** The two warning rules their author filtered are already on the approved removal list in `docs/wiki/decisions.md`. *Done when* the next delivery's record shows no flag without an outcome, or names who left it.
7. **A check's fixture is cut from a real job file.** *Check:* the next new check case in `bench/` names the real file its fixture came from.
8. **The commit helper** for files several sessions write: the diff read in the same command, the commit refused when it is not what its author expects. *Done when* the next day has no carried line. *Wrong premise if* it has one anyway.
9. **mrwolf's queued readings:** read them or drop them, one line on its card.

Two habits of the day cost nothing and found more than every rule written that day: writing down what a result should be before reading it, and handing the real files to a session that has not been told the answer.

*Annotation 2026-10-10, later the same night:* the owner read the sessions' answers to section 5's questions and to two more rounds about the statement of who is in a scene, and marked four things to actively avoid: a new place for a fact that is already written; a record with no way to say "I looked and I am still not sure"; the three costs of answering a picture (the cuts must be right, one picture is one frame, a wrong answer is inherited by every later job); and a rule for every fault. What each needs, and which of them exist, is in `docs/wiki/decisions.md`, 2026-10-10, under the contact sheets entry. Item 3 above is overtaken in form by the same entry: the answers are kept once a video, as a person's answer to a numbered picture, and a tool drafts nothing into that file.

## 10. Where the sessions agreed, where they split, and where each saw it from its own seat

*Annotation 2026-10-10, later the same night, at the owner's ask.* After the day closed the seven open sessions answered three rounds by message, each writing before reading the others: the five questions of section 5; the best case for a confidence number on a written statement of who is in a window, with its risks; and what else is wrong with that statement and a better way to write it down. The session that wrote this page answered the last two as an eighth voice. This is how the answers fell.

**Unanimous, or nearly.**

- **What helped most.** All seven named a second session reading or running their work on real files. None named a check.
- **Their own memory.** All seven said, unasked, that their morning survives only as their own summary of it.
- **Keep the declaration.** All eight, for one reason: every wrong take of the day was an "in" that nobody wrote down.
- **It belongs to the video, by shot, in the source's own frames.** All eight. In a job file it dies with the job; tied to a window it dies with the next plan.
- **Grow the cast file and add no new form** (`bench/cast_file.py`). All eight in substance. Five of the seven counted the form stopped that night as the fifth, sixth or seventh place the same fact is written, with none retired.
- **The case for saying how sure.** All eight found the same one: it is the only place a person can say "I looked, and I am still not sure".
- **The conditions on it.** All eight: only a person writes it, and no tool thresholds it or blends it with a tracker's figure.
- **What the better way costs.** Nearly all named the same three: the cuts must be right; one picture is one frame, so a subject who enters mid-shot needs a second answer; and a wrong answer is inherited by every later job.
- **A rule for every fault was wrong.** Nobody defended it, and two said they had doubted it since mid-afternoon.

**Split down the middle.**

- **Should a tool draft the tracker's take into rows nobody has answered?** Four yes, marked as nobody's word (mrsheep, mrcorn, mrpop, mrdeer); four no, show it beside the row (mrfrog, mrnemo, mrghost, and this page's author after reading them). The owner settled it the same night: the file holds only what a person said, and the tracker's opinion stays in its own table.
- **A number or words for how sure.** Three for a number, kept only if it is scored against what later proved true (mrpop, mrcorn, mrsheep); five for a few steps or words (mrghost, mrfrog, mrnemo, mrdeer, this page's author). Not decided.
- **May a tool list the least sure first.** Five yes (mrpop, mrsheep, mrcorn, mrghost, this page's author); three say no tool should sort by it at all (mrfrog, mrnemo, mrdeer). Not decided.
- **What "not sure" is worth.** One would count it as not answered (mrghost); most say a recorded doubt is the most useful line in the file.

**No agreement: each saw it from its own seat.**

- **The day's biggest miss.** Four named the texts (two outright, one as a guess, one as its second choice). The rest named what was nearest their own work: data that passed its checks and was wrong, the shot left as the source, one measurement arranged three ways, and nobody looking at one load whole.
- **The idea that came too late.** Three pairs and a single: the one-change variant of a built load (mrpop, mrsheep), the statement of who is in a shot (mrcorn, mrdeer), the owner's two ideas for who is who (mrghost, mrfrog), and tying a render to its analysis (mrnemo).
- **Landing every removal the same night.** Two doubted it and did not say so (mrfrog, mrnemo); the lead session's handover lists it among the right calls.

**Raised by one or two, and contested by nobody.** That the owner's eyes on playback are the real "seen" and should be kept apart from a session's look at stills (mrcorn, mrpop). That one load a shot was never tested against the alternative (mrsheep). That the question to ask is "who are the numbered people on this picture" and never "is this subject in", because a pre-filled "not in" was wrong on the two shots where a wrong take hid somebody (mrfrog); the owner called it a good idea.

What the owner marked from these to actively avoid, and what was built that night on the strength of them, is the annotation at the end of section 9 and `docs/wiki/decisions.md`, 2026-10-10.
