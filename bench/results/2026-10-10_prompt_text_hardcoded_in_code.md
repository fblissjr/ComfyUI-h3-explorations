# Prompt text that lives in code: what it asserts, and whether the code can know it (2026-10-10)

lane: prompting
verdict: read, not rendered. One file, `masked_prompt_text.py`, writes eleven assertions about a clip's sound, look, people and movement that no input can remove and the code cannot know; every shipped graph that wires the Masked Prompt node carries some of them. The same file writes about as many sentences that talk about an edit (what stays, what is kept, what is "the scene's own"), which is not how a caption of a finished shot reads. The generator's reference composer also writes prose, for scenes the model makes from nothing or for the pack's own assets. Nothing here shows that any of these sentences caused a fault in a render.

**Read this first.** This is an audit of code, made by reading and counting.
Nothing was rendered and no wording was changed. It was asked for when a
prompt was found to say a quiet room about a track that is a song. Whether
that sentence is why a mouth stayed shut in a render is not shown here.

## What was asked

The owner, relayed by the lead session: make sure prompts are not hardcoded
in code "unless theres a damn good reason"; a prompt never says anything
about a window that is not known to be so; and for the sound fields the
owner's hunch is "generic and concise and enough to let the model infer one
way or another". Read by a session that wrote none of the code.

## How it was searched, so a miss can be found

- **Every string of six words or more** in the 73 tracked `.py` files
  outside `bench/`, `docs/`, `archive/`, `standalone/` and the vendored
  code, found with `ast`; docstrings, tooltips, errors, logs and
  command-line help left out. 769 candidates. All but two files' worth are
  reports, notes and file names.
- **Read one by one:** every candidate in `masked_prompt_text.py` and
  `masked_prompt.py`, and the reference composer of
  `workflows/build_workflows.py` (`_ref_prompt`, `_IMAGE_ROLE_PROSE`,
  `REF_SCENE_SHOTS`, `REF_SCENE_AUDIO`, the keyframe alignment sentences,
  `_SONG_FLICKER_LISTS`).
- **Every node input a prompt goes into** (`prompt`, `timeline`,
  `add_to_shot`, `subject`): none has a default that is prose, except
  `subject`, whose default is the word `person`.
- **Carriers counted by walking** `workflows/h3_config.py::graph_paths`
  (168 graphs) and `prompt_bank/` (167 entries). The Masked Prompt node's
  text is in no graph file, so for each graph that wires the node it was
  assembled from that graph's own inputs.
- **Not searched as closely:** `bench/` and `standalone/` (a grep for the
  section names only); texts typed by hand under `data/`; the vendored
  trees.

The candidates, the counts and the two scripts are in
`data/2026-10-10_prompt_text_in_code/` (untracked).

## What the vendor guide says about the sound fields

- `overall_soundscape` is one to four sentences of ambience, physical
  action sound and non-verbal human sound. Dialogue, singing and diegetic
  music belong in the description and are not repeated there. `N/A` only
  when the user asks for complete silence (`vendor_guides/base_en.md`, 4.6,
  lines 152-154).
- `non_diegetic_music` is music only the audience hears. Singing,
  instruments, a radio or a phone the characters can hear are diegetic and
  go in the description. `N/A` when there is no such music
  (`base_en.md`, 4.7, lines 160-162).
- A subject who speaks or sings carries a speaker id, and the words go in
  `<d>` (`base_en.md`, 4.4, line 127; `vendor_guides/ref_en.md`, 5.4, lines
  260-266).
- Where reference audio is used, its copy or reference relationship is
  stated in the field that matches the layer: ambience in the soundscape,
  an audience-only score in the music field (`ref_en.md`, 6, lines 296-301).

## The inventory

"Carried by" is graphs of the 168 walked, then bank entries of 167.

### `masked_prompt_text.py`: the Masked Prompt node's text

The node takes four choices from a person (`subject`, `voice`,
`picture_gives`, `add_to_shot`) and reads two from the Masked Source it is
wired to (`replace`, `motion_reference`). `voice` defaults to "the main
voice on the track" in the node (`masked_prompt.py:68`) and in all three
prompt dicts of `workflows/h3_config.py`. Twelve shipped graphs wire the
node.

| line | the sentence, shortened | what it asserts | can the code know it | carried by | behind a choice |
|---|---|---|---|---|---|
| 108 | `SOUNDSCAPE`: "The quiet ambience of the scene sits underneath, with the soft brush of clothing as the performer moves." | the sound: quiet, clothing | no: the lane runs on the user's own track and never hears it | 12, 6 | no |
| 110 | `MUSIC`: "non_diegetic_music: N/A" | that there is no score | no | 12 through the node | no |
| 126, 182, 206 | "The target video is photorealistic live-action" | the look | no | 12, 12 | no |
| 129, 210 | "is in the scene for the whole take"; the head "for the whole take, at the same size and in the same place" | presence and size through the window | no: the node does not know which window a render takes | 10, 7 | by role |
| 162-164, 237-238 | "the other people, the walls, the furniture and the ground are visible exactly as before, steady and in focus" | that those things are there; that the plate is steady and sharp | no | 9, 5 | by role |
| 137 | "The shadow <Subject 1> casts lies where that light sends it" | a visible shadow | no | 7, 4 | by role |
| 218-219 | "none of the original hair remains on the shoulders, the chest or the back, where the clothing, its print and its folds carry on unbroken" | long hair on the original; a print | no | 2, 1 | by role |
| 149, 156 | "A breath lifts the chest and shoulders before each new phrase."; "The breathing is slow and even" | breathing | no | 7, 4; 0, 1 | under either voice |
| 106, 150, 157, 229, 234 | `EYES`: "The eyes blink naturally and hold a steady line of sight", with the brows, the face, or the head added | gaze and expression | no | 9, 5 | under either voice |
| 151-152, 158-159, 229 | "The head tips and turns with the delivery, the shoulders loosen and move with the rhythm, and the hands move with the phrasing"; "move a little and often"; "the head tips and nods with the delivery" | what the head, shoulders and hands do | no | 7, 4; 0, 1; 2, 2 | under either voice |
| 99-102 | `LIP_SYNC`: the lips open "on the first syllable of every sung or spoken phrase" | that the subject voices every phrase | a person chose "is the voice"; "every phrase" goes further than the choice | 9, 8 | yes: `voice` |
| 103-105, 191-192 | `LIPS_STILL`: does not speak or sing; no voice on the track is theirs | silence of the subject | a person chose it | 0, 1 | yes: `voice` |
| 96, 190 | "performs the main voice heard on the track" | that the subject is the voice | a person chose it | 12, 5 | yes: `voice` |
| 86-94 | `MOTION_CLAUSE`, `VIDEO_DEFINITION`, `VIDEO_RETENTION`, `MOVES` | a relationship to `<Video 1>` | yes: the Masked Source's `motion_reference` | 6, 3 | follows a wired input |
| 107 | `CAMERA`: "The framing and every movement of the camera are the scene's own and do not change." | a relationship: nothing of the camera is new | holds for any clip | 12, 6 | no |
| 133-136, 186-187, 220-222 | the scene's light falls on the subject as on what is beside them | a relationship | holds for any clip | 12 | by role |
| 118-125, 177-181, 197-205 | the definition, the summary and the retention line of each role | what the still provides and what is kept | yes: `picture_gives` and `replace` | 12 | yes |
| 138-144, 223-224 | "Where the frame shows the whole figure ... where the frame shows only the head and shoulders ..."; "Seen from the side or from behind ..." | nothing: each is an "if" | holds for any clip | 7 | by role |

The head-and-upper-body role (lines 176-195) is the short one. Its own
comment says nothing in it says what the subject will do, because the text
is written without seeing the clip. Of the rows above it carries only the
sound fields, "photorealistic live-action", and the voice sentence.

### `workflows/build_workflows.py`: the reference composer

`_ref_prompt` (line 2939) composes a full-reference prompt from the roles a
graph declares. Its result is returned from `prompt_bank/` by
`_composed_from_bank`, so every text it can write is also a bank entry and
is graded there.

| line | the sentence, shortened | what it asserts | can the code know it | carried by | behind a choice |
|---|---|---|---|---|---|
| 3199 | soundscape default: "Natural ambient atmosphere continues throughout the shot." | ambience, in general words | the model makes the sound; no source is contradicted | 23, 6 | the graph's author |
| 3201 | "The ambience of <Audio 1> continues under the shot." | a relationship to reused audio | yes: the graph wires the video's audio | 10, 7 | follows a wired input |
| 3206 | music default: `N/A` | no score | for a generated scene it is an instruction; over a reused track it is a claim | with line 3199 | the graph's author |
| 3208 | "A slow instrumental score follows the tempo and instrumentation of <Audio N>." | a relationship, and "slow" | the relationship yes; "slow" no | 4, 2 | the graph's author |
| 3156-3170 | "enters from the left and stops at the center of the frame"; "The camera trucks right with small amplitude at slow speed" | an action and a camera move | scripted for footage the model makes | 19, 7; 33, 25 | the graph's author |
| 3194-3197 | the line "I thought you would have gone by now." | a spoken line | scripted | 1, 1 | the graph's author |
| 3125-3137 | "performs the exact movements and actions of the original character from <Video 1>"; "continues walking out of frame to the right as the camera holds" | what is in `<Video 1>` | only for the pack's own asset | 1, 1 each | the graph's author |
| 3060-3064, 3132 | the red waxed-cotton jacket | a garment | scripted | 1, 2 | the graph's author |
| 2574-2592 | `_IMAGE_ROLE_PROSE`: what each still is for | a role | yes: declared by the graph; the comment above it records the day a generic line named "architecture" and built a chalet | many | the graph's author |
| 3006-3100 | the definitions and retention lines for a video's role | a role | yes: declared by the graph | by arm | the graph's author |
| 2687-2790 | `REF_SCENE_SHOTS`, `REF_SCENE_AUDIO`: two scripted scenes of four shots | a whole scene | scripted | 2 bank scenes | the graph's author |
| 437-451 | `_SONG_FLICKER_LISTS`: places and movements for one song graph | a scene | scripted | 1 graph | the graph's author |
| 750 and 793 | the keyframe alignment sentence, with seconds computed from the frame count | the guide's own sentence | yes | the keyframe graphs | no, and rightly |

### Everything else

- `masked_prompt.py`: the node; no prose of its own goes into the text.
- `body_pose.py::legend` (line 738): one sentence saying what the colours
  of a marked mesh mean, returned for a person to put in a text. It says
  what each colour is and nothing about what anyone does. The marked style
  is on the owner's list of things to remove.
- `audio_freeze_song.py`, `conditioning.py`, `reference_conditioning.py`,
  `prompt_lists.py`: they pass a prompt through or substitute a user's own
  lists. None adds a sentence.
- `workflows/h3_config.py`: no prose; the three masked prompt dicts hold
  the choices (`subject`, `voice`, `picture_gives`).
- `marker_arms.py:57`: a tokenizer probe string, never a prompt.
- **Texts typed by hand this week** (not code): all seven masked texts
  under `data/2026-10-10_fun_job_inputs/jobs/` end with line 108's sentence
  and `N/A`, copied.

## The sort

**1. Asserts something the code cannot know, and no input removes it: the
fault.** Eleven assertions, all in `masked_prompt_text.py`:

| | the assertion | the smallest change |
|---|---|---|
| 1 | the sound is quiet ambience and the brush of clothing | not a constant at all: the line comes from what is measured of that video's track (is there music, ambience, a voice, and where). No wording is proposed here. Not a cut: the six sections are required |
| 2 | there is no score | the same |
| 3 | the picture is photorealistic live-action | cut, or make it an input |
| 4 | the subject is there for the whole take, at one size | cut "for the whole take" and "at the same size and in the same place" |
| 5 | other people, walls, furniture and ground are there | cut the list; "everything else is untouched" holds without it |
| 6 | the plate is steady and in focus | cut |
| 7 | the subject casts a shadow | cut, or make the sentence an "if" like its neighbours |
| 8 | the original has hair on the shoulders and the clothing has a print | cut the clause after "the only hair" |
| 9 | how the subject breathes | cut, as the head-and-upper-body role did |
| 10 | where the eyes look and what the face shows | cut |
| 11 | what the head, shoulders and hands do | cut |

Nine to eleven are written under both values of `voice`, so choosing the
other one does not remove them. They are also what the owner's rule of
2026-10-07 names: a generated text never says what the subject will do.

**2. Asserts something a person chose.** The subject is the voice, or is
silent (`voice`); what the still provides (`picture_gives`); the words for
the subject (`subject`); anything in `add_to_shot`. One caution: the
lip-sync sentence says "every sung or spoken phrase", which is more than
"is the main voice" and is false in a window where another voice sings.
And all of the reference composer's scripted scenes: a graph's author
wrote them for footage the model makes, or for an asset the pack ships.
Two of those would become the fault above if a user wired their own video
and kept the text: lines 3125-3137, which say what the person in
`<Video 1>` does, and the `N/A` of line 3206 over a reused track.

**3. Structure and relationships only.** Section names; labels; the task
tag; retention markers; the `<Video 1>` lines; the "where the frame shows"
sentences; the alignment sentence. The camera and light sentences were
here in the first draft of this record and are in the next sort.

**4. Edit-talk: a sentence about an edit, where the model was trained on
captions of finished shots.** The owner's sort, given while this record
was being written: the model does not know anything is being replaced; it
generates what the text describes in some places while the original track
plays. Its training captions were written for footage it generates or for
labelled references. A sentence that says what STAYS, what is KEPT, what
is "the scene's own", or that a thing is "exactly as before" is then read
as a description of the picture, if it is read as anything. The guide's
own edit vocabulary (`<Video 1> is the source video for the target video
edit`, `ref_en.md` 2.5) is tied to a wired `<Video N>`; the masked lane
wires none for the plate, so these sentences have nothing to point at.

| line | the sentence, shortened | is there a plain caption of the finished shot that says it |
|---|---|---|
| 107 | `CAMERA`: "The framing and every movement of the camera are the scene's own and do not change." | only from per-video data: what the camera does in that window. Otherwise nothing |
| 133-136, 186-187, 220-222 | "The scene's own light falls on <Subject 1> exactly as it falls on what is beside ..." | only from per-video data: what the light is. Otherwise nothing |
| 121-122, 178, 200-202 | summaries: "takes the place of one person in a scene that is already lit, framed and cut ... while everything else in the scene stays as it is" | yes, and the file has it: "[Shot 1] <Subject 1>, the {who}, is in the scene." (line 183) |
| 126-128, 206-209 | "its setting, its lighting, its framing and every other person and object in it stay exactly as they already are from the first frame to the last"; "Only that person's head changes" | no |
| 129-131 | "in the place the scene holds for one person, at that place's distance from the lens" | no |
| 162-164, 237-238 | "Everyone and everything else is untouched ... visible exactly as before" | no |
| 118-120, 197-199 | definitions: "The background, lighting and framing of <Picture 1> are not present in the target video." | the guide's definitions say what a reference provides, never what it does not; the generator's own comment at `_ref_prompt` says the same of its negative clauses |
| 123-125, 203-205 | retention: "only the setting changes"; "the body, the clothing and the setting are the scene's own" | the marker and what is retained, without the second half |
| 210-212, 214-217, 229-230 | head role: "sits on that person's neck"; "the body beneath it"; "the body carries on moving exactly as before" | no: "that person" is somebody the model was never told about |
| 218 | "none of the original hair remains" | no |

Several rows are in both sorts: the close and the scene sentence assert
what the code cannot know AND speak of an edit.

**The same sort applies to texts typed by hand**, which are outside this
audit. Tonight's carry "a scene that is already lit, framed and cut",
"the hair beside and above her face is the scene's own and stays as it
is", and "that woman is not <Subject 1> and keeps her own face, hair and
clothes".

**One case under test as this is written, the lead session's.** After a
note of this session's that a face text named no telephone
(`2026-10-10_preprocessing_data_read_by_a_second_session.md`, finding 7),
the lead session added a sentence saying the handset, its cord and the
hand "are the scene's own and stay as they are". By its account the
subject's mouth is shut on every frame of that piece in three arms, and
open in the other half of the same shot on the text without the sentence;
the arm with only that sentence removed had not been read. The note was
right that the text was silent about the handset; the sentence that
filled the gap was in this register. Not a result until the arm is read,
and one piece of one clip then.

## The sentences that should stay in code, each with its reason

- **The six section names and their order.** The guide requires them and
  `bench/check_prompt_guide_conformance.py` grades them.
- **Labels and what each reference is for** (`<Subject 1> is the ... shown
  in <Picture 1>`, the retention markers, `_IMAGE_ROLE_PROSE`). The node
  or the graph knows what is wired; a person retyping it gets a label
  wrong, and a label with no role carries no meaning.
- **The `<Video 1>` relationship**, written only when a motion reference
  is on. A text that names a video nobody wired, or leaves a wired one
  unnamed, has happened here; the code knows which it is.
- **`MOVES`: "moves as the person in <Video 1> moves, at the same
  moments."** It is the relationship and nothing else; the comment above
  it records the render where a list of movements was read as an
  instruction.
- **The pronouns, read off the user's own words.**
- **The keyframe alignment sentence with its seconds computed.** It is the
  guide's own sentence, and a typed duration disagrees with the graph the
  moment a length changes.
- **The voice sentences, behind the choice**, once trimmed to what the
  choice says.

## What this does not show

- That any sentence here changed a render. No pair was rendered.
- What the sound fields should say. The owner and the lead session read
  the guide's sound sections together the same night: the guide writes a
  reused track as a relationship to an `<Audio N>` label, which this lane
  does not wire, so no replacement wording is proposed; the lines are to
  come from what is measured of each video's track. Any wording is the
  owner's to approve, and a change to a shipped prompt needs the prompt
  checks and a rebuild.
- That edit-talk harms a render. It is a sort by register, with one case
  under test.
- Anything about `bench/` or `standalone/` beyond a grep for the section
  names.
