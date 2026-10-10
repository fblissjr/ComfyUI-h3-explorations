# The preprocessing data of twelve renders, read by a second session (2026-10-10)

lane: masked video to video
verdict: not judged as video. Twelve loads' masks, parts, mesh videos, texts and frame arithmetic were read from their files by a session with no tool in the pipeline, after `verify` had passed and the gate was clear on both captures. Four loads have data that is wrong on a run of frames, one is unsure on five frames, one writes a frame too many, four have a text or a region that leaves out something in the picture (two of them among those already counted), and four were found without a fault. None of the faults was at a level that blocks. One clip, one evening, stills and counts only.

**Read this first.** Nobody played a clip for this, and nothing here says what
a render will draw. It says what the files a render reads hold, frame by
frame, against the source picture. "By eye" below means a still looked at; a
still cannot show what must hold over time. Where a load was looked at on
every second or fourth frame and not on every frame, the table says so.

## What was asked

The owner, relayed by the lead session: the preprocessing data is to be
verified straight after it is made, and "not blindly": by eyes as well as by
code, by someone independent, "so we can document gaps in code and data".
This session has built nothing in the masked lane; the lead gave it the
data and not its own conclusions.

## What was read

Two captures made by `bench/capture_masked_run.py` from two no-sampling
previews built by `bench/masked_job.py`:
`data/2026-10-10_fun_rest_a4_1039_capture` (source frames 1039-1465) and
`data/2026-10-10_fun_rest_b2_1466_capture` (1466-1919), both untracked. For
each of three subjects (`lead`, `third`, `second`): the track video, the
part video a face pass takes (`parts_held`), the class map. For each load:
`<load>.render.built.json` and `<load>.render.json` in the two job folders
beside the captures, the load's plan file in its windows folder, the
region the node planned (`runs/<load>/region.npz`), and for the five
whole-person loads the mesh video. Also the gate's `flags.json` and
`outcomes.json`, `verify.json`, the two job files and the lead session's
reading of who is in each shot
(`data/2026-10-10_fun_rest_who_is_where_truth.json`). The source was fitted
as the loader fits it (a centred crop to 1424x1068, then 1024x768).

## How it was looked at

- **Counts, on every frame of both captures (881 frames):** each subject's
  track and part in pixels, its overlap with its own previous frame, the
  share of the part inside its own track, the part's pixels on another
  subject's track, the pixels two tracks share.
- **Overview sheets, every second frame of both captures:** the source with
  the three tracks tinted and the lead's part filled.
- **Close-ups of the lead's head, part as an outline only:** `lead_1134`
  on every frame; `lead_1061`, `lead_1093`, `lead_1156` and `lead_1306`
  on every second frame; `lead_1222` on every fourth; `lead_1466` on every
  second frame to 1529, every frame of 1530-1586, every third after.
- **Mesh beside source, the subject's track outlined on both:**
  `second_1039` every second frame, `third_1222` every sixth, `third_1306`
  every fourth and every frame of 1403-1418, `third_1419` every second,
  `third_1466` every fourth. And the mesh's silhouette against the track on
  every frame, at offsets of zero, one and two frames each way.
- **Plain frames with outlines only,** every frame of 1407-1418.
- **A load's region laid on the source:** `third_1419` every second frame;
  `second_1039` on three frames.

So no load was looked at by eye on every frame from end to end. A jump of
a mask to another person and back would show in the counts as a broken
overlap with the frame before; none did inside a load.

The figures are in `data/2026-10-10_fun_rest_second_read/` (untracked), with
the one-off scripts that made them. The sheets themselves were made in a
scratchpad and are not kept.

**What this reader had seen, and when.** The overview sheets and the
close-ups of `lead_1061` to `lead_1156` were read before the gate's flags.
To know what a job leaves out on purpose the job files had to be read, and
`outcomes.json` was read with them, so everything after the first two
findings below was read knowing the lead session's override notes.

## Load by load

| load | frames | kind | found |
|---|---|---|---|
| `lead_1061` | 1061-1092 | face | no fault. The part is on her face on every frame looked at; on 1067-1069 her own arm crosses it and the part is what shows above the arm; one stray speck of the part on her wrist on 1087 |
| `lead_1093` | 1093-1133 | face | **unsure on 1093-1097.** Her hair is thrown over her face; the part (6,000 to 12,000 px) lies on a dark area under the hair where no face can be seen, and on 1097 on a blur. On her face from 1099 |
| `lead_1134` | 1134-1155 | face | **wrong on 1142-1155.** Finding 1 |
| `lead_1156` | 1156-1188 | face | the part is on her face. **The plan writes 34 frames for a load of 33** (finding 6), and the text does not name the hat she puts on (finding 7) |
| `lead_1222` | 1222-1305 | face | no fault |
| `lead_1306` | 1306-1402 | face | no fault. A second small piece of the part sits on her earring on about 1360-1394; the part shrinks over 1396-1402 as she turns and is still on her face |
| `second_1039` | 1039-1060 | whole person | the track is on her. The mesh shows a whole body where a table hides her below the chest, and she moves too little for the data to show which frame the mesh is on. **The region takes the things on the table in front of her** (finding 8) |
| `third_1222` | 1222-1305 | whole person | no fault. The mesh follows him; offset zero is the best fit |
| `third_1306` | 1306-1418 | whole person | **wrong on 1411-1418.** Finding 4 |
| `third_1419` | 1419-1465 | whole person | **wrong on about 1425-1461.** Finding 3 |
| `lead_1466` | 1466-1729 | face | **wrong on 1536-1574.** Finding 2. Also finding 7 |
| `third_1466` | 1466-1631 | whole person | the track is on him and the mesh follows him as he turns and walks away. On 1627-1631 he is a strip passing a doorway and the mesh is a whole small body, so the two barely overlap. The text has him standing beside the lead for the whole load |

## The findings

**1. `lead_1134`: from 1142 the part is present and is not a face.** She
turns away at 1142. The gate's top-level flags name the two frames where
the part is empty (1145-1146). On the others:

| frames | the part | of it inside her own track |
|---|---|---|
| 1134-1141 | on her face, 6,900 to 14,900 px | 0.78 to 0.91 |
| 1142-1143 | a sliver at the edge of her face, 3,700 then 1,200 px | 0.65, 0.45 |
| 1144 | a loop of 280 px on her outline | 0.004 |
| 1145-1146 | empty | |
| 1147-1151 | 130 to 1,400 px on her hair, her outline and the wall behind | 0.0 to 1.0 |
| 1152-1155 | 2,000 to 3,700 px at the front of her turned head; a speck on her forearm on 1154-1155 | 0.78 to 0.95 |

**2. `lead_1466`: thirty-six frames of pieces while her back is to the
camera.** On her face to 1535; a sliver on 1536; empty on 1537; on
1538-1573 pieces of 240 to 5,100 px. On 1538-1557 most of them are loops
on the dark red band between her hair and her collar, which is clothing or
the nape and is inside her own track, so "inside her track" does not see
them. On 1558-1573 the part is a narrow strip along the edge of her head
beside the third subject, and on 1566-1568 none of it is inside her track
and most of it is on his. On her face again from 1575. Where her face
shows the part is 9,000 to 79,000 px. From about 1636 she holds a
telephone handset across her lower face; the part goes round it.

**3. `third_1419`: his region takes the lead's body, which no file
holds.** In this shot the lead's track is her head alone (4,500 to 26,500
px). Her class map labels only what is round her head, and the owner map
calls hers only what her track holds. His region is 1,050 to 1,480 of the
frame's 3,072 cells on every frame, with 130,000 to 230,000 px of it
outside his own track. By eye, on every second frame: part of her arm and
side on 1425-1427; most of her visible body on 1437-1457; all of her but
her head on 1447-1453, where she stands in front of him and his track
rightly has a hole that his region fills; clear of her by 1463. The gate's
`region_over_another_subject` for this load measures against her mask,
which is her head, and its override rests on an owner map that can give
back only her head.

**4. `third_1306`: on its last eight frames his track and his mesh take
the lead's raised arm.** On 1403-1410 his track is on him. From 1411 an
arm rises behind his head and stays up to 1416; it runs into the striped
sleeve of the lead's top on 1413-1416 and has her red cuff at the wrist on
1413, and his own shoulders are at the left with his arms down. The two
tracks share 10,000 to 47,000 px on 1411-1417 and nothing before 1411 but
a few pixels. On 1412-1416 his mesh has an arm stretched up and to the
right where her arm is. Fairly sure, not certain: 1411 and 1412 are
blurred and show only skin.

**5. Small things about the parts that are not faults by themselves.**
A piece of the lead's part on her wrist (1087) and on her earring (about
1360-1394). On 1576-1585 between three and seven in a hundred of her part's
pixels are also inside the third subject's track, along her profile.

**6. `lead_1156` writes a frame past its shot.** Its plan file gives
`frames_written` 34; the load is 33 frames, 1156-1188, and its loaders are
capped at 33. Frame 1189 is the first of the next shot, which has nobody
in it. Every other load's plan writes the load's own count. The gate's
`region_shared_across_a_cut` named 1186-1189 for this load at "iffy".

**7. Two texts leave out something the region holds.** `lead_face_hall`
is used for `lead_1156`, where she puts on a hat over 1157-1163 and wears
it to the end; the text says the hair above her face is the scene's own
and stays, and names no hat. `lead_face_night` is used for `lead_1466`,
where a telephone handset crosses her lower face from about 1636; the text
names no telephone. `third_whole_hall` has him standing beside the lead in
a hallway on all four of his loads; on `third_1466` he walks away into the
room behind from about 1534.

**8. `second_1039`: things on the table are inside the region.** With a
margin of 64 px about two thirds of the region is not on her track
(122,000 to 132,000 of 184,000 to 199,000 px). On the three frames looked
at it covers a bunch of flowers in a vase, a bowl and a dish in front of
her, and wall and cupboards behind. The text names a kitchen table and
none of them. This is the fault `data/CAPTURE_GAPS.md` records from the
same day's first job.

**9. Frame arithmetic, stills, audio.** For every loader in every graph
the start is the load's first frame, as (frame less 0.4) over 24 seconds
or zero, against the source for the source and against the capture's first
frame for every mask and mesh file; every cap is the load's frame count;
every size is the canvas. The song node's extent is the load's count in
seconds in all twelve. Every mask input is wired to the file the built
record names. Each subject has one still on all its loads, and each still
shows one person of the kind its text describes. The audio of each graph
comes from the same loader node as its frames, so it is the load's stretch
by construction.

**10. The lead session's truth file.** Read against the overview sheets.
Who it names in each shot agrees with what the sheets show, with three
notes. In the first capture's third shot there are several small people at
a table at the back, not one, and I cannot tell which is the second
subject. Its row for the ninth shot describes the tile's frame, where the
lead is a head behind the third subject; from about 1425 she is wholly in
view. In the second capture's last five shots the large person the job
does not name is a different person in each shot. Its tile numbers were
not checked: no tile was opened.

**Not in any load, seen on the way:** the lead's inset moves to its new
place on 1848, one frame before the cut the shot table has at 1849.

## What the code checks, against what was checked by eye

`verify` checks that a capture reads what the nodes wrote: every mask video
of a sighting written in one run, at the canvas, reaching the span; a track
not empty where its shot table says taken; a class map where a plan carries
a part; saved regions against the review reading. It passed on both
captures. It does not look at what is under a mask. The gate's rules are
what could have. "Said here" is what they said on these two captures.

| checked by eye | does any code check it today | said here | what plain data would have said so |
|---|---|---|---|
| a part that is present and is not a face (findings 1, 2) | only an EMPTY part blocks (`part_empty_on_the_subject`). `part_not_visible`, `part_off_the_subject`, `part_in_pieces`, `part_changes_size` are "iffy" | they named all of 1142-1155, and 16 of the 36 frames of 1538-1573: a rule that compares a part with its own recent frames goes quiet once small is usual | the part's size against its size where her face shows in the same load; its share inside her own track; its pixels on another subject's track. All three are in the capture's tables |
| a region over a person whose body no track holds (finding 3) | `region_over_another_subject` measures against the other subject's mask | "up to 68% of lead's mask", top level, overridden: her mask is her head | her track's size in this shot against her track's size in the shots either side; nothing compares them |
| two tracks claiming one limb, and whose it is (finding 4) | the owner map counts contested pixels; `region_over_another_subject` named 1409-1417, top level, overridden | it could not say whose | nothing here can; a person looks |
| a mesh that shows another person's limb (finding 4) | no. `flag_pose` reads the box, the count of bodies, hands and joints outside the frame | nothing | the share of the mesh's silhouette on another subject's track, a frame at a time |
| the mesh is on the same frame as the track | no | nothing | the mesh's silhouette against the track at offsets of a frame or two: zero fitted best on four loads; on `second_1039` she moves too little to tell |
| the mask is on the right person on every frame | no. The shot table, typed corrections and a person's reading | the gate asked and a person answered | the who-is-who decision's file, where it can speak |
| a mask that jumps inside a load | the capture's table has the step between frames; no flag fired on a load | nothing | the same column |
| things inside a region that are not the subject (finding 8) | `segment_inside_region`, for the subject's own classes only, "iffy", 81 flags on one capture | hair, hands and clothing of the subject; never an object | nothing: no object is tracked |
| a text that names what the region holds (finding 7) | `flag_text` reads voices only | nothing | the part model's `Apparel` class for a hat; nothing for a telephone |
| a load that writes past its shot (finding 6) | `region_shared_across_a_cut`, "iffy" | named 1186-1189 | the plan's `frames_written` against the load's frames: two numbers nothing compares |
| loader starts and caps (finding 9) | the builder computes them; nothing reads a built graph back | nothing | the graph itself, as read here |
| the still is that subject's | no | nothing | nothing; a person looks |
| the truth file | no | nothing | nothing |

**What the table says.** Every fault found was either named by a rule at a
level that does not block, or measured against a mask that was itself
short of the person. The data to block on was in the capture in three of
the four. And the eye found two things no table holds: whose arm it is,
and what stands on a table.

## What was done with it

Told to the lead session as each was found. As of this record: `lead_1134`,
`lead_1466` and `third_1419` were pulled from the queue while still
pending; `lead_1466` is to be made again on 1466-1535 and 1575-1729, and
the lead and the third subject stay the source on 1134-1155 and 1419-1465.
`third_1306` was with the lead session to decide.

## What could not be checked, and why

- **Anything in motion, and any sound.** No clip was played and no audio
  was heard.
- **What a render will draw** on the frames called wrong.
- **Whether the loader lands on the frame its start names.** The
  arithmetic was read; no graph was run.
- **Every frame by eye.** See "How it was looked at".
- **Who the small people at the back are,** and the truth file's tile
  numbers.
- **Which frame `second_1039`'s mesh is on.**
- **The `others` masks as the node fits them to the window.** They are the
  other subjects' track files at the same offsets; what the node does with
  them was not run.
