# Sol `video_reference_rows`: shipped against routed, on one `<Video 1>` render (2026-11-17)

**Question.** Step 1 of "Landing `video_reference_rows`" in [`SOLATTN.md`](../../docs/wiki/attention/SOLATTN.md):
on a ref2va graph with a `<Video N>` reference, does routing only the video-reference query rows through Sol's sparse
path (`h3_config.SOL_ROWS_VIDEO_REFERENCES_ROUTED`) hold up against the shipped Sol settings? Picture and voice are the
owner's call on playback. This record holds the sampler times and where the renders are.

**Answer.**
- **Sampler time:** routed is faster, by the margin in the table below.
- **Picture and voice:** the owner saw no difference between the arms on playback, the same day.
- **Outcome:** routed became the default on every shipped Sol graph (`0e43de1d`, the owner's word: keep the faster).

## The runs

**Graph.** `workflows/h3_ref_video_audio_api.json` (base ref2va int8 checkpoint, 16 steps, euler, simple,
`MiniMaxH3SigmaShift` 12 / 3, seed as shipped) on the running `:8188`. The graph attaches no LSE reference balance,
so both arms run Sol on every block but 49. Only these inputs changed, the same in both arms:
- `VHS_LoadVideoFFmpeg.video`: `exp8_sunny_room_door_siren_guide_f1_s1.65_s20261112_00001-audio.mp4` from
  `$H3_INPUT_DIR`, the owner's pick. It is 1344x768, 24 fps, 124 frames, 32 kHz stereo.
- `MiniMaxH3Resolution`: `shape="wide"`, `1344x768`, `length=141`, which is on both clocks.
- `MiniMaxH3ReferenceConditioning.prompt`: the prompt below. It was written for the clip.
  `bench/grade_prompt_text.py --mode ref2va --length 141 --like h3_ref_video_audio_api` reports 0 FAIL, 0 WARN.

**The arms.** The only difference between them is the Sol node's `rows` inputs:

| arm | Sol `rows` | server log, `[h3-sol] MiniMaxH3Sol on:` |
|---|---|---|
| A, shipped | not set | `token routing on 0 block(s), dense blocks [49], sink exact_kv_and_all_rows` |
| B, routed | `SOL_ROWS_VIDEO_REFERENCES_ROUTED` | the same, plus `rows {'text': 'exact', 'reference': 'exact', 'audio': 'exact', 'video_reference': 'routed'}` |

The preflight line prices the sequence at 87,493 tokens: 42,336 target video rows and 37,296 `<Video 1>` rows.

## Times

From the server log (`journalctl -u comfyui.service`, the sampler's progress bar and "Prompt executed"):

| arm | sampling, 16 steps | per step | prompt executed |
|---|---|---|---|
| A, shipped | 10:11 | 38.20 s | 00:10:54 |
| B, routed | 9:00 | 33.75 s | 551.14 s |

Arm A also ran the encoders and loaded the models. Arm B reused A's cached conditioning, so only the sampler rows
compare.

## Where the arms differ

Per-frame PSNR between the two arms' videos (`ffmpeg -lavfi psnr`). It measures how far the arms are from each
other, not which one is better:

| frames | min dB | mean dB |
|---|---|---|
| 0–16 | 40.8 | 44.3 |
| 17–33 | 42.8 | 44.4 |
| 34–50 | 41.5 | 43.1 |
| 51–67 | 43.2 | 44.4 |
| 68–84 | 42.9 | 43.8 |
| 85–101 | 39.0 | 43.3 |
| 102–118 | 35.7 | 38.6 |
| 119–135 | 41.3 | 43.1 |
| 136–140 | 44.3 | 44.6 |

## The files

All under `$H3_OUTPUT_DIR/Video/sol_video_ref_rows_ab_2026-11-17/`:
- `armA_shipped_00001-audio.mp4` and `armB_video_refs_routed_00001-audio.mp4`, each arm with its own sound;
- `stacked_A_over_B_case1.mp4`, A above B, each labelled, with the frame number drawn on, carrying arm A's sound.

A second case (`02_ref2va_film16mm_default_00001-audio.mp4`, two spoken lines) was prepared and dropped at the
owner's word before it ran.

## What this cannot see

- One graph, one seed, one prompt, one reference clip, one length.
- The base 16-step ref2va graph, not Pika. No shipped graph pairs Pika with a `<Video N>` reference.
- The reference voice is a whisper, so this is a weak test of carrying a voice over.
- The harness (`sol_video_ref_ab.py`) was a throwaway in the session's scratchpad and was not kept. It changed only
  the inputs listed above and submitted arm A, then arm B, to `/prompt`.

## The prompt

```text
subject_definitions:
<Subject 1> is the woman in <Video 1>, whose face, long wavy blonde hair, black knit sweater, and brown hardcover book are carried into the target video.
<Audio 1> is the synchronized audio track of <Video 1>; its door knocks, the whispered line of <Subject 1> (S1), and the distant siren are reused in the target video.
<Video 1> is the source of the shot's locked-off medium framing, the sunlit room, its light, and the pacing of the moment.

summary:
[reference generation + audio reuse] The target video is a single static medium shot of <Subject 1> standing in the sunlit room of <Video 1>; knocking at an unseen door makes her turn toward it and whisper a warning, and a siren rises in the distance, with the knocks, the line, and the siren taken from <Audio 1>.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - face, hair, sweater, and the book she holds are retained from <Video 1>.
<Video 1> (framing, room, light, and pacing): partially_preserved - the locked-off medium framing, the room, and the warm side light are kept, while the shot runs slightly longer than <Video 1>.
<Audio 1>: partially_copy - the knocks, the whispered line, and the siren are copied in their original order, and quiet room tone carries the end of the shot.

detailed_description:
The target video is in a cinematic live-action style, lit by warm, low late-afternoon sunlight with soft amber tones.
[Shot 1] The camera holds a static medium shot throughout the entire five-point-nine-second duration, framing a quiet living room. <Subject 1>, a young woman in her twenties with long, wavy blonde hair falling past her shoulders, wearing a black knit sweater, stands just left of center from the waist up, holding a brown hardcover book low against her body with both hands. Her voice, when she speaks, is a soft, breathy whisper of medium-high pitch, slow and tense. Behind her, low sunlight from the left lays a bright rectangular window pattern across a pale wall; a framed black-and-white print hangs on the wall to the left, a wooden bookshelf filled with colorful book spines, pale ceramic vases, and a vase of dried grasses stands on the right, and the corner of a gray sofa with a cushion sits at the lower left edge of the frame. The sunlight catches the loose strands of her hair and leaves the right side of her face in soft shadow, and the room behind her stays slightly soft in focus while she remains sharp. At the opening of the shot she is looking slightly toward the right side of the frame, lips parted, as if she had just been reading. In the first second, two heavy knocks sound from a door off-screen to the left, taken from <Audio 1>, and <Subject 1> freezes, her eyes widening and her shoulders tensing. She turns her head toward the left edge of the frame, toward the sound, her grip tightening on the book so that her knuckles press into its cover, and <Subject 1> (S1) whispers, <d>[English] Wait. Someone is right outside.</d> Her lips then close and her jaw stops moving, and she stays still, listening, her gaze fixed on the left edge of the frame and her breathing shallow and silent. A distant emergency siren from <Audio 1> begins to rise outside and grows steadily louder until the end of the shot. She does not move toward the door or speak again; she produces no further vocal sound, her body stays turned toward the camera while only her head faces left, and the warm light on the wall and the shelves behind her stays unchanged through the final frame.

overall_soundscape:
Quiet indoor room tone; two heavy knocks on a wooden door off-screen to the left at the start; a single whispered line; then a distant emergency siren rising outside and holding until the end, all taken from <Audio 1>.

non_diegetic_music:
N/A
```
