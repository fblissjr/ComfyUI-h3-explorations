# What can be made here: a library of uses

last updated: 2026-10-10 (the body-mesh row: this pack's own nodes); 2026-10-05 (first written; what each unshipped use needs, added the same day)

Written by hand, at the owner's ask (2026-10-05: "a library of potential
uses given what we have done here and can do here holistically with
everything we've built and with what minimax h3 itself can do with it").
One row per thing a person can make: what H3 itself does for it, which
nodes and graph do it today, its state, and where the evidence is.

This page is a map, not an authority. It carries no measurement and no
verdict: a record named in the last column holds those. A graph named here
exists under `workflows/` with `_api.json` after its name; whether it is
still the best one for the job is the owning document's to say. Where this
page and the code disagree, the code is right.

**State**, as used below:

| word | meaning |
|---|---|
| shipped | a graph in the shipped set (`workflows/h3_config.py::GRAPH_DIRS`) does it |
| probe | only a probe or experiment graph does it |
| built | the nodes exist and are registered; no shipped graph uses them |
| wireable | the sockets exist today; nobody has rendered it |
| proposed | an idea with no code |
| closed | parked, declined or ruled out; the row says where |

**What H3 gives, in one paragraph.** The release trains three tasks: text
to video with sound (t2va), video between keyframes (fl2va, which also
serves a first frame alone, a last frame alone and no frame at all), and
reference to video (ref2va), where stills, videos and audio clips are
labelled in the prompt and each is given a role. Everything else on this
page is one of those three, or one of two mechanisms core exposes on top of
them: a noise mask per stream and per token, which freezes part of a latent
while the rest is generated, and guide rows. `../prompting.md` section 1
is the authority on the modes; `../h3_references.md` on references;
`../h3_audio_freeze.md` section 1 on the mask.

## 1. From text and stills

| use | what H3 does for it | nodes and graph today | state | evidence |
|---|---|---|---|---|
| A clip from text alone, with sound | t2va | `MiniMaxH3Conditioning`, `MiniMaxH3Resolution`; `h3_text_to_video`, `daily/h3_t2v_pdd8_flashgen_finish` | shipped | `../h3_distills.md`; `bench/results/2026-09-27_finisher_grid.md` |
| Scripted dialogue between speakers; singing; text on screen | the prompt's speaker ids and `<d>` blocks; no node | `h3_text_to_video_dialogue` | shipped | `../prompting.md` sections 5 and 6; `bench/results/2026-09-19_dialogue_transcription.md` |
| A clip that starts on a given still | fl2va with a first frame | `MiniMaxH3Conditioning` with the canvas taken from the keyframe; `h3_first_frame_to_video`, `daily/h3_i2v_pdd8_flashgen_finish` | shipped | `../h3_geometry_and_nodes.md`; `bench/results/2026-09-26_flashgen_tasks_s1.md` |
| A clip between two stills | fl2va | `MiniMaxH3Conditioning`; `h3_first_last_frame_to_video` | shipped | `../prompting.md` 3.5 and 10.3 |
| A clip that ends on a given still | fl2va with a last frame only | `MiniMaxH3Conditioning`; `h3_last_frame_to_video` | shipped | `../prompting.md` 10.4 |
| Portrait, ultrawide and square shapes; long clips | the trained family of canvases and the frame grid | `MiniMaxH3Resolution`, `MiniMaxH3Preflight`; `h3_probe_canvas_portrait`, `h3_probe_canvas_ultrawide`, `h3_probe_square_canvas` | probe | `../h3_resolutions.md` |

## 2. From references

The reference chain is ordered: each Append node adds one still, video or
audio clip, and the prompt names it by its label.

| use | what H3 does for it | nodes and graph today | state | evidence |
|---|---|---|---|---|
| A person, several people or a place from stills, in a new scene | ref2va, `<Picture N>` and `<Subject N>` | `MiniMaxH3AppendRefImage`, `MiniMaxH3ReferenceConditioning`; `h3_image_ref_plus_text_to_video`, `h3_ref2v_market`, `daily/h3_ref2v_pdd8_flashgen_finish` | shipped | `../h3_references.md`, "The four reference types"; `bench/results/2026-10-03_r2v_finish_time_budget.md` |
| A clip that follows a reference video's camera, cuts and pacing | ref2va, `<Video N>` as a weak reference | `MiniMaxH3AppendRefVideo`; `h3_ref_video_only`, `h3_ref_video_to_video` | shipped | `../h3_references.md`, "Follow camera movement, cuts and rhythm only" |
| The movement of a reference video on a subject from a still | ref2va, attribute transfer from `<Video N>` | the same nodes; `h3_ref_video_motion` | shipped | `../h3_references.md`, "Transfer motion onto a different subject" |
| Edit a video by handing it in as a reference (change a garment, keep the rest) | ref2va, the video partly preserved. The whole frame is regenerated; nothing is kept exactly | the same nodes; `h3_ref_video_edit`, `h3_ref_video_image_edit` | shipped | `../h3_references.md`, "Edit a source video" |
| Replace a character, with the video as the plate | ref2va, a video and one still | the same nodes; `h3_ref_video_swap` | shipped | `../h3_references.md`, "Replace a character, keeping the video as the plate"; `bench/results/2026-08-22_swap_prompt_verdict.json` |
| Continue from the end of a video | ref2va, the video as a continuation source | the same nodes; `h3_ref_video_continue` | shipped | `../h3_references.md`, "Continue from the end of a source video" |
| Speech in a referenced voice | ref2va, `<Audio N>` referenced, not copied | `MiniMaxH3AppendRefAudio`; `h3_ref_audio_voice` | shipped | `../h3_references.md`, "Reference a voice" |
| Music in the style of a referenced track | ref2va, `<Audio N>` referenced | the same node; `h3_ref_image_audio` | shipped | `../h3_references.md`, "Reference a music style" |
| A video reference with its own soundtrack carried over | ref2va, the video's audio as part of the reference | `h3_ref_video_audio`, `h3_ref_image_video_audio` | shipped | `../h3_references.md`, "Video references" |
| References on the keyframe checkpoint, or seen by the text encoder only | the encoder's view of a reference without the video model's copy | `use_vae` on the Append nodes; `h3_probe_ref_pathway_fl2va_encoder` | probe | `../h3_references.md`, "Encoder-only references"; `bench/results/2026-10-03_encoder_only_reference.md` |
| Encode the references once and change only the prompt | none; a split of the conditioning | `MiniMaxH3EncodeReferences`, `MiniMaxH3PromptOnReferences` | built | `../h3_references.md`, "Encoding references apart from the prompt" |

## 3. To a known audio track

The track's audio rows are written into the latent and frozen, the video is
generated against them, and the original waveform is put back at the end.
Not a trained task.

| use | what H3 does for it | nodes and graph today | state | evidence |
|---|---|---|---|---|
| A video that follows a given song or speech, the audio kept exactly | the per-stream mask, audio frozen | `MiniMaxH3FreezeAudio`; `h3_text_to_video_audio_freeze` | shipped | `../h3_audio_freeze.md` sections 2 and 4; `bench/results/2026-09-12_audio_freeze_step2_verdict.json` |
| The same, starting on a given still | a first frame plus the frozen audio | `h3_first_frame_to_video_audio_freeze` | shipped | `../h3_audio_freeze.md` section 4 |
| The same, with the track also shown to the model as guide rows | core's guide rows plus the freeze | `h3_candidate_t2v_pdd8_baked_audio_freeze_guide` | shipped | `../h3_audio_freeze.md` section 5 |
| A whole song as one video, in joined windows | the freeze per window, with the end of each window's video frozen as the start of the next | `MiniMaxH3AudioFreezeSong`; `h3_text_to_video_audio_freeze_song_pdd8` | shipped | `../h3_audio_freeze.md` section 4; `audio_freeze_song.py` |
| A whole song with one subject held by a still | reference stills given with every window's prompt | the song node's `references`; `h3_text_to_video_audio_freeze_song_ref_pdd8` | shipped | `../h3_audio_freeze.md` section 4 |
| A different place or action per section of the song | none; the prompt is filled from lists, and the song node takes a timeline of sections | `MiniMaxH3PromptList`; `h3_text_to_video_audio_freeze_song_lists_pdd8` | shipped | `prompt_lists.py`; `../h3_audio_freeze.md` section 4 |
| Prompt variations from lists on any graph | none | `MiniMaxH3FillPromptLists` | built | `prompt_lists.py` |

## 4. Changing part of a video that exists

A mask over part of each frame: the masked tokens are generated and
everything else is the source's own pixels. Built on the song node, so the
source's audio is kept too. Not a trained task: the release's own edit is
the reference route in section 2, which regenerates the whole frame. [`masked_v2v.md`](masked_v2v.md) is this lane's
page and has its limits; the board the owner keeps for it lists every idea
below with what would accept or reject it.

| use | what H3 does for it | nodes and graph today | state | evidence |
|---|---|---|---|---|
| Replace one person with the person in a still, everything else kept | the per-token mask plus a reference still | `MiniMaxH3SubjectTrack`, `MiniMaxH3MaskedSource`, the song node; `h3_video_to_video_masked_song_pdd8` | shipped | [`masked_v2v.md`](masked_v2v.md); `bench/results/2026-10-04_masked_v2v_band.md` |
| A matte of one person through a clip with cuts, for use anywhere | none; SAM 3 from core | `MiniMaxH3SubjectTrack` alone: a mask, a labelled preview and a report | built | `bench/results/2026-10-04_subject_track_three_clips.md` |
| Replace only the head and hair, the body kept | the same mask, cut to a part | `replace` on the Masked Source | built; oversized head on a long-haired original | `../h3_audio_freeze.md` section 4 |
| Give the model the original's movement without naming it | the source, with everything but the subject greyed, as `<Video N>` | `motion_reference` on the Masked Source | built; first renders pending | `../research/masking/2026-10-05_mryellow.md` |
| Masks of hair, face, clothing, hands or mouth on the tracked person, and a soft-edged matte | none; Sapiens2 | `MiniMaxH3Sapiens2Loader`, `MiniMaxH3SubjectParts` | built | `bench/results/2026-10-05_sapiens2_first_frame.md` |
| Replace one garment, or only the hair | the same mask, from a part | the parts mask wired into the Masked Source's `mask` | wireable | none |
| A masked render through two samplers | the mask, with the kept part restored between them | `MiniMaxH3RestorePlate` | built; not run on H3 | `plate_restore.py` |
| Reshoot one shot, the rest untouched | the mask over whole frames of one shot | a shot or frame range on the Masked Source | proposed | the board |
| Keep the person, replace the world | the mask inverted, a still of the setting as the reference | an inverted `replace` and composite | proposed | the board |
| Redub: a new vocal on an existing video | the mask on the mouth, the new track frozen | the parts node's mouth mask, the song node | proposed | the board |
| Remove a person; the original's shadow gone | no route | VOID was tested on 2026-10-05 and removed | ruled out | `../../bench/results/2026-10-05_void_plate_turn.md` |
| Two people replaced from two stills | two masks, two labelled stills | two Subject Track nodes and a union | proposed | the board |
| A thing that is not a person | the same mask | a mode on the Subject Track without the head comparison | proposed | the board |
| Correct one shot by hand | none: it is the mask that is corrected | `MiniMaxH3SubjectTrack`'s `corrections`: `shot 3: person 2` or `shot 3: none`, with the numbers the preview shows | built | `../../subject_track.py`, "A correction"; `../../bench/check_subject_track.py`, item 7. Not yet run on a clip |
| One review per clip before any render | none | a dry run that emits masks, parts, shots and captions | proposed | the board |
| Movement from a body mesh of the original, with none of its look | the mesh rendered as the video reference | `MiniMaxH3BodyModelLoader`, `MiniMaxH3BodyPose` on `MiniMaxH3SubjectBoxes`' boxes, `MiniMaxH3BodyMeshVideo` (`../../body_pose.py`): the crop sampled as Meta's code samples it, no ComfyUI SAM node in the graph, and a table of what each hand's decoder did | built (2026-10-10); no shipped graph wires it yet | `../../bench/results/2026-10-10_sam3d_body_core_against_meta.md`; `../../bench/check_body_pose.py`; [`meta_perception_models.md`](meta_perception_models.md) |
| Movement from per-shot captions a model writes and the user can edit | the prompt, written by machine | core's text generation over the loaded encoder | proposed; it is still prompting | the board |
| Movement from a control adapter | a trained mask-and-source path | none | closed: declined by the owner | [`decisions.md`](decisions.md), 2026-10-04 |
| Movement from a late start on the schedule | the source showing through at the first step | none in the tree | closed: carries the original's look with its pose | `bench/results/2026-10-04_masked_v2v_turn_soft_arms.md` |

## 5. After a render

| use | what H3 does for it | nodes and graph today | state | evidence |
|---|---|---|---|---|
| Make the sound again and keep the picture exactly | the per-stream mask the other way: video frozen, audio reopened | `MiniMaxH3AudioRefineMask`, `MiniMaxH3FrozenVideoCache`; `distill_experiments/h3_probe_t2v_pdd8_audio_refine` | probe; refines a latent sampled in the same graph | `bench/results/2026-09-25_distill_audio_s1.md` |
| Decode a saved latent again without sampling | none | core's nodes; `distill_experiments/h3_decode_saved_latent` | probe | `bench/results/2026-09-26_vae_decoders_345f.md` |

## 6. Combinations the pieces allow

Each of these needs no new node: the sockets are there. None has been
rendered, so each is a guess about what the model will do, and the row says
which sockets make it possible.

| use | what makes it possible | state |
|---|---|---|
| A masked swap whose prompt changes by section of the song | the song node takes `source` and `lists` together | wireable |
| A masked swap with more than one still of the new person | the song node's `references` takes a reference chain; one still is all that has been rendered | wireable |
| A new garment on the original performer | the parts node's clothing mask into the Masked Source, a still of the garment as the reference | wireable |
| The song as a reference as well as frozen rows | the hybrid in `../h3_audio_freeze.md` section 5; no graph | proposed |
| Continue a finished render | three mechanisms: the reference route in section 2, the song node's frozen prefix, and guide rows (`../research/2026-09-25_continuation_guide_rows.md`, not built) | shipped, shipped, proposed |

## 7. Speed and quality, across every use

These are not uses; they change what a use costs and how it looks.
`../h3_distills.md` says which suits which kind of shot.

| option | what it is | where |
|---|---|---|
| PDD8, and its faster rungs | a few-step distill for the keyframe and reference checkpoints | `MiniMaxH3PDDLoRA`; `../h3_pdd.md` |
| The PDD8 baked checkpoint | the distill built into the weights; the song and masked lanes run on it | `workflows/h3_config.py::MODELS`; `../h3_distills.md` |
| FlashGen, and PDD8 finished by FlashGen | a second distill, applied at the call; the daily graphs use the pair | `MiniMaxH3LoRABranch`; `../research/2026-09-26_flashgen.md` |
| FastH3 | its own checkpoint, text to video only | `../h3_distills.md` |
| Sol-Attn | sparse attention, on in the shipped video graphs | `MiniMaxH3Sol`; `../SOLATTN.md` |

## 8. What each unshipped use needs

One line each, for the rows above that are not shipped. A probe is listed
only where something more than promoting its graph is needed.

| use | what it needs |
|---|---|
| Encode references once, change only the prompt | a shipped graph that uses the two split nodes |
| Prompt variations from lists on any graph | a graph outside the song lane that uses the fill node |
| The tracker's matte on its own | nothing to use it; a clip it has not seen before it is trusted on one |
| Head and hair only | a region that does not oversize the head; an original with short hair to try it on |
| The original's movement as a video reference | its first renders, against the generic-prompt control |
| Parts and a soft matte | a graph that consumes them: the parts mask as the Masked Source's mask, or the matte in its composite |
| One garment, or only the hair | one render; the wiring exists |
| A masked render through two samplers | a two-sampler masked graph, and a run on H3 |
| Reshoot one shot | a shot or a frame range as the Masked Source's region, from the cut finder the tracker already has |
| Keep the person, replace the world | the region and the composite inverted on the Masked Source; a still of the setting |
| Redub | the mouth mask wired as the region with a new track frozen; one render to see whether a small hole holds |
| Remove a person, and the shadow | no route: VOID was tested on 2026-10-05 and removed (`../../bench/results/2026-10-05_void_plate_turn.md`) |
| Two people from two stills | two picks and a union of their masks |
| A thing that is not a person | a choice on the Subject Track that turns the head comparison off |
| Correct one shot by hand | built on 2026-10-05 (section 4); a person the detector never found still needs points or boxes passed to it |
| One review per clip before a render | a path that stops before sampling; the tracker's preview and the song node's `preview` are two parts of it |
| Movement from a body mesh | a node that renders the mesh over the frames, then one render |
| Movement from captions | a node that captions each shot and assembles the prompt, shown for editing |
| Make the sound again on a finished clip | a graph that starts from a saved latent, not from a fresh sample |
| The combinations in section 6 | one render each |

## 9. Not available here

So that nobody plans on them. `../roadmap.md`, "Closed lanes", is the list
and the reasons.

- Single-frame image generation and editing: parked; `../h3_image_editing.md`.
- Keyframes at arbitrary points inside a clip: not built. Core's guide rows
  are used in one graph, for audio at the start.
- Upscaling or frame interpolation of a finished video: nothing here.
- TaoMate-H3, PDMD and the Turbo LoRAs: retired.
