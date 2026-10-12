# The masked lane's page held to the clips and files it names

lane: masked video to video
verdict: no render. Every clip, strip, sheet and data file `docs/wiki/masked_v2v.md` names was opened and its sentence held to it; the pictures hold, and the sentences that had lost to a file, to the code or to a saved graph are corrected in the page: the rows that say "does not hold", and the second table.

2026-10-12, session mrprompt, written at 18:06Z by the clock (`date -u`).
A reading: nothing was rendered, queued or measured again, and no
server was used. It was made while the page was reorganized, at the
owner's word relayed by the lead: "Have them validate against actual clips
where theyre referenced."

## What was done

Each path the page names on the share or under `data/` was listed
(`grep -n -o` over the page for media and data paths), checked for
existence, and opened. A video was read with `ffprobe -count_packets` for
its size and frame count, and a few of its frames were cut out with
`ffmpeg` into a scratch folder outside the tree and looked at. Renders the
page describes without a path were found through the record the page
points at. Frame numbers below are a file's own, counted from zero, unless
the row says source.

The share is `/mnt/hub/ai/img/output/`; `Video/...` below is under it.

## What it cannot see

- A look at a handful of frames says what is on those frames. It does not
  say a level, a floor or a difference: every figure in the page's records
  stands as its record states it and was not measured again.
- The reader is one session and was not blind: it had read the page's
  sentence before it opened the file.
- The owner's verdicts on playback are not in a clip. Where a sentence is
  one, the row says so.
- Faces were looked at as stills scaled down from the canvas. Where an eye
  could not be told open from shut at that size the row says so and the
  sentence is left as its record has it.

## The table

| the page's sentence, in short | the file | what was looked at | holds |
|---|---|---|---|
| the typed arm and the node's wording "are stacked on the output share" ("What it is") | `Video/compare_stacked/body_345_p7_typed_beside_node_over_mask.mp4` | frames 0, 172, 344 | holds. The typed arm is on the left and the node's on the right, each over its own mask (the parts in red, the margin in cyan), the legs the source's in both. Titles are burned in and frame numbers are not; the page now says the layout |
| "The owner watched the pair on 2026-10-07: both good, and not told apart" | the same clip | nothing | not checked: a verdict, which the clip does not hold |
| the reference video is "the size of the canvas, one frame for each frame of the load" | `data/2026-10-11_fun_gate_inputs/second_mesh_with_lids_0014_x3.mp4` beside `gate_p4_0014.json` in that folder | `ffprobe` against the job file's `canvas` and `frames` | holds |
| "A plain grey body model on a black ground", a neutral face that never changes, "its mouth is shut" | the same video | frames 0, 35, 70, 105, 140 | holds on each |
| "over each eye, a white shape with a thin black edge", "an almond where her eye is open, a sliver where it is lowered or shut", nothing for the mouth, brows or pupils, the two reaching "across the bridge of the nose" | the same video, the face cut out at twice its size | frames 0, 35, 70, 105, 140 | holds: an almond on 0, slivers on the others, the two shapes meeting over the nose on each |
| "the factor was three (the drawing script's `--grow`)" | `data/2026-10-11_fun_gate_inputs/draw_lids_on_mesh.py`, the video's name, the job's note | the script's `--grow` argument and the `_x` tag it writes | holds as far as a name and a note say; the factor was not measured off the picture |
| the four places of the fragment, "Copied from the picked render's text" | `data/2026-10-11_fun_gate_inputs/second_hf_0014_p4.txt` | the whole text, each of the four strings searched for | holds, word for word |
| the whole text "is kept in one place" as an example | `bench/examples/masked_job_lid_marks/text.example.txt` | `diff` against the text above | holds: the two files are the same |
| the example "still holds sentences about what "stays as it is" and what is "the scene's own"" | the same text | the whole text | holds |
| the render's graph "as the builder wrote it": the Masked Source on `a video I wire`, `motion_vae` on, the video at the canvas's short edge, `start_from` on `noise`, the song node, Sol, the still appended, loaders for the source, the masks and the reference video | `data/2026-10-11_gate_p4_0014_job/second_hf_0014_p4.render.json` | every node's class and its typed inputs | holds |
| `edge`: "Nothing has rendered with it" | the same graph | the Masked Source's `edge` | does not hold. The graph carries `latent cells`, and its render is on the share. Corrected in the page with the old words beside it |
| the kitchen shot is "one person, seated, her face AND her hair redrawn" | `Video/mrfire/gate_p4_0014/second_hf_0014_p4/second_hf_0014_p4_00001_with_mask.mp4` | frames 20, 100 | holds: the parts coloured below are her face and hair |
| the mask review is "the render on top and, below it, the source with what was regenerated coloured in", its legend naming the mask | the same file | frames 20, 100 | holds: the legend reads "the parts taken" |
| "her lids come down to about the source's level where the source's are lowered" | the same file | frame 100, where the source's are lowered | holds by eye on that one frame. The level is the lid-marks record's and was not measured again |
| the lead's close-up is "her face only" | `Video/mrfire/set_close_1575/lead_1575_lids/lead_1575_lids_00001_with_mask.mp4` | frames 23, 45, 60, 110 | holds: the part coloured is her face, and the hair is the source's |
| her eyes are "open through the long stretch where the source's are shut, and less than wide where the source's are wide" | `lead_1575_lids_00001.mp4` in that folder, with the stretches the close-up record gives in source frames | frames 23, 40, 45, 60, 75, 90, 110, 120 | holds on 23 (open, less than wide) and on 60, 75, 90 and 120 (open). On 40 and 45 her head is down and her eyes lowered, and on 110 they are narrowed: not told from shut at the size looked at. The sentence is left as the record's tables and its readers have it |
| "Her mouth sings" | the same file | frames 23, 60, 110 | holds: her mouth is open on each |
| a reader "found a light streak on each upper lid, over a run of frames in the middle of the shot, in the load with enlarged marks, where the render without marks has a small bright point inside each eye and no streak" | `data/2026-10-11_mrtie_gate_o_read/consecutive_picture_left_eye_0100_0129.png` and `eye_glint_check_out.txt` | the sheet's frames of one eye, both loads | holds on that eye: a bright point in the rows without marks, a light line along the upper lid in the rows with them. The other eye's sheet was not opened |
| the owner's saved lines "are in the local page's untracked `data/fun/poc/answers.jsonl`" | that file | the rows that name a load of that shot | holds that the file is there and has such rows. Which row is the pick was not read |
| the kitchen shot's video was drawn "by the one-off the tool was written from", and the day's points by another | `data/2026-10-11_fun_gate_inputs/draw_lids_on_mesh.py`, `data/2026-10-11_sapiens2_pose_consecutive/run.py` | existence | holds |
| the first zoomed render "ran clean and wrote its zoomed picture and the box beside it"; "the box set the zoom and not the setting"; the files carry `source_size` and `fixed` | `Video/mrpop/fun_street_0438/street_0438_zoom2/street_0438_zoom2_00001_zoomed.mp4` and `_zoomed.json` | the json whole; `ffprobe` of the video | holds: `set_by` is the box and `zoom` is under `largest_zoom` |
| "the face is replaced where it shows and no edge of the box shows" | `Video/mrpop/fun_street_0438/street_0438_source_vs_whole_frame_vs_zoomed.mp4` | frames 5, 40 | holds on both, in stills, as the page says of the lead's own look |
| a render leaves `<name>_window_N_region.npz` beside each window, and a preview `<name>_plan.json` and `<name>_window_N_planned_region.npz` | `Video/mrfire/gate_p4_0014/second_hf_0014_p4/second_hf_0014_p4_windows/`; `find Video/mrfire -name '*_plan.json'` | the file names | holds |
| the 2026-10-07 captures are folders "with a README that gives its layout: per-frame tables of the track, the part and the region, and every node's report" | `data/2026-10-07_capture_windows/`, `data/2026-10-07_zoomout_window/` | each README's head, one window's files | holds |
| a frame of an input video is found with `bench/contact_sheets.py find <video> <frame>` | `contact_sheets/fun/` | `find fun 114` | holds: it names a sheet. The video's name there is `fun`, not the file's stem |
| `data/CAPTURE_GAPS.md` U5, U6 and 30 | that file | each entry's first line | holds |
| the composite's hold was "measured, `bench/results/2026-10-10_saved_windows_laid_again.json`" | that path | existence; `git log --diff-filter=D` | does not hold: the file left the tree (commit a3aeab08) and is under `data/2026-10-10_recomposite_decodes/`. The page now names the record's `.md` |
| "the statement of who is in each window" is in `handover.md`, "After the close" | `docs/wiki/handover.md` | its headings | does not hold: no section of that name. Said beside the sentence, with where the fact lives now |
| the old notes file `docs/research/masking/2026-10-04_mrhf.md` is "no longer in the tree" | that path | existence | holds |

## Held to the code, not to a clip

Found while the page was read with the code beside it; each is corrected
in the page with what it used to say, and logged in
`docs/wiki/decisions.md` under 2026-10-12.

| the page said | the observable | now |
|---|---|---|
| a preview tracks the subject "the first time; the mask is kept and the render that follows tracks nothing" | `masked_prompt.py`, the docstring's "What the `source` wire costs" | no mask is kept since 2026-10-10 |
| the `MASK_VERSION` numbers are on `subject_track.py`, `sapiens2_parts.py` and `video_mask.py` | `grep -n MASK_VERSION *.py` | the trackers' files; the Masked Source has none |
| "None of the captures or tools above carries a field for which person a file is about" | `bench/capture_masked_run.py`, the docstring: a `subject` on every row and file | the capture and a delivery row carry one; the node's own files do not |

Every backticked path and identifier in the page was also searched for in
the tree and in core, once, by hand: each was found, or is a file the page
itself says was removed. `bench/check_doc_links.py` holds the same on every
run for the forms it parses (`symbols_exist`, `citations_resolve`). Either
says a name exists, not that the sentence around it is right.
