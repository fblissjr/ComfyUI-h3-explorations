# What was knowable before the render

**Most faults found after a masked render were in data that existed before it; what was missing was a rule that
blocks, and running the cheap passes on the source before the first render.** Of the fifteen faults of 2026-10-10,
eleven were in data on disk before the render that showed them. In four the tool had raised a flag (one at a level
that did not block, one queued past, two read as good news); in the others the two columns that predict the fault
existed and no rule joined them. Two needed a pass that was not run up front (the pose pass on the source) or a field
nobody recorded (which shot a text was written from). Two have no known predictor.

A row is not finished until its last column names a rule that raises it before sampling. The rules are
`bench/capture_masked_run.py`'s (`preflight --gate`), reading the song node's own plan (`<name>_plan.json` and the
planned region files a preview writes). The working rule behind this page is [`working_rules.md`](working_rules.md),
"The rule: data before a render, and the same data after".

## The four sources, all available before sampling

| Source | What it costs | What it holds |
|---|---|---|
| the trackers' shot tables and tracks | a no-sampling preview | who was taken on each shot and at what likeness, how many people were detected, each subject's size and box per frame |
| the plan | nothing: arithmetic on the loads | each window's frames, where windows join, which frames are kept from the window before, which are held past the load's end, which latent step each frame is in, where the cuts fall |
| the class maps and the owner map | the same preview | which part is where, whose a contested pixel is, where a part is empty |
| a pose pass and the voice table on the SOURCE | a short pass on the card; the voice table needs none | the head's angle, the chin, each wrist against the nose, per frame; which frames are voiced; the mouth's opening |

## The faults of 2026-10-10

| Fault, as found after a render | What would have said so before | Did we have it | The rule that raises it now |
|---|---|---|---|
| a whole shot left as the source for both subjects; a viewer saw it | the shot table: "absent", people detected, the likeness a few hundredths under the line | yes, flagged "iffy", no outcome recorded | `absent_with_people_on_screen` and `load_has_a_shot_without_the_subject`, top level |
| a short load fades back to the original toward its end | the plan: most of the window is the load's last frame held as plate, with the subject on it; the track: the subject small | yes, both, joined by nothing | the cause is removed: the held tail is always open (`video_mask.held_mask`; [`window_context.md`](window_context.md)). `original_shown_in_the_held_tail` (`bench/capture_masked_run.py::flag_held_tail`) reads the node's own plan (`held_tail_not_checked` on a what-if plan); `changed`'s `ends_on_the_source` grades it after a render, and found a third such load nobody had seen |
| a head turn not made on a continuation | the pose pass on the source: a large change of the head's angle just after a join, behind kept frames that hold the old pose | no: the pose pass was run on the renders | `kept_frames_far_from_the_pose_ahead`, iffy (`flag_kept`; `kept_frames_not_checked` without a 3D pose table); not yet run on a real continuation's plan |
| the source's picture repainted on the far side of a cut | the plan: a cut inside a latent step that holds the subject on one side only | yes, as arithmetic nobody had written | `region_carried_across_a_cut`, `region_shared_across_a_cut`; the node's cut gate leaves those frames alone |
| a face drawn for one frame where the part was emptied | the plan: a frame with an empty mask inside a latent step whose other frames have one | yes | `region_on_a_frame_with_no_mask` |
| the original's face or hair back where pixels of the subject were kept | the masks: kept pixels inside the part | yes; the flag was in the list and the render was queued with it standing (no gate then) | `kept_pixels_inside_the_part` (`KEPT_IN_PART` lowered from that measurement, 96f204f6) |
| a mouth that stays at its ordinary opening where the source's opens wide | the source's mouth opening against the voice table | yes, two columns nothing joined | `mouth_open_with_no_voice`, iffy (`flag_mouth`), with the subject's class map and the voice table ([`state_signals.md`](state_signals.md), "A mouth open with no voice") |
| a small subject that does not take its mesh's pose | the track: her share of the frame, and the tokens a mesh at that size gives her head | yes | `subject_small_for_the_whole_load` (its line passed her by a hair: read the size) |
| two trackers on one person; one subject's region over another | the tracks' overlap, the owner map | yes | `two_tracks_on_one_person`, `region_over_another_subject`, `contested_by_class_too` |
| a filled part that drew a face on hair, a hat or an arm | the class map under the filled mask | yes | the graded fill (`part_grades.json`) |
| a sentence written from one shot drawn in another | which shot's frames each load's text was written from | no: nothing recorded it | the job file records a text's frames and refuses one text for loads of different shots ([`running_a_job.md`](running_a_job.md)) |
| held frames written past the shot because the audio outlasted the load | the audio's length against the load's | yes | the builder cuts every load's audio short of the load (`workflows/masked_job_graph.py`, `AUDIO_SHORT_BY`; held by `bench/check_masked_job.py`). The song node writes as many frames as its audio is long, rounded up (`loop_plan.frames_kept`), so audio a sample long wrote the next shot's first frame. Not built: a gate flag for a plan that writes more frames than its load holds |
| the motion video at half the canvas | the graph: `motion_short_edge` against the canvas's short edge | yes | not a rule; a pose pass showed the size was not what failed that render |
| the first window of a pass in the original's look, on one seed | nothing known | no | none: a short first render per subject is the only test |
| long hair drawn over a face in profile for a long stretch | partly: the pose pass says how long the head is in profile; nothing measures the reference's hair | partly | none |

## Found later the same day

| fault | what would have said so before | the rule now |
|---|---|---|
| a small face is replaced and does not read as the reference's person (the owner, on playback of the first joined file) | the face part's height in the model's tokens, in the gate's own table, about half that of the face the owner had accepted | `face_small_in_tokens`, iffy (`bench/capture_masked_run.py::flag_face_size`, `FACE_TOKENS`; the line is unmeasured). It warns and does not predict: nothing reads likeness to the reference ([`state_signals.md`](state_signals.md)). The remedy built is the zoomed render ([`pieces.md`](pieces.md)) |
| a whole-person pass moves the hands off what they are doing and draws what the reference holds (a hand that does not reach the bowl; a strap from the reference) | partly: the gate's list of what lies inside a whole-person region; nothing compares the reference with what the scene needs | none; that subject is replaced as face and hair only (the owner's decision). A still read for one question does not answer another: the strap was in stills a session had looked at |
| three people, fifteen shots, no corrections typed: a named subject never found, a take under the tracker's own line, a shot the who-is-who file leaves open | the shot tables | `subject_never_found`, `taken_under_its_own_line`, `who_is_who_unanswered`, top level. **Not named by anything: a take of the wrong person over the tracker's line**; size does not separate right takes from wrong, and likeness does on that stretch for a reason that does not carry ([`bench/results/2026-10-10_take_size_and_likeness_against_known_answers.md`](../../bench/results/2026-10-10_take_size_and_likeness_against_known_answers.md)). `shots` prints state, likeness and size beside every "clear"; the gate does not read the job's `left_out` |
| a part that was fragments on most frames (on hair, an outline, a wall); a neighbour's track that was a head alone | a reader who did not make the masks, walking every frame; the gate had only warnings, in a list too long to read | the grade empties a part under `PART_LEAST_PX` (`bench/capture_masked_run.py::grade_holds`), so such a frame is named at the top level; a capture prints each part's size against it and takes another from `least=` on the `--mask`; `sizes` prints the same for an existing capture. "Is the part on its own subject" does not separate. The look by someone who did not make the data is recorded with `bench/capture_masked_run.py looked` |
