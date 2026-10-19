# Masked video to video

Rules first: [`data/AGENTS.md`](../../data/AGENTS.md). Then these pages, from the overview down.

## The lane

| page | what it is for |
|---|---|
| [`overview.md`](overview.md) | **start here**: what the lane is, the shipped graphs, how a job runs step by step, the four pieces in brief, where to look |
| [`pipeline.md`](pipeline.md) | the data flow: every stage from a new video to the cut, its tool, what it reads and writes, the record that says it ran; where a frame's content is decided |
| [`working_rules.md`](working_rules.md) | the lane's working habits and their reasons: data before a render and after, reading one window whole, the rules of the lane (canvas, one shot one load, window length) |
| [`running_a_job.md`](running_a_job.md) | how a load is made with the chain and the builder, the recipe for a shot made whole, two shots in one load, depth, and where to see what a run was given and wrote |
| [`pieces.md`](pieces.md) | the four nodes in the order a render meets them, nothing kept by a key, the latent grid and the matte law, the zoomed render |
| [`text.md`](text.md) | the frame every masked text is written in, say the least first, who writes a load's text, the lid-marks fragment |

## Finding and tracking a subject

| page | what it is for |
|---|---|
| [`sam31.md`](sam31.md) | **SAM 3.1 in this pack**: what to type, how to choose one person, what ComfyUI does differently with a phrase, the evidence |
| [`sam31_core.md`](sam31_core.md) | the code map: runtime patches (`sam31_corrections.py`), what is left as core computes it, and our shot-aware tracker against core's, file and line |
| [`who_is_who.md`](who_is_who.md) | recognizing a named person again: the owner's answer and the cast file today, what is built, what the evidence showed, the two-vote plan, the design further ahead |
| [`state_signals.md`](state_signals.md) | what code can say about a subject (head direction, face showing, a hand at the face, who is in front, an open mouth) and what it cannot yet |
| [`perception_models.md`](perception_models.md) | what SAM-Audio, PE-AV, SAM 3D Body, Sapiens2 and the depth models each give the lane, what is on disk, the trials |

## Windows and renders

| page | what it is for |
|---|---|
| [`window_context.md`](window_context.md) | what ties a window to the one before it, every lever that loosens it, the held tail, and the dated results |
| [`window_edges.md`](window_edges.md) | figures per frame of the sound and the picture, for choosing a window's first and last frame |
| [`renders_by_form.md`](renders_by_form.md) | what is known from renders, by form; the body model with marks on its face, how it was made and what it carried |
| [`known_before_a_render.md`](known_before_a_render.md) | every fault found after a render, against the data that existed before it, and the rule that raises it now |
| [`limits_and_directions.md`](limits_and_directions.md) | what the lane cannot do, with evidence, and the directions worth trying |

## Records

- Dated studies: [`research/masking/`](../research/masking/), among them how ComfyUI's SAM 3.1 compares with Meta's
  stage by stage and the minimal core fixes of 2026-10-18.
- Postmortems of the lane's days: [`research/postmortems/`](../research/postmortems/).
- Measured runs: `bench/results/`, by month and lane in [`bench/results/INDEX.md`](../../bench/results/INDEX.md).
