# Masked video to video

Rules first: [`data/AGENTS.md`](../../data/AGENTS.md). Then these pages, from the overview down.

## The lane

| page | what it is for |
|---|---|
| [`overview.md`](overview.md) | **start here**: how a masked job is run, the pieces in the order a render meets them, the limits each has shown, where to go next |
| [`pipeline.md`](pipeline.md) | the lane end to end as a data flow: every stage from a new video to the cut, its tool, what it reads and writes, the record that says it ran |
| [`known_before_a_render.md`](known_before_a_render.md) | every fault found after a render, against the data that existed before it: what would have said so in advance |

## Finding and tracking a subject

| page | what it is for |
|---|---|
| [`sam31.md`](sam31.md) | **SAM 3.1 in this pack**: what to type, our nodes, the runtime patches (`sam31_corrections.py`), the shared tracking helpers, and how each differs from ComfyUI core |
| [`who_is_who.md`](who_is_who.md) | recognizing a named person again across cuts, shots, loads and sessions: the design and its premortems |
| [`who_is_who_evidence.md`](who_is_who_evidence.md) | what the who-is-who decision may look at and how it is weighed |
| [`perception_models.md`](perception_models.md) | what SAM-Audio, PE-AV, SAM 3D Body and Sapiens2 each give the lane, and what they do not |
| [`state_signals.md`](state_signals.md) | what code can say about a subject (head direction, face showing, a hand at the face, who is in front) and what it cannot yet |

## Windows

| page | what it is for |
|---|---|
| [`window_context.md`](window_context.md) | what ties a window to the one before it, every lever that loosens it, and the dated results |
| [`window_edges.md`](window_edges.md) | figures per frame of the sound and the picture, for choosing a window's first and last frame |

## Records

- Dated studies: [`research/masking/`](../research/masking/), among them how ComfyUI's SAM 3.1 compares with Meta's
  stage by stage and the minimal core fixes of 2026-10-18.
- Postmortems of the lane's days: [`research/postmortems/`](../research/postmortems/).
- Measured runs: `bench/results/`, by month and lane in [`bench/results/INDEX.md`](../../bench/results/INDEX.md).
