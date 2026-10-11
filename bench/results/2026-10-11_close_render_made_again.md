# The close render of 2026-10-10, made again through the tools as they stand

2026-10-11, session mrfire. One load, one window, one seed. **Not judged on
playback by the owner when this was written**; what follows is read from
files, and two stills looked at by the session that made the run.

## What it buys

The owner ruled that the masked lane is redesigned around segments
(`docs/wiki/decisions.md`, 2026-10-11) and that nothing changes before the
one render they called close is made again unchanged. This run says whether
the tools at `9de6210d` still make that piece, so that a later difference is
the design's and not drift nobody saw. It applies to this load only.
Accepted when the owner, with the two clips stacked on the same frame
numbers, says it is the same piece.

## The load

The second subject, face and hair replaced from the still, source frames 14
to 154 of `fun_0000_1919_24p.mov`, one window. Every input is that day's own
file, handed in by name through `bench/masked_job.py` with nothing posted by
hand:

| input | 2026-10-10 | this run |
|---|---|---|
| graph | a session script on a lifted graph, read back from `mom_open_hair_face_mesh_00001.png` | `bench/masked_job.py build` at `9de6210d` |
| subject's mask, part, keep-out | three mask videos loaded by path | the same three, decoded equal on every frame: the mask and the part as the capture's sighting, the keep-out as a capture subject of its own named in the load's `others` |
| motion video | the body mesh made inside the graph | the mesh video that run saved, loaded as a file |
| audio | the loader's, whole | cut by the builder to just under the load's length (`workflows/masked_job_graph.py`, `AUDIO_SHORT_BY`) |
| text, still, seed, steps, sampler, shift, Sol and its sink, margin, feather, edge, window, context | | the same, compared input by input |
| node code | the tree that afternoon | `9de6210d`, after that night's removals |

Two inputs of the old graph no longer exist on the nodes (`reuse_mask`,
`reuse_windows`); both were off.

## What the files say

Compared by `numpy` on the saved files; the scripts were one-offs and the
figures are in the untracked notes of the session.

| what | result |
|---|---|
| region the node planned before sampling, against the region the old render saved (`tokens`, every latent step; `mask`, every frame) | equal, no cell differs |
| region this render saved, against the same | equal, no cell differs |
| the window's saved video latent, against the old one | cosine 0.9998; mean absolute difference under a hundredth of the mean absolute value |
| the window's saved audio latent, against the old one | cosine 0.993 |
| pictures, per frame, this render against the old, mean absolute difference in grey levels of 255 | inside the region about 2 at the median and about 3 at most; outside about 1 |
| each render against the source inside the region, same measure | about 21 for both |

The capture's gate was clear before the render with eight flags, none
blocking and none with an outcome entered yet. One of them is worth
carrying: the node's region covers up to about a fifth of the keep-out mask
on the frames where a hand is near the face, because a latent step shares
four frames. The region being equal to the old one, that was so on
2026-10-10 as well.

## What was looked at

A second session that did not make the capture looked twice before
anything sampled and recorded both looks itself (the capture's
`looks.json`): the four files the load reads decoded whole and checked
against that day's preview for which frame each starts on; the text read
whole against the frames; then the node's region. The maker looked at
source frames 72 and 148 of both renders side by side and could not tell
them apart at that size.

## What it does not say

That the piece is good. The owner's words on the old render were that it
was close and had no facial expression or head movement, and a run that
reproduces it reproduces that. Nothing here was watched as a clip. The
text is the old one unchanged and carries sentences the owner has ruled
out since.

## What it showed the tools cannot do

- No kept code writes the two special masks this render depends on (a
  cleaned face-and-hair mask and a hands, arms and held-things mask). They
  exist as two files and cannot be made again.
- `bench/capture_masked_run.py files` takes a `--loads` plan and a `--run` of
  the same name together, keeps both, and prints every flag twice.
- A capture's sighting cannot hold files of two runs
  (`inputs_written_together`), which is right, and means a mask derived
  later from a run's class map has to be made within that rule or handed in
  beside it.

## Where things are

Untracked: the job file and text, `data/2026-10-11_fun_gate_inputs/`; the
capture, `data/2026-10-11_gate_hf_0014/`; the job's records,
`data/2026-10-11_gate_hf_0014_job/`. On the share: the render under
`Video/mrfire/gate_hf_0014/second_hf_0014/`, and both renders stacked with
masks and frame numbers under `Video/mrfire/review_2026-10-11/`.
