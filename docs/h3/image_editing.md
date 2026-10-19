# H3 as an image editor: the parked single-frame path

**Parked on 2026-08-27, by the owner.** The single-frame path is not shipped and not maintained; this page records
what it was. Nothing in the live tree depends on it, and with its shim archived this pack modifies ComfyUI core not at
all. The arithmetic for one frame is core's and still holds (`h3_rules.SINGLE_FRAME`, `is_single_frame`), and
`MiniMaxH3Resolution` refuses `length=1` on a core that floors H3 at 5 frames rather than silently render five.

| was | now |
|---|---|
| `workflows/image/` (18 graphs) | `archive/workflows/image/`, no longer generated |
| `single_frame.py` (the core patch) | `archive/single_frame.py`, no longer imported |
| `bench/check_single_frame.py` | `archive/bench/check_single_frame.py` |
| `bench/bench_image_edit_refs.py` | `archive/bench/bench_image_edit_refs.py` |

`h3_config.GRAPH_DIRS` no longer carries `image`, so those graphs are neither discovered nor graded. The generator's
`_image_graphs()` was deleted on 2026-09-14, so un-parking means restoring it from git history plus the moves above.

## What it was

At one frame H3 behaves like a capable reference-driven image editor. Core's H3 nodes floor `length` at 5, and
`ComfyUI/execution.py::validate_inputs` rejects a lower literal before any node runs, so no node could lift it: the
path rested on a process-global patch, opt-in from 2026-08-27 through `H3_EXPLORATIONS_SINGLE_FRAME=1`. The mechanism
(the 5-frame floor, what the patch did and refused, the single-image VAE, its retirement) is the module docstring of
[`archive/single_frame.py`](../../archive/single_frame.py), and the note on
[`archive/workflows/image/h3_image_edit.json`](../../archive/workflows/image/h3_image_edit.json), the only copy, carries
the single-image VAE round trip against the video VAE and core's `t == 1` decode branch (`comfy/ldm/minimax/vae.py`).

## What it found, 2026-08-16

- **Prompt format.** A format ladder at one seed on one scene: all six guide sections with the audio ones `N/A`
  (`av`), the four visual sections (`sections`, the default), and one flat paragraph (`flat`). `av` differed from
  `sections` by 3.45 (grey-scale mean absolute, 0-255) against an h264 noise floor of about 1.6 and cost 15 tokens;
  `flat` differed by about 44.6, a materially different picture. The designed failure, the style reference bringing
  its cottage, fired in no arm. `docs/state/open_experiments.md` #16f has the numbers and the arms that would
  discriminate.
- **`allow_upscale` off was taken**: 18.1 s against 89.1 s on a two-reference scene, identity, freckling and medium
  held, reproducing #16e on a second subject. With it off all eight graphs rendered in about two minutes against
  eleven.
- **8 steps was not taken**: indistinguishable on one and two references, but at three it dropped the freckling and
  the pendant the scene's `partially_preserved` entry named. Measured only on the one-reference portrait, 8 steps
  would have looked free everywhere.
- **The canvas** was the in-family 768x1152 against the community's 1024x1536, 52% over the area cap; unresolved
  (#16d). At one frame the video is about 9% of the sequence and the reference nearly all the rest, so the canvas is
  not the cost lever.
- **A preflight read caught what the checks passed**: six of eight scenes defined `<Subject N>` and never cited it in
  `detailed_description`, which led to the check case "every defined subject is cited where it acts" in
  `bench/check_ref_prompt_labels.py`; and a row estimate in the generator's note was 58% high against the measured
  3,282.
- **The word budget stayed a visible warning.** The guide's 350-500 words is for a video generation task; the
  scenes ran 99-146, and padding would have been filler. Unlike the audio sections, which
  `check_prompt_guide_conformance._audio_sections_optional` waives by reading the graph for an audio decoder, the
  word budget is waived nowhere.
- **`crop` cannot be retained on a 1:1 reference** onto a 768x1152 canvas; a scene can override the canvas through
  `scene()`'s `extra`, which `h3_image_swap` used.

The scenes named assets under `h3_refs/` documented in `internal/reference_library.md`. `LoadImage` builds its list
from a non-recursive listing of the input directory, so a subfolder value renders (its `VALIDATE_INPUTS` resolves it)
but the frontend will not offer it in the dropdown; the repo's validators exempt values carrying a subfolder, and only
those.
