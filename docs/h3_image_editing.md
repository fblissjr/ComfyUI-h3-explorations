# H3 as an image gen/edit model

Last updated: 2026-08-16.

Video is this repo's primary use case. This one is **experimental**: H3 renders
a single frame if you ask it for one, and at one frame it behaves like a
capable reference-driven image editor. It rests on a temporary patch, it is
moving fast upstream, and it gets its own folder so the two cases do not blur
into each other.

**Everything about the mechanism** -- the 5-frame floor, the shim, the
single-image VAE and the 15.2 dB that settles it -- is in `README.md` under
"One frame: H3 as an image editor" and in the note drawn on
`workflows/image/h3_image_edit.json` itself. This file is about the **prompts**
and the **layout**.

---

## Layout: `workflows/image/`

Graphs are foldered by use case. Video sits at the root of `workflows/`; every
single-frame graph is written to `workflows/image/`.

**The routing is derived, not declared.** `build_workflows._graph_dir()` sends
a graph to `image/` when its `GRAPHS` entry sets `single_frame=True`, which is
exactly what makes it an image graph. A separate `image=True` flag would be a
second source of truth for one fact, and the two would eventually disagree.

**The reading side is `h3_config.GRAPH_DIRS`**, and every check that walks the
shipped graphs goes through `h3_config.graph_paths()`. This is not tidiness.
Six checks used a bare `workflows/*.json`, which is non-recursive, and on
2026-08-16 that was **demonstrated** rather than reasoned about: pointed at a
stale `GRAPH_DIRS`, `check_ref_prompt_labels` and
`check_prompt_guide_conformance` both **exited 0 while covering 20 ref graphs
instead of 28**. No error. No warning. Just a smaller number on a line nobody
has a prior for.

So `check_ref_prompt_labels` now carries a case -- *no graph directory is
invisible to discovery* -- that compares the discovered set against what is on
disk. `bench/` and `archive/` are excluded there by name, deliberately.

Naming inside the folder follows the repo's existing convention rather than a
new one: `h3_image_*.json` is a scene worth rendering, `h3_image_probe_*.json`
exists to answer a question and says which one in the note drawn on its canvas.

---

## The prompts, and what changed

**Until 2026-08-16 this path shipped one prompt in flat prose, and the
docstring argued the guide format could not apply to a still.** Two of the
guide's six sections are audio, and `detailed_description` is specified as
`[Shot 1]` with camera movement and shot timing -- none of which a one-frame
render has.

What changed is evidence. The r/StableDiffusion write-up this path follows
published a second set of prompts on 2026-08-15, and **between their two posts
the author switched formats**: post 1 is flat `Task: Reference-guided
generation. ...` prose, post 2 is the guide's structure with the audio sections
dropped. That is the direction the old docstring argued against, from someone
who had rendered a couple of thousand images this way.

**It is a practitioner's revealed preference, not a measurement.** Neither post
held the scene or the references fixed, so nothing there isolates the format.
That is a reason to test, not a reason to believe, and it is why the ladder
below exists instead of a rewrite.

The half of the old argument that survives: the audio sections describe
something a single-frame graph structurally cannot produce. It has no
`VAEDecodeAudio` at all. That is why four sections is the default and six is
the arm.

### The format ladder

Three formats, each rung removing exactly one thing, so a difference between
two arms has one candidate cause.

| format | what it is | shipped as |
|---|---|---|
| `av` | all six guide sections, audio ones present as `N/A` | `h3_image_probe_format_av.json` |
| `sections` | the four visual sections | **the default**, every `h3_image_*.json` |
| `flat` | one paragraph, no headers, no `[Shot 1]`, markers in English | `h3_image_probe_format_flat.json` |

Both probes render the **same scene as `h3_image_style.json`** at the same
seed, with the same references. Read all three together; each is worthless
alone.

**Rendered 2026-08-16, and the short version is that none of them failed.**
`av` differs from `sections` by 3.45 (grey-scale mean absolute, 0-255) against
an h264 noise floor of ~1.6; `flat` differs from both by ~44.6. So the audio
sections are free and the scaffolding is a large lever on the picture -- but
the designed failure, the style reference bringing its cottage, did not fire in
any arm. Numbers, limits and the arms that would discriminate: `#16f`.

**Content is written once per scene and rendered into all three formats.**
Hand-writing a flat variant would let the arms differ in wording as well as in
structure, and the comparison would measure the writing. Two consequences worth
knowing before reading a result:

- **`flat` keeps `<Subject N>`** even though the community's post-1 prompts do
  not. The subject labels are the only place the reference roles are stated, so
  dropping them would change what the arm says. If the structured arms win,
  whether the subject indirection specifically is what did it is a separate
  follow-up.
- **`flat` renders the retention markers as English** (`<Subject 2> supplies an
  attribute transfer:` rather than `<Subject 2>: attribute_transfer -`). A
  paragraph carrying raw marker vocabulary mid-sentence is a form nobody
  writes, and beating a strawman would tell us nothing. So that rung removes
  the guide's formal apparatus as a unit -- headers, shot marker, and marker
  vocabulary.

### What every format guarantees

These are the parts that are not stylistic, and they are checked:

- **Every `<Picture N>` the graph wires gets a job, and no prompt names one it
  does not wire.** `check_ref_prompt_labels` enforces this in every format,
  unwaived. Naming an absent reference is the failure that reads as a model
  problem: the render succeeds and quietly ignores the instruction.
- **A reference supplying technique says what it does *not* supply.** The
  official guide never writes a negative clause -- every relationship there is
  stated as what a reference provides. This comes from general prompting
  research and from the community write-ups, where the reported failure is a
  style reference dragging its own content along. **Untested here**, same as
  the identical technique in `_ref_prompt`'s swap arm.
- **Retention markers stay inside the guide's visual set.**
  `check_prompt_guide_conformance` grades image graphs on markers, task types,
  section order and dialogue placement exactly like video graphs. Only the two
  audio sections are excused, and only for graphs with no audio decoder.

---

## The scenes

Chosen so each exercises a different retention marker rather than a different
subject. Most are drawn from the two write-ups; `h3_image_swap` is not, and is
the image-path twin of `h3_ref_video_swap`. Every one names an `h3_refs/` asset
from `internal/reference_library.md`, so a result is attributable to a
documented subject rather than to whatever was in the input root that day.

| graph | refs | marker under test | what fails first |
|---|---|---|---|
| `h3_image_edit` | 1 | `partially_preserved` | the camera move, or the far side of the face |
| `h3_image_style` | 2 | `attribute_transfer` | the style reference bringing its cottage |
| `h3_image_composite` | 2 | `partially_preserved` + `fully_preserved` | the cutout: no contact shadow, studio light still on |
| `h3_image_multiperson` | 3 | two `partially_preserved` | the two faces blending into one person |
| `h3_image_recolor` | 1 | `partially_preserved`, strict | everything that was *not* asked to change |
| `h3_image_sheet` | 1 | `fully_preserved` | the rear view, which has no source pixels |
| `h3_image_swap` | 3 | two `attribute_transfer` + `fully_preserved` | a face landing on the wrong one of the two people |

**Two rules the scenes are written against**, both learned the hard way here:

- **A scene the reference already satisfies cannot fail.** The first version of
  the camera scene asked to age a subject to 60 against a reference of a man
  well past 70. It rendered, it looked like a working edit, and it demonstrated
  only that the pipeline runs. Before trusting a passing case, ask what the
  input would have to look like for it to fail.
- **One primary change per beat.** Two changes collapse into whichever is
  easier to render, so `recolor` names exactly two attributes and spends the
  rest of its words on what must not move.

### Reference files, and one UI wrinkle

The scenes name subfolder paths (`h3_refs/face_young_man_glasses_1024x1024.png`)
because `internal/reference_library.md` documents those assets and the input
root is the owner's own media.

`LoadImage` builds its combo from a **non-recursive** `os.listdir` of the input
directory (`ComfyUI/nodes.py::LoadImage.INPUT_TYPES`), so nothing in a subfolder ever
reaches `/object_info`. It also defines `VALIDATE_INPUTS` ->
`folder_paths.exists_annotated_filepath`, and ComfyUI's executor skips its own
combo check for any input the node validates itself. **So these render
correctly**, and both of this repo's validators were rejecting them -- being
stricter than the server, which is the same class of defect as being looser.
Both now exempt values carrying a subfolder, and only those; a typo in a bare
filename still fails.

**The cost, which is real and is not a bug:** the frontend populates that
dropdown from the same list, so opening one of these graphs shows the subfolder
value in the widget but will not offer it in the menu. Re-picking it from the
dropdown is the thing you cannot do.

---

## What is not settled

- **Which format is right -- partly answered 2026-08-16, and the answer is "no
  arm failed".** All three rendered at one seed: the audio sections move the
  image by 3.45 (grey-scale mean, near the ~1.6 h264 noise floor) and cost 15
  tokens; `flat` moves it by 44.61, a materially different picture. But **no
  cottage appeared in any arm** -- the `attribute_transfer` role bound without
  the scaffolding, and identity, freckling and medium held in all three. n=1
  per arm on one scene, and `flat` keeps the negative clause, so this does not
  say format is irrelevant. `open_experiments` #16f has the numbers and the
  three arms that would actually discriminate.
- **Whether the negative clauses earn their tokens.** Same open question as on
  the video path.
- **The canvas**, `docs/open_experiments.md` #16d. Every image graph ships the
  in-family 768x1152; the write-up uses 1024x1536, 52% over H3's area cap. At
  one frame the canvas is nearly free either way, which makes this cheap and
  unresolved rather than expensive and unresolved.
- **Whether `allow_upscale=False` holds across subjects.** It is now the
  default here (see below), on two paired renders rather than one. That is
  still a small n.
- **Whether a lower step count is safe on the simple scenes.** Measured as
  unsafe on the complex one, which is enough to leave the default alone, but
  a per-scene step count was not explored.


---

## What it costs, measured 2026-08-16

Both arms rendered on this box, same seed, nothing else changed. These are
wall clock including queue and any model load, so read the ratios rather than
the absolute seconds.

### `allow_upscale` -- taken, 4.9x

`h3_image_style`, two references:

| | secs |
|---|---:|
| `allow_upscale=True` (the old default) | 89.1 |
| `allow_upscale=False` (**ships now**) | 18.1 |

The two images hold the same identity, freckle pattern, head angle, expression
and hairstyle, checked against the source reference rather than against each
other. The graphite medium transferred in both, and in neither did the style
reference bring its own cottage. This reproduces the 84s/18s ladder in #16e on
a second subject and seed, which is why the default moved here.

### `steps` -- NOT taken, and the reason is the interesting part

| scene | refs | 16 steps | 8 steps | verdict |
|---|---:|---:|---:|---|
| `h3_image_edit` | 1 | 13.0s | 4.0s | indistinguishable |
| `h3_image_style` | 2 | 18.0s | 7.0s | freckling and medium both hold |
| `h3_image_multiperson` | 3 | 25.0s | 10.0s | **8 loses the freckling and the pendant** |

At three references, 8 steps drops exactly the fine detail that scene's
`partially_preserved` entry names as the thing being retained. The saving is
~15s on the one graph where the detail is the whole point, so `steps` stays 16.

**Measured only on the one-reference portrait, 8 steps looks free everywhere.**
That is this repo's own trap -- a check whose input already satisfies the
expected outcome cannot fail -- and it is why the ladder was run on the hard
scenes before touching the default.

One paired render per scene. Consistent with the expected mechanism, not a
sweep.

### The whole set

With `allow_upscale=False`, all eight graphs render in roughly **two minutes
total**, against about eleven before.

---

## Two defects another session's tool found in these prompts

`bench/preflight_graph.py`, written in a parallel session, was pointed at
`workflows/image/` on 2026-08-16 and immediately found something every check
here had passed.

**1. Six of the eight scenes defined `<Subject N>` and then never cited it in
`detailed_description`.** Guide 5.3 asks for each label at its first real
appearance *and where its role applies*, not merely once in
`subject_definitions`. The bodies said "he", "the likeness", "the same man" --
so the binding between a reference and the action it drives was left to
inference, on the exact path whose whole point is saying what each reference
does. Fixed: every scene body now cites its labels.

**`check_ref_prompt_labels` could not see it, and that was the real gap.** Its
`subjects_resolve` case checks that every subject USED is defined -- the
dangling direction -- and passes clean on a prompt that defines two subjects
and cites neither. A new case, *every defined subject is cited where it acts*,
closes it, and was shown red by reverting one scene's body to its pre-fix
wording.

**2. A row count in the generator's own note was 58% high.** It said "~5,200
rows" for the single-reference graph from arithmetic; measured, it is 3,282.
Now a measured table rather than an estimate.

Both were caught by a second reader with independent access to the artifact,
which is the finding `CLAUDE.md` records and this is another instance of it.

### The one warning left, and why it is not being fixed

Every scene trips `detailed_description is N words; the guide asks 350-500 for
generation tasks`, at 99-146 words.

**Deliberate.** That figure is the guide's for a *video* generation task, where
the body has to carry shot timing, camera movement over time, and action in
chronological order. A still frame has none of those. The community's own
working image prompts run 150-200 words, and the first-post set is shorter
still. Padding these to 350 would mean writing filler to satisfy a checker.

**But it is NOT the same case as the audio sections, and an earlier version of
this paragraph said it was.** The distinction is load-bearing and it came from
the session that wrote `preflight_graph.py`:

| | audio sections | word budget |
|---|---|---|
| status | **structurally impossible** | **possible, and judged wrong** |
| why | no `VAEDecodeAudio`, so there is no track to describe | a still frame can carry 350 words; they would be padding |
| exemption | real, and read off the graph | would be invented -- the guide exempts editing tasks explicitly and says nothing about single frames |

So the audio pair is genuinely waived by
`check_prompt_guide_conformance._audio_sections_optional`, which reads the
graph for a decoder. The word budget is **not waived anywhere**, and should not
be: scoping it by latent-frame count would mean inventing an exemption the
guide does not contain, which is the specific thing this repo criticised the
Custom-GPT pack for.

Left as a visible warning: it is correct about the guide, and the deviation is
ours to defend. `preflight_graph.py` prints a note beside it pointing here, so
a reader sees an argued deviation rather than an unexplained red. If the format
arms ever show longer bodies help, this is the number to revisit.

---

## Two things the renders turned up that are not about speed

- **`crop` cannot be retained on a 1:1 reference.** `h3_image_style` and
  `h3_image_recolor` both say the crop is preserved, and both references are
  1024x1024 while the canvas is 768x1152. The model widens the frame -- it has
  to. The claim is unachievable as written and should either drop `crop` or the
  scenes should render on a square canvas. Still unfixed for those two, but the
  **mechanism now exists**: since 2026-08-22 a scene can override the shared
  canvas through `scene()`'s `extra`, which `h3_image_swap` uses to render its
  16:9 plate on a 16:9 canvas rather than promise a framing a portrait output
  cannot hold. The same edit is available to these two and has not been made.
- **The style scene works.** No cottage, in any arm, at any step count. The
  `attribute_transfer` role bound, which is the one thing that scene exists to
  find out, and it is the first evidence here that naming what a reference does
  *not* supply does something.
