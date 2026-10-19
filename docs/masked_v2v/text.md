# The masked text and its frame

What a masked load's text is for, how little a first text says, who writes it, and the one hand-written form kept as
a pattern. The lane's settled rules on a text (to H3 a masked load is not an edit; "sings" only where the person
sings) are in [`data/AGENTS.md`](../../data/AGENTS.md), "Settled for the masked lane"; this page is the reasoning
behind them and how a text is made. The vendor's reference format is [`docs/prompting/prompting.md`](../prompting/prompting.md).

## The frame every masked text is written in

The model is not editing a video it can see. It generates new pixels in the masked region, and nothing is replaced
from its side. What steers it is what its training makes of the frozen frames and the sound round the hole, the
prompt, and the references. So the text gives it enough to infer what happens there: each reference asset
described, and what of it is preserved, partially preserved or transferred (`retention_analysis`); what is
generated (a face, or a face and hair); and the movements and expressions from the body-model video. A sentence that
describes the scene as already lit, framed and cut, or a thing as "the scene's own", is addressed to an editor, not
to this model (the owner, 2026-10-10; made core 2026-10-15).

### Say the least first, and add a sentence only when a render asks for it

The frozen picture round the hole and the frozen audio already tell the model most of what a text would: the setting,
the light, the framing, the pose where the region is small, and whether a voice is on the track. So a first text
discloses in this order and stops (the owner, 2026-10-10, as progressive disclosure):

1. **Who is drawn and from which reference, and this part is never minimal**: the subject's definition, what
   `<Picture 1>` provides, and the vendor guide's marker for it (`vendor_guides/ref_en.md`, section 4.1):
   `fully_preserved` for a whole person; `partially_preserved` when only a part of the still is taken, with what is
   kept and what is not used said in the line; `attribute_transfer` for a motion video.
2. **Nothing about the frozen picture.** Where a person stands, what she wears and the room are frozen context the
   model already has, and naming them does not tell it which face goes in which hole.
3. **Nothing about action or expression, and of voice only "sings", where the person sings.** Who sings is plan data,
   `subjects.<s>.voice`; where the plan names no voice on a load, `bench/load_text.py` falls back to the subject's
   `sings` flag and the video's voice table, so a session watches and listens to the shot and writes the plan's key
   where the fallback would be wrong. An open mouth in a still is not a voice.

Then render, look, and add one sentence for what was missing: only what is plainly visible in the source at the
frames the sentence is about, or measured, and never the frozen picture. An added sentence is checked as a default
is, against the source, and it is the first thing deleted when a render misbehaves.

What the first day of this showed (one clip, one seed each, read from stills, a direction and not a record): a
whole-person swap with hands inside the hole did not draw a two-handed action until one sentence named it; a
sentence placing a gesture "near the end" of a window was drawn through most of the window; and a voice sentence
was written for shots whose track has no voice. Two of the three are a sentence added before a render asked for it.

## Who writes a load's text

**`bench/load_text.py` is the one home of a load's fixed lines** (since 2026-10-14). `bench/shot_to_load.py text
--plan PLAN.json` makes the text from the reference's record (what was drawn: the parts, the eye's form, the mouth
where its frames draw one), the plan's words for the subject and the video (its looks, its hair change, its voice),
and the voice table. It takes no place-and-action sentence (`bench/load_text.py::NO_SCENE`). A text written whole by
a person is the exception (`--by-hand "why"`), and is the only way a two-subject text is made. Its examples are the
kept loads it reproduces, in `bench/check_load_text.py`.

**Before it renders, the text is read in full** from the graph as queued (`bench/what_h3_got.py`), by the session that
queues it (root [`AGENTS.md`](../../AGENTS.md), "A render's prompt is read in FULL before it renders"). In the job file
a text is a file beside it with the source frames it was written from and who wrote it; one text for loads of
different shots is refused ([`running_a_job.md`](running_a_job.md)). Reading a text against its frames, its mask and
its audio is [`working_rules.md`](working_rules.md), "Reading one window whole".

**What the Masked Prompt node writes** is a different text, for the shipped graphs: [`pieces.md`](pieces.md),
piece 3.

## The text: the fragment

The form of 2026-10-11 ("shapes drawn on its face"), kept as what the lane wrote for the body model with lid marks;
it describes no load the chain makes now, whose lines are `bench/load_text.py`'s. A reference the text does not speak
of is one the model cannot know to use, and a text that says what her face does leaves a render unable to say
whether the text or the reference did it. So the text says what the video is and that she follows it, in four
places, and nothing else about it. Copied from the picked render's text
(`data/2026-10-11_fun_gate_inputs/second_hf_0014_p4.txt`):

The end of the subject's definition, after who the subject is:

```text
, who follows the movements and the facial expressions of the figure in <Video 1>.
```

The line that says what the video is:

```text
<Video 1> is a plain grey body model on a black ground, with shapes drawn on its face that are part of its facial expression; <Subject 1> follows its movements and its facial expressions, and its grey look and the shapes themselves are not drawn.
```

Its retention line:

```text
<Video 1> (motion source): attribute_transfer - the movements and the facial expressions of the figure are transferred to <Subject 1>; the grey body model's look, the shapes themselves and its black ground are not.
```

One sentence in the shot:

```text
She follows the movements and the facial expressions of the figure in <Video 1>.
```

What to hold to when the pattern is used:

- **It is a pattern, not a paste** (the owner's ruling of 2026-10-12, in [`docs/state/decisions.md`](../state/decisions.md)).
  The four places fit a load whose reference video is the marked mesh and whose region is a face, or a face and head.
  A plain mesh gets a plainer line; on any other load they are not assumed, and no tool inserts them.
- **The second line describes that video**, as the model sees it: if the reference is drawn another way, the line
  says what that one is.
- **Nothing names a feature, a turn of the head or what she does**, and the shapes are given no colour.
- **The pronoun and the subject's number are the load's.**
- **A line for `<Video 1>` as a source of movement is this pack's form**, measured on 2026-10-05
  (`../../bench/results/2026-10-05_masked_v2v_motion_arms.md`). The vendor guide keeps `<Video N>` for a relationship
  to a whole video ([`docs/prompting/prompting.md`](../prompting/prompting.md), section 9.4); the same content laid
  out the guide's way rendered once, with marks at true size, and the lid-marks record has what it did.

### One whole text, as an example

An example, not a text to copy: the whole text of the render the owner picked, beside an example of that load's job
file, [`bench/examples/masked_job_lid_marks/text.example.txt`](../../bench/examples/masked_job_lid_marks/text.example.txt),
with [`README.md`](../../bench/examples/masked_job_lid_marks/README.md) going through it section by section. It still
holds sentences about what "stays as it is" and what is "the scene's own", kept the same across that day's loads so
one thing changed at a time; do not carry them to a new text.

What that form carried on its shots, and what it did not, is [`renders_by_form.md`](renders_by_form.md).
