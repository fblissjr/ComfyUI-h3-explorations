# Self-Attention in H3: What the Self-Mask Reads, and What Reading It Showed

Last updated: 2026-11-27.

H3's video model looks at the prompt, the picture and the sound in one attention. The self-mask
([`self_steering.md`](self_steering.md)) reads a small part of that attention to find where a named thing is.
This page says whose attention that is, what a day of probing it showed, what was built from it, and what is
still open. Every figure lives in the dated record
[`2026-11-27_self_mask_harvest_step_and_heads.md`](../../bench/results/2026-11-27_self_mask_harvest_step_and_heads.md);
nothing is restated here.

**Quick links:** [the sensor's code (`steering/live_policy.py`)](../../steering/live_policy.py) ·
[the read-only probe (`bench/probe_h3_sensors.py`)](../../bench/probe_h3_sensors.py) ·
[how the shipped map is made, step by step](self_steering.md) ·
scripts and arrays for this page: `data/self_mask_probe_2026-11-27/` ·
clips and sheets: `$H3_OUTPUT_DIR/Video/self_mask_late_harvest_2026-11-27/`

---

## 1. Whose Attention This Is

The attention being read is the video model's (the DiT's). Qwen3-VL does not run during sampling, and its own
attention is never read.

- **Qwen3-VL runs once, before sampling.** It reads the prompt as raw text with no chat template, with a vision
  block spliced in for each wired picture or video, and hands over one hidden state per token from the layer where
  H3 cuts it. The layout for each mode is in the module docstring of core `comfy/text_encoders/minimax.py`; on a
  text-only render it sees text and nothing else. [`steering.md`](steering.md) section 2 describes the two text
  stages.
- **The DiT turns those states into text rows** and packs them at the front of one sequence with the video and
  audio rows (`comfy/ldm/minimax/model.py::PackedLayout`, `comfy/ldm/minimax/model.py::MiniMaxH3Model._embed_and_pack`).
- **Every block runs one attention over that whole sequence**, with no mask between kinds of row
  (`comfy/ldm/minimax/model.py::Attention.forward`). Text, video and audio rows all query and key each other, and
  every block rewrites all of them. By the blocks the sensor reads, the row for a word is no longer the vector
  Qwen wrote: it started there and has been mixed with picture and sound in every block since.
- **Qwen's tokenizer is used once more, as a lookup,** to find which rows spell the target words. That is the
  only reason the node takes a `clip` wire.

So "self-attention" here means: for each patch of the picture, how strongly its query scores against the key of
a word row, inside the DiT. The model is not asked anything and reports nothing. We read it from outside.

---

## 2. The Recipe That Came Out of It

For a moving subject on the six-step Pika recipe, read one head at a late block at the first step, and read it
once more while the first routed pass runs:

```python
attach_self_mask_router(
    model, region_conditioning=..., mask_source="text_phrase", target_phrase="woman, crimson, coat",
    replay_step0=True, harvest_blocks=[45], harvest_heads=[43], reharvest="first_routed_step",
)
```

On the walk clip this changes the whole figure for the whole walk, where the shipped reading changed head and
shoulders for the first half (record, section 8). Both options are on `attach_self_mask_router` only. The node
has no input for them and its defaults are unchanged: the head index was found on one checkpoint and LoRA, and
section 5 lists what has not been tried.

---

## 3. What Reading It Showed

All of this is one kind of shot: one or two small figures crossing a locked-off wide frame.

### When you read matters more than how you clean the map

At the first sampler step, on the blocks the node reads, the map for the target words does not follow a moving
subject. No word's map does. Two or three steps later the same words follow her on nearly every frame (record,
section 4).

Our reading of why, not something measured directly: depth and step trade against each other. A late block at
the first step seems to hold what a middle block holds two steps later, because the first step's own prediction
is already a rough layout by the time it reaches the last blocks.

### A later harvest gives a better mask and a weaker edit

The node harvests on its first active step, so `start_percent` moves the harvest with no code change. Harvesting
one, two or three steps later puts the mask on her for the whole walk. But routing that only begins at the third
step or later no longer changes how she looks (record, section 5).

### One head already knows at the first step

At block `45`, head `43` marks the moving subject at the first step, well enough to cover nearly all of her for
the whole clip. It was chosen on one clip and then came out as the best head of every block read on three clips
it was not chosen on: another seed, a different subject walking the other way, and a different scene (record,
section 6).

The average over all heads at that block is unusable, which is why the earlier calibration left block `45` out.
The signal was inside the block the whole time, in one head, and the average buried it.

### Routing changes the motion, and a second reading follows it

In the routed renders the walker ends farther along than in the plain one. A mask read from the unrouted first
pass describes a walk the routed render no longer takes, so it trails her late in the clip and she fades back.
Reading the same head once more while the first routed pass runs fixes that: the mask comes from the render the
routing is producing. Reading on every routed pass also holds, with a larger mask (record, section 8).

### With two people, the head finds people and the region prompt picks one

Head `43` marks both people, whichever person's words it is read against. In the renders the mask covered both
and one changed: the one walking the direction the region prompt names. Swapping the target words did not change
who; swapping the direction in the region prompt did (record, section 9).

So today the target phrase says "where the subjects are" and the region prompt says "which one". That gave a
clean result when the region prompt described her walk, and it means a region prompt that fits both people
cannot be aimed by the phrase alone.

Heads that do tell the two apart by word exist at the blocks after `45`. They are different heads, they are poor
at finding people at all, and they were picked on the one clip they were scored on. A mask that needs both kinds
of head to agree separated the two cleanly on that clip; that is a lead and not a result.

### Fixes that did not work

- **Subtracting a control map at the first step.** On the shipped blocks it makes the mask worse: there is
  nothing generic in the map to take out. From the third step on it helps a little.
- **Judging each frame against the whole clip** instead of against itself changes nothing measurable.
- **Scoring the target words against all text to pick one of two people.** It narrowed the mask, weakened the
  edit on her, and did not spare him more. It was built, tried in one render, and not landed.

### Which words point where

Once the maps have settled, the words about the subject and what she does all follow her, and the subject noun
is not the best of them. Words for the setting point away from her (record, section 4).

---

## 4. What Is Next

1. **A mask that picks by word.** Check the word-telling heads on a clip they were not picked on, then try the
   mask that needs the subject head and one of them to agree. Until then, aim with the region prompt.
2. **Wider shots before any node default.** Still subjects, close-ups, references, and `MiniMax-H3-Image`, where
   the head index is not known to hold. A node input or a changed default waits on that.
3. **A region prompt cue other than direction** (a color, a size) to see how far "the region prompt picks" goes.

A node-code change is built in a worktree outside `custom_nodes` (root `AGENTS.md`, operative rule 3).

---

## 5. What This Does Not Show

- **One kind of shot, one seed per clip, one render per arm.** No still subjects, no close-ups, no single-frame
  model, no references. The two-person finding rests on one prompt.
- **The subject is found by its difference from the empty scene,** which needs a locked-off camera and a subject
  that moves, and counts a cast shadow as part of the subject.
- **Raw scores, not attention weights.** No softmax over all keys was computed, so nothing here shows how a word
  competed with every other row for a patch.
- **The edits were read on stills.** `compare_harvest_step_and_head_grid.mp4` and `two_people_grid.mp4` in the
  clips folder stack the arms with their masks for playback.
- **In a mask strip the green is the mask, not the attention.**

---

## 6. Where the Ideas Came From

Three of the tests are borrowed from an outside study of a language model's attention
(`coderef/sauers/attention_introspection`, `README.md` and `mechinterp/RESULT.md`): compare against a control
condition, expect a handful of heads to carry an effect, and check picked heads against random sets of the same
size. The second and third carried over. The first did not, at the step where we needed it.

That study asks whether a model can move and report its own attention. Nothing here is that. The shared part is
the instrument: reading scores from one set of rows to another in a band of layers, and adding a bias to them.
The query router's firewall is the same bias (`steering/hooks.py::_build_aud_attn_bias`).

---

## 7. Run It Again

With the ComfyUI venv's python, from the repo root, after `set -a; . ./h3.env; set +a`:

- `data/self_mask_probe_2026-11-27/run_control_probe.py --repo . --out <folder>`: one plain render with a map per
  word, per step.
- `data/self_mask_probe_2026-11-27/run_late_harvest.py --repo . --out <folder> --clips <folder>`: per-head
  readings and the three late-harvest renders.
- `data/self_mask_probe_2026-11-27/run_head43.py --repo . --out <folder> --clips <folder>`: the three held-out
  clips and the first single-head render.
- `data/self_mask_probe_2026-11-27/run_reharvest_two_subjects.py --repo . --part a|b ...`: the second-reading
  renders, the two-person probe, and the two-person arms.

Each frees the card through `:8188` first and refuses when its queue is busy. The `analyze_*.py` files beside
them turn the arrays into the record's tables on the CPU.
