# Self-Attention in H3: What the Self-Mask Reads, and What Reading It Showed

Last updated: 2026-11-27.

H3's video model looks at the prompt, the picture and the sound in one attention. The self-mask
([`self_steering.md`](self_steering.md)) reads a small part of that attention to find where a named thing is.
This page says whose attention that is, what a day of probing it showed, and what to build from it. Every
figure lives in the dated record
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

## 2. What Reading It Showed

All of this is one kind of shot: a single small figure crossing a locked-off wide frame, on the six-step Pika
recipe. Section 4 lists what that leaves out.

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
step or later no longer changes how she looks. Harvesting at the second step kept part of the effect across the
walk (record, section 5).

### One head already knows at the first step

At block `45`, head `43` marks the moving subject at the first step, well enough to cover nearly all of her for
the whole clip. It was chosen on one clip and then came out as the best head of every block read on three clips
it was not chosen on: another seed, a different subject walking the other way, and a different scene (record,
section 6).

The average over all heads at that block is unusable, which is why the earlier calibration left block `45` out.
The signal was inside the block the whole time, in one head, and the average buried it.

A render that harvested from that head at the first step gave the strongest effect of any arm through the first
half of the walk: the whole figure changed, where the shipped reading changed only head and shoulders. Why it
fades after that is under "Routing changes the motion" below.

### That head follows the subject more than the words

Scored against the mean of all text rows, the same head marks her almost as well as against her own three words.
On a clip with one subject that makes a good mask. It also means the head may not be able to pick one of two
people by name. That is the first thing to test before leaning on it.

### Two fixes that did not work

- **Subtracting a control map.** At the first step it makes the mask worse: there is nothing generic in the map
  to take out. From the third step on it helps a little at the blocks the node reads, and at block `45` it hurts.
- **Judging each frame against the whole clip** instead of against itself changes nothing measurable.

### Which words point where

Once the maps have settled, the words about the subject and what she does all follow her, and the subject noun
is not the best of them. Words for the setting point away from her. A phrase built from an action or a sound
word is worth trying when the noun alone gives a weak mask (record, section 4).

### Routing changes the motion

In the routed renders the walker ends farther along than in the plain one. The mask is read from a pass with no
routing, so it describes a walk the routed render no longer takes, and it trails her late in the clip. This is
what now limits the single-head arm, and it is a limit of reading once, not of the reading.

---

## 3. What to Build From It

In the order the measurements support:

1. **Let the harvest choose its block and heads.** `harvest_blocks` already exists on `attach_self_mask_router`;
   heads do not, and the harvester means over all of them. The single-head arm ran by narrowing that mean in the
   probe process only.
2. **Read again on the routed pass.** With `replay_step0` the first step runs twice. Reading the mask again on the
   routed pass, or on each later step, would follow the walk the render actually takes.
3. **Test two subjects.** One prompt with two people and a mask aimed at one of them, read from head `43` against
   that person's words, against all text, and as the words' share among all text.
4. **Only then** widen to still subjects, close-ups, `MiniMax-H3-Image` and `ref2va`. The head index was found on
   one checkpoint and LoRA and is not known to hold on another.

A node-code change is built in a worktree outside `custom_nodes` (root `AGENTS.md`, operative rule 3).

---

## 4. What This Does Not Show

- **One kind of shot, one seed per clip, one render per arm.** No still subjects, no two-subject scenes, no
  close-ups, no single-frame model, no references.
- **The subject is found by its difference from the empty scene,** which needs a locked-off camera and a subject
  that moves, and counts her cast shadow as her.
- **Raw scores, not attention weights.** No softmax over all keys was computed, so nothing here shows how a word
  competed with every other row for a patch.
- **The edit was read on stills.** `compare_harvest_step_and_head_grid.mp4` in the clips folder stacks every arm
  with its mask for playback.
- **In a mask strip the green is the mask, not the attention.**

---

## 5. Where the Ideas Came From

Three of the tests are borrowed from an outside study of a language model's attention
(`coderef/sauers/attention_introspection`, `README.md` and `mechinterp/RESULT.md`): compare against a control
condition, expect a handful of heads to carry an effect, and check picked heads against random sets of the same
size. The second and third carried over. The first did not, at the step where we needed it.

That study asks whether a model can move and report its own attention. Nothing here is that. The shared part is
the instrument: reading scores from one set of rows to another in a band of layers, and adding a bias to them.
The query router's firewall is the same bias (`steering/hooks.py::_build_aud_attn_bias`).

---

## 6. Run It Again

With the ComfyUI venv's python, from the repo root, after `set -a; . ./h3.env; set +a`:

- `data/self_mask_probe_2026-11-27/run_control_probe.py --repo . --out <folder>`: one plain render with a map per
  word, per step.
- `data/self_mask_probe_2026-11-27/run_late_harvest.py --repo . --out <folder> --clips <folder>`: per-head
  readings and the three late-harvest renders.
- `data/self_mask_probe_2026-11-27/run_head43.py --repo . --out <folder> --clips <folder>`: the three held-out
  clips and the single-head render.

Each frees the card through `:8188` first and refuses when its queue is busy. The `analyze_*.py` files beside
them turn the arrays into the record's tables on the CPU.
