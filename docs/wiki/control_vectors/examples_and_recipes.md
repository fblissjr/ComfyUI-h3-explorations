# Worked examples

last updated: 2026-11-09

[Back to the start page](../control_vectors.md)

Only examples a session has run are here, each with where its result is. The earlier version of this page
held recipes nobody had run, several naming pair files that do not exist; they are gone. "The plan" is
[`../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md).

`<python>` below is the ComfyUI venv's python (`docs/comfy_notes.md`). A build needs the card free.

## 1. Build a vector from a pair file in the bank

```bash
<python> bench/control_vectors.py build prompt_bank/control_vectors/film_16mm_1974.toml --families both --overwrite
```

It writes one file per family into the models folder's `control_vectors/` and prints, for each, the
recommended strength for the node's default mode. The shipped control-vector graphs load files built this
way.

## 2. Build a vector from your own prompt, with its reference picture

```bash
<python> bench/control_vectors.py from-prompt \
  --target-prompt my_prompt.txt \
  --words "old wording that occurs once in the prompt -> new wording" \
  --ref-images my_reference.jpg \
  --families ref2va --name my_dial --overwrite
```

The old wording must occur exactly once in the prompt. The picture is given by path, or by name when it is
in ComfyUI's input folder. Run once on 2026-11-09, on a reference shot of one subject, to make a film-stock
dial from that shot's own style sentence (the plan, section 25). Two things it showed:

- The vector's J-Lens words read correctly for the dial.
- Among the bank's vectors it was closest to the bank's own film-stock vector, and it carried more of the
  component all bank vectors share, because its new wording was much longer than the old. Keep the two
  sides about the same length.

## 3. Compare a vector with an edit on one shot

`bench/edit_guidance_arms.py` takes a shipped graph, a prompt and one edit, and builds a set of renders
that differ in one thing each: the prompt as written, the edited prompt, edit guidance, and with
`--vector-arm` the original prompt under a control vector.

```bash
<python> bench/edit_guidance_arms.py \
  --base workflows/h3_image_ref_plus_text_to_video_pika_control_vector_api.json \
  --prompt-file my_prompt.txt --ref-image my_reference.jpg \
  --edit "old wording -> new wording" \
  --name my_shot --out data/results/my_round --seeds 1 2 \
  --node-arms --only base rewrite node_s1.5 bank \
  --vector-arm "bank=<a vector file>:1.0"
```

It prints both prompts in full and writes the graphs. Add `--queue` to render them. A third field on
`--vector-arm` sets the mask. This is the round recorded in the plan, section 25: one dial, on a reference
shot, as an edited prompt, edit guidance, a bank vector and an in-place vector, on three masks and two
seeds. The owner has not judged it.

## 4. Steer with an edited prompt and no vector

The two generated graphs and the nodes are on [`../edit_guidance.md`](../edit_guidance.md).

## Not run by a session

Building from two pictures, building from words through J-Lens, `sculpt`, `grade` and the two build nodes
inside a graph. They may work; nothing here says so.
