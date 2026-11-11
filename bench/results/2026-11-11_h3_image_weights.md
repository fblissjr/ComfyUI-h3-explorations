# MiniMax-H3-Image: what the weights are, and whether Comfy-Org's int8 file is a faithful copy (2026-11-11)

lane: image
verdict: the int8 file is the public int8 recipe run on the release, code for code; on an edit it renders as close to the bf16 reference as the vendor's own servers render to each other; the image DiT is not a fine-tune of the released fl2va or ref2va

MiniMax-H3-Image was released this day. The owner had the release itself and
Comfy-Org's int8 file and asked two things of the weights: which H3 video
checkpoint the image model descends from, and whether the int8 file is good
enough or should be converted again from the release.

Data, beside this file on the box that measured it (this folder's `.json` and
`.jsonl` files are not tracked; the commands below remake them):

- `2026-11-11_h3_image_weights.jsonl`: one row per tensor, from
  `bench/measure_h3_image_weights.py` on the release, the H3 release's `FL2VA`
  and `Ref2VA` transformers, and Comfy-Org's
  `minimax_h3_image_pruned_int8_convrot.safetensors`. Its docstring defines
  every field. `bench/measure_h3_image_weights.py --summarize <that file>`
  prints the table the statements below were read from.
- `2026-11-11_h3_image_renders.json`: the render arms, their files and wall
  times.

## The int8 file against the release

- **The int8 codes are reproducible from the release, exactly.** For every
  quantized linear of every block, quantizing the release weight with
  comfy-kitchen's quantizer in the file's declared format gives the file's
  codes (`codes_equal`, whose minimum over all rows is the figure to read).
  The fused attention projection has to be put in core's row order first;
  left in the release's order the same comparison fails by a wide margin
  (`err_release_order` against `err_shipped`), which is the control that the
  reordering is real and that the comparison can fail.
- **What int8 costs at the weight** is `err_shipped`: nearly the same in every
  block and every kind of linear, with the attention output projection the one
  that varies. This is the same picture as the fl2va file's
  (`2026-09-29_int8_weight_error_fl2va.md`). It is a weight error, not an
  output error.
- **Every tensor the file leaves unquantized is the release's bytes**
  (`bench/convert_h3_image_dit.py --verify`, run this day): the norms, the
  text projection, the token refiner, the patch projection and the six output
  heads. The only tensors that differ are the AdaLN projections, which the
  file stores over a small table, and the timestep embedder, which the table
  replaces.
- **The AdaLN table's error** is in the `.modulation` rows: the table's
  modulation against the release's own, at the timesteps this model is run at
  (`adaln_err_at_used_t`) and over the table's whole grid. It is well below
  the int8 weight error. The part of the modulation that moves with the
  timestep is reproduced less closely than the whole
  (`adaln_err_moving_part_at_used_t`), because most of the modulation is a
  constant the table carries exactly.
- **The VAE** in Comfy-Org's repository is the release's fp32 VAE cast to
  fp16, tensor for tensor (compared this day with a one-off script: every
  tensor equal to the cast). **The text encoder** is not in Comfy-Org's image
  repository; the release's own shards carry the same hub hashes as the H3
  release's.

So converting the release again with the same recipe would give the same file.
The file also asks, per block, for comfy-kitchen's INT8 attention; that is not
a property of the weights and a graph can override it.

## fl2va or ref2va

- **Far from both, and about equally.** `img_fl` and `img_ref` are many times
  `fl_ref` for every kind of tensor: the image weights differ from each video
  checkpoint far more than the two video checkpoints differ from each other.
  fl2va is the nearer in every linear and every AdaLN tensor, by a margin much
  smaller than `fl_ref`.
- **Between the two on the line that joins them.** `alpha` is neither near 0
  (a fine-tune of fl2va) nor near 1 (a fine-tune of ref2va) through most of
  the depth, and falls to about 0 over the last blocks (the per-block line the
  summary prints). `cos_delta` is near 0 throughout: the image model's move
  away from fl2va is almost at right angles to the direction ref2va lies in.

Reading, labeled as one: this is what three models trained on from a shared
starting point look like, with the image model trained much further than
either video model, and with fl2va having moved less from that start than
ref2va did, least of all in the last blocks. The weights cannot name the
starting checkpoint; the model card says only that the design is H3's with the
audio branch removed.

## Our own conversion, and three render arms

`bench/convert_h3_image_dit.py` wrote the release as one bf16 file ComfyUI
loads, with nothing quantized and the full AdaLN kept. It is the reference arm,
not a file sized for the card.

Rendered through the live server on the release's two worked cases, at the
release's sizes and seeds (the JSON lists each file):

- **Against the model card.** The int8 text-to-image render has the composition
  of the card's "local" text-to-image picture, object for object, and the edit
  renders have the composition of the card's "local" edit picture. The chain
  (core's conditioning node, the release grid handed to the sampler, Euler)
  therefore reproduces the vendor's own local results from the same prompt and
  seed.
- **Between the arms, by eye.** The int8 file as shipped, the int8 file with
  exact attention and the bf16 file give the same picture in each case, with
  different fine detail. Read at full size on crops of the bread, the
  vegetables and the two hosts, none shows a defect the others lack. That is a
  reading of stills on two scenes by a session; it does not rank the arms.

## The shipped graphs, and the int8 file on an edit by numbers

After the server was restarted to load the size node, the six graphs in
`workflows/image/` were queued exactly as they are on disk
(`shipped_graphs_run_as_on_disk` in the renders JSON):

- **The two default graphs give, pixel for pixel, the pictures the hand-built
  int8 arm gave** (`psnr_db`, the two "vs the earlier hand-built int8 arm"
  entries are infinite). The size node, the generated graph and the earlier
  test are one chain.
- **The edit is the case numbers can speak to**, because MiniMax's own servers
  report that a text-to-image picture moves at a fixed seed with any small
  numerical change while an edit holds
  (`coderef/vllm-omni/recipes/MiniMaxAI/MiniMax-H3-Image-H200.md`,
  "Correctness guard"; `coderef/vllm-omni/docs/user_guide/quantization/fp8.md`,
  the MiniMax-H3-Image row). The three `edit:` entries of `psnr_db` are the
  default graph against the bf16 reference, the int8 linears alone, and INT8
  attention alone. All three sit in the range that recipe reports between its
  own bf16 GPU layouts on an edit, and above the figure the quantization page
  gives for fp8. Those two pages were measured at another size on other
  hardware, so this is a comparison of ranges, not of like with like.
- **The three `t2i:` entries are low and say nothing**, as the same pages lead
  one to expect: they are in the range those pages give for text to image
  between any two runs.

## What this does not show

- Anything about output error inside the model. No activation was captured.
- More than one edit. One reference picture, one prompt, one seed.
- Whether the int8 file holds up on fine text, dense diagrams or many
  references, which is where a quantized model would be expected to show it
  first. The probe graphs in `workflows/image/` are there to ask.
- Which arm the owner prefers.
