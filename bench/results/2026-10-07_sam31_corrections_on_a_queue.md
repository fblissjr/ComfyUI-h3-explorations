# The SAM 3.1 Corrections node on a server's queue, with an H3 render between its reads (2026-10-07)

lane: masked
verdict: on a server with dynamic VRAM, a stock SAM 3.1 pair and its corrected clone sharing one model each get their own first layer in both orders, the node wired twice corrects once, and both still hold after a real H3 sample evicted SAM from the card and after all models were freed; not covered: SAM called while the video model holds most of the card

**What was asked.** The owner, 2026-10-07, on patch nodes in general:
"every offload that happens in comfy ... i *believe* they lose their
patches in the process - but verify this". And the lead's condition for
registering the node: a stock model and its corrected clone in one graph
on the server's patcher class, in both orders, because ComfyUI's
`ModelPatcher.clone_has_same_weights` does not compare object patches.

**What the node is.** [`sam31_corrections.py`](../../sam31_corrections.py):
the model and text encoder from any checkpoint loader in, clones carrying
two object patches out (the image range at the trunk's first layer, the
text encoder's activation; the two departures of
[`2026-10-07_sam3_core_against_meta.md`](2026-10-07_sam3_core_against_meta.md)).

**How.** [`bench/sam31_corrections_queue_test.py`](../sam31_corrections_queue_test.py).
A scratch server on its own port, started with the main server's launch
script and a scratch base directory, in default mode; its log says dynamic
VRAM was detected and enabled. It served a COPY of this pack at the commit
before this record's, in which alone the node and a test-only probe node
([`bench/sam31_probe_node.py`](../sam31_probe_node.py)) were registered, so
nothing unaccepted was on the main server. The probe calls ComfyUI's own
`SAM3_Detect` on a frame it makes (a ramp with a block of exact black and
one of exact white) and reports, from inside the server, the range that
reached the trunk's first convolution, whether the text encoder ran exact
GELU, the patcher class, and whether SAM was on the card before the call.
One graph: ComfyUI's `CheckpointLoaderSimple` feeding six probes wired to
run in a fixed order. Then the shipped text-to-video graph cut to two
steps, which loaded the text encoder and the video model and sampled. Then
the six probes again, then all models freed, then the six a third time.
The numbers are in
[`2026-10-07_sam31_corrections_on_a_queue.json`](2026-10-07_sam31_corrections_on_a_queue.json);
the table is printed from it by the tool's `render`.

The H3 prompt between the rounds: `h3_text_to_video_api.json` at 2 steps and 73 frames, status success. Overall: PASSED.

| round | probe | patcher class | SAM on the card before the call | first layer saw | text ran exact GELU | verdict |
|---|---|---|---|---|---|---|
| before the H3 prompt | stock | ModelPatcherDynamic | False | [0.0, 1.0] | False | ok |
| before the H3 prompt | corrected | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| before the H3 prompt | stock again | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |
| before the H3 prompt | corrected twice over | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| before the H3 prompt | corrected again | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| before the H3 prompt | stock last | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |
| after the H3 prompt | stock | ModelPatcherDynamic | False | [0.0, 1.0] | False | ok |
| after the H3 prompt | corrected | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after the H3 prompt | stock again | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |
| after the H3 prompt | corrected twice over | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after the H3 prompt | corrected again | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after the H3 prompt | stock last | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |
| after /free | stock | ModelPatcherDynamic | False | [0.0, 1.0] | False | ok |
| after /free | corrected | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after /free | stock again | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |
| after /free | corrected twice over | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after /free | corrected again | ModelPatcherDynamic | True | [-1.0, 1.0] | True | ok |
| after /free | stock last | ModelPatcherDynamic | True | [0.0, 1.0] | False | ok |

## What this shows

- **The corrections are not lost when ComfyUI evicts SAM and loads it
  again** (measured: the first probe after the H3 prompt, and after the
  free, reports SAM off the card before its call, and every read after is
  right). They live on the patcher, and ComfyUI applies a patcher's object
  patches on every load (read: `comfy/model_patcher.py`, `patch_model` and
  the dynamic class's `partially_load`).
- **A stock pair and its corrected clone do not leak into each other**
  (measured, both orders, three rounds), on the dynamic patcher class.
- **Wired twice in a row, the node corrects once** (measured).
- The same on the real model in one process, through the track node and a
  direct trunk call as well, and after `unload_all_models`:
  `bench/check_sam31_corrections_on_card.py` (six cases, green on
  2026-10-07; it needs the card).

## What it does not show

- **SAM called while the video model holds most of the card**, where the
  dynamic class loads SAM in part under pressure. Here SAM was evicted
  whole and reloaded whole.
- A long render, another video model, or another server mode than default.
- That the corrected model segments better: this record is about whether
  the corrections are in force, not what they are worth. For that, the
  comparison record above.

## Files

- `2026-10-07_sam31_corrections_on_a_queue.json`: the three rounds' probes
  and the H3 prompt's status.
