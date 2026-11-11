# MiniMax-H3-Image: MiniMax's own still-image model

Last updated: 2026-11-11, the day it was released and first run here.

MiniMax-H3-Image makes a picture from text, or edits one or more pictures, on
one checkpoint, in six steps with no guidance. It is a different model from
the two image paths this repo already had: PrunaAI's P-H3-Image
([`wiki/p_h3_image.md`](wiki/p_h3_image.md)) and a single frame of H3 video on
Pika ([`h3_image_editing.md`](h3_image_editing.md)). This page is what the
release's files are, what we checked about them, and how the workflows here run
it. The release's model card is the authority on the model; it is read from the
download, not kept in this repo.

## What it is, from the model card and the files

- **The DiT is the H3 video DiT without its audio layers**, the same depth and
  width. Its tensor names are the fl2va release's minus the audio projection
  and the audio output head (the two releases' shard indexes, compared
  2026-11-11).
- **The text encoder is H3's, unchanged.** The release's shards carry the same
  hub hashes as the H3 release's, and its tokenizer and processor files are
  byte for byte the same (checked 2026-11-11 by comparing the two downloads).
  The graphs load the encoder the video graphs load.
- **The VAE is its own file.** Its encoder weights are the H3 video VAE's
  (equal to fp16 rounding on every encoder tensor compared, 2026-11-11); its
  decoder is a smaller single-picture decoder with different shapes. The video
  VAE cannot stand in for it.
- **It samples on a fixed grid of six steps** with one output head per step,
  and the grid is part of the checkpoint (`workflows/h3_config.py::H3_IMAGE_MANUAL_SIGMAS`).
  There is no step count, guidance scale or negative prompt to set.
- **It is built for large pictures.** The card gives a working range in
  megapixels and a best size (`H3_IMAGE_MEGAPIXELS_RANGE`, `H3_IMAGE_MEGAPIXELS`
  in `h3_config.py`), with both sides a multiple of 32.

## Is it fl2va or ref2va underneath?

Neither, as released. `bench/measure_h3_image_weights.py` compared every block
of the three DiTs, and the record is
[`bench/results/2026-11-11_h3_image_weights.md`](../bench/results/2026-11-11_h3_image_weights.md):

- The image weights are far from both video checkpoints, many times the
  distance the two video checkpoints are from each other, and about equally
  far from each.
- Along the one direction that separates fl2va from ref2va, the image weights
  sit between the two through most of the depth, nearer fl2va, and at fl2va's
  end in the last blocks. Their own move away from fl2va is nearly at right
  angles to that direction.

That is what a model trained on from the family's shared starting point looks
like, not a fine-tune of either released video checkpoint. In how it is *run*
it is ref2va's design: reference pictures ride in the sequence ahead of the
picture being made, each named `<Picture N>`, held just short of clean, and
text to image is the same thing with no pictures. Which checkpoint MiniMax
actually started from is not something the weights can say.

## The files on this box

| file | what it is | made by |
|---|---|---|
| `H3_IMAGE_UNET` | the DiT, int8 linears, pruned AdaLN; the graphs' default | Comfy-Org |
| `H3_IMAGE_UNET_BF16` | the DiT as released, nothing quantized or pruned. Not kept on disk: written, rendered and removed on 2026-11-11 | `bench/convert_h3_image_dit.py` writes it again from the release |
| `H3_IMAGE_VAE` | the VAE in fp16 | Comfy-Org, a plain cast of the release's |
| `H3_IMAGE_VAE_FP32` | the VAE as released | the release |

The names are constants in `workflows/h3_config.py`. The files on disk are
links from ComfyUI's model folders into the model store.

## Is Comfy-Org's int8 file a faithful copy? Yes

The same record holds the evidence:

- **Its int8 weights are the public recipe run on the release.** Quantizing the
  release's weights with comfy-kitchen's own quantizer, in the format the file
  declares, gives the file's int8 codes exactly, in every linear of every
  block. Converting it again ourselves the same way would produce the same
  bytes.
- **Everything it does not quantize is the release's bytes**, the fused
  attention rows reordered the way core needs them and the six output heads
  stacked the way core reads them.
- **The one approximation besides int8 is the pruned AdaLN table**, and at the
  timesteps this model is ever run at its error is well below the int8 error.
- **It also carries a request for INT8 attention** in every block: on a graph
  with no attention node, core runs comfy-kitchen's INT8 attention on this
  file. That is the pack's own dense kernel (`h3_config.DENSE_BACKEND_NODE`),
  and the default graphs name it. The owner approved the file's request on
  2026-11-11 (`bench/results/model_contents_baseline.json`), and
  `bench/check_model_contents.py` now fingerprints this file and the VAE, so
  a file replaced under the same name shows.

So there was nothing to gain from redoing the same conversion. What we did
write is the opposite end: the release whole in bf16, as the reference the int8
file can be judged against. The record has the comparison renders. On the
release's two worked cases every arm gives the same picture with different
fine detail, and a session reading crops at full size found no defect in one
arm that the others lack. That does not rank them, and for text to image a
pixel comparison cannot: MiniMax's own servers report a text-to-image picture
moving at a fixed seed with any small numerical change, while an edit holds
("What the vendor's servers do" below). On the edit the arms were compared by
numbers too, and the record has them.

## The workflows

In [`workflows/image/`](../workflows/image/), written by
`workflows/build_workflows.py::build_h3_image_api`:

| graph | what it does |
|---|---|
| `h3_image_t2i_api.json` | text to image |
| `h3_image_edit_api.json` | an edit of a picture, at the picture's shape |
| `h3_image_t2i_heylook_api.json` | short text -> `MiniMax H3 Image Prompt Expander (heylook)` -> text to image |

Four probe graphs rendered the same two cases on the bf16 file, with the same
INT8 attention and with core's exact attention, so the int8 linears and INT8
attention could each be judged alone. They and the bf16 file were removed on
2026-11-11 at the owner's word once the comparison was recorded; the builder
still takes `unet=H3_IMAGE_UNET_BF16` and `attention=None` to make them again.

The chain follows the vendor's diffusers pipeline
(`coderef/diffusers/src/diffusers/modular_pipelines/minimax_h3_image/`) on
core's nodes. Core's conditioning node was read against that pipeline and
agrees with it on what matters: the sequence layout, the picture labels, the
level references are held at, the reference sizes for the VAE and for the text
encoder, and a sampled rather than mean VAE encoding of each reference. Three
small things differ and were left alone: core draws the reference noise from a
fixed seed where the pipeline uses the request's, rounds an in-bounds
reference to a multiple of 32 where the pipeline crops it, and skips the
pipeline's rounding of reference latents to fp16. The chain differs from
Comfy-Org's shipped templates in three places:

1. **The six-step grid is handed to the sampler as written.** The templates
   ask a scheduler for six steps, which lands near the grid the output heads
   were distilled on rather than on it. The heads are one per step of this
   grid under plain Euler; no other sampler, step count or grid is valid.
2. **The size is megapixels and a shape** (`MiniMax H3 Image Size`,
   [`h3_image_nodes.py`](../h3_image_nodes.py)), at the card's best size by
   default. The templates start at the bottom of the working range.
3. **A picture being edited reaches the model untouched.** The edit template
   resamples it first. The vendor's pipeline keeps a reference's own pixels
   whenever its short side is within bounds, and core's node already does the
   bounding.

Attention is the same as the templates': comfy-kitchen's INT8 attention, named
in the graph. The vendor's pipeline computes exact attention, which is what
the bf16 reference arm ran.

The two default graphs carry the release's own worked cases at the release's
seeds. Rendered here on 2026-11-11, each gives the composition of the
corresponding "local" picture on the model card, as a session read the stills:
the farm table, and the two podcast hosts with the headphones gone and fists
touching over the sign.

Both graphs, and the four bf16 probes while they existed, were run as they
were on disk on 2026-11-11, after the server was restarted to load
`MiniMax H3 Image Size`.

## What the vendor's servers do

MiniMax's release names sglang and vllm-omni as the servers to run it on, and
the sglang code is the closest thing to the vendor's own reference. Both were
read on 2026-11-11 (the `coderef/sglang` and `coderef/vllm-omni` clones, by a
subagent, the lines below then checked by hand). They agree with the diffusers
pipeline on everything the chain above takes from it. What they add:

- **A text-to-image picture is not stable at a fixed seed; an edit is.** The
  vllm-omni recipe for this model
  (`coderef/vllm-omni/recipes/MiniMaxAI/MiniMax-H3-Image-H200.md`,
  "Correctness guard") says text-to-image outputs of its different GPU
  layouts differ from each other and that the same spread appeared inside the
  sglang reference; its quantization page
  (`coderef/vllm-omni/docs/user_guide/quantization/fp8.md`) says the same of
  fp8 against bf16, with edits staying close. So a text-to-image pair at one
  seed says nothing about a quantized file. Compare edits, or many seeds.
- **Precision.** sglang runs the DiT in bf16 with its fp32 islands kept,
  encodes references with the VAE's fp32 weights and decodes in bf16
  (`coderef/sglang/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3_image.py`).
  The edit graphs here do the same since 2026-11-11 (the owner's decision):
  `MiniMaxH3VAEPrecision` promotes the encoder to fp32 for the picture being
  edited and leaves the decoder in half precision. The encoder is promoted
  from the fp16 file, because rounding the release's fp32 encoder weights to
  fp16 loses nothing that matters (compared tensor by tensor that day), so the
  fp32 VAE file is not loaded. On the release's edit case the change moves the
  render about as much as the int8 file does against bf16
  (`edit_graphs_with_fp32_reference_encode` in the renders JSON), which is to
  say within the spread the vendor's servers show between their own layouts.
  It is the vendor's way of encoding; nothing here shows it looks better.
- **On the four small points where core differs from the diffusers pipeline,
  the two servers side with the pipeline**: the reference's posterior sample
  and its noise come from the request's seed, the sampled latent is rounded to
  fp16, and an in-bounds reference is cropped rather than resized. sglang and
  diffusers draw those in a different order from each other, so there is no
  single bit-exact edit to match; the differences are in the noise a
  reference carries at its conditioning level, not in what the model is shown.
- **The noise is drawn on the CPU in the release's own example requests**
  (`generator_device=cpu`), which is what ComfyUI does. That is why the default
  text-to-image graph, at the release's seed, lands on the model card's
  "local" picture.
- **vllm-omni's advice for edits** (same recipe): give references at the
  canvas size of the requested shape, and make several changes in one edit
  rather than chaining edits. The first is not what the release's own example
  does (a small picture, a large output) and has not been tried here.
- **Neither server contains the prompt expander.** It is its own model
  (`MiniMax-H3-Image-PE`), served as a language model; its repository's README,
  kept at [`../vendor_guides/h3_image/README.md`](../vendor_guides/h3_image/README.md),
  says how each request is put together and how the reply is used.

## Prompts

The model is given one long, literal English description, used as it is.
MiniMax write that description with a language model, and published the
instructions they give it in [`../vendor_guides/h3_image/`](../vendor_guides/h3_image/)
alongside the release's worked examples.
[`wiki/system_prompts/h3_image/`](wiki/system_prompts/h3_image/README.md) holds
our standalone, English-only system-prompt suite for local thinking LLMs/VLMs
(general scenes vs. design/typography/infographics for T2I, and `1–16`
`<Picture N>` edits, each available as plain prose or with a trailing
`aspect_ratio:` line parsed by `h3_image_nodes.py::parse_expander_reply`).

- **Text to image**: the description opens with what the image is and walks the
  frame (`"double-quoted"` visible text). The ratio variant appends
  `aspect_ratio: <10 presets>`.
- **An edit**: one continuous 3-part paragraph (`'single-quoted'` visible text)
  that binds independent source elements as `<Subject N>`, states what the
  target image shows and how each `<Subject N>` is preserved or transferred, and
  describes only changed or newly introduced details. The ratio variant appends
  `aspect_ratio: <same as the picture | W:H>`.

The two default graphs (`h3_image_t2i_api.json`, `h3_image_edit_api.json`) take
a finished description directly. `h3_image_t2i_heylook_api.json` wires
`MiniMax H3 Image Prompt Expander (heylook)` (`h3_image_nodes.py::MiniMaxH3ImagePromptExpander`)
ahead of conditioning and `MiniMax H3 Image Size.shape_from_expander`, expanding
a short request over `POST /v1/messages` on a local or LAN `heylookitsanllm`
server (`base_url` defaults to `http://127.0.0.1:8080`; model ids are resolved
at runtime from `GET /v1/models`). Not established here: how much a short plain
prompt loses without an expander.

### Prompts written for H3 video

Tried on 2026-11-11 at the owner's ask: six cases with prompts in the two H3
video guides' formats, unchanged, each on the int8 workflow and on the bf16
reference at one seed (`videoprompt_cases` in the renders JSON; side-by-side
sheets `videoprompt_sheet_*.jpg` beside the renders). What a session saw in
the stills, for the owner to check:

- **The model takes the format.** Every case gave a coherent picture of the
  scene the prompt describes. No field name, speaker tag or line of dialogue
  was drawn as lettering, and the sound and music fields left no visible
  trace.
- **`[Shot N]` is read as panels.** A one-location prompt with three shots
  (the covered market) came out as one frame. Two prompts whose three shots are
  different framings (a diner, a stairwell) came out as one image divided into
  panels, one per shot, in the prompt's order.
- **The reference format works for an edit.** With one picture and the six
  sections of the reference guide, the woman in the picture stands in the
  diner the prompt asks for, with her face, hair and freckles. With two
  pictures and a three-shot script, the result is a grid of four panels, the
  first shot drawn twice across the top and the two close-ups below, the
  elderly man taken from his picture without his cap or the mark on his
  cheek. The same script as text to image gave three panels.
- **Words in the prompt outrank the picture where they disagree.** That
  two-picture prompt describes the woman with dark shoulder-length hair; the
  picture wired for her is of a woman with long blonde hair and freckles. The
  render gave her dark hair, and her face does not read as the picture's
  either (crops of the picture, the diner edit and the stairwell edit side by
  side: the diner edit is plainly the same person, the stairwell woman is
  not). The man, whose description agrees with his picture, kept his face.
- **Picture numbering did not decide who was who in that case.** Wired in
  either order, the two pictures gave nearly the same image: each subject
  followed its description to the matching picture.
- **int8 and bf16 agree in layout** on all six. Within it they can differ in
  more than detail: on the diner prompt both drew two interior panels over a
  wide exterior, but which of the couple shares the first panel with the cook
  is swapped between them, and the exterior shows both people on one and the
  woman alone on the other.

One seed each, so none of this is a rule. It is consistent with the image
model having been trained on the family's prompt conventions, which is what
the shared text encoder and design would lead one to expect.

## Not done

- Edits with more than one picture have not been rendered here. The builder
  wires up to six; the card says the model takes sixteen.
- No LoRA, control vector or edit guidance has been tried on this model.
- The release's fp32 VAE file (`H3_IMAGE_VAE_FP32`) is linked and not loaded
  by any graph. Nobody has compared an fp32-encoded edit with an fp16-encoded
  one by eye.
- No control vector made on a video checkpoint is expected to carry over as it
  is: the vectors live after the token refiner, and this model's text
  projection and refiner are its own weights (the record, "The same holds
before the first block"). One
  would have to be made on this model; whether the vector nodes run on it has
  not been tried.
