# The sister checkouts: what each one is good for

last updated: 2026-08-28

`coderef/` holds the reference implementations. `ls -l coderef/` is the list of
what is currently on disk — some symlinks, some real clones — and this page is
what each one is *for*: what it implements, what has actually been compared
against it, and what it is not evidence of.

**Written by a person. Not generated** — the generator that builds
[`index.md`](index.md) never touches this file.

**Revisions are an observation point, not a pin.** Every one below was read on
the date in the header. A sister checkout moves under you; re-read before
quoting. Two of the H3 engines moved during the 2026-08-28 comparison pass
itself.

**Do not import Python from `coderef/`.** CLAUDE.md's rule, with the escaped
instance that earns it: requiring the clone and prepending it to `sys.path` is
how a bench script made itself unrunnable on a box that had the wheel and no
checkout. Use the clone for sources you cannot import; import the rest.

---

## The four H3 implementations worth comparing against

These are the ones that implement MiniMax H3 end to end. All four were compared
against our node chain on 2026-08-28; the findings live in
[`../custom_node_gaps.md`](../custom_node_gaps.md), which this page routes to
rather than restating.

| checkout | revision read | what it is | reach for it when |
|---|---|---|---|
| `sglang` | `803b4fb31c` | **the vendor's own serving path.** The closest thing to ground truth for what MiniMax intended | you need to know what the release actually does at a stage |
| `LightX2V` | `5169278f` | inference engine; **origin of the SLA work and the Turbo LoRAs we load** | anything about SLA, DMD step distillation, offload, or what a LoRA was distilled under |
| `DiffSynth-Studio` | `102fe99` | model library with a native H3 pipeline, its own converters and a LoRA path | you need a second opinion on a state-dict namespace or a converter |
| `diffusers` | `9f7aee482` | model library with a native H3 pipeline, a named H3 scheduler, and a conversion script | you need the canonical tensor namespace, or a clean statement of the sampler |

Two owner documents already exist for the first of these and are the authority
over anything here: [`../research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md)
for what sglang does stage by stage, and
[`../research/sglang_comparison.md`](../research/sglang_comparison.md) for what
its serving path does that we do not.

### What the 2026-08-28 pass established about them

Recorded here because it is a property of the *references*, not of our code:

- **Three-against-one is a real signal and it fired twice.** sglang, DiffSynth
  and diffusers agree on feeding one prepared reference tensor to both towers,
  and on running the video VAE more precisely than we do. Both are open.
- **Agreement is broad and worth banking.** The reference label rules, the
  encoder layer, the absence of a chat template, and the VAE normalisation
  statistics all agree across implementations. When four implementations agree,
  a fifth reading is not the cheapest next step.
- **Neither model library is independent evidence about the seven markers.**
  Both inherit the release tokenizer without touching the ids in code. Two more
  implementations is not two more votes.
- **No engine implements PDD.** diffusers, LightX2V, DiffSynth and sglang were
  each searched. See [`../research/pdd/pdd_implementations.md`](../research/pdd/pdd_implementations.md).

---

## The ComfyUI-side references

| checkout | revision read | what it is |
|---|---|---|
| `comfy-kitchen-sol` | `bd3fc78` | **the most-cited clone here.** Its `.cu` files ship in no wheel, so `morton.md` and `sol_upstream.md` quote it by path. The built branch is installed, so import the Python rather than requiring the clone |
| `comfy-kitchen` | `7490d87` | the upstream of the above |
| `ComfyUI-UtilsCollection` | `5bac35b` | a third-party pack with its own PDD path. Two of our guards were **adopted from it** |
| `Minimax-H3-Turbo` | `02e26d5` | the vendor README that publishes the distilled sigma grid `bench/check_distill_grid.py` grades against — a grid from the vendor, not one we computed |
| `sage-fork` | `56a5be4` | our SageAttention fork |
| `SLA` | `7db4039` | the sparse top-k attention reference |
| `TurboDiffusion` | `e3d6136` | step-distillation reference |

---

## The upstream and infrastructure clones

Not H3 implementations. Listed so nobody mistakes one for a comparison target.

| checkout | what it is | what it is not |
|---|---|---|
| `MiniMax-H3` | the release repository | not a runnable pipeline for our purposes |
| `MiniMax-Music3` | a different model | not H3 |
| `transformers`, `vllm`, `llm-compressor` | encoder-side and quantisation infrastructure | say nothing about the DiT |
| `triton`, `flashinfer`, `nanobind` | kernel infrastructure | |
| `Sana`, `h3-turbo-eval` | adjacent research | |

---

## The trap this page exists to prevent

**Two models live in this repo, and the words for their parts do not
disambiguate the stage.** "Attention" and "capture" each name something at the
DiT *and* at the Qwen3-VL encoder. A fact about one is not a fact about the
other, and three instances of carrying a DiT-side fact to an encoder-side
conclusion happened in a single day. CLAUDE.md holds the full rule; the tell is
always a type or a module prefix, never the vocabulary of the claim.

This applies with extra force to the sister checkouts, because a clone gives you
a confident, well-written source for the wrong stage.
