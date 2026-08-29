# How far apart fl2va and ref2va actually are

last updated: 2026-08-29

[`official_weights_metadata.md`](official_weights_metadata.md) records that the
partition split is something the release **declares** — two `model_index.json`
blocks with different task lists. This file measures what that split is
**worth in the weights**, which nothing had.

Everything here is *measured* by
[`../../bench/compare_h3_partitions.py`](../../bench/compare_h3_partitions.py)
into
[`../../bench/results/2026-08-29_h3_partition_distance.json`](../../bench/results/2026-08-29_h3_partition_distance.json):
relative Frobenius distance and cosine, in float64, over every shared
unquantized tensor. Quantized payloads are skipped and reported as skipped —
`int8_convrot` stores a rotated representation, and comparing two files through
it needs the rotation undone.

## The one-line answer is wrong, whichever line you pick

The partitions are **nearly identical where the model ends** and **unrelated in
what was learned on top**. Reporting either alone misleads.

### The base checkpoints are close

| component | relative | cosine |
|---|---|---|
| block norms (n=210) | 0.0041 | +1.0000 |
| `adaln_t_table` | 0.0183 | +0.9998 |
| input projections | 0.0226 | +0.9998 |
| **output heads** | **0.0793** | +0.9987 |

Key sets are **identical**, which is the whole reason a wrong-partition load is
silent: it matches every key and renders.

### The PDD LoRAs are not

| component | relative | cosine |
|---|---|---|
| `lora_A` (down) | 0.779 | **+0.695** |
| `lora_B` (up) | 1.390 | **+0.010** |
| adaln baked `diff` | 1.175 | **+0.0002** |
| adaln baked `diff_b` | 1.281 | +0.137 |
| head bank | 0.081 | +0.9986 |
| partition fingerprint | 0.050 | +0.9987 |

A relative distance near `sqrt(2)` with cosine near zero is the signature of
two similar-magnitude vectors pointing in unrelated directions. That is what
`lora_B` and the adaln delta are: not a degraded version of the right
correction, a different one.

## Two findings the summary hides

**The down-projections are shared and the up-projections are not.** `lora_A` at
cosine **+0.70** against `lora_B` at **+0.01** says both distillations attend to
a substantially similar input subspace and then map it somewhere completely
different. The product `B @ A` is what reaches the model, and it inherits `B`'s
orthogonality — spot-checked at cosine +0.035, −0.0001 and +0.327 on blocks 0,
25 and 49.

**The divergence has depth structure, and it runs one way.**

| blocks | mean cosine of the qkv `lora_B` delta |
|---|---|
| 0–9 | +0.006 |
| 10–19 | **−0.0002** |
| 20–29 | +0.005 |
| 30–39 | +0.055 |
| 40–49 | **+0.189** (max +0.371 at block 49) |

The two distillations are *perfectly* unrelated through the first thirty blocks
and converge steadily toward the output.

*Inference, mechanism named so it can be refuted:* the late blocks sit closest
to an output projection the partitions share to within 8%, so whatever
correction they need is constrained by geometry both partitions have. The early
blocks process the packed sequence, where fl2va's keyframe condition rows and
ref2va's reference blocks are structurally different inputs — so there is no
shared constraint there, and nothing makes the corrections resemble each other.
Nothing here tests that; it is the reading, not the result.

## What this means for the guard

`MiniMaxH3PDDLoRA` fingerprints `final_layer.video_out` and refuses beyond
`PARTITION_TOLERANCE`. The measurement says something uncomfortable about that
design and then vindicates it:

**The guard fingerprints the component that differs LEAST** — 5% — as a proxy
for a swap whose damage is in the components that differ MOST. That is sound,
because both follow from the same fact (the file is the other partition), so
they are perfectly correlated. And 5% sits an order of magnitude above the
~0.3% a dtype cast moves, which is what makes the threshold workable at all.

But **nobody should read the fingerprint distance as the size of the error it
prevents.** The heads would be 5% wrong; the learned correction would be a
full-magnitude vector pointing elsewhere.

It also explains the failure's shape. The heads are 99.9% aligned, so the
output projection is nearly right and the render looks structurally normal —
`docs/h3_pdd.md`'s "renders and looks entirely normal". What is wrong sits
underneath and has no visual signature announcing which partition it came from.

## What this means for merging the two

It does not work, and now for a stated reason rather than caution. Averaging
two orthogonal vectors gives roughly `0.7x` magnitude pointing at neither, so a
merged LoRA would be a weakened correction that is right for no partition. The
head banks would merge harmlessly at cosine 0.9986 — but the heads are not
where the distillation lives.

One incidental result: the banks being that aligned means a **single head bank
would serve both partitions**. Not worth doing at 42 MiB, but it says the
per-interval head structure is a property of H3 rather than of the task.

## Caveats

- **Quantized tensors are excluded.** 600 of 932 keys in each base checkpoint.
  Every base-checkpoint conclusion rests on the unquantized remainder, which is
  the norms, both adaln surfaces, the input projections and the output heads.
- **This is a distance, not a behaviour.** Nothing was rendered. An orthogonal
  weight delta is strong evidence the corrections are unrelated; it is not a
  measurement of what a wrong-partition render looks like, and no such render
  has been made deliberately.
- **One conversion pair.** Both PDD artifacts came from
  `bench/convert_pdd_lora.py` at the same revision, so a converter-side
  systematic would not show up as a difference here.
