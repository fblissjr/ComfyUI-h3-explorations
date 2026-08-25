# Checkpoint and resume for a sequential AWQ calibration run

last updated: 2026-08-25

Design note. A mid-run failure on a ten-hour calibration currently costs the
run; the goal is that it costs one layer. This records **what state actually
exists at a layer boundary**, verified against the installed pipeline rather
than reasoned from the API, and what therefore has to be written down.

Scope: no GPU. Every result below came from driving the real
`SequentialPipeline` with the real recipe from `bench/h3_awq_recipe.py` over a
three-layer random Qwen3 on CPU, on 2026-08-25, in the pinned
`llm-compressor` environment. Claims are labelled SOURCE where they come from
reading the installed package and MEASURED where a live object was inspected.

## The pipeline's shape, measured

MEASURED. A three-layer model traces to **four** subgraphs, of
`[1, 15, 15, 17]` modules: a prologue, one per decoder layer, and the last
carrying the trailing norm and head. `LifecycleCallbacks.sequential_epoch_end`
fires once per subgraph including the prologue, and **subgraph `k` smooths
layer `k-1`** — the prologue's epoch end smooths nothing, subgraph 1's smooths
layer 0, and so on. There is no lag beyond that offset.

The offset is a property of tracing, not a constant. A checkpoint therefore
indexes by **subgraph**, and records which layers are complete as a derived
fact; nothing should hard-code "layer N is subgraph N+1".

## What is at a boundary, and what is not

The interesting question is which modifier state a fresh process would fail to
rebuild. MEASURED, at every one of the four boundaries:

| state | at a boundary | checkpoint it? |
|---|---|---|
| `AWQModifier._smooth_activation_stats` | empty — each layer's stats are deleted by its own `_apply_smoothing` | no |
| `AWQModifier._parent_args_cache` | **registered but carrying zero filled batches**, for every parent that survives | no |
| `AWQModifier._resolved_mappings` | present, rebuilt from the model in `on_calibration_start` | no |
| `AWQModifier._error_metrics` | accumulates for the whole run, reporting only | yes, so a resumed run's report is not short |
| completed layers' weights and qparams | on the modules | **yes** |
| the pipeline's `IntermediatesCache` | see the boundary section below | **yes** |

The second row is the one worth having measured. The cache is a
`dict[parent module, IntermediatesCache]` and it is **pre-populated for every
layer** when hooks are installed, so at a boundary it looks non-empty — nine
entries for a three-layer model, shrinking to six then three then zero as
smoothing consumes each layer. Counting entries suggests live state carried
across the boundary. Counting *filled batches inside them* gives zero at every
boundary, for every surviving parent. It is structure, not data, and a fresh
process rebuilds it in `on_calibration_start`.

So the brief's reading holds, and now on evidence: resolved mappings are
recomputable and per-layer activation state is transient. The one correction
is that `_error_metrics` is neither — it is cheap to carry and a resumed run
that drops it silently produces a shorter report than an uninterrupted one.

## Where the boundary actually is

Corrected 2026-08-25 after building it: the first version of this note said the
cache holds the inputs to the *next* subgraph at a boundary. It does not, and
the resume failed on exactly that.

With `propagate_error` at its default, each iteration is: the calibration pass,
then `sequential_epoch_end`, then the propagation pass that writes the
subgraph's outputs into the cache. So at the top of subgraph `k`'s epoch end
the cache still holds the inputs to subgraph `k`, not `k+1`, and layer `k-1`
has not yet been smoothed. That instant is resumable at subgraph `k` with
layers `0..k-2` complete, and it is where the checkpoint is taken.

The placement relative to the callback has one consequence, measured, and it is
not the one guessed first. Taking the snapshot *after* the callback does not
double-smooth anything, because the layer that callback smooths is not among
the ones restored either way — the weights come out identical. What it does is
record `_error_metrics` that already include that layer while
`completed_layers` excludes it, so the resumed run re-runs the layer and the
report counts it twice. A mutation refuted the double-smoothing guess and
`bench/check_calibration_checkpoint.py::the_report_covers_exactly_the_completed_layers`
owns the real consequence.

## The completed layers' state

MEASURED. After a layer is calibrated its modules carry, per quantized linear,
`weight`, `weight_scale` and `weight_zero_point`; the scale and zero point are
new parameters that did not exist before the run. `quantization_status` is
`CALIBRATION` during the run and `FROZEN` after `end_calibration`.

**Save the whole layer, not a curated list of tensors.** AWQ smoothing writes
to the smooth layer as well as the balance layers, and on this fixture the
resolved mappings name `input_layernorm`, `post_attention_layernorm` and
`mlp.up_proj` as smooth targets — so a norm weight is a legitimate output of
calibration, not an untouched input. On the random fixture no pre-existing
tensor moved at all, which the error metrics explain: every mapping reported
`reduction: 1.0`, the grid choosing the identity scale on random weights. A
curated list built from that observation would have been wrong for the first
real run. Saving every parameter and buffer under the layer costs nothing
extra and does not depend on the observation being general.

Not the disk tier's staged files: those are named by tensor object id and do
not survive the process, as the brief says.

## The cache

SOURCE. `IntermediatesCache` is `batch_intermediates: list[dict[str,
IntermediateValue]]` plus an offload device, and `IntermediateValue` is
`(value, device)` where value is one of: tensor, list, tuple, dict, dataclass,
or a primitive from `int | str | float | bool | torch.dtype | torch.device |
None`. That closed grammar is what a serializer has to cover, and the dataclass
case is the one that needs its type recorded by module and qualname to be
rebuilt.

MEASURED. At a boundary the keys are traced FX node names — on the fixture,
`model_layers_0`, `model_rotary_emb` and two `getitem_*`. They come from
tracing, so a resume must check the restored keys against what subgraph N+1
declares in `input_names` rather than trusting that the same model produces the
same names.

## Resume, and the one real change

Everything above is readable and writable from outside the package. The part
that is not is entry at a subgraph other than the first: `SequentialPipeline`
iterates all subgraphs from zero. That needs a start index and a preloaded
cache, installed as a patch from our module, because the installed
`llm-compressor` is not edited in place.

The resumed run then: loads a fresh model through the bridge, applies the saved
layer states to layers at or below N including their quant params, restores the
cache, enters the loop at subgraph N+1, and still fires `calibration_start`
so hooks and mappings are built for the whole model. Skipped subgraphs never
see data, so their observers never fire and their restored qparams stand.

SOURCE, and the thing the bit-identical proof exists to confirm:
`end_calibration` freezes and removes observers rather than recomputing, so
restored qparams should survive to the artifact. That is a read, not a result.

## Proof, and the red control

An uninterrupted probe produces A; the same run killed at a boundary and
resumed produces B; every tensor must be bit-identical.

**That proof now runs on CPU at fixture scale**, in
`bench/check_calibration_checkpoint.py`, over the real pipeline and the real
recipe on a small random model — 78 tensors identical, resuming at subgraph 3
of 4 with 2 layers restored. It exercises every seam the card run uses:
tracing, the subgraph slice, the restored cache, the layer restore, the
observers. Four mutations of the module go red on it: the off-by-one in the
boundary index, a restore that does not load the saved layers, a resume that
does not slice, and one that rebuilds the cache from the dataloader.

The fixture needed one deliberate property. Uniform random weights give AWQ
nothing to do — every mapping reports `reduction: 1.0`, the grid picks the
identity scale, and no weight moves — so the equivalence would hold even if
resuming re-smoothed a layer, because re-applying the identity is idempotent.
The fixture is therefore built with the per-channel imbalance AWQ exists to
correct.

The card run after Gate 5 is still required and is now checking scale and the
real weights rather than the mechanism.

The red control is cheap and CPU-only: a resume pointed at a checkpoint from a
different bundle or a different recipe must refuse. The checkpoint therefore
carries the recipe's description and the bundle's identity, and the resume
compares before it loads anything.

## What this note does not establish

- That the AWQ arm runs on CPU. It does not: `_apply_smoothing` calls
  `IntermediatesCache.pin_memory`, which raises without CUDA. Every result here
  was obtained with that call neutered, which is legitimate for reading state —
  pinning is a host-to-device transfer optimisation — and is why the numerical
  proof needs the card.
- Any claim about the candidate, the run in flight, or timing. Checkpoint
  cadence is a measurement to make on the fixture, not a number to pick here.
- That smoothing moves a norm weight in a real run. On the fixture it did not,
  for the reason given, and the design deliberately does not depend on either
  answer.
