# Distills: fewer steps

| page | what it is for |
|---|---|
| [`which_distill.md`](which_distill.md) | **start here**: which few-step distill suits which shot (PDD8, FlashGen, FastH3), what each is good and bad at, and why |
| [`pdd.md`](pdd.md) | Parallel Decoding Distillation: what the Acc LoRAs are, the converter and node contract, the silent traps |
| [`pdd_artifacts.md`](pdd_artifacts.md) | every PDD weight file on the box: which to load on which checkpoint, the converter and artifact changelogs |
| [`ref2v_distillation.md`](ref2v_distillation.md) | why ref2v resists step distillation |

Studies under [`../research/`](../research/index.md): the `pdd/` folder (its README orders the handoffs), FlashGen,
distill routing within a clip, the FastVideo VSA checkpoint, merge requantization of a LoRA on int8.
