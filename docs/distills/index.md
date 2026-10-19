# Distills: fewer steps

| page | what it is for |
|---|---|
| [`which_distill.md`](which_distill.md) | **start here**: which few-step distill suits which shot (PDD8, PDD6, FlashGen, FastH3), what each is good and bad at, and why |
| [`pdd.md`](pdd.md) | Parallel Decoding Distillation: what it is, what ships on which checkpoint, the contract (Euler, the emitted schedule, legal step counts, shift, strength), the traps that render normally, what a bake pins, what is measured |
| [`pdd_node.md`](pdd_node.md) | how the PDD node, `pdd_math` and the converter work, and what each was checked against: the paper, Kijai's conversion, core, the other implementations |
| [`pdd_artifacts.md`](pdd_artifacts.md) | every PDD weight file on the box: which to load on which checkpoint, the sidecar tensors, the generated inventory, the converter and artifact changelogs |
| [`ref2v_distillation.md`](ref2v_distillation.md) | why an fl2v distill does not transfer to ref2v, and what to look for if one is run there |

Studies under [`../research/`](../research/index.md): the `pdd/` folder (its README orders the handoffs), FlashGen,
distill routing within a clip, the FastVideo VSA checkpoint, merge requantization of a LoRA on int8.
