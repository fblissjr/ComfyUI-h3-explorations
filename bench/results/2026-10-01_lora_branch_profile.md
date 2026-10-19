# Where a LoRA at the call spends its time (2026-10-01)

The owner asked for an engineering pass on the at-call LoRA cost after a review
priced it as the second-largest speed lever (FlashGen's branch against the
merge: `2026-09-26_flashgen_lora_path_s1.md`, "Second scene", which calls the
path "not optimized"). This splits the cost before anything is changed.

## How

`bench/profile_lora_branch.py`, on an idle RTX 4090 with no server running,
kitchen `0.2.36+sol.aade8d5.up.3f7210f`, torch 2.14.0:

    profile_lora_branch.py --lora <file> --tokens 32768 --scale-to 110000 --iters 10 --json <row file>

It times `lora_branch._Branch.add_into` and `_mlp_forward`'s swiglu on block
25's real A and B, at the widths the DiT feeds each module, with random
activations. Rows: `2026-10-01_lora_branch_profile_flashgen.json` (FlashGen r64
for fl2va) and `2026-10-01_lora_branch_profile_pdd.json` (PDD8 for fl2va, whose
qkv branch is rank 192). The scaling to 110000 rows assumes the cost is linear
in rows; a 16384-row run on the same day agreed with the 32768-row one to within
a few percent (console output, not saved), which is the only check of that
assumption.

## What it says

Per forward at 110000 rows, from the tool's output:

- **The branch costs about 3 s per forward on both files.** Most of that is
  memory traffic, not arithmetic: the ranks are 64 or 192, so the matmuls are
  thin and each one reads or rewrites a full-width activation.
- **The host-to-device copy of A and B is negligible on its own** (`copy`),
  and the difference between `branch` and `resident` (A and B already on the
  card) is a small fraction of the branch. Keeping A and B resident would cost
  about a gigabyte of VRAM that ComfyUI does not account for, which is why
  `lora_branch.py` copies them; this does not argue for changing that.
- **`mlp.fc2` is the most expensive module, and most of it is recomputing
  swiglu.** The int8 path fuses swiglu into fc2's matmul and never writes the
  activation out, so the branch recomputes it eagerly from fc1's output
  (`swiglu_ms`, `comfy/ops.py::_swiglu_eager`). That recompute is the single
  largest line in the table.
- **qkv and fc1 are the next two**, and theirs is the in-place add: `addmm_`
  reads and rewrites the base output, which for qkv and fc1 is the widest
  tensor in the block.

## What could be changed, and what it would cost

1. **Fuse swiglu into fc2's A projection.** One kernel reading fc1's output
   once and writing only the rank-64 product would remove the recompute and
   its temporary. It is the only piece a small change in this repo can reach.
   Its ceiling is the `swiglu_ms` line plus part of fc2's own matmul, a small
   share of a render's sampler time. **It ends bit-equality with
   `H3ExactLoRA`**, which recomputes swiglu the eager way and is held to this
   node by `bench/check_mutant_parity.py`'s `torch.equal`. The published pack
   would either carry the same kernel or move to a tolerance.
2. **Fold the add into the int8 GEMM's epilogue** for qkv, fc1 and the rest:
   the base matmul writes `W x + B (A x)` in one pass, removing the add's
   read and rewrite. That is a kitchen kernel change, on the fork the owner
   builds from (`docs/working/coderef.md`, comfy-kitchen), and the larger of
   the two.
3. **Not worth doing:** keeping A and B resident (above).

Nothing is changed by this record. *The owner chose neither, 2026-10-01* (`docs/state/decisions.md`).
