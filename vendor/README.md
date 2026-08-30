# vendor/ — third-party code we read but no longer run

Files here are **upstream's, kept verbatim and git-tracked**. They live in this
repo for one reason: before 2026-08-14 the CUDA Sol-Attn node existed in three
untracked copies on this machine and nothing could say which one was running.

## The arrangement, changed 2026-08-30

`sol_attn_minimax.py` is now a **read-only reference**. The node that runs is
`sol_attn_h3.py`, a first-class module in this pack, tracked by git like every
other node here and free to change.

```
vendor/sol_attn_minimax.py     <- upstream's, PRISTINE at the last genuine drop (v3)
                                  read to compare against; never loaded
sol_attn_h3.py                 <- ours, a fork of it, what every graph wires
```

The installed pack is renamed `ComfyUI-SolAttn-cuda.disabled`, which ComfyUI
skips (`nodes.py`, `module_path.endswith(".disabled")`). Its symlink still
points here, so renaming it back restores the old arrangement exactly. Nothing
was deleted.

**Why the change.** The value of a vendored file is that it is upstream's: when
it and our expectations disagree, that disagreement is a finding rather than a
merge artifact. **That property was spent on 2026-08-29**, when the merged
kernel's API change was absorbed by editing this file in place — v3.1 and v3.2
in the table below are OURS, not drops, and while either was here the repo had
no pristine upstream copy at all. The remedy is the third option in the rules
below, which is the one they say not to reach for casually: fork it, rename it
so the fork is obvious, record the divergence. `sol_attn_h3.py`'s header is
that record.

**The restored file does not run on the installed kernel, and that is fine.**
v3 passes `centroid_tail` unconditionally, which comfy-kitchen#117 removed, so
loading it would raise inside the attention override's catch-all and become a
silent full-dense render. That is exactly why it is disabled rather than merely
superseded.

`bench/check_sol_kernel.py` asserts all of it: that the hash is a recorded
version, that it is NOT one of ours, and that ComfyUI is not loading it.

## The rule: do not edit these files

Same rule as `bench/_sol_attn_reference.py`, for the same reason. The value of
a vendored file is that it is upstream's, so when it and our expectations
disagree, **that disagreement is a finding** rather than a merge artifact.
Editing it converts every future upstream drop from a `cp` into a merge, and
converts every bug report into "does that reproduce on stock?".

**If we need a change**, in order of preference:

1. Send it upstream. This is not theoretical — the `sink_q` narrowing in v2
   below is a change this repo proposed in `docs/SOLATTN.md`, which upstream
   implemented. Cost: one message.
2. Do it outside the file — a wrapper, a `transformer_options` key, a
   composing patch. Our own `attention.py` composes with this node rather
   than modifying it.
3. Only if neither works: fork it, rename the file to make the fork obvious,
   and record every local change in a header block so re-syncing is a known
   diff rather than a surprise. **We have not done this and should not start
   casually** — the node's `node_id` is provisional upstream, so a fork buys
   a maintenance burden on something expected to be replaced.

## Version lineage

Provenance is by hand because upstream publishes this file through
conversation, not a repository. `kijai/comfy-kitchen`'s `sol_attn` branch
carries the *kernel*; there is no public repo for the *node*. Both were
checked on 2026-08-14 and neither `kijai/ComfyUI-SolAttn_triton` nor any
`ComfyUI-SolAttn-cuda` repo contains it.

| sha256 (first 16) | label | received | what changed |
|---|---|---|---|
| `3a5f0051fce61d9d` | v1 | 2026-08-14 10:48 | first drop. `sink_q = sink_blocks`, so `exact_kv_and_rows` runs every conditioning query row dense, references included. |
| `d856ba83557d18fb` | v2 | 2026-08-14 14:19 | `sink_q` narrowed to the **target audio** segment only. `PackedLayout` patch also captures the `audio` bounds, `_SPANS` carries them, `rope_freqs` publishes `sol_h3_audio_span`, `_sink_blocks` uses it and falls back to v1 behaviour when absent. **No schema change** — same 14 inputs, same order, same defaults, so no graph regeneration is needed across this upgrade. |
| `7805cf3706bf9b91` | v3 | 2026-08-22 | **Schema change, unlike v1 to v2.** `tau` and `tau_profile` fold into a `selection` DynamicCombo whose other option is `top-k (SLA)` (`keep_percent`), and `routed_cap_percent` is gone. Every graph regenerated; a graph carrying the old inputs passes ComfyUI's prompt validation and dies at execute. Requires a kernel with `topk_ratio` -- it passes the argument unconditionally -- so the node and `comfy-kitchen` `0.2.31+sol.23d1a66` move together. |
| `7805cf3706bf9b91` | **v3, RESTORED 2026-08-30** | — | The file reverted to this. Not a new drop: the last genuine upstream one, recovered from `e18bbc0`. Everything below it in this table is ours and is no longer on disk. |
| `1c55a4b51011041a` | v3.1 | 2026-08-29 | **OURS, not a drop from upstream** -- the first entry in this table we wrote rather than received, and the reason is that upstream MERGED. Sol-Attn landed in comfy-kitchen main as #117 (`dae00a1`) with a reshaped API: `centroid_tail`, `reuse_qkv_memory` and `max_blocks` gone from both entries, `tail`, `block_len` and `coarse_gate` added. The kernel call now reads `sol_attn`'s SIGNATURE and passes only what that build accepts, so one node drives both the branch build and the merged one -- necessary rather than tidy, because both declare version `0.2.31`. `centroid_tail=False` RAISES instead of being dropped: the merged kernel always evaluates the pooled tail at the query block's centroid, which is what `True` did, so our config is behaviour-preserving, but `False` is a computation this build cannot express and swallowing the request would change the math silently. **No schema change** -- same inputs, same order, same defaults, so no graph regeneration is needed. `centroid_tail` and `reuse_qkv_memory` remain as widgets and are now inert on this kernel; they are kept because removing a widget re-points every later value in every saved graph. |

**Every measurement in `docs/SOLATTN.md` dated 2026-08-14 was taken on v1**,
including the 345-frame e2e run and the entire reference-load analysis. v2
changes what `exact_kv_and_rows` costs at reference load, so those numbers
describe v1 semantics and are labelled as such.

## Re-syncing a new drop

```bash
cp <new file> vendor/sol_attn_minimax.py
sha256sum vendor/sol_attn_minimax.py          # add to the table above
python bench/check_sol_kernel.py --require    # will FAIL on an unrecorded hash
```

The check failing on an unknown hash is the point, not an obstacle: it forces
the version to be named and dated before it can be run, which is exactly what
was missing when three copies existed and none was authoritative.

Then restart ComfyUI (this is node code), confirm the reload by reading a
default back out of `/object_info`, and run the smoke — the node's own
composition seam is not covered by any static check.
