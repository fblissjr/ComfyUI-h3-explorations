# vendor/ — third-party code we run but do not own

Files here are **upstream's, kept verbatim and git-tracked**. They live in this
repo for one reason: before 2026-08-14 the CUDA Sol-Attn node existed in three
untracked copies on this machine and nothing could say which one was running.

## The arrangement

```
vendor/sol_attn_minimax.py                              <- tracked, THE source of truth
  ^
  | symlink
custom_nodes/ComfyUI-SolAttn-cuda/sol_attn_minimax.py   <- what ComfyUI loads
```

The installed path is a **symlink**, not a copy, so the tracked file and the
running file cannot diverge. Verified: ComfyUI's loader imports through it
(`spec_from_file_location` follows symlinks) and reports the same 14 inputs.

Consequence worth knowing: the node pack now depends on this repo being
present at that relative path. That is a real coupling and it is the price of
the guarantee.

`bench/check_sol_kernel.py` fails if the installed path stops resolving here,
or if the file's hash is not one of the versions recorded below.

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
