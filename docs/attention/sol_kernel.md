# The Sol kernel: install, verify, and what it exposes

Detail page of [`sol_attn.md`](sol_attn.md). The kernel is `comfy_kitchen.sol_attn`; the node that drives it is
`MiniMaxH3Sol` in `sol_attn_h3.py`.

## Which build runs

`comfy_kitchen.sol_attn` ships in the stock wheel ComfyUI pins (since kitchen `0.2.33`; Comfy-Org/comfy-kitchen#117
merged 2026-08-29). This box runs a **local build of the owner's kitchen fork** instead, because three things the node
uses are not upstream:

- `blk_cnt`, the route observer's out-parameter (open as kitchen PR 168). A stock wheel fails
  `bench/check_sol_kernel.py`'s `observable` case.
- `qk_balance`, the per-head q/k channel rebalancing inside Sol's INT8 quantizers (`block49_quant_error.md`; the PR
  text is [`sol_upstream_qk_balance_pr.md`](sol_upstream_qk_balance_pr.md), not opened).
- `rotate`, the Hadamard rotation of q/k before INT8, which the `rotated` quantizer selects.

**Read the build, never a sentence about it.** `bench/check_sol_kernel.py` prints the importable build;
`comfy_kitchen_build.json` beside the ComfyUI venv names the branch, our commit and the upstream base it came from,
and the installed version carries both (`+sol.<ours>.up.<base>`). Since 2026-09-27 the fork's build branch is
`h3-frontier`, upstream main plus our commits, checked out in the clone itself; the "The build branch" header of
`vendor/rebuild_kernel.sh` owns the fork's layout. The wheel is installed, not editable, so changing branches in the
clone does nothing until the next rebuild.

| what | where |
|---|---|
| source | the owner's fork clone `coderef/comfy-kitchen` (a symlink to the workspace checkout); the script finds the branch by name |
| the local version tag | a sed on `pyproject.toml`'s version line inside `vendor/rebuild_kernel.sh`, reverted on every exit path; the local segment is the built commit's short sha |
| wheel | `<source>/dist/`, one per build, never cleaned |
| installed | the ComfyUI venv's `site-packages`, as a built wheel |
| current against ComfyUI's pin? | `vendor/rebuild_kernel.sh --check` |

The clone is also where the `.cu` sources live that [`sol_upstream.md`](sol_upstream.md) and the research pages cite
by path; they ship in no wheel.

## Install it

On this box, use the script. It derives the version tag, reverts its patch on every exit path including a failed
build, and runs `bench/check_sol_kernel.py --require` afterwards so a build that installed but is not usable fails
loudly:

```bash
vendor/rebuild_kernel.sh 89        # your arch here
```

The manual equivalent, for a box without this repo, as it was run before the script existed (it clones kijai's
`sol_attn` branch, which predates the upstream merge; a build of upstream main or the owner's fork takes the same
steps):

```bash
git clone -b sol_attn https://github.com/kijai/comfy-kitchen.git
cd comfy-kitchen
git submodule update --init --depth 1 third_party/flash-attention third_party/cutlass
uv pip install 'nanobind>=2.0.0'
COMFY_CUDA_ARCHS=89 uv build --wheel --no-build-isolation .   # your arch here
uv pip install --force-reinstall --no-deps dist/comfy_kitchen-<version>.whl
```

Four things that bite:

- **`--no-build-isolation` is required**, or arch detection sees no GPU and compiles every target. Build deps are then
  not installed for you; `nanobind` is the one that is missing.
- **Submodules are not optional.** `flash_decode.cu` includes `flash.h` from `third_party/flash-attention`.
- **Tag your build.** A source build declares the same version as the wheel ComfyUI pins, so nothing can tell them
  apart. PEP 440 still matches a local segment against the pin, so a requirements install stays satisfied.
- **Install the wheel by exact version**, not `comfy_kitchen-*.whl`: `dist/` keeps every build.

Requires head_dim 128 and sm_80 or newer. The direct entry takes bf16, and fp16 since kitchen PR 150, which is why
`MiniMaxH3Sol` accepts an fp16 DiT (`sol_attn_h3.py::_apply_sol`).

## Verify it, because failure is silent

If the kernel is swapped out after the fact (a `--force-reinstall`, a Manager repair, a fresh venv) nothing errors:
the override declines every call and falls back to dense, and the render succeeds slower and numerically different.

```bash
python bench/check_sol_kernel.py --require     # kernel present, right backend, signature intact
python bench/check_solattn_correctness.py      # the CUDA kernel against the eager reference
python bench/smoke_h3.py --workflow h3_text_to_video_api.json --log <comfyui.log>
```

The smoke is the only one that submits a prompt. The prefix is `[h3-sol]` (`sol_attn_h3.py::_log_once`; it was
`[sol_attn]` until 2026-09-14). What to look for:

- `chaining onto an existing attention override`: Sol found the dense backend's override and chains onto it (the
  ordering check, [`sol_wiring.md`](sol_wiring.md)).
- `sparse (1, <T>, 56, 128) tau=... cuda-int8`: the CUDA kernel ran. `cuda-int8` is the difference between the kernel
  running and a silent fallback.
- `conditioning sink: KV blocks (a, b) exact, dense query blocks (c, d)`: the sink ranges
  ([`sol_sink.md`](sol_sink.md)).
- the `[h3-sol] on:` line printed once per patch: the sigma window with the percents behind it, the selection rule,
  the quantizer, the token-routing and dense block counts, and what the fallback is.

## What the kernel exposes, and what the node does with each

Read from `sol_attn`'s signature and `coderef/comfy-kitchen-kijai/comfy_kitchen/constraints.py::sol_attn_common_call_rule`; the node reads the
signature at patch time and passes what the build accepts.

| kernel argument | status here | what it does |
|---|---|---|
| `tau` | exposed | threshold selection |
| `topk_ratio` | always 0 since 2026-09-27 (the SLA lane closed) | fixed-density selection |
| `scale`, `sink_blocks`, `sink_q` | derived from core's packed layout | the conditioning sink |
| `tail` | always on | unselected blocks contribute one pooled term each: Sol's own correction |
| `tau_map` | passed only when `tau_table` or `rows` asks for it | a tau per (head, query block); also how per-segment exact rows are expressed |
| `key_bias` | not exposed, deliberately | a per-key log-space bias, legal only where the biased keys are sink-covered (on H3, the conditioning rows). An untrained prompt-adherence knob; `sol_attn_h3.py` documents why it is not offered |
| `block_len` | not exposed, inert here | live rows per block, for a caller that pads. H3's packed sequence is contiguous |
| `coarse_gate` | not exposed, by choice | VSA's gated coarse branch, a learned projection of the block input. Reachable from an override through a pre-hook that stashes the block input; it lives with VSA because VSA also needs its own reorder and padding and needs a checkpoint that carries the gate |
| `sol_attn_chunked` | unreachable from the override; `MiniMaxH3SolChunked` (`sol_chunked_h3.py`) reaches it from the attention module's forward | never materialises Q/K/V, but by the time an override runs the projection and RoPE are done, so its saving is spent. Upstream reports about 5 GB less peak at 113k tokens. No shipped graph uses it |
| `blk_cnt` | ours, passed only when `H3_SOL_OBSERVE` is armed | an int32 `(B, H, ceil(T/64))` out-parameter: how many key blocks each query block attended exactly. Unarmed, nothing is passed, so an older wheel keeps rendering |
| `qk_balance`, `rotate` | ours, exposed through `quantizer` | `qk_balance`: q scaled by f and k by 1/f per (batch, head, channel) inside the INT8 quantizers, f from the call's own channel rms (alpha 0.5, gated on the top-4 K energy share). `rotate`: a fixed sign-diagonal plus Hadamard rotation of every q/k row before INT8. Both exact for q.k; the threshold, kmean, kcvar and the coarse branch stay in the unrotated, unbalanced space. CUDA only; HIP refuses True; `sol_attn_chunked` takes neither |

**The exact branch is all-INT8.** `sol_attn_exact.cu` runs `mma_s8` for QK into int32 and `mma_u8s8` for PV (uint8 P,
int8 V) with fp32 accumulation and bf16 I/O. There is no 16-bit MMA anywhere in the Sol kernels and no option for
one.

**Dense calls take core's container entry, since 2026-10-03.** Core hands attention backends q, k and v in
single-owner containers (`comfy/ldm/modules/attention.py::AttentionTensorContainer`), and kitchen's INT8 backend frees
the bf16 tensors once it has quantized them. The override now carries that entry whenever the fallback under it does
(core's Model Attention Backend on the shipped graphs), and a declined call passes the containers on. Output is
unchanged bit for bit; the saving is memory on dense calls only, and a Sol call still sets attention's peak
(`bench/results/2026-10-03_sol_container_protocol.md`; `bench/check_sol_node_equivalence.py` holds the hand-off).

## The node's history, in brief

- **Through 2026-08-29** the node was kijai's vendored `SolAttnMiniMax` (`vendor/sol_attn_minimax.py`, now a
  read-only reference kept byte-identical to its last upstream drop; `vendor/README.md` says why).
- **2026-08-30 to 2026-09-27** it was `MiniMaxH3SolAttn`, this pack's fork of it; the migration was output-neutral,
  measured (`bench/results/2026-08-30_sol_node_equivalence.json`).
- **Since 2026-09-27** it is `MiniMaxH3Sol` (the redesign record above, evidence in
  `bench/results/2026-09-27_sol_node_compound_audit.md`). The sink reads core's `minimax_h3_layout`; nothing in the
  node patches core's `PackedLayout` or `_forward` any more.

**The Triton backend was deleted on 2026-08-16** (commit `6872dfd`); it is recoverable from
`github.com/kijai/ComfyUI-SolAttn_triton` at `842c4ea`. Every Sol number older than 2026-08-14 was taken on it, so
those numbers are unreproducible without recovering it. Two consequences that still apply: its `SolAttnBlockProbe`
(per-block sparse-against-dense error) went with it, and its replacement is the capture graders and the route
observer, not a port; and `bench/check_solattn_correctness.py` now grades CUDA alone, having been decoupled from the
Triton import before the deletion so it could not turn into a permanent skip. Its knob vocabulary (`int8_qk`,
`int8_pv`, `use_tma`) survives only in `h3_config.SOL_RECOMMENDED` and `SOL_BASELINE_124F`, and `int8_qk` selected a
different kernel rather than a dtype (`842c4ea:__init__.py:214`), so a bf16 arm against an int8 arm varied three
things at once.
