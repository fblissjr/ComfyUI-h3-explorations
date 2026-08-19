#!/usr/bin/env bash
# Build NVLabs' SM89 CuTe Sol-Attn kernel into the ComfyUI venv, and prove that
# it -- not the Triton fallback -- is what runs.
#
# THERE IS NO ARTIFACT. The SM89 backend is CuTe DSL Python, JIT-compiled by
# cute.compile() on the first eligible call, in-process, with nothing persisted
# to disk. So "build" is three steps -- install the DSL runtime, install the
# sol-attn package, force that first compile on this card -- and only the third
# is a compile. Re-running this script does not make a later ComfyUI process
# skip its own first-call compile; nothing can.
#
# Why this is not vendor/rebuild_kernel.sh: that one builds comfy-kitchen's
# Sol-Attn, a different implementation of the same paper. This is NVLabs' own.
# docs/roadmap.md wants the two compared on this card, and that needs both
# importable in one process -- which is why this installs into the ComfyUI venv
# rather than an isolated one.
#
# The checkout at coderef/Sana is SOMEONE ELSE'S REPO and this script leaves it
# as it found it, for the reason vendor/rebuild_kernel.sh spells out at length.
# Two consequences worth stating:
#
#   - The install is a WHEEL, not `pip install -e`. update-coderef.sh at the
#     repo root pulls every coderef checkout; an editable install would let a
#     routine pull silently swap the kernel under a measurement.
#   - Build leftovers (dist/, build/, *.egg-info) are cleaned on every exit
#     path, including a failed build. Sana's .gitignore covers none of them.
#
# Usage:  vendor/build_sana_sol_sm89.sh
#
#   DSL_VERSION=4.7.0     CuTe DSL pin. Upstream asks for >= 4.5; 4.7.0 is what
#                         their own LTX-2.5 refiner runtime records, and a
#                         version mismatch fails at compile time rather than
#                         falling back -- deliberately, so a dense run is never
#                         reported as a sparse one.
#   TVM_FFI_VERSION=...   apache-tvm-ffi pin. Not named in any Sol-Attn
#                         requirements list, and not a dependency of the DSL
#                         wheel either, but the SM89 path is unrunnable without
#                         it: common/runtime.py passes enable_tvm_ffi=True on
#                         every tensor and interface.py compiles with
#                         --enable-tvm-ffi, so the first call dies on
#                         ModuleNotFoundError. Observed 2026-08-19, DSL 4.7.0.
#   PY=/path/to/python    interpreter to install into.
#   SKIP_INSTALL=1        re-run the verify against whatever is installed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/coderef/Sana"
PKG="$SRC/techniques/sparse_backends"
DSL_VERSION="${DSL_VERSION:-4.7.0}"
TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.13.post3}"
# Two levels up from a repo that sits at <comfy>/custom_nodes/<pack>. Derived,
# not typed. Override with PY= for any other layout.
PY="${PY:-$REPO/../../.venv/bin/python}"

[ -x "$PY" ] || { echo "no interpreter at $PY -- set PY=/path/to/python"; exit 1; }
[ -d "$PKG" ] || { echo "no sol-attn source at $PKG -- is coderef/Sana checked out?"; exit 1; }

# This kernel is architecture-specific by construction: interface.py maps
# (8, 9) to cute_sm89 and every other capability elsewhere. Refuse early rather
# than install a package that would dispatch somewhere we did not build for.
"$PY" - <<'PYARCH'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("no CUDA device visible")
cap = torch.cuda.get_device_capability()
if cap != (8, 9):
    sys.exit(f"this script builds the SM89 backend; this device is SM{cap[0]}{cap[1]}")
PYARCH

echo "== source: Sana $(git -C "$SRC" rev-parse --abbrev-ref HEAD) @ $(git -C "$SRC" rev-parse --short HEAD)"
echo "== package: $(grep -m1 '^version' "$PKG/pyproject.toml")"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | sed 's/^/== gpu: /'
"$PY" -c "import torch; print(f'== torch: {torch.__version__} (cuda {torch.version.cuda})')"

# Sana's tree has no ignore rule for either of these, so a build that left them
# behind would show up as untracked files in someone else's repo.
cleanup() { rm -rf "$PKG/dist" "$PKG/build" "$PKG"/*.egg-info; }
trap cleanup EXIT

if [ -z "${SKIP_INSTALL:-}" ]; then
    cleanup
    echo "== installing CuTe DSL $DSL_VERSION"
    uv pip install --python "$PY" \
        "nvidia-cutlass-dsl==$DSL_VERSION" \
        "apache-tvm-ffi==$TVM_FFI_VERSION" \
        cuda-python

    echo "== building and installing the sol-attn wheel"
    uv build --wheel --out-dir "$PKG/dist" "$PKG"
    uv pip install --python "$PY" --force-reinstall --no-deps "$PKG"/dist/sol_attn-*.whl
    cleanup
fi

echo "== verifying"
"$PY" - <<'PYVERIFY'
import sys

import torch
import torch.nn.functional as F

import cutlass
import tvm_ffi
from sol_attn import get_sol_attn_backend, sol_attn
from sol_attn.triton_ref import sol_attn as triton_sol_attn

print(f"== cutlass dsl: {getattr(cutlass, '__version__', 'unknown')}")
print(f"== tvm ffi: {getattr(tvm_ffi, '__version__', 'unknown')}")

# The load-bearing assertion. _backend_for_arch() returns "triton" whenever
# cutlass.cute fails to import, silently and by design -- so a run that
# produced sane numbers is not by itself evidence the CuTe kernel ran.
backend = get_sol_attn_backend()
print(f"== backend: {backend}")
if backend != "cute_sm89":
    sys.exit(f"dispatch chose {backend!r}, so the SM89 CuTe kernel is not what runs")

# Small enough to compile and run beside a resident ComfyUI. This is a build
# verification, not a benchmark; the shape is chosen for headroom, and the
# script prints no timings on purpose -- see docs/hardware.md on why a figure
# taken at an unrecorded power state is not comparable to anything.
torch.manual_seed(0)
shape = (1, 2048, 8, 128)
q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))

out = sol_attn(q, k, v, tau=1.0, thresh_type="exact")
torch.cuda.synchronize()

if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
    sys.exit(f"contract broken: got {tuple(out.shape)} {out.dtype} on {out.device}")
if not torch.isfinite(out).all():
    sys.exit("output contains non-finite values")

# Two controls, because either alone passes for the wrong reason.
#
# triton_ref is upstream's own implementation of the same routing, reached
# through an entry point that bypasses backend selection: an independent
# implementation rather than a number this script computed itself.
#
# Dense SDPA is the second control, and the one that makes the first mean
# something. On random gaussian Q/K the softmax is near-uniform, so sparse
# attention approximates dense badly on purpose -- which is what gives the
# comparison a scale. A kernel that quietly ran dense attention would agree
# with the Triton sparse reference no better than dense does.
ref = triton_sol_attn(q, k, v, tau=1.0, thresh_type="exact")
dense = F.scaled_dot_product_attention(
    q.transpose(1, 2).float(),
    k.transpose(1, 2).float(),
    v.transpose(1, 2).float(),
).transpose(1, 2)
torch.cuda.synchronize()


def mean_abs(a, b):
    return (a.float() - b.float()).abs().mean().item()


norm_ratio = (out.float().norm() / ref.float().norm()).item()
vs_triton = mean_abs(out, ref)
vs_dense = mean_abs(out, dense)
triton_vs_dense = mean_abs(ref, dense)
print(f"== mean abs deviation, cute vs triton: {vs_triton:.6f}")
print(f"== mean abs deviation, cute vs dense:  {vs_dense:.6f}")
print(f"== mean abs deviation, triton vs dense: {triton_vs_dense:.6f}")
print(f"== output norm, cute / triton: {norm_ratio:.6f}")

# Both bounds were placed from an observation on this box, 2026-08-19, DSL
# 4.7.0, seeds 0-2: the first ratio came out at 0.02 against a 0.1 bound, and
# the second within a percent of 1.0 against a bound of 2x. They are limits,
# not measurements -- what was measured is printed on the lines above.
if not vs_triton < 0.1 * triton_vs_dense:
    sys.exit(
        "CuTe SM89 does not track the Triton reference: it is no closer to it "
        "than dense attention is"
    )
if not 0.5 < vs_dense / triton_vs_dense < 2.0:
    sys.exit(
        "CuTe SM89 departs from dense by an amount the Triton reference does "
        "not -- it is not running the same sparse routing"
    )
# The two above are ratio tests and a uniformly mis-scaled output slips through
# both: a 5% gain error was measured passing them on 2026-08-19, which is why
# this third one exists. The norm ratio is what separates the arms sharply --
# it sat inside a thousandth of 1.0 across seeds, against 1.05 for the mutant.
if not abs(norm_ratio - 1.0) < 0.01:
    sys.exit("CuTe SM89 output is mis-scaled relative to the Triton reference")

# Not an accuracy claim. This says the compiled kernel implements the same
# algorithm as the reference on synthetic input; it says nothing about either
# one on real activations, which is what bench/grade_sage_on_capture.py is for.
print("== ok: cute_sm89 compiled, ran, and tracks the Triton reference")
PYVERIFY

cleanup; trap - EXIT
echo "== Sana checkout restored: $(git -C "$SRC" status --porcelain | wc -l) modified files (want 0)"

echo
echo "Installed into the venv, not into a running process. ComfyUI must be"
echo "restarted before it can import this, and its first Sol call then pays"
echo "its own cute.compile()."
