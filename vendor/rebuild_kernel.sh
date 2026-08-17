#!/usr/bin/env bash
# Rebuild and install comfy_kitchen's Sol-Attn kernel from kijai's branch.
#
# The checkout at coderef/comfy-kitchen-sol is SOMEONE ELSE'S REPO and this
# script leaves it exactly as it found it. That is the whole design:
#
#   2026-08-14: the version-tag edit was left as a working-tree modification
#   on the theory that a future `git pull` would then conflict loudly rather
#   than silently reverting it. It does not conflict -- it BLOCKS:
#
#       error: cannot pull with rebase: You have unstaged changes.
#
#   A guard that stops the owner updating a dependency is worse than the drift
#   it was guarding against. The change lives in vendor/patches/ instead, and
#   is applied only for the duration of a build.
#
# Why the patch exists at all: the branch declares version "0.2.31", identical
# to the PyPI wheel ComfyUI pins, so a fork build and the stock wheel are
# indistinguishable to `pip list` -- and a stock wheel silently has no
# sol_attn, which makes every Sol call fall back to dense with no error.
#
# Usage:  vendor/rebuild_kernel.sh [CUDA_ARCH]     (default 89, this box's 4090)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/coderef/comfy-kitchen-sol"
PATCH="$REPO/vendor/patches/001-local-version-tag.patch"
ARCH="${1:-89}"
# Derived from this checkout, not typed: the repo sits at
# <comfy>/custom_nodes/<pack>, so the venv is two levels up. Override with
# PY=... for any other layout.
PY="${PY:-$REPO/../../.venv/bin/python}"
[ -x "$PY" ] || { echo "no interpreter at $PY -- set PY=/path/to/python"; exit 1; }

[ -d "$SRC" ] || { echo "no checkout at $SRC"; exit 1; }
cd "$SRC"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: $SRC has local changes. This script needs a clean tree so it"
    echo "can guarantee it leaves one. Resolve them first:"
    git status --short
    exit 1
fi

echo "== source: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

# Revert the patch no matter how we leave, including on a failed build. Without
# this a compile error strands the tree dirty and blocks the next pull, which
# is the exact failure this script exists to prevent.
cleanup() { git -C "$SRC" checkout -- pyproject.toml 2>/dev/null || true; }
trap cleanup EXIT

git apply "$PATCH"
echo "== version: $(grep -m1 '^version' pyproject.toml)"

COMFY_CUDA_ARCHS="$ARCH" uv build --wheel --no-build-isolation .
uv pip install --force-reinstall --no-deps dist/comfy_kitchen-*.whl

cleanup; trap - EXIT
echo "== checkout restored: $(git status --porcelain | wc -l) modified files (want 0)"

echo "== verifying the kernel is actually present"
"$PY" "$REPO/bench/check_sol_kernel.py" --require || {
    echo "check_sol_kernel FAILED -- the build installed but is not usable"; exit 1; }

echo
echo "Restart ComfyUI (this is node code), then confirm the reload by reading a"
echo "changed default back out of /object_info before trusting any measurement."
