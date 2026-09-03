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
#   it was guarding against. The edit is made by this script for the duration
#   of a build and reverted on every exit path.
#
# Why the edit exists at all: the source declares a plain version ("0.2.31",
# "0.2.32", ...), identical to the PyPI wheel ComfyUI pins, so a fork build
# and the stock wheel are indistinguishable to `pip list` -- and a stock wheel
# silently has no sol_attn, which makes every Sol call fall back to dense with
# no error. PEP 440 still matches `X.Y.Z+sol.<sha>` against `==X.Y.Z`, so a
# plain requirements install stays satisfied and will not clobber it.
#
#   2026-09-03: this used to be `git apply vendor/patches/001-local-version-tag.patch`,
#   a diff hardcoded against `version = "0.2.31"`. Upstream released 0.2.32 on
#   2026-09-02 and the patch stopped applying on any checkout based on it; the
#   first build of our blk_cnt branch rebased onto v0.2.32 found that. The
#   version line is now rewritten by sed, whatever it says, so the script no
#   longer carries a copy of upstream's version number that has to be kept
#   in step with upstream.
#
# Usage:  vendor/rebuild_kernel.sh [CUDA_ARCH]     (default 89, this box's 4090)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Default is kijai's branch checkout, which is where Sol-Attn lived while it
# was unmerged. **Overridable since 2026-08-29**, when PR #117 landed the
# kernel in comfy-kitchen main (dae00a1) and the branch stopped being the only
# place to get it. Point SRC at any clean checkout or worktree:
#
#   SRC=/path/to/worktree vendor/rebuild_kernel.sh 89
#
# A worktree is the polite way to build a specific upstream commit: both
# existing checkouts under coderef/ belong to somebody else and this script's
# whole design is to leave them as it found them.
SRC="${SRC:-$REPO/coderef/comfy-kitchen-sol}"
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

# Revert the edit no matter how we leave, including on a failed build. Without
# this a compile error strands the tree dirty and blocks the next pull, which
# is the exact failure this script exists to prevent.
cleanup() { git -C "$SRC" checkout -- pyproject.toml 2>/dev/null || true; }
trap cleanup EXIT

# Append the local segment to whatever version the checkout declares; the
# tag is the built commit's short sha, derived rather than typed, so it
# cannot go stale on an update.
SHA="$(git rev-parse --short=7 HEAD)"
sed -i "s/^version = \"\([0-9][^\"+]*\)\"/version = \"\1+sol.$SHA\"/" pyproject.toml
if ! grep -q "^version = \".*+sol.$SHA\"" pyproject.toml; then
    echo "ERROR: could not tag the version line in pyproject.toml:"
    grep -n '^version' pyproject.toml; exit 1
fi
VER="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
echo "== version: $VER"

# `--no-build-isolation` means the build uses an EXISTING environment rather
# than a fresh one, and uv picks that up from VIRTUAL_ENV. Neither checkout has
# a .venv of its own, so without this uv finds nothing, and the failure is
# reported as a MISSING BUILD DEPENDENCY ("No module named 'setuptools'")
# rather than as a missing environment -- which sends you off installing
# setuptools somewhere it already is. Derived from $PY so it follows the
# interpreter override rather than being a second place to configure the venv.
export VIRTUAL_ENV="$(cd "$(dirname "$PY")/.." && pwd)"
echo "== building against $VIRTUAL_ENV"
COMFY_CUDA_ARCHS="$ARCH" uv build --wheel --no-build-isolation .
# By exact version, not a glob: dist/ keeps every wheel ever built here, so
# `comfy_kitchen-*.whl` grew to match more than one the first time this script
# ran twice, and uv would have been handed both.
WHL=(dist/comfy_kitchen-"$VER"-*.whl)
[ -f "${WHL[0]}" ] || { echo "ERROR: no wheel built for $VER"; exit 1; }
uv pip install --force-reinstall --no-deps "${WHL[0]}"

# The build record: ONE file beside the venv saying which build is installed
# and where its source is, written at the only moment both are known. start.sh
# prints it on every launch and cross-checks it against the installed wheel,
# so "which comfy-kitchen is running" has one answer and one path to it
# (owner's ask, 2026-09-03). A stock reinstall replaces the wheel but not this
# file, which is exactly the mismatch start.sh is there to shout about.
RECORD="$VIRTUAL_ENV/comfy_kitchen_build.json"
"$PY" - "$RECORD" "$VER" "$SRC" "${WHL[0]}" "$ARCH" <<'PYEOF'
import json, subprocess, sys, time
record, ver, src, whl, arch = sys.argv[1:6]
def git(*a):
    return subprocess.run(["git", "-C", src, *a], capture_output=True, text=True).stdout.strip()
json.dump({
    "version": ver,
    "sha": git("rev-parse", "HEAD"),
    "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "source": src,
    "origin": git("remote", "get-url", "origin"),
    "wheel": whl,
    "cuda_arch": arch,
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "written_by": "vendor/rebuild_kernel.sh",
}, open(record, "w"), indent=2)
print(f"== build record: {record}")
PYEOF

cleanup; trap - EXIT
echo "== checkout restored: $(git status --porcelain | wc -l) modified files (want 0)"

echo "== verifying the kernel is actually present"
"$PY" "$REPO/bench/check_sol_kernel.py" --require || {
    echo "check_sol_kernel FAILED -- the build installed but is not usable"; exit 1; }

echo
echo "Restart ComfyUI (this is node code), then confirm the reload by reading a"
echo "changed default back out of /object_info before trusting any measurement."
