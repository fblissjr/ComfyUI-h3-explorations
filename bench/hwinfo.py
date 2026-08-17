"""The host this repo's numbers were measured on, printed rather than written down.

NOT a check. It asserts nothing and always exits 0 when it can read the box.
It exists because `docs/hardware.md` describes a machine whose driver version,
power limit and clock state all drift, and CLAUDE.md's rule is that a number
which changes as the project evolves does not belong in prose. So the doc
carries the shape and this carries the values.

    python bench/hwinfo.py

Deliberately **not** `uv run`: this repo has a `pyproject.toml` but is meant to
run inside ComfyUI's venv, so `uv run` without `--active` builds a second one
and writes the `uv.lock` that `docs/comfy_notes.md` says must not exist here.
Nothing below imports anything outside the standard library, so a bare
interpreter is enough and the question does not arise.

Needs no CUDA, no model, no ComfyUI server -- it reads `nvidia-smi`, sysfs and
`/proc`. Safe to run during a render; it takes one sample and does not allocate.

**Sample the PCIe line under load.** The link downtrains when the GPU is idle,
so `current_link_speed` reads gen 1 on a card that negotiates gen 4 the moment
work arrives. Width does not downtrain and is the trustworthy half at idle.

**The line that matters is `power.limit vs default`.** Every timing number in
`docs/bench_plan.md` and `docs/SOLATTN.md` was taken at the stock limit. A
board power limit is invisible in a workflow JSON, absent from the capture
manifest, and survives reboots once someone installs a systemd unit for it --
so a run at a changed limit is not comparable to any of them, and nothing else
in this repo would notice. That is the whole reason this file prints rather
than a human remembering.

Exit codes: 0 read the box, 2 no `nvidia-smi` on PATH (did not run, which is
not the same as passed -- the pattern `check_distill_settings.py` set).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

GPU_FIELDS = [
    "name",
    "driver_version",
    "pstate",
    "power.draw",
    "power.limit",
    "power.default_limit",
    "power.min_limit",
    "power.max_limit",
    "clocks.current.sm",
    "clocks.max.sm",
    "clocks.current.memory",
    "clocks.max.memory",
    "temperature.gpu",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "persistence_mode",
]


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def gpu():
    raw = _run(
        ["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}",
         "--format=csv,noheader"]
    )
    if raw is None:
        return None
    # One GPU per line; this repo is single-card, but do not assume it.
    return [
        dict(zip(GPU_FIELDS, [c.strip() for c in line.split(",")]))
        for line in raw.splitlines()
    ]


def host():
    rows = []
    rows.append(("kernel", _run(["uname", "-r"]) or "?"))

    model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            model = line.split(":", 1)[1].strip()
            break
    rows.append(("cpu", model or "?"))

    mem = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        key, _, val = line.partition(":")
        mem[key] = val.strip()
    for key, label in (("MemTotal", "ram total"),
                       ("Cached", "ram page cache"),
                       ("SwapTotal", "swap total"),
                       ("SwapFree", "swap free")):
        if key in mem:
            kb = int(mem[key].split()[0])
            rows.append((label, f"{kb / 1024 / 1024:.1f} GiB"))
    return rows


def pcie_topology():
    """Every PCI function's negotiated link, for the ones with a link at all.

    Width below capability is the interesting case and is not always
    intentional: seating and firmware both produce it, and neither announces
    itself. `docs/hardware.md` records what this box currently negotiates.
    """
    rows = []
    root = Path("/sys/bus/pci/devices")
    if not root.is_dir():
        return rows
    for dev in sorted(root.iterdir()):
        cur_w = _read(dev / "current_link_width")
        max_w = _read(dev / "max_link_width")
        if not cur_w or cur_w == "0":
            continue
        cur_s = (_read(dev / "current_link_speed") or "?").replace(" PCIe", "")
        max_s = (_read(dev / "max_link_speed") or "?").replace(" PCIe", "")
        cls = _read(dev / "class") or ""
        # 0x030000 display, 0x010802 nvme. Everything else is noise here.
        if not (cls.startswith("0x0300") or cls.startswith("0x0108")):
            continue
        kind = "gpu" if cls.startswith("0x0300") else "nvme"
        flag = "  <-- below capability" if cur_w != max_w else ""
        rows.append(
            (dev.name, f"{kind:4s} x{cur_w} @ {cur_s}"
                       f"  (max x{max_w} @ {max_s}){flag}")
        )
    return rows


def main():
    if shutil.which("nvidia-smi") is None:
        print("SKIP hwinfo: no nvidia-smi on PATH -- did not run, not passed.")
        return 2

    print("host")
    for label, val in host():
        print(f"  {label:16s} {val}")

    cards = gpu()
    if not cards:
        print("\nSKIP hwinfo: nvidia-smi present but returned nothing.")
        return 2

    for i, g in enumerate(cards):
        print(f"\ngpu {i}")
        print(f"  {'name':16s} {g['name']}")
        print(f"  {'driver':16s} {g['driver_version']}")
        print(f"  {'pstate':16s} {g['pstate']}")
        print(f"  {'persistence':16s} {g['persistence_mode']}")
        print(f"  {'memory':16s} {g['memory.used']} / {g['memory.total']}")
        print(f"  {'utilization':16s} sm {g['utilization.gpu']}, "
              f"mem-interface-busy {g['utilization.memory']}")
        print(f"  {'temperature':16s} {g['temperature.gpu']} C")
        print(f"  {'clock sm':16s} {g['clocks.current.sm']} "
              f"(max {g['clocks.max.sm']})")
        print(f"  {'clock memory':16s} {g['clocks.current.memory']} "
              f"(max {g['clocks.max.memory']})")
        print(f"  {'pcie':16s} gen {g['pcie.link.gen.current']} "
              f"x{g['pcie.link.width.current']} "
              f"(max gen {g['pcie.link.gen.max']} "
              f"x{g['pcie.link.width.max']})")

        limit, default = g["power.limit"], g["power.default_limit"]
        print(f"  {'power draw':16s} {g['power.draw']}")
        print(f"  {'power limit':16s} {limit} "
              f"(default {default}, range {g['power.min_limit']} "
              f"to {g['power.max_limit']})")
        if limit != default:
            print("  " + "-" * 60)
            print(f"  POWER LIMIT IS NOT STOCK: {limit} against a "
                  f"default of {default}.")
            print("  Timing taken now is not comparable to any number in")
            print("  docs/bench_plan.md or docs/SOLATTN.md. See docs/hardware.md.")
            print("  " + "-" * 60)

    topo = pcie_topology()
    if topo:
        print("\npcie links")
        for addr, desc in topo:
            print(f"  {addr}  {desc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
