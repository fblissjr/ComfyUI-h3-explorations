"""What a ComfyUI prompt loaded, moved and spent, end to end, as one JSONL record.

Inert unless the server process starts with `H3_TELEMETRY` set; then every
prompt the server runs writes `<dir>/<date>_<time>_<prompt>.jsonl`. The
schema, the reading tool (`bench/telemetry_report.py`) and what each field
can and cannot say are in `docs/pipeline_telemetry.md`. Arming key:

    H3_TELEMETRY=dir=<path>[,interval_ms=250][,blocks=1]

Three sources, each read where it is safe to read:

- **Node boundaries** through core's cache-provider API
  (`comfy_execution/cache_provider.py`), the sanctioned hook: nothing here
  patches core. Core calls `should_cache(context)` synchronously on a local
  cache miss, just before the node executes, and `should_cache(context, value)`
  synchronously when the node's output is stored. We record both and return
  False, so no lookup or store is ever scheduled and caching is unchanged. At
  each boundary, on the executor thread, a snapshot of every loaded model: its
  size, what dynamic VRAM holds on the card, what is pinned in host RAM, and,
  with `blocks=1`, the resident fraction of each DiT block.
- **A sampler thread** every `interval_ms`: the card from NVML (memory, PCIe
  bytes each way, utilisation, clocks, power, temperature), dynamic VRAM's own
  total, the process from /proc (anonymous vs file-backed RSS, disk bytes,
  major faults, CPU time) and the host's page cache and free memory. Only
  thread-safe reads: no VBAR residency here, that is the boundaries' job.
- **Core's own log lines** about loading, unloading, staging and eviction,
  parsed where their format allows, each stamped with the node executing then.

NVML is bound through ctypes to the driver's library, so the pack takes no
dependency; the device is matched by UUID so `CUDA_VISIBLE_DEVICES` cannot
point it at the wrong card.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ENV = "H3_TELEMETRY"
SCHEMA = 1
_LOG = "[h3 telemetry]"
#: dynamic VRAM's page (`comfy_aimdo/model_vbar.py`, VBAR_PAGE_SIZE). Inherited.
VBAR_PAGE = 32 << 20
#: logging's DETAIL level, where core logs "Model loaded", pin eviction and
#: AIMDO frees (`comfy/model_management.py`). Inherited.
DETAIL = 15

_state: dict = {"armed": None, "spec": {}, "installed": False}


# ---------------------------------------------------------------------------
# Arming
# ---------------------------------------------------------------------------

def parse_spec(raw: str) -> dict:
    out = {"dir": None, "interval_ms": 250, "blocks": True}
    for part in (raw or "").split(","):
        key, _, val = part.partition("=")
        key, val = key.strip(), val.strip()
        if key == "dir" and val:
            out["dir"] = os.path.expanduser(val)
        elif key == "interval_ms" and val:
            out["interval_ms"] = max(20, int(val))
        elif key == "blocks" and val:
            out["blocks"] = val.lower() not in ("0", "false", "no", "off")
    return out


def arm(raw: str | None) -> bool:
    spec = parse_spec(raw or "")
    _state["spec"] = spec
    _state["armed"] = bool(raw) and bool(spec["dir"])
    if raw and not spec["dir"]:
        print(f"{_LOG} {ENV} set but no dir=; telemetry disabled")
    return _state["armed"]


def install_from_env() -> bool:
    """Called once at pack import. Registers nothing unless armed."""
    if _state["installed"]:
        return True
    if not arm(os.environ.get(ENV, "")):
        return False
    try:
        from comfy_execution.cache_provider import register_cache_provider
    except Exception as exc:  # an older core without the provider API
        print(f"{_LOG} {ENV} set but core has no cache-provider API ({exc}); disabled")
        _state["armed"] = False
        return False
    Path(_state["spec"]["dir"]).mkdir(parents=True, exist_ok=True)
    rec = _Recorder(_state["spec"])
    register_cache_provider(rec.provider)
    root = logging.getLogger()
    root.addHandler(rec.log_handler)
    # Lowering the root level lets DETAIL records reach our handler; the
    # console handler keeps its own level, so the console does not change.
    if root.level > DETAIL:
        root.setLevel(DETAIL)
    _state["installed"] = True
    _state["recorder"] = rec
    print(f"{_LOG} ARMED: dir={_state['spec']['dir']} "
          f"interval_ms={_state['spec']['interval_ms']} blocks={_state['spec']['blocks']}")
    return True


# ---------------------------------------------------------------------------
# NVML through ctypes
# ---------------------------------------------------------------------------

class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                ("used", ctypes.c_ulonglong)]


class _NvmlUtil(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class Nvml:
    """The dozen NVML calls this needs. Every call returns None on failure."""

    PCIE_TX, PCIE_RX = 0, 1          # nvmlPcieUtilCounter_t: TX = GPU to host
    CLOCK_SM, CLOCK_MEM = 1, 2       # nvmlClockType_t
    TEMP_GPU = 0

    def __init__(self, uuid: str | None):
        self.lib = None
        self.handle = None
        try:
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return
            handle = ctypes.c_void_p()
            ok = False
            if uuid:
                ok = lib.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(handle)) == 0
            if not ok:
                ok = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0
            if ok:
                self.lib, self.handle = lib, handle
        except OSError:
            pass

    def _uint(self, fn, *args):
        out = ctypes.c_uint()
        if getattr(self.lib, fn)(self.handle, *args, ctypes.byref(out)) != 0:
            return None
        return out.value

    def static(self) -> dict:
        if self.lib is None:
            return {}
        name = ctypes.create_string_buffer(96)
        self.lib.nvmlDeviceGetName(self.handle, name, 96)
        drv = ctypes.create_string_buffer(80)
        self.lib.nvmlSystemGetDriverVersion(drv, 80)
        return {"name": name.value.decode(), "driver": drv.value.decode(),
                "pcie_gen_max": self._uint("nvmlDeviceGetMaxPcieLinkGeneration"),
                "pcie_width_max": self._uint("nvmlDeviceGetMaxPcieLinkWidth"),
                "power_limit_mw": self._uint("nvmlDeviceGetEnforcedPowerLimit")}

    def sample(self) -> dict:
        if self.lib is None:
            return {}
        mem, util = _NvmlMemory(), _NvmlUtil()
        out = {}
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem)) == 0:
            out["used"], out["total"] = mem.used, mem.total
        if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(util)) == 0:
            out["util_gpu"], out["util_mem"] = util.gpu, util.memory
        # KB/s over NVML's own short window, not an integral over our interval
        out["pcie_tx_kbs"] = self._uint("nvmlDeviceGetPcieThroughput", self.PCIE_TX)
        out["pcie_rx_kbs"] = self._uint("nvmlDeviceGetPcieThroughput", self.PCIE_RX)
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        out["power_mw"] = self._uint("nvmlDeviceGetPowerUsage")
        out["sm_mhz"] = self._uint("nvmlDeviceGetClockInfo", self.CLOCK_SM)
        out["mem_mhz"] = self._uint("nvmlDeviceGetClockInfo", self.CLOCK_MEM)
        out["temp_c"] = self._uint("nvmlDeviceGetTemperature", self.TEMP_GPU)
        return out


# ---------------------------------------------------------------------------
# /proc readers
# ---------------------------------------------------------------------------

def _kv_file(path: str, keys: tuple, scale: int = 1) -> dict:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in keys:
                    out[k] = int(v.split()[0]) * scale
    except OSError:
        pass
    return out


_STATUS_KEYS = ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSwap", "Threads")
_MEMINFO_KEYS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Dirty",
                 "Writeback", "SwapTotal", "SwapFree", "Mlocked", "Unevictable")


def host_sample() -> dict:
    status = _kv_file("/proc/self/status", _STATUS_KEYS)
    out = {f"proc_{k.lower()}": v * 1024 for k, v in status.items() if k != "Threads"}
    out["proc_threads"] = status.get("Threads")
    out.update({f"host_{k.lower()}": v * 1024 for k, v in
                _kv_file("/proc/meminfo", _MEMINFO_KEYS).items()})
    io = _kv_file("/proc/self/io", ("read_bytes", "write_bytes", "rchar"))
    out.update({f"proc_io_{k}": v for k, v in io.items()})
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                if line.startswith("pgmajfault "):
                    out["host_pgmajfault"] = int(line.split()[1])
                    break
    except OSError:
        pass
    t = os.times()
    out["proc_cpu_user_s"], out["proc_cpu_sys_s"] = t.user, t.system
    return out


# ---------------------------------------------------------------------------
# Model snapshots (executor thread only)
# ---------------------------------------------------------------------------

def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _block_ranges(dm, vbar) -> list:
    """(block index, [(first page, last page), ...]) for each `blocks.N` of a
    diffusion model, from the VBAR slice each castable module was given."""
    per_block: dict = {}
    for name, m in dm.named_modules():
        v = getattr(m, "_v", None)
        if not v or v[0] is not vbar:
            continue
        mt = re.match(r"blocks\.(\d+)\.", name + ".")
        if not mt:
            continue
        lo = (v[1] - vbar.base_addr) // VBAR_PAGE
        hi = (v[1] + v[2] - 1 - vbar.base_addr) // VBAR_PAGE
        per_block.setdefault(int(mt.group(1)), []).append((lo, hi))
    return sorted(per_block.items())


class _Snapshotter:
    def __init__(self, blocks: bool):
        self.blocks = blocks
        self._ranges: dict = {}      # (id(model), vbar ptr) -> block ranges

    def models(self) -> list:
        import comfy.model_management as mm
        out = []
        for lm in list(mm.current_loaded_models):
            patcher = _safe(lambda: lm.model)
            if patcher is None:
                continue
            model = patcher.model
            row = {
                "id": f"{id(model):x}",
                "class": type(model).__name__,
                "patcher": type(patcher).__name__,
                "dynamic": bool(_safe(patcher.is_dynamic, False)),
                "device": str(getattr(lm, "device", "")),
                "currently_used": bool(getattr(lm, "currently_used", False)),
                "size": _safe(patcher.model_size),
                "loaded": _safe(patcher.loaded_size),
                "ram_loaded": _safe(patcher.loaded_ram_size),
                "pinned": _safe(patcher.pinned_memory_size),
                "force_loaded": getattr(model, "model_loaded_weight_memory", None),
            }
            vbars = getattr(model, "dynamic_vbars", None) or {}
            for dev, vbar in vbars.items():
                res = _safe(vbar.get_residency, [])
                row["vbar"] = {
                    "device": str(dev), "pages": len(res),
                    "resident_pages": sum(1 for r in res if r & 1),
                    "pinned_pages": sum(1 for r in res if r & 2),
                    "watermark": _safe(vbar.get_watermark),
                    "loaded": _safe(vbar.loaded_size),
                }
                dm = getattr(model, "diffusion_model", None)
                if self.blocks and dm is not None and res:
                    key = (id(model), vbar._ptr)
                    if key not in self._ranges:
                        self._ranges[key] = _safe(lambda: _block_ranges(dm, vbar), [])
                    blk = []
                    for _idx, spans in self._ranges[key]:
                        pages = {p for lo, hi in spans for p in range(lo, hi + 1)}
                        n = sum(1 for p in pages if p < len(res) and res[p] & 1)
                        blk.append(round(n / len(pages), 3) if pages else None)
                    row["vbar"]["block_resident"] = blk
                break
            out.append(row)
        return out

    def device(self) -> dict:
        import torch
        import comfy.model_management as mm
        out = {"pinned_total": getattr(mm, "TOTAL_PINNED_MEMORY", None)}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            out.update(torch_free=free, torch_total=total,
                       torch_allocated=torch.cuda.memory_allocated(),
                       torch_reserved=torch.cuda.memory_reserved(),
                       torch_peak_allocated=torch.cuda.max_memory_allocated())
        out["aimdo_vram"] = _aimdo_total()
        return out


def _aimdo_total():
    try:
        import comfy_aimdo.control as c
        return c.get_total_vram_usage()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

_LOAD_PATTERNS = [
    ("requested_load", re.compile(r"Requested to load (\S+)")),
    ("dynamic_prepared", re.compile(
        r"Model (\S+) prepared for dynamic VRAM loading\. (\d+)MB Staged\. (\d+) patches attached\."
        r"(?: Force pre-loaded (\d+) weights: (\d+) KB\.)?")),
    ("loaded_completely", re.compile(r"loaded completely;? (.*)")),
    ("loaded_partially", re.compile(r"loaded partially;? (.*)")),
    ("models_unloaded", re.compile(r"(\d+) models unloaded")),
    ("unloaded_partially", re.compile(r"[Uu]nloaded partially(.*)")),
    ("model_loaded", re.compile(r"^Model loaded (.*)")),
    ("pin_eviction", re.compile(r"Pin eviction(.*)")),
    ("aimdo_free", re.compile(r"AIMDO free(.*)")),
    ("vae_device", re.compile(r"VAE load device: (\S+), offload device: (\S+), dtype: (\S+)")),
    ("oom_fallback", re.compile(r"(out of memory.*|retrying with tiled.*)", re.I)),
]


class _Recorder:
    def __init__(self, spec: dict):
        self.spec = spec
        self.lock = threading.Lock()
        self.fh = None
        self.prompt_id = None
        self.seq = 0
        self.node = None
        self.t0: float | None = None
        self.stop = threading.Event()
        self.thread = None
        self.snap = _Snapshotter(spec["blocks"])
        self.nvml = None
        self.provider = _make_provider(self)
        self.log_handler = _LogHandler(self)

    # -- writing ------------------------------------------------------------
    def emit(self, event_type: str, **fields):
        with self.lock:
            if self.fh is None or self.t0 is None:
                return
            self.seq += 1
            row = {"t": round(time.monotonic() - self.t0, 4), "seq": self.seq,
                   "type": event_type, **fields}
            self.fh.write(json.dumps(row, default=str) + "\n")

    # -- prompt lifecycle ----------------------------------------------------
    def start(self, prompt_id: str):
        self.close()
        graph = _running_graph(prompt_id)
        stamp = time.strftime("%Y-%m-%d_%H%M%S")
        path = Path(self.spec["dir"]) / f"{stamp}_{prompt_id[:8]}.jsonl"
        with self.lock:
            self.fh = open(path, "w", buffering=1)
            self.prompt_id, self.seq, self.node = prompt_id, 0, None
            self.t0 = time.monotonic()
        if self.nvml is None:
            self.nvml = Nvml(_cuda_uuid())
        self.emit("header", schema=SCHEMA, prompt_id=prompt_id,
                  wall_start=time.time(), pid=os.getpid(), spec=self.spec,
                  substrate=_substrate(self.nvml), graph=graph)
        self.boundary("prompt_start", None, None)
        self.stop.clear()
        self.thread = threading.Thread(target=self._sample_loop, name="h3-telemetry",
                                       daemon=True)
        self.thread.start()

    def end(self, prompt_id: str):
        if prompt_id != self.prompt_id:
            return
        self.boundary("prompt_end", None, None)
        self.close()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
        with self.lock:
            if self.fh is not None:
                self.fh.close()
                self.fh = None

    # -- boundaries (executor thread) ----------------------------------------
    def boundary(self, kind: str, node_id, class_type):
        if kind == "node_start":
            self.node = node_id
        models = _safe(self.snap.models, [])
        device = _safe(self.snap.device, {})
        self.emit(kind, node=node_id, class_type=class_type, models=models, device=device)
        if kind == "node_end":
            self.node = None

    # -- sampler thread --------------------------------------------------------
    def _sample_loop(self):
        period = self.spec["interval_ms"] / 1000.0
        while not self.stop.is_set():
            gpu = self.nvml.sample() if self.nvml else {}
            self.emit("sample", node=self.node, gpu=gpu, host=host_sample(),
                      aimdo_vram=_aimdo_total())
            self.stop.wait(period)


def _make_provider(rec: _Recorder):
    from comfy_execution.cache_provider import CacheProvider

    class TelemetryProvider(CacheProvider):
        """Observes; never caches. `should_cache` returns False every time."""

        async def on_lookup(self, context):
            return None

        async def on_store(self, context, value):
            return None

        def should_cache(self, context, value=None):
            try:
                kind = "node_start" if value is None else "node_end"
                rec.boundary(kind, context.node_id, context.class_type)
            except Exception as exc:
                logging.getLogger(__name__).debug(f"{_LOG} boundary failed: {exc}")
            return False

        def on_prompt_start(self, prompt_id):
            try:
                rec.start(prompt_id)
            except Exception as exc:
                print(f"{_LOG} could not start a record: {exc}")

        def on_prompt_end(self, prompt_id):
            try:
                rec.end(prompt_id)
            except Exception as exc:
                print(f"{_LOG} could not close the record: {exc}")

    return TelemetryProvider()


class _LogHandler(logging.Handler):
    def __init__(self, rec: _Recorder):
        super().__init__(level=DETAIL)
        self.rec = rec

    def emit(self, record):
        if self.rec.fh is None:
            return
        try:
            msg = record.getMessage()
        except Exception:
            return
        for kind, pat in _LOAD_PATTERNS:
            m = pat.search(msg)
            if m:
                self.rec.emit("log", kind=kind, node=self.rec.node, logger=record.name,
                              level=record.levelname, groups=list(m.groups()), msg=msg[:400])
                return


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _cuda_uuid():
    try:
        import torch
        return "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    except Exception:
        return None


def _git(path: Path):
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _substrate(nvml: Nvml) -> dict:
    import torch
    here = Path(__file__).resolve().parent
    out = {"gpu": nvml.static() if nvml else {}, "torch": torch.__version__,
           "cuda": torch.version.cuda, "argv": sys.argv[1:],
           "comfy_commit": _git(here.parents[1]), "pack_commit": _git(here)}
    for mod in ("comfy_aimdo", "comfy_kitchen"):
        m = sys.modules.get(mod)
        out[mod] = getattr(m, "__version__", None) if m else None
    try:
        import comfy.memory_management as cmm
        out["dynamic_vram"] = bool(getattr(cmm, "aimdo_enabled", False))
    except Exception:
        out["dynamic_vram"] = None
    out["host"] = {k: v * 1024 for k, v in
                   _kv_file("/proc/meminfo", ("MemTotal", "SwapTotal")).items()}
    return out


def _running_graph(prompt_id: str) -> dict:
    """The prompt's node classes and a hash of its canonical JSON, read from the
    queue's running item: (number, prompt_id, prompt, extra, outputs, ...)."""
    try:
        from server import PromptServer
        for item in PromptServer.instance.prompt_queue.currently_running.values():
            if item[1] == prompt_id:
                prompt = item[2]
                canon = json.dumps(prompt, sort_keys=True, separators=(",", ":"))
                return {"sha256": hashlib.sha256(canon.encode()).hexdigest(),
                        "nodes": {k: v.get("class_type") for k, v in prompt.items()}}
    except Exception:
        pass
    return {}
