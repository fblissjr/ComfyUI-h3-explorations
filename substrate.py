"""What produced a number: the substrate block, and nothing else.

DRAFT, 2026-08-17. Written independently of the schema being drafted in
parallel, deliberately -- two implementations meeting is the control this repo
prefers over one side asserting against values it computed itself. Where this
and the schema disagree, the disagreement is the finding.

## What this is

One function, `substrate()`, returning the answer to "what produced this
number" as a plain dict. A capture manifest, a render stamp and a bench record
all embed it verbatim.

**It returns the block and nothing else.** No `kind`, no payload, no filename,
no writing to disk. The moment this module knows that captures have tensors and
bench runs have timings, it is a second place the schema is encoded, and the
two drift. Callers add their own `kind` and payload around what they get here.

## Why it lives at the repo root and imports nothing from ComfyUI

`provenance.py` already implements three of these readers -- `_git_head`,
`_comfy_kitchen_version`, `_sage_version` -- and it CANNOT be imported outside a
running ComfyUI: it needs `folder_paths`. Most of `bench/` deliberately needs
neither CUDA nor ComfyUI, so a bench script cannot reach those readers.

That fixes the direction of the extraction. This module depends on nothing but
the standard library, and `provenance.py` imports IT. Writing fresh readers in
`bench/` instead would create the exact duplication this repo keeps paying for.

## Half merged with `provenance.py`, deliberately

**Step 1 done 2026-08-17.** `provenance.py` imports `git_head` from here and no
longer defines its own. It wraps it in an adapter rather than calling it
directly, because the two differed in their failure sentinel -- this returns
`None`, the stamp records `"not detected"` -- and that string is part of the
stamp's output. Translating at the boundary keeps one implementation of the git
reading while leaving every stamp byte-for-byte what it was, so
`STAMP_SCHEMA_VERSION` did not move for a refactor.

Verified against BOTH consumers, because they load this differently and only one
of them is obvious:

  package-relative   ComfyUI loads `provenance.py` as a package member. Checked
                     by starting the server and confirming the node registers.
  by path, no parent `bench/check_provenance_stamp.py` loads it under a bare
                     module name via `spec_from_file_location`. A plain relative
                     import raises there, and that check catches its own import
                     failure and returns 2 -- so it would have gone from green
                     to silently skipped. Hence the try/except ImportError pair
                     in `provenance.py`; neither spelling is redundant.

**Step 2 is still open, and is blocked on the schema, not on effort.**
`provenance._comfy_kitchen_version` and `_sage_version` perform the same reads
as `_package()` here, in a different shape -- duplicated logic rather than a
duplicated name, so no tool flags it. Their output format is part of the stamp's
schema: the comfy_kitchen string carries a ` (NO sol_attn)` suffix and sage
carries `version@githead`. Reshaping them IS a schema change and must bump
`STAMP_SCHEMA_VERSION` deliberately, alongside the stamp adopting the substrate
block wholesale, rather than riding in on a refactor.

## The three states, which are the whole point

Every group reports `state`, following `provenance.py::_sol_state`, which
already established the idiom here:

    present       read it, value follows
    absent        looked, and it is genuinely not set
    unobservable  cannot be read from this context, or by any means available

The distinction between the last two is not pedantry. A field that reads
`absent` when the truth is `unobservable` is a recorded claim that the thing was
off, and a validator downstream will treat it as fact. That is the same defect
as reading a missing power limit as "presumably stock", which is what this whole
line of work exists to close.

A caller that cannot supply context gets `unobservable` with a reason, never a
silent omission and never a plausible default.
"""

from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

SUBSTRATE_VERSION = 1

_REPO = Path(__file__).resolve().parent


def _run(cmd, timeout=15):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_head(path):
    """Short HEAD plus a dirty marker. Same contract as provenance.py's."""
    head = _run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], 5)
    if head is None:
        return None
    dirty = _run(["git", "-C", str(path), "status", "--porcelain"], 5)
    return head + ("-dirty" if dirty else "")


def host():
    """GPU and host power state. Reads nvidia-smi; needs no CUDA context.

    Safe to call during a render: one sample, no allocation.
    """
    if shutil.which("nvidia-smi") is None:
        return {"state": "unobservable", "why": "no nvidia-smi on PATH"}

    fields = [
        "name", "driver_version", "power.limit", "power.default_limit",
        "power.min_limit", "power.max_limit", "clocks.max.sm",
        "clocks.max.memory", "persistence_mode",
    ]
    raw = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits"])
    if not raw:
        return {"state": "unobservable", "why": "nvidia-smi returned nothing"}

    # Single-card box, but do not bake that in: record every card and let the
    # caller decide. A second GPU appearing must not silently change field
    # meanings from "the GPU" to "the first GPU".
    cards = []
    for line in raw.splitlines():
        v = [c.strip() for c in line.split(",")]
        g = dict(zip(fields, v))

        def num(key):
            try:
                return float(g[key])
            except (KeyError, ValueError):
                return None

        cards.append({
            "name": g.get("name"),
            "driver_version": g.get("driver_version"),
            "power_limit_watts": num("power.limit"),
            "power_limit_default_watts": num("power.default_limit"),
            "power_limit_min_watts": num("power.min_limit"),
            "power_limit_max_watts": num("power.max_limit"),
            "clock_max_sm_mhz": num("clocks.max.sm"),
            "clock_max_memory_mhz": num("clocks.max.memory"),
            "persistence_mode": g.get("persistence_mode"),
        })

    return {
        "state": "present",
        "cuda_version": _cuda_version(),
        "gpus": cards,
        "clock_locks": clock_locks(),
    }


def clock_locks():
    """Whether -lgc / -lmc locks are set. Currently: NOT OBSERVABLE.

    **This is a finding, not a stub.** The parallel schema draft proposes
    `gpu_clock_lock_graphics` and `gpu_clock_lock_memory` as required fields
    with `null` allowed, `null` meaning "confirmed no lock". On this driver
    that value cannot be produced honestly:

      - `nvidia-smi -q -d CLOCK` returns "Requested functionality has been
        deprecated" for both Applications Clocks and Default Applications
        Clocks.
      - `--help-query-gpu` exposes no field for `-lgc` / `-lmc` lock state.
        `clocks_event_reasons.applications_clocks_setting` reports whether
        application clocks are *limiting*, which is a different question and is
        also tied to the deprecated mechanism.

    So an emitter asked for that field can only guess, and the guess would be
    written down as `null` -- "confirmed no lock" -- which is precisely the
    absent-vs-unobservable collapse this module exists to prevent. Requiring a
    field the emitter cannot observe forces it to lie.

    Recorded as unobservable until someone finds a real read. The honest
    alternative, if the locks matter enough, is to record observed clock
    behaviour under load rather than the lock setting.
    """
    return {
        "state": "unobservable",
        "why": "no nvidia-smi query field for -lgc/-lmc; applications clocks "
               "deprecated on this driver",
    }


def _cuda_version():
    """CUDA version, which the query interface does not expose at all.

    Only in the banner, and **the field was renamed**: driver 610.x reports
    `CUDA UMD Version` and deprecates `CUDA Version`. A reader matching only the
    old spelling returns None on a current driver and looks like "not
    installed" -- found here by the value coming back null on a box that
    plainly has CUDA. Both spellings are matched, newest first.
    """
    full = _run(["nvidia-smi"])
    if not full:
        return None
    for pattern in (r"CUDA UMD Version:\s*([0-9.]+)", r"CUDA Version:\s*([0-9.]+)"):
        m = re.search(pattern, full)
        if m:
            return m.group(1)
    return None


def _checkout(path):
    """A git checkout's identity, with the same three states as everything else.

    Returning a bare string here (and `None` on any failure) was the original
    shape, and it was wrong twice over. It gave the schema two shapes for one
    concept -- structured records for packages, bare strings for checkouts --
    and it collapsed "this is not a git checkout" into "git failed to run",
    which is the absent-versus-unobservable distinction this module exists to
    keep. Found by reading this module's own `--keys` output before handing it
    to a schema author, which is the argument for generating that output rather
    than describing it.
    """
    if path is None:
        return {"state": "unobservable", "why": "path could not be resolved"}
    if not (path / ".git").exists():
        return {"state": "absent", "why": "not a git checkout"}
    head = git_head(path)
    if head is None:
        return {"state": "unobservable", "why": "git rev-parse failed"}
    return {"state": "present", "head": head}


def _comfy_root():
    """The ComfyUI checkout, or None -- established by a marker, not by depth.

    `_REPO.parents[1]` only means ComfyUI under the
    `<comfy>/custom_nodes/<pack>/` layout. In a worktree, a bare clone, or a
    symlinked node pack, the grandparent is some unrelated repository whose HEAD
    would be written into the record as `builds.comfyui` -- a fabricated fact
    presented as a read, in the module whose thesis is never a plausible
    default. Checked against files only ComfyUI has.
    """
    if len(_REPO.parents) < 2:
        return None
    root = _REPO.parents[1]
    if any((root / marker).exists()
           for marker in ("comfyui_version.py", "comfy", "main.py")):
        return root
    return None


def builds():
    """Identities of the code that ran.

    Needs no ComfyUI and no CUDA *context*, but note it does IMPORT the packages
    it reports -- reading torch's version means importing torch. Callers that
    must stay light should call `host()` alone, which shells out to nvidia-smi
    and imports nothing.
    """
    comfy_root = _comfy_root()
    return {
        "state": "present",
        "h3_explorations": _checkout(_REPO),
        "comfyui": _checkout(comfy_root),
        "comfy_kitchen": _package("comfy_kitchen", probe_attr="sol_attn"),
        "sageattention": _package("sageattention", with_git=True),
        "torch": _package("torch"),
    }


def _package(name, probe_attr=None, with_git=False):
    """Version of an installed package, plus what distinguishes a local build.

    `comfy_kitchen`'s fork declares the same version string as the stock wheel,
    so the local tag is the entire signal -- provenance.py records this and
    `bench/check_sol_kernel.py` is built on it. `probe_attr` additionally
    reports whether the symbol that makes the build useful is actually present:
    a stock wheel swapped in by --force-reinstall renders successfully with
    every Sol call silently falling back to dense.
    """
    # ImportError means not installed, which is `absent` -- a complete answer.
    # Any OTHER exception means it IS installed and failed to load: a torch
    # whose CUDA extension mismatches the driver, a sageattention whose
    # compiled op is missing. Recording that as `absent` says "this was not
    # installed" about a run that plainly used it, which is the exact
    # absent-versus-unobservable collapse this module argues against -- so it
    # was worth catching here, in the file making the argument.
    try:
        mod = __import__(name)
    except ImportError as exc:
        return {"state": "absent", "why": f"not importable: {exc}"}
    except Exception as exc:
        return {"state": "unobservable",
                "why": f"installed but raised on import: "
                       f"{type(exc).__name__}: {exc}"}
    try:
        from importlib.metadata import version
        ver = version(name)
    except Exception:
        ver = getattr(mod, "__version__", None)
    rec = {"state": "present", "version": ver}
    if probe_attr is not None:
        rec["has_" + probe_attr] = hasattr(mod, probe_attr)
    if with_git:
        # Routed through _checkout for the same reason builds() is: an
        # installed package's source tree may or may not be a checkout, and
        # "installed from a wheel" must not read the same as "git failed".
        try:
            rec["checkout"] = _checkout(Path(mod.__file__).resolve().parents[1])
        except Exception:
            rec["checkout"] = {"state": "unobservable",
                               "why": "package has no resolvable source path"}
    return rec


def graph(prompt=None):
    """The submitted graph's identity, hashed the way provenance.py hashes it."""
    if prompt is None:
        return {
            "state": "unobservable",
            "why": "caller supplied no prompt; a bench client has the graph it "
                   "submitted, a node has it from hidden inputs",
        }
    # Spelled out rather than shared via a kwargs dict, so that a diff against
    # `provenance.py`'s identical call is legible. Two implementations of "the
    # graph's identity" that disagree on separators produce different hashes for
    # one graph and nothing reports it -- the records simply never join.
    blob = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
    return {"state": "present", "sha256": hashlib.sha256(blob).hexdigest()}


def weights(prompt=None):
    """Which weight files the graph loads.

    Matched on the VALUE looking like a weight file, not on node class names.
    Class names drift and a hardcoded list goes quietly incomplete -- the same
    failure mode as a check that enumerates graph directories instead of using
    the shared walk.

    Quantization is reported under a key that says it was inferred from the
    filename, because that is what it is. The repo's builds are distinguished
    only by name (`_int8_convrot`, `_fp8_scaled`, `_w4a8_mixed`), and
    `workflows/h3_config.py` records that the loader prints `torch.float16` for
    both int8 and dequantized-fallback builds -- so the file name is the only
    available signal and must not be presented as a read of the tensor data.

    **This returns LESS than the capture manifest already records**, and the
    commit that introduced it claimed otherwise. `docs/capture_manifest_schema.md`
    has `audio_vae` and a `loras` array, and the existing manifest populates
    both -- with `strength` and `rank` per LoRA, which nothing here recovers,
    because strength is a node input rather than a filename. The claim that
    those slots were missing came from reading the schema's `required` list and
    inferring the properties did not exist. `required` is a constraint list, not
    the property inventory, and that is the second claim in one session built on
    that misreading -- the first said the manifest had no field for host power
    state, when `provenance.gpu_power_limit_watts` was already there.

    So treat this function as a cross-check on a graph, not as the authority on
    what ran. The manifest is richer where a manifest exists; this is for the
    contexts that have no manifest at all.
    """
    if prompt is None:
        return {
            "state": "unobservable",
            "why": "caller supplied no prompt",
        }
    found = []
    for node_id, node in sorted((prompt or {}).items(), key=_node_sort_key):
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        files = [(k, v) for k, v in inputs.items()
                 if isinstance(v, str) and v.endswith((".safetensors", ".ckpt", ".pt"))]
        if not files:
            continue
        # Every other literal input on the same node. In API format a wired
        # input is [node_id, slot], so "not a list" is what distinguishes a
        # setting from a link. Captured generically rather than by naming
        # `strength_model`, so a knob that matters on some future loader node
        # is recorded without anyone remembering to add it.
        file_keys = {k for k, _ in files}
        settings = {k: v for k, v in sorted(inputs.items())
                    if k not in file_keys and not isinstance(v, list)}
        for key, val in files:
            found.append({
                "node_id": node_id,
                "class_type": node.get("class_type"),
                "input": key,
                "file": val,
                "quantization_inferred_from_filename": infer_quantization(val),
                "rank_inferred_from_filename": _infer_rank(val),
                "node_settings": settings,
            })
    if not found:
        return {"state": "absent", "why": "no weight-file input found in graph"}
    return {"state": "present", "files": found}


def _node_sort_key(item):
    """Numeric where the id is numeric. Otherwise '10' sorts before '2'."""
    key = item[0]
    return (0, int(key), "") if str(key).isdigit() else (1, 0, str(key))


def _infer_rank(filename):
    """LoRA rank, if the filename states it. Inferred, and named as inferred.

    Deliberately NOT defaulted. `bench/generate_capture_manifest.py:133` writes
    `"rank": 256` as a literal for every LoRA it finds, which is correct for the
    one LoRA this repo currently ships and silently wrong for any other -- a
    constant presented as a recorded observation. None here means the filename
    does not say, which is a fact; a number means the filename said so, which is
    a different and weaker fact than reading the tensor shapes.
    """
    m = re.search(r"rank[_-]?(\d+)", filename)
    return int(m.group(1)) if m else None


def infer_quantization(filename):
    for tag in ("int8_convrot", "fp8_scaled", "w4a8_mixed", "bf16", "fp16", "fp32"):
        if tag in filename:
            return tag
    return None


def substrate(*, prompt=None):
    """The block. Callers add their own `kind` and payload around it."""
    return {
        "substrate_version": SUBSTRATE_VERSION,
        "host": host(),
        "builds": builds(),
        "graph": graph(prompt),
        "weights": weights(prompt),
    }


def key_paths(prompt=None):
    """Every key path the block can emit, for a schema author to draft against.

    Exists so the schema is written against what the emitter ACTUALLY returns
    rather than against a list someone pasted into a message. A pasted key set
    is a second copy of this structure and drifts from it silently -- the
    schema would then validate fields nothing emits, and miss fields something
    does, with both sides green.

    Call with a real prompt. Without one, `graph` and `weights` short-circuit to
    `unobservable` and their key paths never appear, so a schema drafted from
    the promptless output would be missing two whole subtrees.
    """
    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k in sorted(node):
                yield from walk(node[k], f"{prefix}.{k}" if prefix else k)
        elif isinstance(node, list):
            # UNION every element, not element 0. Sampling the first was the
            # original shape and it defeated the function's whole purpose: on a
            # real graph `weights.files[]` holds a UNETLoader with
            # `node_settings.weight_dtype`, a CLIPLoader with `device` and
            # `type`, two VAELoaders with none, and a LoraLoaderModelOnly with
            # `strength_model`. Reporting only the first hid three of those from
            # the schema author, and `rank_inferred_from_filename` typed as
            # NoneType because the first entry happened to have no rank. A key
            # surface that samples is not a key surface.
            if not node:
                yield prefix + "[]"
                return
            for element in node:
                yield from walk(element, prefix + "[]")
        else:
            yield f"{prefix}  ({type(node).__name__})"

    # Types are unioned too: one path seen as both int and NoneType reports
    # `int|NoneType`, which is the difference between "optional integer" and
    # "always null" to whoever is writing the schema.
    types_by_path = {}
    for entry in walk(substrate(prompt=prompt)):
        path, _, tname = entry.rpartition("  (")
        path, tname = (path, tname.rstrip(")")) if path else (entry, "")
        bucket = types_by_path.setdefault(path, [])
        if tname and tname not in bucket:
            bucket.append(tname)
    # Sorted, not discovery-ordered: a path first seen on the fifth list element
    # would otherwise appear far from its siblings, which is exactly where a
    # schema author stops reading.
    return [f"{p}  ({'|'.join(sorted(types_by_path[p]))})" if types_by_path[p] else p
            for p in sorted(types_by_path)]


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    graph_arg = next((a for a in args if not a.startswith("-")), None)
    prompt_obj = json.loads(Path(graph_arg).read_text()) if graph_arg else None

    if "--keys" in args:
        if prompt_obj is None:
            print("# NOTE: no graph given, so `graph` and `weights` report "
                  "unobservable and their subtrees are absent below.\n"
                  "# Pass an API-format graph JSON to see the full surface.\n")
        for path in key_paths(prompt_obj):
            print(path)
    else:
        print(json.dumps(substrate(prompt=prompt_obj), indent=2, sort_keys=True))
