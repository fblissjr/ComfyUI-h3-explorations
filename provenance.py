"""Bench-only: stamp what a render's settings actually RESOLVED to.

ComfyUI already records everything you *typed*. `/history/{prompt_id}` carries
the whole prompt graph with every widget value and the output filenames, for
free. So this node deliberately does not re-record any of that. It records only
what `/history` structurally cannot know:

- resolved sigmas from `sol_compose`, and the sparse-step count computed from
  them against the actual schedule
- the eleven Sol closure values, which are what really ran if anything
  downstream replaced the override after it was installed
- node-pack HEADs and the sage build, which live outside the graph entirely
- the snapped frame count and resolved canvas, because requested != actual

The single field this exists for is `n_sparse`. It is not a setting anywhere:
it is the intersection of the sigma window with the sampler's schedule, so two
schedulers with identical `sol_compose` bounds can run a different number of
sparse steps and nothing in the graph, the logs or `/history` says so.

## Two cautions, both load-bearing

**A well-provenanced number is not a verified one.** A stamp makes bookkeeping
failures visible and does nothing about invented mechanisms — and it makes the
latter *more* dangerous, because a number with a full provenance record beside
it reads as more trustworthy while being exactly as capable of carrying a wrong
causal story. Recording state and explaining a result are different jobs.

**This records what settings resolved to, never why a number came out the way
it did.** Those get confused precisely because the stamp sits next to the
number. A mechanism claim still needs both arms instrumented.

## Why bench-only

It reads another node pack's closure internals, so it breaks when that pack
changes. On the bench surface that breakage is cheap and expected. In a shipped
workflow it would break in a user's render, which is the wrong place. Wire it
in benches; keep it out of the shipped workflows.

## Joining a stamp to a render

ComfyUI does not expose `prompt_id` to nodes (`io.Hidden` has `unique_id`,
`prompt`, `extra_pnginfo`, `dynprompt` and no id), so the stamp keys on a
canonical hash of the prompt graph instead. `/history` entries carry the same
graph, so hashing each one's `prompt[2]` the same way joins the two and gets you
the graph, the timings and the output filename together.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# `substrate.py` owns the shared readers. Both spellings are needed and neither
# is redundant: ComfyUI loads this file as a package member, where the relative
# import is the correct one, while `bench/check_provenance_stamp.py` loads it by
# path under a bare module name with no parent package, where a relative import
# raises and would have turned that check into a silent exit 2. Both paths are
# exercised -- the check runs green, and the node loads in a live ComfyUI.
try:
    from .substrate import git_head as substrate_git_head
except ImportError:  # loaded by path, not as a package member
    from substrate import git_head as substrate_git_head

import torch

import folder_paths
from comfy_api.latest import io

logger = logging.getLogger(__name__)

STAMP_SCHEMA_VERSION = 2

# Every Sol setting that exists only inside the override closure. Anything here
# that introspection cannot reach becomes an explicit "not detected" IN THE
# RECORD -- a reader diffing two sidecars never sees a log line, and a silently
# absent key is indistinguishable from a setting that was never on.
# Corrected 2026-08-16 against `vendor/sol_attn_minimax.py:497-501`, the CUDA
# node's real `make_override` signature. This list was written in the TRITON
# vocabulary and never updated when the graphs migrated on 2026-08-14, so it was
# wrong in both directions at once: it asked for `int8_qk`, `use_tma` and
# `int8_pv`, which do not exist on the CUDA node and so recorded "not detected"
# on every render forever, and it omitted `routed_cap_percent`, `centroid_tail`
# and `reuse_qkv_memory`, which do run and were therefore absent from the record
# entirely. `centroid_tail` is the one that stings -- it has a live A/B with a
# deadline on it (upstream may remove the toggle) and no stamped render says
# which way it was set.
#
# The failure mode is the one this file's own docstring warns about: three
# permanently-absent keys read as introspection failure, and a reader diffing
# two sidecars cannot tell that from a setting that was never on.
SOL_CLOSURE_KEYS = (
    "tau", "min_tokens", "sigma_start", "sigma_end", "verbose",
    "sink_conditioning", "dense_blocks", "tau_profile",
    "routed_cap_percent", "centroid_tail", "reuse_qkv_memory",
)

NOT_DETECTED = "not detected"


def _git_head(path: Path) -> str:
    """Adapter over `substrate.git_head`, which is the one implementation.

    This file had its own copy until 2026-08-17, identical in behaviour and
    differing only in its failure sentinel. `substrate.py` is the shared reader
    because it depends on nothing outside the standard library, where this
    module needs `folder_paths` and drags in `comfy_api` -- so `bench/` can
    reach that one and could never have reached this one.

    **The adapter is the point, not overhead.** `substrate.git_head` returns
    `None` on failure and this returns `NOT_DETECTED`, and that string is part
    of the stamp's recorded output. Translating here keeps one implementation of
    the git reading while leaving every stamp byte-for-byte what it was, so
    `STAMP_SCHEMA_VERSION` does not move for a refactor. When the stamp adopts
    the substrate block wholesale, this adapter and the sentinel go together, and
    that IS a schema change.
    """
    return substrate_git_head(path) or NOT_DETECTED


def _jsonable(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sol_state(transformer_options, sigmas):
    """Three states, and the caller must be able to tell them apart.

    absent   -- no override installed; nothing to record, not an error
    broken   -- override installed but its settings are unreachable; raise
    present  -- resolved values recorded
    """
    override = transformer_options.get("optimized_attention_override")
    if override is None:
        return {"state": "absent"}

    # An override being present is not the same as SOL being present. Our own
    # sage node installs one on every graph it patches (`attention.py`,
    # `make_sage_override`), so a sage-only graph -- every dense baseline, both
    # capture graphs -- reached the closure read below, found none of Sol's
    # names, and reported "broken". That is the third-state failure CLAUDE.md
    # warns about: correctly absent read as broken, on exactly the graphs whose
    # job is to have no Sol. Found 2026-08-21 when the first
    # open-experiment-22 forward raised on a graph that never wired Sol.
    #
    # Walk our own chain to whatever we wrapped before deciding. If nothing
    # under it is foreign, Sol is absent and the render is fully attributable.
    probe, ours = override, []
    while getattr(probe, "h3_kernel", None) is not None:
        ours.append(probe.h3_kernel)
        probe = probe.h3_previous
    if probe is None and ours:
        return {"state": "absent", "override_installed_by": ours}
    if ours:
        override = probe

    state: dict[str, object] = {"state": "present"}
    compose = transformer_options.get("sol_compose")
    state["sol_compose"] = {k: _jsonable(v) for k, v in compose.items()} if compose else NOT_DETECTED
    state["morton"] = bool(transformer_options.get("sol_morton", False))
    state["morton_curve"] = _jsonable(transformer_options.get("sol_morton_curve"))

    freevars = getattr(getattr(override, "__code__", None), "co_freevars", ()) or ()
    cells = getattr(override, "__closure__", None) or ()
    reached = {}
    for name, cell in zip(freevars, cells):
        if name in SOL_CLOSURE_KEYS:
            try:
                reached[name] = _jsonable(cell.cell_contents)
            except ValueError:  # empty cell
                pass
    # setdefault, not a plain dict build: an upstream rename drops the name out
    # of co_freevars, and this turns that into an explicit "cannot tell" rather
    # than a key that quietly disappears from the record.
    for key in SOL_CLOSURE_KEYS:
        reached.setdefault(key, NOT_DETECTED)
    state["closure"] = reached

    if all(v == NOT_DETECTED for v in reached.values()):
        state["state"] = "broken"
        return state

    # n_sparse: the whole reason this node exists. Not readable anywhere -- it
    # is the window intersected with the schedule.
    s_start, s_end = reached.get("sigma_start"), reached.get("sigma_end")
    if sigmas is None or not isinstance(s_start, (int, float)) or not isinstance(s_end, (int, float)):
        state["n_sparse"] = NOT_DETECTED
    else:
        # The model is evaluated at sigmas[0..steps-1]; the terminal sigma never
        # gets an eval. Counting the full tensor overcounts by one whenever the
        # terminal sigma falls inside the window -- which it does exactly when
        # end_percent == 1.0, since percent_to_sigma(1.0) is 0.0. That is the
        # widest-window arm, i.e. the one most likely to be measured.
        evals = sigmas[:-1]
        inside = (evals <= s_start) & (evals >= s_end)
        state["n_sparse"] = int(inside.sum())
        state["n_evals"] = int(evals.numel())
    return state


class MiniMaxH3ProvenanceStamp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ProvenanceStamp",
            display_name="MiniMax H3 Provenance Stamp (bench)",
            category="model/debug/minimax",
            description=(
                "BENCH ONLY. Writes a JSON sidecar recording what this render's "
                "settings resolved to -- sparse-step count, Sol closure values, "
                "node-pack HEADs, snapped frame count and canvas. Does not record "
                "what you typed; /history already has that. Wire the sampler's "
                "LATENT through it so it runs after sampling."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip=(
                    "Pass the sampler's output through. Required for ordering: "
                    "ComfyUI orders by dependency, not graph position, so without "
                    "a real data dependency this can legally run BEFORE sampling.")),
                io.Model.Input("model"),
                io.Sigmas.Input("sigmas", optional=True, tooltip=(
                    "From BasicScheduler. Without it n_sparse cannot be computed, "
                    "which is the one field this node exists for.")),
                io.String.Input("note", default="", multiline=False, optional=True),
            ],
            outputs=[io.Latent.Output(display_name="latent")],
            hidden=[io.Hidden.prompt],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, model, sigmas=None, note="") -> io.NodeOutput:
        to = (model.model_options or {}).get("transformer_options", {}) or {}
        sol = _sol_state(to, sigmas)

        here = Path(__file__).resolve().parent
        record = {
            "stamp_schema_version": STAMP_SCHEMA_VERSION,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": note,
            "sol": sol,
            # `sol_attn` was the TRITON pack's git HEAD until 2026-08-16, which
            # stamped the wrong thing: no graph has wired the Triton node since
            # 2026-08-14, so every record named a pack that did not run while
            # saying nothing about the kernel that did. The CUDA path's identity
            # is the comfy_kitchen build, and that is the one worth having --
            # the fork build declares a version identical to the stock PyPI
            # wheel, so without the local tag nothing distinguishes them.
            # Schema bumped to 2 for the key change.
            "builds": {
                "h3_explorations": _git_head(here),
                "sol_attn_cuda": cls._comfy_kitchen_version(),
                "comfyui": _git_head(Path(folder_paths.base_path)),
                "sageattention": cls._sage_version(),
            },
            "resolved": cls._geometry(latent),
        }

        prompt = getattr(cls.hidden, "prompt", None)
        if prompt is not None:
            blob = json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
            record["graph_sha256"] = hashlib.sha256(blob).hexdigest()
        else:
            record["graph_sha256"] = NOT_DETECTED

        out_dir = Path(folder_paths.get_output_directory()) / "provenance"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Second-resolution plus graph hash is NOT unique: re-running an
        # identical graph for variance collides, and the later stamp silently
        # replaces the earlier one -- a provenance tool losing provenance.
        stem = record["utc"].replace(":", "").replace("-", "")
        base = f"stamp_{stem}_{record['graph_sha256'][:8]}"
        path = out_dir / f"{base}.json"
        serial = 0
        while path.exists():
            serial += 1
            path = out_dir / f"{base}_{serial:02d}.json"
        record["stamp_serial"] = serial
        path.write_text(json.dumps(record, indent=2, sort_keys=True))

        if sol["state"] == "broken":
            raise RuntimeError(
                "Sol-Attn's attention override is installed but none of its settings "
                f"could be read from the closure. The stamp at {path.name} records "
                "'broken' rather than a hollow set of defaults, but treat any "
                "measurement from this render as unattributed -- most likely the "
                "pack renamed its parameters and SOL_CLOSURE_KEYS needs updating."
            )

        logger.info(
            "[h3] provenance -> %s (sol=%s, n_sparse=%s)",
            path.name, sol["state"], sol.get("n_sparse", "n/a"),
        )
        return io.NodeOutput(latent)

    @staticmethod
    def _comfy_kitchen_version():
        """The CUDA Sol kernel's identity: the installed `comfy_kitchen` build.

        This is the only field that can tell the fork build apart from the
        stock wheel. Both declare `0.2.31`, so the local tag
        (`0.2.31+sol.c04ef20`) is the whole signal -- see
        `bench/check_sol_kernel.py`. Reports whether `sol_attn` is actually
        present too, because a stock wheel swapped in by a `--force-reinstall`
        makes every Sol call fall back to dense and renders successfully.
        """
        try:
            import comfy_kitchen
            try:
                from importlib.metadata import version
                ver = version("comfy_kitchen")
            except Exception:
                ver = getattr(comfy_kitchen, "__version__", NOT_DETECTED)
            has_sol = hasattr(comfy_kitchen, "sol_attn")
            return f"{ver}{'' if has_sol else ' (NO sol_attn)'}"
        except Exception:
            return NOT_DETECTED

    @staticmethod
    def _sage_version():
        try:
            import sageattention
            ver = getattr(sageattention, "__version__", NOT_DETECTED)
            src = Path(sageattention.__file__).resolve().parent.parent
            return f"{ver}@{_git_head(src)}"
        except Exception:
            return NOT_DETECTED

    @staticmethod
    def _geometry(latent):
        """Resolved, not requested -- today's own finding is that they differ."""
        try:
            samples = latent["samples"]
            video = samples[0] if not torch.is_tensor(samples) else samples
            _, _, latent_t, lh, lw = video.shape
            # video latent frames are 5n+2 for 17n+5 pixel frames
            frames = 17 * ((latent_t - 2) // 5) + 5 if latent_t >= 2 else NOT_DETECTED
            return {
                "canvas": [int(lw) * 16, int(lh) * 16],
                "latent_t": int(latent_t),
                "frame_count": frames,
                "duration_s": round(frames / 24, 4) if isinstance(frames, int) else NOT_DETECTED,
                "packed_rows_per_frame": int(lh // 2) * int(lw // 2),
            }
        except Exception as exc:
            # Type only, never the message. A sidecar is meant to be shared
            # next to a render, and an exception string is the one field here
            # that can carry a filesystem path out of the machine.
            return {"error": type(exc).__name__}
