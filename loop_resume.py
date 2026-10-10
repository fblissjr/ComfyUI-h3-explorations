"""Resume a frozen-audio loop from the first window that changed.

`docs/h3_audio_freeze.md` owns the lane; `audio_freeze_song.py` is the node
that uses this. Imports nothing from the pack, so
`bench/check_audio_freeze.py` exercises it without the node.

**What a window's key says.** A window is reused when the key stored beside
it equals the key this run computes for it, and a key is built from what
produced the window:

- the root: a hash of everything upstream of the song node in the queued API
  prompt (models, LoRA, attention settings, loaders, sampler, sigmas,
  references) plus the song node's own literal inputs that reach every
  window (canvas, context, mask, level, crf), minus `SONG_PER_WINDOW` and the
  file-only switches; and a hash of the track's samples;
- the window's own text, frame count, start and seed;
- the previous window's key, because a window's head is the previous
  window's tail.

The song node's prompt text, timeline, `extent`, window length and seed are
deliberately not in the root. They reach each window through its own text,
frames, start and seed, so editing a later prompt block or covering more of
the track leaves the earlier windows' keys alone.

**Reuse is a prefix.** Windows are reused in order until the first one whose
key does not match, and everything from there renders again. A window that
renders again need not reproduce its old latent bit for bit, so a later
stored window, keyed on the same inputs, could sit on a slightly different
head; the prefix rule never reuses past a render.

**What the key cannot see.** A file replaced on disk under the same name (a
checkpoint, a LoRA, a reference still) leaves every key unchanged. The node's
`reuse_windows` switch is the way out.

**A key says how a window was sampled, not how its file was cut.** The frames
a window's video holds are stored beside the key (`written`), and a window is
reused only when that is what this run would write (`stored_frames`). It
matters for one window, the last, when the track ends inside it.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

import comfy.nested_tensor
import comfy.utils

#: Song-node inputs that reach a window only through that window's own text,
#: frames, start and seed, or that change files beside the windows and not the
#: windows. `lists` fills the text and `timeline` places the windows, so a
#: change to either moves only the windows whose text, length or start moved;
#: `preview` renders nothing, so a preview and the render after it share keys.
#: Reasoned, from `MiniMaxH3AudioFreezeSong.execute`.
SONG_PER_WINDOW = ("prompt", "timeline", "preview", "extent", "extent.seconds", "window_frames", "seed",
                   "filename_prefix", "save_metadata_png", "keep_windows", "reuse_windows", "lists",
                   "save_mask_review", "continue_from")


def _is_link(value) -> bool:
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], (str, int)) and isinstance(value[1], int))


def graph_signature(prompt, node_id, skip=()) -> str | None:
    """A hash of `node_id` and everything upstream of it in an API prompt.

    Node ids are replaced by visit order, as core's own cache key does
    (`comfy_execution/caching.py::CacheKeySetInputSignature`), so renumbering
    a graph changes nothing. `skip` names inputs of `node_id` itself to leave
    out. None when the prompt or the node is missing: no key, no reuse.
    """
    if not isinstance(prompt, dict) or node_id is None or str(node_id) not in prompt:
        return None
    order: dict[str, int] = {}
    records = []

    def visit(nid) -> int:
        nid = str(nid)
        if nid in order:
            return order[nid]
        node = prompt.get(nid)
        if not isinstance(node, dict):
            raise KeyError(nid)
        order[nid] = len(order)
        inputs = {}
        for name, value in sorted((node.get("inputs") or {}).items()):
            if nid == str(node_id) and name in skip:
                continue
            inputs[name] = ["link", visit(value[0]), value[1]] if _is_link(value) else value
        records.append((order[nid], node.get("class_type"), inputs))
        return order[nid]

    try:
        visit(node_id)
    except KeyError:
        return None
    blob = json.dumps(sorted(records, key=lambda r: r[0]), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def track_hash(waveform: torch.Tensor, rate: int) -> str:
    h = hashlib.sha256(str(int(rate)).encode())
    h.update(waveform.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def root_key(signature: str | None, track: str) -> str | None:
    return None if signature is None else hashlib.sha256(f"{signature}:{track}".encode()).hexdigest()


def window_key(root: str, number: int, text: str, frames: int, start: float, seed: int,
               previous: str | None) -> str:
    blob = json.dumps({"root": root, "window": int(number), "text": text, "frames": int(frames),
                       "start": round(float(start), 6), "seed": int(seed), "previous": previous},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def window_paths(work_dir: str, filename: str, number: int) -> tuple[str, str]:
    """(video, latent) for window `number`, counted from 1."""
    base = os.path.join(work_dir, f"{filename}_window_{int(number)}")
    return base + ".mp4", base + ".safetensors"


def review_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`'s mask review, beside its video: the render stacked over what was regenerated."""
    return window_paths(work_dir, filename, number)[0][:-len(".mp4")] + "_with_mask.mp4"


def region_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`'s region, beside its latent: the mask and the token region its composite was run with
    (`video_mask.save_window_region`). Only a window rendered over a source has one."""
    return window_paths(work_dir, filename, number)[0][:-len(".mp4")] + "_region.npz"


def planned_region_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`'s PLANNED region, written by a preview of the song node and by nothing else: the file a
    render of the same graph would write as `region_path`, made with nothing sampled. Its own name, so a preview
    never replaces a render's file and the two can be set side by side."""
    return window_paths(work_dir, filename, number)[0][:-len(".mp4")] + "_planned_region.npz"


def plan_path(work_dir: str, filename: str) -> str:
    """The plan a preview of the song node writes beside its planned regions: every window's frames, what it
    writes, and what a cut inside a latent step does to it."""
    return os.path.join(work_dir, f"{filename}_plan.json")


def save_window(work_dir: str, filename: str, number: int, key: str, samples, trim: int,
                next_start: float, written: int) -> str:
    """Store a rendered window's sampled latent and what the next window needs from it.

    Written after the window's video, so a latent on disk means its video
    finished. `written` is how many frames that video holds (`stored_frames`).
    """
    _video, latent_path = window_paths(work_dir, filename, number)
    streams = samples.unbind() if getattr(samples, "is_nested", False) else [samples]
    tensors = {f"stream_{i}": s.detach().cpu().contiguous() for i, s in enumerate(streams)}
    meta = {"key": key, "trim": str(int(trim)), "next_start": repr(float(next_start)),
            "nested": str(bool(getattr(samples, "is_nested", False))), "written": str(int(written))}
    comfy.utils.save_torch_file(tensors, latent_path, metadata=meta)
    return latent_path


def read_window(work_dir: str, filename: str, number: int) -> dict | None:
    """The stored key, trim, next start and frame count of window `number`, without loading its tensors.

    `written` is None for a store from before 2026-10-06, which did not record it."""
    video_path, latent_path = window_paths(work_dir, filename, number)
    if not (os.path.isfile(video_path) and os.path.isfile(latent_path)):
        return None
    try:
        from safetensors import safe_open
        with safe_open(latent_path, framework="pt") as f:
            meta = f.metadata() or {}
        return {"key": meta["key"], "trim": int(meta["trim"]), "next_start": float(meta["next_start"]),
                "written": int(meta["written"]) if "written" in meta else None,
                "video": video_path, "latent": latent_path}
    except Exception:  # noqa: BLE001 -- an unreadable store is a store to render over, not an error
        return None


def stored_key(latent_path: str) -> str:
    """The key a stored window's latent was saved under, for a run that continues from it.

    That run's first window is keyed on it as a window is keyed on the one before it in its own
    run, so where the continued file sits does not matter (`continue_from` is in
    `SONG_PER_WINDOW`) and what it holds does: a first stretch rendered again changes the key and
    the stretch after it renders again. Refused when the file is not a window this pack stored.
    """
    try:
        from safetensors import safe_open
        with safe_open(latent_path, framework="pt") as f:
            meta = f.metadata() or {}
        return str(meta["key"])
    except Exception as exc:  # noqa: BLE001 -- a missing file, another kind of file, a store with no key
        raise ValueError(f"{latent_path} is not a stored window's latent (the `.safetensors` beside a window's video "
                         f"in a run's _windows folder): {type(exc).__name__}: {exc}") from exc


def stored_frames(stored: dict, frames: int) -> int:
    """How many frames a stored window's video holds, for a window `frames` long.

    What it was written with; or, for a store from before 2026-10-06, which did not record
    it, the whole window less the context trimmed from its head, since nothing came off a
    tail then. The song node reuses a stored window only when this equals what the run
    would write (`loop_plan.frames_written`). Without it a last window stored whole, under
    a track that ends inside it, has a key that matches for ever and a file that is too
    long: it never renders again, and the join cannot cut a copied stream cleanly
    (`loop_output.join_and_mux`).
    """
    return int(stored["written"]) if stored.get("written") is not None else int(frames) - int(stored["trim"])


def load_window_latent(latent_path: str) -> dict:
    """A stored window's sampled latent as the `previous` the window node takes."""
    sd, meta = comfy.utils.load_torch_file(latent_path, return_metadata=True)
    streams = [sd[k] for k in sorted(sd, key=lambda k: int(k.split("_")[1]))]
    nested = (meta or {}).get("nested") == "True"
    samples = comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]
    return {"samples": samples}
