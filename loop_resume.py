"""The files a frozen-audio loop leaves beside its windows, and a window's stored latent.

`docs/h3/audio_freeze.md` owns the lane; `audio_freeze_song.py` is the node
that uses this. Imports nothing from the pack, so
`bench/check_audio_freeze.py` exercises it without the node.

**Nothing here hands a window back by a key.** Until 2026-10-10 this module
keyed every stored window on the queued graph, the track, the text and the
seed, and a run reused the stored windows whose keys matched
(`reuse_windows`). The key could not see a file replaced under the same name
or a change to this pack's code, the reuse had been switched off in code
since 2026-10-07 after a stale stored mask hid a fix, and the owner's rule
of 2026-10-10 removed it: nothing an earlier run made is reused by a key; a
result is made again, or handed in explicitly as a named file or a wired
input (`AGENTS.md`, "Operative rules"; `docs/state/decisions.md`).

What is left is explicit. Every window's sampled latent is written beside
its video (`save_window`), and it is read again in two cases, each named by
a person: `continue_from` on the song node, which takes its first window's
context from it, and `bench/recomposite_window.py`, which decodes it.
"""

from __future__ import annotations

import os

import comfy.nested_tensor
import comfy.utils


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


def zoomed_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`'s zoomed picture, beside its video: what the window laid in the zoomed space, frame for
    frame with the video. Only a window of a run with `render_region` zoomed has one (`video_mask.zoomed_source`)."""
    return window_paths(work_dir, filename, number)[0][:-len(".mp4")] + "_zoomed.mp4"


def planned_region_path(work_dir: str, filename: str, number: int) -> str:
    """Window `number`'s PLANNED region, written by a preview of the song node and by nothing else: the file a
    render of the same graph would write as `region_path`, made with nothing sampled. Its own name, so a preview
    never replaces a render's file and the two can be set side by side."""
    return window_paths(work_dir, filename, number)[0][:-len(".mp4")] + "_planned_region.npz"


def plan_path(work_dir: str, filename: str) -> str:
    """The plan a preview of the song node writes beside its planned regions: every window's frames, what it
    writes, and what a cut inside a latent step does to it."""
    return os.path.join(work_dir, f"{filename}_plan.json")


def save_window(work_dir: str, filename: str, number: int, samples, trim: int,
                next_start: float, written: int) -> str:
    """Store a rendered window's sampled latent and what the next window needs from it.

    Written after the window's video, so a latent on disk means its video
    finished. `written` is how many frames that video holds.
    """
    _video, latent_path = window_paths(work_dir, filename, number)
    streams = samples.unbind() if getattr(samples, "is_nested", False) else [samples]
    tensors = {f"stream_{i}": s.detach().cpu().contiguous() for i, s in enumerate(streams)}
    meta = {"trim": str(int(trim)), "next_start": repr(float(next_start)),
            "nested": str(bool(getattr(samples, "is_nested", False))), "written": str(int(written))}
    comfy.utils.save_torch_file(tensors, latent_path, metadata=meta)
    return latent_path


def check_window_latent(latent_path: str) -> None:
    """Refuse a file that is not a window this pack stored: a missing file, another kind of file, a latent with
    no trim beside it. Only the file's header is read, so a run that names one (`continue_from`) is refused at
    its start, before a preview's plan and before any encode, and not minutes in."""
    try:
        from safetensors import safe_open
        with safe_open(latent_path, framework="pt") as f:
            if "trim" not in (f.metadata() or {}) or not any(k.startswith("stream_") for k in f.keys()):
                raise KeyError("no stored window's streams or trim in it")
    except Exception as exc:  # noqa: BLE001 -- whatever is wrong with it, the answer is the same sentence
        raise ValueError(f"{latent_path} is not a stored window's latent (the `.safetensors` beside a window's video "
                         f"in a run's _windows folder): {type(exc).__name__}: {exc}") from exc


def load_window_latent(latent_path: str) -> dict:
    """A stored window's sampled latent as the `previous` the window node takes; refused as
    `check_window_latent` refuses."""
    check_window_latent(latent_path)
    sd, meta = comfy.utils.load_torch_file(latent_path, return_metadata=True)
    streams = [sd[k] for k in sorted(sd, key=lambda k: int(k.split("_")[1]))]
    nested = (meta or {}).get("nested") == "True"
    samples = comfy.nested_tensor.NestedTensor(streams) if nested else streams[0]
    return {"samples": samples}
