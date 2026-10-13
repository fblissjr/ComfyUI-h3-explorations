#!/usr/bin/env python3
"""Make and fill an input video's analysis folder: the workflow that runs before any job on that video.

    <python> bench/clip_analysis.py init fun.mp4 [--copy fun_24p.mov --made-by "how" --checked "what the check said"]
    <python> bench/clip_analysis.py voice fun.mp4 --per-frame VOICE_PER_FRAME.csv --spans VOICE_SPANS.json --by WHO

**What it buys.** A rule that depends on the picture takes its value from this clip, measured before anything
renders (the owner, 2026-10-10; AGENTS.md). So the measuring is a workflow of its own that runs first, its
data sits in one folder named by the input video's FILE NAME (`data/<stem>/`, never a path), and every later
step reads that folder and refuses without it. `clip_folder.py` holds the rules every reader shares; this
tool is the writer.

**init** is the first step and the only one built so far. It finds the input video by its name under the
server's input folder (`H3_INPUT_DIR`), takes its identity (name, byte size, frames, rate, size, a hash of
its head) and writes `data/<stem>/manifest.json`. `--copy` names the every-frame copy at the model's rate
that jobs render from: it is found and identified the same way and recorded under `derived`, with how it was
made and what its frame-for-frame check said, when the caller knows; a copy made before this tool existed
says so in words. Run again on the same file it changes nothing that a render reads; given a file of the
same name with other bytes it makes the folder named by the whole file name, and the old one stays the old
file's. It also writes `README.md`, which says what the folder is and names the older, dated folders under
`data/` that hold earlier work on the same video: nothing existing is moved or deleted.

**voice** names the video's two voice tables in its manifest (`voice`: the per-frame table a capture takes
with `--voice` and the spans file its preflight takes with `--voice-spans`, each by a path from the repo's
root, with who said so and when). They are facts of the video, measured once: until 2026-10-13 their paths
were typed into every session's script. A tool that needs them reads them here (`voice_tables`) and refuses
when the manifest names none. Naming them changes nothing a render reads, so the fingerprint does not move.

To come, in this order: the subjects' pass (trackers, parts, pose) queued for the whole clip and gathered
into `analysis/<run>/` with checksums and the manifest's mask table; `profile.json`, what the clip measured
and what was set from it; a census that needs no picks.

Run it from ComfyUI's environment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import REPO  # noqa: E402

sys.path.insert(0, str(REPO))
import clip_folder as cf  # noqa: E402
import masked_job  # noqa: E402

REFUSED = masked_job.REFUSED
#: What `derived` says of a copy whose making or checking the caller did not state.
NOT_STATED = "not stated when the folder was made: the copy existed before this tool did"


def older_folders(name: str, data: Path) -> list[str]:
    """The dated folders under `data/` that earlier work on this video left: `<date>_<stem>_...`."""
    stem = cf.stem(name)
    return sorted(p.name for p in Path(data).glob(f"*_{stem}_*") if p.is_dir() and p.name[:4].isdigit())


def readme(manifest: dict, older: list[str]) -> str:
    derived = manifest.get("derived")
    lines = [f"# The analysis folder of `{manifest['source']['name']}`", "",
             "Everything measured or decided about this input video before anything renders from it. Made and",
             "filled by `bench/clip_analysis.py`; the rules every reader shares are in `clip_folder.py`. Untracked,",
             "like the rest of `data/`.", "",
             "- `manifest.json`: the video's identity (by name, size, frames and a hash of its head, never a path),",
             "  the copy jobs render from, every mask video with its size and first frame, and one fingerprint that",
             "  changes when anything a render reads changes.",
             "- `analysis/<run>/`: what one analysis run saved. A second run is a new folder.",
             "- `cast/`: who is in the video (`docs/wiki/who_is_who.md`).",
             "- `capture/`: `bench/capture_masked_run.py`'s tables, and `rules_input.json`.",
             "- `profile.json`: what this clip measured and what was set from it.",
             "- `jobs/<job>/`: one job's graphs, records and hand-set values.", "",
             "Every frame number here is a frame of "
             + (f"`{derived['name']}`, the copy made from the input." if derived else "the input video itself."), ""]
    if older:
        lines += ["## Earlier work on this video, where it was left", "",
                  "These folders were made before this one existed. Nothing in them was moved or deleted.", ""]
        lines += [f"- `data/{name}`" for name in older]
    return "\n".join(lines) + "\n"


def init(name: str, *, data: Path, inputs: Path, copy: str = "", made_by: str = "", checked: str = "") -> tuple[Path, dict]:
    """Find a video by its name, identify it, and write or refresh its folder's manifest; returns both."""
    ident = cf.identity(cf.find_source(name, inputs))
    folder = cf.folder_for(name, data, ident)
    manifest = cf.read_manifest(folder) if (folder / cf.MANIFEST).is_file() else {"masks": []}
    manifest["source"] = ident
    if copy:
        if copy == name:
            raise cf.Refused(f"the copy is named {copy!r}, as the input is: a copy made from the input is another file")
        manifest["derived"] = {**cf.identity(cf.find_source(copy, inputs)),
                               "made_by": made_by or (manifest.get("derived") or {}).get("made_by") or NOT_STATED,
                               "checked": checked or (manifest.get("derived") or {}).get("checked") or NOT_STATED}
    manifest = cf.write_manifest(folder, manifest)
    (folder / "README.md").write_text(readme(manifest, older_folders(name, data)))
    return folder, manifest


def frames_of_table(path: Path) -> list[int]:
    """[first, last] source frame a per-frame voice table has a row for, read from its own `frame` column. Refused
    when it has no row, no such column, or a frame missing between its first and its last: a table is said to
    cover the frames it holds, never the frames somebody meant it to."""
    import csv
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    try:
        held = sorted(int(r["frame"]) for r in rows)
    except (KeyError, TypeError, ValueError):
        raise cf.Refused(f"the voice table {path} has no `frame` column of whole numbers") from None
    if not held:
        raise cf.Refused(f"the voice table {path} has no row")
    if held != list(range(held[0], held[-1] + 1)):
        raise cf.Refused(f"the voice table {path} has {len(held)} rows for frames {held[0]} to {held[-1]}: a frame is missing or given twice")
    return [held[0], held[-1]]


def voice(name: str, *, data: Path, per_frame: str, spans: str, by: str) -> tuple[Path, dict]:
    """Name a video's voice tables in its manifest, with the frames the per-frame table covers (read from the table,
    never typed). Both must be files under the repo; they are named from its root."""
    folder = cf.folder_of_file(name, data)
    named = {}
    for key, given in (("per_frame", per_frame), ("spans", spans)):
        path = Path(given) if Path(given).is_absolute() else REPO / given
        if not path.is_file():
            raise cf.Refused(f"the voice table {given!r} is not a file")
        if not path.resolve().is_relative_to(REPO.resolve()):
            raise cf.Refused(f"the voice table {given!r} is not under the repo: a video's tables are named from the repo's root")
        named[key] = str(path.resolve().relative_to(REPO.resolve()))
    import datetime
    manifest = cf.read_manifest(folder)
    manifest["voice"] = {**named, "frames": frames_of_table(REPO / named["per_frame"]), "frames_are": "the per-frame table's own first and last frame",
                         "named_by": by, "named_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return folder, cf.write_manifest(folder, manifest)


def voice_tables(source_name: str, data: Path) -> dict:
    """A video's voice tables as its manifest names them: {"per_frame", "spans"}, each a path that is a file, and
    `frames`, the [first, last] source frame the per-frame table covers, read from the table now. Refused when the
    manifest names none, with the command that names them; when one is gone; and when the table no longer covers
    the frames the manifest says it did when it was named."""
    folder = cf.folder_of_file(source_name, data)
    named = cf.read_manifest(folder).get("voice")
    if not named or not named.get("per_frame") or not named.get("spans"):
        raise cf.Refused(f"{folder.name}'s manifest names no voice tables: run `bench/clip_analysis.py voice {source_name} --per-frame ... "
                         f"--spans ... --by WHO` once for the video")
    out = {}
    for key in ("per_frame", "spans"):
        path = REPO / named[key]
        if not path.is_file():
            raise cf.Refused(f"{folder.name}'s manifest names the voice table {named[key]!r}, which is not a file")
        out[key] = path
    out["frames"] = frames_of_table(out["per_frame"])
    if named.get("frames") is not None and list(named["frames"]) != out["frames"]:
        raise cf.Refused(f"{folder.name}'s manifest says its voice table covered frames {named['frames'][0]}-{named['frames'][1]} when it was named, and "
                         f"{named['per_frame']} now covers {out['frames'][0]}-{out['frames'][1]}: it was changed since. Name it again (`clip_analysis.py voice`)")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("init", help="identify an input video and make its folder; no server, no card")
    i.add_argument("name", help="the input video's FILE NAME, as it is under the server's input folder")
    i.add_argument("--copy", default="", help="the file name of the every-frame copy jobs render from")
    i.add_argument("--made-by", default="", help="how the copy was made")
    i.add_argument("--checked", default="", help="what the copy's frame-for-frame check against the input said")
    v = sub.add_parser("voice", help="name the video's voice tables in its manifest; no server, no card")
    v.add_argument("name", help="the input video's file name, or its copy's")
    v.add_argument("--per-frame", required=True, help="the per-frame voice table a capture takes with --voice")
    v.add_argument("--spans", required=True, help="the voice spans file a capture's preflight takes with --voice-spans")
    v.add_argument("--by", required=True, help="who names them")
    a = p.parse_args()
    try:
        if a.command == "voice":
            folder, manifest = voice(a.name, data=REPO / "data", per_frame=a.per_frame, spans=a.spans, by=a.by)
            print(f"{folder}\n  voice tables: {manifest['voice']['per_frame']} and {manifest['voice']['spans']}, named by {manifest['voice']['named_by']}")
            return 0
        folder, manifest = init(a.name, data=REPO / "data", inputs=masked_job.input_dir(), copy=a.copy,
                                made_by=a.made_by, checked=a.checked)
    except cf.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return REFUSED
    source, derived = manifest["source"], manifest.get("derived")
    print(f"{folder}\n  {source['name']}: {source['frames']} frames at {source['rate']:g} fps, {source['size'][0]}x{source['size'][1]}, "
          f"{source['bytes']} bytes")
    if derived:
        print(f"  jobs render from {derived['name']}: {derived['frames']} frames at {derived['rate']:g} fps; made by: "
              f"{derived['made_by']}; checked: {derived['checked']}")
    print(f"  {len(manifest['masks'])} mask(s) in the table; fingerprint {manifest['fingerprint'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
