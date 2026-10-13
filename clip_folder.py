"""The analysis folder of an input video: where it is, whose it is, and whether it is still that video's.

One folder holds everything measured or decided about one input video before anything renders from it:
`data/<stem>/` under this pack, `<stem>` the video's FILE NAME without its extension. The owner's rule
(2026-10-10): a rule that depends on the picture takes its value from this clip, measured up front, in a
workflow of its own that runs first; its data sits in the data folder named by the input video, by file name
and never by path; and the renders read from it. This module is the part of that every reader shares: the
bench tools that write the folder and the ones that read it import the same functions, so "which folder",
"is it this file's" and "has anything a render reads changed" have one answer each. (No node imports it: a
job's graph is built by `bench/masked_job.py`, which does. This sentence said "the node that reads it" until
2026-10-13.)

Nothing here imports ComfyUI or torch, and nothing here writes outside the folder it is given.

**The manifest**, `data/<stem>/manifest.json`:

    {"schema": "h3_clip_analysis", "version": 1,
     "source":  {"name", "bytes", "frames", "rate", "size": [w, h], "head_sha256"},
     "derived": {"name", ..., "made_by": how, "checked": what the frame-for-frame check said} or null,
     "copies_before": [{"name", ..., "first_source_frame", "replaced_by", "replaced_at", "why", ...}],
     "masks":   [{"subject", "kind", "file", "size": [w, h], "first_source_frame", "frames", "sha256", "by"}],
     "fingerprint": "..."}

`source` is the input video as it was when the folder was made, by name, size, frame count, rate and a hash
of its first bytes, and never by a path: a file is found again by its name (`find_source`) and told apart by
the rest (`verify`). `derived` is the every-frame copy at the model's rate that a job renders from, the same
way: it is a file made from the input, not a second input. `masks` is every mask video in the folder with
the size it has and the source frame its first frame is on, so that a reader need not trust a file's name.
`fingerprint` is one hash over everything a render reads from the folder (`fingerprint`); it changes when
any of that changes and for no other reason, and a render carries it in its own keys. The value in the file
is the one worked out when the manifest was last written; `profile.json` is hashed into it and is written by
others, so the file's value can be behind. `read_manifest` therefore returns the fingerprint
as it is NOW, and a reader takes it from there and never from the file.

**One clock.** Every frame number in the folder (a mask's `first_source_frame`, a shot's frames, a load's) is
a frame of the file the masks were made from: the `derived` copy when the manifest has one, else the source.
A job renders from that same file, so a job's frame numbers are the folder's.

**The copies a video has had** (`copies_before`, 2026-10-13). A video's working copy is replaced now and then: a
longer one, one with its sound made again. The manifest held one copy's name, so the day a copy was replaced
every job that named the one before it found no folder and could not be built again, though its file was where
it had been. `copies_before` keeps each earlier copy as `derived` kept it (its name and identity, how it was
made, what was checked) with `first_source_frame` (the source frame its frame 0 is: its frame clock),
`replaced_by`, `replaced_at` and `why`. A folder is found by any of those names (`folder_of_file`), and a file
is held to the entry OF ITS OWN NAME (`verify`): a job from an earlier copy builds again and is still refused
if that file's bytes have changed. A copy whose frame 0 is not the present copy's is on another clock
(`same_clock`): its frame numbers are not the folder's, and a reader that carries frame numbers across copies
asks first. The fingerprint does not hold `copies_before`: retiring a copy changes nothing a render from the
present copy reads, and the present copy is in the fingerprint as before.

**Two layers of settings**, never merged in a file. What the analysis decided for a load is in the folder's
`profile.json`, each key with the figure it read and the line it compared with, and `line_is` one of
`LINE_IS`. What a person set by hand for a load of a job is `jobs/<job>/<load>.hand.json`
(`HAND_SCHEMA`). A reader resolves hand over analysis over its own default, and says which it took.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
from pathlib import Path

SCHEMA, VERSION = "h3_clip_analysis", 1
HAND_SCHEMA, HAND_VERSION = "h3_load_hand_settings", 1
MANIFEST = "manifest.json"
#: Where a rule's line comes from: measured on this clip, measured on another clip and carried as a starting
#: value, or reasoned. A closed list: a writer and a reader both refuse a fourth.
LINE_IS = ("this clip", "one clip carried", "reasoned")
#: The kinds of mask video a folder holds. `parts_held` is the part with its doubted frames filled and graded
#: by the capture; `held_things` is the mask of what a subject holds, which is not a part.
MASK_KINDS = ("track", "parts", "parts_held", "classes", "held_things", "mesh")
#: How much of a file's head is hashed. Reasoned: enough that two videos of one name and size differ in it,
#: little enough to read from a share every time a render starts.
HEAD_BYTES = 16 * 1024 * 1024
#: The fields of an identity that say "the same file". The name is compared apart: a renamed file is the same
#: bytes and another folder.
SAME_FILE = ("bytes", "frames", "head_sha256")


class Refused(ValueError):
    """The folder is missing, is another file's, or holds something this code does not know how to read."""


def head_hash(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(HEAD_BYTES)).hexdigest()


def probe(path: Path) -> dict:
    """A video's frame count, rate and size, from ffprobe: the frames counted as packets, which reads no picture."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=nb_read_packets,r_frame_rate,width,height", "-of", "json", str(path)],
                         capture_output=True, text=True)
    streams = json.loads(out.stdout or "{}").get("streams") or []
    if out.returncode or not streams:
        raise Refused(f"{Path(path).name} could not be read as a video: {out.stderr.strip()[:300] or 'no video stream'}")
    stream = streams[0]
    over, under = (stream["r_frame_rate"].split("/") + ["1"])[:2]
    return {"frames": int(stream["nb_read_packets"]), "rate": round(int(over) / int(under), 6),
            "size": [int(stream["width"]), int(stream["height"])]}


def identity(path: Path) -> dict:
    """What a file is, with no word of where it is: its name, byte size, frames, rate, size and head hash."""
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, **probe(path), "head_sha256": head_hash(path)}


def same_file(a: dict, b: dict) -> list[str]:
    """The fields in which two identities differ as files; empty when they are the same file."""
    return [k for k in SAME_FILE if a.get(k) != b.get(k)]


def stem(name: str) -> str:
    return Path(name).stem


def copies(manifest: dict) -> list[dict]:
    """Every copy the video has or has had, the present one first: each as its entry, with `present` and
    `first_source_frame` (0 for the present copy unless its entry says otherwise)."""
    out = []
    if manifest.get("derived"):
        out.append({"first_source_frame": 0, **manifest["derived"], "present": True})
    out += [{**c, "present": False} for c in manifest.get("copies_before") or []]
    return out


def copy_named(manifest: dict, name: str) -> dict | None:
    """The entry of the copy of this file name, present or earlier, or None."""
    return next((c for c in copies(manifest) if c["name"] == name), None)


def same_clock(manifest: dict, name: str) -> bool:
    """Whether a file's frame numbers are the folder's: it is the source, or a copy whose frame 0 is the present
    copy's. Refused for a name the manifest does not hold."""
    if name == manifest["source"]["name"]:
        return True
    entry = copy_named(manifest, name)
    if entry is None:
        raise Refused(f"{name!r} is not this video's source or a copy it has had")
    present = next((c for c in copies(manifest) if c["present"]), None)
    return int(entry["first_source_frame"]) == int(present["first_source_frame"] if present else 0)


def retire(manifest: dict, replaced_by: str, why: str, at: str, first_source_frame: int = 0) -> dict:
    """Move the manifest's present copy into `copies_before`, as it stands, with what replaced it and why. The
    caller then sets the new `derived`. Refused with no present copy, with no reason in words, or when a copy of
    that name is already among the earlier ones."""
    present = manifest.get("derived")
    if not present:
        raise Refused("the manifest has no copy to retire")
    if not str(why).strip():
        raise Refused(f"the copy {present['name']!r} is being replaced by {replaced_by!r}: say why, in words; it is kept with the copy")
    if replaced_by == present["name"]:
        raise Refused(f"the copy {present['name']!r} cannot be replaced by a file of its own name")
    return add_copy_before(manifest, {**present, "first_source_frame": int(present.get("first_source_frame", first_source_frame)),
                                      "replaced_by": replaced_by, "replaced_at": at, "why": str(why).strip()}, retiring=True)


def add_copy_before(manifest: dict, entry: dict, retiring: bool = False) -> dict:
    """Enter an earlier copy in the manifest. Refused when its name is the source's, the present copy's (unless it
    is the present copy being retired, `retire`) or one already entered, or when its frames do not lie inside the
    source's."""
    name = entry.get("name")
    if name == manifest["source"]["name"] or any(c["name"] == name for c in manifest.get("copies_before") or []) \
            or (name == (manifest.get("derived") or {}).get("name") and not retiring):
        raise Refused(f"{name!r} is already the source, the present copy or an earlier copy of this video")
    first = int(entry.get("first_source_frame", -1))
    if first < 0 or first + int(entry.get("frames", 0)) > int(manifest["source"]["frames"]):
        raise Refused(f"the copy {name!r} is said to begin on source frame {entry.get('first_source_frame')} and holds {entry.get('frames')} frames: "
                      f"that is not inside the source's {manifest['source']['frames']}")
    manifest["copies_before"] = [*(manifest.get("copies_before") or []), dict(entry)]
    return manifest


def find_source(name: str, input_folder: Path) -> Path:
    """The one file of this name under the server's input folder: directly in it, else anywhere below it."""
    if Path(name).name != name:
        raise Refused(f"{name!r} is a path: a video is named by its file name alone")
    direct = Path(input_folder) / name
    if direct.is_file():
        return direct
    # the name is a name, not a pattern: a bracket in it must not be read as a character class
    found = sorted(p for p in Path(input_folder).rglob(glob.escape(name)) if p.is_file())
    if len(found) == 1:
        return found[0]
    if not found:
        raise Refused(f"no file named {name!r} under the input folder {input_folder}")
    raise Refused(f"{len(found)} files are named {name!r} under {input_folder}: "
                  + ", ".join(str(p.relative_to(input_folder)) for p in found) + ". Keep one, or rename one.")


def read_manifest(folder: Path) -> dict:
    path = Path(folder) / MANIFEST
    if not path.is_file():
        raise Refused(f"{folder} holds no {MANIFEST}: this video has not been analysed. Run "
                      f"bench/clip_analysis.py on it first.")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("version") != VERSION:
        said = (manifest.get("schema"), manifest.get("version")) if isinstance(manifest, dict) else ("not a manifest", None)
        raise Refused(f"{path} is {said[0]!r} version {said[1]!r}; this code reads {SCHEMA!r} version {VERSION}")
    check_manifest(manifest)
    manifest["fingerprint"] = fingerprint(folder, manifest)
    return manifest


def folder_for(name: str, data: Path, ident: dict | None = None) -> Path:
    """The folder of a video named `name`: `data/<stem>/`, or `data/<file name>/` when `<stem>` is another file's.

    Two inputs can share a stem (`fun.mp4`, `fun.mov`). The first to be analysed has `data/<stem>/`; the second
    has the folder named by its whole file name. Which is which is read from each folder's manifest, never
    guessed: with `ident` (the file's identity) the folder returned is the one whose manifest names this file,
    or the first that does not exist yet; without it, the one whose manifest names this file name."""
    short, whole = Path(data) / stem(name), Path(data) / name
    for folder in (short, whole):
        if not (folder / MANIFEST).is_file():
            if ident is not None or folder == whole:
                if ident is None:
                    raise Refused(f"no analysis folder holds a video named {name!r}: neither {short} nor {whole}. "
                                  f"Run bench/clip_analysis.py on it first.")
                return folder
            continue
        source = read_manifest(folder)["source"]
        if source["name"] == name and (ident is None or not same_file(source, ident)):
            return folder
    raise Refused(f"both {short} and {whole} belong to other files than this {name!r}: the folder of a video is named "
                  f"by its file name, and two different files cannot share one")


def folder_of_file(name: str, data: Path) -> Path:
    """The folder whose manifest names a file of this name, as its source, as its derived copy, or as a copy it
    had before (`copies_before`).

    A job is given the file it renders from, which is as a rule the copy and not the input; this is how a job
    finds its video's folder from that file's name alone."""
    held = []
    for path in sorted(Path(data).glob(f"*/{MANIFEST}")):
        # other tools keep a manifest.json of their own under data/, and make and remake their folders while
        # this runs: one that does not parse, or is gone between the listing and the read, is not ours
        try:
            manifest = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        # by the source's name, the present copy's, or the name of any copy the video has had: a job that renders
        # from a copy since replaced is still this video's
        if isinstance(manifest, dict) and manifest.get("schema") == SCHEMA and name in (
                (manifest.get("source") or {}).get("name"), (manifest.get("derived") or {}).get("name"),
                *((c or {}).get("name") for c in manifest.get("copies_before") or [])):
            held.append(path.parent)
    if len(held) == 1:
        return held[0]
    if not held:
        raise Refused(f"no analysis folder under {data} names a file called {name!r}, as its source, as the "
                      f"copy made from it, or as a copy it had before: this video has not been analysed, or this copy "
                      f"was replaced before copies were kept. Run bench/clip_analysis.py init on its input video, or "
                      f"`copy-before` to enter an earlier copy.")
    raise Refused(f"{len(held)} analysis folders name a file called {name!r}: {', '.join(p.name for p in held)}")


def verify(manifest: dict, path: Path, frames: bool = False) -> None:
    """Refuse when `path` is not the file the folder was made from: another file of that name, or the file changed.

    `path` is the source, the derived copy or a copy the video had before, told by its name, and it is held to
    the entry of that name: an earlier copy to its own identity, never to the present copy's. The byte size and
    the head hash are read every time; the frame count, which needs ffprobe, when asked."""
    path = Path(path)
    derived = manifest.get("derived") or {}
    source = copy_named(manifest, path.name) or manifest["source"]
    if path.name != source["name"]:
        before = [c["name"] for c in manifest.get("copies_before") or []]
        raise Refused(f"the folder is {manifest['source']['name']!r}'s"
                      + (f" (its copy is {derived['name']!r}" + (f"; before it: {', '.join(before)}" if before else "") + ")" if derived else "")
                      + f" and the file is named {path.name!r}")
    now = {"bytes": path.stat().st_size, "head_sha256": head_hash(path)}
    if frames:
        now["frames"] = probe(path)["frames"]
    differs = [k for k in now if now[k] != source[k]]
    if differs:
        raise Refused(f"{path.name} is not the file this folder was made from (it differs in "
                      f"{', '.join(differs)}): the analysis is another file's or is stale. Run "
                      f"bench/clip_analysis.py on this file again.")


def check_manifest(manifest: dict) -> None:
    """Refuse a manifest a reader could misread: a source or a copy lacking a field of its identity, an earlier
    copy lacking its identity, its frame clock or what replaced it and why, two files of one name, an unknown mask
    kind, a mask with no size or first frame, a mask named by a path."""
    for block in ("source", "derived"):
        about = manifest.get(block)
        if about is None and block == "derived":
            continue
        lacking = [k for k in ("name", "bytes", "frames", "rate", "size", "head_sha256") if not isinstance(about, dict) or k not in about]
        if lacking:
            raise Refused(f"the manifest's `{block}` lacks {', '.join(lacking)}")
    seen = {manifest["source"].get("name"), (manifest.get("derived") or {}).get("name")}
    for n, copy in enumerate(manifest.get("copies_before") or []):
        lacking = [k for k in ("name", "bytes", "frames", "rate", "size", "head_sha256", "first_source_frame", "replaced_by", "replaced_at", "why")
                   if not isinstance(copy, dict) or k not in copy]
        if lacking:
            raise Refused(f"earlier copy {n} of the manifest lacks {', '.join(lacking)}")
        if copy["name"] in seen:
            raise Refused(f"earlier copy {n} of the manifest is named {copy['name']!r}, as the source, the present copy or another earlier copy is")
        seen.add(copy["name"])
    for n, mask in enumerate(manifest.get("masks") or []):
        lacking = [k for k in ("subject", "kind", "file", "size", "first_source_frame", "frames", "sha256", "by") if k not in mask]
        if lacking:
            raise Refused(f"mask {n} of the manifest lacks {', '.join(lacking)}")
        if mask["kind"] not in MASK_KINDS:
            raise Refused(f"mask {n} of the manifest is of kind {mask['kind']!r}; one of {MASK_KINDS}")
        if Path(mask["file"]).is_absolute() or ".." in Path(mask["file"]).parts:
            raise Refused(f"mask {n} of the manifest names {mask['file']!r}: a mask is named relative to its folder")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(folder: Path, manifest: dict) -> str:
    """One hash over everything a render reads from the folder: the source's identity, the mask table (each
    mask by its own hash) and `profile.json` when it exists.

    It changes when a mask is made again or a rule sets another value, and for no other reason: a record, a
    tile or a report written into the folder does not move it. A hand-set file of a job is a person's and is
    not in it; a reader carries that file's own hash beside this one.

    THE CAST IS NOT IN IT (2026-10-10, after a second session's cold read). Until then this hashed
    `cast/state.json`, a file nothing writes: the cast as built is `cast/cast.jsonl` (`bench/cast_file.py`), so
    a cast answer never moved the fingerprint although this docstring said it did. It is left out and not
    repointed at the real file on purpose: one answer about one shot would then refuse every graph built on
    the clip, for every subject and shot. Whoever wires the cast in decides its grain first."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"schema": SCHEMA, "version": VERSION, "source": manifest["source"],
                              "derived": manifest.get("derived"),
                              "masks": sorted(manifest.get("masks") or [], key=lambda m: (m["subject"], m["kind"], m["file"]))},
                             sort_keys=True).encode())
    path = Path(folder) / "profile.json"
    digest.update(b"\nprofile.json:" + (file_hash(path).encode() if path.is_file() else b"none"))
    return digest.hexdigest()


def write_manifest(folder: Path, manifest: dict) -> dict:
    """Write the folder's manifest with its fingerprint worked out now; returns it as written."""
    manifest = {"schema": SCHEMA, "version": VERSION, **{k: v for k, v in manifest.items() if k not in ("schema", "version", "fingerprint")}}
    manifest.setdefault("derived", None)
    manifest.setdefault("masks", [])
    check_manifest(manifest)
    manifest["fingerprint"] = fingerprint(folder, manifest)
    Path(folder).mkdir(parents=True, exist_ok=True)
    part = Path(folder) / (MANIFEST + ".part")
    part.write_text(json.dumps(manifest, indent=1) + "\n")
    os.replace(part, Path(folder) / MANIFEST)
    return manifest


def read_hand(folder: Path, job: str, load: str) -> tuple[dict, str]:
    """A load's hand-set layer and the hash a render carries for it: `({}, "none")` when there is no such file."""
    for what, name in (("job", job), ("load", load)):
        if not name or name in (".", "..") or any(sep in name for sep in ("/", "\\")):
            raise Refused(f"the {what} is named {name!r}: a name, with no path in it")
    path = Path(folder) / "jobs" / job / f"{load}.hand.json"
    if not path.is_file():
        return {}, "none"
    hand = json.loads(path.read_text())
    if hand.get("schema") != HAND_SCHEMA or hand.get("version") != HAND_VERSION:
        raise Refused(f"{path} is {hand.get('schema')!r} version {hand.get('version')!r}; this code reads "
                      f"{HAND_SCHEMA!r} version {HAND_VERSION}")
    return hand.get("set") or {}, file_hash(path)
