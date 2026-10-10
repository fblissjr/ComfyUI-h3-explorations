#!/usr/bin/env python3
"""Check the analysis folder's shared rules: which folder is a video's, and whether it still is.

`clip_folder.py` is read by the tools that write an input video's analysis folder and by the node that renders
from it. If its answers drift, a render takes another video's masks, or an old analysis's, and nothing else
in the chain can see it: the masks are the right size and the graph validates.

Claims, i.e. what breaks if a case is deleted:

  an identity names a file and no place    name, byte size, frames, rate, size and head hash of a made
                        clip, the same from another folder, and no field that holds a path
  a video is found by its name    directly in the input folder, else below it; a name that is a path, no
                        file and two files of one name are each refused, the two named
  the folder is named by the file name    `data/<stem>/`; a second file of the same stem and another
                        extension gets `data/<file name>/`; the first file is still found in its own; a third
                        claimant is refused; a folder is looked up without the file only by a name it holds
  a folder that is another file's or stale is refused    the same name with other bytes, a changed head, and
                        with `frames` a changed frame count; the file it was made from passes; the copy a
                        job renders from finds the folder by its own name and is held to its own identity
  the fingerprint moves with what a render reads, and with nothing else    a mask made again, a changed
                        profile, a changed cast state each move it; a tile or a record written into the
                        folder, and the order masks are listed in, do not
  a manifest a reader could misread is refused    an unknown mask kind, a mask with no first frame, a mask
                        named by a path, an unknown schema or version
  the hand layer is apart    no file is no layer and the word "none"; a file is its `set` and its own hash;
                        an unknown schema is refused; and `LINE_IS` is the three values agreed and no more

No card, no server, no ComfyUI. It makes two small clips with ffmpeg.

    <python> bench/check_clip_folder.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import REPO, case, finish, skip  # noqa: E402

sys.path.insert(0, str(REPO))
import clip_folder as cf  # noqa: E402


def _clip(path: Path, frames: int = 24, level: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{level:02x}{level:02x}{level:02x}:s=32x16:r=24",
                    "-frames:v", str(frames), "-c:v", "ffv1", str(path)], check=True)
    return path


def _refused(fn, *words: str) -> None:
    try:
        fn()
    except cf.Refused as e:
        lacking = [w for w in words if w not in str(e)]
        assert not lacking, f"refused without saying {lacking}: {e}"
        return
    raise AssertionError("was not refused")


def _tmp():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        skip("no ffmpeg or ffprobe on the path")
    return tempfile.TemporaryDirectory()


def an_identity_names_a_file_and_no_place():
    with _tmp() as t:
        tmp = Path(t)
        one = cf.identity(_clip(tmp / "here" / "fun.mkv", frames=30))
        assert (one["name"], one["frames"], one["rate"], one["size"]) == ("fun.mkv", 30, 24.0, [32, 16]), one
        assert one["bytes"] == (tmp / "here" / "fun.mkv").stat().st_size and len(one["head_sha256"]) == 64, one
        shutil.copy(tmp / "here" / "fun.mkv", tmp / "fun.mkv")
        assert cf.identity(tmp / "fun.mkv") == one, "the same file in another folder has another identity"
        assert not any(str(tmp) in str(v) or "/" in str(v) for v in one.values()), f"an identity holds a path: {one}"
        other = cf.identity(_clip(tmp / "there" / "fun.mkv", frames=31))
        assert set(cf.same_file(one, other)) >= {"frames"} and cf.same_file(one, dict(one, name="x.mkv")) == [], \
            "a file is told apart by something other than its bytes, frames and head"


def a_video_is_found_by_its_name():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for path in ("fun.mp4", "a/deep/song.mov", "a/twice.mp4", "b/twice.mp4"):
            (tmp / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp / path).write_bytes(b"x")
        assert cf.find_source("fun.mp4", tmp) == tmp / "fun.mp4"
        assert cf.find_source("song.mov", tmp) == tmp / "a/deep/song.mov", "a file below the input folder is not found by its name"
        _refused(lambda: cf.find_source("a/deep/song.mov", tmp), "path")
        _refused(lambda: cf.find_source("absent.mp4", tmp), "absent.mp4")
        _refused(lambda: cf.find_source("twice.mp4", tmp), "2 files", "a/twice.mp4", "b/twice.mp4")
        # a name is a name: brackets in it are not a pattern (found by the node's author reading this cold)
        (tmp / "a" / "clip [final].mp4").write_bytes(b"x")
        (tmp / "a" / "clip f.mp4").write_bytes(b"x")
        assert cf.find_source("clip [final].mp4", tmp) == tmp / "a" / "clip [final].mp4", "a bracketed name below the input folder"


def the_folder_is_named_by_the_file_name():
    with _tmp() as t:
        tmp = Path(t)
        data = tmp / "data"
        first, second, third = (cf.identity(_clip(tmp / "in" / name, frames=n)) for name, n in
                                (("fun.mp4", 24), ("fun.mov", 25), ("fun.mkv", 26)))
        home = cf.folder_for("fun.mp4", data, first)
        assert home == data / "fun", f"the first file of a stem goes to {home}"
        cf.write_manifest(home, {"source": first})
        assert cf.folder_for("fun.mp4", data, first) == home and cf.folder_for("fun.mp4", data) == home, "a file is not found in its own folder"
        other = cf.folder_for("fun.mov", data, second)
        assert other == data / "fun.mov", f"a second file of the stem goes to {other}"
        cf.write_manifest(other, {"source": second})
        assert cf.folder_for("fun.mov", data, second) == other and cf.folder_for("fun.mov", data) == other
        assert cf.folder_for("fun.mp4", data, first) == home, "the first file lost its folder to the second"
        # the same name with other bytes is another file: its stem's folder is taken, so it gets its whole name
        changed = cf.identity(_clip(tmp / "in2" / "fun.mp4", frames=40))
        assert cf.folder_for("fun.mp4", data, changed) == data / "fun.mp4", "a changed file is handed the old file's folder"
        cf.write_manifest(data / "fun.mp4", {"source": changed})
        again = cf.identity(_clip(tmp / "in3" / "fun.mp4", frames=41))
        _refused(lambda: cf.folder_for("fun.mp4", data, again), "other files")
        _refused(lambda: cf.folder_for("never.mp4", data), "never.mp4", "clip_analysis")
        assert cf.folder_for("fun.mkv", data, third) == data / "fun.mkv", "a third extension of the stem"


def a_folder_that_is_another_files_or_stale_is_refused():
    with _tmp() as t:
        tmp = Path(t)
        clip = _clip(tmp / "in" / "fun.mkv", frames=24)
        manifest = cf.write_manifest(tmp / "data" / "fun", {"source": cf.identity(clip)})
        assert cf.read_manifest(tmp / "data" / "fun") == manifest
        cf.verify(manifest, clip)
        cf.verify(manifest, clip, frames=True)
        _refused(lambda: cf.verify(manifest, _clip(tmp / "other" / "fun.mkv", frames=24, level=200)), "not the file", "head_sha256")
        _refused(lambda: cf.verify(manifest, _clip(tmp / "longer" / "fun.mkv", frames=48)), "not the file")
        _refused(lambda: cf.verify(manifest, _clip(tmp / "in" / "sun.mkv", frames=24)), "sun.mkv")
        # a file with the same size and head and another frame count is told only when the frames are counted
        liar = dict(manifest, source=dict(manifest["source"], frames=23))
        cf.verify(liar, clip)
        _refused(lambda: cf.verify(liar, clip, frames=True), "frames")
        _refused(lambda: cf.read_manifest(tmp / "data" / "nowhere"), "not been analysed", "clip_analysis")
        # the copy a job renders from is the folder's too: found by its name, and held to its own identity
        copy = _clip(tmp / "in" / "fun_24p.mkv", frames=25, level=90)
        both = cf.write_manifest(tmp / "data" / "fun", {"source": cf.identity(clip), "derived": dict(cf.identity(copy), made_by="a recipe", checked="frame for frame")})
        assert cf.folder_of_file("fun_24p.mkv", tmp / "data") == tmp / "data" / "fun" == cf.folder_of_file("fun.mkv", tmp / "data"), \
            "a folder is not found from its copy's name"
        cf.verify(both, copy, frames=True)
        cf.verify(both, clip)
        _refused(lambda: cf.verify(both, _clip(tmp / "elsewhere" / "fun_24p.mkv", frames=25, level=10)), "not the file")
        _refused(lambda: cf.folder_of_file("never_24p.mkv", tmp / "data"), "never_24p.mkv", "not been analysed")
        cf.write_manifest(tmp / "data" / "twin", {"source": dict(cf.identity(clip), name="twin.mkv"), "derived": dict(cf.identity(copy))})
        _refused(lambda: cf.folder_of_file("fun_24p.mkv", tmp / "data"), "2 analysis folders", "fun", "twin")


def _mask(folder: Path, name: str, body: bytes, **more) -> dict:
    (folder / name).parent.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(body)
    return {"subject": "a", "kind": "track", "file": name, "size": [32, 16], "first_source_frame": 0, "frames": 24,
            "sha256": cf.file_hash(folder / name), "by": "run1", **more}


def the_fingerprint_moves_with_what_a_render_reads_and_with_nothing_else():
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t) / "fun"
        source = {"name": "fun.mp4", "bytes": 1, "frames": 24, "rate": 24.0, "size": [32, 16], "head_sha256": "0" * 64}
        masks = [_mask(folder, "analysis/run1/a_track.mkv", b"one"), _mask(folder, "analysis/run1/b_track.mkv", b"two", subject="b")]
        was = cf.write_manifest(folder, {"source": source, "masks": masks})["fingerprint"]
        assert cf.write_manifest(folder, {"source": source, "masks": masks[::-1]})["fingerprint"] == was, "the order masks are listed in moves it"
        (folder / "analysis/run1/tile.png").write_bytes(b"a picture")
        (folder / "jobs/j").mkdir(parents=True)
        (folder / "jobs/j/load.hand.json").write_text("{}")
        assert cf.write_manifest(folder, {"source": source, "masks": masks})["fingerprint"] == was, "a tile or a job's file moves it"
        moved = {}
        moved["a mask made again"] = cf.write_manifest(folder, {"source": source, "masks": [_mask(folder, "analysis/run1/a_track.mkv", b"ONE"), masks[1]]})["fingerprint"]
        cf.write_manifest(folder, {"source": source, "masks": [_mask(folder, "analysis/run1/a_track.mkv", b"one"), masks[1]]})
        moved["a mask on another first frame"] = cf.write_manifest(folder, {"source": source, "masks": [dict(masks[0], first_source_frame=5), masks[1]]})["fingerprint"]
        (folder / "profile.json").write_text('{"loads": {}}')
        moved["a profile"] = cf.write_manifest(folder, {"source": source, "masks": masks})["fingerprint"]
        (folder / "profile.json").write_text('{"loads": {"x": {}}}')
        moved["a changed profile"] = cf.write_manifest(folder, {"source": source, "masks": masks})["fingerprint"]
        (folder / "cast").mkdir()
        (folder / "cast/state.json").write_text('{"revision": 1}')
        moved["a cast"] = cf.write_manifest(folder, {"source": source, "masks": masks})["fingerprint"]
        moved["another source"] = cf.write_manifest(folder, {"source": dict(source, frames=25), "masks": masks})["fingerprint"]
        same = [why for why, value in moved.items() if value == was]
        assert not same and len(set(moved.values())) == len(moved), f"the fingerprint did not move for: {same or 'two different changes'}"
        assert cf.read_manifest(folder)["fingerprint"] == moved["another source"], "the manifest on disk does not carry the fingerprint written"
        # the file's value falls behind when somebody else writes the profile; a reader is given the value as it is now
        (folder / "profile.json").write_text('{"loads": {"y": {}}}')
        on_disk = json.loads((folder / cf.MANIFEST).read_text())["fingerprint"]
        now = cf.read_manifest(folder)["fingerprint"]
        assert on_disk == moved["another source"] and now != on_disk and now == cf.fingerprint(folder, cf.read_manifest(folder)), \
            "a reader is handed the fingerprint of the last write, not of what is in the folder"


def a_manifest_a_reader_could_misread_is_refused():
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t) / "fun"
        source = {"name": "fun.mp4", "bytes": 1, "frames": 24, "rate": 24.0, "size": [32, 16], "head_sha256": "0" * 64}
        good = _mask(folder, "a_track.mkv", b"one")
        _refused(lambda: cf.write_manifest(folder, {"source": source, "masks": [dict(good, kind="held")]}), "kind", "'held'")
        _refused(lambda: cf.write_manifest(folder, {"source": source, "masks": [{k: v for k, v in good.items() if k != "first_source_frame"}]}), "first_source_frame")
        _refused(lambda: cf.write_manifest(folder, {"source": source, "masks": [dict(good, file="/somewhere/a_track.mkv")]}), "relative")
        _refused(lambda: cf.write_manifest(folder, {"source": source, "masks": [dict(good, file="../a_track.mkv")]}), "relative")
        assert not (folder / cf.MANIFEST).exists(), "a refused manifest was written"
        cf.write_manifest(folder, {"source": source, "masks": [good]})
        (folder / cf.MANIFEST).write_text(json.dumps({"schema": cf.SCHEMA, "version": cf.VERSION + 1}))
        _refused(lambda: cf.read_manifest(folder), "version")
        (folder / cf.MANIFEST).write_text(json.dumps({"schema": "something else", "version": cf.VERSION}))
        _refused(lambda: cf.read_manifest(folder), "something else")
        # what the writer refuses the reader refuses too: a manifest somebody else wrote is not taken on trust
        written = cf.write_manifest(folder, {"source": source, "masks": [good]})
        for label, wrong, word in (("a mask of an unknown kind", {**written, "masks": [dict(good, kind="nonsense")]}, "nonsense"),
                                   ("a mask named by a path", {**written, "masks": [dict(good, file="../x.mkv")]}, "relative"),
                                   ("a source with no frame count", {**written, "source": {k: v for k, v in source.items() if k != "frames"}}, "frames"),
                                   ("a list", [1, 2], "not a manifest")):
            (folder / cf.MANIFEST).write_text(json.dumps(wrong))
            _refused(lambda: cf.read_manifest(folder), word)
        # and another tool's manifest.json under data/, of any shape, is not ours and stops nothing
        (folder / cf.MANIFEST).write_text(json.dumps(written))
        for name, body in (("a_capture", '{"name": "x", "frames": 3}'), ("a_list", "[1, 2]"), ("broken", "{")):
            (folder.parent / name).mkdir()
            (folder.parent / name / cf.MANIFEST).write_text(body)
        assert cf.folder_of_file("fun.mp4", folder.parent) == folder, "another tool's manifest stops a folder being found"


def the_hand_layer_is_apart():
    assert cf.LINE_IS == ("this clip", "one clip carried", "reasoned"), f"the closed list is {cf.LINE_IS}"
    with tempfile.TemporaryDirectory() as t:
        folder = Path(t) / "fun"
        assert cf.read_hand(folder, "job", "load") == ({}, "none"), "no hand file is not an empty layer and the word none"
        path = folder / "jobs" / "job" / "load.hand.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema": cf.HAND_SCHEMA, "version": cf.HAND_VERSION,
                                    "set": {"grow_pixels": {"value": 32, "by": "somebody", "why": "a close shot"}}}))
        hand, key = cf.read_hand(folder, "job", "load")
        assert hand == {"grow_pixels": {"value": 32, "by": "somebody", "why": "a close shot"}} and key == cf.file_hash(path), (hand, key)
        path.write_text(json.dumps({"schema": cf.HAND_SCHEMA, "version": cf.HAND_VERSION + 1, "set": {}}))
        _refused(lambda: cf.read_hand(folder, "job", "load"), "version")
        # a job or a load is a name typed into a node: it does not lead out of the folder
        outside = Path(t) / "outside" / "x.hand.json"
        outside.parent.mkdir()
        outside.write_text(json.dumps({"schema": cf.HAND_SCHEMA, "version": cf.HAND_VERSION, "set": {"grow_pixels": {"value": 1}}}))
        for job, load in (("../../outside", "x"), ("job", "../../../outside/x"), ("", "load"), ("job", ".."), ("a/b", "load")):
            _refused(lambda: cf.read_hand(folder, job, load), "no path")


def main() -> int:
    for fn in (an_identity_names_a_file_and_no_place, a_video_is_found_by_its_name, the_folder_is_named_by_the_file_name,
               a_folder_that_is_another_files_or_stale_is_refused,
               the_fingerprint_moves_with_what_a_render_reads_and_with_nothing_else,
               a_manifest_a_reader_could_misread_is_refused, the_hand_layer_is_apart):
        case(fn.__name__.replace("_", " "), fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
