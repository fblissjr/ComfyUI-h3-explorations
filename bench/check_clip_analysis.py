#!/usr/bin/env python3
"""Check the tool that makes an input video's analysis folder.

`bench/clip_analysis.py` writes `data/<stem>/`; `bench/check_clip_folder.py` holds the rules of that folder,
and this file holds what the tool does with them.

Claims, i.e. what breaks if a case is deleted:

  init makes the folder by the file name    a video found below the input folder gets `data/<stem>/`, a
                        manifest with its identity and the copy's under `derived`, and no path of the input
                        folder anywhere in the manifest or the README; a copy whose making was not stated
                        says so in words
  init again keeps what a render reads    the same file again leaves the mask table and the fingerprint as
                        they were; what was stated of the copy before is kept; a file of the same name with
                        other bytes gets its own folder and the first folder is still the first file's
  the README names earlier work    the dated folders under `data/` of the same stem are listed by name, and
                        folders of another stem are not
  what init refuses    a name that is a path, a video that is not there, a copy that is not there, a copy
                        named as the input

No card, no server, no ComfyUI. It makes small clips with ffmpeg.

    <python> bench/check_clip_analysis.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _lib import case, finish, skip  # noqa: E402

import clip_analysis as tool  # noqa: E402
import clip_folder as cf  # noqa: E402


def _clip(path: Path, frames: int = 24, level: int = 40) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=0x{level:02x}{level:02x}{level:02x}:s=32x16:r=24",
                    "-frames:v", str(frames), "-c:v", "ffv1", str(path)], check=True)
    return path


def _tmp():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        skip("no ffmpeg or ffprobe on the path")
    return tempfile.TemporaryDirectory()


def _refused(fn, *words: str) -> None:
    try:
        fn()
    except cf.Refused as e:
        lacking = [w for w in words if w not in str(e)]
        assert not lacking, f"refused without saying {lacking}: {e}"
        return
    raise AssertionError("was not refused")


def init_makes_the_folder_by_the_file_name():
    with _tmp() as t:
        tmp = Path(t)
        where = dict(data=tmp / "data", inputs=tmp / "input")
        video, copy = _clip(tmp / "input" / "fun.mkv", frames=25), _clip(tmp / "input" / "session" / "fun_24p.mkv", frames=24, level=90)
        folder, manifest = tool.init("fun.mkv", copy="fun_24p.mkv", **where)
        assert folder == tmp / "data" / "fun", folder
        assert manifest["source"] == cf.identity(video) and manifest["source"]["frames"] == 25, manifest["source"]
        derived = manifest["derived"]
        assert {k: derived[k] for k in manifest["source"]} == cf.identity(copy), derived
        assert derived["made_by"] == derived["checked"] == tool.NOT_STATED, "a copy whose making nobody stated does not say so"
        assert cf.read_manifest(folder) == manifest and manifest["masks"] == [], "the manifest on disk is not the one returned"
        written = (folder / cf.MANIFEST).read_text() + (folder / "README.md").read_text()
        assert str(tmp) not in written and "session" not in written, "a path of the input folder is written into the folder"
        assert "fun_24p.mkv" in (folder / "README.md").read_text(), "the README does not say which file the frame numbers are of"
        assert cf.folder_of_file("fun_24p.mkv", tmp / "data") == folder, "the folder is not found from the copy's name"


def init_again_keeps_what_a_render_reads():
    with _tmp() as t:
        tmp = Path(t)
        where = dict(data=tmp / "data", inputs=tmp / "input")
        _clip(tmp / "input" / "fun.mkv", frames=25)
        _clip(tmp / "input" / "fun_24p.mkv", frames=24, level=90)
        folder, first = tool.init("fun.mkv", copy="fun_24p.mkv", made_by="a recipe", checked="every frame equal", **where)
        (folder / "analysis").mkdir()
        (folder / "analysis" / "a_track.mkv").write_bytes(b"a mask")
        mask = {"subject": "a", "kind": "track", "file": "analysis/a_track.mkv", "size": [32, 16], "first_source_frame": 0, "frames": 24,
                "sha256": cf.file_hash(folder / "analysis" / "a_track.mkv"), "by": "run1"}
        held = cf.write_manifest(folder, {**first, "masks": [mask]})
        again_folder, again = tool.init("fun.mkv", **where)
        assert again_folder == folder and again == held, "the same file again changed the manifest"
        assert again["derived"]["made_by"] == "a recipe" and again["masks"] == [mask], "what was stated or gathered before was dropped"
        _, restated = tool.init("fun.mkv", copy="fun_24p.mkv", **where)
        assert restated["derived"]["checked"] == "every frame equal" and restated["fingerprint"] == held["fingerprint"], \
            "naming the copy again lost what its check said, or moved the fingerprint"
        # the same name, other bytes: another file, another folder; the first stays the first file's
        (tmp / "input" / "fun.mkv").unlink()
        _clip(tmp / "input" / "fun.mkv", frames=40, level=200)
        other_folder, other = tool.init("fun.mkv", **where)
        assert other_folder == tmp / "data" / "fun.mkv" and other["masks"] == [] and other["source"]["frames"] == 40, other_folder
        assert cf.read_manifest(folder) == held, "the first file's folder was rewritten for the second file"


def the_readme_names_earlier_work():
    with _tmp() as t:
        tmp = Path(t)
        _clip(tmp / "input" / "fun.mkv")
        for name in ("2026-10-10_fun_kitchen_0604", "2026-10-08_fun_job_inputs", "2026-10-10_funny_other", "2026-10-10_sun_kitchen", "notes_fun_x"):
            (tmp / "data" / name).mkdir(parents=True)
        (tmp / "data" / "2026-10-09_fun_a_file").write_text("not a folder")
        folder, _ = tool.init("fun.mkv", data=tmp / "data", inputs=tmp / "input")
        text = (folder / "README.md").read_text()
        listed = sorted(line.split("`")[1] for line in text.splitlines() if line.startswith("- `data/"))
        assert listed == ["data/2026-10-08_fun_job_inputs", "data/2026-10-10_fun_kitchen_0604"], listed
        assert all((tmp / "data" / name).is_dir() for name in ("2026-10-10_fun_kitchen_0604", "2026-10-08_fun_job_inputs")), "an older folder was moved"


def what_init_refuses():
    with _tmp() as t:
        tmp = Path(t)
        where = dict(data=tmp / "data", inputs=tmp / "input")
        _clip(tmp / "input" / "fun.mkv")
        _refused(lambda: tool.init("input/fun.mkv", **where), "path")
        _refused(lambda: tool.init("absent.mkv", **where), "absent.mkv")
        _refused(lambda: tool.init("fun.mkv", copy="absent_24p.mkv", **where), "absent_24p.mkv")
        _refused(lambda: tool.init("fun.mkv", copy="fun.mkv", **where), "copy")
        assert not (tmp / "data").exists(), "a refused init left a folder behind"


def main() -> int:
    for fn in (init_makes_the_folder_by_the_file_name, init_again_keeps_what_a_render_reads, the_readme_names_earlier_work,
               what_init_refuses):
        case(fn.__name__.replace("_", " "), fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
