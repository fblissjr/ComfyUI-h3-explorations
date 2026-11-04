#!/usr/bin/env python
"""Controls for the prompt grader's rules, and its copies of the guide's closed sets.

A rule that has never been seen to fail is a requirement, not a control. Three
escapes earned this file (all 2026-10-01, found by reading the code against the
rules it claimed to enforce, not by any check):

1. **The stamped-header FAIL fired on one spelling.** The owner's rule is that a
   shot header carries no timestamp. The test was "the body opens with a capital
   `At` and a digit", so `at 00:05.200,`, `The shot cuts at 00:05.200 to` and a
   malformed `[Shot 2, 00:05.200]` header all passed, the last by emptying the
   shot list and switching off every rule that reads shots. Each spelling is a
   case here, and so are the three legal forms that must stay green: a mid-shot
   time in its own sentence, a clock on a sign in quotes, and a time of day.
2. **The retention-line speaker-id test read only `(S1)`**, so the compound
   `(S1,S2)` passed where a single id failed.
3. **The song node's prompts were not graded at all**, and its audio sections
   were waived because the graph has no `VAEDecodeAudio`. The song graph is
   graded through `bench/preflight_graph.py::grade_template`; each way it can
   fail is mutated in memory here.

The second half pins the closed sets the grader keeps COPIES of against the guide
they come from. They agree today; nothing made them disagree loudly. Sections,
the two marker sets and the task types are compared with
`check_prompt_guide_conformance.parse_guide` (the guide parsed at run time);
amplitude, speed and the cut phrases are parsed from `vendor_guides/base_en.md`
here; the marker strings are compared with the release's declared special tokens.

    CUDA_VISIBLE_DEVICES= python bench/check_prompt_rule_controls.py

No model, no GPU, no server. ComfyUI importable, for the song graph the
generator builds in memory.
"""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bench"))
sys.path.insert(0, str(REPO / "workflows"))

import build_prompt_bank as bank  # noqa: E402
import build_prompt_catalogue as catalogue  # noqa: E402
import check_camera_vocabulary as camera  # noqa: E402
import check_prompt_guide_conformance as conf  # noqa: E402
import grade_prompt_text as gpt  # noqa: E402
import preflight_graph as pf  # noqa: E402
import prompts  # noqa: E402
import vendor_config  # noqa: E402

BASE_GUIDE_TEXT = (REPO / "vendor_guides" / "base_en.md").read_text(encoding="utf-8")

# A two-shot base prompt that grades clean; the cases below change one thing.
BASE = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium shot frames a dock "
        "worker on a ferry deck across the take. The worker, on-screen, in her forties, with a low alto "
        "(S1), says: <d>[English] Last crossing.</d> Her lips close and her jaw stops moving. The camera "
        "holds a static shot. {SHOT2} a close shot of a young man folding a timetable. He produces no "
        "vocal sound.\n\n"
        "overall_soundscape: Diesel rumble under water against the hull. Boots scuff on wet steel.\n\n"
        "non_diegetic_music: N/A")

problems: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        problems.append(f"{label}" + (f": {detail}" if detail else ""))


def fails_of(text: str) -> list[str]:
    r = gpt.grade_text(text, "t2va", None, 243)
    return [msg for level, msg in r["findings"] if level == "FAIL"]


def header_cases() -> None:
    must_fail = {
        "a capital At and a time": "[Shot 2] At 00:05.200, the shot cuts to",
        "a lowercase at and a time": "[Shot 2] at 00:05.200, the shot cuts to",
        "At and a whole number": "[Shot 2] At 5 seconds, the shot cuts to",
        "a time later in the opening sentence": "[Shot 2] The shot cuts at 00:05.200 to",
        "a malformed header with a time inside the brackets": "[Shot 2, 00:05.200] The shot cuts to",
        "a malformed header with a label": "[Shot 2: dock] The shot cuts to",
    }
    for name, shot2 in must_fail.items():
        check(f"header: {name} must FAIL", bool(fails_of(BASE.replace("{SHOT2}", shot2))))
    must_pass = {
        "a header with no time": "[Shot 2] The shot cuts to",
        "a mid-shot time in its own sentence": "[Shot 2] The shot cuts to a medium shot of the dock. At 00:07.500, the camera cuts to",
        "a clock on a sign, in quotes": '[Shot 2] The shot cuts to a sign reading "10:45.30" over the door. The camera holds.',
        "a time of day with no decimals": "[Shot 2] The shot cuts at 4:30 in the afternoon to",
    }
    for name, shot2 in must_pass.items():
        fails = fails_of(BASE.replace("{SHOT2}", shot2))
        check(f"header: {name} must pass", not fails, "; ".join(fails)[:120])
    # the pure function agrees with the grader it feeds
    check("header: header_problems is empty on a clean body", pf.header_problems("[Shot 1] A. [Shot 2] The shot cuts to B.") == [])


def speaker_cases() -> None:
    check("speaker: compound ids count each speaker", pf.speaker_numbers("(S1,S2) say, then (S3) and (S1)") == [1, 2, 3])
    check("speaker: the pattern reads a compound id", bool(pf.SPEAKER_ID.search("x (S1,S2) y")))
    # a ref2va bank entry, graded at its declared donor and length, with an id planted in retention_analysis
    entry = prompts.entry("ref2va_image_ref_default")
    text = prompts.text("ref2va_image_ref_default")
    mode, like, frames = entry["mode"], entry.get("donor"), int(entry["frames"])
    clean = [m for lv, m in gpt.grade_text(text, mode, like, frames)["findings"] if lv == "FAIL"]
    check("retention: the unmodified ref2va entry grades without a FAIL", not clean, "; ".join(clean)[:120])
    lines = text.split("\n")
    k = next(i for i, l in enumerate(lines) if l.startswith("retention_analysis:")) + 1
    for planted, name in (("(S1)", "a single id"), ("(S1,S2)", "a compound id")):
        mutated = lines[:k] + [lines[k].replace(" - ", f" {planted} - ", 1) if " - " in lines[k] else lines[k] + f" {planted}"] + lines[k + 1:]
        got = [m for lv, m in gpt.grade_text("\n".join(mutated), mode, like, frames)["findings"] if lv == "FAIL"]
        check(f"retention: {name} in retention_analysis must FAIL", any("retention_analysis" in m for m in got))


def _song_lists_graph() -> dict:
    """The prompt-list song graph, built in memory by the generator.

    Until 2026-11-04 this read `h3_text_to_video_audio_freeze_song_lists_pdd8`,
    the one shipped graph with prompt lists, which was retired with PDD's
    primary use. The generator's own constants and `build_api` make the same
    song and list nodes (only the output prefix differs), on the base chain, so
    the fixture cannot drift from what a lists graph would ship. Needs ComfyUI
    importable, as `check_generator_constants.py` does, with the same import
    order: core before the generator, on its CPU path when the card is masked.
    """
    sys.path.insert(0, str(REPO.parent.parent))  # ComfyUI root
    import torch
    if not torch.cuda.is_available():
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
    import comfy_extras.nodes_minimax_h3  # noqa: F401
    import build_workflows as bw
    return bw.build_api("t2v", prompt=bw._bank_prompt("t2va_song_flicker_lists"),
                        freeze_song=True, freeze_song_seconds=None,
                        freeze_song_timeline=bw._SONG_FLICKER_TIMELINE,
                        freeze_song_lists=bw._SONG_FLICKER_LISTS, freeze_track="just-a-flicker.mp3",
                        freeze_mask=0.25, freeze_context=39, length=bw.LONG_WINDOW)


def song_cases() -> None:
    graph = _song_lists_graph()
    car = prompts.carriers(graph)
    check("song: the song node is read as a template carrier", len(car) == 1 and car[0].template and len(car[0].texts) > 1)
    if not car:
        return
    node_id = car[0].node_id

    def graded(mutator):
        g = copy.deepcopy(graph)
        g[node_id]["inputs"]["prompt"] = mutator(g[node_id]["inputs"]["prompt"])
        lines = pf.grade_template(prompts.carriers(g)[0], g, "control")
        return [l for l in lines if "FAIL" in l]

    check("song: the shipped prompt grades clean", not graded(lambda t: t))
    check("song: dropping the audio sections must FAIL (a song graph has no VAEDecodeAudio)",
          bool(graded(lambda t: t.split("\n\noverall_soundscape:")[0])))
    check("song: a stamped header must FAIL", bool(graded(lambda t: t.replace("[Shot 2] The shot", "[Shot 2] At 00:05.200, the shot", 1))))
    check("song: a placeholder with no list must FAIL", bool(graded(lambda t: t.replace("__place__", "__ghost__", 1))))
    check("song: a list no text uses must FAIL", bool(graded(lambda t: t.replace("__motion__", "slowly"))))


def closed_set_pins() -> None:
    ref = conf.parse_guide(conf.GUIDE.read_text(encoding="utf-8"))
    base_sections = conf.parse_base_guide(conf.BASE_GUIDE.read_text(encoding="utf-8"))
    check("pin: preflight REF_SECTIONS is the reference guide's sections, in order", list(pf.REF_SECTIONS) == list(ref["sections"]),
          f"{pf.REF_SECTIONS} against {ref['sections']}")
    check("pin: preflight BASE_SECTIONS is the base guide's core fields, in order", list(pf.BASE_SECTIONS) == list(base_sections),
          f"{pf.BASE_SECTIONS} against {base_sections}")
    check("pin: preflight VISUAL_MARKERS is the guide's visual set", set(pf.VISUAL_MARKERS) == set(ref["visual"]))
    check("pin: preflight AUDIO_MARKERS is the guide's audio set", set(pf.AUDIO_MARKERS) == set(ref["audio"]))
    check("pin: the bank's TASK_TYPES is the guide's six, in order", tuple(bank.TASK_TYPES) == tuple(ref["task_types"]),
          f"{bank.TASK_TYPES} against {tuple(ref['task_types'])}")

    amp = tuple(re.findall(r"\|\s*Amplitude\s*\|\s*`([^`]+)`", BASE_GUIDE_TEXT))
    spd = tuple(re.findall(r"\|\s*Speed\s*\|\s*`([^`]+)`", BASE_GUIDE_TEXT))
    check("pin: the guide's amplitude and speed rows parsed", len(amp) == 2 and len(spd) == 2, f"{amp} {spd}")
    check("pin: check_camera_vocabulary AMPLITUDE is the guide's", tuple(camera.AMPLITUDE) == amp, f"{camera.AMPLITUDE} against {amp}")
    check("pin: check_camera_vocabulary SPEED is the guide's", tuple(camera.SPEED) == spd, f"{camera.SPEED} against {spd}")
    check("pin: the bank's MODIFIERS is the guide's amplitude then speed", tuple(bank.MODIFIERS) == amp + spd)

    sentence = next(l for l in BASE_GUIDE_TEXT.splitlines() if l.startswith("For ordinary cuts, use"))
    head, _, tail = sentence.partition("When explicitly requested")
    cuts = tuple(re.findall(r"`([^`]+)`", head))
    listed = re.search(r"user,\s*(.*?)\s+may also be used", tail)
    check("pin: the guide's on-request transitions parsed", listed is not None)
    requested = tuple(w.strip() for w in re.split(r",|\bor\b", listed.group(1) if listed else "") if w.strip())
    check("pin: the bank's CUTS is the guide's five cut phrases", tuple(bank.CUTS) == cuts, f"{bank.CUTS} against {cuts}")
    check("pin: the bank's REQUESTED is the guide's on-request transitions", tuple(bank.REQUESTED) == requested,
          f"{bank.REQUESTED} against {requested}")

    declared = set(vendor_config.additional_special_tokens())
    check("pin: every marker preflight reads is a token the release declares", set(pf.ALL_MARKERS) <= declared,
          f"{sorted(set(pf.ALL_MARKERS) - declared)}")
    extra = {m for m in catalogue.MARKERS if m not in declared} - {"<scenetrans>"}
    check("pin: every marker the catalogue records is declared (bar the guide's `<scenetrans>`, which the release does not)",
          not extra, f"{sorted(extra)}")


def main() -> int:
    header_cases()
    speaker_cases()
    song_cases()
    closed_set_pins()
    if problems:
        print(f"  FAIL  {len(problems)} problem(s):")
        for p in problems:
            print(f"    - {p}")
        return 1
    print("  ok    every stamped-header spelling fails and the three legal forms pass; a compound id "
          "in retention_analysis fails; a song graph's prompts are graded and each mutation fails; the "
          "grader's copies of the guide's sets match the guide")
    return 0


if __name__ == "__main__":
    sys.exit(main())
