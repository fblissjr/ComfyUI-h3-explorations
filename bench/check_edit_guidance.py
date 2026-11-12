#!/usr/bin/env python3
"""Check the edit-guidance nodes: the prompt edit's refusals and the guider's arithmetic.

What it checks, with no model and no card:

- `edit_guidance.apply_edits` places an edit in a `prompt_bank/` prompt and refuses, in its own words, a line
  with no arrow, wording that is absent, wording that occurs more than once, wording inside dialogue, wording
  that carries a tag or header, and a `soundscape:` edit aimed at the description.
- `edit_guidance.EditGuider.predict_noise` returns the original pass at scale 0, the edited pass at scale 1,
  `original + scale * (edited - original)` otherwise, and makes one pass on the original before `first_step`.
- The guider marks the two passes' payloads so they differ, and with `align_positions` moves the shorter
  prompt's media rows to the longer prompt's origin, leaving the text rows and the layout's signature alone.
- `edit_guidance.check_same_but_text` refuses conditioning with no token tags and a pair that differs in
  references.

What it cannot see: what a render looks like, or whether the aligned positions are what the model should be
given. Those take renders.

    <comfy-venv-python> bench/check_edit_guidance.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() not in (REPO_ROOT, BENCH_DIR)]
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(REPO_ROOT))

from bench._lib import bootstrap  # noqa: E402

bootstrap()

# CPU, stated before the sampler module loads: otherwise loading it asks the card for its free memory,
# which fails when a running server already holds the card. This check needs no card.
import comfy.cli_args  # noqa: E402

comfy.cli_args.args.cpu = True

import comfy.conds  # noqa: E402
import comfy.samplers  # noqa: E402
import torch  # noqa: E402

import edit_guidance as eg  # noqa: E402

BANK_PROMPT = REPO_ROOT / "prompt_bank" / "t2va_noodle_bar.txt"
failures: list[str] = []


def ok(label: str, cond: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if cond else 'FAIL'}  {label}" + (f": {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(label)


def refuses(label: str, fn, must_contain: str) -> None:
    try:
        fn()
    except ValueError as exc:
        ok(f"refuses {label}", must_contain in str(exc), f"said {str(exc)[:120]!r}, expected {must_contain!r}")
    except Exception as exc:  # a refusal is the tool's own words, never another error
        ok(f"refuses {label}", False, f"raised {type(exc).__name__}: {exc}")
    else:
        ok(f"refuses {label}", False, "it did not refuse")


def check_prompt_edit() -> None:
    prompt = BANK_PROMPT.read_text(encoding="utf-8").strip()
    opening = re.search(r"\[Shot 1\] ([^,]+, [^,]+,)", prompt).group(1)  # the style words after the shot header
    edited, records = eg.apply_edits(prompt, f"{opening} -> Stop-motion, clay,")
    ok("an edit is placed once and nothing else changes",
       edited == prompt.replace(opening, "Stop-motion, clay,", 1) and len(records) == 1)
    words = re.findall(r"[a-z]{5,}", prompt.split("overall_soundscape:")[0])
    twice = next(w for w in words if prompt.count(w) > 1 and "<d>" not in w)
    spoken = re.search(r"<d>\[[A-Za-z]+\] (.+?)</d>", prompt).group(1)
    sound = prompt.split("overall_soundscape:", 1)[1].split("non_diegetic_music:")[0].strip().split(".")[0]
    refuses("a line with no arrow", lambda: eg.apply_edits(prompt, "just some words"), "old wording -> new wording")
    refuses("wording that is absent", lambda: eg.apply_edits(prompt, "a brass sundial -> a clock"), "occurs 0 time")
    refuses("wording that occurs more than once", lambda: eg.apply_edits(prompt, f"{twice} -> thing"), "exactly once")
    refuses("wording inside dialogue", lambda: eg.apply_edits(prompt, f"{spoken} -> hello"), "inside <d>")
    refuses("wording that carries a shot header", lambda: eg.apply_edits(prompt, f"[Shot 1] {opening} -> x y"),
            "tag, a speaker id or a header")
    refuses("a soundscape edit aimed at the description",
            lambda: eg.apply_edits(prompt, f"soundscape: {opening} -> Clay,"), "inside the overall_soundscape")
    out, _ = eg.apply_edits(prompt, f"soundscape: {sound} -> A kettle whistles")
    ok("a soundscape edit lands in the soundscape section", "A kettle whistles" in out.split("overall_soundscape:")[1])
    out_music, _ = eg.apply_edits(prompt, "music: N/A -> A solo cello holds long low notes.")
    ok("a music: edit replaces N/A inside non_diegetic_music:",
       "solo cello" in out_music.split("non_diegetic_music:")[1])
    ref2va_prompt = (REPO_ROOT / "prompt_bank" / "ref2va_look_anchor.txt").read_text(encoding="utf-8").strip()
    out_sum, _ = eg.apply_edits(ref2va_prompt, "summary: crossing -> striding across")
    ok("a summary: edit replaces prose strictly inside summary:",
       "striding across" in out_sum.split("summary:")[1].split("retention_analysis:")[0])
    refuses("a summary: edit aimed at detailed_description",
            lambda: eg.apply_edits(ref2va_prompt, "summary: chiaroscuro lighting -> neon lighting"),
            "inside the summary")
    street_ref2va = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
    out_rel_toggle, recs_rel = eg.apply_edits(street_ref2va, "", release_subject_attire=True)
    ret_block = out_rel_toggle.split("retention_analysis:")[1].split("detailed_description:")[0]
    ok("release_subject_attire=True flips <Subject 1> to partially_preserved and releases clothing with zero edits line",
       "partially_preserved" in ret_block and "fully_preserved" not in ret_block
       and "clothing is not retained" in ret_block and len(recs_rel) == 1)
    music_ref2va = (REPO_ROOT / "prompt_bank" / "ref2va_image_audio_music.txt").read_text(encoding="utf-8").strip()
    out_rel_dir, _ = eg.apply_edits(music_ref2va, "release_attire: <Subject 1>")
    ret_music = out_rel_dir.split("retention_analysis:")[1].split("detailed_description:")[0]
    ok("release_attire strips ', and clothing' cleanly so retention_analysis never contradicts itself",
       "face and hair are retained; clothing is not retained" in ret_music
       and "<Subject 2> (appears in [Shot 1]): fully_preserved" in ret_music)
    out_sec_all, _ = eg.apply_edits(
        street_ref2va,
        "subject_definitions: a black chunky ribbed knit sweater -> an olive canvas field coat\n"
        "retention_analysis: the black chunky knit sweater -> the olive canvas field coat\n"
        "description: a black chunky knit sweater -> an olive canvas field coat",
    )
    ok("subject_definitions:, retention_analysis:, and description: prefixes scope edits cleanly to each section",
       out_sec_all.count("olive canvas field coat") == 3)
    refuses("release_subject_attire=True on a t2va prompt with no retention_analysis:",
            lambda: eg.apply_edits(prompt, "", release_subject_attire=True),
            "retention_analysis:")
    refuses("release_attire naming a nonexistent <Subject 9>",
            lambda: eg.apply_edits(street_ref2va, "release_attire: <Subject 9>"),
            "<Subject 9> was not found")
    semi_prompt = "integrated_multimodal_description:\n[Shot 1] Ambient sound; old radio hum fills the hallway.\noverall_soundscape:\nLow radio hum.\nnon_diegetic_music:\nN/A"
    out_semi, _ = eg.apply_edits(semi_prompt, "sound; old radio hum -> quiet room")
    ok("prose old-wording starting with 'sound;' is kept intact as a single edit rather than split as a section prefix",
       "Ambient quiet room fills the hallway." in out_semi)


class _Patcher:
    model_options: dict = {}


def _guider(scale: float, first_step: int = 0, align: bool = False, last_step: int = 1000) -> eg.EditGuider:
    g = eg.EditGuider(_Patcher())
    g.inner_model = None
    g.conds = {"negative": [{"which": "original", "model_conds": {}}],
               "positive": [{"which": "edited", "model_conds": {}}]}
    g.set_edit(scale, first_step, align, last_step=last_step)
    return g


def check_guider_arithmetic() -> None:
    seen: list[list[str]] = []
    real = comfy.samplers.calc_cond_batch

    def fake(model, conds, x, timestep, model_options):
        seen.append([c[0]["which"] for c in conds])
        return [torch.full((2,), 1.0 if c[0]["which"] == "original" else 3.0) for c in conds]

    comfy.samplers.calc_cond_batch = fake
    try:
        sig = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.0])
        opts = {"transformer_options": {"sample_sigmas": sig}}

        def run(g, step=0):
            seen.clear()
            return float(g.predict_noise(None, sig[step:step + 1], opts)[0])

        ok("scale 0 is the original pass, alone", run(_guider(0.0)) == 1.0 and seen == [["original"]])
        ok("scale 1 is the edited pass, alone", run(_guider(1.0)) == 3.0 and seen == [["edited"]])
        ok("scale 1.5 is original + 1.5 * (edited - original), in two passes",
           run(_guider(1.5)) == 4.0 and seen == [["original", "edited"]])
        g = _guider(2.0, first_step=2)
        ok("before first_step only the original runs", run(g, 0) == 1.0 and seen == [["original"]]
           and run(g, 1) == 1.0 and seen == [["original"]])
        ok("from first_step on, both run", run(g, 2) == 5.0 and seen == [["original", "edited"]])
        g_win = _guider(2.0, first_step=1, last_step=2)
        ok("windowed guider runs [original] before first_step, 2-pass inside [1..2], and 1-pass [edited] after last_step",
           run(g_win, 0) == 1.0 and seen == [["original"]]
           and run(g_win, 1) == 5.0 and seen == [["original", "edited"]]
           and run(g_win, 2) == 5.0 and seen == [["original", "edited"]]
           and run(g_win, 3) == 3.0 and seen == [["edited"]])
        refuses("last_step < first_step", lambda: _guider(1.5, first_step=3, last_step=2), "last_step")
        ok("with no schedule in the options the edit is on from the start",
           float(_guider(2.0, first_step=3).predict_noise(None, sig[:1], {})[0]) == 5.0)
    finally:
        comfy.samplers.calc_cond_batch = real


class _Layout:
    def __init__(self, text_len: int, media: int):
        self.signature = (text_len, 1, 1, 1, 1)
        self.position_ids = torch.zeros(text_len + media, 3, dtype=torch.float64)
        self.position_ids[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
        self.position_ids[text_len:, 0] = float(text_len)


def check_prepare() -> None:
    def conds(text_len):
        return [{"model_conds": {"minimax_payload": comfy.conds.CONDConstant({"seed": 0, "layout": _Layout(text_len, 4)})}}]

    for align in (False, True):
        g = _guider(1.5, align=align)
        g.conds = {"negative": conds(10), "positive": conds(13)}
        short_before = g.conds["negative"][0]["model_conds"]["minimax_payload"].cond["layout"]
        g._prepare()
        po = g.conds["negative"][0]["model_conds"]["minimax_payload"].cond
        pe = g.conds["positive"][0]["model_conds"]["minimax_payload"].cond
        if not align:
            ok("the two payloads are marked and differ", po[eg.PASS_KEY] == "original" and pe[eg.PASS_KEY] == "edited")
            ok("without align_positions the layouts are untouched", po["layout"] is short_before)
        else:
            lo = po["layout"]
            ok("align_positions moves the shorter prompt's media rows to the longer origin",
               bool((lo.position_ids[10:, 0] == 13.0).all()) and bool((pe["layout"].position_ids[13:, 0] == 13.0).all()))
            ok("the text rows, the signature and the layout handed in are left alone",
               bool((lo.position_ids[:10, 0] == torch.arange(10)).all()) and lo.signature[0] == 10
               and bool((short_before.position_ids[10:, 0] == 10.0).all()))


def check_inputs() -> None:
    tags = torch.ones(5)
    good = [[torch.zeros(1, 5, 4), {"minimax_token_tags": tags}]]
    longer = [[torch.zeros(1, 7, 4), {"minimax_token_tags": torch.ones(7)}]]
    ok("two text-only conditionings pass and report their lengths", eg.check_same_but_text(good, longer) == (5, 7))
    refuses("conditioning with no token tags",
            lambda: eg.check_same_but_text([[torch.zeros(1, 5, 4), {}]], good), "no H3 token tags")
    refuses("a pair that differs in references",
            lambda: eg.check_same_but_text(good, [[torch.zeros(1, 5, 4), {"minimax_token_tags": tags, "minimax_refs": [1]}]]),
            "differ in minimax_refs")
    refuses("a list of more than one prompt", lambda: eg.check_same_but_text(good + good, good), "single prompt")


def check_static_reader() -> None:
    """A static graph walker reads, from an edit node, the text the node would hand on."""
    sys.path.insert(0, str(REPO_ROOT / "workflows"))
    import prompts as wf_prompts  # workflows/prompts.py

    prompt = BANK_PROMPT.read_text(encoding="utf-8").strip()
    opening = re.search(r"\[Shot 1\] ([^,]+, [^,]+,)", prompt).group(1)
    edits = f"{opening} -> Stop-motion, clay,"
    graph = {"130": {"class_type": "MiniMaxH3PromptEdit", "inputs": {"original": prompt, "edits": edits}}}
    text, source = wf_prompts._follow(["130", 0], graph)
    ok("a walker following a prompt linked from an edit node reads the edited text",
       text == eg.apply_edits(prompt, edits)[0] and text != prompt and source is graph["130"])
    for name in ("h3_text_to_video_pika_edit_guidance_api.json",
                 "h3_image_ref_plus_text_to_video_pika_edit_guidance_api.json"):
        shipped = json.loads((REPO_ROOT / "workflows" / name).read_text(encoding="utf-8"))
        got = wf_prompts.describe(shipped)
        as_written, text_node = wf_prompts._follow(shipped["5"]["inputs"]["prompt"], shipped)
        ok(f"{name}: the graph is described by its prompt as written, not by the edited copy",
           isinstance(as_written, str) and got["prompt_id"] is not None
           and got["prompt_sha256"] == wf_prompts.sha256(as_written.rstrip()),
           f"prompt_id {got['prompt_id']!r}")
        edit_nodes = [n for n in shipped.values() if n["class_type"] == "MiniMaxH3PromptEdit"]
        literals = [k for k, n in shipped.items() for v in n["inputs"].values() if v == as_written]
        ok(f"{name}: the prompt is written once, on a text node that feeds the conditioning node and the edit node",
           len(edit_nodes) == 1 and text_node is not None and len(literals) == 1
           and shipped["5"]["inputs"]["prompt"] == edit_nodes[0]["inputs"]["original"],
           f"written on node(s) {literals}")
    graph["130"]["inputs"]["edits"] = "a brass sundial -> a clock"
    ok("and reads nothing, without raising, where the node would refuse the edit",
       wf_prompts._follow(["130", 0], graph) == (None, None))
    street_ref2va = (REPO_ROOT / "prompt_bank" / "ref2va_woman_blonde_street.txt").read_text(encoding="utf-8").strip()
    rel_graph = {
        "130": {
            "class_type": "MiniMaxH3PromptEdit",
            "inputs": {"original": street_ref2va, "edits": "", "release_subject_attire": True},
        }
    }
    rel_text, _ = wf_prompts._follow(["130", 0], rel_graph)
    ok("static walker resolves MiniMaxH3PromptEdit with release_subject_attire=True and empty edits",
       isinstance(rel_text, str) and "partially_preserved" in rel_text and "clothing is not retained" in rel_text)

    import importlib
    import importlib.util
    import nodes as _comfy_nodes  # load ComfyUI core's `nodes` before `build_workflows` puts REPO_ROOT at sys.path[0]
    del _comfy_nodes
    import build_workflows as bw  # workflows/build_workflows.py
    import h3_config as cfg  # workflows/h3_config.py

    if "h3x" not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            "h3x", REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)]
        )
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["h3x"] = _mod
        _spec.loader.exec_module(_mod)
    rc_ = importlib.import_module("h3x.reference_conditioning")
    re_ = importlib.import_module("h3x.reference_encode")

    sf_ref_eg = bw.build_api(
        "r2v",
        prompt=bw.PIKA_IMAGE_EDIT_PROMPT,
        ref_images=(bw.PLACEHOLDER_IMAGE_A,),
        ref_image_count=1,
        ref_size_policy="match",
        single_frame=True,
        lora=(cfg.PIKA_REF2VA_SVD995_LORA, cfg.PIKA_STRENGTH),
        lora_branch=True,
        steps=cfg.PIKA_STEPS,
        sampler_name=cfg.PIKA_SAMPLER,
        manual_sigmas=cfg.PIKA_MANUAL_SIGMAS,
        width=768,
        height=768,
        edit_guidance={"edits": "summary: portrait -> close portrait"},
    )
    ok("single-frame r2v edit guidance splits into 1x MiniMaxH3EncodeReferences(single_frame=True) + 2x PromptOnReferences + Preflight.edited_conditioning",
       sf_ref_eg["132"]["class_type"] == "MiniMaxH3EncodeReferences"
       and sf_ref_eg["132"]["inputs"].get("single_frame") is True
       and sf_ref_eg["5"]["class_type"] == "MiniMaxH3PromptOnReferences"
       and sf_ref_eg["131"]["class_type"] == "MiniMaxH3PromptOnReferences"
       and sf_ref_eg["26"]["inputs"].get("edited_conditioning") == ["131", 0])
    vid_rec = rc_.RuntimeVideoReference(frames=torch.zeros((5, 64, 64, 3)), loaded_fps=24.0, soundtrack=None)
    refuses("MiniMaxH3EncodeReferences(single_frame=True) with a video reference",
            lambda: re_.MiniMaxH3EncodeReferences.execute(None, (vid_rec,), width=64, height=64, single_frame=True),
            "still references only")

    guide_eg = bw.build_api(
        "r2v",
        prompt=bw.PIKA_STILL_EDIT_PROMPT,
        ref_images_on=False,
        single_frame=True,
        guide_image=True,
        guide_file=bw.PLACEHOLDER_IMAGE_A,
        lora=(cfg.PIKA_REF2VA_SVD995_LORA, cfg.PIKA_STRENGTH),
        lora_branch=True,
        steps=cfg.PIKA_STEPS,
        sampler_name=cfg.PIKA_SAMPLER,
        manual_sigmas=cfg.PIKA_MANUAL_SIGMAS,
        width=768,
        height=768,
        edit_guidance={"edits": "evening -> twilight"},
    )
    ok("guide-image edit guidance uses MiniMaxH3Conditioning, 1x MiniMaxH3GuideLatent.edited_positive (slot 2), no node 134, and Preflight.edited_conditioning",
       guide_eg["5"]["class_type"] == "MiniMaxH3Conditioning"
       and "4" not in guide_eg
       and "134" not in guide_eg
       and guide_eg["70"]["inputs"].get("edited_positive") == ["131", 0]
       and guide_eg["26"]["inputs"].get("edited_conditioning") == ["70", 2]
       and guide_eg["9"]["inputs"].get("edited") == ["70", 2])


def main() -> int:
    print("edit guidance: the prompt edit, the guider's arithmetic, pass marking and position alignment")
    check_prompt_edit()
    check_static_reader()
    check_guider_arithmetic()
    check_prepare()
    check_inputs()
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nall ok -- the edit lands where it is told to, and the guider computes what it says")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
