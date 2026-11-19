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
  `minimax_refs`, while allowing differing `minimax_keyframes` (unguided `original` vs guided `edited`).

What it cannot see: what a render looks like, or whether the aligned positions are what the model should be
given. Those take renders.

    <comfy-venv-python> bench/check_edit_guidance.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
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
import comfy.nested_tensor  # noqa: E402
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
    out_sec_all, _ = eg.apply_edits(
        street_ref2va,
        "subject_definitions: a black chunky ribbed knit sweater -> an olive canvas field coat\n"
        "retention_analysis: the black chunky knit sweater -> the olive canvas field coat\n"
        "description: a black chunky knit sweater -> an olive canvas field coat",
    )
    ok("subject_definitions:, retention_analysis:, and description: prefixes scope edits cleanly to each section",
       out_sec_all.count("olive canvas field coat") == 3)
    semi_prompt = "integrated_multimodal_description:\n[Shot 1] Ambient sound; old radio hum fills the hallway.\noverall_soundscape:\nLow radio hum.\nnon_diegetic_music:\nN/A"
    out_semi, _ = eg.apply_edits(semi_prompt, "sound; old radio hum -> quiet room")
    ok("prose old-wording starting with 'sound;' is kept intact as a single edit rather than split as a section prefix",
       "Ambient quiet room fills the hallway." in out_semi)


class _Patcher:
    model_options: dict = {}

    def is_dynamic(self) -> bool:
        return False


def _guider(
    scale: float,
    first_step: int = 0,
    align: bool = False,
    last_step: int = 1000,
    streams: str = "both",
    audio_scale: float = 1.0,
    mask: torch.Tensor | None = None,
    temporal_curve: str = "whole_clip",
) -> eg.EditGuider:
    g = eg.EditGuider(_Patcher())
    g.inner_model = None
    g.conds = {"negative": [{"which": "original", "model_conds": {}}],
               "positive": [{"which": "edited", "model_conds": {}}]}
    g.set_edit(
        scale,
        first_step,
        align,
        last_step=last_step,
        streams=streams,
        audio_scale=audio_scale,
        mask=mask,
        temporal_curve=temporal_curve,
    )
    return g


def check_guider_arithmetic() -> None:
    seen: list[list[str]] = []
    real = comfy.samplers.calc_cond_batch

    def fake(model, conds, x, timestep, model_options):
        seen.append([c[0]["which"] for c in conds])
        if getattr(x, "is_nested", False):
            out = []
            for c in conds:
                if c[0]["which"] == "original":
                    out.append(comfy.nested_tensor.NestedTensor((torch.full((2,), 1.0), torch.full((3,), 10.0))))
                else:
                    out.append(comfy.nested_tensor.NestedTensor((torch.full((2,), 3.0), torch.full((3,), 20.0))))
            return out
        lshapes = getattr(model, "latent_shapes", None)
        if lshapes is not None and len(lshapes) >= 2:
            out = []
            for c in conds:
                v_val, a_val = (1.0, 10.0) if c[0]["which"] == "original" else (3.0, 20.0)
                packed, _ = comfy.utils.pack_latents([torch.full(lshapes[0], v_val), torch.full(lshapes[1], a_val)])
                out.append(packed)
            return out
        return [torch.full((2,), 1.0 if c[0]["which"] == "original" else 3.0) for c in conds]

    comfy.samplers.calc_cond_batch = fake
    try:
        sig = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.0])
        opts = {"transformer_options": {"sample_sigmas": sig}}

        def run(g, step=0):
            seen.clear()
            return float(g.predict_noise(None, sig[step:step + 1], opts)[0])

        def run_av(g, step=0):
            seen.clear()
            x_av = comfy.nested_tensor.NestedTensor((torch.zeros((2,)), torch.zeros((3,))))
            res = g.predict_noise(x_av, sig[step:step + 1], opts)
            v, a = res.unbind()
            return round(float(v[0]), 5), round(float(a[0]), 5)

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
        ok("NestedTensor streams='both' steers video and audio identically in 2 passes",
           run_av(_guider(1.5, streams="both")) == (4.0, 25.0) and seen == [["original", "edited"]])
        g_ao = _guider(1.8, first_step=1, last_step=2, streams="audio_only")
        ok("NestedTensor streams='audio_only' locks video to original across all steps and steers audio only",
           run_av(g_ao, 0) == (1.0, 10.0) and seen == [["original"]]
           and run_av(g_ao, 1) == (1.0, 28.0) and seen == [["original", "edited"]]
           and run_av(g_ao, 3) == (1.0, 20.0) and seen == [["original", "edited"]])
        g_vo = _guider(1.8, first_step=1, last_step=2, streams="video_only")
        ok("NestedTensor streams='video_only' steers video and locks audio to original across all steps",
           run_av(g_vo, 0) == (1.0, 10.0) and seen == [["original"]]
           and run_av(g_vo, 1) == (4.6, 10.0) and seen == [["original", "edited"]]
           and run_av(g_vo, 3) == (3.0, 10.0) and seen == [["original", "edited"]])
        g_ind = _guider(1.85, first_step=1, last_step=2, streams="independent", audio_scale=1.0)
        ok("NestedTensor streams='independent' (scale=1.85, audio_scale=1.0) caps audio at edited in 2 passes and drops to 1-pass [edited] after last_step",
           run_av(g_ind, 0) == (1.0, 10.0) and seen == [["original"]]
           and run_av(g_ind, 1) == (4.7, 20.0) and seen == [["original", "edited"]]
           and run_av(g_ind, 3) == (3.0, 20.0) and seen == [["edited"]])
        g_kf_vo = _guider(1.6, first_step=0, last_step=2, streams="video_only")
        g_kf_vo._release_to_original_after = True
        ok("pure guide-differential (identical text, 0 vs K keyframes) hands over to 1-pass [original] after last_step even under streams='video_only'",
           run_av(g_kf_vo, 1) == (4.2, 10.0) and seen == [["original", "edited"]]
           and run_av(g_kf_vo, 3) == (1.0, 10.0) and seen == [["original"]])

        # Real EditGuider.set_conds prefix keyframe detection:
        # Identical text tokens + edited appending a mid-clip keyframe onto [first, last]
        # sets _release_to_original_after=True; mismatched prefix keyframes set False.
        tags_eq = torch.ones((1, 5), dtype=torch.long)
        toks_eq = torch.zeros((1, 5, 4), dtype=torch.float32)
        kf_first = {"resolved_frame_index": 0, "latent": torch.ones((1, 24, 1, 2, 2))}
        kf_last = {"resolved_frame_index": 89, "latent": torch.full((1, 24, 1, 2, 2), 2.0)}
        kf_mid = {"resolved_frame_index": 34, "latent": torch.full((1, 24, 1, 2, 2), 3.0)}
        cond_orig_fl = [[toks_eq, {"minimax_token_tags": tags_eq, "minimax_keyframes": [kf_first, kf_last]}]]
        cond_edit_flm = [[toks_eq.clone(), {"minimax_token_tags": tags_eq, "minimax_keyframes": [dict(kf_first), dict(kf_last), kf_mid]}]]
        g_pfx = eg.EditGuider(_Patcher())
        g_pfx.inner_model = None
        g_pfx.set_conds(cond_orig_fl, cond_edit_flm)
        g_pfx.set_edit(1.5, 0, False, last_step=2, streams="video_only")
        g_pfx_rel_flag = bool(g_pfx._release_to_original_after)
        # Swap to mock cond markers to verify 1-pass [original] after last_step
        g_pfx.conds = {"negative": [{"which": "original", "model_conds": {}}], "positive": [{"which": "edited", "model_conds": {}}]}
        ok("EditGuider.set_conds enables _release_to_original_after when edited appends mid-clip guides onto identical [first, last] keyframes",
           g_pfx_rel_flag
           and run_av(g_pfx, 1) == (4.0, 10.0) and seen == [["original", "edited"]]
           and run_av(g_pfx, 3) == (1.0, 10.0) and seen == [["original"]])
        cond_edit_mismatch = [[toks_eq.clone(), {"minimax_token_tags": tags_eq, "minimax_keyframes": [{"resolved_frame_index": 0, "latent": kf_mid["latent"]}, kf_last, kf_mid]}]]
        g_mismatch = eg.EditGuider(_Patcher())
        g_mismatch.set_conds(cond_orig_fl, cond_edit_mismatch)
        ok("EditGuider.set_conds disables _release_to_original_after when prefix keyframes differ in latent values",
           not g_mismatch._release_to_original_after)

        class _PackedInnerModel:
            latent_shapes = [torch.Size([1, 24, 2, 4, 4]), torch.Size([1, 32, 2, 5])]

        def run_packed_av(g, step=0):
            seen.clear()
            g.inner_model = _PackedInnerModel()
            x_packed, lshapes = comfy.utils.pack_latents([torch.zeros(l) for l in _PackedInnerModel.latent_shapes])
            res = g.predict_noise(x_packed, sig[step:step + 1], opts)
            v, a = comfy.utils.unpack_latents(res, lshapes)
            return round(float(v.flatten()[0]), 5), round(float(a.flatten()[0]), 5)

        g_p_vo = _guider(1.6, first_step=0, last_step=2, streams="video_only")
        g_p_ao = _guider(1.8, first_step=0, last_step=2, streams="audio_only")
        g_p_ind = _guider(1.85, first_step=0, last_step=2, streams="independent", audio_scale=1.0)
        ok("packed CFGGuider.sample latents (inner_model.latent_shapes) isolate video_only, audio_only, and independent",
           run_packed_av(g_p_vo, 1) == (4.2, 10.0) and seen == [["original", "edited"]]
           and run_packed_av(g_p_ao, 1) == (1.0, 28.0) and seen == [["original", "edited"]]
           and run_packed_av(g_p_ind, 1) == (4.7, 20.0) and seen == [["original", "edited"]])
        ok("NestedTensor streams='independent' with scale=1.0, audio_scale=1.0 short-circuits to 1-pass [edited]",
           run_av(_guider(1.0, streams="independent", audio_scale=1.0)) == (3.0, 20.0) and seen == [["edited"]])
        ok("single-stream tensor with streams='audio_only' short-circuits to 1-pass [original]",
           run(_guider(1.8, streams="audio_only")) == 1.0 and seen == [["original"]])
        refuses("last_step < first_step", lambda: _guider(1.5, first_step=3, last_step=2), "last_step")
        refuses("unknown streams mode", lambda: _guider(1.5, streams="bogus"), "streams")
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
    guided_edit = [[torch.zeros(1, 5, 4), {"minimax_token_tags": tags, "minimax_keyframes": [{"resolved_frame_index": 24}]}]]
    ok("two text-only conditionings pass and report their lengths", eg.check_same_but_text(good, longer) == (5, 7))
    ok("a pair that differs in minimax_keyframes (unguided original vs guided edited) passes",
       eg.check_same_but_text(good, guided_edit) == (5, 5))
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
    sec_graph = {
        "130": {
            "class_type": "MiniMaxH3PromptEdit",
            "inputs": {
                "original": street_ref2va,
                "edits": "retention_analysis: fully_preserved -> partially_preserved",
            },
        }
    }
    sec_text, _ = wf_prompts._follow(["130", 0], sec_graph)
    ok("static walker resolves MiniMaxH3PromptEdit with retention_analysis: section edit",
       isinstance(sec_text, str) and "partially_preserved" in sec_text)

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
        edit_guidance={"edits": "evening -> twilight", "streams": "independent", "scale": 1.85, "audio_scale": 1.0},
    )
    ok("guide-image edit guidance uses MiniMaxH3Conditioning, 1x MiniMaxH3GuideLatent.edited_positive (slot 2), no node 134, Preflight.edited_conditioning, and streams/audio_scale",
       guide_eg["5"]["class_type"] == "MiniMaxH3Conditioning"
       and "4" not in guide_eg
       and "134" not in guide_eg
       and guide_eg["70"]["inputs"].get("edited_positive") == ["131", 0]
       and guide_eg["26"]["inputs"].get("edited_conditioning") == ["70", 2]
       and guide_eg["9"]["inputs"].get("edited") == ["70", 2]
       and guide_eg["9"]["inputs"].get("streams") == "independent"
       and guide_eg["9"]["inputs"].get("audio_scale") == 1.0)


def check_guider_mask() -> None:
    seen: list[list[str]] = []
    real = comfy.samplers.calc_cond_batch

    class _PackedAVModel:
        latent_shapes = [torch.Size([1, 24, 7, 4, 4]), torch.Size([1, 32, 2, 37])]

    class _Image5DModel:
        latent_shapes = [torch.Size([1, 24, 1, 4, 4])]

    def fake(model, conds, x, timestep, model_options):
        seen.append([c[0]["which"] for c in conds])
        if getattr(x, "is_nested", False):
            v_in, a_in = x.unbind()
            out = []
            for c in conds:
                v_val, a_val = (1.0, 10.0) if c[0]["which"] == "original" else (3.0, 20.0)
                out.append(comfy.nested_tensor.NestedTensor((torch.full(v_in.shape, v_val), torch.full(a_in.shape, a_val))))
            return out
        lshapes = getattr(model, "latent_shapes", None)
        if lshapes is not None and len(lshapes) >= 2:
            out = []
            for c in conds:
                v_val, a_val = (1.0, 10.0) if c[0]["which"] == "original" else (3.0, 20.0)
                packed, _ = comfy.utils.pack_latents([torch.full(lshapes[0], v_val), torch.full(lshapes[1], a_val)])
                out.append(packed)
            return out
        return [torch.full(x.shape, 1.0 if c[0]["which"] == "original" else 3.0) for c in conds]

    def approx(a: float, b: float, tol: float = 1e-4) -> bool:
        return abs(a - b) <= tol

    # High-res 64x64 spatial mask downsampled via adaptive_max_pool2d to 4x4:
    # top-left = 0.0 (protected), top-right = 0.5 (feathered), bottom = 1.0 (full edit).
    spatial_64 = torch.zeros((1, 64, 64), dtype=torch.float32)
    spatial_64[:, :32, 32:] = 0.5
    spatial_64[:, 32:, :] = 1.0

    sig = torch.tensor([1.0, 0.9, 0.8, 0.5, 0.0])
    opts = {"transformer_options": {"sample_sigmas": sig}}

    comfy.samplers.calc_cond_batch = fake
    try:
        # 1a. NestedTensor spatial mask
        seen.clear()
        g_nest = _guider(1.8, mask=spatial_64)
        x_nest = comfy.nested_tensor.NestedTensor((torch.zeros((1, 24, 7, 4, 4)), torch.zeros((1, 32, 2, 37))))
        v_n, a_n = g_nest.predict_noise(x_nest, sig[0:1], opts).unbind()
        ok("NestedTensor spatial mask preserves 0.0/0.5/1.0 video weights and leaves audio unmasked at scalar 1.8",
           seen == [["original", "edited"]]
           and bool((v_n[:, :, :, :2, :2] == 1.0).all())
           and approx(float(v_n[0, 0, 0, 0, 2]), 2.8)
           and approx(float(v_n[0, 0, -1, 3, 3]), 4.6)
           and approx(float(a_n.min()), 28.0) and approx(float(a_n.max()), 28.0))

        # 1b. Live packed_split 3D AV path (exercises C=24 reshape, not broken pack_latents)
        seen.clear()
        g_pack = _guider(1.8, mask=spatial_64)
        g_pack.inner_model = _PackedAVModel()
        x_p, lshapes = comfy.utils.pack_latents([torch.zeros(s) for s in _PackedAVModel.latent_shapes])
        res_p = g_pack.predict_noise(x_p, sig[0:1], opts)
        v_p, a_p = comfy.utils.unpack_latents(res_p, lshapes)
        ok("packed_split 3D spatial mask reshapes to latent_shapes[0] cleanly and leaves audio scalar",
           seen == [["original", "edited"]]
           and v_p.shape == _PackedAVModel.latent_shapes[0]
           and a_p.shape == _PackedAVModel.latent_shapes[1]
           and bool((v_p[:, :, :, :2, :2] == 1.0).all())
           and bool(torch.allclose(v_p[:, :, :, :2, 2:], torch.full_like(v_p[:, :, :, :2, 2:], 2.8)))
           and bool(torch.allclose(v_p[:, :, :, 2:, :], torch.full_like(v_p[:, :, :, 2:, :], 4.6)))
           and bool(torch.allclose(a_p, torch.full_like(a_p, 28.0))))

        # 1c. Plain 5D unpacked tensor (MiniMax-H3-Image)
        seen.clear()
        g_img = _guider(1.8, mask=spatial_64)
        g_img.inner_model = _Image5DModel()
        v_img = g_img.predict_noise(torch.zeros((1, 24, 1, 4, 4)), sig[0:1], opts)
        ok("plain 5D image tensor spatial mask keeps continuous weights on T_lat=1",
           seen == [["original", "edited"]]
           and v_img.shape == (1, 24, 1, 4, 4)
           and bool((v_img[:, :, :, :2, :2] == 1.0).all())
           and approx(float(v_img[0, 0, 0, 0, 2]), 2.8)
           and approx(float(v_img[0, 0, 0, 3, 3]), 4.6))

        # 2. Pass counts and step boundaries with mask
        seen.clear()
        g_one = _guider(1.0, mask=spatial_64)
        g_one.inner_model = _PackedAVModel()
        v_one, a_one = comfy.utils.unpack_latents(g_one.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("scale=1.0 with mask runs 2 passes and protects m=0 at original",
           seen == [["original", "edited"]]
           and bool((v_one[:, :, :, :2, :2] == 1.0).all())
           and bool(torch.allclose(v_one[:, :, :, 2:, :], torch.full_like(v_one[:, :, :, 2:, :], 3.0)))
           and bool(torch.allclose(a_one, torch.full_like(a_one, 20.0))))

        g_win = _guider(1.8, first_step=1, last_step=2, mask=spatial_64)
        g_win.inner_model = _PackedAVModel()
        seen.clear()
        v_b, a_b = comfy.utils.unpack_latents(g_win.predict_noise(x_p, sig[0:1], opts), lshapes)
        s_b = list(seen)
        seen.clear()
        v_a, a_a = comfy.utils.unpack_latents(g_win.predict_noise(x_p, sig[3:4], opts), lshapes)
        s_a = list(seen)
        ok("step < first_step runs 1-pass [original] and step > last_step (same_text=False) runs 2 passes pinning m=0 to [original]",
           s_b == [["original"]] and bool((v_b == 1.0).all()) and bool((a_b == 10.0).all())
           and s_a == [["original", "edited"]]
           and bool((v_a[:, :, :, :2, :2] == 1.0).all())
           and bool(torch.allclose(v_a[:, :, :, 2:, :], torch.full_like(v_a[:, :, :, 2:, :], 3.0)))
           and bool(torch.allclose(a_a, torch.full_like(a_a, 20.0))))

        g_rel = _guider(1.8, first_step=0, last_step=2, mask=spatial_64)
        g_rel._release_to_original_after = True
        g_rel.inner_model = _PackedAVModel()
        seen.clear()
        v_r, a_r = comfy.utils.unpack_latents(g_rel.predict_noise(x_p, sig[3:4], opts), lshapes)
        ok("_release_to_original_after=True with mask drops to 1-pass [original] after last_step",
           seen == [["original"]] and bool((v_r == 1.0).all()) and bool((a_r == 10.0).all()))

        # 3. 2D [H, W] vs [1, H, W] broadcast and causal [1, 4, 4, 4, 4] pooling on 22 frames (T_lat = 7)
        m_2d = spatial_64[0]
        g_2d = _guider(1.8, mask=m_2d)
        g_2d.inner_model = _PackedAVModel()
        v_2d, _ = comfy.utils.unpack_latents(g_2d.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("2D [H, W] mask broadcasts identically to [1, H, W] across all 7 latent steps",
           bool(torch.allclose(v_2d, v_p)))

        for hit_frame, expected_t_lat in ((0, 0), (3, 1), (17, 5)):
            m_kf = torch.zeros((22, 4, 4), dtype=torch.float32)
            m_kf[hit_frame] = 1.0
            g_kf = _guider(1.8, mask=m_kf)
            g_kf.inner_model = _PackedAVModel()
            v_kf, _ = comfy.utils.unpack_latents(g_kf.predict_noise(x_p, sig[0:1], opts), lshapes)
            other_steps = [t for t in range(7) if t != expected_t_lat]
            ok(f"[22, 4, 4] causal max-pool maps frame {hit_frame} exclusively to latent step t_lat={expected_t_lat}",
               bool(torch.allclose(v_kf[:, :, expected_t_lat], torch.full_like(v_kf[:, :, expected_t_lat], 4.6)))
               and bool((v_kf[:, :, other_steps] == 1.0).all()))

        m_aud_gate = torch.zeros((22, 4, 4), dtype=torch.float32)
        m_aud_gate[10:13, 1:3, 1:3] = 1.0
        g_aud_gate = _guider(1.8, mask=m_aud_gate)
        g_aud_gate.inner_model = _PackedAVModel()
        v_ag, a_ag = comfy.utils.unpack_latents(g_aud_gate.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("[22, 4, 4] spatio-temporal mask gates audio onto active window while keeping quiet ends at original (10.0)",
           approx(float(a_ag[0, 0, 0, 0]), 10.0)
           and approx(float(a_ag[0, 0, 0, -1]), 10.0)
           and approx(float(a_ag.max()), 28.0)
           and bool((v_ag[:, :, :3] == 1.0).all())
           and bool((v_ag[:, :, 4:] == 1.0).all()))

        # 4. [22, 1, 1] temporal envelope modulates both video and audio
        env = torch.full((22, 1, 1), 0.25, dtype=torch.float32)
        env[5:] = 1.0
        g_env = _guider(2.0, mask=env)
        g_env.inner_model = _PackedAVModel()
        v_e, a_e = comfy.utils.unpack_latents(g_env.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("[22, 1, 1] temporal envelope modulates video causally (1.0 + 0.5*2.0 vs 5.0) and audio continuously",
           approx(float(v_e[0, 0, 0, 0, 0]), 2.0)
           and approx(float(v_e[0, 0, 1, 0, 0]), 2.0)
           and approx(float(v_e[0, 0, 2, 0, 0]), 5.0)
           and approx(float(a_e[0, 0, 0, 0]), 15.0)
           and approx(float(a_e[0, 0, 0, -1]), 30.0))

        # Refusal on mismatched frame count
        g_bad_t = _guider(1.5, mask=torch.ones((10, 4, 4)))
        g_bad_t.inner_model = _PackedAVModel()
        refuses("wrong frame count ([10, 4, 4] on 22-frame T_lat=7)",
                lambda: g_bad_t.predict_noise(x_p, sig[0:1], opts), "expected 1 or 22")

        # 4b. build_temporal_curve + temporal_curve on video & still
        c_ramp = eg.build_temporal_curve("ramp_in", 22)
        c_mid = eg.build_temporal_curve("sudden_midclip", 22)
        c_second = eg.build_temporal_curve("second_half", 22)
        ok("build_temporal_curve shapes and boundary values (whole_clip/still -> None, ramp_in/sudden_midclip/second_half)",
           eg.build_temporal_curve("whole_clip", 22) is None
           and eg.build_temporal_curve("ramp_in", 1) is None
           and c_ramp is not None and c_ramp.shape == (22, 1, 1)
           and approx(float(c_ramp[0, 0, 0]), 0.0) and approx(float(c_ramp[-1, 0, 0]), 1.0)
           and c_mid is not None and float(c_mid[:11].max()) == 0.0 and float(c_mid[11:].min()) == 1.0
           and c_second is not None and float(c_second[:11].max()) == 0.0
           and float(c_second[11:].min()) > 0.0 and approx(float(c_second[-1, 0, 0]), 1.0))

        seen.clear()
        g_still_curve = _guider(1.0, temporal_curve="ramp_in")
        g_still_curve.inner_model = _Image5DModel()
        v_sc = g_still_curve.predict_noise(torch.zeros((1, 24, 1, 4, 4)), sig[0:1], opts)
        ok("temporal_curve collapses to no-op on single-frame still (T_lat=1, 1x shortcut at scale=1.0)",
           seen == [["edited"]] and bool(torch.allclose(v_sc, torch.full_like(v_sc, 3.0))))

        seen.clear()
        g_ramp = _guider(2.0, temporal_curve="ramp_in")
        g_ramp.inner_model = _PackedAVModel()
        v_rmp, a_rmp = comfy.utils.unpack_latents(g_ramp.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("temporal_curve='ramp_in' runs 2 passes and ramps video causally & audio continuously from 0 to scale",
           seen == [["original", "edited"]]
           and approx(float(v_rmp[0, 0, 0, 0, 0]), 1.0)
           and approx(float(v_rmp[0, 0, -1, 0, 0]), 5.0)
           and approx(float(a_rmp[0, 0, 0, 0]), 10.0, tol=0.5)
           and approx(float(a_rmp[0, 0, 0, -1]), 30.0, tol=0.5))

        seen.clear()
        g_combo = _guider(2.0, mask=spatial_64, temporal_curve="sudden_midclip")
        g_combo.inner_model = _PackedAVModel()
        v_cmb, a_cmb = comfy.utils.unpack_latents(g_combo.predict_noise(x_p, sig[0:1], opts), lshapes)
        ok("spatial_64 * temporal_curve='sudden_midclip' holds protected region & step-0 at 1.0 and applies 5.0 late",
           seen == [["original", "edited"]]
           and bool((v_cmb[:, :, :, :2, :2] == 1.0).all())
           and approx(float(v_cmb[0, 0, 0, 3, 3]), 1.0)
           and approx(float(v_cmb[0, 0, -1, 3, 3]), 5.0)
           and approx(float(a_cmb[0, 0, 0, 0]), 10.0)
           and approx(float(a_cmb[0, 0, 0, -1]), 30.0))

        early_only = torch.zeros((22, 4, 4), dtype=torch.float32)
        early_only[:5] = 1.0
        g_disjoint = _guider(1.5, mask=early_only, temporal_curve="second_half")
        g_disjoint.inner_model = _PackedAVModel()
        refuses("disjoint spatial mask (frames 0..4) and temporal_curve='second_half' (frames 11..21)",
                lambda: g_disjoint.predict_noise(x_p, sig[0:1], opts), "do not overlap")
    finally:
        comfy.samplers.calc_cond_batch = real

    # 5. Normalization refusals
    refuses("all-zero LoadImage dummy mask", lambda: _guider(1.5, mask=torch.zeros((1, 64, 64))), "all zero")
    refuses("empty [0, 16, 16] mask", lambda: _guider(1.5, mask=torch.zeros((0, 16, 16))), "empty")
    bad_nan = torch.ones((1, 4, 4))
    bad_nan[0, 0, 0] = float("nan")
    refuses("NaN mask", lambda: _guider(1.5, mask=bad_nan), "NaN or Inf")
    bad_inf = torch.ones((1, 4, 4))
    bad_inf[0, 0, 0] = float("inf")
    refuses("Inf mask", lambda: _guider(1.5, mask=bad_inf), "NaN or Inf")
    refuses("unsupported rank [1, 3, 16, 16]", lambda: _guider(1.5, mask=torch.ones((1, 3, 16, 16))), "must have shape")

    # 6. Package-imported video_mask parity (center crop + causal run_lengths)
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO_ROOT)]
    sys.modules.setdefault("_h3pack", pkg)
    vm = sys.modules.get("_h3pack.video_mask")
    if vm is None:
        spec = importlib.util.spec_from_file_location("_h3pack.video_mask", REPO_ROOT / "video_mask.py")
        assert spec is not None and spec.loader is not None
        vm = importlib.util.module_from_spec(spec)
        sys.modules["_h3pack.video_mask"] = vm
        spec.loader.exec_module(vm)
    aspect_ok = True
    for (w, h), (tw, th) in (((1280, 720), (42, 24)), ((720, 1280), (42, 24)),
                              ((1000, 1000), (42, 24)), ((1344, 768), (24, 42))):
        x0, y0 = vm._fit_crop(w, h, tw, th)
        cropped = eg._fit_crop_hw(torch.ones((1, h, w)), th, tw)
        if tuple(cropped.shape[-2:]) != (h - 2 * y0, w - 2 * x0):
            aspect_ok = False
    runs_ok = all(
        ([1] if t_lat == 1 else [eg.FRAME_PER_TOKEN[k % 5] for k in range(t_lat)]) == list(vm.run_lengths(t_lat))
        for t_lat in (1, 2, 7, 27, 37)
    )
    ok("edit_guidance center-crop and FRAME_PER_TOKEN runs match _h3pack.video_mask", aspect_ok and runs_ok)

    # 7. Schema socket order preservation & EditPlan overrides
    schema = eg.MiniMaxH3EditGuider.define_schema()
    ok("MiniMaxH3EditGuider keeps mask at index 9 and appends optional edit_plan strictly last (index 10)",
       len(schema.inputs) == 11
       and schema.inputs[9].id == "mask" and schema.inputs[9].optional is True
       and schema.inputs[10].id == "edit_plan" and schema.inputs[10].optional is True)

    orig_c = [[torch.ones((1, 10, 4)), {"minimax_token_tags": torch.zeros(10, dtype=torch.long)}]]
    edit_c = [[torch.ones((1, 12, 4)), {"minimax_token_tags": torch.zeros(12, dtype=torch.long)}]]
    out_auto = eg.MiniMaxH3EditGuider.execute(
        _Patcher(),
        orig_c,
        edit_c,
        1.75,
        0,
        edit_plan={
            "first_step": 1,
            "last_step": 4,
            "streams": "audio_only",
            "audio_scale": 1.25,
            "temporal_curve": "second_half",
            "force_scale": None,
            "manual_widgets": False,
        },
    ).result[0]
    ok("edit_plan overrides first_step/last_step/streams/audio_scale/temporal_curve when manual_widgets=False",
       out_auto.first_step == 1 and out_auto.last_step == 4 and out_auto.streams == "audio_only"
       and approx(out_auto.edit_scale, 0.0) and approx(out_auto.edit_audio_scale, 1.75)
       and out_auto._temporal_curve == "second_half")

    out_ind = eg.MiniMaxH3EditGuider.execute(
        _Patcher(),
        orig_c,
        edit_c,
        1.75,
        0,
        edit_plan={
            "first_step": 1,
            "last_step": 3,
            "streams": "independent",
            "audio_scale": 1.0,
            "temporal_curve": "ramp_in",
            "force_scale": None,
            "manual_widgets": False,
        },
    ).result[0]
    ok("edit_plan keeps scale as master visual dial on independent streams",
       approx(out_ind.edit_scale, 1.75) and approx(out_ind.edit_audio_scale, 1.0)
       and out_ind._temporal_curve == "ramp_in")

    out_fast = eg.MiniMaxH3EditGuider.execute(
        _Patcher(),
        orig_c,
        edit_c,
        2.0,
        1,
        streams="video_only",
        edit_plan={
            "first_step": 0,
            "last_step": 3,
            "streams": "both",
            "audio_scale": 1.0,
            "temporal_curve": "whole_clip",
            "force_scale": 1.0,
            "manual_widgets": False,
        },
    ).result[0]
    ok("fast_1x EditPlan forces 1-pass scale=1.0, first_step=0, last_step=3, streams='both'",
       approx(out_fast.edit_scale, 1.0) and approx(out_fast.edit_audio_scale, 1.0)
       and out_fast.first_step == 0 and out_fast.last_step == 3 and out_fast.streams == "both"
       and out_fast._temporal_curve == "whole_clip")

    refuses("fast_1x EditPlan paired with spatial mask",
            lambda: eg.MiniMaxH3EditGuider.execute(
                _Patcher(), orig_c, edit_c, 1.5, 0, mask=spatial_64,
                edit_plan={"force_scale": 1.0, "manual_widgets": False},
            ),
            "cannot be combined with a spatial mask")

    out_manual = eg.MiniMaxH3EditGuider.execute(
        _Patcher(),
        orig_c,
        edit_c,
        1.75,
        2,
        last_step=5,
        streams="video_only",
        audio_scale=0.5,
        edit_plan={
            "first_step": 0,
            "last_step": 3,
            "streams": "audio_only",
            "audio_scale": 1.0,
            "temporal_curve": "ramp_in",
            "force_scale": 1.0,
            "manual_widgets": True,
        },
    ).result[0]
    ok("manual_widgets=True preserves manual MiniMaxH3EditGuider widget values",
       out_manual.first_step == 2 and out_manual.last_step == 5 and out_manual.streams == "video_only"
       and approx(out_manual.edit_scale, 1.75) and approx(out_manual.edit_audio_scale, 0.0)
       and out_manual._temporal_curve == "whole_clip")


def main() -> int:
    print("edit guidance: the prompt edit, the guider's arithmetic, pass marking and position alignment")
    check_prompt_edit()
    check_static_reader()
    check_guider_arithmetic()
    check_guider_mask()
    check_prepare()
    check_inputs()
    if failures:
        print(f"\n{len(failures)} problem(s)")
        return 1
    print("\nall ok -- the edit lands where it is told to, and the guider computes what it says")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
