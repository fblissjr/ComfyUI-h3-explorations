#!/usr/bin/env python3
"""Does routing `<d>` to its real token id change what the encoder produces?

Run it with the ComfyUI venv python and a free GPU (`docs/comfy_notes.md`).
Loads the H3 text encoder; no sampler, no DiT, no VAE.

**Why this is not a render.** `CLAUDE.md`'s different-sample rule: two arms that
differ in their conditioning produce two different samples, not a good one and a
degraded one. A dialogue prompt rendered with and without the markers routed
would diverge completely and answer nothing. So this compares at the encoder
output, where the arms are comparable by construction.

Three arms on the same prompt, same weights throughout:

    vendor    <d> and </d> as ids 151669 / 151670, what the release emits
    comfy     the BPE fragments ComfyUI emits today
    stripped  the markers deleted, the words left alone

**`stripped` is not decoration; it is the scale.** Any change to a token
sequence moves the hidden states somewhat, so `vendor` against `comfy` is
uninterpretable on its own. `stripped` says what "the marker is simply not
there" costs. If comfy sits near vendor, its fragments are already carrying the
delimiter and the fix is cosmetic. If comfy sits near stripped, ComfyUI is
effectively not marking dialogue at all.

**Two controls, and the run is void without them.**

1. *Determinism.* The vendor arm is encoded twice and the two must be
   bit-identical. A non-deterministic forward makes every number below noise,
   and this is the cheapest way to find that out.
2. *A marker-free prompt.* Tokenized through both tokenizers it must produce
   identical ids and therefore identical states. If it does not, the harness is
   measuring something other than the markers and nothing else in the run
   stands.

**What this cannot answer.** Whether the DiT cares. The encoder is frozen and
the marker's embedding row is untrained
(`bench/audit_h3_token_embeddings.py`), so the encoder is not where any learned
meaning would live. This is the cheap gate in front of that question: if the
representations barely move, there is nothing downstream to chase.

Alignment is by `difflib` over the id sequences rather than by position. The
arms have different lengths -- ComfyUI emits one more token than the release on
the shipped dialogue line -- so a positional diff would compare unrelated
tokens and report a large difference for the wrong reason.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
# insert(0): `comfy_extras.nodes_minimax_h3` does a bare `import nodes`, and
# this repo's own nodes.py would win from position 0. Same trap preflight
# documents.
_COMFY = Path.home() / "ComfyUI"
sys.path.insert(0, str(_COMFY))

RELEASE_TOKENIZER = _REPO / "coderef" / "MiniMax-H3" / "tokenizer"
ENCODER = (Path.home() / "ComfyUI" / "models" / "text_encoders"
           / "qwen3vl_32b_minimax_h3_int8_convrot.safetensors")

# int8_convrot rather than nvfp4_awq on purpose: nvfp4 stores an INT8 embedding
# table, which perturbs the initial vector of every literal token. That is
# noise on exactly the quantity being measured. int8_convrot keeps the BF16
# table, so the arms differ only in which ids were emitted.

DIALOGUE = ("The woman turns to the camera and says, "
            "<d>[English] We need to leave now.</d> "
            "She walks away toward the door without looking back.")

# The line `workflows/build_workflows.py` writes into the shipped voice graph,
# in its surrounding sentence. Two prompts because a single one cannot show
# whether the effect is prompt-shaped.
SHIPPED = ("<Subject 1> (S1) turns toward the camera and says, in the clear "
           "timbre referenced from <Audio 1>, "
           "<d>[English] I thought you would have gone by now.</d>")

# Fourteen marker pairs in the base guide's three-field t2v format, which is
# where the markers land earliest. Built to be the strongest stressor an
# on-format prompt can be: a space before every `<d>` (three BPE pieces, where
# a newline costs none), punctuation before every `</d>` as the guide requires,
# and short alternating lines so marker density is as high as prose allows.
STRESS = "integrated_multimodal_description: [Shot 1] Live-action, cinematic, handheld on 35mm, a medium two-shot frames a woman with dark shoulder-length hair in a charcoal wool coat facing a man with a close-cropped beard in a navy jacket on a concrete stairwell landing, lit hard from a caged bulb overhead. The woman with the low measured voice (S1) says, <d>[English] You said tomorrow.</d> The man with the lower gravelled voice (S2) answers, <d>[English] It moved.</d> She says, <d>[English] Moved to when?</d> He says, <d>[English] Tonight.</d> She says, <d>[English] That is not possible.</d> He says, <d>[English] I know.</d> She shifts her weight back half a step and her jaw tightens while the camera drifts a few degrees with the operator's breathing. [Shot 2] At 00:04.000, the camera cuts to a close-up of the woman against painted cinderblock. She (S1) says, <d>[English] Who else knows?</d> Off screen he (S2) says, <d>[English] Nobody yet.</d> She says, <d>[English] Keep it that way.</d> She looks past the camera toward the stairs below. [Shot 3] At 00:08.000, the camera cuts to a close-up of the man, the caged bulb throwing a hard edge across his cheek. He (S2) says, <d>[English] I cannot promise that.</d> Off screen she (S1) says, <d>[English] Then do not.</d> He says, <d>[English] Fine.</d> He adjusts the satchel strap on his shoulder and looks down. [Shot 4] At 00:11.500, the camera returns to the medium two-shot. She (S1) says, <d>[English] We leave separately.</d> He (S2) says, <d>[English] Understood.</d> She turns and descends the stairs out of frame while he stays on the landing, watching the empty stairwell until the final frame.\n\noverall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant, a faint electrical hum from the caged bulb overhead, and footsteps on gritty concrete during the final descent. Two speaking voices trade short lines with almost no gap between them.\n\nnon_diegetic_music: N/A"

# The control: no markers anywhere, so both tokenizers must agree exactly.
CONTROL = ("A medium shot establishes the room, then the camera trucks right "
           "with small amplitude at slow speed.")

MARKERS = ("<d>", "</d>")


def _strip(text: str) -> str:
    for m in MARKERS:
        text = text.replace(m, "")
    return text


def _load():
    import comfy.sd
    from transformers import AutoTokenizer
    if not ENCODER.exists():
        raise SystemExit(f"encoder absent: {ENCODER.name}")
    if not RELEASE_TOKENIZER.exists():
        raise SystemExit("coderef/MiniMax-H3/tokenizer is absent (coderef is "
                         "gitignored; this needs the clone)")
    rel = AutoTokenizer.from_pretrained(str(RELEASE_TOKENIZER))
    clip = comfy.sd.load_clip(ckpt_paths=[str(ENCODER)])
    return rel, clip


def _comfy_ids(clip, text: str) -> list[int]:
    """The ids ComfyUI's own path emits, not a reimplementation of it."""
    entries = clip.tokenize(text)["qwen3vl_32b"][0]
    return [int(t[0]) for t in entries]


def _encode(clip, ids: list[int]):
    """[L, 5120] layer-50 states for a literal id sequence."""
    tokens = {"qwen3vl_32b": [[(int(i), 1.0) for i in ids]]}
    cond = clip.encode_from_tokens(tokens)
    return cond[0].float().cpu()


def _aligned_delta(ids_a, ids_b, ha, hb):
    """Relative L2 between two arms over the token spans whose ids agree.

    Returns (mean relative L2, aligned position count). Positions whose ids
    differ are excluded: comparing the encoder's output for `<d>` against its
    output for `'<d'` is not a question anyone asked.
    """
    import torch
    sm = difflib.SequenceMatcher(a=ids_a, b=ids_b, autojunk=False)
    ia, ib = [], []
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            ia.append(blk.a + k)
            ib.append(blk.b + k)
    if not ia:
        return None, 0
    va, vb = ha[ia], hb[ib]
    num = torch.linalg.vector_norm(va - vb, dim=-1)
    den = torch.linalg.vector_norm(va, dim=-1).clamp_min(1e-12)
    return float((num / den).mean()), len(ia)


def _run_prompt(label, text, rel, clip, record):
    print(f"\n=== {label}")
    arms = {
        "vendor": rel(text, add_special_tokens=False)["input_ids"],
        "comfy": _comfy_ids(clip, text),
        "stripped": _comfy_ids(clip, _strip(text)),
    }
    for name, ids in arms.items():
        print(f"  {name:<9} {len(ids):>3} ids")

    states = {name: _encode(clip, ids) for name, ids in arms.items()}

    # Control 1: the same arm twice must be bit-identical.
    again = _encode(clip, arms["vendor"])
    deterministic = bool((again == states["vendor"]).all())
    print(f"  determinism control: {'holds' if deterministic else 'FAILED'}")

    out = {"ids": {k: len(v) for k, v in arms.items()},
           "deterministic": deterministic, "deltas": {}}
    pairs = (("vendor", "comfy"), ("vendor", "stripped"), ("comfy", "stripped"))
    for a, b in pairs:
        d, n = _aligned_delta(arms[a], arms[b], states[a], states[b])
        out["deltas"][f"{a}_vs_{b}"] = {"rel_l2": d, "aligned_positions": n}
        print(f"  {a:>8} vs {b:<9} rel L2 {d:.5f}   over {n} aligned positions")

    vc = out["deltas"]["vendor_vs_comfy"]["rel_l2"]
    vs = out["deltas"]["vendor_vs_stripped"]["rel_l2"]
    if vs and vs > 0:
        frac = vc / vs
        out["comfy_share_of_missing_marker"] = frac
        verdict = ("comfy is close to the release; its fragments carry most of "
                   "the delimiter" if frac < 0.25 else
                   "comfy is close to having no marker at all" if frac > 0.75
                   else "comfy sits between the two; read the numbers")
        print(f"  -> vendor-vs-comfy is {frac:.2f}x vendor-vs-stripped: {verdict}")
        out["verdict"] = verdict
    record[label] = out


def main() -> int:
    import torch  # noqa: F401  # fail fast if the venv is wrong
    rel, clip = _load()
    record = {}

    # Control 2, first, because a failure here voids everything after it.
    print("=== control: a prompt with no markers")
    c_rel = rel(CONTROL, add_special_tokens=False)["input_ids"]
    c_cfy = _comfy_ids(clip, CONTROL)
    ids_match = c_rel == c_cfy
    print(f"  ids identical across tokenizers: {ids_match}")
    if ids_match:
        s1, s2 = _encode(clip, c_rel), _encode(clip, c_cfy)
        states_match = bool((s1 == s2).all())
        print(f"  states identical: {states_match}")
    else:
        states_match = False
        print("  CONTROL FAILED: the tokenizers disagree on a marker-free "
              "prompt, so this harness is not measuring the markers")
    record["control"] = {"ids_identical": ids_match,
                         "states_identical": states_match}
    if not (ids_match and states_match):
        print("\ncontrol failed; not reporting arm deltas")
        return 1

    _run_prompt("dialogue", DIALOGUE, rel, clip, record)
    _run_prompt("shipped_voice_line", SHIPPED, rel, clip, record)
    _run_prompt("t2v_stress_14_pairs", STRESS, rel, clip, record)

    out = _REPO / "bench" / "results" / "2026-08-21_h3_marker_token_states.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
