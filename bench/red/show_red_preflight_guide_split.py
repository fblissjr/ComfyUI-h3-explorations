"""Show `bench/preflight_graph.py`'s guide split red, for the right reason.

The subject grades a prompt against one of the release's TWO prompt guides,
chosen from the graph. Before 2026-08-21 it knew only ref-en's six sections and
only `MiniMaxH3ReferenceToVideo`, so every base-format graph came back
`no MiniMaxH3ReferenceToVideo; nothing to grade` -- a clean line that looks
exactly like a pass, over the graphs this repo now ships. Widening it is what
makes a harness necessary: the new rules all pass on every shipped graph, and a
rule that has only ever been seen passing is a rule nobody has seen work.

Every case mutates an in-memory copy, so no graph on disk is touched. The
verdict is FAIL-only: the ref-en word budget is a WARN and several shipped
prompts carry it, which would drown the signal.

Two cases are NEAR_MISS and they are the point of the file, not padding:

  the word budget must NOT reach a base prompt   ref-en.txt:242 has no
    counterpart in base-en, so a 10-word base prompt must stay green. If it
    goes red, the split is cosmetic and the subject is still grading base
    prompts against the wrong guide.
  a keyframe's own <Picture 1> must NOT be flagged   `wired_labels` counted
    only `ref_*` sockets, so a correct keyframe prompt was reported as naming a
    label no socket wires. That false failure is what this case pins down.
"""
import copy
import json

from harness import Harness, MUTATION, NEAR_MISS, REPO, main, subject

PF = subject("bench/preflight_graph.py")

T2V = "h3_probe_split_base_first_api.json"      # base guide, no keyframe
KEYFRAME = "h3_first_frame_to_video_api.json"   # base guide, first_frame wired
FL2V = "h3_first_last_frame_to_video_api.json"  # both keyframes wired
MAIN = "integrated_multimodal_description"


def _graph(name, cls="MiniMaxH3Conditioning"):
    wf = json.loads((REPO / "workflows" / name).read_text())
    nid = next(k for k, n in wf.items() if n.get("class_type") == cls)
    return wf, nid


def _fails(wf, nid):
    return [m for lvl, m in PF.grade(wf[nid], wf, "") if lvl == "FAIL"]


def _mutate(name, fn):
    """A case that rewrites the prompt of a fresh copy, then grades it."""
    def run():
        wf, nid = _graph(name)
        wf[nid]["inputs"]["prompt"] = fn(wf[nid]["inputs"]["prompt"])
        return _fails(wf, nid)
    return run


def _drop_section(prompt):
    return prompt.replace("overall_soundscape:", "ambience:")


def _reorder(prompt):
    i, j = prompt.index("overall_soundscape:"), prompt.index("non_diegetic_music:")
    return prompt[:i] + prompt[j:] + "\n\n" + prompt[i:j]


def _add_instruction(prompt):
    return "For the target video, <Picture 1> is fully referenced.\n\n" + prompt


def _strip_instruction(prompt):
    return prompt[prompt.index(MAIN + ":"):]


def _replace_instruction(prompt, instruction):
    return instruction + "\n\n" + prompt[prompt.index(MAIN + ":"):]


def _linked_core_fl2v(*, prompt_mutation=lambda value: value,
                      duration_seconds=15.0):
    """HF-shaped core node: linked prompt plus duration math expression."""
    wf, nid = _graph(FL2V)
    prompt = prompt_mutation(wf[nid]["inputs"]["prompt"])
    wf[nid]["class_type"] = "MiniMaxH3ImageToVideo"
    wf[nid]["inputs"]["prompt"] = ["900", 0]
    wf[nid]["inputs"]["length"] = ["901", 1]
    wf["900"] = {
        "class_type": "PrimitiveStringMultiline",
        "inputs": {"value": prompt},
    }
    wf["901"] = {
        "class_type": "ComfyMathExpression",
        "inputs": {
            "expression": (
                "max(5, round(a * 24)) + "
                "(5 - (max(5, round(a * 24)) % 17)) % 17"
            ),
            "values.a": ["902", 0],
        },
    }
    wf["902"] = {
        "class_type": "PrimitiveFloat",
        "inputs": {"value": duration_seconds},
    }
    return _fails(wf, nid)


def build():
    h = Harness("bench/preflight_graph.py")
    h.fixture(REPO / "workflows" / T2V, "the base-format t2v graph under test")
    h.fixture(REPO / "workflows" / KEYFRAME, "the keyframe graph under test")
    h.fixture(REPO / "workflows" / FL2V, "the FL2VA graph under test")
    h.baseline(lambda: _fails(*_graph(T2V)))

    h.case("base: a core field renamed away", MUTATION,
           _mutate(T2V, _drop_section))
    h.case("base: core fields out of guide order", MUTATION,
           _mutate(T2V, _reorder))
    h.case("T2VA: an alignment instruction base-en.txt:14 forbids", MUTATION,
           _mutate(T2V, _add_instruction))
    h.case("keyframe: alignment instruction deleted (base-en.txt:19-29)",
           MUTATION, _mutate(KEYFRAME, _strip_instruction))
    h.case("keyframe: names <Picture 2> with one frame wired", MUTATION,
           _mutate(KEYFRAME, lambda p: p.replace("<Picture 1>", "<Picture 2>")))
    h.case("FL2VA: I2VA's plausible alignment sentence", MUTATION,
           _mutate(FL2V, lambda p: _replace_instruction(
               p, PF.BASE_ALIGNMENT["I2VA"])))
    h.case("FL2VA: duration differs from the graph", MUTATION,
           _mutate(FL2V, lambda p: p.replace(
               "15.08-second mark", "15.07-second mark", 1)))
    h.case("FL2VA: final-shot placeholder resolves to the wrong shot", MUTATION,
           _mutate(FL2V, lambda p: p.replace(
               "[Shot 1] Live-action", "[Shot 2] Live-action", 1)))
    h.case("HF core: linked prompt loses a required section", MUTATION,
           lambda: _linked_core_fl2v(prompt_mutation=_drop_section))
    h.case("HF core: duration primitive disagrees with alignment", MUTATION,
           lambda: _linked_core_fl2v(duration_seconds=14.0))

    # The two that must NOT move. See the module docstring.
    h.case("base: a 10-word prompt, far outside ref-en's 350-500", NEAR_MISS,
           _mutate(T2V, lambda _p: (
               f"{MAIN}: [Shot 1] A short line.\n\n"
               "overall_soundscape: Quiet.\n\nnon_diegetic_music: N/A")))
    h.case("keyframe: its own correct <Picture 1>", NEAR_MISS,
           lambda: _fails(*_graph(KEYFRAME)))
    h.case("FL2VA: exact guide line with graph-derived N and S.SS", NEAR_MISS,
           lambda: _fails(*_graph(FL2V)))
    h.case("HF core: linked prompt and duration expression", NEAR_MISS,
           _linked_core_fl2v)
    return h


if __name__ == "__main__":
    main(build)
