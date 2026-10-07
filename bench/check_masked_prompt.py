#!/usr/bin/env python3
"""The masked lane's prompt node: what it writes, and that the graders read the same text.

`masked_prompt_text.py` holds the sentences and `masked_prompt.py` the node.
Each case is a way a render could run on a text nobody meant:

  bank_copies_are_what_it_writes  the bank holds a copy of the node's text
                                  for each combination in `BANK_COPIES`, one
                                  of them the text the motion arms measured.
                                  A sentence edited in the module without
                                  the copy, or the copy without the module,
                                  is red. `--write` rewrites the copies.
  every_combination_is_well_formed  all the combinations of the choices: six
                                  sections in order, no unfilled slot,
                                  `<Video 1>` named exactly when a motion
                                  reference is on, `(S1)` exactly when the
                                  subject is the voice.
  extra_lands_in_the_shot         `add_to_shot` appears once, on the shot's line,
                                  on one line whatever whitespace it held.
  subject_is_the_users_words      `subject` lands once, in the subject's
                                  definition, as one phrase whatever article,
                                  punctuation and whitespace it came with;
                                  an empty one is "person"; the first man or
                                  woman word in it sets the pronouns; a brace
                                  or a list placeholder passes through.
  summary_says_what_was_worked_out  the lines the node shows above the text
                                  name the pronouns and the word that
                                  decided them, where `picture_gives` came
                                  from, and where the movement comes from.
  wired_parts_must_be_named       the Masked Source's `the wired parts` can
                                  be any part, so following it is refused
                                  with the input to set named.
  copies_match_the_masked_source  the module's copies of the Masked
                                  Source's `replace` values and its "no
                                  motion reference" value equal
                                  `video_mask.py`'s.
  config_is_the_defaults          `h3_config.MASKED_PROMPT` is the module's
                                  default choices.
  graders_read_what_the_node_writes  `workflows/prompts.py::carriers`, which
                                  every grader goes through, resolves a song
                                  node's prompt through the node, reading
                                  `replace` and `motion_reference` off the
                                  Masked Source it is wired to.
  node_reads_the_source           the node itself: the Masked Source's
                                  output carries `replace` and
                                  `motion_reference`, and the node's text
                                  follows them.
  shipped_graphs_wire_the_node    every shipped graph whose song node takes
                                  a Masked Source takes its prompt from a
                                  Masked Prompt node wired to that same
                                  source, and the text is a bank entry.

What this cannot check: that any text renders well. The module's docstring
says which sentences have rendered and which have not. No model, no CUDA,
no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_masked_prompt.py [--write]
"""

from __future__ import annotations

import ast
import importlib
import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "workflows"))

from _lib import case, finish  # noqa: E402

import h3_config  # noqa: E402
import prompts  # noqa: E402

m = prompts._masked_text()

SECTIONS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:",
            "overall_soundscape:", "non_diegetic_music:")
MOTION_ON = "subject only"
#: bank id -> the choices whose text the bank holds a copy of. The first is
#: the text the motion arms rendered (bench/results/2026-10-05_masked_v2v_motion_arms.md).
BANK_COPIES = {
    "ref2va_masked_subject_motion": dict(subject=m.SUBJECT_MAN, motion_reference=MOTION_ON),
    "ref2va_masked_person_swap": dict(),
    "ref2va_masked_person_motion": dict(motion_reference=MOTION_ON),
    "ref2va_masked_person_head": dict(replace=m.REPLACE_PART),
    "ref2va_masked_person_upper_motion": dict(picture_gives=m.GIVES_UPPER, replace=m.REPLACE_PARTS,
                                              motion_reference=MOTION_ON),
    "ref2va_masked_person_silent": dict(voice=m.VOICE_SILENT),
}
SONG, SOURCE, NODE = "MiniMaxH3AudioFreezeSong", "MiniMaxH3MaskedSource", h3_config.MASKED_PROMPT_NODE


def bank_copies_are_what_it_writes():
    for pid, choices in BANK_COPIES.items():
        assert prompts.text(pid) == m.assemble(**choices), f"prompt_bank/{pid}.txt is not what the node writes for {choices}"
    return f"{len(BANK_COPIES)} copies"


def every_combination_is_well_formed():
    count = 0
    for subject, voice, gives, motion in itertools.product((m.SUBJECT_PERSON, m.SUBJECT_MAN, m.SUBJECT_WOMAN, "man wearing a red cap"),
                                                         m.VOICES, m.ROLES, (None, m.MOTION_NONE, MOTION_ON)):
        text = m.assemble(subject, voice, gives, motion_reference=motion)
        what = (subject, voice, gives, motion)
        at = [text.find(name) for name in SECTIONS]
        assert -1 not in at and at == sorted(at), f"{what}: sections missing or out of order"
        assert "{" not in text and "}" not in text, f"{what}: an unfilled slot"
        assert ("<Video 1>" in text) == (motion == MOTION_ON), f"{what}: <Video 1> named without a reference, or not named with one"
        assert ("(S1)" in text) == (voice == m.VOICE_MAIN), f"{what}: the speaker tag does not follow the voice choice"
        assert f"is the {subject} shown in <Picture 1>" in text, f"{what}: the subject"
        assert text == text.strip() and "  " not in text, f"{what}: stray whitespace"
        count += 1
    return f"{count} combinations"


def extra_lands_in_the_shot():
    added = "<Subject 1> moves in step with\n  the people on either side. "
    text = m.assemble(add_to_shot=added)
    want = "<Subject 1> moves in step with the people on either side."
    assert text.count(want) == 1
    line = next(ln for ln in text.split("\n") if ln.startswith("[Shot 1]"))
    assert want in line and line.index(want) < line.index("Everyone and everything else")
    assert m.assemble(add_to_shot="  \n ") == m.assemble()


def subject_is_the_users_words():
    for gives in m.ROLES:
        text = m.assemble(subject=" A blonde haired woman,\n wearing a grey T-shirt. ", picture_gives=gives)
        want = "<Subject 1> is the blonde haired woman, wearing a grey T-shirt shown in <Picture 1>, preserving her"
        assert text.count(want) == 1 and text.count("blonde haired") == 1, gives
        assert m.assemble(subject=" \n", picture_gives=gives) == m.assemble(picture_gives=gives)
        assert m.assemble(subject="a person", picture_gives=gives) == m.assemble(picture_gives=gives)
    # the first man or woman word decides the pronouns, and none means "their"
    for subject, want in (("man wearing a red cap", "his"), ("Woman in a man's hat", "her"), ("lady", "her"),
                          ("tall boy", "his"), ("person in a mantle", "their"), ("singer", "their")):
        assert m.pronouns(subject)[0] == want, (subject, m.pronouns(subject))
        assert f"preserving {want} facial identity" in m.assemble(subject=subject)
    # the user's words are not a template: a brace or a list placeholder passes through as typed
    assert "is the {a} __who__ shown in" in m.assemble(subject="{a} __who__")
    assert "He {waves} __x__." in m.assemble(add_to_shot="He {waves} __x__.")


def summary_says_what_was_worked_out():
    plain = m.summary()
    assert "pronouns: their" in plain and "no Masked Source wired" in plain and "added to the shot: nothing" in plain
    wired = m.summary("blonde haired woman", replace=m.REPLACE_PART, motion_reference=MOTION_ON, add_to_shot="She waves.")
    for want in ('"the blonde haired woman shown in <Picture 1>"', 'pronouns: her (from "woman")',
                 "the still gives: the head and hair (read from the Masked Source's `replace`)",
                 "movement: from <Video 1>", "added to the shot: She waves."):
        assert want in wired, (want, wired)
    assert "(set here)" in m.summary(picture_gives=m.GIVES_HEAD, replace=m.REPLACE_PARTS)


def wired_parts_must_be_named():
    try:
        m.assemble(replace=m.REPLACE_PARTS)
    except ValueError as e:
        assert "picture_gives" in str(e), e
    else:
        raise AssertionError("following `the wired parts` wrote a text")
    assert m.assemble(replace=m.REPLACE_PARTS, picture_gives=m.GIVES_HEAD) == m.assemble(replace=m.REPLACE_PART)


def upper_body_role_keeps_the_legs():
    # chosen here, never followed: the Masked Source has no `replace` that means the upper body
    assert m.GIVES_UPPER in m.GIVES and m.resolve_gives(m.GIVES_UPPER, m.REPLACE_PARTS) == m.GIVES_UPPER
    for replace in (None, m.REPLACE_WHOLE, m.REPLACE_PART):
        assert m.resolve_gives(m.GIVES_FOLLOW, replace) != m.GIVES_UPPER, replace
    moving = m.assemble("man", picture_gives=m.GIVES_UPPER, replace=m.REPLACE_PARTS, motion_reference=MOTION_ON)
    still = m.assemble("man", picture_gives=m.GIVES_UPPER, replace=m.REPLACE_PARTS, motion_reference=m.MOTION_NONE)
    for text in (moving, still):
        for want in ("From the waist up that person is <Subject 1>", "on that person's own legs",
                     "the legs, what is worn below the waist and the setting are the scene's own",
                     "fully_preserved"):
            assert text.count(want) == 1, want
    # with no <Video 1> the sentence that ties the upper body to the kept legs stands where MOVES would
    unreferenced = m.ROLES[m.GIVES_UPPER]["unreferenced"]
    assert unreferenced in still and m.MOVES not in still
    assert m.MOVES in moving and unreferenced not in moving
    # from the waist up the performance is the whole person's, both voices
    assert m.ROLES[m.GIVES_UPPER]["performance"] is m.ROLES[m.GIVES_WHOLE]["performance"]
    assert "the head and upper body (set here)" in m.summary(picture_gives=m.GIVES_UPPER, replace=m.REPLACE_PARTS)


def copies_match_the_masked_source():
    # read from the source, so this check needs neither torch nor ComfyUI for it
    tree = ast.parse((REPO / "video_mask.py").read_text(encoding="utf-8"))
    theirs = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant)}
    for name in ("REPLACE_WHOLE", "REPLACE_PART", "REPLACE_PARTS", "MOTION_NONE"):
        assert theirs.get(name) == getattr(m, name), f"{name}: video_mask.py has {theirs.get(name)!r}, the prompt module {getattr(m, name)!r}"


def config_is_the_defaults():
    assert h3_config.MASKED_PROMPT == dict(subject=m.SUBJECT_PERSON, voice=m.VOICE_MAIN,
                                           picture_gives=m.GIVES_FOLLOW, add_to_shot=""), h3_config.MASKED_PROMPT
    assert m.assemble(**h3_config.MASKED_PROMPT) == m.assemble()


def _graph(source_inputs: dict | None, **node_inputs) -> dict:
    g = {"74": {"class_type": SONG, "inputs": {"prompt": ["106", 0]}},
         "106": {"class_type": NODE, "inputs": {**h3_config.MASKED_PROMPT, **node_inputs}}}
    if source_inputs is not None:
        g["104"] = {"class_type": SOURCE, "inputs": source_inputs}
        g["106"]["inputs"]["source"] = ["104", 0]
    return g


def _read(graph: dict) -> str:
    (carrier,) = prompts.carriers(graph)
    return carrier.text


def graders_read_what_the_node_writes():
    assert _read(_graph(None)) == m.assemble()
    assert _read(_graph(dict(h3_config.MASKED_SOURCE))) == m.assemble()
    assert _read(_graph(dict(h3_config.MASKED_MOTION_SOURCE))) == m.assemble(motion_reference=MOTION_ON)
    assert _read(_graph(dict(h3_config.MASKED_SOURCE, replace=m.REPLACE_PART))) == m.assemble(replace=m.REPLACE_PART)
    assert _read(_graph(dict(h3_config.MASKED_SOURCE), subject="blonde haired woman", add_to_shot="She waves.")) == \
        m.assemble(subject="blonde haired woman", add_to_shot="She waves.")
    # a combination the node refuses is reported as unresolved, never as some other text
    (refused,) = prompts.carriers(_graph(dict(h3_config.MASKED_SOURCE, replace=m.REPLACE_PARTS)))
    assert refused.texts == () and refused.note, refused
    # and the record side identifies it
    described = prompts.describe(_graph(dict(h3_config.MASKED_SOURCE)))
    assert described["prompt_id"] == "ref2va_masked_person_swap", described


def node_reads_the_source():
    import torch
    # the tracker's check puts ComfyUI on its CPU path and makes the stand-in package
    import check_subject_track  # noqa: F401
    vm = importlib.import_module("_h3pack.video_mask")
    node = importlib.import_module("_h3pack.masked_prompt").MiniMaxH3MaskedPrompt
    frames, mask = torch.zeros(2, 64, 64, 3), torch.ones(2, 64, 64)
    plain = vm.MiniMaxH3MaskedSource.execute(frames, mask, reuse_mask=False).args[0]
    moving = vm.MiniMaxH3MaskedSource.execute(frames, mask, reuse_mask=False, motion_reference=vm.MOTION_SUBJECT,
                                              motion_short_edge=32).args[0]
    assert plain["replace"] == vm.REPLACE_WHOLE and moving["motion_reference"] == vm.MOTION_SUBJECT
    assert node.execute(source=plain).args[0] == m.assemble()
    assert node.execute(source=moving).args[0] == m.assemble(motion_reference=MOTION_ON)
    assert node.execute().args[0] == m.assemble()
    assert node.execute(source=dict(plain, replace=vm.REPLACE_PART)).args[0] == m.assemble(replace=m.REPLACE_PART)
    assert node.execute(source=plain, subject="man in a red cap").args[0] == m.assemble(subject="man in a red cap")
    # the Masked Source's warning about the part mask is shown above the prompt and never enters it
    pc = importlib.import_module("_h3pack.part_coverage")
    warned = dict(plain, **{pc.RECORD_KEY: "the part mask is in doubt on 3 of 9 frames"})
    quiet, loud = node.execute(source=plain), node.execute(source=warned)
    assert loud.args[0] == quiet.args[0] == m.assemble(), "the warning changed the prompt"
    shown_quiet, shown_loud = ("\n".join(str(v) for v in out.ui.as_dict()["text"]) for out in (quiet, loud))
    assert "the Masked Source warns: the part mask is in doubt on 3 of 9 frames" in shown_loud, shown_loud
    assert "the Masked Source warns" not in shown_quiet, shown_quiet
    assert shown_loud.index("the Masked Source warns") < shown_loud.index(m.assemble()[:40]), "the warning is not above the prompt"
    schema = node.define_schema()
    defaults = {i.id: getattr(i, "default", None) for i in schema.inputs if i.id != "source"}
    assert defaults == h3_config.MASKED_PROMPT, defaults


def shipped_graphs_wire_the_node():
    wired = 0
    for path in h3_config.graph_paths(REPO / "workflows"):
        graph = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(graph, dict) or isinstance(graph.get("nodes"), list):
            continue
        for nid, node in graph.items():
            if not isinstance(node, dict) or node.get("class_type") != SONG:
                continue
            source = node["inputs"].get("source")
            if not (isinstance(source, list) and graph.get(str(source[0]), {}).get("class_type") == SOURCE):
                continue
            link = node["inputs"].get("prompt")
            writer = graph.get(str(link[0]), {}) if isinstance(link, list) else {}
            assert writer.get("class_type") == NODE, f"{path.name}: the masked song node's prompt is not a Masked Prompt node's"
            assert writer["inputs"].get("source") == source, f"{path.name}: the prompt node reads another source than the song node"
            (carrier,) = [c for c in prompts.carriers(graph) if c.node_id == str(nid)]
            assert prompts.identify(carrier.text), f"{path.name}: the node's text is not a bank entry"
            wired += 1
    assert wired, "no shipped graph wires a Masked Source into a song node: nothing was checked"
    return f"{wired} graph(s)"


def main() -> int:
    if "--write" in sys.argv[1:]:
        for pid, choices in BANK_COPIES.items():
            (prompts.BANK / f"{pid}.txt").write_text(m.assemble(**choices), encoding="utf-8")
            print(f"wrote prompt_bank/{pid}.txt")
        return 0
    for fn in (bank_copies_are_what_it_writes, every_combination_is_well_formed, extra_lands_in_the_shot,
               subject_is_the_users_words, summary_says_what_was_worked_out, wired_parts_must_be_named, upper_body_role_keeps_the_legs,
               copies_match_the_masked_source, config_is_the_defaults,
               graders_read_what_the_node_writes, node_reads_the_source, shipped_graphs_wire_the_node):
        case(fn.__name__, fn)
    return finish()


if __name__ == "__main__":
    sys.exit(main())
