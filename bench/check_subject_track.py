#!/usr/bin/env python3
"""Following one person across cuts: the node's own logic, with stand-ins for SAM 3.

`subject_track.py` is the module. It fails quietly in the ways that matter: a
missed cut carries a mask into a shot the subject is not in, a wrong match
replaces somebody else, and a shot left empty by mistake shows the original.
The model work is behind three callables (`detect`, `sign`, `track`), so each
item drives the module's own functions on made-up frames and masks.

1. **A cut is found and a lighting change is not.** Two different pictures
   back to back score above the default threshold; the same picture with its
   colour and brightness changed scores far below it. The control is the plain
   mean difference, which the lighting change must move more than the cut
   score does, or the gradient score is not buying anything.
2. **The shots cover every frame once**, whatever the cuts given, including a
   cut at 0, one past the end and a repeated one.
2a. **The phrase asks core for every person, not one.** Core's SAM 3 prompt
   parser reads a bare phrase as one detection and `name:N` as up to N.
   `counted` writes the second form, keeps a count the user wrote, and the
   check reads the result back through core's own parser. The control: core
   still reads a bare phrase as one.
3. **The pick is the one the rule names**: the largest mask, the mask nearest
   the frame's centre, the highest score. No detections is no pick.
3a. **The signature is taken from the head and shoulders.** `top_third`
   keeps the top third of the rows a mask covers; two people with different
   heads over the same clothes are alike on the whole mask and unlike on the
   top third.
4. **The subject is followed, and nobody else is.** With three people whose
   signatures are known, the picked one is taken in each shot they are in, a
   shot holding only the others is left empty, a subject who enters after a
   shot's first frame is still found and tracked back to the shot's start, and
   the picked shot is tracked both ways from the pick frame. The comparison is
   relative to the other people on the pick frame, who then score 0; picked on
   a frame with nobody else it is the plain similarity, and the report says
   which.
5. **The mask is one per frame, at the frames' size, and empty where the
   subject is absent.** A threshold nothing can pass leaves every shot but the
   picked one empty; nothing on the pick frame leaves everything empty and the
   report says so.
6. **The node declares the phrase and every threshold as inputs**, with the
   module's constants as their defaults, and four outputs with the mask
   first (the last is the shot table, `bench/check_shot_table.py`). It is
   not an output node and it declares a `MASK_VERSION` (the two things the
   kept-mask store needed, until that store was removed on 2026-10-10; the
   number is still what the job builder's corrections rule reads).

7. **A correction takes the person its number names on the tile, and nothing
   else moves.** `shot N: person K` seeds shot N from person K as
   `shot_table` numbers the people on the frame the shot's tile shows, where
   the detector's own order is a different one; `shot N: none` empties the
   shot. The frame each tile shows is the same with and without corrections,
   so a number read off a tile is the number that applies. A corrected shot
   is tracked once; the other shots are tracked as before; no corrections is
   the automatic result. A person the frame does not have, a shot the clip
   does not have, a shot named twice and a line that is not a correction are
   refused by name.
8. **A thing does not win the automatic pick.** An opening shot that is the
   longest and starts on a microphone alone, the shape of the one-person
   clip's first window: the microphone has no head, so its shot does not
   vote, the person is picked in another shot and taken in the opening from
   where they come in, and the report names what was left out. Two controls:
   with no head anywhere every favourite votes as before, and a frame the
   user names is taken as named.
9. **A picture inside bars keeps its cuts.** A clip padded at the sides, at
   the top and bottom, or both, as a 4:3 picture on a wide canvas is: the
   bars' edges are the same in every frame, and scored with the picture they
   pull every cut's score down until the automatic threshold finds none
   (measured on a real padded file, `bench/results/2026-10-06_bordered_source_cuts.md`).
   The score is taken on the picture, and the cut is found where the bare
   picture's is. Controls: a clip with no border scores exactly what the
   formula without the border step gives, so nothing moves for one; a strip
   that is flat in one shot and not in the next is picture, not border; bars
   with a little coding noise are still bars.

10. **A subject let go inside a shot is found again, and nobody else is
   taken for them.** One shot with no cut; the stand-in tracker cannot follow
   through a stretch where the subject is hidden, as core's tracker cannot.
   The track is seeded again on the first probed frame where the subject is
   back, judged against a gallery of the shot's own tracked frames, runs
   both ways over the empty run, and the frames they were hidden on stay
   empty; the report names where the track let go and where it was found,
   and the shot table lists the frames with no subject, the gallery's frames
   and what every probe scored. The same when the run is at the shot's
   start, searched backward from the tracked side. Controls: a subject who
   never comes back is searched for and not found, the person beside them is
   not taken, and the report says so; a track with no empty run detects and
   tracks exactly what it did before (items 4 and 7 above assert the calls);
   a tracker that keeps letting go is seeded again at most `REGAIN_MOST`
   times. On a clip with one person: the only person the detector returns
   after the loss is taken when they are the subject, and NOT taken when
   they are somebody else, which is what core's detector returned for a
   phrase on the card (2026-10-06); a thing with no head is nobody; and two
   people who both look like the subject, closer together than
   `REGAIN_MARGIN`, are a gap and not a guess.
11. **An earlier run's subject is recognised in a later load.** A run keeps
   a gallery of its picked shot's track, the shot table carries it, and it
   reads back as it was. Handed to a run on another load, it makes the
   pick: a person who is alone and the largest on the probe frame, whom the
   pick rule takes, is refused, and the subject is picked on the first
   frame looked at that shows them. Controls: without the gallery the rule
   does take that person; with it and no subject in the load, nobody is
   picked and every mask is empty; a table with no gallery, a missing file
   and text that is not a table are refused by name.

What this cannot check: that SAM 3's features tell real people apart, that
its tracker follows them, or that the text encoder loads. Those need the card;
`docs/research/masking/2026-10-04_mrhf.md` has what was measured.

No model, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_subject_track.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "nodes.py").exists() and (p / "workflows").is_dir())
COMFY = REPO.parent.parent
# a draft of the module checked beside its own copy of this file, before it replaces the repo's
MODULE = HERE / "subject_track.py" if (HERE / "subject_track.py").exists() else REPO / "subject_track.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import torch  # noqa: E402

import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True  # no CUDA context for a logic check; the sibling checks do the same


def _load():
    """`subject_track` as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(MODULE.parent), str(REPO)]   # a draft's own `shot_table.py` beside it comes first
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.subject_track", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.subject_track"] = module
    spec.loader.exec_module(module)
    return module


st = _load()

H, W = 72, 128


def _padded(picture: torch.Tensor, side: int = 0, top: int = 0, value: float = 0.0) -> torch.Tensor:
    """`picture` centred on a larger frame of one flat `value`: bars at the sides, the top and bottom, or both."""
    h, w, c = picture.shape
    out = torch.full((h + 2 * top, w + 2 * side, c), float(value))
    out[top:top + h, side:side + w] = picture
    return out


def _plain_cut_scores(frames: torch.Tensor) -> torch.Tensor:
    """The cut score over the whole frame, written out again: what `cut_scores` was before it looked for borders."""
    w, h = st.CUT_SIZE
    grey = frames[..., :3].to(torch.float32).mean(dim=-1, keepdim=True).movedim(-1, 1)
    small = torch.nn.functional.interpolate(grey, size=(h, w), mode="area")[:, 0]
    edges = (small[:, 1:, 1:] - small[:, 1:, :-1]).abs() + (small[:, 1:, 1:] - small[:, :-1, 1:]).abs()
    e = edges.flatten(1)
    e = e - e.mean(dim=1, keepdim=True)
    e = e / e.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return 1.0 - (e[1:] * e[:-1]).sum(dim=1)


def check_borders(problems):
    """A picture inside bars: the cut is found where the bare picture's is, and a clip without bars is untouched."""
    a, b = _picture(1, 6, 8), _picture(2, 9, 5)
    shots = [a, torch.roll(a, 2, dims=1), torch.roll(a, 4, dims=1), b, torch.roll(b, 2, dims=1), torch.roll(b, 4, dims=1)]
    bare = torch.stack(shots, dim=0)
    want = st.find_cuts(st.cut_scores(bare), st.auto_cuts(st.cut_scores(bare)))
    if want != [3]:
        problems.append(f"the bare stand-in's cut is {want}, not [3]: the bordered cases below have nothing to stand against")
        return
    for name, side, top in (("bars at the sides", W // 6, 0), ("bars at the top and bottom", 0, H // 6), ("bars all round", W // 8, H // 8)):
        frames = torch.stack([_padded(f, side, top) for f in shots], dim=0)
        seen: dict = {}
        scores = st.cut_scores(frames, seen) if _takes_found() else st.cut_scores(frames)
        got = st.find_cuts(scores, st.auto_cuts(scores))
        if got != want:
            problems.append(f"{name}: cuts {got} at the automatic threshold {st.auto_cuts(scores):.2f}, not {want} "
                            f"(scores {[round(float(v), 2) for v in scores]}); the bars' edges are in every frame and "
                            "must not be scored")
        t, bt, l, r = seen.get("borders", (0, 0, 0, 0))
        if (bool(l and r), bool(t and bt)) != (bool(side), bool(top)):
            problems.append(f"{name}: borders found are top {t}, bottom {bt}, left {l}, right {r} (rows and columns at "
                            "the cut score's size)")
    # nothing moves for a clip without borders: the same numbers as the formula with no border step
    relit = (a * torch.tensor([0.4, 1.0, 0.7]) + 0.15).clamp(0, 1)
    for name, frames in (("the bare stand-in", bare), ("a relit shot and a cut", torch.stack([a, a, relit, relit, b, b], dim=0))):
        if not torch.equal(st.cut_scores(frames), _plain_cut_scores(frames)):
            problems.append(f"{name} has no border and its scores are not the plain formula's: a clip without bars must "
                            "score exactly as before")
    if not _takes_found():
        problems.append("cut_scores does not report what it left out: it takes no second argument for the borders found")
        return
    # a strip that is flat in one shot and not in the next is picture
    dark = [f.clone() for f in shots]
    for f in dark[:3]:
        f[: H // 6] = 0.0
    seen = {}
    st.cut_scores(torch.stack(dark, dim=0), seen)
    if seen.get("borders") != (0, 0, 0, 0):
        problems.append(f"a strip that is black in the first shot only was taken as a border: {seen.get('borders')}")
    # bars with a little coding noise are still bars
    g = torch.Generator().manual_seed(7)
    noisy = torch.stack([_padded(f, W // 6, 0) for f in shots], dim=0)
    noisy = (noisy + (torch.rand(noisy.shape, generator=g) - 0.5) * (2.0 / 255.0)).clamp(0, 1)
    seen = {}
    scores = st.cut_scores(noisy, seen)
    if st.find_cuts(scores, st.auto_cuts(scores)) != want or not all(seen.get("borders", (0, 0, 0, 0))[2:]):
        problems.append(f"bars with coding noise: cuts {st.find_cuts(scores, st.auto_cuts(scores))}, borders {seen.get('borders')}")
    line = st.borders_line(seen, W + 2 * (W // 6), H)
    if "left" not in line or "right" not in line or "top" in line:
        problems.append(f"the report's line about borders does not name the sides left out: {line!r}")
    if st.borders_line({"borders": (0, 0, 0, 0)}, W, H) != "":
        problems.append("the report speaks of borders on a clip that has none")


def _takes_found() -> bool:
    import inspect
    return len(inspect.signature(st.cut_scores).parameters) >= 2


def _picture(seed: int, rows: int, cols: int) -> torch.Tensor:
    """A frame of blocks, [H, W, 3] in 0..1: structure a cut score can see.

    Two pictures on the same block grid share where their edges are, which is
    what a held shot looks like to the score, so a cut needs another grid.
    """
    g = torch.Generator().manual_seed(seed)
    small = torch.rand((1, 3, rows, cols), generator=g)
    return torch.nn.functional.interpolate(small, size=(H, W), mode="nearest")[0].movedim(0, -1)


def _box(x0: int, y0: int, x1: int, y1: int) -> torch.Tensor:
    m = torch.zeros((H, W), dtype=torch.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def check_cuts(problems):
    a, b = _picture(1, 6, 8), _picture(2, 9, 5)
    relit = (a * torch.tensor([0.4, 1.0, 0.7]) + 0.15).clamp(0, 1)      # same picture, another light
    frames = torch.stack([a, a, relit, relit, b, b], dim=0)
    scores = st.cut_scores(frames)
    if tuple(scores.shape) != (5,):
        problems.append(f"cut_scores returned {tuple(scores.shape)} for six frames; one score per step is five")
        return
    cuts = st.find_cuts(scores, st.CUT_THRESHOLD)
    if cuts != [4]:
        problems.append(f"cuts at the default threshold are {cuts}; the only cut is into frame 4 "
                        f"(scores {[round(float(s), 3) for s in scores]})")
    if not float(scores[1]) < 0.5 * st.CUT_THRESHOLD:
        problems.append(f"a lighting change scores {float(scores[1]):.3f}, not far below the threshold {st.CUT_THRESHOLD}")
    plain = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))
    if not float(plain[1]) / float(plain[3]) > float(scores[1]) / float(scores[3]):
        problems.append("the control failed: the plain difference is no more fooled by the lighting change than "
                        "the gradient score is, so the score buys nothing on this case")
    if st.find_cuts(st.cut_scores(frames[:1]), st.CUT_THRESHOLD) != []:
        problems.append("a one-frame clip has a cut")
    if st.find_cuts(torch.tensor([0.2, 0.9, 0.89]), 0.9) != [2]:
        problems.append("a step scoring exactly the threshold is not a cut: the comparison must be inclusive")
    # the threshold from the scores. The two lists are the highest steps of the two clips measured on 2026-10-04:
    # the band segment, whose cuts score 0.99 and above, and the one-person clip, two of whose cuts score under 0.9.
    low = [0.17] * 40
    band = [1.08, 1.02, 1.02, 1.01, 1.01, 1.01, 1.00, 0.99, 0.76, 0.71, 0.65, 0.64, 0.63] + low
    solo = [0.99, 0.98, 0.95, 0.93, 0.90, 0.84, 0.46, 0.46, 0.45, 0.44] + low
    for name, scores, n_cuts in (("the band clip", band, 8), ("the one-person clip", solo, 6)):
        at = st.auto_cuts(scores)
        found = len(st.find_cuts(torch.tensor(scores), at))
        if found != n_cuts:
            problems.append(f"the automatic cut threshold on {name}'s scores is {at:.2f} and finds {found} cut(s), not {n_cuts}")
    if len(st.find_cuts(torch.tensor(solo), st.CUT_THRESHOLD)) == 6:
        problems.append("the control failed: the fixed threshold finds all six cuts of the one-person clip, "
                        "so the automatic threshold buys nothing on this case")
    for scores, why in (([0.46, 0.41, 0.30, 0.05], "no step near a cut"), ([0.95, 0.93, 0.90, 0.86, 0.83, 0.79], "no clear gap"),
                        ([], "no steps")):
        if st.auto_cuts(scores) != st.CUT_THRESHOLD:
            problems.append(f"the automatic cut threshold is {st.auto_cuts(scores):.2f} with {why}; it falls back to {st.CUT_THRESHOLD}")
    if "0.99 0.95 | 0.40" not in st.cuts_line([0.4, 0.99, 0.95], 0.9, False) or "automatic, 0.90" not in st.cuts_line([0.4], 0.9, False):
        problems.append("the report's cut line does not print the steps in order with the threshold marked")


def check_ranges(problems):
    for n, cuts in ((10, []), (10, [4]), (10, [0, 4, 4, 10, 12]), (1, [])):
        ranges = st.shot_ranges(n, cuts)
        covered = [f for s, e in ranges for f in range(s, e)]
        if covered != list(range(n)):
            problems.append(f"shot_ranges({n}, {cuts}) = {ranges} does not cover each frame once")
    if st.shot_ranges(10, [4]) != [(0, 4), (4, 10)]:
        problems.append(f"shot_ranges(10, [4]) = {st.shot_ranges(10, [4])}")


def check_counted(problems):
    """The phrase reaches core asking for more than one detection, in core's own syntax."""
    from comfy.text_encoders.sam3_clip import _parse_prompts  # core's parser: the independent answer
    for phrase, most, want in (("person", 16, [("person", 16)]), ("person:3", 16, [("person", 3)]),
                               ("lead singer, drummer:2", 8, [("lead singer", 8), ("drummer", 2)])):
        got = _parse_prompts(st.counted(phrase, most))
        if got != want:
            problems.append(f"counted({phrase!r}, {most}) is read by core as {got}, not {want}")
    if _parse_prompts("person") != [("person", 1)]:
        problems.append("the control failed: core no longer reads a bare phrase as one detection, so `counted` "
                        "may not be needed and this item is not testing what it says")
    # one phrase, one detection: core's tokenizer encodes that text as typed, so it must reach it with no `:1`.
    # Driven through core's own tokenizer: the tokens of what `counted` writes are the tokens of the bare phrase.
    from comfy.text_encoders.sam3_clip import SAM3TokenizerWrapper
    tokenizer = SAM3TokenizerWrapper()
    bare = tokenizer.tokenize_with_weights("person")
    for phrase, most in (("person", 1), ("person:1", 16), ("person : 1", 4)):
        text = st.counted(phrase, most)
        if text != "person" or tokenizer.tokenize_with_weights(text) != bare:
            problems.append(f"counted({phrase!r}, {most}) is {text!r}: one phrase asking for one detection must reach core "
                            "bare, or core's tokenizer encodes the `:1` as part of the phrase")
    if tokenizer.tokenize_with_weights("person:1") == bare:
        problems.append("the control failed: core's tokenizer now drops the `:1` of a lone phrase itself, so the bare "
                        "form is no longer needed and this item is not testing what it says")
    if st.counted("lead singer:1, drummer", 1) != "lead singer:1, drummer:1":
        problems.append("two phrases asking for one each lost their counts: core encodes each part of a list separately, "
                        "and only a lone phrase goes bare")
    try:
        st.counted(" , ", 4)
    except ValueError:
        pass
    else:
        problems.append("an empty phrase was accepted")


def check_choose(problems):
    big, centre, corner = _box(0, 0, 60, 40), _box(56, 30, 72, 42), _box(110, 0, 126, 10)
    masks = torch.stack([corner, big, centre], dim=0)
    scores = [0.9, 0.6, 0.7]
    for pick, want in ((st.PICK_LARGEST, 1), (st.PICK_CENTRAL, 2), (st.PICK_SCORE, 0)):
        got = st.choose(masks, scores, pick)
        if got != want:
            problems.append(f"choose with `{pick}` took detection {got}; it is {want}")
    if st.choose(torch.zeros((0, H, W)), [], st.PICK_LARGEST) is not None:
        problems.append("choose picked something out of no detections")
    try:
        st.choose(masks, scores, "tallest")
    except ValueError:
        pass
    else:
        problems.append("choose accepted an unknown rule")


def _world():
    """Four shots of 8 frames. Person 0 is the subject; 1 and 2 are others.

    shot 0 (frames 0-7): 0 and 1.   shot 1 (8-15): 1 and 2 only.
    shot 2 (16-23): 0 enters at frame 20, with 2.   shot 3 (24-31): 0 alone.
    Each person has a fixed signature and a box that says who they are.
    """
    sigs = {0: torch.tensor([1.0, 0.0, 0.0]), 1: torch.tensor([0.0, 1.0, 0.0]), 2: torch.tensor([0.0, 0.0, 1.0])}
    boxes = {0: _box(40, 10, 80, 60), 1: _box(0, 20, 20, 50), 2: _box(100, 20, 120, 50)}
    def present(f: int) -> list[int]:
        if f < 8: return [1, 0]
        if f < 16: return [1, 2]
        if f < 24: return [2] + ([0] if f >= 20 else [])
        return [0]
    def who(mask: torch.Tensor) -> int:
        return next(p for p, b in boxes.items() if torch.equal(b, mask))
    calls = {"detect": [], "track": []}
    def detect(f: int):
        calls["detect"].append(f)
        people = present(f)
        if not people:
            return torch.zeros((0, H, W)), []
        return torch.stack([boxes[p] for p in people], dim=0), [0.9 - 0.1 * i for i in range(len(people))]
    def sign(_frame: int, mask: torch.Tensor):
        return sigs[who(mask)]
    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls["track"].append((start, end, seed, who(mask)))
        return mask[None].repeat(end - start, 1, 1)
    return boxes, detect, sign, track, calls


def check_follow(problems):
    boxes, detect, sign, track, calls = _world()
    cuts = [8, 16, 24]
    # a frame named, a value named; each shot is first judged `offset` frames in
    got = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    shots = got.shots
    if got.pick_frame != 3 or got.others != 1 or abs(got.match - 0.8) > 1e-9:
        problems.append(f"named frame 3 and value 0.8 gave pick_frame {got.pick_frame}, {got.others} other(s), cut {got.match}")
    want = [(0, 8, 3, 0), (16, 24, 20, 0), (24, 32, 25, 0)]
    if calls["track"] != want:
        problems.append(f"tracked {calls['track']}; the subject's shots, seeds and identity are {want}")
    if [s.seed for s in shots] != [3, None, 20, 25] or [s.probe for s in shots] != [1, 9, 17, 25]:
        problems.append(f"seeds {[s.seed for s in shots]} and probe frames {[s.probe for s in shots]}; shot 2 holds only "
                        "other people, shot 3's subject enters at frame 20, and each shot is first judged one frame in")
    # relative to the other person on the pick frame, that person scores 0 and a stranger 0.5: neither is the subject
    if not abs(shots[1].best - 0.5) < 1e-4:
        problems.append(f"the shot holding the two others has best similarity {shots[1].best:.3f}, not 0.5")
    mask = st.assemble(32, H, W, got.pieces)
    if tuple(mask.shape) != (32, H, W):
        problems.append(f"the mask is {tuple(mask.shape)}, not one per frame at the frames' size")
    if float(mask[8:16].sum()) != 0.0:
        problems.append("the shot the subject is not in has a mask")
    for f in (0, 7, 16, 23, 31):
        if not torch.equal(mask[f], boxes[0]):
            problems.append(f"frame {f} does not carry the subject's mask")
            break
    text = st.report(got, cuts, st.PICK_LARGEST, "person", True, True, 1.0,
                     cutting=st.cuts_line([0.99, 0.95, 0.4], 0.9, True))
    for need in ("cuts at frame(s) [8, 16, 24]", "relative to the 1 other", "absent (best similarity 0.50", "taken on frame 20",
                 "the value named, 0.80", "1.00 1.00 | 0.50", "cut threshold: the value named, 0.90"):
        if need not in text:
            problems.append(f"the report lacks {need!r}: {text!r}")
    tiles = st.preview(torch.rand((32, H, W, 3)), mask, shots, detect)
    if tiles.shape[0] != 4 or tiles.shape[2] != st.TILE_WIDTH or tiles.shape[3] != 3:
        problems.append(f"the preview is {tuple(tiles.shape)}, not one tile per shot at the tile width")
    if not (float(tiles.min()) >= 0.0 and float(tiles.max()) <= 1.0):
        problems.append("the preview leaves 0..1")


def check_corrections(problems):
    cuts = [8, 16, 24]

    def run(corrections):
        boxes, detect, sign, track, calls = _world()
        got = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, corrections=corrections)
        return boxes, got, calls, st.assemble(32, H, W, got.pieces)

    boxes, base, base_calls, base_mask = run(None)
    shown = [s.shown for s in base.shots]
    _, same, same_calls, same_mask = run({})
    if not torch.equal(same_mask, base_mask) or same_calls["track"] != base_calls["track"] or any(s.corrected for s in same.shots):
        problems.append("with no corrections the result is not the automatic one")

    # shot 3's tile shows frame 20, where the detector lists person 2 of the world first and the subject second;
    # left to right the subject is person 1 and the other is person 2
    _, got, calls, mask = run({3: 2})
    if [s.shown for s in got.shots] != shown:
        problems.append(f"a correction changed the frames the tiles show: {[s.shown for s in got.shots]}, not {shown}")
    if (16, 24, 20, 2) not in calls["track"] or sum(1 for c in calls["track"] if c[0] == 16) != 1:
        problems.append(f"`shot 3: person 2` tracked {[c for c in calls['track'] if c[0] == 16]}: the corrected shot is "
                        "tracked once, from the person the tile numbers 2 on the frame it shows")
    if not torch.equal(mask[16], boxes[2]) or not torch.equal(mask[:16], base_mask[:16]) or not torch.equal(mask[24:], base_mask[24:]):
        problems.append("`shot 3: person 2` did not put that person's mask on shot 3 and leave the other shots alone")
    s3 = got.shots[2]
    if (s3.corrected, s3.seed, s3.index, st._state(s3)) != ("person 2", 20, 0, "taken (corrected)"):
        problems.append(f"the corrected shot records {(s3.corrected, s3.seed, s3.index, st._state(s3))}")

    # a shot the automatic pass left empty, taken by hand
    _, got, calls, mask = run({2: 1})
    s2 = got.shots[1]
    if s2.seed != shown[1] or not torch.equal(mask[8], boxes[1]) or s2.corrected != "person 1":
        problems.append("`shot 2: person 1` did not take the leftmost person of the frame shot 2's tile shows")

    # a shot emptied by hand: not tracked at all, and its tile still shows who had been taken
    _, got, calls, mask = run({3: None})
    s3 = got.shots[2]
    if float(mask[16:24].sum()) != 0.0 or any(c[0] == 16 for c in calls["track"]):
        problems.append("`shot 3: none` left a mask on the shot, or tracked it first")
    if (s3.seed, s3.shown, s3.index, s3.corrected, st._state(s3)) != (None, shown[2], base.shots[2].index, "none", "absent (corrected)"):
        problems.append(f"the emptied shot records {(s3.seed, s3.shown, s3.index, s3.corrected, st._state(s3))}")

    # the picked shot corrected: the others are still matched against the pick
    _, got, calls, mask = run({1: 1})
    if not torch.equal(mask[0], boxes[1]) or sum(1 for c in calls["track"] if c[0] == 0) != 1 or not got.shots[0].picked:
        problems.append("`shot 1: person 1` on the picked shot did not take that person, or tracked the shot twice")
    if not torch.equal(mask[8:], base_mask[8:]):
        problems.append("correcting the picked shot changed how the other shots were matched")

    text = st.report(run({2: 2, 3: None})[1], cuts, st.PICK_LARGEST, "person", True, True, 1.0)
    for need in (f"[2] frames 8-15: corrected by hand, person 2 of the 2 detection(s) on frame {shown[1]}",
                 "[3] frames 16-23: corrected by hand, nobody taken"):
        if need not in text:
            problems.append(f"the report lacks {need!r}: {text!r}")

    try:
        run({2: 3})
        problems.append("a person the frame does not have was accepted")
    except ValueError as exc:
        if "2 person(s)" not in str(exc) or f"frame {shown[1]}" not in str(exc) or "person 3" not in str(exc):
            problems.append(f"the refusal of a missing person does not say how many there are and on which frame: {exc}")
    try:
        run({9: 1})
        problems.append("a correction for a shot the clip does not have was accepted")
    except ValueError as exc:
        if "shot 9" not in str(exc) or "4 shot(s)" not in str(exc):
            problems.append(f"the refusal of a missing shot does not name it: {exc}")

    read = st.parse_corrections("shot 2: person 1\n  Shot 3 = none ; shot 4 person 2\n\n", 4)
    if read != {2: 1, 3: None, 4: 2} or st.parse_corrections("", 4) != {} or st.parse_corrections(None, 4) != {}:
        problems.append(f"parse_corrections read {read}")
    for bad, said in (("shot 2", "`shot 2`"), ("person 2", "`person 2`"), ("shot 5: none", "4 shot(s)"),
                      ("shot 0: none", "shot 0"), ("shot 2: person 0", "person 0"),
                      ("shot 2: none\nshot 2: person 1", "twice"), ("shot two: none", "`shot two: none`")):
        try:
            st.parse_corrections(bad, 4)
            problems.append(f"the correction {bad!r} was accepted")
        except ValueError as exc:
            if said not in str(exc):
                problems.append(f"the refusal of {bad!r} does not say {said!r}: {exc}")


def check_automatic(problems):
    """Nothing named: the pick is the person the rule favours for most of the clip, the cut comes from the scores."""
    boxes, detect, sign, track, calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, None, None, detect, sign, track, stride=4, offset=1)
    # the largest person on each shot's probe frame: 0, 1 or 2 (a tie in size, the first), 2, 0. Person 0 wins the most frames
    if got.pick_frame != 1 or got.others != 1:
        problems.append(f"automatic pick took frame {got.pick_frame} with {got.others} other(s); person 0 is the largest in "
                        "shots 1 and 4, and of those frames the one showing the most people is frame 1")
    if [s.seed for s in got.shots] != [1, None, 20, 25]:
        problems.append(f"automatic: seeds {[s.seed for s in got.shots]}; the subject is in shots 1, 3 and 4")
    if not abs(got.match - st.MATCH_FLOOR) < 1e-9 and not (0.5 < got.match <= 1.0):
        problems.append(f"automatic cut is {got.match}")
    text = st.report(got, [8, 16, 24], st.PICK_LARGEST, "person", False, False, 1.0)
    if "chosen automatically" not in text or "match: automatic" not in text:
        problems.append(f"the report does not say the pick and the match were automatic: {text!r}")
    # the cut rule alone
    floor = st.MATCH_FLOOR
    for scores, want, why in (
            ([0.94, 0.92, 0.71, 0.71, 0.66, 0.51], (0.92 + 0.71) / 2, "a clear gap above the floor moves the cut to its middle"),
            ([0.94, 0.92, 0.90, 0.85], floor, "the subject in every shot: no clear gap, so everything above the floor"),
            ([0.71, 0.71, 0.66, 0.51, 0.46], floor, "the subject in no other shot: a gap among wrong people is below the floor"),
            ([0.93], floor, "one other shot"),
            ([], floor, "no other shot"),
            ([0.95, 0.83, 0.40], floor, "the widest gap decides, and here it lies below both: the two above the floor are kept"),
            ([0.95, 0.83, 0.78], (0.95 + 0.83) / 2, "the widest gap lies above the floor, so the cut moves up into it"),
            ([0.82, 0.40], floor, "a gap whose middle is under the floor does not lower the cut")):
        got_cut = st.auto_match(scores, floor)
        if abs(got_cut - want) > 1e-9:
            problems.append(f"auto_match({scores}) is {got_cut:.3f}, not {want:.3f}: {why}")
    # picked where the subject is alone: nothing to subtract, the plain floor applies
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 26, None, detect, sign, track, stride=4, offset=1)
    if got.others != 0 or abs(got.match - st.PLAIN_FLOOR) > 1e-9 or [s.seed for s in got.shots] != [1, None, 20, 26]:
        problems.append(f"picked on a frame with nobody else: {got.others} other(s), cut {got.match}, seeds "
                        f"{[s.seed for s in got.shots]}; the plain floor applies and the subject is in shots 1, 3 and 4")
    if "plain similarity" not in st.report(got, [8, 16, 24], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("the report does not say that the plain similarity was used")


def _solo_world(layout=None):
    """One person, framed differently from shot to shot, and two things the detector takes for a person.

    Four shots of 8 frames. Shot 1: a microphone alone on frames 0-3, with the person on 4-5, the person alone
    on 6-7. Shot 2: the person, full length. Shot 3: the person in close-up. Shot 4: a lamp and nobody.
    The person in close-up scores 0.75 against the full-length pick, the microphone 0.70, the lamp 0.3.
    Each signature is two views, and only the person has the second, the head.

    `layout(frame, full, close, mic, lamp)` puts the same four masks on other frames.
    """
    def unit(c: float, axis: int) -> torch.Tensor:
        v = torch.zeros(4); v[0] = c; v[axis] = (1 - c * c) ** 0.5
        return v
    full, close = _box(50, 5, 70, 65), _box(20, 5, 100, 70)
    mic, lamp = _box(0, 0, 10, 30), _box(110, 0, 125, 30)
    sigs = [(full, torch.tensor([1.0, 0, 0, 0])), (close, unit(0.75, 1)), (mic, unit(0.70, 2)), (lamp, unit(0.3, 3))]
    def present(f: int) -> list[torch.Tensor]:
        if layout is not None: return layout(f, full, close, mic, lamp)
        if f < 4: return [mic]
        if f < 6: return [mic, close]
        if f < 8: return [close]
        if f < 16: return [full]
        if f < 24: return [close]
        return [lamp]
    tracked = []
    def detect(f: int):
        return torch.stack(present(f), dim=0), [0.9] * len(present(f))
    def sign(_frame: int, mask: torch.Tensor):
        v = next(v for m, v in sigs if torch.equal(m, mask))
        return v, (v if torch.equal(mask, close) or torch.equal(mask, full) else None)
    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        tracked.append((start, end, seed, "person" if torch.equal(mask, close) or torch.equal(mask, full) else "thing"))
        return mask[None].repeat(end - start, 1, 1)
    return detect, sign, track, tracked


def check_alone(problems):
    """A clip with one person: taken in every shot they are in, whatever the framing, and a thing is not."""
    detect, sign, track, tracked = _solo_world()
    cuts = [8, 16, 24]
    got = st.follow(32, cuts, st.PICK_LARGEST, 9, None, detect, sign, track, stride=2, offset=1)
    want = [(0, 8, 4, "person"), (8, 16, 9, "person"), (16, 24, 17, "person")]
    if tracked != want:
        problems.append(f"one person, framed differently per shot: tracked {tracked}, not {want}. Shot 1 opens on a "
                        "microphone alone and then shows it beside the person: a thing has no head, so the person is the "
                        "one with a head from frame 4; shot 4 holds a lamp, which scores under the line and has no head")
    if [s.lone for s in got.shots] != [True, False, True, False] or got.others != 0:
        problems.append(f"lone flags {[s.lone for s in got.shots]} with {got.others} other(s) on the pick frame")
    if not abs(got.match - st.PLAIN_FLOOR) < 1e-9:
        problems.append(f"the line is {got.match}; with nobody else on the pick frame it is the plain floor {st.PLAIN_FLOOR}")
    if got.views != 2 or got.views_used != 2:
        problems.append(f"the subject is compared in {got.views_used} of {got.views} places; this world gives two and the person has both")
    text = st.report(got, cuts, st.PICK_LARGEST, "person", True, False, 1.0)
    for need in ("taken on frame 4, as the only person there, similarity 0.75", "[4] frames 24-31: absent (up to 1 detection(s), none that could be compared)",
                 "framed differently, the mask is 4.0 times as wide", "one person with a head is taken", "the lower one counts"):
        if need not in text:
            problems.append(f"the one-person report lacks {need!r}: {text!r}")
    if st._state(got.shots[0]) != "taken (only person)":
        problems.append("the tile does not say a shot was taken as the only person")
    # the control: with a value named the rule is off, and the same clip loses the two shots framed differently
    detect, sign, track, tracked = _solo_world()
    got = st.follow(32, cuts, st.PICK_LARGEST, 9, st.PLAIN_FLOOR, detect, sign, track, stride=2, offset=1)
    if [s.seed for s in got.shots] != [None, 9, None, None] or any(s.lone for s in got.shots):
        problems.append(f"with a value named, seeds are {[s.seed for s in got.shots]}: the lone rule must be off, and without "
                        "it this clip's two other shots are missed, which is the failure the rule exists for")
    # not alone on the pick frame: a lone person in another shot is not presumed to be the subject
    boxes, detect, sign, track, calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    if any(s.lone for s in got.shots) or [s.seed for s in got.shots] != [3, None, 20, 25]:
        problems.append(f"picked among other people, seeds {[s.seed for s in got.shots]} and lone {[s.lone for s in got.shots]}: "
                        "the lone rule applies only when the pick frame shows nobody else")


def _lost_world(visible, n: int = 48, reach: int | None = None):
    """One shot of `n` frames, no cut: the subject and one other person; the tracker cannot cross a hidden stretch.

    `visible(f)` says whether the subject can be seen on frame `f`. The detector returns the other person on every
    frame and the subject where they are visible. A track covers the frames from its seed out to the nearest hidden
    frame on either side and no further, as core's tracker stays empty once it has let an object go; `reach`
    limits it to that many frames forward of the seed, a tracker that keeps letting go.
    """
    subject, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    sigs = [(subject, torch.tensor([1.0, 0.0, 0.0])), (other, torch.tensor([0.0, 1.0, 0.0]))]
    calls = {"detect": [], "track": []}
    def detect(f: int):
        calls["detect"].append(f)
        people = [other] + ([subject] if visible(f) else [])
        return torch.stack(people, dim=0), [0.9] * len(people)
    def sign(_frame: int, mask: torch.Tensor):
        return next(v for m, v in sigs if torch.equal(m, mask))
    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls["track"].append((start, end, seed, "subject" if torch.equal(mask, subject) else "other"))
        out = torch.zeros((end - start, H, W))
        f = seed
        while f < end and visible(f) and (reach is None or f - seed < reach):
            out[f - start] = mask
            f += 1
        f = seed - 1
        while f >= start and visible(f):
            out[f - start] = mask
            f -= 1
        return out
    return subject, detect, sign, track, calls


def _near(regained, want) -> bool:
    """A shot's `regained` against what is expected, the two similarities to three places."""
    return [(a, f, n, round(score, 3), round(second, 3)) for a, f, n, score, second in regained] == want


def check_regain(problems):
    """A track that lets the subject go inside a shot is seeded again where they are back, and only on them."""
    hidden = lambda f: 20 <= f < 24
    subject, detect, sign, track, calls = _lost_world(lambda f: not hidden(f))
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    shot = got.shots[0]
    want = [(0, 48, 3, "subject"), (20, 48, 24, "subject")]
    if calls["track"] != want:
        problems.append(f"hidden on frames 20-23 of one shot: tracked {calls['track']}, not {want}: the first track ends "
                        "at 19, and the empty run 20-47 is seeded again on the first probed frame the subject is back on")
    if not _near(shot.regained, [(20, 24, 2, 1.0, 0.0)]) or shot.searched:
        problems.append(f"regained {shot.regained}, searched {shot.searched}; one regain, let go on 20, found on 24 among "
                        "2 detections at similarity 1.0 to the gallery, the other person at 0.0")
    if shot.gallery != [0, 3, 5, 8, 11, 14, 16, 19] or [p[:2] for p in shot.probes] != [(20, 1), (24, 2)]:
        problems.append(f"the gallery was taken on frames {shot.gallery} and the probes were {shot.probes}; eight frames spread "
                        "over the tracked 0-19, and probes on 20 (the other person alone) and 24")
    mask = st.assemble(48, H, W, got.pieces)
    on = [bool(m.any()) for m in mask]
    if on != [not hidden(f) for f in range(48)]:
        problems.append(f"the mask is on frames {[f for f, v in enumerate(on) if v]}; it should be every frame but 20-23, "
                        "where the subject is hidden")
    if any(v and not torch.equal(m, subject) for m, v in zip(mask, on)):
        problems.append("a frame of the regained track does not carry the subject's mask")
    if shot.shown != 3 or shot.seed != 3:
        problems.append(f"the regain moved the shot's tile to frame {shot.shown} (seed {shot.seed}); a correction reads its "
                        "numbers off that tile, so it must stay on the pick frame")
    text = st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0)
    if "let go on frame 20 and found again on frame 24 (similarity 1.00 to the shot's own track, the next person there 0.00, 2 detection(s) there)" not in text:
        problems.append(f"the report does not say where the track let go and where it was found: {text!r}")
    table = st.shot_table.build(got, detect, mask, state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=[])
    row = table["shots"][0]
    if row.get("frames_without_subject") != [[20, 23]] or row["frames_with_subject"] != 44:
        problems.append(f"the shot table gives frames without the subject {row.get('frames_without_subject')} and "
                        f"{row['frames_with_subject']} with; the subject is hidden on 20-23 of 48")
    if row.get("regained") != [{"lost_from_frame": 20, "seed_frame": 24, "detections": 2, "similarity": 1.0, "next_person": 0.0}]:
        problems.append(f"the shot table's regained is {row.get('regained')}")
    if row.get("gallery_frames") != shot.gallery or [p["frame"] for p in row.get("probes_after_a_loss", [])] != [20, 24]:
        problems.append(f"the shot table's gallery {row.get('gallery_frames')} and probes {row.get('probes_after_a_loss')}")
    if "none on 20-23" not in st.shot_table.as_text(table):
        problems.append("the shot table's text does not name the frames with no subject")

    # the run at the shot's start: searched backward from the side that is tracked
    hidden = lambda f: 10 <= f < 16
    subject, detect, sign, track, calls = _lost_world(lambda f: not hidden(f))
    got = st.follow(48, [], st.PICK_LARGEST, 30, 0.8, detect, sign, track, stride=4, offset=1)
    want = [(0, 48, 30, "subject"), (0, 16, 7, "subject")]
    if calls["track"] != want or not _near(got.shots[0].regained, [(0, 7, 2, 1.0, 0.0)]):
        problems.append(f"hidden on 10-15, picked on 30: tracked {calls['track']} and regained {got.shots[0].regained}; the "
                        f"empty run 0-15 is probed from 15 down (15 and 11 are hidden, 7 is not): {want}")
    on = [bool(m.any()) for m in st.assemble(48, H, W, got.pieces)]
    if on != [not hidden(f) for f in range(48)]:
        problems.append(f"with the run at the shot's start the mask is on {[f for f, v in enumerate(on) if v]}, not every frame but 10-15")

    # control: the subject never comes back. Searched, not found, and the person beside them is not taken
    subject, detect, sign, track, calls = _lost_world(lambda f: f < 20)
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    shot = got.shots[0]
    if calls["track"] != [(0, 48, 3, "subject")] or shot.regained or shot.searched != [(20, 47)]:
        problems.append(f"the subject leaves at frame 20 for good: tracked {calls['track']}, regained {shot.regained}, "
                        f"searched {shot.searched}; nobody is there to find, and the other person must not be taken")
    if float(st.assemble(48, H, W, got.pieces)[20:].sum()) != 0.0:
        problems.append("frames after the subject left carry a mask")
    probed = [f for f in calls["detect"] if f >= 20]
    if sorted(set(probed)) != list(range(20, 48, 4)):
        problems.append(f"the empty run 20-47 was probed on {sorted(set(probed))}, not every 4th frame from 20")
    said = st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0)
    if "searched and not found on frames 20-47 (the best person on any frame looked at scored 0.00; the line is" not in said:
        problems.append(f"the report does not say a run was searched, the subject not found, and what the best person scored: {said!r}")

    # control: a tracker that keeps letting go is seeded again a bounded number of times
    subject, detect, sign, track, calls = _lost_world(lambda f: True, n=200, reach=13)
    got = st.follow(200, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    if len(got.shots[0].regained) != st.REGAIN_MOST or len(calls["track"]) != st.REGAIN_MOST + 1:
        problems.append(f"a tracker that holds 13 frames at a time was seeded again {len(got.shots[0].regained)} times "
                        f"({len(calls['track'])} track calls); the bound is REGAIN_MOST = {st.REGAIN_MOST}")

    # a clip with one person. After the loss the detector returns ONE person for the phrase: taken when they are the
    # subject, not taken when they are somebody else (seen on the card, 2026-10-06), and a thing with no head is nobody
    full, bystander, mic, twin = _box(50, 5, 70, 65), _box(20, 5, 100, 70), _box(0, 0, 10, 30), _box(90, 5, 110, 65)
    def unit(c: float, axis: int) -> torch.Tensor:
        v = torch.zeros(4); v[0] = c; v[axis] = (1 - c * c) ** 0.5
        return v
    sigs = [(full, torch.tensor([1.0, 0, 0, 0])), (bystander, unit(0.85, 1)), (mic, unit(0.99, 2)), (twin, unit(0.985, 3))]
    def solo(later):
        tracked = []
        def detect(f: int):
            people = [full] if f < 20 else ([] if f < 24 else list(later))
            return (torch.stack(people, dim=0) if people else torch.zeros((0, H, W))), [0.9] * len(people)
        def sign(_frame: int, mask: torch.Tensor):
            v = next(v for m, v in sigs if torch.equal(m, mask))
            return v, (None if torch.equal(mask, mic) else v)
        def track(start: int, end: int, seed: int, mask: torch.Tensor):
            tracked.append((start, end, seed))
            out = torch.zeros((end - start, H, W))
            lo, hi = (0, 20) if seed < 20 else (24, 48)
            out[max(lo, start) - start:min(hi, end) - start] = mask
            return out
        return detect, sign, track, tracked
    detect, sign, track, tracked = solo([full])
    got = st.follow(48, [], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    if tracked != [(0, 48, 3), (20, 48, 24)] or not _near(got.shots[0].regained, [(20, 24, 1, 1.0, -1.0)]):
        problems.append(f"one person, back after the loss: tracked {tracked}, regained {got.shots[0].regained}; the one "
                        "detection is the subject, at 1.0 to the gallery, with nobody else to be clear of")
    detect, sign, track, tracked = solo([bystander])
    got = st.follow(48, [], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    shot = got.shots[0]
    if tracked != [(0, 48, 3)] or shot.regained or shot.searched != [(20, 47)] or float(st.assemble(48, H, W, got.pieces)[20:].sum()):
        problems.append(f"a bystander alone after the loss was taken: tracked {tracked}, regained {shot.regained}. They score "
                        f"0.85 to the gallery, under the line {st.REGAIN_SAME}; being the only detection must not be enough, "
                        "which is what core's detector did with the phrase on the card")
    if "searched and not found on frames 20-47 (the best person on any frame looked at scored 0.85" not in st.report(
            got, [], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("the report does not say what the rejected person scored")
    detect, sign, track, tracked = solo([mic])
    got = st.follow(48, [], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    if tracked != [(0, 48, 3)] or got.shots[0].regained:
        problems.append(f"a thing alone after the loss was taken: tracked {tracked}; it scores 0.99 under the top third but "
                        "has no head, and the gallery has one on every frame, so it is nobody")
    detect, sign, track, tracked = solo([full, twin])
    got = st.follow(48, [], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    if tracked != [(0, 48, 3)] or got.shots[0].regained or got.shots[0].searched != [(20, 47)]:
        problems.append(f"two people within the margin of each other (1.00 and 0.985): tracked {tracked}, regained "
                        f"{got.shots[0].regained}; closer than REGAIN_MARGIN = {st.REGAIN_MARGIN} is a gap, not a guess")

    # A CORRECTED SHOT IS SEARCHED TOO (2026-10-07): the subject named by hand, hidden on 20-23, is taken back at 24.
    # Left to right the other person is 1 and the subject 2. Before this the corrected track was the only one, and
    # the shot was empty from frame 20 to its end.
    gone = lambda f: 20 <= f < 24
    subject, detect, sign, track, calls = _lost_world(lambda f: not gone(f))
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, corrections={1: 2})
    shot = got.shots[0]
    if calls["track"] != [(0, 48, 3, "subject"), (20, 48, 24, "subject")] or not _near(shot.regained, [(20, 24, 2, 1.0, 0.0)]):
        problems.append(f"`shot 1: person 2`, hidden on 20-23: tracked {calls['track']}, regained {shot.regained}; a corrected "
                        "shot is looked for again where its track lets go, like any other")
    mask = st.assemble(48, H, W, got.pieces)
    if float(mask[20:24].sum()) or not all(torch.equal(mask[f], subject) for f in (19, 24, 47)) or st._state(shot) != "taken (corrected)":
        problems.append("the corrected shot's mask is not the subject's on 0-19 and 24-47 with 20-23 empty")
    if "corrected by hand, person 2" not in (text := st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0)) \
            or "found again on frame 24" not in text:
        problems.append("the report of a corrected shot does not say where the subject was found again")
    # and a correction to somebody the tracker holds throughout costs no search
    subject, detect, sign, track, calls = _lost_world(lambda f: True)
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, corrections={1: 1})
    if calls["track"] != [(0, 48, 3, "other")] or got.shots[0].regained or got.shots[0].probes:
        problems.append(f"`shot 1: person 1` in a world where nobody is ever hidden: tracked {calls['track']}, probes {got.shots[0].probes}")

    # THE HAND-OVER: an earlier run's gallery picks the subject here, where the pick rule would take somebody else
    subject, detect, sign, track, calls = _lost_world(lambda f: True)
    first = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    table = st.shot_table.build(first, detect, st.assemble(48, H, W, first.pieces), state=st._state, phrase="person",
                                pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=[])
    if table["gallery"]["frames"] != first.gallery_frames or len(table["gallery"]["signatures"]) != st.GALLERY_MOST:
        problems.append(f"the shot table's gallery has frames {table['gallery']['frames']} and "
                        f"{len(table['gallery']['signatures'])} signatures; the run kept {first.gallery_frames}")
    handed = st.shot_table.gallery_from(st.shot_table.as_json(table))
    if len(handed) != st.GALLERY_MOST or not all(abs(float((a[0] * b[0]).sum()) - 1.0) < 1e-4 for a, b in zip(handed, first.gallery)):
        problems.append("a gallery read back from the shot table's JSON is not the one the run kept")
    # the next load: the subject is hidden on its first 20 frames, and the OTHER person is the largest throughout
    big, small = _box(10, 5, 90, 70), _box(100, 20, 120, 50)
    sigs = [(big, torch.tensor([0.0, 1.0, 0.0])), (small, torch.tensor([1.0, 0.0, 0.0]))]
    def later(with_subject):
        tracked = []
        def detect(f: int):
            people = [big] + ([small] if with_subject and f >= 20 else [])
            return torch.stack(people, dim=0), [0.9] * len(people)
        def sign(_frame: int, mask: torch.Tensor):
            return next(v for m, v in sigs if torch.equal(m, mask))
        def track(start: int, end: int, seed: int, mask: torch.Tensor):
            tracked.append((start, end, seed, "subject" if torch.equal(mask, small) else "other"))
            return mask[None].repeat(end - start, 1, 1)
        return detect, sign, track, tracked
    detect, sign, track, tracked = later(True)
    got = st.follow(48, [], st.PICK_LARGEST, None, None, detect, sign, track, stride=4, offset=1, gallery=handed)
    if tracked != [(0, 48, 20, "subject")] or got.pick_frame != 20 or got.gallery_given != st.GALLERY_MOST:
        problems.append(f"with the earlier run's gallery handed in: tracked {tracked}, pick frame {got.pick_frame}; the other "
                        "person is alone and the largest on the probe frame and must be refused, and the subject is picked "
                        "on frame 20, the first frame looked at that shows them")
    said = st.report(got, [], st.PICK_LARGEST, "person", False, False, 1.0)
    if "chosen as the subject an earlier run handed in: similarity 1.00" not in said or "earlier frame(s) refused, the best of them 0.00" not in said:
        problems.append(f"the report does not say the pick came from the handed-in gallery and what was refused: {said!r}")
    detect, sign, track, tracked = later(True)
    got = st.follow(48, [], st.PICK_LARGEST, None, None, detect, sign, track, stride=4, offset=1)
    if tracked != [(0, 48, 1, "other")]:
        problems.append(f"the control: with no gallery the pick rule takes the largest person on the probe frame, {tracked}; "
                        "if it does not, the hand-over case above proves nothing")
    detect, sign, track, tracked = later(False)
    got = st.follow(48, [], st.PICK_LARGEST, None, None, detect, sign, track, stride=4, offset=1, gallery=handed)
    if tracked or got.pick_frame is not None or float(st.assemble(48, H, W, got.pieces).sum()):
        problems.append(f"the subject is not in the load at all: tracked {tracked}; with a gallery handed in nobody is picked "
                        "and every mask is empty, where the pick rule would have taken the other person")
    if "is the subject an earlier run handed in (the best scored 0.00" not in st.report(got, [], st.PICK_LARGEST, "person", False, False, 1.0):
        problems.append("the report does not say nobody matched the handed-in subject, with the best score seen")
    for bad, need in (("{}", "carries no gallery"), ("/no/such/file_shots.json", "no shot table at"), ("{not json", "is not a shot table")):
        try:
            st.shot_table.gallery_from(bad)
        except ValueError as error:
            if need not in str(error):
                problems.append(f"gallery_from({bad!r}) raised {error!r}, which does not say {need!r}")
        else:
            problems.append(f"gallery_from({bad!r}) was accepted")

    frames_of = st.gallery_frames(torch.stack([_box(0, 0, 4, 4) if f % 3 else torch.zeros((H, W)) for f in range(40)], dim=0), 5)
    if len(frames_of) != 5 or any(f % 3 == 0 for f in frames_of) or frames_of[0] != 1 or frames_of[-1] != 38:
        problems.append(f"gallery_frames gave {frames_of}: five frames that carry a mask, from the first (1) to the last (38)")
    # the smallest and the largest mask are in the gallery, wherever in the run they fall
    sized = torch.stack([_box(0, 0, 2, 2) if f == 7 else _box(0, 0, 60, 60) if f == 22 else _box(0, 0, 10, 10) for f in range(40)], dim=0)
    frames_of = st.gallery_frames(sized, 5)
    if len(frames_of) != 5 or 7 not in frames_of or 22 not in frames_of or frames_of[0] != 0 or frames_of[-1] != 39:
        problems.append(f"gallery_frames gave {frames_of} for a run whose mask is smallest on frame 7 and largest on 22: both "
                        "belong in it, with the run's first and last frame")

    runs = st.empty_runs(torch.stack([_box(0, 0, 4, 4) if f in (2, 3, 7) else torch.zeros((H, W)) for f in range(9)], dim=0))
    if runs != [(0, 2), (4, 7), (8, 9)]:
        problems.append(f"empty_runs gave {runs} for a mask on frames 2, 3 and 7 of 9, not [(0, 2), (4, 7), (8, 9)]")


def check_headless_vote(problems):
    """Nothing named: a thing the detector takes for a person does not win the pick by owning the longest shot."""
    # the shape of the one-person clip's first window (bench/results/2026-10-06_subject_track_defaults.md): the
    # opening shot is the longest and starts on the microphone alone; the person walks in later. Shot 1 is frames
    # 0-15, the microphone alone to frame 5, beside the person to 9, the person alone after; shot 2 the person
    # full length; shot 3 the person in close-up.
    def layout(f, full, close, mic, lamp):
        if f < 6: return [mic]
        if f < 10: return [mic, close]
        if f < 16: return [close]
        return [full] if f < 24 else [close]
    detect, sign, track, tracked = _solo_world(layout)
    got = st.follow(32, [16, 24], st.PICK_LARGEST, None, None, detect, sign, track, stride=2, offset=1)
    want = [(0, 16, 6, "person"), (16, 24, 17, "person"), (24, 32, 25, "person")]
    if tracked != want or got.pick_frame != 17:
        problems.append(f"an opening shot that starts on a microphone alone and is the longest: picked on frame "
                        f"{got.pick_frame}, tracked {tracked}, not {want}. The microphone has no head, so its shot's "
                        "sixteen frames do not vote; the person is picked in shot 2 and taken in shot 1 from where "
                        "they come in")
    if getattr(got, "passed_over", None) != [(1, 1)]:
        problems.append(f"passed over for the pick: {getattr(got, 'passed_over', None)}, not shot 1's detection on frame 1")
    text = st.report(got, [16, 24], st.PICK_LARGEST, "person", False, False, 1.0)
    if "not counted for the pick" not in text or "shot 1 (frame 1)" not in text:
        problems.append(f"the report does not say whose vote was left out of the pick: {text!r}")
    # the control: when nothing the rule favours has a head, the pick is made as before and the report says nothing
    detect, sign, track, tracked = _solo_world(lambda f, full, close, mic, lamp: [mic] if f < 16 else [lamp])
    got = st.follow(32, [16], st.PICK_LARGEST, None, None, detect, sign, track, stride=2, offset=1)
    if got.pick_frame != 1 or getattr(got, "passed_over", None) != []:
        problems.append(f"only things on screen: picked on frame {got.pick_frame} with {getattr(got, 'passed_over', None)} passed over; "
                        "with no head anywhere every favourite votes, as before the rule")
    # a frame the user names is theirs: the rule is about the automatic pick only
    detect, sign, track, tracked = _solo_world(layout)
    got = st.follow(32, [16, 24], st.PICK_LARGEST, 1, None, detect, sign, track, stride=2, offset=1)
    if got.pick_frame != 1 or tracked[0] != (0, 16, 1, "thing") or getattr(got, "passed_over", None) != []:
        problems.append(f"a named frame showing only the microphone: picked on {got.pick_frame}, tracked {tracked[:1]}; "
                        "a named frame is taken as named")


def check_two_places(problems):
    """A match has to hold on the head as well: someone alike at the shoulders and not at the head is left alone."""
    subject, twin, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50), _box(100, 20, 120, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()
    # shoulders: the twin is the subject's double. head: the twin is somebody else
    shoulders = [(subject, e(1.0, 0, 0)), (twin, e(1.0, 0.05, 0)), (other, e(0, 1.0, 0))]
    heads = [(subject, e(1.0, 0, 0)), (twin, e(0, 0, 1.0)), (other, e(0, 1.0, 0))]
    def detect(f: int):
        people = [subject, other] if f < 8 else ([twin, other] if f < 16 else [subject, other])
        return torch.stack(people, dim=0), [0.9, 0.8]
    def find(table, mask):
        return next(v for m, v in table if torch.equal(m, mask))
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)
    both = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: (find(shoulders, m), find(heads, m)),
                     track, stride=4, offset=1)
    if [s.seed for s in both.shots] != [2, None, 17]:
        problems.append(f"compared in two places, seeds are {[s.seed for s in both.shots]}: shot 2 holds the subject's double "
                        "at the shoulders with another head, and must be left alone; shot 3 holds the subject")
    one = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: find(shoulders, m), track, stride=4, offset=1)
    if [s.seed for s in one.shots] != [2, 9, 17]:
        problems.append(f"the control failed: compared at the shoulders alone, seeds are {[s.seed for s in one.shots]}; the "
                        "double should be taken there, or the second place decides nothing in this case")
    # a person on whom no head is found is no match, however alike at the shoulders
    none = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect,
                     lambda f, m: (find(shoulders, m), None if 16 <= f else find(heads, m)), track, stride=4, offset=1)
    if [s.seed for s in none.shots] != [2, None, None]:
        problems.append(f"with no head found in shot 3, seeds are {[s.seed for s in none.shots]}: a person without a head is no match")
    # the subject has no head on the pick frame: the shoulders decide alone, and the report says so
    bare = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, lambda _f, m: (find(shoulders, m), None), track, stride=4, offset=1)
    if [s.seed for s in bare.shots] != [2, 9, 17] or bare.views_used != 1:
        problems.append(f"with no head on the subject, seeds are {[s.seed for s in bare.shots]} from {bare.views_used} place(s): "
                        "the shoulders decide alone")
    if "matched by the head and shoulders alone" not in st.report(bare, [8, 16], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("the report does not say that no head was found on the subject")
    # head_of: the head mostly inside the person, the highest of them, and none for a thing
    person = _box(40, 10, 80, 60)
    hat, face, far = _box(50, 10, 70, 20), _box(50, 22, 70, 34), _box(0, 0, 20, 10)
    got = st.head_of(person, torch.stack([far, face, hat], dim=0))
    if got is None or not torch.equal(got, hat):
        problems.append("head_of does not return the highest head lying inside the person")
    if st.head_of(person, torch.stack([far], dim=0)) is not None or st.head_of(person, torch.zeros((0, H, W))) is not None:
        problems.append("head_of found a head for a person who has none inside their mask")


def check_empty(problems):
    boxes, detect, sign, track, _calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 1.5, detect, sign, track, stride=4, offset=1)
    mask = st.assemble(32, H, W, got.pieces)
    if [s.seed for s in got.shots] != [3, None, None, None] or float(mask[8:].sum()) != 0.0:
        problems.append("with a threshold nothing can pass, a shot other than the picked one was still filled")
    if not torch.equal(mask[0], boxes[0]):
        problems.append("the picked shot lost its mask when the threshold was raised")
    none = lambda _frame: (torch.zeros((0, H, W)), [])
    for frame in (3, None):
        got = st.follow(32, [8], st.PICK_LARGEST, frame, None, none, sign, track)
        if got.pick_frame is not None or got.pieces or float(st.assemble(32, H, W, got.pieces).sum()) != 0.0:
            problems.append("with nothing detected something was still masked")
        if "every mask is empty" not in st.report(got, [8], st.PICK_LARGEST, "person", frame is not None, False, 1.0):
            problems.append("the report does not say that nothing was picked")
        tiles = st.preview(torch.rand((32, H, W, 3)), st.assemble(32, H, W, got.pieces), got.shots, none)
        if tiles.shape[0] != 2:
            problems.append("the preview of a clip with no subject is not one tile per shot")
    try:
        st.follow(32, [], st.PICK_LARGEST, 40, 0.5, detect, sign, track)
    except ValueError:
        pass
    else:
        problems.append("a pick_frame past the end of the clip was accepted")


def check_signature(problems):
    """The signature is taken from the head and shoulders, and tells two heads apart over the same clothes."""
    body = _box(40, 12, 80, 72)
    head = st.top_third(body)
    if not (float(head[12:32].sum()) == float(head.sum()) > 0 and float(head[32:].sum()) == 0):
        problems.append("top_third does not keep exactly the top third of the rows a mask covers")
    if not torch.equal(st.top_third(torch.zeros((H, W))), torch.zeros((H, W))):
        problems.append("top_third changed an empty mask")
    feats_a, feats_b = torch.zeros((3, H, W)), torch.zeros((3, H, W))
    feats_a[0, 12:32], feats_b[1, 12:32] = 1.0, 1.0      # two different heads
    feats_a[2, 32:], feats_b[2, 32:] = 1.0, 1.0          # the same clothes
    v = torch.tensor([0.6, 0.8, 0.0])
    r = st.relative(v, torch.tensor([0.6, 0.0, 0.0]))
    if r is None or not torch.allclose(r, torch.tensor([0.0, 1.0, 0.0])) or st.relative(v, None) is not v:
        problems.append("relative does not subtract the centre and return unit length, or changes a signature with no centre")
    whole = st.similarity(st.signature(feats_a, body), st.signature(feats_b, body))
    heads = st.similarity(st.signature(feats_a, head), st.signature(feats_b, head))
    if not (whole > 0.5 and heads < 0.05):
        problems.append(f"two people in the same clothes: similarity {whole:.2f} on the whole mask and {heads:.2f} on "
                        "the top third; the top third is meant to tell them apart where the whole mask cannot")
    if st.signature(feats_a, torch.zeros((H, W))) is not None or st.similarity(None, None) != -1.0:
        problems.append("an empty mask has a signature, or a missing signature has a similarity")


def check_one_detection_each(problems):
    """One person, three nested detections (2026-10-10): numbered once, a candidate once, counted in the report and the table."""
    boxes, detect, sign, track, calls = _world()
    nested = boxes[1].clone()
    rows = (nested > 0.5).any(dim=1).nonzero().flatten()
    nested[int(rows.min()) + max(1, (int(rows.max()) - int(rows.min())) // 2):] = 0     # the top half of person 1

    def doubled(f):
        masks, scores = detect(f)
        if f == 3 and int(masks.shape[0]) >= 2:
            # person 1 again, as a looser detection scored lower and a tighter one scored lower still
            return torch.cat([masks, nested[None], nested[None]]), list(scores) + [0.3, 0.2]
        return masks, scores

    joined: dict[int, int] = {}
    parts: dict[int, float] = {}
    one = st.one_detection_each(doubled, joined, parts=parts)
    masks, scores = one(3)
    plain, plain_scores = detect(3)
    if int(masks.shape[0]) != int(plain.shape[0]) or joined.get(3) != 2 or not torch.equal(masks, plain) or list(scores) != list(plain_scores):
        problems.append(f"one detection each: frame 3 should be the frame's own people with two duplicates left out; "
                        f"{int(masks.shape[0])} detections, joined {joined}")
    if one(3) is not one(3) or one(4)[0].shape[0] != detect(4)[0].shape[0] or joined.get(4) != 0:
        problems.append("one detection each: a frame is not kept once looked at, or a frame with no duplicates lost someone")
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 0.8, one, sign, track, stride=4, offset=1)
    text = st.joined_report(got.shots, joined)
    for need in ("returned the same thing more than once on 1 of", "(2 extra detection(s))", "[1] 2 of them on the tile's frame, 3"):
        if need not in text:
            problems.append(f"one detection each: the report lacks {need!r}: {text!r}")
    label = st.tile_label(1, got.shots[0], joined)
    if not label.endswith("(2 duplicates joined)") or "duplicate" in st.tile_label(2, got.shots[1], joined) \
            or st.tile_label(1, got.shots[0], {3: 1}).count("(1 duplicate joined)") != 1 or "duplicate" in st.tile_label(1, got.shots[0]):
        problems.append(f"one detection each: the tile's header does not say what was joined on its frame, or says it where nothing was: {label!r}")
    # the kept detection is the whole person here, so the join is safe and its share is 1
    if parts != {3: 1.0} or "NOT joined" in st.joined_report(got.shots, joined, parts):
        problems.append(f"one detection each: the kept detection is the largest of its group; parts are {parts}")
    got.shots[0].corrected = "person 2"
    if "This shot is corrected by hand (person 2): the numbers count the people left" not in st.joined_report(got.shots, joined, parts):
        problems.append("one detection each: a corrected shot whose tile frame had a join is not told to check its number")
    got.shots[0].corrected = ""
    # mrcorn: when the best scored of a thing is a PART, nothing is joined: every detection keeps a number, so a
    # correction can take the whole or the part, and the report and the tile say so
    def a_part_scored_highest(f):
        masks, scores = detect(f)
        if f == 3 and int(masks.shape[0]) >= 2:
            small = torch.zeros_like(masks[0])
            rows = (masks[0] > 0.5).any(dim=1).nonzero().flatten()
            cols = (masks[0] > 0.5).any(dim=0).nonzero().flatten()
            small[int(rows.min()):int(rows.min()) + 3, int(cols.min()):int(cols.min()) + 3] = 1.0      # a few pixels of person 1
            return torch.cat([masks, small[None]]), list(scores) + [0.99]
        return masks, scores
    j2, p2, u2 = {}, {}, {}
    cautious = st.one_detection_each(a_part_scored_highest, j2, parts=p2, unsure=u2)
    masks3, scores3 = cautious(3)
    if int(masks3.shape[0]) != int(plain.shape[0]) + 1 or j2.get(3) != 0 or u2 != {3: 2} or p2 != {}:
        problems.append(f"one detection each: a part scored over its person must leave both numbered; {int(masks3.shape[0])} detections, "
                        f"joined {j2}, unsure {u2}, parts {p2}")
    said = st.joined_report(got.shots, j2, p2, u2)
    for need in ("were NOT joined", "a correction can take the whole or the part", "[1] 2 detections on the tile's frame, 3, may be one thing"):
        if need not in said:
            problems.append(f"one detection each: the report lacks {need!r}: {said!r}")
    if not st.tile_label(1, got.shots[0], j2, u2).endswith("(2 may be one, not joined)"):
        problems.append(f"one detection each: the tile's header does not say two detections may be one: {st.tile_label(1, got.shots[0], j2, u2)!r}")
    if abs(st.A_PART_UNDER - 0.5) > 1e-9:
        problems.append("one detection each: A_PART_UNDER moved; read its provenance before changing this check")
    if st.joined_report(got.shots, {3: 0, 9: 0}) != "":
        problems.append("one detection each: a report line with nothing joined")
    table = st.shot_table.build(got, one, st.assemble(32, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=[8, 16, 24], joined=joined, parts=parts)
    first = table["shots"][0]
    if first["duplicates_joined"] != {"on_shown_frame": 2, "on_frames_looked_at": 2, "kept_share_of_largest": 1.0, "may_be_one_not_joined": 0} \
            or len(first["people"]) != int(plain.shape[0]):
        problems.append(f"one detection each: the table's first shot says {first['duplicates_joined']} with {len(first['people'])} people")
    bare = st.shot_table.build(got, one, st.assemble(32, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                               named_frame=True, named_value=True, cuts=[8, 16, 24])
    if bare["shots"][0]["duplicates_joined"] is not None:
        problems.append("one detection each: a table built without the count writes something other than null")


def check_moved_by_a_gap(problems):
    """2026-10-10: the automatic line was put between two shots that both held the subject, and the report said nothing."""
    def found(firsts, bests, floor, picked=0, taken=()):
        shots = [st.Shot(8 * i, 8 * i + 8, probe=8 * i, best=b, first=f) for i, (f, b) in enumerate(zip(firsts, bests))]
        shots[picked].picked = True
        for i in taken:
            shots[i].seed = shots[i].start
        rest = [s.first for s in shots if not s.picked]
        return st.Followed(shots, pick_frame=3, match=st.auto_match(rest, floor), floor=floor, gap=st.auto_gap(rest, floor), others=1)
    if abs(st.auto_match([0.799, 0.904], st.MATCH_FLOOR) - 0.8515) > 1e-6 or st.auto_gap([0.799, 0.904], st.MATCH_FLOOR) != (0.799, 0.904):
        problems.append("gap: the day's two scores should put the automatic line in their middle and name them as its ends")
    # the line is the gap's middle or the floor, exactly as before `auto_gap` was split out
    for scores, floor in (([0.95, 0.83], 0.8), ([0.85, 0.82], 0.8), ([0.82, 0.60, 0.30], 0.8), ([0.7, 0.3], 0.8), ([], 0.8), ([0.9], 0.8), ([-1.0, 0.9, 0.5], 0.91)):
        gap, line = st.auto_gap(scores, floor), st.auto_match(scores, floor)
        if (gap is None and line != floor) or (gap is not None and (abs(line - sum(gap) / 2) > 1e-9 or line <= floor)):
            problems.append(f"gap: {scores} at floor {floor}: the line {line} and the gap {gap} disagree")
    kitchen = found([1.0, 0.799, 0.904], [1.0, 0.799, 0.904], st.MATCH_FLOOR, taken=(2,))
    if st.moved_by_a_gap(kitchen, False) != (0.799, 0.904):
        problems.append(f"gap: the line stands between 0.799 and 0.904 and that is not said: {st.moved_by_a_gap(kitchen, False)}")
    text = st.report(kitchen, [8, 16], st.PICK_LARGEST, "person", True, False, 1.0)
    for need in ("the automatic line is above its floor (0.80)", "on their first look, 0.80 and 0.90", "shot 2, whose best over the whole shot is 0.80",
                 "on its first look it was under the floor as well"):
        if need not in text:
            problems.append(f"gap: the report lacks {need!r}: {text!r}")
    # mrcorn's case: first looks 0.90 and 0.60 put the line at 0.75; the lower shot's later frames lift it to 0.74, or to 0.77 and it is taken
    for best, taken, need in ((0.74, (2,), "shot 2, whose best over the whole shot is 0.74"),
                              (0.77, (1, 2), "shot 2, whose best over the whole shot is 0.77 and which was taken")):
        lifted = found([1.0, 0.60, 0.90], [1.0, best, 0.90], 0.5, taken=taken)
        if st.moved_by_a_gap(lifted, False) != (0.60, 0.90):
            problems.append(f"gap: the ends are the first looks, 0.60 and 0.90, whatever the shot scored later; got {st.moved_by_a_gap(lifted, False)}")
        said = st.report(lifted, [8, 16], st.PICK_LARGEST, "person", True, False, 1.0)
        if need not in said or "so the gap alone refused it there" not in said:
            problems.append(f"gap: the report lacks {need!r}, or that the gap alone refused the first look: {said!r}")
    for name, quiet in (("a named value", st.moved_by_a_gap(kitchen, True)),
                        ("a line at its floor", st.moved_by_a_gap(found([1.0, 0.82, 0.85], [1.0, 0.82, 0.85], st.MATCH_FLOOR), False)),
                        ("nobody picked", st.moved_by_a_gap(st.Followed([st.Shot(0, 8, probe=0)]), False))):
        if quiet is not None:
            problems.append(f"gap: {name} is reported as moved by a gap: {quiet}")
    # a real `follow` whose line a gap moves: alone on the pick frame, one shot of somebody 0.86 alike, one of the subject
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()   # noqa: E731
    me, alike = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    sig = lambda _f, m: e(1.0, 0, 0) if torch.equal(m, me) else e(1.0, 0.593, 0)   # noqa: E731
    seen = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, lambda f: (torch.stack([alike if 8 <= f < 16 else me]), [0.9]), sig,
                     lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1), stride=4, offset=1)
    if seen.gap is None or abs(seen.gap[0] - 0.86) > 0.005 or abs(seen.gap[1] - 1.0) > 1e-4 or abs(seen.match - sum(seen.gap) / 2) > 1e-9 \
            or seen.floor != st.PLAIN_FLOOR:
        problems.append(f"gap: a shot 0.86 alike and a shot of the subject should put the line between them; gap {seen.gap}, line {seen.match}")
    if "the automatic line is above its floor (0.91)" not in st.report(seen, [8, 16], st.PICK_LARGEST, "person", True, False, 1.0):
        problems.append("gap: a real run whose line a gap moved does not say so in its report")
    # the real thing: `follow` keeps each shot's first look and the ends it handed the rule
    boxes, detect, sign, track, calls = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, None, detect, sign, track, stride=4, offset=1)
    rest = [s for s in got.shots if not s.picked]
    if got.gap != st.auto_gap([s.first for s in rest], got.floor) or any(s.first > s.best + 1e-9 for s in rest) \
            or abs(got.match - st.auto_match([s.first for s in rest], got.floor)) > 1e-9:
        problems.append(f"gap: follow's line {got.match}, gap {got.gap} and first looks {[s.first for s in rest]} do not agree")
    table = st.shot_table.build(got, detect, st.assemble(32, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=False, cuts=[8, 16, 24], gap=st.moved_by_a_gap(got, False))
    if abs(table["match_floor"] - got.floor) > 1e-3 or got.floor not in (st.MATCH_FLOOR, st.PLAIN_FLOOR) \
            or table["match_moved_by_gap"] != (None if got.gap is None else [round(v, 3) for v in got.gap]):
        problems.append(f"gap: the table's floor and gap are {table['match_floor']}, {table['match_moved_by_gap']}; the run's floor {got.floor}")


def check_looks_in_the_table(problems):
    """2026-10-10: a shot refused after a cut left one number in the table. Now every look is there, with everybody on it."""
    subject, twin, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50), _box(100, 20, 120, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()   # noqa: E731
    shoulders = [(subject, e(1.0, 0, 0)), (twin, e(1.0, 0.05, 0)), (other, e(0, 1.0, 0))]
    heads = [(subject, e(1.0, 0, 0)), (twin, e(0, 0, 1.0)), (other, e(0, 1.0, 0))]
    def detect(f: int):
        if 8 <= f < 16:      # the detector's order here is NOT left to right: the other first, then the double
            return torch.stack([other, twin], dim=0), [0.8, 0.9]
        return torch.stack([subject, other], dim=0), [0.9, 0.8]
    find = lambda table, mask: next(v for m, v in table if torch.equal(m, mask))   # noqa: E731
    # in shot 3 the other person has no head found: their head likeness is null and so is their lowest
    sign = lambda f, m: (find(shoulders, m), None if (f >= 16 and torch.equal(m, other)) else find(heads, m))   # noqa: E731
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731
    got = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, sign, track, stride=4, offset=1)
    table = st.shot_table.build(got, detect, st.assemble(24, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=False, cuts=[8, 16])
    one, two, three = table["shots"]
    if one["looks"] != [] or one["shown_frame"] != 2 or [len(p.get("signatures") or []) for p in one["people"]] != [2, 2]:
        problems.append("looks: the picked shot has no looks, and its people on the pick frame carry their signatures")
    else:
        # left to right on frame 2: the subject is person 1, the other person 2
        mine, theirs = one["people"][0]["signatures"], one["people"][1]["signatures"]
        if max(abs(a - b) for a, b in zip(mine[0], [1.0, 0.0, 0.0])) > 1e-4 or max(abs(a - b) for a, b in zip(theirs[1], [0.0, 1.0, 0.0])) > 1e-4:
            problems.append(f"looks: on the pick frame each person's signatures are their own: {mine}, {theirs}")
    if "in frame order" not in table["looks_are"]:
        problems.append("looks: the table does not say its looks are in frame order")
    if [look["frame"] for look in two["looks"]] != [8, 9, 12] or any(len(look["people"]) != 2 for look in two["looks"]):
        problems.append(f"looks: the refused shot was judged on frames 9, 8 and 12 with two people each, written in frame order; the table has "
                        f"{[(look['frame'], len(look['people'])) for look in two['looks']]}")
    else:
        first, second = two["looks"][0]["people"], two["looks"][1]["people"]
        # left to right: the twin is person 1, the other person 2
        if set(first[0]["likeness"]) != {"top third", "head"} or first[0]["likeness"]["top third"] < 0.9 \
                or first[0]["likeness"]["head"] > 0.5 or abs(first[0]["lowest"] - first[0]["likeness"]["head"]) > 1e-9:
            problems.append(f"looks: the double is like the subject at the shoulders and not at the head, and the lowest is the head's: {first[0]}")
        if abs(two["subject"]["similarity"] - max(p["lowest"] for look in two["looks"] for p in look["people"])) > 1e-3:
            problems.append("looks: the shot's own best similarity is not the best `lowest` among its looks")
        if [p["same_as"] for p in first] != [None, None] or [p["same_as"] for p in second] != [1, 2]:
            problems.append(f"looks: nobody is linked on a shot's first look, and each is their own number on the next: "
                            f"{[p['same_as'] for p in first]}, {[p['same_as'] for p in second]}")
        if first[0]["box"] != [0, 20, 20, 30] or first[0]["area_px"] != 600 or first[0]["detector_score"] != 0.9:
            problems.append(f"looks: a person's box, area and detector score: {first[0]}")
    signed = [p.get("signatures") for p in two["people"]]
    if any(v is None or len(v) != 2 for v in signed) or max(abs(a - b) for a, b in zip(signed[0][0], e(1.0, 0.05, 0).tolist())) > 1e-4 \
            or max(abs(a - b) for a, b in zip(signed[0][1], [0.0, 0.0, 1.0])) > 1e-4:
        problems.append(f"looks: each person on a refused shot's shown frame carries their own two signatures: {signed}")
    last = three["looks"][0]["people"]
    headless = next(p for p in last if p["person"] == 2)
    if headless["likeness"]["head"] is not None or headless["lowest"] is not None or headless["likeness"]["top third"] is None:
        problems.append(f"looks: a person with no head found has no head likeness and no lowest: {headless}")
    if next(p for p in three["people"] if p["person"] == 2)["signatures"][1] is not None:
        problems.append("looks: a missing place is null among a person's signatures")
    if not table["looks_are"] or not table["signatures_are"]:
        problems.append("looks: the table does not say what its looks and signatures are")
    # a tracker's result without the record (an older one, or a stand-in) builds a table with empty looks and no signatures
    got.judged = {}
    old = st.shot_table.build(got, detect, st.assemble(24, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                              named_frame=True, named_value=False, cuts=[8, 16])
    if any(s["looks"] for s in old["shots"]) or any("signatures" in p for s in old["shots"] for p in s["people"]):
        problems.append("looks: with nothing recorded the table still writes looks or signatures")


def _refusal(call) -> str:
    try:
        call()
    except SystemExit as err:
        return str(err)
    return ""


def check_rows_from_tables(problems):
    """The trackers' tables of one load read together: a row a shot, answered only by a person's word, a question otherwise.

    The made-up clip is 2026-10-10's load in small: shot 1 both people, shot 2 both people looking unlike themselves
    (each tracker alone refuses them), shot 3 the second subject alone. Until that day's cut this tool also summed
    likenesses over all subjects at once; nothing here compares anybody with anybody.
    """
    import importlib
    tool = importlib.import_module("who_is_who_across_shots")
    lead, second = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()   # noqa: E731
    def who(f, mask):
        is_lead = torch.equal(mask, lead)
        if 8 <= f < 16:          # the middle shot: each unlike themselves
            return e(1.0, 0.5, 0.6) if is_lead else e(0.5, 1.0, 0.6)
        return e(1.0, 0, 0) if is_lead else e(0, 1.0, 0)
    def detect(f: int):
        if f < 8:
            return torch.stack([second, lead]), [0.9, 0.8]
        if f < 16:               # not in left-to-right order, so a number is not an index
            return torch.stack([lead, second]), [0.9, 0.8]
        return torch.stack([second]), [0.9]
    sign = lambda f, m: (who(f, m), who(f, m))   # noqa: E731
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731
    def run(corrections=None, held=None):
        got = st.follow(24, [8, 16], st.PICK_LARGEST, 2, 0.99, detect, sign, track, stride=4, offset=1, corrections=corrections, held=held)
        mask = st.assemble(24, H, W, got.pieces)
        table = st.shot_table.build(got, detect, mask, state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                    named_frame=True, named_value=True, cuts=[8, 16], held=held)
        return got, mask, json.loads(st.shot_table.as_json(table))
    got_lead, mask_lead, table_lead = run()
    got_second, _, table_second = run(held=st.held_people(detect, mask_lead))
    if [s.seed for s in got_lead.shots] != [2, None, None] or [s.seed for s in got_second.shots] != [2, None, 17]:
        problems.append(f"rows: the control: alone, the lead's tracker holds shot 1 only and the second's shots 1 and 3; "
                        f"seeds {[s.seed for s in got_lead.shots]}, {[s.seed for s in got_second.shots]}")
    tables = {"lead": table_lead, "second": table_second}
    out = tool.read_rows(tables, first_source_frame=600)
    rows = out["rows"]
    if [r["status"] for r in rows] != ["answered", "ask", "ask"] or out["not_picked"] or out["first_source_frame"] != 600:
        problems.append(f"rows: both picked in shot 1 is answered; shot 2 (both refused) and shot 3 (the second taken by its own tracker, "
                        f"the lead called absent by its own) ask: {[r['status'] for r in rows]}, {out['not_picked']}")
    # the file is exactly the shape the gate reads: nothing of the old arithmetic is in it
    if sorted(out) != ["first_source_frame", "frames_are", "not_picked", "rows", "status_is", "tables", "what"] \
            or sorted(rows[1]) != ["frames", "note", "shot", "status", "trackers", "why"] \
            or sorted(rows[1]["trackers"]["lead"]) != ["answered_by", "ask", "called", "called_person", "numbers_on_its_tile", "tile_frame", "typed_in_this_table"]:
        problems.append(f"rows: the file's keys moved: {sorted(out)}, {sorted(rows[1])}, {sorted(rows[1]['trackers']['lead'])}")
    two, three = rows[1]["trackers"], rows[2]["trackers"]
    if rows[0]["trackers"]["lead"]["answered_by"] != "picked" or any(c["ask"] for c in rows[0]["trackers"].values()):
        problems.append(f"rows: a pick is a person's word and is asked about by nobody: {rows[0]['trackers']}")
    for label in ("lead", "second"):
        ask = two[label]["ask"]
        if two[label]["numbers_on_its_tile"] != [1, 2] or f"which numbered person is {label}? 1, 2, or none" not in ask \
                or "Its tracker has nobody (absent)" not in ask or f"into {label}'s tracker" not in ask:
            problems.append(f"rows: shot 2's question for {label} is in its tile's own numbers and says what its tracker called: {ask!r}")
    if "Its tracker has person 1 (taken)" not in three["second"]["ask"] or "Its tracker has nobody (absent)" not in three["lead"]["ask"] \
            or "nobody has said who lead, second are in this shot" != rows[2]["why"]:
        problems.append(f"rows: a tracker's own take and a tracker's own absence are both nobody's word: {rows[2]['why']!r}, {three}")
    # the answers typed back into the trackers make every row answered, and nothing is asked again
    n_lead = st.shot_table.person_number(detect(9)[0], 0)
    n_second = st.shot_table.person_number(detect(9)[0], 1)
    _, fixed_mask, fixed_lead = run(corrections={2: n_lead, 3: None})
    _, _, fixed_second = run(corrections={2: n_second, 3: 1}, held=st.held_people(detect, fixed_mask))
    done = tool.read_rows({"lead": fixed_lead, "second": fixed_second})
    if [r["status"] for r in done["rows"]] != ["answered"] * 3 or any(c["ask"] for r in done["rows"] for c in r["trackers"].values()):
        problems.append(f"rows: typed answers are a person's word; the rows are answered and ask nothing: {[(r['status'], r['why']) for r in done['rows']]}")
    if done["rows"][1]["trackers"]["lead"]["answered_by"] != "corrected by hand" or "lead (corrected by hand)" not in done["rows"][1]["note"] \
            or done["rows"][2]["trackers"]["lead"]["typed_in_this_table"] != "none":
        problems.append(f"rows: a corrected shot says whose word it is and what was typed: {done['rows'][1]['note']!r}")
    # ONE NUMBERING: the second tracker's tile of shot 1 lists the lead too, marked held, and the two trackers' calls
    # there are different numbers on the same frame
    held_marks = [p["held_by_another_tracker"] for p in table_second["shots"][0]["people"]]
    if len(held_marks) != 2 or sum(held_marks) != 1 or rows[0]["trackers"]["lead"]["called_person"] == rows[0]["trackers"]["second"]["called_person"]:
        problems.append(f"rows: the later tracker's tile lists the held person, and the two picks are two numbers: {held_marks}")
    # two trackers on ONE person on one frame is asked about even when both are a person's word: here the second
    # tracker is typed onto the person the lead's tracker picked, whom the lead's holds
    _, _, on_the_lead = run(corrections={1: rows[0]["trackers"]["lead"]["called_person"]}, held=st.held_people(detect, mask_lead))
    clash = tool.read_rows({"lead": table_lead, "second": on_the_lead})["rows"][0]
    if clash["status"] != "ask" or "two trackers on one person" not in clash["why"] \
            or not all(c["ask"].startswith("TWO TRACKERS ON ONE PERSON: lead and second both have person") for c in clash["trackers"].values()) \
            or not on_the_lead["shots"][0]["subject"]["also_held_by_another_tracker"]:
        problems.append(f"rows: two trackers on one person on one frame must ask both, whoever's word each is: {clash['status']}, {clash['why']!r}")
    # with a THIRD tracker in the clash's shot whose call is its own, all three are asked in one round, and the reason names it
    third = json.loads(json.dumps(table_second))
    third["shots"][0]["subject"].update(state="taken", person=2 if rows[0]["trackers"]["lead"]["called_person"] == 1 else 1)
    three_way = tool.read_rows({"lead": table_lead, "second": on_the_lead, "third": third})["rows"][0]
    if not all(c["ask"] for c in three_way["trackers"].values()) or "and nobody has said who third is in this shot" not in three_way["why"] \
            or "TWO TRACKERS" in three_way["trackers"]["third"]["ask"]:
        problems.append(f"rows: beside two trackers on one person, a third whose call is its own is asked in the same round: {three_way['why']!r}")
    # nobody detected, and only things with no head, are answers; a tracker that took a thing there is asked about
    empty = json.loads(json.dumps(tables))
    for t in empty.values():
        t["shots"][2]["people"], t["shots"][2]["looks"] = [], []
        t["shots"][2]["subject"].update(state="absent", person=None)
    if tool.read_rows(empty)["rows"][2]["status"] != "answered" or "no table lists anybody" not in tool.read_rows(empty)["rows"][2]["note"]:
        problems.append("rows: a shot with nobody detected is answered: every subject has nobody")
    things = json.loads(json.dumps(tables))
    for t in things.values():
        for person in t["shots"][2]["people"] + [p for look in t["shots"][2]["looks"] for p in look["people"]]:
            person["has_head"] = False
    took_a_thing = tool.read_rows(things)["rows"][2]
    if took_a_thing["status"] != "ask" or "second took a detection in a shot where nothing with a head was found" != took_a_thing["why"] \
            or took_a_thing["trackers"]["lead"]["ask"]:
        problems.append(f"rows: a tracker that took a headless detection is asked about, and one that took nothing there is not: {took_a_thing['why']!r}")
    things["second"]["shots"][2]["subject"].update(state="absent", person=None)
    if tool.read_rows(things)["rows"][2]["status"] != "answered":
        problems.append("rows: a shot whose only detections have no head, with nobody taken, is answered")
    if not all(p["has_head"] is True and tool.headed(p) for p in table_lead["shots"][0]["people"]) or tool.headed({"has_head": False}) \
            or not tool.headed({"has_head": None}) or not tool.headed({}):
        problems.append("rows: a detection is a thing only when the table says it has no head; unknown counts as a person, so the shot asks")
    if any("has_head" not in p for t in tables.values() for shot in t["shots"] for p in shot["people"] + [q for look in shot["looks"] for q in look["people"]]):
        problems.append("rows: a version 2 table carries `has_head` on every person of every tile and look")
    # a tracker with its subject on no shot is the first line, and says when held people are on its tiles
    lost = json.loads(json.dumps(tables))
    for shot in lost["second"]["shots"]:
        shot["subject"].update(state="absent", person=None)
    said = tool.read_rows(lost)["not_picked"]
    if list(said) != ["second"] or "held by a tracker that runs before it" not in said["second"] or "a typed correction may name a held person" not in said["second"]:
        problems.append(f"rows: a tracker with its subject on no shot is named first, with the held people on its tiles: {said}")
    # an older table, and tables of two loads, are refused by name
    for breaker, need in ((lambda t: t["lead"].update(version=1), "version 1"), (lambda t: t["second"].update(frames=25), "not tables of one load")):
        bad = json.loads(json.dumps(tables))
        breaker(bad)
        try:
            tool.read_rows(bad)
            problems.append(f"rows: tables that are {need} were read")
        except SystemExit as err:
            if need not in str(err):
                problems.append(f"rows: the refusal does not say `{need}`: {err}")
    # tables of TWO previews (2026-10-12: one of 457 frames and one of 406 for one load of two subjects): with the frames to cut to,
    # each table is cut to them and read; the lead's here is a preview of shot 1 only, the second's the whole clip
    short = {**json.loads(json.dumps(table_lead)), "frames": 8, "cuts": []}
    short["shots"] = short["shots"][:1]
    two = {"lead": short, "second": table_second}
    try:
        tool.read_rows(two, first_source_frame=600)
        problems.append("rows: tables of two previews were read with no frames to cut them to")
    except SystemExit as err:
        if "not tables of one load" not in str(err) or "--frames" not in str(err):
            problems.append(f"rows: the refusal of two previews' tables does not say how to cut them: {err}")
    cut = tool.read_rows(two, first_source_frame=600, frames=8)
    if [(r["frames"], r["status"]) for r in cut["rows"]] != [([0, 7], "answered")] or cut["cut_to"]["frames"] != 8 or "across_previews" not in cut:
        problems.append(f"rows: two previews' tables cut to the frames both hold give one answered row over them: {cut['rows']}")
    # a cut that begins inside the tables: shots clipped and renumbered from the cut's frame 0, a tile before the cut kept and marked,
    # and a question typed by the shot's number in the tracker's OWN table
    late = tool.read_rows(tables, first_source_frame=604, frames=8, firsts={"lead": 600, "second": 600})
    cells = late["rows"][0]["trackers"]
    if [r["frames"] for r in late["rows"]] != [[0, 3], [4, 7]] or cells["lead"]["tile_frame"] != -2 or not cells["lead"]["tile_outside_the_frames"] \
            or late["rows"][1]["trackers"]["lead"]["shot_in_its_table"] != 2 or "shot 2, lead's tile" not in late["rows"][1]["trackers"]["lead"]["ask"]:
        problems.append(f"rows: a cut from frame 4 for 8 frames is shots 1 and 2 clipped to 0-3 and 4-7, the first shot's tile marked outside: "
                        f"{[r['frames'] for r in late['rows']]}, {cells['lead']}")
    # a preview of the middle shot alone numbers it 1, where the whole clip's table numbers it 2: each tracker's question
    # is typed by the number in ITS table
    middle = json.loads(json.dumps(table_lead["shots"][1]))
    middle.update(shot=1, first_frame=0, last_frame=7, shown_frame=middle["shown_frame"] - 8)
    mid = tool.read_rows({"second": table_second, "lead": {**table_lead, "frames": 8, "cuts": [], "shots": [middle]}},
                         first_source_frame=608, frames=8, firsts={"second": 600})["rows"]
    if len(mid) != 1 or mid[0]["shot"] != 2 or "shot 1, lead's tile" not in mid[0]["trackers"]["lead"]["ask"] \
            or "shot 2, second's tile" not in mid[0]["trackers"]["second"]["ask"]:
        problems.append(f"rows: a tracker's question names the shot by its number in that tracker's own table: {mid}")
    # a table that begins on another source frame says so, and is cut by its own clock
    if tool.read_rows(two, first_source_frame=600, frames=8, firsts={"second": 600})["rows"] != cut["rows"]:
        problems.append("rows: a table told its own first frame, the same as the load's, read differently")
    moved_cut = json.loads(json.dumps(tables))
    moved_cut["lead"].update(cuts=[10, 16])
    moved_cut["lead"]["shots"][0]["last_frame"], moved_cut["lead"]["shots"][1]["first_frame"] = 9, 10
    for bad, need, kw in ((two, "does not cover", dict(first_source_frame=600, frames=12)),
                          (two, "does not cover", dict(first_source_frame=600, frames=8, firsts={"lead": 604})),
                          (moved_cut, "the cuts differ", dict(first_source_frame=600, frames=24)),
                          ({"lead": short, "second": {**table_second, "size": [8, 8]}}, "two picture sizes", dict(first_source_frame=600, frames=8)),
                          (two, "give --first-frame", dict(frames=8))):
        try:
            tool.read_rows(bad, **kw)
            problems.append(f"rows: two previews' tables were cut and read where the refusal `{need}` was due")
        except SystemExit as err:
            if need not in str(err):
                problems.append(f"rows: the refusal does not say `{need}`: {err}")
    if "[608, 616]" not in _refusal(lambda: tool.read_rows(moved_cut, first_source_frame=600, frames=24)) \
            or "[610, 616]" not in _refusal(lambda: tool.read_rows(moved_cut, first_source_frame=600, frames=24)):
        problems.append("rows: cuts that differ inside the frames are not named in source frames for both tables")
    # a cut that leaves the differing cut outside is read: only cuts INSIDE the frames must agree
    if len(tool.read_rows(moved_cut, first_source_frame=600, frames=8)["rows"]) != 1:
        problems.append("rows: tables whose cuts differ only after the frames asked for were not read")
    source = (REPO / "bench" / "who_is_who_across_shots.py").read_text(encoding="utf-8")
    for gone in ("hand_out", "the_sum_says", "the_evidence_says", "suggest", "levels", "signatures\"]["):
        if gone in source:
            problems.append(f"rows: the tool still carries `{gone}`, which the cut of 2026-10-10 removed")


def check_seeds(problems):
    """A seed is a correction a tool can write: a frame of the load and a box, not a number off a tile (2026-10-10)."""
    boxes, detect, sign, track, calls = _world()
    cuts = [8, 16, 24]
    shots = st.shot_ranges(32, cuts)
    as_box = lambda p: ", ".join(str(v) for v in st.shot_table._box(boxes[p]))   # noqa: E731
    # parsing: the frame picks the shot; none; and every refusal says which line
    got = st.parse_seeds(f"frame 13: {as_box(2)}\nframe 26: none; frame 2: {as_box(1)}", shots)
    if got != {2: [(13, tuple(float(v) for v in st.shot_table._box(boxes[2])))], 4: [(26, None)], 1: [(2, tuple(float(v) for v in st.shot_table._box(boxes[1])))]}:
        problems.append(f"seeds: three lines should be three shots' seeds, each on its own frame: {got}")
    for name, text in (("not a seed", "shot 2: person 1"), ("a frame the load does not have", "frame 32: none"), ("a box with no area", "frame 3: 4, 4, 0, 9"),
                       ("two `none` in one shot", "frame 9: none\nframe 10: none"), ("three numbers", "frame 3: 1, 2, 3"),
                       ("two boxes on one frame", "frame 9: 1, 1, 5, 5\nframe 9: 2, 2, 5, 5"),
                       ("`none` beside a box", "frame 9: none\nframe 10: 1, 1, 5, 5")):
        try:
            st.parse_seeds(text, shots)
        except ValueError as exc:
            if "seeds:" not in str(exc):
                problems.append(f"seeds: {name} is refused without saying it is a seed's line: {exc}")
        else:
            problems.append(f"seeds: {name} was read")
    if st.parse_seeds("", shots) != {} or st.parse_seeds("  \n ; ", shots) != {}:
        problems.append("seeds: an empty text is not no seeds")
    # the control: with no seeds the tracker does what it did (person 0, the largest, through the shots they are in)
    plain = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    none = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds={})
    if [(s.seed, s.shown, s.index, st._state(s)) for s in plain.shots] != [(s.seed, s.shown, s.index, st._state(s)) for s in none.shots] \
            or any(s.seeded is not None for s in none.shots) or not torch.equal(st.assemble(32, H, W, plain.pieces), st.assemble(32, H, W, none.pieces)):
        problems.append("seeds: with no seeds the tracker's seeds, tiles, states or masks are not what they were")
    # shot 2 (frames 8-15) holds people 1 and 2 and not the subject. A seed on frame 13 naming person 2's box takes
    # person 2 there, whatever frame the tracker's own pass showed; a seed of none empties the last shot
    calls["track"].clear()
    seeds = st.parse_seeds(f"frame 13: {as_box(2)}\nframe 26: none", shots)
    got = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=seeds)
    two, four = got.shots[1], got.shots[3]
    if (8, 16, 13, 2) not in calls["track"] or two.seed != 13 or two.shown != 13 or st._state(two) != "taken (seeded)" \
            or two.seeded["person"] != 2 or two.seeded["overlap"] < 0.999 or two.seeded["refused"]:
        problems.append(f"seeds: a seed on frame 13 naming person 2's box should track person 2 through shot 2 from frame 13: {calls['track']}, {two.seeded}")
    if four.seed is not None or st._state(four) != "absent (seeded)" or four.start in got.pieces or four.shown != 26:
        problems.append(f"seeds: `none` should leave the shot empty and say it was seeded so: {st._state(four)}")
    if [st._state(s) for s in (got.shots[0], got.shots[2])] != [st._state(s) for s in (plain.shots[0], plain.shots[2])]:
        problems.append("seeds: a seed changed a shot it does not name")
    text = st.report(got, cuts, st.PICK_LARGEST, "person", True, True, 1.0)
    for need in ("[2] frames 8-15: seeded on load frame 13: took person 2 of the 2 detection(s) there", "[4] frames 24-31: seeded on load frame 26: nobody"):
        if need not in text:
            problems.append(f"seeds: the report lacks {need!r}: {text!r}")
    table = st.shot_table.build(got, detect, st.assemble(32, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=cuts)
    row = table["shots"][1]
    if row["seeded"]["frame"] != 13 or row["seeded"]["person"] != 2 or row["subject"]["state"] != "taken (seeded)" or row["subject"]["person"] != 2 \
            or row["shown_frame"] != 13 or "seeded" not in row["subject"]["why"] or table["shots"][0]["seeded"] is not None \
            or table["shots"][3]["seeded"]["box"] is not None or table["shots"][3]["subject"]["state"] != "absent (seeded)":
        problems.append(f"seeds: the table's rows do not say what was seeded: {row['seeded']}, {row['subject']}")
    # a box no detection agrees with, and no detection's mask overlaps at SEED_OVERLAPS (three quarters off the person,
    # 0.14), with no box-prompted segmentation to fall back on: the shot is left EMPTY, never the tracker's own guess, and
    # the report gives the nearest detection's box and overlap so the writer can tell a tighter mask from another person
    calls["track"].clear()
    x, y, w, h = st.shot_table._box(boxes[0])
    far = f"{x + 3 * w // 4}, {y}, {w}, {h}"
    off = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1,
                    seeds=st.parse_seeds(f"frame 28: {far}", shots))
    last = off.shots[3]
    if st._state(last) != "absent (seed refused)" or last.seed is not None or last.start in off.pieces or any(c[0] == 24 for c in calls["track"] if c[2] == 28):
        problems.append(f"seeds: a box three quarters off the person must be refused and the shot left empty (plain took it): {st._state(last)}, {calls['track']}")
    if not (0.1 < last.seeded["overlap"] < st.SEED_OVERLAPS) or last.seeded["nearest_box"] != [x, y, w, h] or last.seeded["how"] != st.SEED_REFUSED \
            or "no box-prompted segmentation" not in last.seeded["why"]:
        problems.append(f"seeds: a refused seed keeps the nearest detection's box and its overlap, and says why: {last.seeded}")
    # HALF off the person (2026-10-23): its box agrees under the line, and its mask overlaps the box 0.33, at the
    # mask line or over it: the person is taken, and the record says how
    half = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1,
                     seeds=st.parse_seeds(f"frame 28: {x + w // 2}, {y}, {w}, {h}", shots))
    if st._state(half.shots[3]) != "taken (seeded)" or half.shots[3].seeded["how"] != st.SEED_BY_OVERLAP or half.shots[3].index != 0 \
            or abs(half.shots[3].seeded["mask_overlap"] - 0.333) > 0.01:
        problems.append(f"seeds: a box half on a person is taken by their mask's overlap with it: {half.shots[3].seeded}")
    said = st.report(off, cuts, st.PICK_LARGEST, "person", True, True, 1.0)
    for need in ("SEED REFUSED on load frame 28", f"box {[x, y, w, h]}", "The shot is left empty", "the line is 0.90"):
        if need not in said:
            problems.append(f"seeds: the refusal's report lacks {need!r}: {said!r}")
    # the line itself: a box that agrees at the line is taken, one just under it is refused
    if abs(st.SEED_AGREES - 0.9) > 1e-9:
        problems.append("seeds: SEED_AGREES moved; read its provenance before changing this check")
    # since 2026-10-23 the line decides HOW the person is taken: at it, the boxes agree; just under it, the mask's overlap
    for shift, how in ((2, st.SEED_BY_BOX), (3, st.SEED_BY_OVERLAP)):       # 40 columns wide: 2 off is 38/42 = 0.905, 3 off is 37/43 = 0.860
        near = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1,
                         seeds=st.parse_seeds(f"frame 28: {x + shift}, {y}, {w}, {h}", shots))
        if near.shots[3].seed is None or near.shots[3].seeded["how"] != how:
            problems.append(f"seeds: a box {shift} columns off a {w}-wide person should be taken as {how!r}: {near.shots[3].seeded}")
    # a seed MAY name a person another tracker holds: a person's word outranks a tracker's take, and the shot says so
    held = torch.zeros((32, H, W))
    held[13] = boxes[2]
    theirs = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=seeds, held=st.held_people(detect, held))
    if st._state(theirs.shots[1]) != "taken (seeded)" or not theirs.shots[1].also_held:
        problems.append(f"seeds: a seed naming a person `others` holds is taken and marked as also held: {st._state(theirs.shots[1])}, {theirs.shots[1].also_held}")
    if any(shot.also_held for shot in st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=seeds).shots):
        problems.append("seeds: with no `others` wired a seeded shot is marked as also held")
    # a seed and a typed correction on one shot are two words for it: an error naming both
    try:
        st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=seeds, corrections={2: 1})
    except ValueError as exc:
        if "both a seed" not in str(exc):
            problems.append(f"seeds: a seed and a correction on one shot are refused without saying so: {exc}")
    else:
        problems.append("seeds: a seed and a correction on one shot were both taken")
    # with nothing picked at all (no detection on the pick frame's rule), seeds still apply: a seeded load needs no pick
    empty_then = lambda f: detect(f) if f >= 8 else (torch.zeros((0, H, W)), [])   # noqa: E731
    unpicked = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, empty_then, sign, track, stride=4, offset=1, seeds=seeds)
    if unpicked.pick_frame is not None or st._state(unpicked.shots[1]) != "taken (seeded)":
        problems.append(f"seeds: with nobody picked a seeded shot should still be taken: {st._state(unpicked.shots[1])}")
    # mrcorn's cold read: A SEED ON THE PICKED SHOT IS THE PICK. The rule picks person 0 (the largest on frame 3); a
    # seed there naming person 1 makes person 1 the subject of the WHOLE load: shot 2, which holds 1 and 2, is taken
    # on person 1, and shots 3 and 4, which hold person 0 and never person 1, are left alone. Before this the seed
    # changed shot 1 only and the mask followed person 1 there and person 0 after, with nothing said.
    calls["track"].clear()
    first = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1,
                      seeds=st.parse_seeds(f"frame 5: {as_box(1)}", shots), sign_whole=lambda f, m: sign(f, m))
    who = sorted({c[3] for c in calls["track"]})
    if st._state(first.shots[0]) != "taken (seeded)" or first.pick_seed != "is the pick" or first.pick_frame != 5 or first.rule_pick != (3, 1) \
            or who != [1] or [s.seed is not None for s in first.shots] != [True, True, False, False]:
        problems.append(f"seeds: a seed on the picked shot naming person 1 must make person 1 the subject everywhere: tracked {calls['track']}, "
                        f"states {[st._state(s) for s in first.shots]}, pick {first.pick_frame}, rule's {first.rule_pick}")
    if len(first.gallery) == 0 or len(first.gallery) != len(first.gallery_frames) or len(first.gallery_whole) != len(first.gallery) \
            or any(float((g[0] * sign(0, boxes[1])).sum()) < 0.999 for g in first.gallery):
        problems.append("seeds: when a seed is the pick the gallery is made from the seeded track, of the person the seed names")
    said = st.report(first, cuts, st.PICK_LARGEST, "person", True, True, 1.0)
    for need in ("subject: the `person` a seed names on frame 5; every other shot is judged against them", "frame 3) was somebody else or another frame and is not used"):
        if need not in said:
            problems.append(f"seeds: the report of a seeded pick lacks {need!r}: {said!r}")
    # a seed on the picked shot naming the SAME person as the rule, on the rule's own frame, changes no other shot
    same = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=st.parse_seeds(f"frame 3: {as_box(0)}", shots))
    if same.rule_pick is not None or [s.seed is not None for s in same.shots] != [s.seed is not None for s in plain.shots] \
            or not torch.equal(st.assemble(32, H, W, same.pieces), st.assemble(32, H, W, plain.pieces)):
        problems.append("seeds: seeding the picked shot with the rule's own person on the rule's own frame changed the masks")
    # a seed on the picked shot that takes nobody cannot be a reference: the rule's pick stays one, and the report says so
    for text in ("frame 5: none", f"frame 5: {far}"):
        nobody = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=st.parse_seeds(text, shots))
        if nobody.pick_seed != "took nobody" or nobody.shots[0].seed is not None or nobody.pick_frame != 3 \
                or "THE PICKED SHOT'S SEED TOOK NOBODY" not in st.report(nobody, cuts, st.PICK_LARGEST, "person", True, True, 1.0):
            problems.append(f"seeds: `{text}` on the picked shot must leave it empty, keep the rule's pick as the reference and say so: {nobody.pick_seed}")
    # the tile of a seeded shot prints no likeness of the automatic pass; a seed's own figure is its box's overlap
    labels = [st.tile_label(n, s) for n, s in enumerate(off.shots, 1)] + [st.tile_label(2, got.shots[1]), st.tile_label(4, got.shots[3])]
    if not labels[3].endswith("box 0.14  absent (seed refused)") or not labels[4].endswith("box 1.00  taken (seeded)") \
            or not labels[5].endswith("frames 24-31  absent (seeded)"):
        problems.append(f"seeds: a seeded tile's header carries the automatic pass's score, or not the seed's own overlap: {labels[3:]}")
    row = st.shot_table.build(off, detect, st.assemble(32, H, W, off.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                              named_frame=True, named_value=True, cuts=cuts)
    if row["shots"][3]["subject"]["similarity"] is not None or row["pick_seed"] != "" \
            or st.shot_table.build(first, detect, st.assemble(32, H, W, first.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                   named_frame=True, named_value=True, cuts=cuts)["pick_seed"] != "is the pick":
        problems.append("seeds: the table writes a seeded shot's automatic similarity, or does not say that a seed was the pick")


def _removing(detect, others, most=None):
    """THE OLD FORM, kept here as the reference a take is compared with: `detect` with held people REMOVED from the list.
    It left the tree on 2026-10-10 (`held_people` marks them); the tracker must take exactly whom it took on this."""
    most = st.ON_OTHERS if most is None else most
    kept = {}

    def detect_others_removed(f):
        if f not in kept:
            masks, scores = detect(f)
            keep = [i for i in range(int(masks.shape[0])) if st.share_on(masks[i], others[f]) <= most]
            kept[f] = (masks[keep], [scores[i] for i in keep if i < len(scores)])
        return kept[f]

    return detect_others_removed


def _takes(got, detector):
    """Per shot: the frame it was taken on and the MASK of the person taken (a list's index means nothing across two lists)."""
    return [(shot.seed, None if shot.seed is None else detector(shot.seed)[0][shot.index].flatten().tolist(), round(float(shot.best), 6))
            for shot in got.shots]


def check_only_person_off_with_others(problems):
    """2026-10-10, three trackers: the third took another subject as "the only person" at 0.59, and the second took a
    subject small in an inset twice. With `others` wired a lone person may be any of the job's named people: the rule is off."""
    me, held_person, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50), _box(100, 20, 120, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()   # noqa: E731
    sigs = [(me, e(1.0, 0, 0)), (held_person, e(0, 1.0, 0)), (other, e(0.6, 0, 0.8))]      # the other is 0.6 like me: under any line
    sign = lambda _f, m: next(v for b, v in sigs if torch.equal(b, m))   # noqa: E731
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731
    def detect(f: int):          # shot 1: me alone. shot 2: the held person and the other. shot 3: the other alone
        people = [me] if f < 8 else ([held_person, other] if f < 16 else [other])
        return torch.stack(people), [0.9] * len(people)
    theirs = torch.zeros((24, H, W))
    theirs[8:16] = held_person
    # the control: with no other tracker, a shot holding ONE person takes them under the line as the only person
    plain = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, sign, track, stride=4, offset=1)
    if [st._state(s) for s in plain.shots] != ["picked", "absent", "taken (only person)"]:
        problems.append(f"only person: the control: two people in shot 2 is absent, one in shot 3 is taken as the only person: {[st._state(s) for s in plain.shots]}")
    got = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, sign, track, stride=4, offset=1, held=st.held_people(detect, theirs))
    if [st._state(s) for s in got.shots] != ["picked", "absent", "absent"]:
        problems.append(f"only person: with `others` wired nobody is taken as the only person, in shot 2 (one left beside a held person) "
                        f"or in shot 3 (one person, nobody held there): {[st._state(s) for s in got.shots]}")
    # ...even when the other tracker's mask is empty on every frame: the rule is off because the job names several people
    empty = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, detect, sign, track, stride=4, offset=1, held=st.held_people(detect, torch.zeros((24, H, W))))
    if st._state(empty.shots[2]) != "absent":
        problems.append(f"only person: `others` wired and empty still switches the rule off: {st._state(empty.shots[2])}")
    # the likeness line still decides: the subject themselves, in a shot beside a held person, is taken by likeness
    back = lambda f: (torch.stack([held_person, me]), [0.9, 0.8]) if 8 <= f < 16 else detect(f)   # noqa: E731
    mine = st.follow(24, [8, 16], st.PICK_LARGEST, 2, None, back, sign, track, stride=4, offset=1, held=st.held_people(back, theirs))
    if st._state(mine.shots[1]) != "taken" or mine.shots[1].best < 0.99:
        problems.append(f"only person: the subject beside a held person is still taken by likeness: {st._state(mine.shots[1])}, {mine.shots[1].best}")
    source = (REPO / "subject_track.py").read_text(encoding="utf-8")
    if "alone = match_threshold is None and not others and held is None" not in source or "held=held)" not in source:
        problems.append("only person: the node does not tell `follow` that `others` is wired, or the rule does not read it")


def check_others(problems):
    """Two trackers, one person (2026-10-10): what another tracker holds is LISTED, marked, and never this one's subject."""
    boxes, detect, sign, track, calls = _world()
    cuts = [8, 16, 24]
    held_mask = torch.zeros((32, H, W))
    for f in list(range(0, 8)) + list(range(20, 32)):       # where person 0 is: the other tracker's mask
        held_mask[f] = boxes[0]
    # the control: with nothing held, the largest person on frame 3 is person 0
    plain = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1)
    if [c[3] for c in calls["track"]] != [0, 0, 0]:
        problems.append(f"others: the control did not follow person 0 as the largest: {calls['track']}")
    calls["track"].clear()
    held = st.held_people(detect, held_mask)
    got = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, held=held)
    if calls["track"] != [(0, 8, 3, 1), (8, 16, 9, 1)]:
        problems.append(f"others: with person 0 held by another tracker the node should follow person 1 through the two "
                        f"shots they are in and nobody after; it tracked {calls['track']}")
    if [s.seed for s in got.shots] != [3, 9, None, None]:
        problems.append(f"others: seeds {[s.seed for s in got.shots]}; shot 3 holds a stranger and the held person, shot 4 only "
                        "the held person, so both are absent")
    seen = {f: len(v) for f, v in held.seen.items()}
    if seen.get(3) != 1 or seen.get(9) != 0 or not all(seen[f] == 1 for f in seen if f >= 24):
        problems.append(f"others: the count of held detections per frame looked at is {seen}")
    # THE LIST IS EVERYBODY: the frame's people are not shortened, on any frame, and the held person keeps a place in it
    if int(detect(3)[0].shape[0]) != 2 or got.shots[3].seen != 1 or got.shots[3].best >= 0:
        problems.append(f"others: on the shot holding only the held person they are listed (seen 1) and compared with nobody: "
                        f"seen {got.shots[3].seen}, best {got.shots[3].best}")
    mask = st.assemble(32, H, W, got.pieces)
    if st.frames_on_others(mask, held_mask) != []:
        problems.append("others: the track of person 1 is reported as lying on the held person")
    drifted = mask.clone()
    drifted[5:8] = boxes[0]                                 # the tracker moved onto the held person, no empty frame between
    if st.frames_on_others(drifted, held_mask) != [5, 6, 7]:
        problems.append(f"others: a track that moved onto the held person is on them on {st.frames_on_others(drifted, held_mask)}, not 5 to 7")
    # the line: half on is not held, a pixel more is; and both stay in the list
    half, more = torch.zeros((H, W)), torch.zeros((H, W))
    half[10:20, 0:20] = 1.0
    more[10:20, 0:20] = 1.0
    theirs = torch.zeros((H, W))
    theirs[10:20, 0:10] = 1.0
    if abs(st.share_on(half, theirs) - 0.5) > 1e-6 or st.share_on(torch.zeros((H, W)), theirs) != 0.0:
        problems.append("others: `share_on` is not the share of a mask's own pixels, or an empty mask has one")
    theirs_more = theirs.clone()
    theirs_more[10:20, 10:11] = 1.0
    line = st.held_people(lambda f: (torch.stack([half, more]), [0.9, 0.8]), torch.stack([theirs, theirs_more]))
    if line(0) != frozenset() or line(1) != frozenset({0, 1}):
        problems.append(f"others: exactly half on the held person is not held and a pixel more is; got {line(0)}, {line(1)}")
    text = st.others_report(got.shots, held.seen, [5, 6, 7])
    for need in ("lie mostly on a person another tracker holds: listed and numbered, marked held, never candidates", "[4] no subject",
                 "cut 3 frame(s) where this track lay mostly on another tracker's person: 5-7"):
        if need not in text:
            problems.append(f"others: the report lacks {need!r}: {text!r}")
    table = st.shot_table.build(got, detect, mask, state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=cuts, held=held, on_others=[5, 6, 7])
    rows = table["shots"]
    marks = [[p["held_by_another_tracker"] for p in r["people"]] for r in rows]
    if table["version"] != 2 or "every detection on the frame" not in table["people_are"] or any("others_left_out" in r for r in rows):
        problems.append(f"others: a table that lists everybody is version 2, says so, and has no `others_left_out`: {table['version']}")
    if [len(m) for m in marks[:2]] != [2, 2] or sum(marks[0]) != 1 or any(marks[1]) or marks[3] != [True]:
        problems.append(f"others: the table's people are everybody, with the held one marked: {marks}")
    if rows[0]["subject"]["person"] is None or rows[0]["people"][rows[0]["subject"]["person"] - 1]["held_by_another_tracker"]:
        problems.append("others: the subject's own number is read in the whole list and is not the held person's")
    if rows[0]["frames_on_others"] != [[5, 7]] or rows[1]["frames_on_others"] != [] or any(r["subject"]["also_held_by_another_tracker"] for r in rows):
        problems.append(f"others: where the track sits on others, or a take marked as also held with nothing typed: {[r['frames_on_others'] for r in rows]}")
    if any(p["held_by_another_tracker"] for look in rows[2]["looks"] for p in look["people"] if look["frame"] < 20) \
            or not any(p["held_by_another_tracker"] for look in rows[2]["looks"] for p in look["people"]):
        problems.append("others: a look's people carry the mark where a held person is on that frame, and only there")
    bare = st.shot_table.build(plain, detect, st.assemble(32, H, W, plain.pieces), state=st._state, phrase="person",
                               pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=cuts)
    if any(r["frames_on_others"] is not None or r["track_score"] is not None or any(p["held_by_another_tracker"] for p in r["people"]) for r in bare["shots"]):
        problems.append("others: a tracker with no `others` and no scores writes a mark, or something other than null for them")
    if abs(st.ON_OTHERS - 0.5) > 1e-9:
        problems.append("others: ON_OTHERS moved; read the day's figures beside it before changing this check")
    # the tile numbers everybody, and says which are held
    tiles = st.preview(torch.zeros((32, H, W, 3)), mask, got.shots, detect, held=held)
    if tiles.shape[0] != 4:
        problems.append(f"others: the preview with held people did not draw a tile a shot: {tuple(tiles.shape)}")

    # JUDGING DID NOT MOVE: against the old form, which removed held people from the list, the same person is taken
    # on the same frame at the same score in every shot. Compared by the taken person's MASK, since the two lists
    # number people differently, which is the fault the one list removes.
    old = st.follow(32, cuts, st.PICK_LARGEST, 3, 0.8, _removing(detect, held_mask), sign, track, stride=4, offset=1)
    if _takes(old, _removing(detect, held_mask)) != _takes(got, detect):
        problems.append("others: marking held people changed a take against removing them, with a named pick frame")
    for pick in st.PICKS:
        new_auto = st.follow(32, cuts, pick, None, 0.8, detect, sign, track, stride=4, offset=1, held=st.held_people(detect, held_mask))
        old_auto = st.follow(32, cuts, pick, None, 0.8, _removing(detect, held_mask), sign, track, stride=4, offset=1)
        if _takes(old_auto, _removing(detect, held_mask)) != _takes(new_auto, detect):
            problems.append(f"others: marking changed a take against removing, with the automatic pick `{pick}`")
        if any(s.seed is not None and s.index in held(s.seed) for s in new_auto.shots):
            problems.append(f"others: the automatic pick `{pick}` took a held person")
    # THE MARK IS READ AFTER THE DUPLICATE JOIN, as the removal was (mrcorn's pin). One person detected twice: the
    # whole of them, mostly OFF the other tracker's mask, and a smaller detection of them the detector scored
    # higher, mostly ON it. Joined first, the smaller stands for the person and the person is held; marked first,
    # the whole would have stayed a candidate.
    whole_of, part_of, stranger = _box(0, 0, 40, 60), _box(0, 0, 40, 40), _box(100, 20, 120, 50)
    twice = lambda f: (torch.stack([whole_of, part_of, stranger]), [0.7, 0.9, 0.8])   # noqa: E731
    on_part = torch.zeros((8, H, W))
    on_part[:, 0:25, 0:40] = 1.0            # 25 of the part's 40 rows, and 25 of the whole's 60
    one_each = st.one_detection_each(twice, {})
    people = one_each(0)[0]
    if int(people.shape[0]) != 2 or not any(torch.equal(m, part_of) for m in people):
        problems.append(f"others: the pair's own control: the two detections of one person are joined and the smaller stands for them: {int(people.shape[0])}")
    after = st.held_people(one_each, on_part)(0)
    if {int(i) for i in after} != {next(k for k, m in enumerate(people) if torch.equal(m, part_of))}:
        problems.append(f"others: read after the join, the person the smaller detection stands for is held: {after}")
    if int(_removing(one_each, on_part)(0)[0].shape[0]) != 1 or st.held_people(twice, on_part)(0) != frozenset({1}):
        problems.append("others: the old form removed that person too; and before the join only the smaller detection would be held")
    source = (REPO / "subject_track.py").read_text(encoding="utf-8")
    if source.index("detect = one_detection_each(detect, joined, parts=parts, unsure=unsure)") > source.index("held = held_people(detect, others) if others is not None else None"):
        problems.append("others: the node marks held people before it joins duplicates")

    # EVERY CHOOSER SKIPS THE HELD. A twin: a held person whose signature IS the subject's, so each rule would take them
    me, twin, stranger = _box(40, 10, 80, 60), _box(0, 20, 20, 50), _box(100, 20, 120, 50)
    e = lambda *v: torch.tensor(v, dtype=torch.float32) / torch.tensor(v, dtype=torch.float32).norm()   # noqa: E731
    twin_sigs = [(me, e(1.0, 0, 0)), (twin, e(1.0, 0, 0)), (stranger, e(0.6, 0, 0.8))]
    twin_sign = lambda _f, m: next(v for b, v in twin_sigs if torch.equal(b, m))   # noqa: E731
    def twin_detect(f: int):     # shot 1: me. shot 2: the twin and a stranger. shot 3 (16-39): me, then only the twin from frame 20
        people = [me] if f < 8 else ([twin, stranger] if f < 16 else ([me] if f < 20 else [twin]))
        return torch.stack(people), [0.9] * len(people)
    twin_mask = torch.zeros((40, H, W))
    twin_mask[8:16] = twin
    twin_mask[20:40] = twin
    def losing(start, end, seed, mask):          # a track that lets go from frame 20, for long enough that the re-find looks there
        out = mask[None].repeat(end - start, 1, 1)
        out[max(20 - start, 0):] = 0.0
        return out
    # the control: with nobody held the twin IS taken in shot 2, and re-found in shot 3
    free = st.follow(40, [8, 16], st.PICK_LARGEST, 2, 0.8, twin_detect, twin_sign, losing, stride=4, offset=1)
    if st._state(free.shots[1]) != "taken" or not free.shots[2].regained:
        problems.append(f"others: the twin's own control: unheld, they are taken in shot 2 and re-found in shot 3: {st._state(free.shots[1])}, {free.shots[2].regained}")
    twin_held = st.held_people(twin_detect, twin_mask)
    kept = st.follow(40, [8, 16], st.PICK_LARGEST, 2, 0.8, twin_detect, twin_sign, losing, stride=4, offset=1, held=twin_held)
    if st._state(kept.shots[1]) != "absent" or kept.shots[2].regained:
        problems.append(f"others: a held person exactly like the subject is not the best match of a look, and not a re-find: "
                        f"{st._state(kept.shots[1])}, {kept.shots[2].regained}")
    # ...nor the pick a handed gallery points at: the gallery is the twin's own look, and only the twin is on the frames probed
    only_twin = lambda f: (torch.stack([twin]), [0.9])   # noqa: E731
    all_twin = torch.zeros((24, H, W))
    all_twin[:] = twin
    handed = [(e(1.0, 0, 0), e(1.0, 0, 0))]
    sign2 = lambda f, m: (twin_sign(f, m), twin_sign(f, m))   # noqa: E731
    if st.follow(24, [8, 16], st.PICK_LARGEST, None, 0.8, only_twin, sign2, losing, stride=4, offset=1, gallery=handed).pick_frame is None:
        problems.append("others: the handed gallery's own control: unheld, the twin is picked")
    if st.follow(24, [8, 16], st.PICK_LARGEST, None, 0.8, only_twin, sign2, losing, stride=4, offset=1, gallery=handed,
                 held=st.held_people(only_twin, all_twin)).pick_frame is not None:
        problems.append("others: a handed gallery picked a held person")
    # ...and the scores are relative to the pick frame's UNHELD people: a held person there does not move anybody's score
    crowd = lambda f: (torch.stack([me, twin]), [0.9, 0.8]) if f < 8 else twin_detect(f)   # noqa: E731
    on_pick = torch.zeros((24, H, W))
    on_pick[0:8] = twin
    rel_new = st.follow(24, [8, 16], st.PICK_LARGEST, 2, 0.8, crowd, twin_sign, losing, stride=4, offset=1, held=st.held_people(crowd, on_pick))
    rel_old = st.follow(24, [8, 16], st.PICK_LARGEST, 2, 0.8, _removing(crowd, on_pick), twin_sign, losing, stride=4, offset=1)
    if _takes(rel_old, _removing(crowd, on_pick)) != _takes(rel_new, crowd) or rel_new.others != rel_old.others:
        problems.append(f"others: a held person on the pick frame moved a score or the count of people beside the subject: {rel_new.others} against {rel_old.others}")

    # A PERSON'S WORD MAY NAME A HELD PERSON, and the shot says it: in the state, the report and the table
    number = st.shot_table.person_number(twin_detect(9)[0], 0)          # the twin's number on shot 2's tile, in the whole list
    typed = st.follow(40, [8, 16], st.PICK_LARGEST, 2, 0.8, twin_detect, twin_sign, losing, stride=4, offset=1, held=twin_held,
                      corrections={2: number})
    if st._state(typed.shots[1]) != "taken (corrected)" or not typed.shots[1].also_held or typed.shots[0].also_held:
        problems.append(f"others: a correction naming a held person is taken and marked: {st._state(typed.shots[1])}, {typed.shots[1].also_held}")
    said = st.others_report(typed.shots, twin_held.seen, [])
    if "[2] THE PERSON NAMED BY HAND HERE IS ALSO HELD BY ANOTHER TRACKER" not in said:
        problems.append(f"others: the report does not say a hand-named person is also held: {said!r}")
    # A SEED goes round `_free` too, and that is a change against the old form: between two people who overlap, one
    # of them held, a box on the held one NAMES them now; with held people removed it reached only the free one and
    # was refused. A box read off a tile sits wholly on its own detection, so this is the hand-typed case.
    near, far = _box(40, 10, 80, 60), _box(50, 10, 90, 60)
    pair = lambda f: (torch.stack([near, far]), [0.9, 0.8])   # noqa: E731
    on_far = torch.zeros((8, H, W))
    on_far[:] = far
    people_now, named_now, record_now = st.seed_names(pair, 2, (50, 10, 40, 50))
    _, _, record_then = st.seed_names(_removing(pair, on_far), 2, (50, 10, 40, 50))
    if named_now is None or not torch.equal(people_now[named_now], far) or record_now["refused"] or not record_then["refused"]:
        problems.append(f"others: a seed's box on a held person names them (and was refused when held people were removed): "
                        f"now {record_now}, then {record_then}")
    # ...and on the PICKED shot it becomes the reference, a person the other tracker holds everywhere: said in the loudest line
    both = lambda f: (torch.stack([me, twin]), [0.9, 0.8]) if f < 8 else twin_detect(f)   # noqa: E731
    twin_all = torch.zeros((40, H, W))
    twin_all[0:16] = twin
    twin_all[20:40] = twin
    both_held = st.held_people(both, twin_all)
    seeded = st.follow(40, [8, 16], st.PICK_LARGEST, 2, 0.8, both, twin_sign, losing, stride=4, offset=1, held=both_held,
                       seeds=st.parse_seeds("frame 2: 0, 20, 20, 30", [(0, 8), (8, 16), (16, 40)]))
    if seeded.pick_seed != "is the pick" or not seeded.shots[0].also_held or [s.seed is None for s in seeded.shots[1:]] != [True, True]:
        problems.append(f"others: a seed on the picked shot naming a held person is the pick, is marked, and leaves every other shot "
                        f"with nobody to take: {seeded.pick_seed}, {seeded.shots[0].also_held}, {[st._state(x) for x in seeded.shots]}")
    if "AND THIS IS THE PICK" not in st.others_report(seeded.shots, both_held.seen, []) or "AND THIS IS THE PICK" in said:
        problems.append("others: the report says the pick-shot consequence only for a hand-named held person on the PICKED shot")
    typed_table = st.shot_table.build(typed, twin_detect, st.assemble(40, H, W, typed.pieces), state=st._state, phrase="person",
                                      pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=[8, 16], held=twin_held)
    if [r["subject"]["also_held_by_another_tracker"] for r in typed_table["shots"]] != [False, True, False]:
        problems.append("others: the table's take does not carry `also_held_by_another_tracker` on the shot a held person was typed into")
    if typed.on_others != [] or seeded.on_others != [] or typed_table["shots"][1]["frames_with_subject"] != 8:
        problems.append(f"others: a shot where a person's word named a held person (`also_held`) was cut as on others: "
                        f"typed.on_others={typed.on_others}, seeded.on_others={seeded.on_others}, frames_with_subject={typed_table['shots'][1]['frames_with_subject']}")

    # A TRACK THAT MOVES ONTO ANOTHER TRACKED PERSON IS EMPTIED BEFORE REGAIN RUNS, and reported as cut.
    # On frames 6..17 (12 frames = REGAIN_MIN_RUN) the initial track jumps onto `stranger` (whom `other_mask` holds)
    # while `me` is hidden on 6..9 and returns on 10..23; cutting 6..17 turns them into a 12-frame empty run, so
    # `regain` searches, skips the held stranger on frame 6, and re-seeds `me` on frame 10.
    other_mask = torch.zeros((24, H, W))
    other_mask[:] = stranger
    touching_box = _box(90, 20, 110, 50)   # columns 90..110: exactly half (100..110) overlaps stranger (100..120), <= ON_OTHERS
    drift_sigs = [(me, e(1.0, 0, 0)), (stranger, e(0.0, 1.0, 0)), (touching_box, e(1.0, 0, 0))]
    drift_sign = lambda _f, m: (next(v for b, v in drift_sigs if torch.equal(b, m)),) * 2   # noqa: E731
    def drift_detect(f: int):
        people = [stranger] if 6 <= f < 10 else [me, stranger]
        return torch.stack(people), [0.9] * len(people)
    def drift_track(start, end, seed, mask):
        out = mask[None].repeat(end - start, 1, 1)
        if start == 0 and end == 24:
            out[6:18] = stranger           # the initial track jumped onto the held stranger on frames 6..17
        elif start == 6 and end == 18:
            out[:10 - start] = 0.0         # re-seeded on frame 10: frames 6..9 stay empty where `me` was hidden
        return out
    # 1) no `others` (`held=None`): nothing is cut, so frames 6..17 stay non-empty and `regain` never runs
    no_others = st.follow(24, [], st.PICK_LARGEST, 2, 0.8, drift_detect, drift_sign, drift_track, stride=4, offset=1)
    if no_others.on_others != [] or no_others.shots[0].regained != [] or st.empty_runs(no_others.pieces[0]) != []:
        problems.append(f"others: with no `others`, a track was cut or re-seeded: on_others={no_others.on_others}, regained={no_others.shots[0].regained}")
    # 2) with `others`: frames 6..17 are emptied before `regain`, recorded in `on_others`/report/shot_table, and `regain` recovers 10..17
    drift_held = st.held_people(drift_detect, other_mask)
    cut_got = st.follow(24, [], st.PICK_LARGEST, 2, 0.8, drift_detect, drift_sign, drift_track, stride=4, offset=1, held=drift_held)
    cut_mask = st.assemble(24, H, W, cut_got.pieces)
    if cut_got.on_others != list(range(6, 18)):
        problems.append(f"others: frames where the track moved onto another person were not recorded in `on_others`: {cut_got.on_others}")
    if [a for a, *_ in cut_got.shots[0].regained] != [6] or [f for _, f, *_ in cut_got.shots[0].regained] != [10]:
        problems.append(f"others: emptying the frames on others did not trigger `regain` from 6 to re-seed at 10: {cut_got.shots[0].regained}")
    if st.empty_runs(cut_mask) != [(6, 10)] or not all(torch.equal(cut_mask[f], me) for f in range(10, 18)):
        problems.append(f"others: after cutting 6..17 and regaining at 10, empty_runs={st.empty_runs(cut_mask)}")
    cut_report = st.others_report(cut_got.shots, drift_held.seen, cut_got.on_others)
    if "cut 12 frame(s) where this track lay mostly on another tracker's person: 6-17" not in cut_report:
        problems.append(f"others: report did not record the 12 cut frames: {cut_report!r}")
    cut_table = st.shot_table.build(cut_got, drift_detect, cut_mask, state=st._state, phrase="person",
                                    pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=[], held=drift_held)
    if cut_table["shots"][0]["frames_on_others"] != [[6, 17]] or cut_table["shots"][0]["frames_without_subject"] != [[6, 9]]:
        problems.append(f"others: shot table after cutting and regaining has frames_on_others={cut_table['shots'][0]['frames_on_others']}, "
                        f"frames_without_subject={cut_table['shots'][0]['frames_without_subject']}")
    # 3) a track that merely touches another person (`<= ON_OTHERS`, here exactly 0.5) is untouched
    def touch_track(start, end, seed, mask):
        out = mask[None].repeat(end - start, 1, 1)
        out[6:18] = touching_box
        return out
    touch_held = st.held_people(drift_detect, other_mask)
    touch_got = st.follow(24, [], st.PICK_LARGEST, 2, 0.8, drift_detect, drift_sign, touch_track, stride=4, offset=1, held=touch_held)
    touch_mask = st.assemble(24, H, W, touch_got.pieces)
    touch_table = st.shot_table.build(touch_got, drift_detect, touch_mask, state=st._state, phrase="person",
                                      pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=[], held=touch_held)
    if touch_got.on_others != [] or st.empty_runs(touch_mask) != [] or touch_table["shots"][0]["frames_on_others"] != []:
        problems.append(f"others: a track touching another person at `<= ON_OTHERS` was cut: on_others={touch_got.on_others}, "
                        f"empty_runs={st.empty_runs(touch_mask)}")




class _SteppingTracker:
    """Stands in for ComfyUI's tracker object: `track_step` returns the score it is told to, per frame."""
    def __init__(self, objects: int = 1, twice: int | None = None, skip_seed: bool = False):
        self.objects, self.twice, self.skip_seed = objects, twice, skip_seed

    def track_step(self, frame_idx, value=0.0, **_):
        return {"object_score_logits": torch.full((self.objects, 1), float(value))}

    def run(self, values):
        for k, v in enumerate(values):
            if k == 0 and self.skip_seed:
                continue
            self.track_step(frame_idx=k, value=v)
            if k == self.twice:
                self.track_step(frame_idx=k, value=v)


def check_track_score(problems):
    """The tracker's own score per frame (2026-10-10): read in step with the frames, or not at all."""
    tracker = _SteppingTracker()
    with st.ScoreTap(tracker) as tap:
        tracker.run([4.0, 3.5, -1.0])
    if tap.seen != {0: 4.0, 1: 3.5, 2: -1.0} or tap.trouble:
        problems.append(f"score: the tap read {tap.seen} with trouble {tap.trouble!r}; the tracker's own numbers, per frame")
    if "track_step" in vars(tracker) or tracker.track_step.__func__ is not _SteppingTracker.track_step:
        problems.append("score: the tap left its wrapper on the tracker object")
    try:
        with st.ScoreTap(tracker):
            raise KeyError("inside")
    except KeyError:
        pass
    if "track_step" in vars(tracker):
        problems.append("score: an error inside the tap left its wrapper on the tracker object")
    # a wrapper the object already carried, and a tap inside a tap, are put back and not deleted (mrcorn)
    theirs = lambda *a, **k: _SteppingTracker.track_step(tracker, *a, **k)  # noqa: E731
    tracker.track_step = theirs
    with st.ScoreTap(tracker) as outer:
        with st.ScoreTap(tracker) as inner:
            tracker.run([1.0, 2.0])
    if vars(tracker).get("track_step") is not theirs or inner.seen != {0: 1.0, 1: 2.0} or outer.seen != {0: 1.0, 1: 2.0}:
        problems.append("score: a tap deleted a wrapper that was on the object before it, or a tap inside a tap did not read")
    del tracker.track_step
    for made, need in ((_SteppingTracker(twice=1), "more than once"), (_SteppingTracker(objects=2), "object scores on a frame"),
                       (object(), "no `track_step`"), (None, "no `track_step`")):
        with st.ScoreTap(made) as tap:
            if hasattr(made, "run"):
                made.run([1.0, 2.0, 3.0])
        if need not in tap.trouble:
            problems.append(f"score: a tracker that should be refused ({need}) gave trouble {tap.trouble!r}")
        if st.scores_for(tap.seen, tap.trouble, 3)[0] != [None, None, None]:
            problems.append(f"score: a record with trouble ({need}) still gave scores")
    if st.scores_for({0: 1.0, 1: 2.0, 2: 3.0}, "", 3) != ([1.0, 2.0, 3.0], ""):
        problems.append("score: a complete run is not returned as it was read")
    if st.scores_for({1: 2.0, 2: 3.0}, "", 3) != ([None, 2.0, 3.0], ""):
        problems.append("score: the seed frame alone may be missing, and is null")
    if st.scores_for({0: 1.0, 1: 2.0, 2: 3.0}, "", 3, backwards=True) != ([3.0, 2.0, 1.0], ""):
        problems.append("score: a run fed backwards is not turned round into clip order")
    for seen in ({0: 1.0, 2: 3.0}, {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}):
        values, why = st.scores_for(seen, "", 3)
        if values != [None, None, None] or not why:
            problems.append(f"score: a record out of step with its run ({sorted(seen)} for 3 frames) gave {values} and {why!r}")

    # the node's own track call, on stand-ins for ComfyUI's three node classes: frame f of the clip scores f
    class _Out:
        def __init__(self, *args):
            self.args = args

    def _nodes(skip_seed=False, twice=None):
        module = types.ModuleType("comfy_extras.nodes_sam3")

        class SAM3_Detect:
            @staticmethod
            def execute(*_a, **_k):
                return _Out(torch.zeros((0, H, W)), [[]])

        class SAM3_VideoTrack:
            @staticmethod
            def execute(images, model, initial_mask=None, **_k):
                tracker = model.model.diffusion_model.tracker
                for k in range(int(images.shape[0])):
                    if k == 0 and skip_seed:
                        continue
                    tracker.track_step(frame_idx=k, value=float(images[k].mean()) * 100.0)
                    if k == twice:
                        tracker.track_step(frame_idx=k, value=0.0)
                return _Out({"n": int(images.shape[0]), "mask": initial_mask})

        class SAM3_TrackToMask:
            @staticmethod
            def execute(data, _which):
                return _Out(data["mask"].repeat(data["n"], 1, 1))

        module.SAM3_Detect, module.SAM3_VideoTrack, module.SAM3_TrackToMask = SAM3_Detect, SAM3_VideoTrack, SAM3_TrackToMask
        return module

    class _Clip:
        def tokenize(self, text):
            return text

        def encode_from_tokens_scheduled(self, tokens):
            return tokens

    segmenter = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(tracker=_SteppingTracker())))
    clip_frames = torch.stack([torch.full((H, W, 3), f / 100.0) for f in range(16)])
    seed_mask = torch.ones((H, W))
    kept_module = sys.modules.get("comfy_extras.nodes_sam3")
    try:
        for skip_seed, twice, name in ((False, None, "every frame stepped"), (True, None, "the seed frame not stepped"), (False, 2, "a frame stepped twice")):
            sys.modules["comfy_extras.nodes_sam3"] = _nodes(skip_seed, twice)
            scores: dict = {}
            _d, _s, track = st._sam_callables(segmenter, _Clip(), clip_frames, "person", 0.5, track_scores=scores)
            piece = track(4, 12, 8, seed_mask)
            if tuple(piece.shape) != (8, H, W):
                problems.append(f"score ({name}): the track's masks are {tuple(piece.shape)}, not one per frame of 4 to 11")
            read = {f: scores.get(f) for f in range(4, 12)}
            if twice is not None:
                if any(v is not None for v in read.values()) or len(scores.get("trouble", [])) != 2:
                    problems.append(f"score ({name}): scores {read} and trouble {scores.get('trouble')}; a run out of step gives none and says why")
                continue
            want = {f: (None if f == 8 else float(f)) for f in range(4, 12)}    # the seed frame is told, not judged: null
            if any((read[f] is None) != (want[f] is None) or (want[f] is not None and abs(read[f] - want[f]) > 1e-4) for f in want):
                problems.append(f"score ({name}): frame f should score f, forwards and backwards from the seed; got {read}")
            if any(f in scores for f in (3, 12)) or scores.get("trouble"):
                problems.append(f"score ({name}): scores outside the frames tracked, or trouble where there was none: {scores}")
            scores[5] = 99.0
            track(4, 8, 6, seed_mask)                        # a later track over some of the same frames replaces them
            if abs(scores[5] - 5.0) > 1e-4 or abs(scores[9] - 9.0) > 1e-4 or scores[6] is not None or scores[8] is not None:
                problems.append(f"score ({name}): a later track did not replace the scores of the frames it covers, and only those")
    finally:
        if kept_module is None:
            sys.modules.pop("comfy_extras.nodes_sam3", None)
        else:
            sys.modules["comfy_extras.nodes_sam3"] = kept_module
    full = {f: float(f) for f in range(0, 8)}
    full["trouble"] = ["frames 8-15: a frame was stepped more than once"]
    boxes, detect, sign, track2, _ = _world()
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 0.8, detect, sign, track2, stride=4, offset=1)
    table = st.shot_table.build(got, detect, st.assemble(32, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=[8, 16, 24], track_scores=full)
    if table["shots"][0]["track_score"] != [float(f) for f in range(8)] or table["shots"][1]["track_score"] != [None] * 8:
        problems.append(f"score: the shot table's per-frame list is not the frames of the shot in order, null where none: {table['shots'][0]['track_score']}")
    if table["track_score_trouble"] != full["trouble"] or "over 0" not in table["track_score_is"]:
        problems.append("score: the table does not carry why a run gave no score, or what the number is")


def check_schema(problems):
    schema = st.MiniMaxH3SubjectTrack.define_schema()
    inputs = {i.id: i for i in schema.inputs}
    for name, default in (("subject_phrase", st.SUBJECT_PHRASE), ("head_phrase", st.HEAD_PHRASE),
                          ("max_people", st.MAX_PEOPLE), ("detection_threshold", st.DETECTION_THRESHOLD),
                          ("pick", st.PICK_CENTRAL)):     # the default since 2026-10-07 (the owner)
        if name not in inputs:
            problems.append(f"the node has no `{name}` input: what SAM is asked and how it is judged must be visible")
        elif getattr(inputs[name], "default", None) != default:
            problems.append(f"`{name}` defaults to {getattr(inputs[name], 'default', None)!r}, not the module's {default!r}")
    for name in ("frames", "segmenter", "segmenter_clip"):
        if name not in inputs:
            problems.append(f"the node has no `{name}` input")
    # the two choices: automatic first, and the named value under its own option
    for name, other, nested, default in (("pick_on", st.PICK_ON_FRAME, "pick_frame", 0),
                                         ("match", st.AT_VALUE, "match_threshold", st.MATCH_THRESHOLD),
                                         ("cuts", st.AT_VALUE, "cut_threshold", st.CUT_THRESHOLD)):
        combo = inputs.get(name)
        if combo is None:
            problems.append(f"the node has no `{name}` choice")
            continue
        options = {o.key: o for o in combo.options}
        if list(options) != [st.AUTOMATIC, other]:
            problems.append(f"`{name}` offers {list(options)}, not automatic first and then `{other}`")
            continue
        under = {i.id: i for i in options[other].inputs}
        if list(under) != [nested] or getattr(under[nested], "default", None) != default:
            problems.append(f"`{name}` -> `{other}` does not reveal `{nested}` at {default!r}")
        if options[st.AUTOMATIC].inputs:
            problems.append(f"`{name}` -> automatic asks for something")
    for flat in ("pick_frame", "match_threshold", "cut_threshold"):
        if flat in inputs:
            problems.append(f"`{flat}` is also a top-level input: two inputs for one thing")
    if len(schema.outputs) != 5 or schema.outputs[0].io_type != "MASK" or schema.outputs[4].io_type != "MASK":
        problems.append("the node's outputs are not mask, preview, report, shot_table, head_mask with the mask first")
    if getattr(schema, "is_output_node", False):
        problems.append("the node is an output node: core would run the tracker on every queue, kept mask or not")
    source = (REPO / "subject_track.py").read_text(encoding="utf-8")
    if "subject_tracks.drop_specks(assemble(n, h, w, found.pieces), in_place=True)" not in source:
        problems.append("the node's mask is not `drop_specks` of the assembled pieces: a stray speck of the tracker's "
                        "would reach the tiles, the shot table and every node after it")
    if not isinstance(getattr(st.MiniMaxH3SubjectTrack, "MASK_VERSION", None), int):
        problems.append("the node declares no integer MASK_VERSION, so a kept mask would survive a change to how it is made")
    fix, handed, last, seeds, track_head = schema.inputs[-5], schema.inputs[-4], schema.inputs[-3], schema.inputs[-2], schema.inputs[-1]
    if track_head.id != "track_head" or getattr(track_head, "default", None) is not True or not getattr(track_head, "optional", False) or getattr(track_head, "advanced", False):
        problems.append("`track_head` is not the node's last input, a visible optional boolean that is True by default")
    if seeds.id != "seeds" or getattr(seeds, "default", None) != "" or not getattr(seeds, "multiline", False) \
            or not getattr(seeds, "optional", False) or "frame 410: 543, 253, 157, 186" not in (seeds.tooltip or ""):
        problems.append("`seeds` is not the node's last input but one, an optional multi-line text that is empty by default, with the format in its tooltip")
    if fix.id != "corrections" or getattr(fix, "default", None) != "" or not fix.tooltip or not getattr(fix, "multiline", False):
        problems.append("`corrections` is not the node's fifth input from the end, a multi-line text that is empty by default, with a tooltip")
    # each appended after the one before, so a graph saved before it keeps every widget where it was
    if handed.id != "subject_from" or getattr(handed, "default", None) != "" or not handed.tooltip or not getattr(handed, "optional", False):
        problems.append("`subject_from` is not the node's fourth input from the end, an optional text that is empty by default, with a tooltip")
    if last.id != "others" or last.io_type != "MASK" or not last.tooltip or not getattr(last, "optional", False):
        problems.append("`others` is not the node's third input from the end, an optional mask with a tooltip")
    import inspect
    if inspect.signature(st.MiniMaxH3SubjectTrack.execute).parameters["track_head"].default is not True:
        problems.append("`track_head` must default to True in execute")
    if 'head track (track_head=True): {head_frames} of {n} frame(s)' not in source or 'table["track_head"] = bool(track_head)' not in source:
        problems.append("`execute` does not report `track_head` in both the text report and `shot_table`")
    if inspect.signature(st.MiniMaxH3SubjectTrack.execute).parameters["seeds"].default != "" \
            or "seeded = parse_seeds(seeds, shot_ranges(n, found_cuts))" not in source or "seeds=seeded," not in source:
        problems.append("`seeds` left empty must be the empty text in execute, parsed against the load's shots and handed to `follow`")
    if inspect.signature(st.MiniMaxH3SubjectTrack.execute).parameters["others"].default is not None:
        problems.append("`others` left unwired must be None in execute: nobody is held by another tracker")
    for need in ("held = held_people(detect, others) if others is not None else None", "track_scores=track_scores)",
                 "on_others = frames_on_others(mask, others)"):
        if need not in source:
            problems.append(f"the node no longer does `{need}`: the `others` input or the tracker's score is not wired")
    sel = st._selection({"match": st.AT_VALUE, "match_threshold": 0.5}, "match", "match_threshold")
    if sel != (st.AT_VALUE, 0.5) or st._selection(st.AUTOMATIC, "match", "match_threshold") != (st.AUTOMATIC, None):
        problems.append("a DynamicCombo's nested dict, or a bare selection, is not read as the choice and its value")


def check_track_head(problems: list[str]) -> None:
    """When `track_head` is False, `head_pieces` is None and the person track is byte-identical; when True (the node's
    default), each shot's head mask is propagated from `sign.head` on its seed frame and clipped to the person track."""
    p_mask = _box(4, 4, 16, 16)
    h_seed = _box(6, 4, 14, 8)
    # Head tracker drifts outside the person mask (cols 14..18) so clipping to the person track is exercised.
    h_drifted = _box(6, 4, 18, 8)

    def detect(f: int):
        return p_mask.unsqueeze(0), [0.95]

    def sign(f: int, m: torch.Tensor):
        v = torch.tensor([1.0, 0.0, 0.0, 0.0])
        return v, v

    sign.head = lambda f, m: h_seed

    def track(start: int, end: int, seed: int, m: torch.Tensor):
        is_head = bool((m[8:16, :] > 0.5).sum() == 0)
        piece = h_drifted if is_head else p_mask
        return piece.unsqueeze(0).expand(end - start, -1, -1).clone()

    off = st.follow(8, [4], st.PICK_LARGEST, 0, 0.5, detect, sign, track, track_head=False)
    on = st.follow(8, [4], st.PICK_LARGEST, 0, 0.5, detect, sign, track, track_head=True)
    if off.head_pieces:
        problems.append("follow with track_head=False produced head_pieces")
    if not torch.equal(st.assemble(8, H, W, off.pieces), st.assemble(8, H, W, on.pieces)):
        problems.append("track_head=True changed the person track")
    if len(on.head_pieces) != 2:
        problems.append(f"track_head=True produced {len(on.head_pieces)} head pieces for 2 shots")
    else:
        assembled_head = st.assemble(8, H, W, on.head_pieces)
        assembled_person = st.assemble(8, H, W, on.pieces)
        if bool(((assembled_head > 0.5) & ~(assembled_person > 0.5)).any()):
            problems.append("head_mask was not clipped to the person track")
        expected_clipped = ((h_drifted > 0.5) & (p_mask > 0.5)).to(torch.float32)
        if not torch.equal(assembled_head[0], expected_clipped) or not torch.equal(assembled_head[5], expected_clipped):
            problems.append("head_mask did not match the propagated head clipped to the person track")
    # A correction to "none" drops that shot's head_pieces as well as its person piece.
    cleared = st.follow(8, [4], st.PICK_LARGEST, 0, 0.5, detect, sign, track, corrections={2: None}, track_head=True)
    if 4 in cleared.head_pieces or 0 not in cleared.head_pieces:
        problems.append(f"correction to none did not clear shot 2's head_pieces: {sorted(cleared.head_pieces)}")


def _boxes_world(n: int = 30, strength: dict | None = None):
    """One shot of `n` frames: the subject (person 0, box A) and another (person 1, box B) on every frame.

    The tracker carries its seed mask over the whole call, TAGGED with the seed frame (one pixel in row 0 at that
    column) so a frame shows which seed's track was kept, and gives the tracker's own score `strength[seed]` less a
    tenth for each frame away from the seed (None on the seed frame, as the node writes it) through `track.scored`.
    Heads are the top ten rows of a mask."""
    subject, other = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    sigs = [(subject, torch.tensor([1.0, 0.0, 0.0])), (other, torch.tensor([0.0, 1.0, 0.0]))]
    strength = strength or {}
    calls = {"track": [], "scored": [], "kept": [], "head": []}

    def detect(_f: int):
        return torch.stack([other, subject], dim=0), [0.9, 0.9]

    def sign(_frame: int, mask: torch.Tensor):
        found = next((v for m, v in sigs if torch.equal(m, mask)), torch.tensor([0.5, 0.5, 0.0]))
        return found, found

    def head(f: int, mask: torch.Tensor):
        calls["head"].append(int(f))
        on = (mask > 0.5).nonzero()
        if not len(on):
            return None
        out = mask.clone()
        out[int(on[:, 0].min()) + 10:] = 0.0
        return out

    sign.head = head

    def carried(start: int, end: int, seed: int, mask: torch.Tensor) -> torch.Tensor:
        out = mask[None].repeat(end - start, 1, 1).clone()
        out[:, 0, seed % W] = 1.0
        return out

    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls["track"].append((start, end, seed))
        return carried(start, end, seed, mask)

    def scored(start: int, end: int, seed: int, mask: torch.Tensor):
        calls["scored"].append((start, end, seed, int((mask > 0.5).sum())))
        base = float(strength.get(seed, 5.0))
        return carried(start, end, seed, mask), [None if f == seed else base - 0.1 * abs(f - seed) for f in range(start, end)], []

    track.scored = scored
    track.keep_scores = lambda first, values, trouble: calls["kept"].append((first, list(values), list(trouble)))
    return subject, other, detect, sign, track, calls


def _tag(mask_frame: torch.Tensor) -> list[int]:
    """The seed frames a tracked frame is tagged with (`_boxes_world`)."""
    return (mask_frame[0] > 0.5).nonzero().flatten().tolist()


def check_every_box(problems):
    """Every owner box in a shot seeds, and the tracks are kept frame by frame by the tracker's own score (2026-10-23)."""
    subject, other, detect, sign, track, calls = _boxes_world(30, {5: 3.0, 20: 9.0})
    a = ", ".join(str(v) for v in st.shot_table._box(subject))
    shots = st.shot_ranges(30, [])
    # three boxes: 5 and 20 agree with the subject; 12 lies on nobody and there is nothing to segment with
    seeds = st.parse_seeds(f"frame 20: {a}\nframe 12: 100, 60, 20, 10\nframe 5: {a}", shots)
    if [f for f, _ in seeds[1]] != [5, 12, 20]:
        problems.append(f"every box: the seeds of one shot are not read in frame order: {seeds}")
    got = st.follow(30, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1, seeds=seeds, track_head=True, head_per_stretch=True)
    shot = got.shots[0]
    # each taken box tracked out to its NEIGHBOURS' frames, never past them, from its own frame
    person_px = int((subject > 0.5).sum())
    person_calls = [c[:3] for c in calls["scored"] if c[3] == person_px]
    if person_calls != [(0, 21, 5), (5, 30, 20)]:
        problems.append(f"every box: each taken box should be tracked from its frame out to its neighbours' frames: {calls['scored']}")
    # kept by SCORE: seed 20's track scores higher wherever both hold her, so it is kept on 6-19, even on 6-12, nearer 5;
    # a split at the midpoint would have kept seed 5's there. Seed 5 keeps its own frame and the frames only it covers
    piece = got.pieces[0]
    tags = [_tag(piece[f]) for f in range(30)]
    if tags != [[5]] * 6 + [[20]] * 24:
        problems.append(f"every box: the merged track should keep seed 5 on 0-5 and seed 20 on 6-29 by the tracker's score: {tags}")
    recs = shot.seeded_all or []
    if [r["frame"] for r in recs] != [5, 12, 20] or [r["refused"] for r in recs] != [False, True, False] \
            or [r.get("kept_on") for r in recs] != [[[0, 5]], [], [[6, 29]]] \
            or [(r.get("first_frame"), r.get("last_frame")) for r in recs] != [(0, 20), (None, None), (5, 29)] \
            or recs[1]["how"] != st.SEED_REFUSED or "segmentation" not in recs[1]["why"] or shot.seeded is not recs[0]:
        problems.append(f"every box: each box's record should say what it took, the frames it ran over and was kept on, and why one was refused: {recs}")
    if st._state(shot) != "taken (seeded)" or shot.seed != 5 or shot.shown != 5:
        problems.append(f"every box: the shot is taken and shown on the first box taken: {st._state(shot)}, {shot.seed}, {shot.shown}")
    # the merged scores are what the record keeps, once, over the whole shot: the kept track's on each frame
    if len(calls["kept"]) != 1 or calls["kept"][0][0] != 0 or len(calls["kept"][0][1]) != 30 \
            or calls["kept"][0][1][5] is not None or calls["kept"][0][1][20] is not None or abs(calls["kept"][0][1][8] - 7.8) > 1e-6:
        problems.append(f"every box: the record should keep the merged scores once (null on each seed's frame): {calls['kept']}")
    # the head is tracked from each box's head inside the one stretch she is held on, and kept the same way
    head_calls = [c[:3] for c in calls["scored"] if c[3] != person_px]
    heads = st.assemble(30, H, W, got.head_pieces)
    if head_calls != [(0, 21, 5), (5, 30, 20)] or any(not bool((heads[f] > 0.5).any()) for f in range(30)) \
            or bool(((heads > 0.5) & ~(st.assemble(30, H, W, got.pieces) > 0.5)).any()):
        problems.append(f"every box: the head should be tracked from each box's head and clipped to her track: {head_calls}")
    text = st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0)
    for need in ("seeded from 3 boxes", "SEED REFUSED on load frame 12", "kept on 0-5", "kept on 6-29", "tracked over 5-29"):
        if need not in text:
            problems.append(f"every box: the report lacks {need!r}: {text!r}")
    if not st.tile_label(1, shot).endswith("2 of 3 boxes  taken (seeded)"):
        problems.append(f"every box: the tile should say how many boxes were taken: {st.tile_label(1, shot)!r}")
    table = st.shot_table.build(got, detect, st.assemble(30, H, W, got.pieces), state=st._state, phrase="person", pick=st.PICK_LARGEST,
                                named_frame=True, named_value=True, cuts=[])
    row = table["shots"][0]
    if [s["frame"] for s in row.get("seeds") or []] != [5, 12, 20] or not row["subject"]["why"].startswith("seeded from 3 boxes") \
            or "kept on 6-29" not in row["subject"]["why"] or "seed refused on frame 12" not in row["subject"]["why"]:
        problems.append(f"every box: the table's row should list every seed and say which were taken and kept, and which refused: {row['subject']['why']}")
    # the merge on its own, two tracks of ONE person (their masks overlap 0.78): no score read, the nearer seed; a tie,
    # the earlier; a read score outranks none; a seed's own frame, its own
    one, two = _box(0, 0, 8, 8), _box(1, 0, 9, 8)
    runs = [(2, 0, 9, one[None].repeat(9, 1, 1), [None] * 9), (6, 0, 9, two[None].repeat(9, 1, 1), [None] * 9)]
    _, _, kept, disagreed = st.merge_tracks(0, 9, runs)
    if kept != [2, 2, 2, 2, 2, 6, 6, 6, 6] or disagreed:
        problems.append(f"every box: with no score the nearer seed is kept, a tie the earlier: {kept}, {disagreed}")
    runs[0] = (2, 0, 9, one[None].repeat(9, 1, 1), [9.0] * 9)
    _, scores, kept, _ = st.merge_tracks(0, 9, runs)
    if kept != [2, 2, 2, 2, 2, 2, 6, 2, 2] or scores[6] is not None:
        problems.append(f"every box: a read score outranks none, and a seed keeps its own frame: {kept}")
    empty = torch.zeros((9, H, W))
    _, scores, kept, _ = st.merge_tracks(0, 9, [(2, 0, 9, empty, [-1.0] * 9), (6, 0, 9, empty, [-3.0] * 9)])
    if kept[0] is not None or scores[0] != -1.0 or kept[2] != 2:
        problems.append(f"every box: a frame no track holds anybody on stays empty with the best score there: {kept}, {scores}")
    # two tracks holding two DIFFERENT people, their scores alternating frame to frame (the blind read's world): never a
    # flicker between them; every frame both hold is left empty and listed, each seed keeping its own frame
    a_m, b_m = _box(10, 10, 30, 60), _box(80, 10, 100, 60)
    flick = [(0, 0, 10, a_m[None].repeat(10, 1, 1), [None] + [0.9 if k % 2 else 0.8 for k in range(1, 10)]),
             (9, 0, 10, b_m[None].repeat(10, 1, 1), [0.8 if k % 2 else 0.9 for k in range(0, 9)] + [None])]
    out, _, kept, disagreed = st.merge_tracks(0, 10, flick)
    if kept != [0] + [None] * 8 + [9] or disagreed != list(range(1, 9)) or bool((out[1:9] > 0.5).any()):
        problems.append(f"every box: two tracks of two people should never be merged, the frames they both hold left empty: {kept}, {disagreed}")
    # the continuity test: one person, but the higher-scoring track's mask has jumped (overlapping the mask kept a frame
    # before under MERGE_CONTINUOUS): the kept seed stays; where the jump is small it switches by score
    big, small = _box(0, 0, 40, 40), _box(0, 0, 12, 12)            # 144 px on 1,600: one person, IoU 0.09
    jump = [(0, 0, 4, big[None].repeat(4, 1, 1), [None, 1.0, 1.0, 1.0]), (3, 0, 4, small[None].repeat(4, 1, 1), [5.0, 5.0, 5.0, None])]
    _, _, kept, _ = st.merge_tracks(0, 4, jump)
    near = [(0, 0, 4, big[None].repeat(4, 1, 1), [None, 1.0, 1.0, 1.0]), (3, 0, 4, _box(0, 0, 40, 38)[None].repeat(4, 1, 1), [5.0, 5.0, 5.0, None])]
    _, _, kept_near, _ = st.merge_tracks(0, 4, near)
    if kept != [0, 0, 0, 3] or kept_near != [0, 3, 3, 3] or abs(st.MERGE_CONTINUOUS - 0.5) > 1e-9:
        problems.append(f"every box: a switch to a mask that jumped should be refused and a near one made: {kept}, {kept_near}")


def check_box_tiers(problems):
    """A box drawn by hand: a detection's mask overlapping it, else what SAM 3 segments inside it, never the rectangle."""
    a, b = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    beside = _box(80, 10, 120, 60)              # another person, right beside the subject
    part = _box(50, 10, 70, 25)                 # a part of the subject the detector also returned
    world = {"people": [b, a]}
    detect = lambda f: (torch.stack(world["people"], dim=0), [0.9] * len(world["people"]))  # noqa: E731
    shifted = (43.0, 10.0, 40.0, 50.0)          # three columns off: the boxes agree 0.86, under SEED_AGREES
    got = st.name_seed(detect, 4, shifted)
    if got.record["how"] != st.SEED_BY_OVERLAP or got.index != 1 or abs(got.record["mask_overlap"] - 0.86) > 0.01 or got.mask is None:
        problems.append(f"tiers: a box three columns off should take the subject by their mask's overlap with it: {got.record}")
    # a person another tracker holds is never taken by the overlap (a box that agrees may still name them, as before)
    held = lambda f: frozenset({1})              # noqa: E731
    got = st.name_seed(detect, 4, shifted, held)
    if got.mask is not None or not got.record["refused"]:
        problems.append(f"tiers: a held person was taken by the mask overlap: {got.record}")
    if st.name_seed(detect, 4, (40.0, 10.0, 40.0, 50.0), held).record["how"] != st.SEED_BY_BOX:
        problems.append("tiers: a box that agrees with a held person no longer names them")
    # a box over two people side by side: both overlap it, neither is a part of the other, so it is refused
    world["people"] = [a, beside]
    got = st.name_seed(detect, 4, (40.0, 10.0, 80.0, 40.0))
    if got.mask is not None or "more than one person" not in got.record["why"]:
        problems.append(f"tiers: a box holding two people should be refused as such: {got.record}")
    # a part of the subject that also overlaps the box is not a second person: the subject is taken
    world["people"] = [a, part]
    got = st.name_seed(detect, 4, (40.0, 10.0, 40.0, 25.0))
    if got.mask is None or got.index != 0 or got.record["how"] != st.SEED_BY_OVERLAP:
        problems.append(f"tiers: a part inside the subject made the box ambiguous: {got.record}")
    # no detection's mask overlaps the box enough: what SAM 3 segments there, CLIPPED TO THE BOX (a tile's box is
    # exact), taken only as one person a detection found too (here a close-up, larger than the box), LESS another
    # tracker's people
    big = _box(10, 0, 128, 72)                   # her, close: her mask overlaps the 2,000 px box only 0.24
    world["people"] = [b, big]
    blob = _box(30, 5, 75, 55)                   # runs out of the box on the left and the top
    others = torch.zeros((8, H, W))
    others[4] = _box(70, 30, 90, 60)             # another tracker's person over its right edge
    held = st.held_people(detect, others)
    seen = []

    def in_box(f, box):
        seen.append((int(f), tuple(box)))
        return blob
    box = (40.0, 10.0, 40.0, 50.0)
    got = st.name_seed(detect, 4, box, held, in_box)
    want = (_box(40, 10, 75, 55) > 0.5) & ~(others[4] > 0.5)
    if got.record["how"] != st.SEED_BY_SEGMENT or got.index is not None or got.mask is None or got.record["doubtful"] \
            or not torch.equal(got.mask > 0.5, want) or got.record["px"] != int(want.sum()) or seen != [(4, box)]:
        problems.append(f"tiers: the segmentation should be clipped to the box and less others' people: {got.record}, {seen}")
    if got.mask is not None and torch.equal(got.mask > 0.5, st.box_region(box, H, W)):
        problems.append("tiers: the seed is the raw rectangle")
    # refused: nothing segmented, or what was segmented lies on another tracked subject's track over PROMPT_ON_OTHERS
    for made, need in ((lambda f, bx: None, "nothing inside the box"), (lambda f, bx: others[4], "another tracked subject's track"),
                       (lambda f, bx: _box(65, 25, 90, 60), "another tracked subject's track")):
        got = st.name_seed(detect, 4, box, held, made)
        if got.mask is not None or need not in got.record["why"]:
            problems.append(f"tiers: a segmentation that should be refused ({need}) gave {got.record}")
    # the blind read's worlds (2026-10-23), each refused and recorded as DOUBTFUL, none taken as "segmented":
    # A. a box over the lead's dress, the lead held by an earlier tracker whose track covers only her top
    lead = _box(20, 0, 70, 72)
    world["people"] = [lead]
    top = torch.zeros((8, H, W))
    top[:, 0:50, 20:70] = 1.0
    got = st.name_seed(detect, 4, (50.0, 40.0, 30.0, 32.0), st.held_people(detect, top), lambda f, bx: _box(50, 45, 70, 72))
    if got.mask is not None or not got.record["doubtful"] or "another tracked subject" not in got.record["why"]:
        problems.append(f"tiers (A): a box prompt on the lead's dress, the lead held, should be refused as doubtful: {got.record}")
    # C. a loose box over two people each overlapping it under SEED_OVERLAPS, SAM segmenting both as one blob
    p1, p2 = _box(10, 10, 30, 60), _box(40, 10, 60, 60)
    world["people"] = [p1, p2]
    got = st.name_seed(detect, 4, (5.0, 5.0, 60.0, 60.0), None, lambda f, bx: ((p1 + p2) > 0).float())
    if got.mask is not None or not got.record["doubtful"] or "more than one person" not in got.record["why"]:
        problems.append(f"tiers (C): a box prompt holding two people as one blob should be refused as doubtful: {got.record}")
    # D. nobody detected in the box (one person far off), and SAM segments a chair from it
    for people in ([], [_box(0, 0, 10, 10)]):
        world["people"] = people
        got = st.name_seed(lambda f: (torch.stack(world["people"]) if world["people"] else torch.zeros((0, H, W)), [0.9] * len(world["people"])),
                           4, (85.0, 25.0, 30.0, 40.0), None, lambda f, bx: _box(90, 30, 110, 60))
        if got.mask is not None or not got.record["doubtful"] or "matches no person detection" not in got.record["why"]:
            problems.append(f"tiers (D): a box prompt that matches no person detection should be refused as doubtful: {got.record}")
    # B. the FIRST tracker of a job, given no `others`: the lead's own box on that frame, carried as another subject's
    #    (`other`), names her, so the box prompt on her dress is refused there too; without it, it is taken (the limit)
    world["people"] = [lead]
    lead_box = ", ".join(str(v) for v in st.shot_table._box(lead))
    track_b = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731

    def sign_b(_f, _m):
        return torch.tensor([1.0, 0.0]), None
    sign_b.in_box = lambda f, bx: _box(50, 45, 70, 72)
    for text, refused in ((f"frame 4: 50, 40, 30, 32\nframe 4: {lead_box} other", True), ("frame 4: 50, 40, 30, 32", False),
                          (f"frame 4: 50, 40, 30, 32\nframe 4: 15, 0, 60, 72 drawn other", True)):
        got = st.follow(8, [], st.PICK_LARGEST, 0, 0.5, detect, sign_b, track_b, seeds=st.parse_seeds(text, [(0, 8)]),
                        other_boxes=st.parse_other_boxes(text, [(0, 8)]), track_head=False)
        if (got.shots[0].seed is None) != refused or (refused and "another tracked subject" not in got.shots[0].seeded["why"]):
            problems.append(f"tiers (B): with the lead's box carried as another subject's the box prompt on her dress should be refused: "
                            f"{text!r} -> {got.shots[0].seeded}")
    if st.parse_seeds(f"frame 4: {lead_box} other", [(0, 8)]) != {} \
            or st.parse_other_boxes(f"frame 4: {lead_box} drawn other", [(0, 8)]) != {4: [((20.0, 0.0, 50.0, 72.0), True)]}:
        problems.append("tiers: an `other` line is a seed, or is not read as another subject's box")
    world["people"] = [b, big]
    # through `follow`: the shot is taken from the segmented seed, no detection is called the subject, and it is said
    calls = []

    def track(start, end, seed, mask):
        calls.append((start, end, seed, int((mask > 0.5).sum())))
        return mask[None].repeat(end - start, 1, 1)

    def sign(_f, _m):
        return torch.tensor([1.0, 0.0]), None
    sign.in_box = in_box
    got = st.follow(8, [], st.PICK_LARGEST, 0, 0.5, detect, sign, track, seeds=st.parse_seeds("frame 4: 40, 10, 40, 50", [(0, 8)]),
                    track_head=False)
    shot = got.shots[0]
    if st._state(shot) != "taken (seeded)" or shot.index is not None or calls != [(0, 8, 4, int((_box(40, 10, 75, 55) > 0.5).sum()))] \
            or got.pick_seed != "names no detection" or not st.tile_label(1, shot).endswith("segmented in its box  taken (seeded)"):
        problems.append(f"tiers: a seed segmented inside its box should seed the shot with no detection named: {calls}, {got.pick_seed}, "
                        f"{st.tile_label(1, shot)!r}")
    text = st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0)
    for need in ("SAM 3 segmented inside it", "THE PICKED SHOT'S SEEDS NAME NO DETECTION"):
        if need not in text:
            problems.append(f"tiers: the report lacks {need!r}: {text!r}")


def check_head_runs(problems):
    """The head is tracked within each stretch the person track holds her, seeded inside it (2026-10-23, 1419d's shape).

    One shot of 47 frames; the person track holds her on 9-17 and 35-46 and nowhere between (on 1419d SAM held her
    on 1428-1436 and again from 1454). On 9-17 only her head peeks out: her person mask there is small and no head is
    found in it; on 35-46 a head is found. The head tracker cannot carry a head across frames the person is not on."""
    held_on = set(range(9, 18)) | set(range(35, 47))
    big = _box(40, 10, 80, 60)                  # her person mask on 35-46; her head there is its top ten rows, 400 px

    def run(peek: torch.Tensor, per_stretch: bool = True):
        calls = []

        def detect(f: int):
            if 18 <= f < 35:                     # hidden: nobody to find there again
                return torch.zeros((0, H, W)), []
            return (peek if f < 30 else big)[None], [0.9]

        def sign(_f: int, mask: torch.Tensor):
            return torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])

        def head(f: int, mask: torch.Tensor):
            if f < 30:
                return None
            out = mask.clone()
            on = (mask > 0.5).nonzero()
            out[int(on[:, 0].min()) + 10:] = 0.0
            return out
        sign.head = head

        def track(start: int, end: int, seed: int, mask: torch.Tensor):
            is_person = torch.equal(mask, peek) or torch.equal(mask, big)
            calls.append((start, end, seed, "person" if is_person else "head"))
            out = torch.zeros((end - start, H, W))
            if is_person:
                for f in range(start, end):
                    if f in held_on:
                        out[f - start] = peek if f < 30 else big
                return out
            f = seed                      # a head is carried only through frames the person is held on
            while f < end and f in held_on:
                out[f - start] = mask
                f += 1
            f = seed - 1
            while f >= start and f in held_on:
                out[f - start] = mask
                f -= 1
            return out
        a = ", ".join(str(v) for v in st.shot_table._box(peek))
        got = st.follow(47, [], st.PICK_LARGEST, 12, 0.5, detect, sign, track, seeds=st.parse_seeds(f"frame 12: {a}", [(0, 47)]),
                        track_head=True, head_per_stretch=per_stretch)
        return got, calls

    small = _box(50, 10, 70, 30)                # 400 px: as large as her head on 35-46, under HEAD_SIZED times it
    # THE DEFAULT (the owner, 2026-10-23): main's span head track, exactly as main, 1419d's fault and all: seeded on 35, the
    # nearest headed gallery frame, tracked over the whole shot, nothing on 9-17; per stretch only on `head: per stretch`
    got, calls = run(small, per_stretch=False)
    heads = st.assemble(47, H, W, got.head_pieces)
    if got.head_mode != "span" or [c[:3] for c in calls] != [(0, 47, 12), (0, 47, 35)] \
            or [f for f in range(47) if bool((heads[f] > 0.5).any())] != list(range(35, 47)):
        problems.append(f"head runs: the default head track should be main's span head track exactly: {calls}")
    if st.parse_head_mode("frame 3: 1, 2, 3, 4\nhead: per stretch: the owner, to test 1419") != ("per stretch", "the owner, to test 1419") \
            or st.parse_head_mode("frame 3: 1, 2, 3, 4") != ("span", "") \
            or st.parse_seeds("frame 3: 1, 2, 3, 4\nhead: per stretch: why", [(0, 8)]) != {1: [(3, (1.0, 2.0, 3.0, 4.0))]}:
        problems.append("head runs: the `head: per stretch: WHY` line is not read, or is read as a seed")
    source = MODULE.read_text(encoding="utf-8")
    if 'table["head_mode"] = head_mode' not in source or "head_per_stretch=head_mode == HEAD_PER_STRETCH" not in source:
        problems.append("head runs: the node does not read the head mode from its seeds, or does not record it in the table")
    got, calls = run(small)
    heads = st.assemble(47, H, W, got.head_pieces)
    on = [f for f in range(47) if bool((heads[f] > 0.5).any())]
    if on != sorted(held_on):
        problems.append(f"head runs: the head should be held on both stretches, 9-17 and 35-46: {on}")
    # the person track once over the shot from the box; then each stretch's head, from inside it
    if [c[:3] for c in calls] != [(0, 47, 12), (9, 18, 12), (35, 47, 35)]:
        problems.append(f"head runs: each stretch's head should be seeded and tracked inside it, never across 18-34: {calls}")
    if not torch.equal(heads[12] > 0.5, small > 0.5):
        problems.append("head runs: where no head is found in the stretch and her person mask is head-sized, it is the head's seed")
    # the control: a person mask more than HEAD_SIZED times her head is not taken for a head, and nor is its top third,
    # since the shot has a head elsewhere (35-46): the top third stands in only where no head is found on any frame
    wide = _box(40, 10, 80, 50)                 # 1,600 px against a 400 px head: four times
    got, calls = run(wide)
    heads = st.assemble(47, H, W, got.head_pieces)
    if any(bool((heads[f] > 0.5).any()) for f in range(9, 18)) or any(got.head_from.get(0, {}).get(f) for f in range(9, 18)) \
            or not bool((heads[40] > 0.5).any()):
        problems.append("head runs: a person mask larger than HEAD_SIZED times her head, in a shot with a head elsewhere, made a head")
    if abs(st.HEAD_SIZED - 2.0) > 1e-9:
        problems.append("head runs: HEAD_SIZED moved; read its provenance before changing this check")


def check_regain_at_a_box(problems):
    """An empty run holding an owner's box is searched whatever its length, on the box's frame first (2026-10-23)."""
    visible = lambda f: not 8 <= f < 11                              # noqa: E731  hidden 8-10
    for with_box in (True, False):
        subject, detect, sign, track, calls = _lost_world(visible, n=30)
        x, y, w, h = st.shot_table._box(subject)
        text = (f"frame 4: {x + 3 * w // 4}, {y}, {w}, {h}\n" if with_box else "") + f"frame 15: {x}, {y}, {w}, {h}"
        got = st.follow(30, [], st.PICK_LARGEST, 15, 0.8, detect, sign, track, stride=4, offset=1,
                        seeds=st.parse_seeds(text, [(0, 30)]), track_head=False)
        shot = got.shots[0]
        mask = st.assemble(30, H, W, got.pieces)
        held = [f for f in range(30) if bool((mask[f] > 0.5).any())]
        if with_box:
            # the box on 4 is refused (three quarters off her, nothing to segment with); the run 0-10, eleven frames, is
            # under REGAIN_MIN_RUN and holds it: probed on 4 first, she is there, and the track is seeded again from 4
            if not shot.regained or shot.regained[0][:2] != (0, 4) or shot.probes[0][0] != 4 or held != list(range(0, 8)) + list(range(11, 30)):
                problems.append(f"regain at a box: the run holding the refused box should be probed on its frame and her track seeded "
                                f"again there: regained {shot.regained}, probes {shot.probes}, held {held}")
        elif shot.regained or shot.probes or held != list(range(11, 30)):
            problems.append(f"regain at a box: the control, the same run with no box, is under REGAIN_MIN_RUN and is not searched: "
                            f"{shot.regained}, {shot.probes}, {held}")


def check_one_box_calls(problems):
    """A shot of one box whose track has no empty frame makes the calls it made before 2026-10-23, in the same order.

    The calls pinned here are what main's `subject_track.py` (9c130c2a) made on these three worlds, read by running it
    beside this one (the build's differential of sixteen worlds, every call compared; the two that differ are a
    refusal's words and a head tracked per stretch across a gap)."""
    nowhere = [("track", 0, 48, 30, 0)] + [("head", f) for f in (30, 27, 34, 20, 40, 13, 47, 7, 0)] + [("track", 0, 48, 30, "head")]
    for head_missing, want in (((), [("track", 0, 48, 30, 0), ("head", 30), ("track", 0, 48, 30, "head")]),
                               ((30,), [("track", 0, 48, 30, 0), ("head", 30), ("head", 27), ("track", 0, 48, 27, "head")]),
                               # no head on any frame: the top third, after the same nine looks, and no look again for the head-sized step
                               (tuple(range(48)), nowhere)):
        subject = _box(40, 10, 80, 60)
        other = _box(0, 20, 20, 50)
        log = []

        def detect(f: int):
            return torch.stack([other, subject], dim=0), [0.9, 0.8]

        def sign(_f: int, mask: torch.Tensor):
            v = torch.tensor([1.0, 0.0, 0.0]) if torch.equal(mask, subject) else torch.tensor([0.0, 1.0, 0.0])
            return v, v

        def head(f: int, mask: torch.Tensor, missing=head_missing):
            log.append(("head", int(f)))
            if f in missing:
                return None
            out = mask.clone()
            out[16:] = 0.0
            return out
        sign.head = head

        def track(start: int, end: int, seed: int, mask: torch.Tensor):
            log.append(("track", start, end, seed, 0 if torch.equal(mask, subject) else "head"))
            return mask[None].repeat(end - start, 1, 1)
        x, y, w, h = st.shot_table._box(subject)
        # main's calls in the default head track, and with `head: per stretch` too: one box with no gap is main either way
        for per_stretch in (False, True):
            log.clear()
            st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect, sign, track, stride=4, offset=1,
                      seeds=st.parse_seeds(f"frame 30: {x}, {y}, {w}, {h}", [(0, 48)]), track_head=True, head_per_stretch=per_stretch)
            if log != want:
                problems.append(f"one box: the calls of a one-box shot with no gap moved (head missing on {head_missing}, "
                                f"{'per stretch' if per_stretch else 'default'}): {log}, was {want}")


def check_body_box(problems):
    """A box round her body, not her head (`body`), seeds her person track and never her head track (2026-10-23, the lead's
    reading of the owner's sheets: on 4647 and 4912 boxes sit on her arm at the table where her head is hidden)."""
    arm, face = _box(40, 30, 60, 60), _box(80, 5, 100, 25)
    shots = [(0, 20)]
    a = ", ".join(str(v) for v in st.shot_table._box(arm))
    f_ = ", ".join(str(v) for v in st.shot_table._box(face))
    if st.parse_seeds(f"frame 8: {a} body", shots) != {1: [(8, (40.0, 30.0, 20.0, 30.0), "body")]} \
            or st.parse_seeds(f"frame 8: {a}", shots) != {1: [(8, (40.0, 30.0, 20.0, 30.0))]}:
        problems.append("body box: a seed line ending in `body` is not read as a body box, or a plain one is")
    try:
        st.parse_seeds(f"frame 8: {a} arm", shots)
        problems.append("body box: a seed line with another word after its box was read")
    except ValueError as exc:
        if "`body`" not in str(exc):
            problems.append(f"body box: the refusal of another word does not name `body`: {exc}")

    def run(text: str, heads_from: int | None, between: torch.Tensor | None = None, faces_from: int | None = None):
        calls = []

        def detect(f: int):
            if between is not None and 5 <= f < 15:
                return torch.stack([between, arm, face]), [0.9, 0.8, 0.8]
            late = heads_from if heads_from is not None else faces_from   # her face detected as a person, with or without a head
            return (face if late is not None and f >= late else arm)[None], [0.9]

        def sign(_f: int, _m: torch.Tensor):
            return torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])

        def head(f: int, mask: torch.Tensor):           # a head is found only where the face is, and none on her arm
            return (mask > 0.5).to(torch.float32) if heads_from is not None and f >= heads_from and torch.equal(mask, face) else None
        sign.head = head

        def track(start: int, end: int, seed: int, mask: torch.Tensor):
            calls.append((start, end, seed))
            return mask[None].repeat(end - start, 1, 1)
        got = st.follow(20, [], st.PICK_LARGEST, 8, 0.5, detect, sign, track, seeds=st.parse_seeds(text, shots), track_head=True, head_per_stretch=True)
        return got, calls

    # a body box alone, no head detected anywhere: her person track is seeded from it, and her head track is EMPTY
    got, calls = run(f"frame 8: {a} body", None)
    person = st.assemble(20, H, W, got.pieces)
    heads = st.assemble(20, H, W, got.head_pieces)
    if calls != [(0, 20, 8)] or not torch.equal(person[8] > 0.5, arm > 0.5) or bool((heads > 0.5).any()) \
            or got.shots[0].seeded.get("encloses") != "body" or "her person track only, never her head" not in st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0):
        problems.append(f"body box: a body box should seed her person track and leave her head track empty: calls {calls}, "
                        f"head px {int((heads > 0.5).sum())}, record {got.shots[0].seeded}")
    # the control: the same box not marked `body` stands in for a head by the top third of her mask, as any box does
    got, calls = run(f"frame 8: {a}", None)
    heads = st.assemble(20, H, W, got.head_pieces)
    if calls != [(0, 20, 8), (0, 20, 8)] or not torch.equal(heads[8] > 0.5, st.top_third(arm) > 0.5):
        problems.append(f"body box: the control, a plain box with no head found, should seed the head from its top third: {calls}")
    # a body box and a head box in one stretch: the head is seeded from the head box's head alone, never on the body box
    got, calls = run(f"frame 4: {a} body\nframe 15: {f_}", 12)
    head_calls = calls[2:]                               # after the two person tracks, out to each other's frames
    # her arm (tracked from the body box) and her face (from the head box) are disjoint and no detection here holds both,
    # so the frames both tracks hold are left empty, and the head is tracked in the stretch the head box is in alone
    # (main's span head, seeded from the head box too, adds nothing on her arm: (0, 20, 15) holds her face, not her arm)
    if calls[:2] != [(0, 16, 4), (4, 20, 15)] or head_calls != [(15, 20, 15), (0, 20, 15)] or got.shots[0].disagreed != list(range(5, 15)) \
            or any(bool((st.assemble(20, H, W, got.head_pieces)[f] > 0.5).any()) for f in range(0, 5)):
        problems.append(f"body box: beside a head box, only the head box's head should seed the head track: {calls}, {got.shots[0].disagreed}")
    table = st.shot_table.build(got, lambda f: (arm[None], [0.9]), st.assemble(20, H, W, got.pieces), state=st._state, phrase="person",
                                pick=st.PICK_LARGEST, named_frame=True, named_value=True, cuts=[])
    if table["shots"][0].get("seed_tracks_disagree_on") != [[5, 14]] \
            or "THE BOXES' TRACKS HELD DIFFERENT PEOPLE on 5, 6" not in st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0):
        problems.append(f"body box: the frames two boxes' tracks disagree on should be in the table and the report: {table['shots'][0].get('seed_tracks_disagree_on')}")
    # a body box FIRST and a head box second, no head found on any frame: the stand-in runs, decided by the shot's boxes as a
    # whole (one is a head box), and is taken from the head box (its frame, her mask there), never from the body box before it
    got, calls = run(f"frame 4: {a} body\nframe 15: {f_}", None, faces_from=10)
    heads = st.assemble(20, H, W, got.head_pieces)
    head_calls = calls[2:]
    if got.head_from.get(0, {}).get(15) != "span, top third" or head_calls != [(0, 20, 15)] or not torch.equal(heads[15] > 0.5, st.top_third(face) > 0.5) \
            or any(bool((heads[f] > 0.5).any()) for f in range(0, 5)):
        problems.append(f"body box: a body box first and a head box second, no head anywhere, should take the top third from the head box: "
                        f"{head_calls}, from {got.head_from.get(0, {}).get(15)}")
    # where one detection holds both her arm and her face, they are parts of her: the frame keeps both, and the head is
    # still seeded from the head box alone, over the one stretch
    whole = ((arm + face) > 0).float()
    got, calls = run(f"frame 4: {a} body\nframe 15: {f_}", 12, whole)
    person = st.assemble(20, H, W, got.pieces)
    if got.shots[0].disagreed or not torch.equal(person[8] > 0.5, whole > 0.5) or calls[2:] != [(0, 20, 15)]:
        problems.append(f"body box: two parts of her one detection holds should be kept together: {got.shots[0].disagreed}, {calls}")


def check_drawn_box(problems):
    """A box the owner DREW is approximate, never an outline (2026-10-23, the owner: "the boxes i drew were approximates!!!"):
    it names the one person lying mostly inside it, is refused across two people or on another tracked subject, and only
    where nobody qualifies asks SAM 3's box prompt, whose result is not clipped to the box."""
    a, b = _box(40, 10, 80, 60), _box(0, 20, 20, 50)
    beside, part = _box(80, 10, 120, 60), _box(50, 10, 70, 25)
    world = {"people": [b, a]}
    detect = lambda f: (torch.stack(world["people"], dim=0), [0.9] * len(world["people"]))  # noqa: E731
    loose = (25.0, 0.0, 100.0, 72.0)             # round the subject, roughly: 7,200 px of box, her 2,000 all inside it
    got = st.name_seed(detect, 4, loose, drawn=True)
    if got.index != 1 or got.record["how"] != st.SEED_BY_INSIDE or got.record["inside"] != 1.0 or abs(got.record["covers"] - 0.278) > 0.001 \
            or not got.record["drawn"]:
        problems.append(f"drawn: a loose box round one person should pick that person, with both shares recorded: {got.record}")
    # the control: the same box read as a tile's box is refused, its mask overlapping the box under SEED_OVERLAPS
    if st.name_seed(detect, 4, loose).mask is not None:
        problems.append("drawn: the control, the loose box read as a tile's, should not take anybody")
    # straddling two people, each mostly inside it: refused, never a guess
    world["people"] = [a, beside]
    got = st.name_seed(detect, 4, (50.0, 10.0, 60.0, 50.0), drawn=True)
    if got.mask is not None or "straddles" not in got.record["why"]:
        problems.append(f"drawn: a box straddling two people should be refused: {got.record}")
    # a part of her the detector also returned is not a second person: she is taken
    world["people"] = [a, part]
    got = st.name_seed(detect, 4, (35.0, 5.0, 50.0, 60.0), drawn=True)
    if got.index != 0 or got.record["how"] != st.SEED_BY_INSIDE:
        problems.append(f"drawn: a part inside her made the box ambiguous: {got.record}")
    # the person lying most inside it is another tracked subject's: refused (the lead's dress under 4647's 4674-4677),
    # whether they qualify or, as here with a fifth of them inside, do not, and before any box prompt is asked
    world["people"] = [b, a]
    asked = []
    for drawn_box in ((35.0, 5.0, 50.0, 60.0), (60.0, 40.0, 30.0, 20.0)):
        got = st.name_seed(detect, 4, drawn_box, lambda f: frozenset({1}), lambda f, bx: asked.append(bx) or a, drawn=True)
        if got.mask is not None or not got.record["why"].startswith("the person lying most inside the drawn box") or asked:
            problems.append(f"drawn: a box lying most on another tracked subject should be refused before the box prompt: {got.record}, {asked}")
    # nobody mostly inside (her, close, is a quarter inside): SAM 3's box prompt, NOT clipped to the box (a rough box would
    # cut her off), held to her detection, less others' pixels
    big = _box(10, 0, 128, 72)
    world["people"] = [b, big]
    box = (40.0, 10.0, 40.0, 50.0)
    blob = _box(30, 5, 90, 65)                    # 3,600 px, running out of the 2,000 px box on every side
    others = torch.zeros((8, H, W))
    held = st.held_people(detect, others)
    got = st.name_seed(detect, 4, box, held, lambda f, bx: blob, drawn=True)
    if got.record["how"] != st.SEED_BY_PROMPT or got.mask is None or not torch.equal(got.mask > 0.5, blob > 0.5) or got.record["px"] != 3600:
        problems.append(f"drawn: the box prompt's result should be taken whole, not clipped to the box: {got.record}")
    others[4] = _box(30, 5, 60, 65)               # another tracked subject under half of it: over DRAWN_ON_OTHERS
    got = st.name_seed(detect, 4, box, st.held_people(detect, others), lambda f, bx: blob, drawn=True)
    if got.mask is not None or "another tracked subject's track" not in got.record["why"]:
        problems.append(f"drawn: a box prompt lying on another tracked subject should be refused: {got.record}")
    others[4] = _box(85, 5, 90, 65)               # a strip along its edge, 300 of 3,600 px: under the line, taken less the strip
    got = st.name_seed(detect, 4, box, st.held_people(detect, others), lambda f, bx: blob, drawn=True)
    if got.mask is None or got.record["px"] != 3300:
        problems.append(f"drawn: a box prompt sharing an edge with another tracked subject should be taken less their pixels: {got.record}")
    # someone with a quarter of their mask in the box does not qualify: the box prompt is asked instead, and its result,
    # lying on her and running onto him only 0.14, is taken
    world["people"] = [beside, big]
    got = st.name_seed(detect, 4, (40.0, 10.0, 50.0, 50.0), None, lambda f, bx: blob, drawn=True)
    if got.record["how"] != st.SEED_BY_PROMPT or got.index is not None:
        problems.append(f"drawn: a person a quarter inside the box should not qualify (DRAWN_INSIDE): {got.record}")
    # the seed line's words, through `follow`: `drawn` (and `body` after it) reach the box's record and the report
    if st.parse_seeds("frame 4: 25, 0, 100, 72 drawn body", [(0, 8)]) != {1: [(4, loose, "drawn", "body")]}:
        problems.append("drawn: a seed line ending `drawn body` is not read as both")
    world["people"] = [b, a]
    sign = lambda f, m: (torch.tensor([1.0, 0.0]), None)    # noqa: E731
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731
    got = st.follow(8, [], st.PICK_LARGEST, 0, 0.5, detect, sign, track, seeds=st.parse_seeds("frame 4: 25, 0, 100, 72 drawn", [(0, 8)]),
                    track_head=False)
    if got.shots[0].index != 1 or not got.shots[0].seeded.get("drawn") \
            or "the one person lying mostly inside the box drawn by hand" not in st.report(got, [], st.PICK_LARGEST, "person", True, True, 1.0):
        problems.append(f"drawn: a `drawn` seed line should be named as a drawn box, and said so: {got.shots[0].seeded}")


def check_head_anchors(problems):
    """The head-sized and top-third stand-ins run only where an owner's head box anchors the stretch (2026-10-23, the blind
    read): never in a stretch with no box, never in a stretch found again, and a body box is never the pick's reference."""
    body, arm = _box(40, 10, 80, 60), _box(60, 40, 75, 60)        # her body, its head the top 8 rows; then only her arm
    held_on = set(range(0, 10)) | set(range(20, 30))
    calls = []

    def detect(f: int):
        return (body if f < 15 else arm)[None], [0.9]

    def sign(_f, _m):
        return torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])

    def head(f: int, mask: torch.Tensor):
        if f >= 15:
            return None                                          # her head hidden: none found on her arm
        out = mask.clone()
        out[int((mask > 0.5).nonzero()[:, 0].min()) + 8:] = 0.0
        return out
    sign.head = head

    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls.append((start, end, seed))
        out = torch.zeros((end - start, H, W))
        for f in range(start, end):
            if f in held_on:
                if torch.equal(mask > 0.5, body > 0.5) or torch.equal(mask > 0.5, arm > 0.5):
                    out[f - start] = body if f < 15 else arm
                elif f < 15:
                    out[f - start] = mask            # a head is carried while it shows; her head is hidden from 15
        return out
    bx = ", ".join(str(v) for v in st.shot_table._box(body))
    got = st.follow(30, [], st.PICK_LARGEST, 5, 0.5, detect, sign, track, seeds=st.parse_seeds(f"frame 5: {bx}", [(0, 30)]), track_head=True, head_per_stretch=True)
    heads = st.assemble(30, H, W, got.head_pieces)
    if any(bool((heads[f] > 0.5).any()) for f in range(20, 30)) or not all(bool((heads[f] > 0.5).any()) for f in range(0, 10)):
        problems.append(f"head anchors: a stretch holding only her arm and no box should have no head (the blind read's 300 px "
                        f"arm against a 320 px head): head px {[int(heads[f].sum()) for f in (5, 20, 25)]}")
    # a subject found again (`regain`), with no head found there: main's span head, here main's own top third of the person
    # found, as main had it (2026-10-23, the lead on the reader's census: never less than main)
    hidden = lambda f: 20 <= f < 24                                # noqa: E731
    subject, detect2, sign2, track2, _ = _lost_world(lambda f: not hidden(f), n=48)
    sign2_head = lambda f, m: None if f >= 24 else (m > 0.5).float() * (torch.arange(H)[:, None] < 20).float()   # noqa: E731
    sign2.head = sign2_head
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect2, sign2, track2, stride=4, offset=1, track_head=True, head_per_stretch=True)
    heads = st.assemble(48, H, W, got.head_pieces)
    if not got.shots[0].regained or not torch.equal(heads[30] > 0.5, st.top_third(subject) > 0.5) \
            or got.head_from.get(0, {}).get(30) != "span, top third" or got.head_from.get(0, {}).get(10) != "stretch":
        problems.append(f"head anchors: a stretch found again with no head found should keep main's span head: regained {got.shots[0].regained}, "
                        f"head px {[int(heads[f].sum()) for f in (30, 40)]}")
    # an automatic shot with NO head found on any frame keeps main's stand-in, the top third of her mask from the seed frame
    # (the owner, 2026-10-23: with no head anywhere the hair pass has nothing nearer to take her room from)
    subject5, detect5, sign5, track5, calls5 = _lost_world(lambda f: True, n=16)
    sign5.head = lambda f, m: None
    got = st.follow(16, [], st.PICK_LARGEST, 3, 0.8, detect5, sign5, track5, stride=4, offset=1, track_head=True)
    heads = st.assemble(16, H, W, got.head_pieces)
    if any(not torch.equal(heads[f] > 0.5, st.top_third(subject5) > 0.5) for f in range(16)):
        problems.append(f"head anchors: an automatic shot with no head anywhere should keep the top third: {[int(heads[f].sum()) for f in (0, 8)]}")
    # a subject found again whose mask is head-sized beside her head elsewhere (400 px against 200) is still not a head:
    # her head-sized mask stands in only where an owner's head box anchors the stretch
    small = _box(40, 10, 60, 30)
    visible6 = lambda f: not 20 <= f < 24                            # noqa: E731
    others6 = _box(0, 20, 20, 50)
    calls6 = []

    def detect6(f: int):
        return torch.stack([others6] + ([small] if visible6(f) else [])), [0.9] * (1 + visible6(f))

    def sign6(_f: int, mask: torch.Tensor):
        v = torch.tensor([1.0, 0.0, 0.0]) if torch.equal(mask, small) else torch.tensor([0.0, 1.0, 0.0])
        return v, v
    sign6.head = lambda f, m: ((m > 0.5) & (torch.arange(H)[:, None] < 20)).float() if f < 20 else None

    def track6(start: int, end: int, seed: int, mask: torch.Tensor):
        calls6.append((start, end, seed))
        out = torch.zeros((end - start, H, W))
        f = seed
        while f < end and visible6(f):
            out[f - start] = mask
            f += 1
        f = seed - 1
        while f >= start and visible6(f):
            out[f - start] = mask
            f -= 1
        return out
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect6, sign6, track6, stride=4, offset=1, track_head=True, head_per_stretch=True)
    heads = st.assemble(48, H, W, got.head_pieces)
    if not got.shots[0].regained or any(bool((heads[f] > 0.5).any()) for f in range(24, 48)) or not bool((heads[5] > 0.5).any()):
        problems.append(f"head anchors: a head-sized mask found again should not stand in for a head: {[int(heads[f].sum()) for f in (5, 30)]}")
    # the tracker's own pick, head-sized beside her head found later in the shot (400 px against 200), no box: not a head
    held_on8 = set(range(0, 10)) | set(range(20, 30))
    small8, big8 = _box(40, 10, 60, 30), _box(40, 10, 80, 60)

    def detect8(f: int):
        return (small8 if f < 15 else big8)[None], [0.9]

    def sign8(_f: int, _m: torch.Tensor):
        return torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    sign8.head = lambda f, m: (((m > 0.5) & (torch.arange(H)[:, None] < 15)).float() if f >= 15 else None)

    def track8(start: int, end: int, seed: int, mask: torch.Tensor):
        out = torch.zeros((end - start, H, W))
        for f in range(start, end):
            if f in held_on8:
                out[f - start] = (small8 if f < 15 else big8) if (torch.equal(mask, small8) or torch.equal(mask, big8)) else mask
        return out
    got = st.follow(30, [], st.PICK_LARGEST, 3, 0.5, detect8, sign8, track8, track_head=True, head_per_stretch=True)
    heads = st.assemble(30, H, W, got.head_pieces)
    # (her head-sized mask is not taken whole for a head: where main's span head reaches, that is what the frame takes)
    if torch.equal(heads[3] > 0.5, small8 > 0.5) or got.head_from.get(0, {}).get(3) != "span" or not bool((heads[25] > 0.5).any()):
        problems.append(f"head anchors: the tracker's own pick, head-sized, should not stand in for a head: {[int(heads[f].sum()) for f in (3, 25)]}, "
                        f"{got.head_from.get(0, {}).get(3)}")
    # nobody picked at all (nothing on the pick frame), a seeded shot, no head anywhere: the top third from its seed
    boxes7, detect7_all, sign7, _track7, _ = _world()
    track7 = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731  (its world's names only boxes)
    detect7 = lambda f: detect7_all(f) if f >= 8 else (torch.zeros((0, H, W)), [])   # noqa: E731
    sign7.head = lambda f, m: None
    two = ", ".join(str(v) for v in st.shot_table._box(boxes7[2]))
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 0.8, detect7, sign7, track7, stride=4, offset=1,
                    seeds=st.parse_seeds(f"frame 13: {two}", st.shot_ranges(32, [8, 16, 24])), track_head=True)
    heads = st.assemble(32, H, W, got.head_pieces)
    if got.pick_frame is not None or got.head_from.get(8, {}).get(13) != "span, top third" or not torch.equal(heads[13] > 0.5, st.top_third(boxes7[2]) > 0.5):
        problems.append(f"head anchors: with nobody picked, a seeded headless shot should keep the top third: {got.head_from.get(8, {}).get(13)}")
    # a box round her body alone on the picked shot is never the reference the other shots are judged against
    boxes, detect3, sign3, track3, _ = _world()
    shots = st.shot_ranges(32, [8, 16, 24])
    one = ", ".join(str(v) for v in st.shot_table._box(boxes[1]))
    got = st.follow(32, [8, 16, 24], st.PICK_LARGEST, 3, 0.8, detect3, sign3, track3, stride=4, offset=1,
                    seeds=st.parse_seeds(f"frame 5: {one} body", shots))
    if got.pick_seed != "body boxes only" or got.pick_frame != 3 \
            or "THE PICKED SHOT'S BOXES ARE ROUND HER BODY ONLY" not in st.report(got, [8, 16, 24], st.PICK_LARGEST, "person", True, True, 1.0):
        problems.append(f"head anchors: a body box on the picked shot should not be the reference: {got.pick_seed}, {got.pick_frame}")
    # two boxes whose head searches land on one frame make ONE head seed there, tracked once over the stretch
    face = _box(40, 10, 80, 30)
    log = []

    def head4(f: int, mask: torch.Tensor):
        return face if f == 5 else None
    sign4 = lambda f, m: (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))   # noqa: E731
    sign4.head = head4

    def track4(start: int, end: int, seed: int, mask: torch.Tensor):
        log.append((start, end, seed, "head" if torch.equal(mask > 0.5, face > 0.5) else "person"))
        return mask[None].repeat(end - start, 1, 1)
    got = st.follow(20, [], st.PICK_LARGEST, 8, 0.5, lambda f: (body[None], [0.9]), sign4, track4,
                    seeds=st.parse_seeds(f"frame 5: {bx}\nframe 8: {bx}", [(0, 20)]), track_head=True, head_per_stretch=True)
    if [c for c in log if c[3] == "head"] != [(0, 20, 5, "head")]:
        problems.append(f"head anchors: two boxes' heads found on one frame should make one head seed: {log}")


def check_review_gaps(problems):
    """The independent review of 2026-10-23: three paths that could keep another person's pixels or a stand-in head."""
    # (1) an unclipped box prompt that bleeds onto another person's arm, under PROMPT_ON_OTHERS, for the FIRST tracker of a
    # job (no earlier tracker's mask): the arm's pixels are taken out of the seed; a part of her detected apart is kept
    her, arm_b, her_head = _box(40, 10, 80, 60), _box(80, 30, 90, 60), _box(45, 10, 75, 25)
    for people, keep_part in (([_box(0, 0, 10, 10), her, arm_b], False), ([_box(0, 0, 10, 10), her, arm_b, her_head], True)):
        detect = lambda f, p=people: (torch.stack(p), [0.9] * len(p))   # noqa: E731
        bleed = _box(40, 10, 88, 60)                     # all of her, and 240 px of his arm: 0.10 of it, under the line
        got = st.name_seed(detect, 4, (30.0, 40.0, 15.0, 15.0), None, lambda f, bx: bleed, drawn=True)
        if got.mask is None or bool(((got.mask > 0.5) & (arm_b > 0.5)).any()) or bool(((her > 0.5) & ~(got.mask > 0.5)).any()):
            problems.append(f"review (1): a box prompt bleeding onto another person's arm should not keep the arm "
                            f"({'with' if keep_part else 'without'} her head detected apart): {got.record}")
    # (2) another subject's box that their own tracker names by the box prompt (their drawn box catches under half of them):
    # it is resolved the same way here, so this tracker's box lying on them is refused, not taken
    other_p = _box(60, 10, 120, 70)                      # the other subject, 3,600 px
    people = [_box(0, 0, 10, 10), other_p]
    detect = lambda f: (torch.stack(people), [0.9, 0.9])  # noqa: E731
    theirs_box = (100.0, 10.0, 20.0, 20.0)               # catches 400 of their 3,600 px: not mostly inside, so the prompt
    ours_box = (60.0, 40.0, 30.0, 30.0)

    def in_box(f, bx):
        return _box(90, 10, 120, 40) if tuple(bx) == theirs_box else _box(60, 40, 90, 70)

    def sign(_f, _m):
        return torch.tensor([1.0, 0.0]), None
    sign.in_box = in_box
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731
    text = "frame 4: 60, 40, 30, 30 drawn\nframe 4: 100, 10, 20, 20 drawn other"
    got = st.follow(8, [], st.PICK_LARGEST, 0, 0.5, detect, sign, track, seeds=st.parse_seeds(text, [(0, 8)]),
                    other_boxes=st.parse_other_boxes(text, [(0, 8)]), track_head=False)
    if got.shots[0].seed is not None or "another tracked subject" not in got.shots[0].seeded["why"]:
        problems.append(f"review (2): another subject named by their box prompt should not be taken by this tracker: {got.shots[0].seeded}")
    # (3) a headless shot keeps main's head everywhere main had one (2026-10-23, the lead on the reader's census: never less
    # than main, the owner's never-leave-originals): its own seed's top third, and on a stretch found again the top third of
    # the person found, as main tracked it there; recorded as main's span head
    subject, detect3, sign3, track3, calls3 = _lost_world(lambda f: not 20 <= f < 24, n=48)
    sign3.head = lambda f, m: None
    got = st.follow(48, [], st.PICK_LARGEST, 3, 0.8, detect3, sign3, track3, stride=4, offset=1, track_head=True)
    heads = st.assemble(48, H, W, got.head_pieces)
    head_calls = [c for c in calls3["track"] if c[3] == "other"]          # the head seed is no detection: "other" to the world
    if not got.shots[0].regained or head_calls != [(0, 48, 3, "other"), (20, 48, 24, "other")] \
            or {got.head_from.get(0, {}).get(f) for f in (10, 30)} != {"span, top third"} \
            or not all(torch.equal(heads[f] > 0.5, st.top_third(subject) > 0.5) for f in (10, 30)):
        problems.append(f"review (3): a headless shot should keep main's top third where main had it: {head_calls}, {got.head_from.get(0, {}).get(30)}")


def check_ran_code(problems):
    """Which code ran (2026-10-23, the owner: "verify mask version is even right"): the stamp is bound at module top level
    when the module is imported, is the sha256 of the file it was imported from and the loaded node's MASK_VERSION, the
    node writes that bound value into every shot table, and changing the file on disk afterwards does not change it."""
    import hashlib
    stamp = st.RAN_CODE
    if stamp != {"file": MODULE.name, "sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
                 "mask_version": st.MiniMaxH3SubjectTrack.MASK_VERSION}:
        problems.append(f"ran code: the stamp is not the file the module was imported from: {stamp}")
    # import a copy of the module, then change that file on disk: what the imported module reports stays what it read
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp)
        for name in ("subject_track.py", "shot_table.py", "subject_tracks.py"):
            (pack / name).write_bytes(((MODULE.parent if (MODULE.parent / name).exists() else REPO) / name).read_bytes())
        before = hashlib.sha256((pack / "subject_track.py").read_bytes()).hexdigest()
        pkg = types.ModuleType("_h3ran")
        pkg.__path__ = [str(pack)]
        sys.modules["_h3ran"] = pkg
        spec = importlib.util.spec_from_file_location("_h3ran.subject_track", pack / "subject_track.py")
        copy = importlib.util.module_from_spec(spec)
        sys.modules["_h3ran.subject_track"] = copy
        spec.loader.exec_module(copy)
        with open(pack / "subject_track.py", "ab") as f:
            f.write(b"\n# edited on disk after the import\n")
        after = hashlib.sha256((pack / "subject_track.py").read_bytes()).hexdigest()
        if copy.RAN_CODE["sha256"] != before or copy.RAN_CODE["sha256"] == after or copy.ran_code()["sha256"] != after:
            problems.append(f"ran code: the stamp should stay what was imported when the file changes on disk: {copy.RAN_CODE}")
        for name in ("_h3ran.subject_track", "_h3ran", "_h3ran.shot_table", "_h3ran.subject_tracks"):
            sys.modules.pop(name, None)
    # execute reports the bound value and reads no file: the hash is taken nowhere but at import
    source = MODULE.read_text(encoding="utf-8")
    body = source[source.index("    def execute(cls, frames, segmenter"):source.index("def ran_code(path=None)")]
    if 'table["ran_code"] = dict(RAN_CODE)' not in body or "hashlib" in body or "read_bytes" in body \
            or "RAN_CODE = ran_code()" not in source.split("def ran_code(path=None)")[1]:
        problems.append("ran code: execute should write the stamp bound at import, and hash nothing itself")


def check_span_fill(problems):
    """Never less than main (2026-10-23, the lead on the reader's census): a stretch with a head detection takes its own head;
    a stretch with none keeps main's span head, the head main tracked across the empty run, recorded per frame."""
    body = _box(40, 10, 80, 60)
    held_on = set(range(0, 10)) | set(range(20, 30))
    calls = []

    def detect(f: int):
        return body[None], [0.9]

    def sign(_f, _m):
        return torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])
    own = _box(40, 10, 80, 20)                                  # the head found on 0-9
    sign.head = lambda f, m: own if f < 10 else None            # none found on 20-29: she is turned away there

    def track(start: int, end: int, seed: int, mask: torch.Tensor):
        calls.append((start, end, seed, "person" if torch.equal(mask > 0.5, body > 0.5) else "head"))
        out = torch.zeros((end - start, H, W))
        for f in range(start, end):
            if f in held_on:
                out[f - start] = mask                           # SAM carries what it was given across the empty run
        return out
    bx = ", ".join(str(v) for v in st.shot_table._box(body))
    got = st.follow(30, [], st.PICK_LARGEST, 5, 0.5, detect, sign, track, seeds=st.parse_seeds(f"frame 5: {bx}", [(0, 30)]), track_head=True, head_per_stretch=True)
    heads = st.assemble(30, H, W, got.head_pieces)
    marks = got.head_from.get(0, {})
    head_calls = [c[:3] for c in calls if c[3] == "head"]
    if [marks.get(f) for f in (0, 9, 20, 29)] != ["stretch", "stretch", "span", "span"] \
            or not all(torch.equal(heads[f] > 0.5, own > 0.5) for f in (5, 25)) or head_calls != [(0, 10, 5), (0, 30, 5)]:
        problems.append(f"span fill: the stretch with a head should take its own, the one without main's span head: "
                        f"{ {f: marks.get(f) for f in (0, 20)} }, {head_calls}")
    # where main's span head does not reach (SAM loses it across the empty run), the headless stretch keeps nothing, as main
    def track_lost(start: int, end: int, seed: int, mask: torch.Tensor):
        out = torch.zeros((end - start, H, W))
        for f in range(start, end):
            if f in held_on and (torch.equal(mask > 0.5, body > 0.5) or f < 10):
                out[f - start] = mask
        return out
    got = st.follow(30, [], st.PICK_LARGEST, 5, 0.5, detect, sign, track_lost, seeds=st.parse_seeds(f"frame 5: {bx}", [(0, 30)]), track_head=True, head_per_stretch=True)
    heads = st.assemble(30, H, W, got.head_pieces)
    if any(bool((heads[f] > 0.5).any()) for f in range(20, 30)) or any(got.head_from.get(0, {}).get(f) for f in range(20, 30)):
        problems.append("span fill: where main's span head is lost across the gap, the stretch should hold no head, as main did")


def check_blind_prompt(problems):
    """The first tracker of a job, which has no earlier tracker's mask, refuses a box prompt as doubtful on a frame where it
    knows none of the job's other subjects (no `other` box there), since it cannot hold the prompt to them (2026-10-23, the
    blind re-read: the reader's dress world, 540 px of the lead's dress taken through a tile box and a drawn one)."""
    lead = _box(20, 0, 70, 72)
    detect = lambda f: (lead[None], [0.9])                       # noqa: E731
    track = lambda start, end, seed, mask: mask[None].repeat(end - start, 1, 1)   # noqa: E731

    def sign(_f, _m):
        return torch.tensor([1.0, 0.0]), None
    sign.in_box = lambda f, bx: _box(50, 45, 70, 72)            # the lead's dress, under a box drawn for the second subject
    if st.parse_job_others("frame 4: 1, 2, 3, 4\njob: 2 other subjects") != 2 or st.parse_job_others("frame 4: 1, 2, 3, 4") != 0 \
            or st.parse_seeds("frame 4: 1, 2, 3, 4\njob: 1 other subject", [(0, 8)]) != {1: [(4, (1.0, 2.0, 3.0, 4.0))]}:
        problems.append("blind: the `job: N other subjects` line is not read, or is read as a seed")
    lead_box = ", ".join(str(v) for v in st.shot_table._box(lead))
    for box_line in ("frame 4: 50, 40, 30, 32", "frame 4: 50, 40, 30, 32 drawn"):
        for job, other, taken in ((1, False, False), (0, False, True), (1, True, False)):
            text = box_line + (f"\nframe 4: {lead_box} other" if other else "") + (f"\njob: {job} other subject" if job else "")
            got = st.follow(8, [], st.PICK_LARGEST, 0, 0.5, detect, sign, track, seeds=st.parse_seeds(text, [(0, 8)]),
                            other_boxes=st.parse_other_boxes(text, [(0, 8)]), job_others=st.parse_job_others(text), track_head=False)
            seeded = got.shots[0].seeded
            # (with the lead's box carried the box is refused on her: as another tracked subject, before or by the prompt)
            if (got.shots[0].seed is not None) != taken or (not taken and not other and not seeded.get("doubtful")) \
                    or (job and not other and "knows none of them on this frame" not in seeded["why"]):
                problems.append(f"blind: {text!r} should be {'taken' if taken else 'refused as doubtful'}: {seeded}")


def main() -> int:
    problems: list[str] = []
    for check in (check_cuts, check_borders, check_ranges, check_counted, check_choose, check_signature, check_follow, check_corrections,
                  check_automatic, check_alone, check_regain, check_headless_vote, check_two_places, check_empty, check_one_detection_each, check_moved_by_a_gap, check_looks_in_the_table, check_rows_from_tables, check_seeds, check_only_person_off_with_others, check_others,
                  check_track_score, check_schema, check_track_head, check_every_box, check_box_tiers, check_head_runs,
                  check_regain_at_a_box, check_one_box_calls, check_body_box, check_drawn_box,
                  check_head_anchors, check_review_gaps, check_ran_code, check_span_fill,
                  check_blind_prompt):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the subject track finds a cut and not a lighting change, covers every frame once, picks by the "
              "rule, follows the subject and nobody else across shots, takes a correction by the tile's numbers, "
              "finds a subject again that the track let go inside a shot and takes nobody else for them, leaves "
              "absent shots empty, never takes a person another tracker holds, reads the tracker's own score per frame "
              "in step or not at all, and declares what it asks SAM as inputs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
