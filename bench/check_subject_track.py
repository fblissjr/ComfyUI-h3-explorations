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
   not an output node and it declares a `MASK_VERSION`, the two
   things `mask_store.py` needs for a kept mask to spare the tracker and to
   go stale when the node's method changes.

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
import sys
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


def check_schema(problems):
    schema = st.MiniMaxH3SubjectTrack.define_schema()
    inputs = {i.id: i for i in schema.inputs}
    for name, default in (("subject_phrase", st.SUBJECT_PHRASE), ("head_phrase", st.HEAD_PHRASE),
                          ("max_people", st.MAX_PEOPLE), ("detection_threshold", st.DETECTION_THRESHOLD),
                          ("pick", st.PICK_LARGEST)):
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
    if len(schema.outputs) != 4 or schema.outputs[0].io_type != "MASK":
        problems.append("the node's outputs are not mask, preview, report, shot_table with the mask first")
    if getattr(schema, "is_output_node", False):
        problems.append("the node is an output node: core would run the tracker on every queue, kept mask or not")
    if not isinstance(getattr(st.MiniMaxH3SubjectTrack, "MASK_VERSION", None), int):
        problems.append("the node declares no integer MASK_VERSION, so a kept mask would survive a change to how it is made")
    fix, last = schema.inputs[-2], schema.inputs[-1]
    if fix.id != "corrections" or getattr(fix, "default", None) != "" or not fix.tooltip or not getattr(fix, "multiline", False):
        problems.append("`corrections` is not the node's last input but one, a multi-line text that is empty by default, with a tooltip")
    # appended after it, so a graph saved before it keeps every widget where it was
    if last.id != "subject_from" or getattr(last, "default", None) != "" or not last.tooltip or not getattr(last, "optional", False):
        problems.append("`subject_from` is not the node's last input, an optional text that is empty by default, with a tooltip")
    sel = st._selection({"match": st.AT_VALUE, "match_threshold": 0.5}, "match", "match_threshold")
    if sel != (st.AT_VALUE, 0.5) or st._selection(st.AUTOMATIC, "match", "match_threshold") != (st.AUTOMATIC, None):
        problems.append("a DynamicCombo's nested dict, or a bare selection, is not read as the choice and its value")


def main() -> int:
    problems: list[str] = []
    for check in (check_cuts, check_borders, check_ranges, check_counted, check_choose, check_signature, check_follow, check_corrections,
                  check_automatic, check_alone, check_regain, check_headless_vote, check_two_places, check_empty, check_schema):
        check(problems)
    for p in problems:
        print(f"FAIL  {p}")
    if not problems:
        print("ok    the subject track finds a cut and not a lighting change, covers every frame once, picks by the "
              "rule, follows the subject and nobody else across shots, takes a correction by the tile's numbers, "
              "finds a subject again that the track let go inside a shot and takes nobody else for them, leaves "
              "absent shots empty, and declares what it asks SAM as inputs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
