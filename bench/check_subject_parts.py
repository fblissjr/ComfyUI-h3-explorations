#!/usr/bin/env python3
"""The Sapiens2 part node's geometry and bookkeeping, on stand-ins for the two models.

`sapiens2_parts.py` is the module. Its mask decides what a masked render
regenerates, so a wrong one fails quietly: a neighbour's hair is replaced, or
the original's is left in. Each item is one way that could happen. The models
are stood in for by two functions that read a painted frame's colours, so what
is graded is everything around the forward pass and nothing in it.

1. **The crop holds the subject and has the model's shape.** `mask_boxes`
   gives a mask's box and marks an empty frame; `crop_box` covers the box
   widened by the margin, is centred on it, and is never squashed.
2. **A map made on the crop lands on the frame where the pixels came from**,
   exactly when the crop is the model's size, closely when it is resized, and
   also when the box leaves the frame (the part off the frame is filled going
   in and dropped coming back).
3. **A neighbour inside the crop gives nothing.** A part lies only on the
   subject's mask widened by `subject_margin`. The control: with a margin
   wide enough to reach the neighbour, the same pixels are taken, so the cut
   is what keeps them out.
4. **The menu names the classes it says.** Each menu entry and each name in
   `other_classes` resolves to the classes of that name; an unknown name and
   an empty choice are refused by name; the node's boolean inputs are the
   menu, with the menu's defaults. `CLASS_NAMES` is compared with upstream's
   own list in `coderef/sapiens2/docs/SEG.md` when that checkout is present.
   Whether the checkpoint's indices follow that list is not something a check
   without weights can say: `bench/results/2026-10-05_sapiens2_first_frame.md`.
5. **A missed frame takes its neighbour's part, moved with the subject, and
   never leaves the subject.** A frame the subject is not in stays empty.
   Turned off, the missed frame is left empty. With the part found nowhere
   the node refuses; with the subject nowhere it returns empty masks.
   **A frame in doubt is held as a missed one is** (2026-10-08). A frame
   whose own labels are a sliver of the subject, and one whose labels lie
   mostly off the subject's mask, each take the nearest trusted frame's part
   at this frame's subject; they are named apart from the missed frames, in
   `Found.doubted` and in the report; nothing they held is kept. The control
   is the node as it was: with `hold_missing` off the sliver is what comes
   back, and `part_coverage` doubts the frame. A clip on which every found
   frame is in doubt has nothing to hold from and is left as found.
6. **The matte is soft only at the person's edge.** It is the alpha on the
   part and on background within `matte_reach` of it, inside the widened
   subject; a different part beside it gets nothing; with no matting model it
   is the part mask.
7. **The node refuses a mask that is not its frames'**: another frame count,
   by name, and another size.
8. **The result does not depend on the batch size.**
9. **The loader lists a folder by the architecture its config names**, and
   reads the working size from the folder's own file.
10. **The preview and the report** come out for the frames kept, and the
   report names the frames held.
11. **The report says how much of the subject the parts cover**, with
   `part_coverage.py`'s figures: exact shares on hand-made masks; a frame
   where the part covers nothing, one where it lies mostly off the subject
   and one far under the clip's median are each named, and a clip with none
   says so; the node's own figures are that function's on the masks it
   returns, so a second caller cannot print another number. A part that
   covers a sliver no longer reads as "found on every frame" and nothing
   else.
12. **The models are shown the subject alone.** What reaches the stand-in
   model is the picture within `ALONE_MARGIN` of the subject's mask and the
   fill colour everywhere else, on the crop's surroundings too; a second
   person beside the subject, in the taken class's colour, is in the label
   map without it (the control) and is not with it; where nobody overlaps
   the subject, the mask and the matte are the same with and without; the
   margin is never under `subject_margin`, so a label that counts was read
   from the picture; both models are handed the same crop; and the frames
   given are not written to.

No model, no weights, no CUDA, no server.

    CUDA_VISIBLE_DEVICES= <comfy venv python> bench/check_subject_parts.py
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from _lib import bootstrap, case, finish  # noqa: E402

bootstrap(cpu=True)  # no device is used: the same masked or not, and no CUDA context beside a render

import torch  # noqa: E402


def _load():
    """`sapiens2_parts` as a module of a stand-in package (`check_audio_freeze.py` says why)."""
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules.setdefault("_h3pack", pkg)
    spec = importlib.util.spec_from_file_location("_h3pack.sapiens2_parts", REPO / "sapiens2_parts.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_h3pack.sapiens2_parts"] = module
    spec.loader.exec_module(module)
    return module


sp = _load()

# The stand-in models read a painted frame. Each colour is a class, and the
# grey is a soft fringe: background to the label map, half alpha to the matte.
HAIR, FACE, UPPER = sp.CLASS_NAMES.index("Hair"), sp.CLASS_NAMES.index("Face_Neck"), sp.CLASS_NAMES.index("Upper_Clothing")
CODES = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]])
CODE_CLASS = torch.tensor([sp.BACKGROUND, HAIR, FACE, UPPER, sp.BACKGROUND])
CODE_ALPHA = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.5])
RED, GREEN, BLUE, GREY = CODES[1], CODES[2], CODES[3], CODES[4]

SIZE = (64, 48)                    # the stand-in model's working size, height by width
H, W = 96, 160
MEAN, STD = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
# A subject whose box, widened by MARGIN, is exactly SIZE: the crop is not resampled.
MARGIN = 8
TOP, LEFT, TALL, WIDE = 24, 56, 48, 32


def _nearest(crops: torch.Tensor) -> torch.Tensor:
    """Per pixel, the index of the nearest code colour. [B, h, w]."""
    return (crops.movedim(1, -1)[..., None, :] - CODES).square().sum(dim=-1).argmin(dim=-1)


def seg(crops: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.one_hot(CODE_CLASS[_nearest(crops)], len(sp.CLASS_NAMES)).movedim(-1, 1).float() * 10.0


def matting(crops: torch.Tensor) -> torch.Tensor:
    return CODE_ALPHA[_nearest(crops)][:, None]


def person(frame: torch.Tensor, mask: torch.Tensor, left: int, hair: bool = True) -> None:
    """Paint one person: hair over face over clothing, a grey fringe above the hair, and their mask."""
    frame[TOP - 4:TOP, left:left + WIDE] = GREY
    frame[TOP:TOP + 12, left:left + WIDE] = RED if hair else BLUE
    frame[TOP + 12:TOP + 24, left:left + WIDE] = GREEN if hair else BLUE
    frame[TOP + 24:TOP + TALL, left:left + WIDE] = BLUE
    mask[TOP:TOP + TALL, left:left + WIDE] = 1.0


def scene(n: int = 1):
    return torch.zeros(n, H, W, 3), torch.zeros(n, H, W)


def run(frames, mask, classes=(HAIR,), **kwargs):
    kwargs.setdefault("crop_margin", MARGIN)
    kwargs.setdefault("subject_margin", 2)
    kwargs.setdefault("size", SIZE)
    use_matting = kwargs.pop("use_matting", True)
    return sp.subject_parts(frames, mask, classes, seg, matting if use_matting else None, mean=MEAN, std=STD, **kwargs)


def check_boxes(problems):
    mask = torch.zeros(3, H, W)
    mask[0, 10:20, 30:70] = 1.0
    mask[2, 0:H, 0:5] = 1.0
    boxes = sp.mask_boxes(mask).tolist()
    if boxes != [[30, 10, 70, 20], [-1, -1, -1, -1], [0, 0, 5, H]]:
        problems.append(f"mask_boxes gave {boxes}")
    for box in ([30, 10, 70, 20], [0, 0, 5, H], [56, 24, 88, 72], [3, 3, 4, 90]):
        for margin in (0, 8, 33):
            x0, y0, x1, y1 = sp.crop_box(box, margin, SIZE)
            wide, tall = x1 - x0, y1 - y0
            if x0 > box[0] - margin or y0 > box[1] - margin or x1 < box[2] + margin or y1 < box[3] + margin:
                problems.append(f"crop_box {box} margin {margin} does not cover the widened box: {(x0, y0, x1, y1)}")
            # the model's shape to within the one pixel a whole-pixel box allows
            if abs(tall * SIZE[1] - wide * SIZE[0]) >= max(SIZE):
                problems.append(f"crop_box {box} margin {margin} is {wide}x{tall}, not the shape of {SIZE[1]}x{SIZE[0]}")
            if abs((x0 + x1) - (box[0] + box[2])) > 1 or abs((y0 + y1) - (box[1] + box[3])) > 1:
                problems.append(f"crop_box {box} margin {margin} is not centred on the subject")
    if sp.crop_box([LEFT, TOP, LEFT + WIDE, TOP + TALL], MARGIN, SIZE) != (LEFT - 8, TOP - 8, LEFT + WIDE + 8, TOP + TALL + 8):
        problems.append("the check's own subject no longer crops to the stand-in model's size; items 2, 3, 5 and 6 rely on it")


def check_round_trip(problems):
    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    found = run(frames, mask, (HAIR, FACE, UPPER), use_matting=False)
    truth = torch.zeros(H, W)
    truth[TOP:TOP + TALL, LEFT:LEFT + WIDE] = 1.0
    if not torch.equal(found.parts[0], truth):
        problems.append("with the crop at the model's size, the parts did not land on the pixels they were read from")
    hair = run(frames, mask, (HAIR,), use_matting=False).parts[0]
    want = torch.zeros(H, W)
    want[TOP:TOP + 12, LEFT:LEFT + WIDE] = 1.0
    if not torch.equal(hair, want):
        problems.append("choosing Hair did not give the stand-in's hair pixels and only those")
    # resized: the same scene through a model twice the size
    big = run(frames, mask, (HAIR,), use_matting=False, size=(SIZE[0] * 2, SIZE[1] * 2)).parts[0]
    both = float((big * want).sum()) / max(float(((big + want) > 0).sum()), 1.0)
    if both < 0.95:
        problems.append(f"through a resized crop the hair overlaps its place by {both:.2f}, under 0.95")
    # at the frame's edge: the crop box leaves the frame on the left and the top
    frames, mask = scene()
    edge = torch.zeros(H, W)
    frames[0, 0:12, 0:WIDE] = RED
    frames[0, 12:TALL, 0:WIDE] = BLUE
    mask[0, 0:TALL, 0:WIDE] = 1.0
    edge[0:12, 0:WIDE] = 1.0
    box = sp.crop_box(sp.mask_boxes(mask)[0], MARGIN, SIZE)
    if box[0] >= 0 or box[1] >= 0:
        problems.append("the edge case's box does not leave the frame, so it checks nothing")
    if not torch.equal(run(frames, mask, (HAIR,), use_matting=False).parts[0], edge):
        problems.append("a subject at the frame's edge: the parts landed shifted, or the fill was read as a part")
    crop = sp.take_crops(frames, [box], SIZE, (0.25, 0.5, 0.75))[0]
    if not torch.allclose(crop[:, 0, 0], torch.tensor([0.25, 0.5, 0.75])):
        problems.append("where the crop box leaves the frame the crop is not the fill colour")
    back = sp.paste_back(torch.ones(box[3] - box[1], box[2] - box[0]), box, H, W)
    if tuple(back.shape) != (H, W) or float(back.sum()) != float((box[2]) * (box[3])):
        problems.append("paste_back did not drop exactly the part of the box that lies off the frame")


def check_neighbour(problems):
    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    # a neighbour's hair inside the crop, four pixels to the right of the subject
    frames[0, TOP:TOP + 12, LEFT + WIDE + 4:LEFT + WIDE + 8] = RED
    theirs = torch.zeros(H, W, dtype=torch.bool)
    theirs[TOP:TOP + 12, LEFT + WIDE + 4:LEFT + WIDE + 8] = True
    box = sp.crop_box(sp.mask_boxes(mask)[0], MARGIN, SIZE)
    if not (box[0] <= LEFT + WIDE + 4 and LEFT + WIDE + 8 <= box[2]):
        problems.append("the neighbour is not inside the crop, so item 3 checks nothing")
    tight = run(frames, mask, (HAIR,), subject_margin=2)
    if bool(tight.parts[0][theirs].any()) or bool(tight.matte[0][theirs].any()):
        problems.append("a neighbour's hair inside the crop was taken as the subject's")
    wide = sp.grow(mask, 2)[0] > 0.5
    if bool(tight.parts[0][~wide].any()) or bool(tight.matte[0][~wide].any()):
        problems.append("something was returned outside the subject's mask widened by subject_margin")
    loose = run(frames, mask, (HAIR,), subject_margin=8)
    if not bool(loose.parts[0][theirs].all()):
        problems.append("control failed: with a margin that reaches the neighbour their hair was still not taken, "
                        "so the cut to the subject is not what kept it out")


def check_menu(problems):
    names = sp.CLASS_NAMES
    if len(names) != len(sp.PALETTE) or len(set(names)) != len(names) or names[sp.BACKGROUND] != "Background":
        problems.append("CLASS_NAMES and PALETTE are not one distinct name and one colour per class, background first")
    doc = REPO / "coderef" / "sapiens2" / "docs" / "SEG.md"
    if doc.is_file():
        listed = dict((int(i), n) for i, n in re.findall(r"(\d+): ([A-Z][A-Za-z_]+)", doc.read_text().split("## Model Zoo")[0]))
        if [listed.get(i) for i in range(len(names))] != list(names):
            problems.append("CLASS_NAMES is not upstream's list in coderef/sapiens2/docs/SEG.md")
    else:
        print("note  coderef/sapiens2 is not checked out: CLASS_NAMES was not compared with upstream's list")
    for entry, classes in sp.PARTS.items():
        if any(c not in names for c in classes):
            problems.append(f"the menu's {entry} names a class that does not exist: {classes}")
    off = dict.fromkeys(sp.PARTS, False)
    cases = [
        ({**off, "hair": True}, "", {"Hair"}),
        ({**off, "mouth": True}, "", {"Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"}),
        ({**off, "hands": True}, "", {"Left_Hand", "Right_Hand"}),
        (off, " upper clothing , LEFT_shoe,", {"Upper_Clothing", "Left_Shoe"}),
        ({**off, "hair": True}, "Hair, background", {"Hair"}),
    ]
    for menu, other, want in cases:
        got = {names[c] for c in sp.chosen_classes(menu, other)}
        if got != want:
            problems.append(f"the choice {[k for k, v in menu.items() if v]} with {other!r} took {sorted(got)}, not {sorted(want)}")
    face = {names[c] for c in sp.chosen_classes({**off, "face_and_neck": True})}
    if not {"Face_Neck", "Eyeglass", "Lower_Lip", "Tongue"} <= face or "Hair" in face:
        problems.append(f"face_and_neck took {sorted(face)}: a face with its glasses or mouth cut out, or with the hair")
    for menu, other, said in ((off, "", "no part is chosen"), (off, "Shoes", "'Shoes'"), ({"beard": True}, "", "'beard'")):
        try:
            sp.chosen_classes(menu, other)
            problems.append(f"the choice {menu} with {other!r} was accepted")
        except ValueError as exc:
            if said not in str(exc):
                problems.append(f"the refusal of {other!r} does not name it: {exc}")
    schema = sp.MiniMaxH3SubjectParts.define_schema()
    from comfy_api.latest import io
    toggles = {i.id: i.default for i in schema.inputs if isinstance(i, io.Boolean.Input)}
    toggles.pop("hold_missing", None)
    if toggles != {entry: entry in sp.PARTS_ON for entry in sp.PARTS}:
        problems.append(f"the node's part inputs {toggles} are not the menu with its defaults")
    if any(not i.tooltip for i in schema.inputs):
        problems.append("an input of the Subject Parts node has no tooltip")
    for name in ("MiniMaxH3Sapiens2Loader", "MiniMaxH3SubjectParts"):
        if re.search(rf"[\s\[]{name}[,\]]", (REPO / "nodes.py").read_text()) is None:
            problems.append(f"{name} is not in the pack's node list (nodes.py)")
    if not isinstance(getattr(sp.MiniMaxH3SubjectParts, "MASK_VERSION", None), int):
        problems.append("the Subject Parts node has no MASK_VERSION, so a kept mask downstream would not notice a change in it")


def check_hold(problems):
    frames, mask = scene(4)
    person(frames[0], mask[0], LEFT)
    person(frames[1], mask[1], LEFT + 10, hair=False)        # on screen, no hair or face to find
    frames[2, TOP:TOP + 12, LEFT:LEFT + WIDE] = RED            # hair on screen, but not the subject's: no mask
    person(frames[3], mask[3], LEFT + 20)
    held = run(frames, mask, (HAIR, FACE), hold_missing=True)
    want = torch.zeros(H, W)
    want[TOP:TOP + 24, LEFT + 10:LEFT + 10 + WIDE] = 1.0
    if held.held != [1]:
        problems.append(f"the frames held were {held.held}, not the one missed frame")
    if not torch.equal(held.parts[1], want):
        problems.append("a held part did not move with the subject: it is not the neighbour's part at this frame's subject")
    if bool((held.matte[1] * (1.0 - (sp.grow(mask[1:2], 2)[0] > 0.5).float())).any()):
        problems.append("a held matte reaches outside this frame's subject")
    if bool(held.parts[2].any()) or bool(held.matte[2].any()):
        problems.append("a frame the subject is not in was given a part")
    if held.present.tolist() != [True, True, False, True] or held.found.tolist() != [True, False, False, True]:
        problems.append(f"present {held.present.tolist()} and found {held.found.tolist()} are not what the scene holds")
    left = run(frames, mask, (HAIR, FACE), hold_missing=False)
    if left.held or bool(left.parts[1].any()):
        problems.append("with hold_missing off a missed frame was still given a part")
    try:
        run(frames[1:2], mask[1:2], (HAIR, FACE))
        problems.append("a part found on no frame was accepted, which would leave the original in every frame")
    except ValueError as exc:
        if "Hair" not in str(exc):
            problems.append(f"the refusal of a part found nowhere does not name it: {exc}")
    empty = run(frames, torch.zeros_like(mask), (HAIR,))
    if bool(empty.parts.any()) or bool(empty.matte.any()) or bool(empty.present.any()):
        problems.append("with the subject on no frame the node did not return empty masks")
    if tuple(sp.hold(want, (0, 0, W, H), (0, 0, W, H), hard=True).shape) != (H, W):
        problems.append("hold changed the frame's shape")


def check_hold_doubted(problems):
    """Item 5, second half. Frames 1 and 2 are found (one pixel is) and are not the subject's part."""
    frames, mask = scene(5)
    person(frames[0], mask[0], LEFT)
    # frame 1: the subject is there, and all the model labels on them is two rows of hair: a sliver
    mask[1, TOP:TOP + TALL, LEFT + 10:LEFT + 10 + WIDE] = 1.0
    frames[1, TOP + 4:TOP + 6, LEFT + 10:LEFT + 10 + WIDE] = RED
    # frame 2: the labels lie on a neighbour's edge, in the margin beside the subject's mask, one column on them
    mask[2, TOP:TOP + TALL, LEFT + 14:LEFT + 14 + WIDE] = 1.0
    frames[2, TOP:TOP + 24, LEFT + 12:LEFT + 15] = RED
    person(frames[3], mask[3], LEFT + 20)
    person(frames[4], mask[4], LEFT + 20)
    was = run(frames, mask, (HAIR, FACE), hold_missing=False)                 # the control: the node as it was
    doubt = sp.summarise(sp.coverage(mask, was.parts)).suspect
    if doubt != [1, 2] or not all(was.found[[1, 2]].tolist()):
        problems.append(f"the control does not hold two found frames in doubt (doubted {doubt}, found "
                        f"{was.found.tolist()}); the case tests nothing")
    if was.doubted or int(was.parts[1].sum()) != 2 * WIDE:
        problems.append("with hold_missing off a doubted frame did not keep the part it was found with")
    held = run(frames, mask, (HAIR, FACE), hold_missing=True)
    if held.doubted != [1, 2] or held.held != [1, 2] or held.found.tolist() != [True, False, False, True, True]:
        problems.append(f"doubted {held.doubted}, held {held.held}, found {held.found.tolist()}: not the two frames in doubt")
    for f, left in ((1, LEFT + 10), (2, LEFT + 14)):
        want = torch.zeros(H, W)
        want[TOP:TOP + 24, left:left + WIDE] = 1.0
        if not torch.equal(held.parts[f], want):
            problems.append(f"a doubted frame ({f}) did not take the nearest trusted frame's part at its own subject")
        if bool((held.parts[f] * (1.0 - (sp.grow(mask[f:f + 1], 2)[0] > 0.5).float())).any()):
            problems.append(f"a doubted frame's held part ({f}) reaches outside its subject")
    if sp.summarise(held.coverage).suspect:
        problems.append(f"after the hold the node's own coverage still doubts {sp.summarise(held.coverage).suspect}")
    for f in (0, 3, 4):
        if not torch.equal(held.parts[f], was.parts[f]):
            problems.append(f"a trusted frame ({f}) changed when its neighbours were held")
    text = sp.report(held, (HAIR, FACE), MARGIN, 2, sp.MATTE_REACH, True, SIZE, "stand-in")
    if "found but not trusted on 2 frames" not in text or "1-2" not in text or "not found on" in text:
        problems.append("the report does not name the two doubted frames apart from missed ones")
    # every found frame in doubt (all of them mostly off the subject): nothing to hold from, left as found
    lone_frames, lone_mask = scene(2)
    for f in range(2):
        lone_mask[f, TOP:TOP + TALL, LEFT + 14:LEFT + 14 + WIDE] = 1.0
        lone_frames[f, TOP:TOP + 24, LEFT + 12:LEFT + 15] = RED
    lone = run(lone_frames, lone_mask, (HAIR, FACE), hold_missing=True)
    if lone.doubted or lone.held or not all(lone.found.tolist()):
        problems.append("a clip whose every found frame is in doubt was not left as found")


def check_matte(problems):
    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    hair = torch.zeros(H, W, dtype=torch.bool)
    hair[TOP:TOP + 12, LEFT:LEFT + WIDE] = True
    face = torch.zeros(H, W, dtype=torch.bool)
    face[TOP + 12:TOP + 24, LEFT:LEFT + WIDE] = True
    fringe = torch.zeros(H, W, dtype=torch.bool)
    fringe[TOP - 4:TOP, LEFT:LEFT + WIDE] = True
    soft = run(frames, mask, (HAIR,), subject_margin=2, matte_reach=8).matte[0]
    if not bool((soft[hair] == 1.0).all()):
        problems.append("the matte is not the alpha on the part itself")
    near = fringe.clone()
    near[:TOP - 2] = False                                     # the two fringe rows inside the widened subject
    if not bool((soft[near] == 0.5).all()):
        problems.append("the matte dropped the soft edge beside the part")
    if bool(soft[fringe & ~near].any()):
        problems.append("the matte's soft edge reaches outside the subject's mask widened by subject_margin")
    if bool(soft[face].any()):
        problems.append("the matte gave weight to another part of the same person: the border between two parts must stay hard")
    short = run(frames, mask, (HAIR,), subject_margin=6, matte_reach=1).matte[0]
    last = fringe.clone()
    last[:TOP - 1] = False
    if not bool((short[last] == 0.5).all()) or bool(short[fringe & ~last].any()):
        problems.append("matte_reach does not bound how far from the part the soft edge extends")
    hard = run(frames, mask, (HAIR,), use_matting=False)
    if not torch.equal(hard.matte, hard.parts):
        problems.append("with no matting model the matte is not the part mask")


def check_refusals(problems):
    node = sp.MiniMaxH3SubjectParts
    cases = (
        (torch.zeros(3, 8, 8, 3), torch.zeros(2, 8, 8), ("subject_mask", "2", "3")),
        (torch.zeros(2, 8, 8, 3), torch.zeros(2, 8, 16), ("subject_mask", "16x8")),
        (torch.zeros(2, 8, 8, 3), torch.zeros(8, 8), ("subject_mask",)),
    )
    for frames, mask, said in cases:
        try:
            node.execute(None, frames, mask)
            problems.append(f"a mask {tuple(mask.shape)} was accepted for frames {tuple(frames.shape)}")
        except ValueError as exc:
            if any(word not in str(exc) for word in said):
                problems.append(f"the refusal of a mask {tuple(mask.shape)} does not name it and the mismatch: {exc}")


def check_batch(problems):
    frames, mask = scene(5)
    for f in range(5):
        person(frames[f], mask[f], LEFT + 6 * f, hair=f != 2)
    one, many = run(frames, mask, (HAIR, FACE), batch=1), run(frames, mask, (HAIR, FACE), batch=4)
    if not (torch.equal(one.parts, many.parts) and torch.equal(one.matte, many.matte) and one.held == many.held
            and torch.equal(one.seen, many.seen)):
        problems.append("the result depends on the batch size")
    if int(one.seen[0, HAIR]) != 12 * WIDE or int(one.seen[2, HAIR]) != 0:
        problems.append("the per-frame class counts are not the subject's pixels of each class")


def check_loader(problems):
    with tempfile.TemporaryDirectory() as root:
        for name, arch in (("seg-a", sp.SEG_ARCH), ("matte-a", sp.MATTE_ARCH), ("other", "SomethingElse")):
            folder = Path(root) / name
            folder.mkdir()
            (folder / "config.json").write_text(json.dumps({"architectures": [arch]}))
        (Path(root) / "empty").mkdir()
        (Path(root) / "seg-a" / "preprocessor_config.json").write_text(json.dumps(
            {"size": {"height": 1024, "width": 768}, "image_mean": [0.1, 0.2, 0.3], "image_std": [0.4, 0.5, 0.6]}))
        if sp.list_models(sp.SEG_ARCH, [root]) != ["seg-a"] or sp.list_models(sp.MATTE_ARCH, [root]) != ["matte-a"]:
            problems.append("the loader does not list a folder by the architecture its config names")
        if sp.working_size(sp.model_folder("seg-a", [root])) != ((1024, 768), (0.1, 0.2, 0.3), (0.4, 0.5, 0.6)):
            problems.append("the working size and normalisation are not read from the folder's preprocessor_config.json")
        try:
            sp.model_folder("missing", [root])
            problems.append("a model folder that does not exist was accepted")
        except ValueError:
            pass


def check_shown(problems):
    frames, mask = scene(4)
    person(frames[0], mask[0], LEFT)
    person(frames[1], mask[1], LEFT + 10, hair=False)
    person(frames[3], mask[3], LEFT + 20)
    keep = sp.preview_frames(sp.mask_boxes(mask)[:, 0] >= 0)
    if keep != (0, 1, 3):
        problems.append(f"the preview's frames {keep} are not the frames the subject is in")
    if len(sp.preview_frames(torch.ones(200, dtype=torch.bool))) != sp.PREVIEW_TILES:
        problems.append("the preview is not bounded on a long clip")
    found = run(frames, mask, (HAIR, FACE), keep=keep)
    sheet = sp.preview(frames, found, SIZE)
    if sheet.ndim != 4 or sheet.shape[0] != 1 or sheet.shape[-1] != 3 or not (0.0 <= float(sheet.min()) and float(sheet.max()) <= 1.0):
        problems.append(f"the preview is not one picture in 0..1: {tuple(sheet.shape)}")
    if sheet.shape[2] != len(keep) * round(sp.TILE_HEIGHT * SIZE[1] / SIZE[0]):
        problems.append("the preview does not hold one tile per kept frame")
    text = sp.report(found, (HAIR, FACE), MARGIN, 2, 8, True, SIZE, None)
    for said in ("3 of 4 frames", "Hair, Face_Neck", "nearest found frame's part: 1", "no matting model"):
        if said not in text:
            problems.append(f"the report does not say `{said}`:\n{text}")
    if sp.ranges([3, 4, 5, 9]) != "3-5, 9":
        problems.append("ranges does not write runs of frames")


def check_coverage(problems):
    pc = sys.modules["_h3pack.part_coverage"]
    subject = torch.zeros(6, 20, 20)
    parts = torch.zeros(6, 20, 20)
    subject[:5, 0:10, 0:10] = 1.0                              # 100 px on frames 0 to 4; frame 5 has no subject
    parts[0:3, 0:9, 0:10] = 1.0                                # 90 of them covered
    parts[3, 0:2, 0:10] = 1.0                                  # 20 covered: under a quarter of the median
    parts[4, 12:14, 0:10] = 1.0                                # none covered, 20 px beside the subject
    parts[2, 10:20, 0:10] = 1.0                                # frame 2 also holds 100 px off the subject
    got = pc.coverage(subject, parts)
    want_covered = [0.9, 0.9, 0.9, 0.2, 0.0, 0.0]
    if not torch.allclose(got.covered, torch.tensor(want_covered, dtype=torch.float64)):
        problems.append(f"covered is {got.covered.tolist()}, not {want_covered}")
    if not torch.allclose(got.outside, torch.tensor([0.0, 0.0, 100 / 190, 0.0, 1.0, 0.0], dtype=torch.float64)):
        problems.append(f"outside is {got.outside.tolist()}")
    if got.present.tolist() != [True] * 5 + [False]:
        problems.append("a frame with no subject is counted as one the subject is in")
    told = pc.summarise(got)
    if (told.nothing, told.low, told.outside, told.suspect) != ([4], [3], [2, 4], [2, 3, 4]):
        problems.append(f"the frames in doubt are nothing {told.nothing}, low {told.low}, outside {told.outside}")
    if abs(told.median - 0.9) > 1e-9 or told.lowest != 0.0 or told.lowest_frame != 4 or told.present != 5:
        problems.append(f"the clip's figures are median {told.median}, lowest {told.lowest} on {told.lowest_frame}")
    text = "\n".join(told.lines())
    for said in ("median of 90%", "lowest 0% on frame 4", "no part on the subject at all on 1 frames: 4",
                 "off the subject's mask on 2 frames", "under 0.25 of the median on 1 frames: 3"):
        if said not in text:
            problems.append(f"the coverage lines do not say `{said}`:\n{text}")
    if told.warning() is None or "3 of 5 frames" not in told.warning():
        problems.append(f"the line for a node downstream is {told.warning()!r}")
    # the control for "low is against the clip's own median": the same frame is not low when every frame is as small
    small = torch.zeros(3, 20, 20)
    small[:, 0:2, 0:10] = 1.0
    if pc.summarise(pc.coverage(subject[:3], small)).suspect:
        problems.append("a part that is a fifth of the subject on every frame is in doubt: low must be against the median")
    # a clip with nothing to doubt says so, and gives no warning; a sliver is printed as a sliver, not as 0%
    fine = pc.summarise(pc.coverage(subject[:3], subject[:3]))
    if fine.suspect or fine.warning() is not None or "no frame is empty" not in "\n".join(fine.lines()):
        problems.append("a part that covers the subject on every frame is still in doubt, or the lines do not say it is not")
    if pc._percent(0.004) != "0.4%" or pc._percent(0.0) != "0%" or pc._percent(0.94) != "94%":
        problems.append("a sliver of coverage is printed as nothing")
    if pc.summarise(pc.coverage(torch.zeros(2, 4, 4), torch.zeros(2, 4, 4))).present != 0:
        problems.append("a clip the subject is never in was summarised as if they were")
    # counted a chunk of frames at a time, with the same figures whatever the chunk
    chunk = pc.CHUNK
    pc.CHUNK = 4
    try:
        small_chunks = pc.coverage(subject, parts)
    finally:
        pc.CHUNK = chunk
    if not (torch.equal(small_chunks.covered, got.covered) and torch.equal(small_chunks.outside, got.outside)
            and torch.equal(small_chunks.present, got.present)) or pc.CHUNK < 6:
        problems.append("the figures depend on how many frames are counted at a time, or the chunk is smaller than this "
                        "case's clip so the comparison is of one path with itself")
    for bad in (torch.zeros(6, 20, 21), torch.zeros(5, 20, 20)):
        try:
            pc.coverage(subject, bad)
            problems.append(f"a part mask {tuple(bad.shape)} was accepted for a subject mask {tuple(subject.shape)}")
        except ValueError:
            pass
    # the node: its figures are that function's on the masks it returns, labelled share included
    frames, mask = scene(3)
    person(frames[0], mask[0], LEFT)
    person(frames[1], mask[1], LEFT + 10)
    person(frames[2], mask[2], LEFT + 20, hair=False)
    frames[2, TOP:TOP + 1, LEFT + 20:LEFT + 22] = RED          # two pixels of hair on the third frame
    # with the hold off, so the frame comes back as it was found: with it on the frame is held (item 5)
    found = run(frames, mask, (HAIR,), use_matting=False, hold_missing=False)
    again = pc.coverage(mask, found.parts)
    if found.coverage is None or not (torch.equal(found.coverage.covered, again.covered)
                                      and torch.equal(found.coverage.outside, again.outside)):
        problems.append("the node's coverage is not coverage() of the subject mask and the part mask it returns")
    if found.found.tolist() != [True, True, True] or found.held:
        problems.append("the sliver frame is not a found frame, so this case does not hold the fault it is for")
    text = sp.report(found, (HAIR,), MARGIN, 2, 8, False, SIZE, None)
    if "under 0.25 of the median on 1 frames: 2" not in text:
        problems.append(f"a part that covers two pixels of the subject on a frame is reported as found and nothing else:\n{text}")
    if "labelled a median of" not in text:
        problems.append("the report does not give the share the part model labelled as a person")
    if abs(float(found.coverage.labelled[0]) - (TALL * WIDE) / float((sp.grow(mask[:1], 2) > 0.5).sum())) > 1e-9:
        problems.append("the labelled share is not the labelled pixels over the widened subject")
    if sp.ranges is not pc.ranges:
        problems.append("the part node and the coverage module write frame runs with two functions")


def check_alone(problems):
    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    # a second person, 12 pixels to the right of the subject: inside the crop, past every margin used here
    there = LEFT + WIDE + 12
    frames[0, TOP:TOP + 12, there:there + 6] = RED
    theirs = torch.zeros(H, W, dtype=torch.bool)
    theirs[TOP:TOP + 12, there:there + 6] = True
    box = sp.crop_box(sp.mask_boxes(mask)[0], 24, (96, 80))
    if (box[2] - box[0], box[3] - box[1]) != (80, 96) or not (box[0] <= there and there + 6 <= box[2]):
        problems.append("the scene's crop is resampled or does not hold the second person, so item 12 checks nothing")
    before = frames.clone()
    given = {}

    def seg_seen(crops):
        given["seg"] = crops.clone()
        return seg(crops)

    def matting_seen(crops):
        given["matting"] = crops.clone()
        return matting(crops)

    def parts(**kwargs):
        return sp.subject_parts(frames, mask, (HAIR,), seg_seen, matting_seen, mean=MEAN, std=STD, size=(96, 80),
                                crop_margin=24, subject_margin=2, keep=(0,), **kwargs)

    plain = parts(show_alone=False)
    shown_plain = given["seg"][0].movedim(0, -1)
    if int(plain.labels[0][theirs].eq(HAIR).sum()) != int(theirs.sum()):
        problems.append("control failed: shown the picture as it is, the second person is not in the label map")
    got = parts()
    shown = given["seg"][0].movedim(0, -1)                     # the crop as the model was handed it, [h, w, 3]
    if not torch.equal(given["seg"], given["matting"]):
        problems.append("the two models were handed different crops")
    if bool((got.labels[0][theirs] != sp.BACKGROUND).any()):
        problems.append("a second person beside the subject is still in the label map: the model was shown them")
    reach = max(sp.ALONE_MARGIN, 2)
    near = sp.grow(mask, reach)[0] > 0.5
    crop_near = near[box[1]:box[3], box[0]:box[2]]
    if not torch.equal(shown[crop_near], shown_plain[crop_near]):
        problems.append("within the margin of the subject's mask the model was not shown the picture")
    fill = torch.tensor(MEAN)
    if not bool((shown[~crop_near] == fill).all()):
        problems.append("further than the margin from the subject the model was shown something other than the fill colour")
    if not (torch.equal(got.parts, plain.parts) and torch.equal(got.matte, plain.matte)):
        problems.append("with nobody overlapping the subject, showing them alone changed the mask or the matte")
    # asked of the function itself: the node hands it a copy of the batch's frames, which would hide a write
    sp.alone(frames, mask, reach, MEAN)
    if not torch.equal(frames, before):
        problems.append("the frames given were written to")
    # a label counts out to subject_margin, so the picture is kept at least that far
    wide = sp.subject_parts(frames, mask, (HAIR,), seg_seen, None, mean=MEAN, std=STD, size=(96, 80),
                            crop_margin=24, subject_margin=sp.ALONE_MARGIN + 6)
    counted = (sp.grow(mask, sp.ALONE_MARGIN + 6)[0] > 0.5)[box[1]:box[3], box[0]:box[2]]
    if not torch.equal(given["seg"][0].movedim(0, -1)[counted], shown_plain[counted]):
        problems.append("with a subject_margin wider than ALONE_MARGIN, a pixel whose label counts was replaced")
    if not bool(wide.found[0]):
        problems.append("with a wide subject_margin the subject's own part was not found")
    text = sp.report(got, (HAIR,), 24, 2, 8, True, (96, 80), None)
    if "shown the subject alone" not in text or f"{reach} px" not in text:
        problems.append(f"the report does not say the models were shown the subject alone, and how far:\n{text}")
    schema = sp.MiniMaxH3SubjectParts.define_schema()
    if any("alone" in i.id for i in schema.inputs):
        problems.append("showing the subject alone became an input of the node; it is how the node works")


def check_held(problems):
    """`Found.holds`: what the subject holds, with no name asked (2026-10-07). Inside the subject's own mask,
    background to the part model, near the lips or a hand. Controls: the same thing far from both, taken only
    when the reach is made long; the same thing beside the hand but outside the mask, never taken."""
    hand, lip = sp.CLASS_NAMES.index("Right_Hand"), sp.CLASS_NAMES.index("Upper_Lip")
    codes = torch.cat([CODES, torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])])
    classes = torch.cat([CODE_CLASS, torch.tensor([hand, lip, sp.BACKGROUND])])
    yellow, magenta, cyan = codes[5], codes[6], codes[7]

    def seg_held(crops):
        nearest = (crops.movedim(1, -1)[..., None, :] - codes).square().sum(dim=-1).argmin(dim=-1)
        return torch.nn.functional.one_hot(classes[nearest], len(sp.CLASS_NAMES)).movedim(-1, 1).float() * 10.0

    def found(frames, mask, **kw):
        return sp.subject_parts(frames, mask, (HAIR,), seg_held, None, mean=MEAN, std=STD, size=SIZE,
                                crop_margin=MARGIN, subject_margin=2, **kw)

    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    frames[0, 44:46, 68:74] = magenta                 # the lips
    frames[0, 60:66, 82:88] = yellow                  # a hand at the body's right edge
    frames[0, 52:58, 82:88] = cyan                    # held: just above the hand, inside the mask
    frames[0, 43:47, 74:82] = cyan                    # held: beside the lips
    frames[0, 64:70, 56:62] = cyan                    # a bag at the hip, far from both
    frames[0, 60:66, 88:94] = cyan                    # beside the hand but outside the subject's mask
    want = torch.zeros(H, W)
    want[52:58, 82:88] = 1.0
    want[43:47, 74:82] = 1.0
    got = found(frames, mask, held_near=8, held_smallest=24).holds[0]
    if not torch.equal(got, want):
        problems.append(f"held: with a reach of 8 px it took {int(got.sum())} px, {int((got * want).sum())} of them "
                        f"on the two held things ({int(want.sum())} px); the bag has {int(got[64:70, 56:62].sum())} "
                        f"and outside the mask {int(got[60:66, 88:94].sum())}")
    far = found(frames, mask, held_near=40, held_smallest=24).holds[0]
    if int(far[64:70, 56:62].sum()) != 36:
        problems.append("held: the control did not work: with a long reach the bag at the hip is still not taken, "
                        "so the reach is not what kept it out")
    if int(far[60:66, 88:94].sum()) != 0:
        problems.append("held: a thing outside the subject's mask was taken at a long reach")
    # nothing to anchor on: the same picture without the hand and the lips holds nothing
    bare, bare_mask = scene()
    person(bare[0], bare_mask[0], LEFT)
    bare[0, 52:58, 82:88] = cyan
    if float(found(bare, bare_mask, held_near=40, held_smallest=1).holds.sum()) != 0.0:
        problems.append("held: something was taken on a frame with no lips and no hand found")
    # a speck is dropped, and the size is what drops it
    speck, speck_mask = scene()
    person(speck[0], speck_mask[0], LEFT)
    speck[0, 60:66, 82:88] = yellow
    speck[0, 57:59, 84:86] = cyan
    if float(found(speck, speck_mask, held_near=8, held_smallest=24).holds.sum()) != 0.0:
        problems.append("held: a 4 px speck beside the hand was taken at a smallest size of 24")
    if float(found(speck, speck_mask, held_near=8, held_smallest=1).holds.sum()) != 4.0:
        problems.append("held: the control did not work: the speck is not taken at a smallest size of 1 either")
    text = sp.report(found(frames, mask, held_near=8, held_smallest=24), (HAIR,), MARGIN, 2, 0, True, SIZE, None)
    if "held: " not in text or "on 1 of 1 frames" not in text:
        problems.append(f"held: the report does not say on how many frames something is held: {text!r}")


def check_classes(problems):
    """Every class on the subject is handed out, and it comes back from a mask exactly."""
    frames, mask = scene()
    person(frames[0], mask[0], LEFT)
    got = run(frames, mask, (HAIR,), subject_margin=2)
    if got.classes is None or tuple(got.classes.shape) != tuple(mask.shape) or got.classes.dtype != torch.uint8:
        problems.append("the part pass returns no per-pixel class map of the frames' size")
        return
    wide = sp.grow(mask, 2)[0] > 0.5
    if bool(got.classes[0][~wide].any()):
        problems.append("the class map names a class outside the subject's mask widened by subject_margin")
    if not torch.equal(got.classes[0] == HAIR, got.parts[0] > 0.5):
        problems.append("the pixels the class map calls hair are not the pixels `parts` took for hair")
    if len(torch.unique(got.classes[0])) < 3:
        problems.append("the class map holds fewer than three classes on a painted person: it is the chosen part, "
                        "not every class")
    # the control for 'every class': a class that was not ticked is in the map and not in `parts`
    other = [int(c) for c in torch.unique(got.classes[0]).tolist() if c not in (0, HAIR)]
    if other and bool(((got.classes[0] == other[0]) & (got.parts[0] > 0.5)).any()):
        problems.append("a class that was not chosen is in `parts`")
    # the mask form: every index back exactly, through a saver that rounds and one that truncates
    every = torch.arange(len(sp.CLASS_NAMES), dtype=torch.uint8).view(1, 1, -1)
    carried = sp.class_mask(every)
    if float(carried.max()) > 1.0 or float(carried[0, 0, 0]) != 0.0:
        problems.append("the class mask leaves 0..1, or class 0 is not an empty mask")
    for name, saved in (("rounds", (carried * 255.0).round()), ("truncates", (carried * 255.0).floor())):
        if not torch.equal(sp.class_indices(saved / 255.0), every):
            problems.append(f"a class index does not come back from a saver that {name}")
    bare = (every.to(torch.float32) / 255.0 * 255.0).floor().to(torch.uint8)
    if torch.equal(bare, every):
        print("note  the bare quotient index/255 also truncates back on this machine: CLASS_NUDGE is not shown needed here")
    outputs = [o.display_name for o in sp.MiniMaxH3SubjectParts.define_schema().outputs]
    if outputs[-1] != "classes" or outputs[:5] != ["parts", "matte", "preview", "report", "held"]:
        problems.append(f"the node's outputs are {outputs}: `classes` must be appended, the five before it unmoved")


def _graded(check):
    def run():
        problems: list[str] = []
        check(problems)
        assert not problems, "; ".join(problems)
    return run


def main() -> int:
    for name, check in (
            ("the crop holds the subject and has the model's shape", check_boxes),
            ("a map made on the crop lands where the pixels came from", check_round_trip),
            ("a neighbour inside the crop gives nothing", check_neighbour),
            ("the menu names the classes it says", check_menu),
            ("a missed frame takes its neighbour's part, moved with the subject", check_hold),
            ("a frame in doubt is held as a missed one is", check_hold_doubted),
            ("the matte is soft only at the person's edge", check_matte),
            ("a mask that is not the frames' is refused by name", check_refusals),
            ("the result does not depend on the batch size", check_batch),
            ("the loader lists a folder by its architecture", check_loader),
            ("the preview and the report", check_shown),
            ("the report says how much of the subject the parts cover", check_coverage),
            ("the models are shown the subject alone", check_alone),
            ("what the subject holds is found by where it is, with no name", check_held),
            ("every class on the subject is handed out and comes back exactly", check_classes)):
        case(name, _graded(check))
    return finish()


if __name__ == "__main__":
    sys.exit(main())
