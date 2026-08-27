#!/usr/bin/env python3
"""What a single-frame edit can hold: reference count, sizing, and canvas.

Every render on the single-frame path so far used ONE reference image, of one
subject, at one canvas. That is enough to prove the path works and nothing
about where it stops. This sweeps the three axes that decide the shipped
defaults, at one frame each, which is seconds rather than minutes.

**The axes, and why each one is here.**

`count` -- core caps reference images at 9. The arithmetic says the card is not
what stops you: nine references through `MiniMaxH3ReferenceFit` at 2048 is
36,864 reference rows and a 42,008-row sequence, against 82,686 for the
124-frame video graph that already fits on this 4090. So the limit is
qualitative -- whether identities stay separate, and whether a prompt naming
`<Picture 7>` is still followed -- and only looking settles it.

`sizing` -- `ref_image_size="max"` is a NO-OP on the shipped graph, which is
worth knowing before reading any result here. Our fit node has already taken
the reference to 2048 short edge and core's `max` is
`min(1.0, 2048 / short_edge)` = 1.0. What actually sets the cost is the fit
node's `allow_upscale`, at 4.9x the reference rows (4,096 against 841 for a
1024x1024 source). `match` undoes the fit node entirely by resizing to the
generation's pixel area. Three settings, three very different costs, and no
measured quality difference anywhere in this repo.

`canvas` -- at one frame the video segment is 9% of the sequence, so the canvas
is nearly free here where it is the dominant lever on a video. That makes the
out-of-family 1024x1536 the community uses cheap to evaluate honestly.

**Reference variety is deliberate and comes from the input `h3_refs/` folder**,
which spans 662x1177 to 2816x1536 and aspects 0.56 to 1.83 across faces,
subjects, scenes, styles and a product. Rendering one subject repeatedly would
measure that subject. The 662x1177 performer is the interesting size case: at
`allow_upscale=True` it is enlarged 3.1x to reach 2048, buying rows and no
detail, which is the exact worry `docs/open_experiments.md` #1 parks on.

Needs a live ComfyUI **with the single-frame shim active** -- if it was started
without `H3_EXPLORATIONS_SINGLE_FRAME=1` every arm renders 5 frames instead of
1, and this script refuses rather than producing a table of the wrong thing.

    python bench/bench_image_edit_refs.py --list
    python bench/bench_image_edit_refs.py --arms count
    python bench/bench_image_edit_refs.py            # everything

Writes a markdown table to stdout and leaves the images in ComfyUI's output
under `Image/bench_refs/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402


REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "workflows" / "image" / "h3_image_edit_api.json"

# Node ids in the shipped API graph. Named rather than repeated as literals so
# a generator change that renumbers them fails loudly here instead of quietly
# editing the wrong node.
N_REF_NODE = "5"        # MiniMaxH3ReferenceToVideo
N_RESOLUTION = "27"     # MiniMaxH3Resolution
N_SAVE = "13"           # SaveImage
N_FIRST_LOAD = "15"     # LoadImage for reference 1
N_FIRST_FIT = "24"      # MiniMaxH3ReferenceFit for reference 1

REFS = "h3_refs"        # subfolder under ComfyUI's input directory

# The library, by role. Sizes are in the filenames and span 662x1177 to
# 2816x1536, which is the point: a sweep on one subject at one size measures
# that subject at that size.
FACE_MAN = f"{REFS}/face_young_man_glasses_1024x1024.png"
FACE_WOMAN = f"{REFS}/face_freckled_woman_redhair_1024x1024.png"
FACE_ELDER = f"{REFS}/face_elderly_man_suit_1024x1024.png"
FACE_GRAIN = f"{REFS}/face_elderly_man_bw_grain_1280x926.png"
SUBJ_PANDA = f"{REFS}/subject_felted_red_panda_1760x990.png"
SUBJ_CAT = f"{REFS}/subject_cat_sushi_costume_1408x768.jpg"
SUBJ_PERFORMER = f"{REFS}/subject_performer_stage_662x1177.png"   # smallest
SUBJ_WOMAN_LOFT = f"{REFS}/subject_woman_loft_portrait_1536x2752.jpeg"
SCENE_ALPINE = f"{REFS}/scene_alpine_lake_meadow_1024x1024.png"
SCENE_CORRIDOR = f"{REFS}/scene_officers_corridor_1376x768.jpeg"
SCENE_LOFT = f"{REFS}/scene_loft_couch_duo_2752x1536.png"          # largest
SCENE_DUEL = f"{REFS}/scene_period_duel_riverbank_2816x1536.jpg"
STYLE_PENCIL = f"{REFS}/style_pencil_cottage_1024x1024.png"
STYLE_IMPASTO = f"{REFS}/style_impasto_lighthouse_storm_1024x1024.png"
PRODUCT_JERSEY = f"{REFS}/product_soccer_jersey_1600x1600.png"

PORTRAIT = (768, 1152)      # shipped, in the trained family
LANDSCAPE = (1344, 768)     # the repo default canvas
SQUARE = (768, 768)         # cheapest in family
COMMUNITY = (1024, 1536)    # out of family, 52% over the area cap


def arm(name, group, refs, prompt, canvas=PORTRAIT, sizing="max",
        upscale=True, note="") -> dict[str, Any]:
    return dict(name=name, group=group, refs=refs, prompt=prompt,
                canvas=canvas, sizing=sizing, upscale=upscale, note=note)


# Each prompt names EXACTLY the labels its arm wires, 1-based in socket order,
# because the tokenizer derives them from the sockets and not from the prompt.
ARMS = [
    # -- count: does composition survive more subjects ----------------------
    arm("count1-face", "count", [FACE_MAN],
        "Task: reference-guided single-image edit. Re-photograph the man from "
        "<Picture 1> in three-quarter view, turned about 45 degrees to his "
        "left, still looking into the lens. Keep his face, glasses, hair, "
        "clothing, the background and the lighting exactly as they are. One "
        "realistic portrait photograph.",
        note="baseline: one subject, no composition"),

    arm("count2-face-scene", "count", [FACE_MAN, SCENE_ALPINE],
        "Task: reference-guided single-image edit. Place the man from "
        "<Picture 1> into the landscape from <Picture 2>, standing in the "
        "meadow in the middle distance and looking toward the camera. Keep his "
        "face, glasses, hair and clothing exactly as in <Picture 1>. Take the "
        "entire setting, weather and light from <Picture 2>, and match his "
        "lighting to it. One realistic photograph.",
        note="identity + environment, the basic composite"),

    arm("count3-face-outfit-scene", "count",
        [FACE_MAN, PRODUCT_JERSEY, SCENE_CORRIDOR],
        "Task: reference-guided single-image edit. The man from <Picture 1> "
        "wears the jersey from <Picture 2> and stands in the corridor from "
        "<Picture 3>, facing the camera. His face, glasses and hair are exactly "
        "those of <Picture 1>. The garment's colour, pattern and crest are "
        "exactly those of <Picture 2>, resized to fit him. The corridor, its "
        "architecture and its light come from <Picture 3>. One realistic "
        "photograph.",
        note="three roles: identity, wardrobe, place"),

    arm("count4-two-people", "count",
        [FACE_MAN, FACE_WOMAN, SCENE_LOFT, PRODUCT_JERSEY],
        "Task: reference-guided single-image edit. The man from <Picture 1> "
        "and the woman from <Picture 2> sit together on the couch in the loft "
        "from <Picture 3>, turned toward each other in conversation. He wears "
        "the jersey from <Picture 4>. Each keeps their own exact face, hair "
        "and build, with no blending between them. The room, furniture and "
        "light come from <Picture 3>. One realistic photograph, both faces "
        "clearly visible and separate.",
        note="TWO identities in one frame -- the first real test of separation"),

    arm("count6-three-people", "count",
        [FACE_MAN, FACE_WOMAN, FACE_ELDER, SCENE_LOFT, PRODUCT_JERSEY,
         STYLE_PENCIL],
        "Task: reference-guided single-image edit. The man from <Picture 1>, "
        "the woman from <Picture 2> and the older man from <Picture 3> are "
        "together in the loft from <Picture 4>, mid-conversation. The younger "
        "man wears the jersey from <Picture 5>. Each person keeps their own "
        "exact face, hair and build, with no blending. Render the whole image "
        "in the drawing style of <Picture 6>: graphite pencil on paper, "
        "hatching for shadow, no colour. One coherent drawing, all three faces "
        "distinct.",
        note="three identities plus a style -- six labels"),

    arm("count9-everything", "count",
        [FACE_MAN, FACE_WOMAN, FACE_ELDER, SUBJ_PANDA, SUBJ_CAT, SCENE_LOFT,
         PRODUCT_JERSEY, STYLE_PENCIL, SCENE_ALPINE],
        "Task: reference-guided single-image edit. In the loft from "
        "<Picture 6>, the man from <Picture 1>, the woman from <Picture 2> and "
        "the older man from <Picture 3> sit together. The felted red panda from "
        "<Picture 4> sits on the couch arm and the costumed cat from "
        "<Picture 5> sits on the floor. The younger man wears the jersey from "
        "<Picture 7>. Through the window behind them is the landscape from "
        "<Picture 9>. Render everything in the pencil drawing style of "
        "<Picture 8>. Each face and each animal keeps its own exact "
        "appearance, with no blending. One coherent drawing.",
        note="core's cap: 9 references, deliberately past the plausible limit"),

    # -- sizing: what the fit node and ref_image_size actually buy ----------
    arm("size-max-upscale", "sizing", [FACE_MAN, SCENE_ALPINE],
        "Task: reference-guided single-image edit. Place the man from "
        "<Picture 1> into the landscape from <Picture 2>, standing in the "
        "meadow and looking toward the camera. Keep his face, glasses, hair "
        "and clothing exactly as in <Picture 1>; take the setting and light "
        "from <Picture 2>. One realistic photograph.",
        sizing="max", upscale=True,
        note="SHIPPED. fit upscales to 2048, core's max is then a no-op"),

    arm("size-max-nofit", "sizing", [FACE_MAN, SCENE_ALPINE],
        "Task: reference-guided single-image edit. Place the man from "
        "<Picture 1> into the landscape from <Picture 2>, standing in the "
        "meadow and looking toward the camera. Keep his face, glasses, hair "
        "and clothing exactly as in <Picture 1>; take the setting and light "
        "from <Picture 2>. One realistic photograph.",
        sizing="max", upscale=False,
        note="references at native size -- 4x cheaper, prices the fit node"),

    arm("size-match", "sizing", [FACE_MAN, SCENE_ALPINE],
        "Task: reference-guided single-image edit. Place the man from "
        "<Picture 1> into the landscape from <Picture 2>, standing in the "
        "meadow and looking toward the camera. Keep his face, glasses, hair "
        "and clothing exactly as in <Picture 1>; take the setting and light "
        "from <Picture 2>. One realistic photograph.",
        sizing="match", upscale=True,
        note="core resizes to the generation's pixel area, undoing the fit node"),

    arm("size-small-source", "sizing", [SUBJ_PERFORMER, SCENE_DUEL],
        "Task: reference-guided single-image edit. Place the performer from "
        "<Picture 1> on the riverbank from <Picture 2>, standing in the open "
        "and facing the camera. Keep the performer's face, costume and pose "
        "from <Picture 1>; take the landscape, weather and light from "
        "<Picture 2>. One realistic photograph.",
        sizing="max", upscale=True,
        note="662x1177 source upscaled 3.1x to reach 2048: rows without detail"),

    # -- canvas: nearly free at one frame, so ask honestly ------------------
    arm("canvas-portrait", "canvas", [FACE_WOMAN, STYLE_IMPASTO],
        "Task: reference-guided single-image edit. Render the woman from "
        "<Picture 1> as a painting in the style of <Picture 2>: thick impasto "
        "oil, visible brush loading, palette-knife texture. Her face, "
        "freckles, red hair and expression stay recognisably hers. One "
        "painted portrait.",
        canvas=PORTRAIT, note="768x1152, in family, shipped"),

    arm("canvas-community", "canvas", [FACE_WOMAN, STYLE_IMPASTO],
        "Task: reference-guided single-image edit. Render the woman from "
        "<Picture 1> as a painting in the style of <Picture 2>: thick impasto "
        "oil, visible brush loading, palette-knife texture. Her face, "
        "freckles, red hair and expression stay recognisably hers. One "
        "painted portrait.",
        canvas=COMMUNITY, note="1024x1536, 52% over the area cap, out of family"),

    arm("canvas-landscape", "canvas", [SUBJ_WOMAN_LOFT, SCENE_DUEL],
        "Task: reference-guided single-image edit. Place the woman from "
        "<Picture 1> on the riverbank from <Picture 2>, walking along the "
        "water's edge and glancing toward the camera. Keep her face, hair and "
        "clothing from <Picture 1>; take the landscape, weather and light from "
        "<Picture 2>. One realistic wide photograph.",
        canvas=LANDSCAPE,
        note="1344x768 from a 1536x2752 portrait source: aspect disagreement"),

    arm("canvas-square", "canvas", [SUBJ_PANDA, STYLE_PENCIL],
        "Task: reference-guided single-image edit. Draw the felted red panda "
        "from <Picture 1> in the graphite pencil style of <Picture 2>: "
        "hatching for shadow, paper grain, no colour. Its pose, proportions "
        "and felted texture stay those of <Picture 1>. One drawing.",
        canvas=SQUARE, note="768x768, cheapest canvas in the family"),
]


def rows_for(width, height, refs, sizing, upscale):
    """Projected reference and video rows, so cost is known before rendering."""
    import math

    from PIL import Image  # only needed for the projection

    inp = _paths.comfy_input()
    if inp is None:
        raise SystemExit(_paths.describe("ComfyUI input", "H3_COMFY_INPUT"))
    ref_rows = 0
    for name in refs:
        try:
            w, h = Image.open(inp / name).size
        except Exception:
            continue
        if upscale:
            s = 2048 / min(w, h)
            w, h = round(w * s), round(h * s)
        if sizing == "max":
            s = min(1.0, 2048 / min(w, h))
        else:
            s = min(1.0, math.sqrt((width * height) / (w * h)))
        tw = max(32, round(w * s / 32) * 32)
        th = max(32, round(h * s / 32) * 32)
        ref_rows += (tw // 32) * (th // 32)
    return ref_rows, (width // 32) * (height // 32)


def build(template, a):
    g = json.loads(json.dumps(template))
    width, height = a["canvas"]

    g[N_REF_NODE]["inputs"]["prompt"] = a["prompt"]
    g[N_REF_NODE]["inputs"]["ref_image_size"] = a["sizing"]
    # Canvas as literals rather than through the Resolution node: this sweep
    # deliberately visits an out-of-family canvas, and the node's dropdown only
    # offers the 95 in-family ones. Length stays wired to Resolution so the
    # single-frame guard still runs.
    g[N_REF_NODE]["inputs"]["width"] = width
    g[N_REF_NODE]["inputs"]["height"] = height
    g[N_RESOLUTION]["inputs"] = {"shape": "custom", "shape.width": width,
                                 "shape.height": height, "length": 1}

    # Reference chain: LoadImage -> MiniMaxH3ReferenceFit -> ref_image_N.
    # The template ships one; the rest are cloned from it so any future change
    # to the fit node's settings propagates to every arm.
    load_tpl = g[N_FIRST_LOAD]
    fit_tpl = g[N_FIRST_FIT]
    for key in list(g[N_REF_NODE]["inputs"]):
        if key.startswith("ref_images."):
            del g[N_REF_NODE]["inputs"][key]
    for i, name in enumerate(a["refs"]):
        lid, fid = f"{900 + i * 2}", f"{901 + i * 2}"
        g[lid] = {"class_type": load_tpl["class_type"], "inputs": {"image": name}}
        g[fid] = {"class_type": fit_tpl["class_type"],
                  "inputs": dict(fit_tpl["inputs"], image=[lid, 0],
                                 allow_upscale=a["upscale"])}
        g[N_REF_NODE]["inputs"][f"ref_images.ref_image_{i}"] = [fid, 0]
    for old in (N_FIRST_LOAD, N_FIRST_FIT):
        g.pop(old, None)

    g[N_SAVE]["inputs"]["filename_prefix"] = f"Image/bench_refs/{a['name']}"
    return g


def submit(base, graph, timeout=900):
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{base}/prompt", body,
                                 {"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
    except urllib.error.HTTPError as exc:
        # Same shape as the success path, so the caller never has to ask which
        # kind of second element it got.
        return "rejected", [exc.read().decode()[:400]], 0.0
    t0 = time.time()
    while time.time() - t0 < timeout:
        q = json.load(urllib.request.urlopen(f"{base}/queue", timeout=10))
        if not q["queue_running"] and not q["queue_pending"]:
            break
        time.sleep(2)
    hist = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=10))
    entry = hist.get(pid, {})
    status = entry.get("status", {}).get("status_str", "missing")
    images = [i["filename"] for o in entry.get("outputs", {}).values()
              for i in o.get("images", [])]
    return status, images, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1:8188")
    ap.add_argument("--arms", help="comma-separated groups: count, sizing, canvas")
    ap.add_argument("--only", help="comma-separated arm names")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    base = f"http://{args.host}"

    wanted = ARMS
    if args.arms:
        groups = {g.strip() for g in args.arms.split(",")}
        wanted = [a for a in wanted if a["group"] in groups]
    if args.only:
        names = {n.strip() for n in args.only.split(",")}
        wanted = [a for a in wanted if a["name"] in names]
    if not wanted:
        print("no arms selected")
        return 2

    if args.list:
        print(f"{'arm':<26}{'refs':>5}{'canvas':>12}{'sizing':>8}{'fit':>7}"
              f"{'ref rows':>10}{'video':>7}   note")
        for a in wanted:
            w, h = a["canvas"]
            rr, vr = rows_for(w, h, a["refs"], a["sizing"], a["upscale"])
            print(f"{a['name']:<26}{len(a['refs']):>5}{f'{w}x{h}':>12}"
                  f"{a['sizing']:>8}{str(a['upscale']):>7}{rr:>10,}{vr:>7,}"
                  f"   {a['note']}")
        return 0

    template = json.loads(TEMPLATE.read_text())

    # Refuse rather than measure the wrong thing: without the shim every arm
    # renders 5 frames and the whole table is about a path nobody asked for.
    try:
        info = json.load(urllib.request.urlopen(
            f"{base}/object_info/MiniMaxH3ReferenceToVideo", timeout=10))
        floor = info["MiniMaxH3ReferenceToVideo"]["input"]["required"]["length"][1]["min"]
    except Exception as exc:
        print(f"no ComfyUI at {base}: {exc}")
        return 2
    if floor != 1:
        print(f"the server reports a length floor of {floor}, so length=1 is "
              f"not available and every arm here would render 5 frames.\n"
              f"Restart ComfyUI with H3_EXPLORATIONS_SINGLE_FRAME=1.")
        return 2

    print(f"| arm | refs | canvas | sizing | fit | ref rows | seq | secs | status |")
    print(f"|---|---:|---|---|---|---:|---:|---:|---|")
    for a in wanted:
        w, h = a["canvas"]
        rr, vr = rows_for(w, h, a["refs"], a["sizing"], a["upscale"])
        graph = build(template, a)
        graph["8"]["inputs"]["steps"] = args.steps
        status, images, secs = submit(base, graph)
        seq = rr + vr
        shown = status if status != "success" else f"ok ({len(images)} img)"
        print(f"| {a['name']} | {len(a['refs'])} | {w}x{h} | {a['sizing']} | "
              f"{a['upscale']} | {rr:,} | ~{seq:,} | {secs:.0f} | {shown} |",
              flush=True)
        if status != "success":
            print(f"|  |  |  |  |  |  |  |  | {str(images)[:160]} |", flush=True)
    print("\nImages under ComfyUI's output in Image/bench_refs/. The sequence "
          "column is projected; the authoritative split is the [h3] preflight "
          "line in the server log for each render.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
