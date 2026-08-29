"""Render the same market scene as three two-shot arms, holding everything but the prompt.

The question is whether visible degradation tracks the number of cuts (and the
large frame-to-frame deltas a cut produces) rather than clip length. Each arm
keeps the full 362-frame canvas and drops one of the three shots, so the two
survivors stretch to fill it and the second one starts at 00:07.500.

Everything else is byte-identical to the tail6 arm: seed, canvas, length,
partition, sigmas, Sol settings, models. Only node 5's prompt differs.
"""
import argparse, json, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workflows import h3_config as cfg

SRV = "http://127.0.0.1:8188"
BASE = Path(__file__).resolve().parents[1] / "bench" / "_shot_ablation_base.json"

STYLE = "Live-action, cinematic, handheld, shallow depth of field. "

# The three shots, each written so it can open a clip (carrying the style
# preamble and introducing its own speakers) or follow a cut.
S1_OPEN = (
    STYLE + "A medium-wide shot frames a covered market aisle in late morning, crates of "
    "citrus stacked along a wooden stall front, dust turning slowly in a shaft of light from "
    "the roof vents and a hanging brass scale swaying gently at the edge of frame. A "
    "stallholder in her fifties with a warm, gravelly alto (S1) sets a crate on the counter, "
    "wipes both palms down her apron, and says: <d>[English] Last of the good ones. After this "
    "it is all imports.</d> Her lips close and she pushes the crate forward with the heel of "
    "her hand. The camera trucks left as a young porter, a lean man in his twenties with a "
    "quick, bright tenor (S2), steps into frame behind her shoulder."
)
S2_OPEN = (
    STYLE + "A close shot over the shoulder of a young porter in a covered market aisle, a lean "
    "man in his twenties with a quick, bright tenor (S2), as he squats, takes a crate of citrus "
    "at its corners, and lifts it to his chest. The porter (S2) says: <d>[English] Then I will "
    "take two.</d> His lips close and he shifts the weight onto his hip, and coins clatter one "
    "after another into a metal tin on the counter."
)
S2_CUT = (
    "the shot cuts to a close shot over the porter's shoulder as he squats, takes the crate at "
    "its corners, and lifts it to his chest. The porter (S2) answers: <d>[English] Then I will "
    "take two.</d> His lips close and he shifts the weight onto his hip, and coins clatter one "
    "after another into a metal tin on the counter."
)
S3_AFTER_S2 = (
    "the camera holds a static shot, wide on the aisle, as he carries both crates away between "
    "the stalls, shoppers stepping aside around him, while a stallholder in her fifties turns "
    "back and stacks fruit into a pyramid with both hands, her lips closed."
)
S3_AFTER_S1 = (
    "the camera holds a static shot, wide on the aisle, as the porter carries the crate away "
    "between the stalls, shoppers stepping aside around him, while the stallholder turns back "
    "and stacks fruit into a pyramid with both hands, her lips closed."
)

TAIL = (
    "\n\noverall_soundscape:\nLoose crowd murmur under a high roof, wooden crates knocking hollow "
    "as they stack, coins dropping one by one into a metal tin, boot steps on swept concrete, and "
    "the dry crackle of paper bags shaken open.\n\nnon_diegetic_music:\nN/A"
)

BASE_LENGTH = 362  # frames, 15.083 s -- the length every arm used at first

def cut_at(length: int) -> str:
    """The second shot starts at the canvas midpoint, rounded down to a half second.

    Shot DURATION is confounded with shot COUNT in the original design: dropping
    a shot from the 362-frame canvas both lowers total demanded content and
    stretches the survivors from ~5 s to ~7.5 s each. Rendering the same two
    shots on a 241-frame (10.04 s) canvas restores ~5 s per shot while keeping
    the count at two, which separates them:

        good at 241 -> total demanded content is the variable
        bad  at 241 -> per-shot duration is, and 5 s is too short either way
    """
    mid = (length / 24.0) / 2.0
    half = int(mid * 2) / 2.0
    return f"00:{int(half // 60):02d}:{half % 60:06.3f}"[3:]

ARMS = {
    "shots23": (S2_OPEN, S3_AFTER_S2),
    "shots13": (S1_OPEN, S3_AFTER_S1),
    "shots12": (S1_OPEN, S2_CUT),
}


def prompt_for(arm: str, length: int = BASE_LENGTH) -> str:
    first, second = ARMS[arm]
    return (
        "integrated_multimodal_description:\n"
        f"[Shot 1] {first}\n"
        f"[Shot 2] At {cut_at(length)}, {second}" + TAIL
    )


def graph_for(arm: str, ref: bool = False, length: int = BASE_LENGTH) -> dict:
    """One arm. `ref=True` builds its 32-evaluation counterpart.

    The reference is NOT "PDD bypassed". It keeps the LoRA and runs block
    width 1, which is the trajectory the coarse partitions are approximations
    of -- at width 1 each fused head spans a single grid point and is exact.
    Construction follows `grade_pdd_partitions.py::build`: drive SIGMAS from
    the PDD node itself rather than the ManualSigmas override.

    It exists to make the detail axis controlled. Absolute detail on a render
    cannot separate a low-detail scene from destroyed detail; detail(arm) over
    detail(ref) holds scene content exactly and so measures destruction alone.
    """
    g = json.loads(BASE.read_text())
    g["5"]["inputs"]["prompt"] = prompt_for(arm, length)
    g["27"]["inputs"]["length"] = length
    suffix = f"{arm}_ref32" if ref else arm
    if length != BASE_LENGTH:
        suffix = f"{arm}_{length}f"
    g["13"]["inputs"]["filename_prefix"] = f"Video/shotablation_{suffix}"
    if ref:
        g["18"]["inputs"]["steps"] = 32
        g["10"]["inputs"]["sigmas"] = ["18", 1]
        g.pop("60", None)
    return g


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{SRV}{path}", json.dumps(payload).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r) if r.length != 0 else {}


def wait(pid: str) -> dict:
    while True:
        with urllib.request.urlopen(f"{SRV}/history/{pid}") as r:
            h = json.load(r)
        if pid in h:
            return h[pid]
        time.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--length", type=int, default=BASE_LENGTH,
                    help="frames; 241 (10.04 s) restores ~5 s per shot at two shots")
    ap.add_argument("--ref", action="store_true",
                    help="render the 32-evaluation counterpart of each named arm "
                         "instead, giving the controlled detail ratio")
    ap.add_argument("--out", default=None, help="write graphs here instead of posting")
    a = ap.parse_args()

    if a.out:
        d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
        for arm in a.arms:
            (d / f"{arm}.json").write_text(json.dumps(graph_for(arm, a.ref, a.length), indent=1))
            print(f"{arm}: {d / f'{arm}.json'}")
        sys.exit(0)

    for arm in a.arms:
        g = graph_for(arm, a.ref, a.length)
        if a.dry_run:
            print(f"[{arm}] {len(prompt_for(arm, a.length))} chars"); continue
        pid = post("/prompt", {"prompt": g})["prompt_id"]
        print(f"{arm}: {pid}", flush=True)
        rec = wait(pid)
        st = rec["status"]["status_str"]
        vids = [f["filename"] for o in rec["outputs"].values()
                for f in o.get("gifs", []) + o.get("videos", [])]
        print(f"  {st}: {vids}", flush=True)
        # Full cache clear between arms. Note that free_memory alone already
        # unloads every model: the server only sets the unload flag when it is
        # truthy, so the executor's `flags.get("unload_models", free_memory)`
        # falls through to free_memory. Passing unload_models=False does not
        # opt out. Both flags are consumed between prompts, never mid-render,
        # so this is safe for anything else sharing the queue -- it just costs
        # that job a reload too.
        post("/free", {"unload_models": True, "free_memory": True})
