#!/usr/bin/env python3
"""Generate h3-mutant-distill's example workflows, in the frontend's own UI format.

The published repo (`standalone/h3_mutant_distill/`) ships one workflow
per recipe in `example_workflows/`. They are UI-format files, not API
prompts, because only a UI file carries each loader's `properties.models`
(name, URL, folder), which is what makes ComfyUI's frontend offer to download
a missing model. So a stranger needs the node pack and nothing typed by hand.

Hand-serializing UI JSON is where these go wrong: widget order, seed
controls, and dynamic combos (`SaveVideo`, `BlockSparseAttention`) all have
to match the frontend's serializer. So the frontend does it. This builds each
recipe as an API graph from `h3_config`, loads it into the live ComfyUI
frontend in headless Chrome (`app.loadApiJson`), serializes the result
(`app.graph.serialize`), adds the model URLs and a note, then loads THAT file
back and asserts `app.graphToPrompt()` gives the API graph it came from. Only
then is it written.

    <comfy venv python> bench/build_mutant_examples.py            # write
    <comfy venv python> bench/build_mutant_examples.py --check    # round-trip the committed files

Needs the ComfyUI server at 127.0.0.1:8188 with `H3ExactLoRA` loaded
(`custom_nodes/h3-mutant-distill` linking to the standalone folder), and
Google Chrome. No GPU work: nothing is queued.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "workflows"))

import h3_config as C  # noqa: E402
from build_workflows import LONG_T2V_PROMPT  # noqa: E402

OUT = REPO / "standalone" / "h3_mutant_distill" / "example_workflows"
SERVER = "http://127.0.0.1:8188"
CHROME = "google-chrome"

#: Where each file is published. The PDD8 files are on the package's HF
#: repo, byte-identical to
#: `h3_config.PDD_FL2VA_LORA` / `PDD_REF2VA_LORA` and to the owner's sidecar
#: repo (sha256 e225a89f... and f1e552ad..., checked 2026-09-28).
HF_PACKAGE = "fbjr/h3-mutant-distill"
COMFY_ORG = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
MODELS = {
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors":
        (f"{COMFY_ORG}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "diffusion_models"),
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors":
        (f"{COMFY_ORG}/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders"),
    "minimax_h3_video_vae_int8_convrot.safetensors":
        (f"{COMFY_ORG}/vae/minimax_h3_video_vae_int8_convrot.safetensors", "vae"),
    "minimax_h3_audio_vae_fp32.safetensors":
        (f"{COMFY_ORG}/vae/minimax_h3_audio_vae_fp32.safetensors", "vae"),
    "minimax_h3_fl2va_pdd_8step_comfy.safetensors":
        (f"https://huggingface.co/{HF_PACKAGE}/resolve/main/"
         "minimax_h3_fl2va_pdd_8step_comfy.safetensors", "loras"),
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors":
        (f"{COMFY_ORG}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models"),
    "minimax_h3_ref2va_pdd_8step_comfy.safetensors":
        (f"https://huggingface.co/{HF_PACKAGE}/resolve/main/"
         "minimax_h3_ref2va_pdd_8step_comfy.safetensors", "loras"),
}
FILE_WIDGETS = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name",
                "VAELoader": "vae_name", "H3ExactLoRA": "lora_name"}

PDD = Path(C.PDD_FL2VA_LORA).name
PDD_R2V = Path(C.PDD_REF2VA_LORA).name

#: The i2v and ref2va prompts are read from the pack graphs these recipes were
#: parity-checked against, so the example cannot drift from them.
I2V_GRAPH = "workflows/distill_experiments/h3_first_frame_to_video_pdd_savelat_api.json"
R2V_GRAPH = "workflows/h3_image_ref_plus_text_to_video_pdd_api.json"
#: Placeholder image names: the user loads their own.
I2V_IMAGE = "first_frame.png"
R2V_IMAGES = ("reference_person.png", "reference_place.png")


def pack_prompt(rel: str, class_type: str) -> str:
    g = json.loads((REPO / rel).read_text())
    return next(n["inputs"]["prompt"] for n in g.values() if n["class_type"] == class_type)
SEED = 730451892
WIDTH, HEIGHT, LENGTH, FPS = 1344, 768, 345, 24.0

HEADER = ("**Experimental. YMMV.** Judged by eye, on one seed, by one person, on an "
          "RTX 4090 at 1344x768 and 345 frames. Those renders also ran Sol-Attn "
          "(sparse attention from ComfyUI-h3-explorations), which this graph does not.")


def base(unet: str, task: str = "t2v") -> dict:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                         "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "5": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": LONG_T2V_PROMPT,
                         "width": WIDTH, "height": HEIGHT, "length": LENGTH}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    if task == "i2v":
        # The canvas comes from the image (`H3KeyframeCanvas`, the release's rule), so
        # core's node never stretches it: a stretched square keyframe drifted on
        # 2026-10-01 (bench/results/2026-10-01_mutant_parity_flashgen_i2v_r2v_finish.md).
        g["15"] = {"class_type": "LoadImage", "inputs": {"image": I2V_IMAGE}}
        g["14"] = {"class_type": "H3KeyframeCanvas", "inputs": {"first_frame": ["15", 0]}}
        g["5"]["inputs"].update(prompt=pack_prompt(I2V_GRAPH, "MiniMaxH3Conditioning"),
                                width=["14", 0], height=["14", 1], first_frame=["14", 2])
    elif task == "r2v":
        g["15"] = {"class_type": "LoadImage", "inputs": {"image": R2V_IMAGES[0]}}
        g["16"] = {"class_type": "LoadImage", "inputs": {"image": R2V_IMAGES[1]}}
        g["5"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                             "prompt": pack_prompt(R2V_GRAPH, "MiniMaxH3ReferenceConditioning"),
                             "width": WIDTH, "height": HEIGHT, "length": LENGTH,
                             "ref_image_size": "max",
                             "ref_images.ref_image_0": ["15", 0],
                             "ref_images.ref_image_1": ["16", 0]}}
    return g


def sampler(g: dict, nid: str, model: list, sigmas: str, latent: list, noise=("6", 0)) -> None:
    """A guider, the sigmas and one SamplerCustomAdvanced, ids `nid`, `nid`+1, `nid`+2."""
    a, b, c = nid, str(int(nid) + 1), str(int(nid) + 2)
    g[a] = {"class_type": "BasicGuider", "inputs": {"model": model, "conditioning": ["5", 0]}}
    g[b] = {"class_type": "ManualSigmas", "inputs": {"sigmas": sigmas}}
    g[c] = {"class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": list(noise), "guider": [a, 0], "sampler": ["7", 0],
                       "sigmas": [b, 0], "latent_image": latent}}


def decode(g: dict, latent: list, prefix: str) -> None:
    g["90"] = {"class_type": "VAEDecode", "inputs": {"samples": latent, "vae": ["3", 0]}}
    g["91"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": latent, "vae": ["4", 0]}}
    g["92"] = {"class_type": "CreateVideo",
               "inputs": {"images": ["90", 0], "fps": FPS, "audio": ["91", 0],
                          "bit_depth": "auto", "color_space": "sRGB", "codec": "none"}}
    g["93"] = {"class_type": "SaveVideo",
               "inputs": {"video": ["92", 0], "filename_prefix": f"video/{prefix}",
                          "format": "auto", "format.codec": "auto", "codec": "auto"}}


def lora(g: dict, nid: str, name: str, blocks: str = "all") -> list:
    """`H3ExactLoRA` on the checkpoint, then kitchen attention, as the pack's graphs wire it."""
    g[nid] = {"class_type": "H3ExactLoRA",
              "inputs": {"model": ["1", 0], "lora_name": name, "strength": 1.0, "blocks": blocks}}
    att = str(int(nid) + 1)
    g[att] = {"class_type": "ModelAttentionBackend",
              "inputs": {"model": [nid, 0], "attention": "comfy kitchen attention"}}
    return [att, 0]


def pdd6() -> dict:
    g = base(C.MODELS["unet_fl2va"])
    sampler(g, "20", lora(g, "10", PDD), C.PDD_MANUAL_SIGMAS, ["5", 1])
    decode(g, ["22", 0], "h3_pdd6")
    return g


def i2v_pdd8() -> dict:
    g = base(C.MODELS["unet_fl2va"], "i2v")
    sampler(g, "20", lora(g, "10", PDD), C.PDD8_SIGMAS, ["5", 1])
    decode(g, ["22", 0], "h3_i2v_pdd8")
    return g


def r2v_pdd8() -> dict:
    g = base(C.MODELS["unet_ref2va"], "r2v")
    sampler(g, "20", lora(g, "10", PDD_R2V), C.PDD8_SIGMAS, ["5", 1])
    decode(g, ["22", 0], "h3_r2v_pdd8")
    return g


# The FlashGen and FastH3 recipes (`h3_t2v_pdd8_flashgen_finish`,
# `h3_t2v_flashgen_late_blocks`, `h3_r2v_flashgen`, `h3_i2v_flashgen`,
# `h3_r2v_pdd8_flashgen_finish`, `h3_t2v_fasth3_contract`) left this builder on
# 2026-11-04 with the pack's FlashGen and FastH3 constants (`docs/roadmap.md`,
# "Closed lanes"); they are in git before that commit. The package's published
# example files are its own release and were not touched here.


#: file stem: (builder, the note shown in the workflow). Notes state what the
#: recipe is and the one-line why, each pointing at the record behind it.
RECIPES = {
    "h3_t2v_pdd6": (pdd6, (
        "## PDD6: close-ups and low motion only (t2v)\n\n"
        "The PDD8 file on a 6-step schedule that keeps PDD8's last, narrow blocks. Use it "
        "only for close-ups and low-motion scenes, and expect it to be iffy even there; it "
        "is here because it is faster, not because it is good.\n\n"
        "Why: PDD's quality follows how coarse its final steps are, not how many there are. "
        "It kept 90-95% of PDD8's fine detail at three quarters of the steps, but PDD8 won "
        "all three fast-motion scenes on artifacts.")),
    "h3_i2v_pdd8": (i2v_pdd8, (
        "## PDD8 alone (image to video)\n\n"
        "PDD8's 8 steps from a first frame. Load your own image: H3 Keyframe Canvas "
        "sizes the video to its aspect, so it is never stretched.\n\n"
        "Why: the pick for i2v. The FlashGen finish brightened the frame at once, "
        "so PDD8 alone won on the one image judged.")),
    "h3_r2v_pdd8": (r2v_pdd8, (
        "## PDD8 alone (reference to video)\n\n"
        "PDD8's 8 steps on the ref2va checkpoint, with two reference images. The "
        "prompt calls <Picture 1> the main character and <Picture 2> the setting; "
        "load your own and edit the subject definitions to match.\n\n"
        "Why: the ref2va PDD8 we run. It has not been compared against the "
        "alternatives.")),
}


# ---- the frontend, in headless Chrome ------------------------------------------

class Frontend:
    """ComfyUI's frontend in headless Chrome, driven over the DevTools protocol."""

    async def __aenter__(self):
        import aiohttp
        self._dir = tempfile.mkdtemp(prefix="mutant_examples_chrome_")
        self._proc = subprocess.Popen(
            [CHROME, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={self._dir}",
             "--no-first-run", "--no-default-browser-check", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(self._dir) / "DevToolsActivePort"
        for _ in range(100):
            if port_file.exists() and port_file.read_text().strip():
                break
            time.sleep(0.1)
        port = port_file.read_text().split()[0]
        self._http = aiohttp.ClientSession()
        async with self._http.get(f"http://127.0.0.1:{port}/json/list") as r:
            page = next(t for t in await r.json() if t["type"] == "page")
        self._ws = await self._http.ws_connect(page["webSocketDebuggerUrl"], max_msg_size=0)
        self._id = 0
        await self._call("Page.navigate", url=SERVER)
        deadline = time.time() + 60
        while not await self.eval("!!(window.app && window.app.graph && window.app.vueAppReady !== false)"):
            if time.time() > deadline:
                raise RuntimeError(f"the frontend at {SERVER} did not come up")
            await asyncio.sleep(0.5)
        await asyncio.sleep(2)
        return self

    async def __aexit__(self, *exc):
        await self._ws.close()
        await self._http.close()
        self._proc.terminate()
        self._proc.wait()
        shutil.rmtree(self._dir, ignore_errors=True)

    async def _call(self, method, **params):
        self._id += 1
        mid = self._id
        await self._ws.send_json({"id": mid, "method": method, "params": params})
        while True:
            msg = await self._ws.receive_json()
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    async def eval(self, expr: str):
        # DevTools `Runtime.evaluate` in a throwaway headless Chrome profile. The
        # expressions are built by this script from its own JSON; nothing
        # outside it reaches here.
        res = await self._call("Runtime.evaluate", expression=expr, awaitPromise=True,
                               returnByValue=True)
        if "exceptionDetails" in res:
            raise RuntimeError(res["exceptionDetails"].get("exception", {}).get("description")
                               or res["exceptionDetails"])
        return res["result"].get("value")

    async def api_to_ui(self, api: dict) -> dict:
        return await self.eval(
            f"(async () => {{ await app.loadApiJson({json.dumps(api)}, 'x.json');"
            f" return app.graph.serialize(); }})()")

    async def ui_to_api(self, ui: dict) -> dict:
        return await self.eval(
            f"(async () => {{ await app.loadGraphData({json.dumps(ui)}, true, true, 'x');"
            f" return (await app.graphToPrompt()).output; }})()")

    async def missing_nodes(self, api: dict) -> list:
        types = sorted({n["class_type"] for n in api.values()})
        return await self.eval(f"{json.dumps(types)}.filter(t => !LiteGraph.registered_node_types[t])")


def dress(ui: dict, note: str, stem: str) -> dict:
    """Add each loader's model URL and the recipe's note to a serialized UI graph.

    The frontend gives every serialized graph a random `id`; a UUID5 of the
    file name keeps a regenerated file byte-identical when its recipe is."""
    ui = copy.deepcopy(ui)
    ui["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"h3-mutant-distill/{stem}"))
    for n in ui["nodes"]:
        widget = FILE_WIDGETS.get(n["type"])
        if widget is None:
            continue
        name = n["widgets_values"][0]
        if name not in MODELS:
            raise ValueError(f"{n['type']} loads {name!r}, which has no published URL here")
        url, folder = MODELS[name]
        n.setdefault("properties", {})["models"] = [{"name": name, "url": url, "directory": folder}]
    right = max(n["pos"][0] + n["size"][0] for n in ui["nodes"])
    top = min(n["pos"][1] for n in ui["nodes"])
    nid = max(n["id"] for n in ui["nodes"]) + 1
    ui["nodes"].append({"id": nid, "type": "MarkdownNote", "pos": [right + 60, top], "size": [480, 360],
                        "flags": {}, "order": len(ui["nodes"]), "mode": 0, "inputs": [], "outputs": [],
                        "properties": {}, "widgets_values": [f"{HEADER}\n\n{note}"]})
    ui["last_node_id"] = max(ui.get("last_node_id", 0), nid)
    return ui


def normalize(api: dict) -> dict:
    """Class and inputs by node id, links as [str id, slot], floats as floats."""
    out = {}
    for nid, n in api.items():
        ins = {}
        for k, v in n["inputs"].items():
            if isinstance(v, list) and len(v) == 2:
                v = [str(v[0]), int(v[1])]
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                v = float(v)
            ins[k] = v
        out[str(nid)] = {"class_type": n["class_type"], "inputs": ins}
    return out


def diff(want: dict, got: dict) -> list[str]:
    w, g = normalize(want), normalize(got)
    probs = [f"node {k} missing" for k in w if k not in g] + [f"extra node {k}" for k in g if k not in w]
    for k in w:
        if k in g:
            if w[k]["class_type"] != g[k]["class_type"]:
                probs.append(f"node {k}: {g[k]['class_type']} for {w[k]['class_type']}")
            for name in sorted(set(w[k]["inputs"]) | set(g[k]["inputs"])):
                a, b = w[k]["inputs"].get(name, "<absent>"), g[k]["inputs"].get(name, "<absent>")
                if a != b:
                    probs.append(f"node {k} {w[k]['class_type']}.{name}: {str(b)[:60]!r} for {str(a)[:60]!r}")
    return probs


#: The nodes an audio refine pass is built from: the pack's own, and the noise
#: masks a core-only refine would need. Owner, 2026-10-01: audio refine is
#: usually not worth its cost, so it is never a stage of an example; it goes in
#: a workflow of its own, whose stem says `audio_refine`. Inherited from the
#: pack's refine graphs (`distill_experiments/h3_probe_t2v_*_audio_refine*`).
AUDIO_REFINE_CLASSES = ("MiniMaxH3AudioRefineMask", "MiniMaxH3FrozenVideoCache",
                        "SetLatentNoiseMask")


async def run(check: bool) -> int:
    fails = 0
    async with Frontend() as fe:
        for stem, (build, note) in RECIPES.items():
            api = build()
            refine = sorted({n["class_type"] for n in api.values()} & set(AUDIO_REFINE_CLASSES))
            if refine and "audio_refine" not in stem:
                print(f"  RED   {stem}: carries an audio refine stage ({refine}); give it a workflow "
                      "of its own named *_audio_refine")
                fails += 1
                continue
            missing = await fe.missing_nodes(api)
            if missing:
                print(f"  RED   {stem}: the frontend has no {missing}; is h3-mutant-distill loaded?")
                fails += 1
                continue
            path = OUT / f"{stem}.json"
            if check:
                if not path.exists():
                    print(f"  RED   {stem}: {path.name} is missing")
                    fails += 1
                    continue
                ui = json.loads(path.read_text())
            else:
                ui = dress(await fe.api_to_ui(api), note, stem)
            probs = diff(api, await fe.ui_to_api(ui))
            if probs:
                fails += 1
                print(f"  RED   {stem}: the UI file does not load back as the recipe")
                for p in probs[:12]:
                    print(f"          {p}")
                continue
            if not check:
                OUT.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(ui, indent=2) + "\n")
            print(f"  ok    {stem}: {len(api)} nodes round-trip{'' if check else ', written'}")
    print(f"\n{'RED' if fails else 'GREEN'}: {fails} failure(s)")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="h3-mutant-distill's example workflows.")
    ap.add_argument("--check", action="store_true",
                    help="round-trip the committed files instead of writing them")
    args = ap.parse_args(argv)
    return asyncio.run(run(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
