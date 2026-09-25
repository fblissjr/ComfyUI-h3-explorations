"""Record: bench/results/2026-09-25_upstream_pdd_comparison.md section 1. Network (HTTP range
requests to Hugging Face) and CPU; no GPU.

Range-fetch the final_layer bank tensors of Kijai's current PDD files and classify rows 1..31
as deltas from head 0 or absolute heads, against alibaba-pai's raw stacks. CPU only."""
import json, os, struct, urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
HF = "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/f94b1bcc94"
REPO_DIR = Path(__file__).resolve().parents[1]
AP = os.environ.get("ALIBABA_DIR", str(REPO_DIR / "coderef/alibaba-pai_MiniMax-H3-Acc-LoRAs"))
FILES = {"MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors": "FL2VA",
         "MiniMax-H3-FL2VA-Acc-8Step_pruned_comfy.safetensors": "FL2VA",
         "MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors": "Ref2VA",
         "MiniMax-H3-Ref2VA-Acc-8Step_pruned_comfy.safetensors": "Ref2VA"}
def rng(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b-1}"})
    return urllib.request.urlopen(req, timeout=120).read()
DT = {"F32": torch.float32, "BF16": torch.bfloat16, "F16": torch.float16}
out = {}
for fn, part in FILES.items():
    url = f"{HF}/loras/{fn}"
    n = struct.unpack("<Q", rng(url, 0, 8))[0]; hdr = json.loads(rng(url, 8, 8 + n)); base = 8 + n
    raw = {}
    with safe_open(f"{AP}/MiniMax-H3-{part}-Acc-8Step.safetensors", "pt") as f:
        raw = {"video": f.get_tensor("proj_out.weight").double(), "audio": f.get_tensor("audio_proj_out.weight").double()}
    res = {"header_metadata_conversion": hdr.get("__metadata__", {}).get("conversion", "")[:160]}
    for st, ck in (("video", "video_out"), ("audio", "audio_out")):
        p = f"diffusion_model.final_layer.{ck}"
        t = {}
        for k in ("lora_up.weight", "lora_down.weight"):
            e = hdr[f"{p}.{k}"]; a, b = e["data_offsets"]
            t[k] = torch.frombuffer(bytearray(rng(url, base + a, base + b)), dtype=DT[e["dtype"]]).reshape(e["shape"]).double()
        prod = t["lora_up.weight"] @ t["lora_down.weight"]           # [32*out, in]
        hw = raw[st]; r = prod.reshape(hw.shape)
        res[st] = {"rows1_31_vs_delta_from_head0_rel": ((r[1:] - (hw[1:] - hw[:1])).norm() / (hw[1:] - hw[:1]).norm()).item(),
                   "rows1_31_vs_absolute_heads_rel": ((r[1:] - hw[1:]).norm() / hw[1:].norm()).item()}
    out[fn] = res; print(fn, json.dumps(res))
json.dump(out, open(os.environ.get("OUT", REPO_DIR / "bench/results/2026-09-25_kijai_pdd_bank_encoding.json"), "w"), indent=1)
