# Upstream survey checks, 2026-09-25

Four checks run during the 2026-09-25 upstream survey
(`docs/sol_upstream.md`, "comfy-kitchen and core, 2026-09-25"). Each check
is here with its command and outcome, so the prose can point at a record
instead of carrying the result. ComfyUI checkout at `88ab4a06`, run with its
venv interpreter, on this box's RTX 4090. No server was running, and no core
file was edited.

## 1. `comfy.audio.resample` against `torchaudio.functional.resample`

Core #16457 replaced torchaudio with `comfy/audio.py::resample`. This pack
still calls torchaudio at its resample sites (`audio_freeze.py`,
`reference_conditioning.py`).

```python
import torch, torchaudio, comfy.audio as ca
torch.manual_seed(0)
for sr in (44100, 48000, 16000, 24000):
    w = torch.randn(2, sr * 3)
    a = torchaudio.functional.resample(w, sr, 32000)
    b = ca.resample(w, sr, 32000)
    print(sr, a.shape == b.shape, (a - b).abs().max().item(), torch.equal(a, b))
```

Outcome: `torch.equal` is True at every rate listed, with a max absolute
difference of zero. That was CPU, fp32, default arguments on both sides,
into the audio VAE's rate. Not checked: CUDA tensors, fp16 input, and
other rates. The swap is a drop-in on this evidence, for this configuration.

## 2. Per-block attention key in shipped model files

Core #16419 lets a checkpoint route a block's attention through a
`<module>.comfy_attention.config` tensor.

```python
# every .safetensors named in workflows/h3_config.py::MODELS, found under ComfyUI/models
from safetensors import safe_open
with safe_open(path, "pt") as f:
    hits = [k for k in f.keys() if "comfy_attention" in k]
```

Outcome: no hits in any file in `h3_config.MODELS`. That covers the DiT
checkpoints, the encoder and both VAEs.

## 3. Quantisation format of the shipped DiT and encoder files

Same loop, reading each `*.comfy_quant` metadata tensor. Outcome: every DiT
file and the encoder in `h3_config.MODELS` reports `"format":
"int8_tensorwise"` with `convrot`. None is W6A8, which is what core #16483 and
kitchen #191 add.

## 4. Core's dtype choice under open PR 16508

PR 16508 adds `torch.float16` to H3's `supported_inference_dtypes`. This
install launches with `--fast fp16_accumulation` (`start.sh`), which sets
`comfy.model_management.PRIORITIZE_FP16`. For a quantized checkpoint,
`comfy/sd.py` passes `weight_dtype=None` to `unet_dtype` and `None` to
`unet_manual_cast`. Called in-process with the flag forced on and off:

```python
import torch, comfy.model_management as mm
dev = mm.get_torch_device()
for pf in (False, True):
    mm.PRIORITIZE_FP16 = pf
    for sup in ([torch.bfloat16, torch.float32],                 # today
                [torch.bfloat16, torch.float16, torch.float32]): # PR 16508
        print(pf, sup,
              mm.unet_dtype(model_params=16_500_000_000, supported_dtypes=sup, weight_dtype=None),
              mm.unet_manual_cast(None, dev, sup))
```

Outcome: bf16 for both in three of the four cases. fp16 for both in exactly
one case, with the flag on and the PR's list. A walk of
`h3_config.graph_paths(..., include_bench=True)` finds every DiT loader at
`weight_dtype` default, so no graph pins a dtype. The PR diff touches
`comfy/ldm/minimax/model.py`, `comfy/model_base.py` and
`comfy/supported_models.py`, and neither `sd.py` nor `model_management.py`.

So if 16508 merged as written, this launcher would run the DiT in fp16.
`sol_attn_h3.py::_ineligible` refuses non-bf16 `q` ("kernel is bf16-only"),
so every Sol call would fall back to dense. **That last step is reasoned,
not run**: no render was made with the PR applied.
