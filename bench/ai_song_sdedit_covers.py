#!/usr/bin/env python3
"""Zero-memorization real-world AI-song validation of the 2-Pass Audio SDEdit Cover Pipeline.

Architecture (proven winner of `bench/music_cover_shootout.py` on zero-memorization Blues-covers-Disco):
  - Pass 1: Pika 6-step `ref2va` + `MiniMaxH3FreezeAudio(song_x, audio_mask=0.0)`
            -> renders photorealistic Target-B video with frame-accurate lip-sync & guitar strumming
               locked to Song X, and stores Song X's clean latent in `latent["samples"][1]`.
  - Pass 2: Undistilled Base `ref2va` (16-step `euler/simple`) + `MiniMaxH3AudioRefineMask(0.0, 1.0)`
            + `BasicScheduler(steps=16, denoise=0.45..0.52)` (exact `sigma_a0 = 0.71..0.77`),
            NO `cond_audio`, ONLY `<Audio 1>` (clean solo Kurt vocal+guitar) + verbatim lyrics prompt!
            Video stays pixel-exact frozen (`MAE = 0.0`) while audio denoises 100% freely from
            Song X's coarse melody/rhythm latent into Target B's raw voice & acoustic guitar.

Arms (all 1024x768 exact 4:3 MTV Unplugged, zero pretraining memorization):
  - `ai1_whatevercomes_d45`      : whatevercomes.wav raw ballad (243f = 10.125s), denoise=0.45
  - `ai2_whatevercomes_d52`      : whatevercomes.wav raw ballad (243f = 10.125s), denoise=0.52 (Pass 1 cached)
  - `ai3_idontwannacry_harm_d45` : idontwannacry.wav HPSS pop-punk (192f = 8.000s), denoise=0.45
  - `ai4_violet_v1_harm_d45`     : violet.mp3 HPSS verse 1 (192f = 8.000s), denoise=0.45
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap, comfy_client  # noqa: E402

bootstrap(cpu=True)
sys.path = [p for p in sys.path if p != str(REPO)]
if str(REPO / "workflows") not in sys.path:
    sys.path.insert(1, str(REPO / "workflows"))

import h3_config  # noqa: E402
import probe_yin_f0_drift as yin_probe  # noqa: E402

DEFAULT_URL = comfy_client.DEFAULT_COMFY_URL
OUT_PREFIX_DIR = "Video/ai_song_sdedit_covers_2026-11-27"
DATA_DIR = REPO / "data" / "music_cover_gate"
GRAPHS_DIR = DATA_DIR / "graphs_ai_sdedit"

PIC_FACE = "music_cover_gate/unplugged_b_face.png"
PIC_STAGE = "music_cover_gate/unplugged_b_stage.png"
AUDIO_TARGET_B = "music_cover_gate/unplugged_b_solo_vocal_243f.wav"

WIDTH = 1024
HEIGHT = 768
SEED = 910427311

UNET_REF2VA = h3_config.MODELS["unet_ref2va"]
ENCODER_BF16 = h3_config.MODELS["clip"]
VIDEO_VAE = h3_config.MODELS["video_vae"]
AUDIO_VAE = h3_config.MODELS["audio_vae"]
LORA_PIKA_REF2VA = h3_config.PIKA_REF2VA_SVD995_LORA
PIKA_SIGMAS = h3_config.PIKA_MANUAL_SIGMAS

ARM_SPECS: dict[str, dict[str, Any]] = {
    "ai1_whatevercomes_d45": {
        "song_x": "music_cover_gate/whatevercomes_243f.wav",
        "song_x_raw": "music_cover_gate/whatevercomes_243f.wav",
        "prompt": DATA_DIR / "ref2va_unplugged_whatevercomes.txt",
        "length": 243,
        "denoise": 0.45,
    },
    "ai2_whatevercomes_d52": {
        "song_x": "music_cover_gate/whatevercomes_243f.wav",
        "song_x_raw": "music_cover_gate/whatevercomes_243f.wav",
        "prompt": DATA_DIR / "ref2va_unplugged_whatevercomes.txt",
        "length": 243,
        "denoise": 0.52,
    },
    "ai3_idontwannacry_harm_d45": {
        "song_x": "music_cover_gate/idontwannacry_harm_192f.wav",
        "song_x_raw": "music_cover_gate/idontwannacry_192f.wav",
        "prompt": DATA_DIR / "ref2va_unplugged_idontwannacry.txt",
        "length": 192,
        "denoise": 0.45,
    },
    "ai4_violet_v1_harm_d45": {
        "song_x": "music_cover_gate/violet_v1_harm_192f.wav",
        "song_x_raw": "music_cover_gate/violet_v1_192f.wav",
        "prompt": DATA_DIR / "ref2va_unplugged_violet_v1.txt",
        "length": 192,
        "denoise": 0.45,
    },
}
ALL_ARMS: tuple[str, ...] = tuple(ARM_SPECS.keys())


def _require_dir(env_name: str) -> Path:
    val = os.environ.get(env_name)
    if not val:
        raise SystemExit(f"{env_name} is unset; run `set -a; . ./h3.env; set +a` first")
    p = Path(val)
    if not p.is_dir():
        raise SystemExit(f"{env_name} does not exist: {p}")
    return p


def _wire_sol(g: dict[str, Any], *, in_id: str, shift_id: str, attn_id: str, sol_id: str) -> str:
    g[shift_id] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": [in_id, 0], "shift_video": 12.0, "shift_audio": 3.0},
    }
    g[attn_id] = {
        "class_type": "ModelAttentionBackend",
        "inputs": {"model": [shift_id, 0], "attention": "comfy kitchen attention"},
    }
    g[sol_id] = {
        "class_type": "MiniMaxH3Sol",
        "inputs": {
            "model": [attn_id, 0],
            "tau": 1.0,
            "start_percent": 0.2,
            "end_percent": 1.0,
            "min_tokens": 12288,
            "sink_conditioning": "exact_kv_and_all_rows",
            "verbose": True,
            "dense_blocks": "49",
            "quantizer": "rotated",
            "token_routing": "off",
            "rows": "per segment",
            "rows.text_rows": "exact",
            "rows.reference_rows": "exact",
            "rows.audio_rows": "exact",
            "rows.video_reference_rows": "routed",
        },
    }
    return sol_id


def build_arm_graph(arm: str) -> dict[str, Any]:
    spec = ARM_SPECS[arm]
    prompt_text = spec["prompt"].read_text(encoding="utf-8").strip()
    length = int(spec["length"])
    g: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET_REF2VA, "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": ENCODER_BF16, "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": LORA_PIKA_REF2VA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
    }
    pika_sol = _wire_sol(g, in_id="18", shift_id="19", attn_id="58", sol_id="21")
    base_sol = _wire_sol(g, in_id="1", shift_id="119", attn_id="158", sol_id="121")

    # References: <Picture 1> face + <Picture 2> stage + <Audio 1> clean solo vocal+guitar Kurt
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": PIC_FACE}}
    g["16"] = {"class_type": "LoadImage", "inputs": {"image": PIC_STAGE}}
    pic_policy = {"size_policy": "tiered", "qwen_view": "separate", "qwen_view.qwen_short_edge": 512}
    g["50"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["15", 0], **pic_policy}}
    g["51"] = {"class_type": "MiniMaxH3AppendRefImage", "inputs": {"image": ["16", 0], "references": ["50", 0], **pic_policy}}
    g["33"] = {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_TARGET_B}}
    g["52"] = {"class_type": "MiniMaxH3AppendRefAudio", "inputs": {"audio": ["33", 0], "references": ["51", 0]}}
    g["132"] = {
        "class_type": "MiniMaxH3EncodeReferences",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "audio_vae": ["4", 0],
            "references": ["52", 0],
            "width": WIDTH,
            "height": HEIGHT,
            "length": length,
            "video_policy": "comfy",
            "image_policy": "comfy",
        },
    }
    g["5"] = {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": prompt_text}}
    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}}

    # Pass 1: Freeze Song X to sync Kurt's lip/guitar video and seed audio latent
    g["48"] = {"class_type": "LoadAudio", "inputs": {"audio": spec["song_x"]}}
    g["49"] = {
        "class_type": "MiniMaxH3FreezeAudio",
        "inputs": {
            "latent": ["26", 1],
            "audio_vae": ["4", 0],
            "audio": ["48", 0],
            "start_seconds": 0.0,
            "audio_mask": 0.0,
            "level": "clip_guard",
        },
    }
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
    g["7"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}}
    g["60"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": PIKA_SIGMAS}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": [pika_sol, 0], "conditioning": ["26", 0]}}
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["60", 0],
            "latent_image": ["49", 0],
        },
    }

    # Pass 2: True Audio SDEdit with undistilled Base ref2va (video frozen, audio_mask=1.0, NO cond_audio!)
    g["152"] = {"class_type": "MiniMaxH3AudioRefineMask", "inputs": {"latent": ["10", 0], "video_mask": 0.0, "audio_mask": 1.0}}
    g["106"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED + 17}}
    g["107"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    g["160"] = {
        "class_type": "BasicScheduler",
        "inputs": {"model": [base_sol, 0], "scheduler": "simple", "steps": 16, "denoise": float(spec["denoise"])},
    }
    g["109"] = {"class_type": "BasicGuider", "inputs": {"model": [base_sol, 0], "conditioning": ["26", 0]}}
    g["110"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["106", 0],
            "guider": ["109", 0],
            "sampler": ["107", 0],
            "sigmas": ["160", 0],
            "latent_image": ["152", 0],
        },
    }
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {"latent": ["110", 0], "model": [base_sol, 0], "sigmas": ["160", 0], "clip": ["2", 0], "note": arm},
    }
    g["111"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
    g["112"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
    g["113"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["111", 0],
            "audio": ["112", 0],
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": f"{OUT_PREFIX_DIR}/{arm}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    }
    return g


def _preflight(gp: Path) -> tuple[int, str]:
    p = subprocess.run([sys.executable, str(REPO / "bench" / "preflight_graph.py"), str(gp)], capture_output=True, text=True)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def _comfy_pid() -> int | None:
    try:
        for line in subprocess.check_output(["pgrep", "-f", r"main\.py.*--port[= ]8188"], text=True).splitlines():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return None


def _wait_idle(url: str) -> None:
    pid = _comfy_pid()
    while True:
        q = json.loads(urllib.request.urlopen(f"{url}/queue", timeout=5).read().decode("utf-8"))
        hogs = []
        try:
            for line in subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True
            ).splitlines():
                p_s, m_s = [x.strip() for x in line.split(",")[:2]]
                if int(float(m_s)) >= 1000 and (pid is None or int(p_s) != pid):
                    hogs.append((int(p_s), int(float(m_s))))
        except Exception:
            pass
        if not q.get("queue_running") and not q.get("queue_pending") and not hogs:
            return
        time.sleep(10.0)


def _extract_videos(hist: dict[str, Any], out_dir: Path) -> list[Path]:
    res: list[Path] = []
    for no in (hist.get("outputs") or {}).values():
        if not isinstance(no, dict):
            continue
        for key in ("gifs", "videos"):
            for it in no.get(key) or []:
                if isinstance(it, dict) and it.get("filename"):
                    p = out_dir / (it.get("subfolder") or "") / it["filename"]
                    if p.is_file():
                        res.append(p)
    return res


def _load_mono_np(p: Path, sr: int = 24000) -> np.ndarray:
    return np.frombuffer(
        subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(p), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]),
        dtype=np.float32,
    ).copy()


def _chroma(y: np.ndarray, sr: int = 24000) -> np.ndarray:
    spec = torch.stft(torch.from_numpy(y), n_fft=4096, hop_length=600, window=torch.hann_window(4096), return_complex=True).abs().numpy() ** 2
    freqs = np.fft.rfftfreq(4096, 1.0 / sr)
    valid = (freqs >= 65.0) & (freqs <= 2000.0)
    pc = np.round(69.0 + 12.0 * np.log2(freqs[valid] / 440.0)).astype(int) % 12
    C = np.zeros((12, spec.shape[1]), dtype=np.float64)
    for k in range(12):
        C[k] = spec[valid][pc == k].sum(axis=0)
    C = np.log1p(C * 1e3)
    C = C - C.mean(axis=0, keepdims=True)
    return C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-8)


def _chroma_sim(a: np.ndarray, b: np.ndarray, any_key: bool = False) -> tuple[float, int, int]:
    n = min(a.shape[1], b.shape[1])
    best, blag, btr = -1.0, 0, 0
    for tr in (range(12) if any_key else (0,)):
        bt = np.roll(b, tr, axis=0) if tr else b
        for lag in range(-12, 13):
            aa = a[:, lag:n] if lag >= 0 else a[:, :n + lag]
            bb = bt[:, :n - lag] if lag >= 0 else bt[:, -lag:n]
            if aa.shape[1] < 20:
                continue
            s = float(np.mean(np.sum(aa * bb, axis=0)))
            if s > best:
                best, blag, btr = s, lag, tr
    return best, blag, btr


def _onset_corr(a_y: np.ndarray, b_y: np.ndarray) -> tuple[float, int]:
    def flux(y):
        S = np.log1p(torch.stft(torch.from_numpy(y), n_fft=1024, hop_length=600, window=torch.hann_window(1024), return_complex=True).abs().numpy())
        return np.maximum(0.0, np.diff(S, axis=1)).mean(axis=0)
    fa, fb = flux(a_y), flux(b_y)
    n = min(len(fa), len(fb))
    best, blag = -1.0, 0
    for lag in range(-12, 13):
        aa = fa[lag:n] if lag >= 0 else fa[:n + lag]
        bb = fb[:n - lag] if lag >= 0 else fb[-lag:n]
        aa, bb = aa - aa.mean(), bb - bb.mean()
        r = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
        if r > best:
            best, blag = r, lag
    return best, blag


def _ltas_cos(a_y: np.ndarray, b_y: np.ndarray) -> float:
    m = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=64)
    va = torch.log(m(torch.from_numpy(a_y)) + 1e-6).mean(dim=-1)
    vb = torch.log(m(torch.from_numpy(b_y)) + 1e-6).mean(dim=-1)
    va, vb = va - va.mean(), vb - vb.mean()
    return float(torch.dot(va, vb).item() / (float((va.norm() * vb.norm()).item()) + 1e-12))


def analyze(input_dir: Path, output_dir: Path, arm_files: dict[str, Path]) -> dict[str, Any]:
    tb = _load_mono_np(input_dir / AUDIO_TARGET_B)
    rep: dict[str, Any] = {"arms": {}}
    for name, mp4 in arm_files.items():
        spec = ARM_SPECS[name]
        sx = _load_mono_np(input_dir / spec["song_x_raw"])
        y = _load_mono_np(mp4)
        c, sx_c = _chroma(y), _chroma(sx)
        cx0, lx0, _ = _chroma_sim(c, sx_c, False)
        cxT, lxT, tr = _chroma_sim(c, sx_c, True)
        ox, olx = _onset_corr(y, sx)
        f0, _, _ = yin_probe.yin_f0(torch.from_numpy(y))
        f0_np = f0.numpy()
        valid = np.isfinite(f0_np) & (f0_np > 0)
        med = float(np.median(f0_np[valid])) if np.any(valid) else float("nan")
        rep["arms"][name] = {
            "file": str(mp4.relative_to(output_dir)),
            "median_f0_hz": round(med, 2) if math.isfinite(med) else None,
            "chroma_sameKey_x": round(cx0, 3),
            "chroma_lag_x": lx0,
            "chroma_anyKey_x": round(cxT, 3),
            "chroma_tr_st": tr,
            "onset_x": round(ox, 3),
            "onset_lag_x": olx,
            "ltas_cos_unplugged_b": round(_ltas_cos(y, tb), 3),
            "ltas_cos_song_x": round(_ltas_cos(y, sx), 3),
        }
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=list(ALL_ARMS))
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args(argv)

    input_dir = _require_dir("H3_INPUT_DIR")
    output_dir = _require_dir("H3_OUTPUT_DIR")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

    graphs: dict[str, dict[str, Any]] = {}
    if args.dry_run or args.run:
        for arm in args.arms:
            g = build_arm_graph(arm)
            gp = GRAPHS_DIR / f"{arm}.json"
            gp.write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
            rc, out = _preflight(gp)
            print(f"[preflight] {arm}: {'PASS' if rc == 0 else f'FAIL({rc})'}")
            if rc != 0:
                print(out)
                return 1
            graphs[arm] = g
        if args.dry_run and not args.run:
            return 0

    arm_files: dict[str, Path] = {}
    timings: dict[str, float] = {}
    if args.run:
        cid = str(uuid.uuid4())
        for arm in args.arms:
            _wait_idle(args.url)
            print(f"\n[run] queuing {arm}...", flush=True)
            t0 = time.monotonic()
            pid = comfy_client.submit_prompt(graphs[arm], base_url=args.url, client_id=cid)
            hist = comfy_client.wait_for_prompt(pid, base_url=args.url, timeout=1800.0)
            dt = comfy_client.execution_duration_s(hist) or (time.monotonic() - t0)
            timings[arm] = round(dt, 2)
            vids = _extract_videos(hist, output_dir)
            print(f"[run] {arm} finished in {dt:.1f}s -> {[v.name for v in vids]}", flush=True)
            if vids:
                arm_files[arm] = vids[-1]

    if args.analyze or args.run:
        vid_dir = output_dir / OUT_PREFIX_DIR
        if vid_dir.is_dir():
            for arm in ALL_ARMS:
                if arm not in arm_files:
                    m = sorted(vid_dir.glob(f"{arm}_*-audio.mp4"))
                    if m:
                        arm_files[arm] = m[-1]
        rep = analyze(input_dir, output_dir, arm_files)
        if timings:
            rep["timings_s"] = timings
        rp = DATA_DIR / "ai_song_sdedit_triage_report.json"
        rp.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
