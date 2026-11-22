#!/usr/bin/env python3
"""Reproducible Benchmark & 8-Part Ablation Suite for Multi-Vector Pole-A Subspace Erasure.

Stages (`--stage`):
  - `census`     : (< 5s CPU) Full 38-concept bank (76 `.safetensors`) spectral, Grassmannian
                   cross-family (`fl2va` <-> `ref2va`) subspace overlap, and J-Lens X-ray census.
  - `cpu_math`   : (< 5s CPU) Exact `float64` mathematical ablations on the 7 saved `.pt` banks:
                   (M1) tau_attn sweep (MaxSim -> 1D) x {raw, de-coned cosine},
                   (M2) SVD(A_c) vs. ColBERT Skiplist SVD vs. Contested Shift-Weighted SVD,
                   (M3) Monte-Carlo Pole-A pair-count coverage curve (N = 1..all).
  - `ablations`  : (GPU stills) Controlled RMS-matched (`101.25 @ s=1.5`) renders for Ablations 1-5
                   (`tau`, `V_r` construction on `fl2va` + `h3_image`, clean rank `r` & ridge `lam`,
                   pair count `N`, and denoising step-window decoupling).
  - `word_media` : (GPU stills + 90f AV) Ablation 6 (`MiniMaxH3WordVector` legacy 1D vs. `Model2Vec`
                   + `against`-SVD erasure vs. 10-pair GPU) & Ablation 7 (`MakeMediaControlVector`
                   `ColQwen`-style patch erasure vs. 1D media contrast).
  - `audio_video`: (GPU 192f/90f joint AV) Ablation 8a (fixed Qwen BPE `pieces` `Clean SIF` &
                   `M5 Ward+WARP` on 192f `fl2va` + `ref2va`), Ablation 8b (40 Hz audio-stream
                   erasure on `reverb_cathedral`, `whisper_asmr`, `underwater_submerged`), and
                   Ablation 8c (stratified bank concepts).
  - `base50`     : (GPU undistilled 50-step Base `fl2va`) `noir` & `paper_origami` 2-seed stills
                   + `39f` speaking AV `1D` vs. `Erasure` check.
  - `all`        : Runs all stages sequentially.

All JSON records write to `data/exp_unpooled_sliders/paper_2026_11_21/` (zero media in `data/`),
and all contact sheets / `.mp4` / `.1fps` companions write strictly to
`$H3_OUTPUT_DIR/{Image,Video}/paper_erasure_2026-11-21/`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path = [p for p in sys.path if Path(p or ".").resolve() != (REPO_ROOT / "bench").resolve()]
sys.path.insert(0, str(REPO_ROOT / "bench"))
sys.path.insert(0, str(REPO_ROOT))

CPU_ONLY_STAGES = {"census", "cpu_math", "figures"}


def _is_cpu_only_invocation(argv: list[str]) -> bool:
    for i, arg in enumerate(argv):
        if arg == "--stage" and i + 1 < len(argv):
            return argv[i + 1] in CPU_ONLY_STAGES
        if arg.startswith("--stage="):
            return arg.split("=", 1)[1] in CPU_ONLY_STAGES
    return False


def _wait_for_comfy_idle_and_free(comfy_url: str = "http://127.0.0.1:8188") -> None:
    """Politely wait until `:8188/queue` AND any peer CLI GPU process are idle, then free cached VRAM."""
    import urllib.request

    my_pid = os.getpid()
    idle_streak = 0
    while idle_streak < 2:
        n_run, n_pend = 0, 0
        try:
            with urllib.request.urlopen(f"{comfy_url}/queue", timeout=5) as resp:
                q = json.loads(resp.read().decode("utf-8"))
            n_run = len(q.get("queue_running") or [])
            n_pend = len(q.get("queue_pending") or [])
        except Exception:
            pass

        # Also check nvidia-smi for any non-ComfyUI peer GPU process using > 1500 MiB
        peer_busy = False
        try:
            smi = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            )
            comfy_pids: set[int] = set()
            ss_out = subprocess.run(["ss", "-ltnp", "sport = :8188"], capture_output=True, text=True, check=False).stdout
            import re as _re

            for m_p in _re.finditer(r"pid=(\d+)", ss_out):
                comfy_pids.add(int(m_p.group(1)))
            for ln in smi.strip().splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    pid_i, mem_i = int(parts[0]), int(parts[1])
                    if pid_i in comfy_pids or pid_i == my_pid:
                        continue
                    try:
                        cmdline_i = (
                            Path(f"/proc/{pid_i}/cmdline")
                            .read_bytes()
                            .replace(b"\x00", b" ")
                            .decode("utf-8", errors="replace")
                        )
                    except Exception:
                        cmdline_i = ""
                    if "main.py" in cmdline_i and (
                        "--output-directory" in cmdline_i or "--listen" in cmdline_i or "8188" in cmdline_i
                    ):
                        comfy_pids.add(pid_i)
                        continue
                    if mem_i > 1500:
                        peer_busy = True
        except Exception:
            pass

        if n_run == 0 and n_pend == 0 and not peer_busy:
            idle_streak += 1
            if idle_streak < 2:
                time.sleep(3.0)
        else:
            idle_streak = 0
            print(
                f"  [gpu-wait] GPU busy (:8188 run={n_run}/pend={n_pend}, peer_cli={peer_busy}); waiting 10s...",
                flush=True,
            )
            time.sleep(10.0)
    try:
        req = urllib.request.Request(
            f"{comfy_url}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10):
            pass
        time.sleep(1.5)
    except Exception:
        pass


_IS_CPU_STAGE = _is_cpu_only_invocation(sys.argv[1:])
if not _IS_CPU_STAGE and "--help" not in sys.argv and "-h" not in sys.argv:
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    _wait_for_comfy_idle_and_free()

from _lib import bootstrap, server_memory_mode  # noqa: E402

bootstrap(cpu=_IS_CPU_STAGE)
if not _IS_CPU_STAGE and "--help" not in sys.argv and "-h" not in sys.argv:
    import importlib as _importlib
    import comfy.cli_args as _cli_args

    _cli_args.args.mmap_torch_files = True
    _cli_args.args.supports_fp8_compute = True
    _cli_args.args.cuda_malloc = True
    _cli_args.args.reserve_vram = 1.0
    _importlib.import_module("cuda_malloc")
    server_memory_mode()

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
import scipy.cluster.hierarchy as sch  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import control_vectors as cv  # noqa: E402
import h3_spans  # noqa: E402
import j_lens  # noqa: E402
from workflows import h3_config  # noqa: E402

DATA_OUT = REPO_ROOT / "data" / "exp_unpooled_sliders" / "paper_2026_11_21"
BANKS_DIR = REPO_ROOT / "data" / "exp_unpooled_sliders" / "banks"
PROMPTS_DIR = REPO_ROOT / "data" / "exp_unpooled_sliders" / "prompts"
PHASE1_SPECTRA = REPO_ROOT / "data" / "exp_unpooled_sliders" / "phase1_spectra.json"

NORM_TARGET = 67.5
SEED_MAIN = 730451892
SEEDS_3 = [730451892, 101, 2026]

SKIPLIST_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
        "by", "from", "into", "across", "under", "over", "through", "while", "as",
        "is", "are", "was", "were", "its", "his", "her", "their", "shot", "shots",
        "filmed", "captured", "recorded", "photographed", "framed", "frames",
        "camera", "cinematic", "scene", "view", "medium", "wide", "close",
    }
)


def _out_dirs() -> tuple[Path, Path]:
    out_base = Path(os.environ["H3_OUTPUT_DIR"])
    img_dir = out_base / "Image" / "paper_erasure_2026-11-21"
    vid_dir = out_base / "Video" / "paper_erasure_2026-11-21"
    img_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "strips").mkdir(parents=True, exist_ok=True)
    return img_dir, vid_dir


def rms_row_norm(x: torch.Tensor) -> float:
    return float(torch.sqrt((x.to(dtype=torch.float64) ** 2).sum(dim=-1).mean()).clamp(min=1e-12).item())


def hsv_sat(im: Image.Image) -> float:
    return float(np.asarray(im.convert("HSV"), dtype=np.float32)[..., 1].mean() / 255.0)


def red_excess(im: Image.Image) -> float:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32)
    return float((arr[..., 0] - 0.5 * (arr[..., 1] + arr[..., 2])).mean())


def grassmann_overlap(vr_a: torch.Tensor, vr_b: torch.Tensor) -> float:
    """Normalized Grassmannian subspace overlap `||Vr_a @ Vr_b.T||_F^2 / min(r_a, r_b)` in `[0, 1]`."""
    a64 = vr_a.detach().to(device="cpu", dtype=torch.float64)
    b64 = vr_b.detach().to(device="cpu", dtype=torch.float64)
    r_min = min(int(a64.shape[0]), int(b64.shape[0]))
    if r_min < 1:
        return 0.0
    cross = a64[:r_min] @ b64[:r_min].T
    return float((cross ** 2).sum().item() / r_min)


def make_contact_sheet(
    rows_imgs: list[tuple[str, list[tuple[str, Image.Image]]]],
    out_path: Path,
    *,
    thumb_w: int = 384,
) -> None:
    if not rows_imgs or not rows_imgs[0][1]:
        return
    first_im = rows_imgs[0][1][0][1]
    thumb_h = int(round(first_im.height * (thumb_w / first_im.width)))
    n_rows = len(rows_imgs)
    n_cols = max(len(r[1]) for r in rows_imgs)
    pad_top = 30
    label_w = 190
    canvas = Image.new("RGB", (label_w + n_cols * thumb_w, n_rows * (thumb_h + pad_top)), (18, 18, 22))
    draw = ImageDraw.Draw(canvas)
    for r_idx, (row_title, col_items) in enumerate(rows_imgs):
        y0 = r_idx * (thumb_h + pad_top)
        draw.text((8, y0 + pad_top + thumb_h // 2 - 8), row_title[:26], fill=(235, 235, 240))
        for c_idx, (col_title, im) in enumerate(col_items):
            x0 = label_w + c_idx * thumb_w
            draw.text((x0 + 6, y0 + 7), col_title[:44], fill=(210, 225, 255))
            canvas.paste(im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), (x0, y0 + pad_top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------------------
# Stage 1: Full 38-Concept Bank Spectral & Cross-Family Subspace Census (CPU)
# ---------------------------------------------------------------------------
def run_census() -> dict[str, Any]:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    store_root = os.environ.get("H3_MODEL_STORE", "").strip()
    cv_root = (Path(store_root) / "control_vectors" / "bank") if store_root else (REPO_ROOT.parent.parent / "models" / "control_vectors" / "bank")
    if not cv_root.is_dir():
        cv_root = REPO_ROOT.parent.parent / "models" / "control_vectors" / "bank"

    fl_files = sorted(cv_root.glob("*__fl2va*.safetensors"))
    ref_files = {p.name.split("__ref2va", 1)[0]: p for p in sorted(cv_root.glob("*__ref2va*.safetensors"))}

    phase1_lookup: dict[str, dict[str, Any]] = {}
    if PHASE1_SPECTRA.is_file():
        p1_raw = json.loads(PHASE1_SPECTRA.read_text(encoding="utf-8"))
        if isinstance(p1_raw, dict):
            phase1_lookup = p1_raw.get("concepts", p1_raw)
        elif isinstance(p1_raw, list):
            for item in p1_raw:
                if isinstance(item, dict):
                    k_nm = item.get("concept") or item.get("name")
                    if k_nm:
                        phase1_lookup[str(k_nm)] = item

    records: list[dict[str, Any]] = []
    b_jl = j_lens.load_j_lens_bundle()
    for fl_p in fl_files:
        stem = fl_p.name.split("__fl2va", 1)[0]
        t_fl, m_fl = cv.load_vector(fl_p)
        v1d = t_fl["vector.post_refiner"]
        mud = t_fl["erasure.mu_d"]
        vr = t_fl["erasure.vr"]
        diag = t_fl["erasure.diag"]

        v1d_nrm = float(torch.linalg.vector_norm(v1d.double()).item())
        mud_nrm = float(torch.linalg.vector_norm(mud.double()).item())
        cos_mv = float(F.cosine_similarity(v1d.flatten().double(), mud.flatten().double(), dim=0).item())

        # Invert diag_i = s_i^2 / (s_i^2 + 0.15 * s_0^2) -> rel_eig_i = s_i^2 / s_0^2
        d64 = diag.double().clamp(1e-6, 0.999999)
        rel_eigs = (0.15 * d64 / (1.0 - d64)).tolist()
        sum_e = sum(rel_eigs)
        sum_e2 = sum(x * x for x in rel_eigs)
        pr8 = (sum_e * sum_e / max(sum_e2, 1e-12)) if rel_eigs else 0.0

        qrep = m_fl.get("quality_report", {})
        pairs_m = m_fl.get("pairs", {})
        spans = pairs_m.get("pair_spans") or []
        n_a_total = sum(int(s.get("pooled_rows_a", 0)) for s in spans)
        n_b_total = sum(int(s.get("pooled_rows_b", 0)) for s in spans)

        # Cross-family fl2va <-> ref2va comparison
        cos_fl_ref = None
        vr_overlap_fl_ref = None
        if stem in ref_files:
            t_ref, _ = cv.load_vector(ref_files[stem])
            cos_fl_ref = round(
                float(F.cosine_similarity(v1d.flatten().double(), t_ref["vector.post_refiner"].flatten().double(), dim=0).item()),
                4,
            )
            if "erasure.vr" in t_ref:
                vr_overlap_fl_ref = round(grassmann_overlap(vr, t_ref["erasure.vr"]), 4)

        v1d_ro = j_lens.decode_vector(v1d, space="post_refiner", family="fl2va", top_k=8, bundle=b_jl)
        er_ro = j_lens.decode_erasure_payload(
            v1d, vr, diag, erasure_mu_d=mud, v1d_readout=v1d_ro, family="fl2va", top_k=8, bundle=b_jl
        )

        # Exact unpooled bank metrics when cached in data/exp_unpooled_sliders/banks/
        exact_1d_e = None
        exact_coherence = None
        exact_eff_rank = None
        bp_cand = BANKS_DIR / f"{stem}.pt"
        if bp_cand.is_file():
            b_pt = torch.load(bp_cand, map_location="cpu", weights_only=False)
            a_all = b_pt["A_all"].double()
            b_all = b_pt["B_all"].double() if "B_all" in b_pt else None
            if b_all is not None:
                mu_a = a_all.mean(dim=0, keepdim=True)
                shifts = b_all - mu_a
                mean_sh = shifts.mean(dim=0, keepdim=True)
                exact_1d_e = round(float((mean_sh ** 2).sum().item() / (shifts ** 2).sum(dim=-1).mean().item()), 4)
            if "D_matched" in b_pt:
                dm = b_pt["D_matched"].double()
                exact_coherence = round(
                    float(torch.linalg.vector_norm(dm.mean(dim=0)).item() / torch.linalg.vector_norm(dm, dim=-1).mean().item()),
                    4,
                )
            ac = a_all - a_all.mean(dim=0, keepdim=True)
            _u, sv, _vh = torch.linalg.svd(ac, full_matrices=False)
            p_sv = (sv ** 2) / (sv ** 2).sum().clamp(min=1e-12)
            p_nz = p_sv[p_sv > 1e-12]
            exact_eff_rank = round(float(torch.exp(-(p_nz * torch.log(p_nz)).sum()).item()), 2)
        elif stem in phase1_lookup and isinstance(phase1_lookup[stem], dict):
            p1_i = phase1_lookup[stem]
            exact_1d_e = p1_i.get("frac_1d_energy") or p1_i.get("rank1_energy_frac") or p1_i.get("1d_energy_frac")
            exact_eff_rank = p1_i.get("B_eff_rank") or p1_i.get("effective_rank") or p1_i.get("eff_rank")

        mpc = float(qrep.get("mean_pairwise_cosine", 0.0))
        regime = (
            "multi_facet_bundle"
            if (exact_1d_e is not None and exact_1d_e < 0.30) or mpc < 0.62
            else ("monolithic_override" if (exact_1d_e is not None and exact_1d_e >= 0.42) or mpc >= 0.72 else "intermediate")
        )
        records.append(
            {
                "concept": stem,
                "regime": regime,
                "n_pairs": int(pairs_m.get("n_pairs", 0)),
                "n_a_tokens": n_a_total,
                "n_b_tokens": n_b_total,
                "differing_sections": pairs_m.get("differing_sections", []),
                "mpc": round(mpc, 4),
                "split_half_cosine": round(float(qrep.get("split_half_cosine", 0.0)), 4),
                "v1d_raw_norm": round(v1d_nrm, 2),
                "mu_d_norm": round(mud_nrm, 2),
                "mu_d_over_v1d_norm": round(mud_nrm / max(v1d_nrm, 1e-9), 4),
                "cos_mu_d_v1d": round(cos_mv, 4),
                "pole_a_pr8": round(pr8, 2),
                "diag": [round(float(x), 3) for x in diag.tolist()],
                "exact_1d_energy_share": exact_1d_e,
                "exact_coherence_ratio": exact_coherence,
                "exact_pole_a_eff_rank": exact_eff_rank,
                "cos_fl2va_ref2va_1d": cos_fl_ref,
                "vr_grassmann_overlap_fl2va_ref2va": vr_overlap_fl_ref,
                "v1d_top_words": [x["word"] for x in v1d_ro.get("positive_words", ())[:6]],
                "mu_d_top_words": [x["word"] for x in er_ro.get("mu_d_positive_words", ())[:6]],
                "rescued_words": er_ro.get("rescued_words", []),
                "erased_pole_a_words": er_ro.get("erased_pole_a_words", []),
            }
        )

    summary = {
        "n_concepts": len(records),
        "n_safetensors_checked": len(fl_files) + len(ref_files),
        "regime_counts": {
            r: sum(1 for x in records if x["regime"] == r)
            for r in ("multi_facet_bundle", "intermediate", "monolithic_override")
        },
        "median_mpc": round(float(np.median([x["mpc"] for x in records])), 4),
        "median_cos_mu_d_v1d": round(float(np.median([x["cos_mu_d_v1d"] for x in records])), 4),
        "min_cos_mu_d_v1d": min((x["cos_mu_d_v1d"], x["concept"]) for x in records),
        "median_cos_fl2va_ref2va_1d": round(
            float(np.median([x["cos_fl2va_ref2va_1d"] for x in records if x["cos_fl2va_ref2va_1d"] is not None])),
            4,
        ),
        "median_vr_grassmann_overlap_fl2va_ref2va": round(
            float(
                np.median(
                    [x["vr_grassmann_overlap_fl2va_ref2va"] for x in records if x["vr_grassmann_overlap_fl2va_ref2va"] is not None]
                )
            ),
            4,
        ),
        "concepts_with_rescued_words": [
            {"concept": x["concept"], "cos_mu_d_v1d": x["cos_mu_d_v1d"], "rescued": x["rescued_words"]}
            for x in records
            if x["rescued_words"]
        ],
        "records": records,
    }
    out_p = DATA_OUT / "census_38_bank.json"
    out_p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[census] Checked {summary['n_concepts']} concepts ({summary['n_safetensors_checked']} files): "
        f"regimes={summary['regime_counts']}, median cos(fl,ref)={summary['median_cos_fl2va_ref2va_1d']}, "
        f"median Vr Grassmann overlap(fl,ref)={summary['median_vr_grassmann_overlap_fl2va_ref2va']}, "
        f"concepts w/ rescued words={len(summary['concepts_with_rescued_words'])}/{summary['n_concepts']}"
    )
    return summary


# ---------------------------------------------------------------------------
# Helpers for Vr Variants & Tau Matching
# ---------------------------------------------------------------------------
_CPU_TOKENIZER_CACHE: dict[str, Any] = {}


def _resolve_bank_pieces(stem: str, a_spans: list[torch.Tensor]) -> list[list[str]] | None:
    """Resolve exact Qwen3-VL BPE pieces per `diff_span` pair on CPU (`< 0.3s`, zero GPU weights)."""
    import control_vector_nodes as cv_nodes
    import h3_rules

    if "tok" not in _CPU_TOKENIZER_CACHE:
        from comfy.text_encoders.minimax import MiniMaxH3Tokenizer

        t_obj = MiniMaxH3Tokenizer()
        hf_tok = t_obj.qwen3vl_32b.tokenizer
        _CPU_TOKENIZER_CACHE["tok"] = t_obj
        _CPU_TOKENIZER_CACHE["dec"] = lambda ids: hf_tok.decode(list(ids), skip_special_tokens=False)

    tok = _CPU_TOKENIZER_CACHE["tok"]
    dec_fn = _CPU_TOKENIZER_CACHE["dec"]
    toml_path: Path | None = None
    for cand_dir in (
        REPO_ROOT / "prompt_bank" / "control_vectors",
        REPO_ROOT / "data" / "control_vectors",
        REPO_ROOT / "internal" / "internal_prompt_bank" / "control_vectors_v2",
        REPO_ROOT / "internal" / "internal_prompt_bank" / "control_vectors",
    ):
        cand = cand_dir / f"{stem}.toml"
        if cand.is_file():
            toml_path = cand
            break
    if toml_path is None:
        return None
    ps = cv.load_pairs(toml_path, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    if len(ps.pairs) != len(a_spans):
        return None
    pieces_per_pair: list[list[str]] = []
    for pair, ap in zip(ps.pairs, a_spans):
        ea = cv_nodes._extract_token_entries(tok.tokenize_with_weights(h3_rules.normalize_prompt(pair.a)))
        eb = cv_nodes._extract_token_entries(tok.tokenize_with_weights(h3_rules.normalize_prompt(pair.b)))
        ids_a = [int(t[0]) for t in ea]
        ids_b = [int(t[0]) for t in eb]
        fd, end_a, _end_b = h3_spans.differing_token_bounds(ids_a, ids_b)
        sp_a = h3_spans.spans(ids_a, [1] * len(ids_a), dec_fn)
        ma = h3_spans.pooling_mask(sp_a, "diff_span", first_diff=min(fd, len(ids_a) - 1), end_diff=end_a, other_ids=ids_b)
        pcs = [dec_fn([tid]) for tid, keep in zip(ids_a, ma.tolist()) if keep]
        if len(pcs) != int(ap.shape[0]):
            return None
        pieces_per_pair.append(pcs)
    return pieces_per_pair


def build_erasure_variant(
    a_spans: list[torch.Tensor],
    b_spans: list[torch.Tensor],
    *,
    a_pieces_per_pair: list[list[str]] | None = None,
    tau: float = 0.15,
    decone_sim: bool = False,
    vr_mode: str = "svd_ac",  # "svd_ac" | "skiplist_svd" | "contested_svd"
    rank: int = 8,
    ridge_rel: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Build `(v1d, mu_d, vr, diag)` under controlled tau, cosine-centering, and Vr extraction variants."""
    a_pool = torch.stack([ap.double().mean(dim=0) for ap in a_spans], dim=0)
    b_pool = torch.stack([bp.double().mean(dim=0) for bp in b_spans], dim=0)
    v1d = (b_pool - a_pool).mean(dim=0)

    a_rows_for_svd: list[torch.Tensor] = []
    d_weights_for_svd: list[torch.Tensor] = []
    d_sum = torch.zeros_like(v1d)
    n_tok_total = 0.0

    # Global mean across all A+B rows for optional de-coned similarity
    if decone_sim:
        ab_cat = torch.cat([x.double() for x in a_spans + b_spans], dim=0)
        mu_ab = ab_cat.mean(dim=0, keepdim=True)
    else:
        mu_ab = None

    for p_idx, (ap_raw, bp_raw) in enumerate(zip(a_spans, b_spans)):
        ap = ap_raw.double()
        bp = bp_raw.double()
        if math.isinf(tau):
            # tau -> inf: uniform attention weights 1 / N_B -> mean(D_p) == mean(B_p) - mean(A_p)
            b_match = bp.mean(dim=0, keepdim=True).expand_as(ap)
        else:
            ap_sim = (ap - mu_ab) if mu_ab is not None else ap
            bp_sim = (bp - mu_ab) if mu_ab is not None else bp
            au = ap_sim / torch.linalg.vector_norm(ap_sim, dim=-1, keepdim=True).clamp(min=1e-12)
            bu = bp_sim / torch.linalg.vector_norm(bp_sim, dim=-1, keepdim=True).clamp(min=1e-12)
            p_mat = torch.softmax((au @ bu.T) / float(tau), dim=-1)
            b_match = p_mat @ bp
        dp = b_match - ap
        d_sum = d_sum + dp.sum(dim=0)
        n_tok_total += float(ap.shape[0])

        keep_mask = torch.ones(ap.shape[0], dtype=torch.bool)
        if vr_mode in ("skiplist_svd", "contested_svd") and a_pieces_per_pair and p_idx < len(a_pieces_pair := a_pieces_per_pair):
            pcs = a_pieces_pair[p_idx]
            if len(pcs) == ap.shape[0]:
                for t_i, raw_pc in enumerate(pcs):
                    w_clean = raw_pc.strip().lower().strip(".,;:!?\"'()[]{}")
                    if not w_clean or not any(ch.isalnum() for ch in w_clean) or w_clean in SKIPLIST_WORDS:
                        keep_mask[t_i] = False
                if int(keep_mask.sum().item()) < 2:
                    keep_mask = torch.ones(ap.shape[0], dtype=torch.bool)

        if vr_mode == "contested_svd":
            # Pair-center A_p to strip scene-setting drift across pairs, then weight by token shift ||dp_i||
            ap_use = (ap - ap.mean(dim=0, keepdim=True))[keep_mask]
            d_nrm = torch.linalg.vector_norm(dp[keep_mask], dim=-1)
            w_row = (d_nrm / d_nrm.mean().clamp(min=1e-9)).clamp(0.25, 2.5)
            a_rows_for_svd.append(ap_use * w_row.unsqueeze(-1))
        elif vr_mode == "skiplist_svd":
            a_rows_for_svd.append(ap[keep_mask])
        else:
            a_rows_for_svd.append(ap)

    mu_d = v1d.clone() if math.isinf(tau) else (d_sum / max(n_tok_total, 1.0))
    a_cat = torch.cat(a_rows_for_svd, dim=0)
    ac = a_cat if vr_mode == "contested_svd" else (a_cat - a_cat.mean(dim=0, keepdim=True))

    if rank <= 0 or ac.shape[0] < 2:
        vr = torch.zeros((0, v1d.numel()), dtype=torch.float64)
        diag = torch.zeros((0,), dtype=torch.float64)
    else:
        _u, sv, vh = torch.linalg.svd(ac, full_matrices=False)
        r_eff = min(int(rank), int((sv > 1e-9 * sv[0]).sum().item()))
        vr = vh[:r_eff].contiguous()
        lam = float(ridge_rel) * float(sv[0].item() ** 2)
        diag = ((sv[:r_eff] ** 2) / (sv[:r_eff] ** 2 + lam)).clamp(0.0, 1.0).contiguous()

    return {
        "v1d": v1d.float(),
        "mu_d": mu_d.float(),
        "vr": vr.float(),
        "diag": diag.float(),
    }


# ---------------------------------------------------------------------------
# Stage 2: Exact Float64 CPU Math Ablations on Saved .pt Banks
# ---------------------------------------------------------------------------
def run_cpu_math() -> dict[str, Any]:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    b_jl = j_lens.load_j_lens_bundle()
    bank_pts = sorted(BANKS_DIR.glob("*.pt"))
    if not bank_pts:
        raise FileNotFoundError(f"No .pt banks in {BANKS_DIR}")

    # M1: Tau sweep + raw vs deconed cosine
    taus = [0.02, 0.05, 0.10, 0.15, 0.30, 0.75, float("inf")]
    m1_tau: dict[str, Any] = {}
    for bp in bank_pts:
        stem = bp.stem
        b_data = torch.load(bp, map_location="cpu", weights_only=False)
        a_sp = b_data["a_spans_per_pair"]
        b_sp = b_data["b_spans_per_pair"]
        rows_tau = []
        for decone in (False, True):
            for t_val in taus:
                op = build_erasure_variant(a_sp, b_sp, tau=t_val, decone_sim=decone)
                v1d_ro = j_lens.decode_vector(op["v1d"], space="post_refiner", family="fl2va", top_k=8, bundle=b_jl)
                mud_ro = j_lens.decode_vector(op["mu_d"], space="post_refiner", family="fl2va", top_k=8, bundle=b_jl)
                v1d_set = {x["word"] for x in v1d_ro["positive_words"]}
                rescued = [x["word"] for x in mud_ro["positive_words"] if x["word"] not in v1d_set]
                cos_mv = float(F.cosine_similarity(op["v1d"].double(), op["mu_d"].double(), dim=0).item())
                rows_tau.append(
                    {
                        "tau": "inf" if math.isinf(t_val) else t_val,
                        "decone": decone,
                        "cos_mu_d_v1d": round(cos_mv, 4),
                        "mu_d_norm": round(float(torch.linalg.vector_norm(op["mu_d"].double()).item()), 2),
                        "top_words": [x["word"] for x in mud_ro["positive_words"][:6]],
                        "rescued_vs_1d": rescued,
                    }
                )
        m1_tau[stem] = rows_tau

    # M2: Vr construction variants (svd_ac vs skiplist_svd vs contested_svd)
    m2_vr: dict[str, Any] = {}
    for bp in bank_pts:
        stem = bp.stem
        b_data = torch.load(bp, map_location="cpu", weights_only=False)
        a_sp = b_data["a_spans_per_pair"]
        b_sp = b_data["b_spans_per_pair"]
        a_pcs = b_data.get("a_pieces_per_pair") or _resolve_bank_pieces(stem, a_sp)
        var_map: dict[str, Any] = {}
        ops_by_v: dict[str, dict[str, torch.Tensor]] = {}
        for vm in ("svd_ac", "skiplist_svd", "contested_svd"):
            op = build_erasure_variant(a_sp, b_sp, a_pieces_per_pair=a_pcs, vr_mode=vm)
            ops_by_v[vm] = op
            er_ro = j_lens.decode_erasure_payload(
                op["v1d"], op["vr"], op["diag"], erasure_mu_d=op["mu_d"], family="fl2va", max_axes=4, bundle=b_jl
            )
            var_map[vm] = {
                "diag": er_ro["diag"],
                "erased_pole_a_words": er_ro["erased_pole_a_words"],
                "axes": [
                    {"axis": ax["axis"], "diag": ax["diag"], "erases": ax["erases_summary"]}
                    for ax in er_ro["vr_axes"]
                ],
            }
        var_map["grassmann_svd_vs_skiplist"] = round(grassmann_overlap(ops_by_v["svd_ac"]["vr"], ops_by_v["skiplist_svd"]["vr"]), 4)
        var_map["grassmann_svd_vs_contested"] = round(grassmann_overlap(ops_by_v["svd_ac"]["vr"], ops_by_v["contested_svd"]["vr"]), 4)
        m2_vr[stem] = var_map

    # M3: Monte-Carlo Pair-Count Pole-A Coverage Curve
    rng = np.random.default_rng(20261121)
    m3_coverage: dict[str, Any] = {}
    for stem in ("noir", "paper_origami", "chaos_panic", "wes_anderson_symmetry"):
        bp = BANKS_DIR / f"{stem}.pt"
        if not bp.is_file():
            continue
        b_data = torch.load(bp, map_location="cpu", weights_only=False)
        a_sp = b_data["a_spans_per_pair"]
        b_sp = b_data["b_spans_per_pair"]
        n_total = len(a_sp)
        full_op = build_erasure_variant(a_sp, b_sp)
        curve = []
        for k_sub in sorted({1, 2, 3, 5, 8, n_total}):
            if k_sub > n_total:
                continue
            n_trials = 1 if k_sub == n_total else 24
            overlaps, mud_coss = [], []
            for _ in range(n_trials):
                idxs = list(range(n_total)) if k_sub == n_total else sorted(rng.choice(n_total, size=k_sub, replace=False).tolist())
                sub_op = build_erasure_variant([a_sp[i] for i in idxs], [b_sp[i] for i in idxs])
                overlaps.append(grassmann_overlap(sub_op["vr"], full_op["vr"]))
                mud_coss.append(float(F.cosine_similarity(sub_op["mu_d"].double(), full_op["mu_d"].double(), dim=0).item()))
            curve.append(
                {
                    "n_pairs": k_sub,
                    "vr_grassmann_overlap_mean": round(float(np.mean(overlaps)), 4),
                    "vr_grassmann_overlap_std": round(float(np.std(overlaps)), 4),
                    "cos_mu_d_full_mean": round(float(np.mean(mud_coss)), 4),
                }
            )
        m3_coverage[stem] = {"n_total_pairs": n_total, "curve": curve}

    out = {"m1_tau_anisotropy": m1_tau, "m2_vr_variants": m2_vr, "m3_pair_coverage_curve": m3_coverage}
    out_p = DATA_OUT / "cpu_math_ablations.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[cpu_math] Saved {out_p.name} across {len(bank_pts)} banks.")
    return out


# ---------------------------------------------------------------------------
# GPU Sampling Helpers (Stills, 90f/192f AV Video, Audio Metrics, Base-50)
# ---------------------------------------------------------------------------
def apply_erasure_shift_rows(
    rows: torch.Tensor,
    op: dict[str, torch.Tensor],
    strength: float,
    *,
    erase_active: bool = True,
) -> torch.Tensor:
    """Apply `add_norm_matched_erasure` (or 1D when `erase_active=False` / `vr` empty) matched to `|s| * 67.5`."""
    if abs(strength) < 1e-9:
        return torch.zeros_like(rows)
    target_rms = abs(strength) * NORM_TARGET
    vr = op.get("vr")
    diag = op.get("diag")
    has_er = erase_active and vr is not None and diag is not None and vr.numel() > 0 and strength > 0.0 and rows.shape[0] > 1
    base_v = op["mu_d"] if (has_er and "mu_d" in op) else op["v1d"]
    b64 = base_v.double()
    b_norm = b64 * (NORM_TARGET / float(torch.linalg.vector_norm(b64).clamp(min=1e-12).item()))
    if not has_er:
        sh = strength * b_norm.unsqueeze(0).expand_as(rows)
        return sh * (target_rms / rms_row_norm(sh))

    r64 = rows.double()
    centered = r64 - r64.mean(dim=0, keepdim=True)
    vr64 = vr.double()
    d64 = diag.double().clamp(0.0, 1.0)
    e_own = ((centered @ vr64.T) * d64.unsqueeze(0)) @ vr64
    e_rms = rms_row_norm(e_own)
    if e_rms < 1e-9:
        sh = strength * b_norm.unsqueeze(0).expand_as(rows)
        return sh * (target_rms / rms_row_norm(sh))
    e_cap = min(1.0, NORM_TARGET / e_rms)
    clamp_s = max(0.0, min(float(strength), 1.0))
    raw = strength * b_norm.unsqueeze(0) - clamp_s * e_cap * e_own
    return raw * (target_rms / rms_row_norm(raw))


def _sol_execute_kwargs() -> dict[str, Any]:
    """Unflatten `h3_config.SOL_RECOMMENDED_CUDA` DynamicCombo keys for direct `MiniMaxH3Sol.execute(...)`."""
    raw = dict(h3_config.SOL_RECOMMENDED_CUDA)
    tr_mode = raw.pop("token_routing", "off")
    tr_dict: dict[str, Any] = {"token_routing": tr_mode}
    rows_mode = raw.pop("rows", "per segment")
    rows_dict: dict[str, Any] = {"rows": rows_mode}
    for k in list(raw.keys()):
        if k.startswith("token_routing."):
            tr_dict[k.split(".", 1)[1]] = raw.pop(k)
        elif k.startswith("rows."):
            rows_dict[k.split(".", 1)[1]] = raw.pop(k)
    raw["token_routing"] = tr_dict
    raw["rows"] = rows_dict
    return raw


def _load_pika_fl2va_model(*, with_lora: bool = True, shift_v: float = 12.0, shift_a: float = 3.0):
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    import control_vector_nodes as cv_nodes

    _wait_for_comfy_idle_and_free()
    lora_branch = cv_nodes._repo("lora_branch")
    vae_precision = cv_nodes._repo("vae_precision")

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_fl2va"])
    model = comfy.sd.load_diffusion_model(unet_path)
    if with_lora:
        (model,) = lora_branch.MiniMaxH3LoRABranch.execute(
            model=model, lora_name=h3_config.PIKA_FL2VA_SVD995_LORA, strength=1.0
        )
    (model,) = MiniMaxH3SigmaShift.execute(model=model, shift_video=shift_v, shift_audio=shift_a)
    (model,) = ModelAttentionBackend.execute(model=model, attention="comfy kitchen attention")

    vae_path = folder_paths.get_full_path_or_raise("vae", h3_config.IMAGE_GRAPH_VAE)
    img_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
    (img_vae,) = vae_precision.MiniMaxH3VAEPrecision.execute(vae=img_vae, encoder="fp32", decoder="unchanged")
    return model, img_vae


def _load_nonmarket_prompts() -> dict[str, dict[str, Any]]:
    """Load `indoor_neutral` + encode/cache non-market `sitcom_kitchen` & `hardware_aisle` prompts."""
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    cache_pt = DATA_OUT / "nonmarket_prompts.pt"
    out: dict[str, dict[str, Any]] = {
        "indoor_neutral": torch.load(PROMPTS_DIR / "indoor_neutral.pt", map_location="cpu", weights_only=False),
    }
    if cache_pt.is_file():
        cached = torch.load(cache_pt, map_location="cpu", weights_only=False)
        out.update(cached)
        return out

    import control_vector_nodes as cv_nodes

    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    print("[prompts] Encoding non-market prompts (t2va_sitcom_kitchen, t2va_hardware_aisle_short)...")
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    fresh: dict[str, dict[str, Any]] = {}
    for key, fname in [
        ("sitcom_kitchen", "t2va_sitcom_kitchen.txt"),
        ("hardware_aisle", "t2va_hardware_aisle_short.txt"),
    ]:
        txt = (REPO_ROOT / "prompt_bank" / fname).read_text(encoding="utf-8").strip()
        enc_2d, tags_1d, tok_ids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, txt)
        pieces = [dec_fn([int(tid)]) for tid in tok_ids]
        sp_m = h3_spans.spans(tok_ids, tags_1d.tolist(), dec_fn)
        fresh[key] = {
            "name": key,
            "prompt": txt,
            "enc": enc_2d.detach().cpu().to(torch.float32),
            "tags": tags_1d.detach().cpu().to(torch.int64),
            "tok_ids": list(tok_ids),
            "pieces": pieces,
            "scene_mask": h3_spans.inference_mask(sp_m, "scene_only").detach().cpu(),
            "sound_mask": h3_spans.inference_mask(sp_m, "soundscape_only").detach().cpu(),
        }
    del clip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.save(fresh, cache_pt)
    out.update(fresh)
    return out


def _refine_context_if_needed(executor, context: torch.Tensor, transformer_options: dict[str, Any]) -> torch.Tensor:
    if context.ndim == 3 and context.shape[-1] != cv.POST_REFINER_DIM:
        dm_exec = getattr(getattr(executor, "__self__", None), "diffusion_model", None)
        if dm_exec is not None and hasattr(dm_exec, "token_refiner") and hasattr(dm_exec, "condition_proj"):
            with torch.no_grad():
                return torch.stack(
                    [
                        dm_exec.token_refiner(
                            dm_exec.condition_proj(context[b_idx]),
                            transformer_options=transformer_options,
                        )
                        for b_idx in range(int(context.shape[0]))
                    ],
                    dim=0,
                )
    return context


def _sample_still_pika(
    model,
    img_vae,
    pr_dict: dict[str, Any],
    *,
    op: dict[str, torch.Tensor] | None,
    strength: float,
    seed: int,
    sigmas: torch.Tensor,
    sampler_name: str = "lcm",
    erase_window: tuple[float, float] = (0.0, 1.0),
    add_window: tuple[float, float] = (0.0, 1.0),
    width: int = 1152,
    height: int = 768,
) -> Image.Image:
    import comfy.nested_tensor as nt
    from comfy.patcher_extension import WrappersMP
    import comfy.sample
    import comfy.samplers

    enc_3d = pr_dict["enc"].to(torch.bfloat16).unsqueeze(0)
    tags_2d = pr_dict["tags"].unsqueeze(0)
    sm = pr_dict["scene_mask"]
    m_run = model
    n_steps = max(len(sigmas) - 1, 1)

    if op is not None and abs(strength) > 1e-9:
        m_run = model.clone()
        step_counter = {"idx": 0, "last_t": None}

        def _dm_wrap(executor, x, timestep, context, transformer_options={}, **kwargs):
            t_cur = round(float(timestep[0].flatten()[0].item() if isinstance(timestep, (list, tuple)) else timestep.flatten()[0].item()), 6)
            if step_counter["last_t"] is None:
                step_counter["idx"] = 0
            elif t_cur != step_counter["last_t"]:
                step_counter["idx"] += 1
            step_counter["last_t"] = t_cur
            pct = step_counter["idx"] / float(n_steps)
            in_add = add_window[0] <= pct < add_window[1]
            in_er = erase_window[0] <= pct < erase_window[1]
            ctx_use = context
            if in_add and context is not None:
                ctx_ref = _refine_context_if_needed(executor, context, transformer_options)
                ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                rows = ref64[sm]
                sh = apply_erasure_shift_rows(rows, op, strength, erase_active=in_er)
                out = ref64.clone()
                out[sm] = rows + sh
                ctx_use = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
            return executor(x, timestep, ctx_use, transformer_options, **kwargs)

        m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["eval_bench"] = [_dm_wrap]

    lat_h, lat_w = height // 16, width // 16
    nested = nt.NestedTensor((torch.zeros((1, 24, 1, lat_h, lat_w)), torch.zeros((1, 32, 2, 2))))
    noise = comfy.sample.prepare_noise(nested, seed)
    guider = comfy.samplers.CFGGuider(m_run)
    guider.inner_set_conds({"positive": [[enc_3d, {"minimax_token_tags": tags_2d.to(torch.long)}]]})
    guider.set_cfg(1.0)
    samples = guider.sample(
        noise, nested, comfy.samplers.sampler_object(sampler_name), sigmas,
        denoise_mask=None, callback=None, disable_pbar=True, seed=seed,
    )
    vid_lat = (samples.unbind()[0] if isinstance(samples, nt.NestedTensor) else samples).detach()
    decoded = img_vae.decode(vid_lat)
    if decoded.ndim == 5:
        decoded = decoded[:, 0]
    arr = np.clip(decoded[0].detach().cpu().float().numpy() * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Stage 3: GPU Controlled Stills Ablations 1-5
# ---------------------------------------------------------------------------
def run_ablations() -> dict[str, Any]:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, _ = _out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    prompts = _load_nonmarket_prompts()
    banks = {
        cn: torch.load(BANKS_DIR / f"{cn}.pt", map_location="cpu", weights_only=False)
        for cn in ("noir", "paper_origami", "watercolor_wash")
    }

    print("[ablations] Loading Pika FL2VA SVD995 + PH3_IMAGE_VAE...")
    model, img_vae = _load_pika_fl2va_model(with_lora=True)
    results: dict[str, Any] = {}

    # Ablation 1: Tau sweep + raw vs deconed on noir (3 seeds) & paper_origami (seed main)
    print("[Ablation 1] Running tau_attn & decone sweep...")
    tau_arms = [
        ("1D (tau=inf)", float("inf"), False),
        ("tau=0.50", 0.50, False),
        ("tau=0.15 (shipped)", 0.15, False),
        ("tau=0.07", 0.07, False),
        ("MaxSim (tau=0.03)", 0.03, False),
        ("tau=0.15 deconed", 0.15, True),
    ]
    ab1_out: list[dict[str, Any]] = []
    sheet1_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for sd in SEEDS_3:
        cols = []
        for label, t_v, dec in tau_arms:
            op = build_erasure_variant(
                banks["noir"]["a_spans_per_pair"],
                banks["noir"]["b_spans_per_pair"],
                tau=t_v,
                decone_sim=dec,
                rank=(0 if math.isinf(t_v) else 8),
            )
            im = _sample_still_pika(model, img_vae, prompts["indoor_neutral"], op=op, strength=1.5, seed=sd, sigmas=pika_sigmas)
            sat = round(hsv_sat(im), 4)
            ab1_out.append({"concept": "noir", "seed": sd, "arm": label, "hsv_sat": sat, "is_bw": sat < 0.06})
            cols.append((f"{label} S={sat:.3f}", im))
        sheet1_rows.append((f"noir s={sd}", cols))
    make_contact_sheet(sheet1_rows, img_dir / "ablation1_tau_sweep_noir.png")
    results["ablation1_tau_sweep"] = ab1_out

    # Ablation 2: Vr construction (svd_ac vs skiplist_svd vs contested_svd) on fl2va
    print("[Ablation 2] Running Vr construction variants (svd_ac vs skiplist vs contested)...")
    ab2_out: list[dict[str, Any]] = []
    sheet2_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for cn, pn in [("noir", "indoor_neutral"), ("paper_origami", "sitcom_kitchen")]:
        b_d = banks[cn]
        a_pcs_cn = b_d.get("a_pieces_per_pair") or _resolve_bank_pieces(cn, b_d["a_spans_per_pair"])
        for sd in SEEDS_3[:2]:
            cols = []
            im_base = _sample_still_pika(model, img_vae, prompts[pn], op=None, strength=0.0, seed=sd, sigmas=pika_sigmas)
            cols.append((f"Base S={hsv_sat(im_base):.2f}", im_base))
            for vm_label, vm_key, r_k in [
                ("1D (r=0)", "svd_ac", 0),
                ("svd_ac (r=8)", "svd_ac", 8),
                ("skiplist_svd (r=8)", "skiplist_svd", 8),
                ("contested_svd (r=8)", "contested_svd", 8),
            ]:
                op = build_erasure_variant(
                    b_d["a_spans_per_pair"],
                    b_d["b_spans_per_pair"],
                    a_pieces_per_pair=a_pcs_cn,
                    vr_mode=vm_key,
                    rank=r_k,
                )
                im = _sample_still_pika(model, img_vae, prompts[pn], op=op, strength=1.5, seed=sd, sigmas=pika_sigmas)
                sat = round(hsv_sat(im), 4)
                mae_base = round(float(np.abs(np.asarray(im, dtype=np.float32) - np.asarray(im_base, dtype=np.float32)).mean()), 2)
                ab2_out.append({"concept": cn, "seed": sd, "arm": vm_label, "hsv_sat": sat, "mae_vs_base": mae_base})
                cols.append((f"{vm_label} S={sat:.3f}", im))
            sheet2_rows.append((f"{cn[:10]} s={sd}", cols))
    make_contact_sheet(sheet2_rows, img_dir / "ablation2_vr_variants_fl2va.png")
    results["ablation2_vr_variants_fl2va"] = ab2_out

    # Ablation 3: Clean rank sweep r in [0, 1, 2, 4, 8, 16] & ridge_rel in [0.01, 0.15, 0.50] on S1_own_Vr
    print("[Ablation 3] Running clean rank r & ridge lambda sweep on S1_own_Vr...")
    ab3_out: list[dict[str, Any]] = []
    sheet3_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for cn, pn in [("noir", "indoor_neutral"), ("paper_origami", "sitcom_kitchen"), ("watercolor_wash", "hardware_aisle")]:
        b_d = banks[cn]
        cols_r = []
        for r_k in [0, 1, 2, 4, 8, 16]:
            op = build_erasure_variant(b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], rank=r_k, ridge_rel=0.15)
            im = _sample_still_pika(model, img_vae, prompts[pn], op=op, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)
            sat = round(hsv_sat(im), 4)
            ab3_out.append({"concept": cn, "sweep": "rank", "rank": r_k, "ridge_rel": 0.15, "hsv_sat": sat})
            cols_r.append((f"r={r_k} S={sat:.3f}", im))
        sheet3_rows.append((f"{cn[:12]} rank", cols_r))
        if cn in ("noir", "paper_origami"):
            cols_l = []
            for lam_r in [0.01, 0.05, 0.15, 0.50, 1.50]:
                op = build_erasure_variant(b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], rank=8, ridge_rel=lam_r)
                im = _sample_still_pika(model, img_vae, prompts[pn], op=op, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)
                sat = round(hsv_sat(im), 4)
                ab3_out.append({"concept": cn, "sweep": "ridge", "rank": 8, "ridge_rel": lam_r, "hsv_sat": sat})
                cols_l.append((f"lam={lam_r} S={sat:.3f}", im))
            sheet3_rows.append((f"{cn[:12]} lam", cols_l))
    make_contact_sheet(sheet3_rows, img_dir / "ablation3_clean_rank_and_ridge.png")
    results["ablation3_rank_and_ridge"] = ab3_out

    # Ablation 4: Pair-count Pole-A coverage N in [1, 2, 5, all]
    print("[Ablation 4] Running pair-count coverage renders...")
    ab4_out: list[dict[str, Any]] = []
    sheet4_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for cn, pn in [("noir", "indoor_neutral"), ("paper_origami", "sitcom_kitchen")]:
        b_d = banks[cn]
        a_sp, b_sp = b_d["a_spans_per_pair"], b_d["b_spans_per_pair"]
        cols_n = []
        for n_sub in [1, 2, 5, len(a_sp)]:
            op = build_erasure_variant(a_sp[:n_sub], b_sp[:n_sub], rank=8)
            im = _sample_still_pika(model, img_vae, prompts[pn], op=op, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)
            sat = round(hsv_sat(im), 4)
            ab4_out.append({"concept": cn, "n_pairs": n_sub, "erasure_rank": int(op["vr"].shape[0]), "hsv_sat": sat})
            cols_n.append((f"N={n_sub} (r={op['vr'].shape[0]}) S={sat:.3f}", im))
        sheet4_rows.append((f"{cn[:12]} N-pairs", cols_n))
    make_contact_sheet(sheet4_rows, img_dir / "ablation4_pair_count_coverage.png")
    results["ablation4_pair_count"] = ab4_out

    # Ablation 5: Denoising step window decoupling
    print("[Ablation 5] Running denoising step-window decoupling...")
    ab5_out: list[dict[str, Any]] = []
    sheet5_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    win_arms = [
        ("1D [0..1]", 0, (0.0, 1.0), (0.0, 1.0)),
        ("Both [0..1] (default)", 8, (0.0, 1.0), (0.0, 1.0)),
        ("Erase [0..0.34] + mu_d [0..1]", 8, (0.0, 0.34), (0.0, 1.0)),
        ("Erase [0..0.50] + mu_d [0..1]", 8, (0.0, 0.51), (0.0, 1.0)),
        ("Both [0..0.50]", 8, (0.0, 0.51), (0.0, 0.51)),
        ("Late Erase [0.34..1]", 8, (0.34, 1.0), (0.34, 1.0)),
    ]
    for cn, pn in [("noir", "indoor_neutral"), ("paper_origami", "sitcom_kitchen")]:
        b_d = banks[cn]
        cols_w = []
        for w_lbl, r_k, er_w, add_w in win_arms:
            op = build_erasure_variant(b_d["a_spans_per_pair"], b_d["b_spans_per_pair"], rank=r_k)
            im = _sample_still_pika(
                model, img_vae, prompts[pn], op=op, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas,
                erase_window=er_w, add_window=add_w,
            )
            sat = round(hsv_sat(im), 4)
            rx = round(red_excess(im), 2)
            ab5_out.append({"concept": cn, "arm": w_lbl, "hsv_sat": sat, "red_excess": rx})
            cols_w.append((f"{w_lbl[:22]} S={sat:.3f}", im))
        sheet5_rows.append((f"{cn[:12]} window", cols_w))
    make_contact_sheet(sheet5_rows, img_dir / "ablation5_step_window_decoupling.png")
    results["ablation5_step_windows"] = ab5_out

    del model, img_vae
    gc.collect()
    torch.cuda.empty_cache()

    out_p = DATA_OUT / "gpu_ablations_1_to_5.json"
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[ablations] Saved {out_p.name} and 5 contact sheets under {img_dir.name}.")
    return results


# ---------------------------------------------------------------------------
# Stage 4: WordVector (Model2Vec + against-SVD) & MakeMedia (ColQwen Patch)
# ---------------------------------------------------------------------------
def run_word_media() -> dict[str, Any]:
    import comfy.sd
    import folder_paths
    import control_vector_nodes as cv_nodes
    from bench.control_vector import helpers as _h

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, _ = _out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)
    prompts = _load_nonmarket_prompts()
    banks = {cn: torch.load(BANKS_DIR / f"{cn}.pt", map_location="cpu", weights_only=False) for cn in ("noir", "paper_origami")}

    # First build a real <Picture 1> media-contrast vector pair (Ablation 7) while encoder is loaded
    print("[word_media] Building <Picture 1> visual media-contrast vector (ColQwen patch erasure)...")
    model, img_vae = _load_pika_fl2va_model(with_lora=True)
    im_sunlit = _sample_still_pika(model, img_vae, prompts["indoor_neutral"], op=None, strength=0.0, seed=SEED_MAIN, sigmas=pika_sigmas)
    op_noir_gpu = build_erasure_variant(banks["noir"]["a_spans_per_pair"], banks["noir"]["b_spans_per_pair"], rank=8)
    im_noir_ref = _sample_still_pika(model, img_vae, prompts["indoor_neutral"], op=op_noir_gpu, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    arr_a = torch.from_numpy(np.asarray(im_sunlit, dtype=np.float32) / 255.0).unsqueeze(0)
    arr_b = torch.from_numpy(np.asarray(im_noir_ref, dtype=np.float32) / 255.0).unsqueeze(0)
    _wait_for_comfy_idle_and_free()
    clip = _h._load_clip(h3_config.MODELS["clip"], device="default")
    dit_fl = Path(folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_fl2va"]))
    loras_fl = _h._parse_lora_specs([f"{h3_config.PIKA_FL2VA_SVD995_LORA}:1.0"])

    media_scene_prompt = (
        "subject_definitions:\n<Subject 1> is the person in <Picture 1>.\n\n"
        "summary:\n[reference generation] <Subject 1> stands in a room.\n\n"
        "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - retained.\n\n"
        "detailed_description:\nThe target video captures the optical look and lighting of <Picture 1>.\n"
        "[Shot 1] Medium shot of <Subject 1> standing beside a table in an apartment room. <Subject 1> produces no vocal sound.\n\n"
        "overall_soundscape:\nQuiet room tone.\n\nnon_diegetic_music:\nN/A"
    )
    with tempfile.TemporaryDirectory() as tmp_d:
        media_vec_path, media_rep = cv_nodes.make_control_vector_from_media_pairs(
            image_a=arr_a,
            image_b=arr_b,
            model=dit_fl,
            clip=clip,
            output_name="media_noir_colqwen",
            pooling="all_rows",
            short_edge=512,
            scene_prompt=media_scene_prompt,
            output_dir=tmp_d,
            overwrite=True,
            loras=loras_fl,
            refiner_device="cuda",
        )
        media_tensors, _ = cv.load_vector(media_vec_path)

    import comfy.model_management as comfy_mm

    del clip
    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    torch.cuda.empty_cache()

    # Now reload Pika FL2VA to evaluate Ablation 6 (WordVector) & Ablation 7 (MakeMedia)
    model, _ = _load_pika_fl2va_model(with_lora=True)

    # Ablation 6: WordVector (1D vs Model2Vec + against-SVD erasure vs 10-pair GPU)
    print("[Ablation 6] Evaluating MiniMaxH3WordVector (1D vs Model2Vec+Erasure vs 10-pair GPU)...")
    word_cases = [
        (
            "noir",
            "indoor_neutral",
            "noir, chiaroscuro, monochrome, silver gelatin, 35mm",
            "@replace_digital, sunlit, warm, daylight, tungsten, color",
        ),
        (
            "paper_origami",
            "sitcom_kitchen",
            "paper origami, folded paper, papercraft",
            "@replace_live_action, realistic, photograph, human, flesh",
        ),
    ]
    ab6_out: list[dict[str, Any]] = []
    sheet6_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for cn, pn, pos_w, neg_w in word_cases:
        synth = j_lens.synthesize_from_words(pos_w, neg_w, family="fl2va")
        op_w_1d = {"v1d": synth["vector.post_refiner"], "mu_d": synth["vector.post_refiner"], "vr": torch.zeros((0, 5376)), "diag": torch.zeros((0,))}
        op_w_er = {
            "v1d": synth["vector.post_refiner"],
            "mu_d": synth["erasure.mu_d"],
            "vr": synth["erasure.vr"],
            "diag": synth["erasure.diag"],
        }
        op_gpu = build_erasure_variant(banks[cn]["a_spans_per_pair"], banks[cn]["b_spans_per_pair"], rank=8)

        cols = []
        im_b = _sample_still_pika(model, img_vae, prompts[pn], op=None, strength=0.0, seed=SEED_MAIN, sigmas=pika_sigmas)
        cols.append((f"Base S={hsv_sat(im_b):.3f}", im_b))
        for lbl, op_use in [
            ("WordVec 1D", op_w_1d),
            ("WordVec Model2Vec+Erase", op_w_er),
            ("10-Pair GPU+Erase", op_gpu),
        ]:
            im_c = _sample_still_pika(model, img_vae, prompts[pn], op=op_use, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)
            sat = round(hsv_sat(im_c), 4)
            ab6_out.append(
                {
                    "concept": cn,
                    "arm": lbl,
                    "used_model2vec_lexicon": synth.get("used_model2vec_lexicon", False),
                    "hsv_sat": sat,
                }
            )
            cols.append((f"{lbl} S={sat:.3f}", im_c))
        sheet6_rows.append((f"{cn[:12]} WordVec", cols))
    make_contact_sheet(sheet6_rows, img_dir / "ablation6_wordvector_model2vec_erasure.png")

    # Ablation 7: MakeMediaControlVector (1D media vs ColQwen patch erasure) on unseen sitcom_kitchen!
    print("[Ablation 7] Evaluating MakeMediaControlVector (1D vs ColQwen patch erasure) on unseen sitcom_kitchen...")
    op_med_1d = {"v1d": media_tensors["vector.post_refiner"], "mu_d": media_tensors["vector.post_refiner"], "vr": torch.zeros((0, 5376)), "diag": torch.zeros((0,))}
    op_med_er = {
        "v1d": media_tensors["vector.post_refiner"],
        "mu_d": media_tensors["erasure.mu_d"],
        "vr": media_tensors["erasure.vr"],
        "diag": media_tensors["erasure.diag"],
    }
    ab7_out: list[dict[str, Any]] = []
    sheet7_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = [
        ("Ref Pair (<Pic 1>)", [("image_a (sunlit)", im_sunlit), ("image_b (noir)", im_noir_ref)])
    ]
    for sd in SEEDS_3[:2]:
        cols_m = []
        im_mb = _sample_still_pika(model, img_vae, prompts["sitcom_kitchen"], op=None, strength=0.0, seed=sd, sigmas=pika_sigmas)
        cols_m.append((f"Unseen Kitchen Base S={hsv_sat(im_mb):.3f}", im_mb))
        for m_lbl, m_op in [("MakeMedia 1D", op_med_1d), ("MakeMedia ColQwen+Erase", op_med_er)]:
            im_m = _sample_still_pika(model, img_vae, prompts["sitcom_kitchen"], op=m_op, strength=1.5, seed=sd, sigmas=pika_sigmas)
            sat = round(hsv_sat(im_m), 4)
            ab7_out.append({"seed": sd, "arm": m_lbl, "hsv_sat": sat})
            cols_m.append((f"{m_lbl} S={sat:.3f}", im_m))
        sheet7_rows.append((f"Kitchen s={sd}", cols_m))
    make_contact_sheet(sheet7_rows, img_dir / "ablation7_makemedia_colqwen_erasure.png")

    del model, img_vae
    gc.collect()
    torch.cuda.empty_cache()

    res = {
        "ablation6_wordvector": ab6_out,
        "ablation7_makemedia": {
            "erasure_rank": int(media_tensors["erasure.vr"].shape[0]),
            "erasure_diag": [round(float(x), 3) for x in media_tensors["erasure.diag"].tolist()],
            "renders": ab7_out,
        },
    }
    out_p = DATA_OUT / "gpu_word_and_media_ablations.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[word_media] Saved {out_p.name}.")
    return res


# ---------------------------------------------------------------------------
# Audio Metrics & Video Mux Helper
# ---------------------------------------------------------------------------
def mux_and_measure_av(
    vid_frames_u8: np.ndarray,
    audio_waveform: torch.Tensor,
    sample_rate: int,
    mp4_path: Path,
    strip_path: Path,
    *,
    fps: int = 24,
) -> dict[str, float]:
    """Write `.mp4` (H.264 + AAC) + 6-frame `.png` strip and compute objective video/audio metrics."""
    import wave

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    strip_path.parent.mkdir(parents=True, exist_ok=True)
    n_f, h, w, _ = vid_frames_u8.shape

    # Save 6-frame review strip
    idxs = np.linspace(0, n_f - 1, min(6, n_f), dtype=int).tolist()
    tw = 288
    th = int(round(h * (tw / w)))
    strip = Image.new("RGB", (tw * len(idxs), th + 24), (16, 16, 20))
    dr = ImageDraw.Draw(strip)
    for ci, fi in enumerate(idxs):
        dr.text((ci * tw + 6, 4), f"f={fi} ({fi / fps:.2f}s)", fill=(230, 235, 255))
        strip.paste(Image.fromarray(vid_frames_u8[fi]).resize((tw, th), Image.Resampling.LANCZOS), (ci * tw, 24))
    strip.save(strip_path)

    # Video saturation & temporal smoothness
    sats = [hsv_sat(Image.fromarray(vid_frames_u8[i])) for i in range(0, n_f, 4)]
    diffs = np.abs(vid_frames_u8[1:].astype(np.float32) - vid_frames_u8[:-1].astype(np.float32)).mean()

    # Audio waveform [B, C, T] or [C, T]
    wav = audio_waveform.detach().cpu().float()
    if wav.ndim == 3:
        wav = wav[0]
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    wav_np = np.clip(wav.numpy(), -1.0, 1.0)
    mono = wav_np.mean(axis=0)
    rms_all = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    dbfs = round(20.0 * math.log10(max(rms_all, 1e-7)), 2)

    # Spectral centroid (Hz) & late-vs-peak envelope tail ratio (reverb/sustain indicator)
    spec = np.abs(np.fft.rfft(mono))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / float(sample_rate))
    centroid_hz = float((freqs * spec).sum() / max(spec.sum(), 1e-9))
    win = max(int(0.10 * sample_rate), 1)
    n_env = len(mono) // win
    if n_env >= 4:
        env = np.array([np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2) + 1e-12) for i in range(n_env)])
        p90 = float(np.percentile(env, 90))
        p25 = float(np.percentile(env, 25))
        tail_sustain_ratio = round(p25 / max(p90, 1e-7), 4)
    else:
        tail_sustain_ratio = 0.0

    with tempfile.TemporaryDirectory() as td:
        raw_vid = Path(td) / "vid.rgb"
        raw_vid.write_bytes(vid_frames_u8.tobytes())
        wav_p = Path(td) / "aud.wav"
        pcm16 = (wav_np.T * 32767.0).astype(np.int16)
        with wave.open(str(wav_p), "wb") as wf:
            wf.setnchannels(int(wav_np.shape[0]))
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", str(raw_vid),
                "-i", str(wav_p),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest",
                str(mp4_path),
            ],
            check=True,
        )

    return {
        "sat_mean": round(float(np.mean(sats)), 4),
        "sat_std": round(float(np.std(sats)), 4),
        "temp_l1_mean": round(float(diffs), 3),
        "audio_dbfs": dbfs,
        "spectral_centroid_hz": round(centroid_hz, 1),
        "tail_sustain_ratio": tail_sustain_ratio,
    }


def _clean_sif_weights(rows: torch.Tensor, mu_c: torch.Tensor, pieces: list[str], mask: torch.Tensor) -> torch.Tensor:
    ru = rows.double() / torch.linalg.vector_norm(rows.double(), dim=-1, keepdim=True).clamp(min=1e-12)
    mu_u = mu_c.double() / torch.linalg.vector_norm(mu_c.double()).clamp(min=1e-12)
    dist = (1.0 - (ru @ mu_u).clamp(-1.0, 1.0)).clamp(min=1e-4)
    w = dist.pow(0.75)
    act_idxs = torch.nonzero(mask.flatten().cpu().bool(), as_tuple=False).flatten().tolist()
    for local_i, tok_i in enumerate(act_idxs):
        pc = pieces[tok_i] if tok_i < len(pieces) else ""
        if not any(ch.isalnum() for ch in pc):
            w[local_i] = 0.5 * w.mean()
    w = w / w.mean().clamp(min=1e-12)
    return w.clamp(0.35, 2.2)


# ---------------------------------------------------------------------------
# Stage 5: 192f/90f Joint AV Video (Clean SIF + M5 Fix & Audio-Stream Erasure)
# ---------------------------------------------------------------------------
def run_audio_video() -> dict[str, Any]:
    import comfy.nested_tensor as nt
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend
    import control_vector_nodes as cv_nodes

    h3_cond_mod = cv_nodes._repo("conditioning")
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    h3_lora_mod = cv_nodes._repo("lora_branch")
    h3_sol_mod = cv_nodes._repo("sol_attn_h3")

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    _, vid_dir = _out_dirs()
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)

    # Encode 192f t2va_hardware_aisle_short & 90f t2va_sitcom_kitchen with TRUE Qwen BPE pieces!
    print("[audio_video] Encoding non-market 192f (hardware_aisle) & 90f (sitcom_kitchen) AV conditioning...")
    from comfy.patcher_extension import WrappersMP

    _wait_for_comfy_idle_and_free()
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vvae_path))
    avae_path = folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(avae_path))

    hw_txt = (REPO_ROOT / "prompt_bank" / "t2va_hardware_aisle_short.txt").read_text(encoding="utf-8").strip()
    _enc_2d, tags_1d, tok_ids, dec_fn = cv_nodes.encode_prompt_for_vector(clip, hw_txt)
    true_pieces = [dec_fn([int(tid)]) for tid in tok_ids]
    sp_masks = h3_spans.spans(tok_ids, tags_1d.tolist(), dec_fn)
    scene_mask = h3_spans.inference_mask(sp_masks, "scene_only").detach().cpu()
    non_alnum_count = sum(1 for i in torch.nonzero(scene_mask).flatten().tolist() if not any(c.isalnum() for c in true_pieces[i]))
    print(f"  Verified Qwen BPE pieces (hardware_aisle): len={len(true_pieces)}, scene_rows={int(scene_mask.sum())}, non-alnum clamped={non_alnum_count}")

    kit_txt = (REPO_ROOT / "prompt_bank" / "t2va_sitcom_kitchen.txt").read_text(encoding="utf-8").strip()
    _enc_k, tags_k, tok_ids_k, dec_fn_k = cv_nodes.encode_prompt_for_vector(clip, kit_txt)
    sp_masks_k = h3_spans.spans(tok_ids_k, tags_k.tolist(), dec_fn_k)
    kit_scene_mask = h3_spans.inference_mask(sp_masks_k, "scene_only").detach().cpu()
    kit_sound_mask = h3_spans.inference_mask(sp_masks_k, "soundscape_only").detach().cpu()

    cond_192, lat_192 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=video_vae, prompt=hw_txt, width=1152, height=768, length=192, single_frame=False
    )[:2]
    cond_90, lat_90 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=video_vae, prompt=kit_txt, width=1152, height=768, length=90, single_frame=False
    )[:2]

    import comfy.model_management as comfy_mm

    del clip
    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    torch.cuda.empty_cache()

    # Held-out PCA on training Pole-A banks (strictly zero test-prompt leakage)
    a_train = torch.cat([torch.load(p, map_location="cpu", weights_only=False)["A_all"].double() for p in sorted(BANKS_DIR.glob("*.pt"))], dim=0)
    mu_corp = a_train.mean(dim=0)
    _u_c, _s_c, vh_c = torch.linalg.svd(a_train - mu_corp.unsqueeze(0), full_matrices=False)
    pcs_2 = vh_c[:2]

    # Build Ward+WARP keys/residuals for noir
    noir_bank = torch.load(BANKS_DIR / "noir.pt", map_location="cpu", weights_only=False)
    noir_op = build_erasure_variant(noir_bank["a_spans_per_pair"], noir_bank["b_spans_per_pair"], rank=8)
    aw_all = (noir_bank["A_all"].double() - mu_corp.unsqueeze(0))
    aw_all = aw_all - (aw_all @ pcs_2.T) @ pcs_2
    aw_u = aw_all / torch.linalg.vector_norm(aw_all, dim=-1, keepdim=True).clamp(min=1e-12)
    z_link = sch.linkage(aw_u.numpy(), method="ward")
    lbls = torch.from_numpy((sch.fcluster(z_link, t=8, criterion="maxclust") - 1).astype(np.int64))
    d_matched = noir_bank["D_matched"].double()
    mu_d_noir = noir_op["mu_d"].double()
    w8_keys, w8_res = [], []
    for cid in sorted(lbls.unique().tolist()):
        m_c = lbls == cid
        ck = aw_u[m_c].mean(dim=0)
        w8_keys.append(ck / torch.linalg.vector_norm(ck).clamp(min=1e-12))
        w8_res.append(d_matched[m_c].mean(dim=0) - mu_d_noir)
    w8_keys_t = torch.stack(w8_keys, dim=0)
    w8_res_t = torch.stack(w8_res, dim=0)

    # Load Pika FL2VA + Sol for fast 192f/90f AV generation
    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_fl2va"])
    base_model = comfy.sd.load_diffusion_model(unet_path)
    (base_model,) = h3_lora_mod.MiniMaxH3LoRABranch.execute(
        model=base_model, lora_name=h3_config.PIKA_FL2VA_SVD995_LORA, strength=1.0
    )
    (base_model,) = MiniMaxH3SigmaShift.execute(model=base_model, shift_video=12.0, shift_audio=3.0)
    (base_model,) = ModelAttentionBackend.execute(model=base_model, attention="comfy kitchen attention")
    (base_model,) = h3_sol_mod.MiniMaxH3Sol.execute(model=base_model, **_sol_execute_kwargs())

    def _render_av_clip(cond_in, lat_in, shift_fn, seed: int, out_stem: str) -> dict[str, Any]:
        m_run = base_model
        if shift_fn is not None:
            m_run = base_model.clone()
            cache_ctx: dict[int, torch.Tensor] = {}

            def _dm_wrap_av(executor, x, timestep, context, transformer_options={}, **kwargs):
                ctx_use = context
                if context is not None:
                    ctx_ref = _refine_context_if_needed(executor, context, transformer_options)
                    k = id(ctx_ref)
                    if k not in cache_ctx:
                        ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                        cache_ctx[k] = shift_fn(ref64).to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
                    ctx_use = cache_ctx[k]
                return executor(x, timestep, ctx_use, transformer_options, **kwargs)

            m_run.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["eval_av"] = [_dm_wrap_av]

        lat_t = lat_in["samples"]
        noise = comfy.sample.prepare_noise(lat_t, seed)
        guider = comfy.samplers.CFGGuider(m_run)
        guider.inner_set_conds({"positive": cond_in})
        guider.set_cfg(1.0)
        samples = guider.sample(
            noise, lat_t, comfy.samplers.sampler_object("lcm"), pika_sigmas,
            denoise_mask=None, callback=None, disable_pbar=True, seed=seed,
        )
        v_lat, a_lat = samples.unbind()
        v_lat = v_lat.detach().cpu()
        a_lat = a_lat.detach().cpu()
        del samples, noise, guider, m_run
        from comfy_extras.nodes_audio import vae_decode_audio

        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        aud_obj = vae_decode_audio(audio_vae, {"samples": a_lat})
        aud_wav = aud_obj["waveform"].detach().cpu()
        aud_sr = int(aud_obj["sample_rate"])
        del aud_obj, a_lat
        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        vid_dec = video_vae.decode(v_lat)
        if vid_dec.ndim == 5:
            vid_dec = vid_dec[0]
        vid_u8 = vid_dec.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
        del vid_dec, v_lat
        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        mp4_p = vid_dir / f"{out_stem}.mp4"
        strip_p = vid_dir / "strips" / f"strip_{out_stem}.png"
        metrics = mux_and_measure_av(vid_u8, aud_wav, aud_sr, mp4_p, strip_p, fps=24)
        metrics["mp4"] = mp4_p.name
        metrics["strip"] = strip_p.name
        return metrics

    # Ablation 8a: Clean 192f AV re-run of A2_1d_clean_sif & M5_ward_warp_hybrid (with true pieces!) vs S1_own_Vr
    print("[Ablation 8a] Running clean 192f AV Qwen-BPE pieces verification for Clean SIF & M5 Ward+WARP...")
    ab8a_out: list[dict[str, Any]] = []

    def _shift_clean_sif(ref64: torch.Tensor) -> torch.Tensor:
        rows = ref64[scene_mask]
        w = _clean_sif_weights(rows, mu_corp, true_pieces, scene_mask)
        sh = 1.5 * noir_op["v1d"].double().unsqueeze(0) * w.unsqueeze(-1)
        sh = sh * ((1.5 * NORM_TARGET) / rms_row_norm(sh))
        out = ref64.clone()
        out[scene_mask] = rows + sh
        return out

    def _shift_m5_clean(ref64: torch.Tensor) -> torch.Tensor:
        rows = ref64[scene_mask]
        rw = rows - mu_corp.unsqueeze(0)
        rw = rw - (rw @ pcs_2.T) @ pcs_2
        rw_u = rw / torch.linalg.vector_norm(rw, dim=-1, keepdim=True).clamp(min=1e-12)
        sims = rw_u @ w8_keys_t.T
        top_v, top_i = torch.topk(sims, k=2, dim=-1)
        gates = torch.relu(top_v - 0.05)
        w_sp = torch.zeros_like(sims)
        w_sp.scatter_(1, top_i, (gates / gates.sum(dim=-1, keepdim=True).clamp(min=1e-6)) * top_v[:, :1].clamp(0.0, 1.0))
        raw_d = mu_d_noir.unsqueeze(0) + 1.5 * (w_sp @ w8_res_t)
        w = _clean_sif_weights(rows, mu_corp, true_pieces, scene_mask).pow(0.5 / 0.75)
        w = w / w.mean().clamp(min=1e-12)
        sh = 1.5 * raw_d * w.unsqueeze(-1)
        sh = sh * ((1.5 * NORM_TARGET) / rms_row_norm(sh))
        out = ref64.clone()
        out[scene_mask] = rows + sh
        return out

    def _shift_s1_erasure(ref64: torch.Tensor) -> torch.Tensor:
        rows = ref64[scene_mask]
        sh = apply_erasure_shift_rows(rows, noir_op, 1.5, erase_active=True)
        out = ref64.clone()
        out[scene_mask] = rows + sh
        return out

    for arm_name, fn_s in [
        ("A2_1d_clean_sif_fixed_pieces", _shift_clean_sif),
        ("M5_ward_warp_clean_sif_fixed_pieces", _shift_m5_clean),
        ("S1_pole_a_erasure", _shift_s1_erasure),
    ]:
        t_c = time.perf_counter()
        m_av = _render_av_clip(cond_192, lat_192, fn_s, SEED_MAIN, f"ab8a_192f_noir_{arm_name}")
        m_av["arm"] = arm_name
        m_av["non_alnum_clamped_tokens"] = non_alnum_count
        m_av["elapsed_s"] = round(time.perf_counter() - t_c, 1)
        ab8a_out.append(m_av)
        print(f"  [8a] {arm_name}: sat={m_av['sat_mean']:.4f}, dbfs={m_av['audio_dbfs']} ({m_av['elapsed_s']}s)")

    # Ablation 8b: 40 Hz Audio-Stream Pole-A Erasure (90f speaking sitcom_kitchen)
    print("[Ablation 8b] Running 40 Hz Audio-Stream Pole-A Erasure (reverb_cathedral, whisper_asmr, underwater_submerged)...")
    ab8b_out: list[dict[str, Any]] = []
    m_base90 = _render_av_clip(cond_90, lat_90, None, SEED_MAIN, "ab8b_90f_base")
    m_base90["concept"] = "base"
    m_base90["mode"] = "none"
    ab8b_out.append(m_base90)

    for aud_stem, mask_use, mask_name in [
        ("reverb_cathedral", kit_sound_mask, "soundscape_only"),
        ("whisper_asmr", kit_scene_mask | kit_sound_mask, "scene+soundscape"),
        ("underwater_submerged", kit_scene_mask | kit_sound_mask, "scene+soundscape"),
    ]:
        vp = cv_nodes.resolve_vector_file_path(aud_stem, preferred_family="fl2va")
        vt, _ = cv.load_vector(vp)
        op_aud = {
            "v1d": vt["vector.post_refiner"],
            "mu_d": vt["erasure.mu_d"],
            "vr": vt["erasure.vr"],
            "diag": vt["erasure.diag"],
        }
        for m_lbl, er_flag in [("add_norm_matched_1d", False), ("add_norm_matched_erasure", True)]:
            def _mk_aud_shift(op_local=op_aud, m_b=mask_use, ef=er_flag):
                def _fn(ref64: torch.Tensor) -> torch.Tensor:
                    rows = ref64[m_b]
                    sh = apply_erasure_shift_rows(rows, op_local, 1.4, erase_active=ef)
                    out = ref64.clone()
                    out[m_b] = rows + sh
                    return out
                return _fn

            m_res = _render_av_clip(cond_90, lat_90, _mk_aud_shift(), SEED_MAIN, f"ab8b_90f_{aud_stem}_{m_lbl}")
            m_res["concept"] = aud_stem
            m_res["mask"] = mask_name
            m_res["mode"] = m_lbl
            ab8b_out.append(m_res)
            print(
                f"  [8b] {aud_stem} ({m_lbl}): centroid={m_res['spectral_centroid_hz']}Hz, "
                f"tail={m_res['tail_sustain_ratio']}, sat={m_res['sat_mean']:.3f}"
            )

    del base_model, video_vae, audio_vae
    gc.collect()
    torch.cuda.empty_cache()

    res = {"ablation8a_clean_pieces_192f": ab8a_out, "ablation8b_audio_erasure_90f": ab8b_out}
    out_p = DATA_OUT / "gpu_audio_video_ablations.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[audio_video] Saved {out_p.name}.")
    return res


# ---------------------------------------------------------------------------
# Stage 6: Undistilled 50-Step Base H3 Check + Stratified Bank Stills
# ---------------------------------------------------------------------------
def run_base50() -> dict[str, Any]:
    import comfy.samplers
    import control_vector_nodes as cv_nodes

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    img_dir, _ = _out_dirs()
    prompts = _load_nonmarket_prompts()
    banks = {cn: torch.load(BANKS_DIR / f"{cn}.pt", map_location="cpu", weights_only=False) for cn in ("noir", "paper_origami")}

    # Part A: Stratified Bank Concepts on Pika 6-step (4 extra concepts from the 38-bank!)
    print("[base50] Part A: Stratified bank concepts (1D vs Erasure)...")
    pika_sigmas = torch.tensor([float(x.strip()) for x in h3_config.PIKA_MANUAL_SIGMAS.split(",")], dtype=torch.float32)
    pika_model, img_vae = _load_pika_fl2va_model(with_lora=True)
    strat_out: list[dict[str, Any]] = []
    strat_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for c_stem, p_key in [
        ("film_16mm_1974", "hardware_aisle"),
        ("cursed_cctv", "indoor_neutral"),
        ("bioluminescent_glow", "indoor_neutral"),
        ("felt_muppet", "sitcom_kitchen"),
    ]:
        vp = cv_nodes.resolve_vector_file_path(c_stem, preferred_family="fl2va")
        vt, _ = cv.load_vector(vp)
        op_c = {"v1d": vt["vector.post_refiner"], "mu_d": vt["erasure.mu_d"], "vr": vt["erasure.vr"], "diag": vt["erasure.diag"]}
        cols = []
        im_0 = _sample_still_pika(pika_model, img_vae, prompts[p_key], op=None, strength=0.0, seed=SEED_MAIN, sigmas=pika_sigmas)
        cols.append((f"Base S={hsv_sat(im_0):.3f}", im_0))
        for lbl, ef in [("1D add_norm_matched", False), ("Pole-A Erasure", True)]:
            op_sub = op_c if ef else {**op_c, "vr": torch.zeros((0, 5376)), "diag": torch.zeros((0,))}
            im_s = _sample_still_pika(pika_model, img_vae, prompts[p_key], op=op_sub, strength=1.5, seed=SEED_MAIN, sigmas=pika_sigmas)
            sat = round(hsv_sat(im_s), 4)
            strat_out.append({"concept": c_stem, "prompt": p_key, "arm": lbl, "hsv_sat": sat})
            cols.append((f"{lbl} S={sat:.3f}", im_s))
        strat_rows.append((c_stem[:16], cols))
    make_contact_sheet(strat_rows, img_dir / "ablation8c_stratified_bank_concepts.png")
    del pika_model
    gc.collect()
    torch.cuda.empty_cache()

    # Part B: Undistilled 50-Step Base FL2VA (loras=[])
    print("[base50] Part B: Loading undistilled 50-step Base FL2VA (loras=[])...")
    base50_model, _ = _load_pika_fl2va_model(with_lora=False, shift_v=12.0, shift_a=3.0)
    sigmas_50 = comfy.samplers.calculate_sigmas(base50_model.get_model_object("model_sampling"), "normal", 50).cpu()
    b50_out: list[dict[str, Any]] = []
    b50_rows: list[tuple[str, list[tuple[str, Image.Image]]]] = []
    for cn, pn in [("noir", "indoor_neutral"), ("paper_origami", "sitcom_kitchen")]:
        op_full = build_erasure_variant(banks[cn]["a_spans_per_pair"], banks[cn]["b_spans_per_pair"], rank=8)
        op_1d = {**op_full, "vr": torch.zeros((0, 5376)), "diag": torch.zeros((0,))}
        for sd in SEEDS_3[:2]:
            cols = []
            im_b = _sample_still_pika(base50_model, img_vae, prompts[pn], op=None, strength=0.0, seed=sd, sigmas=sigmas_50, sampler_name="euler")
            cols.append((f"Base50 S={hsv_sat(im_b):.3f}", im_b))
            for lbl, op_u in [("Base50 1D", op_1d), ("Base50 Pole-A Erasure", op_full)]:
                im_x = _sample_still_pika(base50_model, img_vae, prompts[pn], op=op_u, strength=1.5, seed=sd, sigmas=sigmas_50, sampler_name="euler")
                sat = round(hsv_sat(im_x), 4)
                b50_out.append({"concept": cn, "seed": sd, "arm": lbl, "hsv_sat": sat})
                cols.append((f"{lbl} S={sat:.3f}", im_x))
                print(f"  [Base50] {cn} s={sd} {lbl}: sat={sat:.4f}")
            b50_rows.append((f"{cn[:10]} 50st s={sd}", cols))
    make_contact_sheet(b50_rows, img_dir / "ablation9_base50_undistilled_h3.png")
    del img_vae
    gc.collect()
    torch.cuda.empty_cache()

    # Part C: 39f (`on_both_clocks`) speaking AV check on undistilled 50-step Base FL2VA
    print("[base50] Part C: Running 39f speaking AV check on undistilled 50-step Base FL2VA (sitcom_kitchen)...")
    from comfy.patcher_extension import WrappersMP
    import comfy.sample
    import comfy.sd
    import comfy.utils
    import folder_paths

    h3_cond_mod = cv_nodes._repo("conditioning")
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    h3_sol_mod = cv_nodes._repo("sol_attn_h3")

    _, vid_dir = _out_dirs()
    _wait_for_comfy_idle_and_free()
    (clip_b,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae_b = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])))
    avae_b = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["audio_vae"])))
    kit_b_txt = (REPO_ROOT / "prompt_bank" / "t2va_sitcom_kitchen.txt").read_text(encoding="utf-8").strip()
    _enc_b, tags_b, ids_b, dec_b = cv_nodes.encode_prompt_for_vector(clip_b, kit_b_txt)
    sm_b = h3_spans.inference_mask(h3_spans.spans(ids_b, tags_b.tolist(), dec_b), "scene_only").detach().cpu()
    cond_39, lat_39 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip_b, vae=vvae_b, prompt=kit_b_txt, width=1152, height=768, length=39, single_frame=False
    )[:2]
    import comfy.model_management as comfy_mm

    del clip_b
    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    torch.cuda.empty_cache()

    (base50_sol,) = h3_sol_mod.MiniMaxH3Sol.execute(model=base50_model, **_sol_execute_kwargs())
    op_noir_b = build_erasure_variant(banks["noir"]["a_spans_per_pair"], banks["noir"]["b_spans_per_pair"], rank=8)
    b50_av_out: list[dict[str, Any]] = []
    for arm_lbl, er_on in [("base50_39f_noir_1d", False), ("base50_39f_noir_erasure", True)]:
        m_av = base50_sol.clone()

        def _dm_wrap_b50(executor, x, timestep, context, transformer_options={}, ef=er_on, **kwargs):
            ctx_use = context
            if context is not None:
                ctx_ref = _refine_context_if_needed(executor, context, transformer_options)
                ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                rows = ref64[sm_b]
                sh = apply_erasure_shift_rows(rows, op_noir_b, 1.5, erase_active=ef)
                out = ref64.clone()
                out[sm_b] = rows + sh
                ctx_use = out.to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
            return executor(x, timestep, ctx_use, transformer_options, **kwargs)

        m_av.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["eval_b50"] = [_dm_wrap_b50]
        lat_t = lat_39["samples"]
        noise = comfy.sample.prepare_noise(lat_t, SEED_MAIN)
        guider = comfy.samplers.CFGGuider(m_av)
        guider.inner_set_conds({"positive": cond_39})
        guider.set_cfg(1.0)
        samples = guider.sample(
            noise, lat_t, comfy.samplers.sampler_object("euler"), sigmas_50,
            denoise_mask=None, callback=None, disable_pbar=True, seed=SEED_MAIN,
        )
        v_l, a_l = samples.unbind()
        v_l = v_l.detach().cpu()
        a_l = a_l.detach().cpu()
        del samples, noise, guider, m_av
        from comfy_extras.nodes_audio import vae_decode_audio

        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        aud_obj = vae_decode_audio(avae_b, {"samples": a_l})
        aud_wav = aud_obj["waveform"].detach().cpu()
        aud_sr = int(aud_obj["sample_rate"])
        del aud_obj, a_l
        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        vid_dec = vvae_b.decode(v_l)
        if vid_dec.ndim == 5:
            vid_dec = vid_dec[0]
        vid_u8 = vid_dec.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
        del vid_dec, v_l
        comfy_mm.unload_all_models()
        comfy_mm.cleanup_models_gc()
        gc.collect()
        torch.cuda.empty_cache()
        mp4_p = vid_dir / f"{arm_lbl}.mp4"
        strip_p = vid_dir / "strips" / f"strip_{arm_lbl}.png"
        m_39 = mux_and_measure_av(vid_u8, aud_wav, aud_sr, mp4_p, strip_p, fps=24)
        m_39["arm"] = arm_lbl
        m_39["mp4"] = mp4_p.name
        b50_av_out.append(m_39)
        print(f"  [Base50 39f AV] {arm_lbl}: sat={m_39['sat_mean']:.4f}, dbfs={m_39['audio_dbfs']}")

    del base50_sol, base50_model, vvae_b, avae_b
    gc.collect()
    torch.cuda.empty_cache()

    res = {"stratified_bank_pika": strat_out, "base50_undistilled": b50_out, "base50_39f_av": b50_av_out}
    out_p = DATA_OUT / "gpu_base50_and_stratified.json"
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"[base50] Saved {out_p.name}.")
    return res


def run_dit_blocks() -> dict[str, Any]:
    """1-pass zero-render mechanistic activation trace across H3's 50 DiT blocks (`blocks.0..49`).

    Captures exact step-0 (`sigma = 1.0`) per-block incremental update RMS and cumulative residual
    divergence separately on `text`, `audio` (40 Hz), and `video` (24 fps) segments of the packed
    single-stream sequence (`comfy/ldm/minimax/model.py:649-671`) on `39f` `t2va_sitcom_kitchen`.
    """
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    import comfy.model_management as comfy_mm
    from comfy.patcher_extension import WrappersMP
    import comfy.sample
    import comfy.sd
    import comfy.utils
    import control_vector_nodes as cv_nodes
    import folder_paths

    h3_cond_mod = cv_nodes._repo("conditioning")
    h3_enc_mod = cv_nodes._repo("h3_encoder_loader")
    h3_lora_mod = cv_nodes._repo("lora_branch")
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from comfy_extras.nodes_model_advanced import ModelAttentionBackend

    print("[dit_blocks] Encoding 39f sitcom_kitchen packed AV conditioning for 50-block trace...")
    _wait_for_comfy_idle_and_free()
    (clip,) = h3_enc_mod.MiniMaxH3EncoderLoader.execute(encoder_name=h3_config.MODELS["clip"], device="default")
    vvae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", h3_config.MODELS["video_vae"])))
    kit_txt = (REPO_ROOT / "prompt_bank" / "t2va_sitcom_kitchen.txt").read_text(encoding="utf-8").strip()
    _enc_k, tags_k, ids_k, dec_k = cv_nodes.encode_prompt_for_vector(clip, kit_txt)
    sp_k = h3_spans.spans(ids_k, tags_k.tolist(), dec_k)
    scene_m = h3_spans.inference_mask(sp_k, "scene_only").detach().cpu()
    sound_m = h3_spans.inference_mask(sp_k, "soundscape_only").detach().cpu()
    cond_39, lat_39 = h3_cond_mod.MiniMaxH3Conditioning.execute(
        clip=clip, vae=vvae, prompt=kit_txt, width=1152, height=768, length=39, single_frame=False
    )[:2]
    del clip, vvae
    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    torch.cuda.empty_cache()

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", h3_config.MODELS["unet_fl2va"])
    model = comfy.sd.load_diffusion_model(unet_path)
    (model,) = h3_lora_mod.MiniMaxH3LoRABranch.execute(
        model=model, lora_name=h3_config.PIKA_FL2VA_SVD995_LORA, strength=1.0
    )
    (model,) = MiniMaxH3SigmaShift.execute(model=model, shift_video=12.0, shift_audio=3.0)
    (model,) = ModelAttentionBackend.execute(model=model, attention="comfy kitchen attention")

    lat_t = lat_39["samples"]
    noise = comfy.sample.prepare_noise(lat_t, SEED_MAIN)
    # 1 single step-0 forward pass (`sigmas = [1.0, 0.0]`, 1 NFE!) so every arm sees identical input `x_0`
    one_step_sigmas = torch.tensor([1.0, 0.0], dtype=torch.float32)

    def _capture_50blocks(shift_fn) -> tuple[list[tuple[int, int, str]], list[torch.Tensor], list[torch.Tensor]]:
        m_probe = model.clone()
        segs_holder: list[list[tuple[int, int, str]]] = []
        blk_outs: list[torch.Tensor] = []
        blk_deltas: list[torch.Tensor] = []

        for bi in range(50):
            def _make_blk_hook(idx: int):
                def _blk_hook(args: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
                    h_in_cpu = args["img"].detach().to(dtype=torch.float32, device="cpu", copy=True)
                    if idx == 0 and not segs_holder:
                        segs_holder.append([(int(a), int(b), str(k)) for a, b, k in args["layout"].segments])
                    res_d = extra["original_block"](args)
                    h_out_cpu = res_d["img"].detach().to(dtype=torch.float32, device="cpu", copy=True)
                    blk_deltas.append(h_out_cpu - h_in_cpu)
                    blk_outs.append(h_out_cpu)
                    return res_d
                return _blk_hook

            m_probe.set_model_patch_replace(_make_blk_hook(bi), "dit", "double_block", bi)

        if shift_fn is not None:
            def _dm_wrap_probe(executor, x, timestep, context, transformer_options={}, **kwargs):
                ctx_use = context
                if context is not None:
                    ctx_ref = _refine_context_if_needed(executor, context, transformer_options)
                    ref64 = ctx_ref[0].detach().cpu().to(torch.float64)
                    ctx_use = shift_fn(ref64).to(device=ctx_ref.device, dtype=ctx_ref.dtype).unsqueeze(0)
                return executor(x, timestep, ctx_use, transformer_options, **kwargs)

            m_probe.wrappers.setdefault(WrappersMP.DIFFUSION_MODEL, {})["probe_50"] = [_dm_wrap_probe]

        guider = comfy.samplers.CFGGuider(m_probe)
        guider.inner_set_conds({"positive": cond_39})
        guider.set_cfg(1.0)
        _ = guider.sample(
            noise, lat_t, comfy.samplers.sampler_object("euler"), one_step_sigmas,
            denoise_mask=None, callback=None, disable_pbar=True, seed=SEED_MAIN,
        )
        del guider, m_probe
        return segs_holder[0], blk_outs, blk_deltas

    print("[dit_blocks] Capturing unsteered baseline across 50 DiT blocks...")
    segs, base_outs, base_deltas = _capture_50blocks(None)
    print(f"  Packed sequence segments: {segs}")

    def _load_op(stem: str) -> dict[str, torch.Tensor]:
        vp = cv_nodes.resolve_vector_file_path(stem, preferred_family="fl2va")
        vt, _ = cv.load_vector(vp)
        return {"v1d": vt["vector.post_refiner"], "mu_d": vt["erasure.mu_d"], "vr": vt["erasure.vr"], "diag": vt["erasure.diag"]}

    probe_specs = [
        ("noir_1d", "noir", scene_m, False),
        ("noir_erasure", "noir", scene_m, True),
        ("paper_origami_1d", "paper_origami", scene_m, False),
        ("paper_origami_erasure", "paper_origami", scene_m, True),
        ("reverb_cathedral_soundscape_erasure", "reverb_cathedral", sound_m, True),
    ]

    traces: dict[str, Any] = {"num_blocks": 50, "segments": segs, "arms": {}}
    for arm_key, c_stem, m_bool, er_flag in probe_specs:
        op_c = _load_op(c_stem)

        def _sfn(ref64: torch.Tensor, op=op_c, mask=m_bool, ef=er_flag) -> torch.Tensor:
            rows = ref64[mask]
            sh = apply_erasure_shift_rows(rows, op, 1.5, erase_active=ef)
            out = ref64.clone()
            out[mask] = rows + sh
            return out

        _, s_outs, s_deltas = _capture_50blocks(_sfn)
        per_block: list[dict[str, float | int]] = []
        for bi in range(50):
            row_rec: dict[str, float | int] = {"block": bi}
            for a0, b0, kind in segs:
                if kind in ("text", "audio", "video"):
                    s_seg = s_outs[bi][a0:b0].double()
                    b_seg = base_outs[bi][a0:b0].double()
                    # Pre-RMSNorm geometry: row-unit-normalized L2 / angular shift (1 - mean_cos)
                    s_u = s_seg / torch.linalg.vector_norm(s_seg, dim=-1, keepdim=True).clamp(min=1e-12)
                    b_u = b_seg / torch.linalg.vector_norm(b_seg, dim=-1, keepdim=True).clamp(min=1e-12)
                    ang_div = float((1.0 - (s_u * b_u).sum(dim=-1).mean()).item())
                    # Per-block relative update shift ||delta_s - delta_b||_F / ||delta_b||_F
                    ds = s_deltas[bi][a0:b0].double()
                    db = base_deltas[bi][a0:b0].double()
                    rel_inc = float((torch.linalg.norm(ds - db) / torch.linalg.norm(db).clamp(min=1e-12)).item())
                    row_rec[f"{kind}_ang_div"] = round(ang_div, 5)
                    row_rec[f"{kind}_rel_inc"] = round(rel_inc, 5)
            per_block.append(row_rec)
        del s_outs, s_deltas
        traces["arms"][arm_key] = per_block
        peak_v = max(per_block, key=lambda r: float(r["video_rel_inc"]))
        peak_a = max(per_block, key=lambda r: float(r["audio_rel_inc"]))
        print(
            f"  [50-block] {arm_key:36s}: video_peak=blk {peak_v['block']} (rel={peak_v['video_rel_inc']:.3f}), "
            f"audio_peak=blk {peak_a['block']} (rel={peak_a['audio_rel_inc']:.3f}), "
            f"final_v_ang={per_block[-1]['video_ang_div']:.4f}, final_a_ang={per_block[-1]['audio_ang_div']:.4f}"
        )

    del base_outs, base_deltas, model
    comfy_mm.unload_all_models()
    comfy_mm.cleanup_models_gc()
    gc.collect()
    torch.cuda.empty_cache()

    out_p = DATA_OUT / "dit_blocks_50_trace.json"
    out_p.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    print(f"[dit_blocks] Saved {out_p.name}.")
    return traces


def run_figures() -> list[str]:
    """Generate 4 publication-grade vector SVG plots in `docs/wiki/steering/research/figures/`."""
    fig_dir = REPO_ROOT / "docs" / "wiki" / "steering" / "research" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ab15 = json.loads((DATA_OUT / "gpu_ablations_1_to_5.json").read_text(encoding="utf-8"))
    census = json.loads((DATA_OUT / "census_38_bank.json").read_text(encoding="utf-8"))
    cpu_m = json.loads((DATA_OUT / "cpu_math_ablations.json").read_text(encoding="utf-8"))
    written: list[str] = []

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # --- Figure 1: Soft-MaxSim tau_attn Goldilocks U-Curve & De-Coning Trap ---
    arms_order = [
        ("MaxSim (tau=0.03)", "Hard MaxSim\n(τ=0.03)"),
        ("tau=0.07", "Sharp\n(τ=0.07)"),
        ("tau=0.15 (shipped)", "Shipped\n(τ=0.15)"),
        ("tau=0.50", "Soft\n(τ=0.50)"),
        ("1D (tau=inf)", "1D / Sinkhorn OT\n(τ=∞)"),
        ("tau=0.15 deconed", "De-coned\n(τ=0.15 -μ)"),
    ]
    by_arm: dict[str, list[float]] = {k: [] for k, _ in arms_order}
    for r in ab15["ablation1_tau_sweep"]:
        if r["arm"] in by_arm:
            by_arm[r["arm"]].append(float(r["hsv_sat"]))

    svg1 = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 360" font-family="Inter, system-ui, -apple-system, sans-serif">',
        '<rect width="820" height="360" rx="12" fill="#0f172a"/>',
        '<text x="410" y="32" fill="#f8fafc" font-size="16" font-weight="700" text-anchor="middle">Ablation 1: ColBERT Soft-MaxSim Temperature (τ) Goldilocks Window on Noir (3 Seeds, RMS = 101.25)</text>',
        '<text x="410" y="52" fill="#94a3b8" font-size="12" text-anchor="middle">Lower HSV saturation = cleaner silver-gelatin monochrome (dashed green line: B&amp;W threshold S = 0.06)</text>',
    ]
    x0, y0, pw, ph = 85, 75, 690, 210
    ymax = 0.25
    # Gridlines + B&W threshold zone
    bw_y = y0 + ph * (1.0 - 0.06 / ymax)
    svg1.append(f'<rect x="{x0}" y="{bw_y:.1f}" width="{pw}" height="{y0 + ph - bw_y:.1f}" fill="#064e3b" opacity="0.32"/>')
    svg1.append(f'<line x1="{x0}" y1="{bw_y:.1f}" x2="{x0 + pw}" y2="{bw_y:.1f}" stroke="#34d399" stroke-dasharray="5,4" stroke-width="1.5"/>')
    svg1.append(f'<text x="{x0 + pw - 6}" y="{bw_y - 6:.1f}" fill="#34d399" font-size="11" text-anchor="end">True B&amp;W Zone (S &lt; 0.06)</text>')
    for yv in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        yy = y0 + ph * (1.0 - yv / ymax)
        svg1.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#334155" stroke-width="1"/>')
        svg1.append(f'<text x="{x0 - 10}" y="{yy + 4:.1f}" fill="#cbd5e1" font-size="11" text-anchor="end">{yv:.2f}</text>')
    svg1.append(f'<text x="24" y="{y0 + ph / 2:.1f}" fill="#cbd5e1" font-size="12" text-anchor="middle" transform="rotate(-90 24 {y0 + ph / 2:.1f})">Rendered HSV Saturation (S)</text>')

    pts_main = []
    for idx, (arm_k, lbl) in enumerate(arms_order):
        vals = by_arm[arm_k]
        mean_v = sum(vals) / len(vals)
        cx = x0 + (idx + 0.5) * (pw / len(arms_order))
        cy = y0 + ph * (1.0 - mean_v / ymax)
        if idx < 5:
            pts_main.append(f"{cx:.1f},{cy:.1f}")
        is_win = all(v < 0.06 for v in vals)
        col = "#34d399" if is_win else ("#f59e0b" if idx in (0, 5) else "#f87171")
        for si, v in enumerate(vals):
            vy = y0 + ph * (1.0 - v / ymax)
            ox = (si - 1) * 9
            svg1.append(f'<circle cx="{cx + ox:.1f}" cy="{vy:.1f}" r="4.5" fill="{col}" opacity="0.65"/>')
        svg1.append(f'<rect x="{cx - 22:.1f}" y="{cy - 3:.1f}" width="44" height="6" rx="3" fill="{col}"/>')
        svg1.append(f'<text x="{cx:.1f}" y="{cy - 11:.1f}" fill="{col}" font-size="11" font-weight="700" text-anchor="middle">{mean_v:.3f}</text>')
        l_parts = lbl.split("\n")
        svg1.append(f'<text x="{cx:.1f}" y="{y0 + ph + 20}" fill="#e2e8f0" font-size="11" font-weight="600" text-anchor="middle">{_esc(l_parts[0])}</text>')
        svg1.append(f'<text x="{cx:.1f}" y="{y0 + ph + 35}" fill="#94a3b8" font-size="10.5" text-anchor="middle">{_esc(l_parts[1])}</text>')
    svg1.insert(len(svg1) - 18, f'<polyline points="{" ".join(pts_main)}" fill="none" stroke="#60a5fa" stroke-width="2" stroke-dasharray="4,3"/>')
    svg1.append('</svg>')
    p1 = fig_dir / "fig1_tau_goldilocks_and_deconing.svg"
    p1.write_text("\n".join(svg1), encoding="utf-8")
    written.append(p1.name)

    # --- Figure 2: 38-Concept Bank Spectral Regime Map & Target-Word Rescues ---
    svg2 = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 390" font-family="Inter, system-ui, -apple-system, sans-serif">',
        '<rect width="840" height="390" rx="12" fill="#0f172a"/>',
        '<text x="420" y="30" fill="#f8fafc" font-size="16" font-weight="700" text-anchor="middle">38-Concept Production Bank Census: Three Geometric Regimes &amp; Target-Word Rescues (28/38)</text>',
        '<text x="420" y="50" fill="#94a3b8" font-size="12" text-anchor="middle">fl2va ↔ ref2va cross-family 1D cosine median = 0.9892 | rank-8 Vr Grassmann overlap median = 0.9323</text>',
    ]
    x0, y0, pw, ph = 80, 72, 710, 255
    xmin, xmax = 0.40, 0.85
    ymin, ymax = 5.2, 7.6
    # Regime background bands (mpc < 0.62, 0.62..0.72, >= 0.72)
    b1_x = x0 + pw * ((0.62 - xmin) / (xmax - xmin))
    b2_x = x0 + pw * ((0.72 - xmin) / (xmax - xmin))
    svg2.append(f'<rect x="{x0}" y="{y0}" width="{b1_x - x0:.1f}" height="{ph}" fill="#7f1d1d" opacity="0.22"/>')
    svg2.append(f'<rect x="{b1_x:.1f}" y="{y0}" width="{b2_x - b1_x:.1f}" height="{ph}" fill="#1e3a8a" opacity="0.20"/>')
    svg2.append(f'<rect x="{b2_x:.1f}" y="{y0}" width="{x0 + pw - b2_x:.1f}" height="{ph}" fill="#065f46" opacity="0.22"/>')
    svg2.append(f'<text x="{(x0 + b1_x)/2:.1f}" y="{y0 + 18}" fill="#fca5a5" font-size="11" font-weight="700" text-anchor="middle">Multi-Facet Bundles (10)</text>')
    svg2.append(f'<text x="{(b1_x + b2_x)/2:.1f}" y="{y0 + 18}" fill="#93c5fd" font-size="11" font-weight="700" text-anchor="middle">Intermediate Structural (21)</text>')
    svg2.append(f'<text x="{(b2_x + x0 + pw)/2:.1f}" y="{y0 + 18}" fill="#6ee7b7" font-size="11" font-weight="700" text-anchor="middle">Monolithic Overrides (7)</text>')

    for xv in [0.45, 0.55, 0.62, 0.72, 0.80]:
        xx = x0 + pw * ((xv - xmin) / (xmax - xmin))
        svg2.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0 + ph}" stroke="#334155" stroke-dasharray="3,3"/>')
        svg2.append(f'<text x="{xx:.1f}" y="{y0 + ph + 18}" fill="#cbd5e1" font-size="11" text-anchor="middle">{xv:.2f}</text>')
    for yv in [5.5, 6.0, 6.5, 7.0, 7.5]:
        yy = y0 + ph * (1.0 - (yv - ymin) / (ymax - ymin))
        svg2.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" stroke="#334155"/>')
        svg2.append(f'<text x="{x0 - 10}" y="{yy + 4:.1f}" fill="#cbd5e1" font-size="11" text-anchor="end">{yv:.1f}</text>')
    svg2.append(f'<text x="{x0 + pw / 2:.1f}" y="{y0 + ph + 42}" fill="#cbd5e1" font-size="12" text-anchor="middle">Mean Pairwise Cosine (mpc across B_p − A_p pairs)</text>')
    svg2.append(f'<text x="24" y="{y0 + ph / 2:.1f}" fill="#cbd5e1" font-size="12" text-anchor="middle" transform="rotate(-90 24 {y0 + ph / 2:.1f})">Pole-A Subspace Participation Ratio (PR_8)</text>')

    highlight_names = {
        "noir": (-8, -10),
        "wes_anderson_symmetry": (8, 13),
        "paper_origami": (-12, -10),
        "film_16mm_1974": (-38, 14),
        "reverb_cathedral": (-20, -10),
        "whisper_asmr": (-10, -9),
        "claymation_stopmotion": (8, -8),
        "balloon_head_inflate": (8, -6),
    }
    for rec in census["records"]:
        mpc_v = float(rec["mpc"])
        pr_v = max(ymin + 0.08, min(ymax - 0.08, float(rec["pole_a_pr8"])))
        xx = x0 + pw * ((mpc_v - xmin) / (xmax - xmin))
        yy = y0 + ph * (1.0 - (pr_v - ymin) / (ymax - ymin))
        rescued = bool(rec["rescued_words"])
        col = "#38bdf8" if rescued else "#94a3b8"
        rad = 6.0 if rec["concept"] in highlight_names else 4.5
        svg2.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{rad}" fill="{col}" stroke="#0f172a" stroke-width="1.2"/>')
        if rec["concept"] in highlight_names:
            dx, dy = highlight_names[rec["concept"]]
            tag = rec["concept"] + (f" (+{rec['rescued_words'][0]})" if rec["rescued_words"] else "")
            svg2.append(f'<text x="{xx + dx:.1f}" y="{yy + dy:.1f}" fill="#f8fafc" font-size="10.5" font-weight="600">{_esc(tag)}</text>')
    svg2.append(f'<circle cx="{x0 + 16}" cy="{y0 + ph - 14}" r="5" fill="#38bdf8"/><text x="{x0 + 28}" y="{y0 + ph - 10}" fill="#e2e8f0" font-size="11">Rescues target words in μ_d (28/38)</text>')
    svg2.append(f'<circle cx="{x0 + 260}" cy="{y0 + ph - 14}" r="5" fill="#94a3b8"/><text x="{x0 + 272}" y="{y0 + ph - 10}" fill="#94a3b8" font-size="11">Same top-8 readout as v_1D (10/38)</text>')
    svg2.append('</svg>')
    p2 = fig_dir / "fig2_38bank_regimes_and_grassmann.svg"
    p2.write_text("\n".join(svg2), encoding="utf-8")
    written.append(p2.name)

    # --- Figure 3: Pole-A Coverage Law (Left) & Denoising Step Lock-In (Right) ---
    svg3 = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 360" font-family="Inter, system-ui, -apple-system, sans-serif">',
        '<rect width="860" height="360" rx="12" fill="#0f172a"/>',
        '<text x="225" y="30" fill="#f8fafc" font-size="14.5" font-weight="700" text-anchor="middle">(a) Pole-A Coverage Law (Noir N = 1..12 Pairs)</text>',
        '<text x="645" y="30" fill="#f8fafc" font-size="14.5" font-weight="700" text-anchor="middle">(b) Early-Step Palette Lock-In (Shift = 12.0)</text>',
    ]
    # Left panel: N = 1, 2, 5, 12
    lx0, ly0, lpw, lph = 65, 60, 320, 230
    curve_n = {r["n_pairs"]: float(r["vr_grassmann_overlap_mean"]) for r in cpu_m["m3_pair_coverage_curve"]["noir"]["curve"]}
    sat_n = {r["n_pairs"]: float(r["hsv_sat"]) for r in ab15["ablation4_pair_count"] if r["concept"] == "noir"}
    ns = [1, 2, 5, 12]
    for yv in [0.0, 0.25, 0.50, 0.75, 1.0]:
        yy = ly0 + lph * (1.0 - yv)
        svg3.append(f'<line x1="{lx0}" y1="{yy:.1f}" x2="{lx0 + lpw}" y2="{yy:.1f}" stroke="#334155"/>')
        svg3.append(f'<text x="{lx0 - 8}" y="{yy + 4:.1f}" fill="#60a5fa" font-size="10.5" text-anchor="end">{yv:.2f}</text>')
    pts_gr, pts_st = [], []
    for idx, nv in enumerate(ns):
        xx = lx0 + (idx + 0.4) * (lpw / len(ns))
        gy = ly0 + lph * (1.0 - curve_n[nv])
        sy = ly0 + lph * (1.0 - (sat_n[nv] / 0.20))
        pts_gr.append(f"{xx:.1f},{gy:.1f}")
        pts_st.append(f"{xx:.1f},{sy:.1f}")
        svg3.append(f'<text x="{xx:.1f}" y="{ly0 + lph + 20}" fill="#e2e8f0" font-size="11" text-anchor="middle">N={nv}</text>')
    svg3.append(f'<polyline points="{" ".join(pts_gr)}" fill="none" stroke="#60a5fa" stroke-width="2.5"/>')
    svg3.append(f'<polyline points="{" ".join(pts_st)}" fill="none" stroke="#f43f5e" stroke-width="2.5"/>')
    for idx, nv in enumerate(ns):
        xx = lx0 + (idx + 0.4) * (lpw / len(ns))
        gy = ly0 + lph * (1.0 - curve_n[nv])
        sy = ly0 + lph * (1.0 - (sat_n[nv] / 0.20))
        svg3.append(f'<circle cx="{xx:.1f}" cy="{gy:.1f}" r="5" fill="#60a5fa"/>')
        svg3.append(f'<text x="{xx:.1f}" y="{gy - 9:.1f}" fill="#93c5fd" font-size="10.5" font-weight="700" text-anchor="middle">{curve_n[nv]:.2f}</text>')
        svg3.append(f'<circle cx="{xx:.1f}" cy="{sy:.1f}" r="5" fill="#f43f5e"/>')
        svg3.append(f'<text x="{xx:.1f}" y="{sy + 16:.1f}" fill="#fda4af" font-size="10.5" font-weight="700" text-anchor="middle">S={sat_n[nv]:.3f}</text>')
    svg3.append(f'<text x="{lx0 + lpw / 2:.1f}" y="{ly0 + lph + 44}" fill="#94a3b8" font-size="11" text-anchor="middle">Blue: Vr Grassmann Coverage | Coral: Noir HSV Sat</text>')

    # Right panel: Step windows bar chart
    rx0, ry0, rpw, rph = 470, 60, 350, 230
    win_arms = [
        ("1D [0..1]", "1D\n[0..1]"),
        ("Both [0..1] (default)", "Full\n[0..1]"),
        ("Erase [0..0.34] + mu_d [0..1]", "Erase\n[0..0.34]"),
        ("Erase [0..0.50] + mu_d [0..1]", "Erase\n[0..0.50]"),
        ("Both [0..0.50]", "Both\n[0..0.50]"),
        ("Late Erase [0.34..1]", "Late\n[0.34..1]"),
    ]
    w_map = {r["arm"]: float(r["hsv_sat"]) for r in ab15["ablation5_step_windows"] if r["concept"] == "noir"}
    for yv in [0.0, 0.10, 0.20, 0.30]:
        yy = ry0 + rph * (1.0 - yv / 0.32)
        svg3.append(f'<line x1="{rx0}" y1="{yy:.1f}" x2="{rx0 + rpw}" y2="{yy:.1f}" stroke="#334155"/>')
        svg3.append(f'<text x="{rx0 - 8}" y="{yy + 4:.1f}" fill="#cbd5e1" font-size="10.5" text-anchor="end">{yv:.2f}</text>')
    for idx, (wk, wlbl) in enumerate(win_arms):
        sv = w_map[wk]
        bx = rx0 + idx * (rpw / len(win_arms)) + 10
        bw = (rpw / len(win_arms)) - 20
        bh = rph * (sv / 0.32)
        by = ry0 + rph - bh
        col = "#34d399" if sv < 0.06 else ("#fbbf24" if sv < 0.10 else "#f87171")
        svg3.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="4" fill="{col}"/>')
        svg3.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 6:.1f}" fill="{col}" font-size="10.5" font-weight="700" text-anchor="middle">{sv:.3f}</text>')
        lp = wlbl.split("\n")
        svg3.append(f'<text x="{bx + bw / 2:.1f}" y="{ry0 + rph + 18}" fill="#e2e8f0" font-size="10.5" text-anchor="middle">{_esc(lp[0])}</text>')
        svg3.append(f'<text x="{bx + bw / 2:.1f}" y="{ry0 + rph + 32}" fill="#94a3b8" font-size="10" text-anchor="middle">{_esc(lp[1])}</text>')
    svg3.append('</svg>')
    p3 = fig_dir / "fig3_coverage_law_and_step_window.svg"
    p3.write_text("\n".join(svg3), encoding="utf-8")
    written.append(p3.name)

    # --- Figure 4: 50-Block Single-Stream DiT Modality Coupling Trace ---
    trace_p = DATA_OUT / "dit_blocks_50_trace.json"
    if trace_p.is_file():
        tr = json.loads(trace_p.read_text(encoding="utf-8"))
        svg4 = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 370" font-family="Inter, system-ui, -apple-system, sans-serif">',
            '<rect width="860" height="370" rx="12" fill="#0f172a"/>',
            '<text x="430" y="30" fill="#f8fafc" font-size="15.5" font-weight="700" text-anchor="middle">1-Pass Mechanistic Trace Across H3\'s 50 Joint AV DiT Blocks (blocks.0..49, σ = 1.0)</text>',
            '<text x="430" y="50" fill="#94a3b8" font-size="11.5" text-anchor="middle">(a) Video Per-Block Relative Update Δ (Noir 1D vs Erasure &amp; Origami) | (b) RMSNorm Angular Divergence 1−cos (Scene vs Soundscape Target)</text>',
        ]
        # Panel A: Video relative incremental update across blocks 0..49
        ax0, ay0, apw, aph = 65, 72, 355, 225
        n1d = [float(r["video_rel_inc"]) for r in tr["arms"]["noir_1d"]]
        ner = [float(r["video_rel_inc"]) for r in tr["arms"]["noir_erasure"]]
        oer = [float(r["video_rel_inc"]) for r in tr["arms"]["paper_origami_erasure"]]
        vmax_a = max(max(n1d), max(ner), max(oer), 1e-6) * 1.08
        for bi_tick in [0, 10, 20, 30, 40, 49]:
            xx = ax0 + apw * (bi_tick / 49.0)
            svg4.append(f'<line x1="{xx:.1f}" y1="{ay0}" x2="{xx:.1f}" y2="{ay0 + aph}" stroke="#334155" stroke-dasharray="2,3"/>')
            svg4.append(f'<text x="{xx:.1f}" y="{ay0 + aph + 18}" fill="#cbd5e1" font-size="10.5" text-anchor="middle">{bi_tick}</text>')
        for frac in [0.0, 0.5, 1.0]:
            yy = ay0 + aph * (1.0 - frac)
            svg4.append(f'<line x1="{ax0}" y1="{yy:.1f}" x2="{ax0 + apw}" y2="{yy:.1f}" stroke="#334155"/>')
            svg4.append(f'<text x="{ax0 - 6}" y="{yy + 4:.1f}" fill="#cbd5e1" font-size="10" text-anchor="end">{vmax_a * frac:.2f}</text>')
        for arr, col in [(n1d, "#94a3b8"), (ner, "#38bdf8"), (oer, "#f59e0b")]:
            pts = " ".join(f"{ax0 + apw * (i / 49.0):.1f},{ay0 + aph * (1.0 - v / vmax_a):.1f}" for i, v in enumerate(arr))
            svg4.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.1"/>')
        svg4.append(f'<text x="{ax0 + apw / 2:.1f}" y="{ay0 + aph + 38}" fill="#cbd5e1" font-size="10.5" text-anchor="middle">Block i ∈ [0..49] (Grey: Noir 1D | Cyan: Noir Erasure | Amber: Origami)</text>')

        # Panel B: Angular divergence 1 - cos across blocks 0..49
        bx0, by0, bpw, bph = 475, 72, 355, 225
        rev_a = [float(r["audio_ang_div"]) for r in tr["arms"]["reverb_cathedral_soundscape_erasure"]]
        rev_v = [float(r["video_ang_div"]) for r in tr["arms"]["reverb_cathedral_soundscape_erasure"]]
        noir_v = [float(r["video_ang_div"]) for r in tr["arms"]["noir_erasure"]]
        noir_a = [float(r["audio_ang_div"]) for r in tr["arms"]["noir_erasure"]]
        vmax_b = max(max(rev_a), max(rev_v), max(noir_v), max(noir_a), 1e-6) * 1.08
        for bi_tick in [0, 10, 20, 30, 40, 49]:
            xx = bx0 + bpw * (bi_tick / 49.0)
            svg4.append(f'<line x1="{xx:.1f}" y1="{by0}" x2="{xx:.1f}" y2="{by0 + bph}" stroke="#334155" stroke-dasharray="2,3"/>')
            svg4.append(f'<text x="{xx:.1f}" y="{by0 + bph + 18}" fill="#cbd5e1" font-size="10.5" text-anchor="middle">{bi_tick}</text>')
        for frac in [0.0, 0.5, 1.0]:
            yy = by0 + bph * (1.0 - frac)
            svg4.append(f'<line x1="{bx0}" y1="{yy:.1f}" x2="{bx0 + bpw}" y2="{yy:.1f}" stroke="#334155"/>')
            svg4.append(f'<text x="{bx0 - 6}" y="{yy + 4:.1f}" fill="#cbd5e1" font-size="10" text-anchor="end">{vmax_b * frac:.3f}</text>')
        for arr, col, dash in [(rev_a, "#a855f7", ""), (rev_v, "#34d399", ' stroke-dasharray="5,3"'), (noir_v, "#38bdf8", ""), (noir_a, "#f43f5e", ' stroke-dasharray="3,3"')]:
            pts = " ".join(f"{bx0 + bpw * (i / 49.0):.1f},{by0 + bph * (1.0 - v / vmax_b):.1f}" for i, v in enumerate(arr))
            svg4.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.1"{dash}/>')
        svg4.append(f'<text x="{bx0 + bpw / 2:.1f}" y="{by0 + bph + 38}" fill="#cbd5e1" font-size="10.2" text-anchor="middle">Purple/Green: Reverb Audio vs Video | Cyan/Coral: Noir Video vs Audio</text>')
        svg4.append('</svg>')
        p4 = fig_dir / "fig4_dit_50block_modality_trace.svg"
        p4.write_text("\n".join(svg4), encoding="utf-8")
        written.append(p4.name)

    print(f"[figures] Wrote {len(written)} SVG figures to {fig_dir.relative_to(REPO_ROOT)}: {written}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        choices=("census", "cpu_math", "ablations", "word_media", "audio_video", "base50", "dit_blocks", "figures", "all"),
        default="census",
    )
    args = ap.parse_args()

    if args.stage in ("census", "all"):
        run_census()
    if args.stage in ("cpu_math", "all"):
        run_cpu_math()
    if args.stage in ("ablations", "all"):
        run_ablations()
    if args.stage in ("word_media", "all"):
        run_word_media()
    if args.stage in ("audio_video", "all"):
        run_audio_video()
    if args.stage in ("base50", "all"):
        run_base50()
    if args.stage in ("dit_blocks", "all"):
        run_dit_blocks()
    if args.stage in ("figures", "all"):
        run_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
