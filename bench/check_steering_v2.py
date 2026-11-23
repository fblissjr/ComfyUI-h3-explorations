#!/usr/bin/env python3
"""Offline CPU verification gate (`0 GB` VRAM) for the clean-slate `steering/` v2 package.

What it checks:
1. `< 1e-6` CPU numerical parity against `models/control_vectors/bank/*.safetensors` (when
   present), Point C prelude math (`s=0` exact identity, `h_bar`, `peer_ros=True`, smooth
   `95.5 -> 98.5` RMS knee), and isolated `stamp_vector_routing.py --output-dir` tensor parity.
2. Exact unpooled `fp32` `SpanCache` (`__spans.safetensors`) round-trip (`< 1e-6`) across
   `span_a`, `span_b`, `lens_a`, `lens_b`, `hold_a/b`, `enc_a/b_rows`, and `bf16` refusal.
3. `steering.mixer.MixerCompiler` & `director/compiler.py` routing (`1x` static BasicGuider vs
   `2x` animated/rewrite `MiniMaxH3EditGuider`, `needs_ref_release -> 0.33`, `h3_image` family
   filter, Macro Channel Presets, fader bounds `[-1.5, +2.0]`) spanning `>= 5` non-film concept
   families (`zero_gravity`/`sudden_takeoff_flight`, `youtube_overhype_influencer`/`infomercial_fail`,
   `operatic_bellow`/`whisper_asmr`/`reverb_cathedral`, `knitted_wool_world`/`mercury_liquid_chrome`,
   `power_outage_flashlight`/`blacklight_uv_rave`, `heavy_blizzard`).
4. `[provenance]` 14-key TOML & `.safetensors` round-trip (`None` `temperature`/`seed`, `int|float`
   `temperature`) and Day-1 v1 `regime = ("a",)` enforcement.
5. Stratified hash-seeded `prompt_bank/bank.json` pair compiler diversity & excluded stems.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from _lib import REPO, bootstrap

bootstrap(cpu=True)
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import folder_paths  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save as save_safetensors_bytes  # noqa: E402

import h3_spans  # noqa: E402
import stamp_vector_routing  # noqa: E402
import steering  # noqa: E402
from control_vector import pair_compiler  # noqa: E402
from director import compiler as director_compiler  # noqa: E402
from steering import constants as sc  # noqa: E402
from steering import data as sd  # noqa: E402
from steering import extract as se  # noqa: E402
from steering import mixer as sm  # noqa: E402
from steering import runtime as sr  # noqa: E402

REPO_ROOT = REPO
TOML_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"

DIVERSE_CONCEPT_FAMILIES: dict[str, tuple[str, ...]] = {
    "physics_levitation": ("zero_gravity", "sudden_takeoff_flight"),
    "behavioral_comedy": ("youtube_overhype_influencer", "infomercial_fail"),
    "vocal_acoustic": ("operatic_bellow", "whisper_asmr", "reverb_cathedral"),
    "tactile_material": ("knitted_wool_world", "mercury_liquid_chrome"),
    "extreme_lighting": ("power_outage_flashlight", "blacklight_uv_rave"),
    "atmosphere_weather": ("heavy_blizzard", "env_torrential_rain_drench"),
}


def _bank_dir() -> Path:
    models_root = Path(getattr(folder_paths, "models_dir", REPO_ROOT.parent.parent / "models"))
    return models_root / "control_vectors" / "bank"


def _ref_single_erasure_shift64(
    rows64: torch.Tensor,
    mu_d: torch.Tensor,
    vr: torch.Tensor,
    diag: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    """Independent closed-form Point C reference formula for single-vector erasure."""
    mu64 = mu_d.to(dtype=torch.float64)
    mu_n = mu64 * (67.5 / float(torch.linalg.vector_norm(mu64).item()))
    add_rows = (strength * mu_n).unsqueeze(0).expand_as(rows64)
    if strength <= 0.0:
        return add_rows
    vr64 = vr.to(dtype=torch.float64)
    d64 = diag.to(dtype=torch.float64)
    centered = rows64 - rows64.mean(dim=0, keepdim=True)
    e_own = ((centered @ vr64.T) * d64.unsqueeze(0)) @ vr64
    e_own_rms = float(torch.sqrt((e_own ** 2).sum(dim=-1).mean()).item())
    e_cap = min(1.0, 67.5 / e_own_rms)
    e = e_own * (min(max(strength, 0.0), 1.0) * e_cap)
    raw = add_rows - e
    raw_rms = float(torch.sqrt((raw ** 2).sum(dim=-1).mean()).item())
    return raw * ((abs(strength) * 67.5) / raw_rms)


def check_constants_and_toml_routing() -> str:
    assert sc.POST_REFINER_DIM == 5376, f"expected 5376, got {sc.POST_REFINER_DIM}"
    assert sc.ENCODER_DIM == 5120, f"expected 5120, got {sc.ENCODER_DIM}"
    assert sc.NORM_MATCHED_TARGET == 67.5
    assert sc.ERASURE_MAX_RANK == 8
    assert sc.ERASURE_RIDGE_REL == 0.15
    assert sc.ERASURE_MATCH_TAU == 0.15
    assert (sc.STACK_SMOOTH_KNEE_NORM, sc.STACK_SMOOTH_CEIL_NORM) == (95.5, 98.5)
    assert sc.MIN_ERASURE_PAIRS == 6
    assert sc.STEER_BUSES == ("look", "motion", "audio", "hold")
    assert sc.STEER_REGIMES == ("a",), f"Day-1 v1 core must restrict regime to ('a',), got {sc.STEER_REGIMES}"

    for bad_regime in ("b", "conditional"):
        try:
            sd.parse_routing_table({"bus": "look", "regime": bad_regime})
            raise AssertionError(f"expected ValueError for unverified regime={bad_regime!r}")
        except ValueError:
            pass

    tomls = sorted(TOML_DIR.glob("*.toml"))
    assert len(tomls) == 50, f"expected 50 active TOMLs, found {len(tomls)}"
    buses_seen: set[str] = set()
    for tp in tomls:
        ps = sd.load_pairs(tp)
        assert ps.routing is not None, f"{tp.name} missing [routing]"
        assert ps.routing.regime == "a"
        buses_seen.add(ps.routing.bus)
        exp_fams = ("fl2va", "ref2va") if ps.name in sc.NON_H3_IMAGE_STEMS else ("fl2va", "ref2va", "h3_image")
        assert ps.routing.families == exp_fams, f"{tp.name} families={ps.routing.families} != {exp_fams}"

    assert {"look", "motion", "audio"}.issubset(buses_seen), f"expected look/motion/audio buses, got {buses_seen}"
    return f"50 TOMLs verified (POST_REFINER_DIM=5376, ENCODER_DIM=5120, buses={sorted(buses_seen)}, regime='a')"


def check_span_cache_roundtrip() -> str:
    gen = torch.Generator(device="cpu").manual_seed(20261123)
    n_pairs = 7
    encoded_pairs: list[dict[str, Any]] = []
    refined_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    shared_dir = torch.randn((sc.POST_REFINER_DIM,), generator=gen, dtype=torch.float32)
    shared_enc = torch.randn((sc.ENCODER_DIM,), generator=gen, dtype=torch.float32)

    for idx in range(n_pairs):
        len_a = 18 + idx * 2
        len_b = 20 + idx * 3
        first_diff = 5
        end_a = len_a - 3
        end_b = len_b - 3

        enc_a = torch.randn((len_a, sc.ENCODER_DIM), generator=gen, dtype=torch.float32)
        enc_b = torch.randn((len_b, sc.ENCODER_DIM), generator=gen, dtype=torch.float32)
        enc_b[first_diff:end_b] += 1.5 * shared_enc

        ref_a = torch.randn((len_a, sc.POST_REFINER_DIM), generator=gen, dtype=torch.float32) * 40.0
        ref_b = torch.randn((len_b, sc.POST_REFINER_DIM), generator=gen, dtype=torch.float32) * 40.0
        ref_b[first_diff:end_b] += 12.0 * shared_dir

        tags_a = torch.ones((len_a,), dtype=torch.long)
        tags_b = torch.ones((len_b,), dtype=torch.long)
        ids_a = [100 + i for i in range(first_diff)] + [500 + idx * 30 + i for i in range(end_a - first_diff)] + [900, 901, 902]
        ids_b = [100 + i for i in range(first_diff)] + [700 + idx * 30 + i for i in range(end_b - first_diff)] + [900, 901, 902]

        spans_a = {
            "scene": torch.ones((len_a,), dtype=torch.bool),
            "vision": torch.zeros((len_a,), dtype=torch.bool),
            "dialogue": torch.zeros((len_a,), dtype=torch.bool),
        }
        spans_b = {
            "scene": torch.ones((len_b,), dtype=torch.bool),
            "vision": torch.zeros((len_b,), dtype=torch.bool),
            "dialogue": torch.zeros((len_b,), dtype=torch.bool),
        }
        item = sd.PairItem(
            index=idx,
            weight=1.0 + 0.1 * (idx % 3),
            a=f"scene_a_{idx}",
            b=f"scene_b_{idx}",
            a_span=f"span_a_{idx}",
            b_span=f"span_b_{idx}",
        )
        encoded_pairs.append(
            {
                "idx": idx,
                "item": item,
                "enc_a": enc_a,
                "tags_a": tags_a,
                "ids_a": ids_a,
                "spans_a": spans_a,
                "enc_b": enc_b,
                "tags_b": tags_b,
                "ids_b": ids_b,
                "spans_b": spans_b,
                "first_diff": first_diff,
                "end_a": end_a,
                "end_b": end_b,
            }
        )
        refined_pairs.append((ref_a, ref_b))

    # Add 2 hold items
    encoded_holds: list[dict[str, Any]] = []
    refined_holds: list[tuple[torch.Tensor, torch.Tensor]] = []
    for h_idx in range(2):
        hlen = 16
        h_enc_a = torch.randn((hlen, sc.ENCODER_DIM), generator=gen, dtype=torch.float32)
        h_enc_b = torch.randn((hlen, sc.ENCODER_DIM), generator=gen, dtype=torch.float32)
        h_ref_a = torch.randn((hlen, sc.POST_REFINER_DIM), generator=gen, dtype=torch.float32) * 30.0
        h_ref_b = torch.randn((hlen, sc.POST_REFINER_DIM), generator=gen, dtype=torch.float32) * 30.0
        h_tags = torch.ones((hlen,), dtype=torch.long)
        h_ids_a = [100, 101, 102] + [2000 + h_idx * 20 + i for i in range(10)] + [900, 901, 902]
        h_ids_b = [100, 101, 102] + [3000 + h_idx * 20 + i for i in range(10)] + [900, 901, 902]
        h_spans = {
            "scene": torch.ones((hlen,), dtype=torch.bool),
            "vision": torch.zeros((hlen,), dtype=torch.bool),
            "dialogue": torch.zeros((hlen,), dtype=torch.bool),
        }
        encoded_holds.append(
            {
                "h_item": sd.HoldItem(index=h_idx, group=f"grp_{h_idx}", weight=1.0, a="ha", b="hb", a_span="sa", b_span="sb"),
                "enc_a": h_enc_a,
                "tags_a": h_tags,
                "ids_a": h_ids_a,
                "spans_a": h_spans,
                "enc_b": h_enc_b,
                "tags_b": h_tags,
                "ids_b": h_ids_b,
                "spans_b": h_spans,
                "first_diff": 3,
                "end_a": 13,
                "end_b": 13,
            }
        )
        refined_holds.append((h_ref_a, h_ref_b))

    direct = se.pool_pair_set(
        encoded_pairs,
        refined_pairs,
        encoded_holds,
        refined_holds,
        pooling="diff_span",
        unit_mean=False,
        include_encoder=True,
        min_erasure_pairs=sc.MIN_ERASURE_PAIRS,
    )

    with tempfile.TemporaryDirectory(prefix="h3_spancache_") as tmp:
        tmp_dir = Path(tmp)
        cache = se.build_span_cache(
            encoded_pairs,
            refined_pairs,
            encoded_holds,
            refined_holds,
            pooling="diff_span",
            include_encoder=True,
            median_ref_row_norm=164.25,
            median_enc_row_norm_ex0=291.75,
        )
        span_file = tmp_dir / "zero_gravity__fl2va__spans.safetensors"
        se.save_span_cache(span_file, cache, overwrite=True)
        reloaded = se.load_span_cache(span_file)

        assert reloaded.span_a.dtype == torch.float32 and reloaded.span_b.dtype == torch.float32
        assert reloaded.lens_a.dtype == torch.int32 and reloaded.lens_b.dtype == torch.int32
        assert reloaded.span_a.shape[1] == sc.POST_REFINER_DIM
        assert reloaded.enc_a_rows is not None and reloaded.enc_a_rows.shape == (n_pairs, sc.ENCODER_DIM)
        assert reloaded.metadata["median_ref_row_norm"] == 164.25
        assert reloaded.metadata["median_enc_row_norm_ex0"] == 291.75

        from_cache = se.pool_from_span_cache(
            reloaded,
            unit_mean=False,
            apply_holds=True,
            include_encoder=True,
            min_erasure_pairs=sc.MIN_ERASURE_PAIRS,
        )

        diffs = {
            "vector": float((direct.vector - from_cache.vector).abs().max().item()),
            "enc_raw_vec": float((direct.enc_raw_vec - from_cache.enc_raw_vec).abs().max().item()),
            "hold_basis": float((direct.hold_basis - from_cache.hold_basis).abs().max().item()),
            "erasure_mu_d": float((direct.erasure_mu_d - from_cache.erasure_mu_d).abs().max().item()),
            "erasure_vr": float((direct.erasure_vr - from_cache.erasure_vr).abs().max().item()),
            "erasure_diag": float((direct.erasure_diag - from_cache.erasure_diag).abs().max().item()),
        }
        max_diff = max(diffs.values())
        assert max_diff < 1e-6, f"SpanCache round-trip exceeded 1e-6: {diffs}"

        # Verify bf16 SpanCache refusal on both save and load
        bad_cache = se.SpanCache(
            span_a=cache.span_a.to(dtype=torch.bfloat16),
            span_b=cache.span_b,
            lens_a=cache.lens_a,
            lens_b=cache.lens_b,
            pair_weights=cache.pair_weights,
            hold_a=cache.hold_a,
            hold_b=cache.hold_b,
            hold_weights=cache.hold_weights,
            enc_a_rows=cache.enc_a_rows,
            enc_b_rows=cache.enc_b_rows,
            enc_hold_a=cache.enc_hold_a,
            enc_hold_b=cache.enc_hold_b,
            metadata=cache.metadata,
        )
        try:
            se.save_span_cache(tmp_dir / "bad_bf16__spans.safetensors", bad_cache)
            raise AssertionError("expected save_span_cache to refuse bf16 span_a")
        except ValueError:
            pass

        bf16_raw_path = tmp_dir / "raw_bf16__spans.safetensors"
        bf16_raw_path.write_bytes(
            save_safetensors_bytes(
                {
                    "span_a": cache.span_a.to(dtype=torch.bfloat16),
                    "span_b": cache.span_b.to(dtype=torch.bfloat16),
                    "lens_a": cache.lens_a,
                    "lens_b": cache.lens_b,
                    "pair_weights": cache.pair_weights,
                    "hold_a": cache.hold_a,
                    "hold_b": cache.hold_b,
                    "hold_weights": cache.hold_weights,
                },
                metadata={"format": json.dumps(sc.SPAN_CACHE_FORMAT_VERSION)},
            )
        )
        try:
            se.load_span_cache(bf16_raw_path)
            raise AssertionError("expected load_span_cache to refuse bf16 tensors")
        except ValueError:
            pass

    return f"unpooled fp32 SpanCache round-trip max_abs_diff={max_diff:.2e} < 1e-6 (bf16 refused)"


def check_bank_numerical_parity_and_stamping() -> str:
    bdir = _bank_dir()
    gen = torch.Generator(device="cpu").manual_seed(42)
    ctx = torch.randn((1, 32, sc.POST_REFINER_DIM), generator=gen, dtype=torch.float32) * 2.2
    tags = torch.ones((1, 32), dtype=torch.long)
    tags[0, :4] = 0  # vision prefix rows must stay bitwise untouched

    if bdir.is_dir():
        sampled_paths: list[Path] = []
        for _fam_label, stems in DIVERSE_CONCEPT_FAMILIES.items():
            for st in stems:
                matches = sorted(bdir.glob(f"{st}__fl2va__diff_span*.safetensors"))
                if matches:
                    sampled_paths.append(matches[0])
        assert len(sampled_paths) >= 10, f"expected >=10 diverse bank vectors, found {len(sampled_paths)}"

        max_load_diff = 0.0
        max_math_diff = 0.0
        loaded_items: list[tuple[Path, dict[str, torch.Tensor], dict[str, Any]]] = []
        for sp in sampled_paths:
            tensors, meta = se.load_vector(sp)
            loaded_items.append((sp, tensors, meta))
            with safe_open(str(sp), framework="pt", device="cpu") as sf:
                for k in sf.keys():
                    raw_t = sf.get_tensor(k)
                    d = float((tensors[k] - raw_t).abs().max().item())
                    max_load_diff = max(max_load_diff, d)

            # Verify s=0 exact bitwise identity
            out_zero = sr.apply_post_refiner(
                ctx,
                tags,
                [(tensors["vector.post_refiner"], 0.0, "add_norm_matched_erasure")],
                erasure_specs=[(tensors.get("erasure.vr"), tensors.get("erasure.diag"), tensors.get("erasure.mu_d"))],
            )
            assert torch.equal(out_zero, ctx), f"s=0 was not bitwise identity on {sp.name}"

            # Verify Point C single-vector erasure parity (< 1e-6) against independent formula
            if "erasure.vr" in tensors and "erasure.diag" in tensors and "erasure.mu_d" in tensors:
                out_1 = sr.apply_post_refiner(
                    ctx,
                    tags,
                    [(tensors["vector.post_refiner"], 1.15, "add_norm_matched_erasure")],
                    erasure_specs=[(tensors["erasure.vr"], tensors["erasure.diag"], tensors["erasure.mu_d"])],
                )
                assert torch.equal(out_1[0, :4], ctx[0, :4]), "tag-0 vision rows modified!"
                base64 = ctx[0, 4:].to(dtype=torch.float64)
                exp_shift64 = _ref_single_erasure_shift64(
                    base64,
                    tensors["erasure.mu_d"],
                    tensors["erasure.vr"],
                    tensors["erasure.diag"],
                    1.15,
                )
                exp_rows32 = (base64 + exp_shift64).to(dtype=torch.float32)
                err = float((out_1[0, 4:] - exp_rows32).abs().max().item())
                max_math_diff = max(max_math_diff, err)

        # Verify stacked smooth RMS knee (95.5 -> 98.5) across 3 orthogonal concept families
        v_a = loaded_items[0][1]["vector.post_refiner"]
        v_b = loaded_items[3][1]["vector.post_refiner"]
        v_c = loaded_items[6][1]["vector.post_refiner"]
        er_specs = [
            (it[1].get("erasure.vr"), it[1].get("erasure.diag"), it[1].get("erasure.mu_d"))
            for it in (loaded_items[0], loaded_items[3], loaded_items[6])
        ]
        out_stack = sr.apply_post_refiner(
            ctx,
            tags,
            [
                (v_a, 1.25, "add_norm_matched_erasure"),
                (v_b, 1.25, "add_norm_matched_erasure"),
                (v_c, 1.25, "add_norm_matched_erasure"),
            ],
            erasure_specs=er_specs,
        )
        delta_rms = float(
            torch.sqrt(((out_stack[0, 4:] - ctx[0, 4:]).to(dtype=torch.float64) ** 2).sum(dim=-1).mean()).item()
        )
        assert sc.STACK_SMOOTH_KNEE_NORM < delta_rms <= sc.STACK_SMOOTH_CEIL_NORM + 1e-4, (
            f"expected stacked RMS inside (95.5, 98.5], got {delta_rms:.4f}"
        )

        # Verify isolated stamp_vector_routing --output-dir bit-exact tensor preservation
        with tempfile.TemporaryDirectory(prefix="h3_stamp_test_") as tmp:
            in_sub = Path(tmp) / "in_bank"
            out_sub = Path(tmp) / "out_bank"
            in_sub.mkdir()
            for sp, _t, _m in loaded_items[:6]:
                shutil.copy2(sp, in_sub / sp.name)
            dry = stamp_vector_routing.stamp_directory(toml_dir=TOML_DIR, input_dir=in_sub, dry_run=True)
            assert len(dry["stamped"]) == 6 and not out_sub.exists()
            res = stamp_vector_routing.stamp_directory(
                toml_dir=TOML_DIR, input_dir=in_sub, output_dir=out_sub, dry_run=False
            )
            assert len(res["stamped"]) == 6
            max_stamp_diff = 0.0
            for sp, orig_tensors, _m in loaded_items[:6]:
                st_tensors, st_meta = se.load_vector(out_sub / sp.name)
                assert "routing" in st_meta and st_meta["routing"]["regime"] == "a"
                rt_loaded = sm.resolve_vector_routing(out_sub / sp.name)
                assert rt_loaded.bus == st_meta["routing"]["bus"]
                for k, tv in orig_tensors.items():
                    max_stamp_diff = max(max_stamp_diff, float((st_tensors[k] - tv).abs().max().item()))
            assert max_stamp_diff == 0.0, f"stamp_vector_routing altered tensor bytes: {max_stamp_diff}"

        assert max_load_diff < 1e-6 and max_math_diff < 1e-6, (
            f"bank parity exceeded 1e-6: load={max_load_diff}, math={max_math_diff}"
        )
        return (
            f"{len(sampled_paths)} live bank vectors across 6 families verified "
            f"(load_diff={max_load_diff:.2e}, point_c_diff={max_math_diff:.2e} < 1e-6, "
            f"stacked_rms={delta_rms:.2f}, stamp_diff=0.0)"
        )

    # Synthetic fallback if models/control_vectors/bank is absent
    v_syn = torch.randn((sc.POST_REFINER_DIM,), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(torch.randn((sc.POST_REFINER_DIM, 4), generator=gen, dtype=torch.float64))
    vr_syn = q.T.to(dtype=torch.float32).contiguous()
    diag_syn = torch.tensor([0.85, 0.65, 0.45, 0.25], dtype=torch.float32)
    out_syn = sr.apply_post_refiner(
        ctx,
        tags,
        [(v_syn, 1.1, "add_norm_matched_erasure")],
        erasure_specs=[(vr_syn, diag_syn, v_syn)],
    )
    exp_syn = (
        ctx[0, 4:].to(dtype=torch.float64)
        + _ref_single_erasure_shift64(ctx[0, 4:].to(dtype=torch.float64), v_syn, vr_syn, diag_syn, 1.1)
    ).to(dtype=torch.float32)
    diff_syn = float((out_syn[0, 4:] - exp_syn).abs().max().item())
    assert diff_syn < 1e-6
    return f"synthetic Point C parity verified (diff={diff_syn:.2e} < 1e-6; bank dir absent)"


def check_mixer_and_director_routing() -> str:
    mc = sm.MixerCompiler(toml_dirs=(TOML_DIR,))

    # 1. Static whole-clip faders across 3 distinct families -> 1x BasicGuider + ReleaseReferences=0.33
    plan_1x = mc.compile(
        strips=[
            sm.ChannelStrip(vector="zero_gravity", strength=1.20),          # physics_levitation (motion, release=True)
            sm.ChannelStrip(vector="reverb_cathedral", strength=1.05),      # vocal_acoustic (audio, soundscape_only, audio_only)
            sm.ChannelStrip(vector="knitted_wool_world", strength=1.15),    # tactile_material (look, video_only, release=True)
        ],
        family="ref2va",
        has_picture_ref=True,
    )
    assert plan_1x.pass_mode == "1x" and plan_1x.guider_class == "BasicGuider"
    assert plan_1x.peer_ros is True and plan_1x.smooth_knee == (95.5, 98.5)
    assert plan_1x.needs_ref_release is True and math.isclose(plan_1x.release_end_percent, 0.33, abs_tol=1e-6)
    by_vec_1x = {n.vector: n for n in plan_1x.steer_nodes}
    assert by_vec_1x["zero_gravity"].bus == "motion" and by_vec_1x["zero_gravity"].target == "description_only"
    assert by_vec_1x["reverb_cathedral"].bus == "audio"
    assert by_vec_1x["reverb_cathedral"].target == "soundscape_only"
    assert by_vec_1x["reverb_cathedral"].streams == "audio_only"
    assert by_vec_1x["knitted_wool_world"].bus == "look" and by_vec_1x["knitted_wool_world"].streams == "video_only"
    assert all(n.edit_pass == "both" for n in plan_1x.steer_nodes)

    # 2. Static audio-only / vocal vectors alongside <Picture 1> do NOT release <Picture 1>
    plan_vocal = mc.compile(
        strips=[
            sm.ChannelStrip(vector="operatic_bellow", strength=1.10),
            sm.ChannelStrip(vector="whisper_asmr", strength=-0.50),
        ],
        family="ref2va",
        has_picture_ref=True,
    )
    assert plan_vocal.pass_mode == "1x"
    assert plan_vocal.needs_ref_release is False and plan_vocal.release_end_percent == 1.0

    # 3. Timeline [F,1,1] / spatial [F,H,W] animated fader + [concept] rewrite -> 2x MiniMaxH3EditGuider
    plan_2x = mc.compile(
        strips=[
            sm.ChannelStrip(vector="power_outage_flashlight", strength=1.10),  # static anchor on both passes
            sm.ChannelStrip(
                vector="youtube_overhype_influencer",
                strength=1.30,
                envelope_kind="timeline",
            ),
        ],
        family="fl2va",
        has_concept_rewrite=True,
        has_picture_ref=False,
    )
    assert plan_2x.pass_mode == "2x" and plan_2x.guider_class == "MiniMaxH3EditGuider"
    by_vec_2x = {n.vector: n for n in plan_2x.steer_nodes}
    assert by_vec_2x["power_outage_flashlight"].edit_pass == "both"
    assert by_vec_2x["youtube_overhype_influencer"].edit_pass == "edited_only"

    # 4. Macro Channel Presets (uv_club_rave, cathedral_opera, pyro_action, wool_whisper)
    plan_macro = mc.compile(
        macros=["uv_club_rave", "pyro_action"],
        family="ref2va",
        has_picture_ref=True,
    )
    assert len(plan_macro.steer_nodes) == 4
    assert len(plan_macro.prompt_patches) == 1 and "sub-bass" in plan_macro.prompt_patches[0]
    assert plan_macro.needs_ref_release is True and math.isclose(plan_macro.release_end_percent, 0.33, abs_tol=1e-6)

    # 5. Fader bounds [-1.5, +2.0] validation
    for oob in (-1.55, 2.05):
        try:
            sm.validate_fader_strength(oob)
            raise AssertionError(f"expected fader out-of-bounds ValueError for {oob}")
        except ValueError:
            pass

    # 6. [Image] (h3_image) family filtering hides exact 7 NON_H3_IMAGE_STEMS and keeps 43 visual stems
    all_stems = sorted(p.stem for p in TOML_DIR.glob("*.toml"))
    img_stems = mc.filter_for_family(all_stems, "h3_image")
    assert len(img_stems) == 43, f"expected 43 h3_image stems, got {len(img_stems)}"
    assert not (set(img_stems) & sc.NON_H3_IMAGE_STEMS)
    for req_vis in (
        "zero_gravity",
        "sudden_takeoff_flight",
        "infomercial_fail",
        "youtube_overhype_influencer",
        "knitted_wool_world",
        "mercury_liquid_chrome",
        "power_outage_flashlight",
        "blacklight_uv_rave",
    ):
        assert req_vis in img_stems, f"expected {req_vis} selectable in h3_image"

    # Rejects audio-only vector when compiling for h3_image
    try:
        mc.compile(strips=[sm.ChannelStrip(vector="reverb_cathedral", strength=1.0)], family="h3_image")
        raise AssertionError("expected ValueError compiling reverb_cathedral on h3_image")
    except ValueError:
        pass

    # 7. director/compiler.py integration (resolves routing + auto-lowers ReleaseReferences to 0.33)
    sample_events = json.dumps(
        {
            "portal": [
                {"frame": 0, "x": 0.3, "y": 0.5, "rx": 0.2, "ry": 0.2, "shape": "circle", "label": "P"},
                {"frame": 89, "x": 0.7, "y": 0.5, "rx": 0.2, "ry": 0.2, "shape": "circle", "label": "P"},
            ],
            "targets": [],
            "portal_cues": [],
        }
    )
    base_dir_payload = {
        "mode": "r2v_portal",
        "length": 90,
        "seed": 7,
        "ref_image": "anchor.png",
        "events_json": sample_events,
    }
    res_rel = director_compiler.compile_spatial_take(
        {
            **base_dir_payload,
            "control_vectors": [
                {"vector": "sudden_takeoff_flight", "strength": 1.25},
                {"vector": "mercury_liquid_chrome", "strength": 1.10},
            ],
        }
    )
    g_rel = res_rel["graph"]
    assert g_rel["115"]["inputs"]["target"] == "description_only"
    assert g_rel["115"]["inputs"]["streams"] == "both"
    assert g_rel["115"]["inputs"]["block_band"] == "all"
    assert g_rel["116"]["inputs"]["target"] == "scene_only"
    assert g_rel["116"]["inputs"]["streams"] == "video_only"
    assert math.isclose(float(g_rel["135"]["inputs"]["end_percent"]), 0.33, abs_tol=1e-6)

    res_keep = director_compiler.compile_spatial_take(
        {
            **base_dir_payload,
            "control_vectors": [{"vector": "reverb_cathedral", "strength": 1.15}],
        }
    )
    g_keep = res_keep["graph"]
    assert g_keep["115"]["inputs"]["target"] == "soundscape_only"
    assert g_keep["115"]["inputs"]["streams"] == "audio_only"
    assert math.isclose(float(g_keep["135"]["inputs"]["end_percent"]), 1.0, abs_tol=1e-6)

    return "MixerCompiler (1x/2x, macros, release=0.33, h3_image 43/50 filter) & director/compiler verified across 6 families"


def check_provenance_and_stratified_compiler() -> str:
    # 1. 14-key canonical [provenance] round-trip (including None temperature/seed and int|float temperature)
    prov_full = {
        "author_kind": "heylook_llm",
        "requested_model": "Qwen3.5-27B",
        "resolved_model": "Qwen3.5-27B-exl3-4bpw",
        "preset": "control_vector_pairs",
        "system_source": "docs/wiki/system_prompts/h3_video",
        "system_prompt_sha256": "a" * 64,
        "thinking": True,
        "reasoning_effort": "medium",
        "temperature": None,
        "seed": None,
        "repair_used": False,
        "input_tokens": 912,
        "output_tokens": 1480,
        "created_utc": "2026-11-23T20:00:00Z",
    }
    assert tuple(prov_full.keys()) == sc.CANONICAL_PROVENANCE_KEYS
    parsed_prov = sd.parse_provenance_table(prov_full)
    assert parsed_prov == prov_full
    # Also verify int temperature (e.g. TOML `temperature = 0` or `temperature = 1`)
    parsed_int_temp = sd.parse_provenance_table({**prov_full, "temperature": 1, "seed": 42})
    assert parsed_int_temp["temperature"] == 1 and parsed_int_temp["seed"] == 42

    with tempfile.TemporaryDirectory(prefix="h3_prov_rt_") as tmp:
        tmp_dir = Path(tmp)
        md_sample = """#### Pair 1
##### Prompt A
integrated_multimodal_description: [Shot 1] Live-action medium shot inside a quiet woodshop as a carpenter planes a cedar board smoothly across the five-second take. Normal gravity holds wood shavings on the bench. The camera holds steady. No vocal sound.

overall_soundscape: Crisp hand-plane curls hiss softly across dry cedar grain.

non_diegetic_music: N/A

##### Prompt B
integrated_multimodal_description: [Shot 1] Live-action medium shot inside a quiet woodshop as a carpenter planes a cedar board smoothly across the five-second take. Weightless zero-gravity levitation lifts cedar curls and mallets drifting upward in midair. The camera holds steady. No vocal sound.

overall_soundscape: Crisp hand-plane curls hiss softly across dry cedar grain.

non_diegetic_music: N/A
"""
        rt_spec = sd.RoutingSpec(
            bus="motion",
            regime="a",
            target="description_only",
            streams="both",
            block_band="all",
            needs_ref_release=True,
            families=("fl2va", "ref2va", "h3_image"),
            default_strength=1.0,
        )
        toml_out = tmp_dir / "prov_test.toml"
        sd.write_markdown_to_toml(
            md_sample,
            toml_out,
            name="prov_test",
            quality="zero-gravity workshop levitation",
            a_side="normal downward shop gravity",
            b_side="weightless zero-gravity levitation",
            mode="t2va",
            routing=rt_spec,
            provenance=prov_full,
        )
        ps_loaded = sd.load_pairs(toml_out)
        assert ps_loaded.routing == rt_spec
        assert ps_loaded.provenance["author_kind"] == "heylook_llm"
        assert "temperature" not in ps_loaded.provenance and "seed" not in ps_loaded.provenance

        # Merge full Python 14-key dict (with None temperature/seed) into .safetensors metadata and reload
        ps_with_nulls = sd.PairSet(
            path=ps_loaded.path,
            sha256=ps_loaded.sha256,
            name=ps_loaded.name,
            quality=ps_loaded.quality,
            a_side=ps_loaded.a_side,
            b_side=ps_loaded.b_side,
            mode=ps_loaded.mode,
            frame=ps_loaded.frame,
            exclude_shots=ps_loaded.exclude_shots,
            pairs=ps_loaded.pairs,
            holds=ps_loaded.holds,
            routing=ps_loaded.routing,
            provenance=prov_full,
        )
        vec_meta: dict[str, Any] = {
            "format": sc.FORMAT_VERSION,
            "name": "prov_test",
            "quality": ps_loaded.quality,
            "family": "fl2va",
            "pairs": se.pairs_record(ps_with_nulls, path="prov_test.toml", pair_spans=[], n_holds=0, inline=False),
            "recipe": {
                "checkpoint": {"file": "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "sha256": "0" * 64},
                "loras": [],
                "encoder": {"file": "qwen_3_vl_32b_minimax_h3.safetensors", "sha256": "0" * 64},
            },
            "measured_under": {"mode": "t2va", "frame": "scene", "prefix": "none", "pooling": "diff_span"},
            "points": {"post_refiner": {"unit": "mean_pair_difference", "raw_norm": 65.0, "median_row_norm": 164.0}},
            "quality_report": {"mean_pairwise_cosine": None, "split_half_cosine": None, "per_pair_cosine": []},
            "provenance": {
                "core_revision": "unknown",
                "pack_commit": "unknown",
                "created_utc": "2026-11-23T20:00:00Z",
                "made_by": "check_steering_v2",
            },
        }
        se.merge_toml_metadata_into_vector(vec_meta, ps_with_nulls)
        vec_file = tmp_dir / "prov_test__fl2va__diff_span.safetensors"
        se.save_vector(
            vec_file,
            {"vector.post_refiner": torch.ones((sc.POST_REFINER_DIM,), dtype=torch.float32)},
            vec_meta,
            write_sidecar_md=False,
        )
        _t_rt, m_rt = se.load_vector(vec_file)
        assert m_rt["routing"] == rt_spec.to_dict()
        for k in sc.CANONICAL_PROVENANCE_KEYS:
            assert k in m_rt["provenance"], f"missing canonical provenance key {k!r} after safetensors round-trip"
        assert m_rt["provenance"]["temperature"] is None and m_rt["provenance"]["seed"] is None

    # 2. Stratified hash-seeded pair_compiler over prompt_bank/bank.json
    c1 = pair_compiler._stratified_candidate_texts(
        REPO_ROOT / "prompt_bank",
        mode="t2va",
        prefixes=("t2va_",),
        clause_seed="weightless zero-gravity orbital levitation",
    )
    c2 = pair_compiler._stratified_candidate_texts(
        REPO_ROOT / "prompt_bank",
        mode="t2va",
        prefixes=("t2va_",),
        clause_seed="weightless zero-gravity orbital levitation",
    )
    c_other = pair_compiler._stratified_candidate_texts(
        REPO_ROOT / "prompt_bank",
        mode="t2va",
        prefixes=("t2va_",),
        clause_seed="knitted wool yarn stop-motion world",
    )
    stems_1 = [s for s, _ in c1]
    assert stems_1 == [s for s, _ in c2], "stratified sampler must be deterministic"
    assert stems_1[:6] != sorted(stems_1)[:6], "stratified sampler must not slice alphabetically"
    assert stems_1[:6] != [s for s, _ in c_other[:6]], "distinct concepts should draw distinct hash-seeded scenes"

    c_ref = pair_compiler._stratified_candidate_texts(
        REPO_ROOT / "prompt_bank",
        mode="ref2va",
        prefixes=("ref2va_",),
        clause_seed="mercury liquid chrome",
    )
    all_sampled_stems = set(stems_1) | {s for s, _ in c_ref}
    for banned in ("t2va_covered_market", "ref2va_market_stallholder", "ref2va_woman_blonde_street"):
        assert banned not in all_sampled_stems, f"excluded stem {banned!r} leaked into compiler pool!"

    ps_compiled = pair_compiler.compile_words_to_pairset(
        "weightless zero-gravity orbital levitation",
        name="zg_compiled",
        mode="t2va",
        bank_scenes=6,
        bank_dir=REPO_ROOT / "prompt_bank",
    )
    assert len(ps_compiled.pairs) == 6
    has_dlg = sum(1 for p in ps_compiled.pairs if "<d>" in p.a)
    has_non_dlg = sum(1 for p in ps_compiled.pairs if "<d>" not in p.a)
    assert has_dlg >= 1 and has_non_dlg >= 1, "expected stratified mix of dialogue and non-dialogue bank scenes"

    # Verify root h3_spans was untouched and still works cleanly
    assert callable(h3_spans.spans) and callable(h3_spans.pooling_mask)
    assert steering.__all__

    return "14-key [provenance] round-trip + stratified bank.json sampler (excluded stems blocked, multi-stratum verified)"


def main() -> int:
    checks = [
        ("constants_and_tomls", check_constants_and_toml_routing),
        ("span_cache_fp32", check_span_cache_roundtrip),
        ("bank_cpu_parity", check_bank_numerical_parity_and_stamping),
        ("mixer_and_director", check_mixer_and_director_routing),
        ("provenance_and_sampler", check_provenance_and_stratified_compiler),
    ]
    failed = 0
    print("=== Steering v2 Clean-Slate Verification (`0 GB` VRAM CPU) ===")
    for name, fn in checks:
        try:
            msg = fn()
            print(f"  ok    {name:24s} {msg}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name:24s} {exc}")
            import traceback
            traceback.print_exc()

    if failed:
        print(f"\nFAILED: {failed}/{len(checks)} check group(s) failed")
        return 1
    print(f"\nok: all {len(checks)} steering-v2 check groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
