#!/usr/bin/env python3
"""Generate publication vector SVG figures for Paper v5 in `docs/wiki/steering/paper/figures/`.

Runs with zero external data dependencies by default to build the architectural
flagship diagram `fig_v5_4d_causal_architecture.svg`. When local untracked JSON
metrics under `data/exp_unpooled_sliders/` are available, `--from-data` reports
readiness for regenerating empirical charts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "control_vectors.py").is_file():
            return p
    return Path.cwd()


PAPER_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PAPER_DIR / "figures"
REPO_ROOT = _find_repo_root()


def build_fig_v5_architecture() -> Path:
    """Build Paper v5 Figure 1: Affine Prelude Transport + Causal 4D AV DiT Routing."""
    w, h = 960, 470
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="Inter, system-ui, -apple-system, sans-serif">',
        f'<rect width="{w}" height="{h}" rx="14" fill="#0b1220"/>',
        # Header
        '<text x="480" y="34" fill="#f8fafc" font-size="16.5" font-weight="700" text-anchor="middle">'
        'Causal 4D Prelude Steering in MiniMax H3 (1x Single-Pass AV Architecture)'
        '</text>',
        '<text x="480" y="54" fill="#94a3b8" font-size="11.3" text-anchor="middle">'
        'Zero-weight 193.5 KB operators: Affine Prelude Clearing + Reduced-Interference ROS + 50-Block Depth &amp; Q_aud Routing'
        '</text>',

        # ---------------- PANEL A: Why 1D Hits Veto + Affine Prelude Composition ----------------
        '<rect x="24" y="72" width="442" height="374" rx="11" fill="#111c33" stroke="#334155" stroke-width="1.5"/>',
        '<text x="245" y="96" fill="#38bdf8" font-size="13.5" font-weight="700" text-anchor="middle">'
        '(a) 1x Affine Prelude Transport &amp; Multi-Slider Algebra'
        '</text>',

        # Problem callout
        '<rect x="42" y="110" width="406" height="62" rx="8" fill="#1e1b2e" stroke="#f87171" stroke-width="1.2"/>',
        '<text x="245" y="129" fill="#fca5a5" font-size="11.2" font-weight="700" text-anchor="middle">'
        'Frozen-Anchor Prompt Veto on 5376-D TokenRefiner States'
        '</text>',
        '<text x="245" y="146" fill="#e2e8f0" font-size="10.2" text-anchor="middle">'
        'H is computed once and re-attended across all 50 blocks x K denoising steps.'
        '</text>',
        '<text x="245" y="162" fill="#cbd5e1" font-size="10.2" text-anchor="middle">'
        'Rank-1 +s v_1D leaves Pole-A baseline cues intact; raw stacks collide up to 42.4%.'
        '</text>',

        # Step box: Pipeline
        '<rect x="42" y="186" width="406" height="148" rx="8" fill="#0c2d48" stroke="#38bdf8" stroke-width="1.4"/>',
        '<text x="245" y="206" fill="#e0f2fe" font-size="11.5" font-weight="700" text-anchor="middle">'
        'In-Place 1x Operator (L\' = L exact 3D RoPE cursor invariance)'
        '</text>',
        '<text x="58" y="228" fill="#fda4af" font-size="10.5" font-weight="600">'
        '1. Notch-Erase (PCRR V_r):'
        '</text>',
        '<text x="206" y="228" fill="#e2e8f0" font-size="10.2">'
        '-clip(s,0,1) (H - h_bar) V_r^T diag(g) V_r'
        '</text>',
        '<text x="58" y="250" fill="#6ee7b7" font-size="10.5" font-weight="600">'
        '2. Target Push (+mu_d):'
        '</text>',
        '<text x="206" y="250" fill="#e2e8f0" font-size="10.0">'
        '+ s mu_d (tau=0.15) [1A/1C Phase 2 GPU]'
        '</text>',
        '<text x="58" y="272" fill="#fde68a" font-size="10.5" font-weight="600">'
        '3. Per-Slider ROS + Clamp:'
        '</text>',
        '<text x="206" y="272" fill="#e2e8f0" font-size="10.2">'
        'V_r^(j) perp mu_d^(-j), union lam_max &lt;= 1.0'
        '</text>',
        '<text x="58" y="294" fill="#c4b5fd" font-size="10.5" font-weight="600">'
        '4. Smooth Norm Knee:'
        '</text>',
        '<text x="206" y="294" fill="#e2e8f0" font-size="10.0">'
        '67.5 RMS match + 95.5-&gt;98.5 (4MP fixed)'
        '</text>',
        '<text x="58" y="316" fill="#7dd3fc" font-size="10.5" font-weight="600">'
        '5. Bus-Scoped V_r (CPU):'
        '</text>',
        '<text x="206" y="316" fill="#e2e8f0" font-size="10.2">'
        '7.94x look/aud overlap cut (100% Look)'
        '</text>',

        # Bottom summary of Panel A
        '<rect x="42" y="348" width="406" height="82" rx="8" fill="#0f172a" stroke="#475569"/>',
        '<text x="245" y="368" fill="#34d399" font-size="11" font-weight="700" text-anchor="middle">'
        'Stage 1 + Gate 4 Tested Cases (Zero Prompt Rewrite)'
        '</text>',
        '<text x="245" y="386" fill="#cbd5e1" font-size="10.1" text-anchor="middle">'
        '4.0 MP lighthouse clay+noir: diptych eliminated (clip 0.07255 -&gt; 0.00790, 3.07s)'
        '</text>',
        '<text x="245" y="403" fill="#cbd5e1" font-size="10.1" text-anchor="middle">'
        'box_office stack3 saturation: 0.3187 -&gt; 0.1661 with resolved-mask ROS'
        '</text>',
        '<text x="245" y="420" fill="#94a3b8" font-size="9.5" text-anchor="middle">'
        'CPU: Path 1B &lt;1.1% MSE (falsified); Path 1A/1C unproven on GPU'
        '</text>',

        # ---------------- PANEL B: 50-Block AV DiT Causal Anatomy & 4D Routing ----------------
        '<rect x="486" y="72" width="450" height="374" rx="11" fill="#111c33" stroke="#34d399" stroke-width="1.5"/>',
        '<text x="711" y="96" fill="#34d399" font-size="13.5" font-weight="700" text-anchor="middle">'
        '(b) Causal 4D Depth &amp; Query Routing Across 50 AV DiT Blocks'
        '</text>',

        # Three block bands inside the DiT
        '<rect x="504" y="112" width="128" height="114" rx="8" fill="#0f172a" stroke="#64748b" stroke-width="1.2"/>',
        '<text x="568" y="132" fill="#e2e8f0" font-size="11.2" font-weight="700" text-anchor="middle">Blocks 0..19</text>',
        '<text x="568" y="149" fill="#93c5fd" font-size="10" font-weight="600" text-anchor="middle">Layout &amp; Identity</text>',
        '<text x="568" y="166" fill="#cbd5e1" font-size="9.3" text-anchor="middle">Unsteered under 20_49</text>',
        '<text x="568" y="181" fill="#6ee7b7" font-size="9.3" font-weight="600" text-anchor="middle">Stage D N=1 DINOv3:</text>',
        '<text x="568" y="196" fill="#6ee7b7" font-size="9.3" font-weight="600" text-anchor="middle">1.52x mean / 1.75x peak</text>',
        '<text x="568" y="211" fill="#94a3b8" font-size="9.0" text-anchor="middle">(vs 0..49; below base)</text>',

        '<rect x="642" y="112" width="138" height="114" rx="8" fill="#064e3b" stroke="#34d399" stroke-width="1.4"/>',
        '<text x="711" y="132" fill="#ecfdf5" font-size="11.2" font-weight="700" text-anchor="middle">Blocks 20..29</text>',
        '<text x="711" y="149" fill="#6ee7b7" font-size="10" font-weight="600" text-anchor="middle">Visual Commitment</text>',
        '<text x="711" y="167" fill="#d1fae5" font-size="9.6" text-anchor="middle">24 fps spatial/material</text>',
        '<text x="711" y="182" fill="#d1fae5" font-size="9.6" text-anchor="middle">breakout window;</text>',
        '<text x="711" y="197" fill="#fde68a" font-size="9.6" font-weight="600" text-anchor="middle">10-block lead before</text>',
        '<text x="711" y="212" fill="#fde68a" font-size="9.6" text-anchor="middle">audio branch wakes up</text>',

        '<rect x="790" y="112" width="128" height="114" rx="8" fill="#172554" stroke="#60a5fa" stroke-width="1.2"/>',
        '<text x="854" y="132" fill="#eff6ff" font-size="11.2" font-weight="700" text-anchor="middle">Blocks 30..49</text>',
        '<text x="854" y="149" fill="#93c5fd" font-size="10" font-weight="600" text-anchor="middle">40 Hz Audio + Sink</text>',
        '<text x="854" y="167" fill="#dbeafe" font-size="9.6" text-anchor="middle">Audio emerges in</text>',
        '<text x="854" y="182" fill="#dbeafe" font-size="9.6" text-anchor="middle">blocks 31..47;</text>',
        '<text x="854" y="197" fill="#fda4af" font-size="9.6" font-weight="600" text-anchor="middle">Block 49 dense sink</text>',
        '<text x="854" y="212" fill="#93c5fd" font-size="9.4" text-anchor="middle">(SOL_DENSE_TAIL)</text>',

        # Query-split box
        '<rect x="504" y="240" width="414" height="104" rx="8" fill="#0f293a" stroke="#38bdf8" stroke-width="1.3"/>',
        '<text x="711" y="260" fill="#e0f2fe" font-size="11.5" font-weight="700" text-anchor="middle">'
        'Stage 2 Compiler-Safe Hooks (H3SteerStepWorkspace, 340e8ded)'
        '</text>',
        '<text x="711" y="280" fill="#bae6fd" font-size="10.2" text-anchor="middle">'
        '1x Query Split: Q_vid -&gt; K_txt_steered  |  Q_aud -&gt; K_txt_base'
        '</text>',
        '<text x="711" y="298" fill="#6ee7b7" font-size="9.8" font-weight="600" text-anchor="middle">'
        'Gate 1 (90f): 1.003x, -0.014 st | Showcase C: +0.63 st (Q_aud-&gt;K_vid)'
        '</text>',
        '<text x="711" y="316" fill="#cbd5e1" font-size="9.7" text-anchor="middle">'
        '+0 graph breaks across 4 backends | Mode 3 ref2va: +1.06 dB audio'
        '</text>',
        '<text x="711" y="333" fill="#94a3b8" font-size="9.3" text-anchor="middle">'
        '4D axes: Span x Depth (20..49) x Step [0,0.33] Dual-Lock x Stream Q_aud'
        '</text>',

        # Bottom summary of Panel B
        '<rect x="504" y="356" width="414" height="74" rx="8" fill="#0f172a" stroke="#475569"/>',
        '<text x="711" y="376" fill="#fde68a" font-size="10.8" font-weight="700" text-anchor="middle">'
        'Verified Boundaries &amp; Phase 2 Queue'
        '</text>',
        '<text x="711" y="394" fill="#cbd5e1" font-size="9.9" text-anchor="middle">'
        'Dual-Lock: tags==0 VLM vision + 24-ch VAE pins RGB in [0, 0.33]; Path 1C queued.'
        '</text>',
        '<text x="711" y="411" fill="#cbd5e1" font-size="9.5" text-anchor="middle">'
        'FrozenVideoCache: Mode 0 only | 1x Q_vid gated vs 2x Sol'
        '</text>',
        '</svg>',
    ]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "fig_v5_4d_causal_architecture.svg"
    out_txt = "\n".join(svg) + "\n"
    ET.fromstring(out_txt)
    out_path.write_text(out_txt, encoding="utf-8")
    return out_path


def inspect_optional_data() -> None:
    data_dir = REPO_ROOT / "data" / "exp_unpooled_sliders"
    needed = (
        data_dir / "phase5_cpu_audits" / "bank_cross_interference_audit.json",
        data_dir / "phase5_cpu_audits" / "multivector_pole_b_audit.json",
        data_dir / "phase5_cpu_audits" / "stage3_cpu_prototypes.json",
        data_dir / "stage1_2026_11_22",
        data_dir / "stage2_2026_11_22",
    )
    for p in needed:
        state = "FOUND" if p.exists() else "MISSING (untracked local artifact)"
        print(f"  {p.relative_to(REPO_ROOT)}: {state}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build tracked Paper v5 vector figures.")
    ap.add_argument("--from-data", action="store_true", help="Inspect optional untracked data/ JSON readiness")
    args = ap.parse_args(argv)

    out_path = build_fig_v5_architecture()
    print(f"Wrote {out_path.relative_to(REPO_ROOT)} (XML verified)")
    if args.from_data:
        print("Optional empirical data inventory:")
        inspect_optional_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
