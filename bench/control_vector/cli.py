r"""Argument parser and CLI command router for `bench/control_vectors.py`.

The Four Control Vector Creation Methods & Their Mathematical Objectives
========================================================================
MiniMax H3 extracts text conditioning from `Qwen3-VL-32B-Instruct` truncated after
`layers.49` (`5120`-d), projects it through the DiT's `condition_proj` (`5120 -> 5376`),
and refines it through the 2-block bidirectional `TokenRefiner` (`5376 -> 5376`)
before the 50-block audio-video DiT reads it via cross-attention.

Every H3 control vector is a direction $\\mathbf{v}_{\\text{PR}} \\in \\mathbb{R}^{5376}$
added to the post-refiner token stream inside `MiniMaxH3ApplyControlVectors` during
DiT denoising. There is no gradient training loop; each creation method computes
$\\mathbf{v}_{\\text{PR}}$ in closed form under a specific mathematical objective:

1. `from-prompt` (In-Place Target-Prompt Contrast; also `build --target-prompt --words`):
   - **Mathematical Objective**:
     $$\\mathbf{v}_{\\text{PR}} = (I - U_{\\text{hold}}^\\top U_{\\text{hold}})\\Big(\\operatorname{pool}\\big(\\operatorname{TokenRefiner}(E(P_B))\\big) - \\operatorname{pool}\\big(\\operatorname{TokenRefiner}(E(P_A))\\big)\\Big)$$
     where $P_A$ and $P_B$ are your exact target scene prompt (or `---`-separated
     multi-scene prompts) differing ONLY at the substituted clause(s) `'pole_a -> pole_b'`
     (plus optional `soundscape: a -> b` and `hold: a -> b`), pooled over the exact
     differing token span (`diff_span`).
   - **When `--ref-images` applies**: Pass `--ref-images <img.png>` on `ref2va` shots so
     `Qwen3-VL` and `TokenRefiner` encode $P_A, P_B$ with the true `<Picture 1>` vision
     prefix present in causal/bidirectional attention.
   - **Strengths**: Exact contextual attention and M-RoPE token alignment for your
     specific shot; zero grammar drift; supports multi-clause `;` edits.
   - **Weaknesses**: Requires a GPU encoder pass (`~7s`); tailored to the supplied
     target prompt rather than a reusable multi-scene library vector.

2. `from-toml` / `build` / `make` (Multi-Scene Reusable Pair-Bank Vector):
   - **Mathematical Objective**:
     $$\\mathbf{v}_{\\text{PR}} = (I - U_{\\text{hold}}^\\top U_{\\text{hold}}) \\frac{\\sum_{i=1}^N w_i \\big(\\mathbf{r}_{B,i} - \\mathbf{r}_{A,i}\\big)}{\\sum_{i=1}^N w_i}$$
     averaged in float64 across $N \\approx 10$ diverse H3 scenes (`[[pair]]`) so
     scene-specific nouns, props, and room geometry cancel out while the shared
     concept direction survives, with nuisance directions (`[[hold]]`) projected out
     via SVD (`control_vectors.py::hold_basis`).
   - **When `--ref-images` applies**: Optional on `ref2va` `.toml` banks when you want
     the multi-scene expectation conditioned on a specific `<Picture 1>` reference prefix.
   - **Strengths**: Scene-invariant, reusable library vector with full reliability
     diagnostics (`mean_pairwise_cosine`, `split_half_cosine`, `per_pair_cosine`).
   - **Weaknesses**: Requires authoring or LLM-generating (`generate-pairs` / `from-md`)
     10 valid single-clause pairs and a GPU encoding pass (`~10s` per stem).

3. `from-words` / `words` (Instant `< 2 ms` CPU Unigram Synthesis via J-Lens; also inline `words` on `MiniMaxH3ApplyControlVectors`):
   - **Mathematical Objective**:
     $$\\hat{\\mathbf{v}}_{\\text{PR}}^{(f)} = 67.5 \\cdot \\operatorname{normalize}\\left(\\bar{J}_{\\text{ref}}^{(f)} \\Sigma_{49, \\text{filt}}^{0.30} \\Delta\\boldsymbol{\\mu}\\right)$$
     where $\\Delta\\boldsymbol{\\mu}$ is the difference between positive and negative
     unit-normalized mean-centered rows of $M_{\\text{valid}} = W_{\\text{head}}[\\text{valid}]\\bar{J}_{49 \\to 63}$,
     pulled through the spectral-filtered `layers.49` covariance $\\Sigma_{49, \\text{filt}}^{0.30}$
     and pushed forward through the expected `TokenRefiner` Jacobian $\\bar{J}_{\\text{ref}}^{(f)}$
     stored in `models/j_lens/j_lens_h3.safetensors` (built via `build-j-lens`).
   - **When `--ref-images` applies**: Not used (operates purely in precomputed vocabulary/Jacobian space on CPU).
   - **Strengths**: `< 2 ms` on CPU, `0 GB` VRAM, zero `.toml` or prompt required; runs
     inline inside `MiniMaxH3ApplyControlVectors` (`words` widget) at inference time.
   - **Weaknesses**: Commutative bag-of-words approximation (`cos_5376 = 0.46..0.64` vs.
     10-pair contextual vectors); cannot represent word order, multi-word syntax, or
     relational trajectories (`A -> B` vs. `B -> A`).

4. `from-images` / `make-media` (Visual `<Picture 1>` Reference-Prefix Contrast):
   - **Mathematical Objective**:
     $$\\mathbf{v}_{\\text{PR}} = \\frac{1}{N}\\sum_{i=1}^N \\Big(\\operatorname{pool}\\big(\\operatorname{TokenRefiner}(E(\\text{Pic}_{B,i}, P_0))\\big) - \\operatorname{pool}\\big(\\operatorname{TokenRefiner}(E(\\text{Pic}_{A,i}, P_0))\\big)\\Big)$$
     where $\\text{Pic}_{A,i}$ and $\\text{Pic}_{B,i}$ are contrasting `<Picture 1>`
     images resized to identical `32`-px M-RoPE grid dimensions (`match_media_pair_geometry`)
     and encoded with the exact same neutral `ref2va` scene prompt $P_0$.
   - **Strengths**: Extracts visual lighting, grade, texture, or material shifts directly
     from images when words cannot describe the look; default `pooling='scene_text'`
     captures the cross-attentive imprint that `<Picture 1>` leaves on downstream tag-1
     scene tokens so the resulting vector steers even text-only prompts at inference.
   - **Weaknesses**: Requires matched `<Picture 1>` A/B image pairs; composition/pose
     differences between Image A and Image B will leak unless scrubbed via `--hold` / `--hold-words`.
"""

from __future__ import annotations

import argparse
import os
import sys

import control_vectors as cv
import h3_spans
from workflows import h3_config

from .build_cmd import cmd_build, cmd_rebuild, cmd_words
from .grading import cmd_grade, cmd_inspect
from .helpers import _resolve_requested_families
from .j_lens_builder import cmd_build_j_lens
from .llm_pairs import DEFAULT_LLM_MAX_TOKENS, SUPPORTED_LLM_PROVIDERS, cmd_generate_pairs
from .make_cmds import cmd_from_md, cmd_make, cmd_make_media
from .sculpt import cmd_sculpt

CLI_DESCRIPTION = """Build, inspect, meter, and grade MiniMax H3 control vectors."""

CLI_EPILOG = """Four Control Vector Creation Methods:
  1. from-prompt               In-place contextual contrast ('pole_a -> pole_b') on your exact --target-prompt (~7s GPU)
  2. from-toml (alias: build)  Multi-scene reusable vector from a 10-pair .toml file (or --all) (~10s GPU)
  3. from-words (alias: words) Instant <2 ms CPU synthesis from 3-5 words via J-Lens (0 GB VRAM; also inline in ComfyUI)
  4. from-images (make-media)  Visual <Picture 1> image-pair contrast (--image-a / --image-b) with 32-px M-RoPE matching

J-Lens Operator Bundle:
  build-j-lens                 Verify (--verify-only) or rebuild models/j_lens/j_lens_h3.safetensors (h3_j_lens_v1)

H3 Model Families (--families):
  MiniMax H3 has 2 DiT text-path (condition_proj + token_refiner) families:
    fl2va  : Video T2VA / FL2VA / I2VA  AND  P-H3-Image (both Image T2I and Image Edit)
    ref2va : Video Ref2VA / Guide Video Edit / Masked V2V
    both   : (default) Builds both <name>__fl2va__*.safetensors and <name>__ref2va__*.safetensors
             in a single pass, covering all 4 workflows (Video fl2va, Video ref2va, Image T2I,
             and Image Edit). At runtime, MiniMaxH3ApplyControlVectors automatically swaps to
             the matching family sibling when both files exist in models/control_vectors/."""

_FAMILIES_METAVAR = "{both,fl2va,ref2va}"
_FAMILIES_HELP = (
    "H3 text-path family to build: 'both' (default: writes fl2va + ref2va, covering Video fl2va, "
    "Video ref2va, Image T2I, and Image Edit), 'fl2va' (Video T2VA/FL2VA/I2VA + P-H3-Image T2I/Edit; "
    "aliases: 'image', 't2i', 'image_edit'), or 'ref2va' (Video Ref2VA)"
)


def _validate_families_arg(raw: str) -> str:
    try:
        _resolve_requested_families(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return raw


_MASK_METAVAR = "{" + ",".join(h3_spans.INFERENCE_MASK_MODES) + "}"
_MASK_FIELDS_HELP = (
    "Or one or more prompt fields joined by '+', steering exactly those: "
    + ", ".join(h3_spans.PROMPT_FIELDS)
    + " (the node's 'fields' input)"
)


def _validate_mask_arg(raw: str) -> str:
    if not h3_spans.is_inference_mask(raw):
        raise argparse.ArgumentTypeError(
            f"invalid mask {raw!r}: choose from {', '.join(h3_spans.INFERENCE_MASK_MODES)}, "
            f"or prompt fields joined by '+' ({', '.join(h3_spans.PROMPT_FIELDS)})"
        )
    return raw


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=CLI_DESCRIPTION,
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pui = sub.add_parser(
        "ui",
        help="Launch the self-contained local browser UI for authoring, previewing, and building control vectors",
    )
    pui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    pui.add_argument("--port", type=int, default=0, help="Port (default: 0 picks an unused ephemeral port)")
    pui.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not auto-open a browser tab")
    pui.add_argument(
        "--token",
        default=None,
        help="Session token the launch link carries (default: a new random one each start). Set it to keep one link across restarts",
    )
    pui.add_argument("--toml-dir", dest="toml_dir", default=None, help="Optional directory override for .toml pair packs")
    pui.add_argument("--vec-dir", dest="vec_dir", default=None, help="Optional directory override for .safetensors control vectors")
    pui.add_argument("--encoder", default=None, help="Optional conditioning encoder filename override")
    pui.add_argument("--dit", default=None, help="Optional DiT checkpoint filename override")
    pui.add_argument("--device", default=None, help="Optional conditioning encoder device override")
    pui.add_argument(
        "--llm-provider",
        dest="llm_provider",
        choices=SUPPORTED_LLM_PROVIDERS,
        default=None,
        help="Default generative LLM provider for pair authoring ('transformers' or 'exllamav3')",
    )
    pui.add_argument(
        "--llm-model",
        dest="llm_model",
        default=None,
        help="Default generative LLM model directory path or HuggingFace model ID",
    )

    pfp = sub.add_parser(
        "from-prompt",
        help="Method 1: Build a scene-specific control vector in-place on --target-prompt / --target-prompt-file from '--words \"pole_a -> pole_b\"'",
    )
    pfp_tp = pfp.add_mutually_exclusive_group(required=True)
    pfp_tp.add_argument(
        "--target-prompt",
        dest="target_prompt",
        default="",
        help="Literal inline H3 scene prompt text (separate multiple scenes with '---')",
    )
    pfp_tp.add_argument(
        "--target-prompt-file",
        dest="target_prompt_file",
        default="",
        help="Target H3 scene prompt file path or prompt_bank/<stem>.txt id",
    )
    pfp.add_argument(
        "--words",
        required=True,
        help="In-place 'pole_a -> pole_b' contrast (supports ';' or newline multi-clause, 'soundscape: a -> b', and 'hold: a -> b')",
    )
    pfp.add_argument(
        "--target-weight",
        dest="target_weight",
        type=float,
        default=2.5,
        help="Weight for Pair 0 when multiple '---'-separated target scenes or --bank-scenes are passed (default: 2.5)",
    )
    pfp.add_argument(
        "--bank-scenes",
        dest="bank_scenes",
        type=int,
        default=0,
        help="Optional number of mode-matched prompt_bank scenes to append alongside --target-prompt for multi-scene nuisance cancellation (default: 0)",
    )
    pfp.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pfp.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pfp.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pfp.add_argument("--prompt", default="", help="Optional prompt override for Token-Wise Prompt Receptor Scanning")
    pfp.add_argument("--unit-mean", dest="unit_mean", action="store_true", help="Average unit-normalized pair directions")
    pfp.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    pfp.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode or comma-separated modes ('diff_span', 'from_first_diff', 'scene_text', or 'all'; default: 'diff_span')",
    )
    pfp.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pfp.add_argument("--output-name", dest="output_name", default="", help=argparse.SUPPRESS)
    pfp.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override (auto-detected from --target-prompt if omitted)")
    pfp.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pfp.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pfp.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pfp.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pfp.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also write the compiled .toml file under data/control_vectors/")
    pfp.add_argument("--toml-out", dest="toml_out", default=None, help="Optional output path/dir for generated .toml")
    pfp.add_argument("--dit", dest="dit", default=None, help="Explicit DiT diffusion checkpoint filename override")
    pfp.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pfp.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pfp.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override")
    pfp.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build for stock 30-step base DiT without Pika LoRAs")
    pfp.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va <Picture 1> prefix")
    pfp.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pfp.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pfp.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pfp.add_argument("--seed", type=int, default=0)
    pfp.add_argument("--device", default="default")
    pfp.add_argument("--refiner-device", default="cpu")
    pfp.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")
    pfp.add_argument("--overwrite", action="store_true", help="Overwrite existing .toml and .safetensors files")

    pb = sub.add_parser(
        "build",
        aliases=["from-toml"],
        help="Method 2 (from-toml / build): Build reusable control vector(s) from a .toml/.md pair file (or --all), or in-place --words",
    )
    pb.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Input LLM Markdown (.md) file, pair TOML (.toml) file, or stem under data/control_vectors/ or prompt_bank/control_vectors/ (omit when using --words, --from-words, or --all)",
    )
    pb.add_argument(
        "--words",
        default="",
        help="In-place 'pole_a -> pole_b' contrast (supports ';' or newline multi-clause, 'soundscape:', and 'hold:' lines) compiled directly on --target-prompt",
    )
    pb_tp = pb.add_mutually_exclusive_group(required=False)
    pb_tp.add_argument(
        "--target-prompt",
        dest="target_prompt",
        default="",
        help="Literal inline H3 scene prompt text for in-place --words compilation",
    )
    pb_tp.add_argument(
        "--target-prompt-file",
        dest="target_prompt_file",
        default="",
        help="Target H3 scene prompt file path or prompt_bank/<stem>.txt id for in-place --words compilation",
    )
    pb.add_argument(
        "--target-weight",
        dest="target_weight",
        type=float,
        default=2.5,
        help="Weight for Pair 0 when --target-prompt is passed (default: 2.5)",
    )
    pb.add_argument(
        "--bank-scenes",
        dest="bank_scenes",
        type=int,
        default=0,
        help="Optional number of mode-matched prompt_bank scenes to append alongside --target-prompt when compiling --words (default: 0)",
    )
    pb.add_argument(
        "--all",
        dest="build_all",
        action="store_true",
        help="Build vectors for all .toml banks in prompt_bank/control_vectors/ in a single encoder load",
    )
    pb.add_argument(
        "--from-words",
        dest="from_words",
        default="",
        help="Synthesize a 5376-d control vector directly from simple positive words in <5 ms on CPU via J-Lens (no .toml or Qwen load required)",
    )
    pb.add_argument(
        "--against-words",
        dest="against_words",
        default="",
        help="Optional negative words when using --from-words (defaults to neutral baseline words)",
    )
    pb.add_argument(
        "--hold",
        action="append",
        default=None,
        help="Optional .safetensors control vector(s) to project out (at --hold-alpha) across any build mode",
    )
    pb.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU (at --hold-alpha) across any build mode",
    )
    pb.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when using --hold or --hold-words (default: 1.0)",
    )
    pb.add_argument(
        "--prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning and @prompt_subject",
    )
    pb.add_argument(
        "--unit-mean",
        dest="unit_mean",
        action="store_true",
        help="Average unit-normalized pair directions (equal directional weight per pair) scaled to mean pair norm",
    )
    pb.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    pb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode or comma-separated modes (e.g. 'diff_span', 'from_first_diff', 'scene_text', or 'all'; default: 'diff_span')",
    )
    pb.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pb.add_argument("--output-name", dest="output_name", default="", help=argparse.SUPPRESS)
    pb.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override when importing .md or compiling --words")
    pb.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pb.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pb.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pb.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pb.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also write the compiled .toml file under data/control_vectors/ when using --words")
    pb.add_argument("--toml-out", dest="toml_out", default=None, help="Optional output path/dir for generated .toml")
    pb.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional output path for extracted Test Scene Prompt")
    pb.add_argument("--dit", dest="dit", default=None, help="Explicit DiT diffusion checkpoint filename override")
    pb.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pb.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pb.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override (default: standard Pika H3 Flex LoRA per family)")
    pb.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build for stock 30-step base DiT without Pika LoRAs")
    pb.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pb.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pb.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pb.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--device", default="default")
    pb.add_argument("--refiner-device", default="cpu")
    pb.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")
    pb.add_argument("--overwrite", action="store_true", help="Overwrite existing .toml and .safetensors files")

    pfm = sub.add_parser(
        "from-md",
        aliases=["import-md"],
        help="Convert an LLM-authored control-vector Markdown file into a validated .toml pair file",
    )
    pfm.add_argument("md_path", help="Input Markdown (.md) file path")
    pfm.add_argument(
        "--out",
        default=None,
        help="Output .toml file path or output directory (defaults to data/control_vectors/<stem>.toml)",
    )
    pfm.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pfm.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override (auto-detected if omitted)")
    pfm.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pfm.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pfm.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pfm.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pfm.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional path to write the extracted Test Scene Prompt text")
    pfm.add_argument("--overwrite", action="store_true", help="Overwrite an existing .toml file at the target path")

    pm = sub.add_parser("make", help="Build a control vector from a pair TOML file or in-place --words + --target-prompt")
    pm.add_argument("pairs", nargs="?", default=None, help="Pair TOML file path or stem under data/control_vectors/ or prompt_bank/control_vectors/ (omit when using --words)")
    pm.add_argument("--words", default="", help="In-place 'pole_a -> pole_b' contrast compiled directly on --target-prompt / --target-prompt-file")
    pm_tp = pm.add_mutually_exclusive_group(required=False)
    pm_tp.add_argument("--target-prompt", dest="target_prompt", default="", help="Literal inline H3 scene prompt text for in-place --words compilation")
    pm_tp.add_argument("--target-prompt-file", dest="target_prompt_file", default="", help="Target H3 scene prompt file path or prompt_bank/<stem>.txt id for in-place --words compilation")
    pm.add_argument("--target-weight", dest="target_weight", type=float, default=2.5, help="Weight for Pair 0 when --target-prompt is passed")
    pm.add_argument(
        "--bank-scenes",
        dest="bank_scenes",
        type=int,
        default=0,
        help="Optional number of mode-matched prompt_bank scenes to append alongside --target-prompt when compiling --words (default: 0)",
    )
    pm.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pm.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pm.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pm.add_argument("--pooling", choices=cv.VALID_POOLING, default="diff_span")
    pm.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    pm.add_argument("--unit-mean", dest="unit_mean", action="store_true", help="Average unit-normalized pair directions")
    pm.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pm.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pm.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override (default: standard Pika H3 Flex LoRA per family)")
    pm.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build for stock 30-step base DiT without Pika LoRAs")
    pm.add_argument("--ref-images", default=None, help="Comma-separated reference images for ref2va prefix")
    pm.add_argument("--ref-role", default="subject", help="Role for first reference image")
    pm.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    pm.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    pm.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d) from saved .safetensors")
    pm.add_argument("--name", "--output-name", dest="output_name", default="", help="Optional vector name / file stem override")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--device", default="default")
    pm.add_argument("--refiner-device", default="cpu")
    pm.add_argument("--out", default=None, help="Output .safetensors path or filename")
    pm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file at the target path")

    pmm = sub.add_parser(
        "make-media",
        aliases=["from-images"],
        help="Method 4 (from-images / make-media): Build a control vector from contrasting <Picture 1> image pair(s) with 32-px M-RoPE grid matching",
    )
    pmm.add_argument("--image-a", required=True, help="Baseline/negative pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--image-b", required=True, help="Positive/steered pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--name", "--output-name", dest="output_name", default="media_contrast", help="Output vector stem (default: 'media_contrast')")
    pmm.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="ref2va",
        help="H3 text-path family to build media vectors for ('both', 'fl2va', or 'ref2va'; default: 'ref2va')",
    )
    pmm.add_argument("--pooling", choices=cv.VALID_MEDIA_POOLING, default="scene_text")
    pmm.add_argument("--short-edge", type=int, default=512, help="Target shorter edge in pixels (snapped to 32)")
    pmm.add_argument("--scene-prompt", default="", help="Required H3 ref2va scene prompt")
    pmm.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pmm.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pmm.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pmm.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pmm.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pmm.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pmm.add_argument("--lora", action="append", default=None, help="LoRA file[:strength]")
    pmm.add_argument("--no-lora", dest="no_lora", action="store_true", help="Build without Pika LoRAs")
    pmm.add_argument("--device", default="default")
    pmm.add_argument("--refiner-device", default="cpu")
    pmm.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory")
    pmm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file")

    psculpt = sub.add_parser(
        "sculpt",
        help="Compose multi-vector recipes (add, add_norm_matched, hold) + --hold-words (--hold-alpha) into Ready-to-Render .safetensors files in <5 ms on CPU",
    )
    psculpt.add_argument("vectors", nargs="+", help="One or more .safetensors vector files or stems to compose")
    psculpt.add_argument(
        "--modes",
        default="add",
        help="Comma-separated modes matching 'vectors' ('add', 'add_norm_matched', 'hold'; default: 'add')",
    )
    psculpt.add_argument(
        "--strengths",
        default="1.0",
        help="Comma-separated strengths (or hold alphas in [0.0, 2.0]) matching 'vectors' (default: '1.0')",
    )
    psculpt.add_argument("--invert", action="store_true", help="Invert the primary vector direction (multiply strength by -1.0)")
    psculpt.add_argument("--hold", action="append", default=None, help="Additional .safetensors vector(s) to project out (`mode = 'hold'`)")
    psculpt.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    psculpt.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Attenuation alpha in [0.0, 2.0] for --hold and --hold-words (default: 1.0)",
    )
    psculpt.add_argument("--prompt", default="", help="Optional target H3 scene prompt for prompt-coupling readout")
    psculpt.add_argument("--name", default=None, help="Output vector stem")
    psculpt.add_argument("--output-name", dest="output_name", default="", help=argparse.SUPPRESS)
    psculpt.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    psculpt.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory")
    psculpt.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing output files (default: True)")

    pi = sub.add_parser("inspect", help="Inspect a control vector file and optionally meter prompts")
    pi.add_argument("vector", help="Path or filename in models/control_vectors/")
    pi.add_argument(
        "--hold",
        action="append",
        default=None,
        help="Optional control vector .safetensors file(s) to project out before J-Lens readout",
    )
    pi.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU before J-Lens readout",
    )
    pi.add_argument(
        "--prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning",
    )
    pi.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Soft or over-projected hold attenuation alpha in [0.0, 2.0] when --hold is passed (default: 1.0)",
    )
    pi.add_argument("--check-dit", dest="check_dit", default=None, help="Verify compatibility against a DiT checkpoint")
    pi.add_argument("--check-unet", dest="check_dit", default=None, help=argparse.SUPPRESS)
    pi.add_argument("--check-encoder", default=None, help="Verify compatibility against an encoder checkpoint")
    pi.add_argument("--check-lora", action="append", default=None)
    pi.add_argument("--allow-recipe-mismatch", action="store_true")
    pi.add_argument("--meter", action="append", default=None, help="Prompt-bank id or file path to meter")
    pi.add_argument("--meter-look-family", action="store_true", help="Meter t2va_look_anchor and all 8 look siblings (E2.8)")
    pi.add_argument(
        "--mask",
        type=_validate_mask_arg,
        metavar=_MASK_METAVAR,
        default=None,
        help="Optional inference span mask override when metering prompts (e.g. 'scene_only', 'look_only', 'description_only', 'soundscape_only'). " + _MASK_FIELDS_HELP,
    )
    pi.add_argument("--device", default="default")
    pi.add_argument("--refiner-device", default="cpu")

    pg = sub.add_parser("grade", help="Grade activation/prediction movement against nulls and edit baselines")
    pg.add_argument("--vector", required=True, help="Vector .safetensors path or name")
    pg.add_argument("--prompt", required=True, help="Base prompt id or path")
    pg.add_argument("--edit-prompt", default=None, help="Edit baseline prompt id or path")
    pg.add_argument("--pairs", default=None, help="Pair TOML file to build flip_null")
    pg.add_argument(
        "--mask",
        type=_validate_mask_arg,
        metavar=_MASK_METAVAR,
        default="scene_only",
        help="Inference span mask applied during context and DiT grading (default: 'scene_only', matching runtime nodes). " + _MASK_FIELDS_HELP,
    )
    pg.add_argument("--strengths", default="0,0.5,1.0,2.0,-1.0")
    pg.add_argument("--probe-steps", default="0,8,15", help="Schedule step indices for DiT forward grading")
    pg.add_argument("--no-dit", action="store_true", help="Run post-refiner context movement only without loading the DiT")
    pg.add_argument("--sol", action="store_true", help="Opt in to MiniMaxH3Sol sparse attention during DiT grading (default: stock dense attention)")
    pg.add_argument("--compare-unmasked", action="store_true", help="Also evaluate mask='all_rows' for E2.3")
    pg.add_argument("--width", type=int, default=1152)
    pg.add_argument("--height", type=int, default=768)
    pg.add_argument("--length", type=int, default=h3_config.LENGTH)
    pg.add_argument("--noise-seed", type=int, default=730451892)
    pg.add_argument("--dit", dest="dit", default=None, help="DiT diffusion checkpoint filename")
    pg.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    pg.add_argument("--encoder", default=None)
    pg.add_argument("--lora", action="append", default=None, help="LoRA file[:strength] (defaults to vector recipe.loras)")
    pg.add_argument("--no-lora", action="store_true", help="Ignore vector recipe.loras during grading")
    pg.add_argument("--manual-sigmas", default=None, help="Comma-separated manual sigmas override")
    pg.add_argument("--pika", action="store_true", help="Use Pika H3 Flex 6-step manual sigmas and default probe steps 0,2,5")
    pg.add_argument("--ref-images", default=None)
    pg.add_argument("--ref-role", default="subject")
    pg.add_argument("--seed", type=int, default=0)
    pg.add_argument("--device", default="default")
    pg.add_argument("--refiner-device", default="cpu")
    pg.add_argument("--out", default=None, help="Optional JSON output path for grade report")

    pw = sub.add_parser(
        "words",
        aliases=["from-words"],
        help="Method 3 (from-words / words): Synthesize Ready-to-Render fl2va & ref2va .safetensors vectors directly from 3-5 words in <2 ms on CPU via J-Lens",
    )
    pw.add_argument(
        "pos_words",
        help="Comma- or space-separated positive concept words (e.g. 'blizzard, snowdrift, freezing, snow' or '8-bit, LED, steampunk')",
    )
    pw.add_argument(
        "--against-words",
        dest="against_words",
        default="",
        help="Optional negative words or preset ('@prompt_subject', '@neutral', '@replace_human', '@replace_live_action', '@replace_digital')",
    )
    pw.add_argument(
        "--prompt",
        "--target-prompt",
        dest="prompt",
        default="",
        help="Optional target H3 scene prompt file or text for Token-Wise Prompt Receptor Scanning and @prompt_subject extraction",
    )
    pw.add_argument(
        "--hold-words",
        dest="hold_words",
        default="",
        help="Optional nuisance words to project out on CPU",
    )
    pw.add_argument(
        "--hold-alpha",
        dest="hold_alpha",
        type=float,
        default=1.0,
        help="Attenuation alpha in [0.0, 2.0] for --hold-words (default: 1.0)",
    )
    pw.add_argument("--name", default=None, help="Optional output vector stem override")
    pw.add_argument("--output-name", dest="output_name", default="", help=argparse.SUPPRESS)
    pw.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    pw.add_argument("--out-dir", dest="out_dir", default=None, help="Optional output directory for .safetensors files")

    pgen = sub.add_parser(
        "generate-pairs",
        help="Generate 10 surgical H3 Prompt A / Prompt B pairs using a local generative LLM ('transformers' or 'exllamav3') and h3_control_vector_system_prompt.md",
    )
    pgen.add_argument(
        "concept",
        nargs="?",
        default="",
        help="Target control-vector concept or 'pole_a -> pole_b' description (or pass --concept)",
    )
    pgen.add_argument(
        "--concept",
        dest="concept_flag",
        default="",
        help="Target control-vector concept or 'pole_a -> pole_b' description",
    )
    pgen.add_argument(
        "--llm-provider",
        "--provider",
        dest="llm_provider",
        choices=SUPPORTED_LLM_PROVIDERS,
        default=os.environ.get("H3_LLM_PROVIDER", "transformers") or "transformers",
        help="Generative LLM provider ('transformers' or 'exllamav3'; default: 'transformers' or $H3_LLM_PROVIDER)",
    )
    pgen.add_argument(
        "--llm-model",
        "--model",
        dest="llm_model",
        default=os.environ.get("H3_LLM_MODEL", "") or os.environ.get("H3_EXL3_MODEL_DIR", ""),
        help="Path to local LLM model directory or HuggingFace model ID (default: $H3_LLM_MODEL or $H3_EXL3_MODEL_DIR)",
    )
    pgen.add_argument(
        "--exl3-root",
        dest="exl3_root",
        default=os.environ.get("EXLLAMAV3_ROOT", ""),
        help="Optional path to exllamav3 checkout when using --llm-provider exllamav3",
    )
    pgen.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        help="Enable 4-bit BitsAndBytes quantization when loading a transformers model (default: enabled unless --no-4bit is passed)",
    )
    pgen.add_argument(
        "--no-4bit",
        dest="no_4bit",
        action="store_true",
        help="Disable 4-bit BitsAndBytes quantization when loading a transformers model",
    )
    pgen.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=DEFAULT_LLM_MAX_TOKENS)
    pgen.add_argument("--temperature", type=float, default=0.7)
    pgen.add_argument("--name", default="", help="Optional stem name to also write <name>.md / <name>.toml")
    pgen.add_argument("--out-md", dest="out_md", default="", help="Optional output .md file path")
    pgen.add_argument("--out-toml", dest="out_toml", default="", help="Optional output .toml file path")
    pgen.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also parse and save to data/control_vectors/<name>.toml")
    pgen.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

    pjl = sub.add_parser(
        "build-j-lens",
        help="Verify (--verify-only) or rebuild models/j_lens/j_lens_h3.safetensors (h3_j_lens_v1) from Qwen3-VL layers.49 + TokenRefiner",
    )
    pjl.add_argument(
        "--verify-only",
        dest="verify_only",
        action="store_true",
        help="Verify the existing j_lens_h3.safetensors bundle on CPU without loading GPU models",
    )
    pjl.add_argument("--out", default=None, help="Optional output path for j_lens_h3.safetensors")
    pjl.add_argument("--base-bundle", dest="base_bundle", default=None, help="Existing h3_j_lens_v1 bundle providing upper-Qwen (layers.50..63) tensors")
    pjl.add_argument("--num-prompts", dest="num_prompts", type=int, default=24, help="Number of prompt_bank calibration prompts (default: 24)")
    pjl.add_argument("--max-tokens", dest="max_tokens", type=int, default=64, help="Max tokens per calibration prompt (default: 64)")
    pjl.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    pjl.add_argument("--fl2va-dit", dest="fl2va_dit", default=None, help="fl2va DiT checkpoint filename")
    pjl.add_argument("--ref2va-dit", dest="ref2va_dit", default=None, help="ref2va DiT checkpoint filename")
    pjl.add_argument("--no-lora", dest="no_lora", action="store_true", help="Compute TokenRefiner Jacobians without Pika LoRAs")
    pjl.add_argument("--overwrite", action="store_true", help="Overwrite an existing j_lens_h3.safetensors bundle at --out / default bundle path")
    pjl.add_argument("--device", default="default")
    pjl.add_argument("--refiner-device", dest="refiner_device", default="cpu")

    preb = sub.add_parser(
        "rebuild",
        help="Rebuild all control vectors in a folder (defaults to models/control_vectors/ + prompt_bank/control_vectors/) or a single control vector without hardcoded names",
    )
    preb.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional folder path, .toml path, .safetensors path, or vector stem to rebuild (default: rebuild all vectors in the default control_vectors folder)",
    )
    preb.add_argument(
        "--dir",
        dest="vec_dir",
        default=None,
        help="Control vector .safetensors directory to scan/write (defaults to ComfyUI models/control_vectors/)",
    )
    preb.add_argument(
        "--toml-dir",
        dest="toml_dir",
        default=None,
        help="Pair .toml bank directory to scan (defaults to prompt_bank/control_vectors/)",
    )
    preb.add_argument(
        "--vector",
        dest="vector_target",
        default=None,
        help="Explicit single control vector stem, .safetensors file, or .toml file to rebuild",
    )
    preb.add_argument(
        "--families",
        type=_validate_families_arg,
        metavar=_FAMILIES_METAVAR,
        default="both",
        help=_FAMILIES_HELP,
    )
    preb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode when rebuilding from .toml or compiled pairs (default: 'diff_span')",
    )
    preb.add_argument("--unit-mean", dest="unit_mean", action="store_true", help="Average unit-normalized pair directions when rebuilding from pairs")
    preb.add_argument("--dit", dest="dit", default=None, help="Explicit DiT diffusion checkpoint filename override")
    preb.add_argument("--unet", dest="dit", default=None, help=argparse.SUPPRESS)
    preb.add_argument("--encoder", default=None, help="Text encoder checkpoint filename")
    preb.add_argument("--lora", action="append", default=None, help="Explicit LoRA file[:strength] override")
    preb.add_argument("--no-lora", dest="no_lora", action="store_true", help="Rebuild without Pika LoRAs")
    preb.add_argument("--no-holds", action="store_true", help="Do not project out [[hold]] groups")
    preb.add_argument("--include-encoder", dest="include_encoder", action="store_true", default=True, help="Also save vector.encoder (5120-d; default: True)")
    preb.add_argument("--no-encoder", dest="include_encoder", action="store_false", help="Omit vector.encoder (5120-d)")
    preb.add_argument("--force", action="store_true", help="Force GPU re-encoding in folder mode even for vectors already up to date with their .toml SHA-256, vector.encoder, and j_lens_readout")
    preb.add_argument("--seed", type=int, default=0)
    preb.add_argument("--device", default="default")
    preb.add_argument("--refiner-device", default="cpu")

    return p


def dispatch(args: argparse.Namespace) -> int:
    """Run the handler for a parsed namespace. The one routing of subcommand to handler: `main`
    calls it for the command line and the studio's run route (`ui_server.py`) calls it for the page,
    so a method cannot be wired to one and not the other."""
    if args.cmd == "ui":
        from .ui_server import cmd_ui

        return cmd_ui(args)
    if args.cmd == "generate-pairs":
        return cmd_generate_pairs(args)
    if args.cmd in ("words", "from-words"):
        return cmd_words(args)
    if args.cmd == "sculpt":
        return cmd_sculpt(args)
    if args.cmd in ("build", "from-toml", "from-prompt"):
        return cmd_build(args)
    if args.cmd == "build-j-lens":
        return cmd_build_j_lens(args)
    if args.cmd == "rebuild":
        return cmd_rebuild(args)
    if args.cmd in ("from-md", "import-md"):
        return cmd_from_md(args)
    if args.cmd == "make":
        return cmd_make(args)
    if args.cmd in ("make-media", "from-images"):
        return cmd_make_media(args)
    if args.cmd == "inspect":
        return cmd_inspect(args)
    if args.cmd == "grade":
        return cmd_grade(args)
    raise ValueError(f"no handler for subcommand {args.cmd!r}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(dispatch(args) or 0)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
