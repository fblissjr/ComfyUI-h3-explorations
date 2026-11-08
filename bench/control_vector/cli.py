"""Argument parser and CLI command router for `bench/control_vectors.py`."""

from __future__ import annotations

import argparse
import os
import sys

import control_vectors as cv
from workflows import h3_config

from .build_cmd import cmd_build, cmd_rebuild, cmd_words
from .grading import cmd_grade, cmd_inspect
from .llm_pairs import SUPPORTED_LLM_PROVIDERS, cmd_generate_pairs
from .make_cmds import cmd_from_md, cmd_make, cmd_make_media
from .sculpt import cmd_sculpt

CLI_DESCRIPTION = """Build, inspect, meter, and grade MiniMax H3 control vectors."""

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    sub = p.add_subparsers(dest="cmd", required=True)

    pui = sub.add_parser(
        "ui",
        help="Launch the self-contained local browser UI for authoring, previewing, and building control vectors",
    )
    pui.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    pui.add_argument("--port", type=int, default=0, help="Port (default: 0 picks an unused ephemeral port)")
    pui.add_argument("--no-browser", dest="no_browser", action="store_true", help="Do not auto-open a browser tab")
    pui.add_argument("--toml-dir", dest="toml_dir", default=None, help="Optional directory override for .toml pair packs")
    pui.add_argument("--vec-dir", dest="vec_dir", default=None, help="Optional directory override for .safetensors control vectors")
    pui.add_argument("--encoder", default=None, help="Optional conditioning encoder filename override")
    pui.add_argument("--dit", dest="dit", default=None, help="Optional DiT checkpoint filename override")
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

    pb = sub.add_parser(
        "build",
        help="One-shot end-to-end build: simple words (--words), Markdown (.md), or TOML (.toml) -> fl2va/ref2va .safetensors vectors",
    )
    pb.add_argument(
        "source",
        nargs="?",
        default=None,
        help="Input LLM Markdown (.md) file, pair TOML (.toml) file, or stem under prompt_bank/control_vectors/ (omit when using --words, --placebo, or --all)",
    )
    pb.add_argument(
        "--words",
        default="",
        help="Simple words, short phrase, or 'pole_a -> pole_b' contrast (supports ';' or newline multi-clause, 'soundscape:', and 'hold:' lines) to compile into 10 canonical H3 scene pairs in memory",
    )
    pb.add_argument(
        "--slot",
        default="auto",
        help="Structural H3 slot for --words ('auto', 'opening_framing', 'subject_behavior', 'interpersonal_dynamics', 'camera_motion', 'speech_delivery', 'soundscape')",
    )
    pb.add_argument(
        "--target-prompt",
        dest="target_prompt",
        default="",
        help="Optional target H3 scene prompt (bank id, file path, or text) to compile as Pair 0 at --target-weight alongside 9 canonical frames",
    )
    pb.add_argument(
        "--target-weight",
        dest="target_weight",
        type=float,
        default=2.5,
        help="Weight for Pair 0 when --target-prompt is passed (default: 2.5)",
    )
    pb.add_argument(
        "--placebo",
        action="store_true",
        help="Build a size-matched benign placebo control vector in the same canonical frames",
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
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to build vectors for in one pass (default: 'both' builds fl2va and ref2va)",
    )
    pb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode or comma-separated modes (e.g. 'diff_span', 'from_first_diff', 'scene_text', or 'all'; default: 'diff_span')",
    )
    pb.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pb.add_argument("--output-name", dest="output_name", default="", help="Optional vector stem override")
    pb.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override when importing .md or compiling --words")
    pb.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pb.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pb.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pb.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pb.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also write the compiled 10-pair .toml file under prompt_bank/control_vectors/ when using --words")
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
        help="Output .toml file path or output directory (defaults to prompt_bank/control_vectors/<stem>.toml)",
    )
    pfm.add_argument("--name", default=None, help="Optional vector name / file stem override")
    pfm.add_argument("--mode", choices=cv.VALID_MODES, default=None, help="Explicit mode override (auto-detected if omitted)")
    pfm.add_argument("--frame", choices=cv.VALID_FRAMES, default="scene", help="Pair frame ('scene' or 'bare')")
    pfm.add_argument("--quality", default=None, help="Optional vector.quality description override")
    pfm.add_argument("--a-side", dest="a_side", default=None, help="Optional vector.a_side description override")
    pfm.add_argument("--b-side", dest="b_side", default=None, help="Optional vector.b_side description override")
    pfm.add_argument("--test-prompt-out", dest="test_prompt_out", default=None, help="Optional path to write the extracted Test Scene Prompt text")
    pfm.add_argument("--overwrite", action="store_true", help="Overwrite an existing .toml file at the target path")

    pm = sub.add_parser("make", help="Build a control vector from a pair TOML file or --words")
    pm.add_argument("pairs", nargs="?", default=None, help="Pair TOML file path or stem under prompt_bank/control_vectors/ (omit when using --words)")
    pm.add_argument("--words", default="", help="Simple words or 'pole_a -> pole_b' contrast to compile into 10 canonical H3 pairs in memory")
    pm.add_argument("--slot", default="auto", help="Structural H3 slot for --words ('auto' or one of the 6 canonical slots)")
    pm.add_argument("--target-prompt", dest="target_prompt", default="", help="Optional target H3 scene prompt to compile as Pair 0")
    pm.add_argument("--target-weight", dest="target_weight", type=float, default=2.5, help="Weight for Pair 0 when --target-prompt is passed")
    pm.add_argument("--hold", action="append", default=None, help="Optional .safetensors vector(s) to project out")
    pm.add_argument("--hold-words", dest="hold_words", default="", help="Optional nuisance words to project out on CPU")
    pm.add_argument("--hold-alpha", dest="hold_alpha", type=float, default=1.0, help="Attenuation alpha in [0.0, 2.0] for --hold / --hold-words")
    pm.add_argument("--pooling", choices=cv.VALID_POOLING, default="diff_span")
    pm.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to build when --out and --dit are omitted (default: 'both' writes fl2va and ref2va)",
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
    pm.add_argument("--output-name", default="", help="Optional vector stem override")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--device", default="default")
    pm.add_argument("--refiner-device", default="cpu")
    pm.add_argument("--out", default=None, help="Output .safetensors path or filename")
    pm.add_argument("--overwrite", action="store_true", help="Overwrite an existing vector file at the target path")

    pmm = sub.add_parser(
        "make-media",
        help="Build a control vector from contrasting <Picture 1> image pair(s) with 32-px M-RoPE grid matching",
    )
    pmm.add_argument("--image-a", required=True, help="Baseline/negative pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--image-b", required=True, help="Positive/steered pole image path or data URL (or comma-separated list)")
    pmm.add_argument("--output-name", default="media_contrast", help="Output vector stem (default: 'media_contrast')")
    pmm.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="ref2va",
        help="Which H3 model families to build media vectors for (default: 'ref2va')",
    )
    pmm.add_argument("--pooling", choices=cv.VALID_MEDIA_POOLING, default="scene_text")
    pmm.add_argument("--short-edge", type=int, default=512, help="Target shorter edge in pixels (snapped to 32)")
    pmm.add_argument("--scene-prompt", default="", help="Optional neutral H3 ref2va scene prompt")
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
    psculpt.add_argument("--output-name", dest="output_name", default="", help="Output vector stem override")
    psculpt.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to write (default: 'both' writes fl2va and ref2va)",
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
    pi.add_argument("--device", default="default")
    pi.add_argument("--refiner-device", default="cpu")

    pg = sub.add_parser("grade", help="Grade activation/prediction movement against nulls and edit baselines")
    pg.add_argument("--vector", required=True, help="Vector .safetensors path or name")
    pg.add_argument("--prompt", required=True, help="Base prompt id or path")
    pg.add_argument("--edit-prompt", default=None, help="Edit baseline prompt id or path")
    pg.add_argument("--pairs", default=None, help="Pair TOML file to build flip_null")
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
        help="Synthesize Ready-to-Render fl2va & ref2va .safetensors vectors directly from 3-5 words in <2 ms on CPU via J-Lens (with Pre-Render Readiness Predictor)",
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
    pw.add_argument("--output-name", dest="output_name", default="", help="Optional output vector stem override")
    pw.add_argument(
        "--families",
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to synthesize (default: 'both' writes fl2va and ref2va)",
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
    pgen.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=4800)
    pgen.add_argument("--temperature", type=float, default=0.7)
    pgen.add_argument("--name", default="", help="Optional stem name to also write <name>.md / <name>.toml")
    pgen.add_argument("--out-md", dest="out_md", default="", help="Optional output .md file path")
    pgen.add_argument("--out-toml", dest="out_toml", default="", help="Optional output .toml file path")
    pgen.add_argument("--save-toml", dest="save_toml", action="store_true", help="Also parse and save to prompt_bank/control_vectors/<name>.toml")
    pgen.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")

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
        choices=("both", "fl2va", "ref2va", "auto", "fl2va,ref2va", "ref2va,fl2va"),
        default="both",
        help="Which H3 model families to rebuild (default: 'both' for bank TOMLs or existing families on disk)",
    )
    preb.add_argument(
        "--pooling",
        default="diff_span",
        help="Pooling mode when rebuilding from .toml or compiled pairs (default: 'diff_span')",
    )
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



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "ui":
            from .ui_server import cmd_ui

            return cmd_ui(args)
        if args.cmd == "generate-pairs":
            return cmd_generate_pairs(args)
        if args.cmd == "words":
            return cmd_words(args)
        if args.cmd == "sculpt":
            return cmd_sculpt(args)
        if args.cmd == "build":
            return cmd_build(args)
        if args.cmd == "rebuild":
            return cmd_rebuild(args)
        if args.cmd in ("from-md", "import-md"):
            return cmd_from_md(args)
        if args.cmd == "make":
            return cmd_make(args)
        if args.cmd == "make-media":
            return cmd_make_media(args)
        if args.cmd == "inspect":
            return cmd_inspect(args)
        if args.cmd == "grade":
            return cmd_grade(args)
    except (ValueError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
