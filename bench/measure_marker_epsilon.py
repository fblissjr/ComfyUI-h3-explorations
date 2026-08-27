#!/usr/bin/env python3
"""Does the DiT's prediction change when the seven H3 marker rows change?

## The question this owns

`docs/research/.../2026-08-26_encoder_choice_and_marker_measurement.md` section 4
establishes that the released encoder is byte-identical to stock Qwen3-VL, so
the seven H3 marker rows are untrained stock tail rows, and names the marker
arms as the way to learn which representation the frozen DiT was trained to
read. Its three instruments are encoder-level (deterministic, already run),
DiT-level (renders, a distribution, the owner's judgement) and structure-level
(prefix attention).

None of them answer the cheapest question, which gates the expensive one:
**is the DiT sensitive to the marker rows at all?** If swapping them does not
move the model's prediction, the lane closes without spending a blind session.

## Why the denoised prediction and not a rendered clip

`CLAUDE.md`: a rendered clip cannot A/B a numerical change, because the
trajectory diverges completely and the changed arm's output is a *different
sample* rather than a degraded one. The rule's prescription is to compare knobs
at the call. `bench/grade_sage_on_capture.py` does that for attention kernels;
this is the conditioning equivalent. Identical latents, identical noise,
identical sigma, identical everything but the seven rows, one forward each.

## Why `release` against `mean_init_rows` is the controlled pair

Both tokenize identically -- same ids, same count, same positions, same packed
layout. The ONLY difference is the contents of seven embedding rows. No other
arm pair has that property: `legacy_bpe` retokenizes and `stripped` removes
characters, so both move the prefix length and shift where the video block
starts. Those are still measured here, and read as a scale rather than as the
controlled comparison.

## The ladder, and why an unlabelled number would be uninterpretable

    null       release vs release, re-encoded and re-run  -> MUST be exactly 0.0
    treatment  release vs mean_init_rows                  -> the question
    treatment  release vs legacy_bpe                      -> other representation
    scale      release vs markers stripped from the text  -> prompt-level change
    ceiling    release vs an unrelated scene              -> what "a lot" is

Without the null the harness could be measuring its own nondeterminism, and
without the ceiling a relative L2 has no units a reader can act on.

## What it does not establish

Direction. A large delta says the DiT is sensitive to the marker rows, not
which representation is the one it was trained against. That still needs
renders judged blind under `docs/eval_comparison.md` section 3, and this script
is the gate on whether to spend them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMFY = REPO.parents[1]
# ComfyUI's root goes on LAST so it ends up FIRST. This repo has a `nodes.py`,
# and `comfy_extras/nodes_minimax_h3.py` does a bare `import nodes`: with the
# repo ahead of ComfyUI that resolves to ours and dies on a relative import.
# The trap and its two fixes are in `docs/comfy_notes.md`.
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(COMFY))

import comfy.cli_args  # noqa: E402  (must precede other comfy imports)
import torch  # noqa: E402

import comfy.model_management  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import folder_paths  # noqa: E402
from comfy_extras.nodes_minimax_h3 import _empty_av_latent  # noqa: E402

import marker_arms as M  # noqa: E402
from h3_config import ENCODER_INT8, SAMPLING, SIGMA_SHIFT  # noqa: E402

# The graph this measurement is about. Its prompt is the only one in the repo
# that carries `<d>` dialogue markers on the t2v path, which is what makes the
# marker arms observable at all -- a prompt with no markers would give every
# arm the same token stream and the null control by accident.
GRAPH = REPO / "workflows" / "h3_text_to_video_dialogue_api.json"

# An unrelated scene, for the ceiling row. Deliberately not a variation of the
# dialogue prompt: the ceiling's job is to say what a large prediction change
# looks like, so it must not be a near-neighbour of the treatment.
OTHER_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a "
    "locked-off wide holds on an empty rain-slick loading dock at night, "
    "sodium vapour lights buzzing overhead, puddles rippling under a steady "
    "downpour. No people are present.\n"
    "overall_soundscape: heavy rain on corrugated steel, a distant generator "
    "hum, no speech."
)


def _graph_prompt() -> str:
    graph = json.loads(GRAPH.read_text())
    for node in graph.values():
        if node.get("class_type") == "MiniMaxH3Conditioning":
            return node["inputs"]["prompt"]
    raise SystemExit(f"no MiniMaxH3Conditioning node in {GRAPH.name}")


def _graph_unet() -> str:
    graph = json.loads(GRAPH.read_text())
    for node in graph.values():
        if node.get("class_type") == "UNETLoader":
            return node["inputs"]["unet_name"]
    raise SystemExit(f"no UNETLoader node in {GRAPH.name}")


def _strip_markers(prompt: str) -> str:
    """Remove the seven marker strings, keeping every other character.

    This is the corpus's `stripped` arm applied to one prompt. It is a
    prompt-text transform and touches no weights, which is why it sits on the
    scale rows rather than beside the controlled pair.
    """
    out = prompt
    for token in M.marker_tokens():
        out = out.replace(token, "")
    if out == prompt:
        raise SystemExit(
            "stripping the markers changed nothing, so this prompt carries "
            "none and the whole measurement would read as its own null control"
        )
    return out


def _resolve(kind: str, name: str) -> str:
    path = folder_paths.get_full_path(kind, name)
    if path is None:
        raise SystemExit(f"{kind}/{name} is not installed")
    return path


def encode_arms(prompt: str, stripped: str, encoder: str, rows: list[dict]) -> None:
    """Phase 1: every conditioning, then the encoder is freed.

    Encoding and sampling are not interleaved because the INT8 encoder's H3
    path is 25.28 GiB (`bench/results/2026-08-26_encoder_footprints.json`) and
    the DiT is resident for the forwards. Interleaving would evict one of them
    on every row and time the eviction rather than the model.
    """
    path = _resolve("text_encoders", encoder)
    print(f"[phase 1] loading encoder {encoder}", flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    for row in rows:
        armed = M.apply_arm(clip, row["arm"])
        text = {"graph": prompt, "stripped": stripped, "other": OTHER_PROMPT}[row["text"]]
        t0 = time.time()
        tokens = armed.tokenize(text, images=[])
        cond = armed.encode_from_tokens_scheduled(tokens)
        row["cond"] = cond
        row["prefix_tokens"] = int(cond[0][0].shape[1])
        # Read back what bound, never the arm name that was asked for -- the
        # rule marker_arms.py is built around.
        row["bound"] = M.encoder_arm_record(armed)
        print(f"[phase 1] {row['label']}: prefix {row['prefix_tokens']} tokens, "
              f"{time.time() - t0:.1f}s", flush=True)
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def sigma_shifted(model):
    """MiniMaxH3SigmaShift, replicated rather than imported.

    The node returns an `io.NodeOutput`, and unwrapping that is a second thing
    to get wrong; the body is five lines. `h3_config.SIGMA_SHIFT` is the single
    source for the values, so this cannot drift from the graphs.
    """
    m = model.clone()

    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingAV,
                                comfy.model_sampling.CONST):
        pass

    original = m.get_model_object("model_sampling")
    sampling = ModelSamplingAdvanced(model.model.model_config)
    sampling.set_parameters(shift=SIGMA_SHIFT["shift_video"],
                            audio_shift=SIGMA_SHIFT["shift_audio"])
    if hasattr(original, "noise_scale"):
        sampling.set_noise_scale(original.noise_scale)
    m.add_object_patch("model_sampling", sampling)
    to = m.model_options["transformer_options"] = m.model_options.get(
        "transformer_options", {}).copy()
    to["minimax_h3_sigma_shift_video"] = SIGMA_SHIFT["shift_video"]
    to["minimax_h3_sigma_shift_audio"] = SIGMA_SHIFT["shift_audio"]
    return m


def one_forward(model, cond, noise, latent, sigmas, step: int, seed: int):
    """The model's denoised prediction at one schedule step, and nothing else.

    Driven through `Guider_Basic` and a real `KSAMPLER` rather than by calling
    the DiT directly, so conditioning conversion, the packed layout, model
    loading and every wrapper are the ones a render uses. The sampler function
    evaluates the model once and returns; no step is taken.
    """
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    captured = {}

    def _single(model_fn, x, sigs, extra_args, callback=None, disable=None):
        sigma = sigs[0] * torch.ones((x.shape[0],), device=x.device, dtype=x.dtype)
        out = model_fn(x, sigma, **extra_args)
        captured["out"] = out
        return out

    guider = Guider_Basic(model)
    guider.set_conds(cond)
    result = guider.sample(
        noise, latent, comfy.samplers.KSAMPLER(_single),
        sigmas[step:step + 2], denoise_mask=None, callback=None,
        disable_pbar=True, seed=seed,
    )
    if "out" not in captured:
        raise SystemExit("the sampler never evaluated the model")
    return result


def delta(a, b) -> dict:
    """Relative L2 and cosine per latent block, on CPU float32.

    Reported per block because the video and audio streams have very different
    magnitudes and a single pooled number would be the video block wearing an
    audio label.
    """
    out = {}
    names = ("video", "audio")
    parts_a = a.unbind() if a.is_nested else (a,)
    parts_b = b.unbind() if b.is_nested else (b,)
    for name, pa, pb in zip(names, parts_a, parts_b):
        x = pa.detach().to("cpu", torch.float32).flatten()
        y = pb.detach().to("cpu", torch.float32).flatten()
        if x.shape != y.shape:
            raise SystemExit(f"{name} block shapes differ: {x.shape} vs {y.shape}")
        ref = torch.linalg.vector_norm(x)
        out[name] = {
            "relative_l2": float(torch.linalg.vector_norm(x - y) / ref),
            "cosine": float(torch.nn.functional.cosine_similarity(x, y, dim=0)),
            "max_abs_diff": float((x - y).abs().max()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=362)
    ap.add_argument("--steps", type=int, default=SAMPLING["steps"])
    ap.add_argument("--probe-steps", default="0,8,15",
                    help="schedule indices to evaluate the model at")
    ap.add_argument("--seed", type=int, default=730451892,
                    help="default is the dialogue graph's own noise seed")
    ap.add_argument("--encoder", default=ENCODER_INT8,
                    help="the encoder of record. Its deviation from BF16 "
                         "(0.021-0.027) is BELOW the marker effect at layer 50 "
                         "(0.034-0.16); a W4 artifact's (0.31-0.37) is above "
                         "it, and would bury the thing being measured")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    probe_steps = [int(s) for s in args.probe_steps.split(",") if s.strip()]
    # Everything cheap that can refuse, refuses here -- before an encoder load.
    # A run that dies after phase 1 has spent minutes to learn something a
    # string comparison knew.
    if not probe_steps:
        raise SystemExit("--probe-steps named no steps")
    if max(probe_steps) >= args.steps:
        raise SystemExit(f"probe step {max(probe_steps)} is past a {args.steps}-step schedule")
    prompt = _graph_prompt()
    stripped = _strip_markers(prompt)

    rows = [
        {"label": "release",        "arm": "release",        "text": "graph"},
        {"label": "release_again",  "arm": "release",        "text": "graph"},
        {"label": "mean_init_rows", "arm": "mean_init_rows", "text": "graph"},
        {"label": "legacy_bpe",     "arm": "legacy_bpe",     "text": "graph"},
        {"label": "stripped",       "arm": "release",        "text": "stripped"},
        {"label": "other_scene",    "arm": "release",        "text": "other"},
    ]

    encode_arms(prompt, stripped, args.encoder, rows)

    unet = _graph_unet()
    print(f"[phase 2] loading DiT {unet}", flush=True)
    model = sigma_shifted(comfy.sd.load_diffusion_model(_resolve("diffusion_models", unet)))

    latent, frame_count = _empty_av_latent(args.width, args.height, args.length)
    latent = latent["samples"]
    noise = comfy.sample.prepare_noise(latent, args.seed)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), SAMPLING["scheduler"], args.steps)

    preds: dict = {}
    for step in probe_steps:
        for row in rows:
            t0 = time.time()
            preds[(row["label"], step)] = one_forward(
                model, row["cond"], noise, latent, sigmas, step, args.seed)
            print(f"[phase 2] step {step} {row['label']}: {time.time() - t0:.1f}s",
                  flush=True)

    comparisons = [
        ("null",      "release", "release_again",
         "same arm, re-encoded and re-run. MUST be exactly 0.0"),
        ("treatment", "release", "mean_init_rows",
         "the controlled pair: identical tokens, seven embedding rows differ"),
        ("treatment", "release", "legacy_bpe",
         "other representation; retokenizes, so the prefix length also moves"),
        ("scale",     "release", "stripped",
         "marker strings removed from the text, weights untouched"),
        ("ceiling",   "release", "other_scene",
         "an unrelated scene: what a large prediction change looks like"),
    ]

    record = {
        "measurement": "DiT denoised-prediction delta across marker arms, at "
                       "fixed noise, latents and sigma",
        "question": "is the frozen DiT sensitive to the seven H3 marker rows",
        "does_not_establish": "direction -- which representation the DiT was "
                              "trained against. That needs blind renders.",
        "graph": GRAPH.name,
        "encoder": args.encoder,
        "unet": unet,
        "canvas": {"width": args.width, "height": args.height,
                   "length": args.length, "frame_count": frame_count},
        "schedule": {"scheduler": SAMPLING["scheduler"], "steps": args.steps,
                     "probe_steps": probe_steps,
                     "sigmas_at_probe": [float(sigmas[s]) for s in probe_steps],
                     "sigma_shift": dict(SIGMA_SHIFT)},
        "seed": args.seed,
        "arms": [{k: v for k, v in r.items() if k != "cond"} for r in rows],
        "results": [],
    }

    for kind, left, right, why in comparisons:
        for step in probe_steps:
            record["results"].append({
                "kind": kind, "left": left, "right": right, "why": why,
                "step": step, "sigma": float(sigmas[step]),
                "delta": delta(preds[(left, step)], preds[(right, step)]),
            })

    out = Path(args.out) if args.out else (
        REPO / "bench" / "results" / "2026-08-27_marker_epsilon.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1) + "\n")

    print(f"\n{'kind':<10} {'comparison':<32} {'step':>4} {'video relL2':>12} {'audio relL2':>12}")
    for r in record["results"]:
        print(f"{r['kind']:<10} {r['left'] + ' vs ' + r['right']:<32} {r['step']:>4} "
              f"{r['delta']['video']['relative_l2']:>12.6f} "
              f"{r['delta']['audio']['relative_l2']:>12.6f}")
    try:
        shown = out.relative_to(REPO)
    except ValueError:
        shown = out.name  # a scratch path; never print one into the log
    print(f"\nwrote {shown}")

    null = [r for r in record["results"] if r["kind"] == "null"]
    bad = [r for r in null if r["delta"]["video"]["relative_l2"] != 0.0]
    if bad:
        print("\nNULL CONTROL IS NOT ZERO. Every number above is suspect: the "
              "harness is not deterministic, so a treatment delta cannot be "
              "attributed to the arm.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
