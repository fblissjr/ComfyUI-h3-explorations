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

The denoised prediction is not a compromise for epsilon. At fixed `x` and
sigma the two are related by an affine map whose constants depend only on
sigma, and every arm is compared at the SAME sigma, so a relative L2 here
differs from the epsilon-space quantity by one constant factor per step.
Rankings and ratios within a step are untouched. It is the same information.

## Why no attention patching is a property, not an omission

Nothing here wires sage, Sol or the SLA router, so both arms run whatever
ComfyUI resolves by default. That is not merely "the same handicap on both
arms". Sol is a sparse approximation whose top-k selection over the routed
region depends on sequence CONTENT; `vendor/sol_attn_minimax.py::_sink_blocks`
holds the conditioning rows exact as KV by default (dense conditioning
*queries* are the opt-in `exact_kv_and_rows` mode), but the routed region's
selection is still free to differ between arms. Running Sol would therefore let
the approximation respond to the treatment and fold the kernel's reaction into
the measurement. Running dense removes that failure mode rather than balancing
it. It is also most of the cost: a forward here is ~120 s where a shipped
sage+Sol render is ~37 s/step, and `docs/h3_pdd.md` records dense SDPA at 2.4x
on this workload for the same reason.

## A hazard this measurement walked into, recorded because it generalises

A 640x384x5 smoke of this same script reported the `mean_init_rows` effect as
concentrated in audio at about 4.5x, and that REVERSED at real geometry, where
it is near-symmetric. The audio concentration belongs to the arms that change
the token stream, not to the arm that changes row contents. Anything in this
repo prototyped at a small canvas is exposed to the same reversal: a cheap
geometry is a test of the plumbing and is not evidence about the effect.

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
    ceiling    release vs an unrelated scene              -> a large change, sampled
    purity     stripped vs legacy_bpe-on-stripped         -> MUST be exactly 0.0

**The purity row is what makes `legacy_bpe` readable at all.** That arm
retokenizes, so its prefix length moves and every downstream position with it;
its delta is therefore confounded in a way `mean_init_rows` is not, and the two
must never be compared as if they were like quantities. Running the legacy
tokenizer against a prompt with the markers already removed isolates the
question: the legacy arm empties only the seven H3 token declarations, so off
marker the two tokenizers should agree exactly. A non-zero here means the arm
differs somewhere else as well, and `legacy_bpe`'s delta stops being
attributable to the marker spelling.

**The null and the ceiling are load-bearing as a PAIR, and neither establishes
the harness works alone.** Exactly 0.0 on the null is the EXPECTED result, not a
surprising one -- the noise is drawn once outside the loop, the encode is
deterministic, and two identical CUDA forwards on identical inputs are bitwise
identical; a small-but-nonzero null is what would have needed explaining. But a
check whose input already satisfies the expected outcome cannot fail, so the
null alone cannot show this would DETECT a difference. The ceiling arm is what
shows that. Read them together.

**The ceiling is one sample of "unrelated", not a bound.** A more distant scene
raises it and shrinks any fraction taken against it without anything about the
treatment changing. So the ceiling row is reported beside the treatment rows and
a ratio between them is not computed here: that number would be a property of
the prompt this file happens to carry.

## What it does not establish

Direction. A large delta says the DiT is sensitive to the marker rows, not
which representation is the one it was trained against. That still needs
renders judged blind under `docs/eval_comparison.md` section 3, and this script
is the gate on whether to spend them.
"""

from __future__ import annotations

import argparse
import hashlib
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
        row["token_ids_sha256"] = _token_digest(tokens)
        # Read back what bound, never the arm name that was asked for -- the
        # rule marker_arms.py is built around.
        row["bound"] = M.encoder_arm_record(armed)
        print(f"[phase 1] {row['label']}: prefix {row['prefix_tokens']} tokens, "
              f"{time.time() - t0:.1f}s", flush=True)
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def _token_digest(tokens) -> str:
    """sha256 of the token id sequence a row was encoded from.

    A COUNT is necessary and not sufficient: two tokenizers can emit the same
    number of tokens with different ids. Recording the sequence is what lets a
    purity comparison be settled in phase 1, before an encoder load, rather
    than by two forwards that could only ever have agreed with it.
    """
    seq = []
    for _key, batches in (tokens.items() if isinstance(tokens, dict)
                          else [("x", tokens)]):
        for batch in batches:
            for item in batch:
                seq.append(item[0] if isinstance(item, (list, tuple)) else item)
    return hashlib.sha256(repr(seq).encode()).hexdigest()


def settle_purity(rows: list[dict], comparisons: list[tuple]) -> list[dict]:
    """Resolve every `purity` comparison from phase 1's token ids, and REFUSE
    if one fails.

    A purity row asserts two arms are identical on this input. If their id
    sequences match, the forwards are a formality -- identical ids through a
    deterministic encode give identical conditioning, and the null row already
    established the forward is deterministic. If they do NOT match, the arm
    differs where it claimed not to, every delta attributed to it is
    unattributable, and spending the card on the rest would be spending it on
    numbers already known to be uninterpretable. So this refuses rather than
    reports.
    """
    by_label = {r["label"]: r for r in rows}
    settled = []
    for kind, left, right, why in comparisons:
        if kind != "purity":
            continue
        a, b = by_label[left], by_label[right]
        same = a["token_ids_sha256"] == b["token_ids_sha256"]
        print(f"[phase 1] purity {left} vs {right}: ids "
              f"{'IDENTICAL' if same else 'DIFFER'} "
              f"({a['token_ids_sha256'][:16]} / {b['token_ids_sha256'][:16]})",
              flush=True)
        if not same:
            raise SystemExit(
                f"PURITY FAILED: {left} and {right} were asserted identical on "
                f"this input and their token id sequences differ "
                f"({a['prefix_tokens']} vs {b['prefix_tokens']} tokens). The "
                f"arm differs where it claimed not to, so nothing measured "
                f"against it is attributable. Refusing before the DiT loads."
            )
        settled.append({
            "kind": kind, "left": left, "right": right, "why": why,
            "settled_in": "phase 1, from token ids -- no forward needed",
            "token_ids_sha256": a["token_ids_sha256"],
            "prefix_tokens": a["prefix_tokens"],
        })
    return settled


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
    ap.add_argument("--rows", default=None,
                    help="comma-separated subset of arm labels to run; "
                         "comparisons needing an absent row are skipped")
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
        {"label": "legacy_stripped", "arm": "legacy_bpe",    "text": "stripped"},
        {"label": "other_scene",    "arm": "release",        "text": "other"},
    ]
    if args.rows:
        want = [r.strip() for r in args.rows.split(",") if r.strip()]
        unknown = [w for w in want if w not in {r["label"] for r in rows}]
        if unknown:
            raise SystemExit(f"unknown row(s): {unknown}")
        rows = [r for r in rows if r["label"] in want]


    comparisons = [
        ("null",      "release", "release_again",
         "same arm, re-encoded and re-run. MUST be exactly 0.0"),
        ("treatment", "release", "mean_init_rows",
         "the controlled pair: identical tokens, seven embedding rows differ"),
        ("treatment", "release", "legacy_bpe",
         "other representation; retokenizes, so the prefix length also moves"),
        ("scale",     "release", "stripped",
         "marker strings removed from the text, weights untouched"),
        ("purity",    "stripped", "legacy_stripped",
         "the two tokenizers on a prompt carrying NO markers. The legacy arm "
         "is a fresh tokenizer with only the seven H3 tokens emptied, so this "
         "MUST be 0.0: any other value means the arm differs off-marker too "
         "and legacy_bpe's delta is not attributable to the markers"),
        ("ceiling",   "release", "other_scene",
         "one unrelated scene. NOT a maximum -- see ceiling_is_not_a_bound"),
    ]
    have = {r["label"] for r in rows}
    comparisons = [c for c in comparisons if c[1] in have and c[2] in have]
    if not comparisons:
        raise SystemExit("the selected rows support no comparison")

    encode_arms(prompt, stripped, args.encoder, rows)

    # Purity is settled from token ids here, before the DiT loads. If it
    # fails the run stops: nothing measured against a contaminated arm is
    # attributable, so the card must not be spent on it.
    settled = settle_purity(rows, comparisons)
    comparisons = [c for c in comparisons if c[0] != "purity"]
    needed = {c[1] for c in comparisons} | {c[2] for c in comparisons}
    rows = [r for r in rows if r["label"] in needed]
    skip_phase2 = not rows
    if skip_phase2:
        print("\nevery selected comparison settled in phase 1; "
              "no forward is needed, so the DiT is not loaded.", flush=True)

    unet = _graph_unet()
    _, frame_count = _empty_av_latent(args.width, args.height, args.length)
    sigmas = None
    if not skip_phase2:
        print(f"[phase 2] loading DiT {unet}", flush=True)
        model = sigma_shifted(
            comfy.sd.load_diffusion_model(_resolve("diffusion_models", unet)))
        latent, _ = _empty_av_latent(args.width, args.height, args.length)
        latent = latent["samples"]
        noise = comfy.sample.prepare_noise(latent, args.seed)
        sigmas = comfy.samplers.calculate_sigmas(
            model.get_model_object("model_sampling"), SAMPLING["scheduler"],
            args.steps)

    preds: dict = {}
    for step in probe_steps:
        for row in rows:
            t0 = time.time()
            preds[(row["label"], step)] = one_forward(
                model, row["cond"], noise, latent, sigmas, step, args.seed)
            print(f"[phase 2] step {step} {row['label']}: {time.time() - t0:.1f}s",
                  flush=True)


    record = {
        "measurement": "DiT denoised-prediction delta across marker arms, at "
                       "fixed noise, latents and sigma",
        "question": "is the frozen DiT sensitive to the seven H3 marker rows",
        "does_not_establish": [
            "direction -- which representation the DiT was trained against. "
            "That needs blind renders judged blind.",
            "magnitude at the output -- this bounds sensitivity at one point "
            "on the trajectory with x held fixed. Real sampling compounds, so "
            "a per-step delta can wash out or amplify by the final frame.",
        ],
        "ceiling_is_not_a_bound": (
            "the ceiling row is ONE unrelated prompt, not a maximum. A more "
            "distant scene raises it. Do not quote a treatment as a fraction "
            "of it."
        ),
        "null_and_ceiling_are_a_pair": (
            "exactly 0.0 on the null is expected, not surprising, and cannot "
            "by itself show this would detect a difference. The ceiling row is "
            "what shows that. Neither establishes the harness alone."
        ),
        "graph": GRAPH.name,
        "encoder": args.encoder,
        "unet": unet,
        "canvas": {"width": args.width, "height": args.height,
                   "length": args.length, "frame_count": frame_count},
        "schedule": {"scheduler": SAMPLING["scheduler"], "steps": args.steps,
                     "probe_steps": probe_steps,
                     "sigmas_at_probe": ([float(sigmas[s]) for s in probe_steps]
                                         if sigmas is not None else None),
                     "sigma_shift": dict(SIGMA_SHIFT)},
        "seed": args.seed,
        "arms": [{k: v for k, v in r.items() if k != "cond"} for r in rows],
        "results": list(settled),
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
        pair = f"{r['left']} vs {r['right']}"
        if "delta" not in r:
            # Settled in phase 1 from token ids; it has no step and no forward.
            print(f"{r['kind']:<10} {pair:<32} {'--':>4} "
                  f"{'ids identical':>12} {'(no forward)':>14}")
            continue
        print(f"{r['kind']:<10} {pair:<32} {r['step']:>4} "
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
