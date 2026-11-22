#!/usr/bin/env python3
"""Wave 2 GPU Benchmark Runner & Analyzer: Exp 4, Exp 6 (A/B), and Exp 7.

Executes clean non-market Wave 2 validation and comparative experiments on
ComfyUI ``:8188`` without modifying any shipped workflow or core file:

- **Exp 4 (Zero-Rewrite Inverted-SAM Sin City Dual-World, 1344x768 @ 141f):**
  Segments the talking diner woman from ``02_ref2va_film16mm_default_00001-audio.mp4``
  via ``SAM3_VideoTrack`` + ``MiniMaxH3SAM31Corrections``, asserts background area
  >= 30%, and compares:
    * ``exp4_base_unsteered``        -- matched ``i2v`` baseline (scale=0.0)
    * ``exp4_2x_noir_video_only``    -- 2x ``EditGuider`` + ``InvertMask`` + ``noir`` (``streams="video_only"``)
    * ``exp4_2x_noir_streams_both``  -- 2x ``EditGuider`` + ``InvertMask`` + ``noir`` (``streams="both"`` bleed control)
    * ``exp4_2x_vhs_video_only``     -- 2x ``EditGuider`` + ``InvertMask`` + ``vhs_1991_camcorder`` (``streams="video_only"``)
    * ``exp4_1x_masked_source_noir`` -- 1x ``MiniMaxH3MaskedSource`` + ``MiniMaxH3AudioFreezeSong`` + ``noir``

- **Exp 6A (In-Graph 3-Beat Storyboard-to-Video, 1152x768 @ 192f):**
  Generates Beats 1 -> 2 -> 3 in-graph via zero-swap ``P-H3-Image`` LoRA (8->4 MP)
  and tests ``MiniMaxH3Conditioning(first_frame=Beat1, last_frame=Beat3)`` +
  ``frame 68`` ``MiniMaxH3GuideLatent(Beat2)`` with ``guide_end_percent`` in
  ``{bookends_only, 1.00, 0.33}``.

- **Exp 6B (First Live GPU Multimodal `<Picture>+<Video>+<Audio>` PromptExpander, 1152x768 @ 192f):**
  Wires ``MiniMaxH3AppendRefImage`` + ``MiniMaxH3AppendRefVideo`` + ``MiniMaxH3AppendRefAudio``
  into ``MiniMaxH3PromptExpander`` (requires 5-output schema with ``ref_end_percent`` at
  slot 4) + ``MiniMaxH3EncodeReferences`` -> ``MiniMaxH3PromptOnReferences`` -> ``MiniMaxH3ReleaseReferences``.

- **Exp 7 (FrozenVideoCache + AudioRefineMask Re-Dub + 39f Continuation, 1152x768):**
  Renders speaking Shot 1 (`t2va_box_office.txt`, 192f), then tests:
    * Part 7A: ``compile_audio_redub`` across ``polish (am=1.0)``, ``polish (am=0.25)``, ``fresh (am=1.0)``.
    * Part 7B: ``compile_continue_shot`` across ``ref_tail_39f (39f)``, ``ref_tail_39f (141f)``,
      and ``latent_overlap_39f (141f, seam_taper)``.

All media files land strictly under ``$H3_OUTPUT_DIR/{Image,Video}/wave2_2026-11-21/``.
Scrubbed JSON records, queued prompts, and metrics land under ``data/wave2_2026-11-21/``.

Usage:
  "$PY" bench/run_wave2_director_storyboard_redub.py --dry-run
  "$PY" bench/run_wave2_director_storyboard_redub.py --stage all
  "$PY" bench/run_wave2_director_storyboard_redub.py --stage analyze
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "bench"))
from _lib import bootstrap  # noqa: E402

COMFY_ROOT = bootstrap()
# Keep COMFY ahead of REPO so ComfyUI's own `nodes.py` stays unshadowed, while allowing `director.*`.
sys.path = [p for p in sys.path if p != str(REPO)]
for _extra in (str(REPO / "workflows"), str(REPO / "bench"), str(REPO)):
    if _extra not in sys.path:
        sys.path.append(_extra)

import h3_config  # noqa: E402
import h3_rules  # noqa: E402
import preflight_graph as pf  # noqa: E402
from _lib import comfy_client  # noqa: E402
from director import compiler as dir_compiler  # noqa: E402

RUN_TAG = "wave2_2026-11-21"
COMFY_URL = os.environ.get("COMFY_URL", comfy_client.DEFAULT_COMFY_URL)
DATA_DIR = REPO / "data" / RUN_TAG
OUT_ROOT = h3_config.output_dir()
OUT_VIDEO_DIR = OUT_ROOT / "Video" / RUN_TAG
OUT_IMAGE_DIR = OUT_ROOT / "Image" / RUN_TAG
UPLOAD_SUBFOLDER = "wave2_tmp"

CLIP_DINER = "02_ref2va_film16mm_default_00001-audio.mp4"
STILL_PORTRAIT = "t2i_29_edit_reference_portrait_seed7.png"
VIDEO_DANCER = "motion-transfer-input-1.mp4"
CLIP_SIREN_AUDIO = "exp8_sunny_room_door_siren_guide_f1_s1.65_s20261112_00001-audio.mp4"

CV_NOIR_FL2VA = "bank/noir__fl2va__diff_span__lora_2a8c2b52.safetensors"
CV_NOIR_REF2VA = "bank/noir__ref2va__diff_span__lora_865ee189.safetensors"
CV_VHS_FL2VA = "bank/vhs_1991_camcorder__fl2va__diff_span__lora_2a8c2b52.safetensors"

# ---------------------------------------------------------------------------
# Strictly Non-Market Scene Prompts (pre-validated with preflight_graph.grade)
# ---------------------------------------------------------------------------

EXP4_DINER_CORE = (
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, warm natural sunlit "
    "retro diner tones across the 5.875-second take. Holding the medium close-up framing of the "
    "diner aisle behind red-and-white tufted vinyl booths and sunlit Venetian-blind windows, a "
    "young blonde woman in a black crewneck top, on-screen, twenties, with a confident, teasing "
    "mezzo-soprano (S1), looks directly at camera and says: <d>[English] What are you looking at?</d> "
    "She smiles lightly and adds: <d>[English] You like what you see?</d> Her lips settle closed as "
    "she tilts her head gently over subtle diner background movement.\n\n"
    "overall_soundscape: Quiet retro diner room tone, faint distant kitchen clatter, and subtle booth ambient air.\n\n"
    "non_diegetic_music: N/A"
)

EXP4_PROMPT_I2VA = f"{pf.BASE_ALIGNMENT['I2VA']}\n\n{EXP4_DINER_CORE}"
EXP4_PROMPT_MASKED_REF2VA = (
    "subject_definitions:\n"
    "<Video 1> is the source diner video whose camera framing, background geometry, and ambient room motion are followed across the masked background region.\n\n"
    "summary:\n"
    "[video editing] The target video follows <Video 1> across the sunlit retro diner interior while converting the surrounding booth seating, tiled aisle, walls, and Venetian-blind windows into high-contrast monochrome.\n\n"
    "retention_analysis:\n"
    "<Video 1> (camera and spatial layout): weak_reference - camera framing, window perspective, and background layout of <Video 1> are followed cleanly.\n\n"
    "detailed_description:\n"
    "The target video retains the 24 fps photographic realism and steady medium close-up camera framing of <Video 1>.\n"
    "[Shot 1] Holding the steady medium-close diner perspective of <Video 1>, the red-and-white tufted vinyl booths, pale tabletop forms, ceiling fan blades, recessed menu boards, sunlit Venetian blinds across the wide diner windows, and background patrons maintain the spatial geometry and subtle motion of <Video 1> across the 5.875-second take while remaining silently synchronized around the frozen foreground region.\n\n"
    "overall_soundscape:\n"
    "Quiet retro diner room tone, faint kitchen clatter in the distance, and soft booth ambient air.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)

EXP6A_BEAT1_PROMPT = (
    "Wide architectural photograph looking straight through tall weathered timber doors inside an "
    "old seaside wooden boathouse pier at golden hour, balanced symmetrically along worn cedar "
    "planks toward a calm amber harbor outside. Heavy coiled hemp ropes rest beside the left post, "
    "and an unlit antique brass ship's lantern hangs from an oak piling on the right beside a moored "
    "white wooden sailboat drifting gently on warm golden harbor water, natural afternoon sunlight "
    "streaking across rough wood grain, 35mm lens at f/4."
)

EXP6A_BEAT2_PROMPT = (
    "Edit <Picture 1>. Shift the golden-hour harbor light into cool cobalt-blue evening twilight as "
    "dense silvery sea fog rolls visibly through the open boathouse doors across the harbor water, "
    "obscuring the moored sailboat into a soft misty silhouette, and ignite the brass ship's lantern "
    "on the right oak piling so warm golden flame casts a glowing amber halo onto the nearby cedar "
    "wall while swirling cool blue mist fills the doorway. Keep the exact wooden boathouse architecture, "
    "plank lines, coiled ropes, hanging lantern bracket, and static camera framing from <Picture 1>."
)

EXP6A_BEAT3_PROMPT = (
    "Edit <Picture 1>. Advance the coastal twilight into a deep moonlit midnight storm outside the "
    "open boathouse doorway with foaming whitecap waves breaking over wet dark dock planks, driving "
    "silver rain mist blowing diagonally across deep charcoal-indigo harbor water, and intensify the "
    "lit brass ship's lantern on the right oak piling into a vivid golden beacon warming the wet "
    "foreground timbers. Keep the exact wooden boathouse structure, coiled hemp ropes, hanging brass "
    "lantern position, and fixed straight-on camera perspective from <Picture 1>."
)

EXP6A_VIDEO_PROMPT = (
    "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
    "0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of "
    "the target video.\n\n"
    "integrated_multimodal_description: [Shot 1] Live-action, cinematic, holding a steady straight-on wide "
    "composition inside the weathered coastal wooden boathouse pier across the 8.00-second real-time take. "
    "Opening in the warm golden-hour harbor calm of Picture 1 with calm amber water and unlit brass "
    "lantern, cold silver sea fog rolls briskly inward through the open doorway as golden daylight drains "
    "to cobalt twilight and the hanging brass lantern ignites with warm amber spill across the cedar planks. "
    "By the second half of the continuous take, indigo storm gusts blow sea spray past the wet doorway as "
    "whitecaps swell beneath midnight moonlight, settling smoothly into the stormy rain-swept midnight "
    "harbor scene of Picture 2 while the lantern glows steadily in foreground shelter. No human figures enter "
    "frame and no dialogue is spoken.\n\n"
    "overall_soundscape: Gentle calm harbor water lapping against wooden pilings gives way smoothly to rising "
    "coastal evening wind rushing past timber doors, crisp metal lantern glass ticking lightly, distant muffled "
    "foghorn, and breaking midnight waves showering wet pier planks.\n\n"
    "non_diegetic_music: N/A"
)

EXP7A_POLISH_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Vintage 16mm cinematic frame at the curved cinema box-office "
    "glass booth at dusk under faint neon hum and steady rain pattering softly onto the metal canopy above. "
    "Inside the booth, the woman cashier in her sixties with a flat, papery contralto (S1) taps the glass and "
    "says: <d>[English] Seven thirty, nothing after.</d> The teenage customer at the window, with a thin, hurried "
    "tenor (S2), slides coins into the brass dish and answers: <d>[English] Two, back row.</d> The cashier (S1) "
    "tears two tickets cleanly and replies: <d>[English] Back row is gone, love.</d> Her lips close as tickets "
    "slide over brass.\n\n"
    "overall_soundscape: Crisp brass coin clinks in the dish and sharp paper ticket tears ring cleanly against "
    "soft evening rain pattering on the booth canopy and the warm sixty-cycle hum of an overhead neon sign.\n\n"
    "non_diegetic_music: N/A"
)

EXP7A_FRESH_PROMPT = (
    "integrated_multimodal_description: [Shot 1] Vintage 16mm nocturnal cinema box-office booth engulfed by a "
    "brisk coastal night downpour and resonant street acoustics. Through the curved booth glass, the cashier "
    "(S1), in a hush-toned, gravelly contralto, speaks toward the brass slot: <d>[English] Seven thirty, last screening.</d> "
    "The customer (S2), with a quiet, urgent young tenor, drops coins cleanly across the tray: <d>[English] Two tickets, hurry please.</d> "
    "The cashier (S1) tears two paper stubs sharply and adds: <d>[English] Straight down on the left.</d>\n\n"
    "overall_soundscape: Brisk rain gusts rushing down empty city pavement, heavy brass coins ringing loudly in the steel-rimmed tray, crisp mechanical ticket roll snapping off cleanly, and wet passing tires far down the street.\n\n"
    "non_diegetic_music: N/A"
)

# Strictly within 350-500 words for `guide == "ref"` generation conformance (0 FAIL / 0 WARN).
EXP7B_CONTINUE_PROMPT = (
    "subject_definitions:\n"
    "<Subject 1> is the woman cashier seen inside the ticket booth in <Video 1>, in her sixties with reading glasses and muted vintage cardigan.\n"
    "<Subject 2> is the teenage customer standing at the outer glass booth counter in <Video 1>.\n"
    "<Video 1> is the trailing source clip whose final framing, 16mm colour palette, warm booth lighting, and physical positions the target video resumes directly without cutting.\n"
    "<Audio 1> is the synchronized audio track of <Video 1>, whose vocal timbres and pavement ambience continue in character without copying.\n\n"
    "summary:\n"
    "[video continuation + audio reference] The target video resumes unbroken from the final frame of <Video 1> as <Subject 1> finishes handing over both torn tickets and <Subject 2> pockets his change and steps aside toward the lit theater lobby.\n\n"
    "retention_analysis:\n"
    "<Subject 1> (appears in [Shot 1]): fully_preserved - face, glasses, posture, cardigan, and papery contralto voice continue from <Video 1>.\n"
    "<Subject 2> (appears in [Shot 1]): fully_preserved - jacket, posture at the counter, and hurried young tenor continue seamlessly.\n"
    "<Video 1> (continuation source): fully_preserved - exact curved booth window framing, brass coin dish, warm bulb spill, and 16mm halation resume on frame zero.\n"
    "<Audio 1>: reference - speaker vocal pitch characteristics and neon pavement acoustics continue smoothly.\n\n"
    "detailed_description:\n"
    "The target video retains the exact vintage 16mm colour grading, warm tungsten bulb glow, fine silver film grain, and slight halation around the red-lettered schedule board established in <Video 1>.\n"
    "[Shot 1] Opening seamlessly on the closing composition and exact framing of <Video 1> along the curved pavement side of the cinema box-office booth, the static medium shot holds steadily on the glass speaking grate and recessed brass coin dish beneath the illuminated glass transom. Behind the counter inside the warmly lit wooden interior, <Subject 1> rests her fingertips lightly atop the remaining pink ticket roll while both torn cardboard stubs sit squarely beside the brass dish in clear foreground focus. Facing the curved window from the damp dusk pavement, <Subject 2> leans slightly forward in his dark jacket, sweeps the brass coins cleanly into his palm with a quick deliberate motion, and closes his fingers around the two loose ticket stubs resting near the lower lip of the metal tray. Keeping eye contact briefly through the glass reflections, <Subject 2> (S2) raises his chin, glances toward the warm amber light spilling from the glass lobby doors to the left, and says crisply in the same hurried young tenor (S2): <d>[English] Thanks, just made it.</d> His lips close cleanly as he tucks the torn ticket stubs smoothly between his thumb and forefinger, shifts his weight off the counter edge onto the damp concrete pavement, and turns half a step leftward away from the glass partition toward the recessed lobby entrance. Inside the booth behind the curved pane, <Subject 1> keeps her mouth closed without speaking, picks up a short wooden pencil from beside the ticket spindle, marks a neat single stroke across a lined tally sheet beside the brass tray, and settles her elbows calmly as golden interior bulb light glints across the polished metal rim of the coin dish and window frame. Subtle film weave, drifting cool evening dusk air past the booth corner, faint neon tube flicker above the schedule header, and soft reflections across the curved safety glass persist naturally across the continuous real-time take. As <Subject 2> exits slightly past the left edge of the counter rail into warm lobby spill, the camera holds cleanly on the illuminated counter, the brass coin dish, and <Subject 1> straightening her ticket roll inside the quiet glass booth until the shot concludes.\n\n"
    "overall_soundscape:\n"
    "Loose street air and a sixty-cycle overhead neon transformer hum continue seamlessly from <Audio 1>, joined by brass coins clinking into a closed palm, damp soles pivoting softly over pavement, a dry pencil stroke on ledger paper, and a muffled heavy glass lobby door brushing open softly off-screen.\n\n"
    "non_diegetic_music:\n"
    "N/A"
)


# ---------------------------------------------------------------------------
# Utilities & Clean-Room IO
# ---------------------------------------------------------------------------

def _scrub_paths(obj: Any) -> Any:
    raw = str(OUT_ROOT).rstrip("/")
    inp = str(os.environ.get("H3_INPUT_DIR") or "").rstrip("/")
    comfy = str(COMFY_ROOT).rstrip("/")
    if isinstance(obj, str):
        s = obj.replace(raw, "$H3_OUTPUT_DIR")
        if inp:
            s = s.replace(inp, "$H3_INPUT_DIR")
        s = s.replace(comfy, "$COMFY_ROOT")
        return s
    if isinstance(obj, list):
        return [_scrub_paths(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _scrub_paths(v) for k, v in obj.items()}
    return obj


def _unscrub_paths(obj: Any) -> Any:
    raw = str(OUT_ROOT).rstrip("/")
    inp = str(os.environ.get("H3_INPUT_DIR") or "").rstrip("/")
    comfy = str(COMFY_ROOT).rstrip("/")
    if isinstance(obj, str):
        s = obj.replace("$H3_OUTPUT_DIR", raw).replace("$COMFY_ROOT", comfy)
        if inp:
            s = s.replace("$H3_INPUT_DIR", inp)
        return s
    if isinstance(obj, list):
        return [_unscrub_paths(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _unscrub_paths(v) for k, v in obj.items()}
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_scrub_paths(payload), indent=2) + "\n", encoding="utf-8")


def _http_upload_file(file_path: Path, *, subfolder: str = UPLOAD_SUBFOLDER, base_url: str = COMFY_URL) -> str:
    """Upload PNG/MP4 to ComfyUI `input/<subfolder>/` via `/upload/image` and return relative path."""
    boundary = f"----h3wave2{uuid.uuid4().hex}"
    ctype = "video/mp4" if file_path.suffix.lower() == ".mp4" else "image/png"
    raw = file_path.read_bytes()
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"subfolder\"\r\n\r\n{subfolder}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ninput\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{file_path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode(),
        raw,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    body = b"".join(parts)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
    sub = str(res.get("subfolder") or subfolder).strip("/")
    name = str(res.get("name") or file_path.name)
    return f"{sub}/{name}" if sub else name


def _extract_png_frame(video_path: Path, frame_idx: int, out_png: Path) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video_path),
        "-vf", f"select=eq(n\\,{int(frame_idx)})",
        "-vframes", "1",
        str(out_png),
    ]
    subprocess.run(cmd, check=True)
    if not out_png.is_file():
        raise RuntimeError(f"Failed to extract frame {frame_idx} from {video_path}")
    return out_png


def _probe_video_frame_count(video_path: Path) -> int:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0",
            str(video_path),
        ],
        text=True,
    ).strip()
    return int(out.splitlines()[0])


def _resolve_output_file(entry: dict[str, Any]) -> Path:
    if entry.get("fullpath"):
        return Path(str(entry["fullpath"]))
    sub = str(entry.get("subfolder") or "").strip("/")
    fn = str(entry.get("filename") or "")
    root = OUT_ROOT if entry.get("type", "output") == "output" else (COMFY_ROOT / str(entry.get("type", "temp")))
    return (root / sub / fn) if sub else (root / fn)


def _check_live_schema(base_url: str = COMFY_URL) -> tuple[dict[str, Any], dict[str, Any]]:
    info = comfy_client.get_object_info("MiniMaxH3PromptExpander", base_url=base_url, timeout=10.0)
    pe = info.get("MiniMaxH3PromptExpander")
    if not isinstance(pe, dict):
        raise RuntimeError("MiniMaxH3PromptExpander not registered on live ComfyUI")
    outputs = list(pe.get("output_name") or ())
    if len(outputs) < 5 or outputs[4] != "ref_end_percent":
        raise RuntimeError(
            f"Live MiniMaxH3PromptExpander outputs={outputs!r} (expected slot 4 'ref_end_percent'; restart ComfyUI first)"
        )
    queue = comfy_client.get_queue(base_url=base_url, timeout=10.0)
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError(f"ComfyUI queue is non-empty: {queue!r}")
    all_info = comfy_client.get_object_info(base_url=base_url, timeout=15.0)
    return info, all_info


def _preflight_validate_graph(arm: str, graph: dict[str, Any]) -> dict[str, Any]:
    """Run offline `preflight_graph.grade` over every prompt carrier in `graph`."""
    results: dict[str, Any] = {}
    for nid, node in sorted(graph.items(), key=lambda x: str(x[0])):
        ct = node.get("class_type")
        if ct in h3_config.PROMPT_INPUTS:
            # Skip linked PromptExpander outputs that don't have concrete prompt strings yet.
            p_val = node.get("inputs", {}).get(h3_config.PROMPT_INPUTS[ct])
            if isinstance(p_val, list):
                src = graph.get(str(p_val[0]), {})
                if src.get("class_type") == "MiniMaxH3PromptExpander":
                    results[f"node_{nid}_{ct}"] = [("note", "dynamic MiniMaxH3PromptExpander output validated post-render")]
                    continue
            res = pf.grade(node, graph, stem=arm)
            results[f"node_{nid}_{ct}"] = res
            fails = [m for sev, m in res if sev == "FAIL"]
            warns = [m for sev, m in res if sev == "WARN"]
            if fails:
                raise ValueError(f"{arm} node {nid} ({ct}) failed preflight_graph: {fails}")
            if warns:
                print(f"[WARN] {arm} node {nid} ({ct}): {warns}")
    return results


def _run_and_record(
    arm: str,
    graph: dict[str, Any],
    *,
    client_id: str,
    dry_run: bool,
    resume: bool,
    base_url: str = COMFY_URL,
) -> dict[str, Any]:
    pf_res = _preflight_validate_graph(arm, graph)
    rec_path = DATA_DIR / f"{arm}.json"
    if dry_run:
        return {"arm": arm, "status": "dry_run_ok", "preflight": pf_res, "nodes": len(graph)}
    if resume and rec_path.is_file():
        cached = _unscrub_paths(json.loads(rec_path.read_text(encoding="utf-8")))
        if cached.get("status") == "success":
            print(f"[RESUME] {arm}: skipping completed arm")
            return cached

    t0 = time.monotonic()
    prompt_id = comfy_client.submit_prompt(graph, client_id=client_id, base_url=base_url)
    entry = comfy_client.wait_for_prompt(prompt_id, base_url=base_url, timeout=1800.0)
    wall_s = round(time.monotonic() - t0, 2)
    gpu_s = comfy_client.execution_duration_s(entry)
    files = comfy_client.collect_saved_files(entry)
    out_summary = {
        "arm": arm,
        "status": "success",
        "prompt_id": prompt_id,
        "wall_s": wall_s,
        "gpu_s": gpu_s,
        "files": [
            {
                **f,
                "resolved": str(_resolve_output_file(f)),
            }
            for f in files
        ],
        "outputs_raw": entry.get("outputs") or {},
        "graph": graph,
    }
    _write_json(rec_path, out_summary)
    print(f"[DONE] {arm}: wall={wall_s}s gpu={gpu_s}s files={len(files)}")
    return out_summary


def _make_1fps_companion(mp4_path: Path) -> Path | None:
    if not mp4_path.is_file():
        return None
    try:
        n_frames = _probe_video_frame_count(mp4_path)
    except Exception:
        n_frames = 1
    out = mp4_path.with_name(f"{mp4_path.stem}.1fps.00000-{max(0, n_frames - 1):05d}.mp4")
    if out.is_file():
        return out
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(mp4_path),
        "-vf", "fps=1,scale=960:-2",
        "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
        "-an", str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


# ---------------------------------------------------------------------------
# Shared Base Video Backbone Helpers (FL2VA / P-H3-Image)
# ---------------------------------------------------------------------------

def _fl2va_loaders() -> dict[str, Any]:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": h3_config.MODELS["unet_fl2va"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": h3_config.MODELS["clip"], "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["audio_vae"]}},
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": h3_config.PIKA_FL2VA_SVD995_LORA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "19": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["18", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"}},
        "21": {"class_type": "MiniMaxH3Sol", "inputs": {"model": ["58", 0], **h3_config.SOL_RECOMMENDED_CUDA}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": h3_config.PIKA_SAMPLER}},
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": h3_config.PIKA_MANUAL_SIGMAS}},
    }


def _wire_video_outputs(
    g: dict[str, Any],
    *,
    arm_name: str,
    model_ref: list[Any],
    samples_ref: list[Any],
    sigmas_ref: list[Any],
    note: str,
) -> None:
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {
            "latent": samples_ref,
            "model": model_ref,
            "sigmas": sigmas_ref,
            "clip": ["2", 0],
            "note": note,
        },
    }
    g["11"] = {"class_type": "VAEDecode", "inputs": {"samples": ["30", 0], "vae": ["3", 0]}}
    g["12"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["30", 0], "vae": ["4", 0]}}
    g["13"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["11", 0],
            "audio": ["12", 0],
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": f"Video/{RUN_TAG}/{arm_name}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 19,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
    }


# ---------------------------------------------------------------------------
# Stage builders: Dynamic HUD Spotlights, Exp 4, Exp 6A, Exp 6B, Exp 7
# ---------------------------------------------------------------------------

SPOTLIGHT_SPECS: list[dict[str, Any]] = [
    {
        "arm_name": "spot_chaos_cafe",
        "prompt_path": "prompt_bank/t2va_cafe_kids.txt",
        "seed": 6101,
        "feather": 0.07,
        "scale": 1.8,
        "streams": "both",
        "title": "CHAOS PANIC SPOTLIGHT · EXPANDING & MOVING PORTAL",
        "subtitle": "Inside ring: chaos_panic (+1.35, erasure, scale=1.8) | Outside: calm cafe kids",
        "steers": [
            {
                "vector": "bank/chaos_panic__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.35,
                "edit_pass": "edited_only",
            }
        ],
        "portal": [
            {"frame": 0, "x": 0.22, "y": 0.52, "rx": 0.14, "ry": 0.14, "shape": "circle", "label": "CHAOS CORE"},
            {"frame": 48, "x": 0.48, "y": 0.48, "rx": 0.38, "ry": 0.38, "shape": "circle", "label": "CHAOS SURGE"},
            {"frame": 96, "x": 0.60, "y": 0.50, "rx": 0.56, "ry": 0.56, "shape": "circle", "label": "PEAK PANIC"},
            {"frame": 140, "x": 0.76, "y": 0.52, "rx": 0.17, "ry": 0.17, "shape": "circle", "label": "CHAOS EXIT"},
        ],
    },
    {
        "arm_name": "spot_blizzard_courtyard",
        "prompt_path": "prompt_bank/t2va_spatial_reality_portal.txt",
        "seed": 6102,
        "feather": 0.08,
        "scale": 1.9,
        "streams": "both",
        "title": "HEAVY BLIZZARD PORTAL · GOLDEN COURTYARD BLOOM",
        "subtitle": "Inside ring: heavy_blizzard (+1.45, erasure, scale=1.9) | Outside: golden courtyard",
        "steers": [
            {
                "vector": "bank/heavy_blizzard__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.45,
                "edit_pass": "edited_only",
            }
        ],
        "portal": [
            {"frame": 0, "x": 0.78, "y": 0.42, "rx": 0.13, "ry": 0.13, "shape": "circle", "label": "FROST SEED"},
            {"frame": 54, "x": 0.50, "y": 0.50, "rx": 0.58, "ry": 0.58, "shape": "circle", "label": "BLIZZARD BLOOM"},
            {"frame": 102, "x": 0.30, "y": 0.55, "rx": 0.34, "ry": 0.34, "shape": "circle", "label": "DRIFT WEST"},
            {"frame": 140, "x": 0.20, "y": 0.58, "rx": 0.15, "ry": 0.15, "shape": "circle", "label": "FROST FOCUS"},
        ],
    },
    {
        "arm_name": "spot_bricks_dancer",
        "prompt_path": "prompt_bank/t2va_dancer_stream_243.txt",
        "seed": 6103,
        "feather": 0.07,
        "scale": 1.9,
        "streams": "video_only",
        "title": "TOY BRICK WORLD · BREATHING ELLIPSE PORTAL",
        "subtitle": "Inside ellipse: toy_brick_world (+1.45, video_only) | Outside: live studio dancer",
        "steers": [
            {
                "vector": "bank/toy_brick_world__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.45,
                "edit_pass": "edited_only",
            }
        ],
        "portal": [
            {"frame": 0, "x": 0.30, "y": 0.45, "rx": 0.14, "ry": 0.20, "shape": "ellipse", "label": "BRICK SEED"},
            {"frame": 45, "x": 0.50, "y": 0.48, "rx": 0.46, "ry": 0.56, "shape": "ellipse", "label": "BRICK EXPAND"},
            {"frame": 90, "x": 0.54, "y": 0.42, "rx": 0.18, "ry": 0.26, "shape": "ellipse", "label": "BRICK TIGHT"},
            {"frame": 140, "x": 0.50, "y": 0.52, "rx": 0.56, "ry": 0.64, "shape": "ellipse", "label": "BRICK WORLD"},
        ],
    },
    {
        "arm_name": "spot_cyberpunk_noodle",
        "prompt_path": "prompt_bank/t2va_noodle_bar.txt",
        "seed": 6104,
        "feather": 0.08,
        "scale": 1.9,
        "streams": "video_only",
        "title": "CYBERPUNK NEON RAIN · EXPANDING STREET PORTAL",
        "subtitle": "Inside ring: cyberpunk_neon_rain (+1.45, video_only) | Outside: warm noodle counter",
        "steers": [
            {
                "vector": "bank/cyberpunk_neon_rain__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.45,
                "edit_pass": "edited_only",
            }
        ],
        "portal": [
            {"frame": 0, "x": 0.18, "y": 0.36, "rx": 0.14, "ry": 0.14, "shape": "circle", "label": "NEON SPARK"},
            {"frame": 50, "x": 0.46, "y": 0.50, "rx": 0.40, "ry": 0.40, "shape": "circle", "label": "NEON SWEEP"},
            {"frame": 100, "x": 0.62, "y": 0.52, "rx": 0.62, "ry": 0.62, "shape": "circle", "label": "NEON FLOOD"},
            {"frame": 140, "x": 0.80, "y": 0.46, "rx": 0.16, "ry": 0.16, "shape": "circle", "label": "NEON LOCK"},
        ],
    },
    {
        "arm_name": "spot_dual_watercolor_thermal",
        "prompt_path": "prompt_bank/t2va_pika_balcony_orbit.txt",
        "seed": 6105,
        "feather": 0.07,
        "scale": 1.45,
        "streams": "video_only",
        "title": "DUAL SPLIT-WORLD · WATERCOLOR VS THERMAL PORTAL",
        "subtitle": "Outside: watercolor_wash (+1.30, original_only) | Inside: thermal_infrared (+1.45, edited_only)",
        "steers": [
            {
                "vector": "bank/watercolor_wash__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.30,
                "edit_pass": "original_only",
            },
            {
                "vector": "bank/thermal_infrared__fl2va__diff_span__lora_2a8c2b52.safetensors",
                "strength": 1.45,
                "edit_pass": "edited_only",
            },
        ],
        "portal": [
            {"frame": 0, "x": 0.50, "y": 0.48, "rx": 0.14, "ry": 0.14, "shape": "circle", "label": "THERMAL EYE"},
            {"frame": 55, "x": 0.38, "y": 0.46, "rx": 0.48, "ry": 0.48, "shape": "circle", "label": "THERMAL WAVE"},
            {"frame": 100, "x": 0.66, "y": 0.52, "rx": 0.52, "ry": 0.52, "shape": "circle", "label": "DUAL SWEEP"},
            {"frame": 140, "x": 0.52, "y": 0.50, "rx": 0.16, "ry": 0.16, "shape": "circle", "label": "THERMAL PIN"},
        ],
    },
]

SPOTLIGHT_LONG_SPECS: list[dict[str, Any]] = [
    {
        "arm_slug": "stairwell_noir_chaos",
        "prompt_path": "prompt_bank/t2va_stairwell_dialogue.txt",
        "seed": 6201,
        "feather": 0.07,
        "scale": 1.8,
        "streams": "both",
        "title": "STAIRWELL NOIR + CHAOS PORTAL · STACK2 ROS",
        "subtitle": "Inside ring: noir (+1.25) + chaos_panic (+1.15) | Outside: concrete stairwell",
        "steers": [
            {"vector": "noir", "strength": 1.25, "edit_pass": "edited_only"},
            {"vector": "chaos_panic", "strength": 1.15, "edit_pass": "edited_only"},
        ],
        "portal": [
            {"t_frac": 0.00, "x": 0.22, "y": 0.48, "rx": 0.14, "ry": 0.16, "shape": "circle", "label": "NOIR SEED"},
            {"t_frac": 0.25, "x": 0.46, "y": 0.48, "rx": 0.34, "ry": 0.38, "shape": "circle", "label": "PANIC BLOOM"},
            {"t_frac": 0.55, "x": 0.58, "y": 0.50, "rx": 0.58, "ry": 0.62, "shape": "circle", "label": "PEAK FRENZY"},
            {"t_frac": 0.80, "x": 0.42, "y": 0.48, "rx": 0.30, "ry": 0.34, "shape": "circle", "label": "TIGHTEN"},
            {"t_frac": 1.00, "x": 0.72, "y": 0.50, "rx": 0.16, "ry": 0.18, "shape": "circle", "label": "NOIR EXIT"},
        ],
    },
    {
        "arm_slug": "kitchen_chaos",
        "prompt_path": "prompt_bank/t2va_restaurant_kitchen.txt",
        "seed": 6202,
        "feather": 0.07,
        "scale": 1.8,
        "streams": "both",
        "title": "KITCHEN LINE CHAOS · EXPANDING & MOVING PORTAL",
        "subtitle": "Inside ring: chaos_panic (+1.35, erasure, scale=1.8) | Outside: restaurant line",
        "steers": [
            {"vector": "chaos_panic", "strength": 1.35, "edit_pass": "edited_only"},
        ],
        "portal": [
            {"t_frac": 0.00, "x": 0.20, "y": 0.52, "rx": 0.14, "ry": 0.14, "shape": "circle", "label": "CHAOS CORE"},
            {"t_frac": 0.25, "x": 0.44, "y": 0.48, "rx": 0.36, "ry": 0.36, "shape": "circle", "label": "LINE SURGE"},
            {"t_frac": 0.55, "x": 0.60, "y": 0.50, "rx": 0.58, "ry": 0.58, "shape": "circle", "label": "PEAK PANIC"},
            {"t_frac": 0.80, "x": 0.70, "y": 0.50, "rx": 0.32, "ry": 0.32, "shape": "circle", "label": "PASS SWEEP"},
            {"t_frac": 1.00, "x": 0.78, "y": 0.52, "rx": 0.17, "ry": 0.17, "shape": "circle", "label": "CHAOS EXIT"},
        ],
    },
    {
        "arm_slug": "arena_claymation",
        "prompt_path": "prompt_bank/t2va_arena_rock_video.txt",
        "seed": 6203,
        "feather": 0.07,
        "scale": 1.9,
        "streams": "video_only",
        "title": "ARENA CLAYMATION · BREATHING ELLIPSE PORTAL",
        "subtitle": "Inside ellipse: claymation_stopmotion (+1.45, video_only) | Outside: live rock arena",
        "steers": [
            {"vector": "claymation_stopmotion", "strength": 1.45, "edit_pass": "edited_only"},
        ],
        "portal": [
            {"t_frac": 0.00, "x": 0.32, "y": 0.46, "rx": 0.14, "ry": 0.20, "shape": "ellipse", "label": "CLAY SEED"},
            {"t_frac": 0.25, "x": 0.50, "y": 0.48, "rx": 0.46, "ry": 0.56, "shape": "ellipse", "label": "CLAY EXPAND"},
            {"t_frac": 0.55, "x": 0.54, "y": 0.44, "rx": 0.20, "ry": 0.28, "shape": "ellipse", "label": "STAGE PIN"},
            {"t_frac": 0.80, "x": 0.50, "y": 0.50, "rx": 0.56, "ry": 0.64, "shape": "ellipse", "label": "CLAY WORLD"},
            {"t_frac": 1.00, "x": 0.66, "y": 0.48, "rx": 0.22, "ry": 0.30, "shape": "ellipse", "label": "CLAY LOCK"},
        ],
    },
    {
        "arm_slug": "samurai_blizzard",
        "prompt_path": "prompt_bank/t2va_samurai_bamboo_duel.txt",
        "seed": 6204,
        "feather": 0.08,
        "scale": 1.9,
        "streams": "both",
        "title": "BAMBOO DUEL BLIZZARD · EXPANDING FROST PORTAL",
        "subtitle": "Inside ring: heavy_blizzard (+1.45, erasure, scale=1.9) | Outside: misty bamboo grove",
        "steers": [
            {"vector": "heavy_blizzard", "strength": 1.45, "edit_pass": "edited_only"},
        ],
        "portal": [
            {"t_frac": 0.00, "x": 0.78, "y": 0.44, "rx": 0.13, "ry": 0.13, "shape": "circle", "label": "FROST SEED"},
            {"t_frac": 0.25, "x": 0.56, "y": 0.48, "rx": 0.38, "ry": 0.38, "shape": "circle", "label": "GALE DRIFT"},
            {"t_frac": 0.55, "x": 0.48, "y": 0.50, "rx": 0.60, "ry": 0.60, "shape": "circle", "label": "BLIZZARD PEAK"},
            {"t_frac": 0.80, "x": 0.30, "y": 0.54, "rx": 0.34, "ry": 0.34, "shape": "circle", "label": "WEST SQUALL"},
            {"t_frac": 1.00, "x": 0.20, "y": 0.56, "rx": 0.15, "ry": 0.15, "shape": "circle", "label": "FROST FOCUS"},
        ],
    },
    {
        "arm_slug": "courtroom_rubberhose",
        "prompt_path": "prompt_bank/t2va_courtroom_verdict.txt",
        "seed": 6205,
        "feather": 0.07,
        "scale": 1.9,
        "streams": "video_only",
        "title": "1930S RUBBER-HOSE VERDICT · MOVING CARTOON PORTAL",
        "subtitle": "Inside ring: rubber_hose_1930s (+1.45, video_only) | Outside: live courtroom",
        "steers": [
            {"vector": "rubber_hose_1930s", "strength": 1.45, "edit_pass": "edited_only"},
        ],
        "portal": [
            {"t_frac": 0.00, "x": 0.24, "y": 0.46, "rx": 0.15, "ry": 0.15, "shape": "circle", "label": "INK SPARK"},
            {"t_frac": 0.25, "x": 0.48, "y": 0.48, "rx": 0.40, "ry": 0.40, "shape": "circle", "label": "CEL SWEEP"},
            {"t_frac": 0.55, "x": 0.58, "y": 0.50, "rx": 0.60, "ry": 0.60, "shape": "circle", "label": "TOON FLOOD"},
            {"t_frac": 0.80, "x": 0.42, "y": 0.48, "rx": 0.34, "ry": 0.34, "shape": "circle", "label": "GAVEL IRIS"},
            {"t_frac": 1.00, "x": 0.74, "y": 0.46, "rx": 0.16, "ry": 0.16, "shape": "circle", "label": "INK PIN"},
        ],
    },
]


def _materialize_portal(portal_kfs: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    """Resolve normalized `t_frac` portal keyframes into integer `[0, length - 1]` frames."""
    max_f = max(0, int(length) - 1)
    out: list[dict[str, Any]] = []
    for kf in portal_kfs:
        item = {k: v for k, v in kf.items() if k != "t_frac"}
        if "frame" in item:
            item["frame"] = max(0, min(max_f, int(item["frame"])))
        elif "t_frac" in kf:
            item["frame"] = max(0, min(max_f, int(round(float(kf["t_frac"]) * max_f))))
        else:
            raise ValueError(f"Portal keyframe missing both 'frame' and 't_frac': {kf!r}")
        out.append(item)
    return sorted(out, key=lambda item: int(item["frame"]))


def _expand_long_spotlight_specs(length: int) -> list[dict[str, Any]]:
    """Materialize adult-only long spotlight specs at an `on_both_clocks` video length."""
    len_i = int(length)
    if not h3_rules.on_both_clocks(len_i):
        raise ValueError(f"Spotlight length must sit on on_both_clocks, got {len_i}")
    return [
        {
            **spec,
            "arm_name": f"spot{len_i}_{spec['arm_slug']}",
            "length": len_i,
            "portal": _materialize_portal(spec["portal"], len_i),
        }
        for spec in SPOTLIGHT_LONG_SPECS
    ]


def build_spotlight_arm(spec: dict[str, Any], *, length: int | None = None) -> dict[str, Any]:
    """Build a diverse moving/expanding/shrinking HUD Spotlight graph (`1152x768`, `on_both_clocks`)."""
    target_len = int(length if length is not None else spec.get("length", 141))
    if not h3_rules.on_both_clocks(target_len):
        raise ValueError(f"Spotlight length must sit on on_both_clocks, got {target_len}")
    arm_name = str(spec.get("arm_name") or f"spot{target_len}_{spec['arm_slug']}")
    portal = _materialize_portal(spec["portal"], target_len)
    prompt_text = (REPO / str(spec["prompt_path"])).read_text(encoding="utf-8").strip()
    events_json = json.dumps({"portal": portal, "targets": []}, indent=2)

    g = _fl2va_loaders()
    g["27"] = {
        "class_type": "MiniMaxH3Resolution",
        "inputs": {
            "shape": "wide",
            "shape.wide_resolution": "1152x768  3/2  864 tok/frame  0.73x",
            "length": target_len,
        },
    }
    g["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": prompt_text}}
    g["5"] = {
        "class_type": "MiniMaxH3Conditioning",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": ["133", 0],
            "width": ["27", 0],
            "height": ["27", 1],
            "length": ["27", 2],
            "canvas": "explicit",
        },
    }
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": int(spec["seed"])}}
    g["160"] = {
        "class_type": "MiniMaxH3SpatialDirector",
        "inputs": {
            "mode": "portal_path",
            "width": ["27", 0],
            "height": ["27", 1],
            "length": ["27", 2],
            "feather": float(spec["feather"]),
            "invert_mask": False,
            "events_json": events_json,
        },
    }

    head_ref: list[Any] = ["21", 0]
    for idx, st in enumerate(spec["steers"]):
        nid = str(115 + idx)
        g[nid] = {
            "class_type": "MiniMaxH3Steer",
            "inputs": {
                "model": head_ref,
                "vector": str(st["vector"]),
                "strength": float(st["strength"]),
                "mode": "add_norm_matched_erasure",
                "target": "scene_only",
                "start_percent": 0.0,
                "end_percent": 1.0,
                "streams": "both",
                "hold_words": "",
                "hold_alpha": 1.0,
                "allow_recipe_mismatch": True,
                "edit_pass": str(st["edit_pass"]),
            },
        }
        head_ref = [nid, 0]

    g["26"] = {
        "class_type": "MiniMaxH3Preflight",
        "inputs": {"conditioning": ["5", 0], "samples": ["5", 1], "edited_conditioning": ["5", 0]},
    }
    g["9"] = {
        "class_type": "MiniMaxH3EditGuider",
        "inputs": {
            "model": head_ref,
            "original": ["26", 0],
            "edited": ["5", 0],
            "scale": float(spec["scale"]),
            "first_step": 0,
            "align_positions": False,
            "last_step": 1000,
            "streams": str(spec["streams"]),
            "audio_scale": 1.0,
            "mask": ["160", 0],
        },
    }
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["6", 0],
            "guider": ["9", 0],
            "sampler": ["7", 0],
            "sigmas": ["60", 0],
            "latent_image": ["26", 1],
        },
    }
    _wire_video_outputs(
        g,
        arm_name=f"{arm_name}_clean",
        model_ref=head_ref,
        samples_ref=["10", 0],
        sigmas_ref=["60", 0],
        note=f"spotlights:{arm_name}",
    )
    g["161"] = {
        "class_type": "MiniMaxH3DirectorHUD",
        "inputs": {
            "images": ["11", 0],
            "director_plan": ["160", 3],
            "hud_layout": "director_card",
            "title": str(spec["title"]),
            "subtitle": str(spec["subtitle"]),
            "ring_opacity": 0.85,
        },
    }
    g["162"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": ["161", 0],
            "audio": ["12", 0],
            "frame_rate": 24.0,
            "loop_count": 0,
            "filename_prefix": f"Video/{RUN_TAG}/{arm_name}_hud",
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


def build_exp4_mask_extractor(clip_path: Path, out_prefix: str) -> dict[str, Any]:
    return {
        "1": {
            "class_type": h3_config.MASKED_JOB_LOADER,
            "inputs": {
                "video": str(clip_path),
                "force_rate": h3_config.MASKED_JOB_FORCE_RATE,
                "custom_width": 1344,
                "custom_height": 768,
                "frame_load_cap": 141,
                "start_time": 0.0,
                "format": "None",
            },
        },
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": h3_config.SEGMENTER}},
        "3": {
            "class_type": "MiniMaxH3SAM31Corrections",
            "inputs": {"segmenter": ["2", 0], "segmenter_clip": ["2", 1], **h3_config.MASKED_JOB_SAM_CORRECTIONS},
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 1], "text": "woman"}},
        "5": {
            "class_type": "SAM3_VideoTrack",
            "inputs": {
                "images": ["1", 0],
                "model": ["3", 0],
                "conditioning": ["4", 0],
                "detection_threshold": 0.45,
                "max_objects": 4,
                "detect_interval": 12,
            },
        },
        "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": ""}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 24.0,
                "filename_prefix": out_prefix,
                **h3_config.MASKED_JOB_LOSSLESS,
            },
        },
    }


def build_exp4_2x_arm(
    *,
    arm_name: str,
    first_frame_ref: str,
    mask_mkv_path: Path | None,
    vector: str | None,
    streams: str,
    scale: float,
    seed: int = 730451892,
) -> dict[str, Any]:
    g = _fl2va_loaders()
    g["15"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_ref}}
    g["27"] = {
        "class_type": "MiniMaxH3Resolution",
        "inputs": {"shape": "wide", "shape.wide_resolution": "1344x768  7/4  1008 tok/frame  1.00x", "length": 141},
    }
    g["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": EXP4_PROMPT_I2VA}}
    g["5"] = {
        "class_type": "MiniMaxH3Conditioning",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": ["133", 0],
            "width": ["27", 0],
            "height": ["27", 1],
            "length": ["27", 2],
            "canvas": "explicit",
            "first_frame": ["15", 0],
        },
    }
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}}

    if vector is None:
        g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]}}
        g["9"] = {"class_type": "BasicGuider", "inputs": {"model": ["21", 0], "conditioning": ["26", 0]}}
        g["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]},
        }
        _wire_video_outputs(g, arm_name=arm_name, model_ref=["21", 0], samples_ref=["10", 0], sigmas_ref=["60", 0], note=f"exp4:{arm_name}")
        return g

    g["115"] = {
        "class_type": "MiniMaxH3Steer",
        "inputs": {
            "model": ["21", 0],
            "vector": vector,
            "strength": 1.6,
            "mode": "add_norm_matched_erasure",
            "target": "scene_only",
            "start_percent": 0.0,
            "end_percent": 1.0,
            "streams": "both",
            "hold_words": "",
            "hold_alpha": 1.0,
            "allow_recipe_mismatch": True,
            "edit_pass": "edited_only",
        },
    }
    g["26"] = {
        "class_type": "MiniMaxH3Preflight",
        "inputs": {"conditioning": ["5", 0], "samples": ["5", 1], "edited_conditioning": ["5", 0]},
    }
    g["200"] = {
        "class_type": h3_config.MASKED_JOB_LOADER,
        "inputs": {
            "video": str(mask_mkv_path),
            "force_rate": h3_config.MASKED_JOB_FORCE_RATE,
            "custom_width": 1344,
            "custom_height": 768,
            "frame_load_cap": 141,
            "start_time": 0.0,
            "format": "None",
        },
    }
    g["201"] = {"class_type": "ImageToMask", "inputs": {"image": ["200", 0], "channel": "red"}}
    g["202"] = {"class_type": "GrowMask", "inputs": {"mask": ["201", 0], "expand": 12, "tapered_corners": True}}
    g["203"] = {"class_type": "MaskToImage", "inputs": {"mask": ["202", 0]}}
    g["204"] = {"class_type": "ImageBlur", "inputs": {"image": ["203", 0], "blur_radius": 10, "sigma": 0.45}}
    g["205"] = {"class_type": "ImageToMask", "inputs": {"image": ["204", 0], "channel": "red"}}
    g["206"] = {"class_type": "InvertMask", "inputs": {"mask": ["205", 0]}}

    g["9"] = {
        "class_type": "MiniMaxH3EditGuider",
        "inputs": {
            "model": ["115", 0],
            "original": ["26", 0],
            "edited": ["5", 0],
            "scale": float(scale),
            "first_step": 0,
            "align_positions": False,
            "last_step": 1000,
            "streams": streams,
            "audio_scale": 1.0,
            "mask": ["206", 0],
        },
    }
    g["10"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]},
    }
    _wire_video_outputs(g, arm_name=arm_name, model_ref=["115", 0], samples_ref=["10", 0], sigmas_ref=["60", 0], note=f"exp4:{arm_name}")
    return g


def build_exp4_1x_masked_arm(
    *,
    arm_name: str,
    src_clip_path: Path,
    mask_mkv_path: Path,
    seed: int = 730451892,
) -> dict[str, Any]:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": h3_config.MODELS["unet_fl2va"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": h3_config.MODELS["clip"], "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["audio_vae"]}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": h3_config.PIKA_SAMPLER}},
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": h3_config.PIKA_MANUAL_SIGMAS}},
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": h3_config.PIKA_FL2VA_SVD995_LORA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "19": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["18", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"}},
        "21": {"class_type": "MiniMaxH3Sol", "inputs": {"model": ["58", 0], **h3_config.SOL_RECOMMENDED_CUDA}},
        "115": {
            "class_type": "MiniMaxH3Steer",
            "inputs": {
                "model": ["21", 0],
                "vector": CV_NOIR_FL2VA,
                "strength": 1.6,
                "mode": "add_norm_matched_erasure",
                "target": "scene_only",
                "start_percent": 0.0,
                "end_percent": 1.0,
                "streams": "video_only",
                "hold_words": "",
                "hold_alpha": 1.0,
                "allow_recipe_mismatch": True,
                "edit_pass": "both",
            },
        },
        "28": {
            "class_type": h3_config.MASKED_JOB_LOADER,
            "inputs": {
                "video": str(src_clip_path),
                "force_rate": 24.0,
                "custom_width": 1344,
                "custom_height": 768,
                "frame_load_cap": 141,
                "start_time": 0.0,
                "format": "AnimateDiff",
            },
        },
        "200": {
            "class_type": h3_config.MASKED_JOB_LOADER,
            "inputs": {
                "video": str(mask_mkv_path),
                "force_rate": h3_config.MASKED_JOB_FORCE_RATE,
                "custom_width": 1344,
                "custom_height": 768,
                "frame_load_cap": 141,
                "start_time": 0.0,
                "format": "None",
            },
        },
        "201": {"class_type": "ImageToMask", "inputs": {"image": ["200", 0], "channel": "red"}},
        # Grow protected subject by 20px first, invert to background, then MaskedSource grow=8/feather=8 keeps 12px margin.
        "202": {"class_type": "GrowMask", "inputs": {"mask": ["201", 0], "expand": 20, "tapered_corners": True}},
        "206": {"class_type": "InvertMask", "inputs": {"mask": ["202", 0]}},
        "104": {
            "class_type": "MiniMaxH3MaskedSource",
            "inputs": {
                "frames": ["28", 0],
                "mask": ["206", 0],
                "grow_pixels": 8,
                "grow_by": "a fixed margin",
                "feather_pixels": 8,
                "replace": "whole subject",
                "paint_out": False,
                "part_phrases": "hair, head",
                "part_threshold": 0.5,
                "part_margin": 8,
                "composite": "whole region",
                "change_threshold": 0.05,
                "motion_reference": "none",
                "motion_short_edge": 384,
                "motion_vae": False,
                "start_from": "noise",
                "start_top": 0.3,
                "start_blur": 16,
                "start_knots": 1,
            },
        },
        "74": {
            "class_type": "MiniMaxH3AudioFreezeSong",
            "inputs": {
                "model": ["115", 0],
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "audio": ["28", 2],
                "sampler": ["7", 0],
                "sigmas": ["60", 0],
                "prompt": EXP4_DINER_CORE,
                "timeline": "",
                "preview": False,
                "width": 1344,
                "height": 768,
                "window_frames": 141,
                "context_frames": 39,
                "extent": "whole",
                "seed": seed,
                "audio_mask": 0.0,
                "level": "clip_guard",
                "filename_prefix": f"Video/{RUN_TAG}/{arm_name}",
                "crf": 19,
                "save_metadata_png": True,
                "keep_windows": False,
                "source": ["104", 0],
                "save_mask_review": True,
            },
        },
    }
    return g


def _add_ph3_storyboard_beats(g: dict[str, Any], *, save_beats: bool) -> None:
    g.update({
        "101": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.IMAGE_GRAPH_VAE}},
        "102": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": h3_config.PH3_IMAGE_LORA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "103": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["102", 0], "shift_video": 1.0, "shift_audio": 1.0}},
        "104": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["103", 0], "attention": "comfy kitchen attention"}},
        "105": {"class_type": "ManualSigmas", "inputs": {"sigmas": h3_config.PH3_IMAGE_MANUAL_SIGMAS}},
        "106": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": h3_config.PH3_IMAGE_SAMPLER}},
        # Beat 1 (t = 0)
        "107": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "3:2", "generate_megapixels": 8.0, "enforce_4mp_floor": True}},
        "108": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {"clip": ["2", 0], "vae": ["101", 0], "prompt": EXP6A_BEAT1_PROMPT, "width": ["107", 0], "height": ["107", 1], "canvas": "explicit"},
        },
        "109": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["108", 0]}},
        "113": {"class_type": "RandomNoise", "inputs": {"noise_seed": 6101}},
        "110": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["113", 0], "guider": ["109", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["108", 1]}},
        "114": {"class_type": "VAEDecode", "inputs": {"samples": ["110", 0], "vae": ["101", 0]}},
        "111": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["114", 0], "target_width": ["107", 2], "target_height": ["107", 3]}},
        # Beat 2 (t = 68, edited from Beat 1)
        "120": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "same as the picture", "picture": ["111", 0], "generate_megapixels": 8.0, "enforce_4mp_floor": True}},
        "121": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {"clip": ["2", 0], "vae": ["101", 0], "prompt": EXP6A_BEAT2_PROMPT, "width": ["120", 0], "height": ["120", 1], "canvas": "explicit", "images.image_1": ["111", 0]},
        },
        "122": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["121", 0]}},
        "123": {"class_type": "RandomNoise", "inputs": {"noise_seed": 6102}},
        "124": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["123", 0], "guider": ["122", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["121", 1]}},
        "125": {"class_type": "VAEDecode", "inputs": {"samples": ["124", 0], "vae": ["101", 0]}},
        "126": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["125", 0], "target_width": ["120", 2], "target_height": ["120", 3]}},
        # Beat 3 (t = 191, edited from Beat 2)
        "140": {"class_type": "MiniMaxH3ImageSize", "inputs": {"megapixels": 4.0, "shape": "same as the picture", "picture": ["126", 0], "generate_megapixels": 8.0, "enforce_4mp_floor": True}},
        "141": {
            "class_type": "MiniMaxH3PH3ImageConditioning",
            "inputs": {"clip": ["2", 0], "vae": ["101", 0], "prompt": EXP6A_BEAT3_PROMPT, "width": ["140", 0], "height": ["140", 1], "canvas": "explicit", "images.image_1": ["126", 0]},
        },
        "142": {"class_type": "BasicGuider", "inputs": {"model": ["104", 0], "conditioning": ["141", 0]}},
        "143": {"class_type": "RandomNoise", "inputs": {"noise_seed": 6103}},
        "144": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["143", 0], "guider": ["142", 0], "sampler": ["106", 0], "sigmas": ["105", 0], "latent_image": ["141", 1]}},
        "145": {"class_type": "VAEDecode", "inputs": {"samples": ["144", 0], "vae": ["101", 0]}},
        "146": {"class_type": "MiniMaxH3ImageDownscale", "inputs": {"images": ["145", 0], "target_width": ["140", 2], "target_height": ["140", 3]}},
    })
    if save_beats:
        g["112"] = {"class_type": "SaveImage", "inputs": {"images": ["111", 0], "filename_prefix": f"Image/{RUN_TAG}/exp6a_beat1_t0"}}
        g["127"] = {"class_type": "SaveImage", "inputs": {"images": ["126", 0], "filename_prefix": f"Image/{RUN_TAG}/exp6a_beat2_f68"}}
        g["147"] = {"class_type": "SaveImage", "inputs": {"images": ["146", 0], "filename_prefix": f"Image/{RUN_TAG}/exp6a_beat3_f191"}}


def build_exp6a_arm(*, arm_name: str, use_mid_guide: bool, guide_end_percent: float, save_beats: bool) -> dict[str, Any]:
    g = _fl2va_loaders()
    _add_ph3_storyboard_beats(g, save_beats=save_beats)
    g["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": EXP6A_VIDEO_PROMPT}}
    g["5"] = {
        "class_type": "MiniMaxH3Conditioning",
        "inputs": {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": ["133", 0],
            "width": 1152,
            "height": 768,
            "length": 192,
            "canvas": "explicit",
            "first_frame": ["111", 0],
            "last_frame": ["146", 0],
        },
    }
    cond_ref: list[Any] = ["5", 0]
    lat_ref: list[Any] = ["5", 1]
    model_ref: list[Any] = ["21", 0]

    if use_mid_guide:
        g["70"] = {
            "class_type": "MiniMaxH3GuideLatent",
            "inputs": {
                "positive": ["5", 0],
                "latent": ["5", 1],
                "frame_idx": 68,
                "single_frame": False,
                "vae": ["3", 0],
                "guide": ["126", 0],
            },
        }
        cond_ref = ["70", 0]
        lat_ref = ["70", 1]
        if guide_end_percent < 0.999:
            g["135"] = {
                "class_type": "MiniMaxH3ReleaseReferences",
                "inputs": {
                    "model": ["21", 0],
                    "end_percent": 1.0,
                    "clean_level": 0.999,
                    "audio_end_percent": 1.0,
                    "audio_clean_level": 1.0,
                    "video_end_percent": 1.0,
                    "guide_end_percent": float(guide_end_percent),
                },
            }
            model_ref = ["135", 0]

    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": cond_ref, "samples": lat_ref}}
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 6201}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": model_ref, "conditioning": ["26", 0]}}
    g["10"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]}}
    _wire_video_outputs(g, arm_name=arm_name, model_ref=model_ref, samples_ref=["10", 0], sigmas_ref=["60", 0], note=f"exp6a:{arm_name}")
    return g


def build_exp6b_multimodal_expander(*, arm_name: str = "exp6b_multimodal_prompt_expander") -> dict[str, Any]:
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": h3_config.MODELS["unet_ref2va"], "weight_dtype": "default"}},
        "2": {"class_type": "MiniMaxH3EncoderLoader", "inputs": {"encoder_name": h3_config.MODELS["clip"], "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["video_vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": h3_config.MODELS["audio_vae"]}},
        "15": {"class_type": "LoadImage", "inputs": {"image": STILL_PORTRAIT}},
        "28": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {
                "video": VIDEO_DANCER,
                "force_rate": 24.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 192,
                "start_time": 0.0,
                "format": "AnimateDiff",
            },
        },
        "29": {
            "class_type": "VHS_LoadVideoFFmpeg",
            "inputs": {
                "video": CLIP_SIREN_AUDIO,
                "force_rate": 24.0,
                "custom_width": 0,
                "custom_height": 0,
                "frame_load_cap": 124,
                "start_time": 0.0,
                "format": "AnimateDiff",
            },
        },
        "50": {
            "class_type": "MiniMaxH3AppendRefImage",
            "inputs": {"image": ["15", 0], "size_policy": "max", "size_policy.dit_short_edge": 2048, "size_policy.allow_upscale": False, "qwen_view": "separate", "qwen_view.qwen_short_edge": 512},
        },
        "52": {
            "class_type": "MiniMaxH3AppendRefVideo",
            "inputs": {"frames": ["28", 0], "video_info": ["28", 3], "references": ["50", 0]},
        },
        "53": {
            "class_type": "MiniMaxH3AppendRefAudio",
            "inputs": {"audio": ["29", 2], "references": ["52", 0]},
        },
        "27": {
            "class_type": "MiniMaxH3Resolution",
            "inputs": {"shape": "wide", "shape.wide_resolution": "1152x768  3/2  864 tok/frame  0.73x", "length": 192},
        },
        "14": {
            "class_type": "MiniMaxH3PromptExpander",
            "inputs": {
                "prompt": (
                    "The woman from Picture 1 stands inside a minimalist sunlit concrete gallery, "
                    "performs the continuous spinning turn from Video 1 holding nothing in her hands, "
                    "and pauses sharply with startled focus as she reacts to the urgent knocks, quiet "
                    "spoken whisper, and distant siren carried by Audio 1."
                ),
                "concept": "none",
                "mode": "auto",
                "length": ["27", 2],
                "enabled": True,
                "base_url": "",
                "model": "",
                "thinking": "on",
                "reasoning_effort": "medium",
                "seed": 6301,
                "references": ["53", 0],
                "preset": "h3app_video_ref2va",
            },
        },
        "75": {"class_type": "PreviewAny", "inputs": {"source": ["14", 0]}},
        "76": {"class_type": "PreviewAny", "inputs": {"source": ["14", 3]}},
        "132": {
            "class_type": "MiniMaxH3EncodeReferences",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["3", 0],
                "audio_vae": ["4", 0],
                "references": ["53", 0],
                "width": ["27", 0],
                "height": ["27", 1],
                "length": ["27", 2],
                "video_policy": "comfy",
                "image_policy": "comfy",
            },
        },
        "5": {"class_type": "MiniMaxH3PromptOnReferences", "inputs": {"encoded_references": ["132", 0], "prompt": ["14", 0]}},
        "26": {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["132", 1]}},
        "18": {
            "class_type": "MiniMaxH3LoRABranch",
            "inputs": {
                "model": ["1", 0],
                "lora_name": h3_config.PIKA_REF2VA_SVD995_LORA,
                "strength": 1.0,
                "modules": "all",
                "blocks": "all",
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
        },
        "19": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["18", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "58": {"class_type": "ModelAttentionBackend", "inputs": {"model": ["19", 0], "attention": "comfy kitchen attention"}},
        "21": {"class_type": "MiniMaxH3Sol", "inputs": {"model": ["58", 0], **h3_config.SOL_RECOMMENDED_CUDA}},
        "135": {
            "class_type": "MiniMaxH3ReleaseReferences",
            "inputs": {"model": ["21", 0], "end_percent": ["14", 4], "clean_level": 0.999, "video_end_percent": ["14", 1]},
        },
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": 6302}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": h3_config.PIKA_SAMPLER}},
        "60": {"class_type": "ManualSigmas", "inputs": {"sigmas": h3_config.PIKA_MANUAL_SIGMAS}},
        "9": {"class_type": "BasicGuider", "inputs": {"model": ["135", 0], "conditioning": ["26", 0]}},
        "10": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]}},
    }
    _wire_video_outputs(g, arm_name=arm_name, model_ref=["135", 0], samples_ref=["10", 0], sigmas_ref=["60", 0], note=f"exp6b:{arm_name}")
    return g


def build_exp7_shot1() -> dict[str, Any]:
    prompt_text = (REPO / "prompt_bank" / "t2va_box_office.txt").read_text(encoding="utf-8").strip()
    g = _fl2va_loaders()
    g["27"] = {"class_type": "MiniMaxH3Resolution", "inputs": {"shape": "wide", "shape.wide_resolution": "1152x768  3/2  864 tok/frame  0.73x", "length": 192}}
    g["133"] = {"class_type": "PrimitiveStringMultiline", "inputs": {"value": prompt_text}}
    g["5"] = {
        "class_type": "MiniMaxH3Conditioning",
        "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": ["133", 0], "width": ["27", 0], "height": ["27", 1], "length": ["27", 2], "canvas": "explicit"},
    }
    g["26"] = {"class_type": "MiniMaxH3Preflight", "inputs": {"conditioning": ["5", 0], "samples": ["5", 1]}}
    g["6"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 7101}}
    g["9"] = {"class_type": "BasicGuider", "inputs": {"model": ["21", 0], "conditioning": ["26", 0]}}
    g["10"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["6", 0], "guider": ["9", 0], "sampler": ["7", 0], "sigmas": ["60", 0], "latent_image": ["26", 1]}}
    _wire_video_outputs(g, arm_name="exp7_shot1_box_office", model_ref=["21", 0], samples_ref=["10", 0], sigmas_ref=["60", 0], note="exp7:shot1")
    return g


def build_exp7_redub_arm(
    *,
    arm_name: str,
    shot1_ref: str,
    redub_mode: str,
    audio_mask: float,
    prompt_text: str,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compiled = dir_compiler.compile_audio_redub(
        {
            "video_ref": shot1_ref,
            "length": 192,
            "width": 1152,
            "height": 768,
            "redub_mode": redub_mode,
            "audio_mask": audio_mask,
            "steps": 6,
            "seed": 7201,
            "audio_prompt": prompt_text,
        },
        object_info=object_info,
    )
    g = compiled["graph"]
    g.pop("13", None)
    g["162"]["inputs"]["save_output"] = True
    g["162"]["inputs"]["filename_prefix"] = f"Video/{RUN_TAG}/{arm_name}"
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {
            "latent": ["87", 0],
            "model": ["88", 0],
            "sigmas": ["60", 0] if "60" in g else ["84", 0],
            "clip": ["2", 0],
            "note": f"exp7a:{arm_name}",
        },
    }
    g["12"]["inputs"]["samples"] = ["30", 0]
    return g


def build_exp7_continue_arm(
    *,
    arm_name: str,
    shot1_ref: str,
    last_frame_ref: str | None,
    strategy: str,
    target_length: int,
    object_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compiled = dir_compiler.compile_continue_shot(
        {
            "video_ref": shot1_ref,
            "last_frame_ref": last_frame_ref or "",
            "strategy": strategy,
            "source_frames": 192,
            "length": target_length,
            "context_frames": 39,
            "width": 1152,
            "height": 768,
            "seed": 7301,
            "prompt": EXP7B_CONTINUE_PROMPT,
            "preview_override": False,
        },
        object_info=object_info,
    )
    g = compiled["graph"]
    g["30"] = {
        "class_type": "MiniMaxH3ProvenanceStamp",
        "inputs": {"latent": ["10", 0], "model": ["21", 0], "sigmas": ["60", 0], "clip": ["2", 0], "note": f"exp7b:{arm_name}"},
    }
    g["11"]["inputs"]["samples"] = ["30", 0]
    g["12"]["inputs"]["samples"] = ["30", 0]
    g["162"]["inputs"]["save_output"] = True
    g["162"]["inputs"]["filename_prefix"] = f"Video/{RUN_TAG}/{arm_name}"
    if "13" in g:
        g["13"]["inputs"]["save_output"] = True
        g["13"]["inputs"]["filename_prefix"] = f"Video/{RUN_TAG}/{arm_name}_full_overlap"
    return g


# ---------------------------------------------------------------------------
# Analysis: Region Color/MAE, M1 DINOv3, M2 YIN F0, Storyboard & Seam Metrics
# ---------------------------------------------------------------------------

def _read_video_rgb(path: Path) -> "Any":
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames, axis=0)


def _read_image_rgb(path: Path) -> "Any":
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _mae_psnr(a: "Any", b: "Any") -> tuple[float, float]:
    import numpy as np
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    mae = float(np.mean(np.abs(fa - fb)))
    mse = float(np.mean((fa - fb) ** 2))
    psnr = 99.0 if mse <= 1e-8 else float(20.0 * math.log10(255.0 / math.sqrt(mse)))
    return round(mae, 3), round(psnr, 2)


def _audio_rms_dbfs(path: Path) -> dict[str, Any]:
    import numpy as np
    raw = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "32000", "-f", "f32le", "pipe:1"]
    )
    arr = np.frombuffer(raw, dtype=np.float32)
    if arr.size == 0:
        return {"rms_dbfs": None, "peak_dbfs": None, "active_frac": 0.0}
    rms = float(np.sqrt(np.mean(arr ** 2)))
    peak = float(np.max(np.abs(arr)))
    return {
        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-7)), 2),
        "peak_dbfs": round(20.0 * math.log10(max(peak, 1e-7)), 2),
        "active_frac": round(float(np.mean(np.abs(arr) > 0.015)), 4),
    }


def _analyze_exp4(summary: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    mask_mkv = OUT_VIDEO_DIR / "exp4_sam_woman_mask_00001.mkv"
    if not mask_mkv.is_file():
        cands = sorted(OUT_VIDEO_DIR.glob("exp4_sam_woman_mask*.mkv"))
        if not cands:
            return {"status": "missing_mask"}
        mask_mkv = cands[0]

    mask_rgb = _read_video_rgb(mask_mkv)
    fg = (mask_rgb[..., 0].astype(np.float32) / 255.0) > 0.5
    bg = ~fg
    fg_frac = round(float(np.mean(fg)), 4)
    bg_frac = round(1.0 - fg_frac, 4)

    def _hsv_sat(rgb_u8: "Any", sel: "Any") -> float:
        f = rgb_u8.astype(np.float32) / 255.0
        mx = np.max(f, axis=-1)
        mn = np.min(f, axis=-1)
        sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
        return round(float(np.mean(sat[sel])), 4)

    arms = ["exp4_base_unsteered", "exp4_2x_noir_video_only", "exp4_2x_noir_streams_both", "exp4_2x_vhs_video_only", "exp4_1x_masked_source_noir"]
    base_vid = None
    src_vid = _read_video_rgb(Path(os.environ["H3_INPUT_DIR"]) / CLIP_DINER)
    out: dict[str, Any] = {"fg_area_frac": fg_frac, "bg_area_frac": bg_frac, "bg_ge_30pct": bg_frac >= 0.30, "arms": {}}

    arm_mp4s: dict[str, Path] = {}
    for arm in arms:
        cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}*-audio.mp4")) or sorted(OUT_VIDEO_DIR.glob(f"{arm}*.mp4"))
        cands = [p for p in cands if ".1fps." not in p.name and "_with_mask" not in p.name]
        if not cands:
            continue
        mp4 = cands[0]
        arm_mp4s[arm] = mp4
        vid = _read_video_rgb(mp4)
        n = min(len(vid), len(fg), len(src_vid))
        vid_n = vid[:n]
        if arm == "exp4_base_unsteered":
            base_vid = vid_n
        fg_sat = _hsv_sat(vid_n, fg[:n])
        bg_sat = _hsv_sat(vid_n, bg[:n])
        fg_src_mae = round(float(np.mean(np.abs(vid_n.astype(np.float32) - src_vid[:n].astype(np.float32))[fg[:n]])), 2)
        bg_src_mae = round(float(np.mean(np.abs(vid_n.astype(np.float32) - src_vid[:n].astype(np.float32))[bg[:n]])), 2)
        rec: dict[str, Any] = {
            "mp4": _scrub_paths(str(mp4)),
            "fg_hsv_sat": fg_sat,
            "bg_hsv_sat": bg_sat,
            "fg_to_bg_sat_ratio": round(fg_sat / max(bg_sat, 1e-4), 2),
            "fg_mae_vs_src": fg_src_mae,
            "bg_mae_vs_src": bg_src_mae,
            "audio": _audio_rms_dbfs(mp4),
        }
        if base_vid is not None and arm != "exp4_base_unsteered":
            rec["fg_mae_vs_base"] = round(float(np.mean(np.abs(vid_n.astype(np.float32) - base_vid[:n].astype(np.float32))[fg[:n]])), 2)
            rec["bg_mae_vs_base"] = round(float(np.mean(np.abs(vid_n.astype(np.float32) - base_vid[:n].astype(np.float32))[bg[:n]])), 2)
        out["arms"][arm] = rec

    # Run clean-room M2 YIN F0 comparison comparing steered 2x arms vs exp4_base_unsteered.
    if "exp4_base_unsteered" in arm_mp4s:
        steered_list = [str(arm_mp4s[a]) for a in ("exp4_2x_noir_video_only", "exp4_2x_noir_streams_both", "exp4_2x_vhs_video_only") if a in arm_mp4s]
        if steered_list:
            yin_out = DATA_DIR / "exp4_yin.json"
            subprocess.run(
                [sys.executable, str(REPO / "bench" / "probe_yin_f0_drift.py"),
                 "--baseline", str(arm_mp4s["exp4_base_unsteered"]),
                 "--steered", *steered_list,
                 "--json", str(yin_out)],
                check=False,
            )
            if yin_out.is_file():
                out["yin_f0"] = _scrub_paths(json.loads(yin_out.read_text(encoding="utf-8")))
    return out


def _analyze_exp6a() -> dict[str, Any]:
    import numpy as np
    b1 = sorted(OUT_IMAGE_DIR.glob("exp6a_beat1_t0*.png"))
    b2 = sorted(OUT_IMAGE_DIR.glob("exp6a_beat2_f68*.png"))
    b3 = sorted(OUT_IMAGE_DIR.glob("exp6a_beat3_f191*.png"))
    if not (b1 and b2 and b3):
        return {"status": "missing_beats"}
    img1 = _read_image_rgb(b1[0])
    img2 = _read_image_rgb(b2[0])
    img3 = _read_image_rgb(b3[0])

    from PIL import Image
    def _fit(arr: "Any", w: int, h: int) -> "Any":
        return np.asarray(Image.fromarray(arr).resize((int(w), int(h)), Image.Resampling.LANCZOS))

    res: dict[str, Any] = {"arms": {}}
    for arm in ("exp6a_bookends_only", "exp6a_storyboard_gep100", "exp6a_storyboard_gep033"):
        cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}*-audio.mp4"))
        cands = [p for p in cands if ".1fps." not in p.name]
        if not cands:
            continue
        mp4 = cands[0]
        vid = _read_video_rgb(mp4)
        h_v, w_v = int(vid.shape[1]), int(vid.shape[2])
        t1, t2, t3 = _fit(img1, w_v, h_v), _fit(img2, w_v, h_v), _fit(img3, w_v, h_v)
        adj = np.mean(np.abs(vid[1:].astype(np.float32) - vid[:-1].astype(np.float32)), axis=(1, 2, 3))
        f67_68 = float(adj[67]) if len(adj) > 68 else 0.0
        f68_69 = float(adj[68]) if len(adj) > 68 else 0.0
        mae_f0, psnr_f0 = _mae_psnr(vid[0], t1)
        mae_f68, psnr_f68 = _mae_psnr(vid[68], t2)
        mae_f191, psnr_f191 = _mae_psnr(vid[-1], t3)
        res["arms"][arm] = {
            "mp4": _scrub_paths(str(mp4)),
            "f0_mae": mae_f0,
            "f0_psnr": psnr_f0,
            "f68_mae": mae_f68,
            "f68_psnr": psnr_f68,
            "f191_mae": mae_f191,
            "f191_psnr": psnr_f191,
            "adj_mae_67_68": round(f67_68, 2),
            "adj_mae_68_69": round(f68_69, 2),
            "clip_adj_mae_mean": round(float(np.mean(adj)), 2),
            "audio": _audio_rms_dbfs(mp4),
        }
    return res


def _analyze_exp6b() -> dict[str, Any]:
    rec_path = DATA_DIR / "exp6b_multimodal_prompt_expander.json"
    cands = sorted(OUT_VIDEO_DIR.glob("exp6b_multimodal_prompt_expander*-audio.mp4"))
    cands = [p for p in cands if ".1fps." not in p.name]
    out: dict[str, Any] = {}
    if cands:
        mp4 = cands[0]
        out["mp4"] = _scrub_paths(str(mp4))
        out["audio"] = _audio_rms_dbfs(mp4)
        dino_out = DATA_DIR / "exp6b_dinov3.json"
        subprocess.run(
            [sys.executable, str(REPO / "bench" / "probe_dinov3_copy_freeze.py"),
             "--video", str(mp4),
             "--ref", str(Path(os.environ["H3_INPUT_DIR"]) / STILL_PORTRAIT),
             "--json", str(dino_out)],
            check=False,
        )
        if dino_out.is_file():
            out["dinov3"] = _scrub_paths(json.loads(dino_out.read_text(encoding="utf-8")))
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        raw_out = rec.get("outputs_raw") or {}
        n75 = (raw_out.get("75") or {}).get("text") or []
        n76 = (raw_out.get("76") or {}).get("text") or []
        expanded = str(n75[0]) if n75 else ""
        report = str(n76[0]) if n76 else ""
        out["expander_report"] = report
        out["expanded_prompt_chars"] = len(expanded)
        out["expanded_prompt"] = expanded
        if expanded:
            # Grade expanded prompt against live graph `5` (`MiniMaxH3PromptOnReferences`)
            g = copy.deepcopy(rec.get("graph") or {})
            if "5" in g and "132" in g:
                g["5"]["inputs"]["prompt"] = expanded
                grade_res = pf.grade(g["5"], g, stem="exp6b_expanded")
                out["expanded_preflight"] = grade_res
    return out


def _analyze_exp7() -> dict[str, Any]:
    import numpy as np
    s1_cands = sorted(OUT_VIDEO_DIR.glob("exp7_shot1_box_office*-audio.mp4"))
    s1_cands = [p for p in s1_cands if ".1fps." not in p.name]
    if not s1_cands:
        return {"status": "missing_shot1"}
    s1_mp4 = s1_cands[0]
    s1_vid = _read_video_rgb(s1_mp4)
    out: dict[str, Any] = {"shot1": {"mp4": _scrub_paths(str(s1_mp4)), "frames": int(len(s1_vid)), "audio": _audio_rms_dbfs(s1_mp4)}, "redub": {}, "continue": {}}

    redub_mp4s: list[str] = []
    for arm in ("exp7a_redub_polish_am100", "exp7a_redub_polish_am025", "exp7a_redub_fresh_am100"):
        cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}*-audio.mp4"))
        cands = [p for p in cands if ".1fps." not in p.name]
        if not cands:
            continue
        mp4 = cands[0]
        redub_mp4s.append(str(mp4))
        vid = _read_video_rgb(mp4)
        n = min(len(vid), len(s1_vid))
        v_mae, v_psnr = _mae_psnr(vid[:n], s1_vid[:n])
        out["redub"][arm] = {
            "mp4": _scrub_paths(str(mp4)),
            "video_mae_vs_shot1": v_mae,
            "video_psnr_vs_shot1": v_psnr,
            "audio": _audio_rms_dbfs(mp4),
        }

    if redub_mp4s:
        yin_out = DATA_DIR / "exp7a_yin.json"
        subprocess.run(
            [sys.executable, str(REPO / "bench" / "probe_yin_f0_drift.py"),
             "--baseline", str(s1_mp4),
             "--steered", *redub_mp4s,
             "--json", str(yin_out)],
            check=False,
        )
        if yin_out.is_file():
            out["redub_yin_f0"] = _scrub_paths(json.loads(yin_out.read_text(encoding="utf-8")))

    for arm in ("exp7b_continue_ref_tail39_len39", "exp7b_continue_ref_tail39_len141", "exp7b_continue_latent_overlap39"):
        cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}_0*-audio.mp4")) or sorted(OUT_VIDEO_DIR.glob(f"{arm}_0*.mp4"))
        cands = [p for p in cands if ".1fps." not in p.name and "_full_overlap" not in p.name]
        if not cands:
            continue
        mp4 = cands[0]
        vid = _read_video_rgb(mp4)
        seam_mae, seam_psnr = _mae_psnr(vid[0], s1_vid[-1])
        info: dict[str, Any] = {
            "mp4": _scrub_paths(str(mp4)),
            "frames": int(len(vid)),
            "seam_f_last_to_f0_mae": seam_mae,
            "seam_f_last_to_f0_psnr": seam_psnr,
            "audio": _audio_rms_dbfs(mp4),
        }
        full_cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}_full_overlap*-audio.mp4"))
        full_cands = [p for p in full_cands if ".1fps." not in p.name]
        if full_cands:
            fvid = _read_video_rgb(full_cands[0])
            if len(fvid) > 40:
                s38_39_mae, _ = _mae_psnr(fvid[38], fvid[39])
                info["internal_f38_f39_seam_mae"] = s38_39_mae
                info["internal_f0_f38_vs_shot1_tail39_mae"], _ = _mae_psnr(fvid[:39], s1_vid[-39:])
        out["continue"][arm] = info

    return out


def _eval_portal_spec(portal_kfs: list[dict[str, Any]], frame: int) -> tuple[float, float, float, float]:
    """Pure-CPU cosine-eased portal evaluation matching `spatial_director._eval_portal_frame`."""
    kfs = sorted(portal_kfs, key=lambda item: int(item["frame"]))
    if frame <= int(kfs[0]["frame"]):
        p = kfs[0]
        return float(p["x"]), float(p["y"]), float(p["rx"]), float(p["ry"])
    if frame >= int(kfs[-1]["frame"]):
        p = kfs[-1]
        return float(p["x"]), float(p["y"]), float(p["rx"]), float(p["ry"])
    for i in range(len(kfs) - 1):
        p0, p1 = kfs[i], kfs[i + 1]
        f0, f1 = int(p0["frame"]), int(p1["frame"])
        if f0 <= frame <= f1:
            span = max(1, f1 - f0)
            c = max(0.0, min(1.0, (frame - f0) / float(span)))
            u = 0.5 - 0.5 * math.cos(math.pi * c)
            return (
                float(p0["x"]) + (float(p1["x"]) - float(p0["x"])) * u,
                float(p0["y"]) + (float(p1["y"]) - float(p0["y"])) * u,
                float(p0["rx"]) + (float(p1["rx"]) - float(p0["rx"])) * u,
                float(p0["ry"]) + (float(p1["ry"]) - float(p0["ry"])) * u,
            )
    p = kfs[-1]
    return float(p["x"]), float(p["y"]), float(p["rx"]), float(p["ry"])


def _analyze_spotlights() -> dict[str, Any]:
    import numpy as np

    all_specs = [
        *SPOTLIGHT_SPECS,
        *_expand_long_spotlight_specs(192),
        *_expand_long_spotlight_specs(243),
    ]
    out: dict[str, Any] = {"arms": {}}
    for spec in all_specs:
        arm = str(spec["arm_name"])
        clean_cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}_clean*-audio.mp4")) or sorted(OUT_VIDEO_DIR.glob(f"{arm}_clean*.mp4"))
        clean_cands = [p for p in clean_cands if ".1fps." not in p.name]
        hud_cands = sorted(OUT_VIDEO_DIR.glob(f"{arm}_hud*-audio.mp4")) or sorted(OUT_VIDEO_DIR.glob(f"{arm}_hud*.mp4"))
        hud_cands = [p for p in hud_cands if ".1fps." not in p.name]
        if not clean_cands:
            continue
        clean_mp4 = clean_cands[0]
        vid = _read_video_rgb(clean_mp4)
        n_f, h_v, w_v, _ = vid.shape
        portal_kfs = _materialize_portal(spec["portal"], n_f)
        yy, xx = np.mgrid[0:h_v, 0:w_v]
        xx_n = (xx + 0.5) / float(w_v)
        yy_n = (yy + 0.5) / float(h_v)

        def _hsv_sat_frame(frame_u8: "Any", mask: "Any") -> float | None:
            if not np.any(mask):
                return None
            f = frame_u8.astype(np.float32) / 255.0
            mx = np.max(f, axis=-1)
            mn = np.min(f, axis=-1)
            sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
            return round(float(np.mean(sat[mask])), 4)

        if n_f == 141:
            sample_frames = {"early": 8, "expand": 54, "peak": 96, "tail": 140}
        else:
            max_f = max(0, n_f - 1)
            sample_frames = {
                "early": min(8, max_f),
                "expand": int(round(0.35 * max_f)),
                "peak": int(round(0.65 * max_f)),
                "tail": max_f,
            }
        windows: dict[str, Any] = {}
        for wname, fidx in sample_frames.items():
            px, py, rx, ry = _eval_portal_spec(portal_kfs, fidx)
            norm_d2 = ((xx_n - px) / max(rx, 1e-4)) ** 2 + ((yy_n - py) / max(ry, 1e-4)) ** 2
            inside = norm_d2 <= 1.0
            outside = ~inside
            fr = vid[fidx]
            windows[wname] = {
                "frame": fidx,
                "rx": round(float(rx), 3),
                "ry": round(float(ry), 3),
                "inside_area_frac": round(float(np.mean(inside)), 4),
                "inside_sat": _hsv_sat_frame(fr, inside),
                "outside_sat": _hsv_sat_frame(fr, outside),
                "inside_luma": round(float(np.mean(fr[inside])), 2) if np.any(inside) else None,
                "outside_luma": round(float(np.mean(fr[outside])), 2) if np.any(outside) else None,
            }
        out["arms"][arm] = {
            "clean_mp4": _scrub_paths(str(clean_mp4)),
            "hud_mp4": _scrub_paths(str(hud_cands[0])) if hud_cands else None,
            "frames": int(n_f),
            "audio": _audio_rms_dbfs(clean_mp4),
            "windows": windows,
        }
    return out


# ---------------------------------------------------------------------------
# CLI Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Wave 2 GPU Bench Runner & Analyzer (Spotlights, Exp 4, Exp 6, Exp 7)")
    ap.add_argument("--stage", choices=("spotlights", "spotlights_long", "exp4", "exp6a", "exp6b", "exp7", "analyze", "all"), default="all")
    ap.add_argument("--spotlight-length", type=int, choices=(192, 243), default=192, help="Video frame count for --stage spotlights_long (default: 192)")
    ap.add_argument("--dry-run", action="store_true", help="Compile, preflight-validate, and exit with 0 HTTP mutations")
    ap.add_argument("--resume", action="store_true", help="Skip arms already recorded under data/wave2_2026-11-21/")
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        OUT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    obj_info: dict[str, Any] | None = None
    if not args.dry_run and args.stage != "analyze":
        _, obj_info = _check_live_schema(COMFY_URL)
        print("[OK] Live :8188 verified: 5-output MiniMaxH3PromptExpander + empty queue")

    client_id = f"h3_wave2_{uuid.uuid4().hex[:8]}"
    summary: dict[str, Any] = {"run_tag": RUN_TAG, "stages": {}}

    # Stage: Dynamic HUD Spotlights
    if args.stage in ("spotlights", "all"):
        spot_arms = {spec["arm_name"]: build_spotlight_arm(spec) for spec in SPOTLIGHT_SPECS}
        summary["stages"]["spotlights"] = {
            arm: _run_and_record(arm, g, client_id=client_id, dry_run=args.dry_run, resume=args.resume)
            for arm, g in spot_arms.items()
        }

    # Stage: Adult-Only Long HUD Spotlights (192f / 243f)
    if args.stage == "spotlights_long":
        long_specs = _expand_long_spotlight_specs(args.spotlight_length)
        spot_long_arms = {spec["arm_name"]: build_spotlight_arm(spec, length=args.spotlight_length) for spec in long_specs}
        summary["stages"]["spotlights_long"] = {
            arm: _run_and_record(arm, g, client_id=client_id, dry_run=args.dry_run, resume=args.resume)
            for arm, g in spot_long_arms.items()
        }

    # Stage: Exp 4
    if args.stage in ("exp4", "all"):
        inp_dir = Path(os.environ["H3_INPUT_DIR"])
        src_clip = inp_dir / CLIP_DINER
        mask_prefix = f"Video/{RUN_TAG}/exp4_sam_woman_mask"
        mask_graph = build_exp4_mask_extractor(src_clip, mask_prefix)
        frame0_ref = f"{UPLOAD_SUBFOLDER}/exp4_diner_f0.png"
        mask_mkv_path = OUT_VIDEO_DIR / "exp4_sam_woman_mask_00001.mkv"

        if args.dry_run:
            _run_and_record("exp4_sam_woman_mask", mask_graph, client_id=client_id, dry_run=True, resume=False)
        else:
            f0_png = _extract_png_frame(src_clip, 0, OUT_IMAGE_DIR / "exp4_diner_f0.png")
            frame0_ref = _http_upload_file(f0_png)
            m_res = _run_and_record("exp4_sam_woman_mask", mask_graph, client_id=client_id, dry_run=False, resume=args.resume)
            mkvs = [Path(f["resolved"]) for f in m_res.get("files", []) if str(f.get("resolved", "")).endswith(".mkv")]
            if mkvs:
                mask_mkv_path = mkvs[0]

        exp4_arms = {
            "exp4_base_unsteered": build_exp4_2x_arm(
                arm_name="exp4_base_unsteered", first_frame_ref=frame0_ref, mask_mkv_path=None, vector=None, streams="both", scale=0.0
            ),
            "exp4_2x_noir_video_only": build_exp4_2x_arm(
                arm_name="exp4_2x_noir_video_only", first_frame_ref=frame0_ref, mask_mkv_path=mask_mkv_path, vector=CV_NOIR_FL2VA, streams="video_only", scale=1.6
            ),
            "exp4_2x_noir_streams_both": build_exp4_2x_arm(
                arm_name="exp4_2x_noir_streams_both", first_frame_ref=frame0_ref, mask_mkv_path=mask_mkv_path, vector=CV_NOIR_FL2VA, streams="both", scale=1.6
            ),
            "exp4_2x_vhs_video_only": build_exp4_2x_arm(
                arm_name="exp4_2x_vhs_video_only", first_frame_ref=frame0_ref, mask_mkv_path=mask_mkv_path, vector=CV_VHS_FL2VA, streams="video_only", scale=1.6
            ),
            "exp4_1x_masked_source_noir": build_exp4_1x_masked_arm(
                arm_name="exp4_1x_masked_source_noir", src_clip_path=src_clip, mask_mkv_path=mask_mkv_path
            ),
        }
        summary["stages"]["exp4"] = {
            arm: _run_and_record(arm, g, client_id=client_id, dry_run=args.dry_run, resume=args.resume)
            for arm, g in exp4_arms.items()
        }

    # Stage: Exp 6A
    if args.stage in ("exp6a", "all"):
        exp6a_arms = {
            "exp6a_bookends_only": build_exp6a_arm(arm_name="exp6a_bookends_only", use_mid_guide=False, guide_end_percent=1.0, save_beats=True),
            "exp6a_storyboard_gep100": build_exp6a_arm(arm_name="exp6a_storyboard_gep100", use_mid_guide=True, guide_end_percent=1.0, save_beats=False),
            "exp6a_storyboard_gep033": build_exp6a_arm(arm_name="exp6a_storyboard_gep033", use_mid_guide=True, guide_end_percent=0.33, save_beats=False),
        }
        summary["stages"]["exp6a"] = {
            arm: _run_and_record(arm, g, client_id=client_id, dry_run=args.dry_run, resume=args.resume)
            for arm, g in exp6a_arms.items()
        }

    # Stage: Exp 6B
    if args.stage in ("exp6b", "all"):
        g6b = build_exp6b_multimodal_expander()
        summary["stages"]["exp6b"] = {
            "exp6b_multimodal_prompt_expander": _run_and_record(
                "exp6b_multimodal_prompt_expander", g6b, client_id=client_id, dry_run=args.dry_run, resume=args.resume
            )
        }

    # Stage: Exp 7
    if args.stage in ("exp7", "all"):
        s1_graph = build_exp7_shot1()
        s1_res = _run_and_record("exp7_shot1_box_office", s1_graph, client_id=client_id, dry_run=args.dry_run, resume=args.resume)
        shot1_ref = f"{UPLOAD_SUBFOLDER}/exp7_shot1.mp4"
        last_png_ref = f"{UPLOAD_SUBFOLDER}/exp7_shot1_f191.png"
        if not args.dry_run:
            mp4s = [Path(f["resolved"]) for f in s1_res.get("files", []) if str(f.get("resolved", "")).endswith(".mp4")]
            if not mp4s:
                raise RuntimeError("exp7_shot1_box_office produced no MP4 file")
            s1_mp4 = mp4s[0]
            shot1_ref = _http_upload_file(s1_mp4)
            f191 = _extract_png_frame(s1_mp4, 191, OUT_IMAGE_DIR / "exp7_shot1_f191.png")
            last_png_ref = _http_upload_file(f191)

        exp7_arms = {
            "exp7a_redub_polish_am100": build_exp7_redub_arm(
                arm_name="exp7a_redub_polish_am100", shot1_ref=shot1_ref, redub_mode="polish", audio_mask=1.0, prompt_text=EXP7A_POLISH_PROMPT, object_info=obj_info
            ),
            "exp7a_redub_polish_am025": build_exp7_redub_arm(
                arm_name="exp7a_redub_polish_am025", shot1_ref=shot1_ref, redub_mode="polish", audio_mask=0.25, prompt_text=EXP7A_POLISH_PROMPT, object_info=obj_info
            ),
            "exp7a_redub_fresh_am100": build_exp7_redub_arm(
                arm_name="exp7a_redub_fresh_am100", shot1_ref=shot1_ref, redub_mode="fresh", audio_mask=1.0, prompt_text=EXP7A_FRESH_PROMPT, object_info=obj_info
            ),
            "exp7b_continue_ref_tail39_len39": build_exp7_continue_arm(
                arm_name="exp7b_continue_ref_tail39_len39", shot1_ref=shot1_ref, last_frame_ref=last_png_ref, strategy="ref_tail_39f", target_length=39, object_info=obj_info
            ),
            "exp7b_continue_ref_tail39_len141": build_exp7_continue_arm(
                arm_name="exp7b_continue_ref_tail39_len141", shot1_ref=shot1_ref, last_frame_ref=last_png_ref, strategy="ref_tail_39f", target_length=141, object_info=obj_info
            ),
            "exp7b_continue_latent_overlap39": build_exp7_continue_arm(
                arm_name="exp7b_continue_latent_overlap39", shot1_ref=shot1_ref, last_frame_ref=None, strategy="latent_overlap_39f", target_length=141, object_info=obj_info
            ),
        }
        summary["stages"]["exp7"] = {"exp7_shot1_box_office": s1_res}
        for arm, g in exp7_arms.items():
            summary["stages"]["exp7"][arm] = _run_and_record(
                arm, g, client_id=client_id, dry_run=args.dry_run, resume=args.resume
            )

    if args.dry_run:
        print(json.dumps(_scrub_paths(summary), indent=2))
        return 0

    if args.stage in ("analyze", "all"):
        if OUT_VIDEO_DIR.is_dir():
            for mp4 in sorted(OUT_VIDEO_DIR.glob("*.mp4")):
                if ".1fps." not in mp4.name and "_with_mask" not in mp4.name:
                    _make_1fps_companion(mp4)
        analysis = {
            "spotlights": _analyze_spotlights(),
            "exp4": _analyze_exp4(summary),
            "exp6a": _analyze_exp6a(),
            "exp6b": _analyze_exp6b(),
            "exp7": _analyze_exp7(),
        }
        _write_json(DATA_DIR / "wave2_summary.json", analysis)
        print("[OK] Wrote scrubbed metrics to data/wave2_2026-11-21/wave2_summary.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
