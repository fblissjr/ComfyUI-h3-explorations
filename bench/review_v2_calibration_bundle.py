#!/usr/bin/env python3
"""Independent data review of a native-H3 AWQ calibration bundle.

The bundle under review is produced by `build_native_h3_calibration_batch.py`
from rows chosen by `select_v2_calibration_rows.py`. This script does not
import either one's sizing code: every geometry, label, token and hash claim in
`presentation.json` is recomputed here from the release constants and the
pinned dataset snapshot, so a builder that is wrong in the same way twice
cannot pass. Where a claim can only come from the builder (tensor digests
inside the safetensors), the check says so rather than pretending otherwise.

Arms:

1. **Media identity.** Every declared media file of every bundled row is
   re-hashed from the pinned snapshot and compared against the row's declared
   digest, the bundle's recorded digest, and the accepted pool's digest.
2. **Prompt provenance.** `prompt_sha256` must be the SHA-256 of the source
   row's `target_ir`, must not be the user request, and must carry no chat
   framing.
3. **Presentation order.** `labels_in_order` must be the request order the
   row's own `available_media_labels` declares, with a 1..n counter per type
   and media kinds that agree.
4. **Per-row geometry policy.** Reference stills recomputed under the row's
   declared still policy; keyframes against `adapt_canvas`; reference video
   against the release role policy, its 2 fps sampling and its mean-of-pair
   timestamps.
5. **Token accounting.** Sequence length, text/vision split, per-block merged
   tokens, and the packed patch-row count must agree with each other and with
   the grids.
6. **Split disjointness.** Calibration and holdout media compared by recomputed
   file digest, then by perceptual hash, because exact-hash disjointness was
   the only property the selector enforced.
7. **Distribution.** Achieved role and overlay shares against the accepted
   pool, and exact sequence lengths against the selector's estimates.
8. **Curation read.** Degenerate sources, near-blank media, empty or
   suspiciously short prompts, and prompts whose declared media count does not
   match what they describe.

CPU only. No CUDA, no model, no ComfyUI server. Run it with the ComfyUI venv
python (`docs/comfy_notes.md`): it imports the installed H3 geometry constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH))

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24

MEDIA_LABEL = re.compile(r"<(Picture|Video|Audio) (\d+)>")
KEYFRAME_DECL = re.compile(r"<Picture (\d+)> is the (first|last) frame")
CONTRACT_KEYS = ("duration_seconds", "available_media_labels")


# ---------------------------------------------------------------------------
# release geometry, reimplemented rather than imported


def adapt_canvas(width: int, height: int) -> tuple[int, int]:
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def still_geometry(width: int, height: int, policy: str) -> tuple[int, int, float]:
    ratio = REF_IMAGE_SHORT_EDGE / min(width, height)
    scale = ratio if policy == "upscale_2048" else min(1.0, ratio)
    w = max(CANVAS_MULTIPLE, round(width * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    h = max(CANVAS_MULTIPLE, round(height * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return w, h, scale


def align_frame_count(n: int) -> int:
    while n % 17 != 5:
        n += 1
    return n


def grid_floor(n: int) -> int:
    """Largest 17n+5 length at or below `n`, the direction preparation walks."""
    while n % 17 != 5:
        n -= 1
    return n


def resampled_frame_count(source_count: int, loaded_fps: float) -> int:
    """Frames after nearest-timestamp normalisation to 24 fps."""
    if math.isclose(loaded_fps, FPS, rel_tol=0.0, abs_tol=1e-6):
        return source_count
    return max(1, int(math.floor((source_count - 1) * FPS / loaded_fps + 1e-9)) + 1)


def cross_check_release_constants() -> list[str]:
    """The constants above are copies. Fail loudly if the install disagrees."""
    try:
        # The ComfyUI root, derived from this file's own location rather than
        # from a home directory, and inserted ahead of this repo: comfy_extras
        # does `import nodes`, and this repo has a top-level `nodes.py` that
        # would shadow ComfyUI's (`docs/comfy_notes.md`).
        sys.path.insert(0, str(BENCH.parents[2]))
        import comfy.cli_args  # noqa: PLC0415
        comfy.cli_args.args.cpu = True
        from comfy_extras import nodes_minimax_h3 as m  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment dependent
        return [f"could not import the installed H3 geometry to cross-check: {exc}"]
    bad = []
    for name, mine in (("CANVAS_MULTIPLE", CANVAS_MULTIPLE),
                       ("BASE_SHORT_EDGE", BASE_SHORT_EDGE),
                       ("MAX_PIXELS", MAX_PIXELS),
                       ("REF_IMAGE_SHORT_EDGE", REF_IMAGE_SHORT_EDGE),
                       ("FPS", FPS)):
        theirs = getattr(m, name, None)
        if theirs != mine:
            bad.append(f"{name}: this review uses {mine}, the install declares {theirs}")
    for w, h in ((1920, 1080), (1080, 1920), (1000, 1000), (3134, 1344), (300, 300)):
        if adapt_canvas(w, h) != m.adapt_canvas(w, h):
            bad.append(f"adapt_canvas disagrees at {w}x{h}")
    for n in (1, 5, 6, 22, 100, 121):
        if align_frame_count(n) != m.align_frame_count(n):
            bad.append(f"align_frame_count disagrees at {n}")
    return bad


# ---------------------------------------------------------------------------
# helpers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


CONTRACT_BLOCK = re.compile(
    r"^H3 target contract \(authoritative\):\n(.*?)\n\nOriginal request:", re.S)


def parse_contract(user_content: str) -> dict:
    match = CONTRACT_BLOCK.search(user_content)
    if match is None:
        return {}
    out = {}
    for line in match.group(1).splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            out[key.strip()] = value.strip()
    return out


def declared_order(contract: dict) -> list[tuple[str, int]]:
    labels = MEDIA_LABEL.findall(contract.get("available_media_labels", ""))
    return [(kind.lower(), int(index)) for kind, index in labels]


def dhash_bits(gray, size: int = 8) -> int:
    """64-bit difference hash of a PIL image already reduced to (size+1, size)."""
    px = list(gray.getdata())
    w = size + 1
    bits = 0
    for row in range(size):
        for col in range(size):
            left = px[row * w + col]
            right = px[row * w + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def perceptual_hashes(path: Path, kind: str) -> list[int]:
    """dhash of one still, or of three sampled frames of a clip."""
    from PIL import Image, ImageOps

    if kind == "image":
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
            return [dhash_bits(image.resize((9, 8), Image.Resampling.LANCZOS))]

    import av
    import numpy as np

    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        total = stream.frames or 0
        wanted = {0, max(0, total // 2), max(0, total - 1)} if total else {0, 12, 24}
        for index, frame in enumerate(container.decode(stream)):
            if index in wanted:
                frames.append(np.asarray(frame.to_rgb().to_ndarray()))
            if len(frames) >= 3 or index > max(wanted):
                break
    out = []
    for arr in frames:
        image = Image.fromarray(arr).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        out.append(dhash_bits(image))
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def blankness(path: Path, kind: str) -> dict:
    """Cheap degeneracy read: luma spread and the share of the modal bucket."""
    from PIL import Image, ImageOps

    if kind != "image":
        return {}
    with Image.open(path) as opened:
        gray = ImageOps.exif_transpose(opened).convert("L").resize(
            (64, 64), Image.Resampling.LANCZOS)
    px = list(gray.getdata())
    mean = sum(px) / len(px)
    var = sum((p - mean) ** 2 for p in px) / len(px)
    hist = Counter(p >> 3 for p in px)
    return {"std": round(var ** 0.5, 2),
            "modal_share": round(hist.most_common(1)[0][1] / len(px), 3)}


# ---------------------------------------------------------------------------
# arms


def arm_bundle_files(bundle: Path, presentation: dict) -> list[str]:
    """Every bundled tensor file present, hash-correct, and accounted for."""
    problems = []
    named = set()
    for row in presentation["rows"]:
        for key, digest_key in (("batch_file", "batch_file_sha256"),
                                ("media_file", "media_file_sha256")):
            name = row.get(key)
            if not name:
                problems.append(f"{row['row_id']}: no {key} recorded")
                continue
            named.add(name)
            path = bundle / name
            if not path.is_file():
                problems.append(f"{row['row_id']}: {key} {name} is absent")
                continue
            actual = sha256_file(path)
            if actual != row.get(digest_key):
                problems.append(
                    f"{row['row_id']}: {name} hashes {actual[:12]}, "
                    f"record declares {str(row.get(digest_key))[:12]}")
    on_disk = {p.name for p in bundle.iterdir() if p.suffix == ".safetensors"}
    for extra in sorted(on_disk - named):
        problems.append(f"{extra} is in the bundle but no row names it")
    return problems


def arm_media_identity(presentation: dict, root: Path, pool_by_id: dict) -> tuple[dict, list[str]]:
    """Re-hash every declared media file against the pinned snapshot."""
    problems = []
    digests: dict[str, str] = {}
    checked = 0
    for row in presentation["rows"]:
        pool_row = pool_by_id.get(row["row_id"])
        pool_hashes = (pool_row or {}).get("media_sha256", {})
        for item in row["ordered_media"]:
            rel = item.get("media_path")
            if rel is None:
                continue
            path = root / rel
            if not path.is_file():
                problems.append(f"{row['row_id']}: {rel} absent from the pinned snapshot")
                continue
            actual = digests.get(rel) or sha256_file(path)
            digests[rel] = actual
            checked += 1
            if item.get("declared_sha256") != actual:
                problems.append(
                    f"{row['row_id']}: {rel} declared {str(item.get('declared_sha256'))[:12]}, "
                    f"snapshot is {actual[:12]}")
            if item.get("file_sha256") != actual:
                problems.append(
                    f"{row['row_id']}: {rel} bundle recorded {str(item.get('file_sha256'))[:12]}, "
                    f"snapshot is {actual[:12]}")
            if pool_row is not None and pool_hashes.get(rel) != actual:
                problems.append(
                    f"{row['row_id']}: {rel} pool declares "
                    f"{str(pool_hashes.get(rel))[:12]}, snapshot is {actual[:12]}")
    return {"files_hashed": checked, "distinct_files": len(digests),
            "digests": digests}, problems


def arm_prompt_provenance(presentation: dict, source_by_id: dict) -> list[str]:
    problems = []
    for row in presentation["rows"]:
        src = source_by_id.get(row["row_id"])
        if src is None:
            problems.append(f"{row['row_id']}: not present in the pinned train split")
            continue
        ir = src.get("target_ir") or ""
        if not ir.strip():
            problems.append(f"{row['row_id']}: source target_ir is empty")
            continue
        if hashlib.sha256(ir.encode()).hexdigest() != row["prompt_sha256"]:
            problems.append(
                f"{row['row_id']}: presented prompt is not the row's target_ir")
        if len(ir.encode()) != row["prompt_bytes"]:
            problems.append(
                f"{row['row_id']}: prompt_bytes {row['prompt_bytes']} disagrees "
                f"with target_ir {len(ir.encode())}")
        users = [m for m in src.get("messages", []) if m["role"] == "user"]
        if users:
            request = users[0]["content"]
            if hashlib.sha256(request.encode()).hexdigest() == row["prompt_sha256"]:
                problems.append(f"{row['row_id']}: the user request was presented, not target_ir")
        if "<|im_start|>" in ir or ir.lstrip().startswith("<|"):
            problems.append(f"{row['row_id']}: target_ir carries chat framing")
    return problems


def arm_presentation_order(presentation: dict, source_by_id: dict) -> list[str]:
    problems = []
    for row in presentation["rows"]:
        src = source_by_id.get(row["row_id"])
        if src is None:
            continue
        users = [m for m in src.get("messages", []) if m["role"] == "user"]
        contract = parse_contract(users[0]["content"]) if users else {}
        for key in CONTRACT_KEYS:
            if key not in contract:
                problems.append(f"{row['row_id']}: source contract lacks {key}")
        if contract.get("available_media_labels") != \
                row["target_contract"].get("available_media_labels"):
            problems.append(f"{row['row_id']}: recorded contract labels differ from source")
        order = declared_order(contract)
        expected = [f"<{kind.capitalize()} {index}>" for kind, index in order]
        if expected != row["labels_in_order"]:
            problems.append(
                f"{row['row_id']}: labels_in_order {row['labels_in_order']} is not "
                f"the declared request order {expected}")
        for kind in ("picture", "video", "audio"):
            seen = [i for k, i in order if k == kind]
            if seen != list(range(1, len(seen) + 1)):
                problems.append(f"{row['row_id']}: {kind} labels are not a 1..n run: {seen}")
        kinds = {"picture": "image", "video": "video", "audio": "audio"}
        for (kind, _), item in zip(order, row["ordered_media"]):
            if kinds[kind] != item["type"]:
                problems.append(
                    f"{row['row_id']}: label kind {kind} carries item type {item['type']}")
        images = list(src.get("images") or [])
        videos = list(src.get("videos") or [])
        for (kind, index), item in zip(order, row["ordered_media"]):
            table = images if kind == "picture" else videos if kind == "video" else None
            if table is None:
                continue
            if index > len(table):
                problems.append(f"{row['row_id']}: <{kind} {index}> has no source file")
            elif table[index - 1] != item.get("media_path"):
                problems.append(
                    f"{row['row_id']}: {item['label']} points at {item.get('media_path')}, "
                    f"source declares {table[index - 1]}")
    return problems


def arm_geometry(presentation: dict, source_by_id: dict) -> tuple[dict, list[str]]:
    problems = []
    stats: dict = {"stills_by_policy": Counter(), "keyframes": 0, "videos": 0,
                   "upscaled_stills": 0, "identity_stills": 0}
    for row in presentation["rows"]:
        policy = row["still_policy"]
        if policy not in ("upscale_2048", "max_no_upscale"):
            problems.append(f"{row['row_id']}: unknown still policy {policy!r}")
            continue
        src = source_by_id.get(row["row_id"]) or {}
        keyframes = {int(n): where
                     for n, where in KEYFRAME_DECL.findall(src.get("target_ir") or "")}
        for item in row["ordered_media"]:
            geom = item.get("geometry") or {}
            if item["type"] == "audio":
                if item.get("role") != "reference-audio":
                    problems.append(f"{row['row_id']}: audio item role {item.get('role')!r}")
                continue
            sw, sh = item["decoded"]
            found = re.search(r"(\d+)", item["label"])
            if found is None:
                problems.append(f"{row['row_id']}: label {item['label']!r} has no ordinal")
                continue
            ordinal = int(found.group(1))
            if item["type"] == "image":
                where = keyframes.get(ordinal)
                if where is not None:
                    stats["keyframes"] += 1
                    if item["role"] != f"keyframe-{where}":
                        problems.append(
                            f"{row['row_id']} {item['label']}: role {item['role']!r} "
                            f"but target_ir declares the {where} frame")
                    want = adapt_canvas(sw, sh)
                    if tuple(geom.get("upstream", [])) != want:
                        problems.append(
                            f"{row['row_id']} {item['label']}: keyframe upstream "
                            f"{geom.get('upstream')} != adapt_canvas {list(want)}")
                    want_crop = "disabled" if where == "first" else "center"
                    if geom.get("crop") != want_crop:
                        problems.append(
                            f"{row['row_id']} {item['label']}: crop {geom.get('crop')!r} "
                            f"but the {where} frame takes {want_crop!r}")
                    continue
                stats["stills_by_policy"][policy] += 1
                if item["role"] != "reference-still":
                    problems.append(
                        f"{row['row_id']} {item['label']}: role {item['role']!r} for a "
                        f"picture target_ir does not declare as a keyframe")
                if geom.get("policy") != f"reference-still-{policy.replace('_', '-')}":
                    problems.append(
                        f"{row['row_id']} {item['label']}: geometry policy "
                        f"{geom.get('policy')!r} disagrees with the row's {policy!r}")
                if bool(geom.get("upscaling_allowed")) != (policy == "upscale_2048"):
                    problems.append(
                        f"{row['row_id']} {item['label']}: upscaling_allowed "
                        f"{geom.get('upscaling_allowed')} under {policy}")
                want_w, want_h, scale = still_geometry(sw, sh, policy)
                if tuple(geom.get("upstream", [])) != (want_w, want_h):
                    problems.append(
                        f"{row['row_id']} {item['label']}: still upstream "
                        f"{geom.get('upstream')} != recomputed [{want_w}, {want_h}] "
                        f"from {sw}x{sh} under {policy}")
                if not math.isclose(float(geom.get("scale", -1)), scale, rel_tol=1e-9):
                    problems.append(
                        f"{row['row_id']} {item['label']}: scale {geom.get('scale')} "
                        f"!= recomputed {scale}")
                if scale > 1.0:
                    stats["upscaled_stills"] += 1
                    if policy != "upscale_2048":
                        problems.append(
                            f"{row['row_id']} {item['label']}: upscaled under {policy}")
                if geom.get("resized_upstream") is False:
                    stats["identity_stills"] += 1
                continue
            # video
            stats["videos"] += 1
            if item["role"] != "reference-video":
                problems.append(f"{row['row_id']} {item['label']}: role {item['role']!r}")
            duration = float(row["target_contract"]["duration_seconds"])
            want_frames = align_frame_count(max(5, int(round(duration * FPS))))
            if geom.get("target_frame_count") != want_frames:
                problems.append(
                    f"{row['row_id']} {item['label']}: target_frame_count "
                    f"{geom.get('target_frame_count')} != {want_frames} for {duration}s")
            prepared = geom.get("prepared_frames")
            # `_prepare_reference_video` normalises to 24 fps, truncates to the
            # contract length if the clip is longer, then walks *down* to the
            # model's 17n+5 grid. A clip shorter than the contract is therefore
            # legitimately shorter than `want_frames`; what must hold is the
            # grid, the ceiling, and agreement with the decoded source.
            resampled = resampled_frame_count(int(geom.get("decoded_frames", 0)),
                                              float(geom.get("loaded_fps", FPS)))
            want_prepared = grid_floor(min(resampled, want_frames))
            if prepared != want_prepared:
                problems.append(
                    f"{row['row_id']} {item['label']}: prepared_frames {prepared} "
                    f"!= {want_prepared} recomputed from {geom.get('decoded_frames')} "
                    f"decoded at {geom.get('loaded_fps')} fps against a "
                    f"{want_frames}-frame contract")
            if prepared and prepared % 17 != 5:
                problems.append(
                    f"{row['row_id']} {item['label']}: prepared_frames {prepared} "
                    f"is not on the 17n+5 grid")
            if prepared and prepared > want_frames:
                problems.append(
                    f"{row['row_id']} {item['label']}: prepared_frames {prepared} "
                    f"exceeds the contract's {want_frames}")
            indices = geom.get("sample_indices") or []
            want_indices = list(range(0, int(prepared or 0), FPS // 2))
            if indices != want_indices:
                problems.append(
                    f"{row['row_id']} {item['label']}: sample indices are not a "
                    f"2 fps walk of {prepared} frames")
            stamps = geom.get("timestamps") or []
            want_stamps = [i / 2.0 for i in range(len(indices))]
            if stamps != want_stamps:
                problems.append(
                    f"{row['row_id']} {item['label']}: timestamps {stamps[:4]} are not "
                    f"the 2 fps sequence {want_stamps[:4]}")
            want_canvas = adapt_canvas(*geom.get("source", [0, 0])) if geom.get("source") else None
            if want_canvas and tuple(geom.get("upstream", [])) != want_canvas:
                problems.append(
                    f"{row['row_id']} {item['label']}: video upstream "
                    f"{geom.get('upstream')} != adapt_canvas {list(want_canvas)}")
            qw, qh = geom.get("qwen_view", [0, 0])
            if qw % CANVAS_MULTIPLE or qh % CANVAS_MULTIPLE:
                problems.append(
                    f"{row['row_id']} {item['label']}: qwen view {qw}x{qh} is not "
                    f"patch-aligned")
    stats["stills_by_policy"] = dict(stats["stills_by_policy"])
    return stats, problems


def arm_video_timestamp_presentation(presentation: dict) -> tuple[dict, list[str]]:
    """The scaffold must show one mean-of-pair timestamp per two-frame block.

    Qwen3-VL packs the sampled frames in pairs; an odd sample count is filled by
    repeating the last frame, so the final block's pair mean is that frame's own
    timestamp. The label is rendered `%.1f`, which is why a 2 fps walk shows
    `0.2` rather than `0.25` -- the value is the mean, the display is one
    decimal, and both have to be modelled or the check fires on correct data.
    """
    problems = []
    shown_by_row = {}
    for row in presentation["rows"]:
        videos = [i for i in row["ordered_media"] if i["type"] == "video"]
        if not videos:
            continue
        scaffold = row["presentation_scaffold"].replace("\u0120", " ")
        shown = [s for s in re.findall(r"<(\d+\.\d+)\s*seconds?>", scaffold)]
        shown_by_row[row["row_id"]] = shown
        expected = []
        for item in videos:
            stamps = list((item.get("geometry") or {}).get("timestamps") or [])
            if len(stamps) % 2:
                stamps.append(stamps[-1])
            expected.extend(f"{(stamps[i] + stamps[i + 1]) / 2.0:.1f}"
                            for i in range(0, len(stamps), 2))
        blocks = [b for b in row["vision_blocks"] if b["kind"] == "video_block"]
        if len(expected) != len(blocks):
            problems.append(
                f"{row['row_id']}: {len(expected)} mean-of-pair timestamps for "
                f"{len(blocks)} video blocks")
        if shown != expected:
            problems.append(
                f"{row['row_id']}: scaffold shows {shown} where the mean-of-pair "
                f"sequence is {expected}")
    return shown_by_row, problems


def arm_token_accounting(presentation: dict) -> tuple[dict, list[str]]:
    problems = []
    lengths = {}
    for row in presentation["rows"]:
        length = row["sequence_length"]
        lengths[row["row_id"]] = length
        if row["text_positions"] + row["vision_positions"] != length:
            problems.append(
                f"{row['row_id']}: text+vision {row['text_positions']}+"
                f"{row['vision_positions']} != sequence_length {length}")
        ids = row["batch_tensors"]["input_ids"]
        if ids["shape"] != [1, length]:
            problems.append(f"{row['row_id']}: input_ids shape {ids['shape']} != [1, {length}]")
        for key in ("attention_mask", "mm_token_type_ids"):
            if row["batch_tensors"][key]["shape"] != [1, length]:
                problems.append(
                    f"{row['row_id']}: {key} shape "
                    f"{row['batch_tensors'][key]['shape']} != [1, {length}]")
        merged = sum(b["merged_tokens"] for b in row["vision_blocks"])
        # `token_tags_from_embeds_info` tags each block's `<|vision_start|>` and
        # `<|vision_end|>` as vision alongside its pads, so a block occupies
        # `merged_tokens + 2` tagged positions. Asserting the exact identity
        # rather than an inequality is what makes a dropped or duplicated
        # boundary token visible.
        if merged + 2 * len(row["vision_blocks"]) != row["vision_positions"]:
            problems.append(
                f"{row['row_id']}: vision blocks sum to {merged} over "
                f"{len(row['vision_blocks'])} blocks, which needs "
                f"{merged + 2 * len(row['vision_blocks'])} tagged positions; "
                f"vision_positions is {row['vision_positions']}")
        sizes = [e["size"] for e in row["embeds_info"]]
        if sizes != [b["merged_tokens"] for b in row["vision_blocks"]]:
            problems.append(f"{row['row_id']}: embeds_info sizes disagree with vision blocks")
        patch_rows = 0
        for block in row["vision_blocks"]:
            (t, h, w), = block["grid_thw"]
            if h * w // 4 != block["merged_tokens"]:
                problems.append(
                    f"{row['row_id']}: grid {t}x{h}x{w} gives {h * w // 4} merged tokens, "
                    f"block declares {block['merged_tokens']}")
            patch_rows += t * h * w
            if block["deepstack_features"] != row["deepstack_feature_count"]:
                problems.append(f"{row['row_id']}: block DeepStack count disagrees with the row")
        pixel = row["batch_tensors"].get("pixel_values")
        if pixel and pixel["shape"][0] != patch_rows:
            problems.append(
                f"{row['row_id']}: pixel_values has {pixel['shape'][0]} rows, "
                f"grids require {patch_rows}")
        grids = row["batch_tensors"].get("image_grid_thw")
        if grids and grids["shape"] != [len(row["vision_blocks"]), 3]:
            problems.append(
                f"{row['row_id']}: image_grid_thw shape {grids['shape']} for "
                f"{len(row['vision_blocks'])} blocks")
        spans = row["vision_spans"]
        if len(spans) != len(row["vision_blocks"]):
            problems.append(
                f"{row['row_id']}: {len(spans)} vision spans for "
                f"{len(row['vision_blocks'])} blocks")
        covered = sum(b - a + 1 for a, b in spans)
        if covered != row["vision_positions"]:
            problems.append(
                f"{row['row_id']}: vision spans cover {covered}, "
                f"vision_positions is {row['vision_positions']}")
        if row["marker_ids_present"] and not row["marker_positions"]:
            problems.append(f"{row['row_id']}: marker ids present with no positions")
    return lengths, problems


def foreground_correlation(a, b) -> float:
    """Correlation over the non-background pixels of two 96x96 reductions.

    A large share of this dataset is three-view turnaround sheets on white. A
    plain dhash matches that *layout*, so an unmasked comparison agrees with it
    and the two metrics stop being independent. Dropping near-white pixels is
    what makes the second metric able to disagree -- and it does: four of the
    first six dhash hits fell to it.
    """
    import numpy as np

    mean_a, mean_b = a.mean(axis=2), b.mean(axis=2)
    keep = (mean_a < 235) | (mean_b < 235)
    if int(keep.sum()) < 200:
        return 0.0
    x, y = a[keep].ravel(), b[keep].ravel()
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return float((x * y).mean())


def reduce_image(path: Path):
    from PIL import Image, ImageOps
    import numpy as np

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        return np.asarray(image.resize((96, 96), Image.Resampling.LANCZOS),
                          dtype=np.float64)


def null_hamming_rate(pool: list[dict], root: Path, threshold: int,
                      sample: int, seed: int) -> dict:
    """What the dhash threshold costs on media that share no component.

    A threshold nobody priced is a threshold nobody can read. This draws one
    image from each of `sample` distinct exact-media components and reports how
    often unrelated pairs land inside the threshold anyway.
    """
    import random

    by_component: dict[str, list[str]] = {}
    for row in pool:
        images = [i for i in (row.get("images") or [])]
        if images:
            by_component.setdefault(row["media_component"], []).extend(images)
    rng = random.Random(seed)
    components = sorted(by_component)
    picked = rng.sample(components, min(sample, len(components)))
    bits = []
    for component in picked:
        rel = rng.choice(by_component[component])
        try:
            bits.append(perceptual_hashes(root / rel, "image")[0])
        except Exception:
            continue
    pairs = inside = 0
    for i in range(len(bits)):
        for j in range(i + 1, len(bits)):
            pairs += 1
            if hamming(bits[i], bits[j]) <= threshold:
                inside += 1
    return {"components_sampled": len(bits), "pairs": pairs, "inside": inside,
            "rate": round(inside / pairs, 6) if pairs else None,
            "threshold": threshold}


def arm_split_disjointness(cal_digests: dict, holdouts: dict, root: Path,
                           near_threshold: int, corr_threshold: float,
                           adjudication: dict) -> tuple[dict, list[str]]:
    """Exact-hash disjointness, then an adjudicated perceptual candidate list.

    The exact arm is a plain assertion. The perceptual arm cannot be one: a
    dhash threshold tight enough to miss nothing flags turnaround sheets that
    share only a template, and a red that fires on correct data is worse than
    no check. So it emits ranked candidates and grades them against a recorded
    adjudication. A candidate nobody has ruled on is red as *unreviewed*; one
    ruled `duplicate` is red as a defect; one ruled `distinct` passes and stays
    in the record with its reason.
    """
    problems = []
    hold_digests: dict[str, str] = {}
    for digests in holdouts.values():
        hold_digests.update(digests)
    for rel in sorted(set(cal_digests) & set(hold_digests)):
        problems.append(f"calibration and holdout share media file {rel}")
    cal_by_digest = defaultdict(list)
    for rel, digest in cal_digests.items():
        cal_by_digest[digest].append(rel)
    for rel, digest in hold_digests.items():
        if digest in cal_by_digest and rel not in cal_by_digest[digest]:
            problems.append(
                f"holdout {rel} is byte-identical to calibration "
                f"{cal_by_digest[digest][0]} under a different path")

    def kind_of(rel: str) -> str:
        return "video" if "/videos/" in rel else "image"

    hashes: dict[str, dict[str, list[int]]] = {"calibration": {}, "holdout": {}}
    failed = []
    for side, table in (("calibration", cal_digests), ("holdout", hold_digests)):
        for rel in sorted(table):
            try:
                hashes[side][rel] = perceptual_hashes(root / rel, kind_of(rel))
            except Exception as exc:
                failed.append(f"{side} {rel}: {exc}")

    reductions: dict[str, object] = {}
    candidates = []
    for rel_a, bits_a in hashes["calibration"].items():
        for rel_b, bits_b in hashes["holdout"].items():
            distance = min(hamming(x, y) for x in bits_a for y in bits_b)
            if distance > near_threshold:
                continue
            correlation = None
            if kind_of(rel_a) == "image" and kind_of(rel_b) == "image":
                for rel in (rel_a, rel_b):
                    if rel not in reductions:
                        reductions[rel] = reduce_image(root / rel)
                correlation = round(
                    foreground_correlation(reductions[rel_a], reductions[rel_b]), 4)
            key = f"{rel_a}|{rel_b}"
            ruling = adjudication.get(key, {})
            candidates.append({
                "calibration": rel_a, "holdout": rel_b, "hamming": distance,
                "foreground_correlation": correlation,
                "verdict": ruling.get("verdict", "unreviewed"),
                "reason": ruling.get("reason"),
            })
    candidates.sort(key=lambda c: (-(c["foreground_correlation"] or -1), c["hamming"]))
    for hit in candidates:
        if hit["verdict"] == "duplicate":
            problems.append(
                f"near-duplicate across the split: {hit['calibration']} vs "
                f"{hit['holdout']} (Hamming {hit['hamming']}, foreground "
                f"correlation {hit['foreground_correlation']}): {hit['reason']}")
        elif hit["verdict"] != "distinct":
            strength = ("strong" if (hit["foreground_correlation"] or 0) >= corr_threshold
                        else "weak")
            problems.append(
                f"unreviewed {strength} candidate: {hit['calibration']} vs "
                f"{hit['holdout']} (Hamming {hit['hamming']}, foreground "
                f"correlation {hit['foreground_correlation']})")
    stale = sorted(set(adjudication) - {f"{c['calibration']}|{c['holdout']}"
                                        for c in candidates})
    for key in stale:
        problems.append(f"adjudication names a pair the scan no longer produces: {key}")
    return {"calibration_files": len(cal_digests), "holdout_files": len(hold_digests),
            "unhashable": failed, "candidates": candidates,
            "hamming_threshold": near_threshold,
            "correlation_threshold": corr_threshold}, problems


def arm_within_side_duplicates(digests: dict, root: Path, near_threshold: int) -> list[dict]:
    """Near-duplicates inside the calibration set itself: redundant mass."""
    def kind_of(rel: str) -> str:
        return "video" if "/videos/" in rel else "image"

    entries = []
    for rel in sorted(digests):
        try:
            for bits in perceptual_hashes(root / rel, kind_of(rel)):
                entries.append((rel, bits))
        except Exception:
            continue
    hits = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][0] == entries[j][0]:
                continue
            distance = hamming(entries[i][1], entries[j][1])
            if distance <= near_threshold:
                hits.append({"a": entries[i][0], "b": entries[j][0], "distance": distance})
    return hits


def arm_plan_requirements(presentation: dict, holdouts: dict, pool: list[dict],
                          root: Path) -> tuple[dict, list[str]]:
    """The locked split decisions in `active_plan.md`, which nothing asserted.

    Two of them are stated as musts and had no control: at least two
    small-source components reserved for holdout, and no prompt overlap across
    the split. The first is a plain assertion. The second reports its ranking
    rather than fixing a threshold, because prompt similarity within one
    dataset genre is high by construction and a cutoff would fire on correct
    data; a strong image finding and a top-ranked prompt pair pointing at the
    same two rows is the signal worth reading.
    """
    problems = []
    pool_by_id = {r["id"]: r for r in pool}
    cal_ids = [r["row_id"] for r in presentation["rows"]]
    hold_ids = sorted({r["row_id"] for h in holdouts.values() for r in h["rows"]})

    small = [r["id"] for r in pool if r.get("overlays", {}).get("small_source")]
    small_components = {pool_by_id[i]["media_component"] for i in small}
    held = {pool_by_id[i]["media_component"] for i in small if i in hold_ids}
    if len(held) < 2:
        problems.append(
            f"active_plan.md reserves at least two small-source components for "
            f"holdout; the holdout carries {len(held)} of the pool's "
            f"{len(small_components)}")

    source_by_id = {}
    for line in (root / "data" / "train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] in cal_ids or row["id"] in hold_ids:
            source_by_id[row["id"]] = row

    def shingles(text: str, k: int = 8) -> set:
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {tuple(words[i:i + k]) for i in range(max(0, len(words) - k + 1))}

    cal_shingles = {i: shingles((source_by_id.get(i) or {}).get("target_ir") or "")
                    for i in cal_ids}
    hold_shingles = {i: shingles((source_by_id.get(i) or {}).get("target_ir") or "")
                     for i in hold_ids}
    ranked = []
    for a in cal_ids:
        for b in hold_ids:
            shared = len(cal_shingles[a] & hold_shingles[b])
            if not shared:
                continue
            union = len(cal_shingles[a] | hold_shingles[b])
            ranked.append({"calibration": a, "holdout": b, "shared_8grams": shared,
                           "jaccard": round(shared / union, 5)})
            if cal_shingles[a] and cal_shingles[a] == hold_shingles[b]:
                problems.append(f"identical target_ir across the split: {a} and {b}")
    ranked.sort(key=lambda r: -r["jaccard"])
    return {"small_source_components_in_pool": len(small_components),
            "small_source_components_in_holdout": len(held),
            "small_source_rows_in_calibration":
                [i for i in small if i in cal_ids],
            "holdout_rows": len(hold_ids),
            "prompt_overlap_top": ranked[:10]}, problems


def arm_distribution(presentation: dict, pool: list[dict], selections: dict,
                     lengths: dict) -> tuple[dict, list[str]]:
    notes = []
    pool_roles = Counter(r["primary_role"] for r in pool)
    bundle_roles = Counter(r["primary_role"] for r in presentation["rows"])
    total = len(presentation["rows"])
    pool_total = len(pool)
    share = {}
    for role in sorted(set(pool_roles) | set(bundle_roles)):
        share[role] = {
            "pool_rows": pool_roles.get(role, 0),
            "pool_share": round(pool_roles.get(role, 0) / pool_total, 4),
            "bundle_rows": bundle_roles.get(role, 0),
            "bundle_share": round(bundle_roles.get(role, 0) / total, 4),
        }
    pool_by_id = {r["id"]: r for r in pool}
    overlays = Counter()
    for row in presentation["rows"]:
        ov = (pool_by_id.get(row["row_id"]) or {}).get("overlays", {})
        if ov.get("wide_or_tall"):
            overlays["wide_or_tall"] += 1
        if ov.get("small_source"):
            overlays["small_source"] += 1
        if ov.get("markers"):
            overlays["dialogue_markers"] += 1
        if ov.get("audio_label"):
            overlays["audio_label"] += 1
    pool_overlays = Counter()
    for row in pool:
        ov = row.get("overlays", {})
        if ov.get("wide_or_tall"):
            pool_overlays["wide_or_tall"] += 1
        if ov.get("small_source"):
            pool_overlays["small_source"] += 1
        if ov.get("markers"):
            pool_overlays["dialogue_markers"] += 1
        if ov.get("audio_label"):
            pool_overlays["audio_label"] += 1

    est = {}
    for name, sel in selections.items():
        for entry in sel["calibration"]:
            est[entry["id"]] = entry["tokens_est"]
    deltas = []
    for row_id, actual in lengths.items():
        if row_id in est:
            deltas.append({"row": row_id, "estimate": est[row_id], "actual": actual,
                           "delta": actual - est[row_id]})
    off = [d for d in deltas if abs(d["delta"]) > max(64, 0.05 * d["estimate"])]
    for d in sorted(off, key=lambda d: -abs(d["delta"]))[:10]:
        notes.append(
            f"{d['row']}: exact length {d['actual']} against estimate "
            f"{d['estimate']} ({d['delta']:+d})")
    return {"roles": share,
            "overlays": {k: {"pool": pool_overlays.get(k, 0),
                             "bundle": overlays.get(k, 0)}
                         for k in ("dialogue_markers", "wide_or_tall",
                                   "audio_label", "small_source")},
            "sequence_length_total": sum(lengths.values()),
            "sequence_length_max": max(lengths.values()),
            "estimate_deltas": {"rows_compared": len(deltas),
                                "beyond_tolerance": len(off),
                                "largest": sorted(deltas, key=lambda d: -abs(d["delta"]))[:5]},
            }, notes


def arm_curation(presentation: dict, root: Path, source_by_id: dict,
                 digests: dict) -> tuple[dict, list[str]]:
    findings = []
    degenerate = []
    blank = []
    for row in presentation["rows"]:
        src = source_by_id.get(row["row_id"]) or {}
        ir = src.get("target_ir") or ""
        for item in row["ordered_media"]:
            rel = item.get("media_path")
            if rel is None:
                continue
            sw, sh = item["decoded"]
            geom = item.get("geometry") or {}
            up = geom.get("upstream") or [0, 0]
            area_gain = (up[0] * up[1]) / max(1, sw * sh)
            if item["role"] == "reference-still" and area_gain > 4.0:
                degenerate.append({
                    "row": row["row_id"], "label": item["label"], "source": [sw, sh],
                    "upstream": up, "area_gain": round(area_gain, 2),
                    "policy": row["still_policy"], "path": rel})
            if "/videos/" not in rel:
                stat = blankness(root / rel, "image")
                if stat and (stat["std"] < 12.0 or stat["modal_share"] > 0.6):
                    blank.append({"row": row["row_id"], "label": item["label"],
                                  "path": rel, **stat})
        labels = row["labels_in_order"]
        pictures = sum(1 for label in labels if label.startswith("<Picture"))
        mentioned = {int(n) for _, n in MEDIA_LABEL.findall(ir) if _ == "Picture"}
        missing = sorted(set(range(1, pictures + 1)) - mentioned)
        if missing:
            findings.append(
                f"{row['row_id']}: target_ir never mentions "
                f"{', '.join(f'<Picture {i}>' for i in missing)}")
        ascii_share = sum(1 for c in ir if ord(c) < 128) / max(1, len(ir))
        if ascii_share < 0.9:
            findings.append(
                f"{row['row_id']}: target_ir is {round(1 - ascii_share, 3)} non-ASCII "
                f"by character; check the language")
        required = row["target_contract"].get("required_sections", "")
        for section in [s.strip() for s in required.split(",") if s.strip()]:
            if section not in ir:
                findings.append(
                    f"{row['row_id']}: contract requires section {section!r}, "
                    f"target_ir does not name it")
        if row["prompt_bytes"] < 500:
            findings.append(
                f"{row['row_id']}: target_ir is only {row['prompt_bytes']} bytes")
    for entry in degenerate:
        findings.append(
            f"{entry['row']} {entry['label']}: {entry['source'][0]}x{entry['source'][1]} "
            f"source interpolated to {entry['upstream'][0]}x{entry['upstream'][1]} "
            f"({entry['area_gain']}x area) under {entry['policy']}")
    for entry in blank:
        findings.append(
            f"{entry['row']} {entry['label']}: low-detail source "
            f"(luma std {entry['std']}, modal bucket {entry['modal_share']}) {entry['path']}")
    return {"interpolated_stills": degenerate, "low_detail_media": blank,
            "distinct_media_files": len(digests)}, findings


# ---------------------------------------------------------------------------


# mutation -> (the arm that must gain a problem, what the defect stands for)
MUTATIONS = {
    "declared-hash": ("media_identity",
                      "a row declares a media digest the snapshot does not produce"),
    "bundle-hash": ("bundle_files",
                    "a recorded batch_file digest no longer matches the file"),
    "user-request-prompt": ("prompt_provenance",
                            "the presented prompt is not the row's target_ir"),
    "reordered-labels": ("presentation_order",
                         "labels_in_order no longer follows the request order"),
    "relabelled-media": ("presentation_order",
                         "a label points at another item's media file"),
    "still-geometry": ("geometry",
                       "a reference still records a size its policy does not produce"),
    "policy-swap": ("geometry",
                    "a row's still policy disagrees with its items' geometry"),
    "keyframe-crop": ("geometry",
                      "a last-frame keyframe records the first-frame crop"),
    "timestamp-shift": ("video_timestamps",
                        "the scaffold shows timestamps that are not the pair means"),
    "sequence-length": ("token_accounting",
                        "sequence_length no longer matches the batch tensors"),
    "split-overlap": ("split",
                      "a calibration row points at a holdout media file"),
}


def _mutate(presentation: dict, kind: str, holdout_media: str | None) -> None:
    rows = presentation["rows"]

    def first(predicate):
        for row in rows:
            for item in row["ordered_media"]:
                if predicate(row, item):
                    return row, item
        raise LookupError(f"no row satisfies {kind}")

    if kind == "declared-hash":
        row, item = first(lambda r, i: i.get("declared_sha256"))
        item["declared_sha256"] = "0" * 64
    elif kind == "bundle-hash":
        rows[0]["batch_file_sha256"] = "0" * 64
    elif kind == "user-request-prompt":
        rows[0]["prompt_sha256"] = "0" * 64
    elif kind == "reordered-labels":
        row = next(r for r in rows if len(r["labels_in_order"]) > 1)
        row["labels_in_order"] = list(reversed(row["labels_in_order"]))
    elif kind == "relabelled-media":
        row = next(r for r in rows if len(r["ordered_media"]) > 1)
        a, b = row["ordered_media"][0], row["ordered_media"][1]
        a["media_path"], b["media_path"] = b["media_path"], a["media_path"]
    elif kind == "still-geometry":
        row, item = first(lambda r, i: i.get("role") == "reference-still")
        item["geometry"]["upstream"] = [
            item["geometry"]["upstream"][0] + 32, item["geometry"]["upstream"][1]]
    elif kind == "policy-swap":
        row = next(r for r in rows if r["still_policy"] == "upscale_2048")
        row["still_policy"] = "max_no_upscale"
    elif kind == "keyframe-crop":
        row, item = first(lambda r, i: (i.get("role") or "").startswith("keyframe-last"))
        item["geometry"]["crop"] = "disabled"
    elif kind == "timestamp-shift":
        row = next(r for r in rows
                   if any(i["type"] == "video" for i in r["ordered_media"]))
        # The raw scaffold separates words with U+0120, not a space. Writing the
        # mutation against the decoded form made it a silent no-op and the arm
        # looked green; mutate the bytes the record actually holds.
        scaffold = re.sub(r"<\d+\.\d+(\s|\u0120)seconds>",
                          "<0.0\u0120seconds>", row["presentation_scaffold"])
        if scaffold == row["presentation_scaffold"]:
            raise LookupError("no timestamp text found to shift")
        row["presentation_scaffold"] = scaffold
    elif kind == "sequence-length":
        rows[0]["sequence_length"] += 1
    elif kind == "split-overlap":
        if holdout_media is None:
            raise LookupError("split-overlap needs a holdout to borrow a file from")
        row, item = first(lambda r, i: i.get("media_path"))
        item["media_path"] = holdout_media
        item["declared_sha256"] = item["file_sha256"] = None
    else:
        raise ValueError(kind)


def violation_arm(bundle: Path, holdouts: list[Path], pool_path: Path,
                  adjudication: Path) -> list[str]:
    """Every arm, shown failing. A gate nobody has watched fail is not a gate.

    Each mutation is applied to a scratch copy of `presentation.json` whose
    tensor files are symlinks to the real ones, so the arms run against real
    media and real digests and only the named field is wrong.

    **The mutation must move the arm it targets, not merely the exit code.**
    The first version of this control compared exit codes, and the live bundle
    already carries two reds -- so every mutation "passed" without the review
    having noticed any of them. That is the defect class this whole file exists
    to catch, reproduced inside the file. So the baseline is run first, its
    problems recorded per arm, and a mutation counts as caught only when the
    named arm gains a problem the baseline did not have.
    """
    import shutil
    import subprocess
    import tempfile

    def review(bundle_path: Path) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            out = Path(handle.name)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--bundle", str(bundle_path), "--pool", str(pool_path),
                   "--adjudication", str(adjudication), "--skip-perceptual",
                   "--out", str(out)]
        for path in holdouts:
            command += ["--holdout", str(path)]
        subprocess.run(command, capture_output=True, text=True)
        report = json.loads(out.read_text())
        out.unlink(missing_ok=True)
        return report["problems"]

    baseline = review(bundle)
    failures = []
    holdout_media = None
    if holdouts:
        hold = load_presentation(holdouts[0])
        for row in hold["rows"]:
            for item in row["ordered_media"]:
                if item.get("media_path"):
                    holdout_media = item["media_path"]
                    break
            if holdout_media:
                break

    original = json.loads((bundle / "presentation.json").read_text())
    for kind, (arm, description) in MUTATIONS.items():
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp) / "bundle"
            scratch.mkdir()
            for path in bundle.iterdir():
                if path.suffix == ".safetensors":
                    (scratch / path.name).symlink_to(path)
            mutated = json.loads(json.dumps(original))
            try:
                _mutate(mutated, kind, holdout_media)
            except LookupError as exc:
                failures.append(f"{kind}: could not be applied ({exc})")
                continue
            (scratch / "presentation.json").write_text(json.dumps(mutated))
            problems = review(scratch)
            gained = [p for p in problems.get(arm, [])
                      if p not in baseline.get(arm, [])]
            if not gained:
                failures.append(
                    f"{kind}: {description} -- the {arm} arm gained no problem "
                    f"(baseline had {len(baseline.get(arm, []))})")
            shutil.rmtree(scratch, ignore_errors=True)
    return failures


def load_presentation(path: Path) -> dict:
    return json.loads((path / "presentation.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path,
                        help="calibration bundle directory holding presentation.json")
    parser.add_argument("--holdout", action="append", default=[], type=Path,
                        help="holdout bundle directory; repeatable")
    parser.add_argument("--selection", action="append", default=[], type=Path,
                        help="selector output JSON; repeatable")
    parser.add_argument("--pool", type=Path,
                        default=BENCH / "results" / "2026-08-24_h3_calibration_pool.jsonl")
    parser.add_argument("--near-threshold", type=int, default=12,
                        help="Hamming distance on a 64-bit dhash that makes a pair a "
                             "candidate for review; not by itself a duplicate verdict")
    parser.add_argument("--correlation-threshold", type=float, default=0.6,
                        help="foreground correlation above which an unreviewed "
                             "candidate is reported as strong")
    parser.add_argument("--adjudication", type=Path,
                        default=BENCH / "results" / "2026-08-25_v2_split_near_duplicate_adjudication.json",
                        help="recorded verdicts for cross-split perceptual candidates")
    parser.add_argument("--null-sample", type=int, default=220,
                        help="components sampled to price the Hamming threshold")
    parser.add_argument("--out", type=Path, help="write the JSON record here")
    parser.add_argument("--violation-arm", action="store_true",
                        help="run every mutation control instead of the live review")
    parser.add_argument("--skip-perceptual", action="store_true",
                        help="skip the near-duplicate arms (they decode every file)")
    args = parser.parse_args()

    from build_h3_calibration_pool import pinned_snapshot

    if args.violation_arm:
        failures = violation_arm(args.bundle, args.holdout, args.pool,
                                 args.adjudication)
        for kind, (arm, description) in MUTATIONS.items():
            bad = [f for f in failures if f.startswith(f"{kind}:")]
            print(f"[{'ESCAPED' if bad else 'caught'}] {arm} <- {kind}: {description}")
            for item in bad:
                print(f"    - {item}")
        print(f"\nmutations that escaped: {len(failures)} of {len(MUTATIONS)}")
        return 1 if failures else 0

    root, revision = pinned_snapshot()
    report: dict = {"revision": revision, "arms": {}, "problems": {}, "notes": {}}

    constant_problems = cross_check_release_constants()
    report["problems"]["release_constants"] = constant_problems

    presentation = load_presentation(args.bundle)
    pool = [json.loads(line) for line in args.pool.read_text().splitlines()]
    pool_by_id = {r["id"]: r for r in pool}
    train = root / "data" / "train.jsonl"
    source_by_id = {}
    for line in train.read_text().splitlines():
        row = json.loads(line)
        source_by_id[row["id"]] = row

    report["arms"]["bundle_rows"] = len(presentation["rows"])
    report["problems"]["bundle_files"] = arm_bundle_files(args.bundle, presentation)

    media, problems = arm_media_identity(presentation, root, pool_by_id)
    cal_digests = media.pop("digests")
    report["arms"]["media_identity"] = media
    report["problems"]["media_identity"] = problems

    report["problems"]["prompt_provenance"] = arm_prompt_provenance(presentation, source_by_id)
    report["problems"]["presentation_order"] = arm_presentation_order(presentation, source_by_id)

    geometry, problems = arm_geometry(presentation, source_by_id)
    report["arms"]["geometry"] = geometry
    report["problems"]["geometry"] = problems
    shown, problems = arm_video_timestamp_presentation(presentation)
    report["arms"]["video_timestamps"] = shown
    report["problems"]["video_timestamps"] = problems

    lengths, problems = arm_token_accounting(presentation)
    report["problems"]["token_accounting"] = problems

    holdout_digests = {}
    for path in args.holdout:
        hold = load_presentation(path)
        table = {}
        for row in hold["rows"]:
            for item in row["ordered_media"]:
                rel = item.get("media_path")
                if rel and (root / rel).is_file():
                    table[rel] = sha256_file(root / rel)
        holdout_digests[path.name] = table
        report["arms"].setdefault("holdout_bundles", {})[path.name] = {
            "rows": len(hold["rows"]),
            "policies": dict(Counter(r["still_policy"] for r in hold["rows"])),
            "roles": dict(Counter(r["primary_role"] for r in hold["rows"])),
            "media_files": len(table),
        }

    if args.skip_perceptual:
        shared = sorted(set(cal_digests) & set().union(*map(set, holdout_digests.values()))) \
            if holdout_digests else []
        report["arms"]["split"] = {"exact_shared_paths": shared,
                                   "perceptual": "skipped"}
        report["problems"]["split"] = [f"calibration and holdout share {r}" for r in shared]
    else:
        adjudication = {}
        if args.adjudication and args.adjudication.is_file():
            loaded = json.loads(args.adjudication.read_text())
            adjudication = {f"{e['calibration']}|{e['holdout']}": e
                            for e in loaded["pairs"]}
            report["arms"]["adjudication_source"] = loaded.get("recorded")
        split, problems = arm_split_disjointness(
            cal_digests, holdout_digests, root, args.near_threshold,
            args.correlation_threshold, adjudication)
        split["null_control"] = null_hamming_rate(
            pool, root, args.near_threshold, args.null_sample, 7)
        report["arms"]["split"] = split
        report["problems"]["split"] = problems
        report["arms"]["calibration_internal_near_duplicates"] = \
            arm_within_side_duplicates(cal_digests, root, args.near_threshold)

    holdout_presentations = {p.name: load_presentation(p) for p in args.holdout}
    if holdout_presentations:
        requirements, problems = arm_plan_requirements(
            presentation, holdout_presentations, pool, root)
        report["arms"]["plan_requirements"] = requirements
        report["problems"]["plan_requirements"] = problems

    selections = {p.name: json.loads(p.read_text()) for p in args.selection}
    distribution, notes = arm_distribution(presentation, pool, selections, lengths)
    report["arms"]["distribution"] = distribution
    report["notes"]["distribution"] = notes

    curation, findings = arm_curation(presentation, root, source_by_id, cal_digests)
    report["arms"]["curation"] = curation
    report["notes"]["curation"] = findings

    blocking = sum(len(v) for v in report["problems"].values())
    report["blocking_problem_count"] = blocking
    report["advisory_note_count"] = sum(len(v) for v in report["notes"].values())

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    for arm, items in sorted(report["problems"].items()):
        status = "GREEN" if not items else f"RED ({len(items)})"
        print(f"[{status}] {arm}")
        for item in items[:20]:
            print(f"    - {item}")
        if len(items) > 20:
            print(f"    ... {len(items) - 20} more")
    for arm, items in sorted(report["notes"].items()):
        print(f"[note] {arm}: {len(items)}")
        for item in items[:25]:
            print(f"    - {item}")
        if len(items) > 25:
            print(f"    ... {len(items) - 25} more")
    print(f"\nblocking problems: {blocking}")
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
