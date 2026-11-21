"""Shared pure-stdlib H3 prompt contract for all generation modes and multimodal references.

Owns the deterministic grammar that `auto_edit_prompt.py`, CLI tools, and tests share so
`heylook` only authors creative prose while Python enforces exact vendor guide rules
(`vendor_guides/base_en.md`, `vendor_guides/ref_en.md`, and `bench/preflight_graph.py`):

- Prompt modes: `t2va`, `i2va`, `fl2va`, `l2va`, `ref2va`, `single_frame` (plus `"auto"`).
- Exact Part One alignment stamping and refreshing from `(mode, length, [Shot N])`.
- Multimodal `H3References` inspection via `reference_order.assign_labels` (`<Picture N>`,
  `<Video N>`, shared-counter `<Audio N>` with `<Audio j>` before its `<Video k>`).
- Mechanical validation mirroring `bench/preflight_graph.py` (`header_problems`,
  `marker_rules`, `speaker_id_rules`, retention markers, tag coverage, and Part One match).

Pure stdlib + `h3_rules` + `reference_order`: zero torch or ComfyUI imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Sequence

try:
    from . import h3_rules as _rules
    from .reference_order import AudioRef, ImageRef, VideoRef, assign_labels
except ImportError:  # pragma: no cover - direct repo-root import
    import h3_rules as _rules  # type: ignore[no-redef]
    from reference_order import AudioRef, ImageRef, VideoRef, assign_labels  # type: ignore[no-redef]


PROMPT_MODES: tuple[str, ...] = (
    "auto",
    "ref2va",
    "t2va",
    "i2va",
    "fl2va",
    "l2va",
    "single_frame",
)

BASE_MODES: frozenset[str] = frozenset({"t2va", "i2va", "fl2va", "l2va"})
KEYFRAME_BASE_MODES: frozenset[str] = frozenset({"i2va", "fl2va", "l2va"})

REF2VA_SECTIONS: tuple[str, ...] = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
BASE_SECTIONS: tuple[str, ...] = (
    "integrated_multimodal_description",
    "overall_soundscape",
    "non_diegetic_music",
)

LEGAL_TASK_TAGS: frozenset[str] = frozenset({
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
})
VISUAL_RETENTION_MARKERS: frozenset[str] = frozenset({
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
})
AUDIO_RETENTION_MARKERS: frozenset[str] = frozenset({
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
})

I2VA_PART_ONE = (
    "For the target video, at 0.00 seconds into the target video, "
    "<Picture 1> (from [Shot 1]) is fully referenced."
)
_I2VA_PART_ONE_RE = re.compile(
    r"^For the target video,\s*at\s+0\.00\s+seconds into the target video,\s*"
    r"<Picture 1>\s*\(from \[Shot 1\]\)\s*is fully referenced\.\s*$",
    re.MULTILINE,
)
_FL2VA_PART_ONE_RE = re.compile(
    r"^How the reference pictures align with the target video\s*\u2014\s*"
    r"Picture 1 \(from Shot 1\) aligns with the 0\.00-second mark of the target video;\s*"
    r"Picture 2 \(from Shot (\d+)\) aligns with the (\d+\.\d{2})-second mark of the target video\.\s*$",
    re.MULTILINE,
)
_L2VA_PART_ONE_RE = re.compile(
    r"^How the reference pictures align with the target video\s*\u2014\s*"
    r"<Picture 1> \(from \[Shot (\d+)\]\) aligns with the (\d+\.\d{2})-second mark of the target video\.\s*$",
    re.MULTILINE,
)
_ANY_PART_ONE_LINE_RE = re.compile(
    r"^(?:For the target video,\s*at\s+0\.00\s+seconds into the target video,.*"
    r"|How the reference pictures align with the target video\s*(?:\u2014|--|-).*)\n*",
    re.IGNORECASE,
)

_SHOT_TOKEN_RE = re.compile(r"\[Shot[^\]]*\]")
_SHOT_NUM_RE = re.compile(r"\[Shot\s+(\d+)\]")
_SHOT_HEADER_BODY_RE = re.compile(r"\[Shot\s+(\d+)\](.*?)(?=\[Shot\s+\d+\]|\Z)", re.S)
_HEADER_CLOCK_RE = re.compile(r"\b\d{1,2}:\d{2}\.\d+\b")
_SPEAKER_ID_RE = re.compile(r"\(S\d+(?:\s*,\s*S\d+)*\)")
_DIALOGUE_LANG_RE = re.compile(r"<d>(?!\s*\[[A-Za-z][A-Za-z -]*\])")

_CAPTION_PAIR = ("<|caption_start|>", "<|caption_end|>")
_LYRICS_PAIR = ("<|lyrics_start|>", "<|lyrics_end|>")
_MARKER_PAIRS = (_CAPTION_PAIR, _LYRICS_PAIR)
_ALL_MARKERS = (_CAPTION_PAIR[0], _CAPTION_PAIR[1], _LYRICS_PAIR[0], _LYRICS_PAIR[1], "<|cutoff|>")


@dataclass(frozen=True)
class MediaManifest:
    """Ordered media labels & roles derived from wired `H3References` and/or `images.image_1..N`."""

    labels: tuple[str, ...] = ()
    pictures: tuple[str, ...] = ()
    videos: tuple[str, ...] = ()
    audios: tuple[str, ...] = ()
    soundtrack_pairs: tuple[tuple[str, str], ...] = ()  # (audio_label, video_label)
    standalone_audios: tuple[str, ...] = ()
    synthetic_videos: tuple[str, ...] = ()
    summary_lines: tuple[str, ...] = ()

    @property
    def n_pictures(self) -> int:
        return len(self.pictures)

    @property
    def n_videos(self) -> int:
        return len(self.videos)

    @property
    def n_audios(self) -> int:
        return len(self.audios)

    @property
    def has_any(self) -> bool:
        return bool(self.labels)

    @property
    def has_synthetic_video(self) -> bool:
        return bool(self.synthetic_videos)


def describe_references(
    references: Sequence[Any] | None = None,
    *,
    n_extra_images: int = 0,
) -> MediaManifest:
    """Inspect runtime `H3References` (duck-typed) and/or extra still count via `reference_order.assign_labels`."""
    recs: list[ImageRef | VideoRef | AudioRef] = []
    meta_per_rec: list[dict[str, Any]] = []
    if references:
        for idx, item in enumerate(references):
            if isinstance(item, (ImageRef, VideoRef, AudioRef)):
                recs.append(item)
                meta_per_rec.append({"synthetic": False, "use_vae": True})
            elif hasattr(item, "frames") and hasattr(item, "soundtrack"):
                has_st = getattr(item, "soundtrack", None) is not None
                recs.append(VideoRef(name=f"video_{idx + 1}", has_soundtrack=has_st))
                meta_per_rec.append({
                    "synthetic": getattr(item, "allow_upscale", True) is False,
                    "use_vae": bool(getattr(item, "use_vae", True)),
                    "loaded_fps": getattr(item, "loaded_fps", 24.0),
                })
            elif hasattr(item, "image"):
                recs.append(ImageRef(name=f"image_{idx + 1}"))
                meta_per_rec.append({"synthetic": False, "use_vae": bool(getattr(item, "use_vae", True))})
            elif hasattr(item, "audio"):
                recs.append(AudioRef(name=f"audio_{idx + 1}"))
                meta_per_rec.append({"synthetic": False, "use_vae": True})
            else:
                raise TypeError(f"Unsupported reference record in H3References: {type(item).__name__}")
    elif n_extra_images > 0:
        for i in range(int(n_extra_images)):
            recs.append(ImageRef(name=f"image_{i + 1}"))
            meta_per_rec.append({"synthetic": False, "use_vae": True})

    if not recs:
        return MediaManifest()

    labels = tuple(assign_labels(recs))
    pictures: list[str] = []
    videos: list[str] = []
    audios: list[str] = []
    soundtrack_pairs: list[tuple[str, str]] = []
    standalone_audios: list[str] = []
    synthetic_videos: list[str] = []
    summary_lines: list[str] = []

    lbl_idx = 0
    for rec, rmeta in zip(recs, meta_per_rec):
        if isinstance(rec, ImageRef):
            lbl = labels[lbl_idx]
            lbl_idx += 1
            pictures.append(lbl)
            summary_lines.append(f"- {lbl}: still image reference")
        elif isinstance(rec, VideoRef):
            st_lbl: str | None = None
            if rec.has_soundtrack:
                st_lbl = labels[lbl_idx]
                lbl_idx += 1
                audios.append(st_lbl)
            v_lbl = labels[lbl_idx]
            lbl_idx += 1
            videos.append(v_lbl)
            if st_lbl is not None:
                soundtrack_pairs.append((st_lbl, v_lbl))
            if rmeta.get("synthetic"):
                synthetic_videos.append(v_lbl)
                summary_lines.append(
                    f"- {v_lbl}: synthetic kinematic/sprite trajectory guide (follows path/facing; released at 0.33)"
                )
            else:
                st_note = f" (owned soundtrack: {st_lbl}, emitted right before {v_lbl})" if st_lbl else " (silent)"
                summary_lines.append(f"- {v_lbl}: reference video clip{st_note}")
        elif isinstance(rec, AudioRef):
            a_lbl = labels[lbl_idx]
            lbl_idx += 1
            audios.append(a_lbl)
            standalone_audios.append(a_lbl)
            summary_lines.append(f"- {a_lbl}: standalone audio / voice-timbre reference")

    return MediaManifest(
        labels=labels,
        pictures=tuple(pictures),
        videos=tuple(videos),
        audios=tuple(audios),
        soundtrack_pairs=tuple(soundtrack_pairs),
        standalone_audios=tuple(standalone_audios),
        synthetic_videos=tuple(synthetic_videos),
        summary_lines=tuple(summary_lines),
    )


def detect_part_one_mode(text: str) -> str | None:
    """Detect `i2va`, `fl2va`, or `l2va` from a leading vendor Part One alignment sentence."""
    head = (text or "").lstrip()
    if not head:
        return None
    first_line = head.splitlines()[0].strip()
    if _I2VA_PART_ONE_RE.match(first_line):
        return "i2va"
    if _FL2VA_PART_ONE_RE.match(first_line):
        return "fl2va"
    if _L2VA_PART_ONE_RE.match(first_line):
        return "l2va"
    low = first_line.lower()
    if low.startswith("for the target video, at 0.00 seconds"):
        return "i2va"
    if low.startswith("how the reference pictures align with the target video"):
        if "picture 2" in low:
            return "fl2va"
        return "l2va"
    return None


def strip_part_one(text: str) -> str:
    """Remove any leading vendor Part One alignment sentence from `text`."""
    cleaned = (text or "").lstrip()
    return _ANY_PART_ONE_LINE_RE.sub("", cleaned, count=1).lstrip()


def count_final_shot(text: str) -> int:
    """Highest `[Shot N]` number present in `text`, defaulting to `1`."""
    nums = [int(m) for m in _SHOT_NUM_RE.findall(text or "")]
    return max(nums) if nums else 1


def format_part_one(mode: str, *, final_shot: int = 1, length: int = 141) -> str:
    """Build the exact `vendor_guides/base_en.md` §2.1 Part One sentence for `i2va`/`fl2va`/`l2va`."""
    m = str(mode or "").strip().lower()
    if m == "i2va":
        return I2VA_PART_ONE
    snapped = _rules.snap_length(max(5, int(length)))
    dur_s = _rules.duration_of(snapped)
    n_shot = max(1, int(final_shot))
    if m == "fl2va":
        return (
            f"How the reference pictures align with the target video \u2014 "
            f"Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {n_shot}) aligns with the {dur_s:.2f}-second mark of the target video."
        )
    if m == "l2va":
        return (
            f"How the reference pictures align with the target video \u2014 "
            f"<Picture 1> (from [Shot {n_shot}]) aligns with the {dur_s:.2f}-second mark of the target video."
        )
    return ""


def stamp_part_one(prompt_body: str, *, mode: str, length: int = 141) -> str:
    """Deterministically stamp or refresh the exact Part One line for `i2va`/`fl2va`/`l2va`."""
    m = str(mode or "").strip().lower()
    if m not in KEYFRAME_BASE_MODES:
        return prompt_body
    body = strip_part_one(prompt_body)
    part_one = format_part_one(m, final_shot=count_final_shot(body), length=length)
    return f"{part_one}\n\n{body}" if body else part_one


def resolve_prompt_mode(
    prompt_mode: str = "auto",
    *,
    original: str = "",
    manifest: MediaManifest | None = None,
    length: int = 141,
    has_references_wire: bool = False,
) -> str:
    """Resolve `prompt_mode='auto'` into one concrete mode (`ref2va`/`t2va`/`i2va`/`fl2va`/`l2va`/`single_frame`)."""
    pm = str(prompt_mode or "auto").strip().lower()
    if pm not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
    if pm != "auto":
        return pm
    if int(length) == _rules.SINGLE_FRAME:
        return "single_frame"
    if has_references_wire:
        return "ref2va"
    orig_clean = (original or "").strip()
    if orig_clean:
        p1 = detect_part_one_mode(orig_clean)
        if p1 is not None:
            return p1
        if re.search(r"(?m)^(?:subject_definitions|retention_analysis):", orig_clean) or "<Subject " in orig_clean:
            return "ref2va"
        if re.search(r"(?m)^integrated_multimodal_description:", orig_clean):
            return "t2va"
        # Sectionless prompt with no video headers -> single_frame
        if not re.search(r"(?m)^(?:overall_soundscape|non_diegetic_music|detailed_description):", orig_clean):
            return "single_frame"
    if manifest is not None and manifest.has_any:
        return "ref2va"
    return "t2va"


def _header_problems(main_body: str) -> list[str]:
    out: list[str] = []
    bad = [t for t in _SHOT_TOKEN_RE.findall(main_body) if not re.fullmatch(r"\[Shot \d+\]", t)]
    if bad:
        out.append(f"{bad[0]} is not a valid `[Shot N]` header")
    for n, body in _SHOT_HEADER_BODY_RE.findall(main_body):
        opening = re.split(r"(?<=[.!?])\s+", body.strip(), maxsplit=1)[0]
        opening = re.sub(r'"[^"]*"', "", opening)
        if re.match(r"\s*at\s+\d", body, re.I) or _HEADER_CLOCK_RE.search(opening):
            out.append(f"[Shot {n}] carries a timestamp in its opening header; shot headers carry no timestamps")
            break
    return out


def _marker_fails(prompt: str) -> list[str]:
    fails: list[str] = []
    for op, cl in _MARKER_PAIRS:
        if prompt.count(op) != prompt.count(cl):
            fails.append(f"{op} appears {prompt.count(op)}x against {prompt.count(cl)}x for {cl}")
    for op, cl in _MARKER_PAIRS:
        for m in re.finditer(re.escape(op) + r"(.*?)" + re.escape(cl), prompt, re.S):
            body = m.group(1)
            has_d = "<d>" in body or "</d>" in body
            if has_d and (op, cl) == _CAPTION_PAIR:
                fails.append("<d> sits inside <|caption_start|>...<|caption_end|>; captions never wrap <d>")
            if not has_d and (op, cl) == _LYRICS_PAIR:
                fails.append("<|lyrics_start|>...<|lyrics_end|> wraps no <d> block")
            if any(o in body for o, _ in _MARKER_PAIRS):
                fails.append(f"{op} contains a nested marker pair")
    for m in re.finditer(r"<d>(.*?)</d>", prompt, re.S):
        if any(x in m.group(1) for x in _ALL_MARKERS):
            fails.append("a <|...|> marker sits inside a <d>...</d> block")
    if _DIALOGUE_LANG_RE.search(prompt):
        fails.append("a <d> dialogue block is missing its [Language] tag (expected <d>[English] ...</d>)")
    return fails


def _speaker_id_fails(main_body: str, *, require_ref_sx: bool) -> list[str]:
    if not main_body or not require_ref_sx:
        return []
    fails: list[str] = []
    bounds = [m.start() for m in _SHOT_NUM_RE.finditer(main_body)]
    segments = [main_body] if not bounds else [main_body[a:b] for a, b in zip([0] + bounds, bounds + [len(main_body)])]
    for seg in segments:
        prev_end = 0
        for d in re.finditer(r"<d>", seg):
            before = seg[: d.start()]
            if _SPEAKER_ID_RE.search(before):
                prev_end = d.end()
                continue
            if re.search(r"<Audio\s+\d+>", seg[prev_end : d.start()]):
                prev_end = d.end()
                continue
            head = " ".join(before.split())[-60:]
            fails.append(
                f"a <d> dialogue block has no (Sx) speaker ID earlier in its shot; "
                f"bind `<Subject N> (S1)` before `<d>` near ...{head!r}"
            )
            prev_end = d.end()
    return fails


def _extract_sections(prompt: str) -> dict[str, str]:
    body = strip_part_one(prompt)
    hdr_re = re.compile(
        r"(?m)^(subject_definitions|summary|retention_analysis|detailed_description|"
        r"integrated_multimodal_description|overall_soundscape|non_diegetic_music):[ \t]*"
    )
    matches = list(hdr_re.finditer(body))
    out: dict[str, str] = {}
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        out[m.group(1)] = body[start:end].strip()
    return out


def validate_h3_prompt_mechanics(
    prompt: str,
    *,
    mode: str,
    manifest: MediaManifest | None = None,
    length: int = 141,
    enforce_min_words: bool = False,
) -> tuple[list[str], list[str]]:
    """Run mechanical H3 prompt contract checks mirroring `bench/preflight_graph.py`.

    Returns `(fails, warns)`. Any item in `fails` should trigger 1-shot repair."""
    fails: list[str] = []
    warns: list[str] = []
    m = str(mode or "t2va").strip().lower()
    text = (prompt or "").strip()
    if not text:
        return ["prompt is empty"], []

    if m == "single_frame":
        for forbidden_hdr in ("overall_soundscape:", "non_diegetic_music:", "subject_definitions:", "retention_analysis:"):
            if re.search(rf"(?m)^{re.escape(forbidden_hdr)}", text):
                fails.append(f"single_frame image prompt must be sectionless visual prose (found {forbidden_hdr!r})")
        if manifest is not None and manifest.n_pictures > 0:
            pic_nums = {int(n) for n in re.findall(r"<Picture\s+(\d+)>", text)}
            extra = sorted(n for n in pic_nums if n < 1 or n > manifest.n_pictures)
            if extra:
                fails.append(f"single_frame prompt referenced nonexistent <Picture N> {extra}; only 1..{manifest.n_pictures} attached")
        return fails, warns

    fails.extend(_marker_fails(text))
    sections = _extract_sections(text)

    if m == "ref2va":
        missing = [s for s in REF2VA_SECTIONS if not sections.get(s)]
        if missing:
            fails.append(f"ref2va prompt missing required section(s): {missing}")
        summ = sections.get("summary", "")
        if summ:
            tm = re.match(r"^\[([^\]]+)\]", summ)
            if tm is None:
                fails.append("summary must start with a legal [task] bracket (e.g. [reference generation])")
            else:
                tags = [t.strip().lower() for t in tm.group(1).split("+") if t.strip()]
                bad_tags = [t for t in tags if t not in LEGAL_TASK_TAGS]
                if not tags or bad_tags:
                    fails.append(f"summary task bracket contains unknown tag(s) {bad_tags}; legal: {sorted(LEGAL_TASK_TAGS)}")
        ret = sections.get("retention_analysis", "")
        if ret:
            if _SPEAKER_ID_RE.search(ret):
                fails.append("retention_analysis must NEVER contain speaker IDs like (S1)")
            for raw_ln in ret.splitlines():
                ln = raw_ln.strip()
                if not ln:
                    continue
                if ln.startswith("<Audio "):
                    if "(appears in" in ln:
                        fails.append(f"audio retention line must not include '(appears in ...)': {ln[:80]!r}")
                    if not any(f": {mk}" in ln for mk in AUDIO_RETENTION_MARKERS):
                        fails.append(
                            f"audio retention line must use one of {sorted(AUDIO_RETENTION_MARKERS)}: {ln[:80]!r}"
                        )
                elif ln.startswith(("<Subject ", "<Picture ", "<Video ")):
                    if not any(f": {mk}" in ln for mk in VISUAL_RETENTION_MARKERS):
                        fails.append(
                            f"visual retention line must use one of {sorted(VISUAL_RETENTION_MARKERS)}: {ln[:80]!r}"
                        )
        main_body = sections.get("detailed_description", "")
        if "[Shot 1]" not in main_body:
            fails.append("detailed_description must contain '[Shot 1]'")
        fails.extend(_header_problems(main_body))
        fails.extend(_speaker_id_fails(main_body, require_ref_sx=True))
        desc_words = len(main_body.split())
        editing_task = "video editing" in sections.get("summary", "").lower()
        if not editing_task and not (350 <= desc_words <= 500):
            if desc_words < 350:
                msg = (
                    f"detailed_description is {desc_words} words; target requires "
                    "350-500 words of concrete multi-beat staging, camera motion, lighting, and foley"
                )
            else:
                msg = (
                    f"detailed_description is {desc_words} words (> 500 ceiling); target "
                    "requires 350-500 words — tighten prose without dropping shot structure or speaker tags"
                )
            if enforce_min_words:
                fails.append(msg)
            else:
                warns.append(msg)

        if manifest is not None and manifest.has_any:
            for kind, wired, rx in (
                ("Picture", manifest.pictures, r"<Picture\s+(\d+)>"),
                ("Video", manifest.videos, r"<Video\s+(\d+)>"),
                ("Audio", manifest.audios, r"<Audio\s+(\d+)>"),
            ):
                found = {int(n) for n in re.findall(rx, text)}
                max_n = len(wired)
                extra = sorted(n for n in found if n < 1 or n > max_n)
                if extra:
                    fails.append(f"prompt references nonexistent <{kind} N> {extra}; wired count is {max_n}")
                missing_lbls = [f"<{kind} {i}>" for i in range(1, max_n + 1) if i not in found]
                if missing_lbls:
                    fails.append(f"prompt omits wired reference tag(s): {missing_lbls}")
        return fails, warns

    # Base modes: t2va, i2va, fl2va, l2va
    missing = [s for s in BASE_SECTIONS if not sections.get(s)]
    if missing:
        fails.append(f"{m} prompt missing required section(s): {missing}")
    if "<Subject " in text:
        fails.append(f"{m} is a base-format mode and must not contain <Subject N> tags")
    main_body = sections.get("integrated_multimodal_description", "")
    if "[Shot 1]" not in main_body:
        fails.append("integrated_multimodal_description must contain '[Shot 1]'")
    fails.extend(_header_problems(main_body))
    if m in KEYFRAME_BASE_MODES:
        want_p1 = format_part_one(m, final_shot=count_final_shot(main_body), length=length)
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line != want_p1:
            fails.append(f"{m} Part One mismatch: got {first_line!r}, expected {want_p1!r}")
        if m == "fl2va":
            if re.search(r"<Picture\s+\d+>", main_body):
                fails.append("fl2va prose must use bare 'Picture 1' / 'Picture 2', never '<Picture 1>'")
        elif m in ("i2va", "l2va"):
            if "<Picture 2>" in text:
                fails.append(f"{m} takes only <Picture 1> (found <Picture 2>)")
    return fails, warns
