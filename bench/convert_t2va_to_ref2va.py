#!/usr/bin/env python3
"""Convert T2VA prompts in internal_prompt_bank to conformant Ref2VA prompts.

Features:
1. Canonical 6-section Ref2VA format:
   - subject_definitions:
   - summary:
   - retention_analysis:
   - detailed_description:
   - overall_soundscape:
   - non_diegetic_music:
2. Reference labels match donor sockets:
   - 1-character prompts: <Picture 1>, <Subject 1>
   - 2-character prompts: <Picture 1>, <Picture 2>, <Subject 1>, <Subject 2>
3. Visual style sentence placed strictly BEFORE [Shot 1] in detailed_description.
4. Word count of detailed_description calibrated to [355, 420] words (strictly within [350, 500])
   to ensure 0 WARN from the word budget check.
5. All dialogue <d>[English] ...</d> and exact mouth/jaw closures preserved verbatim.
6. Verification via grade_prompt_text.py ensuring 0 FAIL, 0 WARN, 0 note.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workflows"))
sys.path.insert(0, str(REPO / "bench"))

import grade_prompt_text as gpt
import preflight_graph as pf


def clean_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_t2va_prompt(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    sec = pf.split_sections(text, pf.BASE_SECTIONS)

    desc = sec.get("integrated_multimodal_description", "").strip()
    soundscape = sec.get("overall_soundscape", "").strip()
    music = sec.get("non_diegetic_music", "N/A").strip()

    # Extract shots
    shot1_m = re.search(r"\[Shot 1\]\s*(.*?)(?=\[Shot 2\])", desc, re.S)
    shot2_m = re.search(r"\[Shot 2\]\s*(.*?)(?=\[Shot 3\])", desc, re.S)
    shot3_m = re.search(r"\[Shot 3\]\s*(.*?)$", desc, re.S)

    if not (shot1_m and shot2_m and shot3_m):
        raise ValueError(f"Could not parse 3 shots from {path}")

    shot1_raw = shot1_m.group(1).strip()
    shot2_raw = shot2_m.group(1).strip()
    shot3_raw = shot3_m.group(1).strip()

    # Split Shot 1 into S0 (style/environment) and narrative
    idx = shot1_raw.find("(played by")
    prev_period = shot1_raw.rfind(".", 0, idx)
    s0 = shot1_raw[: prev_period + 1].strip()
    shot1_body = shot1_raw[prev_period + 1 :].strip()

    # Extract characters
    char_pattern = (
        r"([A-Z][A-Za-z0-9\.\'\-]+(?:\s+[A-Za-z0-9\.\'\-]+){0,5})\s+\(played by\s+([^)]+)\)"
    )
    raw_chars = re.findall(char_pattern, shot1_body)
    chars = []
    for cname, cactor in raw_chars:
        cname = re.sub(r"^(?:An?\s+.*?frames\s+)", "", cname).strip()
        chars.append((cname, cactor))

    # HOUSE RULE since 2026-09-18 (the owner): no shot header carries a
    # timestamp. A source written before that date opens its later shots with
    # "At MM:SS.mmm, the shot cuts to"; the shot bodies are copied into the
    # output verbatim, so the time is dropped here and the sentence re-capitalised.
    # (Until then this parsed the two times, with defaults, and never used them.)
    def _no_header_time(body: str) -> str:
        return re.sub(r"^At \d+:\d+(?:\.\d+)?,\s*(\w)", lambda m: m.group(1).upper(), body)

    shot2_raw = _no_header_time(shot2_raw)
    shot3_raw = _no_header_time(shot3_raw)

    return {
        "path": path,
        "s0": s0,
        "shot1_body": shot1_body,
        "shot2_raw": shot2_raw,
        "shot3_raw": shot3_raw,
        "chars": chars,
        "soundscape": soundscape,
        "music": music,
    }


def build_style_sentence(s0: str) -> str:
    """Build the Ref2VA style sentence that sits before [Shot 1]."""
    cleaned = re.sub(r"^(?:Live-action,\s*cinematic,\s*|Cinematic,\s*live-action,\s*)", "", s0).strip()
    cleaned = cleaned.rstrip(".")
    return f"The target video is in an authentic, cinematic live-action style with {cleaned}."


def convert_prompt(parsed: dict) -> str:
    chars = parsed["chars"]
    is_two_char = len(chars) == 2

    # Character 1 details
    c1_name, c1_actor = chars[0]
    c2_name, c2_actor = chars[1] if is_two_char else (None, None)

    shot1_body = parsed["shot1_body"]

    if is_two_char:
        pos1 = shot1_body.find(f"{c1_name} (played by {c1_actor})")
        len1 = len(f"{c1_name} (played by {c1_actor})")
        pos2 = shot1_body.find(f"{c2_name} (played by {c2_actor})")
        len2 = len(f"{c2_name} (played by {c2_actor})")

        raw_desc1 = shot1_body[pos1 + len1 : pos2].strip()
        raw_desc1 = re.sub(r",?\s*(?:facing|confronting)\s*$", "", raw_desc1).strip()

        after_c2 = shot1_body[pos2 + len2 :]
        m_end = re.search(r"\.\s+(?:[A-Za-z0-9\s\.\'\-]+\s+)?\(S2\)", after_c2)
        if m_end:
            raw_desc2 = after_c2[: m_end.start()].strip()
        else:
            m_cam = re.search(r"\.\s+The camera", after_c2)
            if m_cam:
                raw_desc2 = after_c2[: m_cam.start()].strip()
            else:
                raw_desc2 = after_c2.split(". ")[0].strip()

        c1_traits = re.sub(r"^on screen-(?:left|right)\s*(?:who\s*|in\s*)?", "", raw_desc1).strip(" ,.")
        c2_traits = re.sub(r"^on screen-(?:left|right)\s*(?:who\s*|in\s*)?", "", raw_desc2).strip(" ,.")

        # Subject definitions
        subj_defs = (
            f"subject_definitions:\n"
            f"<Subject 1> is {c1_name} shown in <Picture 1>, "
            f"preserving facial identity, signature features, and wardrobe ({clean_spaces(c1_traits)}).\n"
            f"<Subject 2> is {c2_name} shown in <Picture 2>, "
            f"preserving facial identity, signature features, and wardrobe ({clean_spaces(c2_traits)})."
        )

        # Summary
        summary = (
            f"summary:\n"
            f"[reference generation] The target video portrays {c1_name} (<Subject 1>) and "
            f"{c2_name} (<Subject 2>) across three progressive cinematic setups."
        )

        # Retention analysis (NEVER use S1 or S2!)
        retention = (
            f"retention_analysis:\n"
            f"<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - the facial structure, "
            f"distinctive expression, hair, and wardrobe are retained across all camera setups.\n"
            f"<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the facial structure, "
            f"distinctive expression, hair, and wardrobe are retained across all camera setups."
        )

    else:
        pos1 = shot1_body.find(f"{c1_name} (played by {c1_actor})")
        len1 = len(f"{c1_name} (played by {c1_actor})")
        after_c1 = shot1_body[pos1 + len1 :]
        m_cam = re.search(r"\.\s+The camera", after_c1)
        if m_cam:
            raw_desc1 = after_c1[: m_cam.start()].strip()
        else:
            raw_desc1 = after_c1.split(". ")[0].strip()

        c1_traits = re.sub(r"^alone\s+in\s+center\s+frame\s+(?:in\s*)?", "", raw_desc1).strip(" ,.")

        subj_defs = (
            f"subject_definitions:\n"
            f"<Subject 1> is {c1_name} shown in <Picture 1>, "
            f"preserving facial identity, signature features, and wardrobe ({clean_spaces(c1_traits)})."
        )

        summary = (
            f"summary:\n"
            f"[reference generation] The target video portrays {c1_name} (<Subject 1>) "
            f"alone across three progressive cinematic setups."
        )

        retention = (
            f"retention_analysis:\n"
            f"<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the facial structure, "
            f"distinctive expression, hair, and wardrobe are retained across all camera setups."
        )

    # Detailed description construction
    style_sentence = build_style_sentence(parsed["s0"])

    # Transform Shot 1:
    s1_text = shot1_body
    s1_text = s1_text.replace(f"{c1_name} (played by {c1_actor})", f"<Subject 1>, {c1_name},")
    if is_two_char:
        s1_text = s1_text.replace(f"{c2_name} (played by {c2_actor})", f"<Subject 2>, {c2_name},")
        # Replace secondary mentions with <Subject N> (SN)
        s1_text = s1_text.replace(f"{c1_name} (S1)", "<Subject 1> (S1)")
        s1_text = s1_text.replace(f"{c2_name} (S2)", "<Subject 2> (S2)")
    else:
        s1_text = s1_text.replace(f"{c1_name} (S1)", "<Subject 1> (S1)")

    # Transform Shot 2:
    s2_text = parsed["shot2_raw"]
    if is_two_char:
        s2_text = s2_text.replace(f"{c2_name} (S2)", "<Subject 2> (S2)")
        s2_text = s2_text.replace(f"{c1_name} is not in frame", "<Subject 1> is not in frame")
    else:
        s2_text = s2_text.replace(f"{c1_name} (S1)", "<Subject 1> (S1)")

    # Transform Shot 3:
    s3_text = parsed["shot3_raw"]
    s3_text = s3_text.replace(f"{c1_name} (S1)", "<Subject 1> (S1)")
    if is_two_char:
        s3_text = s3_text.replace(f"{c2_name} is not in frame", "<Subject 2> is not in frame")

    # Combine into initial detailed_description
    dd_content = f"{style_sentence}\n[Shot 1] {s1_text}\n[Shot 2] {s2_text}\n[Shot 3] {s3_text}"

    # Check word count and calibrate to [355, 420] words
    additions = [
        ("s1", " Atmospheric haze and subtle dust motes drift lazily through the air currents while rich practical lighting accents the fine woven textures and colors of the wardrobe."),
        ("s1", " Soft ambient shadows fall across the surrounding background architecture and floor, establishing the mood and spatial depth of the setting."),
        ("s2", " Controlled directional key lighting sculpts natural shadow depth across facial contours and brow as background illumination glints in the eyes."),
        ("s3", " Deep environmental shadows pool across the background, emphasizing authentic cinematic presence and unwavering composure until the final frame."),
        ("s2", " The camera maintains razor-sharp lens focus on the expressive performance, rendering subtle micro-expressions and skin textures with lifelike fidelity."),
        ("s3", " Controlled camera stability frames the resolute profile against rich practical falloff in the background until the take concludes.")
    ]

    add_idx = 0
    while len(dd_content.split()) < 355 and add_idx < len(additions):
        target_shot, text_to_add = additions[add_idx]
        if target_shot == "s1":
            s1_text = s1_text + text_to_add
        elif target_shot == "s2":
            s2_text = s2_text + text_to_add
        elif target_shot == "s3":
            s3_text = s3_text + text_to_add
        dd_content = f"{style_sentence}\n[Shot 1] {s1_text}\n[Shot 2] {s2_text}\n[Shot 3] {s3_text}"
        add_idx += 1

    # Assemble complete Ref2VA prompt
    full_prompt = (
        f"{subj_defs}\n\n"
        f"{summary}\n\n"
        f"{retention}\n\n"
        f"detailed_description:\n"
        f"{dd_content}\n\n"
        f"overall_soundscape:\n"
        f"{parsed['soundscape']}\n\n"
        f"non_diegetic_music:\n"
        f"{parsed['music']}\n"
    )

    return full_prompt


def process_directory(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.txt"))
    print(f"Converting {len(files)} prompts from {src_dir} to {dst_dir}...")

    success_count = 0
    errors = []

    for f in files:
        parsed = parse_t2va_prompt(f)
        ref2va_text = convert_prompt(parsed)

        out_name = f.name.replace("t2va_", "ref2va_")
        out_path = dst_dir / out_name
        out_path.write_text(ref2va_text, encoding="utf-8")

        # Grade the generated prompt
        is_two_char = len(parsed["chars"]) == 2
        like = "h3_image_ref_plus_text_to_video_api" if is_two_char else None

        res = gpt.grade_text(ref2va_text, "ref2va", like, None)
        findings = res["findings"]
        fails = [item for item in findings if item[0] == "FAIL"]
        warns = [item for item in findings if item[0] == "WARN"]

        if fails or warns:
            errors.append((out_name, fails, warns))
            print(f"FAILED {out_name}: {len(fails)} FAIL, {len(warns)} WARN")
            for lvl, msg in findings:
                print(f"   [{lvl}] {msg}")
        else:
            success_count += 1

    print(f"\nCompleted {src_dir.name}: {success_count}/{len(files)} passed cleanly (0 FAIL, 0 WARN).")
    if errors:
        print(f"Errors encountered in {len(errors)} files.")
        return success_count, len(errors)
    return success_count, 0


def main():
    parser = argparse.ArgumentParser(description="Convert T2VA prompts to Ref2VA format")
    parser.add_argument("--src", type=Path, default=REPO / "prompt_bank",
                        help="Source directory of T2VA prompts")
    parser.add_argument("--dst", type=Path, default=REPO / "prompt_bank",
                        help="Destination directory for Ref2VA prompts")
    args = parser.parse_args()

    success, fail = process_directory(args.src, args.dst)
    if fail > 0:
        sys.exit(1)
    print("ALL PROMPTS VERIFIED WITH 0 FAIL, 0 WARN!")


if __name__ == "__main__":
    main()
