#!/usr/bin/env python3
"""Does a clip say its scripted lines? Whisper transcription against the prompt's dialogue, word error rate per clip and per pair.

    H3_COMFY_OUTPUT=<share> python bench/measure_dialogue_transcription.py --pairs pairs.json --out record.json
    python bench/measure_dialogue_transcription.py --summarize record.json

Clip paths are relative to ComfyUI's output directory (`bench/_paths.py::comfy_output`)
and name the silent `.mp4`; the sound is read from its `-audio.mp4` sibling. A
pairs file is a JSON list of `{"label", "a", "b"}` with optional
`"a_label"` / `"b_label"`, or a previous record, whose pairs are re-run; a clip
listed in several pairs is transcribed once.

## Why

The evaluation sweep found one audio check with a high reported agreement
with people on speech content: transcription error against the scripted line
(`docs/research/2026-09-19_evaluation_one_judge.md`, item 3). Nothing else in
this pack measures whether the words came out. It is also the one audio
metric that needs no reference render: the prompt is the reference, so it
reads on a pair of two takes as well as on one take.

## What it measures

Per clip, the scripted lines are read from the prompt embedded in the clip
(every `<d>[Language] ...</d>` span, in order). Whisper transcribes the
clip's sound with greedy decoding; script and transcript go through Whisper's
own text normaliser; the error rate is the edit distance over the script's
unit count, with substitutions, deletions and insertions kept apart. Units
are words, except that each Chinese, Japanese or Korean character is a unit
of its own, since those scripts are written without spaces. Per line, the best-matching window of the transcript
gives that line's own error, so a clip that drops one line of three reads
differently from one that mumbles all three. Per pair, B's error minus A's.

The model is read from the local Hugging Face cache; nothing is downloaded
and nothing is installed (`--model`, default `openai/whisper-medium`). CPU
only.

The control: every transcript is also scored against another clip's script,
which must read far worse, or the score is not measuring the words.

Limits: a transcriber's error is not a listener's; it cannot hear accent,
timing, emotion or which character spoke. Lines in a language other than
English are transcribed but scored with the English normaliser. A clip whose
prompt has no dialogue is reported with no score.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import orjson

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _paths  # noqa: E402
from diff_clip_graphs import graph_of  # noqa: E402

RATE = 16000                                      # Whisper's input rate
DIALOGUE = re.compile(r"<d>\s*(?:\[([^\]]+)\])?\s*(.*?)</d>", re.S)
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uff66-\uff9f"
UNITS = re.compile(f"[{_CJK}]|[^\\s{_CJK}\\u3000-\\u303f\\uff00-\\uff65]+")


def resolve(clip: str) -> Path:
    p = Path(clip)
    return p if p.is_absolute() else _paths.comfy_output() / p


def shown(path: Path) -> str:
    try:
        return str(path.relative_to(_paths.comfy_output()))
    except ValueError:
        return path.name


def script_lines(clip: Path) -> list[dict]:
    """Every <d> span of every long text input in the embedded graph, in graph order."""
    lines = []
    for node in graph_of(str(clip)).values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, str) and "<d>" in value:
                lines += [{"language": (lang or "").strip(), "text": text.strip()}
                          for lang, text in DIALOGUE.findall(value)]
    return lines


def audio(clip: Path) -> np.ndarray:
    sib = clip.with_name(clip.stem + "-audio.mp4")
    src = sib if sib.is_file() else clip
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", str(RATE),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def edit_counts(ref: list[str], hyp: list[str]) -> dict:
    """Word-level Levenshtein alignment: substitutions, deletions, insertions."""
    n, m = len(ref), len(hyp)
    cost = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[:, 0] = np.arange(n + 1)
    cost[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1,
                             cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i, j] == cost[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and cost[i, j] == cost[i - 1, j] + 1:
            d, i = d + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return {"ref_words": n, "sub": int(s), "del": int(d), "ins": int(ins),
            "wer": (s + d + ins) / n if n else None}


def best_window(ref: list[str], hyp: list[str]) -> dict:
    """The transcript window that matches one scripted line best, over window lengths near the line's."""
    if not ref:
        return {"wer": None}
    best = None
    for length in range(max(1, len(ref) // 2), len(ref) * 3 // 2 + 2):
        for start in range(0, max(1, len(hyp) - length + 1)):
            c = edit_counts(ref, hyp[start:start + length])
            if best is None or c["wer"] < best["wer"]:
                best = {**c, "window": " ".join(hyp[start:start + length])}
    return best or {"wer": None}


class Transcriber:
    def __init__(self, model: str, threads: int):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        torch.set_num_threads(threads)
        self.torch = torch
        self.processor = WhisperProcessor.from_pretrained(model, local_files_only=True)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, local_files_only=True).eval()
        self.name = model

    def __call__(self, wave: np.ndarray, language: str | None) -> str:
        feats = self.processor(wave, sampling_rate=RATE, return_tensors="pt").input_features
        kwargs = {"task": "transcribe", "do_sample": False, "num_beams": 1}
        if language:
            kwargs["language"] = language
        with self.torch.inference_mode():
            ids = self.model.generate(feats, **kwargs)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    def normalize(self, text: str) -> list[str]:
        """Scoring units: words, except that each CJK character is its own unit.

        Chinese, Japanese and Korean are written without spaces, so splitting on
        whitespace turns a line into one or two "words" and a transcript that
        drops a comma scores as a total miss. Characters are the usual unit there.
        """
        return UNITS.findall(self.processor.tokenizer.normalize(text))


def measure_clip(tr: Transcriber, clip: Path) -> dict:
    lines = script_lines(clip)
    langs = {ln["language"].lower() for ln in lines if ln["language"]}
    language = "english" if langs == {"english"} else None
    text = tr(audio(clip), language)
    hyp = tr.normalize(text)
    row = {"clip": shown(clip), "language_forced": language, "transcript": text, "lines": lines}
    if not lines:
        row["score"] = None
        return row
    ref = tr.normalize(" ".join(ln["text"] for ln in lines))
    row["score"] = edit_counts(ref, hyp)
    row["per_line"] = [{"text": ln["text"], **best_window(tr.normalize(ln["text"]), hyp)} for ln in lines]
    return row


def summarize(path: Path) -> int:
    d = orjson.loads(path.read_bytes())
    clips = {c["clip"]: c for c in d["clips"]}
    print(f"{'pair':34s} {'A wer':>7s} {'B wer':>7s} {'B-A':>7s}   worst line A / B")
    for p in d["pairs"]:
        a, b = clips[p["a"]], clips[p["b"]]
        if a["score"] is None or b["score"] is None:
            print(f"{p['label']:34s}  no dialogue")
            continue
        wa, wb = a["score"]["wer"], b["score"]["wer"]
        la = max(x["wer"] for x in a["per_line"])
        lb = max(x["wer"] for x in b["per_line"])
        print(f"{p['label']:34s} {wa:7.3f} {wb:7.3f} {wb - wa:+7.3f}   {la:.2f} / {lb:.2f}")
    ctrl = [c["control_other_script"]["wer"] for c in d["clips"] if c.get("control_other_script")]
    if ctrl:
        print(f"control, transcript against another scene's script: lowest wer {min(ctrl):.3f} over {len(ctrl)} clips")
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--summarize":
        return summarize(Path(sys.argv[2]))
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--model", default="openai/whisper-medium")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    pairs = orjson.loads(args.pairs.read_bytes())
    if isinstance(pairs, dict):          # a previous record: re-run its pairs
        pairs = [{k: v for k, v in pr.items() if v is not None} for pr in pairs["pairs"]]
    tr = Transcriber(args.model, args.threads)
    done: dict[str, dict] = {}
    for p in pairs:
        for side in ("a", "b"):
            path = resolve(p[side])
            if not path.is_file():
                raise SystemExit(f"refuse: no clip at {shown(path)}")
            if shown(path) not in done:
                done[shown(path)] = measure_clip(tr, path)
                s = done[shown(path)]["score"]
                shown_wer = "-" if s is None else f"{s['wer']:.3f}"
                print(f"{shown(path):90s} wer {shown_wer}", flush=True)
    # The control: each transcript scored against ANOTHER clip's script (the next
    # clip, in order, whose script differs). A tool that reads near zero here is
    # not reading the words.
    scored = [c for c in done.values() if c["score"] is not None]
    for i, c in enumerate(scored):
        other = next((o for o in scored[i + 1:] + scored[:i] if o["lines"] != c["lines"]), None)
        if other is not None:
            ref = tr.normalize(" ".join(ln["text"] for ln in other["lines"]))
            c["control_other_script"] = {"script_of": other["clip"], **edit_counts(ref, tr.normalize(c["transcript"]))}
    record = {"tool": "bench/measure_dialogue_transcription.py", "model": tr.name,
              "decoding": "greedy, task transcribe, English forced when every scripted line is tagged English",
              "normaliser": "the model's own English text normaliser (WhisperTokenizer.normalize)",
              "pairs": [{"label": p["label"], "a": shown(resolve(p["a"])), "b": shown(resolve(p["b"])),
                         "a_label": p.get("a_label"), "b_label": p.get("b_label")} for p in pairs],
              "clips": list(done.values())}
    args.out.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2) + b"\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
