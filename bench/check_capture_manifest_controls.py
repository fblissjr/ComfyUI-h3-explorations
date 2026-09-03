#!/usr/bin/env python3
"""Red controls for the capture-manifest contract: each case is a fixture built
to violate one rule, and the run is green only when every violation is caught
and the two legitimate fixtures pass.

Exists because schema 1.5.0 shipped (dae6864, 2026-09-03) with a checker that
required `prompt_sha256` to be non-empty, never recomputed a tensor's sha256,
and let the generator copy the server stamp from whichever record sorted
first -- three holes Codex found by reading, none of which the checker's own
green could have shown. The rule in CLAUDE.md: a check that cannot go red is
not a check; build the control when the outcome is open. It was.

Cases, all on fixtures of a few megabytes (real captures are 4 GiB a cell):

  gen_same_stamp      two records, one server stamp        -> manifest written
  gen_all_null        two legacy records, no stamp         -> manifest written
  gen_mixed_stamp     two records, different stamps        -> generator refuses
  gen_null_and_stamp  one stamped, one legacy              -> generator refuses
  chk_flipped_byte    a tensor byte changed after hashing  -> checker fails under --verify-hashes
  chk_altered_text    full_prompt_text edited              -> checker fails (hash of the text)
  chk_wrong_bank_id   bank_id names another entry          -> checker fails (identity of the text)

Needs torch for the fixture tensors and the installed ComfyUI checkout for
the generator's audio-row helper (the same dependency preflight has). No
GPU. Runs in seconds.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
GEN = REPO / "bench" / "generate_capture_manifest.py"
sys.path.insert(0, str(REPO / "bench"))
import check_capture_manifest as chk  # noqa: E402
sys.path.insert(0, str(REPO / "workflows"))
import prompts as _prompts  # noqa: E402

DONOR = REPO / "workflows" / "bench" / "h3_text_to_video_stamped_api.json"
SEQ = 2048   # rows per fixture record; must exceed the donor's video + audio rows at length 5


def _graph(length: int) -> dict:
    g = json.loads(DONOR.read_text(encoding="utf-8"))
    for n in g.values():
        if n.get("class_type") == "MiniMaxH3Resolution":
            n["inputs"]["length"] = length
    return g


def _record(path: Path, block: int, step: int, server):
    import torch
    g = torch.Generator().manual_seed(block * 100 + step)
    t = lambda: (torch.randn(1, 2, SEQ, 128, generator=g) * 0.5).bfloat16()  # noqa: E731
    torch.save({"q": t(), "k": t(), "v": t(), "kernel": "sage", "block": block, "step": step,
                "sigma": 0.5, "seq_len": SEQ, "render": 0, "unmerged_blocks": None,
                "server": server}, path)


def _capture(root: Path, name: str, stamps) -> Path:
    d = root / name
    d.mkdir(parents=True)
    for i, st in enumerate(stamps):
        _record(d / f"qkv_L{SEQ}_S{SEQ}_b{i}_s4.pt", i, 4, st)
    return d


def _gen(d: Path, wf: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    return subprocess.run([PY, str(GEN), "--capture-dir", str(d), "--workflow", str(wf)],
                          capture_output=True, text=True, env=env, cwd=str(REPO))


def _check(d: Path, verify: bool) -> str | None:
    chk.VERIFY_HASHES = verify
    try:
        chk.check_manifest(d / "manifest.json")
        return None
    except AssertionError as exc:
        return str(exc)


def main() -> int:
    results = []
    stamp_a = {"argv": ["main.py", "--fast", "fp16_accumulation"], "torch": "x", "cuda": "y",
               "comfy_kitchen": "z", "pid": 1}
    stamp_b = dict(stamp_a, pid=2, argv=["main.py"])
    with tempfile.TemporaryDirectory(prefix="h3_manifest_controls_") as tmp:
        root = Path(tmp)
        wf = root / "donor_len5.json"
        wf.write_text(json.dumps(_graph(5)), encoding="utf-8")

        def case(name, want_ok, run):
            ok, detail = run()
            good = (ok == want_ok)
            results.append(good)
            print(f"  {'ok  ' if good else 'FAIL'}  {name:20} {'passed' if ok else 'refused'} "
                  f"({'as it should' if good else 'WRONG'})  {detail[:110] if detail else ''}")

        # -- generator: server-stamp uniformity --------------------------------
        def gen_case(stamps):
            def run():
                d = _capture(root, f"gen_{len(results)}", stamps)
                r = _gen(d, wf)
                return (d / "manifest.json").is_file(), (r.stderr.strip().splitlines() or [""])[-1] if r.returncode else ""
            return run
        case("gen_same_stamp", True, gen_case([stamp_a, stamp_a]))
        case("gen_all_null", True, gen_case([None, None]))
        case("gen_mixed_stamp", False, gen_case([stamp_a, stamp_b]))
        case("gen_null_and_stamp", False, gen_case([stamp_a, None]))

        # -- checker: a valid manifest, then one violation each ------------------
        good = _capture(root, "chk_good", [stamp_a, stamp_a])
        r = _gen(good, wf)
        assert (good / "manifest.json").is_file(), r.stderr[-400:]
        assert _check(good, verify=True) is None, "the control's own good fixture failed the checker"

        def flipped():
            d = root / "chk_flip"; shutil.copytree(good, d)
            f = next(d.glob("qkv_*.pt"))
            b = bytearray(f.read_bytes()); b[len(b) // 2] ^= 0xFF; f.write_bytes(bytes(b))
            err = _check(d, verify=True)
            return err is None, err or ""
        case("chk_flipped_byte", False, flipped)

        def altered_text():
            d = root / "chk_text"; shutil.copytree(good, d)
            m = json.loads((d / "manifest.json").read_text())
            m["prompt"]["full_prompt_text"] += " one more word"
            (d / "manifest.json").write_text(json.dumps(m))
            err = _check(d, verify=False)
            return err is None, err or ""
        case("chk_altered_text", False, altered_text)

        def wrong_bank():
            d = root / "chk_bank"; shutil.copytree(good, d)
            m = json.loads((d / "manifest.json").read_text())
            other = next(e["id"] for e in _prompts.entries() if e["id"] != m["prompt"]["bank_id"])
            m["prompt"]["bank_id"] = other
            (d / "manifest.json").write_text(json.dumps(m))
            err = _check(d, verify=False)
            return err is None, err or ""
        case("chk_wrong_bank_id", False, wrong_bank)

    n_bad = results.count(False)
    if n_bad:
        print(f"\n{n_bad} control(s) did not behave; the manifest contract has a hole")
        return 1
    print("\nall ok -- every violation is caught and both legitimate fixtures pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
