#!/usr/bin/env python3
"""Stamp `[routing]` (and optional `[provenance]`) from `prompt_bank/control_vectors/*.toml`
into control-vector `.safetensors` `__metadata__` on CPU (`0` GPU needed, bit-exact tensors).

Supports `--dry-run` (default read-only report when neither `--output-dir` nor `--in-place`
is given) and `--output-dir <dir>` so isolated parity tests never touch live `$H3_MODEL_STORE`
or `<comfy>/models/control_vectors/bank` `.safetensors` files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _lib import REPO, bootstrap

bootstrap(cpu=True)
if str(REPO) not in sys.path:
    sys.path.insert(1, str(REPO))

import control_vectors as cv  # noqa: E402
import folder_paths  # noqa: E402

REPO_ROOT = REPO
DEFAULT_TOML_DIR = REPO_ROOT / "prompt_bank" / "control_vectors"


def _default_input_dir() -> Path:
    models_root = Path(getattr(folder_paths, "models_dir", REPO_ROOT.parent.parent / "models"))
    bank = models_root / "control_vectors" / "bank"
    return bank if bank.is_dir() else (models_root / "control_vectors")


def _extract_stem(path: Path, metadata: dict[str, Any]) -> str:
    name = str(metadata.get("name", "")).strip()
    if name:
        return name
    base = path.stem
    for fam in ("fl2va", "ref2va", "h3_image", "fasth3"):
        tag = f"__{fam}"
        if tag in base:
            return base.split(tag, 1)[0]
    return base


def stamp_directory(
    *,
    toml_dir: Path = DEFAULT_TOML_DIR,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    sync_pairs_sha256: bool = True,
    write_sidecars: bool = False,
) -> dict[str, Any]:
    """Stamp `[routing]` and `[provenance]` from `toml_dir/*.toml` onto `.safetensors` in `input_dir`."""
    if in_place and output_dir is not None:
        raise ValueError("pass either output_dir or in_place=True, not both")
    if not dry_run and not in_place and output_dir is None:
        raise ValueError(
            "refusing to modify live control_vectors in place without explicit --in-place "
            "(pass --dry-run to preview or --output-dir <dir> to write copies)"
        )

    eff_in = Path(input_dir) if input_dir is not None else _default_input_dir()
    if not toml_dir.is_dir():
        raise FileNotFoundError(f"TOML directory not found: {toml_dir}")
    if not eff_in.is_dir():
        raise FileNotFoundError(f"input vector directory not found: {eff_in}")

    pair_sets = {}
    for tp in sorted(toml_dir.glob("*.toml")):
        ps = cv.load_pairs(tp)
        pair_sets[ps.name] = ps

    safetensor_files = sorted(eff_in.glob("*.safetensors"))
    stamped: list[str] = []
    skipped: list[str] = []
    unmatched: list[str] = []

    for sf_path in safetensor_files:
        tensors, metadata = cv.load_vector(sf_path)
        stem = _extract_stem(sf_path, metadata)
        ps = pair_sets.get(stem)
        if ps is None or ps.routing is None:
            unmatched.append(sf_path.name)
            continue

        new_meta = json.loads(json.dumps(metadata))
        new_meta["routing"] = ps.routing.to_dict()
        if ps.provenance:
            cv.merge_toml_metadata_into_vector(new_meta, ps)
        if sync_pairs_sha256 and isinstance(new_meta.get("pairs"), dict):
            if int(new_meta["pairs"].get("n_pairs", -1)) == len(ps.pairs):
                new_meta["pairs"]["sha256"] = ps.sha256

        if new_meta == metadata and (in_place or dry_run):
            skipped.append(sf_path.name)
            continue

        stamped.append(sf_path.name)
        if dry_run:
            continue

        dest_dir = eff_in if in_place else Path(output_dir)  # type: ignore[arg-type]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / sf_path.name
        cv.save_vector(
            dest_path,
            tensors,
            new_meta,
            overwrite=True,
            write_sidecar_md=write_sidecars,
        )

    return {
        "toml_count": len(pair_sets),
        "safetensors_scanned": len(safetensor_files),
        "stamped": stamped,
        "skipped": skipped,
        "unmatched": unmatched,
        "dry_run": dry_run,
        "output_dir": str(eff_in if in_place else output_dir) if not dry_run else None,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--toml-dir", type=Path, default=DEFAULT_TOML_DIR, help="Directory of *.toml pair files")
    ap.add_argument("--input-dir", type=Path, default=None, help="Input directory of *.safetensors vectors")
    ap.add_argument("--output-dir", type=Path, default=None, help="Isolated output directory for stamped copies")
    ap.add_argument("--in-place", action="store_true", help="Explicitly overwrite input-dir .safetensors in place")
    ap.add_argument("--dry-run", action="store_true", help="Preview stamping counts without writing any files")
    ap.add_argument(
        "--no-sync-sha256",
        dest="sync_pairs_sha256",
        action="store_false",
        default=True,
        help="Do not update pairs.sha256 to match the [routing]-stamped TOML sha256",
    )
    ap.add_argument("--write-sidecars", action="store_true", help="Also write refreshed .md sidecars")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eff_dry = bool(args.dry_run or (args.output_dir is None and not args.in_place))
    summary = stamp_directory(
        toml_dir=args.toml_dir,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        in_place=bool(args.in_place),
        dry_run=eff_dry,
        sync_pairs_sha256=bool(args.sync_pairs_sha256),
        write_sidecars=bool(args.write_sidecars),
    )
    mode_lbl = "DRY-RUN" if summary["dry_run"] else f"WROTE -> {summary['output_dir']}"
    print(
        f"[{mode_lbl}] scanned={summary['safetensors_scanned']} "
        f"stamped={len(summary['stamped'])} skipped={len(summary['skipped'])} "
        f"unmatched={len(summary['unmatched'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
