#!/usr/bin/env python3
"""Turn a compressed-tensors W4A16 H3 candidate directory into what the adapter loads.

## What this produces, and why that shape

`MiniMaxH3AWQEncoderLoader` accepts one selected `.safetensors` file whose
safetensors metadata declares `scheme=w4a16`, `quantization=awq`, and the
artifact's own `config.json` verbatim, and whose adapted tensor inventory
exactly populates native ComfyUI's H3. A calibration run emits a Hugging Face
*directory* instead: weights in one or more shards, plus the small configs and
a run record beside them. This script closes that gap in one pass, writing:

1. a single consolidated `.safetensors` carrying the contract metadata, named
   for the generation it came from so it can never be confused with another in
   `models/text_encoders`; and
2. the versioned config snapshot under `config/<name>/`, copied byte-for-byte
   from the candidate's own files, with `sha256.json` over the copies and the
   produced artifact.

The single-file form was chosen over teaching the loader to read the directory.
Both were viable; the single file keeps *one* load path and one contract
instead of two, is indifferent to how many shards the candidate was saved in,
gives the artifact a distinctive name in ComfyUI's combo rather than a generic
`model.safetensors` under a directory, and keeps the existing full-file digest
control working. It costs one more copy of the weights on disk.

The consolidation is faithful: layers H3 does not consume are kept, because the
artifact should stay re-derivable from the calibration output and comparable to
it. The adapter drops them at load.

## Peak memory

Consolidation holds the whole state dict in host memory once, because
safetensors serializes from a complete mapping. Budget roughly the candidate's
on-disk size. `--configs-only` writes the snapshot without that.

Examples:

    python bench/convert_h3_awq_candidate.py --candidate-dir <candidate> \\
        --snapshot-name qwen3vl_32b_minimax_h3_w4a16_awq_v2 \\
        --output <models>/text_encoders/qwen3vl_32b_minimax_h3_w4a16_awq_v2-comfy.safetensors

    python bench/convert_h3_awq_candidate.py --candidate-dir <candidate> \\
        --snapshot-name qwen3vl_32b_minimax_h3_w4a16_awq_v2 --configs-only
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY = REPO.parents[1]
sys.path.insert(0, str(COMFY))

# Provenance only; never a runtime config for the loader.
PROVENANCE_FILES = ("recipe.yaml", "h3_v2_run_record.json")


def _adapter():
    """Import the adapter without importing this repo's node package.

    Same shape as `bench/check_h3_awq_encoder.py::_module`: `h3_awq_encoder`
    imports `comfy_api`, and a bare `import nodes` from a custom-node directory
    resolves to the wrong `nodes.py` (`docs/comfy_notes.md`).
    """
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    pkg = types.ModuleType("_h3pack")
    pkg.__path__ = [str(REPO)]
    sys.modules["_h3pack"] = pkg
    spec = importlib.util.spec_from_file_location(
        "_h3pack.h3_awq_encoder", REPO / "h3_awq_encoder.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def shard_paths(candidate: Path) -> list[Path]:
    """Every weight shard, from the index when the candidate has one.

    Reading the index rather than globbing means a candidate saved with an
    unexpected shard naming still converts, and a candidate whose index names a
    file that is missing fails here instead of producing a short artifact.
    """
    index = candidate / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map") or {}
        names = sorted(set(weight_map.values()))
        if not names:
            raise SystemExit(f"{index.name} declares no weight_map entries")
        paths = [candidate / name for name in names]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"{index.name} names shards that are absent: {missing}")
        return paths
    paths = sorted(
        p for p in candidate.glob("*.safetensors")
        if not p.name.endswith("index.safetensors")
    )
    if not paths:
        raise SystemExit(f"no .safetensors weights found in {candidate.name}")
    return paths


def snapshot_files(candidate: Path, adapter) -> list[str]:
    """The candidate files the snapshot copies, in a stable order.

    The still-image processor config is whichever name this candidate uses;
    the adapter owns that list, so a third spelling is added in one place.
    """
    names = ["config.json", "tokenizer_config.json"]
    still = [n for n in adapter.STILL_CONFIG_NAMES if (candidate / n).is_file()]
    if not still:
        raise SystemExit(
            f"{candidate.name} carries none of {adapter.STILL_CONFIG_NAMES}"
        )
    names.extend(still)
    names.append("video_preprocessor_config.json")
    names.extend(name for name in PROVENANCE_FILES if (candidate / name).is_file())
    missing = [name for name in names if not (candidate / name).is_file()]
    if missing:
        raise SystemExit(f"{candidate.name} is missing {missing}")
    return names


def validate_candidate(candidate: Path, adapter) -> dict:
    """Refuse a directory that is not the contract this adapter implements.

    Runs the adapter's own invariants against the candidate's config, so this
    script cannot accept something the loader would later reject.
    """
    config = adapter._quant_contract(candidate)
    source_layers, depth = adapter.artifact_depth(candidate)
    still = adapter._still_settings(candidate)
    video = adapter._snapshot_json(candidate, "video_preprocessor_config.json")
    return {
        "source_layers": source_layers,
        "h3_layers": depth,
        "quantized_linears": depth * 7,
        "group_size": adapter.GROUP_SIZE,
        "ignore_entries": len((config.get("quantization_config") or {}).get("ignore") or []),
        "still_bounds": adapter._bounds_from(still, "candidate still"),
        "video_bounds": adapter._bounds_from(video, "candidate video"),
        "storage_dtype": config.get("dtype"),
    }


def write_snapshot(candidate: Path, out_dir: Path, adapter,
                   artifact_name: str | None, artifact_digest: str | None,
                   today: str) -> list[Path]:
    """Copy the candidate's small files verbatim and hash what was copied.

    Verbatim, not reformatted: the copies then hash equal to the candidate's
    own files, so `sha256.json` is a statement about the artifact and not only
    about this directory. v1's snapshot added a trailing newline to two files
    and had to record the source digests in prose to stay checkable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    names = snapshot_files(candidate, adapter)
    written = []
    for name in names:
        shutil.copyfile(candidate / name, out_dir / name)
        written.append(out_dir / name)
    digests = {name: _sha256(out_dir / name) for name in names}
    if artifact_name and artifact_digest:
        digests[artifact_name] = artifact_digest
    (out_dir / "sha256.json").write_text(
        json.dumps(dict(sorted(digests.items())), indent=2) + "\n"
    )
    written.append(out_dir / "sha256.json")
    facts = validate_candidate(candidate, adapter)
    (out_dir / "README.md").write_text(
        _readme(out_dir.name, candidate.name, names, facts, artifact_name, today)
    )
    written.append(out_dir / "README.md")
    return written


def _readme(snapshot_name: str, candidate_name: str, names: list[str],
            facts: dict, artifact_name: str | None, today: str) -> str:
    still_lo, still_hi = facts["still_bounds"]
    video_lo, video_hi = facts["video_bounds"]
    artifact_line = (
        f"`{artifact_name}`" if artifact_name
        else "the single file this snapshot's converter produces"
    )
    declares = "\n".join((
        f"- {facts['source_layers']} source decoder layers, of which H3 consumes"
        f" {facts['h3_layers']} ({facts['quantized_linears']} W4A16 linears).",
        f"- symmetric group-{facts['group_size']} pack-quantized W4A16, with"
        f" {facts['ignore_entries']} entries on the quantizer's ignore list;"
        " vision tower, mergers, embedding and LM head stay unquantized.",
        f"- top-level storage dtype `{facts['storage_dtype']}`;"
        " the decoder is bfloat16.",
        f"- a still-image budget of {still_lo}..{still_hi} pixels and a video"
        f" budget of {video_lo}..{video_hi}.",
    ))
    return f"""# {snapshot_name}

Settings-preserving snapshot written by `bench/convert_h3_awq_candidate.py` on
{today} from the candidate directory `{candidate_name}`. The weights themselves
stay outside this repository; the adapter loads {artifact_line}.

Every file here is a byte-for-byte copy of the candidate's own, so the digests
in `sha256.json` are digests of the candidate's files as well as of these
copies. Nothing here was retyped or reformatted.

What this snapshot declares, as of the date above:

{declares}

What consumes each file:

{chr(10).join(_file_note(name) for name in names)}

The loader recognizes an artifact by comparing its embedded `config.json`
against every snapshot under `config/`, not by its filename. Editing any file
here therefore changes which artifacts load, and `bench/check_h3_awq_encoder.py`
hashes this directory on every run.
"""


_FILE_NOTES = {
    "config.json": (
        "validates the Qwen3-VL-32B architecture and the symmetric group-128 "
        "W4A16 compressed-tensors contract, and is the record the loader "
        "matches an artifact against"
    ),
    "tokenizer_config.json": (
        "validates that ComfyUI's native MiniMax tokenizer realizes all 20 "
        "declared special tokens, including the seven H3 ids 151669-151675"
    ),
    "processor_config.json": (
        "constructs the still-image processor: pixel bounds, resampling, "
        "normalization and patch geometry"
    ),
    "preprocessor_config.json": (
        "constructs the still-image processor: pixel bounds, normalization and "
        "patch geometry. The release spelling of the same settings, flat rather "
        "than nested inside a processor container"
    ),
    "video_preprocessor_config.json": (
        "owns video pixel bounds and patch geometry"
    ),
    "recipe.yaml": (
        "records how the artifact was calibrated and quantized; provenance, not "
        "a runtime configuration file"
    ),
    "h3_v2_run_record.json": (
        "the calibration run's own record, copied from the candidate; "
        "provenance, read by nothing at load time"
    ),
}


def _file_note(name: str) -> str:
    return f"- `{name}`: {_FILE_NOTES.get(name, 'copied from the candidate')}."


def consolidate(candidate: Path, shards: list[Path], output: Path) -> dict:
    """Write one safetensors carrying the adapter's contract metadata."""
    from safetensors.torch import load_file, save_file

    config_text = (candidate / "config.json").read_text()
    metadata = {
        "format": "pt",
        "quantization": "awq",
        "scheme": "w4a16",
        # The adapter parses this and requires it to equal the snapshot's
        # config exactly. Embedding the parsed-and-redumped form rather than
        # the raw text is deliberate: equality is compared on parsed JSON, and
        # a redump normalizes whitespace the two copies need not share.
        "config": json.dumps(json.loads(config_text)),
    }
    state: dict = {}
    for shard in shards:
        loaded = load_file(str(shard), device="cpu")
        clashes = sorted(set(loaded) & set(state))
        if clashes:
            raise SystemExit(
                f"{shard.name} repeats tensors already read: {clashes[:3]}"
            )
        state.update(loaded)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output), metadata=metadata)
    return {"tensors": len(state), "bytes": output.stat().st_size}


def verify_output(output: Path, adapter, snapshot: Path) -> str:
    """Reopen the produced file and prove the adapter accepts it.

    Against the snapshot directory just written, not against whatever happens
    to be installed: this asserts the pair is coherent, and leaves resolution
    over `config/` to the loader.
    """
    from safetensors import safe_open

    with safe_open(str(output), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        names = set(handle.keys())
    resolved = adapter._validate_metadata(metadata)
    embedded = json.loads(metadata["config"])
    if embedded != adapter._snapshot_json(snapshot, "config.json"):
        raise SystemExit("produced metadata does not match the written snapshot")
    _, depth = adapter.artifact_depth(snapshot)
    reached = f"model.language_model.layers.{depth - 1}.self_attn.q_proj.weight_packed"
    if reached not in names:
        raise SystemExit(f"produced file does not carry {reached}")
    return str(resolved) if resolved is not None else "this module's own snapshot"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-dir", required=True,
                        help="the calibration run's Hugging Face output directory")
    parser.add_argument("--snapshot-name", required=True,
                        help="directory name for the config snapshot under config/")
    parser.add_argument("--snapshot-root", default=None,
                        help="write the snapshot under this root instead of config/")
    parser.add_argument("--output", default=None,
                        help="path of the consolidated .safetensors to write")
    parser.add_argument("--configs-only", action="store_true",
                        help="write the snapshot and skip the multi-gigabyte write")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing snapshot directory or output file")
    parser.add_argument("--date", default=None,
                        help="date recorded in the snapshot README (default: today)")
    args = parser.parse_args(argv)

    candidate = Path(args.candidate_dir).expanduser().resolve()
    if not candidate.is_dir():
        raise SystemExit(f"{args.candidate_dir} is not a directory")
    if not args.configs_only and not args.output:
        parser.error("--output is required unless --configs-only is given")

    adapter = _adapter()
    root = (Path(args.snapshot_root).expanduser().resolve()
            if args.snapshot_root else adapter.SNAPSHOT_ROOT)
    out_dir = root / args.snapshot_name
    if out_dir.exists() and not args.force:
        raise SystemExit(
            f"{out_dir} already exists; pass --force to replace it, and be sure "
            "you are not replacing a snapshot a deployed artifact still matches"
        )

    facts = validate_candidate(candidate, adapter)
    print(f"candidate {candidate.name}: {facts['source_layers']} decoder layers, "
          f"H3 depth {facts['h3_layers']}, {facts['quantized_linears']} W4A16 "
          f"linears, still {facts['still_bounds']}, video {facts['video_bounds']}, "
          f"storage dtype {facts['storage_dtype']}")

    artifact_name = artifact_digest = None
    if not args.configs_only:
        output = Path(args.output).expanduser()
        if output.exists() and not args.force:
            raise SystemExit(f"{output} already exists; pass --force to replace it")
        shards = shard_paths(candidate)
        print(f"consolidating {len(shards)} shard(s)")
        written = consolidate(candidate, shards, output)
        print(f"wrote {output.name}: {written['tensors']} tensors, "
              f"{written['bytes'] / 1e9:.2f} GB")
        artifact_name = output.name
        artifact_digest = _sha256(output)

    today = args.date or dt.date.today().isoformat()
    files = write_snapshot(candidate, out_dir, adapter, artifact_name,
                           artifact_digest, today)
    print(f"wrote snapshot {out_dir.name}: {len(files)} files")

    if not args.configs_only:
        resolved = Path(verify_output(Path(args.output).expanduser(), adapter, out_dir))
        # Repo-relative when it is inside the repo: this line gets pasted into
        # records, and an absolute path names the machine.
        try:
            shown = resolved.resolve().relative_to(REPO)
        except ValueError:
            shown = resolved.name
        print(f"adapter accepts the produced file; snapshot resolved to {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
