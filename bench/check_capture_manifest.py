#!/usr/bin/env python3
"""Check that activation captures contain valid, conforming manifest.json files.

Validates all top-level and nested schema constraints, token arithmetic invariants,
and tensor checksum integrity per docs/capture_manifest_schema.md.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _paths  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows"))
import prompts as _prompts  # noqa: E402


REQUIRED_TOP_KEYS = {
    "schema_version",
    "timestamp",
    "provenance",
    "workload",
    "prompt",
    "references",
    "token_accounting",
    "captured_tensors",
}

REQUIRED_PROVENANCE_KEYS = {
    "git_commit",
    "gpu_device",
    "driver_version",
    "cuda_version",
    "pytorch_version",
    "comfyui_version",
}

REQUIRED_WORKLOAD_KEYS = {"workflow_file", "canvas", "models", "sampling", "attention"}
REQUIRED_CANVAS_KEYS = {"width", "height", "aspect", "length", "latent_frames"}
REQUIRED_MODELS_KEYS = {"unet", "clip", "video_vae"}
REQUIRED_SAMPLING_KEYS = {"sampler_name", "scheduler", "steps", "seed", "cfg"}
REQUIRED_ATTENTION_KEYS = {"sage_mode", "sol_attn", "head_chunks"}
REQUIRED_TOKEN_KEYS = {"total_sequence_length", "video_tokens", "text_tokens", "reference_tokens", "audio_tokens"}
REQUIRED_REFERENCE_KEYS = {"slot", "source_file", "sha256", "raw_dimensions", "fitted_dimensions", "latent_rows"}

# Substrate fields: the KEY is required, and `null` is a legal value.
#
# These are deliberately NOT in the REQUIRED_* sets above, because those assert
# `k in d and d[k]` -- a truthiness test that cannot tell a missing key from a
# present `null`, and would also reject a legitimate 0. For substrate that
# distinction is the whole point:
#
#   key absent        -> NOT RECORDED. Fails. Nobody can say what produced this.
#   key present, null -> CONFIRMED ABSENT. Passes. No clock lock set; weights
#                        not quantized.
#   key present, set  -> recorded. Passes.
#
# Reading absence as "presumably stock" rebuilds the exact hole these fields
# exist to close, and it is CLAUDE.md's rule that anything gaining an absent
# state makes every assertion about it inherit a third case.
# `vae_quantization` is NOT here. It is singular, and a reference graph loads two
# VAEs at different quantizations -- video `int8_convrot` beside audio `fp32` in
# `workflows/h3_probe_capture_ref3_api.json`. Requiring one value over two files
# records something true of neither, and requiring a singular field over a plural
# reality is harder to walk back than to not require. Left optional until it is
# either split per-VAE or dropped.
SUBSTRATE_MODELS_KEYS = {"weight_quantization"}
SUBSTRATE_PROVENANCE_KEYS = {"gpu_power_limit_watts"}


SUBSTRATE_SINCE = "1.1.0"


def substrate_expected(schema_version: str) -> bool:
    """Whether this manifest's version is required to carry the substrate keys.

    The substrate fields were ADDED in 1.1.0. `SCHEMA_VERSIONS` still accepts
    1.0.0, so asserting them unconditionally failed a manifest that conforms
    perfectly to the version it declares -- red on correct state, which
    `CLAUDE.md` rates worse than no check, and it would have fired on exactly
    the older captures a migration is supposed to leave alone.

    Version-gated rather than presence-gated on purpose: "the key is missing, so
    do not check it" would make the assertion unfalsifiable, which is the whole
    defect it was written to close.
    """
    return tuple(int(x) for x in schema_version.split(".")) >= \
        tuple(int(x) for x in SUBSTRATE_SINCE.split("."))


def assert_substrate(block: dict, keys: set, where: str) -> None:
    """Presence-only assertion. `null` passes; a missing key does not."""
    for k in sorted(keys):
        assert k in block, (
            f"Missing substrate key {k!r} in {where}. This is not the same as "
            f"recording it as null: null means confirmed absent, a missing key "
            f"means nobody wrote down what produced this measurement."
        )


# The accepted set and the reported version were two separate literals until
# 2026-08-17, and they had already drifted: the report said v1.0.0 while the
# schema and the only existing manifest were both 1.1.0, so it would have kept
# claiming 1.0.0 through every future bump. One constant for the accepted set,
# and the report states the versions it actually saw rather than a fixed string.
SCHEMA_VERSIONS = ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0", "1.5.0")

#: Model-file hashes were added in 1.2.0, gated the same way and for the same
#: reason as the substrate keys above: a 1.1.0 manifest that never carried them
#: conforms to the version it declares, and failing it would be red on correct
#: state.
MODEL_HASHES_SINCE = "1.2.0"


def model_hashes_expected(schema_version: str) -> bool:
    return tuple(int(x) for x in schema_version.split(".")) >= \
        tuple(int(x) for x in MODEL_HASHES_SINCE.split("."))


def assert_model_hashes(models: dict, where: str) -> None:
    """Every model the manifest NAMES carries a hash, and it looks like one.

    Keyed off the names already in the block rather than a fixed list, so a
    manifest naming a model this file has never heard of still has to account
    for it. A resolution failure is recorded as a reason string and passes here
    -- "could not hash, and here is why" is a fact; a silently absent entry is
    the thing being ruled out.
    """
    digests = models.get("sha256")
    assert isinstance(digests, dict), (
        f"{where}.sha256 must be an object mapping each named model to its "
        f"digest, got {type(digests).__name__}. A manifest that names its "
        f"models but records no bytes identifies them by filename alone.")
    if "_unavailable" in digests:
        return  # generation could not reach folder_paths, and says so
    for field in ("unet", "clip", "video_vae", "audio_vae"):
        if not models.get(field):
            continue
        assert field in digests, (
            f"{where} names {field!r} but {where}.sha256 has no entry for it")
        val = digests[field]
        assert isinstance(val, str) and (
            len(val) == 64 or val.startswith(("unresolved:", "unreadable:"))), (
            f"{where}.sha256.{field} is {val!r}: expected a 64-char sha256 or "
            f"a reason string saying why not")
    named = models.get("loras") or []
    if named:
        got = digests.get("loras")
        assert isinstance(got, list) and len(got) == len(named), (
            f"{where} names {len(named)} lora(s) but sha256.loras holds "
            f"{got!r}; a per-lora hash cannot be matched up by position "
            f"unless the lists are the same length")
        # Length alone accepted a null, which the schema declares as a string.
        # A null says nothing -- not the digest, and not why there isn't one.
        for i, val in enumerate(got):
            assert isinstance(val, str) and (
                len(val) == 64
                or val.startswith(("unresolved:", "unreadable:"))), (
                f"{where}.sha256.loras[{i}] is {val!r}: expected a 64-char "
                f"sha256 or a reason string saying why not")


def check_manifest(manifest_path: Path):
    assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"
    data = json.loads(manifest_path.read_text())

    # Top-level validation
    for k in REQUIRED_TOP_KEYS:
        assert k in data, f"Missing required top-level key {k!r} in {manifest_path}"

    assert data["schema_version"] in SCHEMA_VERSIONS, f"Unexpected schema_version: {data['schema_version']}"

    # Provenance
    prov = data["provenance"]
    for k in REQUIRED_PROVENANCE_KEYS:
        assert k in prov and prov[k], f"Missing required provenance key {k!r}"
    if substrate_expected(data["schema_version"]):
        assert_substrate(prov, SUBSTRATE_PROVENANCE_KEYS, "provenance")

    # Workload
    workload = data["workload"]
    for k in REQUIRED_WORKLOAD_KEYS:
        assert k in workload, f"Missing required workload key {k!r}"

    canvas = workload["canvas"]
    for k in REQUIRED_CANVAS_KEYS:
        assert k in canvas, f"Missing required canvas key {k!r}"
    assert canvas["width"] % 32 == 0 and canvas["height"] % 32 == 0, "Canvas dimensions not multiple of 32"

    models = workload["models"]
    for k in REQUIRED_MODELS_KEYS:
        assert k in models and models[k], f"Missing required models key {k!r}"
    if substrate_expected(data["schema_version"]):
        assert_substrate(models, SUBSTRATE_MODELS_KEYS, "workload.models")
    if model_hashes_expected(data["schema_version"]):
        assert_model_hashes(models, "workload.models")

    # `weight_quantization` is a PROJECTION of the required `unet` filename, not
    # independent information -- `int8_convrot` is readable straight off
    # `minimax_h3_fl2va_pruned_int8_convrot.safetensors`. A second home for one
    # fact is only honest if something asserts the two agree; the precedent is
    # filing.md's rule that an `artifacts` list is a projection of the citations
    # and a disagreement means one of them is wrong. Without this, the field could
    # say fp8_scaled over an int8 filename and both would pass.
    # `vae_quantization` describes the VIDEO vae only. It is optional and its name
    # is singular over a plural reality, so rather than leave it ambiguous it is
    # pinned to the required `video_vae` and documented as saying nothing about the
    # audio vae. Same projection test as below: if present, it must agree.
    vq = models.get("vae_quantization")
    if vq is not None:
        assert vq in models["video_vae"], (
            f"vae_quantization {vq!r} does not appear in the video_vae filename "
            f"{models['video_vae']!r}. This field describes the video vae only -- "
            f"it says nothing about audio_vae, which a reference graph loads at a "
            f"different quantization."
        )

    wq = models.get("weight_quantization")
    if wq is not None:
        assert wq in models["unet"], (
            f"weight_quantization {wq!r} does not appear in the unet filename "
            f"{models['unet']!r}. One of the two is wrong; the filename is the "
            f"file that was actually loaded."
        )

    sampling = workload["sampling"]
    for k in REQUIRED_SAMPLING_KEYS:
        assert k in sampling, f"Missing required sampling key {k!r}"
    assert sampling["steps"] > 0, "Sampling steps must be positive"

    attn = workload["attention"]
    for k in REQUIRED_ATTENTION_KEYS:
        assert k in attn, f"Missing required attention key {k!r}"

    # Prompt
    prompt_obj = data["prompt"]
    assert "full_prompt_text" in prompt_obj and prompt_obj["full_prompt_text"], "Prompt full text missing"
    assert "sections" in prompt_obj and isinstance(prompt_obj["sections"], dict), "Prompt sections missing"
    for s_name, s_content in prompt_obj["sections"].items():
        assert not s_content.endswith("..."), f"Prompt section {s_name!r} contains truncated ellipsis"

    # Token Accounting & Invariants
    tokens = data["token_accounting"]
    for k in REQUIRED_TOKEN_KEYS:
        assert k in tokens, f"Missing required token_accounting key {k!r}"

    sum_tokens = (
        tokens["video_tokens"]
        + tokens["text_tokens"]
        + tokens["reference_tokens"]
        + tokens["audio_tokens"]
    )
    assert tokens["total_sequence_length"] == sum_tokens, (
        f"Token sum invariant violated: total ({tokens['total_sequence_length']}) != "
        f"video({tokens['video_tokens']}) + text({tokens['text_tokens']}) + "
        f"ref({tokens['reference_tokens']}) + audio({tokens['audio_tokens']})"
    )

    # References & Reference Token Invariant
    refs = data["references"]
    # A text-to-video capture has no references, and that is a state, not a
    # defect: the assertion inherited a third case when the Base16 capture
    # landed (2026-09-03). Zero references is legal only when the token
    # accounting agrees there were none.
    assert len(refs) > 0 or data["token_accounting"]["reference_tokens"] == 0, (
        "Manifest lists 0 references but token_accounting.reference_tokens is nonzero")
    if data["schema_version"] not in ("1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"):
        for k in ("bank_id", "prompt_sha256"):
            assert k in data["prompt"], f"Missing prompt key {k!r} (required from schema 1.5.0)"
        assert data["prompt"]["prompt_sha256"], "prompt_sha256 must be set from schema 1.5.0"
        # The hash must be OF the text beside it, and the bank id must be the
        # entry that text is -- required-nonempty was a hole Codex found on
        # 2026-09-03: a stale hash or a wrong id would have passed.
        want = hashlib.sha256(data["prompt"]["full_prompt_text"].rstrip().encode("utf-8")).hexdigest()
        assert data["prompt"]["prompt_sha256"] == want, (
            "prompt_sha256 is not the sha256 of full_prompt_text (rstripped)")
        assert data["prompt"]["bank_id"] == _prompts.identify(data["prompt"]["full_prompt_text"]), (
            f"bank_id {data['prompt']['bank_id']!r} is not the bank entry full_prompt_text identifies as")
        assert data["provenance"]["server"] is None or isinstance(data["provenance"]["server"], dict), (
            "provenance.server must be null or the server stamp dict")
        for k in ("workflow_sha256", "graph_sha256"):
            assert k in data["workload"], f"Missing workload key {k!r} (required from schema 1.5.0)"
        assert data["workload"].get("task") in ("t2va", "i2va", "fl2va", "l2va", "ref2va"), (
            f"workload.task must name the render type from schema 1.5.0, got {data['workload'].get('task')!r}")
        assert "server" in data["provenance"], "provenance.server key missing (null is legal for pre-1.5.0 captures)"
    ref_row_sum = 0
    for r in refs:
        for k in REQUIRED_REFERENCE_KEYS:
            assert k in r, f"Missing required reference key {k!r}"
        ref_row_sum += r["latent_rows"]

    assert tokens["reference_tokens"] == ref_row_sum, (
        f"Reference token sum invariant violated: token_accounting.reference_tokens ({tokens['reference_tokens']}) != "
        f"sum of latent_rows ({ref_row_sum})"
    )

    # Captured Tensors Integrity
    tensors = data["captured_tensors"]
    assert len(tensors) > 0, "Manifest lists 0 captured tensors"
    pids = {t.get("server_pid") for t in tensors}
    assert len(pids) == 1, (
        f"captured tensors came from {len(pids)} distinct server processes ({sorted(map(str, pids))}); "
        f"one capture must be one process")
    cap_dir = manifest_path.parent
    for t in tensors:
        assert "filename" in t and "sha256" in t and "shape" in t and "dtype" in t
        pt_file = cap_dir / t["filename"]
        assert pt_file.is_file(), f"Tensor file listed in manifest missing on disk: {pt_file}"
        assert pt_file.stat().st_size == t["size_bytes"], f"File size mismatch for {pt_file}"
        if VERIFY_HASHES:
            got = _sha256_file(pt_file)
            assert got == t["sha256"], f"sha256 mismatch for {pt_file.name}: manifest {t['sha256'][:12]}, file {got[:12]}"
        assert t["shape"][2] == tokens["total_sequence_length"], (
            f"Tensor shape sequence dimension {t['shape'][2]} does not match total_sequence_length {tokens['total_sequence_length']}"
        )


VERIFY_HASHES = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    global VERIFY_HASHES
    VERIFY_HASHES = "--verify-hashes" in sys.argv[1:]
    # Say which it was, so a green is distinguishable: without the flag the
    # recorded sha256 is never recomputed (Codex, 2026-09-03), and the fast
    # path checks existence, size and sequence shape only.
    print("  note  tensor sha256 " + ("recomputed for every file (--verify-hashes)" if VERIFY_HASHES
                                      else "NOT recomputed; pass --verify-hashes to read every tensor file"))
    capture_base = _paths.capture_root()
    if capture_base is None or not capture_base.is_dir():
        print("  skip  no capture collection: set H3_CAPTURE_ROOT")
        return 0

    # Enumerate CAPTURES, not manifests. Globbing `*/manifest.json` and
    # validating the hits could only ever fail on a malformed manifest, never on
    # a capture nobody recorded -- which is the case this check exists for. It
    # reported ok on a collection where one of two capture directories had no
    # manifest at all, holding twelve multi-GiB tensors and no provenance.
    # Ask what the input would have to look like for a check to fail; if the
    # answer does not include the failure it was written for, it is the wrong
    # enumeration.
    captures = sorted(d for d in capture_base.iterdir()
                      if d.is_dir() and any(d.glob("qkv_*.pt")))
    if not captures:
        print("  skip  no capture directories found to validate")
        return 0

    unmanifested = [d for d in captures if not (d / "manifest.json").is_file()]
    if unmanifested:
        names = "\n".join(f"    {d.name}  ({len(list(d.glob('qkv_*.pt')))} tensors)"
                          for d in unmanifested)
        print(f"  FAIL  {len(unmanifested)} of {len(captures)} capture "
              f"director(ies) have no manifest.json:\n{names}\n"
              f"    A capture with no provenance grades a later comparison "
              f"against a substrate nobody can recover.\n"
              f"    Write one with bench/generate_capture_manifest.py.",
              file=sys.stderr)
        return 1

    seen: set[str] = set()
    for d in captures:
        m = d / "manifest.json"
        check_manifest(m)
        seen.add(json.loads(m.read_text())["schema_version"])

    versions = ", ".join(f"v{v}" for v in sorted(seen))
    print(f"  ok    validated {len(captures)} capture(s) against {versions} "
          f"(every capture carries a manifest, and all invariants hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
