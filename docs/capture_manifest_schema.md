# Activation Capture Manifest Schema (v1.1.0)

Every activation capture directory under `$H3_CAPTURE_ROOT/` must contain a complete, auditable `manifest.json` alongside the `.pt` tensor files.

The manifest guarantees that any captured tensor is 100% traceable to the exact prompt, reference images, model checkpoints, canvas geometry, sampling parameters, and per-head error decomposition analysis.

---

## 1. JSON Schema Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "H3ActivationCaptureManifest",
  "type": "object",
  "required": [
    "schema_version",
    "timestamp",
    "provenance",
    "workload",
    "prompt",
    "references",
    "token_accounting",
    "captured_tensors"
  ],
  "properties": {
    "schema_version": { "type": "string", "enum": ["1.0.0", "1.1.0"] },
    "timestamp": { "type": "string", "format": "date-time" },
    "provenance": {
      "type": "object",
      "required": ["git_commit", "gpu_device", "cuda_version", "driver_version", "comfyui_version", "gpu_power_limit_watts"],
      "properties": {
        "git_commit": { "type": "string" },
        "gpu_device": { "type": "string" },
        "gpu_power_limit_watts": { "type": "number" },
        "driver_version": { "type": "string" },
        "cuda_version": { "type": "string" },
        "pytorch_version": { "type": "string" },
        "comfyui_version": { "type": "string" },
        "comfy_kitchen_version": { "type": "string" }
      }
    },
    "workload": {
      "type": "object",
      "required": ["workflow_file", "canvas", "models", "sampling", "attention"],
      "properties": {
        "workflow_file": { "type": "string" },
        "canvas": {
          "type": "object",
          "required": ["width", "height", "aspect", "length", "latent_frames"],
          "properties": {
            "width": { "type": "integer" },
            "height": { "type": "integer" },
            "aspect": { "type": "string" },
            "length": { "type": "integer" },
            "fps": { "type": "number" },
            "latent_frames": { "type": "integer" }
          }
        },
        "models": {
          "type": "object",
          "required": ["unet", "clip", "video_vae", "weight_quantization"],
          "properties": {
            "unet": { "type": "string" },
            "clip": { "type": "string" },
            "video_vae": { "type": "string" },
            "audio_vae": { "type": "string" },
            "weight_quantization": { "type": "string" },
            "vae_quantization": { "type": "string" },
            "loras": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": { "type": "string" },
                  "strength": { "type": "number" },
                  "rank": { "type": ["integer", "null"] }
                }
              }
            }
          }
        },
        "sampling": {
          "type": "object",
          "required": ["sampler_name", "scheduler", "steps", "seed", "cfg"],
          "properties": {
            "sampler_name": { "type": "string" },
            "scheduler": { "type": "string" },
            "steps": { "type": "integer" },
            "denoise": { "type": "number" },
            "seed": { "type": "integer" },
            "cfg": { "type": "number" }
          }
        },
        "attention": {
          "type": "object",
          "required": ["sage_mode", "sol_attn", "head_chunks"],
          "properties": {
            "sage_mode": { "type": "string" },
            "sol_attn": { "type": "string" },
            "head_chunks": { "type": "integer" }
          }
        }
      }
    },
    "prompt": {
      "type": "object",
      "required": ["full_prompt_text", "sections"],
      "properties": {
        "full_prompt_text": { "type": "string" },
        "sections": {
          "type": "object",
          "properties": {
            "subject_definitions": { "type": "string" },
            "summary": { "type": "string" },
            "retention_analysis": { "type": "string" },
            "detailed_description": { "type": "string" },
            "soundscape": { "type": "string" }
          }
        }
      }
    },
    "references": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["slot", "source_file", "sha256", "raw_dimensions", "fitted_dimensions", "latent_rows"],
        "properties": {
          "slot": { "type": "string" },
          "source_file": { "type": "string" },
          "sha256": { "type": "string" },
          "raw_dimensions": { "type": "array", "items": { "type": "integer" } },
          "fitted_dimensions": { "type": "array", "items": { "type": "integer" } },
          "latent_rows": { "type": "integer" },
          "fit_settings": { "type": "object" }
        }
      }
    },
    "token_accounting": {
      "type": "object",
      "required": ["total_sequence_length", "video_tokens", "text_tokens", "reference_tokens", "audio_tokens"],
      "properties": {
        "total_sequence_length": { "type": "integer" },
        "video_tokens": { "type": "integer" },
        "text_tokens": { "type": "integer" },
        "reference_tokens": { "type": "integer" },
        "audio_tokens": { "type": "integer" }
      }
    },
    "captured_tensors": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["filename", "block", "step", "shape", "dtype", "size_bytes"],
        "properties": {
          "filename": { "type": "string" },
          "block": { "type": "integer" },
          "step": { "type": "integer" },
          "shape": { "type": "array", "items": { "type": "integer" } },
          "dtype": { "type": "string" },
          "size_bytes": { "type": "integer" },
          "sha256": { "type": "string" }
        }
      }
    }
  }
}
```

## Substrate fields: the key is required, `null` is legal

Added 2026-08-17. `weight_quantization` and `gpu_power_limit_watts` are required
**keys** whose value may be `null`, and
`bench/check_capture_manifest.py` asserts them by presence rather than by
truthiness. Three states, not two:

| state | meaning | verdict |
|---|---|---|
| key absent | **not recorded** — nobody wrote down what produced this | fails |
| key present, `null` | **confirmed absent** — no clock lock set, weights not quantized | passes |
| key present, value | recorded | passes |

A validator that read a missing key as "presumably stock" would rebuild the hole
these fields exist to close, which is why they are not in the `REQUIRED_*` sets
that test `k in d and d[k]` — that test cannot tell absence from `null`, and would
also reject a legitimate `0`.

**Why the model build matters and not only the host.** The shipped checkpoints are
`pruned_int8_convrot`, and `convrot` applies a rotation, so any claim about q/k
geometry is a claim about a rotated, pruned, quantized space. `fp8_scaled` and
`w4a8_mixed` builds of the same models sit on disk beside them. See
`docs/open_experiments.md` item 19.

**Migration cost was zero on the day this landed:** the schema gates on
`schema_version`, and the only existing manifest already declared `1.1.0` and
already populated both quantization fields. It gets dearer per capture, which is
why this was done ahead of the record unification rather than behind it.

### `vae_quantization` is deliberately not required

It is singular, and a reference graph loads two VAEs at different quantizations --
`minimax_h3_video_vae_int8_convrot` beside `minimax_h3_audio_vae_fp32` in
`workflows/h3_probe_capture_ref3_api.json`. One value over two files records
something true of neither, and the existing manifest demonstrates it: it says
`int8_convrot`, which is right about the video VAE and wrong about the audio one.

Requiring a singular field over a plural reality is harder to walk back than not
requiring it, so it stays optional until it is either split per-VAE or dropped.

### `weight_quantization` is a projection, and is checked as one

`int8_convrot` is readable straight off
`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, so the field is a second home
for a fact the required `unet` filename already carries. That is only honest if
something asserts the two agree, which `bench/check_capture_manifest.py` now does
-- the precedent is `filing.md`'s rule that an `artifacts` list is a projection of
the citations and a disagreement means one of them is wrong. Without the
assertion the field could read `fp8_scaled` over an int8 filename and both would
pass.

The filename wins any disagreement: it names the file that was actually loaded.

## Where a fact belongs: substrate versus payload

Settled 2026-08-17 while designing the shared substrate block. The seam is what
each half can honestly measure, and it is a sharper rule than it looks:

- **The substrate block carries point-in-time facts about the machine and the
  software.** It is produced by one call at record time.
- **A payload carries whatever its kind can measure over its own duration.** A
  bench run spans a render and can sample across it; a capture cannot.

**The worked case that produced the rule.** GPU clock locks (`-lgc`, `-lmc`) are
not readable on this driver — no `--help-query-gpu` field exposes them, and
`nvidia-smi -q -d CLOCK` answers "Requested functionality has been deprecated".
An emitter asked for the lock state can only guess.

Two wrong answers were considered before the right one:

1. **Require the field and let a guess be written as `null`.** Rejected: `null`
   here means *confirmed absent*, so a guess would be recorded as a confirmation.
   That is the absent-versus-unobservable collapse these fields exist to close,
   reintroduced by the field meant to close it.
2. **Record a clock sample instead of the lock setting.** Also rejected, and this
   is the subtler one. `substrate()` is called once. One sample describes a
   millisecond of a render lasting minutes, sitting in a field named as though it
   characterised the run — a decorative number wearing a descriptive name, and
   worse than an honest gap because it looks like an answer.

So the substrate block records the lock as **`unobservable`**, a first-class state
alongside recorded and confirmed-absent. It keeps the record that the question was
asked and could not be answered, which is the not-covered-versus-correctly-absent
distinction `CLAUDE.md` names.

Clock behaviour under load is worth measuring — it measures the thing rather than
the setting supposed to cause it — and it belongs in the **bench payload**, where
min, max and time-at-cap across a run are meaningful. Not here.

### Four states, then, not three

| state | meaning | verdict |
|---|---|---|
| key absent | not recorded | fails, once required |
| `null` | confirmed absent — no lock set, weights unquantized | passes |
| `"unobservable"` | asked, cannot be answered on this driver or platform | passes |
| a value | recorded | passes |

---

# Step two: the discriminated run record

Drafted 2026-08-17 against a **regenerated** key set, not a pasted one:

    python substrate.py --keys workflows/h3_probe_capture_ref3_api.json

Pass a real API-format graph. Without one the emitter warns and reports `graph`
and `weights` as unobservable, so their subtrees are absent — six leaf paths — and
a schema drafted from that output would validate a shape nothing emits while
missing two whole subtrees, both sides green. Regenerate rather than trust any
list, including the one below.

## Shape

One record, a `kind` discriminator, a shared substrate block, a per-kind payload.
Required-ness is conditional on `kind`, which is what lets the capture variant keep
every requirement it has today while the bench variant never mentions tensors.

```json
{
  "record_version": 2,
  "kind": "capture" | "render" | "bench",
  "timestamp": "<ISO-8601>",
  "substrate": { "...": "verbatim from substrate.py" },
  "payload":   { "...": "per kind" }
}
```

Always required: `record_version`, `kind`, `timestamp`, `substrate`, `payload`.

## The substrate block's requirements are structural

Each group carries a `state`, and **the group-level `state` keys are always
required** — they are the four-state discriminator, so their absence is the one
thing that cannot be recovered later. The subtrees are conditional on the state:

| group `state` | what else must be present |
|---|---|
| `present` | that group's own subtree |
| `absent` | nothing — confirmed absent is a complete answer |
| `unobservable` | `why`, so the reason survives the record |

Required unconditionally: `substrate_version`, `host.state`, `builds.state`,
`graph.state`, `weights.state`.

**`weights` requires nothing below `state`, deliberately.** It infers
quantization and rank from filename shape — the emitter says so in the field
names, `quantization_inferred_from_filename` and `rank_inferred_from_filename`,
which is the evidence kind stated inside the claim rather than beside it. It is
the weakest part of the block and expected to move, so requiring anything under
it now would be requiring a shape we already know is wrong.

**Updated 2026-08-17:** this said the emitter "cannot recover LoRA strength or
rank". It recovers both now. Strength comes from `node_settings`, which captures
every literal input on a node that loads a weight file — generic rather than
named, so `strength_model` is recorded without anyone hardcoding it, and
`weight_dtype` and CLIP `type` came along in the same pass. Rank is inferred
from the filename and is `null` when the filename does not say.

That `null` is why `rank` is typed `["integer", "null"]` here rather than
`integer`. `bench/generate_capture_manifest.py` wrote a literal `256` for every
LoRA until the same day — correct for the one LoRA this repo ships, silently
wrong for any other, and a fabricated value is worse than an absent one because
it validates.

## Payloads by kind

**`capture`** keeps today's requirements unchanged: `workflow_file`, `canvas`,
`models`, `sampling`, `attention`, plus prompt, references, token accounting and
the captured tensors with their checksums. Nothing in the capture variant is
relaxed by unification — that is the point of the discriminator.

**`render`** is a capture without tensors: the graph ran and produced output, and
no activations were kept.

**`bench`** is not specified here. It is step three, it depends on the owner's call
that `bench_e2e_h3.py` must persist at all, and per the substrate-versus-payload
seam it is where **sampled** clock behaviour belongs — min, max and time-at-cap
across a run, which are meaningful where the substrate block's single point-in-time
sample would not be.

## Two names go wrong when the bench kind lands

`docs/capture_manifest_schema.md` and `bench/check_capture_manifest.py` are both
named for one kind of a record that will carry three. The rename waits for step
three rather than churning citations ahead of the thing that needs it —
`check_doc_links.py` will go red on the move, which is the desired behaviour and
the reason the citations were put in `path::symbol` form.
