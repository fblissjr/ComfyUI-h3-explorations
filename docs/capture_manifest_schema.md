# Activation Capture Manifest Schema

**The accepted version set is `bench/check_capture_manifest.py::SCHEMA_VERSIONS`, and what a new manifest is stamped with is `bench/generate_capture_manifest.py`.** This document describes the shape; those two own the number. A version written into prose here drifted against the code once already, which is why the constant exists.

Every activation capture directory under `$H3_CAPTURE_ROOT/` must contain a complete, auditable `manifest.json` alongside the `.pt` tensor files.

The manifest makes a captured tensor traceable to the prompt, reference images and their checksums, model checkpoints and their quantizations, canvas geometry, sampling parameters, token accounting, and the host substrate that produced it.

**It records nothing about error decomposition, per-head or otherwise.** That is a property of the analysis run over a capture, not of the capture, and it is not recoverable from this file. If you need to know which heads were measured or how, that lives in the analysis invocation -- see `bench/analyze_sol_error.py`, whose `--heads` default measures a subset.

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
    "schema_version": { "type": "string", "enum": ["1.0.0", "1.1.0", "1.2.0"] },
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
                  "strength": { "type": ["number", "null"] },
                  "rank": { "type": ["integer", "null"] },
                  "loader": { "type": "string" },
                  "pdd_patch_heads": { "type": ["boolean", "null"] },
                  "low_vram": { "type": ["boolean", "null"] }
                },
                "description": "One entry per REACHABLE loader node, across every class in h3_config.LORA_LOADER_CLASSES. `strength` is nullable because a linked widget is computed upstream and unknowable from the graph -- null there is the honest value and a number would be invented. `loader` names the class, because the three do not mean the same thing: the pack node merges or bypasses per `low_vram`, and `pdd_patch_heads` distinguishes a PDD arm running its parallel heads from the control that is not, which nothing else in this record would show."
              }
            },
            "sha256": {
              "type": "object",
              "description": "since 1.2.0; per-model file digest, or a reason string beginning 'unresolved:' or 'unreadable:'. `_unavailable` alone means generation could not reach folder_paths.",
              "properties": {
                "unet": { "type": "string" },
                "clip": { "type": "string" },
                "video_vae": { "type": "string" },
                "audio_vae": { "type": "string" },
                "loras": { "type": "array", "items": { "type": "string" } },
                "_unavailable": { "type": "string" }
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
            "sol_start_percent": { "type": ["number", "null"] },
            "sol_end_percent": { "type": ["number", "null"], "description": "the Sol window's upper bound. STEP-COUNT DEPENDENT: a fixed band covers a different fraction of an 8-step run than a 16-step one, so this is derived per graph and two renders differing only in it are different experiments." },
            "sol_dense_blocks": { "type": ["string", "null"], "description": "DiT blocks forced dense, NVLabs ship '0-1' on every H3 config they publish." },
            "sol_tau": { "type": ["number", "null"] },
            "sage_mode": { "type": "string", "description": "the wired sage node's mode, or the literal 'absent' / 'orphaned' when no sage node runs. It was a hardcoded 'fp16 (most accurate)' until 2026-08-26, so a render with no sage at all reported a mode for a kernel that never ran -- the same defect this file records for sol_attn, in the same dict literal." },
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

### `models.sha256` records bytes, not the loaded model (1.2.0)

Every other model field is a **filename**, and `weight_quantization` and `rank`
are read back out of one. A name is what the graph asked for. It is not evidence
of what the loader read: a file replaced in place leaves every one of those
fields identical, and the manifest keeps asserting the old identity. Reference
media and the captured tensors were already hashed in this same generator; the
models were the gap.

**What the hash cannot see, stated because a silent limit reads as coverage.**
It pins which bytes sat on disk, not what the sampler ended up holding. A LoRA
at strength 0.75 and the same LoRA at 1.0 hash identically. So does a checkpoint
under a quantization applied after load, and so does any runtime patch. Reading
the live weights instead would close that, but a sampled digest over a few
tensors of a 32B checkpoint is not a sound way to do it and would read as a
stronger claim than it is. This is the file-level half, and it is honest about
being only that.

A model that cannot be resolved or read is recorded as a reason string
(`unresolved: ...`, `unreadable: ...`) rather than omitted, and generation that
could not reach `folder_paths` writes `_unavailable`. An absent hash and an
unhashed model must not look alike, which is the same rule the substrate keys
follow: null means confirmed absent, missing means nobody wrote it down.

`bench/check_capture_manifest.py::assert_model_hashes` keys off the names the
manifest itself carries rather than a fixed list, so a manifest naming a model
this repo has never heard of still has to account for it. Gated on
`schema_version` >= 1.2.0, so the 1.1.0 manifests already on disk conform to the
version they declare and are not failed for a field that did not exist.

### What was rendered, and a capture with no references (1.5.0)

Added 2026-09-03 on the first Base16 text-to-video capture, whose first
manifest claimed a 1024x768 canvas for a 1344x768 render, no text encoder,
no seed, and a token total that disagreed with the capture's own files. Four
reads in the generator had met exactly one implementation (the ref3 capture
graphs) and none of this pack's current nodes: `MiniMaxH3Resolution` carries
`shape`, `shape.<shape>_resolution` and `length`, not `width`/`height`;
`MiniMaxH3EncoderLoader` is the encoder loader every shipped graph wires;
the seed lives on `RandomNoise`; and text and audio rows were typed
constants. All four now derive from the graph or the files, and the
generator refuses to write when the derived text rows go negative, which is
what a wrong canvas produces.

New keys, all required from 1.5.0 and checked by
`bench/check_capture_manifest.py`:

- `prompt.bank_id` (null for a prompt not in `prompt_bank/`; the full text
  is then the only copy) and `prompt.prompt_sha256` -- the owner's rule that
  every record says what was rendered, joined through
  `workflows/prompts.py::describe`.
- `workload.workflow_sha256` (the file's bytes) and `workload.graph_sha256`
  (the canonical hash `provenance.py` and the Sol route record use), so a
  manifest naming a workflow FILE that has since been regenerated still says
  which graph it was.
- `workload.task`: the render type (`t2va`, `i2va`, `fl2va`, `l2va`,
  `ref2va`) by the conditioner's sockets, the rule the prompt graders use;
  required. A manifest that named the DiT and the sidecar but not whether
  the render was a text-to-video or a reference one was the owner's first
  question of the 1.5.0 output.
- `provenance.server`: the launch flags and versions of the process that
  wrote the capture, stamped into every record by `h3_capture.py` from
  2026-09-03 and copied here; null for older captures. `--fast
  fp16_accumulation` changes the numerics a capture holds, and nothing
  offline can recover which mode the server ran in.
- `captured_tensors[].sigma`, `kernel`, `render` and `segments`: the
  record's own top-level scalars, read through a memory map rather than by
  paging in the tensors. `segments` is null when the capture ran with Sol
  absent, because Sol's rope hook is what publishes the table; a consumer
  that needs sink ranges must derive them from geometry, and the tau arm in
  `bench/measure_sol_exact_variants.py` says so.

And one assertion that inherited a third case: `references` may be empty,
but only when `token_accounting.reference_tokens` is zero. A text-to-video
capture has no references and that is a state, not a defect.

### `vae_quantization` is deliberately not required

It is singular, and a reference graph loads two VAEs at different quantizations --
`minimax_h3_video_vae_fp16` beside `minimax_h3_audio_vae_fp32` in
`workflows/h3_probe_capture_ref3_api.json`. One value over two files records
something true of neither, and the existing manifest demonstrates it: it says
`int8_convrot`, which described the video VAE that graph loaded when it was
captured and never described the audio one.

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
