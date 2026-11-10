# The command line, the files and the bank

last updated: 2026-11-10

[Back to the start page](../control_vectors.md)

## The commands

The tool is `bench/control_vectors.py`. Its own help is the reference, for the commands and for every flag
of each:

```bash
<comfy-venv-python> bench/control_vectors.py --help
<comfy-venv-python> bench/control_vectors.py <command> --help
```

Run it with the ComfyUI venv's python, never `uv` (`docs/comfy_notes.md`, "Running things"). A build loads
the text encoder on the card, so run it when no render holds the card.

This table says what each command is for and where its verified record lives. A flag table is not kept
here: the last one drifted from the parser.

| Command | For | Recorded verification |
|---|---|---|
| `build` (also `from-toml`) | A vector from a pair file, or every file in the bank | Yes — all shipped `.toml` bank vectors (`Arms 01..52` in [`2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md), plus `film_16mm_1974` / `film_16mm_1974_neutral` on 2026-11-10) |
| `from-prompt` | A vector from your own prompt and `old -> new` clauses (plus optional `--bank-scenes <K>` and `--ref-images`) | Yes — tested on a reference picture on 2026-11-09; `--bank-scenes` covered in `bench/check_control_vectors.py` |
| `rebuild` | Vectors made again from the pairs recorded in their own files | Yes — single-vector and folder modes verified in `bench/check_control_vectors.py` |
| `inspect` | A vector's metadata, J-Lens words, and token span masks (`--mask`) | Yes — verified across all 3-lever `--mask` modes |
| `make` | A vector from a pair file or from a prompt and clauses; an older twin of `build` / `from-prompt` | Covered in CLI parser checks; `Arms 55..56` tested compiled `--words` |
| `make-media` (also `from-images`) | A vector from two pictures | Covered in `bench/check_control_vectors.py`; no rendered video arm on record |
| `words` (also `from-words`) | A vector from a few words, through J-Lens, on the CPU (`< 2 ms`) | Yes — 7 rendered arms (`Arms 53, 54, 61..65` in [`2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md)) |
| `sculpt` | A new vector from existing ones: added, held out, inverted | Covered in `bench/check_control_vectors.py`; runtime `mode="hold"` rendered in `Arm 58` |
| `grade` | How far a vector moves post-refiner rows and DiT velocities (`--mask`, `--pika`, `--sol`) | Yes — text prompts graded on GPU (`2026-11-05` & `2026-11-10` under `--pika --sol --mask scene_only`); `--ref-images` VAE DiT probe path not yet recorded |
| `from-md` (also `import-md`) | An LLM's Markdown answer turned into a pair file | Covered in `bench/check_control_vectors.py` |
| `generate-pairs` | Pairs written by a local LLM | Not run on record; see the LLM page |
| `build-j-lens` | The J-Lens bundle checked (`--verify-only`), or its refiner half rebuilt (requires `--overwrite` to replace an existing file) | `--verify-only` verified; see the J-Lens page before rebuilding |
| `ui` | The browser UI | Not maintained at present |

The plan is to cut this list down to building from a prompt and rebuilding from a file's own record
([`the plan`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md), section 19).
That has not been done.

## Other tools

- `bench/control_vector_rows.py`: caches the text rows of the bank's prompts and pair files for study on
  the CPU; `--ref-image` encodes prompts on top of one picture.
- `bench/control_vector_pooling.py` and `bench/control_vector_lexicon.py`: the pooling comparison and the
  comparison of two ways to name a vector in words, both over that cache.
- `bench/edit_guidance_arms.py`: builds and queues comparison arms on one shot, including an arm that
  applies a vector (`--vector-arm`).

## The files a build writes

A build writes into the models folder's `control_vectors/`, one file per family:

```text
<name>__<family>__<pooling>__lora_<hash>.safetensors
<name>__<family>__<pooling>__lora_<hash>.md
```

- `<family>` is `fl2va` or `ref2va`.
- `<hash>` identifies the LoRA recipe the refiner ran under. A build with a reference picture adds a
  `__ref_<hash>` part before it.
- The `.safetensors` holds the direction after the refiner and a unit copy before it, with metadata that
  records the pairs, so `rebuild` needs nothing else.
- The `.md` beside it is a report for a reader: norms, the recommended strength for each mode, how well the
  pairs agree, the J-Lens words.

The shipped graphs name vector files by these full names. A name that no longer exists on the server is
caught by `bench/check_model_files.py`, which needs a running server.

## The bank

`prompt_bank/control_vectors/` holds the pair files the starter vectors are built from. Listed from the
folder, with no claim per file:

```text
auctioneer_rapid_patter   balloon_head_inflate  bioluminescent_glow    chaos_panic
claymation_stopmotion     crystallize_ice       cursed_cctv            cyberpunk_neon_rain
elastic_neck_stretch      felt_muppet           fever_melt             film_16mm_1974
film_16mm_1974_neutral    handheld_documentary_zoom head_swell         infomercial_fail
lettering                 magnetic_attraction   miniature_tiltshift    molten_lava_golem
noir                      operatic_bellow       paper_origami          pyro_chain_explosions
pyro_klaxon_audio         reverb_cathedral      rubber_hose_1930s      smoke_dissolve
strobe_club_pulse         sunken_coral_reef     thermal_infrared       underwater_submerged
vhs_1991_camcorder        vocal_strain          wes_anderson_symmetry  whisper_asmr
wind_tunnel_gale          zero_gravity
```

`ls prompt_bank/control_vectors/` is the authority; this list is a copy made on the date above.

The bank is kept as a starter set and is not being grown (owner, 2026-11-09, [`../decisions.md`](../decisions.md)).
Every file is a text-only prompt with no picture. Which of these vectors do what their names say on a
render has been looked at for only a few; the record is
[`../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md`](../../../bench/results/2026-11-05_control_vectors_pika_and_ref2va.md).
