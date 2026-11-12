# Control-vector readouts scored blind: J-Lens words and the rows table, first run (2026-11-12)

lane: control

Date: 2026-11-12

The first run of experiment 1 of
[`docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md)
(section 11). A judge that sees only a vector's readout picks that vector's concept out of five. The judging
format and the controls are WorkspaceBench's (the same page, section 8.1).

This run is a throwaway. Its two scripts have no check and live beside their outputs in the untracked folder
`data/control_vectors/readout_score_2026-11-12/` (`build_packets.py`, `score.py`). The counts below are what
`score.py` printed; `results.json` in that folder holds them.

## 1. What was run

- **Vectors.** Every `*__fl2va__diff_span__lora_*.safetensors` in `<comfy>/models/control_vectors/` that has a
  pair file under `prompt_bank/control_vectors/` or `data/control_vectors/`. `key.json` in the run folder lists
  each vector file and pair file.
- **Readers.** Eight words per pole from each:
  - `jlens`: `j_lens.py::decode_vector` on `vector.post_refiner`, with the bundle
    `<comfy>/models/j_lens/j_lens_h3.safetensors` as it stood on disk (file dated 2026-11-06).
  - `rows`: the rows table of `bench/control_vector_lexicon.py` (the same arithmetic, copied into
    `build_packets.py`), from the row cache `data/control_vectors/rows_2026-11-09` (built at `402c074a`), each
    concept's own pair file left out.
- **Options.** The target's `quality` line from its pair file and four other vectors' `quality` lines, drawn at
  random with a fixed seed, in a seeded order. A vector whose name extends another's is never its distractor.
  The judge may also answer "cannot tell".
- **Items per vector and reader**, all against the same option list:
  - `real`: the vector's own readout.
  - `twin`: the readout of a random direction of the same norm.
  - `shuffle`: another vector's readout, one whose concept is not among the options.
- **Floor.** One options-only item per vector: the five options and no readout, judged by a judge that saw no
  readout at all.
- **Judges.** Sonnet subagents of the Claude Code session that ran this, each given one packet and the
  instructions in `judge_instructions.md` (or `judge_instructions_options_only.md`) and nothing else. Six
  readout packets, each holding one item per vector, so the three variants of a vector's readout were never
  judged side by side. Every pick had to quote words from the readout.
- **One repair.** One judge skipped one item of its packet. A fresh judge was given that item alone.
- **Repo commit at the run:** `a5383b6c`. CPU only; no model was loaded and nothing was rendered.

## 2. Counts

Options only, no readout: the target was picked for 11 of 46 vectors. A uniform pick over five options expects
about 9.

| Reader | Vectors | `real`: target | `real`: cannot tell | `real`: another option | `twin`: target | `shuffle`: target | `real` target and `twin` not |
|---|---:|---:|---:|---:|---:|---:|---:|
| `jlens` | 46 | 41 | 4 | 1 | 1 | 1 | 40 |
| `rows` | 46 | 35 | 10 | 1 | 0 | 1 | 35 |

Every `real` pick of the target carried a quote found in the readout. On `twin` items the judges answered
"cannot tell" for 45 of 46 (`jlens`) and 43 of 46 (`rows`).

The two readers on the same vector, `real` items:

| Both name it | `jlens` only | `rows` only | Neither |
|---:|---:|---:|---:|
| 32 | 9 | 3 | 2 |

Split by whether the vector's pair file is in the row cache (the cache predates the newer vectors):

| Reader | Pair file in the cache: target | Not in the cache: target |
|---|---:|---:|
| `jlens` | 27 of 32 | 14 of 14 |
| `rows` | 22 of 32 | 13 of 14 |

## 3. The items that did not go to the target

`jlens`, `real` (five vectors):

| Vector | Pick | Top plus words | Note |
|---|---|---|---|
| `auctioneer_rapid_patter` | cannot tell | frantic, dazzling, exhilarate, jittery, rapid | no word for speech or a voice in either pole; `handheld_documentary_zoom` was an option |
| `bioluminescent_glow` | `cyberpunk_neon_rain` | hologram, psychedelic, dazzling, neon, fractal | the picked option was the nearest distractor by vector cosine |
| `claymation_stopmotion` | cannot tell | cartoon, tiny, exaggerated, puppet, whimsical | no word for clay or stop-motion in either pole |
| `elastic_neck_stretch` | cannot tell | dizzy, looming, labyrinth, slippery, mesmerism | no word for a neck or for stretching in either pole |
| `zero_gravity` | cannot tell | glitch, psychedelic, floating, hallucinate | `sunken_coral_reef` was an option, and "underwater" was in the plus words |

`rows`, `real`: ten "cannot tell" (`auctioneer_rapid_patter`, `bioluminescent_glow`, `chaos_panic`,
`crystallize_ice`, `magnetic_attraction`, `molten_lava_golem`, `paper_origami`, `smoke_dissolve`,
`underwater_submerged`, `wes_anderson_symmetry`) and one other option (`infomercial_fail`, read as
`wind_tunnel_gale` from "storm" and "violently").

Controls that hit the target: one `jlens` twin (`film_16mm_1974_neutral`, on "grain"), one `jlens` shuffle
(`infomercial_fail`'s readout judged as `rubber_hose_1930s`, on "cartoon" and "hilarious"), one `rows`
shuffle (`operatic_bellow`'s readout judged as `pyro_klaxon_audio`, on "roar" and "roaring").

## 4. What this run cannot see

- **Whether a render shows the concept.** It scores naming, not effect.
- **Anything about a reader better than `jlens`.** With random distractors, `jlens` leaves five vectors of
  room. The distractors were mostly far from the target: `key.json` records each one's cosine with the
  target vector. A reader that is truly better would have little room to show it here.
- **Judge dependence.** One model family judged, once. No second judge, no repeat.
- **Leakage inside a packet.** A judge saw 46 items at once. It was told to judge each alone; nothing enforced
  that.
- **How much the word list was fitted to this bank.** The J-Lens word map was edited while the first vectors'
  readouts were being read (`17593beb`, 2026-11-05). The pair files that are not in the row cache were all
  added after the bundle file's date (`git log --diff-filter=A` on each, and the file dates under
  `data/control_vectors/`), and `jlens` named all of those.
- **Whether the `rows` gap is the method or the cache.** The rows table knows only words the cached prompts
  contain, and it returns token pieces.

## 5. Readings, not findings

- Both readers name the concept far above the options-only floor, and almost never on a random direction or
  on another vector's readout. The controls behaved.
- Three of the five `jlens` misses are concepts whose defining term the single-token word list cannot hold
  (bioluminescent, claymation or plasticine, an elongating neck). Five is too few to call a pattern. It is the
  case experiments 3 and 5 of the plan are aimed at.
- Before this harness can rank a new reader against `jlens`, the option lists need to be harder: the nearest
  concepts by vector cosine as distractors, the way the benchmark uses same-category distractors.
