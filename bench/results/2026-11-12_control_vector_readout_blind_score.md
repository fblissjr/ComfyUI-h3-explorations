# Control-vector readouts scored blind: J-Lens words and the rows table, three runs (2026-11-12)

lane: control

Date: 2026-11-12

The first run of experiment 1 of
[`docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md`](../../docs/research/control_vectors/2026-11-12_chain_lens_digest_and_experiment_plan.md)
(section 11). A judge that sees only a vector's readout picks that vector's concept out of five. The judging
format and the controls are WorkspaceBench's (the same page, section 8.1).

All three runs are throwaways. Their scripts have no check and live beside their outputs in the untracked
folders `data/control_vectors/readout_score_2026-11-12/` (sections 1 to 5),
`data/control_vectors/readout_score_2026-11-12_nearest/` (section 6) and
`data/control_vectors/readout_score_2026-11-12_groups/` (section 7): `build_packets.py` and `score.py` in
each. The counts below are what `score.py` printed; `results.json` in each folder holds them.

The three runs share one set of readouts and differ only in the options. Random options were too easy
(sections 1 to 5). Each vector's nearest concepts gave room but leaked the target to a judge with no readout
(section 6). One shared list per group of near concepts keeps the room and closes the leak (section 7); that
is the form to carry forward.

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

## 6. Second run: the nearest concepts as options

Run the same day, at the owner's word, to give a better reader room to show.

### 6.1 What changed

- **Readouts.** None were recomputed. Every word list was copied from the first run's packets, so the two
  runs differ only in the options and in the `shuffle` source.
- **Options.** The target's `quality` line and those of its four nearest other vectors, by cosine of
  `vector.post_refiner`. `key.json` in the run folder records each cosine.
- **`shuffle`.** The readout of the fifth-nearest vector: the nearest concept that is not among the options.
- **Judges.** As before, with new packets. Every packet came back complete.

### 6.2 Counts

Options only, no readout: the target was picked for 23 of 46 vectors.

| Reader | Vectors | `real`: target | `real`: cannot tell | `real`: another option | `twin`: target | `shuffle`: target |
|---|---:|---:|---:|---:|---:|---:|
| `jlens` | 46 | 31 | 12 | 3 | 0 | 2 |
| `rows` | 46 | 21 | 9 | 16 | 0 | 3 |

On `twin` items the judges answered "cannot tell" for 45 of 46 with each reader. The two readers on the same
vector, `real` items: both named it for 17, `jlens` only for 14, `rows` only for 4, neither for 11.

### 6.3 The option lists leak the target

A judge given the options and nothing else picked the target for half the vectors, where a uniform pick
expects about 9 of 46. The benchmark's leakage standard (the plan page, section 8.1) asks that such a guesser
stay close to uniform, so this option design fails it. The judge reported leaning on which options looked
like near-duplicates of each other: the target is the one option the other four were all chosen to resemble.

The readout judges did not use this. With a random direction's words in front of them they picked the
target for no vector. So the `real` counts above are not inflated by the leak, but a harness that ever forces
a pick would be.

The repair is to give every member of a group the same list: groups of mutually near concepts, one option
list per group, each member the target once. A list then says nothing about which member is asked for.

### 6.4 Where the misses went

`jlens`, another option (three vectors, all to `vhs_1991_camcorder`):

| Vector | Words quoted for the pick |
|---|---|
| `cursed_cctv` | corrupted, outdated, unstable, shaky |
| `film_16mm_1974` | tape |
| `film_16mm_1974_neutral` | amateur, shaky |

`jlens`, cannot tell (twelve): `auctioneer_rapid_patter`, `bioluminescent_glow`, `claymation_stopmotion`,
`elastic_neck_stretch`, `felt_muppet`, `pyro_chain_explosions`, `pyro_klaxon_audio`, `summer_beachwear`,
`underwater_submerged`, `wes_anderson_symmetry`, `wind_tunnel_gale`, `zero_gravity`.

`rows`, another option (sixteen). Eight went to the nearest distractor. The picks were made on words that
belong to the neighbor: `underwater_submerged` was read as `bioluminescent_glow` from the pieces "bi",
"olum", "ines", "cent"; `cursed_cctv` and both film vectors as `vhs_1991_camcorder` from "HS", "-video",
"tape", "cassette"; `claymation_stopmotion`, `felt_muppet` and `wes_anderson_symmetry` as `rubber_hose_1930s`
from "cartoon", "pie", "ose".

`shuffle` items that went to the target: `balloon_head_inflate`'s readout was taken for `head_swell` with
both readers; `operatic_bellow`'s for `pyro_klaxon_audio` (`jlens`); `handheld_documentary_zoom`'s for
`chaos_panic` and `auctioneer_rapid_patter`'s for `wind_tunnel_gale` (`rows`).

### 6.5 What the second run cannot see

Everything in section 4 still holds, and one thing more:

- **Whether a miss is the reader's or the vector's.** The film and VHS vectors have a high cosine with each
  other (`key.json`), and `2026-11-06_j_lens_and_refiner_jacobian.md` section 5.2 already records the 16mm
  direction mixing in tape. A reader that reports a vector faithfully will confuse two vectors that overlap.
  This harness scores the vector and the reader together.

### 6.6 Readings, not findings

- With near concepts as options the `jlens` words leave fifteen vectors unnamed, so there is room for a
  better reader. The misses fall in a few groups: old video (16mm, VHS, CCTV), puppet and cartoon looks,
  loud chaotic energy, and underwater or glowing scenes.
- The rows table does not just say less than `jlens` here; it says the neighbor's words. That is the
  borrowing the simplification plan's section 20.1 describes, counted.
- `head_swell` and `balloon_head_inflate` are written as nearly the same concept and their readouts pass for
  each other, yet the second is only the fifth-nearest vector to the first. The words the two share are a
  small part of each vector.

## 7. Third run: one shared option list per group of near concepts

Run the same day, at the owner's word, as the repair section 6.3 describes.

### 7.1 What changed

- **Readouts.** Again copied word for word from the first run.
- **Vectors.** `film_16mm_1974_neutral` is left out as a variant of `film_16mm_1974`, which leaves a count
  that divides by five.
- **Groups.** Groups of five by a greedy rule on the cosine of `vector.post_refiner`: the closest unassigned
  pair seeds a group, and the unassigned vector with the highest mean cosine to the group joins until it
  holds five. `key.json` in the run folder lists the groups with their mean and lowest cosine. The rule
  makes the early groups tight and leaves the last ones with what remains.
- **Options.** Every member of a group is judged against the same five `quality` lines, in an order drawn
  per item. Each line is the target for exactly one member.
- **`shuffle`.** The readout of the nearest vector outside the group.

### 7.2 Counts

Options only, no readout: the target was picked for 9 of 45 vectors, one in each group. The judge reported
picking the same line each time a list came back, which is what the design allows: that can be right for one
member only.

| Reader | Vectors | `real`: target | `real`: cannot tell | `real`: another option | `twin`: target | `shuffle`: target |
|---|---:|---:|---:|---:|---:|---:|
| `jlens` | 45 | 33 | 9 | 3 | 0 | 7 |
| `rows` | 45 | 23 | 13 | 9 | 0 | 9 |

The two readers on the same vector, `real` items: both named it for 19, `jlens` only for 14, `rows` only for
4, neither for 8.

By group, `real` items that went to the target, of five:

| Group (mean cosine in `key.json`) | Members | `jlens` | `rows` |
|---|---|---:|---:|
| 0, tight | `pirate_crew`, `wild_west_saloon`, `medieval_feast`, `circus_clowns`, `disco_fever` | 5 | 3 |
| 1, tight | `film_16mm_1974`, `vhs_1991_camcorder`, `cursed_cctv`, `rubber_hose_1930s`, `claymation_stopmotion` | 2 | 1 |
| 2, tight | `chaos_panic`, `pyro_chain_explosions`, `wind_tunnel_gale`, `vocal_strain`, `pyro_klaxon_audio` | 2 | 2 |
| 3, tight | `bioluminescent_glow`, `crystallize_ice`, `magnetic_attraction`, `cyberpunk_neon_rain`, `underwater_submerged` | 4 | 1 |
| 4 | `arctic_expedition`, `astronaut_crew`, `summer_beachwear`, `molten_lava_golem`, `sunken_coral_reef` | 4 | 3 |
| 5 | `fever_melt`, `smoke_dissolve`, `strobe_club_pulse`, `infomercial_fail`, `handheld_documentary_zoom` | 3 | 3 |
| 6 | `felt_muppet`, `miniature_tiltshift`, `paper_origami`, `elastic_neck_stretch`, `thermal_infrared` | 4 | 2 |
| 7, loose | `auctioneer_rapid_patter`, `operatic_bellow`, `whisper_asmr`, `wes_anderson_symmetry`, `balloon_head_inflate` | 4 | 3 |
| 8, loose | `head_swell`, `zero_gravity`, `reverb_cathedral`, `lettering`, `noir` | 5 | 5 |

### 7.3 Where the misses went

- `jlens`, another option (three, all in group 1): `film_16mm_1974` and `cursed_cctv` to `vhs_1991_camcorder`,
  `claymation_stopmotion` to `rubber_hose_1930s`. In group 2 its three misses are all "cannot tell".
- `rows`, another option (nine): the same three, plus `vhs_1991_camcorder` to `cursed_cctv`,
  `bioluminescent_glow` to `crystallize_ice`, `underwater_submerged` to `bioluminescent_glow`, `chaos_panic`
  and `vocal_strain` to `wind_tunnel_gale`, and `wild_west_saloon` to `medieval_feast`.
- `shuffle` items that went to the target are near concepts that fell in different groups. With `jlens`:
  `underwater_submerged` and `sunken_coral_reef` each for the other, `operatic_bellow` for
  `reverb_cathedral`, `felt_muppet` for `rubber_hose_1930s`, `miniature_tiltshift` for
  `wes_anderson_symmetry`, `pyro_chain_explosions` for `molten_lava_golem`, `auctioneer_rapid_patter` for
  `strobe_club_pulse`.

### 7.4 Across the three runs

The same `jlens` readout named 29 vectors in all three runs. Ten were named in neither of the two harder
runs: `bioluminescent_glow`, `claymation_stopmotion`, `cursed_cctv`, `elastic_neck_stretch`,
`film_16mm_1974`, `pyro_chain_explosions`, `pyro_klaxon_audio`, `summer_beachwear`, `wes_anderson_symmetry`,
`wind_tunnel_gale`.

### 7.5 What the third run cannot see

Sections 4 and 6.5 still hold. Two more:

- **A judge sees a list more than once in a packet.** Each packet holds one item per vector, so a group's
  list comes back five times, with different words and in a different order. The judges were told that this
  says nothing about any one item. The twin items, where the right answer is "cannot tell", make "each line
  is used once" a poor rule to follow; nothing measured whether a judge tried it.
- **How much the counts would move on a repeat.** Every item was judged once.

### 7.6 Readings, not findings

- This form does what the first two did not: a guesser with no readout lands on chance by construction, and
  the `jlens` words leave twelve vectors of room.
- The misses are concentrated. Two tight groups, old video with cartoon looks and loud chaotic energy, hold
  six of the twelve `jlens` misses. Where the vectors have little to do with each other the words name all
  of them.
- The `shuffle` control now reads as a count of near concepts whose readouts pass for each other.
