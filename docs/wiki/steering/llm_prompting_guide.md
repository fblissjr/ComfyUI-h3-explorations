# Authoring Contrastive Pairs with an LLM

Last updated: 2026-11-21 (added Rule 6: The Pole-A Coverage Law for rank-8 `erasure.vr`).

You can author a validated 6–10 pair `.toml` control-vector pack with any frontier chat LLM (copy-paste workflow) or with a local GPU LLM via `bench/control_vectors.py generate-pairs`.

---

## 1. The two canonical system prompts

| Prompt file | When to use | Output format |
|---|---|---|
| [`docs/wiki/system_prompts/h3_control_vector_standalone_system_prompt.md`](../system_prompts/h3_control_vector_standalone_system_prompt.md) | **Copy-paste into any external chat UI** (Claude, Gemini, ChatGPT, or a local Heylook/OpenWebUI tab) with zero repo context needed | Self-contained Markdown (`### Test Scene Prompt`, `#### Pair 1..10`, optional `#### Hold 1..K`) ready for `bench/control_vectors.py from-md` or `build <file.md>` |
| [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) | Loaded automatically by `bench/control_vectors.py generate-pairs` (and documented with architecture links) | Same validated Markdown contract |

---

## 2. Six rules that decide whether LLM-authored pairs work or fail

Every rule below comes directly from measured failure modes in our 2026-11-05..2026-11-21 builds:

1. **One coherent physical/optical/acoustic mechanism across all pairs—never a grab-bag of different nouns**:
   - *Good*: Every pair in `claymation_stopmotion` turns a different scene (kitchen, street, workshop, harbor) into hand-sculpted plasticine clay with thumbprints and 12-fps stepped motion. Scene nouns cancel; plasticine physics reinforces.
   - *Bad (`wardrobe_1970s`, 2026-11-11)*: Pair 1 targets a mustard blouse, Pair 2 bell-bottoms, Pair 3 a suede vest, Pair 4 a plaid jacket, Pair 5 a floral wrap dress. Averaging them cancels every garment noun into abstract `"vintage, psychedelic, nostalgic"` mood words that tint lighting and never change a character's clothes. (For an exact outfit on a character, skip pair building and use [`MiniMaxH3EditGuider`](../edit_guidance.md).)
2. **Identical surrounding prose and shared grammatical lead-in (Gates A–D discipline)**:
   - Outside the single contrasting clause (and `overall_soundscape:` when audio is coupled), **every word, punctuation mark, `<d>...</d>` dialogue line, `(S1)` tag, and header must be byte-for-byte identical** between Pole A and Pole B (`Gate A` refuses trailing `, while ...` appends; `Gate B` enforces `med_fd >= 10` varied words ahead of the first diff).
   - Keep the lead-in words (`"illuminated by "`, `"captured on "`) identical on both sides so `first_diff` lands directly on the contrasting content words rather than averaging wrapper words, and **never write Pole-A baseline state adjectives (`warm`, `sunny`, `bright`, `glossy`, `_BASELINE_STATE_WORDS` in `control_vectors.py`) inside shared non-`<d>` prefixes** (`Gate D` contradiction lint).
3. **Concrete, rotated Pole A baselines + matched word counts (avoiding length bias & Pole-A cancellation)**:
   - Never write negated baselines (`"no explosions"`, `"not claymation"`—negations still activate the negated noun in `Qwen3-VL`) and avoid bare 2-word Pole A against a 25-word Pole B.
   - Write a concrete, ordinary Pole A of roughly matching clause length (`10–18` words -> `10–22` words) so `B - A` measures the concept rather than sentence length.
   - **Never repeat one baseline garment across Pole A when steering attire/costumes** (`pirate_crew` / `wild_west_saloon` / `arctic_expedition` lesson, 2026-11-11): repeating `"casual wool sweater and shirt"` across pairs bakes `-shirt` / `-wearing` into `(-)` and cancels overlapping attire features on Pole B. Vary Pole A baselines across pairs (or use `from-prompt --bank-scenes 6 --words "<target clause>"` — compiled builds require `>= MIN_ERASURE_PAIRS (6)` pairs to emit `erasure.vr` unless `--force-low-pair-erasure` is passed — or `a1 | a2 | a3 -> b` in [`h3_pair_compiler.py`](../../../h3_pair_compiler.py)).
4. **Stratify across diverse subjects, interiors/exteriors, and lighting**:
   - Vary setting, gender/object, framing, and baseline color palette across the 6–10 pairs so accidental scene correlates average toward zero.
5. **Add a `#### Hold` group whenever your concept has an obvious unwanted twin**:
   - If you ask for 1970s 16mm grain without warm sepia tint, add a `Hold` group contrasting neutral daylight balance against warm amber tungsten cast (`film_16mm_1974_neutral`).
   - If you ask for surveillance grit without panicked movement, add a `Hold` group contrasting calm pedestrians against frantic running crowds.
6. **Obey the Pole-A Coverage Law (`8–12` pairs spanning every conflicting baseline facet for `erasure.vr`)**:
   - Under default `add_norm_matched_erasure`, Pole A (`diff_a`) is not throwaway scaffolding—thin SVD over centered Pole-A tokens builds the 8-dimensional **erasure subspace `erasure.vr`** that zeroes conflicting cues in user prompts at render time.
   - Across Monte-Carlo subsample curves (`bench/control_vector/eval_multivector_erasure.py`), `1–2` pairs capture only `33%–60%` of the full Grassmannian Pole-A subspace, whereas **`8–10` pairs reach `81%–89%+` coverage**. Make Pole A actively span the real-world defaults your slider will have to override (e.g., for `noir`: sunlit windows, warm tungsten bulbs, golden-hour street light, vibrant clothing, natural skin warmth) rather than repeating one verbatim copy-pasted Pole-A phrase across every pair.

---

## 3. Converting and building an LLM Markdown file

Save the LLM's Markdown reply to `data/control_vectors/my_concept.md`, then validate and convert it to `.toml` (CPU, `< 1s`):

```bash
CUDA_VISIBLE_DEVICES= /path/to/comfy-venv/bin/python bench/control_vectors.py from-md \
  data/control_vectors/my_concept.md \
  --out data/control_vectors/my_concept.toml \
  --test-prompt-out data/control_vectors/my_concept_test.txt
```

`from-md` (`control_vectors.write_markdown_to_toml`) validates every pair through `load_pairs`—checking H3 section structure, protected tags/dialogue parity, and non-empty `diff_span`—before writing `.toml`.

Then build the `fl2va` `.safetensors` vector (default `--families fl2va`, shared across `{fl2va, ref2va}` at `cos ~0.99`; pass `--families both` only if you want a separate `ref2va` sibling) and check the J-Lens top words + automatic `b_side` alignment audit (`j_lens.check_b_side_alignment`) printed at the end of the build:

```bash
/path/to/comfy-venv/bin/python bench/control_vectors.py build \
  data/control_vectors/my_concept.toml --families fl2va
```
*(You can also pass `data/control_vectors/my_concept.md` directly to `bench/control_vectors.py build` with `--save-toml`.)*
