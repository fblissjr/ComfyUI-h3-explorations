# 7. Prompting Generative LLMs for Control Vectors

[← Back to Control Vectors Overview](../control_vectors.md)

When you want to author a **reusable 10-pair multi-scene control vector** (`Method 2: from-toml`) or an **in-place scene-paired contrast spec** (`Method 1: from-prompt`) using a local LLM (`transformers` / `exllamav3` via `bench/control_vectors.py generate-pairs`) or an external LLM (Claude, Gemini, ChatGPT), use [`docs/wiki/system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md) as the system prompt and one of the user-prompt templates below.

---

## 1. Template A — Authoring a Reusable 10-Pair Multi-Scene Vector (`Method 2: from-toml`)

Use this template when building a generalizable library vector that works across any prompt:

```text
Write a complete 10-pair MiniMax H3 Control Vector Set (### Test Scene Prompt, ### Contrastive Pairs (1–10), and optional ### Hold Pairs) for the following concept shift:

- Concept name: <e.g., film_noir_rain>
- Baseline (Pole A): <e.g., crisp neutral daylight or balanced indoor lighting with dry surfaces>
- Steered target (Pole B): <e.g., high-contrast 1940s silver-gelatin film noir chiaroscuro with Venetian blind shadow bars, wet rain-streaked glass, and cigarette smoke haze>
- Coupled overall_soundscape shift (optional): <e.g., dry neutral room tone -> steady rain pattering against windowpanes and distant wet tire hiss>
- Nuisance axes to hold constant (optional ### Hold Pairs): <e.g., hold(wardrobe): keep modern everyday clothing rather than forcing 1940s fedora/trenchcoat attire>
- Test scene scenario: <e.g., a ref2va scene where <Subject 1> from <Picture 1> sits at a diner booth and says a short line>
```

Once the LLM outputs the fenced Markdown block, save it as `film_noir_rain.md` and run:
```bash
COMFY="$(cd ../.. && pwd)"
"$COMFY/.venv/bin/python" bench/control_vectors.py from-toml film_noir_rain.md --name film_noir_rain --families both --overwrite
```
This validates all 10 pairs (`lint_pair_set`, `10/10` unique spans, zero banned rate adverbs) and compiles both `fl2va` and `ref2va` `.safetensors` files (or use `from-md film_noir_rain.md --name film_noir_rain` to convert `.md` to `.toml` on CPU first).

---

## 2. Template B — Authoring an In-Place `from-prompt` Contrast Spec (`Method 1: from-prompt`)

Use this template when you already have a specific scene prompt (`t2va` or `ref2va`) and want the LLM to craft clean, high-leverage in-place `pole_a -> pole_b` substitutions on your exact prompt:

```text
Given my exact target prompt below, produce an In-Place Target-Prompt Clause-Diff Spec (`--words`) that shifts the scene toward: <e.g., 1990s multi-camera studio sitcom with punchy comedic delivery and a live studio audience laugh track>.

Rules:
1. Every Pole A before `->` MUST be an exact character-for-character substring of my target prompt.
2. Never repeat Pole A nouns inside Pole B (prevent B - A noun cancellation) and never use generic rate adverbs (slowly, gradually, begins to).
3. Separate multiple clause replacements with `; `, and prefix any overall_soundscape replacement with `soundscape: `.

My target prompt:
<paste your exact t2va or ref2va prompt here>
```

Then pass the returned `--words` spec directly to `from-prompt`:
```bash
"$COMFY/.venv/bin/python" bench/control_vectors.py from-prompt \
  --name my_scene_steer \
  --target-prompt my_prompt.txt \
  --words "<paste spec here>" \
  --overwrite
```
