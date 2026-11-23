# Wave 4 Post-Restart Live Verification (`:8188` + Strict `bank/` Smokes)

**Date:** `2026-11-23`  
**Seed / Canvas / Guard:** `seed = 730451892`, `allow_recipe_mismatch = False`, directly from `../../models/control_vectors/bank/` (`50` active stems / `143` `.safetensors`)  
**Ledger & Outputs:** [`data/exp_unpooled_sliders/wave4_2026_11_23/wave4_results.json`](../../data/exp_unpooled_sliders/wave4_2026_11_23/wave4_results.json) · `$H3_OUTPUT_DIR/Image/wave4_2026_11_23/` · `$H3_OUTPUT_DIR/Video/wave4_2026_11_23/`  
**Boundary Check:** Nested `app/.git` remains frozen and untouched at `825d27d` (`Track App-2.0` boundary intact).

---

## 1. Restarted `:8188` `/object_info` Schema & Live Workflow Audit

- **Live `/object_info` Registry:** `1,117` total ComfyUI nodes loaded; **`79` `MiniMaxH3*` nodes** registered, including `MiniMaxH3PromptExpander`, `MiniMaxH3AutoEditPrompt`, and `MiniMaxH3ImagePromptExpander`.
- **`MiniMaxH3Steer` Parity:**
  - `target` options (`7`): `['scene_only', 'description_only', 'look_only', 'soundscape_only', 'after_dialogue', 'non_vision', 'vision_tokens']` (`vision_tokens` verified at tail index `6`).
  - `vector` options (`51`): `'none'` + all **`50` active bank stems**, including Wave 3B promotions `youtube_overhype_influencer` and `sudden_takeoff_flight`.
  - Root [`heylook_client.py`](../../heylook_client.py#L47): `DEFAULT_HEYLOOK_MODEL == "gemma-4-26B-A4B-it-8bit"`.
- **Shipped API Workflow Audit (`workflows/**/*.json`):** Checked **`135 / 135` JSON graphs** (`0` non-API files skipped, **`2,779` API nodes**, **`20` `MiniMaxH3Steer` nodes**) against live `:8188` `/object_info` (class registration, required inputs present, internal link targets/slots, and `MiniMaxH3Steer` combo enums): **`0` errors**.
- **Heylook `/v1/models` Probe (Non-Fatal):** `configured = true`, returned **`HTTP 502: Bad Gateway`** (`reachable = false`; remote Heylook server intentionally offline during this run — live expander verification can run separately once Heylook is brought back online). Zero raw URLs or hostnames recorded.

---

## 2. Strict `allow_recipe_mismatch=False` Live GPU Smokes (`bank/`)

All 4 smokes ran serially against live `:8188` with `allow_recipe_mismatch = False` loading `.safetensors` directly from `bank/`. All 4 outputs were inspected natively via `read_file` (`2` `.png` files + `2` `.1fps.00000-00024.mp4` companions).

| Smoke ID | Checkpoint / Stack | Vector Config (`allow_recipe_mismatch=False`) | Size / Length | Metrics (`sat`, `luma`, `clip`, `lap`, `temp_l1`, audio) | Time (`exec` / `wall`) | Output & Visual Verdict |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| `smoke1_h3i_takeoff` | `MiniMax-H3-Image` (`h3_image`, `4.0 -> 16.0 MP`) | `sudden_takeoff_flight` (`s = 1.20`, `scene_only`) | `2368x1760` (`4.168 MP`) | `sat 0.4328`, `luma 115.96`, `clip 0.02928`, `lap 1189.9` | `34.83s / 36.07s` | `$H3_OUTPUT_DIR/Image/wave4_2026_11_23/smoke1_h3i_takeoff_00001_.png` — **PASS:** motorcyclist & sport-touring bike launched high airborne into clear blue canyon sky above the twin yellow lines and road shadow with crisp terrain texture. |
| `smoke2_h3i_overhype` | `MiniMax-H3-Image` (`h3_image`, `4.0 -> 16.0 MP`) | `youtube_overhype_influencer` (`s = 1.20`, `scene_only`) | `2368x1760` (`4.168 MP`) | `sat 0.2909`, `luma 69.47`, `clip 0.01678`, `lap 813.7` | `25.17s / 25.53s` | `$H3_OUTPUT_DIR/Image/wave4_2026_11_23/smoke2_h3i_overhype_00001_.png` — **PASS:** calm-prompt studio creator transformed into wide-eyed open-mouth shock expression with splayed hands beside the boom condenser mic (*note:* minor garbled tee lettering artifact on navy shirt). |
| `smoke3_fl2va_stack_39f` | `fl2va` (`MiniMaxH3Sol`, `auto_1x_kv`, [`t2va_stairwell_dialogue.txt`](../../prompt_bank/t2va_stairwell_dialogue.txt)) | Dual stack: `youtube_overhype_influencer` (`1.25`, `scene_only`) + `reverb_cathedral` (`1.15`, `soundscape_only`), `streams="both"` | `1152x768`, `39f` (`1.625s`) | `sat 0.4532`, `luma 95.5`, `clip 0.01252`, `temp_l1 11.542`, `lap 83.91`, `-20.27 dBFS`, `2604.2 Hz` | `23.60s / 24.05s` | `$H3_OUTPUT_DIR/Video/wave4_2026_11_23/smoke3_fl2va_stack_39f_00001-audio.mp4` (`.1fps.00000-00024.mp4`) — **PASS:** intense confrontation in concrete stairwell pushing cleanly from face-to-face two-shot (`f00`) into close-up reaction (`f24`). |
| `smoke4_ref2va_takeoff_39f` | `ref2va` (`MiniMaxH3Sol`, `woman_blonde.jpg` + [`ref2va_woman_blonde_street.txt`](../../prompt_bank/ref2va_woman_blonde_street.txt)) | `sudden_takeoff_flight` (`1.20`, `description_only`), `streams="both"` | `1152x768`, `39f` (`1.625s`) | `sat 0.2219`, `luma 107.2`, `clip 0.00003`, `temp_l1 13.287`, `lap 52.89`, `-16.65 dBFS`, `2320.4 Hz` | `36.63s / 37.57s` | `$H3_OUTPUT_DIR/Video/wave4_2026_11_23/smoke4_ref2va_takeoff_39f_00001-audio.mp4` (`.1fps.00000-00024.mp4`) — **PASS:** `<Subject 1>` blonde hair, face, black sweater, jeans, and green-awning overcast street preserved; airborne mid-stride with windblown hair (`f00`) touching back onto sidewalk (`f24`). |
