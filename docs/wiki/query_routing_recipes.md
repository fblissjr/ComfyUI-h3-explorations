# Cross-Modal Query Routing Recipes & Exact Prompt Diffs (`1x`)

Exact node wiring, prompt diffs, and seeds to reproduce the shipped workflows (`T2V`, `fl2va`, `ref2va`), the `2026-11-24` comparison reels in `$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/`, and the `2026-11-25` Phase 2 showcase renders.

**Companion pages:** [`README.md`](README.md) · [`query_routing.md`](query_routing.md) · [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json) · [`workflows/h3_first_last_frame_to_video_pika_query_router_api.json`](../../workflows/h3_first_last_frame_to_video_pika_query_router_api.json) · [`workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json) · [`docs/wiki/system_prompts/h3_query_router/`](system_prompts/h3_query_router/README.md) (`h3-query-router-dub`, `h3-query-router-soundscape`) · [`bench/results/2026-11-25_query_router_phase2_sowhat.md`](../../bench/results/2026-11-25_query_router_phase2_sowhat.md) · [`bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md`](../../bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md)

---

## 1. Base Setup (Shipped `1x` Workflow)

Every stairwell recipe below uses the exact graph shipped at [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json):

| Graph Parameter | Value / Node ID |
| :--- | :--- |
| **Resolution & length (`"27"`)** | `1344x768` (`7/4` wide), `192` frames (`8.0s` at `24 fps`, Pika H3 Flex `6` steps) |
| **Seed (`"6"`)** | `730451892` (plus `101` and `202` on Thunderstorm / Spanish controls) |
| **Attention (`"58"`, `"21"`)** | `ModelAttentionBackend("comfy kitchen attention")` -> `MiniMaxH3Sol(dense_blocks="49", quantizer="rotated")` |
| **Default prompt (`"5"`)** | Unchanged dry English stairwell prompt (wired to `MiniMaxH3Preflight` `"26"`) |
| **Alternate audio prompt (`"146"`)** | Wired into `MiniMaxH3QueryRouter` (`"145"`) `.audio_conditioning` |
| **Alternate video prompt (`"147"`, optional)** | Wired into `MiniMaxH3QueryRouter` (`"145"`) `.video_conditioning` (`workflows/h3_text_to_video_pika_query_router_api.json` ships `"146"` for thunderstorm; add a `MiniMaxH3Conditioning` node `"147"` for video rewrites) |
| **Router defaults (`"145"`)** | `section_split="off"`, `video_block_band="all"`, `audio_block_band="all"`, `start_percent=0.0`, `end_percent=1.0`, `aud_to_vid_gain=1.0`, `audio_channel="both"`, `ref_audio_stream="both"` |

### Exact Unchanged Base Prompt (Node `"5"`)

```text
integrated_multimodal_description: [Shot 1] Live-action, cinematic, handheld on 35mm, a medium two-shot frames a woman with dark shoulder-length hair in a charcoal wool coat facing a man with a close-cropped beard in a navy jacket on a concrete stairwell landing, lit hard from a caged bulb overhead. The woman with the low measured voice (S1) says, <d>[English] You said tomorrow.</d> The man with the lower gravelled voice (S2) answers immediately, <d>[English] It moved.</d> She says at once, <d>[English] To when?</d> He answers at once, <d>[English] Tonight.</d> Her jaw tightens and the camera shakes slightly with the operator's breathing. [Shot 2] The camera cuts to a close-up of the woman against painted cinderblock, the bulb throwing a hard edge down one cheek. She (S1) says, <d>[English] Who else knows?</d> Off screen he (S2) answers immediately, <d>[English] Nobody.</d> She says at once, <d>[English] Keep it that way.</d> She looks past the camera toward the stairs below. [Shot 3] The camera cuts to a close-up of the man, rain noise faint through a window well behind him. He (S2) says, <d>[English] Understood.</d> He looks down, adjusts the strap on his shoulder, and holds still until the final frame.

overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead.

non_diegetic_music: N/A
```

---

## 2. Prompt Diff Discipline (Why the Plain Controls Are Clean A/Bs)

For every dubbing and soundscape comparison in `$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/reel_plain_vs_1x_router_stairwell_s730451892.mp4`:
- **Base `video_conditioning` (`"5"`) stays unchanged.**
- **`1x` Router run:** `audio_conditioning` (`"146"`) gets the exact diffed prompt shown below while `MiniMaxH3QueryRouter` keeps `Q_vid` reading node `"5"`.
- **Plain-Prompt Control run (no router):** `MiniMaxH3QueryRouter` is removed and node `"5"` itself gets that **exact same** diffed prompt string. Not a single extra word about actor appearance, nationality, wardrobe, hair, or lighting is added anywhere.

---

## 3. Clear-Win Recipes (`1x` Router Beats Same-Seed Plain Prompting)

### Recipe 1 — Zero-Shot Japanese Dub (`reel_plain_vs_1x_router` clips `2–3` & `reel_1x_dubbing` clip `3/4`)

- **Router wiring:** Node `"5"` = Base English prompt; Node `"146"` (`audio_conditioning`) = Base prompt with only `<d>[English] ...</d>` replaced; `video_block_band="all"`, `audio_block_band="all"`.
- **What plain prompting did wrong:** Recast both actors into Japanese/East Asian actors, added a cream knit sweater under her coat, shifted Shot 1's lamp conduit, and replaced the brick building outside Shot 3's window with misty hills.

```diff
--- base / video_conditioning (node "5")
+++ audio_conditioning (node "146")   [identical string used alone in plain control]
@@ Shot 1 @@
-The woman with the low measured voice (S1) says, <d>[English] You said tomorrow.</d>
-The man with the lower gravelled voice (S2) answers immediately, <d>[English] It moved.</d>
-She says at once, <d>[English] To when?</d>
-He answers at once, <d>[English] Tonight.</d>
+The woman with the low measured voice (S1) says, <d>[Japanese] 明日って言ったよね。</d>
+The man with the lower gravelled voice (S2) answers immediately, <d>[Japanese] 変わった。</d>
+She says at once, <d>[Japanese] いつに？</d>
+He answers at once, <d>[Japanese] 今夜だ。</d>
@@ Shot 2 @@
-She (S1) says, <d>[English] Who else knows?</d>
-Off screen he (S2) answers immediately, <d>[English] Nobody.</d>
-She says at once, <d>[English] Keep it that way.</d>
+She (S1) says, <d>[Japanese] 他に誰が知ってるの？</d>
+Off screen he (S2) answers immediately, <d>[Japanese] 誰も。</d>
+She says at once, <d>[Japanese] そのままにして。</d>
@@ Shot 3 @@
-He (S2) says, <d>[English] Understood.</d>
+He (S2) says, <d>[Japanese] 了解。</d>
```

### Recipe 2 — Zero-Shot Spanish Dub (`reel_plain_vs_1x_router` clips `4–5` & `reel_1x_dubbing` clip `2/4`)

- **Router wiring:** Same as Recipe 1; tested at seeds `730451892` and `101`.
- **What plain prompting did wrong:** Drifted both actors' faces (different woman in Shot 2, heavier dark-bearded man in Shot 3) and shifted Shot 1 framing (`28.17` vs `9.84` MAE).

```diff
--- base / video_conditioning (node "5")
+++ audio_conditioning (node "146")   [identical string used alone in plain control]
@@ Shot 1..3 <d> tags only (0 visual words changed) @@
-<d>[English] You said tomorrow.</d>  ->  +<d>[Spanish] Dijiste mañana.</d>
-<d>[English] It moved.</d>           ->  +<d>[Spanish] Se movió.</d>
-<d>[English] To when?</d>            ->  +<d>[Spanish] ¿Para cuándo?</d>
-<d>[English] Tonight.</d>            ->  +<d>[Spanish] Esta noche.</d>
-<d>[English] Who else knows?</d>     ->  +<d>[Spanish] ¿Quién más sabe?</d>
-<d>[English] Nobody.</d>             ->  +<d>[Spanish] Nadie.</d>
-<d>[English] Keep it that way.</d>   ->  +<d>[Spanish] Déjalo así.</d>
-<d>[English] Understood.</d>         ->  +<d>[Spanish] Entendido.</d>
```

### Recipe 3 — Bilingual Per-Speaker Code-Switching `(S1)` ES / `(S2)` EN (`reel_plain_vs_1x_router` clips `6–7`)

- **Router wiring:** Same as Recipe 1, translating only `(S1)`'s four lines to Spanish while keeping `(S2)`'s four lines in English.
- **What plain prompting did wrong:** Even leaving all of `(S2)`'s lines in English still caused facial drift in both the woman and the man (`25.41` vs `11.80` MAE).

```diff
--- base / video_conditioning (node "5")
+++ audio_conditioning (node "146")   [identical string used alone in plain control]
@@ (S1) lines only; (S2) lines stay verbatim [English] @@
-<d>[English] You said tomorrow.</d>  ->  +<d>[Spanish] Dijiste mañana.</d>
-<d>[English] To when?</d>            ->  +<d>[Spanish] ¿Para cuándo?</d>
-<d>[English] Who else knows?</d>     ->  +<d>[Spanish] ¿Quién más sabe?</d>
-<d>[English] Keep it that way.</d>   ->  +<d>[Spanish] Déjalo así.</d>
```

### Recipe 4 — 1970s Sitcom Laugh Track (`reel_plain_vs_1x_router` clips `8–9`)

- **Router wiring:** Node `"5"` unchanged; Node `"146"` changes only `overall_soundscape:`.
- **What plain prompting did wrong (and honest caveat):** Writing `"Bright 1970s three-camera sitcom studio recording..."` strictly inside `overall_soundscape:` still bled straight into `Q_vid` (`35.66` MAE): H3 gave the woman a retro 1970s flipped-out bob hairstyle, added a wall lamp right beside her head in Shot 2, flattened Shot 1 lighting, and softened the man's expression. Routing `Q_aud` keeps the gritty stairwell intact while live studio audience laughter and applause react after deadpan lines.

```diff
--- base / video_conditioning (node "5")
+++ audio_conditioning (node "146")   [identical string used alone in plain control]
@@ overall_soundscape only (0 words changed in integrated_multimodal_description) @@
-overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead.
+overall_soundscape: Bright 1970s three-camera sitcom studio recording; loud live studio audience bursts into warm laughter and scattered applause immediately after every deadpan line and pause, dry studio mic vocal tone
```

### Recipe 5 — Thunderstorm Over Dialogue (`workflows/h3_text_to_video_pika_query_router_api.json` default; `reel_plain_vs_1x_router` clips `10–11`)

- **Router wiring:** Directly loaded by [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json) (`"146"` -> `"145"`). Tested across `3` seeds (`730451892`, `101`, `202`).
- **What plain prompting did wrong:** Swapping only `overall_soundscape:` added/moved the caged wall lamp in Shot 2 and darkened Shot 3's window (`21.76` plain vs `9.91` routed on `s730451892`). Companion `3`-seed result on `tortoise` (`141` frames): plain Cathedral Reverb swapped London rowhouses into heavy church doors (`41.28` vs `19.67`).

```diff
--- base / video_conditioning (node "5")
+++ audio_conditioning (node "146")   [identical string used alone in plain control]
@@ overall_soundscape only (0 words changed in integrated_multimodal_description) @@
-overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead.
+overall_soundscape: A violent thunderstorm hammers the building, heavy rain drumming on a metal roof, rolling thunder close overhead, and wind howling up the stairwell.
```

### Recipe 6 — Underwater Visuals + Dry Stairwell Speech (`aud_to_vid_gain=0.0`, post-`9e5db27c`)

- **Router wiring (inverted roles):** Node `"5"` (default + `Q_vid`) gets the underwater visual rewrite prompt; Node `"146"` (`audio_conditioning`) gets the **unchanged dry base stairwell prompt** (zero underwater/bubble words in `prompt_aud`!); set **`aud_to_vid_gain = 0.0`**.
- **What `9e5db27c` unlocked (`p2b_uwvid_dryaud_a2v_000_post9e5`):** With plain UW+Dry (`-32.02` dBFS) and even routed `aud_to_vid_gain=1.0` (`-32.28` dBFS), `Q_aud` still read the underwater bubbles in `K_vid` and muffled the voices. After `9e5db27c` closed the 1-hop `K_vid -> shadow_text -> Q_aud` leak, `aud_to_vid_gain=0.0` jumped to **`-22.85` dBFS (`+9.43` dB over `g=1.0`, `+9.63` dB over pre-fix `g=0.0`)** — delivering crystal-clear close-mic dry stairwell dialogue over 100% submerged turquoise caustics and rising bubbles.
- **Strict clean-text rule:** Only drop `aud_to_vid_gain` to `0.0` when `prompt_aud` is completely stripped of the conflicting visual-action prose. If `prompt_aud` still mentions the visual explosion/action (as in `p2_cooking_router_zen_g0`), keep the default `aud_to_vid_gain = 1.0`.

```diff
--- plain fair control ("stairwell_plain_uwvid_dryaud") == node "5" + overall_soundscape unchanged
+++ video stream reads UW prompt ("5"); audio_conditioning ("146") reads Base English dry prompt (aud_to_vid_gain=0.0)
@@ What plain prompting and g=1.0 failed to separate @@
 integrated_multimodal_description: [Shot 1] ... deeply submerged underwater inside a flooded concrete stairwell; crisp bright turquoise sunlight caustics ... steady streams of rising silver air bubbles ...
-overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead.
+(In plain prompting and g=1.0, K_vid underwater physics dragged speech to -32.02 / -32.28 dBFS; post-9e5db27c g=0.0 with a clean dry prompt_aud restores -22.85 dBFS dry close-mic speech.)
```

### Recipe 7 — `1x` Binaural Reality-vs-Story Split (`t2va_radio_drama`, `audio_channel="right"`, `late_30_49`)

- **Router wiring:** Use [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json) (`192f`, seed `730451892`); Node `"5"` = verbatim [`prompt_bank/t2va_radio_drama.txt`](../../prompt_bank/t2va_radio_drama.txt); Node `"146"` (`audio_conditioning`) = same prompt with `overall_soundscape:` and `non_diegetic_music:` swapped to storm-lashed Gothic mansion stairwell + rain + horror strings; set **`audio_channel = "right"`**, **`audio_block_band = "late_30_49"`**, **`aud_to_vid_gain = 1.0`**.
- **Why `late_30_49` is mandatory:** Keeping `blocks.0..29` open locks all 3 speakers' speech phase and lip-sync across Left and Right, while cross-masking `blocks.30..49` separates the two ears (`lr_corr = 0.1167`, `L = -13.68` / `R = -14.35` dBFS, `pic_mae = 16.01`, **zero echo**). Left ear stays in the bone-dry 1948 padded broadcast booth while Right ear enters the haunted storm stairwell. *(Verified `2/2` with `stairwell_binaural_right_storm_late30`: `pic_mae = 7.93`, `lr_corr = 0.1122`.)*

### Recipe 8 — Soundscape-to-Architecture Lock (`t2va_cooking_show_meltdown`, Chaos Visuals + Zen Audio)

- **Router wiring:** Node `"5"` = verbatim [`prompt_bank/t2va_cooking_show_meltdown.txt`](../../prompt_bank/t2va_cooking_show_meltdown.txt); Node `"146"` (`audio_conditioning`) = calm whispered dialogue + `"tranquil Zen tea pavilion"` `overall_soundscape:` + hushed solo cello `non_diegetic_music:`; keep **`aud_to_vid_gain = 1.0`**, **`audio_block_band = "all"`**.
- **What plain prompting did wrong:** `"Zen tea pavilion"` inside `overall_soundscape:` tore sunlit French garden windows with green trees and a pink blossom vase straight into the windowless brick TV studio (`47.99` MAE). `1x` routing (`g=1.0`) kept the brick studio, stainless hood, copper pots, and neon sign 100% locked while syncing serene solo cello and meditative whisper to the purple beetroot blender explosion (`-16.89` dBFS, `HF>2k = 0.0247`).

### Recipe 9 — Ensemble `ref2va` Cast & Ticket Lock (`ref2va_scene_kitchen` w/ `woman_blonde.jpg`)

- **Router wiring:** Use [`workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json`](../../workflows/h3_image_ref_plus_text_to_video_pika_query_router_api.json) (shared `MiniMaxH3EncodeReferences` `"132"` -> dual `MiniMaxH3PromptOnReferences` `"5"` / `"146"` -> `MiniMaxH3QueryRouter` `"145"`). Load `$H3_INPUT_DIR/woman_blonde.jpg` as `<Picture 1>`, use `ref2va_scene_kitchen` adapted to female `<Subject 1>` on `"5"`, and translate strictly the 6 `<d>` dialogue/lyric lines to `[Japanese]` on `"146"` (via `h3-query-router-dub`).
- **What plain prompting did wrong:** Plain `[Japanese]` kept `<Subject 1>` blonde because `<Picture 1>` pinned her, **while recasting every unpinned co-star in the kitchen to East Asian cooks**, swapping the plated Western dishes, and erasing `"FIRE"` on the thermal tickets (`40.03` MAE). `1x` routing locked `<Subject 1>`, the Black pan-tossing cook, the curly-haired flat-top cook `(S2)`, Western plates, and `"FIRE"` thermal tickets across all 4 shots (`28.55` MAE) while speaking/singing synced Japanese with `<|cutoff|>`.

---

## 4. Reference Recipes & Honest Neutral Boundaries (Plain Prompting Already Works)

These cases were tested with matched plain controls (`2026-11-24` and `2026-11-25`) and are documented honestly so nobody wastes router complexity where plain prompting already wins:

- **Period-piece `JA -> EN` reverse dub (`t2va_samurai_bamboo_duel`) & heavily anchored `EN -> ES` (`t2va_delta_blues_porch`):** Plain prompting already kept both Japanese samurai actors + `"関ヶ原街道"` vertical kanji (`28.40` vs `25.23` MAE) and the Deep South porch + resonator guitar + `"CLARKSVILLE 5 MILES"` sign (`35.27` vs `31.95` MAE).
- **Tight single-headshot `ref2va` (`t2i_29`), sealed-mouth voiceover, & full-prompt `ref2va` motion-transfer restyle:** When `<Picture 1>` fills the frame or the visual prose already nails the target style/closed lips, plain prompting matches `1x` routing directly.
- **Temporal audio switching (`audio_time_start=0.5`, `late_30_49`):** Beats plain prompting on visual lock (`21.93` vs `41.70`) and H2 horror HF (`5.8x`) on `t2va_radio_drama`, but produces a 1-phrase repeat if `t=0.5` lands flush on spoken dialogue (`"put out the lamp"` / `"Who else knows?"`). Place `audio_time_start` only on silence/music transitions.
- **Blueprint Ink Visual (`video_conditioning`):** Replace node `"5"`/`"147"` with crisp white-on-Prussian-blue architectural drafting descriptions (`BLUEPRINT_LONGER_PROMPT`), keeping node `"5"` dry English on `Q_aud`. Plain prompting with that same prompt already holds dry acoustics (`65.45` plain vs `65.27` routed MAE; `19.44` vs `17.50` dB spec) because ink linework does not fight concrete acoustics.
- **Hydrophone + Sonar & 1930s AM Radio:** Swap `overall_soundscape:` on `"146"` only. Although Hydrophone+Sonar stopped Shot 2's plain-prompt lamp bleed (`24.00` vs `9.59` MAE), both audio effects are subtle on blind watch/listen without priming, so they are omitted from `reel_plain_vs_1x_router_stairwell_s730451892.mp4`.
- **Zero-rewrite single-prompt leak guard (`section_split="soundscape_to_audio"`):** Set `section_split="soundscape_to_audio"` on `"145"` with nothing wired to `"146"` or `"147"` to hide `overall_soundscape:` / `non_diegetic_music:` from `Q_vid` on fresh renders (`stairwell_router_split_base`, `tortoise_router_split_*`). Caveat: if `overall_soundscape:` describes a visible cue omitted from `integrated_multimodal_description` (like `tortoise`'s lit traffic light), hiding it drops that cue (`tortoise` traffic light goes unlit).

---

## 5. Complete Same-Seed (`730451892`) Measurements Table

All `8` matched stairwell pairs (`192` frames, `1344x768`, seed `730451892`, base RMS `-25.87` dBFS), plus `stairwell_plain_uwvid_dryaud`:

| Concept Pair | Plain Pic MAE vs Base | Routed Pic MAE vs Base | Plain vs Routed Pic MAE | Plain Spec MAE vs Base (dB) | Routed Spec MAE vs Base (dB) | Plain RMS (dBFS) | Routed RMS (dBFS) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Japanese Dub** | `26.79` | `11.96` | `25.88` | `16.65` | `16.12` | `-33.82` | `-22.87` | **Clear win** (stops actor/costume/window recast) |
| **Spanish Dub** | `28.17` | `9.84` | `27.24` | `11.04` | `11.43` | `-28.98` | `-29.10` | **Clear win** (stops actor face drift) |
| **Bilingual `(S1)` ES / `(S2)` EN** | `25.41` | `11.80` | `22.47` | `12.22` | `11.99` | `-28.88` | `-30.95` | **Clear win** (stops both-actor face drift) |
| **1970s Sitcom Laugh Track** | `35.66` | `9.95` | `34.65` | `16.07` | `9.47` | `-26.34` | `-26.18` | **Clear win** (stops 70s hair/lamp/lighting bleed) |
| **Thunderstorm (`s730451892`)** | `21.76` | `9.91` | `20.62` | `11.15` | `12.06` | `-20.43` | `-21.94` | **Clear win** (stops Shot 2 lamp + Shot 3 dark window; `3` seeds) |
| **UW Video + Dry Sound** | `47.10` *(plain UW+Dry)* | `48.54` *(`a2v=0.25`)* | `21.73` *(plain UW+Dry vs `0.25`)* | `20.38` *(`-23.60` HF tilt)* | `14.97` *(`-20.79` HF tilt)* | `-31.99` | `-32.09` | **Clear win** (stops UW visual muffling dry speech) |
| **Hydrophone + Sonar** | `24.00` | `9.59` | `23.67` | `15.99` | `15.43` | `-27.56` | `-27.40` | Visual leak fixed, but audio subtle on blind watch |
| **Blueprint Ink Visual** | `65.45` | `65.27` | `18.53` | `19.44` | `17.50` | `-33.49` | `-34.72` | **Neutral** (plain prompt works well without router) |
