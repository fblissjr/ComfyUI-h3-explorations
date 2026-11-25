# Cross-Modal Query Routing Recipes & Exact Prompt Diffs (`1x`)

Exact node wiring, prompt diffs, and seeds to reproduce the shipped workflow and comparison reels in `$H3_OUTPUT_DIR/Video/query_router_reels_2026-11-24/`.

**Companion pages:** [`README.md`](README.md) · [`query_routing.md`](query_routing.md) · [`workflows/h3_text_to_video_pika_query_router_api.json`](../../workflows/h3_text_to_video_pika_query_router_api.json) · [`bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md`](../../bench/results/2026-11-24_query_router_live_probes_and_step2_gates.md)

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

### Recipe 6 — Underwater Visuals + Dry Stairwell Speech (`reel_plain_vs_1x_router` clips `12–13`)

- **Router wiring (inverted roles):** Node `"5"` (default + `Q_vid`) gets the underwater visual rewrite prompt; Node `"146"` (`audio_conditioning`) gets the **unchanged dry base stairwell prompt**; set `aud_to_vid_gain = 0.25` (gives brightest `2–8` kHz vs `100–500` Hz tilt without breaking lip-sync).
- **What the fair plain control proved:** Even when a plain prompt kept the underwater visual shots and wrote the **exact dry concrete `overall_soundscape:`**, plain prompting still pulled the dialogue underwater (`-23.60` dB HF tilt, only `5.87` dB away from underwater hydrophone sound), whereas `1x` routing recovered drier speech (`-20.79` dB HF tilt, `14.85` dB away from plain UW+Dry).

```diff
--- plain fair control ("stairwell_plain_uwvid_dryaud") == node "5" + overall_soundscape unchanged
+++ video stream reads UW prompt ("5"); audio_conditioning ("146") reads Base English dry prompt
@@ What plain prompting failed to separate @@
 integrated_multimodal_description: [Shot 1] ... deeply submerged underwater inside a flooded concrete stairwell; crisp bright turquoise sunlight caustics ... steady streams of rising silver air bubbles ...
-overall_soundscape: Close handheld room tone in a hard concrete stairwell with a long reflective tail on every consonant and a faint electrical hum from the caged bulb overhead.
+(In plain prompting, the underwater image text overrode the dry overall_soundscape text and muffled the speech; 1x routing gives Q_aud the dry base prompt cleanly.)
```

---

## 4. Reference Recipes (Subtle by Ear or Neutral vs. Plain Prompting)

These appear in `reel_1x_soundtracks_and_visuals_stairwell_s730451892.mp4` and earlier notes; they are documented honestly here for completeness:

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
