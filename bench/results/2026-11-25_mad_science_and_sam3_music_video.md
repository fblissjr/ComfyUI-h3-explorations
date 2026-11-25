# Wave 1 Mad-Science & `SAM 3.1` Frozen-Groove Music-Video Probes (`2026-11-25`)

Zero-new-code `:8188` GPU batch (`10` clips, `1344×768`, 6-step Pika H3 Flex `MiniMaxH3Sol`, `seed = 730451892`) testing four zero-training capabilities and honest boundaries directly on pack `main` (`f232b0fb`):

1. **Impossible Foley via `aud_to_vid_gain = 0.0` (1-hop `K_vid -> shadow_text -> Q_aud` firewall):** Armored medieval knight drawing a broadsword to tiny plastic rubber-duck squeaks + fairy harp (`192f`).
2. **Species-Voice Firewall (`aud_to_vid_gain = 0.0`):** Photoreal golden retriever belting Italian *Nessun Dorma* opera (`192f`).
3. **`SAM 3.1` Multi-Frame Subject-vs-Background Splits over Frozen Music (`141f`, `on_both_clocks`):** `MiniMaxH3FreezeAudio` (`$H3_INPUT_DIR/h3_trio_groove_20261123.wav`) + `MiniMaxH3SubjectTrack` (`sam3.1_multiplex_fp32.safetensors`) + same-seed `MiniMaxH3EditGuider(streams="video_only", first_step=1, scale=2.0)` swapping **Background-Only** and **Subject-Only** while keeping lossless audio muxed from `[freeze_node, 1]`.
4. **Hazy-Memory `<Picture 1>` Sweep (`141f`, `ref2va`):** `MiniMaxH3ReleaseReferences(end_percent=1.0, clean_level={0.999, 0.50, 0.35})` on `$H3_INPUT_DIR/old_man.jpg` in bright alpine daylight.

- **Rendered clips & 33.6 s Bonus Showcase Reel:** `$H3_OUTPUT_DIR/Video/mad_science_wave1_2026-11-25/` (`reel_mad_science_and_sam3_mv_2026-11-25.mp4`, `15.7 MB`)
- **Independent verification:** All `10/10` renders and the `33.6 s` reel were independently watched and listened to by two sessions with identical verdicts on every arm.

---

## 1. Summary Table (`seed = 730451892`, `1344×768`, 6-Step Pika `MiniMaxH3Sol`)

| Arm | Group | Frames | Elapsed (s) | Pic MAE vs First | RMS (dBFS) | `HF > 2 kHz` | `YIN F0` (Hz) | Voiced Frac | Honest Verdict |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| `w1_knight_duck_plain` | `knight_duck` | `192` | `94.0` | `0.000` | `-24.95` | `0.3944` | `331.34` | `0.2794` | **Plain fails:** loud metallic steel *SHIIIING!* sword scrape overrides explicit *"zero metallic ring"* prompt (`00:01–00:04`) |
| `w1_knight_duck_router_g000` | `knight_duck` | `192` | `106.1` | `29.320` | `-26.50` | `0.2554` | `296.67` | `0.3089` | **Clear `a2v=0.0` win (5th 1-hop firewall proof):** zero steel scrape; two unmistakable rubber-duck squeaks timed to the sword draw + fairy harp glissando on the swing |
| `w1_mv_freeze_base` | `mv_sam3` | `141` | `68.0` | `0.000` | `-24.62` | `0.1680` | `87.19` | `0.2356` | Clean sunlit white-brick dance studio baseline locked to `$H3_INPUT_DIR/h3_trio_groove_20261123.wav` |
| `w1_mv_sam3_bg_neon` | `mv_sam3` | `141` | `132.1` | `42.505` | `-24.62` | `0.1680` | `87.19` | `0.2356` | **Clear first-shot win:** dancer stays 100% sunlit photoreal with exact beat choreography while magenta/cyan neon graphics & floating particles erupt across the brick loft behind him (`GrowMask(20)+ImageBlur(12)+InvertMask`, `first_step=1`, bit-identical muxed groove) |
| `w1_mv_sam3_subj_chrome` | `mv_sam3` | `141` | `104.1` | `11.328` | `-24.62` | `0.1680` | `87.19` | `0.2356` | **Clear first-shot win:** entire sunlit brick loft stays 100% locked photoreal while only the dancer's head, torso, and arms become mirror-polished liquid chrome grooving frame-for-frame on beat with zero ghosting |
| `w1_opera_dog_plain` | `opera_dog` | `192` | `110.1` | `0.000` | `-15.38` | `0.0702` | `364.26` | `0.7813` | **Honest native win:** plain H3 already keeps a photoreal golden retriever belting full-orchestra *Nessun Dorma* in a booming male tenor with synced jaw motion and zero barking |
| `w1_opera_dog_router_g000` | `opera_dog` | `192` | `108.1` | `30.590` | `-33.71` | `0.0452` | `445.54` | `0.6823` | **Partial/honest shift:** `a2v=0.0` shifts singing up `+3.5` semitones into a female soprano prima donna synced to the dog's jaw, but noticeably quieter (`-33.71` vs `-15.38` dBFS) |
| `w1_hazy_ref_c999` | `hazy_ref` | `141` | `112.1` | `0.000` | `-49.93` | `0.1190` | `68.79` | `0.0077` | Clean bright alpine snow relighting of `old_man.jpg` in a blue parka from Frame 0 with zero original corridor bleed |
| `w1_hazy_ref_c500` | `hazy_ref` | `141` | `98.0` | `28.163` | `-49.02` | `0.1390` | `70.66` | `0.0154` | **Falsified as dreamy dial:** Frame `0` flashes `old_man.jpg`'s original dark hallway + cap + cheek talisman before hard-snapping to the snowy peak |
| `w1_hazy_ref_c350` | `hazy_ref` | `141` | `98.0` | `37.957` | `-52.51` | `0.3045` | — | `0.0000` | **Falsified as dreamy dial:** Stronger Frame-`0` corridor double-exposure flash before snapping to the snowy peak; `00:01–00:05` stays nearly identical to `c999` |

---

## 2. Key Takeaways & Reproducible Wiring Notes

1. **5th Independent Proof of the 1-Hop `aud_to_vid_gain = 0.0` Firewall (`knight_duck`):**
   - Visual steel-sword physics overrode plain H3's explicit *"zero metallic ring, strictly rubber-duck squeaks"* instruction (`0.3944` HF scrape). Stripping knight/sword/metal words from `audio_conditioning` and setting `aud_to_vid_gain = 0.0` cleanly eliminated every trace of metallic sound while preserving grim armored visuals and timing two crisp rubber-duck squeaks + a fairy harp glissando directly to the sword draw and swing.
2. **Verified Zero-Code `SAM 3.1` + `MiniMaxH3FreezeAudio` + `MiniMaxH3EditGuider` Music-Video Recipe (`w1_mv_*`):**
   - Base pass: `LoadAudio` (`h3_trio_groove_20261123.wav`) -> `MiniMaxH3FreezeAudio(start_seconds=0.0, audio_mask=0.0, level="clip_guard")` -> `SamplerCustomAdvanced` -> `VAEDecode` (`141f`, `1344×768`).
   - Track pass: `CheckpointLoaderSimple(sam3.1_multiplex_fp32.safetensors)` -> `MiniMaxH3SubjectTrack(subject_phrase="person", pick="most central", track_head=False)` produces a clean `[141, 768, 1344]` mask directly from the cached base `VAEDecode` frames.
   - **Background-Only Split:** `GrowMask(expand=20, tapered_corners=True)` -> `MaskToImage` -> `ImageBlur(blur_radius=12, sigma=0.45)` -> `ImageToMask(channel="red")` -> `InvertMask` -> `MiniMaxH3EditGuider(streams="video_only", first_step=1, scale=2.0)` with `VHS_VideoCombine.audio` wired to `[freeze_node, 1]` (`clip_audio`).
   - **Subject-Only Split:** `MaskToImage` -> `ImageBlur(blur_radius=6, sigma=0.45)` -> `ImageToMask(channel="red")` -> `MiniMaxH3EditGuider(streams="video_only", first_step=1, scale=2.0)`, keeping the entire studio pixel-locked while turning only the dancer into liquid chrome.
   - Note: `first_step = 1` anchors Step 0's silhouette to the tracked base pose; if a full night/darkness background replacement is wanted instead of neon AR overlays over daylight brick, test `first_step = 0` on the Background-Only pass.
3. **Why `clean_level < 0.999` Was Cut From Studio 2.0 (`hazy_ref`):**
   - Lowering `MiniMaxH3ReleaseReferences.clean_level` (`visual_cond_noise_aug`) from `0.999` to `0.50`/`0.35` brings `<Picture 1>`'s AdaLN timestep closer to noisy early video target rows (`max(t_v, vis_aug)`), causing Frame `0` to briefly flash `<Picture 1>`'s original dark room/wardrobe before snapping to the target scene. Keep `clean_level = 0.999` as default (use `end_percent = 0.35..0.50` when releasing reference layout mid-denoise).
4. **Honest Baseline Check (`opera_dog`):**
   - Plain H3 already sings full-orchestra Italian opera through a photoreal golden retriever without barking (`364.26 Hz` tenor); `a2v = 0.0` shifts the vocal persona `+3.5` semitones higher to a female soprano (`445.54 Hz`), confirming that strong visual-to-audio conflicts (sword scrape, underwater muffling, roaring fire, bearded biker → grandma voice) are where `a2v = 0.0` earns its keep most decisively.
