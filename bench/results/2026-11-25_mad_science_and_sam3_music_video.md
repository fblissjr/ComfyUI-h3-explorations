# Wave 1 Mad-Science & `SAM 3.1` Frozen-Groove Music-Video Probes (`2026-11-25`)

Zero-new-code `:8188` GPU batch (`10` clips, `1344×768`, 6-step Pika H3 Flex `MiniMaxH3Sol`, `seed = 730451892`) testing four zero-training capabilities and honest boundaries directly on pack `main` (`f232b0fb`):

1. **Impossible Foley via `aud_to_vid_gain = 0.0` (1-hop `K_vid -> shadow_text -> Q_aud` firewall):** Armored medieval knight drawing a broadsword to tiny plastic rubber-duck squeaks + fairy harp (`192f`).
2. **Species-Voice Firewall (`aud_to_vid_gain = 0.0`):** Photoreal golden retriever belting Italian *Nessun Dorma* opera (`192f`).
3. **`SAM 3.1` Multi-Frame Subject-vs-Background Splits over Frozen Music (`141f`, `on_both_clocks`):** `MiniMaxH3FreezeAudio` (`$H3_INPUT_DIR/h3_trio_groove_20261123.wav`) + `MiniMaxH3SubjectTrack` (`sam3.1_multiplex_fp32.safetensors`) + same-seed `MiniMaxH3EditGuider(streams="video_only", first_step=1, scale=2.0)` swapping **Background-Only** and **Subject-Only** while keeping lossless audio muxed from `[freeze_node, 1]`.
4. **Hazy-Memory `<Picture 1>` Sweep (`141f`, `ref2va`):** `MiniMaxH3ReleaseReferences(end_percent=1.0, clean_level={0.999, 0.50, 0.35})` on `$H3_INPUT_DIR/old_man.jpg` in bright alpine daylight.

- **Rendered clips:** `$H3_OUTPUT_DIR/Video/mad_science_wave1_2026-11-25/` (all `10` raw clips; hyped `SAM 3.1` mini-reel intentionally withheld until `last_step = 3` / `scale = 1.6` re-test).
- **Independent verification:** All `10/10` renders were independently watched and listened to by two sessions, then re-audited against `docs/wiki/edit_guidance.md` (`last_step = 3` on 6-step Pika) to strip initial over-praise on `w1_mv_sam3_*`.

---

## 1. Summary Table (`seed = 730451892`, `1344×768`, 6-Step Pika `MiniMaxH3Sol`)

| Arm | Group | Frames | Elapsed (s) | Pic MAE vs First | RMS (dBFS) | `HF > 2 kHz` | `YIN F0` (Hz) | Voiced Frac | Honest Verdict |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| `w1_knight_duck_plain` | `knight_duck` | `192` | `94.0` | `0.000` | `-24.95` | `0.3944` | `331.34` | `0.2794` | **Plain fails:** loud metallic steel *SHIIIING!* sword scrape overrides explicit *"zero metallic ring"* prompt (`00:01–00:04`) |
| `w1_knight_duck_router_g000` | `knight_duck` | `192` | `106.1` | `29.320` | `-26.50` | `0.2554` | `296.67` | `0.3089` | **Only clean Wave 1 home run (5th 1-hop firewall proof):** zero steel scrape; two unmistakable rubber-duck squeaks timed to the sword draw + fairy harp glissando on the swing |
| `w1_mv_freeze_base` | `mv_sam3` | `141` | `68.0` | `0.000` | `-24.62` | `0.1680` | `87.19` | `0.2356` | Clean sunlit white-brick dance studio baseline locked to `$H3_INPUT_DIR/h3_trio_groove_20261123.wav` |
| `w1_mv_sam3_bg_neon` | `mv_sam3` | `141` | `132.1` | `42.505` | `-24.62` | `0.1680` | `87.19` | `0.2356` | **Plumbing-only partial (not a visual win):** multi-frame `SAM 3.1` mask kept the dancer isolated and audio bit-identical, **but** `first_step = 1` locked bright sunlit white walls on Step 0 so neon only painted glowing scribbles over bright white brick instead of shifting the room to midnight neon |
| `w1_mv_sam3_subj_chrome` | `mv_sam3` | `141` | `104.1` | `11.328` | `-24.62` | `0.1680` | `87.19` | `0.2356` | **Plumbing-only partial (overcooked `last_step = 1000`):** studio walls stayed locked and mask tracked cleanly, **but** `scale = 2.0, last_step = 1000` (violating `last_step = 3` on 6-step Pika) overcooked steps `4..5` into a burnt-orange chest while leaving the black trousers untouched cloth |
| `w1_opera_dog_plain` | `opera_dog` | `192` | `110.1` | `0.000` | `-15.38` | `0.0702` | `364.26` | `0.7813` | **Plain wins on audio quality:** already keeps a photoreal golden retriever belting rich, full-orchestra *Nessun Dorma* tenor with synced jaw motion and zero barking |
| `w1_opera_dog_router_g000` | `opera_dog` | `192` | `108.1` | `30.590` | `-33.71` | `0.0452` | `445.54` | `0.6823` | **Worse overall mix:** shifted pitch `+3.5` st to female soprano (`445.54` Hz), **but** came out thin/hollow and `18.3` dB quieter than plain (`-33.71` vs `-15.38` dBFS) |
| `w1_hazy_ref_c999` | `hazy_ref` | `141` | `112.1` | `0.000` | `-49.93` | `0.1190` | `68.79` | `0.0077` | Clean bright alpine snow relighting of `old_man.jpg` in a blue parka from Frame 0 with zero original corridor bleed |
| `w1_hazy_ref_c500` | `hazy_ref` | `141` | `98.0` | `28.163` | `-49.02` | `0.1390` | `70.66` | `0.0154` | **Falsified as dreamy dial:** Frame `0` flashes `old_man.jpg`'s original dark hallway + cap + cheek talisman before hard-snapping to the snowy peak |
| `w1_hazy_ref_c350` | `hazy_ref` | `141` | `98.0` | `37.957` | `-52.51` | `0.3045` | — | `0.0000` | **Falsified as dreamy dial:** Stronger Frame-`0` corridor double-exposure flash before snapping to the snowy peak; `00:01–00:05` stays nearly identical to `c999` |

---

## 2. Key Takeaways & Bug Fixes Needed Before Re-Testing `SAM 3.1` Splits

1. **5th Independent Proof of the 1-Hop `aud_to_vid_gain = 0.0` Firewall (`knight_duck` — Sole Clean Wave 1 Home Run):**
   - Visual steel-sword physics overrode plain H3's explicit *"zero metallic ring, strictly rubber-duck squeaks"* instruction (`0.3944` HF scrape). Stripping knight/sword/metal words from `audio_conditioning` and setting `aud_to_vid_gain = 0.0` cleanly eliminated every trace of metallic sound while preserving grim armored visuals and timing two crisp rubber-duck squeaks + a fairy harp glissando directly to the sword draw and swing.
2. **`SAM 3.1` + `MiniMaxH3FreezeAudio` + `MiniMaxH3EditGuider` (`w1_mv_*` — Plumbing Verified, Visual Recipe Needs 2 Fixes):**
   - **What actually worked:** `MiniMaxH3FreezeAudio` -> cached `VAEDecode` -> `MiniMaxH3SubjectTrack` (`sam3.1_multiplex_fp32.safetensors`, `[141, 768, 1344]`) -> `MiniMaxH3EditGuider(streams="video_only")` -> mux `[freeze_node, 1]` ran end-to-end with zero ghosting and bit-identical lossless audio (`-24.62 dBFS`).
   - **Two mistakes that hurt visual quality in Wave 1:** (a) running `scale = 2.0, last_step = 1000` violated `docs/wiki/edit_guidance.md`'s **`last_step = 3`** rule on 6-step Pika and overcooked steps `4..5` into burnt-orange specular patches on `subj_chrome` (while dark trousers stayed cloth); (b) running `first_step = 1` locked Step 0's sunlit white brick walls so `bg_neon` could only paint neon scribbles over bright white brick instead of shifting to dark midnight neon.
   - **Next re-test recipe:** Set `last_step = 3`, `scale = 1.5–1.6`, and either `first_step = 0` or lighting-compatible background edits.
3. **Why `clean_level < 0.999` Was Cut From Studio 2.0 (`hazy_ref`):**
   - Lowering `MiniMaxH3ReleaseReferences.clean_level` (`visual_cond_noise_aug`) from `0.999` to `0.50`/`0.35` brings `<Picture 1>`'s AdaLN timestep closer to noisy early video target rows (`max(t_v, vis_aug)`), causing Frame `0` to briefly flash `<Picture 1>`'s original dark room/wardrobe before snapping to the target scene. Keep `clean_level = 0.999` as default (use `end_percent = 0.35..0.50` when releasing reference layout mid-denoise).
4. **Honest Baseline Check (`opera_dog`):**
   - Plain H3 already sings rich, full-orchestra Italian opera through a photoreal golden retriever without barking (`-15.38 dBFS`, `364.26 Hz` tenor); `a2v = 0.0` shifted pitch `+3.5` st to female soprano (`445.54 Hz`) at the cost of a thin/hollow `-33.71 dBFS` mix — confirming that `a2v = 0.0` is for strong visual-to-audio physics conflicts (`knight_duck`, `uwvid_dryaud`, `shellshock`, `biker`), not scenes plain prompting already handles cleanly.
