# 2026-11-23 — Diverse 4-Shot Sound-Mixer Film (`565f` / `23.54s`) & Live `heylook 2.0.276` Verification

**Runner:** [`bench/run_multishot_film_2026_11_23.py`](../run_multishot_film_2026_11_23.py)  
**Outputs:** `$H3_OUTPUT_DIR/Video/multishot_film_2026_11_23/` (`multishot_radio_studio_master_564f.mp4`, `14.27 MB`, `1152x768`, `24.0 fps`)  
**Anchor Input:** `$H3_INPUT_DIR/multishot_radio_studio_anchor_20261123.png`  
**Transient Telemetry:** [`data/multishot_film_2026_11_23/metrics.json`](../../data/multishot_film_2026_11_23/metrics.json), [`data/multishot_film_2026_11_23/heylook_provenance.json`](../../data/multishot_film_2026_11_23/heylook_provenance.json)

---

## 1. Objective & Anti-Crutch Cast Setup

Verify the clean-slate [`steering/mixer.py`](../../steering/mixer.py) (`MixerCompiler`) pipeline end-to-end on restarted ComfyUI `:8188` and `heylook 2.0.276` across a **4-person deadpan radio-drama studio scene** (`<Subject 1>` silver-haired actor in burgundy tweed waistcoat `(S1)`, `<Subject 2>` emerald-scarf actress `(S2)`, `<Subject 3>` headphone-wearing young actor `(S3)`, `<Subject 4>` deadpan foley artist behind prop gravel tray and tabletop wind drum) with **zero** `noir`, `film_16mm*`, `covered_market*`, or `woman_blonde*` crutches:

1. **Anchor Keyframe (`anchor_radio_studio_t2i`, `seed=8401`, `29.24s`):** `P-H3-Image` `T2I` (`4.0 MP` downscaled from `8.0 MP`, `3:2`) produced crisp four-subject horizontal separation and red `"ON AIR"` signage under practical tungsten studio globes.
2. **Live `heylook 2.0.276` Continuity Audit (`model=""` Auto, `preset=""` -> `h3-director-shot-continue`):** Executed against `$H3_INPUT_DIR/multishot_radio_studio_anchor_20261123.png` with authenticated `H3_HEYLOOK_API_KEY_FILE` and logged all 14 canonical `HeylookProvenance` keys:
   - `resolved_model`: `"gemma-4-31B-it-8bit"`
   - `preset`: `"h3-director-shot-continue"` (`system_source`: `"preset:h3-director-shot-continue"`)
   - `system_prompt_sha256`: `"501deb742a72e11be78595f0f840078936e8d30d48519baa05436d12ea03ff41"`
   - `thinking`: `"model default"`, `temperature`: `1.0`, `seed`: `null`, `repair_used`: `false`
   - `input_tokens`: `2096`, `output_tokens`: `2972`, `created_utc`: `"2026-11-23T22:39:12Z"` (`92,476 ms` gen / `35.92 tok/s`, `40.20 GB` Apple M2 Ultra)

---

## 2. Per-Shot `MixerCompiler` Plans & Quantitative Metrics

All 4 shots ran at `1152x768 @ 141f` (`on_both_clocks`, `24.0 fps`, `5.875s` each) using Pika `Ref2VA` (`minimax_h3_ref2va_pruned_int8_convrot.safetensors` + `pika_h3_flex_ref2va_svd995_bf16.safetensors` + `MiniMaxH3Sol(dense_blocks="49")`) with matched `bank/*__ref2va__diff_span__lora_865ee189.safetensors` vectors and `MiniMaxH3ReleaseReferences(end_percent=0.33, clean_level=0.999)`:

| Shot (`seed`) | `MixerCompiler` Plan | Steered Stems (`bus`, `strength`, `target`, `streams`) | Time (`s`) | Mean Luma | Interframe MAE (`mean / max`) | Audio RMS / Peak (`dBFS`) | Centroid (`Hz`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`shot1_opera_opening`** (`8411`) | `1x BasicGuider` (`cathedral_opera`) | `operatic_bellow` (`audio`, `1.15`, `description_only`, `both`), `reverb_cathedral` (`audio`, `1.10`, `soundscape_only`, `audio_only`) | `123.84` | `73.13` | `7.344 / 10.747` | `-13.31 / +1.99` | `2526.5` |
| **`shot2_timed_blackout`** (`8412`) | `2x MiniMaxH3EditGuider` (`scale=1.65`, `streams="both"`, `MiniMaxH3SpatialDirector` + `MiniMaxH3SoundCues` @ `f48` `audio_end_percent=0.60`) | `power_outage_flashlight` (`look`, `1.30`, `scene_only`, `video_only`, `edited_only`) | `166.40` | `38.49` *(pre `72.66` -> post `20.85`)* | `5.087 / 48.429` | `-16.08 / +2.09` | `2426.0` |
| **`shot3_blizzard_windmachine`** (`8413`) | `1x BasicGuider` (`blizzard_gale`) | `heavy_blizzard` (`look`, `1.10`, `scene_only`, `both`), `wind_tunnel_gale` (`motion`, `1.05`, `description_only`, `both`) | `63.84` | `93.77` | `11.059 / 22.809` | `-10.99 / +0.68` | `2220.5` |
| **`shot4_uv_auctioneer_finale`** (`8414`) | `1x BasicGuider` (`3`-strip cross-bus stack) | `blacklight_uv_rave` (`look`, `1.15`, `scene_only`, `video_only`), `strobe_club_pulse` (`look`, `0.95`, `scene_only`, `video_only`), `auctioneer_rapid_patter` (`audio`, `1.05`, `description_only`, `both`) | `113.03` | `86.74` | `22.499 / 51.682` | `-10.96 / +2.12` | `1984.4` |
| **Concatenated Master** | `multishot_radio_studio_master_564f.mp4` | — | **`467.11`** *(4 video renders)* | — | — | `-12.40 / +2.18` | `2171.1` (`565f`, `23.542s`) |

---

## 3. Native Visual & Audio Inspection (`read_file`, `$0` API Spend)

Inspected `anchor_radio_studio_00001_.png`, all four `.1fps.00000-00140.mp4` strips, and the `24 fps` `14.27 MB` concatenated `multishot_radio_studio_master_564f.mp4` directly via native `read_file`:

- **`anchor_radio_studio_00001_.png`:** Clean 4-subject blocking across left prop table (`<Subject 4>`), three boom stands (`<Subject 1..3>`), gravel tray, wind drum, and sharp `"ON AIR"` wall sign.
- **`shot1_opera_opening` (`00:00–00:06`):** Strong theatrical voice timbre and reverb on `(S1)` (`"Hark! The cellar bolts are giving way tonight!"`) and `(S2)` (`"Then seal the vault before the tide comes in!"`) paired with gravel rustle and padlock drop; minor caveat: silent `<Subject 3>` opens his mouth gasping around `00:02–00:05` despite silent audio.
- **`shot2_timed_blackout` (`00:06–00:12`):** Cleanest technical validation of the run. Warm practical booth illumination holds steady through `f0..f47` (`pre_blackout_luma = 72.66`), then an audible breaker `click` (`MiniMaxH3SoundCues` @ `f48`) drops mean luma by **`-51.81`** (`post_blackout_luma = 20.85`), revealing the glowing red `"ON AIR"` box and sweeping flashlight cone from `<Subject 4>` while `(S3)` delivers `"Wait, did someone trip the main studio breaker just now?"`.
- **`shot3_blizzard_windmachine` (`00:12–00:17`):** Strong multi-bus (`Look + Motion`) composition with zero over-saturation: `<Subject 4>` cranks the wooden wind drum while horizontal frost mist blasts left-to-right, script pages fly airborne, actors shield their eyes, and `(S2)` shouts cleanly over the howling wind roar.
- **`shot4_uv_auctioneer_finale` (`00:17–00:23.5`):** Spoken rapid-fire auctioneer patter (`(S1)`) and crisp final table slap rendered clearly, **but visual inspection caught zebra-stripe UV pattern distortion across actors' faces and bright halos over `00:19–00:22`** caused by stacking two high-gain `Look`-bus transforms (`blacklight_uv_rave=1.15` + `strobe_club_pulse=0.95`) simultaneously (`interframe_mae_mean = 22.499`). **Takeaway:** Keep simultaneous active `Look`-bus fader sums moderate or attenuate per-strip gain (`<= ~0.7` each) when combining two `Look` stems (`mixer.py` stack-headroom guidance).
- **Master framing note:** `ffmpeg -f concat` across `5.875s` video + `5.856s` AAC-padded clips emitted `565` video frames (`23.542s`, audio `23.435s`), adding 1 duplicate boundary frame relative to the `4 x 141 = 564f` nominal sum.
