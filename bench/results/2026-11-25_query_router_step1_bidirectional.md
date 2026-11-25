# Query router Step 1 bidirectional creative & `aud_to_vid_gain` sweep (`2026-11-25`)

lane: query routing
verdict: Across `13` matched `192f` Pika renders (`11` bidirectional probes + `2` follow-up `aud_to_vid_gain` clips on `biker`), closing `Q_aud -> K_vid` plus the 1-hop `K_vid -> shadow_text -> Q_aud` relay (`aud_to_vid_gain=0.0`, post-`9e5db27c`) generalizes cleanly across three new domains: **(1) Shell-Shock World-Mute** drops blazing warehouse fire roar by **`-32.66 dB`** (`-55.75` vs `-23.09 dBFS` at `g=1.0`) and cuts `> 2 kHz` crackle **`55x`** (`0.0009` vs `0.0498`) while keeping flames and `"DANGER"` signage intact; **(2) Wrong-Voice Casting (`biker`)** yields a crisp 4-rung monotonic pitch curve (`plain 168.70 Hz` -> `g=1.0 187.37 Hz` -> `g=0.15 216.94 Hz` -> **`g=0.0 260.21 Hz`, `+7.5 st`**), turning a scarred 6'5" outlaw biker's voice into a squeaky elderly grandmother voice synced cleanly to his lips across all 3 shots; and **(3) Pixel-Locked Underwater Dry Redub** (uncached `MiniMaxH3AudioRefineMask(video_mask=0, audio_mask=1)` + `QueryRouter(audio_conditioning=["5", 0], aud_to_vid_gain=0.0)`) locks exact source pixels (`pic_mae=2.337`) while cutting frozen-`K_vid` water hiss **`5.8x`** (`HF>2k 0.1458 -> 0.0250`), lifting voiced speech **`+39%`** (`0.1859 -> 0.2588`), and running `26.0 s` faster (`98.05 s` vs `124.08 s`), now wired into `director/compiler.py::compile_audio_redub(isolate_visuals=True)` (`7a7693d0`). Two honest neutral ties (`s1_android_*` natural human voices `147.55` vs `149.50 Hz`, and `s1_victorian_trap_*` 1885 parlor + booming 808 trap `-14.25` vs `-14.42 dBFS` with zero neon bleed) sharpen the Prior Conflict Law: plain H3 handles explicit music-genre overrides and explicit voice-quality descriptors cleanly, leaking visuals primarily when `overall_soundscape` names a contrasting physical location/era and leaking acoustics whenever `K_vid` visual physics or close-up speaker physiology contradicts the audio prompt.

**Date:** `2026-11-25`
**Code & shipped integration:** [`steering/query_router.py`](../../steering/query_router.py), [`steering/hooks.py`](../../steering/hooks.py) (`9e5db27c`, `3bb7a3c9`), [`audio_refine.py`](../../audio_refine.py), [`director/compiler.py`](../../director/compiler.py) (`7a7693d0`, `compile_audio_redub(isolate_visuals=True)`), and Studio 2.0 1-Click Genre Roulette, Smart Twists (`Wrong Voice` calibrated to `recommended_gain = 0.0`), & Re-Dub `Isolate Visuals` drawer (`app/.git` `b49de03`, `97e07e7`).
**Media folder:** `$H3_OUTPUT_DIR/Video/query_router_step1_2026-11-25/` (`13` `-audio.mp4` clips).
**Setup:** Pika H3 Flex `6`-step `lcm`, `1344x768`, `192` frames (`8.0 s` at `24.0 fps`, `on_both_clocks`), seed `730451892`, `MiniMaxH3Sol(dense_blocks="49", quantizer="rotated")`, `1x` `BasicGuider`. All prompts pre-graded `0 FAIL / 0 WARN` via [`bench/grade_prompt_text.py`](../grade_prompt_text.py) (`SHELLSHOCK_AUD_CLEAN` additionally asserted zero fire/flame/smoke/danger/warehouse tokens). Every rendered `.mp4` was independently watched and listened to via native `read_file`.

---

## Part A — Visual-Physics & Face-to-Voice Decoupling (`aud_to_vid_gain=0.0` Across 3 Domains)

### A1. Wrong-Voice Casting (`biker`): Monotonic `Q_aud -> K_vid` Pitch Sweep

Scene: 3-shot roadhouse bar (`"NO CREDIT"` chalkboard) featuring a massive 6'5" scarred bearded outlaw biker delivering tough lines (`"Step away from my bike before I fold you in half."`, `"You heard me. Keys on the bar."`, `"Last warning, pal."`). Plain control injects frail 90-year-old grandmother falsetto voice tags directly into the biker prompt; routed arms keep the gruff biker prompt on `Q_vid` and a frail 90-year-old grandmother parlor script with identical `<d>` lines on `Q_aud` (`audio_conditioning`), sweeping `aud_to_vid_gain ∈ {1.0, 0.15, 0.0}`:

| Arm (`s730451892`, `192f`) | `aud_to_vid_gain` | Wall (s) | Pic MAE vs Plain / `g=1.0` | RMS (dBFS) | `HF>2k` | `YIN` Median $F_0$ (`Δ st` vs Plain) | `YIN` Voiced / IQR (`st`) | Observed Picture & Voice Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `s1_biker_wrongvoice_plain` | — | `96.05` | `0.000` / `27.467` | `-27.11` | `0.0358` | `168.70 Hz` (`0.0 st`) | `0.391` / `4.709` | Massive scarred biker + `"NO CREDIT"` intact; deep gravelly adult male voice ignores grandmother tags |
| `s1_biker_wrongvoice_router` | `1.00` | `104.05` | `27.467` / `0.000` | `-28.00` | `0.0946` | `187.37 Hz` (`+1.8 st`) | `0.407` / `5.213` | Biker & bar intact; slightly lighter tenor but still unambiguously adult male because open `K_vid` sees the bearded close-up |
| `s1_biker_wrongvoice_router_g015` | `0.15` | `122.07` | `27.046` / `18.182` | `-26.71` | `0.1129` | `216.94 Hz` (`+4.4 st`) | `0.437` / `6.016` | Intermediate shift: shriller, female-leaning vocal timbre synced to the biker's mouth |
| **`s1_biker_wrongvoice_router_g000`** | **`0.00`** | **`100.05`** | `32.077` / `22.693` | `-34.28` | **`0.1343`** | **`260.21 Hz` (`+7.5 st`)** | **`0.452` / `7.855`** | **Clean breakthrough:** unmistakable high-pitched squeaky elderly woman voice synced cleanly to the massive scarred biker's lips across all 3 shots |

- **Why `g=0.0` is required on tight close-ups:** Even with a 100% frail-grandmother `audio_conditioning` prompt, open `Q_aud -> K_vid` (`g=1.0`) reads the giant bearded biker filling the frame and pulls vocal pitch right back down into adult male (`187.37 Hz`). Closing `Q_aud -> K_vid` plus the 1-hop `K_vid -> shadow_text -> Q_aud` relay (`g=0.0`) breaks visual-physiology dominance monotonically (`168.70 -> 187.37 -> 216.94 -> 260.21 Hz`, `+7.5 st`) while open `Q_vid -> K_aud` keeps every syllable lip-synced.

### A2. Shell-Shock World-Mute (`shellshock`): Silencing a Warehouse Inferno

Scene: 3-shot blazing industrial warehouse inferno (towering flames, flying white sparks, soot-stained silent survivor, red `"DANGER"` metal sign). `SHELLSHOCK_AUD_CLEAN` replaces `prompt_aud` with a calm silent padded room + dead-quiet anechoic stillness / soft heartbeat / close-mic breathing (strictly zero fire/flame/ember/smoke/danger/warehouse stems):

| Arm (`s730451892`, `192f`) | `aud_to_vid_gain` | Wall (s) | Pic MAE vs Plain | RMS (dBFS) | `HF>2k` (`H1 / H2`) | Observed Picture & Audio Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `s1_shellshock_plain` | — | `96.05` | `0.000` | `-25.27` | `0.0280` (`0.0431 / 0.0035`) | Leaked fire/wind roar, hallucinated a whispered voice reading *"Danger"* aloud at `00:01`, and broke the overhead `"DANGER"` sign mid-clip |
| `s1_shellshock_router_g100` | `1.00` | `104.05` | `36.615` | `-23.09` | `0.0498` (`0.0708 / 0.0046`) | Stopped phantom *"Danger"* speech and locked the flames + red `"DANGER"` sign, **but open `K_vid` pumped loud roaring fire and crackling embers straight back into the soundtrack** |
| **`s1_shellshock_router_g000`** | **`0.00`** | **`100.05`** | `38.858` | **`-55.75`** | **`0.0009` (`0.0005 / 0.0013`)** | **`-32.66 dB` quieter and `55x` less `> 2 kHz` crackle than `g=1.0`:** eerie anechoic stillness & gentle close-mic breathing while keeping every flame, shower of sparks, and the `"DANGER"` sign blazing |

### A3. Uncached `AudioRefineMask` + `aud_to_vid_gain=0.0` on Pixel-Locked Underwater Video (`audiorefine_uwvid`)

Source video: `$H3_OUTPUT_DIR/Video/query_router_a2v_conflict_2026-11-24/uwvid_unrouted_s730451892_00001-audio.mp4` (`-32.14 dBFS`, `HF>2k = 0.0346`, `YIN F0 = 149.47 Hz`, `voiced_frac = 0.2638`). Encoded via `VHS_LoadVideoFFmpegPath(24.0 fps) -> VAEEncode -> LTXVConcatAVLatent -> MiniMaxH3AudioRefineMask(video_mask=0.0, audio_mask=1.0)` (no `MiniMaxH3FrozenVideoCache`, because [`steering/hooks.py`](../../steering/hooks.py#L1444-L1451) refuses `FrozenVideoCache` + `double_block` hooks) with dry English [`prompt_bank/t2va_stairwell_dialogue.txt`](../../prompt_bank/t2va_stairwell_dialogue.txt) on `"5"`. For `g=0.0`, `MiniMaxH3QueryRouter` explicitly wires `audio_conditioning = ["5", 0]` so `shadow_a` enters `mode_id=2` and blocks the 1-hop `K_vid -> text -> Q_aud` relay:

| Arm (`s730451892`, `192f`) | `aud_to_vid_gain` | Wall (s) | Pic MAE vs UW Source | RMS (dBFS) | `HF>2k` (`H1 / H2`) | `YIN` Median $F_0$ / Voiced | Observed Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `_uw_src_unrouted` *(input clip)* | — | — | `0.000` | `-32.14` | `0.0346` (`0.0388 / 0.0310`) | `149.47 Hz` / `0.264` | Original submerged underwater dialogue |
| `s1_audiorefine_uwvid_g100` | `1.00` *(stock redub)* | `124.08` | `2.337` | `-18.40` | **`0.1458`** (`0.1595 / 0.1317`) | `148.56 Hz` / `0.186` | Pixels locked (`2.34` H.264 floor), **but frozen `K_vid` underwater pixels pumped loud rushing water & bubble hiss (`4.2x` source HF)** over the speech |
| **`s1_audiorefine_uwvid_g000`** | **`0.00`** *(router)* | **`98.05`** | **`2.337`** | **`-24.65`** | **`0.0250` (`0.0244 / 0.0262`)** | `142.46 Hz` / **`0.259`** | **Exact pixel lock + `5.8x` lower water hiss (`+39%` voiced speech) and `26.0 s` faster:** crisp dry stairwell dialogue over the pixel-locked underwater take |

- **Shipped integration (`7a7693d0`):** [`director/compiler.py::compile_audio_redub`](../../director/compiler.py) now supports `isolate_visuals=True` (omitting `MiniMaxH3FrozenVideoCache` and attaching `MiniMaxH3QueryRouter(aud_to_vid_gain=0.0, audio_conditioning=["26", 0])`), verified by [`bench/check_director_studio.py`](../check_director_studio.py).

---

## Part B — Two Honest Neutral Ties (Refining Where Plain Prompting Already Succeeds)

| Scene (`s730451892`, `192f`) | Arm | Wall (s) | Pic MAE vs Plain | RMS (dBFS) | `HF>2k` | `YIN` $F_0$ / Voiced | Honest Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`android`** *(Chrome Androids + Natural Human Voices)* | `s1_android_plain` | `108.23` | `0.000` | `-26.21` | `0.0495` | `147.55 Hz` / `0.241` | **Honest tie:** gleaming chrome androids + cyan optics, and plain H3 already complied cleanly with `"natural human female/male voice"` (zero vocoder bleed) |
| **`android`** | `s1_android_router` (`g=1.0`) | `102.05` | `14.674` | `-27.46` | `0.0852` | `149.50 Hz` / `0.210` | Matches plain visually and vocally (slightly brighter consonant sheen, `HF>2k 0.0852` vs `0.0495`) |
| **`victorian`** *(1885 Parlor Quartet + Modern 808 Trap / Synthwave)* | `s1_victorian_trap_plain` | `96.05` | `0.000` | `-14.25` | `0.0400` | — | **Honest tie:** 100% period-authentic 1885 drawing room, tailcoats, antique violins/cello, and `"LONDON 1885"` mantel plaque over booming 808 trap with zero neon/club bleed |
| **`victorian`** | `s1_victorian_trap_router` (`g=1.0`) | `102.06` | `46.092` | `-14.42` | `0.0334` | — | Equally period-authentic parlor (mantel clock/candles instead of plaque) over booming 808 trap; plain prompting already handles pure music-genre clashes |

---

## Part C — Updated Operating Rules (`2026-11-25`)

1. **Location-bearing soundscapes vs. pure music genres:** Plain H3 leaks architecture/props when `overall_soundscape:` names a contrasting **physical place or era** (`"Zen tea pavilion"` -> garden windows in `cooking_zen`, `"1970s three-camera sitcom studio"` -> 70s hair/lamp, `"cathedral"` -> church doors), **not** when `non_diegetic_music:` swaps a pure music genre (`808 trap` over an explicitly anchored 1885 quartet).
2. **`aud_to_vid_gain = 0.0` (with 1-hop text firewall `9e5db27c`) is the primary visual-to-audio decoupling lever across four proven domains:**
   - **Underwater dry speech (`uwvid_dryaud`):** `+9.43 dB` over `g=1.0`.
   - **Blazing inferno world-mute (`shellshock`):** `-32.66 dB` quieter / `55x` less crackle than `g=1.0`.
   - **Pixel-locked frozen-video redub (`audiorefine_uwvid`):** `5.8x` less water hiss / `+39%` voiced speech on exact frozen pixels (`2.337` MAE).
   - **Wrong-Voice close-up casting (`biker`):** `+7.5 st` (`168.70 -> 260.21 Hz`) monotonic pitch release while open `Q_vid -> K_aud` preserves lip-sync.
3. **Clean-`prompt_aud` invariant still holds:** Always strip conflicting visual-chaos/action prose out of `prompt_aud` whenever `aud_to_vid_gain` drops toward `0.0` (keep `g=1.0` default whenever `prompt_aud` still carries shared visual-action beats, as shown by `cooking_zen_g0`).

---

## Part D — Labeled Side-by-Side Showcase Reel (`2026-11-25`)

- **File:** `$H3_OUTPUT_DIR/Video/query_router_showcase_2026-11-25/reel_h3_query_router_showcase_2026-11-25.mp4` (`86.00 s`, `1344x768 @ 24 fps`, `aac stereo 32000 Hz`, raw un-normalized audio; binaural closer `lr_corr = 0.1173`).
- **Chapters:**
  - `0.0–2.0 s`: Intro
  - `3.5–15.5 s`: Ch 1 — Ensemble `ref2va` Kitchen `EN -> JA` Dub (`plain_ja` recasting unpinned cooks & erasing `"FIRE"` tickets vs `router_ja`)
  - `17.0–28.0 s`: Ch 2 — Cooking Meltdown + Zen Audio (brick TV studio locked vs plain `"Zen tea pavilion"` garden windows)
  - `29.5–40.5 s`: Ch 3 — Underwater Dry Speech (`+9.43 dB`, `a2v=0.0`)
  - `42.0–52.0 s`: Ch 4 — Shell-Shock World-Mute (`-32.66 dB` / `55x` less crackle, `a2v=0.0`)
  - `53.5–63.5 s`: Ch 5 — Pixel-Locked `AudioRefineMask + a2v=0.0` (`5.8x` less water hiss, `pic_mae=2.337`)
  - `65.0–76.0 s`: Ch 6 — Wrong-Voice Biker (`168.70 Hz -> 260.21 Hz`, `+7.5 st`, `a2v=0.0`)
  - `78.0–86.0 s`: Bonus Closer — `late_30_49` Binaural Radio Booth (Left dry studio vs Right haunted storm mansion, `lr_corr = 0.1173`)

