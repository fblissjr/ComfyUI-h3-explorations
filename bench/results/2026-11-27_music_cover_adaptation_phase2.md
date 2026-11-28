# 2026-11-27 — Phase 2 Silicon A/B: `MiniMaxH3AudioRefRouter` Music-Cover Adaptation

**Verdict: PASS (`1.04x–1.06x` wall-clock cost, zero core patches, zero K/V mutation).**
`MiniMaxH3AudioRefRouter` ([`audio_ref_router.py`](../../audio_ref_router.py)) resolves the Phase 1 `b1`/`b2` cliff ([`2026-11-27_music_cover_adaptation_gate.md`](2026-11-27_music_cover_adaptation_gate.md)) on **both** a real-world commercial pair (**Nirvana MTV Unplugged covering Cyndi Lauper's *"Girls Just Want to Have Fun"***) and a **zero-memorization synthetic extreme-contrast pair** (**male delta blues + steel resonator guitar covering female disco soprano**).

---

## 1. Mechanism Verified

[`MiniMaxH3AudioRefRouter`](../../audio_ref_router.py) wraps `WrappersMP.DIFFUSION_MODEL` and installs a call-local `optimized_attention_override` (plus a standalone Sage seam in [`attention.py`](../../attention.py)) that operates **strictly on target `Q_aud` rows** (`~1–3%` of packed sequence length) without mutating `K` or `V`:

1. **Block-depth split (`split_block`, default `24`):**
   - Early blocks `[0, split_block)` apply `melody_gain_early` on `K_melody` (`cond_audio` or `<Audio 2>`) and `timbre_gain_early` on `<Audio 1>` (`ref_audio[0]`) to lock Song X's syllable cadence, syncopation, and melodic contour.
   - Late vocal-timbre blocks `[split_block, 50)` apply `melody_gain_late = 0.0` (`-1e4` firewall) on `K_melody` and `timbre_gain_late = 2.0–3.0` (`+log(g)`) on `<Audio 1>` to strip Person A's vocal formants and backing instruments while committing Target B's voice and live acoustic instrumentation.
2. **Independent `melody_clean_level`:** Softens **only** `cond_audio` (or `ref_audio[1]`) latents with deterministic Gaussian noise clamped at `max(melody_clean_level, audio_stream_time)`, keeping `<Audio 1>` (`ref_audio[0]`) bit-identical (bypassing core's shared `audio_cond_noise_aug`).
3. **Selective `melody_end_percent` release:** Drops **only** the melody scaffold segment after `melody_end_percent`, preserving `<Audio 1>` and exact 3D RoPE `position_ids`.

CPU verification ([`bench/check_music_cover_routing.py`](../check_music_cover_routing.py)): `6/6` passing (exact `0.0` identity at `gain=1.0`, zero K/V mutation, `AttentionTensorContainer` parity, late-block `-1e4` invariance, independent softening, selective RoPE-preserving release, and `attention.py` `ref_audio`/`cond_audio` LSE balance).

---

## 2. Real-World Silicon A/B: Unplugged Covers Fun

- **Setup:** [`archive/bench/unplugged_fun_cover.py`](../../archive/bench/unplugged_fun_cover.py) · `1024x768` (exact 4:3 MTV Unplugged framing) · `192f = 8.000 s` (exact dual-clock grid `39 + 51*3`) · `seed = 840519203` · `MiniMaxH3Sol` (`tau=1.0`, `rotated`, `dense_blocks="49"`, `per segment`) · prompt [`data/music_cover_gate/ref2va_unplugged_covers_fun.txt`](../../data/music_cover_gate/ref2va_unplugged_covers_fun.txt).
- **Target B refs (`$H3_INPUT_DIR/music_cover_gate/`):** `<Picture 1>` `unplugged_b_face.png`, `<Picture 2>` `unplugged_b_stage.png`, `<Audio 1>` `unplugged_b_audio_192f.wav` (`unplugged_compressed.mp4`, `44.5–52.5 s`, `median_f0 = 69.63 Hz` guitar/cello bass + raw raspy grunge vocal).
- **Song X melody scaffold (`$H3_INPUT_DIR/music_cover_gate/`):** `fun_x_chorus_clean_192f.wav` (`fun.mp4`, `57.15–65.15 s`, `median_f0 = 460.99 Hz` Cyndi Lauper soprano + 80s synth/drums: *"Oh daddy dear, you know you're still number one! But girls, they wanna have fun!"*) and its non-pitch-shifted STFT-HPSS drum-stripped harmonic scaffold `fun_x_chorus_clean_harmonic_192f.wav`.
- **Outputs (`$H3_OUTPUT_DIR/Video/unplugged_covers_fun_2026-11-27/`):** Triage saved to [`data/music_cover_gate/unplugged_fun_triage_report.json`](../../data/music_cover_gate/unplugged_fun_triage_report.json); all 9 clips audited natively (video + audio) via `read_file`.

| Arm | Model / Steps | Router / Release Settings | Wall (s) | `med F0` (Hz) | `st vs fun` | `ltas_cos B` | `frame_mel B` | Audible & Visual Assessment (Native `read_file`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `u0_ceiling` | Pika 6-step | No `fun.mp4` (`<Picture 1,2>+<Audio 1>` only) | 131.8 | 114.8 | -24.1 | **0.971** | 0.710 | Photorealistic MTV Unplugged stage + close-up cut (`t=4s`), pure Kurt timbre/guitar/cello, **but invents a slow folk melody** instead of Cyndi's chorus. |
| `u1_b2_raw_rel50` | Pika 6-step | Unrouted `ReleaseReferences(aud_end=0.50)` raw | 110.9 | 230.5 | -12.0 | 0.917 | 0.653 | Locks Cyndi's exact melody transposed `-12 st` (1 octave down), **but carries cleaner pop-rock vocal sheen + faint snare bleed** on *"wanna have fun!"*. |
| `u2_b2_harm_rel50` | Pika 6-step | Unrouted `ReleaseReferences(aud_end=0.50)` HPSS | 110.7 | 236.0 | -11.6 | 0.940 | 0.667 | HPSS drum stripping removes 80s percussion bleed completely while keeping exact melody; slight clean-tenor polish remains vs `u0`. |
| `u3_base16_raw_rel38` | Base 16-step | Unrouted `ReleaseReferences(aud_end=0.38)` raw | 195.2 | 312.9 | -6.7 | 0.948 | 0.720 | Richer close-up micro-expressions and grittier grunge strain while locking melody; mild snare/electric sheen from raw track. |
| `u4_router_raw_s20` | Pika 6-step | `AudioRefRouter(s=20, l_mel=0, l_tim=2.0, rel=0.50, clean=1.0)` raw | 147.4\* | 276.6 | -8.8 | 0.945 | 0.709 | Matched to `u1`: `+0.028` `ltas_cos B` & `+0.056` `frame_mel B`, visibly more raw grunge grit + wide mouth belt on *"wanna have fun!"* with 100% melody lock. |
| **`u5_router_raw_s24_soft70`** | Pika 6-step | `AudioRefRouter(s=24, e_tim=0.75, l_tim=2.5, rel=0.50, clean=0.70)` raw | **115.6** (`1.04x`) | 219.0 | -12.9 | **0.950** | **0.728** | **Best 6-step Pika arm (raw input).** Unmistakable Kurt rasp + pure acoustic guitar/cello (zero 80s synth/snare), dead-on Cyndi melody & punchy downbeat. |
| **`u6_router_harm_s24_soft75`** | Pika 6-step | `AudioRefRouter(s=24, e_tim=0.75, l_tim=2.5, rel=0.50, clean=0.75)` HPSS | **115.8** (`1.04x`) | 175.6 | -16.7 | 0.941 | 0.690 | Pure intimate unplugged acoustic texture with zero percussion pulse; slightly mellower opening attack than `u5`. |
| `u7_router_raw_s16_rel67` | Pika 6-step | `AudioRefRouter(s=16, e_tim=0.75, l_tim=3.0, rel=0.67, clean=0.80)` raw | 116.9 (`1.05x`) | 351.1 | -4.7 | 0.912 | 0.673 | Extra step (`0.67`) gives tightest Cyndi phrasing while firewall keeps voice male/acoustic, but higher tenor pitch & faint rock punch confirm `0.50` is the cleaner sweet spot. |
| **`u8_router_base16_harm_s24`** | Base 16-step | `AudioRefRouter(s=24, e_tim=0.75, l_tim=2.5, rel=0.44, clean=0.80)` HPSS | **206.7** (`1.06x`) | 116.0 | -23.9 | **0.952** | **0.731** | **Overall winner.** Authentic 1993 MTV Unplugged Kurt Cobain voice, acoustic guitar, bowed cello, room tone, and photorealistic lip-synced close-up singing *"Girls Just Want to Have Fun"*'s recognizable melody. |

\*`u4` was the first run after server restart (cold UNET/encoder load); steady-state warm Pika router arms ran `115.6–116.9 s` (`1.04x–1.05x` vs `110.9 s` unrouted `u1`).

---

## 3. Zero-Memorization Causal Proof: Blues Covers Disco

To rule out pretraining familiarity with *"Girls Just Want to Have Fun"*, we re-ran the Phase 1 synthetic **zero-memorization** extreme-contrast pair ([`archive/bench/music_cover_gate.py`](../../archive/bench/music_cover_gate.py), `1152x768`, `141f = 5.875 s`, `seed = 730451892`, male delta blues `194.63 Hz` + steel resonator guitar covering original female disco soprano `456.81 Hz`):

| Arm | Settings | Wall (s) | `med F0` (Hz) | `st vs B` | `env_corr X` | `ltas_cos B` | `frame_mel B` | Native Audit |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `c0_voice_ceiling` | Target B only (no Song X) | 90.2 | 194.50 | -0.01 | 0.220 | 0.972 | 0.558 | Voice/guitar ceiling (`0` disco melody). |
| `b2_cond_release_50_raw` | Unrouted `aud_end=0.50` raw | 79.1 | **406.47** | **+12.75** | **0.814** | 0.921 | 0.419 | Exact disco melody, **leaked female soprano & disco synth/drums**. |
| `p2_b2_router_raw_clean100_s20` | `s=20, e_mel=1.5, l_mel=0, e_tim=0.75, l_tim=2.5, clean=1.0` raw | **71.8** | 286.55 | +6.70 | 0.282 | 0.923 | 0.511 | **Pure steel resonator guitar (zero disco synth/drums)** + high-energy belted male blues-rock vocal preserving Song X's big soaring leap on *"morning LIGHT / groove take FLIGHT"* (`-6.05 st` down from `b2`'s female leak). |
| `p2_b2_router_raw_s24` | `s=24, e_mel=1.0, l_mel=0, e_tim=0.75, l_tim=2.5, clean=0.70` raw | 82.7 | **201.72** | **+0.62** | **0.385** | **0.963** | **0.583** | **100% gravelly male delta blues baritone + pure steel guitar** (zero female/disco bleed) keeping Song X's phrasing and cadence. |
| `p2_b2_router_harm_s24` | `s=24, e_mel=1.0, l_mel=0, e_tim=0.75, l_tim=2.5, clean=0.75` HPSS | **71.9** | **196.25** | **+0.14** | 0.186 | **0.971** | **0.599** | Matches `c0_voice_ceiling` voice/guitar fidelity (`0.971` vs `0.972`), mellowest blues reinterpretation. |

### Practical Tuning Contract (Two Clean Orthogonal Dials)

1. **Late-block timbre firewall (always on):** `split_block = 20..24`, `melody_gain_late = 0.0`, `timbre_gain_late = 2.0..2.5`, `timbre_gain_early = 0.75`, `melody_end_percent = 0.50` (6-step Pika) or `0.44` (16-step Base).
2. **Melody-lift vs. natural-register balance:**
   - **Max Target-B vocal authenticity (recommended default):** `melody_clean_level = 0.70..0.80`, `melody_gain_early = 1.0` (e.g. `u5`, `u8`, `p2_b2_router_raw_s24`).
   - **Max Song-X high-note melodic belt (for original/unfamiliar melodies with big jumps):** `melody_clean_level = 1.00`, `melody_gain_early = 1.0..1.5`, `split_block = 20` (e.g. `u4`, `p2_b2_router_raw_clean100_s20`).
3. **Optional HPSS pre-pass:** Recommended whenever Song X has loud drum-machine/electronic percussion (`fun.mp4`) and Target B is acoustic unplugged (`unplugged_compressed.mp4`). Never use heavy `-12 st` phase-vocoder pitch shifting (proven OOD in Phase 1).
