# Projects & Portfolio Dossiers (`docs/wiki/projects/`)

Router for structured multi-track dossiers in `docs/wiki/`. Format and invariants live in [`AGENTS.md`](AGENTS.md); master wiki routing lives in [`../index.md`](../index.md).

---

## 1. Dossiers

| Folder / Dossier | Scope | Human & Agent Entry | Machine Ledger | Offline Viewer |
| :--- | :--- | :--- | :--- | :--- |
| `2026-11-13_repo_audit_and_atlas/` | Full codebase, commit (`Nov 4–13`), `agy` (`16` sessions) + Claude (`11` roots, `51` subagents) transcript audit, 8-track matrix, Top 14 unfinished/forgotten ideas, code-vs-prose truth ledger, `$H3_OUTPUT_DIR` renders, & Phase-2 code reorg plan | [`2026-11-13_repo_audit_and_atlas/index.md`](2026-11-13_repo_audit_and_atlas/index.md) | [`ledger.json`](2026-11-13_repo_audit_and_atlas/ledger.json) | [`report.html`](2026-11-13_repo_audit_and_atlas/report.html) |

---

## 2. Active & Parked Track Hubs (LLM Wiki)

| Track | Status (`2026-11-13`) | Primary Wiki Hub | Key Code Entrypoints |
| :--- | :--- | :--- | :--- |
| **Control Vectors & J-Lens / Chain Lens** | Active (benchmarked blind; `exp2`–`exp5` + `§15` next) | [`../control_vectors.md`](../control_vectors.md) | [`control_vector_nodes.py`](../../../control_vector_nodes.py), [`j_lens.py`](../../../j_lens.py), [`h3_pair_compiler.py`](../../../h3_pair_compiler.py) |
| **Edit Guidance (`MiniMaxH3EditGuider`)** | Active (`e0a5bb35` packed split landed; spatial `MASK` next) | [`../edit_guidance.md`](../edit_guidance.md) | [`edit_guidance.py`](../../../edit_guidance.py), [`h3_prompt_edit.py`](../../../h3_prompt_edit.py) |
| **Timeline Sound Cues & 2D Sprite D-Pad** | Active (`0.33` release + 2 Pika workflows shipped; UI & 2-subject next) | [`../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md`](../research/control_vectors/2026-11-12_sound_cues_dpad_and_033_release_law.md) | [`sound_cues.py`](../../../sound_cues.py), [`sprite_trajectory.py`](../../../sprite_trajectory.py), [`control_vector_nodes.py`](../../../control_vector_nodes.py) |
| **Single-Frame Image (`MiniMax-H3-Image` & `P-H3-Image`)** | Shipped (`4 MP` floor, split KV-prefix refs, `heylook` expander) | [`../image/h3_image.md`](../image/h3_image.md), [`../image/p_h3_image.md`](../image/p_h3_image.md) | [`h3_image_nodes.py`](../../../h3_image_nodes.py), [`ph3_image_nodes.py`](../../../ph3_image_nodes.py) |
| **Sol Sparse Attention & Quant Precision** | Active (`SOL_DENSE_TAIL="49"` shipped; `MiniMaxH3SolChunked` retired `2026-11-13`, wheel rebuild open) | [`../attention/SOLATTN.md`](../attention/SOLATTN.md) | [`sol_attn_h3.py`](../../../sol_attn_h3.py), [`attention.py`](../../../attention.py) |
| **Core Conditioning, Frame/Clock Rules & Prompting** | Active (`39+51m` dual-clock & `AGENTS.md` updated) | [`../core/h3_frame_rules.md`](../core/h3_frame_rules.md), [`../prompting/prompting.md`](../prompting/prompting.md) | [`h3_rules.py`](../../../h3_rules.py), [`conditioning.py`](../../../conditioning.py), [`reference_conditioning.py`](../../../reference_conditioning.py) |
| **Audio Freeze & Chained Song Loops** | Stable code, unrendered queue items (`AudioRefineMask`, stem/0.25) | [`../core/h3_audio_freeze.md`](../core/h3_audio_freeze.md) | [`audio_freeze.py`](../../../audio_freeze.py), [`audio_freeze_song.py`](../../../audio_freeze_song.py) |
| **Masked V2V Pipeline & `lane_db` (`'fun'` parked)** | `'fun'` shot backlog **parked/closed**; SQLite Tier-1 `lane_db` shipped for future `'fun2'` | [`../masked_v2v.md`](../masked_v2v.md) | [`video_mask.py`](../../../video_mask.py), [`subject_track.py`](../../../subject_track.py), [`bench/lane_db/db.py`](../../../bench/lane_db/db.py) |
