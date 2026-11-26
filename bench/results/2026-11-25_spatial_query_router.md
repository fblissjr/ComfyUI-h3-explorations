# 1× Spatial Visual Query Routing Gate (`2026-11-25`)

Standalone `4`-clip matched-seed GPU falsification (`90f` / `3.75 s`, `1344×768`, 6-step Pika H3 Flex `MiniMaxH3Sol`, `seed = 730451892`, **100% stock 3D RoPE** + `comfy_kitchen.int8_attention` for large spatial video passes) verifying single-pass `1×` spatial split-screen and free-form portal-box visual routing (`steering/spatial_query_router.py::MiniMaxH3SpatialQueryRouter`).

- **Prompts:** Matching locked-off 35mm wide street framing — **Zone A** (`prompt_bank/t2va_spatial_summer_street.txt`, bright sunlit European cobblestone street under clear blue sky) vs **Zone B** (midnight cyberpunk neon rain street with glowing magenta/cyan holograms, steaming street vents, wet neon-reflecting asphalt, and translucent umbrellas).
- **Rendered clips:** `$H3_OUTPUT_DIR/Video/spatial_qr_lab_2026-11-25/`.
- **Decision:** **Promoted immediately** into `MiniMaxH3SpatialQueryRouter` (`steering/spatial_query_router.py`) + standalone workflow `workflows/h3_text_to_video_pika_spatial_query_router_api.json` (default `region = "right_half"`, `isolate_blocks = "0-24"`, with portal box `0.28, 0.18, 0.72, 0.85` pre-populated for 1-dropdown switch to `"box"`).

---

## 1. Results (`seed = 730451892`, `90f`, `1344×768`, 6-Step Pika Kitchen+Sol)

| Arm | Zone / Isolation | Elapsed (s) | `L / R` Luma (`L - R`) | `Box / Out` Luma | `L / R` Sharpness | `f16→17` Pop (`L / R`) | Honest Verdict |
| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :--- |
| `spatial_base_A` | stock (`none`) | `37.9` | `81.9 / 112.9` (`-31.0`) | `99.2 / 96.7` | `635.1 / 2188.5` | `1.71 / 2.69` | Clean golden sunlit European cobblestone street baseline. |
| `spatial_split_seamless_0_49` | `right_half`, `none` | `45.8` | `92.5 / 55.1` (`+37.5`) | `76.1 / 72.8` | `1617.4 / 645.0` | `2.78 / 2.10` | **Instant home run:** Left half stays bright sunlit limestone + olive trees + sunny sidewalk diners while right half transforms into pouring midnight cyberpunk neon rain + steaming vents + umbrellas, with dry cobblestones morphing seamlessly into wet neon-reflecting asphalt down the center (zero hard line, zero VAE shimmer). |
| `spatial_split_iso_0_24` | `right_half`, `0-24` | `43.1` | `96.0 / 57.8` (`+38.3`) | `72.6 / 78.7` | `1836.2 / 562.4` | `3.33 / 1.94` | **Faster (`-2.7 s` vs seamless) & stronger contrast:** Symmetric half-length K/V isolation on `0..24` (`comfy_kitchen.int8_attention`, `attn_mask = None`) blocks 1-hop cross-world text bleed and then blends lighting seamlessly on `25..49`. |
| `spatial_box_portal_iso_0_24` | `center_box`, `0-24` | `42.1` | `73.8 / 96.7` (`-22.9`) | `69.1 / 92.0` | `539.2 / 1746.5` | `2.43 / 2.25` | **Only `+4.2 s` over stock (`1.11×` wall time):** Both outer sides remain bright golden summer cobblestone street with sidewalk tables while the center box (`x∈[0.28,0.72], y∈[0.18,0.85]`) opens cleanly into a towering midnight cyberpunk neon portal with umbrella-carrying pedestrians emerging into daylight — zero rectangular paste-in seam and zero `17f` pop. |

---

## 2. Why This Worked Cleanly on First Try

1. **100% Stock 3D RoPE (`0` Causal-VAE Pop):** Unlike temporal RoPE warps (`2026-11-25_rope_warp_falsification.md`), routing spatial video queries by text context leaves every `(t, h, w)` RoPE coordinate intact, preserving H3's `[1, 4, 4, 4, 4]` causal Video VAE format completely (`f16→17` pop stays `1.94–3.33`, matching baseline `1.71–2.69`).
2. **Zero Dense `attn_mask` + `comfy_kitchen.int8_attention`:** Precomputing 1D `torch.long` index partitions (`idx_a`, `idx_b`) once per step and slicing K/V cleanly during `blocks 0..24` keeps every large spatial video pass on `comfy_kitchen.int8_attention(q, k, v)` (`attn_mask = None`), delivering `+4.2 s` wall-time overhead (`42.1 s` vs `37.9 s` baseline) with zero OOM risk.
3. **Two-Phase Architecture (`0..24` Isolated -> `25..49` Blended):** Isolating both zones (and their respective `txt_A` / `txt_B` tokens) during early structure blocks `0..24` locks each world's palette, weather, and architecture without 1-hop bleed; reopening shared `K_vid_all` on `25..49` lets H3's own attention naturally unify shadows, reflections, and perspective across any boundary or free-form box.
