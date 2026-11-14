# MiniMax-H3-Image: generate at 16 MP and shrink, against generate at the delivered size (2026-11-14)

lane: image

Date: 2026-11-14

The measured half of [`docs/wiki/research/image/2026-11-14_h3_image_generate_big_deliver_small.md`](../../docs/wiki/research/image/2026-11-14_h3_image_generate_big_deliver_small.md).
The owner asked whether delivering smaller files costs anything, and whether generating big and shrinking is better
than generating at the size wanted. It settled `workflows/h3_config.py::H3_IMAGE_DELIVER_MEGAPIXELS`.

## 1. What was run

- **Cases.** 14: six t2i captions from the 2026-11-14 latent bank (`shadow_study`, `text_menu`, `face_portrait`,
  `texture_knit`, `gradient_dusk`, `edges_tiles`) and eight edits of the owner's pictures, two of them the same edit
  from a resized copy of the picture (`e_figure_1mp`, a 4 MP picture shrunk to 1 MP; `e_workshop_4mp`, a 1 MP
  picture enlarged to 4 MP). Prompts, seeds and sources: `data/size_test_2026-11-14/prompts.json`. The dinner edit's
  first source was removed from the input folder before queuing; a near-identical picture without its caption
  replaced it (`e_dinner`'s `note`).
- **Arms, one seed per case.** Built from the shipped `workflows/image/h3_image_{t2i,edit}_api.json`
  (`data/size_test_2026-11-14/queue_size_test.py`): generated at 16 MP and shrunk by `MiniMaxH3ImageDownscale`
  (Lanczos) to 2, 4 and 8 MP; generated at 4 MP, saved at 4 and shrunk to 2; generated at 8 MP. Delivered sizes from
  `h3_config.h3_image_size`, so both arms of a comparison have the same dimensions. 42 renders, all completed.
- **Stack.** The shipped graphs at repo `c617993e`: `minimax_h3_image_pruned_int8_convrot`, the bf16 32B encoder,
  `p-h3-image_vae_bf16`, comfy-kitchen attention, the six-step grid; RTX 4090.
- **Outputs.** `$H3_OUTPUT_DIR/Image/size_test_2026-11-14/<case>/gen<G>_<D>mp_*.png`; phone-width A/B files
  (shuffled order, key in `phone/key.txt`) in `phone/` (4 MP) and `phone/other_sizes/` (2 and 8 MP);
  `measures.json`.

Text-to-image arms differ in composition, because the latent's spatial size changes the noise layout; the edit arms
are anchored by the picture and are the fair comparison.

## 2. Time and file size

Median wall time per render, from ComfyUI's history (each render samples once; the 16 MP arm also shrinks and saves
three times):

| | generated at 4 MP | 8 MP | 16 MP |
|---|---|---|---|
| edit (n=8 each) | 23.1 s | 29.3 s | 45.3 s |
| t2i (n=6 each) | 13.5 s | 19.0 s | 33.1 s |

Median PNG size (MB):

| | delivered 2 MP | 4 MP | 8 MP |
|---|---|---|---|
| edit, generated at the delivered size (2 MP: generated at 4) | 3.2 | 6.2 | 11.1 |
| edit, generated at 16 and shrunk | 2.8 | 5.3 | 10.4 |
| t2i, generated at the delivered size (2 MP: generated at 4) | 3.3 | 6.4 | 11.7 |
| t2i, generated at 16 and shrunk | 3.2 | 6.4 | 12.6 |

A 16 MP delivery is about 20 to 30 MB (the 2026-11-14 VAE record's renders).

## 3. The owner's eye

- **Phone, 4 MP set, blind** (the owner and another viewer): no difference on any of the 14.
- **Desktop:** the owner found the pictures generated at 16 MP and shrunk clearly better.

## 4. Content checked at full size

- **`e_plaques` (new text on four plates).** Misspellings of "PHASE" as "PHAGE": generated at 4 MP, two plates
  (1 and 4); at 8 MP, two (1 and 4); at 16 MP, one (plate 1, likely pulled toward the source's "STAGE"). "POSABLE
  DRAFT" is spelled right in every arm. Crops: the session's `plaques_strip.png` (not kept; rerun from the
  `gen*_*mp_*.png` files).
- **`e_dinner` (add three candles).** Generated at 4 MP: four lit candles; at 8 and 16 MP: three.
- **`e_poster` headline, `t2i_text_menu` lettering and prices, the `ITALIA` label:** right at every size.

## 5. Others' readings, checked here

Gemini (2026-11-14) audited the 14 phone files and full-size crops. Checked here and holding: the `e_dinner` extra
candle at 4 MP; that 16 MP still misspells plate 1 (this session had first read the phone sheet as all four right,
corrected above). Checked here and not holding: a "POSASLE DRAFT" misspelling at 4 MP; the plate reads "POSABLE"
in every arm. Reported by Gemini and not re-checked here: at full size, 16 MP keeps skin pores and eye colour on
faces where 4 MP crushes eye shadows (`e_workshop`, `e_figure`, `e_ship`), and a desk-edge notch and a clipped
chair in two t2i layouts at 4 MP.

## 6. Server defaults, corrected

An earlier answer the same day said both servers default to 1024x1024. Read again (`coderef/sglang`,
`coderef/vllm-omni`, `coderef/diffusers` at the tips noted in the VAE record):

- sglang's t2i request defaults `size` to `1024x1024` (`runtime/entrypoints/openai/protocol.py`), generated on its
  4 MP canvas floor and shrunk; its edit request has no default size (`image_api.py`) and its request check then
  requires one (`minimax_h3/request_validation.py`).
- vllm-omni defaults to 1024 (`coderef/vllm-omni/vllm_omni/diffusion/models/minimax_h3/pipeline_minimax_h3_image.py`,
  `MINIMAX_H3_IMAGE_DEFAULT_SIZE`); an edit with
  `size=auto` reads the picture's size and then marks the size not provided (`entrypoints/openai/api_server.py`),
  so it falls back to 1024x1024 anyway.
- diffusers defaults to 2048x2048 (`modular_pipelines/minimax_h3_image/before_denoise.py`).

None is a quality choice; MiniMax's card, examples and prompt expander all aim at about 16 MP.

## 7. What this cannot see

- One seed per case; t2i layouts differ between arms, so a t2i preference mixes size with composition.
- Content errors were counted on two edits; the direction (fewer slips when generated big) rests on those two.
- Phone and desktop judgements are the owner's; the desktop one is unblinded.
