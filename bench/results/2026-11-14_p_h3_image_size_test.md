# P-H3-Image: generate at 8 MP and shrink to 4, against generate at 4 (2026-11-14)

lane: image

Date: 2026-11-14

The P-H3-Image half of the generate-big, deliver-small question the MiniMax-H3-Image record answered
([`2026-11-14_h3_image_size_test.md`](2026-11-14_h3_image_size_test.md)). It settled
`workflows/h3_config.py::PH3_IMAGE_DELIVER_MEGAPIXELS` (owner decision, 2026-11-14).

## 1. What was run

- **Cases.** Three edits of the owner's pictures (`e_plaques`: new text on four plates; `e_figure`: a cable-knit
  sweater; `e_dinner`: three candles in brass holders) and two text-to-image captions from the 2026-11-14 latent
  bank (`text_menu`, `face_portrait`). Prompts, seeds and sources: `data/p_h3_size_test_2026-11-14/queued.json`.
- **Arms.** Built from the shipped `workflows/image/h3_p_h3_image_{t2i,edit}_api.json` at repo `7b7e185a`
  (`data/p_h3_size_test_2026-11-14/queue_p_h3_size_test.py`), `canvas="explicit"` at the case's shape
  (`h3_config.h3_image_size`): generated at 8 MP and shrunk by `MiniMaxH3ImageDownscale` (Lanczos) to 4 MP, against
  generated at 4 MP. Both arms of a case have the same delivered dimensions.
- **Seeds.** One seed per case (10 renders); then three more seeds on each edit (18 renders,
  `data/p_h3_size_test_2026-11-14/queue_seeds.py`), asked for by Gemini before the decision.
- **Stack.** `p-h3-image_int8_convrot`, the bf16 32B text encoder (`qwen3vl_32b_minimax_h3_bf16_pruned`),
  `p-h3-image_vae_int8_convrot` decoded in fp16, comfy-kitchen attention, the 8-step grid; RTX 4090.
- **Outputs.** `$H3_OUTPUT_DIR/Image/p_h3_size_test_2026-11-14/<case>/gen<G>_4mp*.png`; phone-width A/B files with
  shuffled order in `phone/` (first seed, `key.txt`) and `phone/seeds/` (extra seeds, `key.txt`); `README.txt`
  for a reader judging blind.

Text-to-image arms differ in composition, and an edit's new content (the knit pattern, the candle holders)
differs between arms too, because the noise is drawn at the latent's shape (`comfy/sample.py::prepare_noise_inner`).
The edits are the fair comparison of how well things are made.

## 2. Time

Median wall time per render, first seed, from ComfyUI's history:

| | generated at 4 MP | generated at 8 MP |
|---|---|---|
| edit (n=3 each) | 29.2 s | 36.0 s |
| t2i (n=2 each) | 22.5 s | 32.5 s |

## 3. Content checked at full size, four seeds per edit

| edit | generated at 8, shrunk | generated at 4 |
|---|---|---|
| `e_plaques`: "PHASE" misspelled "PHAGE", of 16 plates | 6 | 11 |
| `e_dinner`: candle count not three | 1 seed (four) | 0 |
| `e_dinner`: the glass pitcher hidden behind a candelabra | 1 seed | 3 seeds |

"POSABLE DRAFT" and the `ITALIA` label were right in every render. The first plaque count, made from a
phone-width sheet, undercounted; the table is from full-size crops.

`t2i_text_menu`, first seed: generated at 8 MP misspelled "Flat Whrite" and a plant covers the "T" of "Toasted";
generated at 4 MP is clean. A text-to-image layout changes with the canvas, so this mixes size with composition.

## 4. The owner's eye

- `e_figure`, first seed, on the phone: the owner preferred the picture generated at 8 MP and shrunk.
- Decision (owner, 2026-11-14, with Gemini's agreement forwarded): P-H3-Image generates at 8 MP and delivers 4 MP.

## 5. What this cannot see

- Four seeds on three edits, one on each text-to-image case.
- The extra seeds' phone files were sent to the owner; no judgement on them is recorded here.
- Content was counted by this session; Gemini's reading of the first seed agreed on the plaques, the pitcher and the
  menu.
