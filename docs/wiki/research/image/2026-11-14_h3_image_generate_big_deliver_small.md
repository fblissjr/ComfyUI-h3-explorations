# MiniMax-H3-Image: generate big, deliver small (2026-11-14)

- **Date:** 2026-11-14
- **Question (the owner's):** 16 MP files are very large. Is there a cost to delivering something smaller, and is
  generating at 16 MP and shrinking better than generating at the size wanted?
- **Answer:** generate at 16 MP and shrink. On a phone the two cannot be told apart; on a desktop the owner found the
  16 MP-and-shrunk pictures clearly better, and at full size the edits generated at 16 MP made fewer content slips
  (a misspelled plate, an extra candle). Generating at 16 MP takes about twice as long as at 4 MP, which is what a
  16 MP render already took; delivering 4 MP cuts the PNG to about a quarter of a 16 MP one.
- **Decision (owner, 2026-11-14):** the MiniMax-H3-Image graphs generate at 16 MP and deliver 4 MP.
  `MiniMaxH3ImageSize` gained `generate_megapixels` (default `H3_IMAGE_MEGAPIXELS`); `megapixels` is the delivered
  size (default `H3_IMAGE_DELIVER_MEGAPIXELS`). Either can be raised per render.
- **Numbers:** [`bench/results/2026-11-14_h3_image_size_test.md`](../../../../bench/results/2026-11-14_h3_image_size_test.md).
  This page carries directions and pointers only.

**Where the images are:** `$H3_OUTPUT_DIR/Image/size_test_2026-11-14/` (one folder per case with every arm at full
resolution; `phone/` the 4 MP A/B files and `key.txt`; `phone/other_sizes/` the 2 and 8 MP pairs).

---

## 1. Why big helps, as far as the code says

The VAE always decodes 16 px cells and the DiT always sees 32 px tokens, while the DiT's position grid is normalized
by the picture's area (`coderef/diffusers/.../minimax_h3_image/before_denoise.py::_spatial_position_grid`): a 16 MP
canvas spans the same coordinate range as a 4 MP one with four times the tokens. So at 16 MP a letter or an eye covers
more cells and more tokens, and the model card says the model is best near 16 MP. Shrinking afterwards with Lanczos
averages several decoded pixels into one, which also hides decoder speckle
([`2026-11-14_h3_image_vae_inspection.md`](2026-11-14_h3_image_vae_inspection.md)). This is the reading; the test
below is the evidence.

## 2. The test

Fourteen cases, six text-to-image and eight edits of the owner's pictures (two of them from resized copies, so the
input size varied with the picture fixed), each delivered at 2, 4 and 8 MP two ways: generated at 16 MP and shrunk,
or generated at the delivered size (at 4 MP and shrunk for 2 MP, as the servers do). Edits keep the picture's layout
and are the fair comparison; text-to-image layouts change with the canvas size. Record sections 1 and 2.

What it showed (record sections 3 and 4):

- **Phone:** the owner and another viewer saw no difference on any of the 14, blind.
- **Desktop:** the owner found the 16 MP-generated pictures clearly better.
- **Content:** in the plates edit, generated at 4 and 8 MP misspelled two plates, at 16 MP one; in the dinner edit,
  4 MP added a fourth candle where three were asked for. The headline, the menu and the bottle label came out right
  at every size.
- **Time:** about twice as long at 16 MP as at 4 MP, for edits and text-to-image alike.
- **File size:** a 4 MP PNG is the same size whichever way it was made, about a quarter of a 16 MP one.

Gemini audited the same files the same day; its findings are in record section 5, with which held when checked.

## 3. What changed

- `workflows/h3_config.py::H3_IMAGE_DELIVER_MEGAPIXELS` (new, the owner's decision) beside
  `H3_IMAGE_MEGAPIXELS` (generation, inherited from the model card).
- `h3_image_nodes.py::MiniMaxH3ImageSize`: `generate_megapixels` appended as the last input; the canvas is the
  largest of `megapixels`, `generate_megapixels` and the 4 MP floor, at the delivered picture's shape, so the shrink
  never stretches or enlarges (`resolve_canvas_and_target_size`). `megapixels` now defaults to 4.
- The four MiniMax-H3-Image graphs carry both values (`workflows/build_workflows.py::build_h3_image_api`);
  `bench/check_h3_image.py` holds the size rule and the graphs; `bench/node_id_manifest.json` records the input.
- The studio app takes its size controls from the graphs; its own session was told of the new input.

## 4. Server defaults, corrected

An earlier answer the same day said sglang and vllm-omni both default to 1024x1024. sglang does so only for text to
image and refuses an edit without a size; vllm-omni defaults to 1024 and an edit's "auto" size falls back to it
despite reading the picture's size; diffusers defaults to 2048x2048. None is a quality choice. Record section 6.

## 5. Next steps

- Delivered size per use: 4 MP is the default; raise `megapixels` (up to `generate_megapixels`) when a large file is
  wanted. JPEG or WebP delivery would cut a 4 MP file several times further and needs a different save node; not done.
- P-H3-Image took the same split the same day: generated at 8 MP, delivered at 4 (owner decision,
  [`bench/results/2026-11-14_p_h3_image_size_test.md`](../../../../bench/results/2026-11-14_p_h3_image_size_test.md)).
- The edit lead in [`../../image/h3_image.md`](../../image/h3_image.md) (references sized to the canvas) is unchanged by
  this and still untested.

## Pointers

| what | where |
|---|---|
| numbers, conditions, checked readings | [`bench/results/2026-11-14_h3_image_size_test.md`](../../../../bench/results/2026-11-14_h3_image_size_test.md) |
| prompts, seeds, the queue and phone-sheet scripts | `data/size_test_2026-11-14/` (untracked) |
| renders, phone A/B files, key | `$H3_OUTPUT_DIR/Image/size_test_2026-11-14/` |
| the decision's constant and the node | `workflows/h3_config.py::H3_IMAGE_DELIVER_MEGAPIXELS`, [`h3_image_nodes.py`](../../../../h3_image_nodes.py) `MiniMaxH3ImageSize` |
