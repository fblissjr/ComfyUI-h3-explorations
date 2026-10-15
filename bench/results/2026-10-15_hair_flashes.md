# Her hair flashing at step edges: three region rules, measured before the build (2026-10-15)

The second subject's hair came and went in four-frame steps on `second_hfp_0014d` (57-60, 159-166) and `second_hfp_0604d`
(604-652) (the handover's "What not to assume"; the outside reviewer's clips review). The tooling brief of the day asked
for three proposals for `video_mask.py`. Each was measured offline on the two saved windows before anything was built.

The outside reviewer then watched every frame of those spans and placed each flash in a pixel box and in grid cells (G2,
`internal/gemini/2026-10-15_watch_hair_item4.md`). The lead ruled on G2, and this record is what was built and what each
rule does on the two windows.

How it was measured:

- The region is rebuilt from each window's saved region file with the node's own functions. It matches the saved
  region cell for cell on both windows (0 cells differ).
- The rules are taken verbatim from `video_mask.py` and run in the node module's namespace, so what is measured is what
  is built.
- The scripts and their full output are in `internal/claude/2026-10-15_mrlol_scripts/`: `hair_ab_measure.py`,
  `hair_c_measure.py` and `hair_rules_measure.py`. That folder is not shipped.

## Why a speck holes four frames

A keep-out takes a cell out of the region for the whole of its latent step, because `token_mask` pools a step's frames.
So a speck on one frame holes four. A cell given back to the source shows the original's hair there, and the subject's
new hair is drawn on some steps and not on the next.

## What was built (`video_mask.window`, in this order)

1. **`drop_specks`.** A 4-connected piece of the others' mask on one frame, under one cell's area (`SPECK_CELLS` 1.0,
   which is 256 px at a 16 px cell), and present at its place on at most half of its step's frames
   (`SPECK_STEP_SHARE` 0.5), is dropped before the region is cut. The half is the reviewer's guard of 2026-10-14: a real
   face can be one cell on a small shot, and a face stays on most frames of its step.
2. **`close_lone_holes`.** A cell is given back to the region when all of these hold:
   - the others took it out of the grown region;
   - it is not the subject's own;
   - all four of its neighbours are in the region;
   - the others' px in it are under the same area on every frame of the step;
   - the others are on it on at most half of those frames.

   In whole tokens the cell is a token.
3. **`keep`** still wins over everything it touches.
4. **`drop_islands`.** Where a keep-out or `keep` is wired, a 4-connected piece of the region on a step that shares no
   cell with the subject's own mask is given back to the source.

## The flashes, cell by cell (0014d)

| flash (G2) | step, cells | cause | before | after |
|---|---|---|---|---|
| 1B, 57-60 | 13, (22..24, 39) | a strangers speck, 89 px on 57 and 22 on 58 | out | in, by the specks rule |
| 2A, 159-162 | 43, (31,24) (32,23) (32,24) | held specks, 183 px on 161 | out | in, by the specks rule |
| 2B, 159-162 | 43, (31,36) (32,35) (32,36) | held specks, 9-125 px on 161-162 | out | in, by the specks rule |
| 2C, 163-166 | 44, (26,39) | a false strangers piece of 286-418 px, 40 and 105 px in the cell on 164-165 | out | in, by the close |

The 159-162 flash needs one cell's area: its piece on 161 is 183 px. The specks rule alone, at any floor up to 256 px,
mends only 159-162 of 159-166. The close mends 163-166.

## What the rules change on the two windows

| | 0604d | 0014d |
|---|---|---|
| specks: cell-steps given back by the area alone | 253 | 73 |
| specks: cell-steps given back with the guard | 169 | 54 |
| the close: cell-steps closed | 3 | 5 |
| islands: pieces dropped (cell-steps) | 36 (154) | 4 (14) |
| in all: cell-steps added / taken | 153 / 135 | 59 / 14 |

The guard keeps out the pieces that stay on three or four frames of their step, for example 0604d's 821-842 beside her
hair.

Of the cells closed, one holds another subject's px: 0604d step 47, cell (16,33), with 7 px of the lead on 762 alone.
Assembly gives the lead's pixels back on the second subject's rows (`restore`). The node gets one combined keep-out mask,
so no rule there can tell another subject from a held thing; a cell a keep-out holds on three or four frames stays open,
whoever it is.

The islands dropped hold none of her hair or face in her class map, with one exception. On 0604d step 57 (796-799), one
cell holds 237 px of her hair on 799. The specks rule gave that cell back, but it touches none of her mask, so the
islands rule drops it again, and it is outside the region exactly as before. The reviewer saw no rendered hair in the 43
islands of the first measure, which are apron, chest or bowl below her arms.

## What was not built

- **Filling the composite's small pockets** (proposal (c)). Its first half, the weight held over each step, is already the
  node's (`changed_alpha`, since fc2628f9). Small enclosed pockets lie on 33 of 0604d's 345 frames and 10 of 0014d's 154.
  None is on her hair or face, and none is on a flash.
  - 1C's tip shimmer (0014d, 50-65) is the outer edge of the blend, not a pocket. It is a row in `docs/wiki/next_steps.md`.
- **0604d's lower-left hair, stopped at her blouse and apron strap** (608-612, 621-652). This is the large clothes and
  held keep-outs, and it is the owner's ruled form: Open List row 10 of 2026-10-12, "clothes left alone, hair to the
  shoulder line". It is not a fault.

## What this cannot see

- The picture: whether a cell given back draws her hair or a seam. The first readers after the build look at the flashes
  above and at hair tips near hands.
- What the islands rule drops where it matters. The others never take a cell of her own mask, so her own hair is never an
  island. What it can drop is margin room where new hair would fall outside her carried mask (0604d's 799 above), and a
  piece that `keep` cuts off. Neither window holds such a case with hair in it.
- A small face that stays on fewer than half of its step's frames. The specks rule would drop it.
