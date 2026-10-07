# Instances per pair and how long an object is gone, for the lane's own targets, from Meta's benchmark annotations (2026-10-07)

lane: masked
verdict: in the video benchmark a small object (annotated box under a thousandth of the frame) is typically gone only briefly when it is gone: the median gap is 0.12 to 0.17 s in SA-V (24 frames a second), 0.5 s in YT-1B and about 1 to 1.2 s in the egocentric SmartGlasses, and 91% to 93% of SA-V's gaps, 75% to 79% of YT-1B's and about half of SmartGlasses' last one second or less (98%, 87% to 88% and 66% to 72% last two seconds or less); but half or more of the small masklets leave and return at least once (50% to 71%), so a regain has to expect it, and the egocentric footage has gaps of ten seconds and more; the bare words the lane asks for are asked with a median of one to a few instances per pair, more in crowded scenes and for `person` in YT-1B video

An addendum to [`2026-10-07_sam3_benchmark_phrases.md`](2026-10-07_sam3_benchmark_phrases.md): same tool (`bench/sam3_dataset_phrases.py`, command `targets`), same datasets, same licence handling (aggregates only, no phrase listed; the datasets are gated and not in the repository; SAM License, with CC-BY-NC 4.0 for SA-V and YT-Temporal-1B in VEval).

**What was asked.** The lead, for the masked lane's held-object work (the owner's next build keeps a held object by where it is on the subject, not by tracking it, and mregg is measuring regain): for the things the lane asks SAM for (the bare words `person`, `head`, `hand`, `hair`, and held things as a class: any asked noun whose annotations sit under a thousandth of the image), the per-pair count of instances, and in VEval how often an object leaves and returns and for how long.

**Four lines for the lane.** Each says how it is known.

1. **A normal gap for a small object is short, and a quarter of SA-V's are one frame** (counted, VEval "small thing" rows below). Median gap: 0.12 s and 0.17 s in SA-V's two splits, 0.50 s in both YT-1B splits, 1.17 s and 1.00 s in SmartGlasses. A gap of up to one second covers 91% to 93% of SA-V's small-object gaps, 75% to 79% of YT-1B's and 50% to 51% of SmartGlasses'; two seconds covers 98%, 87% to 88% and 66% to 72%; five seconds covers 100%, 94% to 98% and 85% to 92%. The longest are 4 s (SA-V), 15 s (YT-1B) and 18 s (SmartGlasses). For all masklets in SA-V, a quarter of the gaps are a single frame (checked on the test split; masks exist on every frame index, so this is flicker, not a coarse annotation step).
2. **Small objects leave and return often** (counted): 63% and 50% of SA-V's small masklets, 51% and 50% of YT-1B's, 71% and 69% of SmartGlasses'; against 48% / 47%, 34% / 37% and 71% / 71% for all masklets. Many are also gone at the last frame, 47% to 77%.
3. **Instances per pair for the bare words** (counted, tables below): `person` has a median of 4 in Gold Metaclip, 11 in SA-1B and 19 in Crowded, 1 (Ego4D), 3 (SA-V), 4 (YT-1B) and 5.5 (NGA) in Silver, and in VEval 1 to 3 (SA-V) and 12.5 to 15 (YT-1B); `head` 1 in Gold Metaclip and SA-1B, 1 to 2 in Silver, 2 to 4 in SA-V video, and 22 in Gold Crowded (6 pairs); `hand` 2 in Gold Metaclip, 4 in SA-1B, 14 in Crowded, 1 to 2 in Silver (Ego4D 2, a pair of hands) and 2 to 2.5 in SA-V video; `hair` 1 to 2.5 where it has more than a few pairs. Small things as a class: a median of 5 to 6 instances per positive pair in Gold Metaclip and SA-1B and 14 in Crowded, but 1 in SA-V video, 2 in SmartGlasses and 2 to 3 in YT-1B.
4. **What these figures do not say**: "small thing" is a size class, not "held"; an absent frame means the annotators marked no object in it (it left, was hidden or was too small to mark; the files do not say which); the benchmark's clips are 4 to 7 s (SA-V, YT-1B median) and 20 s (SmartGlasses), none are ours; and nothing here measures a model.

**How.** The `targets` command of `bench/sam3_dataset_phrases.py`. A target is a pair whose phrase, after dropping a leading article, is exactly `person`/`people`, `head`/`heads`, `hand`/`hands` or `hair`. A *small thing* is a positive pair whose annotations have a median box share of at most 0.001 of the image (Gold, Silver), or in VEval a masklet whose median box share over the frames it is present in is at most 0.001; its instances per pair are the small masklets of a video and phrase. A masklet is *present* in a frame when it has a box, *gone* when it has none; a *gap* is a run of gone frames between two present frames; frames before the first and after the last appearance are not gaps. Frame rates are the README's folder names (SA-V 24, the others 6), not checked against the media. Silver's two species-name sources are left out (no target is in them).

**Controls.** Every masklet's present, leading, trailing and gap frames add up to its video length (asserted in all six VEval files); the masklets that leave and return, counted by the gap code, equal the number the committed record's run counter gives, in all six files; the same class "all masklets" reproduces the committed 47.7% / 47.3% / 71.4% / 70.6% / 33.8% / 36.5%.

## The tables

Every table is printed from `2026-10-07_sam3_benchmark_targets.json` by `render-targets`.

### Gold: instances per positive pair for the lane's targets (annotator a)

| set | target | pairs asked | negative | positive pairs | instances median | p90 | max | >= 2 | >= 5 | >= 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| metaclip | person | 70 | 4.3% (3/70) | 67 | 4.0 | 14.0 | 38 | 76.1% (51/67) | 44.8% (30/67) | 9.0% (6/67) |
| metaclip | head | 11 | 9.1% (1/11) | 10 | 1.0 | 4.3 | 7 | 40.0% (4/10) | 10.0% (1/10) | 0.0% (0/10) |
| metaclip | hand | 26 | 3.9% (1/26) | 25 | 2.0 | 4.0 | 10 | 80.0% (20/25) | 8.0% (2/25) | 0.0% (0/25) |
| metaclip | small thing |  |  | 276 | 5.0 | 32.0 | 80 | 78.6% (217/276) | 53.3% (147/276) | 23.5% (65/276) |
| sa1b | person | 154 | 1.3% (2/154) | 152 | 11.0 | 38.0 | 157 | 94.7% (144/152) | 69.7% (106/152) | 43.4% (66/152) |
| sa1b | head | 25 | 0.0% (0/25) | 25 | 1.0 | 10.2 | 59 | 48.0% (12/25) | 24.0% (6/25) | 8.0% (2/25) |
| sa1b | hand | 58 | 0.0% (0/58) | 58 | 4.0 | 15.6 | 70 | 94.8% (55/58) | 41.4% (24/58) | 10.3% (6/58) |
| sa1b | hair | 50 | 0.0% (0/50) | 50 | 2.0 | 6.0 | 46 | 60.0% (30/50) | 16.0% (8/50) | 4.0% (2/50) |
| sa1b | small thing |  |  | 493 | 6.0 | 30.0 | 252 | 81.5% (402/493) | 58.8% (290/493) | 23.9% (118/493) |
| attributes | small thing |  |  | 15 | 6.0 | 26.6 | 168 | 80.0% (12/15) | 60.0% (9/15) | 26.7% (4/15) |
| crowded | person | 45 | 2.2% (1/45) | 44 | 19.0 | 31.7 | 88 | 100.0% (44/44) | 97.7% (43/44) | 65.9% (29/44) |
| crowded | head | 7 | 14.3% (1/7) | 6 | 22.0 | 55.5 | 80 | 100.0% (6/6) | 100.0% (6/6) | 66.7% (4/6) |
| crowded | hand | 31 | 0.0% (0/31) | 31 | 14.0 | 22.0 | 60 | 100.0% (31/31) | 90.3% (28/31) | 48.4% (15/31) |
| crowded | hair | 4 | 0.0% (0/4) | 4 | 8.0 | 12.4 | 13 | 100.0% (4/4) | 75.0% (3/4) | 0.0% (0/4) |
| crowded | small thing |  |  | 343 | 14.0 | 45.0 | 113 | 93.0% (319/343) | 81.3% (279/343) | 46.7% (160/343) |
| wiki_common | small thing |  |  | 16 | 3.5 | 22.0 | 81 | 68.8% (11/16) | 37.5% (6/16) | 31.2% (5/16) |
| fg_food | small thing |  |  | 14 | 16.0 | 50.7 | 96 | 100.0% (14/14) | 85.7% (12/14) | 57.1% (8/14) |
| fg_sports_equipment | small thing |  |  | 30 | 4.0 | 16.1 | 27 | 83.3% (25/30) | 46.7% (14/30) | 16.7% (5/30) |


### Silver: instances per positive pair for the lane's targets (annotator a)

| set | target | pairs asked | negative | positive pairs | instances median | p90 | max | >= 2 | >= 5 | >= 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| bdd100k | person | 11 | 0.0% (0/11) | 11 | 2.0 | 15.0 | 16 | 72.7% (8/11) | 36.4% (4/11) | 9.1% (1/11) |
| bdd100k | hair | 1 | 0.0% (0/1) | 1 | 6.0 | 6.0 | 6 | 100.0% (1/1) | 100.0% (1/1) | 0.0% (0/1) |
| bdd100k | small thing |  |  | 549 | 6.0 | 19.0 | 122 | 88.0% (483/549) | 61.9% (340/549) | 15.1% (83/549) |
| droid | person | 32 | 28.1% (9/32) | 23 | 1.0 | 2.8 | 6 | 21.7% (5/23) | 8.7% (2/23) | 0.0% (0/23) |
| droid | head | 4 | 75.0% (3/4) | 1 | 1.0 | 1.0 | 1 | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) |
| droid | hand | 13 | 92.3% (12/13) | 1 | 2.0 | 2.0 | 2 | 100.0% (1/1) | 0.0% (0/1) | 0.0% (0/1) |
| droid | small thing |  |  | 175 | 4.0 | 15.0 | 114 | 76.0% (133/175) | 49.1% (86/175) | 9.7% (17/175) |
| ego4d | person | 508 | 4.9% (25/508) | 483 | 1.0 | 1.0 | 7 | 8.9% (43/483) | 1.5% (7/483) | 0.0% (0/483) |
| ego4d | head | 12 | 41.7% (5/12) | 7 | 1.0 | 3.8 | 8 | 14.3% (1/7) | 14.3% (1/7) | 0.0% (0/7) |
| ego4d | hand | 203 | 2.0% (4/203) | 199 | 2.0 | 2.0 | 6 | 76.4% (152/199) | 1.5% (3/199) | 0.0% (0/199) |
| ego4d | hair | 1 | 0.0% (0/1) | 1 | 1.0 | 1.0 | 1 | 0.0% (0/1) | 0.0% (0/1) | 0.0% (0/1) |
| ego4d | small thing |  |  | 156 | 5.0 | 33.0 | 384 | 75.0% (117/156) | 51.3% (80/156) | 25.0% (39/156) |
| food_rec | person | 20 | 35.0% (7/20) | 13 | 1.0 | 2.0 | 3 | 23.1% (3/13) | 0.0% (0/13) | 0.0% (0/13) |
| food_rec | hand | 65 | 4.6% (3/65) | 62 | 1.0 | 1.0 | 2 | 4.8% (3/62) | 0.0% (0/62) | 0.0% (0/62) |
| food_rec | small thing |  |  | 92 | 12.0 | 77.0 | 312 | 88.0% (81/92) | 75.0% (69/92) | 40.2% (37/92) |
| geode | small thing |  |  | 20 | 6.5 | 17.0 | 38 | 100.0% (20/20) | 70.0% (14/20) | 15.0% (3/20) |
| nga_art | person | 60 | 16.7% (10/60) | 50 | 5.5 | 28.1 | 58 | 80.0% (40/50) | 54.0% (27/50) | 36.0% (18/50) |
| nga_art | head | 62 | 9.7% (6/62) | 56 | 1.0 | 5.0 | 41 | 39.3% (22/56) | 12.5% (7/56) | 5.4% (3/56) |
| nga_art | hand | 59 | 10.2% (6/59) | 53 | 2.0 | 7.6 | 21 | 81.1% (43/53) | 22.6% (12/53) | 1.9% (1/53) |
| nga_art | hair | 36 | 8.3% (3/36) | 33 | 1.0 | 3.6 | 8 | 18.2% (6/33) | 9.1% (3/33) | 0.0% (0/33) |
| nga_art | small thing |  |  | 218 | 5.0 | 40.9 | 294 | 77.1% (168/218) | 54.6% (119/218) | 26.6% (58/218) |
| sav | person | 520 | 12.7% (66/520) | 454 | 3.0 | 8.0 | 51 | 74.9% (340/454) | 31.5% (143/454) | 0.4% (2/454) |
| sav | head | 158 | 6.3% (10/158) | 148 | 2.0 | 5.0 | 9 | 66.9% (99/148) | 11.5% (17/148) | 0.0% (0/148) |
| sav | hand | 173 | 13.9% (24/173) | 149 | 2.0 | 6.0 | 11 | 69.8% (104/149) | 16.8% (25/149) | 0.0% (0/149) |
| sav | hair | 21 | 0.0% (0/21) | 21 | 2.0 | 4.0 | 4 | 71.4% (15/21) | 0.0% (0/21) | 0.0% (0/21) |
| sav | small thing |  |  | 678 | 4.0 | 23.0 | 245 | 78.8% (534/678) | 44.1% (299/678) | 15.8% (107/678) |
| yt1b | person | 112 | 12.5% (14/112) | 98 | 4.0 | 20.0 | 40 | 68.4% (67/98) | 43.9% (43/98) | 17.3% (17/98) |
| yt1b | head | 16 | 12.5% (2/16) | 14 | 1.5 | 11.4 | 34 | 50.0% (7/14) | 35.7% (5/14) | 7.1% (1/14) |
| yt1b | hand | 43 | 9.3% (4/43) | 39 | 1.0 | 7.2 | 14 | 43.6% (17/39) | 23.1% (9/39) | 0.0% (0/39) |
| yt1b | hair | 4 | 0.0% (0/4) | 4 | 2.5 | 7.5 | 9 | 50.0% (2/4) | 25.0% (1/4) | 0.0% (0/4) |
| yt1b | small thing |  |  | 139 | 6.0 | 38.6 | 167 | 79.9% (111/139) | 59.7% (83/139) | 24.5% (34/139) |


### VEval: instances per positive pair for the lane's targets

| split | target | pairs asked | negative | positive pairs | instances median | p90 | max | >= 2 | >= 5 |
|---|---|---|---|---|---|---|---|---|---|
| sav_test | person | 6 | 0.0% (0/6) | 6 | 3.0 | 6.5 | 8 | 66.7% (4/6) | 33.3% (2/6) |
| sav_test | head | 9 | 0.0% (0/9) | 9 | 2.0 | 3.6 | 6 | 88.9% (8/9) | 11.1% (1/9) |
| sav_test | hand | 14 | 0.0% (0/14) | 14 | 2.0 | 4.7 | 8 | 92.9% (13/14) | 14.3% (2/14) |
| sav_test | small thing | 123 |  | 123 | 1.0 | 3.8 | 13 | 46.3% (57/123) | 4.9% (6/123) |
| sav_val | person | 4 | 0.0% (0/4) | 4 | 1.0 | 3.1 | 4 | 25.0% (1/4) | 0.0% (0/4) |
| sav_val | head | 3 | 0.0% (0/3) | 3 | 4.0 | 10.4 | 12 | 100.0% (3/3) | 33.3% (1/3) |
| sav_val | hand | 12 | 0.0% (0/12) | 12 | 2.5 | 4.0 | 12 | 100.0% (12/12) | 8.3% (1/12) |
| sav_val | small thing | 124 |  | 124 | 1.0 | 3.0 | 11 | 40.3% (50/124) | 5.7% (7/124) |
| smartglasses_test | person | 1 | 0.0% (0/1) | 1 | 14.0 | 14.0 | 14 | 100.0% (1/1) | 100.0% (1/1) |
| smartglasses_test | small thing | 113 |  | 113 | 2.0 | 4.8 | 10 | 66.4% (75/113) | 10.6% (12/113) |
| smartglasses_val | person | 2 | 0.0% (0/2) | 2 | 14.5 | 15.7 | 16 | 100.0% (2/2) | 100.0% (2/2) |
| smartglasses_val | small thing | 107 |  | 107 | 2.0 | 5.0 | 12 | 57.9% (62/107) | 16.8% (18/107) |
| yt1b_test | person | 11 | 0.0% (0/11) | 11 | 15.0 | 23.0 | 25 | 100.0% (11/11) | 90.9% (10/11) |
| yt1b_test | small thing | 62 |  | 62 | 2.0 | 7.0 | 12 | 67.7% (42/62) | 27.4% (17/62) |
| yt1b_val | person | 16 | 0.0% (0/16) | 16 | 12.5 | 22.0 | 25 | 100.0% (16/16) | 100.0% (16/16) |
| yt1b_val | small thing | 75 |  | 75 | 3.0 | 9.6 | 15 | 74.7% (56/75) | 34.7% (26/75) |


### VEval: how long a masklet is gone (frames are 24 a second in SA-V, 6 in the others)

| split | class | masklets | present, median fraction of the video | absent, median share of its own span | leaves and returns | gaps | gap seconds: median / p95 / max | gaps <= 0.5 s | <= 1 s | <= 2 s | <= 5 s | gone at the last frame |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sav_test | all masklets | 1557 | 0.77 | 0.00 | 47.7% (743/1557) | 1846 | 0.17 / 1.71 / 4.25 | 78.5% (1449/1846) | 90.1% (1664/1846) | 96.8% (1786/1846) | 100.0% (1846/1846) | 37.5% (584/1557) |
| sav_test | person | 21 | 0.34 | 0.00 | 28.6% (6/21) | 9 | 0.71 / 2.77 / 3.62 | 44.4% (4/9) | 77.8% (7/9) | 88.9% (8/9) | 100.0% (9/9) | 66.7% (14/21) |
| sav_test | head | 23 | 0.87 | 0.00 | 34.8% (8/23) | 15 | 0.12 / 0.50 / 0.71 | 93.3% (14/15) | 100.0% (15/15) | 100.0% (15/15) | 100.0% (15/15) | 30.4% (7/23) |
| sav_test | hand | 43 | 0.74 | 0.10 | 69.8% (30/43) | 104 | 0.12 / 1.24 / 2.79 | 82.7% (86/104) | 93.3% (97/104) | 98.1% (102/104) | 100.0% (104/104) | 37.2% (16/43) |
| sav_test | small thing | 244 | 0.57 | 0.06 | 63.1% (154/244) | 436 | 0.12 / 1.29 / 2.92 | 78.2% (341/436) | 92.7% (404/436) | 98.2% (428/436) | 100.0% (436/436) | 49.6% (121/244) |
| sav_val | all masklets | 1698 | 0.70 | 0.00 | 47.3% (803/1698) | 1869 | 0.17 / 1.61 / 4.17 | 77.3% (1444/1869) | 89.1% (1666/1869) | 97.0% (1812/1869) | 100.0% (1869/1869) | 41.2% (700/1698) |
| sav_val | person | 7 | 0.81 | 0.19 | 71.4% (5/7) | 15 | 0.12 / 1.20 / 1.67 | 86.7% (13/15) | 93.3% (14/15) | 100.0% (15/15) | 100.0% (15/15) | 42.9% (3/7) |
| sav_val | head | 19 | 0.12 | 0.00 | 47.4% (9/19) | 21 | 0.08 / 0.62 / 0.75 | 90.5% (19/21) | 100.0% (21/21) | 100.0% (21/21) | 100.0% (21/21) | 47.4% (9/19) |
| sav_val | hand | 42 | 0.66 | 0.00 | 45.2% (19/42) | 69 | 0.12 / 1.13 / 2.58 | 84.1% (58/69) | 92.8% (64/69) | 98.6% (68/69) | 100.0% (69/69) | 50.0% (21/42) |
| sav_val | small thing | 225 | 0.31 | 0.02 | 50.2% (113/225) | 290 | 0.17 / 1.46 / 3.75 | 76.9% (223/290) | 90.7% (263/290) | 97.6% (283/290) | 100.0% (290/290) | 59.6% (134/225) |
| smartglasses_test | all masklets | 1792 | 0.31 | 0.19 | 71.4% (1279/1792) | 2886 | 1.17 / 8.17 / 19.33 | 27.7% (800/2886) | 46.3% (1335/2886) | 68.3% (1971/2886) | 89.6% (2585/2886) | 65.6% (1176/1792) |
| smartglasses_test | person | 14 | 0.09 | 0.00 | 35.7% (5/14) | 6 | 3.92 / 6.12 / 6.33 | 0.0% (0/6) | 0.0% (0/6) | 0.0% (0/6) | 66.7% (4/6) | 100.0% (14/14) |
| smartglasses_test | small thing | 290 | 0.21 | 0.23 | 70.7% (205/290) | 515 | 1.17 / 7.55 / 11.83 | 34.0% (175/515) | 49.7% (256/515) | 71.8% (370/515) | 91.6% (472/515) | 73.5% (213/290) |
| smartglasses_val | all masklets | 1686 | 0.28 | 0.22 | 70.6% (1190/1686) | 2790 | 1.33 / 8.00 / 18.33 | 28.5% (794/2790) | 44.0% (1229/2790) | 63.4% (1769/2790) | 87.5% (2441/2790) | 71.2% (1201/1686) |
| smartglasses_val | person | 29 | 0.18 | 0.33 | 62.1% (18/29) | 32 | 2.08 / 9.77 / 10.50 | 21.9% (7/32) | 34.4% (11/32) | 50.0% (16/32) | 90.6% (29/32) | 86.2% (25/29) |
| smartglasses_val | small thing | 276 | 0.17 | 0.16 | 68.5% (189/276) | 431 | 1.00 / 9.67 / 18.33 | 38.8% (167/431) | 50.8% (219/431) | 65.7% (283/431) | 85.2% (367/431) | 77.2% (213/276) |
| yt1b_test | all masklets | 2330 | 0.67 | 0.00 | 33.8% (788/2330) | 1348 | 0.50 / 5.44 / 17.00 | 50.5% (681/1348) | 67.0% (903/1348) | 81.0% (1092/1348) | 94.4% (1273/1348) | 40.4% (941/2330) |
| yt1b_test | person | 172 | 0.41 | 0.03 | 54.1% (93/172) | 152 | 0.50 / 4.82 / 10.67 | 51.3% (78/152) | 65.8% (100/152) | 77.0% (117/152) | 95.4% (145/152) | 54.1% (93/172) |
| yt1b_test | small thing | 208 | 0.54 | 0.03 | 51.0% (106/208) | 213 | 0.50 / 6.63 / 14.83 | 54.0% (115/213) | 79.3% (169/213) | 88.3% (188/213) | 93.9% (200/213) | 46.6% (97/208) |
| yt1b_val | all masklets | 2162 | 0.72 | 0.00 | 36.5% (789/2162) | 1380 | 0.50 / 3.83 / 14.67 | 56.4% (778/1380) | 71.7% (989/1380) | 85.7% (1182/1380) | 97.1% (1340/1380) | 36.7% (793/2162) |
| yt1b_val | person | 221 | 0.69 | 0.00 | 43.9% (97/221) | 150 | 0.33 / 3.50 / 4.83 | 60.0% (90/150) | 74.7% (112/150) | 86.0% (129/150) | 100.0% (150/150) | 31.2% (69/221) |
| yt1b_val | small thing | 321 | 0.54 | 0.02 | 50.2% (161/321) | 310 | 0.50 / 3.59 / 9.33 | 55.5% (172/310) | 74.8% (232/310) | 87.4% (271/310) | 98.1% (304/310) | 50.8% (163/321) |

Controls: every masklet's present, leading, trailing and gap frames add up to its video length (asserted); the masklets that leave and return, counted here, equal the committed count: sav_test True, sav_val True, smartglasses_test True, smartglasses_val True, yt1b_test True, yt1b_val True

## What this does not establish

- That a held object behaves like a small one: held things are a subset of the size class, and the size class includes background clutter.
- Why an object is gone. A short gap may be an occlusion by a hand, a flicker of the annotation, or the object turning; the files do not say.
- Anything for our footage: the clips are other people's, at other resolutions and distances.
- A model's regain. It is the figure a regain test can be set against, nothing more.

## Reproducing it

```
python bench/sam3_dataset_phrases.py targets --gold G --silver S --veval V --out <json>
python bench/sam3_dataset_phrases.py render-targets --json <json>
```

## Files

- `2026-10-07_sam3_benchmark_targets.json`: every aggregate behind the tables (49 KB); no phrase.
