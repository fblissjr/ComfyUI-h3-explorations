# What Meta's four SAM 3 datasets say about a valid prompt, counted from their annotation files (2026-10-07)

lane: masked
verdict: Meta's benchmarks ask for things with short plain noun phrases (one to three words in 72% to 92% of Gold's pairs outside its Attributes subset, about three in five with an article), name a relation or a position in a few percent of pairs at most, and ask for a person with a bare person word that is almost never a negative; how often a phrase is a negative is mostly how each subset was built (Gold 84% of pairs, VEval 75%, Silver without its two species sets 52%), with descriptive phrases mildly more often negative in the image sets (odds 1.16 in Gold, 1.09 in Silver) and less often in video (0.72); VEval's annotations carry the phrase and nothing else about a person, so picking one of several people is not something these files show Meta doing with words; tracking sixteen people with `person` is inside what the video benchmark holds and thirty-two is at its edge (one pair of 2,547)

**Five lines to act on.** Each says how it is known; the tables are below.

1. **A person: ask `person`.** It is the form Meta uses when it means every person, and the benchmark almost never says no to it: 269 Gold pairs with the bare word, 2.2% negative; 40 VEval pairs, none; 1,263 Silver pairs, 10.4%. A bare person word is also the commonest way a person is asked (39% of Gold's person-word pairs, 53% of Silver's, 40% of VEval's) and is a negative 12% to 16% of the time, against 44% to 55% for any other two-word form (counted, "People" below).
2. **A prop (a cigarette, an ice cream cone): the plain singular noun, nothing added.** Objects that small are in the benchmark (24% of Gold SA-1B's annotations are under a thousandth of the image, an equal-area square about 32 px on a side at Meta's 1008 canvas; its ice cream cone annotations are 18 to 39 px), but a cigarette is barely tested: 55 asked pairs across the four datasets, 48 of them negatives, 8 annotations. What the benchmark supports for small things is their size, not a cigarette (counted, "Size").
3. **A part (a hand, a face): the part noun alone.** Pairs with a body word and no person word outnumber those with one 27 to 1 in Gold, 8.5 to 1 in Silver and 4.6 to 1 in VEval; "person plus part" exists and is mostly positive (73% to 81%) but is the minority form (counted, "People").
4. **Not worth asking: a description that picks one person out** (what they wear or hold, where they stand). Relation words (wearing, holding, with, in, on) are in 0% to 2.6% of pairs of any Gold subset and up to 0.4% of VEval's, position words in up to 4.3% of any set, and phrases of three or more words outside the modifier patterns are negatives 61% to 98% of the time in every Gold subset. VEval's annotations carry a phrase per object and no per-person text, and Meta's own way to choose one among several is a box or a click (mryolk's guide). This says the benchmark does not test such phrases, not that they fail.
5. **Our use is inside what is covered up to sixteen people and at the edge at thirty-two.** In Gold Crowded the bare `person` has a median of 19 instances and 5 of 44 positive pairs reach 32 or more; in video the largest pair in any VEval file holds 33 masklets (one of 2,547 positive pairs) and the largest `person` pair 25; benchmark clips run about 2 to 30 s. A bare-`person` pair with 32 people is therefore a tail case here, not a typical one.

**What was asked.** The owner, 2026-10-07, via the lead: what the annotation files of Meta's four SAM 3 datasets say about what a valid prompt is and how specific to be, as counts with their denominators: what a phrase looks like, how often it is a negative (asked, nothing annotated: the benchmark's own statement of what the model should refuse), the size of what is annotated, how people and parts of people are asked, and how video handles one person among several.

**How.** [`bench/sam3_dataset_phrases.py`](../sam3_dataset_phrases.py) reads the annotation JSON only (no image or video). A *pair* is one asked (image or video, noun phrase); it is *positive* if an instance is annotated and a *negative* otherwise, as Gold's README defines. Gold has three annotators over the same pairs; tables use annotator a, and the agreement table gives all three. A phrase's *form* is a lexical class, no tagger: bare noun, two words, modifier plus noun (the modifier list has 460 words: colours, three shade words, and every word that opens at least two distinct phrases of Gold's own Attributes subset), relational, positional, of-phrase, or longer; *descriptive* means any form except the first two. Sizes are the annotated box's area over the image's. The pooled contrast is a Mantel-Haenszel odds ratio over subsets, with an interval that resamples distinct phrases (300 resamples, seed 0) because a phrase is asked in many pairs. No picture was looked at and no model was run.

**Controls the numbers carry.** Gold's pair counts equal Meta's README table in all seven subsets, and its mask counts equal annotator a's in six and annotator b's in Food/Drink; VEval's global phrase map has 51,248 phrases and the shares mryolk recorded (3.9 / 23.4 / 39.3 / 19.9 / 12.1 / 1.5 % for 1 / 2 / 3 / 4 / 5-6 / 7+ words, against his 4 / 23 / 39 / 20 / 1.5 for 7+); every VEval pair's `num_masklets` equals its annotation rows and every annotation repeats its pair's phrase (100% in all six files). **A correction to the known figures:** the 51,248 phrases are the whole phrase map every VEval and SA-FARI file carries, not what is asked; the asked pairs per VEval file number 1,416 to 2,237 and are shorter (four or more words in 0.9% to 12.5% of them, against 33.5% of the map).

**Licence.** All four repositories carry the SAM License of 2025-11-19 (a royalty-free right to use, reproduce, distribute and modify the SAM Materials, redistribution with the licence text, acknowledgement in published research; annotation files are not named in its definition), and the READMEs add CC-BY-NC 4.0 for SA-FARI, SA-V and YT-Temporal-1B and CC-BY 4.0 for SmartGlasses. The data are gated and not in this repository; this record and its json hold aggregates only, no phrase list, and quote no phrase. Counts derived from Meta's SA-Co annotations; Meta Platforms, SAM License.

**Reading the numbers.** (1) A negative rate by phrase shape is a statement about how Meta built the benchmark (which phrases it asked where nothing is annotated), not about a model; the datasets show what the benchmark expects to be refused, not what a model refuses. (2) Forms are lexical: a two-word noun compound and an adjective plus noun are told apart only by the learned word list. (3) The two Silver sources named by species (Fathomnet, iNaturalist) are scientific names and pull the Silver totals; the sentences about Silver use the other eight sources where they say so. (4) "Position" words are counted wherever they occur, so a word for a part of an object ("top", "front") counts; I did not separate those. (5) Nothing here measures how well SAM 3.1 does on any of these phrases.

## The tables

Every table below is printed from `2026-10-07_sam3_benchmark_phrases.json` by the tool's `render`; where a sentence and the file disagree, the file is right. Left out here, and in the json and the tool's `render`: Silver's instance and person tables and every SA-FARI table.

### 1. What a phrase looks like

#### Gold: phrase shapes, annotator a

| subset | pairs | distinct | negative | 1 word | 2 | 3 | 4 | 5-6 | 7+ | article | colour | relational | positional | descriptive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| metaclip | 33393 | 22289 | 81.5% | 9.5% | 34.5% | 36.6% | 11.7% | 6.5% | 1.2% | 66.2% | 8.3% | 0.6% | 1.6% | 37.0% |
| sa1b | 13258 | 8163 | 42.2% | 10.9% | 35.2% | 31.5% | 12.9% | 8.4% | 1.0% | 66.0% | 25.4% | 0.7% | 2.4% | 52.1% |
| attributes | 9245 | 2610 | 80.4% | 0.1% | 96.1% | 3.8% | 0.0% | 0.0% | 0.0% | 0.0% | 10.8% | 0.0% | 0.2% | 93.9% |
| crowded | 20687 | 15754 | 76.2% | 8.1% | 30.0% | 34.1% | 16.0% | 10.2% | 1.7% | 64.9% | 17.3% | 0.4% | 1.3% | 51.7% |
| wiki_common | 65502 | 6770 | 96.8% | 13.5% | 42.6% | 36.3% | 6.2% | 1.4% | 0.0% | 63.4% | 0.1% | 0.2% | 0.1% | 19.6% |
| fg_food | 13951 | 2906 | 89.7% | 19.6% | 42.9% | 30.0% | 5.9% | 1.5% | 0.2% | 58.2% | 2.7% | 0.7% | 0.0% | 17.2% |
| fg_sports_equipment | 12166 | 804 | 83.0% | 13.7% | 39.0% | 35.0% | 8.9% | 3.4% | 0.0% | 57.4% | 1.3% | 2.6% | 0.2% | 27.7% |

#### Gold: share of pairs by phrase form

| subset | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | distinct phrases seen in both positive and negative pairs |
|---|---|---|---|---|---|---|---|---|---|
| metaclip | 27.6% | 35.4% | 12.0% | 2.0% | 4.0% | 0.6% | 1.6% | 16.8% | 4.9% (1100/22289) |
| sa1b | 32.5% | 15.4% | 22.4% | 4.3% | 5.6% | 0.7% | 2.4% | 16.7% | 5.9% (485/8163) |
| attributes | 0.1% | 6.0% | 89.9% | 0.7% | 0.0% | 0.0% | 0.2% | 3.1% | 31.7% (828/2610) |
| crowded | 22.7% | 25.5% | 16.4% | 3.9% | 10.8% | 0.4% | 1.3% | 18.9% | 2.9% (461/15754) |
| wiki_common | 36.7% | 43.7% | 8.7% | 0.3% | 0.1% | 0.2% | 0.1% | 10.2% | 14.9% (1009/6770) |
| fg_food | 44.6% | 38.2% | 6.2% | 0.0% | 0.3% | 0.7% | 0.0% | 10.1% | 18.2% (529/2906) |
| fg_sports_equipment | 30.9% | 41.3% | 9.6% | 1.3% | 0.0% | 2.6% | 0.2% | 14.1% | 42.3% (340/804) |


#### Silver: phrase shapes

| source | pairs | distinct | negative | 1 word | 2 | 3 | 4 | 5-6 | 7+ | article | colour | relational | positional | descriptive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bdd100k | 5546 | 2508 | 41.3% | 16.0% | 39.1% | 25.5% | 11.7% | 6.9% | 0.9% | 57.5% | 19.4% | 0.6% | 1.3% | 41.6% |
| droid | 9445 | 5591 | 51.3% | 14.0% | 31.9% | 30.2% | 14.8% | 8.3% | 0.8% | 56.8% | 30.8% | 0.4% | 1.0% | 56.5% |
| ego4d | 12608 | 6246 | 24.5% | 12.3% | 38.3% | 32.0% | 11.5% | 5.3% | 0.5% | 66.1% | 29.8% | 0.3% | 3.5% | 51.7% |
| fathomnet | 287193 | 1962 | 96.3% | 59.5% | 40.3% | 0.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% | 0.0% | 0.0% | 0.3% |
| food_rec | 20985 | 11833 | 55.9% | 15.5% | 29.7% | 24.6% | 16.4% | 11.8% | 2.1% | 55.2% | 18.3% | 0.8% | 0.6% | 51.1% |
| geode | 14850 | 5283 | 67.6% | 18.8% | 38.8% | 21.3% | 10.9% | 8.5% | 1.6% | 45.4% | 5.6% | 0.7% | 1.2% | 36.7% |
| inaturalist | 1439051 | 2845 | 96.8% | 0.0% | 94.9% | 5.1% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 5.1% |
| nga_art | 22294 | 14511 | 74.4% | 17.3% | 31.3% | 25.0% | 14.0% | 9.8% | 2.5% | 49.8% | 8.5% | 1.8% | 1.6% | 44.0% |
| sav | 18337 | 6519 | 29.1% | 17.4% | 46.0% | 25.1% | 7.2% | 3.6% | 0.6% | 67.0% | 23.5% | 0.4% | 1.4% | 37.2% |
| yt1b | 7816 | 6006 | 54.1% | 17.4% | 35.5% | 26.9% | 11.8% | 7.5% | 1.0% | 54.4% | 21.3% | 0.6% | 2.0% | 45.5% |


#### VEval: phrase shapes (asked pairs)

| split | pairs | distinct | negative | 1 word | 2 | 3 | 4 | 5-6 | 7+ | article | colour | relational | positional | descriptive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sav_test | 1454 | 1222 | 55.7% | 11.5% | 44.4% | 31.6% | 8.1% | 4.3% | 0.2% | 69.0% | 33.4% | 0.2% | 3.9% | 48.1% |
| sav_val | 1504 | 1248 | 54.6% | 12.8% | 43.5% | 31.4% | 6.9% | 5.2% | 0.3% | 67.5% | 36.0% | 0.1% | 4.3% | 48.5% |
| smartglasses_test | 2221 | 1551 | 84.3% | 14.4% | 45.8% | 33.8% | 4.8% | 1.1% | 0.1% | 65.5% | 14.7% | 0.0% | 0.4% | 41.1% |
| smartglasses_val | 2237 | 1518 | 85.1% | 15.9% | 48.6% | 31.8% | 3.2% | 0.5% | 0.0% | 65.7% | 15.0% | 0.1% | 0.1% | 36.2% |
| yt1b_test | 1536 | 677 | 81.6% | 28.1% | 55.1% | 15.1% | 1.4% | 0.3% | 0.0% | 64.5% | 4.3% | 0.0% | 0.3% | 12.4% |
| yt1b_val | 1416 | 650 | 82.0% | 29.9% | 55.2% | 13.9% | 0.8% | 0.1% | 0.0% | 61.9% | 4.2% | 0.4% | 0.6% | 11.8% |

#### VEval: share of pairs by phrase form

| split | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | distinct phrases seen in both positive and negative pairs |
|---|---|---|---|---|---|---|---|---|---|
| sav_test | 41.2% | 10.7% | 29.6% | 3.0% | 1.9% | 0.2% | 3.9% | 9.5% | 4.4% (54/1222) |
| sav_val | 41.4% | 10.0% | 30.4% | 2.2% | 1.3% | 0.1% | 4.3% | 10.2% | 6.2% (77/1248) |
| smartglasses_test | 42.6% | 16.2% | 32.8% | 0.7% | 3.0% | 0.0% | 0.4% | 4.1% | 2.1% (33/1551) |
| smartglasses_val | 47.6% | 16.1% | 31.3% | 1.0% | 0.7% | 0.1% | 0.1% | 3.0% | 3.1% (47/1518) |
| yt1b_test | 76.6% | 10.9% | 9.8% | 0.3% | 1.0% | 0.0% | 0.3% | 1.0% | 5.8% (39/677) |
| yt1b_val | 78.2% | 10.0% | 9.3% | 0.1% | 0.2% | 0.4% | 0.6% | 1.3% | 6.0% (39/650) |


### 2. Positive against negative

#### Gold: share of pairs that are negative, by phrase form

| subset | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | descriptive | plain | odds ratio, descriptive vs plain (pairs treated as independent) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| metaclip | 73.4% (6757/9207) | 89.0% (10526/11832) | 68.9% (2757/3999) | 61.9% (421/680) | 85.4% (1132/1325) | 91.6% (196/214) | 71.0% (380/535) | 90.2% (5052/5601) | 80.4% (9938/12354) | 82.2% (17283/21039) | 0.894 (0.845, 0.946) |
| sa1b | 29.3% (1264/4306) | 56.5% (1154/2044) | 39.6% (1177/2974) | 34.0% (196/576) | 44.1% (325/737) | 56.4% (53/94) | 23.5% (74/315) | 60.9% (1347/2212) | 45.9% (3172/6908) | 38.1% (2418/6350) | 1.381 (1.288, 1.48) |
| attributes | 75.0% (6/8) | 69.6% (389/559) | 81.3% (6754/8309) | 71.2% (47/66) | 66.7% (2/3) |  | 100.0% (15/15) | 77.2% (220/285) | 81.1% (7038/8678) | 69.7% (395/567) | 1.871 (1.553, 2.256) |
| crowded | 54.5% (2565/4704) | 88.1% (4654/5281) | 65.5% (2221/3391) | 73.5% (597/812) | 86.1% (1927/2239) | 75.0% (69/92) | 76.2% (202/265) | 90.3% (3525/3903) | 79.8% (8541/10702) | 72.3% (7219/9985) | 1.514 (1.42, 1.615) |
| wiki_common | 97.2% (23396/24062) | 96.4% (27608/28629) | 95.1% (5448/5727) | 99.4% (169/170) | 100.0% (62/62) | 100.0% (105/105) | 97.6% (81/83) | 98.1% (6535/6664) | 96.8% (12400/12811) | 96.8% (51004/52691) | 0.997 (0.894, 1.112) |
| fg_food | 87.5% (5443/6220) | 90.9% (4845/5329) | 89.7% (776/865) |  | 93.0% (40/43) | 97.8% (88/90) |  | 94.0% (1319/1404) | 92.5% (2223/2402) | 89.1% (10288/11549) | 1.519 (1.29, 1.788) |
| fg_sports_equipment | 79.7% (3001/3764) | 86.6% (4353/5029) | 70.5% (826/1171) | 90.6% (144/159) |  | 89.7% (280/312) | 94.7% (18/19) | 86.2% (1476/1712) | 81.3% (2744/3373) | 83.6% (7354/8793) | 0.853 (0.77, 0.946) |

Pooled across the seven subsets (Mantel-Haenszel, descriptive vs plain, odds of being a negative): odds ratio 1.164, 95% interval (1.128, 1.201) treating pairs as independent, (1.089, 1.233) resampling distinct phrases (300 resamples, seed 0)

Gold Attributes, how an attribute phrase begins (distinct two-or-more-word phrases 2607): other 56.3% (1467/2607), material 17.9% (468/2607), colour 14.5% (377/2607), size 5.2% (136/2607), shape 6.1% (159/2607); the ten commonest first words cover 21.2% (554/2607)

#### Gold: three annotators over the same pairs

| subset | negative a / b / c | positive in 0 / 1 / 2 / 3 of 3 | contested (1 or 2 of 3) of pairs any annotator marked | by form: plain noun / modifier + noun / relational / of-phrase |
|---|---|---|---|---|
| metaclip | 81.5% / 81.5% / 81.3% | 26045 / 1173 / 1092 / 5083 | 30.8% (2265/7348) | 25.4% (719/2826) / 28.4% (409/1442) / 56.0% (14/25) / 25.7% (55/214) |
| sa1b | 42.2% / 42.1% / 42.2% | 4937 / 497 / 966 / 6858 | 17.6% (1463/8321) | 9.3% (296/3176) / 24.0% (484/2013) / 35.9% (19/53) / 13.0% (57/439) |
| attributes | 80.4% / 80.8% / 80.4% | 7041 / 422 / 368 / 1414 | 35.8% (790/2204) | 0.0% (0/2) / 35.9% (678/1891) /  / 50.0% (1/2) |
| crowded | 76.2% / 76.2% / 76.2% | 15466 / 242 / 408 / 4571 | 12.4% (650/5221) | 8.2% (183/2219) / 17.6% (223/1265) / 12.0% (3/25) / 17.3% (58/335) |
| wiki_common | 96.8% / 96.8% / 96.8% | 63167 / 199 / 320 / 1816 | 22.2% (519/2335) | 26.3% (194/738) / 19.9% (61/307) /  /  |
| fg_food | 89.7% / 89.5% / 89.7% | 12268 / 245 / 224 / 1214 | 27.9% (469/1683) | 23.4% (206/881) / 33.0% (35/106) / 0.0% (0/2) / 25.0% (1/4) |
| fg_sports_equipment | 83.0% / 83.2% / 83.0% | 9766 / 359 / 301 / 1740 | 27.5% (660/2400) | 25.1% (220/875) / 21.5% (83/386) / 20.6% (7/34) /  |


#### Silver: share of pairs that are negative, by phrase form

| source | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | descriptive | plain | odds ratio, descriptive vs plain (pairs treated as independent) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bdd100k | 29.8% (629/2111) | 58.1% (656/1130) | 23.6% (214/908) | 41.7% (70/168) | 43.3% (127/293) | 93.9% (31/33) | 48.6% (35/72) | 63.5% (528/831) | 43.6% (1005/2305) | 39.6% (1285/3241) | 1.177 (1.056, 1.311) |
| droid | 48.4% (1245/2570) | 74.0% (1135/1534) | 32.3% (792/2454) | 35.2% (228/647) | 71.6% (302/422) | 41.5% (17/41) | 63.0% (58/92) | 63.4% (1068/1685) | 46.2% (2465/5341) | 58.0% (2380/4104) | 0.621 (0.572, 0.674) |
| ego4d | 23.2% (1072/4619) | 24.5% (361/1474) | 24.3% (811/3343) | 25.3% (187/738) | 24.4% (162/664) | 41.7% (15/36) | 18.6% (82/440) | 30.8% (398/1294) | 25.4% (1655/6515) | 23.5% (1433/6093) | 1.107 (1.021, 1.201) |
| fathomnet | 96.6% (165133/170940) | 96.0% (110737/115369) | 75.4% (319/423) |  |  |  |  | 100.0% (461/461) | 88.2% (780/884) | 96.4% (275870/286309) | 0.283 (0.23, 0.347) |
| food_rec | 47.6% (2846/5974) | 67.3% (2887/4290) | 38.5% (1115/2896) | 40.9% (374/914) | 65.0% (1712/2632) | 55.9% (95/170) | 68.9% (91/132) | 65.7% (2614/3977) | 56.0% (6001/10721) | 55.9% (5733/10264) | 1.005 (0.952, 1.061) |
| geode | 35.5% (1888/5313) | 79.9% (3270/4092) | 66.1% (709/1072) | 100.0% (240/240) | 91.1% (1123/1233) | 100.0% (104/104) | 51.8% (87/168) | 99.8% (2622/2628) | 89.7% (4885/5445) | 54.8% (5158/9405) | 7.177 (6.518, 7.903) |
| inaturalist | 94.5% (499/528) | 96.8% (1321546/1365736) |  |  |  |  |  | 97.0% (70585/72787) | 97.0% (70585/72787) | 96.8% (1322045/1366264) | 1.072 (1.026, 1.12) |
| nga_art | 64.3% (4178/6500) | 83.6% (5012/5992) | 57.4% (1272/2215) | 60.1% (388/646) | 83.3% (1771/2126) | 91.6% (362/395) | 52.6% (174/331) | 84.1% (3440/4089) | 75.6% (7407/9802) | 73.6% (9190/12492) | 1.111 (1.046, 1.181) |
| sav | 34.7% (3368/9705) | 23.3% (423/1812) | 21.6% (849/3931) | 19.4% (147/759) | 33.7% (149/442) | 25.0% (17/68) | 19.8% (49/247) | 24.7% (339/1373) | 22.7% (1550/6820) | 32.9% (3791/11517) | 0.6 (0.56, 0.642) |
| yt1b | 46.4% (1223/2637) | 70.9% (1151/1623) | 37.6% (549/1458) | 36.3% (111/306) | 69.0% (254/368) | 66.7% (30/45) | 43.0% (67/156) | 69.0% (844/1223) | 52.2% (1855/3556) | 55.7% (2374/4260) | 0.866 (0.792, 0.947) |

Pooled across sources (Mantel-Haenszel): odds ratio 1.093, 95% interval (1.07, 1.117) treating pairs as independent, (1.028, 1.167) resampling distinct phrases (300 resamples, seed 0)


#### VEval: share of pairs that are negative, by phrase form

| split | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | descriptive | plain | odds ratio, descriptive vs plain (pairs treated as independent) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sav_test | 57.1% (342/599) | 51.3% (80/156) | 60.9% (262/430) | 53.5% (23/43) | 35.7% (10/28) | 66.7% (2/3) | 14.0% (8/57) | 60.1% (83/138) | 55.5% (388/699) | 55.9% (422/755) | 0.984 (0.8, 1.211) |
| sav_val | 58.6% (365/623) | 45.7% (69/151) | 53.3% (244/458) | 54.5% (18/33) | 68.4% (13/19) | 50.0% (1/2) | 30.8% (20/65) | 59.5% (91/153) | 53.0% (387/730) | 56.1% (434/774) | 0.884 (0.722, 1.083) |
| smartglasses_test | 89.4% (847/947) | 81.4% (294/361) | 84.2% (614/729) | 46.7% (7/15) | 70.2% (47/67) |  | 40.0% (4/10) | 64.1% (59/92) | 80.1% (731/913) | 87.2% (1141/1308) | 0.588 (0.468, 0.74) |
| smartglasses_val | 89.8% (957/1066) | 80.0% (288/360) | 84.5% (592/701) | 31.8% (7/22) | 46.7% (7/15) | 50.0% (1/2) | 0.0% (0/3) | 75.0% (51/68) | 81.1% (658/811) | 87.3% (1245/1426) | 0.625 (0.494, 0.791) |
| yt1b_test | 84.0% (989/1177) | 81.5% (137/168) | 74.8% (113/151) | 0.0% (0/4) | 40.0% (6/15) |  | 0.0% (0/5) | 56.2% (9/16) | 67.0% (128/191) | 83.7% (1126/1345) | 0.394 (0.282, 0.55) |
| yt1b_val | 84.4% (934/1107) | 73.9% (105/142) | 79.5% (105/132) | 0.0% (0/1) | 0.0% (0/3) | 100.0% (5/5) | 0.0% (0/8) | 66.7% (12/18) | 73.0% (122/167) | 83.2% (1039/1249) | 0.545 (0.376, 0.79) |

Pooled (Mantel-Haenszel): odds ratio 0.718, 95% interval (0.649, 0.793) treating pairs as independent, (0.642, 0.805) resampling distinct phrases (300 resamples, seed 0)

VEval global phrase map: 51248 phrases; words 1/2/3/4/5-6/7+ = 3.9% / 23.4% / 39.3% / 19.9% / 12.1% / 1.5%; starts with an article 66.3%; equals mryolk's count: True; phrases holding the props' words: {'cigarette': 5, 'ice cream cone': 12}


### 3. Size

#### Gold: annotated box area as a share of the image

| subset | annotations | p05 | median | p95 | <= 1e-4 (about 10 px square at 1008) | <= 1e-3 (about 32 px) | <= 1e-2 (about 100 px) | phrases whose median is <= 1e-3 |
|---|---|---|---|---|---|---|---|---|
| metaclip | 20144 | 0.00019 | 0.011 | 0.48 | 3.0% (607/20144) | 17.8% (3581/20144) | 48.6% (9795/20144) | 8.4% (123/1460) |
| sa1b | 30306 | 0.0001 | 0.0059 | 0.46 | 5.0% (1517/30306) | 24.1% (7320/30306) | 57.6% (17455/30306) | 12.2% (195/1592) |
| attributes | 3663 | 0.00071 | 0.09 | 0.78 | 0.1% (4/3663) | 6.7% (245/3663) | 20.1% (737/3663) | 2.1% (7/337) |
| crowded | 50417 | 0.0003 | 0.0048 | 0.1 | 1.3% (657/50417) | 17.2% (8667/50417) | 65.8% (33162/50417) | 8.8% (189/2147) |
| wiki_common | 6448 | 0.001 | 0.032 | 0.69 | 0.1% (9/6448) | 4.7% (305/6448) | 33.6% (2167/6448) | 1.4% (8/560) |
| fg_food | 10041 | 0.00097 | 0.013 | 0.36 | 0.3% (29/10041) | 5.2% (520/10041) | 45.1% (4530/10041) | 1.5% (6/410) |
| fg_sports_equipment | 5075 | 0.00072 | 0.038 | 0.69 | 0.8% (39/5075) | 6.8% (344/5075) | 30.1% (1529/5075) | 2.6% (7/267) |

Gold: the owner's props (annotations whose phrase has the words; box share median / equal-area side in px at 1008)
- metaclip: cigarette: no annotation; asked in 3 pairs, 3 negative; ice cream cone: no annotation; asked in 1 pairs, 1 negative
- sa1b: cigarette: 1 annotations (median 0.00072, ~27.1 px); asked in 1 pairs, 0 negative
- crowded: cigarette: no annotation; asked in 4 pairs, 4 negative; ice cream cone: 9 annotations (median 0.0015, ~38.9 px); asked in 2 pairs, 0 negative
- wiki_common: cigarette: 4 annotations (median 0.1, ~319.6 px); asked in 31 pairs, 28 negative
- fg_food: ice cream cone: 8 annotations (median 0.00033, ~18.4 px); asked in 6 pairs, 3 negative

#### Gold: instances annotated per positive pair (any phrase)

| subset | positive pairs | median | p90 | max | >= 2 | >= 5 | >= 10 | >= 16 | >= 32 |
|---|---|---|---|---|---|---|---|---|---|
| metaclip | 6172 | 1.0 | 7.0 | 123 | 47.4% (2926/6172) | 16.4% (1012/6172) | 6.5% (398/6172) | 3.1% (193/6172) | 0.9% (54/6172) |
| sa1b | 7668 | 1.0 | 9.0 | 252 | 48.0% (3681/7668) | 19.3% (1477/7668) | 9.2% (706/7668) | 4.9% (373/7668) | 1.4% (105/7668) |
| attributes | 1812 | 1.0 | 3.0 | 168 | 22.4% (405/1812) | 6.0% (109/1812) | 2.9% (52/1812) | 1.4% (26/1812) | 0.3% (6/1812) |
| crowded | 4927 | 5.0 | 26.0 | 142 | 74.8% (3683/4927) | 50.5% (2490/4927) | 33.7% (1659/4927) | 21.4% (1056/4927) | 7.0% (346/4927) |
| wiki_common | 2098 | 1.0 | 5.0 | 175 | 34.2% (717/2098) | 12.2% (255/2098) | 5.5% (115/2098) | 2.9% (60/2098) | 0.9% (20/2098) |
| fg_food | 1440 | 2.0 | 18.1 | 162 | 59.6% (858/1440) | 30.7% (442/1440) | 16.9% (244/1440) | 11.9% (172/1440) | 4.5% (65/1440) |
| fg_sports_equipment | 2068 | 1.0 | 5.0 | 43 | 43.9% (908/2068) | 12.1% (251/2068) | 4.3% (90/2068) | 1.5% (31/2068) | 0.1% (1/2068) |


#### Silver: annotated box area as a share of the image

| source | annotations | p05 | median | p95 | <= 1e-4 (about 10 px square at 1008) | <= 1e-3 (about 32 px) | <= 1e-2 (about 100 px) | phrases whose median is <= 1e-3 |
|---|---|---|---|---|---|---|---|---|
| bdd100k | 13210 | 0.0001 | 0.0021 | 0.36 | 5.0% (660/13210) | 38.0% (5017/13210) | 69.7% (9211/13210) | 24.8% (130/525) |
| droid | 11098 | 0.00028 | 0.012 | 0.49 | 1.3% (146/11098) | 15.9% (1764/11098) | 47.9% (5314/11098) | 8.3% (68/815) |
| ego4d | 24049 | 0.00028 | 0.016 | 0.51 | 0.8% (186/24049) | 11.8% (2830/24049) | 42.2% (10139/24049) | 4.9% (75/1527) |
| fathomnet | 14174 | 0.00071 | 0.014 | 0.49 | 0.0% (4/14174) | 7.9% (1117/14174) | 44.3% (6276/14174) | 1.5% (7/459) |
| food_rec | 28347 | 0.00027 | 0.016 | 0.99 | 1.5% (418/28347) | 11.9% (3379/28347) | 41.9% (11868/28347) | 3.6% (53/1476) |
| geode | 7570 | 0.0016 | 0.18 | 0.71 | 0.3% (23/7570) | 3.3% (250/7570) | 16.1% (1221/7570) | 0.0% (0/120) |
| inaturalist | 48899 | 0.0047 | 0.16 | 0.63 | 0.1% (31/48899) | 1.5% (738/48899) | 8.2% (3997/48899) | 0.1% (3/2765) |
| nga_art | 18991 | 0.00011 | 0.0078 | 0.76 | 4.6% (878/18991) | 21.0% (3982/18991) | 53.2% (10103/18991) | 9.2% (95/1030) |
| sav | 39683 | 0.00025 | 0.0084 | 0.4 | 1.5% (580/39683) | 18.1% (7189/39683) | 52.5% (20845/39683) | 8.0% (153/1923) |
| yt1b | 12221 | 0.0002 | 0.0096 | 0.45 | 2.1% (258/12221) | 17.9% (2190/12221) | 50.5% (6176/12221) | 10.3% (86/833) |

Silver: the owner's props (annotations whose phrase has the words; box share median / equal-area side in px at 1008)
- bdd100k: cigarette: no annotation; asked in 1 pairs, 1 negative
- droid: ice cream cone: no annotation; asked in 2 pairs, 2 negative
- food_rec: cigarette: no annotation; asked in 3 pairs, 3 negative; ice cream cone: no annotation; asked in 1 pairs, 1 negative
- geode: cigarette: no annotation; asked in 4 pairs, 4 negative
- nga_art: cigarette: 3 annotations (median 0.003, ~55.3 px); asked in 5 pairs, 2 negative
- yt1b: cigarette: no annotation; asked in 2 pairs, 2 negative


#### VEval: tracked objects

| split | videos | masklets | present frames p25 / median / p75 | fraction of video present median | present seconds median | leaves and returns | median box share (frame) | frame box <= 1e-3 | masklets per positive pair median / p90 / max |
|---|---|---|---|---|---|---|---|---|---|
| sav_test | 144 | 1557 | 32 / 75 / 111 | 0.77 | 3.1 | 47.7% (743/1557) | 0.0099 | 15.6% (17841/114129) | 2.0 / 4.0 / 20 |
| sav_val | 146 | 1698 | 24 / 60 / 94 | 0.70 | 2.5 | 47.3% (803/1698) | 0.011 | 10.5% (11230/106612) | 2.0 / 4.0 / 25 |
| smartglasses_test | 343 | 1792 | 17 / 36 / 71 | 0.31 | 6.0 | 71.4% (1279/1792) | 0.0062 | 16.1% (13278/82489) | 4.0 / 10.0 / 25 |
| smartglasses_val | 334 | 1686 | 14 / 34 / 64 | 0.28 | 5.7 | 70.6% (1190/1686) | 0.0063 | 15.8% (11171/70614) | 4.0 / 9.7 / 28 |
| yt1b_test | 387 | 2330 | 16 / 30 / 39 | 0.67 | 5.0 | 33.8% (788/2330) | 0.014 | 9.8% (6984/71347) | 7.0 / 15.0 / 33 |
| yt1b_val | 345 | 2162 | 17 / 31 / 41 | 0.72 | 5.2 | 36.5% (789/2162) | 0.0091 | 14.6% (10088/69042) | 7.0 / 14.6 / 30 |

VEval: clip lengths
- sav_test: frames p05 57 / median 110 / p95 148; seconds p05 2.4 / median 4.6 / p95 6.2 / max 9.0; frames are 24 a second
- sav_val: frames p05 43 / median 100 / p95 166; seconds p05 1.8 / median 4.1 / p95 6.9 / max 8.0; frames are 24 a second
- smartglasses_test: frames p05 120 / median 120 / p95 120; seconds p05 20.0 / median 20.0 / p95 20.0 / max 29.7; frames are 6 a second
- smartglasses_val: frames p05 120 / median 120 / p95 120; seconds p05 20.0 / median 20.0 / p95 20.0 / max 26.7; frames are 6 a second
- yt1b_test: frames p05 32 / median 41 / p95 127; seconds p05 5.3 / median 6.8 / p95 21.1 / max 30.0; frames are 6 a second
- yt1b_val: frames p05 31 / median 43 / p95 117; seconds p05 5.2 / median 7.2 / p95 19.5 / max 30.0; frames are 6 a second

VEval: several people in one video
- sav_test: videos with a person masklet 23; with two or more 16; of those, one shared phrase for all 81.2% (13/16); positive person pairs with exactly one masklet 46.2% (12/26); descriptive form among the one-masklet pairs 33.3% (4/12), among the several-masklet pairs 21.4% (3/14)
- sav_val: videos with a person masklet 20; with two or more 10; of those, one shared phrase for all 80.0% (8/10); positive person pairs with exactly one masklet 52.2% (12/23); descriptive form among the one-masklet pairs 33.3% (4/12), among the several-masklet pairs 9.1% (1/11)
- smartglasses_test: videos with a person masklet 22; with two or more 20; of those, one shared phrase for all 100.0% (20/20); positive person pairs with exactly one masklet 9.1% (2/22); descriptive form among the one-masklet pairs 50.0% (1/2), among the several-masklet pairs 15.0% (3/20)
- smartglasses_val: videos with a person masklet 15; with two or more 15; of those, one shared phrase for all 100.0% (15/15); positive person pairs with exactly one masklet 0.0% (0/15); descriptive form among the one-masklet pairs , among the several-masklet pairs 20.0% (3/15)
- yt1b_test: videos with a person masklet 30; with two or more 30; of those, one shared phrase for all 100.0% (30/30); positive person pairs with exactly one masklet 0.0% (0/30); descriptive form among the one-masklet pairs , among the several-masklet pairs 3.3% (1/30)
- yt1b_val: videos with a person masklet 32; with two or more 32; of those, one shared phrase for all 96.9% (31/32); positive person pairs with exactly one masklet 0.0% (0/33); descriptive form among the one-masklet pairs , among the several-masklet pairs 0.0% (0/33)

VEval: the file's own consistency (control)
- sav_test: pairs whose num_masklets equals the annotation rows 100.0% (1454/1454); annotation phrase equals its category name 100.0% (1557/1557); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']
- sav_val: pairs whose num_masklets equals the annotation rows 100.0% (1504/1504); annotation phrase equals its category name 100.0% (1698/1698); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']
- smartglasses_test: pairs whose num_masklets equals the annotation rows 100.0% (2221/2221); annotation phrase equals its category name 100.0% (1792/1792); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']
- smartglasses_val: pairs whose num_masklets equals the annotation rows 100.0% (2237/2237); annotation phrase equals its category name 100.0% (1686/1686); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']
- yt1b_test: pairs whose num_masklets equals the annotation rows 100.0% (1536/1536); annotation phrase equals its category name 100.0% (2330/2330); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']
- yt1b_val: pairs whose num_masklets equals the annotation rows 100.0% (1416/1416); annotation phrase equals its category name 100.0% (2162/2162); annotation fields ['areas', 'bboxes', 'category_id', 'height', 'id', 'iscrowd', 'noun_phrase', 'segmentations', 'video_id', 'width']


### 4. People

#### Gold: person phrases

| subset | exact person/people pairs | negative | instances per positive pair: median / p90 / max | >= 16 | >= 32 | any person word: pairs, negative | person word and a body word: pairs, negative | body word alone: pairs, negative |
|---|---|---|---|---|---|---|---|---|
| metaclip | 70 | 4.3% (3/70) | 4.0 / 14.0 / 38 | 9.0% (6/67) | 1.5% (1/67) | 789, 49.6% | 46, 30.4% | 1049, 74.6% |
| sa1b | 154 | 1.3% (2/154) | 11.0 / 38.0 / 157 | 43.4% (66/152) | 18.4% (28/152) | 1014, 15.4% | 87, 14.9% | 832, 21.0% |
| attributes | 0 |  |  /  /  |  |  | 55, 61.8% | 0,  | 102, 87.2% |
| crowded | 45 | 2.2% (1/45) | 19.0 / 31.7 / 88 | 65.9% (29/44) | 11.4% (5/44) | 512, 43.2% | 20, 70.0% | 964, 85.7% |
| wiki_common | 0 |  |  /  /  |  |  | 374, 94.7% | 0,  | 1018, 96.9% |
| fg_food | 0 |  |  /  /  |  |  | 26, 100.0% | 0,  | 111, 95.5% |
| fg_sports_equipment | 0 |  |  /  /  |  |  | 40, 95.0% | 0,  | 68, 98.5% |

Gold: size of the annotated exact `person`/`people` objects (box share of the image; per masklet median in video)
- metaclip: n 419; p05 0.00024, median 0.032, p95 0.36; median side of the equal-area square at 1008: 181 px
- sa1b: n 2904; p05 0.00015, median 0.0025, p95 0.15; median side of the equal-area square at 1008: 51 px
- crowded: n 913; p05 0.00045, median 0.0087, p95 0.11; median side of the equal-area square at 1008: 94 px


#### Silver: person phrases

| source | exact person/people pairs | negative | instances per positive pair: median / p90 / max | >= 16 | >= 32 | any person word: pairs, negative | person word and a body word: pairs, negative | body word alone: pairs, negative |
|---|---|---|---|---|---|---|---|---|
| bdd100k | 11 | 0.0% (0/11) | 2.0 / 15.0 / 16 | 9.1% (1/11) | 0.0% (0/11) | 67, 20.9% | 0,  | 17, 76.5% |
| droid | 32 | 28.1% (9/32) | 1.0 / 2.8 / 6 | 0.0% (0/23) | 0.0% (0/23) | 77, 46.8% | 17, 52.9% | 483, 50.9% |
| ego4d | 508 | 4.9% (25/508) | 1.0 / 1.0 / 7 | 0.0% (0/483) | 0.0% (0/483) | 993, 8.6% | 294, 14.6% | 963, 16.2% |
| fathomnet | 0 |  |  /  /  |  |  | 0,  | 0,  | 0,  |
| food_rec | 20 | 35.0% (7/20) | 1.0 / 2.0 / 3 | 0.0% (0/13) | 0.0% (0/13) | 95, 64.2% | 23, 30.4% | 869, 75.6% |
| geode | 0 |  |  /  /  |  |  | 132, 100.0% | 7, 100.0% | 569, 83.1% |
| inaturalist | 0 |  |  /  /  |  |  | 0,  | 0,  | 0,  |
| nga_art | 60 | 16.7% (10/60) | 5.5 / 28.1 / 58 | 36.0% (18/50) | 8.0% (4/50) | 1451, 65.3% | 119, 62.2% | 1119, 49.7% |
| sav | 520 | 12.7% (66/520) | 3.0 / 8.0 / 51 | 0.4% (2/454) | 0.4% (2/454) | 1210, 10.2% | 156, 11.5% | 1267, 19.0% |
| yt1b | 112 | 12.5% (14/112) | 4.0 / 20.0 / 40 | 17.3% (17/98) | 3.1% (3/98) | 531, 24.7% | 51, 31.4% | 380, 35.0% |

Silver: size of the annotated exact `person`/`people` objects (box share of the image; per masklet median in video)
- bdd100k: n 59; p05 0.00025, median 0.0014, p95 0.012; median side of the equal-area square at 1008: 37 px
- droid: n 36; p05 0.0031, median 0.029, p95 0.2; median side of the equal-area square at 1008: 172 px
- ego4d: n 563; p05 0.0059, median 0.26, p95 0.74; median side of the equal-area square at 1008: 514 px
- food_rec: n 17; p05 0.024, median 0.24, p95 0.7; median side of the equal-area square at 1008: 496 px
- nga_art: n 601; p05 0.00027, median 0.0031, p95 0.12; median side of the equal-area square at 1008: 57 px
- sav: n 1791; p05 0.00058, median 0.013, p95 0.36; median side of the equal-area square at 1008: 117 px
- yt1b: n 750; p05 0.00067, median 0.01, p95 0.32; median side of the equal-area square at 1008: 101 px


#### VEval: person phrases

| split | exact person/people pairs | negative | instances per positive pair: median / p90 / max | >= 16 | >= 32 | any person word: pairs, negative | person word and a body word: pairs, negative | body word alone: pairs, negative |
|---|---|---|---|---|---|---|---|---|
| sav_test | 6 | 0.0% (0/6) | 3.0 / 6.5 / 8 | 0.0% (0/6) | 0.0% (0/6) | 33, 21.2% | 17, 17.6% | 127, 11.8% |
| sav_val | 4 | 0.0% (0/4) | 1.0 / 3.1 / 4 | 0.0% (0/4) | 0.0% (0/4) | 31, 25.8% | 14, 28.6% | 110, 12.7% |
| smartglasses_test | 1 | 0.0% (0/1) | 14.0 / 14.0 / 14 | 0.0% (0/1) | 0.0% (0/1) | 29, 24.1% | 18, 11.1% | 22, 77.3% |
| smartglasses_val | 2 | 0.0% (0/2) | 14.5 / 15.7 / 16 | 50.0% (1/2) | 0.0% (0/2) | 25, 40.0% | 14, 21.4% | 9, 66.7% |
| yt1b_test | 11 | 0.0% (0/11) | 15.0 / 23.0 / 25 | 45.5% (5/11) | 0.0% (0/11) | 66, 54.5% | 0,  | 12, 8.3% |
| yt1b_val | 16 | 0.0% (0/16) | 12.5 / 22.0 / 25 | 37.5% (6/16) | 0.0% (0/16) | 63, 47.6% | 0,  | 9, 0.0% |

VEval: size of the annotated exact `person`/`people` objects (box share of the image; per masklet median in video)
- sav_test: n 21; p05 0.0015, median 0.014, p95 0.58; median side of the equal-area square at 1008: 121 px
- sav_val: n 7; p05 0.002, median 0.099, p95 0.33; median side of the equal-area square at 1008: 318 px
- smartglasses_test: n 14; p05 0.0032, median 0.026, p95 0.16; median side of the equal-area square at 1008: 163 px
- smartglasses_val: n 29; p05 0.0002, median 0.0026, p95 0.14; median side of the equal-area square at 1008: 51 px
- yt1b_test: n 172; p05 0.00095, median 0.025, p95 0.28; median side of the equal-area square at 1008: 159 px
- yt1b_val: n 221; p05 0.0013, median 0.019, p95 0.22; median side of the equal-area square at 1008: 137 px


### 5. How a specific person among several is handled in video

(The sentences under `VEval: tracked objects` above: clip lengths, several people in one video, the file's own consistency.)


### Controls

#### Gold controls: counts against Meta's README table

| subset | pairs | README pairs | masks a / b / c | README masks | matches |
|---|---|---|---|---|---|
| metaclip | 33393 | 33393 | 20144 / 19991 / 20251 | 20144 | ['a'] |
| sa1b | 13258 | 13258 | 30306 / 30095 / 30021 | 30306 | ['a'] |
| attributes | 9245 | 9245 | 3663 / 3681 / 3715 | 3663 | ['a'] |
| crowded | 20687 | 20687 | 50417 / 50606 / 49683 | 50417 | ['a'] |
| wiki_common | 65502 | 65502 | 6448 / 7761 / 6642 | 6448 | ['a'] |
| fg_food | 13951 | 13951 | 10041 / 9825 / 9838 | 9825 | ['b'] |
| fg_sports_equipment | 12166 | 12166 | 5075 / 4987 / 5355 | 5075 | ['a'] |


## What this establishes

Each line says how it is known.

- **The benchmark phrases are short and plain** (counted, every file). One to three words in 72% to 99.9% of Gold's pairs by subset, 57% to 66% start with an article outside Attributes (which has none), colour words in 0.1% to 25% by subset; Silver's article share runs 45% to 67% outside its two species sources and VEval's 62% to 69%; SA-FARI's asked phrases are 113 distinct animal names with no article.
- **Relations and positions are rare** (counted). Relational words in 0.0% to 2.6% of Gold pairs, 0.3% to 1.8% of Silver's (outside its two species sources), up to 0.4% of VEval's; position words up to 2.4%, 3.5% and 4.3% in the same order.
- **How an attribute is phrased** (counted, Gold Attributes). Two words in 96% of its pairs, an attribute first and a noun second, no article; of the distinct phrases the first word is a colour in 14.5%, a material in 17.9%, a shape in 6.1%, a size in 5.2% and something else (a type, a pattern, a state) in 56.3%. Annotators disagree on whether the phrase applies for 35.8% of the pairs any annotator marked, the highest of the seven subsets.
- **What a crowded scene asks** (counted, Gold Crowded). Longer, article-led, caption-like phrases (about 65% start with an article, 28% have four or more words), 76% negative, and a positive pair has a median of 5 instances, 21% have 16 or more and 7% 32 or more.
- **Negatives are mostly construction** (counted): the share of negatives moves from 42% (Gold SA-1B) to 97% (Wiki-Common) between subsets, far more than between phrase forms inside a subset.
- **Descriptive against plain** (counted, pooled): Gold odds 1.164 (1.089 to 1.233 resampling phrases), Silver 1.093 (1.028 to 1.167), VEval 0.718 (0.642 to 0.805), SA-FARI's interval spans 1. So descriptive phrases are mildly more often negatives in the image sets and less often in video; the direction is not stable across datasets and the size is small next to the subset effect.
- **Size** (counted). Boxes under a thousandth of the image are 5% to 24% of Gold annotations by subset, 38% of BDD100k's, 10% to 16% of VEval's masklet-frames; the objects are not all large.
- **Video objects** (counted, VEval). A masklet is present for a median of 2.5 s (SA-V) to 6 s (SmartGlasses); 34% to 71% leave and return; 2 to 7 masklets per positive pair at the median, up to 33.
- **Picking one person among several** (counted). Annotation rows carry `noun_phrase`, `category_id`, `video_id` and the mask lists; nothing else about identity. In videos with two or more person masklets, all share one phrase in 80% to 81% of cases in SA-V and 97% to 100% in SmartGlasses and YT-1B; positive person pairs with exactly one masklet are 46% to 52% of SA-V's and 0% to 9% of the others, and their phrase is descriptive in a third (4 of 12 in each SA-V split).

## What it does not establish

- **How well SAM 3.1 does on any phrase.** These are annotations, not predictions. Nothing here says a long phrase fails; it says the benchmark mostly asks them as negatives and seldom as positives.
- **That absence from the benchmark means a phrase fails.** Relations, positions and per-person descriptions are untested here, not refuted.
- **Why a negative is a negative.** The files do not say how Meta chose the phrases it asked where nothing is annotated.
- **Anything about a cigarette's detectability.** Eight annotations.
- **Anything for 2D images that carries to our video work beyond the counts above.** Image and video benchmarks are different protocols.
- **The SA-FARI train split beyond the aggregates.** It is read and counted, 870 MB, and used only for its phrase and size statistics.

## Reproducing it

From ComfyUI's environment (numpy; orjson optional), with the four repositories downloaded, `G`, `S`, `V`, `F` their directories:

```
python bench/sam3_dataset_phrases.py run --gold G --silver S --veval V --safari F --out <json>
python bench/sam3_dataset_phrases.py render --json <json>
```

`--skip-train` leaves out SA-FARI's train annotation; `--examples N` prints N example phrases per form to the terminal and writes none. A few minutes, CPU only; the largest file is held in memory while it is counted.

## Files

- `2026-10-07_sam3_benchmark_phrases.json`: every aggregate behind the tables, one key per dataset and subset; counts, shares, quantiles, odds ratios, the modifier-word count and the controls. No phrase.
- The tool is `bench/sam3_dataset_phrases.py`; its docstring says the datasets are gated and not in the repository.
