# EmbeddingGemma 2 as a second reading of who a numbered person is: a first run on 25 crops

2026-10-11, session mrcolor. A throwaway first run on one clip and one stretch
of it (source frames 1039 to 1465), run by the session that wrote it and judged
by nobody yet. Nothing is built on it. It does not reopen the ruling of
2026-10-10 that who is in a scene is a person's answer to a numbered picture
(`docs/wiki/decisions.md`): the question here is only whether a likeness could
sit BESIDE that picture as a second reading.

## What it buys

The owner asked whether an image embedding could tell the same person a cut
later, by more than the face, and fetched Google's EmbeddingGemma 2 to try. A
likeness from the tracker's own features failed on this clip on the 10th
(`2026-10-10_who_is_who_across_shots.md`). If a general image embedding ranked
the right person first on the shots that went wrong then, it would be worth
showing beside the numbered picture; if it does not, the road stays closed
with one more model tried. The plan this belongs to is "The trials" in
`docs/wiki/meta_perception_models.md`, the paragraph on a likeness score.

## How it was run

- Labels: the owner's own answers of 2026-10-11 on the numbered pictures of
  nine shots (the local page's saved lines), the latest for each numbered
  person: 25 people on 14 pictures; 12 the lead, 6 the second subject, 5 the
  third subject, 2 nobody we track.
- Crops: each numbered person's box from the preview's shot table, cut from
  the working copy's frame at the canvas size. Two crops a person: the whole
  box, and its top third (head and shoulders, the place the tracker compares).
  All 25 whole crops were looked at on one sheet before anything was embedded:
  each box holds one person. No mask was used: a box takes in the background.
- Model: the checkpoint in the owner's store (`google_embeddinggemma-2`),
  through sentence-transformers, pictures only, on the CPU, in an environment
  of its own (ComfyUI's was not touched). One vector a crop, unit length.
- The test: for each person, the nearest crop by cosine among the crops of
  OTHER shots. Right when that crop is the same person.

## What was written before it ran

That the third subject would be right nearly always; that the lead would be
right inside one outfit and often wrong across her two outfits of this stretch;
that the second subject would be wrong on her small, soft crops at the back.
And the bar: it passes only if no named person's nearest other-shot crop is a
different named person, the hard shots included. Agreement on the easy ones
says nothing.

## What came out

| | whole box | top third |
|---|---|---|
| named people whose nearest other-shot crop is the same person | 17 of 23 | 16 of 23 |
| the third subject | 5 of 5 | 4 of 5 |
| the lead | 10 of 12 | 10 of 12 |
| the second subject | 2 of 6 | 2 of 6 |
| the lead, matched only against shots where she wears the other outfit | 10 of 11 | 10 of 11 |

The misses, whole box:

| crop | who it is | nearest other-shot crop | what the crop shows |
|---|---|---|---|
| shot 1, both pictures | the second subject | the lead; the third subject | her two large crops at the kitchen table; her other four crops are small figures in the background |
| shot 6 | the lead | a crop of nobody we track | a hand and an object; no face |
| shot 9, frame 1423 | the lead | a crop of nobody we track | a small crop, her head mostly hidden |
| shot 9, both pictures | the second subject | a crop of nobody we track | a few dozen pixels at the back |

The scores do not separate. Over every pair of crops from different shots, the
same person scored between 0.69 and 0.96 (whole box) and two different named
people between 0.65 and 0.83: 122 of the 139 different-person pairs score above
the lowest same-person pair. No single line tells them apart; only the ranking
carries anything, and only for large, clear crops.

## What it says, and what it does not

- **By the bar written beforehand it does not pass.** A named person's nearest
  crop is a different named person on the second subject's two largest crops.
- **One expectation was wrong.** The lead was found across her change of
  outfit on ten of eleven crops, where a miss was expected. One clip, two
  outfits, large crops; her hair is the same in both.
- **It fails where the 10th's features failed:** small, soft or partly hidden
  people, and a person whose only large crops are in another kind of shot.
  Shot 9, which the trackers swapped on the 10th, is right for the third
  subject and for one of the lead's two crops, and wrong for the other.
- **Not tried:** crops cut by the mask, crops of one part (hair, clothing), a
  comparison with the features of the models this pack already loads, any
  other stretch or clip, the model's video or audio inputs.
- One reader has not looked: the crops and the full table are with the data.

## Where the data is

`data/2026-10-11_fun_eg2_who_first_crops/` (untracked): the script, the 50
crops, the two score tables and what the run printed.
