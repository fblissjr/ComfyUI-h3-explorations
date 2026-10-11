# vendor/meta_sapiens2: Meta's names for the Sapiens2 models' points and classes

last updated: 2026-10-11

The Sapiens2 checkpoints do not name what they output: the pose model's config calls its points `LABEL_0`
on, and the segmentation model's config does the same for its classes. The names are in Meta's code. This
folder holds the two lists this pack reads, copied from that code
([facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2); each list records the file and
the commit it was read at).

These files are Meta's, under Meta's licence in [`LICENSE.md`](LICENSE.md).

## What is here

| path | whose | what |
|---|---|---|
| `LICENSE.md` | Meta's | the Sapiens2 License, byte for byte as upstream ships it |
| `pose_keypoints308.json` | Meta's | the pose model's point names in the checkpoint's order, and each name's left-right partner |
| `seg_classes29.json` | Meta's | the segmentation model's class names and preview colours, by index |
| `README.md` | ours | |

## How it is made and what holds it

[`bench/build_meta_sapiens2.py`](../../bench/build_meta_sapiens2.py) writes the three files of Meta's from a
checkout of upstream. It reads Meta's files as text and runs none of them; its docstring gives the rule for
each list. Nothing in the folder is edited by hand.

[`bench/check_meta_sapiens2.py`](../../bench/check_meta_sapiens2.py) holds the folder: the licence file to its
hash and to upstream's; each list to what the builder reads from upstream, where that checkout is present;
the pose list's left-right pairs to the pose checkpoint's own, where the model is present; and the class
list to the Subject Parts node's table.

## Who reads it

- [`bench/pose_along_shot.py`](../../bench/pose_along_shot.py) takes the points' names from
  `pose_keypoints308.json` for a model folder that does not name its points and whose own left-right pairs
  are this list's.
- `CLASS_NAMES` and `PALETTE` in [`sapiens2_parts.py`](../../sapiens2_parts.py) are the table in
  `seg_classes29.json`; the check holds the two to each other.
