# Every render of the fun clip on the share, and how each was made: the ledger's first record

2026-10-13, session mrolaf, at the lead's relay of the owner's ask for a list of what has been rendered and
which of it can be trusted. Read from files by `bench/render_ledger.py`; no clip was watched for this page and
nothing here says a render is good, or better than another.

## What it buys

After four days the video had clips in a dozen session folders, and neither the sessions nor the owner could
say which renders existed for a shot, which could be made again, or which a delivery stood on. The rule that
followed is that a piece is laid only from its own build record, with the file's frames equal to the record's;
a piece whose making cannot be said is rendered again. This page is the count behind that rule, made by a tool
so it is made again after every render and never typed.

## How it was made

    bench/render_ledger.py --shots data/fun/shots.json --plan data/fun/plan.json \
        --share <the server's output root> --inputs <the server's input root> --records data \
        --source fun_0000_1919_24p.mov --source fun_0000_6394_24p.mov --source fun.mp4=23.976023976 \
        --table data/2026-10-12_fun_delivery_tables/t_fun_0000_1919_joined_2026-10-12.txt \
        --verdicts data/fun/verdicts/verdicts.jsonl --answers data/fun/poc/answers.jsonl \
        --items data/fun/review/items.json --spots data/fun/joined_clip_spots_2026-10-10.json \
        --json data/fun/render_ledger_2026-10-13.json --md data/fun/render_ledger_2026-10-13.md --summary <this table>

The ledger itself is the page `data/fun/render_ledger_2026-10-13.md` with its JSON beside it, untracked: it
holds every render's file, its frames three ways, its making, its form, where the table lays it and the owner's
lines. What is tracked here is the count a shot and a subject, which carries no path of the machine and none of
the owner's words. The tool's docstring says what each of the four states of a making means and all it means.

## The count, a shot and a subject

Made 2026-10-13T13:59:45Z at commit 90e0926e by `bench/render_ledger.py` from the share's output root, the job tool's records, the video's shots and plan files, the copies fun_0000_1919_24p.mov, fun_0000_6394_24p.mov, fun.mp4, a delivery table, the owner's lines. Where each was read from is on the page beside `render_ledger_2026-10-13.json`, which is not tracked.

| frames | subject | renders | recorded, the file agrees | recorded, THE FILE DISAGREES | a build record for its load, WHOSE QUEUE RECORD NAMES NO FILE AND NO OUTCOME | NO BUILD RECORD | laid in the table | with a line of the owner's |
|---|---|---|---|---|---|---|---|---|
| 14-167 | second | 26 | 12 | 0 | 0 | 14 | 1 | 14 |
| 14-167 | subject not known | 2 | 0 | 0 | 0 | 2 | 0 | 0 |
| 168-185 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 168-185 | second | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| 168-185 | subject not known | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 186-277 | second | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| 186-277 | subject not known | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 278-293 | second | 5 | 2 | 0 | 0 | 3 | 1 | 2 |
| 278-293 | lead | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 278-293 | subject not known | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 294-414 | lead | 5 | 0 | 0 | 0 | 5 | 0 | 5 |
| 294-414 | second | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| 294-414 | subject not known | 1 | 0 | 0 | 0 | 1 | 0 | 0 |
| 415-437 | second | 5 | 2 | 0 | 0 | 3 | 1 | 2 |
| 415-437 | lead | 3 | 0 | 0 | 0 | 3 | 0 | 3 |
| 438-483 | lead | 4 | 0 | 0 | 0 | 4 | 1 | 3 |
| 438-483 | second | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| 484-515 | second | 4 | 2 | 0 | 0 | 2 | 1 | 2 |
| 484-515 | lead | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 516-603 | lead | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 516-603 | second | 2 | 0 | 0 | 0 | 2 | 0 | 1 |
| 604-1009 | lead | 6 | 0 | 0 | 0 | 6 | 1 | 6 |
| 604-1009 | second | 13 | 1 | 2 | 0 | 10 | 2 | 4 |
| 1010-1038 | lead | 3 | 0 | 0 | 0 | 3 | 2 | 3 |
| 1010-1038 | second | 11 | 0 | 0 | 0 | 11 | 0 | 1 |
| 1039-1060 | second | 12 | 4 | 0 | 1 | 7 | 1 | 4 |
| 1039-1060 | lead | 1 | 0 | 0 | 0 | 1 | 1 | 1 |
| 1061-1092 | lead | 2 | 1 | 0 | 1 | 0 | 1 | 1 |
| 1093-1133 | lead | 3 | 2 | 0 | 1 | 0 | 1 | 3 |
| 1093-1133 | second | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1134-1155 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1156-1188 | lead | 4 | 2 | 0 | 2 | 0 | 1 | 4 |
| 1156-1188 | second | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1189-1221 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1222-1305 | lead | 3 | 2 | 0 | 1 | 0 | 1 | 3 |
| 1222-1305 | third | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| 1306-1418 | lead | 2 | 1 | 0 | 1 | 0 | 1 | 1 |
| 1306-1418 | third | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| 1419-1465 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1419-1465 | second | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1419-1465 | third | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1466-1729 | lead | 9 | 3 | 0 | 2 | 4 | 2 | 3 |
| 1466-1729 | second | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1466-1729 | third | 1 | 0 | 0 | 1 | 0 | 1 | 1 |
| 1730-1754 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1755-1801 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1802-1848 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1849-1897 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1898-1919 | lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

A render that covers several shots is counted in each. A subject the shots file says is not in a shot has a
row there when a render of that subject covers its frames; `subject not known` is a render no file names a
subject for.

## What the first reading of it found

Each of these is a pointer into the page; the figures are the table's and the page's.

- **The two renders whose file disagrees with its record are both loads that continued a stored window** on
  the long kitchen shot (`second_hf_0859c`, `second_hf_0859d`). Each record's load starts earlier and is longer
  than the file, and nothing in the record says which source frames the file holds. One of them is laid in the
  delivery table of 2026-10-12.
- **The builder's renders of 2026-10-10 have a build record that names no file and no outcome**: the job tool
  did not write either into its queue record until the next day. They are tied by the output folder and the
  load's name and are a state of their own. One of them holds a frame more than its record's load
  (`lead_1156`, under 1156-1188).
- **Every render with no build record sits in one session's folder** (`mrpop`), written on 2026-10-10 and
  2026-10-11 by the clock in UTC; no record under `data/*_job` names the file or its folder, so how each was
  queued is not in any file. What is known of each is the graph saved in its picture: what was asked for, not
  what ran or from which commit. In many of them the masks, and in some the body model, were made inside the
  render's own graph, so there is no file of either to hold against anything.
- **The delivery table of 2026-10-12 lays pieces from all four states.** The page says which on every piece's
  own line.
- **No record names an input that has gone from disk**: every recorded render's line `its inputs by path`
  finds each file, the capture folder, the windows folder and the still. That is presence, not sameness: a
  path that still exists may have been rewritten.

## What it cannot see

Whether a render is good. Whether a record's inputs are the bytes that ran. Which source frames a file holds
when its count differs from its record. A render whose picture file is gone: the numbered videos with no
saved graph and no record are counted on the page and listed in the JSON, tied to no video. Who made a render,
beyond the folder it sits in. Whether the shots file's cuts are right.

## What is not placed

The page's last section: build records with no render on the share (loads of the 10th whose output folder
holds no file of their name, and `second_hf_0604` of the 12th, whose run ended in error), and the rows of the owner's verdicts that are tied to no render's file. Those
rows are of stacked or joined viewing copies, whose page items name no render, and of spots on the joined
clip of the 10th that the page's own list gives no frames for. The spots that do have frames are printed
under the renders that clip's table laid there, as said of the joined clip and not of the render alone.
