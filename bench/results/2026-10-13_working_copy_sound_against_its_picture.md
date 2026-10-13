# A working copy's sound against its picture: the copy made without the stretch, and the one that replaces it

2026-10-13, session mrolaf. Found while making the voice tables for the whole video and comparing them, as the
lead asked, with the tables of its first 1920 frames: the new spans came out a frame later than the old, and
two frames later further on. Nothing here was heard or watched; every figure is `bench/copy_sound.py`'s.

## What it buys

A working copy keeps every frame of the source and re-stamps them from 24000/1001 a second to 24, so that a
frame number is a time. The picture then runs a thousandth fast, and the sound has to be stretched by the same
share. The copy of the whole video made that morning was not: its sound is the source's at the source's speed.
Its picture had been held to the source's frame by frame and found equal, and that check is the only one a
copy had, so the copy stood as checked. A masked render freezes the copy's sound and is sampled hearing it; a
load from that copy would have heard the song behind the mouths it was drawing, by more the later in the video
it was. No plan and no render had been built from it when this was found; the previews running on it read no
sound.

## What was measured

At ten places along each copy, a window of four seconds of the copy's sound set against the source's sound at
the time the copy's picture there comes from. Positive is the sound after the picture; in frames of the copy.

| the copy | made | first place | last place | furthest that fits |
|---|---|---|---|---|
| `fun_0000_1919_24p.mov`, frames 0-1919 | 2026-10-10, sound stretched by `atempo` 1.001 | -0.11 at 4 s | +0.12 at 72 s | +0.25 |
| `fun_1920_6395_24p.mov`, frames 1920-6394 | 2026-10-12, no stretch | +0.10 at 4 s | +4.28 at 178 s | +4.28 |
| `fun_0000_6394_24p.mov`, the whole video | 2026-10-13 morning, no stretch | +0.10 at 4 s | +6.20 at 258 s | +6.20 |
| `fun_0000_6394_24p_sound_in_step.mov`, the whole video | 2026-10-13, that copy's picture stream copied, sound stretched by `atempo` 1.001 | -0.11 at 4 s | +0.15 at 258 s | +0.17 |

The two with no stretch read, at every place, the figure the tool prints for "a copy with no stretch": a
thousandth of the elapsed time, to a tenth of a millisecond. The two stretched copies wander by a few
milliseconds either way and do not drift; at 24 places the new copy stays between -0.11 and +0.33 frames.

How well the windows fit, which the tool prints beside each place: 0.91 to 1.00 on the two copies with no
stretch; 0.35 to 1.00 on the new copy and 0.56 to 1.00 on the first, because `atempo` keeps the pitch by
dropping slivers of sound, and a window with a sliver gone fits its source less well. One place of the new copy
fits under the tool's line of 0.5 and is printed and not counted.

## What changed

- `fun_0000_6394_24p_sound_in_step.mov` is the working copy for anything that hears the sound. Its picture's
  packets are the earlier whole copy's, packet for packet (every stamp, size and md5 equal), so that copy's
  frame-by-frame check against the source and every mask made on it stand. Its record is beside it.
- The two copies with no stretch are not changed or removed; each record beside them has a dated line.
- `bench/copy_sound.py` is the tool, with `bench/check_copy_sound.py`. A copy is made, then read by it.
- The voice tables are made from the source's own sound at the source's frame rate, not from a copy's.

## What it cannot see

Whether the stretched sound is right to an ear: nobody has listened to the new copy. A dropout between two
places. Which of two files is wrong when they disagree: the source's picture and sound are taken to be in step.
The first form of the tool set the copy's window against the source's as it is and nothing else; on made sound
with no bass a copy in step then fits at a tenth and reads as unreadable, so it now tries the source's window
at the picture's speed as well and keeps the better fit. Which of the two fitted is printed and is not a
verdict: a stretch by resampling fits the second, `atempo`'s and no stretch at all fit the first.
