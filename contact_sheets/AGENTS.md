# contact_sheets: what is on a frame of an input video, already made

Look here before you sample a video. For every input video this repo has
worked on, its own raw frames are laid out as contact sheets, each tile
with its frame number on it and nothing drawn over it. They were made once
and they are kept. (Owner, 2026-10-10.)

## The rule

- **A question that a frame number and the raw frame can answer is answered
  by opening a sheet.** Who is on screen around frame 1010, where a cut
  falls, what a subject is holding, which way somebody faces: open the
  sheet that holds the frame.
- **Do not make them again, and do not sample the video again into a
  scratch folder to look at a few stills.** That is what every session used
  to do, each at its own spacing, and it is what this folder replaces.
- **Make something else only when a raw frame at this size cannot answer
  it:** a closer look at one frame at full size, a mask or an outline on
  the frame, a render beside its source, anything measured. Those have
  their own tools (`bench/frame_sheet.py`, `bench/capture_masked_run.py`).
- **When in doubt, look.** None of this is a reason to look less; it is a
  reason not to do the same work twice.
- **Nowhere else.** Sheets of an input video are made by
  `bench/contact_sheets.py`, into this folder, and the tool takes no other
  place to write. Not a scratch folder, not `internal/`, not `data/`, not
  the media share.

**Why one set.** So that every session reads the same frames by the same
numbers. Six stills sampled by one session at its own spacing cannot be
checked by the next one, and two sets of sheets are two answers to "what is
on frame N".

## What is here

    contact_sheets/
      AGENTS.md                      this file, the only thing here git tracks
      <video>/                       ONE folder a video, named by its file name
        <video>_<first>_<last>.jpg   a sheet: the frames from <first> to <last>
        sheets.json                  the file's identity, the spacing, the list of sheets

- **One folder a video, named exactly by the video's file name without its
  extension.** `fun.mp4` is `fun/`. There is never a second folder for the
  same video, and the tool refuses a different file that has the same name.
- **A sheet's name is its first and last frame.** `fun_0960_1199.jpg` holds
  frames 960 to 1199. Frame numbers are the input file's own, counted from
  0, and they are the numbers on the tiles.
- **A tile every few frames, not every frame.** `sheets.json` says how
  many; any frame is within a few frames of a tile. The exact frame, when it
  matters, is a closer look and its own tool.

## Can the numbers be trusted

A frame number on a tile is proved when the sheets are made, not assumed:

- **No frame rate is assumed.** The tool reads every frame's own timestamp
  from the file. A number is the frame's place in the file's order, counted
  from 0, every frame. Each sheet's header gives the time span it covers and
  says so in capitals when the frame times are not evenly spaced.
- **The written sheets are read back.** At the tiles that differ most from
  the tile before them (the cuts), the frame is read a second way, by its
  timestamp, and set against the tile. A tile must be its own frame and not
  its neighbour's. `sheets.json` holds what was read, under `check`.
- **A set that cannot be shown right is not left on disk.** If the file
  gives no timestamps, the count of frames read is not the count the file
  reports, or a tile does not match its frame, the tool removes what it
  wrote and says why. Do not then make sheets another way and trust their
  numbers: tell the owner.
- **If you doubt a set that is here:** `bench/contact_sheets.py verify VIDEO`
  runs the same reading again and writes nothing.

What the numbers are not: the numbers of a copy made at another rate, or of
a loader that forces a rate as it loads (that drops or repeats frames).

## Finding and making

    <python> bench/contact_sheets.py list                # which videos have sheets
    <python> bench/contact_sheets.py find fun 1010       # the sheet that holds frame 1010
    <python> bench/contact_sheets.py make VIDEO          # a new video; does nothing if its sheets are there
    <python> bench/contact_sheets.py verify VIDEO        # is the set here this file's, and numbered right

`make` on a video that already has its sheets writes nothing, and says why
and what to do instead. `--again` replaces a set, for a different spacing;
there is still one set.

One fact here is also written elsewhere: `sheets.json` holds the video
file's identity, and so does the video's analysis folder under `data/`.
Both come from the same function (`clip_folder.identity`); it is here only
so that a different file of the same name is refused.

## What these are not

- **Not who is who.** A sheet shows people; it does not say which of them a
  job calls the lead. That is a person's answer, kept in the cast file
  (`bench/cast_file.py`), which holds only what a person said.
- **Not a mask and not a render.** Nothing a tracker or a model made is on
  them.
- **Not for another copy's numbering.** A working copy that keeps every
  frame has the same frame numbers as its source (the video's analysis
  folder under `data/` says so for a copy); a render's frame k is source
  frame `first + k`.

## Never tracked, never uploaded

The pictures are frames of the owner's clips. Git ignores everything in
this folder except this file, and nothing here goes to a hosted service or
onto the board.
