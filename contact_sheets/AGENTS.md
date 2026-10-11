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

## Finding and making

    <python> bench/contact_sheets.py list                # which videos have sheets
    <python> bench/contact_sheets.py find fun 1010       # the sheet that holds frame 1010
    <python> bench/contact_sheets.py make VIDEO          # a new video; does nothing if its sheets are there

`make` on a video that already has its sheets writes nothing and says so.
`--again` replaces a set, for a different spacing; there is still one set.

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
