# Do the clips say their scripted lines? Whisper against the prompt, 2026-09-19

Tool: `bench/measure_dialogue_transcription.py` (CPU; Whisper medium from the
local Hugging Face cache through the ComfyUI venv's `transformers`, nothing
installed or downloaded). Numbers: `bench/results/2026-09-19_dialogue_transcription.json`.
Table:

    python bench/measure_dialogue_transcription.py --summarize bench/results/2026-09-19_dialogue_transcription.json

## What was compared

Every dialogue scene of the dense-against-default panel (noodle bar at 107
frames, diner, market and hardware aisle, dense against their defaults; crowd
and meerkat, whose dense renders predate the panel), the panel's two-seed
decoy (the diner rendered fully dense at two seeds), the two true chain pairs
of `bench/results/2026-09-19_audio_sage_vs_kitchen.md` (meerkat and diner,
kitchen against sage), and the `block49_repro` defaults against their
sage-levers arms. Each clip's scripted lines are read from its own embedded
prompt (`<d>` spans), so a pair of two takes is scored against the same
script.

## What it found, in direction

- **Every dialogue clip says its lines, in every arm.** On every scene with
  English dialogue except meerkat, the transcript matches the script exactly,
  dense and default alike, kitchen and sage alike, and in both seeds of the
  two-seed decoy. The Mandarin noodle bar
  line matches too, once scored by character. On these scenes this check sits
  at its ceiling: it does not separate the arms at all.
- **The meerkat narration runs out of clip, in every arm.** The first line
  is complete; the second stops partway, at the same word in the default, the
  sage chain and the fully dense render. Whisper medium and small stop at the
  same point, while the sound keeps its energy to the end of the clip, so
  what follows is either unintelligible or not speech. Nobody has listened
  for it. Either way it is the same in every arm, which points at the prompt
  (two long narration lines for one clip's length) rather than at attention.
- **The control holds.** Every transcript scored against another scene's
  script reads far worse than against its own, so the score is reading the
  words.

## What this means

For the chain and dense questions, transcription error is not a
discriminating instrument on these scenes: every arm says every line. It
stays worth logging beside a verdict, because it is the one audio check that
needs no reference render and it catches a scene that does not fit its
dialogue, as meerkat shows. It cannot hear accent, timing, emotion or which
character spoke, which is where the owner's "sage sounds more natural" lives.

## Fixed while writing

The first run scored the Mandarin line by whitespace-separated "words", which
turned one missing comma into a total miss on the sage-levers noodle bar.
Chinese, Japanese and Korean text is now scored by character.
