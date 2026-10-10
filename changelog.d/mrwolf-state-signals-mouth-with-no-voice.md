bump: patch

### Changed

- `docs/wiki/state_signals.md` gains an entry, "A mouth open with no voice", written before the case is measured: a face pass on a shot with no voice has no channel for an open mouth (the audio is silent, the mesh has no mouth, a face pass wires no motion video). It says which frames to take by measurement (the source's mouth opening high and the voice table unvoiced), and what to test first and why: the source's own head, zoomed and blurred, to the video model, at a blur of about a sixteenth of the head's width, with an eighth as the control that should lose the mouth; then a mouth map drawn from the part model's classes if the blur brings the face back. All reasoned; nothing rendered.
