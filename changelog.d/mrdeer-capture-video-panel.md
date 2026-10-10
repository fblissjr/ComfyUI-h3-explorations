bump: patch

### Changed

- `bench/capture_masked_run.py video` draws a panel under the stacked rows, read from the capture folder's own tables so the picture and the tables cannot disagree: the frame number large, each subject's mask and part, each run's region against its subject and what it covers of anybody else, the change inside the region since the frame before for source and render, the preflight's flags that are live on that frame in plain words, the motion video the model was shown (`--motion`) or the word none, the reference stills (`--still LABEL=IMAGE`) and the window (`--windows`). One frame pulled as a still explains itself. The form follows two reviews made by session scripts on 2026-10-09 that are gone or tied to one clip; colour by part and contacts between subjects are not in it yet.
