# The vocal stem in the masked lane: where a voice is, by frame

**The lane's one audio model is torchaudio's Hybrid Demucs, which splits a clip's track into a vocals stem and the rest;
`bench/voice_spans.py` (since 2026-10-10) turns the stem into a level and a voiced or unvoiced mark for every video
frame.** It says where a voice is, not whose: who sings on a shot is plan data, a person's word
(`data/<video>/plan.json`, `subjects.<s>.voice`; [`data/AGENTS.md`](../../data/AGENTS.md)). Taking one person's voice
out of the mix by naming or pointing at them is SAM-Audio's designed case, and SAM-Audio is loaded by nothing here
([`perception_trials.md`](perception_trials.md), "SAM-Audio"). The overview of every perception model is
[`perception_models.md`](perception_models.md).

## How it is wired

- **The tool.** `bench/voice_spans.py separate` runs Hybrid Demucs from weights already in the torch hub cache
  (`voice_spans.WEIGHTS`; it never fetches); `spans` writes `voice_per_frame.csv` and `voice_spans.json`, a frame voiced
  above `voice_spans.THRESHOLD_DB` with short gaps closed (`voice_spans.JOIN`); `cut` gives a window's own stretch of
  the stem. Its check is `bench/check_voice_spans.py`.
- **What reads it.** `bench/load_text.py` (a load's text leans on the voiced spans), the capture's preflight and its
  `mouth` mode (`bench/capture_masked_run.py`, `--voice-spans`, `--voice`: a text that names a voice over frames with
  none, a mouth open with no voice), `bench/clip_signals.py` (phrases and dips on the same scale), and
  `bench/copy_sound.py`.
- **What it decides downstream.** A text says a person sings only where the plan says so; whether a head reference's
  mouth is drawn on a window follows the plan's voice there ([`sapiens2_pose.md`](sapiens2_pose.md), "The head
  reference: the face points drawn on the body model"); a mouth open with no voice is
  [`state_signals.md`](state_signals.md), "A mouth open with no voice".

## Limits

- **Its control is printed by `spans` and read first**: the longest unvoiced stretch with the rest stem's level beside
  it. On the clip it was written on, the first 24 seconds held the band with the vocals stem silent.
- One vocals stem: a second voice under the lead, and which person on screen is heard, are not in it. Words are not in
  it either: two models found on the machine were tried as transcribers for singing and neither worked (the tool's
  docstring).
- The level is the two channels' mean, which reads low where they differ ([`docs/state/next_steps.md`](../state/next_steps.md)).
- A table's row is a video frame, so the sound must be the source's own at the source's rate
  ([`2026-10-13_working_copy_sound_against_its_picture.md`](../../bench/results/2026-10-13_working_copy_sound_against_its_picture.md)).
