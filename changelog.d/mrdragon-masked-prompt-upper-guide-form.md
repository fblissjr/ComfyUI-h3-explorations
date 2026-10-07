bump: minor

### Changed

- **The Masked Prompt node's upper-body text is rewritten in the vendor
  guide's form and says nothing it cannot know.** The owner judged a typed
  rewrite better than the node's text on two seeds of one window
  (`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`), with
  the subject facing away at a cut on one, and asked for it adopted without
  wording that guesses what the subject does. `masked_prompt_text.py`'s
  `GIVES_UPPER` role now describes the finished scene, says the user's words
  for the subject in the definition, the retention line and the shot, and
  gives the voice one sentence; it is under half its former length. It was a
  replacement described against "that person", with the whole-person role's
  lip and body choreography.
- **The movement sentence every role shares is the relationship alone**:
  `masked_prompt_text.MOVES` no longer lists turning, facing and gesturing.
  The head and whole-person roles are otherwise unchanged.
- `prompt_bank/` copies of the node's text (`ref2va_masked_person_upper_motion`,
  `ref2va_masked_person_motion`, `ref2va_masked_subject_motion`) and
  `docs/prompt_bank.md` are rebuilt from it. No graph changes: the node
  writes the text when it runs. `bench/check_masked_prompt.py` pins the new
  role: no guessed action, the subject's words three times, one sentence for
  the voice. **What the node writes now has not been rendered.**
