bump: patch

### Added

- `bench/capture_masked_run.py files` writes `parts_held__<by>.mkv` beside each part mask: the part with the frames the gate doubts (empty on the subject, a jump in size, moved within the subject's box) filled from the undoubted frames either side, the nearer one's shape shifted to where a straight line between the two puts its centre and cut to the subject. `hold=FIRST-LAST+FIRST-LAST` in a `--mask` spec fills only the source frames the caller chose after looking, because a part can be rightly empty; a doubted frame further than `HOLD_REACH` from an undoubted one is left and listed in the manifest. The part node's own box-relative `hold` was tried first on one preview and is not used: `held_parts`'s docstring says what it did. The rules are now in one function, `part_faults`, read by the flags and by the fill; `bench/check_capture_masked_run.py` gains a case for it.
