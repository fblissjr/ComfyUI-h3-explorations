bump: patch

### Added

- **`bench/results/2026-10-07_masked_switch_keep_prompt_verdicts.md`: the
  owner's verdicts on four ideas rendered once each on the upper-body
  recipe, and what was measured after.** A region turned on partway through
  a window shows the new subject late, by different amounts at two moments
  and the same on and off the frame grouping; a patch of the source kept
  inside the region keeps the held thing and brings the original person
  back around it; a shadows sentence and a bare-noun subject change nothing
  the owner can see; the prompt rewritten in the vendor guide's form is
  better on two seeds, with one fault at a cut on one. `masked_prompt_text.py`
  is not changed yet.
- **`bench/masked_render_against_source.py`**: two measurements of a masked
  render against its source. `flicker` sizes the frame-to-frame change of
  the residual in bands either side of the region's seam, read from the
  render's own overlay; `landing` finds the frame at which a late region
  takes effect against a control arm. Its docstring says what neither shows.
