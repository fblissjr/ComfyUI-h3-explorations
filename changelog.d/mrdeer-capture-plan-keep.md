bump: patch

### Added

- A plan in `bench/capture_masked_run.py` takes `keep=LABEL+LABEL`: the tokens those masks touch are taken out of the planned region after the others, the subject's own included, as the Masked Source does. The `kept_pixels_inside_the_part` rule reads a plan's keep as well as a rendered run's, against the mask the plan carries.
