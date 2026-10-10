bump: patch

### Added

- A preflight rule in `bench/capture_masked_run.py`, `two_tracks_on_one_person`: two subjects whose tracked masks are nearly the same mask on a frame, which is one tracker having taken the other's person. On the stretch it was written from it names the 95 frames a second tracker spent on the first one's person, where the masks' overlap reads 0.99 to 1.00; two people touching read under 0.1. A kept-out or keep mask given as a subject is not compared.
- The `look` reading's text carries the routine it is for: read the look on window 1's saved file while window 2 samples, and stop the run if it reads as the original.
