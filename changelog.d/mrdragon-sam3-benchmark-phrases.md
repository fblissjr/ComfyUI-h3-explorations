bump: patch

### Added

- **`bench/sam3_dataset_phrases.py`, and the record of its first run**
  (`bench/results/2026-10-07_sam3_benchmark_phrases.md` and its json of
  aggregates): what the annotation files of Meta's four SAM 3 datasets say
  about a valid prompt, counted: phrase shapes, how often each shape is asked
  where nothing is annotated, the size of what is annotated, how people and
  parts of people are asked, and how many instances a pair holds in images
  and in video. Written and run by an independent session. The datasets are
  gated and are not in this repository; the tool reads them from paths, and
  the record holds counts only and quotes no phrase. The record opens with
  five lines to act on for this lane's SAM phrases.
