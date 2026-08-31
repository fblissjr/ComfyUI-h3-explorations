# Archived results

Records from lanes that are closed. They are kept, not deleted: a closed lane's
measurements are still the evidence for why it closed, and several are cited by
the write-ups that closed it.

**What lands here is decided by the lane, not by the date.** A date cutoff was
considered on 2026-08-31 and rejected on measurement: of the results predating
2026-08-26, most were cited by tracked code or docs -- including checks that
read them at runtime -- while several results from the days after it were cited
by nothing. Age and liveness turned out to be close to orthogonal, so the rule
is:

- A file moves here only as part of a lane that is **closed**, with the closure
  recorded somewhere a reader can check.
- A file **stays** in `bench/results/` if anything outside its own lane's
  write-ups cites it, if a kept record names it as an input, or if its lane is
  still open. Being uncited is not evidence a result is dead -- it can equally
  mean the document that should cite it was never written.
- Citations that pointed at a moved file were rewritten to its new path in the
  same commit. `bench/check_doc_links.py` is what proves that landed.

Deliberately **not** archived, and the reason each time:

- The raw arm records (`*_arms.jsonl`, `*_smoke.jsonl`) behind verdicts that
  `docs/evidence.md` and `docs/SOLATTN.md` still cite. Splitting a verdict from
  its substrate is how a result stops being reproducible.
- The `_sla` and `_v11` variants: `h3_probe_turbo_768p_sla_router` is a live
  graph, so that lane is not closed.
- `2026-08-25_refview_b_qwen2048_int8_occupancy.json` and its query log: the
  reference-view ablation is an open experiment (CLAUDE.md), and this is the
  only occupancy measurement of one of its arms.
- `2026-08-22_h3_marker_tokenization.json`: carries a `corrected_by` key, and
  marker tokenization is live territory that this repo has re-derived more than
  once.

Nothing here is generated, walked by a check, or read at runtime. Reading these
files is fine; citing one as current is not -- each lane's README says what
superseded it.
