# PDD diagrams

[`two_pdd_paths.html`](two_pdd_paths.html) — our PDD implementation against
Kijai's, traced end to end through the four shipped PDD graphs. Open it from
disk; it is self-contained apart from web fonts, which have local fallbacks.

Four diagrams: the shared graph spine with the accelerator slot marked, load
time side by side, one sampling step side by side, and the 32-interval grid at
8, 4 and an off-schedule step count. Plus a per-graph table and the four places
the two implementations diverge.

**A teaching surface, not a source.** Every number on that page is owned by
[`docs/h3_pdd.md`](../../h3_pdd.md), and where the two disagree that document is
right. Nothing generates this page and no check reads it, so it is the copy that
goes stale silently — the same standing hazard `CLAUDE.md` records for
`internal/h3_resolution_explainer.html`.

Built 2026-08-27. Its claims about Comfy-Org/ComfyUI#15908 and about the Kijai
artifacts describe their state on that date, and both were moving: his converted
files changed head-bank encoding that day, and the PR was open with
`comfy/ldm/minimax/model.py` alone in its diff.

Also published as an Artifact. That copy and this one are updated by hand and
have no mechanism keeping them in step.
