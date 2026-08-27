# PDD diagrams

[`two_pdd_paths.html`](two_pdd_paths.html) — our PDD implementation against
Kijai's, traced end to end through the four shipped PDD graphs. Open it from
disk; it is self-contained apart from web fonts, which have local fallbacks.

Four diagrams: the shared graph spine with the accelerator slot marked, load
time side by side, one sampling step side by side, and the 32-interval grid at
8, 4 and an off-schedule step count. Plus a per-graph table and the four places
the two implementations diverge.

[`three_pdd_implementations.html`](three_pdd_implementations.html) — the same
mechanism in a third pack, `silveroxides/ComfyUI-UtilsCollection`, against ours
and Kijai's. Three diagrams: where each one gets the block index, which
partitions of the 32-interval grid each will build, and who owns the fused head
tensors at runtime. Plus a guard matrix, which is where the three actually
separate — that pack fails closed on several things ours only warns about, and
on one row (a partial patch-key match) it enforces what `docs/h3_pdd.md` lists
under "Enforced by nothing".

**A teaching surface, not a source.** Every number on those pages is owned by
[`docs/h3_pdd.md`](../../h3_pdd.md), and where a page disagrees with it that
document is right. Nothing generates them and no check reads them, so they are
the copies that go stale silently — the same standing hazard `CLAUDE.md` records for
`internal/h3_resolution_explainer.html`.

Both built 2026-08-27, and every third-party claim on them describes that date,
because all three subjects were moving: the Kijai converted files changed
head-bank encoding that day, Comfy-Org/ComfyUI#15908 was open with
`comfy/ldm/minimax/model.py` alone in its diff, and the third pack had added its
PDD path the day before (`23ab5f2dd4f6`, read at HEAD `5bac35be3d61`). The
three-way page also describes our own `nfe` widget, which was being removed the
same afternoon; it says so where it matters.

Each is also published as an Artifact. Those copies and these are updated by
hand and have no mechanism keeping them in step.
