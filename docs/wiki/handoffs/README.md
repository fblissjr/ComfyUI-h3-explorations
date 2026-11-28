# Handoffs: notes for the next session

A handoff is a short note from a session that stopped mid-work to whichever
session (any agent, any harness) picks the work up next. This folder replaces
the single `handover.md` (archived as
[`../archive/2026-11-28_handover.md`](../archive/2026-11-28_handover.md)).

- **One file per handoff:** `<YYYY-MM-DD>_<slug>.md`, written by one session.
  Say what state the work is in, what the next step is, and which files,
  branches or renders it depends on. Keep it short; link, don't restate.
- **Consumed means deleted.** The session that takes a handoff up deletes the
  file in the commit that finishes or absorbs the work. Git history is the
  archive; nothing moves to `archive/`.
- **Not a record.** What was learned goes where it always goes: dated
  decisions in [`../decisions.md`](../decisions.md), measurements in
  `bench/results/`, the backlog in [`../next_steps.md`](../next_steps.md).
- **An empty folder (this README only) means nothing is waiting.** List it
  at the start of a session: `ls docs/wiki/handoffs/`.
