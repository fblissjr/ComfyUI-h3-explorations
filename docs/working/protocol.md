# How the two leads work together

**In short.** A working day has two leads and the owner. The **operating lead** runs the box: the card, git on `main`,
builds on branches, the ledger and the owner's pages, and its own workers. The **review lead** keeps the method: it
rules when the leads disagree, keeps the Open List and the owner's inbox, clears every landing before it is made, and
writes the changelog, the handover and `docs/state/decisions.md`. They talk only to each other, through two
append-only files a day, and nobody ever waits idle on the other. Where this page and root `AGENTS.md` disagree,
`AGENTS.md` is right until the owner's change is written into it.

The roles are jobs, not models. **Who holds which today** (the owner, 2026-10-17): the operating lead is a Gemini
session and the review lead a Claude session. When one model is not available, one designated lead holds both roles
and asks the owner where it is unclear how to go forward (the owner, 2026-10-18). A lead whose model cannot write the
Open List (today, Gemini) puts its questions to the owner in its terminal on such a day.

## Who does what

| | the operating lead | the review lead |
|---|---|---|
| the card | the only one that starts and stops ComfyUI and queues captures or renders | never touches it |
| git | the only session that runs git on `main`; builds on branches in worktrees | runs no git; sends `COMMIT` for its own files |
| landings | writes `READY` with the branch, its blind read and both ends of the change | clears (`CLEARED`) or refuses with findings, in the order sent |
| the owner | relays what the owner says in its terminal as `OWNER SAID` | the only writer of the H3 Open List and the inbox |
| pages | keeps the ledger, the owner's pages and the plan file current | spot-checks a ledger stamp against `data/<video>/laid.jsonl` and `git log` |
| pictures | reads every frame, still, sheet and clip, and returns text | reads only what comes back as text, and says which reading a ruling rests on |
| writing | its workers' briefs | the changelog, the handover, `docs/state/decisions.md`, and rules on method |

The owner decides. Only the owner can block an item, and only through an Open List row.

## The mailbox

Two files a day under `internal/leads/`, one writer each, append only:

- `<date>_review_to_operate.md`, written only by the review lead;
- `<date>_operate_to_review.md`, written only by the operating lead.

Nothing already written is edited. A wrong entry is corrected by a new entry naming it. One entry per block, with the
time from `date -u`:

```
## R7 | 2026-10-17T14:05Z | ASK | build the one compositing law
what: ...
files: ...
done when: ...
if no answer by 15:00Z: <what the asker does instead>
```

Ids run through the day: `R<n>` from the review lead, `O<n>` from the operating lead. Put tables and readings in their
own files and link them; keep entries short.

| kind | who | what it carries |
|---|---|---|
| `ASK` | either | wants something done or answered; always ends with "if no answer by <time>: <default>", and the default is never "wait" |
| `ANSWER <id>` | either | the answer to an ASK |
| `READY <branch> <commit>` | operating | a build to land, with its note, its blind read and both ends of the change |
| `CLEARED <branch> <commit>` / `REFUSED <branch>` | review | the ruling on a READY, with the findings file when refused |
| `LANDED <commit>` | operating | the landing was committed; the hash read after the commit returned |
| `COMMIT <paths>` | review | the review lead's own files to commit, with a message file under `internal/` |
| `OWNER SAID` | operating | the owner's words in its terminal, verbatim with the time, for the Open List |
| `BLOCKED <Open List row>` | either | an item waits on the owner; the writer moves to its next item |
| `NOW` | either | what the writer is on and what is next: at each stop, and at least every 30 minutes while the operating lead works |

Each lead reads the other's file when it changes (a file watcher) and at least every 30 minutes.

## The rules that keep anyone from waiting

1. **Nobody waits idle on the other.** An ASK carries its own default; past its time the asker does the default and
   says so in a new entry.
2. **Only the owner blocks**, through an Open List row. Everything else has a default.
3. **One owner per item and per file.** An item names who does it; nobody else edits its files. The card and `main`
   are the operating lead's; a branch is its builder's.
4. **No cap on parallel workers** (the owner, 2026-10-17). What a cap was for is the reading: a branch comes to the
   review lead only after its own blind read, and the review lead clears in the order sent.
5. **When the leads disagree**, each writes its figures in one entry and the review lead rules. A call that needs the
   owner becomes an Open List row, and rule 2 applies.
6. **Push back on a premise with the line that breaks it.** When what you checked contradicts the other lead's stated
   premise (a grep that finds an import said not to be there, a reader of the key you are about to write), stop and
   answer with the file and line before building on it. Only that case: it is not a licence to re-argue asks.

## A landing, step by step

1. **The builder** works on its own branch in a worktree outside `custom_nodes` (`h3_worktrees/<builder>-<item>`), cut
   from current `main` and never pushed.
2. **Pre-flight, before `READY`** (the operating lead):
   - `git diff main...<branch>` holds only the intended files;
   - the branch's `bench/check_*.py` are green, and `bench/break_checks.py` output is attached verbatim;
   - a refusal test passes only on the tool's own words with no traceback;
   - a session that did not build it wrote its blind read with no open findings;
   - **both ends of the change** (the owner, 2026-10-17): for any file, key, field, format or folder added or changed,
     every tool that writes it and every tool that reads it, by grep with file:line, each marked updated, unaffected
     (and why) or refusing on purpose (with the refusal's words).
3. **The review lead** re-runs one check from the branch, reads the diff, and for compositing math expands the formula
   at m = 0, between and 1 itself. It clears nothing that lacks both ends. Until it rules, the operating lead carries on
   with its next item and never builds on the unlanded branch.
4. **The operating lead lands it** after `CLEARED`: it takes the branch's files from the merge base (`git checkout
   <branch> -- <path>` for a file `main` has not touched since, `git diff main...<branch> -- <path> | git apply` for one
   it has), runs the branch's checks on `main`, commits by pathspec, reads the hash after the commit returns
   (`git rev-parse --short HEAD`), and writes `LANDED <commit>`.

The git rules are root `AGENTS.md`'s, "The shared tree": `git status` first, `git commit -F <msg> -- <paths>`, never
`git add` except `git add -- <file>` for a new file you made, never `commit -a`, a bare `git commit` or `--amend`.
The owner pushes. Every message is written as already public and says "the owner", never a name.

## The ledger and the owner's pages

After every landing or stopped load, the operating lead makes them new with one command,
`bench/rebuild_delivery.py --video <video>` (`--clip` to assemble a new cut): the ledger, the delivery table, the gap
check, the stacked join, both pages and the close-of-day snapshot of the answers file, stopping at the first refusal.
It writes `NOW: ledger <stamp>` when they are made. [`runbook.md`](runbook.md) has the rest of that routine.

## Workers

Workers are a lead's subagents, or standalone sessions the owner starts. They never write to the other lead's files. A
standalone worker of the operating lead talks to it through `<date>_operate_to_workers.md` (only the lead writes it;
items `W<n>`, each naming its worker and worktree) and `<date>_worker_<name>.md` (only that worker writes it).

## Starting a day (either lead)

1. **Read, in this order, before speaking or running a command:**
   - root `AGENTS.md`, and its operative rules;
   - [`docs/state/handover.md`](../state/handover.md): yesterday's close, what not to assume, today's order;
   - [`working_with_the_owner.md`](working_with_the_owner.md);
   - this page, and `internal/local_env.md` (gitignored: this box's paths for `H3_OUTPUT_DIR`, `H3_INPUT_DIR`,
     `H3_MODEL_STORE` and the ComfyUI venv's python);
   - [`comfy_notes.md`](comfy_notes.md), before the first command;
   - `data/AGENTS.md`, `data/<video>/AGENTS.md` and its `plan.json`, before lane work;
   - today's mailbox file from the other lead, if it has written its first entry.
2. **Check live state:** `date -u +%Y-%m-%dT%H:%M:%SZ`, `git status --short`, `git rev-parse --short HEAD`, and the
   ports `8188` (ComfyUI) and `8765` (the review server).
3. **First entry** (`R1` or `O1`, a `NOW`): what you read, HEAD, `git status`, the ports, and the first item.
