# Working with the owner

last updated: 2026-10-17 (moved here from the Claude sessions' private memory at the owner's word, so the Gemini and
Claude sessions read the same preferences)

**How the owner likes to work, for every session of either model.** AGENTS.md holds the rules of the repo; this page
holds the owner's preferences about being asked, shown and told things. Where the two disagree, AGENTS.md is right and
this page is corrected. Each line carries the date it was asked for; the reasons are in `decisions.md` on that date.

## Their time

- **The owner is on US Central.** Records are stamped with `date -u`; anything said to the owner uses their clock.
  Work out their time of day before writing "this morning", "tonight" or "tomorrow"; otherwise say "next" or "later
  today". The day's end is when their working day ends or they say so, not when UTC reaches evening. (2026-10-13)
- **Every time written down is read from the clock or the file**, never estimated or rounded ahead of it. (2026-10-12)
- **Batch what goes to them.** Finished pieces and questions go in batches on one page, with direct links in order
  and a line each on what to look for, not one at a time. Nothing waits on their judging between batches; interrupt
  only for something obviously wrong that later work would build on. (2026-10-13)
- **Read what they already answered before asking.** Before any page or question, read what they said:
  `data/<video>/owner_answers.jsonl` (`bench/page_answers_ledger.py` lists what no record cites), the cast file and
  the shots record. Where it answers the question, use it. Where it leaves you uncertain, ask again, and say what the
  earlier answer said and why it does not settle this. Who is in each shot of `fun` is answered for the whole video.
  (2026-10-13, 2026-10-17)
- **Updates at issues and milestones**, not a running commentary.

## Where things go, and what they are called

- **Every question says which page it is on.** A decision is a row with buttons on the H3 Open List (written by the
  Claude lead only); a look at a clip is an item on the local review page. Make the row before mentioning it.
  (2026-10-12)
- **Review links go in one place:** `Video/<video>/TO_REVIEW.html` on the share, beside `PROVENANCE.html`, given as
  the review server's address (`http://<tailnet address>:8765/fun_share/...`), never a file path. Its "Start here"
  list is kept most important first. No review link goes on a hosted page or in a tracked file. (2026-10-13,
  2026-10-15)
- **Name things as they hear them.** Every clip heading, page item and mention leads with the load name the sessions
  use (`second_hfp_1010j`), then the plain description; give the direct link. (2026-10-12)
- **A clip shown to be judged shows the shot as the cut would:** every subject of the shot replaced, the day each
  piece was rendered, what to look at and what to compare it with. Everyone (the owner, the sessions, Gemini) watches
  the same stacked view, `bench/stack_delivery.py`: source, what changed, the cut, and per person what H3 saw.
  (2026-10-13, 2026-10-16)
- **A render is handed over stacked** with frame numbers burned in, never bare; something seen on playback is
  reported with the file and the frame.
- **Pages to answer on** are simple: one static page a question, one picture a shot at full size, a row of plain
  buttons a subject whose labels say exactly what a click means, a words box a row, dark, no empty tiles; a click is
  one line appended to the answers file. `bench/who_page.py` is the form. (2026-10-12)

## Judging

- **Their eye on playback accepts a piece.** A reading of stills lists what to look at, with frames, in neutral words,
  and stops: no verdict on the piece, no hold on a reading of likeness, eyes or mouth. A total gap, or something
  obviously wrong that can be fixed, is flagged; something minor for a frame or two is not. When they say a
  deviation is not worth time, stop. (2026-10-13)
- **Judging is comparative and in free text** (stacked pairs at a matched seed), never a score. A still is reported
  as seen, without a quality word. (2026-08-20)
- **A render must decide something**; check the share for one already made first, and the cheap no-card check before
  the card. (2026-09-15)
- **The owner's account of their own work stands.** Where a session's notes differ from what they say happened, say
  so once with the evidence, then write it their way. (2026-10-12)
- **Old clips are examples, not pieces.** "Not possible" is not an answer when it was done before: say what was done,
  in which form, and what the narrower fault is now. A thing a session cannot solve goes up one level (a worker to
  its lead, a lead to the other lead) before it is ever written as "not possible". (2026-10-13)

## How to listen

- **Cause and symptom often come in one sentence.** Re-read the whole sentence and say it back before building on
  half of it. (2026-08-28)
- **A terse priority remark names a lane**: confirm which before claiming the card. (2026-09-15)
- **A thought in the terminal is not a go.** It goes to the Open List's inbox; a go is a row or a line meant as one.
  (AGENTS.md, "How the owner and the sessions talk")

## Writing

- **American spelling** in anything for the owner or the public. (2026-10-12)
- **For a casual viewer**: plain English, no project terms (mask, margin, window, tracker, prompt), every cause tied
  to something visible in the picture. (2026-10-12)
- **People are labels** (the lead, the second subject, the third subject), never who they are to the owner; the
  owner's passing remarks are not quoted in tracked text or commit messages. (2026-10-12)
- **Relative paths only** in anything tracked, in comments and in commit messages; never a home directory, a local
  mount or the owner's name. (2026-10-17)
- **Never upload their media** (renders, clips, stills, audio) to a hosted service without asking and naming what
  leaves the machine. (2026-10-02)

## Tools and nodes

- **A tool measures and says what it measured on and what it cannot see**; it gives no verdict. Green or red only for
  a hard failure; a colour is read in context before anything is reverted. (2026-10-11, 2026-10-16)
- **Simplest first; no precision on a guess**; a rule easily worked around is friction and goes. (2026-10-10/12)
- **No zero sentinels:** a numeric widget never means a mode by being 0. (2026-09-13)
- **A node is easy to use:** no bookkeeping inputs; every tooltip says what the input does. (2026-10-03)
- **A third party's licence is carried whole**, never quoted, summarised or reworded. (2026-10-11)
- **A new tool's first run is a throwaway**: once, on a small target, output read end to end.
- **"Biggest problems and fixes"** goes in one file, priority-ordered, each item labelled in the owner's terms (root
  cause, targeted fix used broadly, assumption, oversight, great idea, not sure yet). (2026-10-16)
