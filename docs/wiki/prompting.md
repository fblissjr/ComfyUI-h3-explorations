# Prompting H3: where the rules come from, and what checks them

last updated: 2026-09-03

This page is a map. It does not state any prompting rule of its own. It tells
you where each rule comes from, how much it binds, where the prompt text lives,
and which check catches a mistake. When it disagrees with one of the files it
points at, that file is right.

## If you are an agent, start with the skill

`.claude/skills/h3-prompt` routes by task: writing a prompt, editing a shipped
one, converting between modes, judging whether one is good. For each task it
names the file that owns the answer and the command that verifies the result.
It repeats no rules, so it cannot become a second authority.

Use the skill when you need to do something. Use this page when you need to
know where a claim came from.

## The sources, ranked

Only the first one binds.

| rank | source | standing | if you break it |
|---|---|---|---|
| 1 | the vendor's two guides, `vendor_guides/base_en.md` and `ref_en.md` | the only authority | the prompt is off the distribution the model was trained on |
| 2 | the vendor's own API payloads, `coderef/MiniMax-H3/scripts/readme/*.sh` | evidence of what the vendor does, not a rule | you are doing something the vendor's pipeline never does |
| 3 | the vendor's prompt-writing skill, `coderef/MiniMax-H3/.claude/skills/h3-prompt-writing/` | a router that states no rule | nothing |
| 4 | [`docs/prompting.md`](../prompting.md) | our reading of the guides, with every rule labelled by how much it binds | depends on the label the rule carries |
| ~~5~~ | ~~`internal/PROMPTING.md`~~ | deleted 2026-09-01; listed so nobody goes looking for it | |

Sources 1 to 3 and 5 are gitignored and do not ship, which is why they appear
in backticks rather than as links.

Two things not worth re-checking, both established on 2026-09-01: the vendor's
skill adds nothing, because its bundled guide files are byte-identical to ours
and its rules all trace back to guide sentences; and our guide corpus is the
vendor's own text, verified by hash.

## Who owns which question

| question | go to |
|---|---|
| How do I write a prompt, in any mode? | [`docs/prompting.md`](../prompting.md). This is the single source of truth for the rules. |
| Where does prompt text live? Where do I get a conformant prompt to start from? | `prompt_bank/`, one file per prompt plus a manifest. Every prompt a shipped graph renders lives there, along with the house-authored ones. [`docs/prompt_bank.md`](../prompt_bank.md) is its generated table. |
| What is each scene for? | the `tests` line on a bank entry, in the same table. It says what the scene is a test of, for example a distant figure, close-up dialogue, sung vocals, or fast motion. Compare rungs within a scene; do not compare scenes against each other. |
| Which prompts can be rendered at a different length? | the `adapt` column in the bank table. A prompt that names no cut time and no duration can take another frame count from the graph alone. The rest are pinned to the count they were written for. |
| What does this repo actually render? | [`docs/prompt_catalogue.md`](../prompt_catalogue.md), generated from the graphs. It judges nothing. Its `bank id` column links each scene to its bank entry. |
| Is what we render any good? | [`docs/prompt_audit.md`](../prompt_audit.md), hand-written, one verdict per scene. |
| What do the reference labels mean, and how are images sized? | [`docs/h3_references.md`](../h3_references.md) |
| Which canvases and lengths are legal? | [`docs/h3_resolutions.md`](../h3_resolutions.md) and [`docs/h3_geometry_and_nodes.md`](../h3_geometry_and_nodes.md) |
| How do I compare two prompts fairly? | [`docs/eval_comparison.md`](../eval_comparison.md), and read the warning near the end of this page first |

## How the pieces connect, and what checks each link

| from | to | how it is kept true |
|---|---|---|
| the vendor guides | the templates and camera vocabulary the graders use | parsed from the guide files when `preflight_graph.py`, `check_prompt_guide_conformance.py` and `check_camera_vocabulary.py` load. A missing guide is a loud error, never a silent pass. |
| `prompt_bank/` | `workflows/build_workflows.py` | since 2026-09-03 the generator loads every shipped prompt from the bank by id, through `workflows/prompts.py`, and refuses to build if an entry is missing. The two keyframe defaults are the exception: their first line carries the clip duration, so the generator re-times the bank text at the graph's length and checks that it matches at the declared one. |
| `prompt_bank/` | `docs/prompt_bank.md` | generated. `build_prompt_bank.py --check` regrades every entry with the same function `grade_prompt_text.py` uses, works out which graphs ship each entry from the graphs themselves, and reports rather than fails an entry whose `recorded_findings` names the audit verdict that already covers it. |
| `build_workflows.py` | `workflows/*.json` | generated. Graphs are never edited by hand, and nothing is true of a graph until it is rebuilt. |
| the graphs | `docs/prompt_catalogue.md` | generated. `build_prompt_catalogue.py --check`. |
| catalogue scenes | `docs/prompt_audit.md` verdicts | hand-written. `check_prompt_docs_sync.py` fails if a scene has no verdict. |
| the guides | the vocabularies quoted in `docs/prompting.md` | retyped for readers, checked instance by instance. |
| the worked examples in `docs/prompting.md` | still grade clean | re-run through `preflight_graph.grade`. |
| the worked examples | the portable standard | checked word for word. |
| `CLAUDE.md` | `docs/wiki/index.md` | generated. `build_wiki_index.py --check`. |
| dated snapshots | frozen | pinned by hash and never graded against current sources. |
| shipped prompts | vendor practice | `bench/diff_prompt_corpus.py`, a report rather than a gate. |

Nothing checks the prose of the manual itself: the rules, their labels, the
reasoning. No tool can. What catches a mistake there is someone working the
answer out again from the guides, not someone reading it a second time.

**Two commands to run before trusting a generated table**, because both have
been stale before:

    python bench/build_prompt_catalogue.py --check
    python bench/build_prompt_bank.py --check

**Experiments name prompts by bank id.** `bench/run_graph_arms.py` accepts
`@bank:<id>` as a widget value, so a manifest never carries a second copy of the
text. Every render row and every capture manifest records the bank id, the
text's hash, the length, the canvas and the seed, using
`workflows/prompts.py::describe`. A render you cannot trace to a bank entry is
a render you cannot compare with anything.

## Worked examples

`docs/prompting.md` §10 has worked examples for every mode, and every one of
them is graded. Shipped prompts are more examples, but they are what we render,
not necessarily what is good; read them alongside the audit's verdict.

| mode | worked examples | shipped |
|---|---|---|
| T2VA | §10.1 | see the catalogue |
| I2VA | §10.2.1 to §10.2.5 | see the catalogue |
| FL2VA | §10.3.1 to §10.3.5 | see the catalogue |
| L2VA | §10.4.1 to §10.4.5 | see the catalogue |
| Ref2VA | §10.5 | see the catalogue |

The shipped column is a pointer on purpose. How many scenes exist per mode is a
property of the graphs, and a count written here would go stale with nothing to
catch it. `bench/grade_prompt_text.py --list-donors` names the canonical graph
per mode; the catalogue lists all of them; the bank table lists every entry per
mode whether or not a graph ships it.

The keyframe modes are where extra examples were most needed. Each shipped only
one prompt, so a reader had one specimen and no sense of the range. The four
extra examples per mode vary shot count, dialogue, register and camera, and
§10.3.3, §10.3.5, §10.4.3 and §10.4.5 are the only multi-shot keyframe prompts
anywhere in this repo.

## Handing the standard to someone outside this repo

The portable standard is published at
[https://claude.ai/code/artifact/daa73be5-0bb7-4d02-8456-5dc107acea54](https://claude.ai/code/artifact/daa73be5-0bb7-4d02-8456-5dc107acea54),
with the source at
[`docs/portable/h3_prompt_standard.html`](../portable/h3_prompt_standard.html).
Republishing the same file keeps the same URL, so the link can be handed out
once.

It is self-contained: the rules with their labels, the structure per mode, the
practical answers about speaker IDs and markers, two graded examples, and nine
failure patterns with the reason each is wrong. It cites no repository path, so
it survives being pasted into a ticket or read by an agent in another repo. It
is derived from `docs/prompting.md` and says so; where they disagree, the manual
is newer.

Because it is a second copy of rules owned elsewhere, it is checked rather than
trusted. `bench/check_prompt_docs_sync.py` compares every quotation on it with
the source that owns it: the three first-line templates against the constants
parsed from the guide, the camera vocabulary against the guide's table, every
worked example against §10 word for word, and each quoted guide sentence against
the guide. Run it after editing either file. A green run means the quotations
still match; the prose around them is not checked.

There is a second portable file,
[`docs/portable/h3_system_prompt.md`](../portable/h3_system_prompt.md), a system
prompt for a writer model with a core block and one block per mode. Two
portable files can disagree with each other. It is not generated from the
standard, which would be the real fix and is not built; instead its three
first-line strings come from the guide-parsed constants and the same check
grades them. It is a draft and its header says so. Outputs written to its rules
pass the grader's stated mechanical rules, and nothing more: a deliberate
control that broke a rule the guide only shows by example stayed green, which is
correct, because the grader refuses to enforce shown rules.

Dated snapshots live beside it, for example
`docs/portable/2026-09-01_h3_prompt_standard.html`: a frozen record of what was
shared on that date, with a banner saying so. A snapshot is never graded
against current sources, since it falls behind the manual by design. It is
checked only for being unmodified, against a hash in
`docs/portable/snapshots.json`; a snapshot-shaped file nothing records also
fails, because it would look authoritative with nothing pinning it. This
caught a real drift once: the T2VA example had been transcribed with
`non_diegetic_music: N/A` where the manual carries a real cue, and seven camera
motion names had been abbreviated.

## Grade a draft before you render it

    python bench/grade_prompt_text.py --mode fl2va --length 345 draft.txt

This wraps loose prompt text in a shipped graph of that mode and runs the same
grader the graphs go through. It adds no rules. It exits nonzero on a FAIL.

Two things that otherwise waste an hour:

- A prompt is conformant at one duration. The clip length in the first line and
  every cut time are checked against the snapped frame count, so the same text
  is right at 345 frames and wrong at 192. Always pass `--length`.
- For Ref2VA, pass `--like` with a graph that wires the references your prompt
  declares. Labels are graded against that graph's sockets, so a prompt naming
  two pictures and an audio clip fails against a one-picture donor. That is a
  property of the donor, not the prompt, and the tool says so when every
  failure has that shape.

For a prompt already in a graph, run `bench/preflight_graph.py <graph.json>`
directly. It reports and never refuses.

## What is known to go wrong

Each of these is owned elsewhere; this is the index.

| what goes wrong | where it is explained |
|---|---|
| The model renders what is described, not what is named. Expand every name into a visual noun phrase once and reuse it word for word. | `prompting.md` §15.3 |
| Nothing puts the duration into your prompt, so a writer picks cut times by feel and overruns the clip. | `prompting.md` §15.2, §3.3 |
| The speech budget's shape is unsettled and may be nearly twice too tight on short shots. Do not cite it as measured. | `prompting.md` §15.4 |
| On-screen text must be typed literally, or the model draws the texture of English. | `prompting.md` §6 |
| `(S4,S5)` means both speak in unison, not "and also this character". | `prompting.md` §15.3 |
| There is no negative prompt; guidance is distilled in. | `prompting.md` §15.1 |
| Camera-motion vocabulary is checked by `bench/check_camera_vocabulary.py`: amplitude and speed fail the check, a motion outside the guide's table only warns. This page said "enforced by nothing" until 2026-09-01. | `prompting.md` §11, §13 |
| Mood words in the music line, such as "lonely" or "bittersweet", break a guide sentence nothing checks. A review on 2026-09-03 found them across a dozen bank entries. | `prompt_audit.md`, the corpus-wide row |

**The trap that outranks all of these: a rendered clip cannot A/B a prompt
change.** Two arms that differ in any way produce different samples, not a
better and a worse version of one sample; they diverge from the first frame.
"Which clip looks better" is a draw from a distribution. A perceptual claim
needs many seeds per arm, judged blind, in aggregate. See
[`docs/eval_comparison.md`](../eval_comparison.md). One clip per arm cannot
support it however carefully the pair is presented.

## Open, closed, and withdrawn questions

`prompting.md` §14.3 owns these; `bench/diff_prompt_corpus.py` re-derives them.

- **Shot line breaks: closed.** The vendor's prompts and a third-party corpus
  both run multi-shot prompts on one line, with no counterexample. Ours were
  split; they were conformed on 2026-09-01. Clips of those scenes from before
  that date are not matched-seed comparable with clips after it.
- **Mouth closing: withdrawn.** The rule is about position: cue the mouth
  closing when the shot continues, never when the line ends the shot. We
  already followed it. The earlier finding was a per-line ratio, which was the
  wrong statistic. The positional pattern is a third-party observation, not
  vendor practice; the vendor's corpus does not confirm it.
- **Turns per shot: open, base format only.** The vendor's base examples use
  one dialogue block per shot and our stairwell dialogue scene does not. In
  reference format the vendor itself puts two turns in one shot, so our
  reference dialogue scenes are within vendor practice. No guide states a cap,
  and nothing has been rendered to settle it.
- **Two house habits, noticed, not defects.** We write `N/A` for the music line
  and a one-sentence soundscape far more often than the vendor does. Both are
  legal. Neither was a decision anyone made.

How these are found: `bench/diff_prompt_corpus.py` makes two passes. The first
reports every feature where the vendor never varies and we do. The second
reports features where both corpora vary but our rate sits far from theirs,
which is a house habit rather than a divergence. The second pass exists because
the first cannot see a skew: it suppresses any feature the vendor varies at all,
which is correct, and which is why our `N/A` rate went uncounted for several
runs. Neither `preflight_graph.py` nor `check_prompt_guide_conformance.py`
could have found any of these, since one encodes stated rules and the other
refuses to assert what the guide does not state.

Its limit is real: the feature list is written by hand, so it finds only the
divergences someone thought to encode. Three of its own statistics were wrong
before they were right. Read a clean report as "nothing on the axes we thought
of".

The distinction that does the work throughout: a sentence the guide states is a
rule; an example the guide shows is not. Two rules were invented here by reading
examples as rules, and both were retracted. `docs/prompting.md` marks every
guide rule as stated or shown for exactly that reason.
