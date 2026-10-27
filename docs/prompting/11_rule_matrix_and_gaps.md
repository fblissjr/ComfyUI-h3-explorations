> **Part of [`docs/prompting.md`](../prompting.md) (Rule Matrix, Guide Disagreements & Known Gaps (§11–§13)).**
> For the core prompt structure (`§1`–`§5.7`) and master router, see [`docs/prompting.md`](../prompting.md).

## 11. Every rule, its layer, and what checks it

`bench/preflight_graph.py` reports and never refuses; `bench/check_*.py` are
red/green. "nothing" means exactly that.

**To grade a prompt that is not in a graph yet** — anything in §10, or a draft —
use `bench/grade_prompt_text.py`, which wraps the text in a shipped graph of the
requested mode and runs `preflight_graph.grade` against it. It adds no rules of
its own, so this table describes it too. It exits nonzero on FAIL only. A prompt
already in a graph goes through `python bench/preflight_graph.py <graph.json>`,
which also prices the sequence, and a bank entry through
`python bench/build_prompt_bank.py --check`.

### What keeps each link true

| link | what holds it |
|---|---|
| guides → the graders' templates and camera vocabulary | parsed from the guide files at load; a missing guide is a loud error |
| `prompt_bank/` → the generator | `workflows/prompts.py` loads every shipped prompt by id and the generator refuses a missing entry; the two keyframe defaults are re-timed at the graph's length and checked at the declared one |
| `prompt_bank/` → `prompt_bank.md` | `build_prompt_bank.py --check` regrades every entry, fails on edge whitespace and on a stale table; a `recorded_findings` entry is reported, not gated |
| generator → `workflows/*.json` | generated; nothing is true of a graph until rebuilt |
| graphs → the graders | `h3_config.PROMPT_INPUTS` is the one list of nodes that carry a prompt, and `check_prompt_guide_conformance.py` fails on a graph or a node that declares a `prompt` input outside it. A song node's template is read as the texts its Prompt Lists expand to (`workflows/prompts.py::carriers`) |
| graphs → `prompt_catalogue.md` | `build_prompt_catalogue.py --check`, and nothing runs it for you: run it before trusting the table (§13) |
| catalogue → `prompt_audit.md` | `check_prompt_docs_sync.py` fails on a scene with no verdict |
| this file → its copies | the same check re-grades every §10 example at the duration its heading names and compares the Part One strings and the camera table with the guide. Its docstring says exactly what is pinned for each portable copy. A prose rule in a copy is pinned only where `RULE_PINS` names its load-bearing sentence, in the manual and both copies: the rules that have drifted once, not the whole standard |
| shipped prompts → vendor practice | `bench/diff_prompt_corpus.py`, a report |
| the prose of this file | nothing |

### The rules

| rule | layer | checked by |
|---|---|---|
| base modes emit three core fields, in order | GUIDE base §2.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| ref2va emits six sections, in order | GUIDE ref §1 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| Part One is the first line, then one blank line | GUIDE base §2.1 | `preflight_graph.py` (position implied by exact-string match) |
| T2VA has no Part One | GUIDE base §2.1 | `preflight_graph.py` |
| the I2VA / FL2VA / L2VA alignment sentence is the mode's own, verbatim, with `N` and `S.SS` resolved | GUIDE base §2.1 | **`preflight_graph.py`**, exact-string against the guide's own text; verified red on a deliberately mode-swapped graph |
| FL2VA uses no angle or square brackets in Part One | GUIDE base §2.1 | `preflight_graph.py` (same exact-string case) |
| `<Picture N>` bracket convention inside the body | GUIDE base §3.1-§3.3, *shown* | `preflight_graph.py` accepts bare `Picture N` only on a two-keyframe graph |
| the prompt names exactly the labels the graph wires | HOUSE (runtime) | `check_ref_prompt_labels.py`, `preflight_graph.py`, both directions |
| label ordinals follow append-chain order; a soundtrack's `<Audio j>` precedes its own `<Video k>`; `<Audio>` is one counter | HOUSE (runtime) | `check_reference_order.py` |
| NO shot header carries a timestamp; a time only splits action inside a shot | OWNER (2026-09-18); departs from GUIDE base §4.2, ref §5.1 | `bench/preflight_graph.py::header_problems` (FAIL), and through it `build_prompt_bank.py --check` and `check_prompt_docs_sync.py`. It fails a body that opens with `At` and a digit in any case, a MM:SS.mmm in a shot's opening sentence outside double quotes, and a header that is not `[Shot N]`; it passes a mid-shot time in a later sentence, a quoted on-screen clock and a time of day. `bench/check_prompt_rule_controls.py` carries each spelling as a case, red-proved |
| mid-shot times strictly increasing | GUIDE base §4.2, applied to the times the house still allows | `preflight_graph.py` |
| mid-shot times fall inside the video duration | GUIDE base §4.2, likewise | `preflight_graph.py` |
| `[Shot N]` header format, sequential numbers | GUIDE ref §5.1, base §4.2, minus their cut time | `bench/preflight_graph.py::header_problems` FAILs a header that is not `[Shot N]` (since 2026-10-01; it used to empty the shot list silently); sequential numbering is checked by nothing |
| the five cut phrasings; dissolve/fade/wipe on request only | GUIDE base §4.2 | nothing |
| camera motion type from the twelve-row table | GUIDE base §4.3 | `check_camera_vocabulary.py`, partially: `vocab_matches_guide` pins the check's own motion list to §4.3 (red/green) and `denied_motion` reports known-bad phrases as WARN and never goes red, so a novel out-of-table phrase is caught by neither. By the owner's ruling (§4) those warnings are informational, not defects. This was the escaped instance: a shipped prompt carried `whip pan`, `tracks left` and `at medium amplitude and moderate speed` and every gate passed it. The check has run since 2026-08-28 and this row said "nothing" until 2026-09-01 |
| amplitude only `with small/large amplitude` | GUIDE base §4.3 | `check_camera_vocabulary.py` (`modifiers_in_set`, red/green over every shipped prompt since 2026-08-28; this row said "nothing" until 2026-09-01) |
| speed only `at slow/fast speed` | GUIDE base §4.3 | `check_camera_vocabulary.py` (`modifiers_in_set`, same case as amplitude) |
| motion written as natural English inside the shot | GUIDE base §4.3 | nothing |
| style and initial composition open `[Shot 1]` (base) | GUIDE base §4.1 | nothing |
| style stated in one or two sentences **before** `[Shot 1]` (ref2va) | GUIDE ref §5.2 | nothing |
| speaker ids `(S1)`, `(S2)`, stable across shots, unconditional | GUIDE base §4.4 | nothing |
| **ref2va: the id is reused at every vocal event** | GUIDE ref §5.4, *stated* | `bench/preflight_graph.py::speaker_id_rules` (FAIL), and through it `build_prompt_bank.py --check` and `grade_prompt_text.py`. It is scoped per shot and deliberately conservative: two speakers alternating inside one shot where only the first is attributed read clean, so the rule is stronger than its check (`docs/wiki/next_steps.md` records the gap). `bench/check_speaker_id_control.py` is its control: red on the bank at `4bd7b429^`, green on the same entries now, green on the soundtrack exemption, and a note and never a FAIL in the base modes |
| **a verbal cue inside a reused soundtrack takes `<Audio N>` and no new `(Sx)`** | GUIDE ref §5.4, *stated* as a prohibition | same rule, as its exemption; `ref2va_soundtrack_fully_copy` is the entry that exercises it |
| the base modes reuse the id per vocal event | **NOT A RULE** in base_en — it states the id and its stability only; ref states the reuse. §12.13 | `speaker_id_rules` reports it as a note and grades nothing |
| non-vocalising characters get no id | GUIDE base §4.4 | nothing |
| a speaker has a stated position in the frame | GUIDE base §4.1 ("subject appearance and position"), ref §5.3 ("position in the frame"), *stated* | nothing — and not mechanizable. An unplaced speaker beside a placed silent one was staged out of frame and lost its line |
| the voice descriptor does not contradict the identity the same sentence establishes | GUIDE base §4.4 for "stable identity"; HOUSE for resolving it by moving the range | nothing |
| every action has a stated agent | HOUSE, observed — neither guide states it | nothing |
| a count is consistent with what the shot shows | HOUSE, observed — neither guide states it | nothing |
| the cast count is stated, and where a test counts people, nobody else appears | HOUSE, observed (§16) — neither guide states it | nothing, and not mechanizable |
| each shot says who is in it, and an exit when someone leaves | HOUSE, observed (§16) — neither guide states it | nothing |
| one geography: places, paths and directions agree across the shots | HOUSE, observed (§16) — neither guide states it | nothing |
| bank text has no whitespace at either end | HOUSE | `build_prompt_bank.py --check` (FAIL); our conditioning nodes strip both ends too (§15.2) |
| "produces no vocal sound" is reserved for a character who never vocalises | HOUSE — the phrase is in neither guide; the category is base §4.4's | nothing |
| compound `(S1,S2)` only for already-numbered speakers | GUIDE base §4.4 | nothing |
| identity established where the speaker first **appears** | GUIDE base §4.4 | **nothing**, and not mechanizable |
| the addressee is named outside `<d>`, and a listener takes no id | GUIDE base §4.4 (*stated*: the slot, and no id for a non-vocaliser); ref §5.4 (*shown*: naming the addressee); HOUSE that it is done whenever a second person is present | nothing |
| `<d>` contains only a language tag and the verbatim words | GUIDE base §4.4 | `preflight_graph.py` checks the tag and that no marker sits inside |
| `<d>` appears only in the main field | GUIDE ref §6 (ref2va); implied base §2.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| `says in an off-screen voiceover`, then lips-remain-closed | GUIDE base §4.4 | nothing |
| `<scenetrans>` in both parts of a line crossing a cut, plus a continuity phrase | GUIDE base §4.4 | nothing — **and the token matches nothing the release declares** [OPEN] |
| `<cutoff>` for speech truncated by the end | GUIDE base §4.4 | nothing — release declares `<\|cutoff\|>`; house writes the piped form [OPEN] |
| lyrics pair wraps `<d>`; caption pair is a sibling; no other nesting; pairs balanced | HOUSE | `preflight_graph.py` |
| no whitespace before `</d>`; no caption padding; markers do not start a line | HOUSE | `preflight_graph.py` (WARN) |
| no full stop before `<\|cutoff\|>` | OPEN — the stated mechanism described a pre-fix tokenizer path | `preflight_graph.py` (WARN) |
| `<\|caption_start\|>` means subtitle | OPEN — undocumented; name is the only evidence; a signage reading was withdrawn | nothing |
| on-screen text typed literally in English double quotes | GUIDE base §4.5 | nothing |
| `overall_soundscape` is 1-4 English sentences in one paragraph | GUIDE base §4.6 | nothing |
| dialogue, singing and diegetic music not repeated in `overall_soundscape` | GUIDE base §4.6 | nothing |
| `overall_soundscape` `N/A` only for requested total silence | GUIDE base §4.6 | nothing |
| `non_diegetic_music` is 1-3 sentences on instrumentation, speed, rhythm, dynamics; no mood words | GUIDE base §4.7 | nothing |
| sequenced/chronological soundscape prose | **NOT A RULE** — example-derived, retracted | n/a |
| physical action sounds excluded from the soundscape | **NOT A RULE** — §4.6 names them as belonging there | n/a |
| label-on-its-own-line layout | OPEN — base *shows* inline, ref_en's own examples disagree with each other | nothing |
| the four label types and their meanings | GUIDE ref §2 | nothing checks the semantics; presence is checked |
| a label keeps one meaning across all six sections | GUIDE ref §2 | nothing |
| identity-only image gets no standalone `<Picture N>` entry | GUIDE ref §2.2 | nothing |
| visible content from a reference video is `<Subject N>`, not `<Video N>` | GUIDE ref §2.3 | nothing |
| an `<Audio N>` bound to a speaker reuses the global `(Sx)` | GUIDE ref §2.4, §5.4 | nothing |
| video and audio ordinals are independent | GUIDE ref §2.5 | `check_reference_order.py` (runtime side) |
| a reference video does not create `<Audio N>` merely by having sound | GUIDE ref §2.5 | nothing |
| `summary` opens with a bracketed task-type prefix | GUIDE ref §3 | `check_prompt_guide_conformance.py` |
| task types come from the six-row table, joined ` + `, no repeats | GUIDE ref §3 | `check_prompt_guide_conformance.py`, parsing the guide's own table |
| `keyframe completion` only where an image is a concrete frame anchor | GUIDE ref §3 | `check_prompt_guide_conformance.py` checks the **graph** wires `MiniMaxH3AddGuide` |
| asset presence alone does not create a task type | GUIDE ref §3 | nothing |
| no new labels introduced in `summary` | GUIDE ref §3 | nothing |
| video-editing summaries open with the fixed sentence | GUIDE ref §3 | nothing |
| one `retention_analysis` line per label | GUIDE ref §4 | `preflight_graph.py` requires a retention line for every defined label |
| visual markers from §4.1's four; audio markers from §4.2's four; sets cross only on `weak_reference` | GUIDE ref §4.1, §4.2 | `check_prompt_guide_conformance.py`, `preflight_graph.py` |
| marker chosen within the role already defined | GUIDE ref §4.2 | **nothing** — a legal marker on the wrong entity is set-legal and backwards |
| added content is not a loss of fidelity | GUIDE ref §4.2 | nothing |
| no `(Sx)` in `retention_analysis` | GUIDE ref §5.4 | `preflight_graph.py` |
| labels inserted in `detailed_description` where their roles apply | GUIDE ref §5.2, §5.3 | `preflight_graph.py` WARNs on an uncited `<Subject N>` |
| `detailed_description` 350-500 words for generation tasks | GUIDE ref §5.2 | `preflight_graph.py` (WARN, base modes correctly exempt) |
| reused source dialogue: verbatim, `[unclear]`, standardised punctuation, terminal `.` `?` `!` | GUIDE ref §5.4, scoped to reuse | nothing |
| closing punctuation on **all** dialogue | OPEN — over-generalises ref §5.4 and collides with base §4.4's verbatim rule | nothing |
| frame grid `17k + 5` at 24 fps; duration honesty | HOUSE (runtime) | `h3_rules.py::snap_length`; preflight uses it for cut-time bounds |
| last cut by `duration - 2.5 s`; speech inside `0.4 s`..`duration - 0.8 s` | HOUSE / third-party | nothing |
| speech budget `2.5 x (shot_seconds - 1.0)` | OPEN — functional form assumed, one anchor, an alternative fits it exactly | nothing |
| turn cap | OPEN — house material states both a cap and no cap | nothing |
| lips close and jaw stops at the end of every line | HOUSE, from a vendor example | nothing |
| explicit "produces no vocal sound" for silent on-screen characters | HOUSE | nothing. *Narrowed 2026-09-20*: reserved for a character who never vocalises in the clip; the row above carries it, and §15.3 item 5 says what to write for someone silent only for now |
| L2VA closed-mouth endpoint rule | HOUSE | nothing |
| "only" as a strict exclusion; wardrobe as preserve/add/replace/modify | HOUSE | nothing |
| cast sheet: same noun phrase verbatim on every mention | HOUSE | nothing |
| name the moves that do not happen to hold a frame | HOUSE / third-party | nothing |
| whole prompt under 7,000 characters | OPEN — no source in either guide | nothing |
| supported-language list | HOUSE — no list in either guide | nothing |

---

## 12. Where the guides are silent, ambiguous, or disagree

Stated so nobody mistakes a house reading for the vendor's.

1. **`<scenetrans>` names no declared token.** base §4.4 asks for it; the release
   declares seven H3 tokens and it is not among them. A guide describing a
   mechanism whose token does not exist. Nothing here has rendered one.
2. **`<cutoff>` versus `<|cutoff|>`.** Both guides print the unpiped spelling and
   the release declares the piped one. House writes the piped form; that is a
   deliberate divergence from guide prose, not an oversight.
3. **Five of the seven declared markers appear in neither guide.**
   `<|cutoff|>`, the lyrics pair and the caption pair are undocumented. Every
   structural rule about them here is house pattern.
4. **`<|caption_start|>` has no documented meaning.** Its name and its position
   in the declared list are the only evidence.
5. **Section-label layout is example-derived and ref_en contradicts itself.**
   §7's complete example puts each label on its own line; §6's snippets are
   inline. base_en is consistently inline and states nothing. Do not enforce.
6. **Neither guide states a whole-output English rule for the base modes.** ref_en
   states it in its preamble. base_en only says "English sentences" of the two
   audio fields and "English double quotation marks" of on-screen text.
7. **Neither guide caps turns, caps characters, or lists supported languages.**
   Any such number is house or third-party.
8. **`retention_analysis` for a cited-only `<Picture N>` is unresolved by the
   text.** ref §4 says one line per reference label; ref §2 says an identity-only
   image gets no separate definition line. The guide's own §7 example resolves it
   in practice — its four `<Picture N>` and two `<Video N>` labels are cited only
   inside `<Subject N>` definitions and receive no retention lines — but that is a
   worked example, not a statement.
9. **base §2.1's L2VA template says "reference picture**s**" for a single
   picture.** Reproduce it as written anyway; it is the string the guide says the
   mode always uses.
10. **The guide never says how a shot's duration relates to a frame count**,
    because frame counts are outside its scope entirely. Every timing number in
    §3.3 and §3.4 is ours.
11. **Neither guide states a line-break rule for the main narrative field.**
    base_en's only multi-shot worked example runs `[Shot 2]` inline in one
    unbroken paragraph, and MiniMax's own reproducible t2va API payload does the
    same -- verified 2026-09-01 by counting escaped newlines inside
    `integrated_multimodal_description` in
    `coderef/MiniMax-H3/scripts/readme/reproducible-768p-t2va-request.sh`, which
    has none. Two independent artifacts, but each is the vendor's *only*
    multi-shot base-mode specimen, and both are **shown**, not stated. Our own
    shipped prompts disagree with it and with each other; §14 records the split.
    Do not enforce it.
12. **Neither guide states a limit on dialogue turns per shot.** Both of
    base_en's dialogue-bearing examples carry exactly one `<d>` block per shot,
    and that is the whole of the vendor evidence -- **shown**, at n=2, with no
    sentence behind it. `internal/PROMPTING.md` §4.3 read it as a rule ("one
    speaking turn per shot") and contradicted itself elsewhere; §14 records where
    our shipped prompts land.
13. **base_en and ref_en disagree on whether the speaker id is repeated.** ref
    §5.4 *states* the reuse — "Reuse the corresponding ID at every actual vocal
    event in `detailed_description`" — and states the soundtrack-cue exemption
    beside it. base_en's §4.4 states that a vocalising subject takes a stable id
    and keeps it across shots, and says nothing about repeating it per line; its
    two dialogue examples show one per line, which is *shown*, not stated. So the
    same prompt text can be conformant as a base prompt and non-conformant as a
    ref2va one. We enforce it where it is stated and report it where it is not
    (§5.1); the bank currently repeats the id in every ref2va entry and in some
    base entries but not all, which is a house inconsistency and not a guide
    violation. **Do not resolve it by reading base_en's examples as a rule** —
    that is how the two retracted rules in §13 were invented. *Added
    2026-09-20.*

## 13. Known gaps in this repo

**Three of the four bullets this section carried until 2026-09-01 had gone
false, and they are recorded here as withdrawn rather than quietly rewritten** —
a reader who remembers the old sentence needs to know it was retracted. Each was
true when written on 2026-08-28 and was overtaken within days, which is this
repo's standard failure: prose stating a fact the code already knows.

### Withdrawn 2026-09-01

Each withdrawn claim is quoted as it stood. **The replacement is a pointer, not
a new count** — that is what made these rot: they cached a fact the graphs
already know, and a corrected count would rot the same way.

- **"L2VA has no shipped prompt, and `scene_prompt()`'s L2VA branch returns
  `[Shot N]` and `S.SS` unsubstituted."** Both halves are false. For what
  L2VA actually ships, read `prompt_catalogue.md`, or ask the graphs:

      python bench/grade_prompt_text.py --list-donors

  The `scene_prompt()` defect was fixed on 2026-08-28, and the function, which
  nothing ever called, was deleted on 2026-10-01 as dead code, so it never
  reached a graph.
- **"No shipped graph carries any marker but `<d>`."** False. The markers each
  scene carries are a column in `prompt_catalogue.md`, derived from the graphs.
- **"Every generated ref2va prompt is one shot at 42-68 words."** False as a
  universal. For the current distribution, over every graph
  `h3_config.graph_paths` yields (a literal `workflows/*.json` glob skips
  `workflows/distill_experiments/`):

      python bench/preflight_graph.py $(python -c "import sys; sys.path.insert(0, 'workflows'); from pathlib import Path; import h3_config; print(*h3_config.graph_paths(Path('workflows')))") | grep 'the guide asks 350-500'

### Still open

- **Camera-motion vocabulary has live WARNINGS, and is not unenforced.**
  `bench/check_camera_vocabulary.py` has graded every shipped prompt's motion
  phrases against base §4.3's sets since 2026-08-28. Run it; do not restate its
  findings here. A warned term is informational by the owner's ruling of
  2026-09-01 (§4), so the warnings are left standing on purpose and a prompt is
  not rewritten to clear one without a rendering reason.

  > **WITHDRAWN 2026-09-01: this bullet previously read "Camera-motion
  > vocabulary is enforced by nothing", and carried a same-day
  > "re-checked" stamp.** The re-check looked in `preflight_graph.py` and
  > `check_prompt_guide_conformance.py` and nowhere else, so it missed a checker
  > that had existed for four days and confirmed the claim it set out to verify.
  > **A date on a claim records when it was written, not whether it is true** —
  > and a narrow re-check is worse than none, because it launders a stale claim
  > into a fresh-looking one. The decidable half of that requirement is built;
  > the speaker-identity half is not mechanizable and remains open.
- **The ref2va word budget is preflight's only recurring WARN**, which is why it
  stopped being read as a finding. That is a reason to fix the prompts or retire
  the WARN, not to keep both.
- **No shipped graph is a multi-shot FL2VA or L2VA.** Every keyframe graph is one
  shot, so the branch that resolves `Shot N` to anything but 1 was dead until
  2026-09-01 and is exercised only by §10.3.2, §10.3.5, §10.4.3 and §10.4.5.
  That dead branch was carrying a defect: see §14.
- **Nothing regenerates `prompt_catalogue.md`.** It has a `--check` mode and no
  caller, and it had gone stale by 2026-09-01. This repo has no test runner by
  design (`docs/checks.md`), so the discipline is to run it before trusting it,
  not to wire a gate.

---

