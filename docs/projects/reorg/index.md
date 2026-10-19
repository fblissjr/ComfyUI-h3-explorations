# The reorganization: docs, bench and folder rules

**Goal.** A session finds what it needs by going down the layers (root `AGENTS.md`, a folder's `AGENTS.md`, its
index, the page or file) instead of searching, every file is on an index, and checks keep it that way. **Where it
stands, 2026-10-19:** the docs are reorganized and rewritten, `h3_config.py`'s comments are current, the folder-rules
standard and its check are in; the bench is inventoried and planned ([`bench.md`](bench.md)) but nothing in it has
moved. **Next is the owner's**: the seven decisions below, each with a recommendation. Nothing else is blocked on
them; the items after them can be done by any session.

## Done

| step | commit |
|---|---|
| `docs/` becomes the wiki: `docs/wiki/` dissolved into topic folders, each with `AGENTS.md`, a `CLAUDE.md` shim and `index.md` | `901bee1` |
| links into gitignored files kept, and routed from `docs/index.md` | `f63d4be` |
| every topic folder rewritten to lead with the answer; long pages split into an overview and detail; canvas and length given one home (`docs/h3/geometry.md`); corrections logged in `docs/state/decisions.md` | `db63687`, `de1c923`, `12413d3` |
| every research record opens with its finding; undated records carry their date; guidance moved out of `research/`, dated records in | `9e4336d`, `edf3f18`, `9699943` |
| session-named records renamed by subject; the PDD session handoffs archived to `archive/docs/` | `74aa6499` |
| `bench/bench_inventory.py`: who depends on each bench script; `archive/` given its own index | `8f0181e6` |
| one agent-neutral protocol for the two leads, `docs/working/protocol.md` (was `internal/gemini/PROTOCOL.md`); the folder-rules standard, `docs/method/agents_and_indexes.md`, and its check, `bench/check_folder_rules.py`; citations of dated session notes past seven days removed from tracked files | `d9316ed8` |
| `docs/masked_v2v/perception_models.md` an overview of each model's status today, from the lane's commits, over a page per model | `56ccfb83` |
| `workflows/h3_config.py`'s comments say what is current: purpose, provenance, when and why each value; 1971 comment lines to 885, the code's AST unchanged | `2562c1b9` |

Every cut in the docs rewrite, with where it went, is
[`../../research/2026-10-19_docs_consolidation.md`](../../research/2026-10-19_docs_consolidation.md).
[`mechanical_commits.txt`](mechanical_commits.txt) lists the commits that only rewrote paths or comments, so
`bench/bench_inventory.py` does not date a script by them.

## Waiting on the owner

Each: the decision, the recommendation, and what a session does on a yes. Answer in any order; a "yes" in the
terminal or on the Open List is enough to start.

| # | decision | recommendation | on a yes, a session |
|---|---|---|---|
| 1 | **Prune the bench by lane, or sort it into subfolders?** | **Prune.** A script's path is an identifier: 115 scripts are imported by bare name from `bench/`, the check runner finds checks by name, and records name paths. Subfolders would break all three for no gain the index does not already give ([`bench.md`](bench.md), "Why not subfolders") | follows the steps in [`bench.md`](bench.md) |
| 2 | **Which lane first?** | **The closed lanes** (`docs/state/roadmap.md`, "Closed lanes"), those with tools behind them: the AWQ encoder lane, our VSA node, Morton's tooling, the turbo LoRAs, TaoMate, PDMD, open experiment #23. Their tools are the safest to archive, since nothing current runs them | lists that lane's candidates in `local/` with what each measured and the record it made; the owner marks keep or archive; one commit per lane |
| 3 | **Archive `bench/analyze_pdd_unmerge_curve.py` now?** | **Yes.** It aggregates two 2026-08-31 records about the merged PDD path, the control arm since 2026-09-26. Nothing imports, calls or checks it | `git mv` to `archive/bench/`, repoints the one prose line in `docs/distills/pdd.md`, rebuilds the index |
| 4 | **Retire a closed lane's checks with its tools?** | **Yes.** A check whose only subject is archived tests retired code; it moves to `docs/method/checks.md`'s retired rows | does it as part of each lane's commit |
| 5 | **Review `docs/method/evidence.md` against the code?** | **Yes, as its own pass.** It is live (about 50 files cite it, `bench/check_retraction_consumers.py` reads its ledger, root `AGENTS.md` cites its "Settled about H3"), but two thirds of its dated entries are from August. Keep the ledger's format; re-check each "holds" and "do not rely" row against the code and records; move what is settled about H3 next to the page that owns its subject | rewrites it as the other pages were, logging each retired row in `docs/state/decisions.md` |
| 6 | **The `unmerged_blocks` tooltip in `pdd_lora.py`** gives an older worst-first block order and recovery figures than `bench/analyze_pdd_unmerge_curve.py`'s | **Shorten it** to what the control arm does and point at `docs/distills/pdd.md`, rather than restate figures in a UI string | a build on a branch, landed by the leads' steps |
| 7 | **Move the leads' files on the owner's machine** (below) | **Yes, at the start of a day**, before either lead writes its first entry | the owner moves them; nothing tracked waits on it |

## Left for a session (no decision needed)

1. **`docs/state/handover.md`** to its rule of about 80 lines, at tonight's rewrite (the review lead's).
2. **Code that disagrees with its own page**, found by the rewrites. Each is a code change, so a build on a branch:
   - `bench/shot_to_load.py::defaults_carried` lists the head tube as opt-in and off, but the capture runs it whenever
     a head track is given, the default since `b7927da1`; the printed defaults of a load omit it.
   - `sol_attn_h3.py` keeps its own copies of the dense-block options, the ridge shield and the historical tail that
     `workflows/h3_config.py` also holds; one copy each is the rule.
   - `h3_config.LORA_LOADER_CLASSES` still lists the retired `MiniMaxH3TurboLoRA`.
   - `bench/analyze_sol_block_grouping.py` writes a dated session path into its record, and
     `bench/bench_e2e_h3.py` and `bench/check_solattn_correctness.py` print one.
3. **Pages that still need a line**: `docs/masked_v2v/sam31.md` does not cover the SAM 3.1 head track;
   `docs/masked_v2v/running_a_job.md` and the 2026-10-18 MoGe-2 record name a `DEPTH_WIN_M` constant the code does
   not have; `h3_config.AUDIO_REFINE` points at a session `docs/state/next_steps.md` no longer has.
4. **A measurement to redo**: the `simple` scheduler's 11-sparse, 5-dense split was measured on Sol's old 0.2-0.9
   window and not re-derived for today's 0.2-1.0 (`workflows/h3_config.py`'s comment says so).
5. **The prompt catalogue** (`docs/prompting/prompt_catalogue.md`) was already stale before the reorganization (a
   reference timing figure); `python bench/build_prompt_catalogue.py` rebuilds it.

## On the owner's machine (not in git)

- The leads' mailbox moves from `internal/gemini/<date>_claude_to_gemini.md` and `..._gemini_to_claude.md` to
  `internal/leads/<date>_review_to_operate.md` and `..._operate_to_review.md`, with ids `R<n>` and `O<n>`
  ([`docs/working/protocol.md`](../../working/protocol.md)). The watcher `internal/gemini/wait_for_change.py` is
  pointed at the new files.
- `internal/gemini/local_env.md` becomes `internal/local_env.md`: it is the box's, not one model's.
- `internal/gemini/PROTOCOL.md` is retired in favor of the tracked page.

## How to pick this up

Read this page, then [`bench.md`](bench.md). `python bench/bench_inventory.py` gives the bench data as it stands
(`--json` for a tool); `python bench/check_folder_rules.py`, `python bench/check_doc_links.py` and
`python bench/build_wiki_index.py` say whether the docs still hold together. Each step's commit is in "Done" above.
