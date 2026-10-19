# The reorganization: docs, bench and folder rules

**Goal.** A session finds what it needs by going down the layers (root `AGENTS.md`, a folder's `AGENTS.md`, its
index, the page or file) instead of searching, every file is on an index, and checks keep it that way. **Where it
stands, 2026-10-19:** the docs are reorganized and rewritten, `h3_config.py`'s comments are current, the folder-rules
standard and its check are in; the bench is inventoried and planned ([`bench.md`](bench.md)) but nothing in it has
moved.

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
| `workflows/h3_config.py`'s comments say what is current: purpose, provenance, when and why each value; 1971 comment lines to 885, the code's AST unchanged | this page's commit |

Every cut in the docs rewrite, with where it went, is
[`../../research/2026-10-19_docs_consolidation.md`](../../research/2026-10-19_docs_consolidation.md).
[`mechanical_commits.txt`](mechanical_commits.txt) lists the commits that only rewrote paths or comments, so
`bench/bench_inventory.py` does not date a script by them.

## Left, in order

1. **The bench** ([`bench.md`](bench.md)): archive what nothing uses, lane by lane, on the owner's list.
2. **`docs/method/evidence.md`**: what still holds, against the code. It is read by `bench/check_retraction_consumers.py`
   (its consumer ledger) and cited by about 50 files, root `AGENTS.md` among them for "Settled about H3"; two thirds of
   its dated entries are from August.
3. **`docs/state/handover.md`** to its rule of about 80 lines, at tonight's rewrite.
4. **Code that disagrees with its own page**, found by the rewrites (each a code change, so a build on a branch):
   - `bench/shot_to_load.py::defaults_carried` lists the head tube as opt-in and off, but the capture runs it whenever
     a head track is given, the default since `b7927da1`; the printed defaults of a load omit it.
   - `sol_attn_h3.py` keeps its own copies of the dense-block options, the ridge shield and the historical tail that
     `workflows/h3_config.py` also holds.
   - `h3_config.LORA_LOADER_CLASSES` still lists the retired `MiniMaxH3TurboLoRA`.
   - The `simple` scheduler's 11-sparse, 5-dense split was measured on the old 0.2-0.9 window and not re-derived for
     today's 0.2-1.0 (the comment now says so).
   - `bench/analyze_sol_block_grouping.py` writes a dated session path into its record, and two checks print one.
5. **Pages that still need a line**: `docs/masked_v2v/sam31.md` does not cover the SAM 3.1 head track;
   `running_a_job.md` and the 2026-10-18 MoGe-2 record name a `DEPTH_WIN_M` constant the code does not have;
   `h3_config.AUDIO_REFINE` points at a session `next_steps.md` no longer has.
6. **The `unmerged_blocks` tooltip in `pdd_lora.py`**: its block order and recovery figures are older than
   `bench/analyze_pdd_unmerge_curve.py`'s. A UI string, so the owner's call.

## On the owner's machine (not in git)

- The leads' mailbox moves from `internal/gemini/<date>_claude_to_gemini.md` and `..._gemini_to_claude.md` to
  `internal/leads/<date>_review_to_operate.md` and `..._operate_to_review.md`, with ids `R<n>` and `O<n>`
  (`docs/working/protocol.md`). The watcher `internal/gemini/wait_for_change.py` follows them.
- `internal/gemini/local_env.md` becomes `internal/local_env.md`: it is the box's, not one model's.
- `internal/gemini/PROTOCOL.md` is retired in favor of the tracked page.

## Waiting on the owner

- Which bench scripts to archive, lane by lane ([`bench.md`](bench.md), "What waits on the owner").
- Whether `bench/analyze_pdd_unmerge_curve.py` goes to `archive/bench/` now (nothing imports or runs it; one prose
  mention in `docs/distills/pdd.md` would be repointed).
