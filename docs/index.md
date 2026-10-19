# The wiki: where each answer lives

`docs/` is the wiki. This page routes; it states no fact about H3. Each folder has its own `index.md` with one row per
page, and a folder's pages run from a short overview down to the detail. Where a page and the code disagree, the code
is right (root `AGENTS.md`, "Do not trust prose").

## Start of a day

Read in this order (root `AGENTS.md`, "How the owner and the sessions talk"):
[`state/handover.md`](state/handover.md), then
[`working/working_with_the_owner.md`](working/working_with_the_owner.md), then [`working/protocol.md`](working/protocol.md), then
[`working/comfy_notes.md`](working/comfy_notes.md) before the first command.

## The folders

| folder | what it answers | start at |
|---|---|---|
| [`state/`](state/index.md) | where things stand: the handover, what is next, what was decided, the roadmap, what is not measured | [`handover.md`](state/handover.md) |
| [`method/`](method/index.md) | how the repo decides something is true: what holds, every check, the A/B process | [`evidence.md`](method/evidence.md) |
| [`working/`](working/index.md) | running ComfyUI and the box, the card and ledger runbook, telemetry, the sister checkouts | [`runbook.md`](working/runbook.md) |
| [`h3/`](h3/index.md) | the model and its ComfyUI path: what can be made, canvas and length, conditioning, references, gaps against the vendor | [`uses.md`](h3/uses.md) |
| [`attention/`](attention/index.md) | Sol-Attn and quantization: knobs, measured arms, Morton, upstream, the per-block precision policy | [`sol_attn.md`](attention/sol_attn.md) |
| [`distills/`](distills/index.md) | few-step distills: which one for which shot, PDD and its files, why ref2v resists | [`which_distill.md`](distills/which_distill.md) |
| [`prompting/`](prompting/index.md) | how to write an H3 prompt, the bank and catalogue of every shipped prompt, the audit | [`prompting.md`](prompting/prompting.md) |
| [`masked_v2v/`](masked_v2v/index.md) | the masked video-to-video lane: the pipeline, SAM 3.1, windows, who is who, what a render needs before it queues | [`overview.md`](masked_v2v/overview.md) |
| [`projects/`](projects/index.md) | work that spans folders and sessions: a plan, what is done, what is left, what waits on the owner | its index |
| [`research/`](research/index.md) | dated studies and postmortems: what was true when written, not current guidance | its index |

Above the docs: [`VISION.md`](../VISION.md) holds the tenets behind every rule; [`data/AGENTS.md`](../data/AGENTS.md)
holds the masked lane's rules and where each video's state lives.

## Code, by generated index

| index | what it lists |
|---|---|
| [`INDEX.md`](../INDEX.md) | every tracked file at the root: the pack's modules and the nodes each defines, and each directory's index |
| [`bench/INDEX.md`](../bench/INDEX.md) | every script under `bench/` by its own docstring |
| [`bench/results/INDEX.md`](../bench/results/INDEX.md) | every record by month and lane |
| [`workflows/INDEX.md`](../workflows/INDEX.md) | every graph, with the chain read off it |

All four come from `bench/build_index.py`; `--check` says whether they are current.

## Outside git, and linked anyway

These are gitignored, so a fresh clone does not have them, but pages link into them because that is where the
evidence or the working notes are. The links stay.

| path | what is there |
|---|---|
| `internal/` | session logs, prompt research and postmortems (start with the newest); the leads' mailbox under `internal/leads/` ([`working/protocol.md`](working/protocol.md)) and this box's paths in `internal/local_env.md` |
| `data/<video>/` | each video's analysis folder and the masked lane's state (`data/AGENTS.md` is tracked); `data/results/` holds the JSON behind older records |
| `bench/results/**/*.json` and `.jsonl` | the data beside each tracked `.md` record |
| `coderef/` | the sister checkouts, read and never imported ([`working/coderef.md`](working/coderef.md)) |
| the share's `output/Video/<video>/` | rendered clips |

## Keeping the wiki

The rules for writing here are [`AGENTS.md`](AGENTS.md); how folders, rules files and indexes are kept, and the checks
that hold them, are [`method/agents_and_indexes.md`](method/agents_and_indexes.md).
