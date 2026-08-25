# Where each marker-corpus arm binds at render time

last updated: 2026-08-25

Design note, not an implementation. `bench/marker_corpus/compiled.json` declares
each arm as a triple — prompt bytes, tokenizer identity, model transform — and
says of the model-side one that nothing there applies it. This answers the
question that leaves open: **at render time, what object does each arm attach
to, what does the render's provenance show for it, and what goes red when an
arm claims a transform that did not happen.**

Scope: no code here, nothing built, nothing rendered. The corpus stays the only
author of prompt text. Nothing proposed touches the deployed artifact or the
seven marker rows on disk. Build after Gate 5.

Every mechanical claim below was executed on this box on 2026-08-25, on CPU,
against the deployed v1 artifact loaded read-only. Where a claim is an
inference it says so.

## The four arms, and where each attaches

| arm | attaches to | mechanism |
|---|---|---|
| `release_id` | nothing | the corpus's prompt bytes go in the graph's text widget |
| `stripped` | nothing | same, with the corpus's stripped bytes |
| `legacy_bpe` | the CLIP's **tokenizer** | a fresh pre-patch tokenizer on a cloned CLIP |
| `mean_init_rows` | the CLIP's **embedding rows** | an offset-keyed patch on a cloned patcher |

The first two need no binding point and no node. They are graph-level, and the
control on them is a digest, not a mechanism.

The second two are both CLIP-in, CLIP-out. That is the whole argument for where
they live.

## Bench nodes, not options on the encoder loader

**Recommendation: one bench-only node per transform, taking CLIP and returning
CLIP. Not an input on `MiniMaxH3AWQEncoderLoader`.**

- The loader ships. It is published as a single downloadable file for people
  who want the encoder and not this research repo, and its contract is
  deliberately narrow — a merely similar artifact must fail loudly rather than
  load. An input meaning "use the tokenizer that was wrong before the upstream
  fix" is a foot-gun on a shipped node, and it widens a contract kept narrow on
  purpose.
- Neither transform needs anything the loader has. Both act on an already-built
  CLIP, and neither requires re-reading the artifact or reloading weights. A
  loader option would have to do its work after construction anyway — which is
  what a separate node is.
- The arm stays visible in the graph. The existing provenance stamp already
  hashes the whole prompt graph into `graph_sha256`, so an arm expressed as a
  node changes that hash for free. An arm expressed as a widget on a node that
  is present in every graph changes it too, but a reader looking at two stamps
  cannot see *which* knob moved without the graphs; a node they can see by name.
- Retirement is cheaper. When the corpus is finished these nodes are deleted;
  an input on the loader would have to be deprecated on a published surface.

The cost is one extra node in each arm's graph, and that only in bench graphs.
No shipped graph gains anything.

## `legacy_bpe`

**Attaches to:** a fresh `MiniMaxH3Tokenizer` assigned onto a cloned CLIP.

`bench/audit_h3_marker_tokenization.py::_unpatched_clip` already builds exactly
this and is the code to reuse rather than re-derive. Two properties of it are
load-bearing and were re-confirmed here:

- It builds a **fresh** tokenizer, not a copy. `CLIP.clone()` assigns
  `n.tokenizer = self.tokenizer` — the same object — so mutating the clone's
  tokenizer would reach every other holder of that CLIP.
- It empties whichever token-list attribute exists rather than one it expects
  by name, and then **verifies by vocabulary** that the result no longer
  declares the markers. That branch exists because keying off the constant's
  name is how a reconstruction returns the patched tokenizer while calling it
  stock. On this checkout the attribute present is the module-level one; the
  class-level name from the retired local branch is absent, and a version
  checking only for the latter would have found nothing to empty and silently
  produced the release tokenizer.

Executed, on this checkout: the release tokenizer resolves the seven markers to
ids 151669 through 151675; the reconstructed arm resolves none of them; a
marker-bearing probe tokenizes to a different length and a different id
sequence under the two; a marker-free probe tokenizes identically; and the
release tokenizer object is unchanged after the reconstruction.

Those last two are the arm's own validity conditions, not extras. The
marker-free case is what distinguishes "this arm changed the markers" from
"this arm is a different tokenizer".

## `mean_init_rows`

**Attaches to:** an offset-keyed weight patch on a cloned CLIP's patcher. Not
an in-place write.

The distinction is the whole of it. `CLIP.clone()` shares `cond_stage_model` by
reference and clones only the patcher (executed). The encoder loader also
installs `cached_patcher_init`, so ComfyUI can hand the same loaded CLIP to a
later prompt. An in-place assignment into `embed_tokens.weight` therefore
reaches every later render that reuses the cached model, which is exactly the
inheritance the brief forbids. A patch on a cloned patcher cannot: the shared
module is never written, and ComfyUI's own unpatch restores whatever it did
apply.

The key, read off the loaded model rather than assumed, is
`qwen3vl_32b.transformer.model.embed_tokens.weight`, shape 151936 by 5120,
BF16. The seven marker rows are contiguous, so the patch addresses only them:

    (key, (0, 151669, 7))  ->  ("set", (rows,))

Executed: `add_patches` matches that offset-keyed form and returns it; applying
it through ComfyUI's own `calculate_weight` replaces exactly rows 151669
through 151675, leaves 151668 and 151676 byte-identical, and leaves the live
model's table unchanged. A patch keyed without the offset would need a
dense table-sized delta; the offset form is why this arm costs seven rows
rather than the whole embedding.

**Undo is structural, not a step to remember.** The arm's CLIP is a clone, the
transform lives on that clone's patcher, and the original is untouched. Nothing
has to be reverted, and the control below asserts that rather than trusting it.

## What the provenance record shows

**There is a gap here that has to be closed before either arm renders.**
`MiniMaxH3ProvenanceStamp` takes latent, model and sigmas. It has no CLIP
input, so today nothing about the encoder arm can reach a stamp at all. The arm
would be visible only as a change in `graph_sha256`, which says something
changed and not what.

The stamp needs an optional CLIP input and an `encoder_arm` block. What that
block records is the part that matters, and the rule is already established in
this repo by `bench/check_provenance_stamp.py::closure_is_read_not_declared`: a key
being present proves it was listed, and only a value that *moves* proves it was
read. So the block records values read off the live CLIP, never the arm name a
node was given:

- **Tokenizer identity.** Which of the seven markers the CLIP's own tokenizer
  resolves to single ids, and the digest of the id sequence it produces for a
  fixed probe string. Both move between the release and legacy arms; neither
  can be satisfied by a node that declared an arm and did nothing.
- **Marker rows.** A digest of rows 151669 through 151675 as the patcher will
  actually produce them, derived through ComfyUI's `calculate_weight` from the
  patcher's own patch list — not from the payload the node was handed. Moves
  when the transform attached, and does not when it did not.
- **The arm name**, recorded alongside as a label only, explicitly not evidence.

The arm name and the read values are then two independent facts, and the red
controls below are all of the form *they must agree*.

Cost: deriving the patched rows through `calculate_weight` needs the full table
because the narrow happens inside it, so the digest is best computed once when
the arm is applied and stamped onto the CLIP for the provenance node to read
back. That is the same shape as `_h3_encoder_contract`, which the loader
already stamps and `reference_geometry.encoder_contract_from_clip` already
reads back — a pattern with a control on it rather than a new one.

## Red control per arm

Each is an arm claiming a transform that was not applied, and each must go red.

| arm | the violation to build | what goes red |
|---|---|---|
| `release_id` | prompt text edited away from the corpus's bytes | the rendered prompt's sha256 does not equal the scene's `canonical_prompt_sha256` |
| `stripped` | a marker string left in, or ordinary text altered while stripping | any of the seven marker strings present, or `ordinary_text_sha256` moved |
| `legacy_bpe` | the arm declared, the tokenizer left as the release one | the stamp shows the seven markers resolving to 151669 through 151675, and the probe digest equal to the release arm's |
| `legacy_bpe` | the reconstruction keyed off an attribute name that does not exist here | nothing to empty, so the arm's vocabulary still declares the markers — the same red as above, which is why the vocabulary is the observable and not the branch |
| `mean_init_rows` | the arm declared, `add_patches` given a key the model does not have | `add_patches` returns an empty match list, and the row digest equals the unpatched one |
| `mean_init_rows` | the transform applied in place instead of as a patch | after the arm, the original CLIP's rows no longer hash to the unpatched digest |
| all | the stamp reading the node's declaration instead of the CLIP | two arms differing only in transform produce identical `encoder_arm` blocks |

The last row is the one that governs the others. It is the generalization of
the Sol closure case: a stamp that records what it was told is indistinguishable
from a stamp that records what happened, until two arms that should differ do
not.

## What this note does not establish

- That any arm's transform changes what the DiT reads. The embedding rows
  behind these ids are untrained (`bench/audit_h3_token_embeddings.py`) and
  whether the markers reach the output is not settled here.
- Any rendered or perceptual claim. Arms rendered from this design are compared
  under `docs/eval_comparison.md` section 3 like every other comparison, and
  the same-sample rule applies: these are different samples, not degraded
  versions of one.
- That the corpus is frozen. It declares itself a seed set, and freezing is the
  owner's act.
- Anything about Gate 5. The binding points above are independent of the
  candidate's numbers, which is why this could be designed before it lands.
