# PDD: five implementations, and how this lane got where it is

last updated: 2026-08-28

Two things nobody had written down: **how our PDD implementation compares to
every other one available**, and **how the lane arrived at its current shape**
across the days it was built.

[`../../h3_pdd.md`](../../h3_pdd.md) owns the converter and node contract and
this file defers to it. Where the two disagree, that file is right — except at
§4 below, which is a list of places this pass found it stale.

**Different is not worse.** Where an implementation does something ours does not,
that is recorded as a difference and a consequence. A verdict appears only where
there is measured evidence.

Evidence labels: *read* (the code at the cited path), *measured* (computed for
this document from an artifact on disk, or a record in `bench/results/`),
*inference* (a conclusion, with the mechanism named so it can be refuted).

---

## 1. The five implementations

| short name | what it is | revision |
|---|---|---|
| **ours** | `pdd_lora.py`, `pdd_math.py`, `bench/convert_pdd_lora.py` | working tree |
| **vendor** | `alibaba-pai/MiniMax-H3-Acc-LoRAs`, the source we converted from | local copy |
| **core PR** | Comfy-Org/ComfyUI#15908 | head `ce1fb009dddc` |
| **UtilsCollection** | `coderef/ComfyUI-UtilsCollection` | `5bac35b` |
| **Mamad8** | a third-party PDD node pack installed on this box | `e8c8c95` |

Mamad8 targets a **different artifact family** — many more intervals, a
different block width, displacement-style heads, and a student LoRA loaded
separately. Its rows below are not like-for-like on the artifact, only on the
mechanism.

**No other engine implements PDD at all** (*read*). diffusers has full H3
support and runs Euler at eta 0, with no distillation. LightX2V ships DMD LoRAs
for H3, which is a different method. DiffSynth registers only SFT for H3.
sglang has no distilled H3 entry — its `parallel_decode` is VAE sharding, and
its speculative-decoding machinery is for language models. Searched for
parallel decoding, PDD, multi-token prediction, replicated heads and step
distillation.

---

## 2. The comparison

| axis | ours | vendor | core PR | UtilsCollection | Mamad8 |
|---|---|---|---|---|---|
| fusion formula | `dt/sum(dt)`, fp64 | same, bf16 cast | same, inline | same, fp64→fp32 | at export |
| when fused | lazily per span, cached | every forward | every forward | at patch time | at export |
| output weight resized | no | module replaced | yes | no | no |
| step index derived from | **`t_emb`** | a hook **counter** | `sample_sigmas` argmin | raw timestep | raw timestep |
| block extent from | `sample_sigmas` | config | `sample_sigmas` | an `nfe` widget | the artifact |
| emits SIGMAS | yes | no | no | yes | yes |
| off-grid step count | warns | undetectable | silently clamps | **raises** | fails closed |
| pruned-base adaln | affine fit, residual refused above a threshold | n/a | n/a | exact composition against a shipped basis | n/a |
| partition fingerprint | **yes** | no | no | no | n/a |
| patch point | two output linears plus a delegating wrapper | `setattr` plus a hook | inside core | copies the body, imports a core helper | copies the body |
| tests | three harnesses | none | none | five unit tests, none touching the maths | none |

**The arithmetic agrees everywhere.** Four independent statements of the fusion
plan agree term for term (*read*). Our three divergences from the vendor — step
index from `t_emb` rather than a call counter, cached fusion rather than
per-forward, fp64 rather than bf16 — are all recorded and defensible, and none
changes the result the plan defines.

### What only we do

- **A partition fingerprint.** The only mechanism in any of the five that
  catches a Ref2VA LoRA applied to an FL2VA base. This matters because the two
  published files are **key-for-key identical** (*measured*), which is one of
  the two silent traps CLAUDE.md names for this lane — nothing about the key set
  can tell them apart.
- Deriving the step index from `t_emb` rather than counting calls, which is what
  makes the node correct under a sampler that evaluates out of order.
- Delegating to the stock modulation instead of copying its body — both other
  ComfyUI packs copy it, and one imports a private core helper to do so.
- Arity-transparent patching, asserted against both core signatures.
- Three-way numeric grading of the conversion, and a measured adaln residual
  with a refusal threshold.
- A `patch_heads=False` control arm.

### What everyone else does and we do not

- **Raise rather than warn on an off-grid step count.** All three other
  implementations refuse, in three different ways. Ours warns. The in-flight
  rewiring (§3, S5) changes the shape of this by making off-grid mostly
  inexpressible rather than merely detectable, which is a stronger fix than a
  raise — but the hand-built graph case remains.
- Restrict block width to the trained envelope (UtilsCollection).
- Treat provenance hashes as load-bearing (Mamad8).
- Let core own the fused heads' VRAM (UtilsCollection).
- Enforce the sigma shift (both ComfyUI packs).
- Model the audio carry of the H3 sampling class (Mamad8).

### On "inefficient in spots"

The owner's reading of the core PR. **Not supportable as a throughput claim**
(*inference*): the per-forward fusion is a small einsum against a very large
sequence, and the PR's current head already wraps it in core's cast context.
What **is** supportable is a lifetime problem — the enlarged output weight
outliving the graph, which produced an observed broadcast error, and which is
why the PR had to change `comfy/model_patcher.py` at all rather than only the
model file.

---

## 3. How the lane got here

Five snapshots, reconstructed from git rather than from memory. Each transition
is a defect or a decision, named.

**Origin.** Converter reads the published file's metadata rather than guessing,
emits the backbone under ComfyUI's generic LoRA naming, the adaln pairs in a
neutral namespace, and the heads **already collapsed** to the fused count. Node
indexes the precomputed stack and recovers the step by nearest row of a table.
No graphs, no checks.

**S1 → S2.** Two defects of one family: a comparison too tight for its own
noise. The partition fingerprint fired on the first *correct* render, because
ComfyUI casts on load — so it became a distance comparison with a tolerance. The
head selector picked the wrong head at two of eight steps, because a step
sitting exactly on a boundary came back a fraction below it, and it was silent
in every direction. Then three structural moves: selection stopped going through
the step index at all and started matching **boundary embeddings directly**,
which deletes the quantisation problem rather than guarding it; the bank began
shipping whole and fusing at load, un-pinning the step count from the artifact;
and the adaln delta was pre-solved into the pruned base's own basis, turning
fifty forward patches into ordinary weight patches. Two escapes landed in this
window and both got guards — a prefix that was not a prefix, so four arms
rendered with **no modulation update at all**, and a tolerance that shipped
briefly below the noise it had to tolerate.

**S2 → S3.** The step-count widget was a requirement backed by a warning that
fires after sampling has started. It was replaced by deriving the block extents
from the sampler's own sigma vector at run time. Split sampling and partial
denoise became expressible for free. Three guards were added, two of them
**adopted from UtilsCollection** and one written against the core PR.

**S3 → S4.** The schedule-observation key collided across four schedulers at one
step count — and because the scheduler node sits *downstream*, changing it did
not re-execute the PDD node, so every block decoded an interval the sampler
never visited. The key became the whole vector. Also fixed: a converter error
that made every real pruned run look failed after writing a correct file, and a
log line claiming heads were patched at zero strength.

**S4 → S5, in flight as of this writing.** The node now **emits** the schedule:
a SIGMAS output and a step input, with the scheduler node deleted from the
non-split PDD graphs. Off-grid stops being expressible rather than merely
detectable. A shared schedule reader was added after a scheduler-only reader
returned nothing on every PDD graph and reported correctly-wired graphs as
wrong. A new check grades inertness against ComfyUI's own sigma computation and
asserts its exactness precondition rather than assuming it.

**This work is uncommitted and being written by another session.** Nothing in §3
S5 should be read as finished.

---

## 4. Stale records this pass found

Four, in descending order of consequence. All are proposed corrections to
[`../../h3_pdd.md`](../../h3_pdd.md), not applied here.

1. **The core PR adopted our head semantics, and our local copies of the vendor
   artifacts are stale** (*measured*). A PR commit changed its head formula to
   one that is correct only if the stored rows are deltas from the first head.
   On the copies on this box they are **not** deltas — they are verbatim heads,
   with an exact zero difference against the published values — and applying the
   PR's current formula to them yields a badly wrong result, a doubled head. The
   upstream repository's last-modified timestamp sits two minutes after that
   commit, so the artifacts were re-uploaded. Consequence: `h3_pdd.md`'s
   statement that the PR's strength parameter "means something different from
   ours" is **no longer true**, and our copies should be re-fetched before any
   further comparison. The per-file hashes added on 2026-08-27 are what made
   this visible at all, and the failure mode is loud rather than silent.
2. `h3_pdd.md` says the PR's diff is one file. It is now two — a later commit
   changes several sites in core's model patcher so that a shape-changing patch
   cannot be deferred. Not H3-specific.
3. The UtilsCollection row reading "unconsumed keys are an error — open,
   converter-side" is stale; the converter enforces it.
4. One characterisation to soften: `h3_pdd.md` describes UtilsCollection's
   pruned-adaln handling as "the same affine solve" and the pair as "two
   independent solutions". It is an exact composition against a basis shipped as
   an asset, with no residual to score.

---

## 5. Still uncontrolled in this lane

`docs/checks.md` holds the standing audit; these are candidates for it.

| requirement | status |
|---|---|
| a PDD arm is consumed by `euler`, not a re-noising sampler | **nothing.** Re-verified: no check reads the sampler name off a shipped graph. No escaped instance — every shipped graph carries it — so per CLAUDE.md that is a reason not to build the check yet |
| the node's own step and fused-head counts agree | **nothing** on the node itself. Covered statically for shipped graphs, which is not the population the rewiring was for |
| `strength` semantics | read into the record, graded only against turbo recipes |
| the runtime adaln injection path | **dead code that announces itself as a safe fallback.** The pruned conversion removes the tensors it reads, so it raises after logging that it is doing the right thing. Unreachable only because another guard refuses first — one guard masking another's failure |
| a legal step count is a *sensible* one | the in-flight work adds an envelope warning; nothing grades it |
| whether the heads change the output visibly | **not established.** No blind session has been run on a PDD arm |
| `docs/research/pdd/`'s README and its three HTML explainers | in no check, and absent from the in-flight diff, so the rewiring has probably staled them |

---

## 6. What to re-check when the in-flight work lands

Ordered by consequence. Carried from the reconstruction, and the reason this
file is dated:

1. Whether the new sigma check was actually run and committed.
2. Whether the graphs were rebuilt **after** the last generator edit — both are
   in the diff, and CLAUDE.md's rule is that nothing is true of a graph until it
   is rebuilt.
3. The UI-writing path, where the schedule can be absent and the writer has no
   node-removal step.
4. The shared schedule reader's positional assumption about widget order,
   against `bench/node_id_manifest.json` — the schema gained both an input and
   an output this round, and positional widget values are exactly what that
   manifest exists to protect.
5. Whether the step/head agreement gained a runtime comparison, or a stated
   reason not to.
6. Whether the dead adaln path was resolved or left.
7. The PDD research documents and explainers named in §5.
