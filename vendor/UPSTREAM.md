# What we want changed in code we don't own

A ledger, not a fork. The point is to capture **our improvements to upstream's
work without duplicating upstream's work** — because a fork drifts, and a
change that lives only in a Discord message is lost the moment the tab closes.

The mechanism has already worked once, end to end, and that entry is kept
below as the template.

## How this is meant to run

1. **Find it and write it here**, in implementable form: what, why, the
   evidence, and the concrete shape of the change. "Implementable" is the bar
   — `sink_q = [audio_start // BLOCK, video_start // BLOCK]` got implemented;
   "the sink is too broad" would not have.
2. **Relay it.** Upstream publishes this node through conversation, so that is
   where it goes.
3. **Verify against real geometry when it lands**, and record the version it
   landed in.
4. **Delete nothing.** A declined proposal is as useful as an accepted one —
   it records that the question was asked and answered.

**No code fork.** `vendor/README.md` has the preference order: upstream it,
else wrap it, and only fork with the divergence recorded in a header. Nothing
here has needed a fork.

### If prose is not enough

Keep a diff at `vendor/patches/NNN-name.patch`, generated against the pinned
version in `vendor/README.md`'s lineage table. Then:

```bash
git apply --check --directory=vendor vendor/patches/001-foo.patch
```

tells you instantly whether it still applies to the current drop. That is
controlled duplication with a drift detector attached, rather than a fork.
Delete the patch when upstream takes it. **There are no proposal patches**,
which is the preferred state.

`vendor/patches/001-local-version-tag.patch` is a different animal and is
deliberately not a ledger entry: it is a **local build config change we never
want upstream to take** (tagging their branch with our build id). It is
recorded as a patch so that an edit to a gitignored checkout of someone else's
tree is reproducible from this repo rather than existing only on one disk, and
`vendor/rebuild_kernel.sh` applies it only for the duration of a build. The
checkout stays clean at rest, because a standing modification there does not
conflict on `git pull`, it blocks it.

---

## Ledger

### LANDED in v2 — narrow `sink_q` to the target audio span

**Status:** proposed 2026-08-14 against v1, implemented by upstream the same
day, verification pending a render.

**What.** `exact_kv_and_rows` ran *every* conditioning query row dense,
references included. The dense-query protection exists for the target audio
rows; reference rows only need the exact-KV side.

**Why, and the evidence that made it concrete.** `final_layer(h, t_emb,
video_seg, audio_seg)` reads out only the target video and audio segments, so
reference rows' outputs are discarded — they matter only as keys and values
for later layers. Target audio is decoded output. The sink treated both
identically. From the measured row counts in `docs/h3_references.md`, one
345-frame video reference put ~58% of attention on the exact path at
`exact_kv_and_rows` against ~35% at `exact_kv`.

**The shape.** Target audio is a single contiguous segment immediately before
video (`PackedLayout` appends target audio then target video, "always the last
two segments"), so `sink_q = [audio_start // BLOCK, video_start // BLOCK]`
isolates it. The kernel already took an arbitrary `sink_q`; only the node tied
it to the KV sink.

**Outcome.** v2 publishes `sol_h3_audio_span` and uses it, falling back to v1
behaviour when absent. Upstream's tooltip: "the cost no longer scales with
reference size." Cost to us: one message. Cost of the fork we did not write:
a merge on every future drop.

---

### OPEN — `min_tokens`' guidance is misleading for H3 specifically

**Status:** found 2026-08-14, not yet relayed.

The input's help says sequences below ~12k stay dense because "below ~12k
tokens dense is usually faster". True in general, and inert for H3: the DiT
has **exactly one** attention site (`comfy/ldm/minimax/model.py`) at the
full packed length, and frame counts satisfy `n % 17 == 5`, so the shortest
clip past 5 frames is already S = 7,194 at 1344x768. Every value from 4096 to
12288 selects the same thing — everything.

Worth telling upstream because a user tuning `min_tokens` on H3 is tuning a
knob that cannot act, and the tooltip invites it.

---

### OPEN — duplicate function definitions in the node

**Status:** found 2026-08-14 on v1, still present in v2, not yet relayed.

`_log_once` and `_log_kernel_failure` are each defined twice, identically
(v1 lines ~113-139). Harmless — the second binding wins — but it is the kind
of thing that suggests a hand-merge and is one deletion to fix.

---

### OPEN (question, not a patch) — INT8 V in the exact branch

**Status:** raised 2026-08-14 with the sage fork, not yet put to upstream.

`sol_attn_exact.cu` runs `mma_s8` for QK and `mma_u8s8` for PV — uint8 P,
int8 V — with fp32 accumulation. No fp16/bf16 MMA anywhere and no option.

The question worth asking is **not** "this is too lossy". Our own headline
evidence for caring was an fp8-vs-fp16 accuracy ratio, and **that figure was
withdrawn on 2026-08-16 as untrusted and removed from this repo**
(`docs/evidence.md`): it came from a synthetic `torch.randn` sweep, and the
competing real-activation figure was never re-derived here. And int8-V
specifically is unmeasured by anyone.

The honest version: *has anyone checked where INT8 PV lands at ~100k tokens?*
Upstream's own eager reference is O(T²) and refuses past 4 GiB, so it cannot
reach H3's real length — a genuine gap in his test coverage rather than a
feature request. Blocked on our real-activation captures.

---

### OPEN — `MiniMaxH3TokenCounter`'s threshold uses the wrong stride

**Status:** found 2026-08-14 by the other h3-explorations session, source read,
not yet relayed. Different repo (KJNodes), same author.

It warns at `seq_len * 7168 >= 2**31` = 299,593 tokens, using the *contiguous*
stride. H3 hands the kernels three views of one fused qkv buffer at stride
21504, so the real int32 crossing is ~99,864 — and every shipped graph here is
past it at 102,816. Our `preflight.py` already names this and our sage fork
carries the int64 specialisation, so nothing of ours is at risk. The point is
that their counter is not a second opinion on the question it appears to
answer. One-line stride read to confirm.


### OPEN upstream, REVERTED locally — H3's seven special tokens in ComfyUI's tokenizer

**Status:** relayed as ComfyUI PR 15808, still OPEN. Applied locally as
`vendor/patches/002-comfyui-h3-special-tokens.patch` from 2026-08-21 and
**reverted on 2026-08-22 by owner decision**: we wait for Comfy to merge it
rather than carry a modification to a core file we do not own. The patch is
kept here so the local state is reproducible from this repo and so it can be
diffed against whatever upstream eventually merges.

**What.** ComfyUI backs the H3 tokenizer with its bundled `qwen25_tokenizer`,
which declares thirteen `additional_special_tokens` where the release declares
twenty. The seven missing ones -- `<d>`, `</d>`, `<|cutoff|>`,
`<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, `<|caption_end|>` --
tokenize as ordinary text, so a prompt containing one gets several BPE pieces
and a different embedding. The fix registers them in
`MiniMaxH3Tokenizer.__init__`, which is the only place that reaches every
consumer, core's `MiniMaxH3ReferenceToVideo` included -- no custom pack can
add an import to that.

**Why it belongs upstream and not here.** That reach is the whole argument. A
pack can only fix its own nodes.

**What carries it meanwhile.** `vendor_tokens.clip_with_vendor_tokens`, which
builds a fresh tokenizer with the tokens added and rebinds it on a clone. It
returns the CLIP unchanged when the tokens are already present, so it is a
no-op on an install that has the core patch and does the work on one that does
not -- written for exactly this state. Every graph here reaches it through
`MiniMaxH3Conditioning`'s `vendor_tokens` input, default True.

**What is NOT covered while the patch is out.** Core's own
`MiniMaxH3ReferenceToVideo`, which no shim of ours can reach. A graph wired to
the core node tokenizes markers as literal text and renders anyway.

**The verifier.** `bench/audit_h3_marker_tokenization.py` runs identically in
both states: with the core patch present it asserts the shim is a no-op, and
without it asserts the shim supplies the tokens, grading both against the
release tokenizer's own ids.
