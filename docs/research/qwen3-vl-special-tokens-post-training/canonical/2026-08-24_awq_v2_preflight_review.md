# Independent review of the Gemini AWQ v2 preflight

**Status:** Authoritative rejection of the reviewed artifact set
**Observation date:** 2026-08-24
**Decision:** **NO-GO. Do not launch quantization from these artifacts.**

This review applies to the compactly archived Gemini submission under
[`archive/rejected/gemini-preflight/`](../archive/rejected/gemini-preflight/README.md)
at the hashes recorded below. Bulky generated manifests were removed after
their hashes were frozen. Machine-local home paths were privacy-redacted in the
archived display copies, so the hashes identify the original submitted bytes.
This does not reject a future native-H3-calibrated
AWQ candidate. It rejects the claim that this particular preflight passed and
the claim that its staged launcher is ready for technical-lead execution.

The current deployed checkpoint and its ComfyUI symlink remain the active
artifact. At review time no `nativecal` candidate directory existed under the
local `llm-compressor/models` directory.

## Review substrate

- `llm-compressor` commit:
  `8357e9459228be5831ff43f9449cdd7733d3d877`
- calibration manifest SHA-256:
  `86c06b706bf8ebd5e33a006ba11a42fc210f3d74b4ab1809ccbd1de6f6a0fcc9`
- evaluation manifest SHA-256:
  `72eb11eaf99a74acae244e3a5ebe2732bcefc813f811008d31a873f36b884dcf`
- staged quantizer SHA-256:
  `bbc02d055a1697a977c8f6328e4f123133437357b4304d48a0177f3eca16a535`
- presentation builder SHA-256:
  `f5c4dbc66121ba77f3e0b17ca291d1813a250fce371298cba6e56b8bcef04f2f`
- presentation test SHA-256:
  `a23bd32519d5d5a115826ec929a313ef7684aac94be3b06f867aa8e266cf6b27`
- geometry test SHA-256:
  `818d345ddc4ac31f9797af06fb845e7255a1774ae553bb9d8bb4c0d62eb63fff`
- trace generator SHA-256:
  `23eb23883636dbc187d5aee8388954c008b37e8ea0075c6f84dd08e19e208ed9`

The review used source inspection, independent JSONL reductions, the installed
ComfyUI implementation, the current AWQ adapter, yesterday's completed
quantizer, and live construction of the proposed modifier objects in the exact
`llm-compressor` virtual environment. It did not load the 32B model or launch
AWQ.

## Launch-blocking findings

### 1. The staged recipe cannot be constructed

The current `llm-compressor` rejects both proposed modifier definitions before
`oneshot` can run:

- `AWQModifier(duality=False)` supplies a nonexistent field. The API field is
  `duo_scaling`; Pydantic reports `Extra inputs are not permitted`.
- the `QuantizationModifier` supplies a `config_groups.group_0` without its
  required `targets` field. It also supplies both `scheme` and `config_groups`,
  which the installed resolver explicitly forbids.

These were reproduced with
`coderef/llm-compressor/.venv/bin/python` against the commit
above. This is a hard launcher failure, not a quality concern.

### 2. The proposed video calibration path does not exist

The manifest labels every selected Malcolmrey output MP4 as an input
`video` reference and fabricates a two-frame count and `[0.0, 0.5]` timestamps.
The inventory code hashes the file but does not decode it or extract those
frames. The local files are generated target videos paired with their
generation prompts; they are not thereby native Ref2VA input references.

The staged quantizer then handles a video record with a literal `pass`. All 106
selected Malcolmrey rows therefore reach `AutoProcessor` as text-only rows.
The handoff's claim that the proposed run covers real video blocks is false.

### 3. The row-level trace uses synthetic random frames

For every selected Malcolmrey row, the trace generator substitutes
`torch.rand((2, 384, 384, 3))`. It does not seed that random tensor and does not
decode the row's MP4. This explains the independently observed result that all
106 Malcolmrey trace rows have the same `[[1, 24, 24]]` grid pattern.

The reported fallback count is initialized to the literal value zero and never
computed. The report's `PASSED` status and its parity-test summaries are written
unconditionally. The script does not inspect the evaluation manifest despite
claiming that scope in its module docstring. Consequently, the generated trace
and acceptance report are not evidence for the real proposed run.

### 4. Five of the six claimed presentation-parity fixtures do not compare the two paths

Only the text-only fixture compares the builder's token IDs to the installed
ComfyUI tokenizer. The single-image and interleaved fixtures create a ComfyUI
result but never compare it. The two-image, odd-video, and special-token
fixtures do not construct a ComfyUI comparison at all. They assert properties
of the new builder against values the same builder produced.

Accordingly, the supported result is one text-only tokenization comparison,
not `6/6` native-H3 presentation parity. There is no demonstrated token,
ordered-media, vision-span, timestamp, grid, patch, or token-tag identity for
the multimodal fixtures required by
[`native_h3_contract.md`](native_h3_contract.md).

### 5. The geometry suite has no independent reference arm

The geometry script calls only `process_still_image` and
`process_video_block` from the new builder, then checks its own shapes and broad
token-count ranges. It never calls the release processor, the AWQ artifact's
snapshotted processor, `MiniMaxH3ReferenceConditioning`, or the installed AWQ
adapter. It therefore records examples from one implementation; it does not
establish processor parity.

The builder is also unused by the staged quantizer. Even if the builder were
validated later, that would not validate the launcher's independent
`AutoProcessor` path.

### 6. The audited sequences are not the declared launch population or governed by an effective sequence policy

The generated trace reports 53 calibration rows longer than the launcher's
`max_seq_length=2048`, including one traced row of 4,226 tokens. The launcher
tokenizes with `truncation=False` and passes a callable collator. Subsequent
source inspection and the measured calibration-seam probe established that, in
the pinned `llm-compressor`, this callable short-circuits both consumers of
`max_seq_length`. The declared 2,048 value is therefore inert; it would not
truncate these rows. This corrects the earlier uncertainty about what that
parameter would do.

The remaining launch blocker is that no manifest-enforced sequence policy or
feasibility measurement governs the full 4,226-token row, and the row-level
trace is not generated from the actual dataloader handed to `oneshot`. See
[`2026-08-24_calibration_input_seam.md`](2026-08-24_calibration_input_seam.md)
for the source and probe evidence.

Silent image-decode exceptions can also turn image rows into text-only rows.
No assertion reconciles the intended manifest with the actual tokenized
dataset.

### 7. The manifest report overstates what was inventoried and how it was selected

The source-inventory file contains 2,662 records, of which 2,658 carry
`unique_primary`; it is not itself a 2,658-row unique-primary manifest. The
report hard-codes 1,560 Malcolmrey rows as scanned even though only 1,552 rows
were written after blank-prompt rows were skipped. It calls 1,548 local rows
decodable even though the code checks only file existence and successful
SHA-256 hashing.

The calibration selector shuffles deterministically and takes the first source
quotas. It does not stratify by task despite saying it does. The resulting
calibration manifest contains 256 unique exact prompt/media records, but only
241 unique nonempty ordered-media hashes across 247 media-bearing rows. The
calibration and evaluation files leave 111 unique-primary records unassigned
without a rejection manifest or reason.

## Findings that did survive independent checking

These narrow properties are real but do not satisfy the preflight gate:

- the calibration and evaluation JSONL files contain 256 and 2,291 rows;
- each file has unique exact `dedup_key` values;
- the two files have zero overlap in their recorded normalized-prompt hashes;
- the two files have zero overlap in their recorded nonempty ordered-media
  hashes; and
- the calibration manifest's literal prompt text contains `<d>` and `</d>` but
  none of the other five H3 marker strings.

The disjointness result is about the recorded hashes. It does not repair the
incorrect media semantics, prove MP4 decoding, validate presentation, or show
what the quantizer would actually consume.

## Required replacement gate

Claude should own the corrected launch path and acceptance decision. Gemini's
raw inventory work may be used as candidate input only after the lead rechecks
its semantics and media. A replacement preflight must, at minimum:

1. classify Malcolmrey rows honestly as generation prompts/output clips; do
   not relabel outputs as reference inputs;
2. use real decoded media for every media-bearing fixture and record decode,
   sampling, pixel/patch, grid, token, and presentation hashes;
3. make the exact validated presentation/processor path be the path passed to
   `oneshot`, rather than testing a builder the launcher does not use;
4. compare all multimodal fixtures against an independent installed-native
   path and include mutations that make each check fail;
5. instantiate the complete recipe successfully in the pinned environment
   before loading the 32B model;
6. enforce the sequence-length policy in the deterministic manifest/dataloader,
   do not rely on the inert `max_seq_length` declaration, and trace the exact
   post-policy dataset actually consumed by calibration;
7. produce deterministic calibration, holdout, and rejection manifests with
   declared task/media targets and reasons for every excluded row; and
8. keep the current checkpoint and symlink untouched until a new candidate has
   passed artifact and encoder-level evaluation.

Until that replacement evidence is independently accepted, the v2
quantization lane remains open but **not launch-authorized**.
