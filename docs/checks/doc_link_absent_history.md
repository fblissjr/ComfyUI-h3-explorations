# Deliberately Absent Citations Ledger — Full Historical Notes

> **Extracted from [`docs/checks.md`](../checks.md) on 2026-10-27.**
> Preserves the full multi-paragraph `WHY:` histories for every `PATH:` in the `doc-link-absent` ledger. The active machine-read `doc-link-absent` block with concise 1-line reasons lives in [`docs/checks.md`](../checks.md).

## What is deliberately not checked

`docs/open_experiments.md` is the other half of this document: seven things
this repo has decided **not** to measure, each with its cost, the decision it
would change, and the actual blocker. Read it before proposing a new check --
several obvious ideas are already there with a reason attached.

A suite of twelve **render** scenes is designed but not run. They are quality
gates for output, not code checks: each carries a pre-registered binary claim
and a predicted per-arm direction, and a human watches and listens. Nothing
executes them, and nothing here can -- judging whether a third shot happened
or two voices stayed distinct is not a `check_*.py`.

> Those live in `internal/`, which is **gitignored and not distributed**. If
> you cloned this repo you do not have them, and that is deliberate: they are
> the owner's working research notes, not shipped content. Everything this
> document describes is in `bench/` and is present in the clone.

One thing there has no code check and cannot get one until the graphs exist:

- **Two-stage split graphs do not exist** (the lightx2v ones were retired
  2026-09-26). When one does, the checks worth writing are: both stages read
  one schedule, stage 2 carries `DisableNoise`, the split point is inside the
  step range, and the two stages' shift nodes agree.


### Citations that cannot resolve, for `check_doc_links.py`

The ledger that check reads, kept here rather than in the script for the same
reason `retraction-consumers` lives in `docs/evidence.md`: the enumeration
belongs beside the prose it governs, and nothing gets a second copy.

An entry asserts that someone looked and the target is *deliberately* gone. It
is not a way to silence a broken link.

```text
PATH: meta_sam3/EDITS.diff
WHY: the declared edits on the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model_builder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/decoder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/io_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/maskformer_segmentation.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/model_misc.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/multiplex_mask_decoder.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_image.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_multiplex_base.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_multiplex_detector_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_multiplex_tracking.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_tracker_utils.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/sam3_video_base.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/text_encoder_ve.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/tokenizer_ve.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/video_tracking_multiplex_demo.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/video_tracking_multiplex.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: meta_sam3/sam3/model/vl_combiner.py
WHY: part of the local copy of Meta's SAM 3.1 inference code; removed by the owner on 2026-10-17
     (r17_05, nothing imports or runs it; upstream is in coderef/sam3), last present at commit a05b676d.
     The 2026-10-07 research notes under docs/research/masking/ keep citing it by line.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/owner_authored_marker_corpus.md
WHY: the brief the marker corpus was compiled to, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_layer50_processor_policy_benchmark.md
WHY: a note on the encoder's processor policies at one layer, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_serving_geometry_composes.md
WHY: a note on how the serving geometry composes, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_keyframe_vs_reference_positioning.md
WHY: a note on how keyframes and references are positioned, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/active_plan.md
WHY: the plan of the encoder calibration work, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: docs/research/qwen3-vl-special-tokens-post-training/canonical/2026-08-24_still_policy_token_cost.md
WHY: a note on what each still policy costs in tokens, in the research folder of the encoder calibration work; that work is not
     carried on and the owner had the folder removed on 2026-10-12, last present at commit
     9f892833. Pages that drew on it keep citing it by name.

PATH: bench/check_calibration_selector.py
WHY: a check of the calibration row selector's selection paths; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/check_marker_corpus.py
WHY: a check of the compiled marker corpus against its brief; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/check_pool_media_integrity.py
WHY: a check of the calibration pool's media verification; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/check_native_h3_presentation.py
WHY: a check of a native-H3 calibration bundle against two reference arms; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/check_released_encoder_is_stock.py
WHY: a check that compared the released text encoder's shards with stock Qwen3-VL's by digest; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/check_calibration_model_mapping.py
WHY: a check that the released checkpoint loads strictly into the calibration model; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/select_v2_calibration_rows.py
WHY: the tool that selected calibration rows from the pool; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/build_h3_calibration_pool.py
WHY: the tool that built the calibration candidate pool; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/review_pool_near_duplicates.py
WHY: the tool that reviewed the pool for near-duplicate media; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: bench/review_v2_calibration_bundle.py
WHY: the tool that reviewed a calibration bundle before a run; the encoder calibration work it belonged to is not
     carried on, and the owner had it removed on 2026-10-12, last present at commit a78ae801.
     Dated records and that work's research notes keep citing it.

PATH: docs/research/masking/2026-10-04_mrblue.md
WHY: a session's research note on the masked lane (a code review of the mask path); removed from the tree by the
     owner on 2026-10-11 with four others of 2026-10-04 to 2026-10-06, last present at commit
     c237d6cf. Records under bench/results, a later note and the index in the same folder, and
     pages of the wiki keep citing it.

PATH: docs/research/masking/2026-10-04_mrhf.md
WHY: a session's research note on the masked lane (which other models help the lane, and the band clip's shot table); removed from the tree by the
     owner on 2026-10-11 with four others of 2026-10-04 to 2026-10-06, last present at commit
     c237d6cf. Records under bench/results, a later note and the index in the same folder, and
     pages of the wiki keep citing it.

PATH: docs/research/masking/2026-10-04_mrpink.md
WHY: a session's research note on the masked lane (the build session's notes on the first clips); removed from the tree by the
     owner on 2026-10-11 with four others of 2026-10-04 to 2026-10-06, last present at commit
     c237d6cf. Records under bench/results, a later note and the index in the same folder, and
     pages of the wiki keep citing it.

PATH: docs/research/masking/2026-10-05_mryellow.md
WHY: a session's research note on the masked lane (the lane read broadly, and what a fractional mask does); removed from the tree by the
     owner on 2026-10-11 with four others of 2026-10-04 to 2026-10-06, last present at commit
     c237d6cf. Records under bench/results, a later note and the index in the same folder, and
     pages of the wiki keep citing it.

PATH: docs/research/masking/2026-10-06_mrsun.md
WHY: a session's research note on the masked lane (a reading of SAM 3.1's video pipeline against core's port); removed from the tree by the
     owner on 2026-10-11 with four others of 2026-10-04 to 2026-10-06, last present at commit
     c237d6cf. Records under bench/results, a later note and the index in the same folder, and
     pages of the wiki keep citing it.

PATH: vendor/sol_attn_minimax.py
WHY: upstream's Sol-Attn node, kept as a read-only reference since 2026-08-30 and loaded by
     nothing; removed from the tree by the owner on 2026-10-11, last present at commit c237d6cf.
     The node that runs is sol_attn_h3.py. docs/SOLATTN.md, docs/morton.md,
     docs/open_experiments.md and one research note cite it by line as where a thing was read.

PATH: void_conditioning.py
WHY: the module of `MiniMaxH3VoidConditioning`; removed 2026-10-05 with the VOID code at the owner's word (0.192.0),
     last present at commit df3bd21b. The two records of the test under bench/results,
     the changelog and docs/wiki/decisions.md keep citing it.

PATH: bench/convert_void_checkpoint.py
WHY: the VOID checkpoint converter; removed 2026-10-05 with the VOID code at the owner's word (0.192.0),
     last present at commit df3bd21b. The two records of the test under bench/results,
     the changelog and docs/wiki/decisions.md keep citing it.

PATH: bench/check_void_conversion.py
WHY: the converter's check; removed 2026-10-05 with the VOID code at the owner's word (0.192.0),
     last present at commit df3bd21b. The two records of the test under bench/results,
     the changelog and docs/wiki/decisions.md keep citing it.

PATH: bench/check_void_conditioning.py
WHY: the conditioning node's check; removed 2026-10-05 with the VOID code at the owner's word (0.192.0),
     last present at commit df3bd21b. The two records of the test under bench/results,
     the changelog and docs/wiki/decisions.md keep citing it.

PATH: bench/prove_calibration_seam.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit 0372a60a. The encoder-lane research docs in docs/research keep citing it.

PATH: bench/pilot_sequential_feasibility.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit b41a13fc. The encoder-lane research docs in docs/research keep citing it.

PATH: bench/probe_calibration_input_seam.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit bd97ea90. The encoder-lane research docs in docs/research keep citing it.

PATH: bench/h3_awq_recipe.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit 09bf8a27. The encoder-lane research docs in docs/research keep citing it.

PATH: bench/h3_gptq_recipe.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit b41a13fc. The encoder-lane research docs in docs/research keep citing it.

PATH: bench/check_calibration_checkpoint.py
WHY: needed `llmcompressor`, which the owner removed as a dependency on 2026-09-29
     (0.175.0), with the AWQ/GPTQ calibration lane it served; last present at
     commit bbc41f09. The encoder-lane research docs in docs/research keep citing it.

PATH: assert_chain.py
WHY: the module of `SageChainAssert`, retired 0.173.0 (owner, 2026-09-29); last
     present at commit f0fbbdba. It left every generated graph on 2026-09-17, and
     docs/wiki/next_steps.md and the sage-chain history keep citing it.

PATH: reference_fit.py
WHY: the module of `MiniMaxH3ReferenceFit`, retired 0.173.0 (owner, 2026-09-29)
     with `reference_video_fit.py`; last present at commit d3ae7c50.
     docs/h3_references.md and docs/custom_node_gaps.md are history that keep
     citing it.

PATH: bench/analyze_morton.py
WHY: Morton block-geometry analysis, deleted 2026-09-27 with the Morton lane
     (docs/research/2026-09-27_sol_node_redesign.md); last present at commit
     e0b8cc55. docs/morton.md is that lane's history and keeps citing it.

PATH: sol_curves.py
WHY: the Hilbert/2d curve helpers behind the Sol node's morton_curve, deleted
     2026-09-27 with the Morton lane; last present at commit e0b8cc55.
     docs/morton.md is that lane's history and keeps citing it.

PATH: _morton.py
WHY: the Triton pack's Wan Morton file, deleted with the pack on 2026-08-16
     (commit 6872dfd). docs/morton.md quotes its docstring as the only stated
     payoff for reordering anywhere in either pack, so the citation has to
     stay. Recover from github.com/kijai/ComfyUI-SolAttn_triton at 842c4ea.

PATH: _morton_h3.py
WHY: same pack, same deletion. Cited for the opposite reason -- it is the H3
     variant and makes NO quality or sparsity claim, which is the whole point
     of the comparison it appears in.

PATH: 2026-08-23_caption_marker_arms.md
WHY: owner removed the whole `internal/prompts/*.md` set on 2026-09-01. It is
     gitignored, so there is no history to recover it from. The marker-corpus
     documents in `docs/research/.../canonical/` cite it as the origin of the
     caption arms, and that provenance has to stay even though the source is
     gone -- a citation to a deleted file is still a truthful statement about
     where something came from.

PATH: 2026-08-22_office_refs_fr_subs.md
WHY: same set, same removal, same reason. Cited as the source of the
     owner-authored marker corpus.

PATH: h3_awq_encoder.py
WHY: the AWQ adapter, deleted 2026-09-13 with the closed encoder lane
     (docs/roadmap.md "Closed lanes"; docs/wiki/decisions.md). The research
     records under docs/research/ cite it as the implementation their
     measurements ran through, and that provenance stays. Recover from git
     history before that date.

PATH: h3_awq_encoder.md
WHY: the adapter's own document, deleted the same day for the same reason.

PATH: qwen3vl_32b_minimax_h3_w4a16_awq
WHY: the W4 artifact's config snapshot directory under config/, deleted with
     the adapter. The records cite it as the bounds their calibration rows
     were sized under.

PATH: check_h3_awq_encoder.py
WHY: the adapter's check, deleted with it. docs/research/official_weights_metadata.md
     cites it as the proof that core rejected the compressed-tensors file.

PATH: build_native_h3_calibration_batch.py
WHY: the calibration-batch builder of the closed lane, deleted 2026-09-13.
     The seam-audit records cite it as what produced their bundles.

PATH: capture_h3_encoder_states.py
WHY: the layer-50 capture tool for the W4 artifacts, deleted 2026-09-13.
     Cited by the holdout records as the producer of their captures.

PATH: measure_still_policy_token_cost.py
WHY: swept the AWQ snapshot's still bounds against core's; deleted with the
     snapshot it read. Its record under docs/research/ stands.

PATH: custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/h3_song_audio_context.py
WHY: seitanism's sibling pack, read 2026-09-12 and no longer installed under
     ComfyUI's custom_nodes/ (found 2026-10-01). docs/h3_audio_freeze.md cites
     it as what the audio-freeze lane was compared against.

PATH: custom_nodes/ComfyUI-H3-Motion-Context-MultiRef/existing_video_extension.py
WHY: the same sibling pack's extension node, cited beside it in
     docs/h3_audio_freeze.md; gone with the pack.
```


