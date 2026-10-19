# `qk_balance` is not inert behind the dense tail (2026-09-27)

Test 1 of the Sol node redesign
(`docs/research/2026-09-27_sol_node_redesign.md`). The question: does
`MiniMaxH3Sol`'s `quantizer` balanced (the old `qk_balance=True`, shipped
everywhere) change output once `dense_blocks` 45,48,49 sends the blocks it
was built for off Sol? The audit had inferred it does not
(`2026-09-27_sol_node_compound_audit.md` §4 item 1, now corrected).

## What ran

- **Manifests:** `bench/sol_redesign_test1a_arms.json` (session A) and
  `bench/sol_redesign_test1b_arms.json` (session B, on a restarted server).
  Rows are in `2026-09-27_sol_redesign_test1.jsonl`. Seed 730451892 is held
  on every arm.
- **Graphs**, each at its shipped settings except `quantizer`:
  - i2va: `workflows/distill_experiments/h3_first_frame_to_video_pdd_api.json`
  - ref2va: `workflows/h3_image_ref_plus_text_to_video_pdd_api.json`
  - t2v: `workflows/distill_experiments/h3_text_to_video_pdd_savelat_api.json`,
    with the slapstick_moving_piano bank prompt at 345 frames. This is test
    0's arm (`2026-09-27_sol_redesign_bitid.md`), whose balanced latent is
    `..._pdd8_00003_`.
- **Kitchen** `0.2.35+sol.fc32da2.up.c8c7825`, main at 0.162.0.
- **Session A:** balanced, then plain, for i2va and ref2va, then t2v plain.
  Adjacent arms differ in graph or quantizer, so nothing is a cache hit.
- **Session B, a fresh server process:** the balanced i2va, ref2va and t2v
  arms again. The t2v arm ran last, so it came in a different position than
  in test 0.
- **Comparison:**
  - The mp4s: MD5 of the decoded streams (`ffmpeg -map 0:v|0:a -f md5`).
    The container bytes carry the graph.
  - The t2v latents: `torch.equal`, and rel L2 against test 0's latent.

## Result

| Mode | balanced, session A vs session B | balanced vs plain |
|---|---|---|
| i2va video | identical (`7e523131254a`) | differs (`0a28bdf4ba16`) |
| i2va audio | identical (`5d4b11408f69`) | differs (`a7158adbfc05`) |
| ref2va video | identical (`e3d980480602`) | differs (`ae609e114406`) |
| ref2va audio | identical (`731c36e75591`) | differs (`069382222ba4`) |
| t2v video latent | `torch.equal` to test 0 | rel L2 2.023e-01 |
| t2v audio latent | `torch.equal` to test 0 | rel L2 1.928e-01 |

The balanced output reproduces bit for bit across server processes and run
orders in all three modes, and plain differs in all three. **The quantizer
changes the render in every mode.**

The rel L2 is a trajectory divergence, not a quality figure: a rendered clip
cannot A/B a numerical change (`CLAUDE.md`).

## Why: the gate opens on every block

`bench/measure_qk_balance_gate_on_capture.py` reads the kernel's per-head
gate off captured K (top-4 channel energy share at least 0.2, the constants
imported from the installed kitchen). Record:
`2026-09-27_qk_balance_gate_on_capture.json`. It covers two t2v sage-chain
captures from 2026-09-19. Those are base-model activations, not PDD8 (see the
caveat under the grade below). The renders above are the current pipeline's
evidence, and these are the mechanism: covered_market (steps 8 and 15) and noodle_bar
(six steps), 56 heads per block.

- **Every captured block has open heads,** so balanced reaches Sol blocks
  the dense tail leaves on Sol.
- **Block 0 is the widest open** of the Sol blocks.
- **Blocks 44, 46 and 47 are next.**
- **Block 49 opens almost every head,** and 45 opens half.

The per-cell counts are in the JSON.

The earlier record did not support the audit's inference either:

- The block-49 page's table (`docs/attention/block49_quant_error.md`, "What it
  measured") moves blocks 0 and 40 at the default gate.
- `2026-09-15_channel_balance_kernel_b0_s15.json` has a `kernel` row that
  differs from its `plain` row.
- The phrase the audit leaned on, "blocks 45, 48 and 49 open on their own",
  named where the gate opens widely, not the only places it opens.

Dated notes now sit beside each of those sentences.

## What it decides

- **Bug #8 of the redesign** (the node default was off while every graph
  shipped on) **is closed by the redesign itself.** `quantizer` defaults to
  balanced (`sol_attn_h3.py::SOL_QUANTIZER_DEFAULT`), so the node default and
  the shipped state already agree.
- **Which quantizer is better is numerical,** so it is graded on captures
  (below). On the captures available it is `rotated`, not `balanced`, but
  those captures predate the shipped pipeline.
- **Scope.** Bit-identity answers "does it change output" for these three
  prompts and this seed. The gate measurement is what generalises it, and
  that covers two t2v scenes. i2va and ref2va activations were not captured.

## The quantizer grade, on the 2026-09-19 captures

`bench/grade_sol_quantizer_on_capture.py` runs the installed Sol kernel
(`0.2.35+sol.fc32da2.up.c8c7825`) under each quantizer on every head of a
captured cell. It reports the quantization term: rel L2 against the
unquantized Sol reference, over the dense norm. On heads whose gate is shut,
it asserts that balanced equals plain bit for bit, and that held on every
cell. Record: `2026-09-27_sol_quantizer_grade.json`, which has 13 cells:

- covered_market, step 15: blocks 0, 8, 16, 24, 32, 40, 44, 46 and 47.
  These are the Sol blocks behind the shipped dense tail.
- noodle_bar, step 4: blocks 0, 44, 46 and 47.

What it shows, per cell against plain:

- **`rotated` lowers the quantization term on all 13 cells.** It gains on
  every cell by a wider margin than `balanced` does on any.
- **`balanced` is within a few percent of `plain`,** and its sign changes
  from cell to cell. The one large move is noodle_bar block 46. On open
  heads alone it is still mixed.
- **`balanced+rotated` tracks `rotated`:** a little better on some cells, a
  little worse on others.

This agrees with audit §4 item 4 (rotation does most of balance's work),
which was measured on block 49. Here it holds on the Sol blocks themselves.

### What these captures are, and why this does not flip the default

The two 2026-09-19 captures are **base-model renders on main `48063012`**:

- no LoRA
- the sage chain
- kitchen `0.2.35+sol.8176242`

The shipped graphs are PDD8, and the LoRA moves the q/k projections the
quantizer sees. The code has also changed a great deal since (owner,
2026-09-27). So this grade is about those activations. The owner's
best-defaults rule wants a measured best on what ships, so the default stays
`balanced` until a capture of the current PDD8 graph is graded with the same
script. Test 2's capture session is where that happens. (Done the same
night: `2026-09-27_sol_redesign_test2.md`. PDD8 ranks the quantizers the
same way.)

Two more things to settle before `rotated` becomes the default:

- **Its kernel time,** which this does not measure.
- **The "all blocks" routing preset,** which requires a balanced quantizer
  (`sol_attn_h3.py::sol_routing_blocks`). With a rotated default it would
  need `balanced+rotated`, or the requirement would have to be re-derived
  (redesign bug #7).
