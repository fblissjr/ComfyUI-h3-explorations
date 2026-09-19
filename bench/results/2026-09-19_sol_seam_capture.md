# Captures work on the default kitchen chain: the hook moves to Sol's override seam

Date: 2026-09-19. Pack at `14f11b8a` (`b3a15bd1` the seam, `14f11b8a` the
removal of the old delegate-path capture), comfy-kitchen `0.2.35+sol.8176242`,
RTX 4090, memory compiler off on every run here. Scripts and logs:
`internal/2026-09-19_seam_acceptance/` (`run_all.sh`, `compare.sh`, the graphs,
`chain.log`, `compare.out`).

Why: the only capture hook lived in this pack's sage node
(`attention.py`), so the DEFAULT chain, which since 2026-09-15 carries core's
`ModelAttentionBackend` and no sage node, could not be captured at all. Every
capture-graded number in this repo is therefore from the sage chain, a caveat
carried by every record that uses one. Sol's override receives every DiT
attention call on both chains, because it chains onto whichever dense node came
first, so it is the one seam that covers both.

What changed, in one line each:

- `h3_capture.seam_begin` counts the call and, for a requested (block, step),
  copies q/k/v to the host BEFORE the kernel runs; `seam_end` writes the file
  after, tagged with the route that ran. The copy moved earlier because core
  hands the override single-owner containers that a backend may consume in
  place.
- The sage forward no longer captures the calls it hands to Sol; the seam owns
  them. Each call is counted once, by exactly one hook.
- The Sol node installs its block-index hook when capture is armed as well.
  Without that, a shipped graph published no `sol_block` and the seam would
  have captured nothing inside Sol's window, which is what the static check
  before this run caught.

## Acceptance, 14 checks, all green

`internal/2026-09-19_seam_acceptance/compare.sh`, output in `compare.out`
beside it. Three renders of the noodle bar at its declared 107 frames, seed
730451892, each on its own server.

| what it shows | how |
|---|---|
| The seam captures on the sage chain exactly what the old path did | 36 files against `2026-09-19_noodle_bar_sage_chain_107f`: every name present, and all 36 byte-identical in q/k/v and in the `kernel`, `block`, `step`, `seq_len`, `render`, `segments` and `sol_morton` fields |
| The hook FIRED inside Sol's window, so the match is not vacuous | 24 of those files carry `_ksol` at steps 8 and 15, which only the seam can write now that the delegate-path capture is gone; the 12 step-1 files are untagged, which only the sage forward writes |
| Arming is inert on the sage chain | the armed clip is md5-identical, video and audio, to this morning's armed sage render of the same graph |
| **The refactor changed no render** | the same graph the 2026-09-18 kitchen clip carries, rendered UNARMED on the new code, is md5-identical in video and audio to `Video/on_length/noodle_bar_plain_107f_*` |
| Arming is inert on the kitchen chain, never tested before | the armed kitchen clip is md5-identical to that unarmed render |
| Captures work on the kitchen chain at all | 36 files: 24 `_ksol` inside the window, 12 `_koutside_range` on the dense steps |

## The two sets this produced

- `2026-09-19_seam_acceptance_sage_107f`: scratch, keep until 2026-09-26. It
  exists to prove the hook.
- `2026-09-19_seam_acceptance_kitchen_107f`: **the first capture ever taken on
  the default chain.** Its 12 dense-step files are the kitchen INT8 kernel's
  own territory on the chain that actually ships, which no capture had held;
  `2026-09-19_depth_profile.md` had to grade the dense kernels on sage-chain
  inputs instead. Keep until 2026-10-31.

Both manifests are schema 1.7.0, whose `workload.attention.dense_node` names
the node that supplied the dense attention, read from the graph, so a file
tagged with a route REASON says which kernel ran
(`docs/capture_manifest_schema.md`).

## What this does not change

- **No number moves.** Nothing here re-grades anything; it makes a class of
  capture possible that was not.
- **The compiler must still be off for a capture run**
  (`--disable-comfy-compiler`): the host copy allocates inside the block's
  recorded scope, as the sage hook's always did. Said in `seam_begin`'s
  docstring.
- **Refiner calls are not captured on either chain** at this seam: they carry
  no `sol_block` label, and are neither counted nor written.
- **Unarmed cost is one attribute read** per call. Not measured as a render;
  the unarmed regression render above is the evidence that it changes nothing.
