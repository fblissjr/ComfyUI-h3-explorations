# The Sol redesign is output-neutral: bit-identical latents (2026-09-27)

Test 0 of the Sol node redesign
(`docs/wiki/archive/2026-09-27_sol_node_redesign.md`).

## What ran

- **Manifest:** `bench/sol_redesign_bitid_arms.json`. It holds the first two
  arms of `bench/int8attn_after_arms.json`, in the same order with the same
  seed (730451892) and patches: the PDD8 warmup at 124 frames, then
  slapstick_moving_piano at 345 frames. Rows are in
  `2026-09-27_sol_redesign_bitid.jsonl`.
- **Server:** a fresh one on main `f69087d4` (0.162.0, `MiniMaxH3Sol`),
  kitchen `0.2.35+sol.fc32da2.up.c8c7825`. The first render after start was
  the warmup.
- **Reference:** the 11:04 render of the same arm in
  `2026-09-27_int8attn_after.jsonl`, on `MiniMaxH3SolAttn` and kitchen
  `0.2.35+sol.863e953.up.c8c7825`. Its latents are
  `latents/text_to_video_pdd_savelat_{video,audio}_slapstick_moving_piano__pdd8_00002_.latent`
  under the output root; this run's are `..._00003_`.
- **The graph:** it differs from the reference run's graph
  (`workflows/distill_experiments/h3_text_to_video_pdd_savelat_api.json` at
  `d45e6789`) only in the Sol node. The settings map one to one:

  | Old node | New node |
  |---|---|
  | `selection` adaptive tau, `selection.tau` 1.0 | `tau` 1.0 |
  | `qk_balance` True, `rotate` False | `quantizer` balanced |
  | `pooled_tail` True | none: the tail is always on |
  | `morton` False | none: Morton is retired |

  Every other input is unchanged.
- **The prompt** was stripped by `run_graph_arms.py`'s `@bank:` handling on
  both runs, so the edge-whitespace change in `a5ab229f` does not touch it.

## Result

| Latent | Tensor | Result |
|---|---|---|
| video | `latent_tensor` (1, 24, 102, 48, 84) float32 | **bit-identical** (`torch.equal`) |
| audio | `latent_tensor` (1, 32, 2, 575) float32 | **bit-identical** |

The files differ only in the embedded `prompt` metadata, which is the graph
itself: 5877 against 5738 bytes, because the new node has fewer inputs.

Sampler time on the slapstick arm: 263.1 s here, against 262.7 s in the
reference row. Both are single renders, so this is not a timing comparison.

## What it establishes

1. **The redesign changes no output at the shipped settings.** The new node
   passes the kernel the same arguments the old one did. Kitchen `fc32da2`
   changes only the token stage, which is off here.
2. **PDD8 on the exact branch reproduces bit for bit across server
   processes.** That was unchecked until now (fastdude, 2026-09-27). It
   covers this seed and scene, after a warmup, with the same load order.

It does not establish anything about settings the graphs do not ship: token
routing, the other quantizers, or `exact_kv_and_all_rows`.
