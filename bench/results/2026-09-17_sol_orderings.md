# Token orderings for Sol-Attn: `3d` beats raster at equal cost; `2d_frame` and `hilbert` do not

Date: 2026-09-17. Model: MiniMax H3, int8 convrot checkpoint, base 16-step t2v
captures at full length (packed 104361, video grid 102 x 24 x 42). RTX 4090,
card alone, comfy-kitchen 0.2.34+sol.b532e28, first 8 heads, the node's own
sinks, shipped recipe (`qk_balance` on, pooled tail), tau swept over
0.6 to 2.0. Script: `bench/sweep_sol_orderings_on_capture.py`. Data:
`2026-09-17_sol_orderings.json`.

Why this was runnable now and not in August: `docs/morton.md` found that a
reorder moves the routing threshold with a sign that changes by block and
sigma, so no matched-tau A/B exists. With the kernel's own routed-block count
and the finding that a Sol call's time is proportional to routed density
(`2026-09-17_sol_stage_profile.md`), each ordering can be swept over tau and
compared as a curve of error against cost. Only the video rows are permuted,
by `sol_attn_h3._perm_for`, roll included; the reference is fp32 dense
attention permuted the same way.

Read at raster's shipped operating point (tau 1.0). "err" is the ordering's
error at raster's density (log-interpolated along its own curve); "dens" is the
density it needs to reach raster's error. Negative is better in both.

| cell | raster density / error | `3d` | `2d_frame` | `hilbert` |
|---|---|---|---|---|
| block 0, step 15 | 0.212 / 0.1249 | err -11.1%, dens -20.2% | err -10.4%, dens -20.8% | err -8.4%, dens -17.2% |
| block 24, step 15 | 0.215 / 0.0958 | err -27.1%, dens -30.4% | err +8.6%, dens +9.9% | err -6.1%, dens -6.5% |
| block 32, step 15 | 0.227 / 0.0922 | err -7.7%, dens -10.2% | err +9.2%, dens +10.5% | err +1.7%, dens +2.0% |
| block 40, step 15 | 0.202 / 0.2614 | err -19.4%, dens -32.5% | err -0.3%, dens -0.6% | err -7.8%, dens -11.9% |
| block 49, step 15 | 0.223 / 0.0377 | err -19.8%, dens -9.5% | err +105.6%, out of range | err +44.6%, dens +33.1% |
| block 32, step 4 | 0.222 / 0.0925 | err -2.9%, dens -4.4% | err -1.4%, dens -2.0% | err -3.3%, dens -4.5% |

Readings:

- `3d` is below raster on every captured cell: a few percent to about a
  quarter less error at equal cost, or up to about a third fewer routed blocks
  at equal error. Its advantage is smallest at the early step.
- `2d_frame` and `hilbert` are not improvements. Both lose to raster on block
  32, and badly on block 49, where ordering within a frame separates tokens
  that attend to each other across frames. Neither beats `3d` on any cell
  that matters. This contradicts the geometric argument `sol_curves.py` was
  written on, which `docs/morton.md` had already warned does not rank
  orderings.
- This supersedes the Triton-era note that Morton turned negative stacked on
  INT8: on the CUDA kernel at full length it is positive.

Limits: eight heads, six cells, one capture set; error against exact attention,
which in this pack has disagreed with the eye before. No clip has been rendered
with `morton` on at full length. The reorder also runs on the dense steps,
where it changes which tokens share an INT8 scale group in the dense kernel;
not measured. A code review the same day found the reorder composes with every
attention route but has one latent bug (per-token modulation indices are not
permuted, which matters only with a non-uniform video denoise mask) and no
test that runs with it on.
