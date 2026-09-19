# Where Sol-Attn's block grouping sends mass to the pooled branch, 2026-09-19

Model: MiniMax H3, int8 convrot checkpoint, base 16-step text to video at the trained canvas.

Script: `bench/analyze_sol_block_grouping.py`, whose docstring is the method.
Data: `2026-09-19_sol_block_grouping.json`. CPU only, fp32, tau 1.0, head prefix of
8 of 56, sink conditioning `exact_kv_and_rows`, block
sizes [64, 16], orderings ['raster', '3d']. Rows are tagged by
capture set: **A** = `2026-09-03_base16_t2v_1344x768`, **B** = `2026-09-10_sol_impl_courtroom`.

The same groupings painted on the latent grid, for the captures `qkv_L104361_S104361_b24_s15.pt`, `qkv_L104361_S104361_b49_s15.pt`, are under `internal/2026-09-19_block_grouping/` with their own README.
That directory is gitignored, so the pictures are local to the box that ran
this; `--figures DIR` rebuilds them from the capture.

## What this is and is not

The Sol ALGORITHM in fp32 and the exact attention it approximates, on captured
post-RoPE q/k/v. No INT8 term, no kernel, no clip.

**Missed mass is not output error.** It is the share of a query's exact
softmax mass that Sol routed through the POOLED branch rather than attending
exactly. Sol still supplies an approximation of that mass, and the fidelity of
the pooled term is a quantity nothing here measures, so a configuration can
push less mass into the pooled branch and still land further from dense
attention. Output error on these same captures lives in
`2026-09-18_sol_block_size.md`.

At block 64, missed mass and that record's output rel_l2 rank the two
orderings the same way on 7 of 9 cells, compared at the density plain
order routes at this tau.
They disagree on `2026-09-03_base16_t2v_1344x768/b40_s15`: rel_l2 moves by 0.803x under `3d` while missed mass moves by 1.056x.
They disagree on `2026-09-10_sol_impl_courtroom/b49_s13`: rel_l2 moves by 1.378x under `3d` while missed mass moves by 0.578x.

At block 16, missed mass and that record's output rel_l2 rank the two
orderings the same way on 9 of 9 cells, compared at the density plain
order routes at this tau.

Those cells are where this record must not be read as a statement about
error. Regenerate the comparison with `--cross-check`.

**Cells are at equal tau, not at equal cost.** The four (ordering, block size)
cells route at four different densities, printed in every table. In this data
they all land within a few percent of one another, so the confound is small,
but that is an observation about these captures rather than a property of tau.
`sweep_sol_block_size_on_capture.py` is the instrument that holds cost fixed by
construction. The rest of the instrument controls are in the JSON under
`instrument`.

## A. How unlike are the tokens that share a block?

Mean squared distance of a block's keys from its mean key over their mean
squared norm: 0 identical, about 1 unrelated. Video blocks only (the block
straddling the conditioning rows and the ragged last block are excluded).
The pooled term replaces every key in a block by that mean, so this is the
raw material of the key side.

| cell | raster_b64 key mean / p90 | raster_b16 key mean / p90 | 3d_b64 key mean / p90 | 3d_b16 key mean / p90 |
|---|---|---|---|---|
| A b0_s15 | 0.259 / 0.446 | 0.211 / 0.409 | 0.314 / 0.555 | 0.229 / 0.405 |
| A b24_s15 | 0.469 / 0.632 | 0.392 / 0.551 | 0.414 / 0.532 | 0.322 / 0.438 |
| A b32_s15 | 0.427 / 0.598 | 0.356 / 0.516 | 0.356 / 0.522 | 0.293 / 0.443 |
| A b40_s15 | 0.485 / 0.673 | 0.418 / 0.590 | 0.432 / 0.593 | 0.371 / 0.530 |
| A b32_s4 | 0.375 / 0.483 | 0.316 / 0.425 | 0.315 / 0.435 | 0.272 / 0.386 |
| B b0_s13 | 0.295 / 0.518 | 0.247 / 0.478 | 0.336 / 0.593 | 0.258 / 0.475 |
| B b24_s13 | 0.486 / 0.644 | 0.409 / 0.555 | 0.424 / 0.530 | 0.341 / 0.454 |
| B b49_s13 | 0.200 / 0.306 | 0.139 / 0.238 | 0.131 / 0.235 | 0.090 / 0.175 |
| A b49_s15 | 0.206 / 0.308 | 0.144 / 0.243 | 0.136 / 0.240 | 0.095 / 0.191 |

| cell | raster_b64 query mean / p90 | raster_b16 query mean / p90 | 3d_b64 query mean / p90 | 3d_b16 query mean / p90 |
|---|---|---|---|---|
| A b0_s15 | 0.246 / 0.595 | 0.167 / 0.423 | 0.195 / 0.310 | 0.124 / 0.171 |
| A b24_s15 | 0.509 / 0.638 | 0.425 / 0.566 | 0.448 / 0.576 | 0.354 / 0.481 |
| A b32_s15 | 0.399 / 0.505 | 0.337 / 0.456 | 0.338 / 0.457 | 0.283 / 0.413 |
| A b40_s15 | 0.367 / 0.468 | 0.323 / 0.428 | 0.340 / 0.431 | 0.297 / 0.398 |
| A b32_s4 | 0.391 / 0.482 | 0.333 / 0.428 | 0.328 / 0.420 | 0.288 / 0.391 |
| B b0_s13 | 0.261 / 0.596 | 0.185 / 0.429 | 0.217 / 0.340 | 0.147 / 0.193 |
| B b24_s13 | 0.525 / 0.666 | 0.442 / 0.583 | 0.465 / 0.565 | 0.382 / 0.495 |
| B b49_s13 | 0.199 / 0.329 | 0.152 / 0.237 | 0.161 / 0.246 | 0.121 / 0.190 |
| A b49_s15 | 0.210 / 0.317 | 0.163 / 0.240 | 0.181 / 0.265 | 0.137 / 0.210 |

## B. Does pooling hide mass the query wanted?

Sampled video query tokens, stratified over latent frames, the same physical
tokens in every cell. `missed` is the share of the query's EXACT softmax mass
sitting in key blocks its query block did not route. `top` is the share of a
pooled block's mass carried by the top eighth of its tokens, mass-weighted.
`density` is the share of key blocks routed, and moves with the cell.

| cell | raster_b64 density / missed mean / missed p90 / top | raster_b16 density / missed mean / missed p90 / top | 3d_b64 density / missed mean / missed p90 / top | 3d_b16 density / missed mean / missed p90 / top |
|---|---|---|---|---|
| A b0_s15 | 0.212 / 0.3373 / 0.8127 / 0.555 | 0.215 / 0.3078 / 0.7418 / 0.484 | 0.218 / 0.3275 / 0.7529 / 0.522 | 0.216 / 0.2688 / 0.6996 / 0.436 |
| A b24_s15 | 0.215 / 0.1472 / 0.4103 / 0.939 | 0.217 / 0.1047 / 0.3016 / 0.894 | 0.224 / 0.1177 / 0.3511 / 0.922 | 0.221 / 0.0824 / 0.2521 / 0.870 |
| A b32_s15 | 0.227 / 0.1654 / 0.4469 / 0.950 | 0.229 / 0.1277 / 0.3646 / 0.906 | 0.236 / 0.1593 / 0.4653 / 0.941 | 0.235 / 0.1076 / 0.3153 / 0.893 |
| A b40_s15 | 0.202 / 0.2388 / 0.4772 / 0.975 | 0.195 / 0.2045 / 0.4086 / 0.959 | 0.198 / 0.2523 / 0.5957 / 0.974 | 0.197 / 0.1802 / 0.4059 / 0.954 |
| A b32_s4 | 0.222 / 0.2111 / 0.6085 / 0.889 | 0.225 / 0.1776 / 0.5447 / 0.848 | 0.226 / 0.1987 / 0.6002 / 0.881 | 0.228 / 0.1633 / 0.5235 / 0.842 |
| B b0_s13 | 0.217 / 0.3578 / 0.7924 / 0.562 | 0.222 / 0.3223 / 0.7453 / 0.488 | 0.225 / 0.3237 / 0.7310 / 0.508 | 0.223 / 0.2735 / 0.6991 / 0.429 |
| B b24_s13 | 0.220 / 0.1821 / 0.4996 / 0.961 | 0.220 / 0.1310 / 0.3725 / 0.928 | 0.222 / 0.1403 / 0.4195 / 0.946 | 0.219 / 0.1098 / 0.3320 / 0.915 |
| B b49_s13 | 0.219 / 0.0483 / 0.1492 / 0.668 | 0.215 / 0.0350 / 0.1212 / 0.517 | 0.221 / 0.0279 / 0.1090 / 0.524 | 0.214 / 0.0218 / 0.0880 / 0.403 |
| A b49_s15 | 0.223 / 0.0437 / 0.1172 / 0.812 | 0.217 / 0.0289 / 0.0833 / 0.678 | 0.231 / 0.0191 / 0.0551 / 0.702 | 0.220 / 0.0142 / 0.0424 / 0.593 |

Tokens needed to cover 90 percent of a query's exact mass, against the tokens
Sol actually attended exactly. The first is a property of the query and does
not move with the cell; the second is the cell's cost.

| cell | tokens for 90% (median / p90) | raster_b64 routed tokens | raster_b16 routed tokens | 3d_b64 routed tokens | 3d_b16 routed tokens |
|---|---|---|---|---|---|
| A b0_s15 | 25142 / 49063 | 21144 | 21526 | 21807 | 21657 |
| A b24_s15 | 263 / 4755 | 21438 | 21631 | 22447 | 22102 |
| A b32_s15 | 303 / 4781 | 22692 | 22946 | 23784 | 23665 |
| A b40_s15 | 234 / 1858 | 20222 | 19391 | 19728 | 19563 |
| A b32_s4 | 972 / 13326 | 22250 | 22586 | 22647 | 22847 |
| B b0_s13 | 22680 / 53580 | 21787 | 22267 | 22591 | 22377 |
| B b24_s13 | 345 / 3384 | 22056 | 22059 | 22239 | 21907 |
| B b49_s13 | 161 / 15609 | 21885 | 21460 | 22193 | 21358 |
| A b49_s15 | 324 / 5558 | 22358 | 21711 | 23165 | 21985 |

## C. Do the queries of one block want the same keys?

Each individual query's own ideal key-block set (top-k by its exact mass, k =
what its block routed) against the block's actual routed set, and against its
neighbours'. Jaccard has a chance floor at these densities, printed beside
every number. `disc` removes the forced blocks (diagonal and sinks) from both
sets and from k; those agree by construction, so `disc` is the number that
answers the question.

| cell | raster_b64 q-vs-block disc (chance) / q-vs-q disc | raster_b16 q-vs-block disc (chance) / q-vs-q disc | 3d_b64 q-vs-block disc (chance) / q-vs-q disc | 3d_b16 q-vs-block disc (chance) / q-vs-q disc |
|---|---|---|---|---|
| A b0_s15 | 0.685 (0.104) / 0.689 | 0.621 (0.106) / 0.640 | 0.590 (0.107) / 0.583 | 0.640 (0.108) / 0.659 |
| A b24_s15 | 0.472 (0.105) / 0.411 | 0.479 (0.107) / 0.405 | 0.482 (0.111) / 0.426 | 0.528 (0.110) / 0.454 |
| A b32_s15 | 0.472 (0.113) / 0.459 | 0.454 (0.115) / 0.417 | 0.470 (0.119) / 0.487 | 0.516 (0.121) / 0.498 |
| A b40_s15 | 0.385 (0.098) / 0.360 | 0.363 (0.092) / 0.322 | 0.382 (0.095) / 0.382 | 0.406 (0.097) / 0.383 |
| A b32_s4 | 0.473 (0.110) / 0.450 | 0.460 (0.113) / 0.413 | 0.482 (0.113) / 0.473 | 0.515 (0.116) / 0.475 |
| B b0_s13 | 0.661 (0.107) / 0.628 | 0.612 (0.110) / 0.599 | 0.600 (0.111) / 0.572 | 0.639 (0.111) / 0.633 |
| B b24_s13 | 0.440 (0.108) / 0.411 | 0.462 (0.110) / 0.403 | 0.449 (0.109) / 0.417 | 0.492 (0.109) / 0.433 |
| B b49_s13 | 0.648 (0.107) / 0.602 | 0.599 (0.106) / 0.543 | 0.649 (0.109) / 0.604 | 0.676 (0.106) / 0.631 |
| A b49_s15 | 0.639 (0.111) / 0.602 | 0.587 (0.108) / 0.531 | 0.657 (0.116) / 0.610 | 0.684 (0.110) / 0.630 |

| cell | raster_b64 q-vs-block all (chance) | raster_b16 q-vs-block all (chance) | 3d_b64 q-vs-block all (chance) | 3d_b16 q-vs-block all (chance) |
|---|---|---|---|---|
| A b0_s15 | 0.646 (0.113) | 0.587 (0.114) | 0.560 (0.115) | 0.608 (0.116) |
| A b24_s15 | 0.471 (0.114) | 0.478 (0.115) | 0.484 (0.120) | 0.526 (0.118) |
| A b32_s15 | 0.478 (0.121) | 0.462 (0.123) | 0.482 (0.128) | 0.523 (0.129) |
| A b40_s15 | 0.399 (0.106) | 0.376 (0.100) | 0.398 (0.103) | 0.418 (0.104) |
| A b32_s4 | 0.477 (0.118) | 0.465 (0.121) | 0.490 (0.121) | 0.519 (0.124) |
| B b0_s13 | 0.620 (0.116) | 0.577 (0.118) | 0.567 (0.120) | 0.605 (0.120) |
| B b24_s13 | 0.441 (0.117) | 0.464 (0.118) | 0.453 (0.119) | 0.494 (0.117) |
| B b49_s13 | 0.632 (0.116) | 0.585 (0.114) | 0.636 (0.118) | 0.658 (0.114) |
| A b49_s15 | 0.620 (0.120) | 0.571 (0.116) | 0.641 (0.124) | 0.663 (0.118) |

## D. Would a better rule at the same granularity do it?

Mean missed mass under four ranking rules, each keeping the same NUMBER of key
blocks Sol kept, on the same sampled queries. `sol` is the shipped centroid
against centred centroid. `quest` is the per-channel min/max upper bound, a
rule a kernel could run. `blockmax` ranks by the TRUE largest token score in
the block, which that bound relaxes, so it separates a loose bound from a
wrong objective. `oracle` ranks by the query's own exact mass per block: the
floor any rule at this granularity could reach for that query.

| cell | raster_b64 sol / quest / blockmax / oracle | raster_b16 sol / quest / blockmax / oracle | 3d_b64 sol / quest / blockmax / oracle | 3d_b16 sol / quest / blockmax / oracle |
|---|---|---|---|---|
| A b0_s15 | 0.3373 / 0.3789 / 0.3335 / 0.3121 | 0.3078 / 0.3419 / 0.2939 / 0.2509 | 0.3275 / 0.3641 / 0.3144 / 0.2521 | 0.2688 / 0.3031 / 0.2530 / 0.2200 |
| A b24_s15 | 0.1472 / 0.1968 / 0.1107 / 0.0368 | 0.1047 / 0.1196 / 0.0618 / 0.0199 | 0.1177 / 0.1146 / 0.0707 / 0.0253 | 0.0824 / 0.0751 / 0.0428 / 0.0157 |
| A b32_s15 | 0.1654 / 0.2172 / 0.1339 / 0.0392 | 0.1277 / 0.1425 / 0.0759 / 0.0213 | 0.1593 / 0.1866 / 0.0861 / 0.0285 | 0.1076 / 0.1086 / 0.0560 / 0.0163 |
| A b40_s15 | 0.2388 / 0.3444 / 0.2168 / 0.0312 | 0.2045 / 0.2444 / 0.1213 / 0.0190 | 0.2523 / 0.3267 / 0.1413 / 0.0247 | 0.1802 / 0.1929 / 0.0786 / 0.0162 |
| A b32_s4 | 0.2111 / 0.2713 / 0.1749 / 0.0910 | 0.1776 / 0.1977 / 0.1174 / 0.0617 | 0.1987 / 0.2182 / 0.1361 / 0.0800 | 0.1633 / 0.1625 / 0.1033 / 0.0575 |
| B b0_s13 | 0.3578 / 0.3872 / 0.3551 / 0.3298 | 0.3223 / 0.3502 / 0.3099 / 0.2572 | 0.3237 / 0.3502 / 0.3095 / 0.2500 | 0.2735 / 0.2973 / 0.2573 / 0.2231 |
| B b24_s13 | 0.1821 / 0.2131 / 0.1483 / 0.0358 | 0.1310 / 0.1181 / 0.0698 / 0.0170 | 0.1403 / 0.1375 / 0.0842 / 0.0229 | 0.1098 / 0.0845 / 0.0454 / 0.0133 |
| B b49_s13 | 0.0483 / 0.1691 / 0.0475 / 0.0334 | 0.0350 / 0.1042 / 0.0335 / 0.0221 | 0.0279 / 0.0797 / 0.0279 / 0.0222 | 0.0218 / 0.0450 / 0.0215 / 0.0180 |
| A b49_s15 | 0.0437 / 0.2121 / 0.0416 / 0.0233 | 0.0289 / 0.1181 / 0.0265 / 0.0136 | 0.0191 / 0.0708 / 0.0186 / 0.0111 | 0.0142 / 0.0371 / 0.0140 / 0.0089 |

## E. Does the routed set survive from one step to the next?

Jaccard of the routed key-block sets per video query block and head, between
two captured steps of one DiT block. `disc` excludes the forced blocks, which
are identical by construction. Chance floors in brackets.

| capture set | DiT block | steps | cell | density | J all (chance) | J disc (chance) |
|---|---|---|---|---|---|---|
| A | 32 | 4-15 | raster_b64 | 0.213/0.217 | 0.527 (0.121) | 0.496 (0.112) |
| A | 32 | 4-15 | raster_b16 | 0.217/0.220 | 0.483 (0.122) | 0.453 (0.115) |
| A | 32 | 4-15 | 3d_b64 | 0.217/0.228 | 0.538 (0.125) | 0.509 (0.117) |
| A | 32 | 4-15 | 3d_b16 | 0.219/0.227 | 0.491 (0.125) | 0.462 (0.117) |
| B | 0 | 5-6 | raster_b64 | 0.219/0.218 | 0.921 (0.123) | 0.908 (0.114) |
| B | 0 | 5-6 | raster_b16 | 0.222/0.221 | 0.888 (0.125) | 0.876 (0.116) |
| B | 0 | 5-6 | 3d_b64 | 0.223/0.222 | 0.934 (0.125) | 0.927 (0.116) |
| B | 0 | 5-6 | 3d_b16 | 0.222/0.221 | 0.896 (0.124) | 0.886 (0.116) |
| B | 0 | 5-12 | raster_b64 | 0.219/0.211 | 0.779 (0.120) | 0.748 (0.111) |
| B | 0 | 5-12 | raster_b16 | 0.222/0.215 | 0.743 (0.123) | 0.719 (0.114) |
| B | 0 | 5-12 | 3d_b64 | 0.223/0.217 | 0.791 (0.124) | 0.770 (0.114) |
| B | 0 | 5-12 | 3d_b16 | 0.222/0.216 | 0.754 (0.123) | 0.732 (0.114) |
| B | 0 | 5-13 | raster_b64 | 0.219/0.208 | 0.748 (0.120) | 0.715 (0.111) |
| B | 0 | 5-13 | raster_b16 | 0.222/0.213 | 0.715 (0.122) | 0.689 (0.113) |
| B | 0 | 5-13 | 3d_b64 | 0.223/0.216 | 0.757 (0.123) | 0.734 (0.114) |
| B | 0 | 5-13 | 3d_b16 | 0.222/0.214 | 0.725 (0.122) | 0.702 (0.114) |
| B | 0 | 6-12 | raster_b64 | 0.218/0.211 | 0.801 (0.120) | 0.769 (0.111) |
| B | 0 | 6-12 | raster_b16 | 0.221/0.215 | 0.766 (0.122) | 0.744 (0.114) |
| B | 0 | 6-12 | 3d_b64 | 0.222/0.217 | 0.810 (0.123) | 0.790 (0.114) |
| B | 0 | 6-12 | 3d_b16 | 0.221/0.216 | 0.776 (0.123) | 0.756 (0.114) |
| B | 0 | 6-13 | raster_b64 | 0.218/0.208 | 0.767 (0.119) | 0.733 (0.110) |
| B | 0 | 6-13 | raster_b16 | 0.221/0.213 | 0.735 (0.122) | 0.710 (0.113) |
| B | 0 | 6-13 | 3d_b64 | 0.222/0.216 | 0.775 (0.123) | 0.752 (0.114) |
| B | 0 | 6-13 | 3d_b16 | 0.221/0.214 | 0.745 (0.122) | 0.723 (0.113) |
| B | 0 | 12-13 | raster_b64 | 0.211/0.208 | 0.930 (0.117) | 0.907 (0.108) |
| B | 0 | 12-13 | raster_b16 | 0.215/0.213 | 0.911 (0.120) | 0.899 (0.111) |
| B | 0 | 12-13 | 3d_b64 | 0.217/0.216 | 0.932 (0.121) | 0.922 (0.112) |
| B | 0 | 12-13 | 3d_b16 | 0.216/0.214 | 0.917 (0.120) | 0.907 (0.112) |
| B | 24 | 5-6 | raster_b64 | 0.212/0.213 | 0.866 (0.119) | 0.854 (0.110) |
| B | 24 | 5-6 | raster_b16 | 0.214/0.214 | 0.807 (0.120) | 0.791 (0.111) |
| B | 24 | 5-6 | 3d_b64 | 0.216/0.216 | 0.829 (0.121) | 0.814 (0.112) |
| B | 24 | 5-6 | 3d_b16 | 0.214/0.214 | 0.784 (0.120) | 0.767 (0.111) |
| B | 24 | 5-12 | raster_b64 | 0.212/0.212 | 0.634 (0.119) | 0.606 (0.109) |
| B | 24 | 5-12 | raster_b16 | 0.214/0.212 | 0.579 (0.119) | 0.550 (0.111) |
| B | 24 | 5-12 | 3d_b64 | 0.216/0.213 | 0.604 (0.120) | 0.575 (0.111) |
| B | 24 | 5-12 | 3d_b16 | 0.214/0.211 | 0.554 (0.119) | 0.524 (0.110) |
| B | 24 | 5-13 | raster_b64 | 0.212/0.211 | 0.605 (0.118) | 0.575 (0.109) |
| B | 24 | 5-13 | raster_b16 | 0.214/0.212 | 0.553 (0.119) | 0.523 (0.110) |
| B | 24 | 5-13 | 3d_b64 | 0.216/0.213 | 0.582 (0.120) | 0.551 (0.111) |
| B | 24 | 5-13 | 3d_b16 | 0.214/0.210 | 0.532 (0.118) | 0.501 (0.110) |
| B | 24 | 6-12 | raster_b64 | 0.213/0.212 | 0.675 (0.119) | 0.649 (0.110) |
| B | 24 | 6-12 | raster_b16 | 0.214/0.212 | 0.621 (0.119) | 0.594 (0.111) |
| B | 24 | 6-12 | 3d_b64 | 0.216/0.213 | 0.646 (0.120) | 0.619 (0.111) |
| B | 24 | 6-12 | 3d_b16 | 0.214/0.211 | 0.596 (0.119) | 0.568 (0.110) |
| B | 24 | 6-13 | raster_b64 | 0.213/0.211 | 0.641 (0.119) | 0.613 (0.110) |
| B | 24 | 6-13 | raster_b16 | 0.214/0.212 | 0.590 (0.119) | 0.562 (0.111) |
| B | 24 | 6-13 | 3d_b64 | 0.216/0.213 | 0.619 (0.120) | 0.591 (0.111) |
| B | 24 | 6-13 | 3d_b16 | 0.214/0.210 | 0.569 (0.118) | 0.540 (0.110) |
| B | 24 | 12-13 | raster_b64 | 0.212/0.211 | 0.917 (0.118) | 0.909 (0.109) |
| B | 24 | 12-13 | raster_b16 | 0.212/0.212 | 0.891 (0.118) | 0.881 (0.110) |
| B | 24 | 12-13 | 3d_b64 | 0.213/0.213 | 0.912 (0.119) | 0.904 (0.110) |
| B | 24 | 12-13 | 3d_b16 | 0.211/0.210 | 0.887 (0.117) | 0.877 (0.109) |
| B | 49 | 5-6 | raster_b64 | 0.230/0.228 | 0.922 (0.130) | 0.916 (0.120) |
| B | 49 | 5-6 | raster_b16 | 0.222/0.220 | 0.893 (0.124) | 0.884 (0.115) |
| B | 49 | 5-6 | 3d_b64 | 0.208/0.207 | 0.894 (0.116) | 0.884 (0.107) |
| B | 49 | 5-6 | 3d_b16 | 0.210/0.209 | 0.879 (0.117) | 0.869 (0.108) |
| B | 49 | 5-12 | raster_b64 | 0.230/0.213 | 0.742 (0.124) | 0.722 (0.115) |
| B | 49 | 5-12 | raster_b16 | 0.222/0.208 | 0.723 (0.120) | 0.703 (0.112) |
| B | 49 | 5-12 | 3d_b64 | 0.208/0.211 | 0.709 (0.117) | 0.686 (0.108) |
| B | 49 | 5-12 | 3d_b16 | 0.210/0.205 | 0.701 (0.116) | 0.679 (0.107) |
| B | 49 | 5-13 | raster_b64 | 0.230/0.210 | 0.708 (0.123) | 0.686 (0.114) |
| B | 49 | 5-13 | raster_b16 | 0.222/0.206 | 0.691 (0.119) | 0.669 (0.111) |
| B | 49 | 5-13 | 3d_b64 | 0.208/0.212 | 0.681 (0.117) | 0.656 (0.108) |
| B | 49 | 5-13 | 3d_b16 | 0.210/0.205 | 0.673 (0.116) | 0.649 (0.107) |
| B | 49 | 6-12 | raster_b64 | 0.228/0.213 | 0.770 (0.124) | 0.752 (0.115) |
| B | 49 | 6-12 | raster_b16 | 0.220/0.208 | 0.750 (0.120) | 0.731 (0.111) |
| B | 49 | 6-12 | 3d_b64 | 0.207/0.211 | 0.739 (0.117) | 0.718 (0.108) |
| B | 49 | 6-12 | 3d_b16 | 0.209/0.205 | 0.730 (0.115) | 0.709 (0.107) |
| B | 49 | 6-13 | raster_b64 | 0.228/0.210 | 0.733 (0.123) | 0.712 (0.114) |
| B | 49 | 6-13 | raster_b16 | 0.220/0.206 | 0.715 (0.119) | 0.694 (0.110) |
| B | 49 | 6-13 | 3d_b64 | 0.207/0.212 | 0.707 (0.117) | 0.684 (0.108) |
| B | 49 | 6-13 | 3d_b16 | 0.209/0.205 | 0.697 (0.115) | 0.675 (0.106) |
| B | 49 | 12-13 | raster_b64 | 0.213/0.210 | 0.934 (0.118) | 0.928 (0.109) |
| B | 49 | 12-13 | raster_b16 | 0.208/0.206 | 0.923 (0.115) | 0.916 (0.107) |
| B | 49 | 12-13 | 3d_b64 | 0.211/0.212 | 0.933 (0.118) | 0.927 (0.109) |
| B | 49 | 12-13 | 3d_b16 | 0.205/0.205 | 0.923 (0.114) | 0.916 (0.105) |

## What this says

**There are two regimes and they want different things.** On DiT block 0 the
attention is diffuse: the median query needs more than twenty thousand
individual key tokens to cover 90 percent of its mass, which is about what Sol
routes anyway, and a pooled block's mass is spread across its tokens rather
than concentrated. On blocks 24, 32, 40 and 49 the median query needs a few
hundred tokens while Sol routes tens of thousands, and a pooled block carries
almost all of its mass in the top eighth of its tokens. Every statement below
is about the second regime; nothing measured here improves the first.

**The three levers do not rest on the same evidence.** The block-size lever is
corroborated by output error on these same captures (`2026-09-18_sol_block_size.md`,
where block 16 is lower on nearly every cell). The query-side and routing-rule
levers below rest on missed mass alone, which is not error. Treat them as
well-motivated directions to measure next, not as established gains.

**The block size is the smallest of the three levers.** Going from 64 to 16 at
the same tau lowers missed mass to roughly three quarters of the block-64
figure. Taking the SAME number of blocks but ranking them by the true largest
token score in each block rather than by the centroid score lowers it further
on the mid blocks. Taking the same number of blocks but letting each query
choose its own set -- the per-query oracle, which no per-block rule can beat --
lowers it to a small fraction of Sol's on those blocks. That ordering is stable
across both clips.

**The query side is where the headroom is, and C says so from the other
direction.** A block's own queries agree with the block's discretionary routed
set at a Jaccard well above the chance floor and far from 1, and they agree
with each other no better. The block with the lowest agreement is the block
where the per-query oracle gains the most. One routing decision per 64 queries
is not right for all 64 of them.

**The Quest bound is the wrong rule here, and that is not an implementation
defect.** The bound dominates the true block maximum on every sampled pair, as
it must, and still ranks worse than the shipped centroid rule on most cells.
The per-channel min/max relaxation is too loose at head dimension 128 over 64
keys: it rewards a block for being wide rather than for being relevant. The
`blockmax` arm is the same objective computed exactly and it does beat the
centroid rule, so the objective is fine and the relaxation is not.

**Routing decisions survive between steps.** Adjacent captured steps keep most
of the routed set, steps six to eight apart keep well over half, and the widest
pair measured -- eleven steps -- still keeps about half, against a chance floor
near an eighth. The `3d` order neither helps nor hurts this.

**What follows is inference, not measurement.** The per-query oracle is a
bound, not a rule: it is what a decision taken per query could reach. The cheap
approximation to it -- splitting a query block into two or four sub-blocks that
route separately against the same key blocks -- is NOT measured here and should
be the next thing measured, because it costs routing work linear in the split
and no change to the key-side layout. `blockmax` is not a rule either:
computing a true per-block maximum per query block is a full score pass. What
it argues is that a per-block key SUMMARY approximating a maximum rather than a
mean is worth designing, and that E is what would pay for a summary too
expensive to rebuild every step. Finer blocks remain the option that needs a
kernel rewrite and buys the least of the three, and their routing cost, which
grows with the square of the block count, is in none of these numbers.

## Limits

Two clips, the captured DiT blocks and steps only. A head PREFIX of the first
8 of 56, not a sample. Equal tau, not equal cost. fp32 with no INT8 term, so
every number is a ceiling on what a grouping change could buy and not a
forecast of a kernel. Routing COST, which grows with the square of the block
count, is in nothing here, so block 16's numbers do not carry their own price.
**Missed mass is not output error**, and on the cells named above the two rank
the orderings oppositely; the pooled term's own fidelity is measured nowhere
here. These are properties of one attention call on captured inputs, not of a
rendered clip, and this pack has a standing example of error against exact
attention and a watched clip disagreeing.
