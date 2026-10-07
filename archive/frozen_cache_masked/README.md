# The frozen-row cache's masked-window use, retired 2026-10-06, restored 2026-10-07

last updated: 2026-10-07

**2026-10-07: the masked use is live again.** The owner restored the
module, the check, the generator's `masked_cache` and the probe graph from
25103f0e, with one change on top (the cached block leaves the attention
override in place). `frozen_video_cache.py` and
`bench/check_frozen_video_cache.py` at the repo's root are the live files;
this folder is the record of the retirement and of what was tried on the
way back, and nothing reads it. "Why it was retired" below is history. The
diff kept here, `dense_options_fix_2026-10-07.diff`, is the change the
corrected pair ran with; it is not what landed: on Sol-Attn's real override
its chain walk goes nowhere and it leaves the override in place, which is
what the live module now does in one line.

Not used, not imported, registered nowhere, in no workflow, walked by no
check. `archive/` is the folder the wiki defines that way
(`docs/wiki/index.md`), and `bench/check_graph_discovery.py` and the import
checks are the observable. Nothing here runs; `git log` on these paths and
on the live files is the history.

## What this was

`MiniMaxH3FrozenVideoCache` (`frozen_video_cache.py`, ported from
ComfyUI-H3-AudioRefine) caches the frozen rows of a sampling run so each
later step computes only the live rows. Its first use is the audio-refine
pass, where the video is frozen whole and the audio reopened; that use
stays live in the pack and is measured positive: the refine sampler 72.0 s
cached against 143.4 s uncached, the video untouched, the finished audio at
cosine 0.9956 (`bench/results/2026-09-25_frozen_cache_s1.md`).

From 1d2d776d to 25103f0e it had a second use: a masked video-to-video
window, where the audio, the previous window's tail and every video token
outside the subject are kept and a region is regenerated. For that it
gained a `halo` input (kept rows recomputed each step around the region),
a live-share gate, `verify`'s ring-and-interior split of the video error, a
second verify pass with the text cached, the generator's `masked_cache`
argument and one probe graph. The three files here are the module, its
check and that graph exactly as they stood at 25103f0e, before the cut.

## Why it was retired

The first run on a masked window
(`bench/results/2026-10-06_frozen_cache_masked_window.md`): the cache
engaged cleanly (one build, eleven cached steps, no rebuild, no decline,
the stock render deterministic bit for bit, no kept row moved) and saved
nothing. A plain cached step was 36.75 s against 33.6 s for a Sol-Attn
stock step and 53.4 s for a dense one; on the window's twelve steps,
463.8 s cached against 462.9 s stock. The band fixed before the run called
that "not worth its approximation on the lane's most favourable window".
The attention call for live queries against every key was 29 s of a 38 s
timed step, more than twice what the design allowed.

The owner, 2026-10-06, on that result: "if it provides negative value
deprecate it but cite why and preserve the code somewhere that's clear it's
not used anywhere or imported or in workflows or nodes code", and on the
split: "keep it audio refine".

## What the live module kept, and why

- The gate's other guards (a context window, a mask off the layout's grid,
  the kept content's sample) and the per-row reading of the audio mask:
  each protects the refine pass as much (mrhand, 2026-10-06, at the cut).
- `LIVE_SHARE_LIMIT`, under refine-pass provenance: it has never fired
  there (the live share is under a fiftieth) and exists for a short clip
  with a long prompt.
- `verify`'s stage timers: the stage-split record is a refine run and reads
  them.

Removed from the live module: `halo` (node input, `h3_config`, `_dilate`,
the halo rows), the video half of `_compare` with `RING_TOKENS`, the
text-cached verify pass, and the gate now sends any call that regenerates a
video row stock with the reason. The check's items 9 to 16 (the masked use)
became one item holding the retirement and the guards on the refine layout.

## 2026-10-07: the run measured torch's kernel, not kitchen's

Read this before "Why it was retired" is taken as a verdict on the design.
The kernel was timed alone the next morning
(`bench/results/2026-10-07_frozen_cache_rectangle_kernel.md`). On the
kitchen int8 kernel the rectangle costs the square's rate per query-key
pair, so the kernel is not slow off the square. The cached block never
reached it: `_dense_options` removes `optimized_attention_override`, which
is where core's Model Attention Backend node puts the kitchen kernel (Sol
chains onto it), and the checkpoint names no backend of its own
(`ComfyAttention.function` is `None`), so the call falls to core's default,
torch's `scaled_dot_product_attention`. Torch's kernel timed alone at the
window's sizes reproduces the attention stage the run measured. The saving
the cost model predicted was therefore never tested. Neither was the
approximation: `verify` compared a cached step on exact attention with a
stock step on Sol's sparse int8 attention, so its figures hold two
differences at once.

The fix is in this module's `_cached_block`: send the call to the dense
backend the override chain carries, not past it. Nothing here is changed;
the retirement is the owner's decision and stands until they reopen it.

**The same day, with the owner's go: the pair was run again with that fix.**
`dense_options_fix_2026-10-07.diff` in this folder is the one change, against
the module as archived. On the same window and seed the cached arm's
sampling came in well under the stock arm's, in the band the design
predicted, with one build and no rebuild
(`bench/results/2026-10-07_frozen_cache_masked_window_fixed.md`). The cached
render's quality is the owner's eye on the stacked pair; until they say so
this folder stays an archive and the live module stays as it is.

## For whoever reads this next

mrhand's end-of-day note (`internal/claude/2026-10-06_mrhand/notes.md`,
"What I would do with it") has the arithmetic on why the miss is one rate
(the rectangular attention call costs about 2.8 times the dense square per
query-key pair on the same kernel) and what would be worth an hour: timing
the kernel alone off the square. The piece worth lifting back first if any
of this is reopened is the row reading from core's masks (`_rows`), which a
Sol-Attn that skips kept rows as queries would need with no cache at all.
The decision to reopen waits on the owner looking at the side-by-side
(`Video/compare_stacked/band_cache_baseline_vs_cache_plain_over_mask.mp4`),
since the cached render is a different sample of the region and nobody has
said whether it is an acceptable one.
