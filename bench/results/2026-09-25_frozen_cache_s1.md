# Frozen-video cache, first live run, 2026-09-25

**A first run, not a comparison.** One seed, one scene, one render per arm. It
tests whether `MiniMaxH3FrozenVideoCache` (`frozen_video_cache.py`, 0.143.0)
runs on the card, what it saves, and how far its audio departs from the
uncached pass. The manifest is `bench/frozen_cache_arms.json`, and the rows are
`2026-09-25_frozen_cache_s1.jsonl`, with graph sha and substrate per row.

**Arms**, all FlashGen 4-step with the audio-refine pass:
- `refine`: the uncached refine graph.
- `refine_cached_verify`: the cached graph with `verify` on.
- `refine_cached`: the cached graph as shipped.

**Setup:** diner scene (`t2va_diner_breakup`), seed 730451892, 1344x768, 345
frames. The server was freshly restarted at 0.143.0.

**Cache state:** the first arm rendered the whole graph. The two cached arms
reused its first pass from ComfyUI's node cache, so their rows time the
refine sampler (node 87) alone.

## Did it run

- **Every arm rendered.** No row carries `error` or `suspect_cache_hit`.
- **The cache engaged.** Each cached arm logged one build over 104,502 packed
  rows, at int4, 13.49 GiB in RAM.
- **Verify ran on all five cached steps.** On the card, the kitchen backend
  took a query shorter than its keys with no fallback logged.
- **The compiler flip engaged** (the server has aimdo on and no
  `--disable-comfy-compiler`), and the pass completed with it.
- **The video is untouched.** The three silent video files hash identical.
- **`verify` changes nothing in the output.** The cached and verify arms'
  audio decodes bit-identical.

## What the cache costs the audio

The `verify` lines give the audio velocity on each cached step against the same
step run stock, from the server log:

| video sigma | cosine | relative L2 |
|---|---|---|
| 0.8957 | 0.998276 | 0.05872 |
| 0.8575 | 0.998427 | 0.05607 |
| 0.8000 | 0.998894 | 0.04706 |
| 0.7064 | 0.999460 | 0.03286 |
| 0.5239 | 0.999608 | 0.02801 |

The departure shrinks as the pass denoises. These are per-step comparisons from
the same state; they do not compound the way the trajectory does.

**On the finished audio**, the cached and uncached arms' decoded waveforms
(48 kHz stereo PCM from each `-audio.mp4`, so through AAC):
- cosine 0.9956;
- SNR 17.3 dB.

For scale, the FlashGen first pass without refine sits at cosine 0.84 from
either refined clip. So the cache moves the audio far less than the refine
pass itself does. Whether the difference is audible is the owner's ear to
decide.

**No noise floor yet.** The earlier session's `diner_flashgen_refine` clip is
not a same-graph repeat: its graph sha differs (0.141.0, before 0.142.0 moved
Sol's defaults). A same-graph repeat after a cache clear would give the floor
the SNR above should be read against.

## Wall time

Per row: `total_s` (submit to finish), node 10 (the FlashGen first pass),
node 87 (the refine sampler) and node 11 (the video decode), from `per_node_s`:

| arm | total | first pass | refine | decode |
|---|---|---|---|---|
| refine | 308.4 s | 124.3 s | 143.4 s | 30.1 s |
| refine_cached_verify | 225.1 s | cached | 191.1 s | 30.0 s |
| refine_cached | 106.2 s | cached | 72.0 s | 29.9 s |

**The cached arms' totals leave out the first pass**, which the node cache
served. A whole cached render from a cold node cache is `refine`'s total with
its refine node swapped: 308.4 - 143.4 + 72.0 = 237.0 s. That figure is
derived, not measured.

The earlier session's uncached refine rows ran 133 to 135 s. The cache roughly
halves the refine pass on this card. The build step is a full step and more
(tqdm shows about 29 s for it), so most of what is left is the build.

The cached steps are not free. Two causes are likely, and neither is timed
here:
- each step moves the int4 store from pageable RAM to the card and rebuilds
  K/V with a full-sequence qkv matmul;
- the 4090 streams weights under dynamic VRAM.

## The owner's listen, 2026-09-26 (unblinded, one seed)

`refine` against `refine_cached`, in the owner's words: "Cant hear a
difference whatsoever". On that the shipped refine graphs took the cache in
0.144.0, and the graphs this run named moved: the uncached arm is now
`h3_probe_t2v_flashgen_4step_audio_refine_uncached`, and the `_cached` graph is
gone because the refine graph carries the cache itself. The rows keep the sha
of the graphs that ran.

## Next

- *Done 2026-09-26:* the owner's listen, above.
- **A same-graph repeat of `refine` after a cache clear**, for the noise floor.
- **Not tried:** fewer refine steps (`h3_config.AUDIO_REFINE` is inherited and
  untuned), a VRAM store, K/V contents instead of the hidden state, or
  pinned transfers, any of which could shorten the cached steps.

Clips on the output share under `Video/`, each with a `-audio.mp4` companion:
- `h3_probe_t2v_flashgen_4step_audio_refine_refine_00001`
- `h3_probe_t2v_flashgen_4step_audio_refine_cached_refine_cached_00001`
- `h3_probe_t2v_flashgen_4step_audio_refine_cached_refine_cached_verify_00001`
