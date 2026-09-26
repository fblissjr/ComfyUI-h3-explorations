# FlashGen on tasks it was not trained for: first renders, 2026-09-26

**A first look, not a verdict.** One render per task at seed 730451892, on each
graph's own default prompt and references. The rows are
`2026-09-26_flashgen_tasks_s1.jsonl`. FlashGen was trained for T2VA on FL2VA,
and vllm-omni refuses it elsewhere (`docs/research/2026-09-26_flashgen.md`).
Both graphs run the shipped t2v settings: rank 64 applied at the call, its own
sigmas on Euler, and the repo's attention default.

**Arms:**
- `i2v_flashgen`: `h3_probe_i2v_flashgen_4step`. First frame, on the fl2va
  checkpoint and file. The canvas follows the keyframe, 768x768, 345 frames.
- `r2v_flashgen`: `h3_probe_r2v_flashgen_4step`. Two image references on the
  ref2va checkpoint, with the file converted for its adaln basis
  (`2026-09-26_flashgen_lora_conversion_ref2va.json`). 1344x768, 345 frames.

## What happened

- **Both rendered.** No row carries `error` or `suspect_cache_hit`.
- **i2v, by eye (four frames across the clip):**
  - it holds the first frame's subject, lighting and framing, with the slow
    push-in the prompt asks for;
  - the face stays coherent and detailed to the last frame.
- **ref2va, by eye:**
  - both references appear: the man's likeness and the mountain lake, which is
    seen through a window behind him;
  - the likeness holds across the clip.
- **Audio is quiet on both, and that is the prompts.** The i2v prompt asks for
  "quiet room tone with a low ambient hum". Earlier PDD renders of the same
  ref2va prompt (`image_ref_plus_text_to_video_pdd_00001`, `_4step_00001`)
  measure as quiet as this one. `volumedetect` mean and max, dB: i2v -65.2 and
  -33.5; ref2va -52.1 and -20.8.
- **Sampler time** is in the JSONL. The ref2va pass carries the reference rows,
  and the i2v canvas is smaller than 1344x768.

## Clips

On the output share under `Video/`, each with a `-audio.mp4` companion:
- `h3_probe_i2v_flashgen_4step_i2v_flashgen_00001`
- `h3_probe_r2v_flashgen_4step_r2v_flashgen_00001`

## Next

- **The owner's look.**
- **A base or PDD render of each graph at the same seed**, to see what FlashGen
  costs on these tasks.
- **Prompts with sound in them**, to hear FlashGen's audio off T2VA.
