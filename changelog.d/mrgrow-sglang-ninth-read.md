bump: patch

### Changed

- **The sglang docs say what its serving code does with frozen audio and
  with video inputs, read at its head of the day, and three stale
  statements are corrected.** The owner asked whether sglang does anything
  with frozen audio or anything special with a video that is edited,
  continued or windowed. It does not, in its own pipeline: a reused source
  track is a reference audio block ahead of the target, the target audio is
  generated from noise, a source video is a reference video and nothing
  else, one request is one clip, and sampling is the same for every task.
  `docs/research/sglang_comparison.md` has the read as its "Ninth read",
  with how that request differs from the whole-frame edit rendered here.
  - `docs/h3_audio_freeze.md` section 8 said no mask path exists anywhere in
    sglang's H3 serving code. One file now has one: the stage that runs a
    DiT step for a ComfyUI sampler applies core's masked-row rule to video
    and audio rows. The sentence is narrowed to sglang's own pipeline, with
    a dated note. The eighth read described that stage without its masks.
  - `docs/h3_references.md`: two citations into sglang files whose lines
    had moved are by name now; a note under "Edit a source video" says the
    vendor path has no edit task and where a reference video's rows sit
    against the target's in time and in space; a note beside the vendor
    comparison table says the condition noise level is a setting sglang
    takes per request and core takes from a conditioning key.
  - `docs/wiki/decisions.md` logs what the corrected sentences used to say.
  No code, graph or default changes. Line citations in the older sections
  of the two sglang research docs were not re-verified.
