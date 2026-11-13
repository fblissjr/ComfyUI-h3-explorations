# The frozen-row probe: the refine pass keeps frozen video exact, 2026-09-26

**Question.** On 2026-09-25 the audio-refine arm's video, frozen at mask 0,
decoded about 46 dB from its base arm's
(`2026-09-25_distill_audio_s1.md`). Was that the refine pass, or the first
pass? It matters beyond the refine graphs: route 4 of
`docs/wiki/archive/2026-09-26_distill_routing.md` rests on frozen rows coming
back exact.

**How.** One execution of `h3_probe_t2v_pdd8_audio_refine_savelat` (label
`frozen_row_probe`), run first in the 2026-09-26 distill run on the armed
server. That one execution saved pass 1's latent, the refine pass's
denoised output and the final latent, so no cache is involved. The graph
also carries `MiniMaxH3DenoiseMaskProbe`, which logs the mask the sampler
uses at every step. The fastdude session ran it and read the log. The latent
comparison was made by that session and repeated independently here (float64).
Record: `2026-09-26_frozen_row_probe.json`.

## Result

- **The mask is exactly 0 on the video and 1 on the audio at all six refine
  steps** (sigma 0.923 down to 0.524, from the probe's log lines).
- **Video:** the refine pass's denoised output is **bit-identical** to pass 1.
  The final latent differs from pass 1 by at most 4.8e-7 (relative L2
  2.0e-8), which is the last Euler step's float rounding.
- **Audio:** changed, as the pass intends (relative L2 0.31).

So the refine pass preserves frozen video exactly. **The 2026-09-25 gap was
pass 1:** the refine arm re-rendered it rather than reusing the base arm's
from cache, and the two renders differed. That record's cache-hit reading was
wrong, and it carries a dated note.

## What this moves

- Route 4 (per-row noise) is no longer blocked: frozen rows are exact.
- **The open question is now pass 1's run-to-run reproducibility.** On
  2026-09-26 the shipped FlashGen graph (graph sha `f11c8f64`, LoRA through
  the branch, no merged weight patches) re-rendered bit for bit on two server
  processes at two seeds, video and audio
  (`2026-09-26_flashgen_rerender_repro_seed891.json`, `_seed892.json`). It
  also reproduced within one process: `text_to_video_flashgen_ship_diner_00001`
  against `h3_probe_t2v_flashgen_r64_4step_branch_r64_branch_00001`, a real
  re-render (sampler 141.4 s, not a cache hit), per the fastdude session.
  *Caveat, 2026-09-26 later: those renders ran before 0.154.8, whose fix stops
  a LoRA branch installing on top of another model's still-applied branch in
  the same process (`2faed1a8`). A matching pair shows both processes built
  the same model; it does not show that model was FlashGen alone. The
  reproducibility reading stands; which weights it reproduced is unverified.*
  The
  2026-09-25 pair was PDD8, with 308 merged weight patches, at 0.141.0.
  Whether PDD8 reproduces today is unmeasured. The merged patches' stochastic
  requantization is seeded per module (`comfy/ops.py`,
  `comfy.utils.string_to_seed(s.seed_key)`), so it is not by itself a source
  of run-to-run change. The decisive test: render the probe graph once more
  on a fresh server and compare the two pass-1 latents.

**Scheduled:** the fastdude session re-renders this probe graph at the same
seed on a fresh server at the end of the distill run. Comparing its `_pass1`
latent with this one decides whether PDD8 reproduces on today's stack.

