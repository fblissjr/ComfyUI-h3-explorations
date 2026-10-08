bump: minor

### Added

- **`MiniMaxH3ReferenceNoise` (`reference_noise.py`): how clean the model is
  shown its picture and video references.** Core mixes a fixed, small amount
  of noise into every visual condition latent and labels its rows as that
  clean (`comfy/ldm/minimax/model.py`, `_cond_video_rows` and the `cond` and
  `ref_img` timesteps), at `VISUAL_COND_TIMESTEP` unless the payload carries
  `visual_cond_noise_aug`; core fills that from a conditioning key no node
  set. The release exposes the same level per request
  (`imgvid_cond_noise_aug_for_inference` in coderef sglang's
  `configs/sample/minimax_h3.py`). The node takes a model and `clean_level`
  (default core's constant, so by default it changes nothing) and installs a
  diffusion-model wrapper that hands core a copy of the payload with the
  level in it; it sits on the model so that it reaches a conditioning built
  inside another node, as the song node builds its own. One level for every
  still and video reference alike, as core has it; reference audio is not
  touched. Written for the whole-frame video-to-video lane, where a
  reference video at the canvas's size carried the source's look with its
  movement. No shipped graph wires it, and nothing has been rendered at any
  level but core's.
