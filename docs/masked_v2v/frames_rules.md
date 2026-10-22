# Frame rules: how a masked load's frames line up with what H3 sees

Last updated: 2026-10-22.

One page for the frame math of a masked load: how long its window is, how the video VAE and the text encoder cut that
window, where the shot's own frames land, what fills the rest, and what is thrown away. Every figure points at the code
that sets it; when this page and the code disagree, the code is right ([`AGENTS.md`](AGENTS.md), and the root
`AGENTS.md`, "Do not trust prose"). The lane's other rules are in [`AGENTS.md`](AGENTS.md); the legal lengths and
their cost for H3 in general are in [`docs/h3_resolutions.md`](../h3_resolutions.md). The whole render path in trace
order, as it stood on 2026-10-21, is Gemini's report (`internal/gemini/2026-10-21_render_path_report.md`, local, not in
git). Two things it describes have changed since: the padding was split evenly before and after (now all after, §4),
and `<Video 1>` was padded with black frames (now held copies of its first and last frames,
`video_mask.py`'s motion padding). Its simplifications (one function for the window plan, the audio sliced to the
window's exact frames, the padding as explicit inputs) stand as proposals.

## The rule

**Every frame of the shot sits in a window with its own shot first.** A short shot's padding all goes after it, so the
shot's first frame is window frame 0; where the frames after the shot belong to another shot, they are held copies of
the shot's last frame; the padding is thrown away after the decode. The owner's words, 2026-10-22: "just add buffer
frames after", "toss the after".

## 1. The window's length

- A masked load renders in one window whose length is on both clocks: on the video's `17n + 5` grid and whole on the
  audio's (`h3_rules.on_both_clocks`, `h3_rules.FPS`, `h3_rules.AUDIO_LATENT_RATE`), the shortest that holds the shot
  (`workflows/masked_job_graph.py::window_for`). The lengths: 39, 90, 141, 192, 243, 294, 345.
- A shot longer than the longest runs as chained windows inside one load (`h3_rules.chain_lengths`,
  `workflows/h3_config.py::MASKED_JOB_CONTEXT`); [`AGENTS.md`](AGENTS.md) §6 has how the overlap is thrown away.

## 2. How the video VAE cuts the window

From `comfy/ldm/minimax/vae.py` (the constructor's `clip_length`, `time_down`, `token_drop`, and `encode_temporal`):

- The window is encoded in **clips of 17 frames, each on its own**. A short last clip is filled out with copies of its
  own last frame.
- Inside a clip, the first frame is one latent step by itself and the other 16 go 4 to a step: 5 steps a clip.
- The last 3 steps of the whole window are dropped, which is why a `17n + 5` window has `5n + 2` steps.

So in a 39-frame window: clip 1 is frames 0-16, clip 2 is 17-33, clip 3 is 34-38 filled with 12 copies of frame 38.

**What goes wrong when a shot's frame is outnumbered in its clip.** `second_hfp_4479e` (31 frames in 39, padded 4 before
and 4 after): its last frame, 4509, was window frame 34, alone at the start of clip 3 with the next shot's 4 frames and
12 copies of the last of them. Everything regenerated round her came out brighter on 4509 alone, a square (the owner,
2026-10-22). `third_fp_4602c` showed no such frame: its last frame sat in a clip of 15 of its own frames and one of the
next shot's.

## 3. How the text encoder sees `<Video 1>`

- Qwen3-VL is given `<Video 1>` (the face drawing) at two frames a second, every 12th frame from **window frame 0**
  (`reference_conditioning.py`, `sample_indices = range(0, frames, FPS // 2)`), and fuses each pair into one picture,
  labelled with their mean time (`comfy/text_encoders/minimax.py`, the `video` branch). So it sees one picture a second,
  each the blend of two frames half a second apart.
- The padding is part of what it samples: the drawing is held at its first and last frames over the padding
  (`video_mask.py`, the motion reference's padding). With the padding after the shot, its first picture is the shot's
  first moment, and any picture past the shot's end is a held copy of its last.
- `<Video 1>` also goes to the video model whole: every frame is VAE-encoded (`motion_vae`, set in
  `workflows/masked_job_graph.py::load`), in the same 17-frame clips as the window (§2). So the padding rule lines up
  `<Video 1>`'s clips with the window's for the video model, and its samples with the shot's start for the text encoder.
- Both see `<Video 1>` at the canvas's size: the text encoder in 16 px patches merged to 32 px, the video model in 32 px
  tokens. At the head crop (`bench/crop_reference_to_head.py`, enlarged 1.654 times on `second_0278_0293_g`) a drawn
  iris is 15-21 px across and, where the lids cut it on a downcast eye, 6-9 px tall: under one 16 px patch tall
  (measured 2026-10-22, `claude-face-crop`'s proof). That is why the `<Video 1>` crop's size matters for gaze.
- This is upstream's design: sglang sizes a reference video like the target canvas, short edge 768, and feeds one
  array to Qwen3-VL (every 12th frame) and the VAE, while a reference still goes to a 2048 short edge
  ([`docs/research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md), §2 and §3). A larger `<Video 1>` is
  off that path; a tighter crop inside the 768 frame is on it.
- `<Picture 1>`, the still, goes to the VAE and to Qwen3-VL as one image (`qwen_view` "shared",
  `workflows/h3_config.py::MASKED_JOB_STILL_VIEW`), so the two see it alike.

## 4. What fills the padding

- **After the shot, always** (`workflows/masked_job_graph.py::window_pad`: before 0, after the window less the shot).
  The one exception: a shot ending so near the source's last frame that the frames after it do not exist takes the
  rest before it, held.
- **Held copies of the shot's last frame, with its masks,** where the frames after the shot belong to another shot
  (the shot's boundaries are `data/<video>/shots.json`'s). **Real frames** where the shot goes on past the load.
- **The guard**: where the shot's last frame has no mask, or less than `video_mask.HELD_EDGE_LEAST` of the shot's median
  region, that side takes real frames instead and the plan row says so (`held_fell_back`). Why: `second_hfp_4912e`, whose
  last frame had no mask and whose held copies filled the encoder's last clip with her untouched hair.
- The audio the model hears is silent under held frames; the audio muxed into the piece is the source's.

## 5. What is thrown away

After the decode, every padding frame is dropped (`audio_freeze_song.py::MiniMaxH3AudioFreezeSong.execute`), so the
piece, its region and its matte are the shot's own frames and nothing else.

## Worked examples

| load | shot | window | padding | encoder clips | what happened |
|---|---|---|---|---|---|
| `second_hfp_4479e` (before the rule) | 4479-4509, 31 | 39 | 4 real before, 4 real after, both other shots | 4479 in a step with 3 frames of another shot; 4509 alone in clip 3 with the next shot | a bright square on 4509 |
| 4479 (under the rule) | 4479-4509, 31 | 39 | 8 held after | clips 1-2 her frames; clip 3 her last frame held | to be rendered |
| `third_fp_4602c` | 4602-4646, 45 | 90 | 22 before, 23 after, other shots | 4646 in a clip of 15 of its own frames | no bright frame |
| `second_hfp_4912e` | 4912-4946 | | held copies of 4946, whose mask was empty | the last clip a frozen frame of her untouched hair | why the guard exists |
