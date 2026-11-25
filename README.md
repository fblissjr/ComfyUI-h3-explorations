<img src="assets/banner.jpg" alt="A hawk perched at the top of a bare tree against an overcast sky" width="100%">

# ComfyUI-h3-explorations

Solo tinkering and research hub for the MiniMax H3 ecosystem (`MiniMax-H3`, `Pika H3 Flex`, `MiniMax-H3-Image`, `P-H3-Image`), shipping [`67` custom ComfyUI nodes](nodes.py), [`123` generated API workflows](workflows/INDEX.md), and the mobile-first [**H3 Studio 2.0**](app/README.md) workbench.

## What's Here

- **`1x` Cross-Modal Query Routing (`MiniMaxH3QueryRouter`)** — route video queries (`Q_vid`) and audio queries (`Q_aud`) to separate prompts or isolate audio from visual leakage (`aud_to_vid_gain = 0.0`) inside a single `1x` forward pass ([`docs/wiki/query_routing.md`](docs/wiki/query_routing.md), [`docs/wiki/query_routing_recipes.md`](docs/wiki/query_routing_recipes.md)).
- **H3 Studio 2.0 (`app/`)** — iPhone Safari & Desktop React workbench with `1x` Query Router Instant Audio Twists / Genre Roulette / Heylook Smart Twists, collapsible `Re-Dub Audio` (`Isolate Visuals`), 4-bus Control Vector mixer, and `/lab` Workshop ([`app/README.md`](app/README.md)).
- **`1x` Orthogonal Control Vectors & Subspace Erasure (`MiniMaxH3Steer`)** — continuous `Look`/`Motion`/`Audio`/`Hold` sliders across `50` active bank stems plus reproducible research manuscript ([`docs/wiki/h3_steering_and_architecture.md`](docs/wiki/h3_steering_and_architecture.md), [`docs/wiki/steering/paper/manuscript.md`](docs/wiki/steering/paper/manuscript.md)).
- **`2x` Spatial World Director & Masked `EditGuider`** — `[F, H, W]` tracked-mask and moving-portal steering, 3-frame `MiniMaxH3SoundCues`, and interactive sandbox ([`docs/wiki/spatial_director_guide.md`](docs/wiki/spatial_director_guide.md), [`docs/wiki/edit_guidance.md`](docs/wiki/edit_guidance.md)).
- **Audio Freeze, Video-Frozen Re-Dub, & High-Res Single-Frame Image** — `MiniMaxH3FreezeAudio`, `MiniMaxH3AudioRefineMask` (+ uncached `a2v=0.0` isolation), `MiniMax-H3-Image` / `P-H3-Image` `4 MP` pipelines, and `MiniMaxH3Sol` attention ([`docs/wiki/core/h3_audio_freeze.md`](docs/wiki/core/h3_audio_freeze.md), [`docs/wiki/image/h3_image.md`](docs/wiki/image/h3_image.md), [`docs/wiki/attention/SOLATTN.md`](docs/wiki/attention/SOLATTN.md)).

[What Works Right Now](docs/wiki/README.md) · [LLM Wiki Index](docs/wiki/index.md) · [Handover](docs/wiki/handover.md) · [H3 Frame & Clock Rules](docs/wiki/core/h3_frame_rules.md) · [Cross-Modal Query Routing](docs/wiki/query_routing.md) · [H3 Studio 2.0](app/README.md) · [Paper draft (v5)](docs/wiki/steering/paper/manuscript.md)

## Licence

This pack is under the MIT licence in `LICENSE`, except the
`vendor/meta_sapiens2/` and `vendor/meta_moge2/` directories: those hold
Meta's code and text, under Meta's licences in
`vendor/meta_sapiens2/LICENSE.md` and `vendor/meta_moge2/LICENSE`.
