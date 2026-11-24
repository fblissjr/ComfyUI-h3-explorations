# Query router, first renders: two prompts in one pass, section split, and the router's own drift

lane: query routing
verdict: Three seeds, two scenes. A second prompt for the sound holds the picture at the router's own floor and changes the sound on the stairwell. Hiding the sound sections from the picture stops written sound wording reaching it and re-rolls the picture. The router is deterministic and is not an exact no-op.

**Date:** `2026-11-24`
**Node:** `MiniMaxH3QueryRouter` ([`steering/query_router.py`](../../steering/query_router.py)), pack HEAD `e46b70ac` loaded on the server for every clip in the tables.
**Figures:** the tables below. The full rows, and how each measure is computed, are in the local `data/results/2026-11-24_query_router_first_renders.json` (untracked, as all of `data/`); a name in backticks after a statement below is a key in that file.
**Clips:** `$H3_OUTPUT_DIR/Video/query_routing_2026-11-24/router_seeds/`; one labelled reel with sound per scene and seed in `.../reels/router_seeds_<scene>_s<seed>.mp4`; one stills sheet per scene and seed in `$H3_OUTPUT_DIR/Image/query_routing_2026-11-24/router_seeds/`.

## What was rendered

Pika H3 Flex text to video, the graph [`workflows/h3_text_to_video_pika_control_vector_api.json`](../../workflows/h3_text_to_video_pika_control_vector_api.json) with no control vector, `1344x768`, with the sparse attention node. Two prompts from the bank: [`t2va_stairwell_dialogue.txt`](../../prompt_bank/t2va_stairwell_dialogue.txt) at `192` frames and [`t2va_offpath_tortoise.txt`](../../prompt_bank/t2va_offpath_tortoise.txt) at `141`. Seeds `730451892`, `101`, `202`. Every clip a comparison needs was rendered at that seed; nothing is compared across seeds.

| arm | what differs from the unsteered graph |
|---|---|
| `*_unsteered` | nothing |
| `*_router_identity` | router on, the sound is given the same prompt as the picture |
| `stairwell_storm_written_plain` | the soundscape section rewritten to a thunderstorm; no router, so both streams read it |
| `stairwell_router_two_prompt_storm` | router: the picture reads the unchanged prompt, the sound reads the storm prompt (`audio_conditioning`) |
| `tortoise_reverb_written_plain` | a cathedral reverb written into the soundscape section; no router |
| `tortoise_router_two_prompt_reverb` | router: the picture reads the unchanged prompt, the sound reads the reverb prompt |
| `*_router_split_base` | router, `section_split="soundscape_to_audio"` on the unchanged prompt |
| `tortoise_router_split_reverb` | the same split on the reverb prompt |

## Figures

Picture distance is on 8-bit levels at `320x180`. A distance says how far one clip is from another, not which is better.

**stairwell**

| seed | arm | picture vs unsteered | sound spectrum vs unsteered (dB) | quiet-moment level (dB) | pitch shift (semitones) |
|---|---|---|---|---|---|
| 730451892 | `stairwell_unsteered` | 0 | 0 | -51.2 | |
| 730451892 | `stairwell_router_identity` | 8.41 | 4.62 | -50.6 | 0.386 |
| 730451892 | `stairwell_storm_written_plain` | 21.841 | 10.897 | -43.5 | 4.728 |
| 730451892 | `stairwell_router_two_prompt_storm` | 9.972 | 11.842 | -43.6 | 1.961 |
| 730451892 | `stairwell_router_split_base` | 20.383 | 5.697 | -48.7 | 0.017 |
| 101 | `stairwell_unsteered` | 0 | 0 | -70.9 | |
| 101 | `stairwell_router_identity` | 9.448 | 3.947 | -70.5 | 0.101 |
| 101 | `stairwell_storm_written_plain` | 20.999 | 16.719 | -48.9 | 2.768 |
| 101 | `stairwell_router_two_prompt_storm` | 11.858 | 15.418 | -53.3 | 2.142 |
| 101 | `stairwell_router_split_base` | 18.374 | 5.583 | -69.4 | 1.575 |
| 202 | `stairwell_unsteered` | 0 | 0 | -51.4 | |
| 202 | `stairwell_router_identity` | 11.052 | 4.987 | -49.9 | 0.175 |
| 202 | `stairwell_storm_written_plain` | 21.729 | 14.849 | -39.5 | 1.526 |
| 202 | `stairwell_router_two_prompt_storm` | 10.081 | 13.407 | -42.9 | 1.883 |
| 202 | `stairwell_router_split_base` | 17.096 | 5.91 | -49.5 | 0.786 |

**tortoise**

| seed | arm | picture vs unsteered | sound spectrum vs unsteered (dB) | quiet-moment level (dB) | pitch shift (semitones) |
|---|---|---|---|---|---|
| 730451892 | `tortoise_unsteered` | 0 | 0 | -37.9 | |
| 730451892 | `tortoise_router_identity` | 19.287 | 2.823 | -37.9 |  |
| 730451892 | `tortoise_reverb_written_plain` | 41.275 | 5.471 | -42.9 |  |
| 730451892 | `tortoise_router_two_prompt_reverb` | 19.669 | 3.954 | -36.6 |  |
| 730451892 | `tortoise_router_split_base` | 43.591 | 6.631 | -43.2 |  |
| 730451892 | `tortoise_router_split_reverb` | 47.173 | 5.273 | -42.7 |  |
| 101 | `tortoise_unsteered` | 0 | 0 | -26.4 | |
| 101 | `tortoise_router_identity` | 17.382 | 4.807 | -26.6 |  |
| 101 | `tortoise_reverb_written_plain` | 34.614 | 3.93 | -27.7 |  |
| 101 | `tortoise_router_two_prompt_reverb` | 23.001 | 3.533 | -26.9 |  |
| 101 | `tortoise_router_split_base` | 35.983 | 6.343 | -27.4 |  |
| 101 | `tortoise_router_split_reverb` | 35.575 | 6.082 | -28.0 |  |
| 202 | `tortoise_unsteered` | 0 | 0 | -27.2 | |
| 202 | `tortoise_router_identity` | 20.008 | 2.885 | -27.8 |  |
| 202 | `tortoise_reverb_written_plain` | 48.136 | 7.145 | -32.5 |  |
| 202 | `tortoise_router_two_prompt_reverb` | 28.882 | 3.984 | -27.6 |  |
| 202 | `tortoise_router_split_base` | 44.294 | 6.841 | -31.8 |  |
| 202 | `tortoise_router_split_reverb` | 45.364 | 6.27 | -30.1 |  |

**tortoise, picture distance between pairs of clips**

| seed | split of the unchanged prompt vs split of the reverb prompt | split of the unchanged prompt vs plain reverb rewrite | router identity vs two prompts |
|---|---|---|---|
| 730451892 | 21.788 | 38.744 | 20.076 |
| 101 | 17.313 | 40.105 | 21.642 |
| 202 | 19.614 | 52.621 | 25.444 |

**repeats and the test without the sparse attention node (seed `730451892`)**

| comparison | picture distance | sound spectrum distance (dB) |
|---|---|---|
| `stairwell_unsteered`: the same graph rendered twice, across a server restart | 0.0 | 0.0 |
| `stairwell_storm_written_plain`: the same graph rendered twice, across a server restart | 0.0 | 0.0 |
| `tortoise_unsteered`: the same graph rendered twice, across a server restart | 0.0 | 0.0 |
| `tortoise_reverb_written_plain`: the same graph rendered twice, across a server restart | 0.0 | 0.0 |
| `tortoise_router_split_reverb`: batch clip vs the same graph first thing on a restarted server | 0.0 | 0.0 |
| `tortoise_router_split_base`: batch clip vs the same graph first thing on a restarted server | 0.0 | 0.0 |
| `tortoise_router_two_prompt_reverb`: batch clip vs the same graph first thing on a restarted server | 0.0 | 0.0 |
| `tortoise_router_identity`: batch clip vs the same graph first thing on a restarted server | 0.0 | 0.0 |
| tortoise without the sparse node: unsteered vs router identity | 17.067 | 3.199 |
| tortoise unsteered: without the sparse node vs with it | 20.769 | 4.612 |

**text refiner alone, CPU: how far one section's rows move when only another section's encoder rows change**

| prompt | change | rows watched | their move as a share of their own token spread |
|---|---|---|---|
| `covered_market` | `soundscape_swapped_for_another_prompts` | description | 0.118 |
| `covered_market` | `soundscape_flattened_to_section_mean` | description | 0.1065 |
| `covered_market` | `description_swapped_for_another_prompts` | soundscape | 0.4333 |
| `covered_market` | `description_flattened_to_section_mean` | soundscape | 0.7536 |
| `indoor_neutral` | `soundscape_swapped_for_another_prompts` | description | 0.175 |
| `indoor_neutral` | `soundscape_flattened_to_section_mean` | description | 0.1509 |
| `indoor_neutral` | `description_swapped_for_another_prompts` | soundscape | 0.3607 |
| `indoor_neutral` | `description_flattened_to_section_mean` | soundscape | 0.7577 |

## What the figures and the stills show

1. **Two prompts in one pass hold the picture.** The two-prompt clip is close to the router's identity clip's distance from the unsteered clip on five of the six scene and seed pairs, and somewhat above it on the sixth (tortoise, seed `202`). The plain rewrite is farther on all six.
2. **The storm prompt changes the stairwell's sound in one pass.** The quiet moments get louder on all three seeds, close to the plain rewrite's level, and the spectrum moves about as far as the plain rewrite's.
3. **Written sound wording reaches the picture.** In the plain reverb rewrite the stills show heavy wooden doors in a stone front on two seeds and a steeple in the distance on the third. The unsteered and two-prompt clips show the plain street.
4. **Hiding the sound sections from the picture stops that, and re-rolls the picture.** The split of the reverb prompt and the split of the unchanged prompt are as close to each other as the router's floor. Both are as far from the unsteered clip as the plain rewrite is.
5. **The picture was using the sound section.** The tortoise soundscape says the crossing signal beeps; the description names a zebra crossing and no signal. The unsteered clips show a lit traffic signal on all three seeds. With the sound sections hidden it is gone or dark on all three.
6. **The router is deterministic and is not an exact no-op.** The same graph rendered twice, across a server restart, is identical (`same_graph_rendered_twice_seed_730451892`). Four router clips rendered again first thing on a restarted server, in the opposite order, are identical to the batch's (`router_clips_repeated_on_a_restarted_server_in_reverse_order`). The identity clip still differs from the unsteered clip on every seed, and still does with the sparse attention node removed (`identity_without_the_sparse_attention_node`), by about as much as adding or removing that node moves a render.

## What this cannot say

- **Whether the reverb is heard.** On the tortoise the two-prompt clip's sound is within the router's floor of the unsteered clip's, and the scene's sound is a steady bed with nothing sharp for a tail to ring on. Nobody has listened to these clips yet.
- **Whether the storm is heard as a storm, and whether lips still match.** The level and spectrum figures cannot tell.
- **Which picture is better.** Items 3 and 5 are one reader's reading of the stills sheets, not blind.
- **Why the identity clip drifts.** Not tested. The main path runs `comfy kitchen attention` and the router's side call for the sound rows is torch's.
- **Anything beyond two prompts and one checkpoint.**

## Earlier the same day

- The first eight router clips were rendered on the build before `e46b70ac` (`36644f6a`). Their two-prompt and identity figures equal this build's exactly. Their section-split clips cut the sound sections after the text refiner and are superseded (`first_look_on_the_earlier_build_36644f6a`).
- A CPU measurement of the text refiner alone, before any of these renders: changing only the soundscape rows moves the description rows a little, and changing the description rows moves the soundscape rows a lot (`refiner_section_mixing_cpu`; run without the Pika LoRA, with another prompt's rows swapped in).

## To run it again

The drivers are in the study folder, which this repo does not track (`data/query_routing_2026_11_24/`): `run_router_seeds.py --seeds 730451892 101 202 --queue`, then `measure_router_seeds.py`; `run_router_identity_dense.py --queue` for the test without the sparse node. They change only the noise seed, the file name and the router node on graphs built by [`bench/edit_guidance_arms.py`](../edit_guidance_arms.py).
