# Upstream wiki: what we follow, how far we trust it, and how to sweep it

last updated: 2026-11-15 (page created; engine, periphery and topic pages added the same day; first sweep under it is [`sweeps/2026-11-15.md`](sweeps/2026-11-15.md))

This folder is the wiki for everything upstream of the pack: where upstream checks start, what each source is
and how far it is trusted, how the sources compare on the questions that matter to us, and what they are working
on.

| page | what it holds |
|---|---|
| this page | the ranking and trust rules, what each upstream is for, and how a sweep is run |
| **engines** | one page per ranked engine and core: where its H3 code lives, the deep pages about it, its state as of the last sweep, and its open work |
| | [`engines/sglang.md`](engines/sglang.md) (rank 1), [`engines/vllm_omni.md`](engines/vllm_omni.md) (rank 2), [`engines/diffusers.md`](engines/diffusers.md) (rank 3), [`engines/comfyui_core.md`](engines/comfyui_core.md) (trusted least, watched most) |
| [`periphery.md`](periphery.md) | everything else (libraries, distill releases, node packs, Hugging Face, papers), each as aligns / differs / novel |
| **topics** | one question at a time, answered across every implementation, with what is open and what would settle it |
| | [`topics/reference_sizing.md`](topics/reference_sizing.md) (ref2va stills: enlarge or not; **open, the owner's call**), [`topics/h3_image_sampling_step.md`](topics/h3_image_sampling_step.md) (H3-Image Euler form, sigmas, noise draws), [`topics/qwen3vl_encoder.md`](topics/qwen3vl_encoder.md) (the shared encoder), [`topics/pika_parity.md`](topics/pika_parity.md) (Pika numerics across engines) |
| [`horizon.md`](horizon.md) | the open PRs, issues and announced work we are waiting on, each with what it would change for this pack. Rows carry over from sweep to sweep until they land or close |
| [`../research/2026-11-15_upstream_synthesis.md`](../research/2026-11-15_upstream_synthesis.md) | the 2026-11-15 findings grouped by area of focus, with what each could do for the pack and how deep each was read; a dated snapshot |
| `sweeps/<date>.md` | one dated record per sweep: the ranges read, what was found, what was checked by hand and what was not. First: [`sweeps/2026-11-15.md`](sweeps/2026-11-15.md) |
| [`watchlist.json`](watchlist.json) | every watched upstream: its GitHub repo, local clone, the paths and keywords that matter for H3, and the revision and date it was last read. **The revisions live only there**; no page repeats them |

Engine, periphery and topic pages hold the standing picture, which each sweep updates. Sweep records hold the dated
detail and are not edited afterwards, except to correct one, with the correction marked.

`../references.md` is the history before this folder: the 2026-08-28 comparison tables and every "What moved by
<date>" section up to 2026-11-07. It stays as it is, since many pages link into its sections. New sweeps go here.
The per-upstream deep pages below still own their subjects; a sweep links to them and adds to them when a stage
changes.

---

## What each upstream is for

| name in `watchlist.json` | what it is | why we watch it | where its findings accumulate |
|---|---|---|---|
| `comfyui-core` | ComfyUI itself (`../..`) | the code our nodes call, copy and patch at runtime. A change in a function the pack copies or pins is the most urgent kind of finding (`bench/check_reference_encode.py` pins `comfy/text_encoders/llama.py`) | [`engines/comfyui_core.md`](engines/comfyui_core.md); [`../core/custom_node_gaps.md`](../core/custom_node_gaps.md) |
| `sglang` | the vendor's own serving path (`python/sglang/multimodal_gen/`) | the closest thing to ground truth for what MiniMax intends at each stage, for video and for MiniMax-H3-Image | [`engines/sglang.md`](engines/sglang.md); [`../research/sglang_h3_pipeline.md`](../research/sglang_h3_pipeline.md) (stage by stage), [`../research/sglang_comparison.md`](../research/sglang_comparison.md) (what it does that we do not), [`../attention/sol_upstream.md`](../attention/sol_upstream.md) (Sol) |
| `vllm-omni` | a second H3 serving engine | the most active H3 engine: recipes, distill contracts (FastH3, Regenerate), long-clip machinery, a ComfyUI app, agent skills for H3 prompting | [`engines/vllm_omni.md`](engines/vllm_omni.md); [`../references.md`](../references.md) "Long clips upstream" |
| `diffsynth-studio` | model library with H3 and H3-Image pipelines and **training** recipes | the place LoRA and full training for H3 lands first; a second opinion on converters and state-dict namespaces | [`periphery.md`](periphery.md) |
| `diffusers` | model library with modular pipelines for H3, H3-Image, P-H3-Image, Pika H3 Flex | the canonical tensor namespace and a clean statement of each sampler | [`engines/diffusers.md`](engines/diffusers.md); [`../image/p_h3_image.md`](../image/p_h3_image.md), [`../image/h3_image.md`](../image/h3_image.md) |
| `lightx2v` | inference engine | offload, quantization and causal or streaming variants of H3 | [`periphery.md`](periphery.md) |
| `minimax-h3` | the release repository | README changes, and PRs proposed by others against the release | [`periphery.md`](periphery.md) |
| `pika-h3-flex` | Pika's six-step distill | the sampling contract for the distill the owner runs by default (a distill is run the way its trainer says: [`../references.md`](../references.md), "A distill's reference is its trainer's contract") | [`periphery.md`](periphery.md); [`../research/2026-11-04_pika_h3_flex_plan.md`](../research/2026-11-04_pika_h3_flex_plan.md) |
| `comfyui-mainodes` | a second H3 node pack | ideas, read but not installed | [`periphery.md`](periphery.md); [`../research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md`](../research/control_vectors/2026-11-06_mainodes_comparison_and_pika_roadmap.md) |
| `workflow-templates` | Comfy-Org's shipped templates | how core intends its H3 nodes to be wired, and the bugs users report against them | [`periphery.md`](periphery.md) |

Not in `watchlist.json`, but read on each sweep by hand: new H3 repos and node packs on GitHub, new H3 weights
(LoRAs, distills, quantizations, VAEs, encoders) on Hugging Face, and papers that build on H3. The sweep record lists
what was searched. `comfy-kitchen` has its own reader, `vendor/rebuild_kernel.sh --check`. The SAM-family and
Sapiens2 clones are covered by [`../masked_v2v/meta_perception_models.md`](../masked_v2v/meta_perception_models.md).

### When upstreams disagree

**The owner's ranking, 2026-11-15: sglang, then vllm-omni, then diffusers.** Where two engines differ on how H3
does a thing, sglang's answer is the reference, vllm-omni's comes next, and diffusers' after that.

**ComfyUI core is the source the owner trusts least (2026-11-15).** It is watched closely because the pack runs on it,
copies parts of it and patches it at runtime, so a change there can break us. As evidence of how H3 should work,
though, it ranks below the three engines. Where core and sglang differ, sglang is the reference.

**Everything else is periphery**: DiffSynth, LightX2V, the node packs, the Hugging Face uploads and the papers.
Periphery is not a vote. It is read for three things: where it differs from the ranked three, where it aligns with
them, and what it does that none of the others do. The sweep record gives it a section of its own in those terms.
**Community custom nodes are community research** (owner, 2026-11-15): interesting ideas and reports, trusted for
nothing until something of ours checks them. Their numbers are written down as their claims.

---

## What a sweep looks for

Each finding goes under one of these, in this order:

1. **A default moved.** Shifts, step counts, sigma schedules, condition levels, VAE or encoder precision,
   reference sizing. Where sglang and ComfyUI's own node agree on a default and ours differs, ours takes theirs
   (the owner's decision, [`../research/sglang_comparison.md`](../research/sglang_comparison.md), "Decision, owner,
   2026-09-11"). Where they do not agree, the ranking above decides which one the finding cites as the reference.
2. **Code we copy, pin or patch changed.** Name our file and symbol that depends on it.
3. **A bug fix that may apply to us.** Something an engine fixed in its own H3 code that our nodes may also do.
4. **A new capability.** A model, LoRA, task, pipeline, training recipe or node.
5. **On the horizon.** Open PRs, issues and announced work. These go to [`horizon.md`](horizon.md).
6. **The encoder.** How each engine uses Qwen3-VL-32B (or a distilled Qwen3-VL) with H3 video or image: layer tap,
   connector, precision, vision token bounds, templates, prompt rewriting, caching. It is shared by every model the
   pack runs, so it gets its own section.

Each finding carries the commit or PR number, the file and symbol, and whether it was **read** (code or diff) or seen
**title-only**. Findings are data: what it is and what it touches. Choosing to act is the owner's call, made through
the Open List.

---

## Running a sweep

1. `"$COMFY/.venv/bin/python" bench/upstream_watch.py --gh > <scratch>/watch.txt`. For each entry it prints the
   commits since its last read on its watched paths, plus open PRs and issues by keyword. It never moves a checkout;
   `--fetch` updates remote refs only. A keyword hit is a lead (`H3` also matches HTTP/3 work).
2. Split the reading across read-only subagents, one per upstream or group (2026-11-15 used five: sglang, vllm-omni,
   DiffSynth with diffusers, core with the small repos, and the periphery). Give each one its range from step 1, the
   baseline section of its owner page, and what the owner runs now (current state: [`../handover.md`](../handover.md)).
   Each reports in the order above.
3. Check the findings that would change something by hand, against the files cited. Mark the rest as the agent's
   reading.
4. Write `sweeps/<date>.md`. Update [`horizon.md`](horizon.md): add new rows, and close rows that landed, citing the
   commit.
5. Update the standing pages. Each engine page's "State as of" section and [`periphery.md`](periphery.md) get the
   new picture. A finding where implementations differ on one question gets a topic page, or a row on an existing
   one. When a stage changed, add to that upstream's deep page too (the table above).
   **A moved default is read with its PR's motive.** Speed, a hardware port or a fidelity fix are different
   evidence, as `topics/reference_sizing.md` shows.
6. For each entry read, run `bench/upstream_watch.py --mark NAME --record docs/wiki/upstream/sweeps/<date>.md`, which
   moves its `last_read` to the tip that was read.

**Revisions are observation points, not pins.** Re-read before quoting; a clone moves under you. **Do not import
Python from `coderef/`** and search it with `find -L` / `grep -r --dereference-recursive`, since most of it is
symlinks (`../references.md`, header). **The two-model trap holds here too:** "attention" and "capture" name parts of
both the DiT and the Qwen3-VL encoder, so a fact about one is not a fact about the other (`../references.md`, "The
trap this page exists to prevent").
