# Spatial Director Vision

> A timeless statement of purpose, enduring principles, tradeoff rules, and long-term direction for **Spatial Director**. Tactical rollout steps, active milestones, and concrete engine or node choices belong in [`director/ROADMAP.md`](ROADMAP.md), [`director/AGENTS.md`](AGENTS.md), [`docs/wiki/spatial_director_guide.md`](../docs/wiki/spatial_director_guide.md), and [`VISION.md`](../VISION.md); this document captures what should still hold true even if every programming language, web runtime, model architecture, attention kernel, segmentation backbone, operating system, network overlay, and accelerator in the studio is replaced five years from now.

---

## 1. Purpose — Why Spatial Director Exists

Conventional tools for moving image and sound force creators to choose among three broken metaphors:

- **Prompt-box slot machines** treat generative video as a global text incantation. Asking a single upper-story window to shatter or one character to glance left re-rolls the room, camera, lighting, and bystanders from scratch — charging full-clip latency before showing whether the composition even survived.
- **Node-graph plumbing** exposes real conditioning levers, latent masks, and sampler schedules at the price of pulling the director's eyes off the scene and into wires, tensor shapes, token math, and multi-section prompt boilerplate.
- **Traditional multi-track timelines and compositors** were built for cutting and layering footage that already exists. They assume rigid keyframe lanes, manual rotoscoping, and mute visuals with sound bolted on afterward — blind to semantic entities, environmental chain reactions, and joint audiovisual synthesis.

Directing a generative world breaks every one of those assumptions.

Directing a scene is neither rolling dice on a global paragraph nor painting dead pixels frame by frame. It is **localized causal intervention inside a living audiovisual world**. A session begins inside a concrete **World State** — a photograph, a painted concept, a generated anchor frame, or the final beat of an earlier shot. You point directly at a place or an actor at a chosen moment; express a terse, natural stage direction (*"shatters outward"*, *"steps out of the rain"*, *"turns toward the sound"*); dial how tightly the change is contained versus how far its environmental halo ripples into neighboring subjects and light; preview the change locally in seconds before paying for motion; steer or cancel as the trajectory forms; hear synchronized foley and room acoustics born from the same event; verify what changed against a matched counterfactual baseline; and roll any frame forward into the next shot.

Just as importantly, a directing stage should belong wholly to its creator — not as a policy promise, but as a structural property verifiable by inspection. **Spatial Director is built for a single owner operating inside a private studio network or private overlay — never exposed to the public internet and never shared across untrusted tenants.** Every interaction, security, and architectural choice rests on that trust boundary:

- **Your world stays anchored unless you ask it to move** — untouched regions and protected subjects hold their composition, identity, and calm while directed regions transform and nearby entities react proportionally.
- **Your intent stays spatial and immediate** — direct canvas gestures, concentric influence halos, entity locks, discrete event pills, and short natural commands compile invisibly into rigorous multimodal conditioning underneath.
- **Your studio stays sovereign and disposable until you commit** — uninhibited experimentation lives in ephemeral scratch space with zero cloud telemetry, promoting only deliberate selects and their reproducible recipes into permanent storage.

---

## 2. Core Beliefs & Guiding Principles

These seven commitments sit beneath implementation details and are meant to outlast any single model family or interface stack.

### 1. World State First, Never a Blank Slot Machine
Every session starts on an edge-to-edge stage anchored to an observable visual state — never an empty form or a hidden demo preset. By default, anchored pixels are a contract: outside the regions, entities, and environmental halos the director activates, the scene holds its ground. Re-imagining an entire shot from scratch is an explicit creative choice, never an accidental side effect of asking for a local event.

### 2. Direct Spatial Intent Over Prompt Incantation
A director should point at *where*, mark *when*, select *whom*, and state *what happens* in a few plain words — never hand-author sprawling multi-section conditioning schemas, token alignments, or negative-prompt folklore. Spatial regions, concentric inner masks and outer reaction halos, semantic entity silhouettes, actor trajectories, and human-readable event capsules on a single clean loop bar belong directly under the cursor, scroll wheel, keyboard, and fingertips. Structured model conditioning is compiler output: inspectable on demand, overridable when wanted, and never mandatory clerical work.

### 3. Progressive Commitment Across Latency Tiers
A studio must never charge full motion-render latency just to reveal a framing, scale, or semantic mistake. Creation flows up a deliberate ladder of feedback speeds:

- **Zero-latency (`0 ms`) stage choreography** — dragging spatial portals, tuning concentric containment versus halo reach, sketching trajectories, repositioning cutouts, and auditioning synthesized timing impulses live on the viewport.
- **Fast localized pre-visualization** — drafting appearances, props, or transformations directly inside the active region or cutout in seconds, ready to drag, resize, accept into the world state, or discard before committing to time.
- **Interruptible live-step simulation** — streaming visual previews and convergence signals step by step as motion and audio form, so misfires are caught and cancelled early rather than paid in full.

### 4. Coupled Entities, Environmental Halos, and Unified Sound
Living scenes are not isolated sticker layers, and motion is never silent. An inner region defines where an event originates; its outer halo defines how light, atmosphere, and nearby actors respond — flinching, turning, or yielding along a path — while explicitly protected subjects and architecture remain invariant.

In the same spirit, visual motion, foley impacts, ambient room tone, and voice are facets of one audiovisual world event, yet remain independently lockable. Polishing or re-dubbing the acoustic landscape must never force a re-roll of picture you already love, and refining visual motion inside a region must never silently overwrite a locked soundtrack.

### 5. Honest Causality Over Decorative Theater
Generative world models are learned probabilistic simulators with real spatial patch grains, discrete temporal clocks, finite attention budgets, and characteristic failure modes — not deterministic rigid-body game engines. Spatial Director refuses decorative UI theater: no invented Newtonian force gauges, fake metric depth meshes, deceptive progress indicators, or hidden prompt scripts steering behind the creator's back.

Every handle on the stage maps to a real causal lever in the underlying generator. Every rendered take can be judged against a matched same-seed counterfactual baseline — through interactive `A|B` spatial wipes, measured in-mask versus out-of-mask localization telemetry, and isolated stream auditioning — so the creator sees what their direction *actually* caused.

### 6. Ephemeral Sandbox, Deliberate Permanence, and Recipe Lineage
Bold direction requires fearless, messy branching: testing ten wild impulses, splitting a timeline from frame `42`, comparing two interpretations side by side, and throwing nine away without polluting permanent libraries or leaving orphan residue on disk.

All scratch previews, masks, cutouts, and draft renders live inside a contained, one-action-forgettable workspace. Only an explicit human export crosses into permanent storage — carrying the clean artifact, optional diagnostic HUD, and complete generative recipe so any moment can be reopened, audited, or continued months later.

### 7. Private-Studio Sovereignty and Calm Surface Craft
Built for a single creator across workstation monitors and personal review screens over a private network or overlay, Spatial Director runs segmentation, local multimodal co-direction, synthesis, and storage on sovereign hardware with zero public exposure, zero telemetry, and clean fallback whenever optional local assistants are offline.

On screen, the world owns the viewport. Light, tactile, low-noise chrome keeps spatial controls right where the eye is working while tucking engine parameters, token budgets, and compiled graphs calmly out of sight until summoned.

---

## 3. The Tradeoff Compass

When two good product or engineering ideas collide, Spatial Director breaks ties using five ordering rules:

1. **Fidelity outside the mask over dramatic global flair.** When an intervention strains the generator, contain the blast radius rather than allowing unmasked architecture, lighting, or protected actors to drift.
2. **Causal honesty over simulated precision.** Surface true spatial/temporal grid snapping, honest capacity ceilings, and measured counterfactual diffs over smooth-looking controls that have no real backing in the model.
3. **Fast steerable feedback over monolithic render perfection.** Favor workflows that prove composition in seconds and stream interruptible previews over opaque pipelines that stay dark until final decode.
4. **Direct spatial gesture over inspector sprawl.** Whenever an intent can be expressed by pointing on the stage, scrolling concentric radii, clicking an entity, or dropping an event pill on the loop capsule, refuse to add another permanent form panel or mode switch.
5. **Disposable scratch isolation over automatic persistence.** Never write drafts, cutouts, or counterfactual baselines into permanent media folders unasked; require deliberate promotion and guarantee clean, verifiable forgetting.

---

## 4. Explicit Non-Goals

Knowing what Spatial Director refuses to become protects the clarity and immediacy of the stage:

- **Not a multi-track NLE or frame-paint compositor:** Spatial Director orchestrates causal world events, entity trajectories, and shot-to-shot continuations; final editorial assembly, color-grading suites, title typography, and pixel-clone cleanup belong in dedicated finishing tools.
- **Not a general-purpose node-graph IDE:** Workflow graphs are compiled automatically to serve spatial intent. Curious creators can inspect or export the compiled graph, but the studio will never turn back into wiring noodles across the viewport.
- **Not a rigid-body CAD or game-physics viewport:** It will not pretend learned latent simulators obey deterministic collision meshes or engineering tolerances where conditioning actually operates through visual, temporal, and semantic fields.
- **Not a toybox of canned preset effects:** There are no static "spawn [canned object]" buttons, hidden prompt templates, or watermarked gimmicks; every world starts from the creator's own scene and answers open-vocabulary direction.
- **Not a multi-tenant cloud platform:** No accounts, credit meters, remote moderation proxies, public feeds, or third-party SaaS dependencies that compromise privacy or local responsiveness.

---

## 5. The Unfinished Horizon

Spatial Director is not chasing a static widget checklist; it is advancing along a clear long-term arc: **evolving from a localized clip-steering canvas into a continuous, branchable audiovisual world sandbox.**

Across future generations of models and hardware, that arc tightens one seamless creative loop:

> **World Anchor -> Point & Sketch Intent -> Instant Localized Pre-Viz -> Live Puppeteered Simulation -> Counterfactual Study -> Branching World Continuity**

In practical terms, that horizon invites deeper craft in four enduring directions — pursued without compromising causal honesty, calm simplicity, or studio sovereignty:

- **Live-Loop Puppeteering and Visual Co-Direction:** Moving beyond pause-and-place markers toward directing *while the world plays* — dropping impulses directly onto a live loop, steering entity paths in real time with cursor or keys, and collaborating with a local visual-language co-director that perceives the scene, suggests context-grounded reactions, and translates terse stage notes into spatial-temporal choreography.
- **Unified Soft Steering and Pixel-Locked Surgery:** Routing fluidly between guidance-space steering (where reflections, shadows, smoke, and room acoustics cross region boundaries naturally) and strict pixel-locked plate restoration with persistent subject identity banks whenever a shot demands untouched exterior pixels.
- **Multi-Entity Relational Choreography:** Authoring richer causal chains directly on the stage — linking an impulse at `t0` to staggered perceptual reactions, gaze shifts, dialogue beats, and trajectories across multiple segmented actors (`t1..tN`), while holding designated bystanders and background structure invariant.
- **Branching World Trees and Long-Horizon Continuity:** Expanding linear frame roll-forwards into navigable shot trees with persistent cast and environment memory across multi-minute sequences, seamless acoustic continuity across window seams, and lossless round-trips into private creative archives with every branch's full spatial recipe intact.

The models, encoders, and machines underneath Spatial Director will be replaced many times. The promise will not: **you direct the world directly on the stage, untouched reality holds its ground, sight and sound move as one, every control on screen is causally real, and your studio remains wholly yours.**
