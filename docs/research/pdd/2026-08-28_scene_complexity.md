# What makes a PDD clip degrade: three accounts refuted, one standing

**Status 2026-08-28.** The owner's account is the only one still standing, and
he stated it before any of us measured anything. Three accounts registered by
the assistant lanes died in the space of an hour. This file records what was
tested, what died, and — more useful than either — the two ways we kept fooling
ourselves.

**Looking for what to watch and what each file tests?** That is
[`artifact_axis.md`](artifact_axis.md) section 0, which is the viewing map for
every arm all three sessions queued this night. This file is the ablation's own
record and does not restate it.

**The question.** Rendered PDD clips show ghosting, melting faces and object
morphing partway through. The owner's observations, over several evenings:
degradation appears at moments of large visual change, the bunker dialogue
scenes look clean, and — the framing that survived — *"theres too much detail in
whats being asked of the scene / frames being generated"*.

## 1. The arms

Base graph frozen from the tail6 render as `bench/_shot_ablation_base.json`;
`bench/run_shot_count_ablation.py` builds each arm from it. Verified by
dict-diff that only the prompt node and the filename prefix differ between
arms, and additionally the length node for the duration control. Seed, canvas,
partition, manual sigmas, Sol settings and models are held.

Each arm drops one of the market scene's three shots so the surviving two
stretch across the same canvas.

| arm | shots | frames | per-shot | verdict |
|---|---|---|---|---|
| `00007` (original) | 3 | 362 | ~5.0s | **bad** — ghosting, melting |
| `shots13` | 1+3 | 362 | ~7.5s | **great** |
| `shots12` | 1+2 | 362 | ~7.5s | **great**, box did not ghost |
| `shots23` | 2+3 | 362 | ~7.5s | rendered, unjudged |
| `shots13_241f` | 1+3 | 241 | ~5.0s | queued |
| `shots12_241f` | 1+2 | 241 | ~5.0s | queued, lands last |

The two `_241f` arms were posted directly rather than through the sequential
runner, so neither depends on a client process surviving. **A consequence worth
knowing when reading them: they did not get the full model unload between arms
that the 362-frame arms each got.** The runner does that between its arms and it
was stopped to post them independently. This canvas is lighter than everything
queued ahead of them and the pipeline measured bit-identical across submissions,
so the exposure is an OOM rather than a changed render — and a crashed arm
cannot write a partial output.

Verdicts are the owner's, free-text, on the muxed file. **One observer, one
seed, one pair at a time, and a coarse verdict rather than a scored
comparison** — this is not `docs/eval_comparison.md` section 3 and must not be
quoted as though it were.

### The bad clip is bracketed on both clip-level metrics

Scored through `bench/score_shot_ablation.py` (320x192 rgb24, flat RGB mean),
which is the same pipeline the per-shot table used:

| arm | delta | detail | verdict |
|---|---|---|---|
| `shots13` | 0.0122 | 0.0640 | great |
| **`00007`** | **0.0149** | **0.0635** | **bad** |
| `shots23` | 0.0165 | 0.0622 | unjudged |
| `shots12` | 0.0209 | 0.0580 | great |

**The bad clip sits inside the range of the good ones on both axes at once.**
Sorted by delta it is second of four; sorted by detail it is second of four. No
threshold on either statistic separates it, and no monotone function of either
can, because good arms lie on both sides. That is the whole refutation of the
two clip-level accounts in one table, and it does not depend on any verdict for
`shots23`.

## 2. What died

**Delta — `mean(|frame[n] - frame[n-1]|)`.** Refuted by `shots12`, which
carries the highest delta of the three scored arms and is fine. The bad clip
sits between two good ones. Registered by this lane after the owner's *"every
damn pixel is moving"*, which turned out to be the symptom he named alongside
the mechanism, not the mechanism.

**Output spatial detail.** No ordering: lowest is good, highest is good, the
bad one sits between. It was ambiguous by construction anyway — ghosted fruit
reads as LOW detail, so a low number cannot distinguish a low-detail scene from
destroyed detail.

**The crate lift.** `shots12` contains the shot the owner had repeatedly named
as the failure point (*"the box being lifted caused it to ghost/disappear/bend"*)
and he reported unprompted that the box did not ghost. So that action is not
sufficient to produce the failure.

## 3. What survives, and the confound inside it

**Total content demanded across the clip.** Three setups in fifteen seconds
breaks; two does not, whichever two.

One measurement constrains it: **`shots12` rendered an unprompted extra cut** —
delta peaks at frames 182 and 219 when only 180 was asked for — so it delivered
three visual segments and was still fine. Whatever breaks the original is
therefore not the number of shots that end up *on screen*. Demanded content
survives that; rendered segment count does not.

**The confound, and it is not resolved as of this writing.** Dropping a shot
changed two things at once. Total demanded content went down, AND each surviving
shot stretched from about 5s to about 7.5s. Both of these fit every observation:

    "three setups is too much for 15s"      -> write fewer setups
    "a setup needs more than 5s to resolve" -> give each one more time

They imply different fixes, and under the second a three-shot clip would be fine
at 22s. The `_241f` arms are the control: the same two pairs at 241 frames
(10.04s) restore ~5s per shot while keeping the count at two. Good at 241 means
total demanded content is the variable; bad means per-shot duration is.

`cut_at(362)` reproduces the original `00:07.500` exactly, which is what
validates the midpoint function against an arm that had already rendered.

## 4. Predictability

Built by the prompts lane from the owner's *"a tracking shot of a boxy house
going from left to right across... like its on a rail"*. `rail_long` and
`churn_long` are one shot each, no cuts, full 15s, both long prompts. The axis
is how hard the next frame is to PREDICT rather than how much it changes: a rail
move is near-maximal delta and near-zero uncertainty.

These are immune to both confounds above — shot count is one, duration is the
whole clip. The rail arm is load-bearing: long AND detailed AND the
highest-delta thing in the repo. Clean exonerates length, detail demand and
delta in a single render.

Their prompt lengths are 518 and 420 words, so **length is not held between
them**, contrary to how the pair was first described. The direction is
favourable — rail is the longer one — but the claim should not be overstated.

## 5. The two ways we fooled ourselves

**An account can look confirmed by exactly the arm that refutes it, when you
score a subset of a set that is still rendering.** Delta was declared confirmed
on the `shots13`/`shots12` pair and refuted by `shots12` once the third verdict
arrived. The crate-lift account was proposed by this lane while `shots23` was
already queued. Both within an hour. The fix is not more caution about which
account to believe — it is **not scoring until the set is complete.**

**Reaching for an instrument before re-reading what was said.** The owner's
sentence contained both the symptom and the mechanism; one lane built a delta
axis from the symptom half, the other measured detail on the OUTPUT and tried to
engineer around an ambiguity that existed only because it was the wrong end of
the causal chain. The cause was in the prompt and needed no render at all.

## 6. Method findings that outlive this

**ComfyUI writes its own log at `user/comfyui_8188.log`, with
`.prev.log` for the previous server session.** This matters because
`start.sh` pipes the server's stdout through `grep` rather than tee-ing it, and
this lane wrongly concluded from that a VRAM-mode switch was *unobservable*, then
proposed a 26-minute reference render partly to work around it. It was
observable the whole time. Across both files, covering every render of the
evening: **zero `lowvram` / `novram` / partial-load switches.** No render tonight
took a degraded memory path.

**This canvas OOMs on residency, not contention.** 362 frames at 1344x768 —
three OOMs across two lanes in one evening, every one with a single process on
the card. A clean start fixes it; a retry does not. A crash in
`SamplerCustomAdvanced` cannot write a partial output, because `VideoCombine`
runs after it in execution order, so a crashed arm never contaminates a set.

**`free_memory` alone unloads every model.** The server sets the unload flag
only when it is truthy, so the executor's `flags.get("unload_models",
free_memory)` falls through. Passing `unload_models: False` does not opt out.
Flags are consumed between prompts, never mid-render, so this is safe for a
shared queue but costs the next job a reload.

**Delta and spatial detail do not survive a resolution change equally.** A
spatial gradient shrinks under downsampling; a temporal one does not. So
`bench/score_shot_ablation.py` deliberately does NOT import
`bench/measure_clip_delta.py::motion` despite the no-second-copy rule — that
file measures at 160x96 BT.601 gray, this one at 320x192 rgb24 with a flat mean,
and mixing them silently produces incomparable numbers on both axes. Two
internally consistent files beat one module that has to pick a scale, because
the day someone changes `W, H` for a good reason every historical number moves.

## 7. Not established

- Every artifact observation feeding this file is a **market render**. Scene-specific
  explanations are unexcluded. The non-market arms (`aisle`, `sortline`) exist
  and are built but are held, because all four are three setups at ~5s each and
  would be uninterpretable until the duration control lands.
- Nothing here is a scored comparison. No distribution, no blinding, one seed.
- **The demand account has no mechanism.** "Too much demanded" is a description
  of when it breaks, not of what breaks. Nothing here says whether the limit is
  attention capacity, the step budget, the latent resolution, or the text
  segment's share of the packed sequence.
- Enforced by nothing: no check asserts any claim in this file.
