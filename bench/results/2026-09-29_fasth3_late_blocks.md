# FastH3's backbone by depth, on top of the gates alone: the owner's blind read (2026-09-29)

`docs/state/open_experiments.md` #48. `MiniMaxH3OverlayLoader` on `fasth3_v2_on_fl2va.h3overlay.safetensors`, gates on,
refiner, adaln and io off: `gates_only` (no backbone diff), `late_30_49` (FastH3's backbone diff on blocks 30-49)
and `early_0_29` (blocks 0-29, the location control). FastH3's contract harness, seed 730451892, three scenes at
their written lengths, one clip per arm per scene. Renders: `2026-09-29_fasth3_late_blocks.jsonl` (nine rows, no
errors, no cache hits). Blinded as the session `fasth3_late_blocks`: six pairs, each late or early against
gates_only on one scene, scored by the owner on the lean pairs-only form
(`docs/method/eval_comparison.md`, "The lean pairs-only session"), one preference per pair with a tie option. Joined
by `bench/score_session.py`: `2026-09-29_fasth3_late_blocks_verdict.json`. Manifest with the predictions written
before any render: `bench/fasth3_late_blocks_arms.json`.

## Result

| pair | scene | contest | verdict | the owner's words, condensed |
|---|---|---|---|---|
| 01 | look_anchor | late vs gates_only | late | "very slightly better ... proportions, maybe a little more detail, but barely" |
| 05 | radio_drama | late vs gates_only | late | slightly better on detail, lighting, colour; a paper floats in the woman's hand in both clips (flagged broken, both) |
| 03 | slapstick_moving_piano | late vs gates_only | same | "both look exactly the same"; heard Clip 1's audio only |
| 02 | look_anchor | early vs gates_only | gates_only | "more extra careful details like a glass jar" |
| 04 | slapstick_moving_piano | early vs gates_only | early | "little better detail but barely" |
| 06 | radio_drama | early vs gates_only | early | in the early clip she drops the paper and it falls; in the gates_only clip it floats in the air |

Tally over the three scenes: late 2, gates_only 0, same 1; early 2, gates_only 1, same 0. Audio was "same" on all
six pairs (both halves heard on five; on pair 03 only Clip 1's).

## Reading

- **No arm is clearly different from gates_only.** Five of the six preferences are described by the owner as slight
  or "barely", one pair is "exactly the same", and the one clear difference (pair 06, the paper falling or
  floating) is a physics or adherence difference between two samples in a scene where both arms and gates_only
  showed floating paper elsewhere. The owner also noted that the floating paper may be a prompt ambiguity these
  distills cannot handle and that the base model can.
- **Both backbone arms lean slightly ahead of gates_only** (four preferences to one, one tie, across both contests).
  That is six pairs and one seed each: a sign test on the five decided pairs gives a two-sided p of about 0.38, so
  it is not evidence, only the direction.
- **Depth does not separate them.** Late and early each took two of three; nothing here says the late blocks matter
  more than the early ones, which the weight-agreement finding (`2026-09-26_fasth3_weights.md`, finding 5)
  would have suggested.
- **Against the predictions** (`bench/fasth3_late_blocks_arms.json`): P1 (late looks like gates_only) is roughly
  what the owner saw, with two slight preferences for late and one tie. P2 (early is closer to gates_only than late
  is) is not supported: early took as many preferences as late.
- **What would falsify the "gates alone carry the look" reading** (late reaching FastH3's own numbers where gates_only
  does not) was not tested: there is no FastH3-as-released arm in this session, and this is a by-eye read.

## Not shown

- Whether any arm approaches FastH3 as released. No such arm here (it exists per scene in the gate-dial results
  from #36, `bench/fasth3_gate_dial_arms.json`, not rendered again for this session).
- More than three scenes at one seed. Each pair is two different samples, so a scene-level preference is one sample.
- Position bias: no identical-clip control pair was in the session.
