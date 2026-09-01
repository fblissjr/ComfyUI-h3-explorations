# The vendor's two prompt-writing guides, verbatim

`base_en.md` and `ref_en.md` are MiniMax's own published H3 prompt-writing
guides, byte for byte. **They are the only authority on how to write an H3
prompt**; [`docs/prompting.md`](../docs/prompting.md) is our reading of them and
says so, and `docs/prompting.md` §14 ranks every other source that claims to
govern a prompt.

## Why they are here rather than in `internal/`

They were gitignored until 2026-09-01, and `bench/preflight_graph.py` reads them
**at import** — it parses the three keyframe alignment templates out of
`base_en.md` rather than retyping them, which is the right design. The
consequence was that the prompt tooling could not run at all on a checkout
without `internal/`. Vendor-published text that a shipped check depends on
belongs in the tracked tree.

## Why not `vendor_config/`

That directory holds files from the **HF model release**, and
`bench/check_vendor_config.py::ORIGIN` maps each one to its path inside that
release (`tokenizer/`, `processor/`, `FL2VA/`). These guides are not in the
model release — they come from MiniMax's GitHub repository. Putting them there
would break that checker's contract, so they get their own directory and their
own hash record.

## Provenance, verified rather than assumed

MiniMax ship these same two files inside their own prompt-writing skill, at
`.claude/skills/h3-prompt-writing/references/` and `.agents/skills/…` in their
GitHub repo. On 2026-09-01 both of ours were confirmed **byte-identical by
SHA-256** to that bundle, and the two bundle copies identical to each other.

`sha256.json` records the hashes. **Do not edit these files.** They are the
comparison basis; an edit makes every check that parses them agree with us
instead of with the vendor.

**A third-party fork exists and is easy to mistake for the original** — see
`docs/prompting.md` §14.2b. One of its two guides is byte-identical to the
vendor's and the other is not, so hashing one and generalising gets the wrong
answer.
