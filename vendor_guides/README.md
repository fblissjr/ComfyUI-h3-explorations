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

`sha256.json` records the hashes, **and `bench/check_prompt_docs_sync.py`
verifies them** — a hash record nothing reads is decoration, and this one went
unread for its first hours. **Do not edit these files.** They are the
comparison basis; an edit makes every check that parses them agree with us
instead of with the vendor, silently, because they all re-derive from whatever
is here. **Revising a guide is a deliberate, visible event**: replace the file,
update `sha256.json`, and expect the downstream checks to report what moved.

**A third-party fork exists and is easy to mistake for the original** — see
`docs/prompting.md` §14.2. One of its two guides is byte-identical to the
vendor's and the other is not, so hashing one and generalising gets the wrong
answer.

## `h3_image/`: the MiniMax-H3-Image prompt-expansion prompts

A different thing from the two guides above, kept here at the owner's word
(2026-11-11): the instructions MiniMax give a language model to turn a short
request into the long description MiniMax-H3-Image is given. They came with the
model, in two repositories. The prompt expander's repository
(`MiniMax-H3-Image-PE`) holds the `prompt/` folder, whose own `manifest.json`
lists a SHA-256 for each file, and a `README.md` that says how each prompt is
sent and how its reply is used. The image model's repository holds the
`examples/` folder. `h3_image/` keeps those text files at the paths they had,
byte for byte, with the release's `LICENSE` whole beside them.

- `README.md`: the expander repository's own README, the authority on how a
  request is put together from the prompts and what to do with the reply. The
  client script it names (`scripts/expand_prompt.py`) was not in the download
  and is not here.

- `prompt/t2i/`: the system prompt, the task prompt and the request settings
  for text to image.
- `prompt/edit/task_prompt.txt`: the one prompt for an edit.
- `examples/`: the release's two worked cases (the short request, the
  description made from it, the size chosen) and the two finished prompts its
  scripts send.

`sha256.json` pins every file, and `bench/check_prompt_docs_sync.py` verifies
them and verifies that the two standalone copies in
[`docs/wiki/system_prompts/`](../docs/wiki/system_prompts/README.md)
(`h3_image_t2i_system_prompt.md`, `h3_image_edit_system_prompt.md`) are the two
task prompts here, byte for byte. The same rule as above holds: never edit them.
Whether the two video guides above also describe how to prompt this model is
not established; these files are the vendor's own statement for it.
