# Asking an LLM to write pairs

last updated: 2026-11-09

[Back to the start page](../control_vectors.md)

Writing prompt pairs is not the main road any more. Steering by an edit of your own prompt needs no pairs
([`../edit_guidance.md`](../edit_guidance.md)), and a vector can be built from your prompt and one
`old -> new` clause. The bank of pair files is kept as a starter set and is not being grown (owner,
2026-11-09, [`../decisions.md`](../decisions.md)). This page is for the case where you still want an LLM to
write a set of pairs.

## The prompt to hand it

[`../system_prompts/h3_control_vector_standalone_system_prompt.md`](../system_prompts/h3_control_vector_standalone_system_prompt.md).
The whole file is the prompt. It names nothing the LLM cannot know: no product name, none of this repo's
files or tools. Examples of good and bad answers are beside it in
[`../system_prompts/h3_control_vector_standalone_system_prompt_examples.md`](../system_prompts/h3_control_vector_standalone_system_prompt_examples.md).

What it asks for, briefly, and why:

- Each pair changes one clause and nothing else, so the difference between the two sides is the thing being
  steered.
- The plain side is specific, not a bare "normal" or "natural", and the two sides are about the same
  length. Vectors built from a short plain clause against a long steered one share a common component with
  every other vector built that way.
- One edit a line.

## Turning the answer into a vector

Save the LLM's answer as a `.md` file and build from it; `bench/control_vectors.py build --help` and
`from-md --help` say how. The loader checks the pairs and refuses a file it cannot use
(`control_vectors.py::load_pairs`).

## What is not wired up

`bench/control_vectors.py generate-pairs`, which runs a local LLM for you, and the browser UI still read
the older page, [`../system_prompts/h3_control_vector_system_prompt.md`](../system_prompts/h3_control_vector_system_prompt.md),
not the standalone prompt above (`bench/control_vector/llm_pairs.py::_extract_system_prompt_text`). A check
pins rules in that older page (`bench/check_prompt_docs_sync.py`). Moving both to the standalone prompt is
listed in [`the plan`](../../research/control_vectors/2026-11-09_control_vector_simplification_plan.md) and
has not been done. No session ran `generate-pairs` on 2026-11-09.
