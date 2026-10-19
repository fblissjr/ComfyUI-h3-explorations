# Working: running the box, and working with the owner

| page | what it is for |
|---|---|
| [`working_with_the_owner.md`](working_with_the_owner.md) | **read before speaking to the owner.** Their clock, batching, never re-asking, where links go, how a clip is shown, writing for them |
| [`comfy_notes.md`](comfy_notes.md) | **read before the first command.** The venv and `uv`, starting and stopping ComfyUI and the review server, arming keys and captures, what is cached, changing a graph, the traps, the `node_id` rule |
| [`runbook.md`](runbook.md) | the card and ledger runbook, pointers only: ComfyUI, one load from preview to land, the review server, provenance, end of day, a render |
| [`hardware.md`](hardware.md) | what bounds this workload and which host settings invalidate a timing |
| [`pipeline_telemetry.md`](pipeline_telemetry.md) | the `H3_TELEMETRY` recorder: what a prompt loaded, moved and spent per node, its schema, what it cannot see |
| [`capture_manifest_schema.md`](capture_manifest_schema.md) | the activation capture manifest; accepted versions are `bench/check_capture_manifest.py::SCHEMA_VERSIONS` |
| [`coderef.md`](coderef.md) | the sister checkouts under `coderef/`: what each implements for H3, its current state at the revision last read, what upstream does that we do not. Read before proposing a borrow |

The cross-team protocol between the Gemini and Claude leads is `internal/gemini/PROTOCOL.md` (gitignored).
