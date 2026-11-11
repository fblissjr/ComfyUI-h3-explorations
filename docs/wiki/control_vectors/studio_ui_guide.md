# The browser UI

last updated: 2026-11-11

[Back to the start page](../control_vectors.md)

The Control Vector Studio is the control vector command line as a page. Every control on it is drawn from the CLI's
own parser, and every button that builds or saves runs the CLI's own handler, so the page and the command line cannot
say different things about a method. It was rebuilt on 2026-11-11 on the owner's word; until then it was a hand-written
page that had fallen behind the code ([`../decisions.md`](../decisions.md), 2026-11-11).

It starts with:

```bash
<comfy-venv-python> bench/control_vectors.py ui
```

## How it stays in step with the CLI

- **The page asks the server what the CLI takes.** `GET /api/schema` is the parser walked
  (`bench/control_vector/ui_schema.py::method_schema`): each subcommand, each argument, its help, default and choices.
  `bench/control_vector/web/form.js` draws a method's form from that. Add a flag to `bench/control_vector/cli.py` and it is on the page.
- **A button posts arguments, and the CLI runs them.** `POST /api/run` takes a method and an `{argument: value}` map,
  writes the argument list the command line would carry (`bench/control_vector/ui_schema.py::argv_for`), parses it with the CLI's parser and
  calls `bench/control_vector/cli.py::dispatch`, the same routing `main` uses. The result shows the command the request
  stood for.
- **A preview runs the method's own code.** "Preview the pairs" returns what `from-prompt` would encode
  (`bench/control_vector/helpers.py::_compile_pair_set_from_args`, which `build` and `make` call too); the words and sculpt previews read
  out what `words` and `sculpt` would save, without saving.
- **The node's lists come from the node.** The sculpt screen's modes are `control_vectors.APPLY_MODES` and the prompt
  meter's masks and default are the node's (`h3_rules.CONTROL_VECTOR_DEFAULTS`). Where a method cannot use a choice, the
  method refuses it and the page shows the reason.

[`bench/check_control_vector_studio.py`](../../../bench/check_control_vector_studio.py) holds all of that with the card
masked: the schema against the parser, a page request against a command line for every method, every route, and the
page's scripts against the routes. It fails if a form control or a list of modes is written into the page by hand.

## The screens

| Screen | What it runs |
|---|---|
| Library | lists the vectors and pair files; `rebuild` |
| Pairs | `from-prompt` with a preview of the compiled pairs; a pair file editor that saves with `from-md`; `build`; `generate-pairs` into the editor |
| Images | `make-media` |
| Words | `words`, with a preview |
| Sculpt | `sculpt`, with a preview |
| Inspect | a saved vector's word readout; the prompt meter; `inspect`; `grade` |
| All methods | every subcommand as its own form, including `make` and `build-j-lens` |

Two things the page decides that the parser does not: which folder the studio writes to (`bench/control_vector/ui_schema.py::SERVER_SET`,
the vector and pair folders given at launch), and how a control is drawn (`WIDGETS`, `PRIMARY`). Neither is an option
or a default.

## Who may call the server

The run route executes any method, so the server answers only its own page
(`bench/control_vector/ui_server.py::request_refusal`):

- **A POST needs the session token.** The server makes a token at start, the page reads it from `/api/init` and sends
  it back in a header. A page on another site cannot read it or send that header.
- **`Host` and `Origin` must be this server.** A request under a DNS name the server was not started with is refused,
  which is what stops another site from re-pointing its own name at this machine. To serve under a name, start it with
  `--host <name>`.
- **A method run from the page writes only inside the studio's two folders.** An output path elsewhere is refused
  before the method runs (`bench/control_vector/ui_schema.py::OUTPUT_PATHS`). Use the command line to write elsewhere,
  or start the studio with `--vec-dir` and `--toml-dir`. Paths a method reads are not restricted.

## What changed for someone who used the old page

- **Edited pairs go through a pair file.** The old page let pairs be edited in the page and built from memory. Now
  "Edit as a pair file" moves the compiled pairs into the editor, the editor saves a `.toml`, and `build` reads it, so
  what was built can always be built again from the command line.
- **Re-pool is gone.** It had no command line form, and with nothing cached it started a full build. It returns when
  the CLI has a method for it.
- **Nothing overwrites unless `--overwrite` is ticked**, as on the command line. The old page always overwrote.
- **The families setting is per method**, at the method's own default. The old page had one setting for every screen,
  which built image contrasts for both families where the method's default is `ref2va`.

## What has been run, and what has not

Run on 2026-11-11: the page in headless Chrome against a server with the card masked (every screen drawn, a words save,
a sculpt save, a pairs preview and a pair file save, a word readout), and one `from-prompt` build on the card through
`/api/run`. Not run through the page on the card: `build` from a pair file, `make-media`, `rebuild`, `inspect`, `grade`,
the prompt meter and `generate-pairs`.

The page shows the CLI as it is, so the CLI's own gaps show too: `from-prompt --frame` has no effect, `--mode l2va` is
offered and refused, `--ref-role` and `generate-pairs --load-in-4bit` are read by nothing, and `grade` applies in mode
`add` where the node's default is `add_norm_matched`. Those are the CLI's to fix.
