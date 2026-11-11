#!/usr/bin/env python3
"""Check that the Control Vector Studio is the CLI: its schema, its run route and its page.

The studio (`bench/control_vectors.py ui`) draws its controls from the CLI's parser and runs
the CLI's handlers. These cases hold that, so the page cannot fall behind the command line the
way the hand-written one did (it described a design the code had retired, and reached a small
part of the CLI's arguments; `internal/` notes of 2026-11-11).

Claims, i.e. what breaks if a case is deleted:
  schema is the parser     every subcommand but `ui`, and every argument its help shows, is in
                           `GET /api/schema` with the parser's own default and choices; an
                           argument dropped from the description goes red (control)
  request is the argv      a `{dest: value}` map parses to the namespace the command line
                           itself parses to, for a line of every method; a value that starts
                           with a dash is not read as a flag; an unknown name, and a folder
                           the studio sets itself, are refused
  run is the dispatch      `POST /api/run` hands `cli.dispatch` that same namespace, plus the
                           studio's folders; a parser refusal comes back as HTTP 400 with the
                           parser's message
  previews share the code  `/api/compile` returns the pair set `from-prompt` would encode
                           (`helpers._compile_pair_set_from_args`), and `/api/preview-recipe`
                           the steps `sculpt` would compose (`sculpt._sculpt_steps_from_args`)
  routes answer            every route in `ui_server.ROUTES` answers with the card masked;
                           static files are served from `web/` only
  page calls the routes    every `/api/...` the page's scripts name is a route, and every
                           route is called by the page
  page writes no options   `web/index.html` has no form control, and no script holds a
                           literal list of a method's modes, masks or poolings
  hints name real things   every name in `ui_schema.WIDGETS`, `PRIMARY` and `SERVER_SET` is an
                           argument the parser has

Not covered: a build on the card through the page, and the page in a browser. The words and
sculpt cases need the J-Lens bundle and two library vectors on disk; without them those cases
say so and are skipped, never passed.

No GPU, no server.

    <comfy-venv-python> bench/check_control_vector_studio.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from _lib import bootstrap  # noqa: E402

bootstrap(cpu=True)

import folder_paths  # noqa: E402

import control_vector_nodes as cv_nodes  # noqa: E402
import control_vectors as cv  # noqa: E402
import h3_spans  # noqa: E402
import j_lens  # noqa: E402
from bench.control_vector import cli, ui_schema, ui_server  # noqa: E402
from bench.control_vector.helpers import WEB_DIR, _compile_pair_set_from_args  # noqa: E402
from bench.control_vector.sculpt import _sculpt_steps_from_args  # noqa: E402
from bench.control_vector.ui_state import ControlVectorUIState  # noqa: E402

SCENE = (
    "integrated_multimodal_description:\n[Shot 1] In an open desert highway overlook filmed under crisp overcast "
    "afternoon light with cool slate shadows, a traveler in a canvas jacket (S1) says in a calm, even voice, "
    "<d>[English] We can cross the valley before sunset.</d>\n\noverall_soundscape:\nQuiet open-air room tone.\n\n"
    "non_diegetic_music:\nN/A"
)
WORDS = "under crisp overcast afternoon light with cool slate shadows -> in a blistering midday desert heatwave"

#: One command line per method, as a person would type it, beside the `{dest: value}` map the
#: page would post for the same thing. The case holds the two to the same namespace.
LINES: dict[str, tuple[list[str], dict]] = {
    "from-prompt": (
        ["from-prompt", "--target-prompt", SCENE, "--words", WORDS, "--name", "heat", "--bank-scenes", "2",
         "--hold", "a.safetensors", "--hold", "b.safetensors", "--no-encoder", "--pooling", "diff_span,scene_text"],
        {"target_prompt": SCENE, "words": WORDS, "name": "heat", "bank_scenes": 2, "hold": ["a.safetensors", "b.safetensors"],
         "include_encoder": False, "pooling": "diff_span,scene_text"},
    ),
    "build": (["build", "film_16mm_1974", "--families", "fl2va", "--unit-mean", "--overwrite"],
              {"source": "film_16mm_1974", "families": "fl2va", "unit_mean": True, "overwrite": True}),
    "from-md": (["from-md", "pairs.md", "--name", "x", "--frame", "bare"], {"md_path": "pairs.md", "name": "x", "frame": "bare"}),
    "make": (["make", "film_16mm_1974", "--pooling", "scene_text"], {"pairs": "film_16mm_1974", "pooling": "scene_text"}),
    "make-media": (["make-media", "--image-a", "a.png", "--image-b", "b.png", "--short-edge", "384"],
                   {"image_a": "a.png", "image_b": "b.png", "short_edge": 384}),
    "sculpt": (["sculpt", "one", "two", "--modes", "add,hold", "--strengths=-1.0,0.5", "--invert"],
               {"vectors": ["one", "two"], "modes": "add,hold", "strengths": "-1.0,0.5", "invert": True}),
    "inspect": (["inspect", "film_16mm_1974", "--meter", "p1", "--meter", "p2", "--mask", "look_only"],
                {"vector": "film_16mm_1974", "meter": ["p1", "p2"], "mask": "look_only"}),
    "grade": (["grade", "--vector", "v", "--prompt", "p", "--no-dit", "--strengths=-1.0,1.0"],
              {"vector": "v", "prompt": "p", "no_dit": True, "strengths": "-1.0,1.0"}),
    "words": (["words", "blizzard, snow", "--against-words", "@neutral", "--hold-alpha", "0.5"],
              {"pos_words": "blizzard, snow", "against_words": "@neutral", "hold_alpha": 0.5}),
    "generate-pairs": (["generate-pairs", "warm to cold", "--no-4bit", "--temperature", "0.2"],
                       {"concept": "warm to cold", "no_4bit": True, "temperature": 0.2}),
    "build-j-lens": (["build-j-lens", "--verify-only"], {"verify_only": True}),
    "rebuild": (["rebuild", "film_16mm_1974", "--force"], {"target": "film_16mm_1974", "force": True}),
}


def _state(tmp: Path) -> ControlVectorUIState:
    (tmp / "toml").mkdir(parents=True, exist_ok=True)
    (tmp / "vec").mkdir(parents=True, exist_ok=True)
    state = ControlVectorUIState(toml_dir=tmp / "toml", vec_dir=tmp / "vec")
    state.get_clip = lambda *a, **k: "<encoder not loaded in this check>"  # type: ignore[method-assign]
    return state


def _call(state: ControlVectorUIState, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    raw = json.dumps(body).encode() if body is not None else b""
    code, ctype, payload = ui_server.handle_ui_request(state, method, path, raw)
    return code, (json.loads(payload) if ctype == "application/json" else {"text": payload})


def _public(ns: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(ns).items() if not k.startswith("_")}


def _parser_args() -> dict[str, dict[str, list[argparse.Action]]]:
    """`{method: {dest: [actions]}}` for every argument a subcommand's help shows."""
    out: dict[str, dict[str, list[argparse.Action]]] = {}
    for name, (sp, _aliases) in ui_schema._subparsers(cli.build_parser()).items():
        by_dest: dict[str, list[argparse.Action]] = {}
        for a in sp._actions:
            if isinstance(a, argparse._HelpAction) or a.help is argparse.SUPPRESS:
                continue
            by_dest.setdefault(a.dest, []).append(a)
        out[name] = by_dest
    return out


def _schema_gaps(schema: dict) -> list[str]:
    gaps: list[str] = []
    parser_args = _parser_args()
    want_methods = set(parser_args) - set(ui_schema.SKIP_METHODS)
    if set(schema["methods"]) != want_methods:
        gaps.append(f"methods {sorted(schema['methods'])} != the parser's {sorted(want_methods)}")
    for name in sorted(want_methods & set(schema["methods"])):
        sp = ui_schema._subparsers(cli.build_parser())[name][0]
        described = {a["dest"]: a for a in schema["methods"][name]["args"]}
        expected = set(parser_args[name]) - set(ui_schema.SERVER_SET.get(name, {}))
        if set(described) != expected:
            gaps.append(f"{name}: described {sorted(set(described) ^ expected)} differ from the parser")
        for dest, entry in described.items():
            acts = parser_args[name].get(dest)
            if not acts:
                continue
            default = sp.get_default(dest)
            if entry["kind"] == "bool":
                default = bool(default)
            elif entry["kind"] == "list":
                default = list(default or [])
            if entry["default"] != default:
                gaps.append(f"{name}.{dest}: default {entry['default']!r} != the parser's {default!r}")
            choices = [str(c) for c in acts[0].choices] if acts[0].choices else None
            if entry["choices"] != choices:
                gaps.append(f"{name}.{dest}: choices {entry['choices']} != the parser's {choices}")
    return gaps


def check_schema_is_the_parser() -> None:
    schema = ui_schema.method_schema()
    gaps = _schema_gaps(schema)
    assert not gaps, "; ".join(gaps[:6])
    assert schema["node"]["apply_modes"] == list(cv.APPLY_MODES)
    assert schema["node"]["mask_modes"] == list(h3_spans.INFERENCE_MASK_MODES)
    # Control: the comparison can fail. Drop one argument from one method and it must say so.
    broken = json.loads(json.dumps(schema, default=str))
    dropped = broken["methods"]["from-prompt"]["args"].pop()
    assert _schema_gaps(broken), f"dropping {dropped['dest']} from the description went unnoticed"


def check_request_is_the_argv() -> None:
    missing = set(ui_schema.method_schema()["methods"]) - set(LINES)
    assert not missing, f"no command line in LINES for {sorted(missing)}: a new method is not held here yet"
    parser = cli.build_parser()
    for method, (argv, values) in LINES.items():
        want = _public(parser.parse_args(argv))
        got = _public(ui_schema.parse_request(method, values))
        diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if got.get(k) != want.get(k)}
        assert not diff, f"{method}: the page's request and the command line differ in {diff}"
    # A value that starts with a dash stays a value.
    ns = ui_schema.parse_request("from-prompt", {"target_prompt": SCENE, "words": "--not-a-flag -> -x"})
    assert ns.words == "--not-a-flag -> -x", ns.words
    ns = ui_schema.parse_request("words", {"pos_words": "-cold"})
    assert ns.pos_words == "-cold", ns.pos_words
    for bad, needle in (
        ({"pos_words": "snow", "nope": 1}, "no argument"),
        ({"pos_words": "snow", "out_dir": "/elsewhere"}, "set by the studio"),
        ({"pos_words": "snow", "hold_alpha": True}, "takes a value"),
    ):
        try:
            ui_schema.parse_request("words", bad)
        except ValueError as exc:
            assert needle in str(exc), f"{bad}: refused with {exc!r}, expected {needle!r}"
        else:
            raise AssertionError(f"{bad} was accepted")
    try:
        ui_schema.parse_request("ui", {})
    except ValueError:
        pass
    else:
        raise AssertionError("the studio's own subcommand was accepted as a method")


def check_run_is_the_dispatch() -> None:
    seen: list[argparse.Namespace] = []
    real = cli.dispatch
    cli.dispatch = lambda ns: seen.append(ns) or 0  # type: ignore[assignment]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            state = _state(Path(tmp))
            parser = cli.build_parser()
            for method, (argv, values) in LINES.items():
                seen.clear()
                code, res = _call(state, "POST", "/api/run", {"method": method, "args": values})
                assert code == 200 and res["ok"], f"{method}: {code} {res.get('error')}"
                assert len(seen) == 1, f"{method}: dispatch called {len(seen)} time(s)"
                want = _public(parser.parse_args(argv))
                got = _public(seen[0])
                placed = ui_schema.SERVER_SET.get(method, {})
                for dest, attr in placed.items():
                    assert got[dest] == str(getattr(state, attr)), f"{method}: {dest} is {got[dest]!r}, not the studio's {attr}"
                skip = set(placed) | {"toml_out", "out"}
                diff = {k: (got.get(k), want.get(k)) for k in set(got) | set(want) if k not in skip and got.get(k) != want.get(k)}
                assert not diff, f"{method}: dispatch got a namespace that differs from the command line's in {diff}"
                assert res["command"].startswith(f"bench/control_vectors.py {method}"), res["command"]
            code, res = _call(state, "POST", "/api/run", {"method": "from-prompt", "args": {"words": "a -> b"}})
            assert code == 400 and "--target-prompt" in res["error"], (code, res)
            code, res = _call(state, "POST", "/api/run", {"method": "nope", "args": {}})
            assert code == 400 and "unknown method" in res["error"], (code, res)
    finally:
        cli.dispatch = real  # type: ignore[assignment]


def check_previews_share_the_code() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state = _state(Path(tmp))
        values = {"target_prompt": SCENE, "words": WORDS, "name": "heat"}
        code, res = _call(state, "POST", "/api/compile", {"args": values})
        assert code == 200, res
        ps = _compile_pair_set_from_args(cli.build_parser().parse_args(
            ["from-prompt", "--target-prompt", SCENE, "--words", WORDS, "--name", "heat"]))
        assert [(p["a"], p["b"], p["weight"]) for p in res["pairs"]] == [(p.a, p.b, float(p.weight)) for p in ps.pairs]
        assert (res["name"], res["mode"]) == (ps.name, ps.mode), (res["name"], res["mode"])
        code, res = _call(state, "POST", "/api/compile", {"args": {"words": WORDS}})
        assert code == 400, "a contrast with no target prompt was compiled"

        argv, sculpt_values = LINES["sculpt"]
        want = _sculpt_steps_from_args(cli.build_parser().parse_args(argv))
        assert want[0]["strength"] == -1.0 and want[1] == {"vector": "two", "mode": "hold", "strength": 0.5}, want
        if not j_lens.bundle_available():
            print("        skipped: the recipe preview route (no J-Lens bundle on disk)")
            return
        lib = Path(folder_paths.get_folder_paths(cv_nodes.CONTROL_VECTORS_FOLDER)[0])
        vecs = sorted(lib.glob("*__fl2va__*.safetensors"))[:2]
        if len(vecs) < 2:
            print("        skipped: the recipe preview route (fewer than two fl2va vectors in the library)")
            return
        for v in vecs:
            shutil.copy2(v, state.vec_dir / v.name)
        recipe = {"vectors": [v.name for v in vecs], "modes": "add,add_norm_matched", "strengths": "-1.0,0.5", "families": "fl2va"}
        code, res = _call(state, "POST", "/api/preview-recipe", {"args": recipe})
        assert code == 200, res
        assert res["steps"] == [
            {"vector": vecs[0].name, "mode": "add", "strength": -1.0},
            {"vector": vecs[1].name, "mode": "add_norm_matched", "strength": 0.5},
        ], res["steps"]
        assert not list(state.vec_dir.glob("*sculpt*")), "a preview saved a vector"


def check_routes_answer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        state = _state(Path(tmp))
        bodies: dict[str, dict] = {
            "/api/run": {"method": "words", "args": {}},
            "/api/compile": {"args": {"target_prompt": SCENE, "words": WORDS}},
            "/api/preview": {"text": ""},
            "/api/load-toml": {"file": "film_16mm_1974.toml"},
            "/api/save-toml": {"text": "", "args": {}},
            "/api/preview-words": {"args": {"pos_words": "snow", "families": "fl2va"}},
            "/api/preview-recipe": {"args": {}},
            "/api/j-lens": {},
            "/api/prompt-meter": {},
            "/api/generate-pairs-llm": {"args": {}},
        }
        for (method, path), _fn in ui_server.ROUTES.items():
            if path == "/api/quit":
                continue
            code, res = _call(state, method, path, bodies.get(path))
            assert code in (200, 400), f"{method} {path} answered {code}"
            assert "ok" in res, f"{method} {path} answered without 'ok': {sorted(res)[:6]}"
            assert code == 200 or res.get("error"), f"{method} {path} refused with no reason"
        # A body no route can act on is refused by the route, never by a crash.
        code, res = _call(state, "POST", "/api/run", {"method": "words", "args": "not an object"})
        assert code == 400 and "object" in res["error"], res
        code, _ = _call(state, "POST", "/api/build", {})
        assert code == 404, "a route that is not in ROUTES answered"
        # Loading and saving a pair file through `from-md`, into the studio's own folder.
        code, lt = _call(state, "POST", "/api/load-toml", {"file": "film_16mm_1974.toml"})
        assert code == 200 and lt["n_pairs"] > 0, lt
        code, sv = _call(state, "POST", "/api/save-toml", {"text": lt["markdown"], "args": {"name": "film_copy"}})
        assert code == 200 and Path(sv["toml_path"]).is_file() and Path(sv["toml_path"]).parent == state.toml_dir, sv
        assert sv["command"].startswith("bench/control_vectors.py from-md"), sv["command"]
        code, again = _call(state, "POST", "/api/save-toml", {"text": lt["markdown"], "args": {"name": "film_copy"}})
        assert code == 400 and "already exists" in again["error"], again
        # Static files: the page, its scripts, and nothing outside web/.
        for path, want in (("/", 200), ("/ui/app.js", 200), ("/ui/screens/pairs.js", 200), ("/ui/../cli.py", 404),
                           ("/ui/missing.js", 404), ("/ui/../../check_control_vector_studio.py", 404)):
            code, _ctype, _payload = ui_server.handle_ui_request(state, "GET", path)
            assert code == want, f"GET {path} answered {code}, expected {want}"
        if j_lens.bundle_available():
            code, res = _call(state, "POST", "/api/run", {"method": "words", "args": {"pos_words": "blizzard, snow", "families": "fl2va", "name": "check_words"}})
            assert code == 200 and res["rc"] == 0, res
            assert [b["family"] for b in res["built"]] == ["fl2va"], res["built"]
            assert all(Path(state.vec_dir / b["file"]).is_file() for b in res["built"]), "the vector was not written to the studio's folder"
        else:
            print("        skipped: a words run through /api/run (no J-Lens bundle on disk)")


def _page_scripts() -> dict[str, str]:
    return {str(p.relative_to(WEB_DIR)): p.read_text(encoding="utf-8") for p in sorted(WEB_DIR.rglob("*.js"))}


def check_page_calls_the_routes() -> None:
    scripts = _page_scripts()
    called = set()
    for text in scripts.values():
        called |= set(re.findall(r"['\"`](/api/[a-z\-]+)['\"`]", text))
    routes = {path for (_m, path) in ui_server.ROUTES}
    assert called <= routes, f"the page calls {sorted(called - routes)}, which the server does not answer"
    assert routes <= called, f"the server answers {sorted(routes - called)}, which the page never calls"
    # Every script the page imports is on disk under web/.
    for name, text in scripts.items():
        for rel in re.findall(r"from\s+['\"](\.{1,2}/[^'\"]+)['\"]", text):
            target = (WEB_DIR / name).parent / rel
            assert target.resolve().is_file(), f"{name} imports {rel}, which is not on disk"
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    for el_id in set(re.findall(r"getElementById\('([^']+)'\)", "\n".join(scripts.values()))):
        assert f'id="{el_id}"' in html, f"a script looks up #{el_id}, which index.html does not define"


def _hand_written_lists(html: str, scripts: dict[str, str]) -> list[str]:
    found: list[str] = []
    for tag in ("select", "option", "input", "textarea"):
        if re.search(rf"<{tag}\b", html):
            found.append(f"index.html writes a <{tag}>")
    # A script that names two or more of a list's members as string literals is holding a copy of it.
    lists = {
        "apply modes": [m for m in cv.APPLY_MODES if m != "add"],
        "masks": list(h3_spans.INFERENCE_MASK_MODES),
        "poolings": list(cv.VALID_POOLING),
        "modes": [m for m in cv.VALID_MODES],
    }
    for name, text in scripts.items():
        code = re.sub(r"//[^\n]*", "", text)
        for label, members in lists.items():
            hits = [m for m in members if re.search(rf"['\"`]{re.escape(m)}['\"`]", code)]
            if len(hits) >= 2:
                found.append(f"{name} writes {label} {hits} as literals")
    return found


def check_page_writes_no_options() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    scripts = _page_scripts()
    found = _hand_written_lists(html, scripts)
    assert not found, "; ".join(found)
    # Controls: both halves can fail.
    assert _hand_written_lists(html + '<select id="x"><option value="add">add</option></select>', scripts)
    assert _hand_written_lists(html, {**scripts, "x.js": "const MASKS = ['scene_only', 'look_only'];"})


def check_hints_name_real_things() -> None:
    parser_args = _parser_args()
    every_dest = {d for dests in parser_args.values() for d in dests}
    unknown = sorted(set(ui_schema.WIDGETS) - every_dest)
    assert not unknown, f"WIDGETS names {unknown}, which no method has"
    for table_name, table in (("PRIMARY", ui_schema.PRIMARY), ("SERVER_SET", ui_schema.SERVER_SET)):
        for method, dests in table.items():
            assert method in parser_args, f"{table_name} names the method {method!r}, which the parser does not have"
            hidden = {a.dest for a in ui_schema._subparsers(cli.build_parser())[method][0]._actions}
            missing = sorted(d for d in dests if d not in hidden)
            assert not missing, f"{table_name}[{method!r}] names {missing}, which {method} does not have"


def main() -> int:
    cases = (
        ("the schema is the parser: every method and argument, its default and choices", check_schema_is_the_parser),
        ("a page request parses to the command line's namespace, for a line of every method", check_request_is_the_argv),
        ("/api/run hands cli.dispatch that namespace and the studio's folders", check_run_is_the_dispatch),
        ("the previews return what their method would encode or compose", check_previews_share_the_code),
        ("every route answers with the card masked; static files stay inside web/", check_routes_answer),
        ("the page calls every route and only routes; its imports and ids resolve", check_page_calls_the_routes),
        ("the page writes no control and no option list of its own", check_page_writes_no_options),
        ("the presentation hints name arguments the parser has", check_hints_name_real_things),
    )
    failures = []
    print("Control Vector Studio: the page against the CLI")
    for label, fn in cases:
        try:
            fn()
            print(f"  ok    {label}")
        except Exception as exc:
            failures.append(label)
            print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
