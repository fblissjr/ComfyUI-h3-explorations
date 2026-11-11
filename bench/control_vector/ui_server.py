"""HTTP server and `/api/*` routes for the H3 Control Vector Studio.

**The page runs the CLI.** `POST /api/run` takes a method name and a `{argument: value}` map,
writes the argument list the command line would carry (`ui_schema.argv_for`), parses it with the
CLI's own parser and calls the CLI's own dispatch (`cli.dispatch`). The page's controls are drawn
from `GET /api/schema`, which is that parser walked. So a flag added to the CLI is on the page and
reaches its handler with no edit here or in `web/`.

The other routes are previews and listings: they read out what a method would do without saving
(`/api/compile`, `/api/preview-words`, `/api/preview-recipe`), or they show a file or a vector
(`/api/load-toml`, `/api/preview`, `/api/j-lens`, `/api/prompt-meter`). Each preview parses its
arguments with the same parser and calls the same helper its method's handler calls.

`ROUTES` is the whole list. `bench/check_control_vector_studio.py` calls every one with the card
masked and holds the page's calls to it.

**Who may call it.** The run route executes any CLI method, so nothing is served without the
launch secret. The server makes a token at start and prints a link that carries it; opening the
link sets a cookie (`HttpOnly`, `SameSite=Strict`) and every request after, the page's files
included, must bring it back (`gate`). A page on another site cannot: the browser does not send
a `SameSite=Strict` cookie on its requests, and a name re-pointed at this machine has no cookie
of ours. A machine that can merely reach the port gets nothing either, which is what lets the
studio sit behind an HTTPS front on a tailnet the way the phone studio does. A method run from the
page also writes only inside the studio's vector and pair folders (`ui_schema.OUTPUT_PATHS`,
`_refuse_outside_paths`). Paths a method reads (a prompt file, a pair file, a picture) are not
restricted: reading the files the person at the page names is what the methods are for.
"""

from __future__ import annotations

import argparse
import contextlib
import hmac
import io
import json
from pathlib import Path
import shlex
import tempfile
import threading
from typing import Any, Callable
import urllib.parse

import control_vector_nodes as cv_nodes
import control_vectors as cv
import h3_rules
import h3_spans
import j_lens

from . import cli as _cli
from .helpers import (
    WEB_DIR,
    _compile_pair_set_from_args,
    _display_path,
    _pair_set_to_markdown,
    _recommended_strength_text,
    _resolve_requested_families,
)
from .llm_pairs import (
    SUPPORTED_LLM_PROVIDERS,
    _extract_system_prompt_text,
    _generate_pairs_from_args,
)
from .sculpt import _meter_prompt_tokens, _sculpt_steps_from_args
from .ui_schema import OUTPUT_PATHS, SERVER_SET, argv_for, method_schema, parse_request
from .ui_state import (
    ControlVectorUIState,
    _discover_ui_clips,
    _list_ui_toml_files,
    _list_ui_vectors,
    _preview_markdown_pairs,
)

#: Static files the page may load, by extension. Anything else under `web/` is not served.
_STATIC_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}

#: Methods whose handler takes the studio's loaded encoder and refiner cache, so a second build
#: in one session does not load them again. These are caches the handlers already accept, not
#: arguments: the namespace the page's arguments parse to is otherwise handed over untouched.
_ENCODER_METHODS: frozenset[str] = frozenset({"from-prompt", "build", "make", "make-media", "rebuild"})
#: Methods that load their own encoder or DiT: the studio lets go of its copy first.
_OWN_MODEL_METHODS: frozenset[str] = frozenset({"inspect", "grade", "build-j-lens", "generate-pairs"})

#: One card job at a time: the handlers print to a redirected stdout and share the card.
_RUN_LOCK = threading.Lock()


class RouteError(ValueError):
    """A refusal the page should show as the reason, with HTTP 400."""


def cookie_name(port: int) -> str:
    """Cookies are shared across ports of one host, so each studio's is named for its port."""
    return f"h3_cv_studio_{int(port)}"


def _cookie_value(headers: Any, name: str) -> str:
    for part in str(headers.get("Cookie", "") or "").split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value
    return ""


def gate(state: ControlVectorUIState, method: str, raw_path: str, headers: Any, *, port: int) -> tuple[str, str, str]:
    """Decide what a request gets before any route or file is touched.

    Returns one of:
      `("pass", "", "")`              the request brought the session cookie
      `("enter", location, cookie)`   a GET of the launch link: set the cookie, send it to `location`
      `("refuse", reason, "")`        anything else

    - **The launch link** is a GET with `?token=<the session token>`. It is answered with a
      redirect to the same path without the token, so the secret does not stay in the address
      bar, and a `Set-Cookie` the page's scripts cannot read.
    - **Every other request needs the cookie**, a static file as much as a POST.
    - **`Origin`, when the browser sends one, must be this server**, and a POST must be JSON.
      Both are already implied by the cookie's `SameSite=Strict`; they are kept so the refusal
      does not rest on one browser rule.
    """
    name = cookie_name(port)
    parsed = urllib.parse.urlsplit(raw_path)
    if method == "GET":
        offered = urllib.parse.parse_qs(parsed.query).get("token", [""])[0]
        if offered:
            if not hmac.compare_digest(offered.encode(), state.session_token.encode()):
                return "refuse", "that link's token is not this session's; use the link the studio printed at start", ""
            rest = urllib.parse.urlencode([(k, v) for k, v in urllib.parse.parse_qsl(parsed.query) if k != "token"])
            location = parsed.path + (f"?{rest}" if rest else "")
            return "enter", location, f"{name}={state.session_token}; Path=/; HttpOnly; SameSite=Strict"
    if not hmac.compare_digest(_cookie_value(headers, name).encode(), state.session_token.encode()):
        return "refuse", "open the studio with the link it printed at start (it carries this session's token)", ""
    host_header = str(headers.get("Host", "") or "")
    origin = str(headers.get("Origin", "") or "").strip()
    if origin and origin.lower() not in (f"http://{host_header}".lower(), f"https://{host_header}".lower()):
        return "refuse", f"Origin {origin!r} is not this server", ""
    if method == "POST":
        ctype = str(headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            return "refuse", "a POST must be application/json", ""
    return "pass", "", ""


def _listing(state: ControlVectorUIState) -> dict[str, Any]:
    return {
        "vectors": _list_ui_vectors(state.vec_dir),
        "toml_files": _list_ui_toml_files(state.toml_dir),
    }


def _resolve_vector(state: ControlVectorUIState, name: str, family: str | None = None) -> Path:
    cand = state.vec_dir / str(name)
    if cand.is_file():
        return cand
    return cv_nodes.resolve_vector_file_path(str(name), preferred_family=family) if family else cv_nodes.resolve_vector_file_path(str(name))


def _request_args(req: dict[str, Any]) -> dict[str, Any]:
    args = req.get("args")
    if args is None:
        return {}
    if not isinstance(args, dict):
        raise RouteError("'args' must be an object of {argument: value}")
    return args


def _refuse_outside_paths(state: ControlVectorUIState, method: str, ns: argparse.Namespace) -> None:
    """A method run from the page writes inside the studio's two folders, or not at all."""
    roots = [state.vec_dir.resolve(), state.toml_dir.resolve()]
    for dest in OUTPUT_PATHS.get(method, ()):
        raw = getattr(ns, dest, None)
        if not raw or not str(raw).strip():
            continue
        target = Path(str(raw)).expanduser().resolve()
        if not any(target == root or root in target.parents for root in roots):
            raise RouteError(
                f"{method}: {dest} must be inside the studio's vector folder ({roots[0]}) or pair folder "
                f"({roots[1]}); got {str(raw)!r}. Use the command line to write elsewhere."
            )


def _apply_server_set(state: ControlVectorUIState, method: str, ns: argparse.Namespace) -> None:
    for dest, attr in SERVER_SET.get(method, {}).items():
        setattr(ns, dest, str(getattr(state, attr)))
    # A pair file the method writes goes to the studio's pair folder unless a path was typed.
    if method in ("from-prompt", "build") and not getattr(ns, "toml_out", None):
        writes_toml = bool(getattr(ns, "save_toml", False)) or str(getattr(ns, "source", "") or "").lower().endswith(".md")
        if writes_toml:
            ns.toml_out = str(state.toml_dir)
    if method == "from-md" and not getattr(ns, "out", None):
        ns.out = str(state.toml_dir)


def _command_line(method: str, values: dict[str, Any]) -> str:
    """The command the request stands for, for the page to show. A long value (an uploaded
    picture) is shown by its length."""
    shown = [a if len(a) <= 400 else f"{a.split('=', 1)[0]}=<{len(a)} characters>" for a in argv_for(method, values)]
    return shlex.join(["bench/control_vectors.py", *shown])


def run_method(state: ControlVectorUIState, method: str, values: dict[str, Any]) -> dict[str, Any]:
    """Parse `values` as `method`'s command line and run the CLI's handler for it."""
    ns = parse_request(method, values)
    command = _command_line(method, values)
    _refuse_outside_paths(state, method, ns)
    _apply_server_set(state, method, ns)
    rows_out: list[dict[str, Any]] = []
    ns._summary_rows_out = rows_out
    buf = io.StringIO()
    with _RUN_LOCK:
        if method in _OWN_MODEL_METHODS:
            state.release_clip_vram()
        if method in _ENCODER_METHODS:
            ns._refiner_cache = state._refiner_cache
            jlens_only = method == "build" and str(getattr(ns, "from_words", "") or "").strip()
            if not jlens_only:
                ns._clip_override = state.get_clip(getattr(ns, "encoder", None), getattr(ns, "device", None))
        with contextlib.redirect_stdout(buf):
            rc = int(_cli.dispatch(ns) or 0)
    return {
        "ok": rc == 0,
        "method": method,
        "command": command,
        "rc": rc,
        "log": buf.getvalue(),
        "built": rows_out,
        **_listing(state),
    }


# --- routes -----------------------------------------------------------------------------------


def _r_init(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "toml_dir": _display_path(state.toml_dir),
        "vec_dir": _display_path(state.vec_dir),
        "system_prompt": _extract_system_prompt_text(),
        "clips": _discover_ui_clips(),
        "j_lens_available": j_lens.bundle_available(),
        "llm_providers": list(SUPPORTED_LLM_PROVIDERS),
        "default_llm_provider": state.default_llm_provider,
        "default_llm_model": state.default_llm_model,
        **_listing(state),
    }


def _r_schema(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, **method_schema()}


def _r_run(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    method = str(req.get("method", "") or "").strip()
    if not method:
        raise RouteError("missing 'method'")
    return run_method(state, method, _request_args(req))


def _pair_rows(ps: cv.PairSet) -> list[dict[str, Any]]:
    rows = []
    for idx, p in enumerate(ps.pairs):
        _, section = cv.differing_section_info(p.a, p.b, frame=ps.frame)
        rows.append(
            {"index": idx, "weight": float(p.weight), "a": p.a, "b": p.b, "a_span": p.a_span, "b_span": p.b_span, "section": section}
        )
    return rows


def _r_compile(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """What `from-prompt` would encode for these arguments: the pairs, compiled and not encoded."""
    ns = parse_request("from-prompt", _request_args(req))
    ps = _compile_pair_set_from_args(ns)
    return {
        "ok": True,
        "name": ps.name,
        "mode": ps.mode,
        "frame": ps.frame,
        "quality": ps.quality,
        "a_side": ps.a_side,
        "b_side": ps.b_side,
        "differing_sections": cv.pair_set_differing_sections(ps.pairs, frame=ps.frame),
        "pairs": _pair_rows(ps),
        "holds": [{"index": h.index, "group": h.group, "weight": float(h.weight), "a": h.a, "b": h.b} for h in ps.holds],
        "markdown": _pair_set_to_markdown(None, ps),
    }


def _r_preview(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """Validate a Markdown pair document as the importer will read it."""
    return _preview_markdown_pairs(
        state,
        str(req.get("text", "") or ""),
        frame=str(req.get("frame", "scene") or "scene"),
        name=str(req.get("name", "") or "").strip(),
        mode=str(req.get("mode", "t2va") or "t2va"),
    )


def _r_load_toml(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    fname = str(req.get("file", "") or "").strip()
    if not fname:
        raise RouteError("missing 'file'")
    cand = state.toml_dir / (fname if fname.endswith(".toml") else f"{fname}.toml")
    tpath = cand if cand.is_file() else cv_nodes.resolve_pair_file_path(fname)
    ps = cv.load_pairs(tpath, bank_dir=cv_nodes.PROMPT_BANK_DIR)
    tp_cand = tpath.parent / f"{tpath.stem}_test_prompt.txt"
    return {
        "ok": True,
        "file": tpath.name,
        "path": str(tpath),
        "name": ps.name,
        "frame": ps.frame,
        "mode": ps.mode,
        "quality": ps.quality,
        "a_side": ps.a_side,
        "b_side": ps.b_side,
        "n_pairs": len(ps.pairs),
        "n_holds": len(ps.holds),
        "test_scene_prompt": tp_cand.read_text(encoding="utf-8").strip() if tp_cand.is_file() else "",
        "pairs": _pair_rows(ps),
        "markdown": _pair_set_to_markdown(tpath, ps),
    }


def _r_save_toml(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """`from-md` on the editor's text: the text is written to a file and the method is run on it."""
    text = str(req.get("text", "") or "")
    if not text.strip():
        raise RouteError("no Markdown pairs to save")
    values = dict(_request_args(req))
    name = str(values.get("name", "") or "").strip()
    if not name:
        raise RouteError("a name is required to save a pair file")
    if not cv.NAME_RE.match(name):
        # The name becomes a file name here and in from-md; nothing but a plain stem is taken.
        raise RouteError(f"the name must match [a-z0-9_]+, got {name!r}")
    if "md_path" in values:
        raise RouteError("from-md here reads the editor's text; 'md_path' cannot be sent")
    with tempfile.TemporaryDirectory(prefix="h3_cv_studio_") as tmp:
        md_path = (Path(tmp) / f"{name}.md").resolve()
        if md_path.parent != Path(tmp).resolve():
            raise RouteError(f"the name {name!r} does not stay inside the working folder")
        md_path.write_text(text, encoding="utf-8")
        res = run_method(state, "from-md", {**values, "md_path": str(md_path)})
    out_arg = str(values.get("out", "") or "").strip()
    out_target = Path(out_arg) if out_arg else state.toml_dir
    saved = out_target if out_target.suffix.lower() == ".toml" else out_target / f"{name}.toml"
    return {**res, "name": name, "toml_path": str(saved), "toml": _display_path(saved)}


def _r_preview_words(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """What `words` would synthesize, read out and not saved."""
    if not j_lens.bundle_available():
        raise RouteError("J-Lens bundle models/j_lens/j_lens_h3.safetensors not found")
    ns = parse_request("words", _request_args(req))
    fams = _resolve_requested_families(getattr(ns, "families", "both"))
    out: list[dict[str, Any]] = []
    for fam in fams:
        synth = j_lens.synthesize_from_words(
            str(ns.pos_words).strip(),
            str(getattr(ns, "against_words", "") or "").strip() or None,
            family=fam,
            hold_words=str(getattr(ns, "hold_words", "") or "").strip() or None,
            alpha=float(getattr(ns, "hold_alpha", 1.0)),
            prompt=str(getattr(ns, "prompt", "") or "").strip() or None,
        )
        ro = synth["readout"]
        out.append(
            {
                "family": fam,
                "pos_words_matched": synth["pos_words_matched"],
                "pos_words_unmatched": synth.get("pos_words_unmatched", []),
                "neg_words_matched": synth["neg_words_matched"],
                "hold_words": synth["hold_words"],
                "rec_strength": _recommended_strength_text(float(ro.get("norm_after_holds", ro.get("norm_before_holds", 0.0)))),
                "readout": ro,
            }
        )
    return {"ok": True, "families": out}


def _r_preview_recipe(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """What `sculpt` would compose, read out and not saved."""
    if not j_lens.bundle_available():
        raise RouteError("J-Lens bundle models/j_lens/j_lens_h3.safetensors not found")
    ns = parse_request("sculpt", _request_args(req))
    steps = _sculpt_steps_from_args(ns)
    fams = _resolve_requested_families(getattr(ns, "families", "both"))
    out: list[dict[str, Any]] = []
    for fam in fams:
        resolved = [{**st, "vector": _resolve_vector(state, st["vector"], fam)} for st in steps]
        rep = j_lens.decode_recipe(
            resolved,
            family=fam,
            hold_words=str(getattr(ns, "hold_words", "") or "").strip() or None,
            alpha=float(getattr(ns, "hold_alpha", 1.0)),
            prompt=str(getattr(ns, "prompt", "") or "").strip() or None,
        )
        out.append({"family": fam, "readout": rep.get("readout") or rep})
    return {"ok": True, "steps": [{**st, "vector": str(st["vector"])} for st in steps], "families": out}


def _r_j_lens(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """The J-Lens word readout of a saved vector."""
    if not j_lens.bundle_available():
        raise RouteError("J-Lens bundle models/j_lens/j_lens_h3.safetensors not found")
    vec_name = str(req.get("vector", "") or "").strip()
    if not vec_name:
        raise RouteError("missing 'vector'")
    holds = [_resolve_vector(state, str(h)) for h in (req.get("holds") or []) if str(h).strip()]
    rep = j_lens.decode_safetensors(
        _resolve_vector(state, vec_name),
        holds=holds or None,
        hold_words=str(req.get("hold_words", "") or "").strip() or None,
        alpha=float(req.get("hold_alpha", 1.0)),
        prompt=str(req.get("prompt", "") or "").strip() or None,
    )
    ro = rep.get("post_refiner") or rep.get("readout") or {}
    return {"ok": True, **rep, "readout": ro}


def _r_prompt_meter(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """Per-token projection of a prompt's refined rows on a vector, over the rows `mask` selects."""
    vec_arg = str(req.get("vector", "") or "").strip()
    prompt_arg = str(req.get("prompt", "") or "").strip()
    if not vec_arg or not prompt_arg:
        raise RouteError("both 'vector' and 'prompt' are required")
    mask = str(req.get("mask", "") or "").strip() or str(h3_rules.CONTROL_VECTOR_DEFAULTS["mask"])
    if not h3_spans.is_inference_mask(mask):
        raise RouteError(
            f"mask must be one of {tuple(cv_nodes.MASK_MODES)} or prompt fields joined by '+' "
            f"({tuple(h3_spans.PROMPT_FIELDS)}), got {mask!r}"
        )
    with _RUN_LOCK:
        return _meter_prompt_tokens(
            state,
            vector_arg=vec_arg,
            prompt_arg=prompt_arg,
            pooling=mask,
            family=str(req.get("family", "") or "").strip() or None,
        )


def _r_generate_pairs(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    """`generate-pairs`' LLM call, its Markdown returned for the page's editor."""
    values = dict(_request_args(req))
    values.setdefault("llm_provider", state.default_llm_provider)
    if not str(values.get("llm_model", "") or "").strip() and state.default_llm_model:
        values["llm_model"] = state.default_llm_model
    ns = parse_request("generate-pairs", values)
    with _RUN_LOCK:
        state.release_clip_vram()
        return _generate_pairs_from_args(ns, llm_cache=state._llm_cache)


def _r_quit(state: ControlVectorUIState, req: dict[str, Any]) -> dict[str, Any]:
    state.shutdown_requested = True
    return {"ok": True}


Route = Callable[[ControlVectorUIState, dict[str, Any]], dict[str, Any]]

#: Every API route the server answers. A route not here does not exist.
ROUTES: dict[tuple[str, str], Route] = {
    ("GET", "/api/init"): _r_init,
    ("GET", "/api/schema"): _r_schema,
    ("POST", "/api/run"): _r_run,
    ("POST", "/api/compile"): _r_compile,
    ("POST", "/api/preview"): _r_preview,
    ("POST", "/api/load-toml"): _r_load_toml,
    ("POST", "/api/save-toml"): _r_save_toml,
    ("POST", "/api/preview-words"): _r_preview_words,
    ("POST", "/api/preview-recipe"): _r_preview_recipe,
    ("POST", "/api/j-lens"): _r_j_lens,
    ("POST", "/api/prompt-meter"): _r_prompt_meter,
    ("POST", "/api/generate-pairs-llm"): _r_generate_pairs,
    ("POST", "/api/quit"): _r_quit,
}


def _static(clean_path: str) -> tuple[int, str, str] | None:
    """Serve `index.html` and the files under `web/` at `/ui/<path>`."""
    if clean_path == "/favicon.ico":
        return 204, "image/x-icon", ""
    if clean_path in ("/", "/index.html"):
        rel = "index.html"
    elif clean_path.startswith("/ui/"):
        rel = clean_path[len("/ui/"):]
    else:
        return None
    target = (WEB_DIR / rel).resolve()
    if WEB_DIR.resolve() not in target.parents or target.suffix not in _STATIC_TYPES or not target.is_file():
        return 404, "application/json", json.dumps({"ok": False, "error": f"not found: {clean_path}"})
    return 200, _STATIC_TYPES[target.suffix], target.read_text(encoding="utf-8")


def handle_ui_request(
    state: ControlVectorUIState,
    method: str,
    path: str,
    body: bytes = b"",
) -> tuple[int, str, str]:
    clean_path = path.split("?", 1)[0]
    if method == "GET":
        static = _static(clean_path)
        if static is not None:
            return static

    route = ROUTES.get((method, clean_path))
    if route is None:
        return 404, "application/json", json.dumps({"ok": False, "error": f"not found: {method} {clean_path}"})

    req: dict[str, Any] = {}
    if body:
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception as exc:
            return 400, "application/json", json.dumps({"ok": False, "error": f"invalid JSON: {exc}"})
        if not isinstance(req, dict):
            return 400, "application/json", json.dumps({"ok": False, "error": "the request body must be a JSON object"})
    try:
        res = route(state, req)
    except SystemExit as exc:
        return 400, "application/json", json.dumps({"ok": False, "error": f"the method exited ({exc.code})"})
    except Exception as exc:
        return 400, "application/json", json.dumps({"ok": False, "error": str(exc) or type(exc).__name__})
    code = 200 if res.get("ok", True) else 400
    return code, "application/json", json.dumps(res, default=str)


def cmd_ui(args: argparse.Namespace) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import webbrowser

    toml_dir_override = Path(args.toml_dir) if getattr(args, "toml_dir", None) else None
    vec_dir_override = Path(args.vec_dir) if getattr(args, "vec_dir", None) else None
    state = ControlVectorUIState(
        toml_dir=toml_dir_override,
        vec_dir=vec_dir_override,
        default_encoder=getattr(args, "encoder", None),
        default_dit=getattr(args, "dit", None),
        default_device=getattr(args, "device", None),
        default_llm_provider=getattr(args, "llm_provider", None),
        default_llm_model=getattr(args, "llm_model", None),
    )
    if getattr(args, "token", None):
        state.session_token = str(args.token)
    srv_ref: list[ThreadingHTTPServer] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *log_args: Any) -> None:
            return

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            verdict, detail, cookie = gate(state, method, self.path, self.headers, port=srv_ref[0].server_address[1])
            extra: list[tuple[str, str]] = []
            if verdict == "enter":
                code, ctype, payload = 303, "text/plain; charset=utf-8", ""
                extra = [("Location", detail), ("Set-Cookie", cookie)]
            elif verdict == "refuse":
                code, ctype, payload = 403, "application/json", json.dumps({"ok": False, "error": detail})
            else:
                code, ctype, payload = handle_ui_request(state, method, self.path, body)
            raw_bytes = payload.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw_bytes)))
            self.send_header("Cache-Control", "no-store")
            for key, value in extra:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(raw_bytes)
            if state.shutdown_requested and srv_ref:
                threading.Thread(target=srv_ref[0].shutdown, daemon=True).start()

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    server = ThreadingHTTPServer((str(args.host), int(args.port)), _Handler)
    srv_ref.append(server)
    host, actual_port = server.server_address[:2]
    url = f"http://{host}:{actual_port}/?token={state.session_token}"
    print("=== MiniMax H3 Control Vector Studio ===")
    print(f"  url   : {url}")
    print("  note  : Open that link once per browser; the token in it is this session's key and nothing")
    print("          is served without it. Behind an HTTPS front, open <front's address>/?token=<the same>.")
    print("  note  : Click 'Quit server' on the page or press Ctrl+C here to exit.")
    if not getattr(args, "no_browser", False):
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
