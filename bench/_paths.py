"""Where ComfyUI's media and this box's captures live, resolved rather than typed.

These directories are per-machine. Several scripts here carried them as string
literals, which is wrong twice over: the path names somebody's specific storage
layout, and the script only runs on that one machine. And this box launches
the server with `--output-directory` and `--input-directory` pointing at a
share, which nothing in the HTTP API reports back, so a repo-relative guess
is wrong in exactly the case that matters: a batch built against the local
`output` directory resolves zero clips.

Resolution order for the input and output directories, most authoritative
first:

1. The environment variable, which is the escape hatch for any layout.
2. The launcher's flag, read verbatim from a command line. This process's
   own when it is the server (`/proc/self/cmdline` carries the flag), and
   otherwise the live server's: the owner of `COMFY_PORT` by `ss -ltnp`,
   then `/proc/<pid>/cmdline`, NUL-separated, both the `--flag path` and the
   `--flag=path` spellings. Read only: nothing here connects to the server.
3. ComfyUI's `folder_paths`, only when the server itself is the caller and
   was launched without the flag, because inside any other process it
   answers with the stock directory beside the checkout.
4. Otherwise REFUSE, with a message naming everything that was tried and
   the local directory it would once have fallen back to. A silent local
   answer costs a batch; a refusal costs a shell variable.

The path read from a command line is a runtime input. It is returned to the
caller and never written into a record; the scrubs in the writers refuse it.
Nothing here creates a directory.

    H3_COMFY_INPUT     reference images and other render inputs
    H3_COMFY_OUTPUT    rendered video and stills
    H3_CAPTURE_ROOT    the collection of activation-capture directories

    python bench/_paths.py --controls

Controls, run 2026-09-04 with a live server on the port, all green: the
environment variable wins over a command line carrying a different flag;
the parser reads both spellings from a NUL-joined bytes fixture, returns
None for an absent flag, a flag with no value, and an empty command line;
with the variable unset and no port owner the resolver refuses and its
message names the variable, the port and the local directory; with a port
owner whose command line lacks the flag it refuses and says so; and the
live server's command line, when there is one, parses to a directory that
exists.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# inherited: start.sh passes `--port 8188` and docs/comfy_notes.md finds the
# server by that port's owner. Not measured, not reasoned.
COMFY_PORT = 8188


def _comfy_root():
    """The ComfyUI checkout, by marker rather than by depth."""
    if len(_REPO.parents) < 2:
        return None
    root = _REPO.parents[1]
    markers = ("comfyui_version.py", "comfy", "main.py")
    return root if any((root / m).exists() for m in markers) else None


# ----------------------------------------------------------- command lines

def flag_value(cmdline: bytes | None, flag: str) -> str | None:
    """The value of `--flag` in a NUL-separated command line: the next
    argument, or the text after `=` in the one-argument spelling. None when
    the flag is absent, is the last argument, or the command line is empty.
    The first occurrence wins, as argparse's would not, but a launcher that
    passes the flag twice is a launcher to read by hand."""
    if not cmdline:
        return None
    args = [a.decode("utf-8", "surrogateescape") for a in cmdline.split(b"\0") if a]
    for i, a in enumerate(args):
        if a == flag:
            return args[i + 1] if i + 1 < len(args) else None
        if a.startswith(flag + "="):
            return a[len(flag) + 1:] or None
    return None


def _read_cmdline(pid: int | str) -> bytes | None:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None


def port_owner_pid(port: int = COMFY_PORT) -> int | None:
    """The pid listening on `port`, from `ss -ltnp`; None when nothing is,
    or `ss` is unavailable. The same observable docs/comfy_notes.md uses."""
    try:
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4 or not fields[3].endswith(f":{port}"):
            continue
        marker = "pid="
        at = line.find(marker)
        if at < 0:
            continue
        digits = ""
        for ch in line[at + len(marker):]:
            if not ch.isdigit():
                break
            digits += ch
        if digits:
            return int(digits)
    return None


def _from_own_cmdline(flag: str) -> Path | None:
    value = flag_value(_read_cmdline("self"), flag)
    return Path(os.path.expanduser(value)) if value else None


def _from_server_cmdline(flag: str, port: int = COMFY_PORT) -> tuple[Path | None, str]:
    """The live server's flag value, and a sentence saying what was found
    for the refusal message."""
    pid = port_owner_pid(port)
    if pid is None:
        return None, f"no server owns port {port}"
    value = flag_value(_read_cmdline(pid), flag)
    if not value:
        return None, f"the server on port {port} (pid {pid}) was launched without {flag}"
    return Path(os.path.expanduser(value)), f"the server on port {port} (pid {pid}) carries {flag}"


def _is_server_process() -> bool:
    """True inside ComfyUI itself: `folder_paths` is loaded and the entry
    point is its `main.py`. A bench script that imported comfy modules has
    `folder_paths` loaded too, and that is exactly the caller `folder_paths`
    must not answer, because there it reports the stock directory."""
    if "folder_paths" not in sys.modules:
        return False
    return Path(sys.argv[0]).name == "main.py" if sys.argv and sys.argv[0] else False


def _from_folder_paths(getter: str):
    try:
        import folder_paths  # only meaningful inside a running ComfyUI
        value = getattr(folder_paths, getter)()
    except Exception:
        return None
    return Path(value) if value else None


def _resolve(env_var: str, getter: str, flag: str, fallback: str, name: str) -> Path:
    raw = os.environ.get(env_var)
    if raw:
        return Path(os.path.expanduser(raw))
    own = _from_own_cmdline(flag)
    if own is not None:
        return own
    if _is_server_process():
        via_comfy = _from_folder_paths(getter)
        if via_comfy is not None:
            return via_comfy
    via_server, found = _from_server_cmdline(flag)
    if via_server is not None:
        return via_server
    root = _comfy_root()
    local = (root / fallback) if root else None
    raise SystemExit(
        f"refuse: could not resolve the ComfyUI {name} directory. {env_var} is unset; {found}; "
        f"the local directory {local if local else '(no ComfyUI checkout beside this repo)'} is "
        f"not the answer on a box whose launcher redirects it, so it is not used. "
        f"Set {env_var}, start the server, or pass the directory explicitly where the tool takes one."
    )


def comfy_input() -> Path:
    """ComfyUI's input directory. Refuses rather than guessing."""
    return _resolve("H3_COMFY_INPUT", "get_input_directory", "--input-directory", "input", "input")


def comfy_output() -> Path:
    """ComfyUI's output directory. Refuses rather than guessing."""
    return _resolve("H3_COMFY_OUTPUT", "get_output_directory", "--output-directory", "output", "output")


def capture_root():
    """The activation-capture collection, or None.

    Captures live outside the repo by design -- they are large and unversioned
    -- so there is no repo-relative fallback worth guessing. `H3_CAPTURE_ROOT`
    or nothing.
    """
    raw = os.environ.get("H3_CAPTURE_ROOT")
    return Path(os.path.expanduser(raw)) if raw else None


# ------------------------------------------------------------------ controls

def _controls() -> int:
    import contextlib
    bad = 0

    def case(label, ok, detail=""):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}{('  ' + detail) if detail else ''}")
        bad += 0 if ok else 1

    fixture_dir = "/srv/share/output"
    # a fixture built at runtime, never a path this box uses
    two_arg = b"\0".join([b"python3", b"main.py", b"--output-directory", fixture_dir.encode(), b"--port", b"8188", b""])
    one_arg = b"\0".join([b"python3", b"main.py", b"--output-directory=" + fixture_dir.encode(), b"--port", b"8188"])
    no_flag = b"\0".join([b"python3", b"main.py", b"--port", b"8188"])
    dangling = b"\0".join([b"python3", b"main.py", b"--output-directory"])

    # 1. the parser
    case("parser: `--flag path`", flag_value(two_arg, "--output-directory") == fixture_dir)
    case("parser: `--flag=path`", flag_value(one_arg, "--output-directory") == fixture_dir)
    case("parser: absent flag is None", flag_value(no_flag, "--output-directory") is None)
    case("parser: dangling flag is None", flag_value(dangling, "--output-directory") is None)
    case("parser: empty command line is None", flag_value(b"", "--output-directory") is None and flag_value(None, "--output-directory") is None)
    case("parser: a different flag is not confused", flag_value(two_arg, "--input-directory") is None)

    g = globals()

    @contextlib.contextmanager
    def patched(**subs):
        saved = {k: g[k] for k in subs}
        env_saved = os.environ.get("H3_COMFY_OUTPUT")
        g.update(subs)
        try:
            yield
        finally:
            g.update(saved)
            if env_saved is None:
                os.environ.pop("H3_COMFY_OUTPUT", None)
            else:
                os.environ["H3_COMFY_OUTPUT"] = env_saved

    fake_server = lambda flag, port=COMFY_PORT: (Path(fixture_dir), "fixture server")  # noqa: E731
    no_own = lambda flag: None  # noqa: E731

    # 2. the environment variable wins over a command line carrying a different flag
    with patched(_from_server_cmdline=fake_server, _from_own_cmdline=no_own):
        os.environ["H3_COMFY_OUTPUT"] = "/srv/elsewhere"
        case("env var wins over the server's flag", comfy_output() == Path("/srv/elsewhere"))
        os.environ.pop("H3_COMFY_OUTPUT", None)
        case("server's flag answers when the env var is unset", comfy_output() == Path(fixture_dir))

    # 3. neither: refuse, naming the variable, the port and the local directory
    with patched(port_owner_pid=lambda port=COMFY_PORT: None, _from_own_cmdline=no_own, _is_server_process=lambda: False):
        os.environ.pop("H3_COMFY_OUTPUT", None)
        try:
            comfy_output(); msg = None
        except SystemExit as e:
            msg = str(e)
        local = _comfy_root()
        names = msg is not None and "H3_COMFY_OUTPUT" in msg and f"port {COMFY_PORT}" in msg and (local is None or str(local / "output") in msg)
        case("no env var, no server: refuses naming all three", bool(names), "" if names else repr(msg))

    # 4. a port owner launched without the flag: refuse and say so
    with patched(port_owner_pid=lambda port=COMFY_PORT: 1, _read_cmdline=lambda pid: no_flag, _from_own_cmdline=no_own, _is_server_process=lambda: False):
        os.environ.pop("H3_COMFY_OUTPUT", None)
        try:
            comfy_output(); msg = None
        except SystemExit as e:
            msg = str(e)
        case("port owner without the flag: refuses and says so", msg is not None and "without --output-directory" in msg, "" if msg and "without" in msg else repr(msg))

    # 5. the live server, if there is one: read-only, its flag parses to a directory that exists
    pid = port_owner_pid()
    if pid is None:
        print(f"  skip  live: no server owns port {COMFY_PORT}")
    else:
        live, found = _from_server_cmdline("--output-directory")
        case("live: the port owner's command line carries the flag", live is not None, found)
        if live is not None:
            case("live: the directory it names exists", live.is_dir(), str(live))

    print("\n" + ("every control held" if not bad else f"{bad} control(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    if "--controls" in sys.argv[1:]:
        sys.exit(_controls())
    print(__doc__)
    sys.exit(2)
