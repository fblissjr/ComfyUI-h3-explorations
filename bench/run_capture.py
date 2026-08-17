#!/usr/bin/env python3
"""Submit and track the production reference capture workflow.

Submits workflows/h3_probe_capture_ref3_api.json unmodified to ComfyUI,
monitoring progress and verifying activation capture to disk.

This is the harness that produced the captures every Sol error number rests on,
so its failure modes are the ones that matter most. Two were live until
2026-08-17, and both made it report success without one:

- The poll loop was `while True` with a bare `except Exception: pass`, no timeout
  and no attempt ceiling. An unreachable server, a typo in `--host`, or a killed
  ComfyUI produced a process that sat there forever printing nothing, which reads
  as "still rendering".
- It returned 0 as soon as `prompt_id` appeared in `/history`, reading
  `status_str` only to print it. A render that errored appeared in history like
  any other, so **a failed capture exited 0**. Downstream that is worse than a
  crash: `analyze_sol_error.py` will happily decompose whatever tensors did get
  written, and nothing says they are a partial run.

Both are fixed below. The exit codes are now 0 success, 1 submit or setup
failure, 2 the render reported a non-success status, 3 timed out waiting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", default="workflows/h3_probe_capture_ref3_api.json")
    parser.add_argument("--host", default="127.0.0.1:8188")
    parser.add_argument("--output-prefix", default="Video/capture_ref3_362f")
    # A 362-frame reference render is tens of minutes, so the default ceiling is
    # generous. It exists so an unattended run cannot wait forever, not to bound
    # a render.
    parser.add_argument("--timeout", type=float, default=7200.0,
                        help="seconds to wait for the render before exiting 3")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--max-poll-errors", type=int, default=12,
                        help="consecutive /history failures before exiting 1")
    args = parser.parse_args()

    wf_path = Path(args.workflow)
    if not wf_path.is_file():
        print(f"Error: workflow file not found: {wf_path}", file=sys.stderr)
        return 1

    wf = json.loads(wf_path.read_text())

    # Update output filename prefix if requested
    if args.output_prefix:
        for node in wf.values():
            if node.get("class_type") == "SaveVideo":
                node.get("inputs", {})["filename_prefix"] = args.output_prefix

    base_url = f"http://{args.host}"
    client_id = str(uuid.uuid4())

    payload = json.dumps({"prompt": wf, "client_id": client_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(f"[run_capture] Submitting workflow: {wf_path}")
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read())
    except Exception as exc:
        print(f"[run_capture] Could not submit to {base_url}: {exc!r}",
              file=sys.stderr)
        return 1
    prompt_id = res.get("prompt_id")
    if not prompt_id:
        # A 200 with no prompt_id means the server rejected the graph. Continuing
        # would poll /history/None forever and then time out with a misleading
        # message about the render.
        print(f"[run_capture] Server returned no prompt_id; response was {res!r}",
              file=sys.stderr)
        return 1
    print(f"[run_capture] Enqueued prompt_id: {prompt_id}")

    start_time = time.time()
    deadline = start_time + args.timeout
    consecutive_errors = 0

    while True:
        if time.time() > deadline:
            print(f"[run_capture] TIMEOUT after {args.timeout:.0f}s waiting for "
                  f"{prompt_id}. The render may still be running -- this says "
                  f"only that this process stopped waiting.", file=sys.stderr)
            return 3

        try:
            with urllib.request.urlopen(f"{base_url}/history/{prompt_id}") as resp:
                history = json.loads(resp.read())
            consecutive_errors = 0
        except Exception as exc:
            # Transient poll failures are expected while a long render holds the
            # event loop. A run of them is not, and swallowing every one is how
            # an unreachable server used to look identical to a slow one.
            consecutive_errors += 1
            if consecutive_errors >= args.max_poll_errors:
                print(f"[run_capture] {consecutive_errors} consecutive poll "
                      f"failures against {base_url}; last was {exc!r}. Giving up "
                      f"rather than waiting on a server that may not be there.",
                      file=sys.stderr)
                return 1
            time.sleep(args.poll_interval)
            continue

        if prompt_id in history:
            exec_info = history[prompt_id]
            status = exec_info.get("status", {})
            status_str = status.get("status_str")
            elapsed = time.time() - start_time

            # Appearing in /history means finished, NOT succeeded. A render that
            # errored is in there too, which is why this branches on the status
            # rather than printing it and returning 0 regardless.
            if status_str != "success":
                print(f"[run_capture] Prompt {prompt_id} FAILED after "
                      f"{elapsed:.1f}s (status: {status_str!r}).", file=sys.stderr)
                for msg in status.get("messages", []):
                    print(f"  {msg}", file=sys.stderr)
                print("  Any tensors already written are from a partial run and "
                      "must not be analysed as a capture.", file=sys.stderr)
                return 2

            print(f"[run_capture] Prompt {prompt_id} completed in {elapsed:.1f}s "
                  f"(status: {status_str})")
            return 0

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    sys.exit(main())
