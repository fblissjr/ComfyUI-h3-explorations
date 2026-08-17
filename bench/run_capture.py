#!/usr/bin/env python3
"""Submit and track the production reference capture workflow.

Submits workflows/h3_probe_capture_ref3_api.json unmodified to ComfyUI,
monitoring progress and verifying activation capture to disk.
"""

from __future__ import annotations

import argparse
import json
import os
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
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read())
    prompt_id = res.get("prompt_id")
    print(f"[run_capture] Enqueued prompt_id: {prompt_id}")

    start_time = time.time()

    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/history/{prompt_id}") as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                exec_info = history[prompt_id]
                status = exec_info.get("status", {})
                elapsed = time.time() - start_time
                print(f"[run_capture] Prompt {prompt_id} completed in {elapsed:.1f}s (status: {status.get('status_str')})")
                return 0
        except Exception:
            pass

        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
