# vendor_config/: the release's own config files, verbatim

Loaded when a session opens a file here; the root `AGENTS.md` still governs.

- **Byte for byte as MiniMax released them**, with their digests in `sha256.json`. Never edit, reformat or regenerate
  them; `vendor_config.py` (at the root) is what reads them. A value of ours goes in `workflows/h3_config.py` with
  its provenance, never here.
