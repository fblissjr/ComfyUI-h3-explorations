# Rejected Gemini AWQ v2 preflight

**Status:** Historical NO-GO evidence. Do not execute the archived scripts or
cite their generated `PASSED` claims.

The authoritative review is
[`../../../canonical/2026-08-24_awq_v2_preflight_review.md`](../../../canonical/2026-08-24_awq_v2_preflight_review.md).
Source scripts have the suffix `.rejected` to prevent them from resembling an
active launcher. Small reports and the invalid row trace are retained for
forensic review. The bulky generated manifests were deleted after their hashes
were frozen:

Machine-local home paths were replaced with `<HOME>/` in the archived copies.
The hashes in the authoritative review identify the original submitted bytes,
not these privacy-redacted display copies.

| Removed artifact | SHA-256 |
|---|---|
| `native_h3_cal_manifest.jsonl` | `86c06b706bf8ebd5e33a006ba11a42fc210f3d74b4ab1809ccbd1de6f6a0fcc9` |
| `native_h3_eval_manifest.jsonl` | `72eb11eaf99a74acae244e3a5ebe2732bcefc813f811008d31a873f36b884dcf` |
| `source_inventory.jsonl` | `f4dcde8179e05796a66e7b548eabad78042baf094edb3d8b8f94d09510f85efc` |

The corrected candidate pool is independently built from the pinned H3-IR
snapshot and no longer imports this rejected inventory.
