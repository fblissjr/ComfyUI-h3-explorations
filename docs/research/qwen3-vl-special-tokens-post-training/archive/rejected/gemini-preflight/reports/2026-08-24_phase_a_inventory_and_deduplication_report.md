# Phase A: Deterministic Source Inventory & Deduplication Report

> **NOT AN ACCEPTED CALIBRATION MANIFEST:** Independent review preserved the
> narrow prompt/media-hash disjointness finding but rejected the media
> semantics, decode claims, selection description, and launch readiness. See
> [`../../canonical/2026-08-24_awq_v2_preflight_review.md`](../../canonical/2026-08-24_awq_v2_preflight_review.md).

**Date:** 2026-08-24
**Random Seed:** 42

## 1. Corpus Overview

| Corpus | Total Rows Scanned | Valid Decodable Rows | Unique Primaries |
| :--- | :---: | :---: | :---: |
| **`StellarVoyager/H3-IR`** | 1,110 | 1110 | 1110 |
| **`malcolmrey_various`** | 1,560 | 1548 | 1548 |
| **TOTAL** | **2662** | **2658** | **2658** |

## 2. Deduplication Findings

- **Unique Primary Records:** 2658
- **Exact Duplicates (Same Prompt + Media):** 4
- **Prompt-Only Duplicates (Same Prompt, Different Media):** 2
- **Media-Only Duplicates (Same Media, Different Prompt):** 259

## 3. Disjoint Partitioning Status

| Manifest | Total Rows | H3-IR Share | Malcolmrey Share | Disjoint Proof |
| :--- | :---: | :---: | :---: | :---: |
| **`native_h3_cal_manifest.jsonl`** | 256 | 150 | 106 | 100% Unique |
| **`native_h3_eval_manifest.jsonl`** | 2291 | 850 | 1441 | **0 Overlap (Prompt & Media)** |

## 4. Special Token Coverage in Calibration Manifest

| Special Token | Total Calibration Occurrences | Prompts Containing Token |
| :--- | :---: | :---: |
| `<d>` | 301 | 180 (70.3%) |
| `</d>` | 301 | 180 (70.3%) |
| `<|cutoff|>` | 0 | 0 (0.0%) |
| `<|lyrics_start|>` | 0 | 0 (0.0%) |
| `<|lyrics_end|>` | 0 | 0 (0.0%) |
| `<|caption_start|>` | 0 | 0 (0.0%) |
| `<|caption_end|>` | 0 | 0 (0.0%) |
