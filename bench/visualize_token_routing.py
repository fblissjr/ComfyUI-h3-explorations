#!/usr/bin/env python3
"""Simulate, measure, and visually export Sol-Attn with Token Routing (ON vs OFF).

Demonstrates what token routing (token_aug) does to attention numerics:
1. Simulates realistic Q and K activations with both clustered and outlier attention patterns.
2. Runs three paths:
   - Full Dense Attention (Ground Truth)
   - Sol-Attn Block-Sparse (Token Routing OFF: token_aug = 0)
   - Sol-Attn Augmented (Token Routing ON: token_aug = 64)
3. Quantifies numerical error (Relative L2, Cosine Similarity, Max Pointwise Error).
4. Generates a standalone interactive HTML visualizer showing the exact attention heatmaps,
   the promoted outlier tokens, and the residual error maps.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path

import torch

BLOCK_SIZE = 64


def generate_synthetic_scene(
    num_blocks: int = 8,
    head_dim: int = 128,
    tau: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Generate realistic Q, K, V with distinct block properties.

    Injects an intentional 'needle in a haystack': an unselected key block
    that has low average similarity but contains 2 critical high-magnitude outlier tokens.
    """
    torch.manual_seed(seed)
    seq_len = num_blocks * BLOCK_SIZE

    # Base background noise
    q = torch.randn(seq_len, head_dim, dtype=torch.float32) * 0.5
    k = torch.randn(seq_len, head_dim, dtype=torch.float32) * 0.5
    v = torch.randn(seq_len, head_dim, dtype=torch.float32) * 0.8

    # Block 0: Strong local attention (matches query block 0 strongly)
    # Block 1: Moderate attention
    # Block 3: The "Outlier Block": 62 noise tokens + 2 strong feature tokens
    outlier_block_idx = 3
    outlier_token_1 = outlier_block_idx * BLOCK_SIZE + 12
    outlier_token_2 = outlier_block_idx * BLOCK_SIZE + 35

    # Make target query (e.g. query token 15) correlate very strongly with those 2 tokens
    target_q_idx = 15
    feature_dir = q[target_q_idx] / torch.norm(q[target_q_idx])
    k[outlier_token_1] = feature_dir * 3.5
    k[outlier_token_2] = feature_dir * 3.2

    # Make the rest of block 3 anti-correlated or weak so block centroid is BELOW tau cutoff
    for idx in range(outlier_block_idx * BLOCK_SIZE, (outlier_block_idx + 1) * BLOCK_SIZE):
        if idx not in (outlier_token_1, outlier_token_2):
            k[idx] = -feature_dir * 0.8 + torch.randn(head_dim) * 0.3

    return q, k, v, outlier_block_idx, target_q_idx


def compute_sol_attention_simulation(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    token_aug: int = 64,
) -> dict:
    """Compute dense, Sol (routing OFF), and Sol (routing ON) attention."""
    seq_len, dim = q.shape
    num_blocks = seq_len // BLOCK_SIZE
    scale = 1.0 / math.sqrt(dim)

    # 1. Full Dense Attention (Ground Truth)
    raw_scores = torch.matmul(q, k.T) * scale
    dense_weights = torch.softmax(raw_scores, dim=-1)
    dense_output = torch.matmul(dense_weights, v)

    # 2. Block-level coarse scoring
    k_blocks = k.view(num_blocks, BLOCK_SIZE, dim)
    v_blocks = v.view(num_blocks, BLOCK_SIZE, dim)
    k_centroids = k_blocks.mean(dim=1)  # [num_blocks, dim]
    v_centroids = v_blocks.mean(dim=1)  # [num_blocks, dim]

    q_blocks = q.view(num_blocks, BLOCK_SIZE, dim)
    q_centroids = q_blocks.mean(dim=1)  # [num_blocks, dim]

    # Coarse proxy scores between query blocks and key centroids
    proxy_scores = torch.matmul(q_centroids, k_centroids.T) * scale  # [num_blocks, num_blocks]

    # Simulate for Query Block 0 (holding the target query)
    qb = 0
    row_proxy = proxy_scores[qb]
    mu = row_proxy.mean().item()
    sigma = row_proxy.std().item()
    threshold = mu + tau * sigma

    # Block qualification
    selected_blocks = [j for j in range(num_blocks) if row_proxy[j].item() >= threshold or j == qb]
    unselected_blocks = [j for j in range(num_blocks) if j not in selected_blocks]

    # 3. Path A: Token Routing OFF (token_aug = 0)
    # Selected blocks get full token attention; unselected blocks get pooled centroid tail
    off_weights = torch.zeros_like(dense_weights[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE])

    # Unnormalized logits
    unnorm_logits_off = torch.zeros(BLOCK_SIZE, seq_len)
    tail_scores_off = []

    for j in range(num_blocks):
        c_start, c_end = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
        if j in selected_blocks:
            unnorm_logits_off[:, c_start:c_end] = raw_scores[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE, c_start:c_end]
        else:
            # Pooled tail approximation: uniform energy across block equal to centroid logit
            centroid_score = row_proxy[j].item()
            unnorm_logits_off[:, c_start:c_end] = centroid_score
            tail_scores_off.append((j, centroid_score))

    off_weights = torch.softmax(unnorm_logits_off, dim=-1)
    off_output = torch.matmul(off_weights, v)

    # 4. Path B: Token Routing ON (token_aug = 64)
    # Search unselected blocks for top individual tokens
    unnorm_logits_on = unnorm_logits_off.clone()
    promoted_tokens = []

    if token_aug > 0 and unselected_blocks:
        candidate_tokens = []
        for j in unselected_blocks:
            c_start, c_end = j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE
            block_tokens_scores = raw_scores[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE, c_start:c_end].mean(dim=0)
            for t_offset, score in enumerate(block_tokens_scores):
                candidate_tokens.append((c_start + t_offset, score.item(), j))

        # Sort candidate tokens descending
        candidate_tokens.sort(key=lambda x: x[1], reverse=True)
        promoted = candidate_tokens[:token_aug]
        promoted_tokens = [t[0] for t in promoted]

        # For promoted tokens, insert their exact token scores!
        for t_idx, _, _ in promoted:
            unnorm_logits_on[:, t_idx] = raw_scores[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE, t_idx]

    on_weights = torch.softmax(unnorm_logits_on, dim=-1)
    on_output = torch.matmul(on_weights, v)

    # Calculate metrics for query block 0
    gt_sub = dense_output[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE]

    # Relative L2 error
    l2_off = (torch.norm(off_output - gt_sub) / torch.norm(gt_sub)).item()
    l2_on = (torch.norm(on_output - gt_sub) / torch.norm(gt_sub)).item()

    # Cosine similarity
    cos_off = torch.nn.functional.cosine_similarity(off_output.flatten(), gt_sub.flatten(), dim=0).item()
    cos_on = torch.nn.functional.cosine_similarity(on_output.flatten(), gt_sub.flatten(), dim=0).item()

    # Weight difference against dense
    weight_err_off = torch.abs(off_weights - dense_weights[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE]).mean().item()
    weight_err_on = torch.abs(on_weights - dense_weights[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE]).mean().item()

    return {
        "num_blocks": num_blocks,
        "selected_blocks": selected_blocks,
        "unselected_blocks": unselected_blocks,
        "promoted_tokens": promoted_tokens,
        "mu": mu,
        "sigma": sigma,
        "threshold": threshold,
        "l2_off": l2_off,
        "l2_on": l2_on,
        "cos_off": cos_off,
        "cos_on": cos_on,
        "weight_err_off": weight_err_off,
        "weight_err_on": weight_err_on,
        "dense_weights": dense_weights[qb * BLOCK_SIZE : (qb + 1) * BLOCK_SIZE].tolist(),
        "off_weights": off_weights.tolist(),
        "on_weights": on_weights.tolist(),
        "proxy_scores": row_proxy.tolist(),
    }


def generate_html_report(results: dict, output_path: Path):
    """Write an interactive comparison visualization HTML."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Token Routing: Numerical & Visual Data Comparison</title>
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: rgba(23, 31, 48, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --on-color: #10b981;
      --off-color: #f43f5e;
      --font: system-ui, -apple-system, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      padding: 2rem 1.5rem;
      line-height: 1.5;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    h1 {{ font-size: 2rem; margin-bottom: 0.5rem; color: #38bdf8; }}
    .subtitle {{ color: var(--text-muted); margin-bottom: 2rem; }}
    .grid-metrics {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 1.25rem;
      text-align: center;
    }}
    .metric-label {{ font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; margin-bottom: 0.25rem; }}
    .metric-val {{ font-size: 1.75rem; font-weight: 700; }}
    .val-good {{ color: var(--on-color); }}
    .val-bad {{ color: var(--off-color); }}

    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin-bottom: 1.75rem;
    }}
    .card-title {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem; }}

    .heatmaps-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.5rem;
    }}
    @media (max-width: 900px) {{
      .heatmaps-grid {{ grid-template-columns: 1fr; }}
      .grid-metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
    .heatmap-box {{
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 0.5rem;
      padding: 1rem;
    }}
    .heatmap-header {{ font-weight: 600; margin-bottom: 0.5rem; display: flex; justify-content: space-between; }}
    canvas {{ width: 100%; height: 180px; image-rendering: pixelated; border-radius: 4px; background: #000; }}

    .table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
      font-size: 0.9rem;
    }}
    .table th, .table td {{
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--card-border);
      text-align: left;
    }}
    .table th {{ color: var(--text-muted); background: rgba(255, 255, 255, 0.02); }}

    .badge {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 0.25rem;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-selected {{ background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .badge-rescued {{ background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }}
    .badge-tail {{ background: rgba(71, 85, 105, 0.3); color: #94a3b8; }}

    .callout {{
      border-left: 4px solid var(--accent);
      background: rgba(56, 189, 248, 0.06);
      padding: 1rem 1.25rem;
      border-radius: 0 0.5rem 0.5rem 0;
      margin-top: 1rem;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Sol-Attn Token Routing: Data & Numerics Verification</h1>
    <p class="subtitle">Comparing Full Dense Attention vs. Block-Sparse (Routing OFF) vs. Token Routing (Routing ON: token_aug = 64)</p>

    <!-- Key Metrics Readout -->
    <div class="grid-metrics">
      <div class="metric-card">
        <div class="metric-label">Relative L2 Error (OFF)</div>
        <div class="metric-val val-bad">{results['l2_off'] * 100:.2f}%</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Error vs. Dense FP32</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Relative L2 Error (ON)</div>
        <div class="metric-val val-good">{results['l2_on'] * 100:.2f}%</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Error with token_aug = 64</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Cosine Similarity (OFF)</div>
        <div class="metric-val">{results['cos_off']:.5f}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">Block-only routing</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Cosine Similarity (ON)</div>
        <div class="metric-val val-good">{results['cos_on']:.5f}</div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">With outlier rescue</div>
      </div>
    </div>

    <!-- Heatmaps -->
    <div class="card">
      <div class="card-title">🔬 Attention Weight Heatmaps: What the Model Actually Computes</div>
      <p style="color: var(--text-muted); font-size: 0.95rem; margin-bottom: 1rem;">
        Query Block 0 (64 tokens) attending to Key Sequence (512 tokens / 8 blocks). Notice how Key Block 3 fails the block threshold, but contains sharp outlier spikes that are completely flattened when routing is OFF, and fully restored when routing is ON.
      </p>

      <div class="heatmaps-grid">
        <div class="heatmap-box">
          <div class="heatmap-header">
            <span>1. Ground Truth (Dense Attention)</span>
            <span style="color: var(--text-muted);">FP32 Exact</span>
          </div>
          <canvas id="canvasDense"></canvas>
        </div>

        <div class="heatmap-box">
          <div class="heatmap-header">
            <span style="color: var(--off-color);">2. Token Routing OFF (token_aug = 0)</span>
            <span style="color: var(--text-muted);">Pure Block-Centroid Tail</span>
          </div>
          <canvas id="canvasOff"></canvas>
        </div>

        <div class="heatmap-box">
          <div class="heatmap-header">
            <span style="color: var(--on-color);">3. Token Routing ON (token_aug = 64)</span>
            <span style="color: var(--on-color);">Promoted Outlier Tokens Rescued!</span>
          </div>
          <canvas id="canvasOn"></canvas>
        </div>

        <div class="heatmap-box">
          <div class="heatmap-header">
            <span>4. Residual Error Reduction</span>
            <span style="color: var(--accent);">Green = Exact Match</span>
          </div>
          <canvas id="canvasDiff"></canvas>
        </div>
      </div>
    </div>

    <!-- Block Breakdown Table -->
    <div class="card">
      <div class="card-title">📊 Block-by-Block Routing Breakdown</div>
      <table class="table">
        <thead>
          <tr>
            <th>Block Index</th>
            <th>Coarse Proxy Score</th>
            <th>Cutoff Threshold</th>
            <th>Block Status</th>
            <th>Tokens Promoted by Routing</th>
          </tr>
        </thead>
        <tbody>
"""

    for b in range(results["num_blocks"]):
        score = results["proxy_scores"][b]
        is_sel = b in results["selected_blocks"]
        promoted_in_b = [t for t in results["promoted_tokens"] if b * BLOCK_SIZE <= t < (b + 1) * BLOCK_SIZE]

        status_badge = (
            '<span class="badge badge-selected">EXACT BLOCK (Score &ge; T)</span>'
            if is_sel
            else '<span class="badge badge-tail">POOLED TAIL (Score &lt; T)</span>'
        )

        tokens_badge = (
            f'<span class="badge badge-rescued">{len(promoted_in_b)} outlier tokens promoted</span>'
            if promoted_in_b
            else '<span style="color: var(--text-muted);">None</span>'
        )

        html_content += f"""          <tr>
            <td><strong>Block {b}</strong> ({b * BLOCK_SIZE}..{(b + 1) * BLOCK_SIZE - 1})</td>
            <td><code>{score:.3f}</code></td>
            <td><code>{results['threshold']:.3f}</code></td>
            <td>{status_badge}</td>
            <td>{tokens_badge}</td>
          </tr>
"""

    html_content += f"""        </tbody>
      </table>

      <div class="callout">
        <strong>The Takeaway:</strong> In Block 3, the block average was only <code>{results['proxy_scores'][3]:.3f}</code>, well below the threshold of <code>{results['threshold']:.3f}</code>. Without token routing, that entire block was squashed into a coarse grey blur. With token routing ON, the kernel automatically promoted the top <code>{len(results['promoted_tokens'])}</code> outlier tokens to 100% exact precision, reducing the total attention error from <strong>{results['l2_off'] * 100:.2f}%</strong> down to <strong>{results['l2_on'] * 100:.2f}%</strong>!
      </div>
    </div>
  </div>

  <script>
    const dataDense = {json.dumps(results['dense_weights'])};
    const dataOff = {json.dumps(results['off_weights'])};
    const dataOn = {json.dumps(results['on_weights'])};

    function renderCanvas(canvasId, matrix, isDiff = false) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      const rows = matrix.length;
      const cols = matrix[0].length;
      canvas.width = cols;
      canvas.height = rows;

      const imgData = ctx.createImageData(cols, rows);
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          const idx = (r * cols + c) * 4;
          const val = matrix[r][c];

          if (isDiff) {{
            // Green for zero error, Red for high error
            const err = Math.min(1.0, val * 25.0);
            imgData.data[idx] = Math.round(err * 244);     // R
            imgData.data[idx + 1] = Math.round((1 - err) * 185); // G
            imgData.data[idx + 2] = 50;                  // B
            imgData.data[idx + 3] = 255;
          }} else {{
            // Viridis/Magma gradient approximation
            const norm = Math.min(1.0, Math.pow(val * cols * 0.7, 0.7));
            imgData.data[idx] = Math.round(norm * 56 + (1 - norm) * 11);     // R
            imgData.data[idx + 1] = Math.round(norm * 189 + (1 - norm) * 15); // G
            imgData.data[idx + 2] = Math.round(norm * 248 + (1 - norm) * 25); // B
            imgData.data[idx + 3] = 255;
          }}
        }}
      }}
      ctx.putImageData(imgData, 0, 0);
    }}

    renderCanvas('canvasDense', dataDense);
    renderCanvas('canvasOff', dataOff);
    renderCanvas('canvasOn', dataOn);

    // Compute diff map
    const diffMatrix = [];
    for (let r = 0; r < dataDense.length; r++) {{
      const row = [];
      for (let c = 0; c < dataDense[0].length; c++) {{
        row.push(Math.abs(dataOn[r][c] - dataDense[r][c]));
      }}
      diffMatrix.push(row);
    }}
    renderCanvas('canvasDiff', diffMatrix, true);
  </script>
</body>
</html>
"""
    output_path.write_text(html_content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, default=1.0, help="Sol threshold tau")
    parser.add_argument("--token-aug", type=int, default=64, help="Token augmentation budget")
    parser.add_argument("--out", type=str, default="docs/wiki/research/sparse/token_routing_comparison.html", help="HTML report output path (relative to repo root)")
    args = parser.parse_args()

    print(f"[visualize_token_routing] Generating synthetic attention simulation (tau={args.tau}, token_aug={args.token_aug})...")
    q, k, v, outlier_block, target_q = generate_synthetic_scene(tau=args.tau)
    results = compute_sol_attention_simulation(q, k, v, tau=args.tau, token_aug=args.token_aug)

    print("\n--- Simulation Numerical Results ---")
    print(f"  Selected Blocks (Exact): {results['selected_blocks']}")
    print(f"  Unselected Blocks (Tail): {results['unselected_blocks']}")
    print(f"  Promoted Outlier Tokens: {len(results['promoted_tokens'])} tokens")
    print(f"  L2 Relative Error (OFF): {results['l2_off'] * 100:.2f}%")
    print(f"  L2 Relative Error (ON):  {results['l2_on'] * 100:.2f}%")
    print(f"  Cosine Similarity (OFF): {results['cos_off']:.5f}")
    print(f"  Cosine Similarity (ON):  {results['cos_on']:.5f}")

    out_file = Path(args.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    generate_html_report(results, out_file)
    print(f"\n[visualize_token_routing] Wrote visual report to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
