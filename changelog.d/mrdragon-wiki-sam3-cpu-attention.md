bump: patch

### Changed

- **`docs/wiki/sam3_prompting.md` names a fourth difference in how ComfyUI
  treats a phrase**: where ComfyUI does not use PyTorch's attention (a
  `--cpu` process by default, the split and sub-quadratic options), the SAM 3
  detector's text padding mask is added to the attention scores and hides
  nothing, so detections for longer phrases differ from the card's. A CPU
  probe of SAM 3 detection needs `--use-pytorch-cross-attention`. The open
  question it replaces (why the CPU and the card disagreed) is removed from
  "What is not known". Localised by the session reviewing the upstream
  activation fix; not reported upstream.
