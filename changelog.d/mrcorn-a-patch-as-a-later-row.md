bump: patch

### Added

- **`bench/check_assemble_delivery.py`, a fourteenth case: a patch as a
  later row.** A redone stretch of a render is a file made from that
  render, not from the original, and the table's rule for rows that share
  frames was written for passes made from the original. The case pins that
  it holds for a patch too: given as a later row over its hole's frames,
  the patch shows wherever it differs from the original, which is what the
  render changed there and what the patch redrew, the render shows on the
  other frames, and the shared pixels are flagged as settled by order. So a
  redo is one more row of a delivery's table, with no join first.
