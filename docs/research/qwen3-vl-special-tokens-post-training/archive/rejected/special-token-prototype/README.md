# Retired special-token prototype

**Status:** Historical source only. Not a training recipe.

The archived generator and trainer are retained with a `.rejected` suffix. The
generated `h3_contrastive_pairs_1k.jsonl` was deleted after recording:

- SHA-256: `cb0f429b1697d56f33e32b75d69f8b5315471f77d1aa3e530cf2909c8bffd78a`;
- 1,000 rows total;
- 340 rows with no contrast; and
- 119 distinct `(positive, contrast)` pairs among the remaining rows.

The trainer was a stub and the generator changed requested semantics between
arms. The accepted replacement corpus contract is
[`../../../canonical/owner_authored_marker_corpus.md`](../../../canonical/owner_authored_marker_corpus.md).

Original source hashes before machine-local paths were privacy-redacted in the
archive:

| Source | SHA-256 |
|---|---|
| `generate_h3_post_training_data.py.rejected` | `18806227b1fe7a9d997b6db8757740df793158c1e2a1ae4578e1425eb28a50c0` |
| `train_h3_bpe_transplant.py.rejected` | `dd2de286f17a4a831df77a1d0049c54c40d87f6dfc44234a34bf9c265a506493` |
