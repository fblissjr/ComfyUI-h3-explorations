"""Show contracts 4 and 5 of `check_reference_contracts.py` red.

Both live in ComfyUI's own files -- `comfy/model_base.py` and `comfy/sd.py` --
so a file-level mutation would edit the running install, and the render server
shares it. Instead each mutation recompiles the real function from
`inspect.getsource` with one substitution and binds the result in place for the
duration of the case. That is a source-level mutation with no file touched, and
`_mutate` REFUSES when its substitution matches nothing, so a mutation that
silently failed to apply cannot be reported as a red.

That refusal is the point. On 2026-08-22 an earlier mutation of a different
check printed nothing and was nearly recorded as "the check is blind"; it had
failed to apply and the module would not import. The spine already treats an
exception as ERRORED rather than "differed", and `_mutate` makes the
apply-failure raise instead of quietly returning the original.

**What each mutation is, in the terms the contract is written in:**

  M1  the keyframe and reference blocks in `extra_conds` swap places, so
      reference latents lead the flat lists. Nothing raises; the DiT is handed
      the same tensors in the wrong order.
  M2  `encode_from_tokens`'s `return_dict` branch stops merging the encoder's
      third return value, so `minimax_token_tags` never reaches conditioning.
**One mutation was tried and removed, and its removal is a finding.** Flipping
`return_dict=True` to `False` at `sd.py:341` -- the production caller -- does
not silently drop the tags. It raises `AttributeError: 'tuple' object has no
attribute 'pop'` on the very next line, which assumes a dict. So that half of
contract 5 is protected by a crash rather than needing a check, and the case
was dropped instead of being made to pass: the spine scores an exception as
ERRORED, and rewriting `contract5a_holds` to swallow exceptions would have
conflated "the tags are missing" with "anything at all went wrong". The silent
failures are M2 and M3, and those are the ones that red.

Auditing that call site is still what moved contract 5a onto
`encode_from_tokens_scheduled`. The earlier version drove `encode_from_tokens`
directly with an explicit `return_dict=True`, which asserts the merge works
when asked for and says nothing about whether the production caller asks.
  M3  `extra_conds` stops copying the tags into the payload -- the "dropped
      afterward" half of contract 5, which raises nothing either.
  M5  the merge stores `None` under the right key.
  M6  `extra_conds` copies an all-zero tensor instead of the tags.

**M5 and M6 are the corruption pair, and both were GREEN until 2026-08-22.**
Contracts 5a and 5b asserted that a KEY was present, which `None` and an
all-zero tensor both satisfy -- and an all-zero tag tensor is precisely the
silent "every row is text" failure the contracts exist to prevent. A presence
test passes the exact state it was written to catch. Both seams now compare
`torch.equal` against a retained marker. Found by codex predicting it from the
source; confirmed here by running both mutations green before the fix.

The near-misses matter as much: an unrelated edit to the same function, and a
restore, must both leave the verdict where the baseline is.
"""
import inspect
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "ComfyUI"))

from harness import Harness, MUTATION, NEAR_MISS, main  # noqa: E402

import check_reference_contracts as C  # noqa: E402


class _mutate:
    """Rebind `owner.attr` to its own source with one substitution applied.

    Raises if the substitution matches nothing -- a mutation that did not
    apply is a harness failure, never a red.
    """

    def __init__(self, owner, attr, old, new):
        self.owner, self.attr = owner, attr
        self.original = getattr(owner, attr)
        src = textwrap.dedent(inspect.getsource(self.original))
        if old not in src:
            raise RuntimeError(
                f"mutation anchor not found in {owner.__name__}.{attr}: "
                f"{old!r} -- the source moved and this case would have "
                f"reported a red it never produced")
        mutated = src.replace(old, new, 1)
        if mutated == src:
            raise RuntimeError("substitution produced an identical source")
        # Recompiling a method loses the implicit `__class__` cell that
        # zero-arg `super()` closes over, so the mutant would raise
        # "super(): __class__ cell not found" before reaching any assertion --
        # an ERROR, which the spine correctly refuses to count as a red. Make
        # the call explicit instead. This is harness mechanics and changes no
        # behaviour: `super()` inside a method IS `super(__class__, self)`.
        mutated = mutated.replace("super().", f"super({owner.__name__}, self).")
        ns = dict(sys.modules[self.original.__module__].__dict__)
        ns[owner.__name__] = owner
        exec(compile(mutated, f"<mutant {attr}>", "exec"), ns)
        self.replacement = ns[self.original.__name__]

    def __enter__(self):
        setattr(self.owner, self.attr, self.replacement)
        return self

    def __exit__(self, *exc):
        setattr(self.owner, self.attr, self.original)
        return False


def _with(owner_path, attr, old, new, verdict):
    def run():
        mod = __import__(owner_path[0], fromlist=[owner_path[1]])
        owner = getattr(mod, owner_path[1])
        with _mutate(owner, attr, old, new):
            return verdict()[0]
    return run


MB = ("comfy.model_base", "MiniMaxH3")
SD = ("comfy.sd", "CLIP")


def build():
    # The subject reports ok=True for a HOLDING contract, so the verdict
    # inverts: red is a contract that does not hold.
    h = Harness(subject="bench/check_reference_contracts.py",
                verdict=lambda ok: not ok)
    h.baseline(lambda: C.contract4_holds()[0] and C.contract5a_holds()[0]
               and C.contract5b_holds()[0])

    # --- mutations: each must move the verdict off the baseline ---

    h.case("M1 contract 4: reference latents concatenated BEFORE keyframes",
           MUTATION,
           _with(MB, "extra_conds",
                 'payload["cond_video_latents"] = payload.get("cond_video_latents", []) + [r["latent"] for r in refs if "latent" in r]',
                 'payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r] + payload.get("cond_video_latents", [])',
                 C.contract4_holds))

    h.case("M2 contract 5a: return_dict stops merging the encoder's extras",
           MUTATION,
           _with(SD, "encode_from_tokens",
                 "for k in o[2]:",
                 "for k in []:",
                 C.contract5a_holds))

    h.case("M3 contract 5b: extra_conds stops copying the tags",
           MUTATION,
           _with(MB, "extra_conds",
                 'payload["text_token_tags"] = tags',
                 'pass',
                 C.contract5b_holds))

    h.case("M5 contract 5a: the merge stores None under the right key",
           MUTATION,
           _with(SD, "encode_from_tokens", "out[k] = o[2][k]", "out[k] = None",
                 C.contract5a_holds))

    h.case("M6 contract 5b: the payload gets an all-zero tag tensor",
           MUTATION,
           _with(MB, "extra_conds",
                 'payload["text_token_tags"] = tags',
                 'payload["text_token_tags"] = tags * 0',
                 C.contract5b_holds))

    # --- near-misses: the verdict must NOT move ---

    h.case("G1 an unrelated edit to the same function", NEAR_MISS,
           _with(MB, "extra_conds",
                 'payload["seed"] = kwargs.get("seed", 0)',
                 'payload["seed"] = kwargs.get("seed", 0) or 0',
                 C.contract4_holds))

    h.case("G2 restored: contract 4 unmutated", NEAR_MISS,
           lambda: C.contract4_holds()[0])
    h.case("G3 restored: contract 5a unmutated", NEAR_MISS,
           lambda: C.contract5a_holds()[0])
    h.case("G4 restored: contract 5b unmutated", NEAR_MISS,
           lambda: C.contract5b_holds()[0])
    return h


if __name__ == "__main__":
    main(build)
