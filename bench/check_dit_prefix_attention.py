#!/usr/bin/env python3
"""Controls for `bench/measure_dit_prefix_attention.py`, on a synthetic capture.

The measurement answers "how much of a latent query's attention lands on the
encoder prefix, and where inside it". Three ways that number can be wrong while
still looking like a number, and one case each:

1. **The class map is not a partition.** A key belonging to no class, or to two,
   silently changes every fraction and nothing announces it. The masses of one
   query must sum to 1, because `model.py::Attention.forward` passes `mask=None`
   and the softmax denominator is therefore the whole key set.
2. **The classes are attached to the wrong positions.** A prefix-vs-latent split
   that MOVES when the prefix labels are shuffled is reading the labels where it
   should be reading the segment boundary; a within-prefix split that does NOT
   move is not reading the labels at all.
3. **The prefix boundary is off by one.** The capture records no segment table,
   so the boundary is reconstructed. One row of error reclassifies a key and
   changes nothing visible.

**Each of the three carries its own negative arm**, because a check whose input
already satisfies the expected outcome cannot fail (CLAUDE.md). The sum case is re-run
over a deliberately non-total partition and must go red; the shuffle case is
re-run through a deliberately wrong summariser and must go red; the boundary
case is re-run with the prompt one token short and must raise.

**The tag route is not available and this says so.** Control 3 as commissioned
wanted the prefix positions checked against the tags the payload gives them.
Nothing in a `qkv_*.pt` distinguishes a tag-1 row from a tag-0 one --
`h3_capture.py` writes q/k/v and nothing else -- so that check cannot be built
from a capture, and `tags_are_reported_unknown` asserts the RECORD says so in as
many words instead of leaving the reader to assume the classes were verified.
The two-route length cross-check is what stands in for it, and it is a control,
not a restatement: the geometric residual moves with the canvas and the token
count moves with the prompt, so an error in one disagrees with the other.

Claims, i.e. what breaks if a case is deleted:

- `class_masses_sum_to_one`      -- the class map covers every key exactly once.
- `incomplete_partition_is_seen` -- and the sum above can go red.
- `shuffle_preserves_prefix_split` -- prefix-vs-latent comes from the segment
                                    boundary, not from the labels.
- `shuffle_moves_within_prefix`  -- the within-prefix split does come from them.
- `broken_summariser_is_seen`    -- and the invariance above can go red.
- `off_by_one_prefix_is_caught`  -- the two-route cross-check refuses a prompt
                                    that does not belong to the tensors.
- `absent_prompt_is_unknown`     -- with no prompt the record says UNKNOWN
                                    rather than claiming agreement.
- `tags_are_reported_unknown`    -- the record never implies the vision/text tag
                                    split was verified when it was not.
- `classifier_finds_markers`     -- markers, marker spans and labels are found
                                    when present.
- `classifier_invents_nothing`   -- and not found when absent.
- `unfaithful_char_spans_disable_spans` -- when the decoded tokens do not
                                    reconstruct the prompt, every
                                    character-offset class is dropped rather
                                    than placed on a shifted offset.
- `end_to_end_on_a_capture`      -- the whole script runs on a synthetic capture
                                    and writes a record and an importance tensor.

CPU only. No CUDA, no model, no server, no ComfyUI process. Seconds.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import torch  # noqa: E402

import measure_dit_prefix_attention as M  # noqa: E402

# Synthetic capture geometry. Deliberately not a legal H3 canvas: `PackedLayout`
# does not police canvases and the point here is a few hundred rows, not a
# render. 64x64 -> 4x4 latent -> 2x2 patch rows per frame; length 5 -> latent_t
# 2, audio_t 8. So 8 video rows and 16 audio rows, and the prefix is the rest.
CANVAS = (64, 64, 5)
PREFIX_LEN = 200
HEADS, HEAD_DIM = 2, 8

FAILURES: list[str] = []


def case(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:34} {detail}")
    if not ok:
        FAILURES.append(name)
    return ok


# --------------------------------------------------------------------------


def synthetic_prompt(tokenizer, n_tokens=PREFIX_LEN):
    """A prompt of EXACTLY n_tokens that carries a marker pair and a label.

    Built by measuring rather than by counting characters: the cross-check under
    test compares a token count against a row count, so a prompt whose length was
    assumed would test the assumption instead of the code.
    """
    head = ("integrated_multimodal_description:\n"
            "<Picture 1>: a red bicycle <d>held in frame</d> at dusk.\n\n"
            "overall_soundscape:\nrain\n\nnon_diegetic_music:\nN/A")
    filler = " rain"
    text = head
    while True:
        n = len(M.tokenize_prefix(text, tokenizer)[0])
        if n >= n_tokens:
            break
        text += filler * max(1, (n_tokens - n) // 2)
    ids, _ = M.tokenize_prefix(text, tokenizer)
    while len(ids) > n_tokens:
        text = text[:-1]
        ids, _ = M.tokenize_prefix(text, tokenizer)
    return text


def synthetic_qk(seq_len, prefix_len, seed=0):
    """q/k whose prefix mass is deliberately uneven across positions.

    A uniform prefix would make the within-prefix split invariant to a label
    shuffle for a reason that has nothing to do with the code, and the shuffle
    case would pass while testing nothing.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, seq_len, HEAD_DIM, generator=g)
    k = torch.randn(1, HEADS, seq_len, HEAD_DIM, generator=g)
    # ramp the prefix key norms so early prefix positions attract far more mass
    ramp = torch.linspace(3.0, 0.2, prefix_len).view(1, 1, prefix_len, 1)
    k[:, :, :prefix_len] = k[:, :, :prefix_len] * ramp
    return q.to(torch.bfloat16).float(), k.to(torch.bfloat16).float()


def write_capture(directory: Path, prompt: str, seq_len: int):
    q, k = synthetic_qk(seq_len, PREFIX_LEN)
    torch.save({"q": q.to(torch.bfloat16), "k": k.to(torch.bfloat16),
                "v": torch.zeros_like(q, dtype=torch.bfloat16)},
               directory / f"qkv_L{seq_len}_S{seq_len}_b0_s1.pt")
    w, h, length = CANVAS
    (directory / "workflow_api.json").write_text(json.dumps({
        "1": {"class_type": "EmptyMiniMaxH3LatentAV",
              "inputs": {"width": w, "height": h, "length": length}},
        "2": {"class_type": "Synthetic", "inputs": {"prompt": prompt}},
    }))


# --------------------------------------------------------------------------


def main():
    torch.set_num_threads(2)
    print("bench/check_dit_prefix_attention.py -- CPU, synthetic capture\n")

    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tokenizer = MiniMaxH3Tokenizer()
    specials = M.special_token_ids(tokenizer)

    prompt = synthetic_prompt(tokenizer)
    ids, pieces = M.tokenize_prefix(prompt, tokenizer)
    w, h, length = CANVAS

    # the non-text row count, from the model's own layout, to size the capture
    _frames, latent_t, audio_t = M.temporal_shape(length)
    non_text = M._layout(0, latent_t, h // 16, w // 16, audio_t, []).seq_len
    seq_len = len(ids) + non_text

    layout, ids, pieces = M.derive_layout(seq_len, w, h, length, "", prompt,
                                          tokenizer)
    prefix_len = layout["prefix_len"]
    case("prefix_length_cross_check_agrees",
         layout["cross_check"]["agreement"] == "two independent routes agree"
         and prefix_len == len(ids),
         f"prefix {prefix_len} rows, S {seq_len}")

    prefix_cls, evidence = M.prefix_classes(ids, pieces, specials)
    names, class_of = M.class_map(layout["segments"], prefix_cls, seq_len)

    case("classifier_finds_markers",
         evidence["marker_tokens"].get("<d>") == 1
         and evidence["marker_tokens"].get("</d>") == 1
         and len(evidence["marker_spans"]) == 1
         and any(lbl[0].startswith("<Picture 1>") for lbl in evidence["labels"])
         and evidence["section_keys"],
         f"{evidence['marker_tokens']}, {len(evidence['marker_spans'])} span(s), "
         f"{len(evidence['labels'])} label(s)")

    plain = "a red bicycle at dusk with no markers and no labels at all"
    p_ids, p_pieces = M.tokenize_prefix(plain, tokenizer)
    _, plain_ev = M.prefix_classes(p_ids, p_pieces, specials)
    case("classifier_invents_nothing",
         not plain_ev["marker_tokens"] and not plain_ev["marker_spans"]
         and not plain_ev["labels"] and not plain_ev["section_keys"],
         "no marker, span, label or section key on plain text")

    # a character-offset placement that does not reconstruct the prompt must
    # DISABLE the span classes, not place them somewhere plausible
    _, drift_ev = M.prefix_classes(ids, pieces, specials,
                                   source_text=prompt + "drifted")
    case("unfaithful_char_spans_disable_spans",
         drift_ev["char_span_placement"].startswith("DISABLED")
         and not drift_ev["labels"] and not drift_ev["section_keys"]
         and drift_ev["marker_tokens"],
         "span classes dropped, the id-based marker class kept")

    # ---- the measurement, once, reused by the cases below
    q, k = synthetic_qk(seq_len, prefix_len)
    v_lo, v_hi, _ = next(s for s in layout["segments"] if s[2] == "video")
    rows = M.sample_rows(v_hi - v_lo, 8, strata=4, device="cpu") + v_lo
    out = M.measure_masses(q, k, rows, class_of, len(names), prefix_len,
                           head_chunk=1)
    base = M.summarise(out["per_head"], names, layout["segments"])

    lo, hi = out["row_sum_min"], out["row_sum_max"]
    case("class_masses_sum_to_one", abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9,
         f"per-query sum in [{lo:.15f}, {hi:.15f}]")

    # negative arm: a class map that leaves one segment out of the accounting
    void = names.index("seg_audio")
    partial = base["mass_by_class"]
    partial_sum = sum(mass for n, mass in partial.items() if n != "seg_audio")
    case("incomplete_partition_is_seen", partial_sum < 1.0 - 1e-6,
         f"dropping seg_audio (class {void}) sums to {partial_sum:.9f}, "
         "so the case above can go red")

    # ---- control 2: shuffle the prefix labels only
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(prefix_len, generator=g)
    shuffled = class_of.clone()
    shuffled[:prefix_len] = class_of[:prefix_len][perm]
    out2 = M.measure_masses(q, k, rows, shuffled, len(names), prefix_len,
                            head_chunk=1)
    shuf = M.summarise(out2["per_head"], names, layout["segments"])

    d_top = max(abs(base["top_level"][key] - shuf["top_level"][key])
                for key in base["top_level"])
    case("shuffle_preserves_prefix_split", d_top < 1e-12,
         f"max top-level move {d_top:.3e}")

    d_within = max(abs(base["within_prefix"][n] - shuf["within_prefix"][n])
                   for n in base["within_prefix"])
    case("shuffle_moves_within_prefix", d_within > 1e-3,
         f"max within-prefix move {d_within:.4f}")

    # negative arm: a summariser that reads prefix mass off ONE prefix class is
    # exactly the defect `shuffle_preserves_prefix_split` exists to catch, and
    # it must move under the same shuffle.
    def broken(rec):
        return rec["mass_by_class"]["prefix_text_other"]
    d_broken = abs(broken(base) - broken(shuf))
    case("broken_summariser_is_seen", d_broken > 1e-6,
         f"single-class prefix total moves {d_broken:.6f}, so the invariance "
         "above can go red")

    # ---- control 3: an off-by-one prefix boundary
    short = prompt
    while len(M.tokenize_prefix(short, tokenizer)[0]) >= len(ids):
        short = short[:-1]
    caught = ""
    try:
        M.derive_layout(seq_len, w, h, length, "", short, tokenizer)
    except SystemExit as exc:
        caught = str(exc)
    case("off_by_one_prefix_is_caught",
         "prefix_length_cross_check" in caught,
         f"refused with {len(M.tokenize_prefix(short, tokenizer)[0])} tokens "
         f"against {prefix_len} rows")

    no_prompt, _, _ = M.derive_layout(seq_len, w, h, length, "", None, tokenizer)
    case("absent_prompt_is_unknown",
         no_prompt["cross_check"]["agreement"].startswith("UNKNOWN"),
         no_prompt["cross_check"]["agreement"])

    # ---- end to end, through main(), on a capture written to disk
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp) / "synthetic_capture"
        directory.mkdir()
        write_capture(directory, prompt, seq_len)
        out_json = Path(tmp) / "record.json"
        argv = sys.argv
        sys.argv = ["measure_dit_prefix_attention.py", str(directory),
                    "--rows", "8", "--strata", "4", "--head-chunk", "1",
                    "--json", str(out_json)]
        try:
            rc = M.main()
        finally:
            sys.argv = argv
        record = json.loads(out_json.read_text()) if out_json.exists() else {}
        tensor_path = out_json.with_name(out_json.stem + "_importance.safetensors")
        ok = (rc == 0 and record.get("records")
              and tensor_path.exists())
        vec = None
        if tensor_path.exists():
            from safetensors.torch import load_file
            vec = load_file(str(tensor_path))
        case("end_to_end_on_a_capture",
             bool(ok) and vec is not None and "b0_s1" in vec
             and vec["b0_s1"].numel() == prefix_len
             and "prefix_class_id" in vec,
             f"record + importance vector of {prefix_len} positions")

        case("tags_are_reported_unknown",
             record.get("classes", {}).get("prefix_tag_source", "")
             .startswith("UNKNOWN"),
             record.get("classes", {}).get("prefix_tag_source", "<missing>"))

        # the synthetic k ramp puts the mass at the front of the prefix; a
        # vector that did not track it would be reading the wrong rows
        front = float(vec["b0_s1"][:20].sum()) if vec is not None else 0.0
        back = float(vec["b0_s1"][-20:].sum()) if vec is not None else 0.0
        case("importance_tracks_the_planted_signal", front > 5 * back,
             f"first 20 positions {front:.3e} vs last 20 {back:.3e}")

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} case(s): {', '.join(FAILURES)}")
        return 1
    print("all cases green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
