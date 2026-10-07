# Is a track still on the subject? Five measures of likeness to the subject's own seed, with a neighbour's track as the control (2026-10-07)

lane: masked
verdict: for the most central person on a crowd stretch, followed in a group of sixteen, the track most like the subject's seed is the subject's own on nearly every look by every measure and the nearest neighbour's never is, most cleanly by the lightness of the pixels under the mask; the node's own line is passed on fewer than half the looks against a single seed signature; the node's own measure (top third and head, sixteen heads asked on the frame) has nothing to compare on some looks, and for the largest person, a front-row figure, on most; one pass as fed, no nudged twin, one subject who differs from the people around in lightness

**What was asked.** A mask that stays on and keeps a plausible shape can
be on the person next to the subject
([`2026-10-07_subject_regain_looks.md`](2026-10-07_subject_regain_looks.md)
has a track doing exactly that). Can likeness to the subject's own seed
tell?

**How.** [`bench/subject_likeness_in_a_group.py`](../subject_likeness_in_a_group.py).
One subject and one set of up to sixteen seed masks, found once on the
frames as fed; the group is followed in one call by ComfyUI's SAM 3.1
through [`sam31_corrections.py`](../../sam31_corrections.py). At every
sixth frame every track with a plausible mask is compared with the
SUBJECT's seed. The subject's own track should be the most like it; the
nearest neighbour's track, judged as if it were the subject, is the
control that must fail. Five measures, each with how often it had nothing
to compare:

- the top third of the mask alone (the image trunk's features, as
  `subject_track.py` signs a person, without the head);
- with a head, the lower of the two places, as the node scores a person:
  the head looked for among sixteen heads asked on the frame (the node's
  own measure), among sixty-four, or on a crop around the judged track;
- the lightness under the mask: a coarse histogram of the lightness of
  the pixels under the mask, eroded a little, against the same under the
  subject's seed mask. No model, no head. It has no line of its own, so
  only "first among the tracks" is counted for it.

On the card, outside the server. Data:
[`2026-10-07_subject_likeness_in_a_group.json`](2026-10-07_subject_likeness_in_a_group.json);
the tables are printed from it by the tool's `render`.

**The reading, written before the first run** (the tool's docstring).
Usable as the "still the subject" check only if the subject's own track
is over the node's regain line and first among the tracks on most looks,
and the control fails.

### crowd, hard, the most central

`lotsofpeopledance_0414_0720.mkv` from 138.0 s, 144 frames at 24.0 a second, 1344x760; SAM 3.1 corrected (sam31_corrections), on cuda:0; 16 seeded on frame 4, the subject first; a look every 6 frames; the line 0.88, the margin 0.03. One run per arm.

| run | measure | who is judged against the subject's seed | looks with a plausible mask | nothing to compare | over the line | first among the tracks | over the line and first | and leading by the margin |
|---|---|---|---|---|---|---|---|---|
| as fed | the top third alone | the subject's own track | 20 | 0 | 9 | 19 | 9 | 9 |
| as fed | the top third alone | THE CONTROL: the nearest neighbour's track | 7 | 0 | 0 | 0 | 0 | 0 |
| as fed | with a head, 16 on the frame | the subject's own track | 20 | 4 | 5 | 16 | 5 | 5 |
| as fed | with a head, 16 on the frame | THE CONTROL: the nearest neighbour's track | 7 | 4 | 0 | 0 | 0 | 0 |
| as fed | with a head, 64 on the frame | the subject's own track | 20 | 0 | 6 | 19 | 6 | 6 |
| as fed | with a head, 64 on the frame | THE CONTROL: the nearest neighbour's track | 7 | 2 | 0 | 0 | 0 | 0 |
| as fed | with a head, from a crop | the subject's own track | 20 | 0 | 6 | 19 | 6 | 6 |
| as fed | with a head, from a crop | THE CONTROL: the nearest neighbour's track | 7 | 1 | 0 | 0 | 0 | 0 |
| as fed | the lightness under the mask | the subject's own track | 20 | 0 | no line | 20 | no line | no line |
| as fed | the lightness under the mask | THE CONTROL: the nearest neighbour's track | 7 | 0 | no line | 0 | no line | no line |

as fed, the subject's own track look by look (`none`: no head found to compare, or no plausible mask):

| frame | 10 | 16 | 22 | 28 | 34 | 40 | 46 | 52 | 58 | 64 | 70 | 76 | 82 | 88 | 94 | 100 | 106 | 112 | 118 | 124 | 130 | 136 | 142 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plausible | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes | no | no | no |
| the top third alone: own | 0.958 | 0.897 | 0.781 | 0.811 | 0.850 | 0.899 | 0.793 | 0.801 | 0.895 | 0.948 | 0.941 | 0.931 | 0.856 | 0.923 | 0.917 | 0.859 | 0.866 | 0.793 | 0.833 | 0.817 | none | none | none |
| the top third alone: the best other track | 0.781 | 0.692 | 0.780 | 0.694 | 0.746 | 0.772 | 0.778 | 0.828 | 0.762 | 0.779 | 0.699 | 0.739 | 0.693 | 0.740 | 0.743 | none | none | none | none | 0.507 | 0.652 | 0.559 | none |
| with a head, 16 on the frame: own | 0.950 | 0.897 | 0.781 | 0.811 | 0.848 | 0.859 | none | none | none | 0.922 | 0.935 | 0.892 | 0.816 | 0.861 | 0.858 | 0.806 | 0.837 | 0.793 | 0.833 | none | none | none | none |
| with a head, 16 on the frame: the best other track | 0.781 | 0.692 | 0.780 | 0.630 | 0.604 | 0.753 | 0.778 | 0.828 | 0.636 | 0.754 | none | 0.708 | 0.556 | 0.740 | 0.743 | none | none | none | none | none | 0.652 | none | none |
| with a head, 64 on the frame: own | 0.950 | 0.897 | 0.781 | 0.811 | 0.848 | 0.859 | 0.793 | 0.776 | 0.895 | 0.922 | 0.935 | 0.892 | 0.816 | 0.861 | 0.858 | 0.806 | 0.837 | 0.793 | 0.833 | 0.728 | none | none | none |
| with a head, 64 on the frame: the best other track | 0.781 | 0.692 | 0.780 | 0.630 | 0.667 | 0.753 | 0.778 | 0.828 | 0.636 | 0.754 | 0.697 | 0.724 | 0.655 | 0.740 | 0.743 | none | none | none | none | 0.507 | 0.652 | none | none |
| with a head, from a crop: own | 0.951 | 0.897 | 0.781 | 0.811 | 0.844 | 0.859 | 0.793 | 0.770 | 0.895 | 0.923 | 0.935 | 0.890 | 0.811 | 0.857 | 0.852 | 0.799 | 0.834 | 0.793 | 0.833 | 0.714 | none | none | none |
| with a head, from a crop: the best other track | 0.781 | 0.692 | 0.780 | 0.630 | 0.667 | 0.751 | 0.778 | 0.828 | 0.629 | 0.751 | 0.691 | 0.724 | 0.642 | 0.740 | 0.743 | none | none | none | none | 0.507 | 0.652 | none | none |
| the lightness under the mask: own | 0.922 | 0.786 | 0.684 | 0.702 | 0.713 | 0.842 | 0.707 | 0.757 | 0.732 | 0.876 | 0.891 | 0.853 | 0.863 | 0.900 | 0.921 | 0.767 | 0.800 | 0.682 | 0.706 | 0.878 | none | none | none |
| the lightness under the mask: the best other track | 0.562 | 0.482 | 0.539 | 0.541 | 0.470 | 0.524 | 0.629 | 0.550 | 0.518 | 0.522 | 0.456 | 0.483 | 0.429 | 0.578 | 0.561 | none | none | none | none | 0.351 | 0.441 | 0.477 | none |

## What this shows

- **Rank works; the node's line does not.** By every measure the
  subject's track is first on nearly every look and the neighbour's on
  none. Against a single seed signature the node's line is passed on
  fewer than half the looks. The node compares with a gallery of several
  frames of the track, which reads higher, and a gallery is what a track
  that has moved to another figure poisons.
- **The lightness under the mask separates on every look**, with a wide
  gap, and is always available. It is clothing, not identity: it tells
  this subject from the people around because they differ in lightness,
  and it will not where they do not.
- **The node's own measure sometimes has nothing to compare**: no head
  found for the track among sixteen heads asked on a crowded frame.
  Sixty-four heads, or a head asked on a crop around the track, finds one
  on every look here.

## The first form of this test, kept because it was blind

The first run used the node's measure alone, on the LARGEST person of the
same stretch (a front-row figure; the shipped `pick` at the time), as fed
and nudged. Data:
[`2026-10-07_subject_likeness_in_a_group_first_form.json`](2026-10-07_subject_likeness_in_a_group_first_form.json)
(an earlier layout, which `render` does not read).

| run | looks where the subject's track has a plausible mask | of them with no head found, scored as no match | over the line and first |
|---|---|---|---|
| as fed | 16 | 13 | 1 |
| nudged | 16 | 13 | 0 |

A person with no head found scores as no match (`subject_track.py`,
`gallery_scores`), so on most looks the score said nothing about who the
track was on, and the control "failed" only because the subject did too.
That is why the measures above are reported side by side with a "nothing
to compare" column.

## What it does not show

- A nudged twin: one pass as fed. A pass takes several minutes (a head is
  asked on a crop for every track at every look).
- The front-row figure by the five measures, who does not differ from the
  people around in lightness: the hard case, not run.
- That any of these measures catches a track that has moved: every look
  here is of a track that was on its subject or was empty. The case to
  test is the largest person's track in the regain-looks record.
- Identity. No labels.

## Reproducing it

```
<python> bench/subject_likeness_in_a_group.py run --clip lotsofpeopledance_0414_0720.mkv --second 138 --seconds 6 --width 1344 --rate 24 --subject "most central" --key "crowd, hard, the most central" --json J
<python> bench/subject_likeness_in_a_group.py render --json J
```

The run writes both passes; this record holds the as-fed pass alone (the
run was stopped after it to free the card).

## Files

- `2026-10-07_subject_likeness_in_a_group.json`: the as-fed pass, every
  look, five measures, the subject and the control.
- `2026-10-07_subject_likeness_in_a_group_first_form.json`: the first
  form's run.
