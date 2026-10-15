# The head reference's mouth a frame at a time, on the remade references (2026-10-15)

lane: masked video to video
verdict: data, against the answers known. Under eed4acb9 a frame whose lip lines missed the class map's mouth got no mouth, in either form: the remade references had a mouth on 11 of 51, 10 of 29, 0 of 22, 0 of 16 and 214 of 275 frames (630, 805, 1010, 5114, 2823). A frame at a time, the lines where they lie on it and else the class map's own mouth where a gate of its own passes (on the body model's picture, near the pose model's mouth), they get one on 27, 19, 11, 14 and 275. Of 101 answered frames, 6 go against the answer, all on 805.

**Read this first.** The answers are three blind readers' of 10/14 (630b, 805b, 1010d; each saw the class map's mouth drawn
on every frame) and the lead's word on 2823's open mouth (mrarc's sheets, 2026-10-15). The gate's two values are starting
values, set on these same frames: a fit, not a test. The outside reviewer's frame-by-frame table of open and closed mouths
from picture and sound (request G6) is the second test, still to come.

The answers (the lead's order, mrtall, 2026-10-15): 630 kept on 632-647, dropped on 658-680 (in the air, in profile);
805 kept on 808-812, 819-824 and 828-833, dropped on 806-807, 813-818 and 825 ("open or off her lips"); 1010 dropped on
1026-1030 (on her chest); 2823 drawn on 2862-2867, 2912-2922 and 2957-2970 (wide open). 5114 has none.

## How

`bench/head_reference.py`'s own functions on each reference's own plain cut, box, points and class map (2823's from its
folder moved aside, whose files are those of the failed run; its plain cut and points do not depend on the mouth):

```python
mouths, held = hr.class_mouths(classes, classes_first, first, count, rec["box"], plain_cut_size)
passes, gate = hr.class_mouth_gate(mouths, held.pop("_centres"), plain_cut, points, first, count)
shares = hr.mouth_check(points, classes, classes_first, first, count)["_shares"]
# a frame: the lip lines at MOUTH_CHECK on_share or over, else the class map's own mouth where `passes`, else no mouth
```

`on_body` is the share of the class map's mouth, cut by the box, on the plain cut's pixels over `body_grey`; the distance
is from its centre on the canvas to the pose model's outer-lip ring's centre, over the face's height
(`_lib/face_points.face_height`). The gate passes at `on_body` 0.6 or more and a distance of 0.4 or less
(`CLASS_MOUTH_GATE`).

## The values

- The distance decides no answered frame anywhere from 0.25 to 0.45: 1010's chest frames are 0.47 and more. 0.4 is set
  above the small faces nobody faulted (1010-1025 and 5114-5129, 0.20 to 0.38; 1023 is 0.57) and below the chest.
- `on_body` against the answers, the distance at 0.4: 0.4 goes against them on 7 frames (658, 661, 677, 807, 813, 814,
  818 drawn); 0.45 on 5 (677, 807, 813, 814, 818 drawn); 0.5 on 6 (807, 813, 814, 818 drawn; 822, 823 left out); 0.6 on 6
  (807, 813, 814 drawn; 819, 822, 823 left out); 0.65 on 7 (824 left out as well). 0.6 draws the fewest mouths a reader
  faulted. 807, 813 and 814 sit over 0.7 on the body model and near the pose mouth, so no value drops them; the outside
  reviewer reads them as closed or pursed mouths, drawn closed. 0.45 would keep 819, 822 and 823 (open, three-quarter,
  0.524, 0.499, 0.478) and draw 630's 677 (0.498, in the air) and 805's 818 (0.553). Left out, a frame keeps the body
  model's own lips; drawn wrong, it puts a mouth in the air or off her lips. 0.6 is the starting value.
- The outside reviewer's figures for this gate (`internal/gemini/2026-10-15_outbox.md`, G4) agree with these but for two:
  it keeps 630-657 (630 itself is dropped here: `on_body` 0.39), and it names 664 as the frame 0.45 adds on 630 (664 is
  0.27 here; the frame at 0.498 is 677).

## What else was tried

- The outside reviewer's parked shape (`internal/gemini/2026-10-14_consolidated_recommendations_for_tomorrow.md`, 5A):
  the part model's lips and Face_Neck do not overlap, so the share of the lips inside Face_Neck grown 2 px reads 0.15 to
  0.58 on frames the readers kept and its 0.9 drops every frame; a distance of 0.25 keeps 14 of 630's 23 faulted frames.
- The share inside the face filled (Face_Neck with the mouth classes, holes filled) reads 1.00 on every frame, the chest
  too: filling takes the lips in. The share of a ring round the mouth that is Face_Neck reads 0.54 to 0.97 on 1026-1030:
  the part model labels her face round the lips on her chest.

## What it cannot see

Whether a mouth drawn is open when hers is closed: the class map says what it labelled. The body model's head against its
body: a mouth on the body model's chest passes `on_body`, and only the distance drops it. A pose model off the face on a
frame. The picture: a render, played, is the reading.

## The frames

### lead_0630_0680 (630-680): the lip lines on 11, the class map's own mouth on 16, no mouth on 24 of 51 frames (under eed4acb9: a mouth on 11)

| frame | lines' share | on_body | from the pose mouth (face heights) | drawn | the reader | |
|---|---|---|---|---|---|---|
| 630 | 0.33 | 0.39 | 0.249 | no mouth |  |  |
| 631 | 0.53 | 0.788 | 0.095 | the class map |  |  |
| 632 | 0.73 | 0.88 | 0.127 | the class map | KEEP |  |
| 633 | 0.53 | 0.915 | 0.151 | the class map | KEEP |  |
| 634 | 1.00 | 0.901 | 0.081 | the lip lines | KEEP |  |
| 635 | 0.67 | 0.864 | 0.177 | the class map | KEEP |  |
| 636 | 1.00 | 0.926 | 0.047 | the lip lines | KEEP |  |
| 637 | 1.00 | 0.963 | 0.035 | the lip lines | KEEP |  |
| 638 | 1.00 | 0.908 | 0.042 | the lip lines | KEEP |  |
| 639 | 1.00 | 0.995 | 0.084 | the lip lines | KEEP |  |
| 640 | 0.53 | 1.0 | 0.158 | the class map | KEEP |  |
| 641 | 0.47 | 0.998 | 0.138 | the class map | KEEP |  |
| 642 | 0.33 | 1.0 | 0.159 | the class map | KEEP |  |
| 643 | 0.33 | 1.0 | 0.171 | the class map | KEEP |  |
| 644 | 0.47 | 0.85 | 0.135 | the class map | KEEP |  |
| 645 | 0.87 | 0.942 | 0.111 | the lip lines | KEEP |  |
| 646 | 0.93 | 0.981 | 0.035 | the lip lines | KEEP |  |
| 647 | 0.87 | 1.0 | 0.075 | the lip lines | KEEP |  |
| 648 | 1.00 | 0.997 | 0.008 | the lip lines |  |  |
| 649 | 0.73 | 0.988 | 0.168 | the class map |  |  |
| 650 | 1.00 | 0.76 | 0.077 | the lip lines |  |  |
| 651 | 0.67 | 0.926 | 0.158 | the class map |  |  |
| 652 | 1.00 | 0.936 | 0.115 | the lip lines |  |  |
| 653 | 0.27 | 0.999 | 0.218 | the class map |  |  |
| 654 | 0.73 | 0.909 | 0.105 | the class map |  |  |
| 655 | 0.20 | 0.854 | 0.218 | the class map |  |  |
| 656 | 0.20 | 0.764 | 0.221 | the class map |  |  |
| 657 | 0.27 | 0.836 | 0.196 | the class map |  |  |
| 658 | 0.27 | 0.437 | 0.219 | no mouth | DROP |  |
| 659 | 0.27 | 0.243 | 0.253 | no mouth | DROP |  |
| 660 | 0.40 | 0.39 | 0.223 | no mouth | DROP |  |
| 661 | 0.40 | 0.433 | 0.253 | no mouth | DROP |  |
| 662 | 0.27 | 0.313 | 0.278 | no mouth | DROP |  |
| 663 | 0.33 | 0.221 | 0.248 | no mouth | DROP |  |
| 664 | 0.27 | 0.27 | 0.236 | no mouth | DROP |  |
| 665 | 0.20 | 0.203 | 0.236 | no mouth | DROP |  |
| 666 | 0.27 | 0.254 | 0.23 | no mouth | DROP |  |
| 667 | 0.07 | 0.193 | 0.237 | no mouth | DROP |  |
| 668 | 0.00 | 0.198 | 0.237 | no mouth | DROP |  |
| 669 | 0.00 | 0.181 | 0.228 | no mouth | DROP |  |
| 670 | 0.07 | 0.384 | 0.232 | no mouth | DROP |  |
| 671 | 0.07 | 0.27 | 0.246 | no mouth | DROP |  |
| 672 | 0.00 | 0.339 | 0.247 | no mouth | DROP |  |
| 673 | 0.07 | 0.046 | 0.241 | no mouth | DROP |  |
| 674 | 0.00 | 0.023 | 0.252 | no mouth | DROP |  |
| 675 | 0.07 | 0.081 | 0.22 | no mouth | DROP |  |
| 676 | 0.00 | 0.153 | 0.271 | no mouth | DROP |  |
| 677 | 0.00 | 0.498 | 0.267 | no mouth | DROP |  |
| 678 | 0.00 | 0.0 | 0.261 | no mouth | DROP |  |
| 679 | 0.00 | 0.0 | 0.285 | no mouth | DROP |  |
| 680 | 0.00 | 0.0 | 0.298 | no mouth | DROP |  |

### lead_0805_0833 (805-833): the lip lines on 10, the class map's own mouth on 9, no mouth on 10 of 29 frames (under eed4acb9: a mouth on 10)

| frame | lines' share | on_body | from the pose mouth (face heights) | drawn | the reader | |
|---|---|---|---|---|---|---|
| 805 | 0.13 | 0.0 | 0.323 | no mouth |  |  |
| 806 | 0.40 | 0.219 | 0.205 | no mouth | DROP |  |
| 807 | 0.33 | 0.983 | 0.197 | the class map | DROP | AGAINST THE ANSWER |
| 808 | 0.93 | 0.653 | 0.051 | the lip lines | KEEP |  |
| 809 | 0.93 | 0.984 | 0.046 | the lip lines | KEEP |  |
| 810 | 0.93 | 0.781 | 0.072 | the lip lines | KEEP |  |
| 811 | 1.00 | 0.82 | 0.02 | the lip lines | KEEP |  |
| 812 | 0.53 | 0.72 | 0.196 | the class map | KEEP |  |
| 813 | 0.40 | 0.72 | 0.212 | the class map | DROP | AGAINST THE ANSWER |
| 814 | 0.20 | 0.754 | 0.217 | the class map | DROP | AGAINST THE ANSWER |
| 815 | 0.00 | 0.061 | 0.264 | no mouth | DROP |  |
| 816 | 0.00 | 0.0 | 0.253 | no mouth | DROP |  |
| 817 | 0.20 | 0.244 | 0.239 | no mouth | DROP |  |
| 818 | 0.40 | 0.553 | 0.213 | no mouth | DROP |  |
| 819 | 0.60 | 0.524 | 0.152 | no mouth | KEEP | AGAINST THE ANSWER |
| 820 | 0.73 | 0.654 | 0.113 | the class map | KEEP |  |
| 821 | 0.67 | 0.656 | 0.166 | the class map | KEEP |  |
| 822 | 0.60 | 0.499 | 0.192 | no mouth | KEEP | AGAINST THE ANSWER |
| 823 | 0.60 | 0.478 | 0.182 | no mouth | KEEP | AGAINST THE ANSWER |
| 824 | 0.60 | 0.62 | 0.171 | the class map | KEEP |  |
| 825 | 0.73 | 0.399 | 0.1 | no mouth | DROP |  |
| 826 | 0.73 | 0.701 | 0.089 | the class map |  |  |
| 827 | 0.80 | 0.704 | 0.063 | the lip lines |  |  |
| 828 | 0.67 | 0.731 | 0.119 | the class map | KEEP |  |
| 829 | 0.87 | 0.697 | 0.015 | the lip lines | KEEP |  |
| 830 | 0.87 | 0.698 | 0.024 | the lip lines | KEEP |  |
| 831 | 1.00 | 0.764 | 0.025 | the lip lines | KEEP |  |
| 832 | 0.93 | 0.691 | 0.041 | the lip lines | KEEP |  |
| 833 | 0.87 | 0.897 | 0.09 | the lip lines | KEEP |  |

### second_1010_1031 (1010-1031): the lip lines on 0, the class map's own mouth on 11, no mouth on 11 of 22 frames (under eed4acb9: a mouth on 0)

| frame | lines' share | on_body | from the pose mouth (face heights) | drawn | the reader | |
|---|---|---|---|---|---|---|
| 1010 | 0.00 | 0.686 | 0.315 | the class map |  |  |
| 1011 | 0.00 | 0.987 | 0.354 | the class map |  |  |
| 1012 | 0.00 | 1.0 | 0.378 | the class map |  |  |
| 1013 | 0.00 | 1.0 | 0.3 | the class map |  |  |
| 1014 | 0.00 | 0.991 | 0.302 | the class map |  |  |
| 1015 | 0.00 | 1.0 | 0.349 | the class map |  |  |
| 1016 | 0.00 | 1.0 | 0.309 | the class map |  |  |
| 1017 | 0.00 | 1.0 | 0.35 | the class map |  |  |
| 1018 | 0.00 | 0.712 | 0.322 | the class map |  |  |
| 1019 | 0.00 | 0.073 | 0.299 | no mouth |  |  |
| 1020 | 0.00 | 0.569 | 0.343 | no mouth |  |  |
| 1021 | 0.00 | 0.946 | 0.348 | the class map |  |  |
| 1022 | 0.07 | 0.0 | 0.199 | no mouth |  |  |
| 1023 | 0.00 | 1.0 | 0.574 | no mouth |  |  |
| 1024 | 0.00 | - | no lips | no mouth |  |  |
| 1025 | 0.33 | 1.0 | 0.205 | the class map |  |  |
| 1026 | 0.00 | 1.0 | 1.931 | no mouth | DROP |  |
| 1027 | 0.00 | - | no lips | no mouth | DROP |  |
| 1028 | 0.00 | 1.0 | 0.978 | no mouth | DROP |  |
| 1029 | 0.00 | 1.0 | 0.469 | no mouth | DROP |  |
| 1030 | 0.00 | 1.0 | 0.602 | no mouth | DROP |  |
| 1031 | 0.00 | 1.0 | 1.017 | no mouth |  |  |

### second_5114_5129 (5114-5129): the lip lines on 0, the class map's own mouth on 14, no mouth on 2 of 16 frames (under eed4acb9: a mouth on 0)

| frame | lines' share | on_body | from the pose mouth (face heights) | drawn | the reader | |
|---|---|---|---|---|---|---|
| 5114 | 0.00 | 1.0 | 0.25 | the class map |  |  |
| 5115 | 0.00 | 1.0 | 0.28 | the class map |  |  |
| 5116 | 0.40 | 0.994 | 0.264 | the class map |  |  |
| 5117 | 0.00 | 0.997 | 0.289 | the class map |  |  |
| 5118 | 0.00 | 0.858 | 0.325 | the class map |  |  |
| 5119 | 0.00 | 0.714 | 0.32 | the class map |  |  |
| 5120 | 0.00 | 0.216 | 0.344 | no mouth |  |  |
| 5121 | 0.00 | 0.36 | 0.336 | no mouth |  |  |
| 5122 | 0.00 | 0.625 | 0.315 | the class map |  |  |
| 5123 | 0.00 | 0.717 | 0.317 | the class map |  |  |
| 5124 | 0.00 | 0.751 | 0.3 | the class map |  |  |
| 5125 | 0.00 | 0.876 | 0.302 | the class map |  |  |
| 5126 | 0.00 | 0.946 | 0.318 | the class map |  |  |
| 5127 | 0.00 | 0.867 | 0.29 | the class map |  |  |
| 5128 | 0.00 | 0.94 | 0.295 | the class map |  |  |
| 5129 | 0.00 | 0.901 | 0.295 | the class map |  |  |

### lead_2823_3097_SUPERSEDED_failed_off_track_on_uncommitted_tool (2823-3097): the lip lines on 214, the class map's own mouth on 61, no mouth on 0 of 275 frames (under eed4acb9: a mouth on 214)

| frame | lines' share | on_body | from the pose mouth (face heights) | drawn | the reader | |
|---|---|---|---|---|---|---|
| 2823 | 1.00 | 1.0 | 0.01 | the lip lines |  |  |
| 2824 | 0.93 | 1.0 | 0.011 | the lip lines |  |  |
| 2825 | 1.00 | 1.0 | 0.008 | the lip lines |  |  |
| 2826 | 0.87 | 1.0 | 0.008 | the lip lines |  |  |
| 2827 | 1.00 | 1.0 | 0.011 | the lip lines |  |  |
| 2828 | 0.93 | 1.0 | 0.012 | the lip lines |  |  |
| 2829 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 2830 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 2831 | 0.93 | 1.0 | 0.021 | the lip lines |  |  |
| 2832 | 1.00 | 1.0 | 0.016 | the lip lines |  |  |
| 2833 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 2834 | 1.00 | 1.0 | 0.025 | the lip lines |  |  |
| 2835 | 0.93 | 1.0 | 0.023 | the lip lines |  |  |
| 2836 | 0.93 | 1.0 | 0.02 | the lip lines |  |  |
| 2837 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 2838 | 0.93 | 1.0 | 0.019 | the lip lines |  |  |
| 2839 | 0.93 | 1.0 | 0.019 | the lip lines |  |  |
| 2840 | 0.93 | 1.0 | 0.015 | the lip lines |  |  |
| 2841 | 1.00 | 1.0 | 0.019 | the lip lines |  |  |
| 2842 | 0.93 | 1.0 | 0.027 | the lip lines |  |  |
| 2843 | 0.87 | 1.0 | 0.028 | the lip lines |  |  |
| 2844 | 0.73 | 1.0 | 0.027 | the class map |  |  |
| 2845 | 0.67 | 1.0 | 0.024 | the class map |  |  |
| 2846 | 0.87 | 1.0 | 0.022 | the lip lines |  |  |
| 2847 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 2848 | 0.73 | 1.0 | 0.025 | the class map |  |  |
| 2849 | 0.73 | 1.0 | 0.024 | the class map |  |  |
| 2850 | 0.80 | 1.0 | 0.02 | the lip lines |  |  |
| 2851 | 0.80 | 1.0 | 0.021 | the lip lines |  |  |
| 2852 | 1.00 | 1.0 | 0.02 | the lip lines |  |  |
| 2853 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 2854 | 1.00 | 1.0 | 0.022 | the lip lines |  |  |
| 2855 | 0.93 | 1.0 | 0.021 | the lip lines |  |  |
| 2856 | 0.93 | 1.0 | 0.021 | the lip lines |  |  |
| 2857 | 0.93 | 1.0 | 0.018 | the lip lines |  |  |
| 2858 | 0.93 | 1.0 | 0.019 | the lip lines |  |  |
| 2859 | 0.93 | 1.0 | 0.019 | the lip lines |  |  |
| 2860 | 0.87 | 1.0 | 0.019 | the lip lines |  |  |
| 2861 | 0.80 | 1.0 | 0.024 | the lip lines |  |  |
| 2862 | 0.73 | 1.0 | 0.027 | the class map | KEEP |  |
| 2863 | 0.60 | 1.0 | 0.03 | the class map | KEEP |  |
| 2864 | 0.60 | 1.0 | 0.029 | the class map | KEEP |  |
| 2865 | 0.73 | 1.0 | 0.027 | the class map | KEEP |  |
| 2866 | 0.73 | 1.0 | 0.026 | the class map | KEEP |  |
| 2867 | 0.73 | 1.0 | 0.029 | the class map | KEEP |  |
| 2868 | 0.80 | 1.0 | 0.037 | the lip lines |  |  |
| 2869 | 0.93 | 1.0 | 0.042 | the lip lines |  |  |
| 2870 | 0.93 | 1.0 | 0.048 | the lip lines |  |  |
| 2871 | 0.93 | 1.0 | 0.049 | the lip lines |  |  |
| 2872 | 0.80 | 1.0 | 0.047 | the lip lines |  |  |
| 2873 | 0.87 | 1.0 | 0.043 | the lip lines |  |  |
| 2874 | 0.80 | 1.0 | 0.039 | the lip lines |  |  |
| 2875 | 0.80 | 1.0 | 0.037 | the lip lines |  |  |
| 2876 | 0.93 | 1.0 | 0.036 | the lip lines |  |  |
| 2877 | 0.93 | 1.0 | 0.038 | the lip lines |  |  |
| 2878 | 0.93 | 1.0 | 0.041 | the lip lines |  |  |
| 2879 | 0.93 | 1.0 | 0.037 | the lip lines |  |  |
| 2880 | 0.87 | 1.0 | 0.032 | the lip lines |  |  |
| 2881 | 0.80 | 1.0 | 0.031 | the lip lines |  |  |
| 2882 | 0.87 | 1.0 | 0.025 | the lip lines |  |  |
| 2883 | 0.87 | 1.0 | 0.012 | the lip lines |  |  |
| 2884 | 0.93 | 1.0 | 0.01 | the lip lines |  |  |
| 2885 | 0.67 | 1.0 | 0.004 | the class map |  |  |
| 2886 | 0.87 | 1.0 | 0.008 | the lip lines |  |  |
| 2887 | 0.87 | 1.0 | 0.002 | the lip lines |  |  |
| 2888 | 0.87 | 1.0 | 0.006 | the lip lines |  |  |
| 2889 | 0.87 | 1.0 | 0.008 | the lip lines |  |  |
| 2890 | 0.93 | 1.0 | 0.012 | the lip lines |  |  |
| 2891 | 0.93 | 1.0 | 0.014 | the lip lines |  |  |
| 2892 | 0.93 | 1.0 | 0.015 | the lip lines |  |  |
| 2893 | 0.93 | 1.0 | 0.017 | the lip lines |  |  |
| 2894 | 0.93 | 1.0 | 0.019 | the lip lines |  |  |
| 2895 | 0.93 | 1.0 | 0.015 | the lip lines |  |  |
| 2896 | 1.00 | 1.0 | 0.009 | the lip lines |  |  |
| 2897 | 0.93 | 1.0 | 0.014 | the lip lines |  |  |
| 2898 | 0.80 | 1.0 | 0.015 | the lip lines |  |  |
| 2899 | 1.00 | 1.0 | 0.013 | the lip lines |  |  |
| 2900 | 0.93 | 1.0 | 0.011 | the lip lines |  |  |
| 2901 | 1.00 | 1.0 | 0.011 | the lip lines |  |  |
| 2902 | 1.00 | 1.0 | 0.011 | the lip lines |  |  |
| 2903 | 0.93 | 1.0 | 0.01 | the lip lines |  |  |
| 2904 | 0.73 | 1.0 | 0.005 | the class map |  |  |
| 2905 | 0.87 | 1.0 | 0.002 | the lip lines |  |  |
| 2906 | 0.87 | 1.0 | 0.002 | the lip lines |  |  |
| 2907 | 0.80 | 1.0 | 0.005 | the lip lines |  |  |
| 2908 | 0.87 | 1.0 | 0.01 | the lip lines |  |  |
| 2909 | 0.93 | 1.0 | 0.012 | the lip lines |  |  |
| 2910 | 0.93 | 1.0 | 0.014 | the lip lines |  |  |
| 2911 | 1.00 | 1.0 | 0.018 | the lip lines |  |  |
| 2912 | 0.73 | 1.0 | 0.02 | the class map | KEEP |  |
| 2913 | 0.60 | 1.0 | 0.02 | the class map | KEEP |  |
| 2914 | 0.53 | 1.0 | 0.021 | the class map | KEEP |  |
| 2915 | 0.73 | 1.0 | 0.026 | the class map | KEEP |  |
| 2916 | 0.93 | 1.0 | 0.021 | the lip lines | KEEP |  |
| 2917 | 0.60 | 1.0 | 0.026 | the class map | KEEP |  |
| 2918 | 0.73 | 1.0 | 0.026 | the class map | KEEP |  |
| 2919 | 0.73 | 1.0 | 0.025 | the class map | KEEP |  |
| 2920 | 0.73 | 1.0 | 0.025 | the class map | KEEP |  |
| 2921 | 0.67 | 1.0 | 0.025 | the class map | KEEP |  |
| 2922 | 0.60 | 1.0 | 0.026 | the class map | KEEP |  |
| 2923 | 0.87 | 1.0 | 0.025 | the lip lines |  |  |
| 2924 | 0.87 | 1.0 | 0.019 | the lip lines |  |  |
| 2925 | 0.93 | 1.0 | 0.011 | the lip lines |  |  |
| 2926 | 0.93 | 1.0 | 0.009 | the lip lines |  |  |
| 2927 | 0.93 | 1.0 | 0.005 | the lip lines |  |  |
| 2928 | 0.93 | 1.0 | 0.004 | the lip lines |  |  |
| 2929 | 0.93 | 1.0 | 0.003 | the lip lines |  |  |
| 2930 | 1.00 | 1.0 | 0.004 | the lip lines |  |  |
| 2931 | 1.00 | 1.0 | 0.004 | the lip lines |  |  |
| 2932 | 1.00 | 1.0 | 0.005 | the lip lines |  |  |
| 2933 | 0.87 | 1.0 | 0.009 | the lip lines |  |  |
| 2934 | 0.87 | 1.0 | 0.009 | the lip lines |  |  |
| 2935 | 0.93 | 1.0 | 0.007 | the lip lines |  |  |
| 2936 | 0.93 | 1.0 | 0.005 | the lip lines |  |  |
| 2937 | 1.00 | 1.0 | 0.002 | the lip lines |  |  |
| 2938 | 1.00 | 1.0 | 0.009 | the lip lines |  |  |
| 2939 | 0.80 | 1.0 | 0.011 | the lip lines |  |  |
| 2940 | 1.00 | 1.0 | 0.006 | the lip lines |  |  |
| 2941 | 1.00 | 1.0 | 0.005 | the lip lines |  |  |
| 2942 | 1.00 | 1.0 | 0.009 | the lip lines |  |  |
| 2943 | 0.93 | 1.0 | 0.001 | the lip lines |  |  |
| 2944 | 0.87 | 1.0 | 0.003 | the lip lines |  |  |
| 2945 | 0.87 | 1.0 | 0.005 | the lip lines |  |  |
| 2946 | 0.87 | 1.0 | 0.005 | the lip lines |  |  |
| 2947 | 0.80 | 1.0 | 0.004 | the lip lines |  |  |
| 2948 | 0.87 | 1.0 | 0.004 | the lip lines |  |  |
| 2949 | 0.93 | 1.0 | 0.006 | the lip lines |  |  |
| 2950 | 1.00 | 1.0 | 0.008 | the lip lines |  |  |
| 2951 | 1.00 | 1.0 | 0.012 | the lip lines |  |  |
| 2952 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 2953 | 1.00 | 1.0 | 0.02 | the lip lines |  |  |
| 2954 | 1.00 | 1.0 | 0.023 | the lip lines |  |  |
| 2955 | 0.87 | 1.0 | 0.027 | the lip lines |  |  |
| 2956 | 0.87 | 1.0 | 0.031 | the lip lines |  |  |
| 2957 | 0.67 | 1.0 | 0.035 | the class map | KEEP |  |
| 2958 | 0.73 | 1.0 | 0.035 | the class map | KEEP |  |
| 2959 | 0.73 | 1.0 | 0.037 | the class map | KEEP |  |
| 2960 | 0.87 | 1.0 | 0.038 | the lip lines | KEEP |  |
| 2961 | 0.73 | 1.0 | 0.043 | the class map | KEEP |  |
| 2962 | 0.67 | 1.0 | 0.046 | the class map | KEEP |  |
| 2963 | 0.73 | 1.0 | 0.046 | the class map | KEEP |  |
| 2964 | 0.73 | 1.0 | 0.047 | the class map | KEEP |  |
| 2965 | 0.73 | 1.0 | 0.049 | the class map | KEEP |  |
| 2966 | 0.80 | 1.0 | 0.046 | the lip lines | KEEP |  |
| 2967 | 0.73 | 1.0 | 0.048 | the class map | KEEP |  |
| 2968 | 0.80 | 1.0 | 0.046 | the lip lines | KEEP |  |
| 2969 | 0.67 | 1.0 | 0.046 | the class map | KEEP |  |
| 2970 | 0.73 | 1.0 | 0.046 | the class map | KEEP |  |
| 2971 | 0.87 | 1.0 | 0.042 | the lip lines |  |  |
| 2972 | 0.93 | 1.0 | 0.036 | the lip lines |  |  |
| 2973 | 0.93 | 1.0 | 0.031 | the lip lines |  |  |
| 2974 | 0.93 | 1.0 | 0.027 | the lip lines |  |  |
| 2975 | 0.80 | 1.0 | 0.028 | the lip lines |  |  |
| 2976 | 0.93 | 1.0 | 0.026 | the lip lines |  |  |
| 2977 | 0.80 | 1.0 | 0.024 | the lip lines |  |  |
| 2978 | 0.93 | 1.0 | 0.02 | the lip lines |  |  |
| 2979 | 0.73 | 1.0 | 0.019 | the class map |  |  |
| 2980 | 0.73 | 1.0 | 0.017 | the class map |  |  |
| 2981 | 0.60 | 1.0 | 0.017 | the class map |  |  |
| 2982 | 0.67 | 1.0 | 0.017 | the class map |  |  |
| 2983 | 0.47 | 1.0 | 0.016 | the class map |  |  |
| 2984 | 0.60 | 1.0 | 0.017 | the class map |  |  |
| 2985 | 0.67 | 1.0 | 0.016 | the class map |  |  |
| 2986 | 0.93 | 1.0 | 0.014 | the lip lines |  |  |
| 2987 | 0.93 | 1.0 | 0.016 | the lip lines |  |  |
| 2988 | 0.87 | 1.0 | 0.016 | the lip lines |  |  |
| 2989 | 0.87 | 1.0 | 0.016 | the lip lines |  |  |
| 2990 | 0.80 | 1.0 | 0.016 | the lip lines |  |  |
| 2991 | 0.87 | 1.0 | 0.015 | the lip lines |  |  |
| 2992 | 0.87 | 1.0 | 0.017 | the lip lines |  |  |
| 2993 | 0.87 | 1.0 | 0.019 | the lip lines |  |  |
| 2994 | 0.87 | 1.0 | 0.018 | the lip lines |  |  |
| 2995 | 0.73 | 1.0 | 0.022 | the class map |  |  |
| 2996 | 0.60 | 1.0 | 0.024 | the class map |  |  |
| 2997 | 0.80 | 1.0 | 0.026 | the lip lines |  |  |
| 2998 | 0.73 | 1.0 | 0.027 | the class map |  |  |
| 2999 | 0.73 | 1.0 | 0.027 | the class map |  |  |
| 3000 | 0.80 | 1.0 | 0.027 | the lip lines |  |  |
| 3001 | 0.73 | 1.0 | 0.027 | the class map |  |  |
| 3002 | 0.80 | 1.0 | 0.026 | the lip lines |  |  |
| 3003 | 0.60 | 1.0 | 0.026 | the class map |  |  |
| 3004 | 0.80 | 1.0 | 0.024 | the lip lines |  |  |
| 3005 | 1.00 | 1.0 | 0.02 | the lip lines |  |  |
| 3006 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 3007 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 3008 | 0.60 | 1.0 | 0.027 | the class map |  |  |
| 3009 | 0.60 | 1.0 | 0.028 | the class map |  |  |
| 3010 | 0.67 | 1.0 | 0.026 | the class map |  |  |
| 3011 | 1.00 | 1.0 | 0.012 | the lip lines |  |  |
| 3012 | 0.87 | 1.0 | 0.021 | the lip lines |  |  |
| 3013 | 0.87 | 1.0 | 0.022 | the lip lines |  |  |
| 3014 | 0.73 | 1.0 | 0.026 | the class map |  |  |
| 3015 | 0.80 | 1.0 | 0.027 | the lip lines |  |  |
| 3016 | 0.80 | 1.0 | 0.027 | the lip lines |  |  |
| 3017 | 0.80 | 1.0 | 0.027 | the lip lines |  |  |
| 3018 | 1.00 | 1.0 | 0.026 | the lip lines |  |  |
| 3019 | 1.00 | 1.0 | 0.025 | the lip lines |  |  |
| 3020 | 0.73 | 1.0 | 0.03 | the class map |  |  |
| 3021 | 0.73 | 1.0 | 0.027 | the class map |  |  |
| 3022 | 0.67 | 1.0 | 0.024 | the class map |  |  |
| 3023 | 0.93 | 1.0 | 0.02 | the lip lines |  |  |
| 3024 | 0.93 | 1.0 | 0.013 | the lip lines |  |  |
| 3025 | 1.00 | 1.0 | 0.013 | the lip lines |  |  |
| 3026 | 1.00 | 1.0 | 0.012 | the lip lines |  |  |
| 3027 | 1.00 | 1.0 | 0.011 | the lip lines |  |  |
| 3028 | 1.00 | 1.0 | 0.008 | the lip lines |  |  |
| 3029 | 0.93 | 1.0 | 0.008 | the lip lines |  |  |
| 3030 | 0.87 | 1.0 | 0.01 | the lip lines |  |  |
| 3031 | 1.00 | 1.0 | 0.009 | the lip lines |  |  |
| 3032 | 1.00 | 1.0 | 0.009 | the lip lines |  |  |
| 3033 | 1.00 | 1.0 | 0.011 | the lip lines |  |  |
| 3034 | 1.00 | 1.0 | 0.01 | the lip lines |  |  |
| 3035 | 1.00 | 1.0 | 0.01 | the lip lines |  |  |
| 3036 | 1.00 | 1.0 | 0.007 | the lip lines |  |  |
| 3037 | 1.00 | 1.0 | 0.006 | the lip lines |  |  |
| 3038 | 1.00 | 1.0 | 0.006 | the lip lines |  |  |
| 3039 | 0.93 | 1.0 | 0.004 | the lip lines |  |  |
| 3040 | 0.73 | 1.0 | 0.008 | the class map |  |  |
| 3041 | 0.87 | 1.0 | 0.005 | the lip lines |  |  |
| 3042 | 0.67 | 1.0 | 0.006 | the class map |  |  |
| 3043 | 0.87 | 1.0 | 0.005 | the lip lines |  |  |
| 3044 | 1.00 | 1.0 | 0.005 | the lip lines |  |  |
| 3045 | 0.87 | 1.0 | 0.002 | the lip lines |  |  |
| 3046 | 0.93 | 1.0 | 0.003 | the lip lines |  |  |
| 3047 | 1.00 | 1.0 | 0.003 | the lip lines |  |  |
| 3048 | 0.87 | 1.0 | 0.002 | the lip lines |  |  |
| 3049 | 0.87 | 1.0 | 0.003 | the lip lines |  |  |
| 3050 | 0.87 | 1.0 | 0.005 | the lip lines |  |  |
| 3051 | 0.93 | 1.0 | 0.006 | the lip lines |  |  |
| 3052 | 0.93 | 1.0 | 0.007 | the lip lines |  |  |
| 3053 | 0.87 | 1.0 | 0.008 | the lip lines |  |  |
| 3054 | 0.93 | 1.0 | 0.012 | the lip lines |  |  |
| 3055 | 0.47 | 1.0 | 0.017 | the class map |  |  |
| 3056 | 0.53 | 1.0 | 0.017 | the class map |  |  |
| 3057 | 0.93 | 1.0 | 0.016 | the lip lines |  |  |
| 3058 | 0.93 | 1.0 | 0.015 | the lip lines |  |  |
| 3059 | 0.93 | 1.0 | 0.014 | the lip lines |  |  |
| 3060 | 1.00 | 1.0 | 0.014 | the lip lines |  |  |
| 3061 | 1.00 | 1.0 | 0.014 | the lip lines |  |  |
| 3062 | 0.87 | 1.0 | 0.014 | the lip lines |  |  |
| 3063 | 0.93 | 1.0 | 0.012 | the lip lines |  |  |
| 3064 | 0.93 | 1.0 | 0.011 | the lip lines |  |  |
| 3065 | 0.47 | 1.0 | 0.09 | the class map |  |  |
| 3066 | 0.53 | 1.0 | 0.11 | the class map |  |  |
| 3067 | 0.40 | 1.0 | 0.055 | the class map |  |  |
| 3068 | 0.53 | 1.0 | 0.016 | the class map |  |  |
| 3069 | 0.87 | 1.0 | 0.008 | the lip lines |  |  |
| 3070 | 1.00 | 1.0 | 0.013 | the lip lines |  |  |
| 3071 | 1.00 | 1.0 | 0.013 | the lip lines |  |  |
| 3072 | 1.00 | 1.0 | 0.016 | the lip lines |  |  |
| 3073 | 0.93 | 1.0 | 0.016 | the lip lines |  |  |
| 3074 | 0.93 | 1.0 | 0.018 | the lip lines |  |  |
| 3075 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 3076 | 1.00 | 1.0 | 0.021 | the lip lines |  |  |
| 3077 | 1.00 | 1.0 | 0.019 | the lip lines |  |  |
| 3078 | 1.00 | 1.0 | 0.015 | the lip lines |  |  |
| 3079 | 1.00 | 1.0 | 0.015 | the lip lines |  |  |
| 3080 | 1.00 | 1.0 | 0.015 | the lip lines |  |  |
| 3081 | 0.87 | 1.0 | 0.023 | the lip lines |  |  |
| 3082 | 0.87 | 1.0 | 0.022 | the lip lines |  |  |
| 3083 | 1.00 | 1.0 | 0.022 | the lip lines |  |  |
| 3084 | 0.93 | 1.0 | 0.02 | the lip lines |  |  |
| 3085 | 1.00 | 1.0 | 0.02 | the lip lines |  |  |
| 3086 | 1.00 | 1.0 | 0.018 | the lip lines |  |  |
| 3087 | 0.87 | 1.0 | 0.019 | the lip lines |  |  |
| 3088 | 1.00 | 1.0 | 0.019 | the lip lines |  |  |
| 3089 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 3090 | 0.93 | 1.0 | 0.016 | the lip lines |  |  |
| 3091 | 0.80 | 1.0 | 0.017 | the lip lines |  |  |
| 3092 | 1.00 | 1.0 | 0.01 | the lip lines |  |  |
| 3093 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 3094 | 1.00 | 1.0 | 0.017 | the lip lines |  |  |
| 3095 | 1.00 | 1.0 | 0.014 | the lip lines |  |  |
| 3096 | 1.00 | 1.0 | 0.014 | the lip lines |  |  |
| 3097 | 1.00 | 1.0 | 0.014 | the lip lines |  |  |

6 of 101 answered frames against the answer: [807, 813, 814, 819, 822, 823]
