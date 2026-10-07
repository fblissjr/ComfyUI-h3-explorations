#!/usr/bin/env python3
"""What Meta's four SAM 3 datasets say about what a valid prompt is and how specific to be: counts from the annotation files.

Reads the annotation JSON of SA-Co/Gold, SA-Co/Silver, SA-Co/VEval and SA-FARI from paths you give it, prints the tables and writes one small
json of aggregates. It reads annotations only (no image, no video, no mask is decoded). It copies no dataset content: the json holds counts,
shares and quantiles, never a phrase list; `--examples N` prints up to N example phrases per class to the terminal and never writes them.

THE DATASETS ARE GATED AND ARE NOT IN THIS REPOSITORY. Their repositories on Hugging Face (facebook/SACo-Gold, SACo-Silver, SACo-VEval,
SA-FARI) carry the SAM License of 2025-11-19 and, for SA-FARI and part of VEval, a CC-BY-NC 4.0 note in their README; read those terms before
quoting anything. Formats: coderef/sam3/scripts/eval/{gold,silver,veval}/README.md.

  python bench/sam3_dataset_phrases.py run --gold DIR --silver DIR --veval DIR --safari DIR --out aggregates.json
  python bench/sam3_dataset_phrases.py render --json aggregates.json

Any of the four directories may be left out; without `--gold` the modifier list is only colours and three shade words, so "modifier + noun" counts come out
lower than with it. `run` also prints the controls it carries (Gold's counts against Meta's own README table;
VEval's phrase map against the count recorded in docs/research/masking/2026-10-07_mryolk_phrases.md).

Definitions, so the tables can be read: a PAIR is one asked (image or video, noun phrase); it is POSITIVE if at least one instance is
annotated and a NEGATIVE otherwise (Meta's own definition, Gold README). Gold has three annotators over the same pairs: tables use annotator a
unless they say otherwise. Sizes are the annotated box's area as a share of the image (Gold, Silver: the file's normalised box; VEval, SA-FARI:
pixels over the video frame). Phrase FORM is a lexical class, not a parse (no tagger is used): see FORMS and the word lists below.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

try:
    import orjson
except ImportError:                                   # optional; the stdlib parser is only slower
    orjson = None

# ---- word lists. Provenance of each: reasoned from the lead's question and checked against the datasets' own most frequent words; none is trained.
ARTICLES = {"a", "an", "the"}
#: reasoned; "cream" is left out because "ice cream" is a food, not a colour; "light"/"dark" are shade words and count as modifiers below
COLOURS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey", "silver", "gold", "golden",
           "beige", "tan", "navy", "violet", "teal", "maroon", "turquoise", "magenta", "cyan"}
#: the question named these five; "of" is counted separately because it builds "a row of X" compositions, not a relation to something else
RELATIONAL = {"wearing", "holding", "with", "in", "on"}
#: the question's examples (left, leftmost, behind) plus the position words that are frequent in the files (front, back, top, bottom, middle, centre)
POSITION = {"left", "leftmost", "right", "rightmost", "behind", "front", "back", "top", "bottom", "middle", "center", "centre", "nearest", "closest",
            "farthest", "furthest", "beside", "above", "below", "under", "next", "near"}
#: reasoned
PERSON_WORDS = {"person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "baby",
                "lady", "guy", "player", "pedestrian", "rider", "dancer", "human", "humans", "adult", "toddler"}
#: reasoned; a body-part word alone does not say whose it is, so the tables keep "with a person word" apart
BODY_WORDS = {"hand", "hands", "head", "face", "arm", "arms", "leg", "legs", "foot", "feet", "hair", "eye", "eyes", "finger", "fingers", "nose",
              "mouth", "ear", "ears", "shoulder", "shoulders", "neck", "torso", "knee", "elbow", "thumb", "palm", "wrist", "hip", "lip", "lips",
              "teeth", "tooth", "beard", "chest", "forehead", "cheek", "skin"}
#: the owner's two test props, by word; a phrase counts if it has all the words of an entry
PROPS = {"cigarette": [("cigarette",), ("cigarettes",)], "ice cream cone": [("ice", "cream", "cone"), ("ice", "cream", "cones")]}
#: a word enters the learned modifier lexicon when it is the first word of at least this many distinct Gold-Attributes phrases. Provenance: reasoned. The subset is
#: built as "attribute + noun", so a first word is an attribute by construction; two phrases drops typing slips and one-offs. At five, a third of that subset's own
#: two-word phrases fell outside the lexicon (measured on the first run), which defeats the purpose
ATTR_MIN_PHRASES = 2
#: box-share thresholds reported as shares of annotations, each with the side of the equal-area square on Meta's 1008 x 1008 canvas.
#: Provenance: round powers of ten (1e-4 is about 10 px on a side at 1008, 1e-3 about 32 px, 1e-2 about 100 px)
SIZE_THRESHOLDS = (1e-4, 1e-3, 1e-2)
CANVAS = 1008
#: instance counts a positive pair is checked against; 16 and 32 are the people counts of this pack's tracker work
COUNT_MARKS = (2, 5, 10, 16, 32)
#: minimum annotations a phrase needs before its median size is used
MIN_ANN_PER_PHRASE = 3
#: Gold: Meta's own table (coderef/sam3/scripts/eval/gold/README.md, "Data Stats"): subset -> (# image-NPs, # image-NP-masks). Wiki-Food&Drink's mask
#: count is annotator b's file; the other six are annotator a's (measured when this tool was written)
README_GOLD_STATS = {"metaclip": (33393, 20144), "sa1b": (13258, 30306), "attributes": (9245, 3663), "crowded": (20687, 50417),
                     "wiki_common": (65502, 6448), "fg_food": (13951, 9825), "fg_sports_equipment": (12166, 5075)}
#: VEval's global phrase map size as counted by mryolk (docs/research/masking/2026-10-07_mryolk_phrases.md) and the shares it recorded
MRYOLK_MAP = {"phrases": 51248, "1": 0.04, "2": 0.23, "3": 0.39, "4": 0.20, "7+": 0.015}
VEVAL_FPS = {"sav": 24, "smartglasses": 6, "yt1b": 6, "sa_fari": 6}       # README: JPEGImages_24fps / _6fps
FORMS = ("plain noun", "other two words", "modifier + noun", "two modifiers + noun", "of-phrase", "relational", "positional", "other three or more")
DESCRIPTIVE = {"modifier + noun", "two modifiers + noun", "of-phrase", "relational", "positional", "other three or more"}

#: first-word classes for Gold's Attributes subset, reasoned from its most frequent first words; "other" is everything else (type, pattern, state)
MATERIAL_WORDS = {"metal", "metallic", "wooden", "wood", "plastic", "leather", "glass", "stone", "fabric", "steel", "ceramic", "cotton", "paper", "concrete", "brick",
                  "rubber", "cloth", "wicker", "marble", "silk", "wool", "denim", "foam", "aluminum", "iron", "copper", "brass"}
SHAPE_WORDS = {"round", "flat", "rectangular", "curved", "square", "circular", "oval", "triangular", "cylindrical", "pointed", "spherical", "hexagonal"}
SIZE_WORDS = {"small", "large", "short", "long", "tall", "big", "tiny", "wide", "narrow", "thin", "thick", "huge", "little"}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")


def load_json(path):
    raw = open(path, "rb").read()
    if orjson is not None:
        try:
            return orjson.loads(raw)
        except orjson.JSONDecodeError:                # SA-FARI carries NaN, which strict parsers refuse
            pass
    return json.loads(raw)


def tokens(phrase):
    return TOKEN_RE.findall(phrase.lower())


class Lexicon:
    """The modifier words: colours plus the words Meta's own Attributes subset opens its phrases with."""

    def __init__(self, learned=()):
        self.attr = set(COLOURS) | {"dark", "light", "bright"} | set(learned)
        self.cache = {}                                # phrase -> features; the same phrase is asked thousands of times

    @classmethod
    def learn(cls, gold_attribute_phrases):
        first = defaultdict(set)
        for p in set(gold_attribute_phrases):
            t = tokens(p)
            if len(t) >= 2:
                first[t[0]].add(p)
        return cls(w for w, s in first.items() if len(s) >= ATTR_MIN_PHRASES and w not in ARTICLES)


def features(phrase, lex):
    f = lex.cache.get(phrase)
    if f is None:
        f = lex.cache[phrase] = _features(phrase, lex)
    return f


def _features(phrase, lex):
    t = tokens(phrase)
    art = bool(t) and t[0] in ARTICLES
    c = t[1:] if art else t
    f = {"words": len(t), "article": art, "content": len(c), "colour": any(w in COLOURS for w in t), "relational": any(w in RELATIONAL for w in t),
         "positional": any(w in POSITION for w in t), "of": "of" in t, "possessive": any(w.endswith("'s") for w in t),
         "person": any(w in PERSON_WORDS or re.sub(r"'s$", "", w) in PERSON_WORDS for w in t), "body": any(w in BODY_WORDS for w in t)}
    if f["relational"]:
        form = "relational"
    elif f["positional"]:
        form = "positional"
    elif f["of"]:
        form = "of-phrase"
    elif len(c) == 1:
        form = "plain noun"
    elif len(c) == 2:
        form = "modifier + noun" if c[0] in lex.attr else "other two words"
    elif len(c) == 3 and c[0] in lex.attr and c[1] in lex.attr:
        form = "two modifiers + noun"
    else:
        form = "other three or more"
    f["form"] = form
    f["exact_person"] = c in (["person"], ["people"])
    return f


def wilson(k, n, z=1.96):
    """95% Wilson interval for a share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(centre - half, 4), round(centre + half, 4))


def share(k, n):
    """A count with its denominator; the 95% Wilson interval is left to `wilson(k, n)` so the json stays small."""
    return {"k": int(k), "n": int(n), "share": round(k / n, 4) if n else None}


def mantel_haenszel(tables):
    """Pooled odds ratio and a 95% interval (Robins, Breslow, Greenland) over 2x2 tables (a, b, c, d) = (desc neg, desc pos, plain neg, plain pos)."""
    num = den = 0.0
    sp = sq = sps = 0.0                                # sums for the variance terms
    for a, b, c, d in tables:
        n = a + b + c + d
        if n == 0 or min(a + b, c + d) == 0:
            continue
        R, S = a * d / n, b * c / n
        P, Q = (a + d) / n, (b + c) / n
        num += R
        den += S
        sp += P * R
        sq += Q * S
        sps += P * S + Q * R
    if den == 0 or num == 0:
        return None
    or_ = num / den
    var = sp / (2 * num * num) + sps / (2 * num * den) + sq / (2 * den * den)
    se = math.sqrt(var)
    return {"odds_ratio": round(or_, 3), "ci95": (round(math.exp(math.log(or_) - 1.96 * se), 3), round(math.exp(math.log(or_) + 1.96 * se), 3))}


def woolf(a, b, c, d):
    """Odds ratio with a 95% interval for one 2x2 table, 0.5 added to each cell. The interval treats pairs as independent, which they are not (a phrase repeats)."""
    a, b, c, d = a + .5, b + .5, c + .5, d + .5
    r = a * d / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return {"odds_ratio": round(r, 3), "ci95_pairs": (round(math.exp(math.log(r) - 1.96 * se), 3), round(math.exp(math.log(r) + 1.96 * se), 3))}


#: bootstrap resamples and seed for the interval that resamples distinct phrases. Provenance: reasoned (enough for a 95% percentile interval to settle to a few thousandths); fixed seed
BOOTSTRAP_B, BOOTSTRAP_SEED = 300, 0


def cluster_bootstrap_or(strata):
    """Pooled Mantel-Haenszel odds ratio, resampling DISTINCT PHRASES with replacement inside each stratum, so a phrase asked in thousands of pairs counts once.
    strata: list of (descriptive[bool per phrase], negative_pairs[int per phrase], positive_pairs[int per phrase])."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = []
    for _ in range(BOOTSTRAP_B):
        tabs = []
        for desc, neg, pos in strata:
            i = rng.integers(0, len(desc), len(desc))
            d, n_, p_ = desc[i], neg[i], pos[i]
            tabs.append((int(n_[d].sum()), int(p_[d].sum()), int(n_[~d].sum()), int(p_[~d].sum())))
        r = mantel_haenszel(tabs)
        if r:
            vals.append(r["odds_ratio"])
    if not vals:
        return None
    return {"ci95_resampling_distinct_phrases": (round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)), "resamples": len(vals), "seed": BOOTSTRAP_SEED}


def pooled(tables, clusters):
    r = mantel_haenszel(tables)
    if r:
        r = dict(r) | (cluster_bootstrap_or(clusters) or {})
    return r


def phrase_counts(phrases, positive, lex):
    """Per distinct phrase: descriptive?, negative pairs, positive pairs."""
    per = defaultdict(lambda: [0, 0])
    for p, ps in zip(phrases, positive):
        per[p][1 if ps else 0] += 1
    keys = list(per)
    desc = np.array([features(k, lex)["form"] in DESCRIPTIVE for k in keys], dtype=bool)
    return desc, np.array([per[k][0] for k in keys]), np.array([per[k][1] for k in keys])


def quantiles(x, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95)):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return None
    return {f"p{int(q * 100):02d}": float(np.quantile(x, q)) for q in qs} | {"min": float(x.min()), "max": float(x.max()), "n": int(x.size)}


def small_shares(share_arr):
    s = np.asarray(share_arr, dtype=float)
    out = {"n": int(s.size)}
    for t in SIZE_THRESHOLDS:
        out[f"<= {t:g}"] = share(int((s <= t).sum()), s.size)
    return out


# ---- the shape of a set of pairs ------------------------------------------------------------------------------------------

def shape_table(phrases, positive, exhaustive, lex, examples=0):
    """Phrase-shape statistics of one set of pairs: lengths, flags, forms, and the negative rate by form and flag."""
    n = len(phrases)
    feats = [features(p, lex) for p in phrases]
    cache = {p: features(p, lex) for p in set(phrases)}
    pos = np.asarray(positive, dtype=bool)
    words = np.array([f["words"] for f in feats])
    out: dict = {"pairs": n, "distinct_phrases": len(cache), "positive_pairs": int(pos.sum()), "negative_pairs": int((~pos).sum())}
    wb = {"1": words == 1, "2": words == 2, "3": words == 3, "4": words == 4, "5-6": (words >= 5) & (words <= 6), "7+": words >= 7}
    out["words_share_of_pairs"] = {k: round(float(v.mean()), 4) for k, v in wb.items()}
    dist = {p: f["words"] for p, f in cache.items()}
    dv = np.array(list(dist.values()))
    out["words_share_of_distinct"] = {"1": round(float((dv == 1).mean()), 4), "2": round(float((dv == 2).mean()), 4), "3": round(float((dv == 3).mean()), 4),
                                      "4": round(float((dv == 4).mean()), 4), "5-6": round(float(((dv >= 5) & (dv <= 6)).mean()), 4),
                                      "7+": round(float((dv >= 7).mean()), 4)}
    for flag in ("article", "colour", "relational", "positional", "of", "possessive", "person", "body", "exact_person"):
        arr = np.array([f[flag] for f in feats])
        out.setdefault("flag_share_of_pairs", {})[flag] = round(float(arr.mean()), 4)
        if flag in ("article", "colour", "relational", "positional"):          # the flags the question names; the others only add size
            out.setdefault("flag_negative_rate", {})[flag] = {"with": share(int((arr & ~pos).sum()), int(arr.sum())), "without": share(int((~arr & ~pos).sum()), int((~arr).sum()))}
    forms = np.array([f["form"] for f in feats])
    out["form_share_of_pairs"] = {k: round(float((forms == k).mean()), 4) for k in FORMS}
    out["form_negative_rate"] = {k: share(int(((forms == k) & ~pos).sum()), int((forms == k).sum())) for k in FORMS}
    desc = np.isin(forms, list(DESCRIPTIVE))
    out["descriptive_share_of_pairs"] = round(float(desc.mean()), 4)
    out["descriptive_negative_rate"] = {"descriptive": share(int((desc & ~pos).sum()), int(desc.sum())), "plain": share(int((~desc & ~pos).sum()), int((~desc).sum()))}
    tab = [int((desc & ~pos).sum()), int((desc & pos).sum()), int((~desc & ~pos).sum()), int((~desc & pos).sum())]
    out["_desc_table"] = tab
    out["descriptive_odds_ratio"] = woolf(*tab)
    out["_clusters"] = phrase_counts(phrases, positive, lex)
    # a phrase seen in several pairs: negative in all of them, positive in all, or mixed
    per = defaultdict(lambda: [0, 0])                   # per phrase: [negative pairs, positive pairs]
    for p, ps in zip(phrases, pos):
        per[p][1 if ps else 0] += 1
    kinds = Counter("all negative" if v[1] == 0 else ("all positive" if v[0] == 0 else "mixed") for v in per.values())
    out["distinct_phrase_outcomes"] = {k: int(kinds.get(k, 0)) for k in ("all negative", "all positive", "mixed")}
    fm = {}
    for p, v in per.items():
        fm.setdefault(cache[p]["form"], Counter())["all negative" if v[1] == 0 else ("all positive" if v[0] == 0 else "mixed")] += 1
    out["distinct_phrase_all_negative_by_form"] = {k: share(fm.get(k, Counter())["all negative"], sum(fm.get(k, Counter()).values())) for k in FORMS}
    if exhaustive is not None:
        ex = np.asarray(exhaustive, dtype=bool)
        out["non_exhaustive_positive_by_form"] = {k: share(int(((forms == k) & pos & ~ex).sum()), int(((forms == k) & pos).sum())) for k in FORMS}
    if examples:
        out["_examples"] = {k: sorted({p for p, f in cache.items() if f["form"] == k})[:examples] for k in FORMS}
    return out


def attribute_breakdown(phrases):
    """Gold Attributes: what the first word of an attribute phrase is (colour, material, shape, size or other)."""
    c = Counter()
    n_distinct = 0
    for p in set(phrases):
        t = tokens(p)
        if len(t) < 2:
            continue
        n_distinct += 1
        w = t[0]
        c["colour" if w in COLOURS else "material" if w in MATERIAL_WORDS else "shape" if w in SHAPE_WORDS else "size" if w in SIZE_WORDS else "other"] += 1
    first = Counter(tokens(p)[0] for p in set(phrases) if len(tokens(p)) >= 2)
    top = sum(v for _, v in first.most_common(10))
    return {"distinct_phrases": n_distinct, "first_word_class": {k: share(v, n_distinct) for k, v in c.items()},
            "ten_commonest_first_words_cover": share(top, n_distinct)}


def instance_stats(counts):
    c = np.asarray(counts, dtype=int)
    if c.size == 0:
        return None
    out = {"positive_pairs": int(c.size), "median": round(float(np.median(c)), 1), "p90": round(float(np.quantile(c, 0.9)), 1), "max": int(c.max()), "mean": round(float(c.mean()), 2)}
    for m in COUNT_MARKS:
        out[f">= {m}"] = share(int((c >= m).sum()), c.size)
    return out


def people_table(phrases, positive, ninst, lex):
    """Person phrases: how many pairs, how many are negative, and instances per positive pair."""
    feats = [features(p, lex) for p in phrases]
    pos = np.asarray(positive, dtype=bool)
    ninst = np.asarray(ninst, dtype=int)
    sel = {"exact person/people": np.array([f["exact_person"] for f in feats]),
           "any person word": np.array([f["person"] for f in feats]),
           "person word and a body word": np.array([f["person"] and f["body"] for f in feats]),
           "body word without a person word": np.array([f["body"] and not f["person"] for f in feats]),
           "possessive": np.array([f["possessive"] for f in feats])}
    out = {}
    for k, m in sel.items():
        out[k] = {"pairs": int(m.sum()), "negative": share(int((m & ~pos).sum()), int(m.sum()))}
        if k in ("exact person/people", "any person word"):
            out[k]["instances_per_positive_pair"] = instance_stats(ninst[m & pos])
    forms = np.array([f["form"] for f in feats])
    pm = sel["any person word"]
    out["person_pairs_by_form"] = {f: share(int((pm & (forms == f) & ~pos).sum()), int((pm & (forms == f)).sum())) for f in FORMS}   # share = negatives
    return out


def has_prop(toks, entries):
    return any(set(e) <= toks for e in entries)


def prop_table(ph_ann, share_arr, pair_phrases=None, pair_positive=None):
    """For the owner's props: annotation counts and box-share quantiles (ph_ann: phrase per annotation), and how often the pair is asked and is a negative."""
    out = {}
    toks = [set(tokens(p)) for p in ph_ann]
    s = np.asarray(share_arr, dtype=float)
    ptoks = [set(tokens(p)) for p in pair_phrases] if pair_phrases is not None else None
    pos = np.asarray(pair_positive, dtype=bool) if pair_positive is not None else None
    for name, entries in PROPS.items():
        m = np.array([has_prop(t, entries) for t in toks], dtype=bool)
        out[name] = {"annotations": int(m.sum()), "box_share": quantiles(s[m]) if m.any() else None,
                     "side_px_at_1008_median": round(math.sqrt(float(np.median(s[m]))) * CANVAS, 1) if m.any() else None}
        if ptoks is not None and pos is not None:
            pm = np.array([has_prop(t, entries) for t in ptoks], dtype=bool)
            out[name]["pairs_asked"] = share(int((pm & ~pos).sum()), int(pm.sum()))          # the share is the negatives
    return out


def per_phrase_size(ph_ann, share_arr):
    """Phrase-level: how many distinct phrases (with enough annotations) have a median box share under each threshold."""
    by = defaultdict(list)
    for p, s in zip(ph_ann, share_arr):
        by[p.lower()].append(s)
    meds = np.array([np.median(v) for v in by.values() if len(v) >= MIN_ANN_PER_PHRASE])
    out = {"phrases_with_enough_annotations": int(meds.size)}
    for t in SIZE_THRESHOLDS:
        out[f"median <= {t:g}"] = share(int((meds <= t).sum()), meds.size)
    return out


# ---- image datasets (Gold, Silver) ------------------------------------------------------------------------------------------

def image_set(d, lex, examples):
    ids = [i["id"] for i in d["images"]]
    phrases = [i["text_input"] for i in d["images"]]
    exh = [bool(i.get("is_instance_exhaustive", 1)) for i in d["images"]]
    ninst = Counter(a["image_id"] for a in d["annotations"])
    positive = [i in ninst for i in ids]
    counts = [ninst.get(i, 0) for i in ids]
    img = {i["id"]: i["text_input"] for i in d["images"]}
    ann_ph = [img[a["image_id"]] for a in d["annotations"]]
    box = np.array([a["bbox"][2] * a["bbox"][3] for a in d["annotations"]], dtype=float)
    mask = np.array([a["area"] for a in d["annotations"]], dtype=float)
    out = shape_table(phrases, positive, exh, lex, examples)
    out["people"] = people_table(phrases, positive, counts, lex)
    out["instances_per_positive_pair"] = instance_stats([c for c in counts if c])
    out["annotations"] = len(d["annotations"])
    out["box_share"] = quantiles(box)
    out["mask_share"] = quantiles(mask)
    out["box_share_thresholds"] = small_shares(box)
    out["props"] = prop_table(ann_ph, box, phrases, positive)
    exact = np.array([features(p, lex)["exact_person"] for p in ann_ph], dtype=bool)
    out["exact_person_box_share"] = quantiles(box[exact]) if exact.any() else None
    out["per_phrase_size"] = per_phrase_size(ann_ph, box)
    out["iscrowd_share"] = share(int(sum(a.get("iscrowd", 0) for a in d["annotations"])), len(d["annotations"]))
    return out, positive, ids, phrases


GOLD_SUBSETS = ("metaclip", "sa1b", "attributes", "crowded", "wiki_common", "fg_food", "fg_sports_equipment")


def run_gold(root, lex_holder, examples):
    out = {"subsets": {}, "annotator_agreement": {}, "controls": {}}
    # the lexicon comes from the Attributes subset itself, so read it first
    first = load_json(os.path.join(root, "gold_attributes_merged_a_release_test.json"))
    lex = lex_holder["lex"] = Lexicon.learn([i["text_input"] for i in first["images"]])
    out["modifier_lexicon_size"] = len(lex.attr)
    tables, clusters = [], []
    for sub in GOLD_SUBSETS:
        D = {ab: (first if (sub == "attributes" and ab == "a") else load_json(os.path.join(root, f"gold_{sub}_merged_{ab}_release_test.json"))) for ab in "abc"}
        res, positive, ids, phrases = image_set(D["a"], lex, examples)
        out["subsets"][sub] = res
        tables.append(res.pop("_desc_table"))
        clusters.append(res.pop("_clusters"))
        # three annotators over the same pairs
        same = all([i["id"] for i in D[ab]["images"]] == ids and [i["text_input"] for i in D[ab]["images"]] == phrases for ab in "bc")
        pos = {ab: {a["image_id"] for a in D[ab]["annotations"]} for ab in "abc"}
        k = np.array([sum(i in pos[ab] for ab in "abc") for i in ids])
        forms = np.array([features(p, lex)["form"] for p in phrases])
        agree = {"same_pairs_in_a_b_c": bool(same), "negative_rate_by_annotator": {ab: round(1 - len(pos[ab]) / len(ids), 4) for ab in "abc"},
                 "positive_in_k_of_3": {str(i): int((k == i).sum()) for i in range(4)},
                 "contested_share_of_pairs": share(int(((k == 1) | (k == 2)).sum()), len(ids)),
                 "contested_share_of_pairs_positive_to_any": share(int(((k == 1) | (k == 2)).sum()), int((k >= 1).sum())),
                 "contested_by_form": {f: share(int((((k == 1) | (k == 2)) & (forms == f)).sum()), int(((k >= 1) & (forms == f)).sum())) for f in FORMS}}
        out["annotator_agreement"][sub] = agree
        # control: the pair count and mask count against Meta's README
        n_images, n_masks = README_GOLD_STATS[sub]
        masks = {ab: len(D[ab]["annotations"]) for ab in "abc"}
        out["controls"][sub] = {"pairs": len(ids), "readme_pairs": n_images, "pairs_match": len(ids) == n_images, "masks_by_annotator": masks,
                                "readme_masks": n_masks, "readme_masks_match_annotator": [ab for ab in "abc" if masks[ab] == n_masks]}
        del D
    out["descriptive_vs_plain_negative_mantel_haenszel"] = pooled(tables, clusters)
    out["attributes_subset"] = attribute_breakdown([i["text_input"] for i in first["images"]])
    return out


def run_silver(root, lex, examples):
    out = {"sources": {}}
    tables, clusters = [], []
    for f in sorted(glob.glob(os.path.join(root, "silver_*.json"))):
        key = re.sub(r"^silver_|(_merged)?_test\.json$", "", os.path.basename(f))
        d = load_json(f)
        res, *_ = image_set(d, lex, examples)
        tables.append(res.pop("_desc_table"))
        clusters.append(res.pop("_clusters"))
        out["sources"][key] = res
        del d
    out["descriptive_vs_plain_negative_mantel_haenszel"] = pooled(tables, clusters)
    return out


# ---- video datasets (VEval, SA-FARI) -------------------------------------------------------------------------------------------

def runs_of(present):
    """Number of contiguous runs of True in a boolean list."""
    return int(sum(1 for i, v in enumerate(present) if v and (i == 0 or not present[i - 1])))


def video_set(d, lex, examples, fps):
    pairs = d["video_np_pairs"]
    phrases = [p["noun_phrase"] for p in pairs]
    positive = [p["num_masklets"] > 0 for p in pairs]
    counts = [p["num_masklets"] for p in pairs]
    out = shape_table(phrases, positive, None, lex, examples)
    out["people"] = people_table(phrases, positive, counts, lex)
    out["masklets_per_positive_pair"] = instance_stats([c for c in counts if c])
    present_frames, fraction, med_share, min_share, runs, frame_shares = [], [], [], [], [], []
    ph_for_share = []                                   # phrase of each masklet that has at least one present frame, parallel to med_share
    for a in d["annotations"]:
        area_px = float(a["width"]) * float(a["height"])
        pr = [b is not None for b in a["bboxes"]]
        sh = [(b[2] * b[3]) / area_px for b in a["bboxes"] if b is not None]
        n = len(a["bboxes"])
        present_frames.append(sum(pr))
        fraction.append(sum(pr) / n if n else 0.0)
        runs.append(runs_of(pr))
        if sh:
            med_share.append(float(np.median(sh)))
            min_share.append(float(min(sh)))
            frame_shares.extend(sh)
            ph_for_share.append(a["noun_phrase"])
    out["masklets"] = len(d["annotations"])
    out["videos"] = len(d["videos"])
    out["masklet_frames_present"] = quantiles(present_frames, (0.05, 0.25, 0.5, 0.75, 0.95))
    out["masklet_fraction_of_video_present"] = quantiles(fraction, (0.05, 0.25, 0.5, 0.75, 0.95))
    out["masklet_present_seconds"] = quantiles([f / fps for f in present_frames], (0.05, 0.25, 0.5, 0.75, 0.95))
    out["masklet_runs"] = {"one_run": share(sum(1 for r in runs if r == 1), len(runs)), "leaves_and_returns": share(sum(1 for r in runs if r > 1), len(runs)),
                           "max_runs": int(max(runs)) if runs else 0}
    out["frame_box_share"] = quantiles(frame_shares)
    out["frame_box_share_thresholds"] = small_shares(frame_shares)
    out["masklet_median_box_share"] = quantiles(med_share)
    out["masklet_min_box_share"] = quantiles(min_share)
    out["masklet_median_box_share_thresholds"] = small_shares(med_share)
    out["fps_of_frames"] = fps
    out["props"] = prop_table(ph_for_share, med_share, phrases, positive)
    exact = np.array([features(p, lex)["exact_person"] for p in ph_for_share], dtype=bool)
    out["exact_person_masklet_median_box_share"] = quantiles(np.asarray(med_share)[exact]) if exact.any() else None
    lengths = [v["length"] for v in d["videos"]]
    out["video_length_frames"] = quantiles(lengths, (0.05, 0.25, 0.5, 0.75, 0.95))
    out["video_length_seconds"] = quantiles([n / fps for n in lengths], (0.05, 0.25, 0.5, 0.75, 0.95))
    # pairs per video, and masklets per video-phrase pair, for the question of several people
    ppv = Counter(p["video_id"] for p in pairs)
    pos_ppv = Counter(p["video_id"] for p in pairs if p["num_masklets"] > 0)
    out["pairs_per_video"] = {"median": float(np.median(list(ppv.values()))), "max": int(max(ppv.values()))}
    out["positive_pairs_per_video_with_any"] = {"median": float(np.median(list(pos_ppv.values()))) if pos_ppv else None, "max": int(max(pos_ppv.values())) if pos_ppv else 0}
    return out


def video_people_selection(d, lex):
    """Question 5: how a specific person among several is addressed. Everything an annotation carries about identity is its phrase."""
    feats = {}
    pf = lambda p: feats.setdefault(p, features(p, lex))
    by_video = defaultdict(list)
    for a in d["annotations"]:
        by_video[a["video_id"]].append(a["noun_phrase"])
    person_videos = {v: [p for p in ps if pf(p)["person"]] for v, ps in by_video.items()}
    multi = {v: ps for v, ps in person_videos.items() if len(ps) >= 2}
    same_phrase = sum(1 for ps in multi.values() if len(set(ps)) == 1)
    forms = Counter()
    for ps in multi.values():
        for p in set(ps):
            forms[pf(p)["form"]] += 1
    n_pos = sum(1 for p in d["video_np_pairs"] if p["num_masklets"] > 0 and pf(p["noun_phrase"])["person"])
    single = sum(1 for p in d["video_np_pairs"] if p["num_masklets"] == 1 and pf(p["noun_phrase"])["person"])
    one_desc = sum(1 for p in d["video_np_pairs"] if p["num_masklets"] == 1 and pf(p["noun_phrase"])["person"] and pf(p["noun_phrase"])["form"] in DESCRIPTIVE)
    many_desc = sum(1 for p in d["video_np_pairs"] if p["num_masklets"] >= 2 and pf(p["noun_phrase"])["person"] and pf(p["noun_phrase"])["form"] in DESCRIPTIVE)
    # the file's own consistency: a pair's num_masklets is the number of annotation rows for (video, phrase), and each row repeats the pair's phrase
    rows = Counter((a["video_id"], a["category_id"]) for a in d["annotations"])
    cat = {c["id"]: c["name"] for c in d["categories"]}
    pairs_ok = sum(1 for p in d["video_np_pairs"] if p["num_masklets"] == rows.get((p["video_id"], p["category_id"]), 0))
    text_ok = sum(1 for a in d["annotations"] if a["noun_phrase"] == cat.get(a["category_id"]))
    return {"consistency": {"pairs_whose_num_masklets_equals_annotation_rows": share(pairs_ok, len(d["video_np_pairs"])),
                            "annotation_phrase_equals_its_category_name": share(text_ok, len(d["annotations"])),
                            "annotation_fields": sorted(d["annotations"][0]) if d["annotations"] else []},
            "positive_person_pairs_with_one_masklet_descriptive_form": share(one_desc, single),
            "positive_person_pairs_with_several_masklets_descriptive_form": share(many_desc, n_pos - single),
            "videos_with_a_person_masklet": sum(1 for ps in person_videos.values() if ps),
            "videos_with_two_or_more_person_masklets": len(multi),
            "of_those_all_masklets_share_one_phrase": share(same_phrase, len(multi)),
            "forms_of_distinct_person_phrases_in_those_videos": {k: int(forms.get(k, 0)) for k in FORMS},
            "positive_person_pairs": n_pos, "of_those_exactly_one_masklet": share(single, n_pos)}


def run_video(root_files, lex, examples, fps_key):
    out = {"splits": {}}
    tables, clusters = [], []
    for name, path in root_files:
        d = load_json(path)
        fps = VEVAL_FPS[fps_key(name)]
        res = video_set(d, lex, examples, fps)
        tables.append(res.pop("_desc_table"))
        clusters.append(res.pop("_clusters"))
        res["person_selection"] = video_people_selection(d, lex)
        res["info"] = d["info"].get("description")
        if len(d.get("categories", [])) and "phrase_map" not in out:
            out["phrase_map"] = phrase_map_controls(d["categories"], lex)
        out["splits"][name] = res
        del d
    out["descriptive_vs_plain_negative_mantel_haenszel"] = pooled(tables, clusters)
    return out


def phrase_map_controls(categories, lex):
    """The global phrase map every VEval and SA-FARI file carries; the control against mryolk's count of the same list."""
    names = [c["name"] for c in categories]
    w = np.array([len(n.split()) for n in names])          # whitespace words, as mryolk counted
    return {"phrases": len(names), "words_share": {"1": round(float((w == 1).mean()), 4), "2": round(float((w == 2).mean()), 4),
                                                   "3": round(float((w == 3).mean()), 4), "4": round(float((w == 4).mean()), 4),
                                                   "5-6": round(float(((w >= 5) & (w <= 6)).mean()), 4), "7+": round(float((w >= 7).mean()), 4)},
            "starts_with_article": round(float(np.mean([bool(tokens(n)) and tokens(n)[0] in ARTICLES for n in names])), 4),
            "mryolk_phrases": MRYOLK_MAP["phrases"], "matches_mryolk_count": len(names) == MRYOLK_MAP["phrases"],
            "phrases_with_the_props_words": {k: sum(has_prop(set(tokens(n)), e) for n in names) for k, e in PROPS.items()}}


# ---- command line ---------------------------------------------------------------------------------------------------------------

def veval_files(root):
    return [(os.path.basename(f)[len("saco_veval_"):-len(".json")], f) for f in sorted(glob.glob(os.path.join(root, "annotation", "saco_veval_*.json")))]


def safari_files(root, skip_train):
    names = ["sa_fari_test.json"] + ([] if skip_train else ["sa_fari_train.json"])
    return [(n[len("sa_fari_"):-len(".json")], os.path.join(root, "annotation", n)) for n in names if os.path.exists(os.path.join(root, "annotation", n))]


def cmd_run(a):
    out = {"tool": "bench/sam3_dataset_phrases.py", "note": "aggregates only; the datasets are gated and are not in the repository"}
    holder = {}
    if a.gold:
        print("gold ...", file=sys.stderr, flush=True)
        out["gold"] = run_gold(a.gold, holder, a.examples)
    lex = holder.get("lex") or Lexicon()
    if a.silver:
        print("silver ...", file=sys.stderr, flush=True)
        out["silver"] = run_silver(a.silver, lex, a.examples)
    if a.veval:
        print("veval ...", file=sys.stderr, flush=True)
        out["veval"] = run_video(veval_files(a.veval), lex, a.examples, lambda n: n.split("_")[0])
    if a.safari:
        print("safari ...", file=sys.stderr, flush=True)
        out["safari"] = run_video(safari_files(a.safari, a.skip_train), lex, a.examples, lambda n: "sa_fari")
    ex = {}

    def strip(o, path=""):
        if isinstance(o, dict):
            for k in [k for k in o if k == "_examples"]:
                ex[path] = o.pop(k)
            for k, v in o.items():
                strip(v, f"{path}/{k}")
    strip(out)
    json.dump(out, open(a.out, "w"), separators=(",", ":"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes)", file=sys.stderr)
    if a.examples:
        for k, v in ex.items():
            print(k, v)
    cmd_render(argparse.Namespace(json=a.out))


def pct(x):
    return "" if x is None else f"{100 * x:.1f}%"


def sh(s):
    return "" if not s or s.get("share") is None else f"{100 * s['share']:.1f}% ({s['k']}/{s['n']})"


def render_set(title, S, hdr_name):
    print(f"\n#### {title}\n")
    print(f"| {hdr_name} | pairs | distinct | negative | 1 word | 2 | 3 | 4 | 5-6 | 7+ | article | colour | relational | positional | descriptive |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        w = s["words_share_of_pairs"]
        f = s["flag_share_of_pairs"]
        print(f"| {k} | {s['pairs']} | {s['distinct_phrases']} | {pct(s['negative_pairs'] / s['pairs'])} | " + " | ".join(pct(w[b]) for b in ("1", "2", "3", "4", "5-6", "7+")) +
              f" | {pct(f['article'])} | {pct(f['colour'])} | {pct(f['relational'])} | {pct(f['positional'])} | {pct(s['descriptive_share_of_pairs'])} |")


def fmt_pooled(r):
    if not r:
        return "not computable"
    return (f"odds ratio {r['odds_ratio']}, 95% interval {tuple(r['ci95'])} treating pairs as independent, {tuple(r.get('ci95_resampling_distinct_phrases', ()))} "
            f"resampling distinct phrases ({r.get('resamples', 0)} resamples, seed {r.get('seed')})")


def render_forms(title, S, hdr_name):
    print(f"\n#### {title}\n")
    print(f"| {hdr_name} | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | distinct phrases seen in both positive and negative pairs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        f = s["form_share_of_pairs"]
        o = s["distinct_phrase_outcomes"]
        print(f"| {k} | " + " | ".join(pct(f[x]) for x in FORMS) + f" | {sh(share(o['mixed'], s['distinct_phrases']))} |")


def render_negatives(title, S, hdr_name):
    print(f"\n#### {title}\n")
    print(f"| {hdr_name} | plain noun | other two words | modifier + noun | two modifiers + noun | of-phrase | relational | positional | other 3+ | descriptive | plain | odds ratio, descriptive vs plain (pairs treated as independent) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        r = s["form_negative_rate"]
        d = s["descriptive_negative_rate"]
        o = s["descriptive_odds_ratio"]
        print(f"| {k} | " + " | ".join(sh(r[f]) for f in FORMS) + f" | {sh(d['descriptive'])} | {sh(d['plain'])} | {o['odds_ratio']} {tuple(o['ci95_pairs'])} |")


def cmd_render(a):
    D = json.load(open(a.json))
    if "gold" in D:
        G = D["gold"]
        print(f"### Gold ({G['modifier_lexicon_size']} modifier words learned from the Attributes subset)")
        render_set("Gold: phrase shapes, annotator a", G["subsets"], "subset")
        render_forms("Gold: share of pairs by phrase form", G["subsets"], "subset")
        render_negatives("Gold: share of pairs that are negative, by phrase form", G["subsets"], "subset")
        print("\nPooled across the seven subsets (Mantel-Haenszel, descriptive vs plain, odds of being a negative): " + fmt_pooled(G["descriptive_vs_plain_negative_mantel_haenszel"]))
        ab = G["attributes_subset"]
        print(f"\nGold Attributes, how an attribute phrase begins (distinct two-or-more-word phrases {ab['distinct_phrases']}): " +
              ", ".join(f"{k} {sh(v)}" for k, v in ab["first_word_class"].items()) + f"; the ten commonest first words cover {sh(ab['ten_commonest_first_words_cover'])}")
        print("\n#### Gold: three annotators over the same pairs\n\n| subset | negative a / b / c | positive in 0 / 1 / 2 / 3 of 3 | contested (1 or 2 of 3) of pairs any annotator marked | by form: plain noun / modifier + noun / relational / of-phrase |\n|---|---|---|---|---|")
        for k, v in G["annotator_agreement"].items():
            n = v["negative_rate_by_annotator"]
            c = v["contested_by_form"]
            kk = v["positive_in_k_of_3"]
            print(f"| {k} | {pct(n['a'])} / {pct(n['b'])} / {pct(n['c'])} | {kk['0']} / {kk['1']} / {kk['2']} / {kk['3']} | {sh(v['contested_share_of_pairs_positive_to_any'])} | "
                  f"{sh(c['plain noun'])} / {sh(c['modifier + noun'])} / {sh(c['relational'])} / {sh(c['of-phrase'])} |")
        print("\n#### Gold controls: counts against Meta's README table\n\n| subset | pairs | README pairs | masks a / b / c | README masks | matches |\n|---|---|---|---|---|---|")
        for k, v in G["controls"].items():
            m = v["masks_by_annotator"]
            print(f"| {k} | {v['pairs']} | {v['readme_pairs']} | {m['a']} / {m['b']} / {m['c']} | {v['readme_masks']} | {v['readme_masks_match_annotator']} |")
        render_sizes("Gold", G["subsets"], "subset")
        render_instances("Gold", G["subsets"], "subset")
        render_people("Gold", G["subsets"], "subset")
        render_person_sizes("Gold", G["subsets"], "subset")
    if "silver" in D:
        S = D["silver"]
        render_set("Silver: phrase shapes", S["sources"], "source")
        render_forms("Silver: share of pairs by phrase form", S["sources"], "source")
        render_negatives("Silver: share of pairs that are negative, by phrase form", S["sources"], "source")
        print("\nPooled across sources (Mantel-Haenszel): " + fmt_pooled(S["descriptive_vs_plain_negative_mantel_haenszel"]))
        render_sizes("Silver", S["sources"], "source")
        render_instances("Silver", S["sources"], "source")
        render_people("Silver", S["sources"], "source")
        render_person_sizes("Silver", S["sources"], "source")
    for key, title in (("veval", "VEval"), ("safari", "SA-FARI")):
        if key in D:
            V = D[key]
            render_set(f"{title}: phrase shapes (asked pairs)", V["splits"], "split")
            render_forms(f"{title}: share of pairs by phrase form", V["splits"], "split")
            render_negatives(f"{title}: share of pairs that are negative, by phrase form", V["splits"], "split")
            print("\nPooled (Mantel-Haenszel): " + fmt_pooled(V["descriptive_vs_plain_negative_mantel_haenszel"]))
            pm = V.get("phrase_map")
            if pm:
                print(f"\n{title} global phrase map: {pm['phrases']} phrases; words 1/2/3/4/5-6/7+ = " + " / ".join(pct(pm["words_share"][b]) for b in ("1", "2", "3", "4", "5-6", "7+")) +
                      f"; starts with an article {pct(pm['starts_with_article'])}; equals mryolk's count: {pm['matches_mryolk_count']}; phrases holding the props' words: {pm['phrases_with_the_props_words']}")
            render_video(title, V["splits"])
            render_people(title, V["splits"], "split")
            render_person_sizes(title, V["splits"], "split")


def render_sizes(name, S, hdr):
    print(f"\n#### {name}: annotated box area as a share of the image\n")
    print(f"| {hdr} | annotations | p05 | median | p95 | <= 1e-4 (about 10 px square at 1008) | <= 1e-3 (about 32 px) | <= 1e-2 (about 100 px) | phrases whose median is <= 1e-3 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        b = s["box_share"]
        t = s["box_share_thresholds"]
        pp = s["per_phrase_size"]
        print(f"| {k} | {s['annotations']} | {b['p05']:.2g} | {b['p50']:.2g} | {b['p95']:.2g} | {sh(t['<= 0.0001'])} | {sh(t['<= 0.001'])} | {sh(t['<= 0.01'])} | {sh(pp['median <= 0.001'])} |")
    print(f"\n{name}: the owner's props (annotations whose phrase has the words; box share median / equal-area side in px at 1008)")
    for k, s in S.items():
        pr = s.get("props") or {}
        bits = [f"{n}: {v['annotations']} annotations (median {v['box_share']['p50']:.2g}, ~{v['side_px_at_1008_median']} px)" if v["annotations"] else f"{n}: no annotation"
                for n, v in pr.items() if v.get("pairs_asked", {}).get("n")]
        bits = [b + f"; asked in {pr[n]['pairs_asked']['n']} pairs, {pr[n]['pairs_asked']['k']} negative" for b, n in zip(bits, [n for n, v in pr.items() if v.get("pairs_asked", {}).get("n")])]
        if bits:
            print(f"- {k}: " + "; ".join(bits))


def render_instances(name, S, hdr):
    print(f"\n#### {name}: instances annotated per positive pair (any phrase)\n")
    print(f"| {hdr} | positive pairs | median | p90 | max | >= 2 | >= 5 | >= 10 | >= 16 | >= 32 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        i = s["instances_per_positive_pair"]
        print(f"| {k} | {i['positive_pairs']} | {i['median']} | {i['p90']} | {i['max']} | " + " | ".join(sh(i[f">= {m}"]) for m in COUNT_MARKS) + " |")


def render_people(name, S, hdr):
    print(f"\n#### {name}: person phrases\n")
    print(f"| {hdr} | exact person/people pairs | negative | instances per positive pair: median / p90 / max | >= 16 | >= 32 | any person word: pairs, negative | person word and a body word: pairs, negative | body word alone: pairs, negative |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        p = s["people"]
        e = p["exact person/people"]
        i = e["instances_per_positive_pair"] or {}
        a, b, c = p["any person word"], p["person word and a body word"], p["body word without a person word"]
        print(f"| {k} | {e['pairs']} | {sh(e['negative'])} | {i.get('median', '')} / {i.get('p90', '')} / {i.get('max', '')} | {sh(i.get('>= 16'))} | {sh(i.get('>= 32'))} | "
              f"{a['pairs']}, {pct(a['negative']['share'])} | {b['pairs']}, {pct(b['negative']['share'])} | {c['pairs']}, {pct(c['negative']['share'])} |")


def render_person_sizes(name, S, hdr):
    print(f"\n{name}: size of the annotated exact `person`/`people` objects (box share of the image; per masklet median in video)")
    for k, s in S.items():
        q = s.get("exact_person_box_share") or s.get("exact_person_masklet_median_box_share")
        if q:
            print(f"- {k}: n {q['n']}; p05 {q['p05']:.2g}, median {q['p50']:.2g}, p95 {q['p95']:.2g}; median side of the equal-area square at 1008: {math.sqrt(q['p50']) * CANVAS:.0f} px")


def render_video(name, S):
    print(f"\n#### {name}: tracked objects\n")
    print("| split | videos | masklets | present frames p25 / median / p75 | fraction of video present median | present seconds median | leaves and returns | median box share (frame) | frame box <= 1e-3 | masklets per positive pair median / p90 / max |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, s in S.items():
        f = s["masklet_frames_present"]
        fr = s["masklet_fraction_of_video_present"]
        sec = s["masklet_present_seconds"]
        m = s["masklets_per_positive_pair"] or {}
        print(f"| {k} | {s['videos']} | {s['masklets']} | {f['p25']:.0f} / {f['p50']:.0f} / {f['p75']:.0f} | {fr['p50']:.2f} | {sec['p50']:.1f} | {sh(s['masklet_runs']['leaves_and_returns'])} | "
              f"{s['frame_box_share']['p50']:.2g} | {sh(s['frame_box_share_thresholds']['<= 0.001'])} | {m.get('median', '')} / {m.get('p90', '')} / {m.get('max', '')} |")
    print(f"\n{name}: clip lengths")
    for k, s in S.items():
        L, T = s["video_length_frames"], s["video_length_seconds"]
        print(f"- {k}: frames p05 {L['p05']:.0f} / median {L['p50']:.0f} / p95 {L['p95']:.0f}; seconds p05 {T['p05']:.1f} / median {T['p50']:.1f} / p95 {T['p95']:.1f} / max {T['max']:.1f}; frames are {s['fps_of_frames']} a second")
    print(f"\n{name}: several people in one video")
    for k, s in S.items():
        ps = s["person_selection"]
        print(f"- {k}: videos with a person masklet {ps['videos_with_a_person_masklet']}; with two or more {ps['videos_with_two_or_more_person_masklets']}; "
              f"of those, one shared phrase for all {sh(ps['of_those_all_masklets_share_one_phrase'])}; positive person pairs with exactly one masklet {sh(ps['of_those_exactly_one_masklet'])}; "
              f"descriptive form among the one-masklet pairs {sh(ps['positive_person_pairs_with_one_masklet_descriptive_form'])}, among the several-masklet pairs {sh(ps['positive_person_pairs_with_several_masklets_descriptive_form'])}")
    print(f"\n{name}: the file's own consistency (control)")
    for k, s in S.items():
        c = s["person_selection"]["consistency"]
        print(f"- {k}: pairs whose num_masklets equals the annotation rows {sh(c['pairs_whose_num_masklets_equals_annotation_rows'])}; annotation phrase equals its category name {sh(c['annotation_phrase_equals_its_category_name'])}; annotation fields {c['annotation_fields']}")


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gold", help="directory with gold_*_merged_{a,b,c}_release_test.json (facebook/SACo-Gold)")
    r.add_argument("--silver", help="directory with silver_*.json (facebook/SACo-Silver)")
    r.add_argument("--veval", help="directory with annotation/saco_veval_*.json (facebook/SACo-VEval)")
    r.add_argument("--safari", help="directory with annotation/sa_fari_*.json (facebook/SA-FARI)")
    r.add_argument("--skip-train", action="store_true", help="SA-FARI: leave out the 870 MB train annotation")
    r.add_argument("--out", required=True, help="the aggregates json to write")
    r.add_argument("--examples", type=int, default=0, help="print up to N example phrases per form to the terminal (never written to the json)")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("render")
    p.add_argument("--json", required=True)
    p.set_defaults(fn=cmd_render)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
