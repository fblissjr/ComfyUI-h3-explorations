# J-lens variants and extensions: reading notes

Fetched 2026-11-12. Raw fetches, text conversions and figures are in
a session scratch folder (`refs_fetch_variants`, not kept).
Everything below comes from the sources unless it is marked "(from memory, not in the source)" or
"(my inference)". Numbers marked "read from figure" were read off a chart by eye, not from a stated table.

Notation shared by the sources: `h_{l,s}` is the residual at layer l, position s; `W_U` the unembedding;
`W_E` the embedding; the plain J-lens (Gurnee et al. 2026) is

    J_l = E_{prompt, s} [ E_{s <= t < T} [ d h_final,t / d h_l,s ] ]      (J++ post, eq. 1)
    read(h) = top-k of W_U · J_l · h       (neuron-lens writes softmax(W_U · norm(J_l h)), with the final RMSNorm)

"Lens vectors" are the rows of `W_U J_l`, one per vocabulary token (J++ post).

---

## 1. R-lens (camilablank, coauthors agam_bhatia and Neel Nanda; posted 2026-08-05)

URL: https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/ . Lenses released at
https://huggingface.co/camilablank/workspace-lenses .

### Method
- Identical to the J-lens except that the backward pass uses Layer-wise Relevance Propagation (LRP)
  rules, following Relevance Patching (RelP). "What gets transported and averaged is a relevance
  coefficient rather than a raw gradient." It is implemented purely with stop-gradients, so the forward
  pass is unchanged.
- Three rules:
  - **LN-rule**: treat the normalization denominator `sqrt(eps + Var[x])` as a constant (the norm becomes
    linear in the backward pass). Applied to the residual-stream RMSNorms.
  - **Identity-rule**: replace `GELU(x)` with `x * stop_grad(GELU(x)/x)` (same for SiLU), so the
    activation's backward pass is a per-element linear map. For SiLU "the gradient is just sigmoid(z)
    instead of the full SiLU derivative".
  - **Half-rule**: in a gated MLP, split relevance evenly between gate path and up path instead of
    double-counting through the product.
- Not modified: linear layers (LRP 0-rule equals the ordinary gradient), attention, q/k norms.
- MoE models: LRP rules on all routed experts; freeze routing weights (softmax/sigmoid scoring, router
  logits, gate projection); for a shared expert, multiply its output by a constant and sweep a few values
  ("amplifies the shared expert's importance"). DeepSeek-V4-Flash's mHC residual-mixing coefficients are
  frozen too.
- Fit data/size: not stated in the R-lens post. The J++ post says the released R-lenses were "fitted with
  25 prompts"; the neuron-lens design doc says the published R-lens "reads the penultimate layer from 25
  Pile prompts, skipping 4 positions".

### What it needs
Weights, a backward pass per output direction (as the J-lens), stop-gradients patched into norms and MLPs.
Cost: "negligible" over the J-lens (authors' word; no timing given).

### Claimed fix and evidence
Claim: J-lens is fit by backpropagating from the final layer to the readout layer; "errors are likely to
accumulate over the course of layers", so early-layer readouts are noisy ("trash tokens": 锁定, 尷,
`＊＊＊＊`, `......`, euw, tav, zinho). R-lens limits that accumulation.
- **Evals** (adapted from J-lens paper App. A.6): multihop, multilingual, association, typo, poetry;
  metric is per-layer pass@10 (does the intermediate appear in the top 10 at the target position),
  averaged over layers and categories. Items filtered to those the model answers correctly (multihop,
  multilingual).
- **Mean pass@10** (read from figure IMG 3; first-half-of-layers / all layers):
  DeepSeek-V4-Flash R 0.23/0.29 vs J 0.03/0.16; Gemma-3-27B R 0.10/0.21 vs J 0.04/0.13;
  Qwen3.6-27B R 0.12/0.16 vs J 0.09/0.14; Qwen3.5-27B R 0.12/0.16 vs J 0.08/0.14;
  Qwen3.5-9B R 0.09/0.12 vs J 0.07/0.10; Qwen3.5-4B R 0.05/0.08 vs J 0.06/0.09;
  Qwen3.5-122B R 0.02/0.16 vs J 0.01/0.14; Qwen3.6-A3B R 0.09/0.18 vs J 0.13/0.20.
  Text: "no R-lens advantage for both the smallest dense and MoE models"; advantage grows with size.
- **Earlier surfacing (qualitative)**: typo "aganst" -> "against" rank 1 at layer 4 (J-lens never on that
  position); "sushi" -> "Japan" at layer 2 (J-lens layer 14); "Jordan" -> "basketball" at L4 (J-lens L20).
  "Verona" -> "Italy" rank 1 around L5, rank >1000 under J-lens.
- **Probe upper bound**: linear probes over 10 concepts per category; earliest layer with 0.99-1.00 probe
  accuracy taken as a crude bound; R-lens lands closer to it than J-lens (landmarks: R-lens beats it).
- **CKA**: R-space CKA on Qwen3.6-27B shows 2-3 bands vs J-space's 4-5.
- **Appendix, MLP gain** (as J-lens paper Fig. 32): on Qwen3.6-27B, J- and R-lens directions are amplified
  similarly by the next MLP, both more than a standard MLP neuron in later workspace layers.

### WRITE direction
- **Ablation**: 30 multihop questions; project out the R-, J- or logit-lens direction of the intermediate
  token from the activation at the penultimate prompt position, either at the first half of layers or at
  all layers; sample 8 times before and after; accuracy judged by a GPT 5.4 nano autorater; report relative
  accuracy loss, with a random-direction control. Text: on nearly all models (except the two smaller MoEs on
  first-half layers) ablating R-lens directions costs much more accuracy than J- or logit-lens. Read from
  figures IMG 7/8 (all layers): DeepSeek-V4-Flash R ~0.53 vs J ~0.47; Gemma-3-27B R ~0.48 vs J ~0.18;
  Qwen3.6-27B R ~0.43 vs J ~0.37; Qwen3.6-35B-A3B R ~0.58 vs J ~0.59; random controls near 0. Error bars
  overlap R and J on most models.
- No additive steering, no synthesis from words, no round trip.

### Limits / open
- R-lens "is primarily a tool for more faithfully tracking computation over layers"; concepts it finds
  early usually show up later under J-lens anyway.
- Comments (no author replies in the thread):
  - StellaAthena (tuned-lens author): it is hard to know the right metric for "what the model thinks after
    layer K"; many metrics can be gamed by reporting layer K+2's answer; asks how the authors view this,
    since they treat their probe "upper bound" lightly.
  - Anne K. Halsall: on Gemma 2, per-prompt margin gradients (VJPs at a fixed readout position) are
    mutually coherent at early layers (median cosine to the population mean ~0.72-0.80) exactly where
    J-lens is noisiest: "there is a stable direction to find there". Also warns that on Gemma 2 the leading
    component of any `W_U`-derived object is dominated by near-zero-frequency (undertrained) vocab rows;
    asks whether the trash tokens were checked against that.
  - Pranav Viswanath asks for the attention-gain (broadcast) measure as well (J-space paper 4.3.2).
  - Clément Dumas: the identity-rule formula fails to render on LW (the formula is as written above).

---

## 2. J++ lens (Kola Ayonrinde, with Jack Lindsey (Anthropic) named in the body; posted 2026-10-08)

URL: https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/ . Code https://github.com/safety-research/jpp_lens ,
lenses https://huggingface.co/koayon/jpp-lenses , Neuronpedia demo for Qwen3.6-27B.

### Method (eq. 2a-2c of the post)

    J_{i,l} = E_{prompt,s} [ E_{s<=t<T} [ d_LRP h_final,t / d h_l,s ] | h_l,s in C_{i,l} ]    (2a)
    J_l     = sum_{i=1..E} w_{i,l} · J_{i,l}                                               (2b)
    readout_k(h_l,s) = top-k over v in V_sem of ( W_U J_l h_l,s )                          (2c)

Three changes over the J-lens; removing all three recovers eq. 1:
1. **Jacobian Filtering**: k-means partition of each layer's activation space into E clusters; one
   "Expert Jacobian" averaged only over source activations in that cluster; combine with learned weights
   `w_{i,l}` (may be zero or negative). Weights learned on a small dev set of latent-variable extraction
   problems (the only use of labelled data; clusters and Jacobians are fit on unlabelled text). The result
   is one linear map per layer, same shape as the J-lens, so no inference cost.
2. **LRP backward pass**, as R-lens: LN-rule on residual RMSNorms, identity and half rules on gated MLPs,
   plain gradient through attention, Gated DeltaNet and linear layers; MoE routing weights frozen; mHC
   coefficients frozen.
3. **Readout Filtering**: restrict the readout vocabulary to `V_sem` = all tokens except those with no
   letter or digit (6,116 tokens removed for Qwen3.6-27B). Motivated by the transport being homogeneous
   linear, not affine, so it cannot capture input-independent reweighting.
Also: skip the first 16 positions (attention sinks are "one source of biased noise") and the final position.

Why: averaging cancels unbiased noise but not biased noise. Some experts give poor readouts even on their
own clusters; hypothesis: the transport map near those activations is singular or highly curved, so the
Jacobian there is a poor linear approximation.

### What it needs (Table 5, Qwen3.6-27B: 64 blocks, d_model 5120)
- Fit data: WikiText-103 train, 64 sequences x 128 tokens, first 16 and final position dropped: 111
  positions per sequence, 7,104 total.
- Source layers 8,16,...,56 (7 layers); target layer 63.
- E = 8 experts per layer; k-means per layer on 64-dim PCA projections of unit-normalised, centred
  activations, fitted on 1,000 WikiText-103 sequences.
- Weights: Adam, lr 0.05, on a dev set disjoint from the eval set.
- Compute for all 63 layers: ≈9 H200 GPU-hours (7.3 Jacobians, ≈1 clustering, ≈0.5 merge and weights);
  ≈1 h wall-clock on 16 GPUs. "About 7% more compute than fitting a J-Lens on the same data", plus clustering.

### Results
- Readout: recall@10 = target appears in top 10 at any of 7 equally spaced layers, macro-averaged over five
  tasks. Table 1 (recall@10 %, all lenses scored with Readout Filtering):

  | model | Logit | J | R | J++ |
  |---|---|---|---|---|
  | Qwen3.5 9B | 29.7 | 34.2 | 37.5 | 55.0 |
  | Qwen3.6 27B | 31.1 | 35.7 | 37.7 | 55.2 |
  | Gemma 4 31B | 34.8 | 34.9 | 46.4 | 58.0 |
  | Olmo 3 32B | 35.2 | 32.6 | 36.2 | 50.0 |
  | Qwen3.5 122B-A10B | 21.7 | 24.4 | 23.2 | 48.4 |
  | DeepSeek-V4-Flash 284B | 31.2 | 30.5 | 47.4 | 61.4 |

  Median uplift over J +63.5%, over R +42.2%.
- Ablation on Qwen3.6-27B (Table 2/4): full 55.2; without Readout Filtering 52.3 (−2.9pp); without LRP
  48.6 (−6.6); without Jacobian Filtering ("Pooled Jacobian") 43.5 (−11.7, 60% of the gain). E=4 51.5,
  E=16 58.1, E=32 59.9. k-means routing (each item to its own cluster's expert) 44.1; random routing 47.7;
  best single expert per layer 50.2; one layer-8 router for all layers 56.0; pooled with two worst experts
  dropped 46.9; leave-one-task-out dev set 54.4. Token-type filters instead of clusters: semantic sources
  and targets only 47.4; middle-of-word sources only 48.1; end-of-word only 48.6; three-class token router 50.2.
- Per task (Qwen3.6-27B, recall@10, J -> J++): multihop 56.1 -> 67.3; multilingual 47.4 -> 74.6; typo 60.4
  -> 91.7; association 10.8 -> 37.3; poetry 3.7 -> 4.9. Recall@1 macro: J 18.2, R 18.4, J++ 32.4.
- Expert analysis: only 24 of 52 Expert Jacobians beat the Pooled Jacobian; 17 score below 2% per-layer
  recall@10; at layers 8 and 16, 10 of 16 experts score below 2%. Bad experts often fitted mainly on
  punctuation ("formatting experts"), but not exclusively; cluster size does not predict quality.
- Gradient norm: with the standard backward pass per-position gradient norms explode toward early
  layers; with LRP they grow up to 5x less.
- Rank: averaged Jacobian numerically full rank at every layer (≥5,119 of 5,120 singular values above the
  fp32 floor), but entropy effective rank 1,746 at L8 vs 4,765 at L56 (J++). Expert Jacobians disagree
  (cosine) at L22 and largely agree with the J-lens Jacobian at L52 except experts that got <2% of fit
  positions: "the transport map becomes more globally linear in later layers".
- Fit scaling: at a fixed ~7,100 source positions, longer sequences give a better lens and it is not
  plateauing; cost scales with positions, not sequence length.
- Null results: linear-regression transport using future tokens did poorly vs J-lens; adding a bias or
  z-scoring readouts had mixed effect; final block as "extended unembedding" did not work; per-token
  (unaveraged) Jacobians "work very poorly"; gradient clipping helped J-lens but gave nothing on top of J++.

### WRITE direction
- **Intermediate swap** (adapted from Gurnee et al.; code name "probe-swap"): patch in a swapped-in
  concept's lens vector; success = the model's top answer word changes as intended. 67 trials on
  Qwen3.6-27B: J++ 59.7%, J 49.3%, R 50.7%, logit 22.4%. Top-5 criterion: J++ 75.9, J 70.7, R 70.7, logit 65.5.
- **Clamp ablation** in the case study: clamp out a concept's lens vector at layers 16-28; under a
  "suppress" instruction its later rank rises instead of falling.
- **Naming vs avoiding** (Gurnee et al. App. A.14 replication): ablate the implied concept's lens vector at
  layers 24-35 or 48-59, control = another concept of the same category. Under the avoiding question,
  early ablation of the J++ vector raises the implied concept's probability by 2.8pp beyond control vs
  0.36pp for J-lens (not significant); R-lens in between. Late ablation drives it near zero.
- Case-study finding, "retrieve-then-suppress": instructed concept enters the workspace by layer 24 under
  focus, neutral and suppress prompts; only focus keeps rising (26% by L56); suppress plateaus at 7-8%.
  With J-lens the concept only becomes visible from layer 40.
- The post positions J++ lens vectors as usable for "linear probing or precise interventions" and as an
  initialisation for multi-token lenses (Template Lens, Oracle Lens). Appendix D: under the "unsupervised
  probe" view, one expects lens vectors to be useful as steering vectors too (footnote cites Marks and
  Tegmark on why the optimal probe and steering vector may differ).
- No synthesis-from-words or round-trip test.

### Limits / open (stated)
- Single vocabulary tokens only; bag of concepts, not propositions ("I believe that Paris is warm in the
  spring" vs [Paris], [warm], [spring]); tokenizer-dependent (no arithmetic eval on Qwen: digits split);
  may miss non-human concepts and non-textual representations "such as mental imagery".
- Eval ceiling unclear (models may shortcut without the intermediate). Poetry stays hard for every lens.
- Clustering is coarse; source of noisy gradients open (outlier tokens, curved regions, or tokens with
  little workspace content).
- Comments, author replies (Kola Ayonrinde, 2026-11-06):
  - To niranjandeshpande (does filtering overfit to single-target tasks?): tested on WorkspaceBench; on
    the Agentic Misalignment scenario J++ scores 0.9 vs 0.73 for previous lenses; on core multi-token
    tasks 7.2 vs 4.5 (J) and 5.2 (R). "I don't think we see evidence of overfitting."
  - To Hope1234 (are expert Jacobians rotated copies; align with orthogonal Procrustes before averaging?):
    they ran a version: aligning experts rotationally to the best expert and averaging recovers ~30% of
    J++'s gain; best single expert alone beats it; the R-lens's rank exceeds the experts' ranks. Conclusion:
    divergence is "more likely to be curvature or linearisation error" than rotation.

---

## 3. Meta-tokens in the J-space (agam_bhatia, coauthors camilablank and Neel Nanda; posted 2026-07-20)

URL: https://www.lesswrong.com/posts/6ek6n7yZ5DzfarJHy/ .

### What a meta-token is
"Tokens that appear in J-lens readouts and give non-obvious evidence of the type of processing/algorithm
occurring in the model." Need not be causal themselves, "but should at least correspond to some causal
concept". Found mostly in workspace layers (~30-90% depth). In Qwen they are often Chinese multi-character
tokens, which the authors attribute to Chinese tokens being more information-dense, so a single-token
lens can name more.

### Lens used
Own J-lens for Qwen3.6-27B (64 layers): Jacobian with respect to the penultimate layer, averaged over 25
sequences of 128 tokens from NeelNanda/pile-10k (some experiments wikitext), first 4 tokens skipped
(anomalously high norm). Replications (appendix): verbal-report correlation slightly better than logit lens
early-mid (not significant); CKA bands less clean than the paper's; directed modulation replicated.
Footnote: meta-tokens "more prevalent in the official Qwen-3.6-27B J-Lens readouts" (Neuronpedia).

### How meta-tokens are found
- By hand from case studies, then an "autoresearch loop":
  1. corpus-wide sweep: J-lens readout at every (layer, position), rank of every token; a token "fires"
     on a sample if rank ≤ 20 somewhere in L20-62; prevalence = fraction of samples.
  2. an LLM proposes candidates plus a hypothesis of the computation each indexes.
  3. filters: prevalence; position type (structural slots: terminators, newlines, commas, chat-template
     tags); obviousness (a probe predicts the firing token from surface features: previous-turn embedding
     similarity, lexical overlap); LM judge keeps firings with no surface relation to the text.
  4. human inspection.
  Corpora: Pile, WikiText, FineWeb (en/zh/ja/ko/es/ar), TriviaQA, HotpotQA, 2Wiki, MuSiQue, TruthfulQA,
  CommonsenseQA, GSM8K, CaseHOLD, DailyDialog, persuasion, news summaries, Alpaca, Dolly, SWE-chat, Dolci;
  500-1000 samples each.
- Result: "fairly disappointed"; most high-prevalence candidates were surface-predictable. Binding
  constraint stated as the J-lens itself (one vocab item per direction).

### Case studies and WRITE-direction results
- **Interpretative** (什么意思, 是什么意思, 这句话, 是何含义; "what does it mean", "this sentence"): fire on
  ambiguous text (poems, jokes, crossword hints) just before genre tokens like "poem"; a clarifying prefix
  ("A poem:") largely removes them.
  Steering protocol: negative steering along the meta-token's J-lens direction, a separate vector per layer,
  at (1) all workspace layers or (2) the layers where the token is in the top 10 at any position;
  positions = punctuation and following chat-template tokens (all positions for the rhyme prompt); sweep
  the coefficient to the largest still-coherent value, per prompt and per vector; LLM autorater on
  coherent, on-topic rollouts; 50 rollouts per prompt; matched random-token vectors as control.
  Results (read from figure IMG 4, target-behaviour rate): puns baseline 1.0, salient-layer steer ~0.29,
  workspace steer ~0.38, controls ~0.92-0.97; rhymes baseline ~0.87, steer ~0.50/~0.45, controls ~0.77-0.89;
  wordplay-hint baseline ~0.99, steer ~0.68/~0.66, controls ~1.0. Projection ablation (coefficient 1.0)
  "generally had no effect" (puns ~0.79); the authors read this as the J-lens vector being "an imperfect
  approximation to the true direction", leaving a residual that negative steering can cancel.
- **GCD**: "gcd" fires on all LCM prompts tested. The J-lens vector for "gcd" itself is "not causal". The
  causal test instead uses difference-of-means class vectors: `d_g` = mean final-prompt-token residual over
  20 prompts with gcd = g, minus the grand mean of the 10 class centroids, per layer. Two swaps:
  "pinv": `c = V⁺(h − grand)` with `V = [d_g, d_t]`, write back `h + γ·V(σ(c) − c)` (σ swaps the
  coordinates), orthogonal complement untouched; "surgical": project out `d_g`, add `α·d_t`. Single layers,
  all prefill positions, 50 rollouts x 10 prompt variants, T=0.7, no CoT, <10% degenerate. Example 9->3:
  "The LCM of 27 and 90 is" -> 810 (true 270). Flip rates read from figure IMG 7: 6->2 ~0.97 surgical /
  ~0.76 pinv; 4->2 ~0.58/~0.37; 9->3 ~0.37/~0.33; 8->4 ~0/~0.56; 6->3 ~0.01/~0.11; "refine" 2->4
  ~0.36/~0.19, 3->9 small; "sideways" (7->3, 6->7, 6->8) ~0. A linear probe on final-layer activations
  predicts the GCD. Other candidates seen, not studied: "stack" (parenthesis balancing), "xor" (find repeats).
- **Hedging** (大概率, "most likely"): appears on open questions with many answers; negative steering at
  all prefill positions and salient layers. Commit-to-one-answer rate (read from figure IMG 8, five
  prompts): baseline 0-0.12, random 0-0.08, meta 0.26-0.56. Example "John ran out of pencils..." ->
  "He went to the stationery store."
- Unvalidated candidates from the search: 更新于, 相信在 (fires on compaction-summary prompts), 追问,
  反问, 引用.

### Limits / open
Single token, Chinese phrases, one 27B model; meta-concepts without a corresponding token are invisible;
multi-token lenses named as the next step. One comment (Burny), no corrections.

---

## 4. Neuron Lens repository (stdlogits/neuron-lens; README credits @_a9lim and @treetowntree, "with Claude")

URL: https://github.com/stdlogits/neuron-lens (created 2026-10-30, last push 2026-11-11). Read: README,
docs/design.md (most), docs/findings.md §1, §3, §4, §6 and Open questions, `neuron_lens/transport.py`
docstring, `neuron_lens/readout.py`. Lenses at `stdlogits/rs-lenses`, catalog dataset `stdlogits/neuron-lens`.
Models: Qwen3-8B (36 layers, d 4,096, 12,288 SwiGLU neurons/layer) and Gemma 4 12B (48 layers, d 3,840,
text, images, audio); Qwen3.5-35B-A3B estimator support "in progress".

### The two lenses (design.md, "one estimator in two directions")

| | R-lens | S-lens |
|---|---|---|
| Jacobian | `J_L = mean_t Σ_{t'≥t} ∂h_final(t')/∂h_L(t)` | `K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)` |
| reads with | `W_U` | `W_E` |
| read of h | `softmax(W_U · norm(J_L h))` | token v scores `h · K_L (W_E[v] − mean(W_E))` |
| mode | reverse (one cotangent per output basis direction) | forward (one tangent per input basis direction) |
| origin | J-lens under RelP rules (= the R-lens) | K-lens under RelP rules |

- `h_final` is the last layer's output before the final norm; `norm` is the model's final RMSNorm with its
  weight, so an R-lens read is scale-free (describes a direction, not an effect size).
- Summed over offsets (t' ≥ t, or s ≤ t), averaged over positions; first 16 positions and the last are
  excluded; documents weigh equally and alternate even/odd halves for split-half checks.
- RelP rules: decoder RMSNorms hold `rsqrt` constant; `silu(g)` linearized as `sigmoid(g)·dg`; `silu(g)·up`
  passes half of each factor's change (β = 0.5); attention, q/k norms and RoPE keep the true derivative.
  Under these rules each layer is, per document, a fixed linear map, and "the R-lens's map is exactly the
  transpose of the S-lens's", so one estimator fits both.
- Each lens is fitted at two points per layer: `out` (layer output: reads MLP writes) and `mid` (the MLP
  input after attention: reads MLP read directions and attention writes). "A read anywhere else
  extrapolates, and the catalog ships none."
- **K-lens** (README): "Theia Vogel's K-lens (@voooooogel, 2026, unpublished) takes the Jacobian the other
  way, back to the input embeddings ... answering which input tokens put that direction there." The
  **S-lens** is "introduced here": "the K-lens under RelP's rules, as the R-lens is the J-lens under them."

### Cost and precision (design.md)
- Fit: 1,000 DCLM documents x 128 tokens per lens; 11-16 s per document per lens on one H100; ~3 h per
  text lens. Qwen3-8B per document on H100: R 11.2 s (autograd 27.2 s), S 11.5 s (autograd 63.6 s), using
  hand-written per-layer linear maps under `torch.compile`, not autograd. Costs `d_model` passes per
  document, 256 directions at a time.
- bf16 is 0.72% (R) / 0.65% (S) from fp32; two documents' estimates differ by 65%, so sampling error
  dominates (~1.5% at 1,000 docs). FP8 adds a 3.8-4.5% bias that does not average out. Random-probe
  sketching argued not to pay.
- Split-half difference (`‖even − odd‖/‖mean‖`): R-lens 0.04-0.06 at L0-L18 down to ≤0.01 from L28; S-lens
  0.00-0.05. On a Qwen3-0.6B prototype, RelP halved the S-lens split-half noise vs true gradient (L7: 0.15
  vs 0.26).
- Corpus swap (chat-fitted R-lens vs DCLM): top-1 token agreement only 31-51% through L16, but "the corpus
  changes tokens, not concepts" and every causal score is the same within ~0.04. Weak, diffuse writes
  drift toward the corpus's register ("register tokens are the lens's, not the neuron's").

### How a neuron's write vector is turned into a readout
Neuron n at layer L: reads normalized MLP input `x̂` through `a = γ ⊙ W_gate[n]` and `b = γ ⊙ W_up[n]`,
writes `act · W_down[:, n]`, `act = silu(a·x̂)(b·x̂)`. Because `up` is linear, `act` takes both signs, so
each neuron is catalogued as two **poles**: `+` writes `+W_down[:, n]` and reads along `â + b̂`; `−` writes
`−W_down[:, n]` and reads along `â − b̂` (unit rows; a heuristic, activations give the true poles).

Reads per pole (`08_reads.py`, `w = ±W_down[:, n]`, `c = â ± b̂`):
- `write_R = softmax(W_U · norm(J_L^out w))`: what the write makes the model say.
- `write_S = w · K_L^out (W_E[v] − ē)`: the tokens whose "footprint" the write imitates (a token's
  footprint is the average change it makes to the stream at that layer).
- `read_R = softmax(W_U · norm(J_L^mid c))`: what the residual says when it drives the pole.
- `read_S = c · K_L^mid (W_E[v] − ē)`: the input tokens that drive the pole.
- `_z` variants: each token's score z-scored across the layer's poles ("layer-contrastive"), which removes
  **hub tokens** that top many units at once (e.g. at L1 ` the`, ` in`, ` and` sit in 30-31% of write reads'
  top 10s). Raw `read_S` keeps literal tokens early; `read_S_z` gives concepts later but loses the literal
  peak token.
- `write_orth_*`: the write with its components along the gate and up rows removed (what the neuron adds
  beyond what it reads).
- Token mask: only tokens seen 5+ times in the activation corpora, no byte-fallback pieces (103,830 of
  151,669 for Qwen3-8B), because undertrained tokens drive some early neurons to huge activations.
README code (Qwen3-8B): `h = J[L] @ w; h = h * rsqrt(mean(h²)+1e-6) * norm.weight; topk(lm_head.weight @ h)`.

Which read works where (Qwen3-8B): input side, S-lens L0-L7, both at L8, R-lens from L9; write side,
S-lens through L8, R-lens from about L4.

### How readouts are validated
- **Steering vs lens (findings §3)**: add `±α·W_down[:, n]` to layer L's MLP output at positions 16-63 of 16
  held-out DCLM documents (64 tokens), at 1x and 4x the pole's typical top activation; record mean change in
  next-token log-probabilities; score = Pearson r over kept tokens between that change and the lens's logits
  for the same write. Control: another neuron's read from the same layer. 1,194 neurons. Median r:

  | layer | logit | J | R | control |
  |---|---|---|---|---|
  | L0 | 0.08 | 0.37 | 0.33 | −0.01 |
  | L4 | 0.09 | 0.37 | 0.45 | 0.00 |
  | L8 | 0.12 | 0.39 | 0.49 | 0.01 |
  | L12 | 0.25 | 0.57 | 0.65 | −0.02 |
  | L16 | 0.36 | 0.66 | 0.74 | 0.01 |
  | L20 | 0.59 | 0.81 | 0.87 | 0.01 |
  | L24 | 0.83 | 0.92 | 0.95 | 0.00 |
  | L28 | 0.90 | 0.96 | 0.97 | −0.01 |
  | L32 | 0.94 | 0.98 | 0.99 | −0.01 |

  R leads J from L4 to L23 by 0.03-0.13; J leads at L0; they converge from L24. From L5, 94-100% of random
  poles' writes predict their own steering effect (r past 0.24, the 99th percentile of mismatched pairs).
  Gemma 4 12B: R-lens write read predicts steering from L11 (median r 0.53 at L12-L23, 0.82 at L24-L35,
  0.97 at L36-L47); before L11 natural-size pushes move the output too little to predict.
- **Ablation where the pole fires** agrees far less before L21 (r <0.15 to L16, 0.87 at L32), and the
  lenses tie there. §4 explains: per-context linearization (scale activation by 1, 0, 1.25, 0.75) shows a
  firing's effect is near-linear (0.92-1.00) but context-specific before L21 (one context vs the pole's other
  23: r 0.05-0.11 at L0-L8; 0.78 at L28-L35). "The lens and steering agree because both average over contexts."
  "A write read is the pole's average push."
- **Input side (§6)**: exact drivers = every kept token substituted at position 24 of 8 documents; share of a
  read's top 10 among the 20 exact drivers: `read_S` 0.32 at L0, 0.10 at L1-L4, 0.06 at L8, ≤0.004 by L12-L16;
  `read_R` 0.011 at L0, 0.069 at L8, 0.053-0.080 at L9-L29 (control ≤0.003). Why S fades after L8 is open:
  R² of the S-lens footprint vs the exact token footprint is 0.63 at L0, 0.20 at L1, 0.09 at L5, 0.10 at L15,
  but even the exact in-context linearization explains only 0.86/0.30/0.12/0.07; RelP beats the true
  derivative from L1 (0.17 vs 0.07 at L15).
- **Reads at firings (§7)**: S-lens names the firing token for 85% of poles at L0; R-lens read names exact
  drivers for 44-51% at L27-L31; write names the next token for about a third at L28-L30.
- **Blind label tests (§8)**: Claude explainers label 160 poles from lens reads only; detection AUROC 0.80
  for firing labels; write labels picked out of four from steered text 65% (chance 25%).
- Tests: `tests/test_transport.py` checks both lenses against brute-force Jacobians of the patched model
  (1e-4) on a tiny Qwen3; same for small Gemma 4 and Qwen3.5 models.

### Other WRITE-direction items
- Steering a neuron's write (above) is the main write test. Images/audio (Gemma 4): a neuron's read direction
  carried back to pixels/samples "raises the neuron only in the first six layers" when added to held-out
  images; started from the read and optimised by gradient ascent, a single patch drives its neuron through L23.
- Heads (§11): each vocabulary token's S-lens footprint fed through a head's OV and QK circuits gives
  token-by-token copy and attend maps; "a head reads what a position holds, not what it is about to say",
  so keys/values take the S-lens, queries the R-lens.

### Qwen3.5 rules (design.md "in progress"), relevant for hybrid Qwen encoders
Gated DeltaNet written as attention `o = A v` with `A = W L⁻¹ diag(β)`; RelP rules by operation: LN rule on
residual norms and DeltaNet's output norm; identity rule on SiLU (experts, after the conv, DeltaNet's
`silu(z)`); half rule on elementwise gates (SwiGLU, DeltaNet `norm(o)·silu(z)`, attention output gate
`o·σ(gate)`); true derivative for softmax attention, DeltaNet `A`, router weights, shared-expert gate.
Table compares with the published R-lens rules (J++ code `lrp.py`, preset `r+moe`), which hold the router
and shared gate (and scale the shared branch ×4) and keep plain autograd inside token mixers.

### Limits / open (stated)
Two models; one English-heavy text lens corpus; 128-token fit documents; token-level tests; labels by
Claude models; steering adds a write at every position "up to many times its natural size: it measures a
direction's average effect". Open: why S-lens reads fall after L8; "whether J++ filtering ... or a longer
context improves early and mid-network reads"; Gemma 4's weak early writes; others.

---

## 5. K-lens tweet (@voooooogel, 2026-08-05)

Fetched via `https://api.fxtwitter.com/voooooogel/status/2084858861551526002` and
`https://api.fxtwitter.com/2/conversation/2084858861551526002` (thread plus 15 replies); the image from
pbs.twimg.com. api.vxtwitter.com returned a Cloudflare challenge.

- Text: "been playing around with anthropic's jacobian lens and my own variant, the k-lens. here are both
  lenses showing some internal states from qwen 3.6-27b on the thunderword." Single tweet, not a thread.
- The only definition from the author: a reply asked "is k-lens just to input embeddings instead?"; the
  author answered "yep basically".
- Image: Finnegans Wake's first thunderword split into token pieces, chips labelled `J@layer` and
  `K@layer`. Caption: "J (above — what the workspace is poised toward) and K (below — what the surface
  deposited)". Footer: "Layers 0-62 sampled at 18 depths ... Instrument: backwards lens (fit 2026-08-04,
  n=200 wikitext) + neuronpedia jacobian lens." So the K-lens is described as a "backwards lens" fit on 200
  wikitext samples. No math, no metrics, no steering.
- The formal K-lens definition used above (`K_L = mean_t Σ_{s≤t} ∂h_L(t)/∂e(s)`, scored with
  `W_E[v] − mean(W_E)`) is the neuron-lens README's description of it, not the tweet's.

---

## 6. Tuned lens (Belrose et al. 2023, arXiv 2303.08112) — background

Abstract and method sections read (arXiv API + HTML). Train one affine "translator" `(A_l, b_l)` per layer
of a frozen model: `TunedLens_l(h_l) = LogitLens(A_l h_l + b_l)`, initialised to identity, trained to
minimise `KL(f_{>l}(h_l) || TunedLens_l(h_l))` against the model's final-layer distribution at the same
position (a distillation loss). Motivations: residual updates have nonzero mean (hence the bias `b_l`) and
hidden-state covariance drifts across layers (hence `A_l`). Tested up to 20B parameters; "more predictive,
reliable and unbiased than the logit lens". Causal check: causal basis extraction finds directions important
to the lens; their influence on the model correlates (Spearman 0.89, Pythia 410M layer 18). Lenses trained on
a base model transfer to its fine-tunes (Vicuna 13B from LLaMA 13B). The current version notes Muon training
gives much lower loss than the paper's SGD lenses ("severely undertrained"). Training a full set takes under
an hour on one 8xA40 node. J++ (App. G) reports that a regression transport using future tokens did poorly
compared to the J-lens and that a bias term had mixed effects.

---

## Not read / could not read
- J-lens paper itself (Gurnee et al. 2026) was out of scope; its definitions above are as restated by the sources.
- J++ code (safety-research/jpp_lens) and its `lrp.py`; R-lens and J++ released lens files; WorkspaceBench post
  (cited in J++ author replies).
- J++ figures (numbers taken from the text and tables, which state them); R-lens figures 2, 4-6, 9-19 not viewed.
- Neuron-lens findings §2, §5, §7-§18 beyond the README's summaries; most scripts.
- The K-lens has no published write-up; the tweet carries no math.
