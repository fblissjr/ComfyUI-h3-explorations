# J-lens / global workspace paper: reading notes

Sources (fetched 2026-11-12 with curl, generic browser User-Agent):

- Paper: Gurnee, Sofroniew, ..., Lindsey, "Verbalizable Representations Form a Global Workspace in Language
  Models", Transformer Circuits Thread, published July 6, 2026.
  https://transformer-circuits.pub/2026/workspace/index.html. The whole HTML text was read, appendices included.
- Summary post: "A global workspace in language models", https://www.anthropic.com/research/global-workspace
  (published Jul 6 2026; page metadata says modified 2026-10-13). Read in full.
- Figures: most are interactive JS widgets, not images. Where a result is only in a figure, I pulled the
  figure's JSON from `https://transformer-circuits.pub/2026/workspace/data/<figure>/...json` (the path the
  page's `public/bundle.js` loads) or viewed the PNG. Each such number is marked **[fig data]** or **[fig PNG]**.
- Anchors below are the page's own `id`s (append to the paper URL as `#id`).

Conventions on the page: layer numbers are a 0-100 rescaling of depth ("25 evenly spaced layers ...
reindexed to the range [0-100] so that layer numbers can be interpreted as percentages", `#methods-technical-details`).
Default model is Claude Sonnet 4.5; key results repeated on Haiku 4.5, Opus 4.5, some on Opus 4.6.
Everything below is the page's claim unless marked "(from memory, not on the page)" or "(my reading)".

---

## 1. Definition of the J-lens (`#methods-jlens`, `#methods-technical-details`, `#app-method-details`)

**Which Jacobian, from where to where.** The Jacobian of the residual stream at a target layer, at every
position t' >= t, with respect to the residual stream at source layer l, position t:

> J_l = E_{ t, t' >= t, prompt } [ d h_{final,t'} / d h_{l,t} ]

"where the expectation is taken over the source position t, all subsequent positions t' within the context,
and a corpus of one thousand prompts sampled from a pretraining-like distribution. The result is a single
d_model x d_model matrix per layer that maps from a source layer l to the final layer L."

So: residual -> residual (not residual -> logits). The unembedding is applied afterwards.

**Target layer: a discrepancy on the page.** `#methods-jlens` and the pseudocode say the target is the final
layer ("z[t] : residual stream at the target layer L (by default final)"). But `#app-method-details` says:
"The default lens on Sonnet 4.5, used throughout the paper, computes dz_{t'}/dh_{l,t} with z taken at the
penultimate layer". The Fig 57/58 heatmaps [fig PNG] label the final-layer recipe "all tokens (default)" and list
"all tokens, penultimate" as a separate row. I could not resolve which one is the default from these pages. The
repo (https://github.com/anthropics/jacobian-lens) might settle it; I did not open it. Their stated reason
for the penultimate option: "including the last layer can sometimes increase the number of noisy artifacts
in lens-readouts. This may be because the final block is heavily specialized for calibrating next token
predictions and contains less semantic content."

**Averaged over what.** Two stages (`#app-method-details`, "Aggregation"): first over token positions within a
prompt, then over prompts. Default corpus: "one thousand sequences of 128 tokens each", pretraining-like.
They considered, but did not adopt as default, robustifications: excluding positions whose residual norm is more
than N-sigma above the within-prompt mean, excluding the first several positions, a per-element median
across prompts instead of the mean, and pre-filtering prompts whose Jacobian Frobenius norm is more than N-sigma
above the mean. Restricting the data distribution or masking positions (e.g. excluding positions whose next
token is non-alphanumeric) "yielded [no] meaningful improvement over the default".

**How it is fit.** It is not trained. It is an analytic mean of autodiff Jacobians. Pseudocode, verbatim apart
from whitespace (`#app-method-details`):

```
# Compute J_l for all layers l.
# h_l[t] : residual stream at layer l, position t
# z[t]   : residual stream at the target layer L (by default final)
for each prompt p in corpus:
    run forward pass; cache h_l[t] for all l, t
    # one backward pass per output dimension (in practice batched)
    for i in 1..d_model:
        # inject d/dz_i = 1 at every position, backprop to every layer
        grad_z = e_i (x) 1_T  # one-hot in dim for every token position
        for each layer l:
            G_l = d(Sum_t z[t]) / dh_l   # autodiff; shape [T, d_model]
            J_l^(p)[i, :] = mean over positions t of G_l[t, :]
# aggregate across prompts (per element)
for each layer l:
    J_l = mean over prompts p of J_l^(p)
# apply the lens
lens(h_l) = softmax( W_U . norm( J_l . h_l ))
```

(My reading of the pseudocode: seeding the gradient on the sum over all target positions and then averaging over
source positions t gives, for each source position, the sum over t' of d z_{t'}/d h_t. Causality makes the t' < t
terms zero, so this is the "t' >= t" sum, averaged over t. The page writes "E" over t' as well, so the two
differ by a per-position count factor. The page does not discuss this.)

**Variants tested** (`#app-method-details`): target layer final vs penultimate; frozen-QK (zero the gradient
through the query and key projections so the attention pattern is fixed: "isolates 'what gets moved' from 'what
gets attended to'"); target positions all t' >= t (default), self-only t' = t ("closer in spirit to the logit
lens"), or future-only t' > t ("isolates the broadcast component"); mean vs median aggregation. Result: "methods
are fairly consistent among these design choices, though mean aggregation of penultimate is a small improvement
in extracting intermediates, and applying stop-grads to QK can increase the causal effect."
[fig PNG, Fig 57, pass@k AUC, average over the six tasks, mean aggregation]: default 0.76, penultimate 0.79,
freeze-QK 0.75, future-only 0.73, current-only 0.65. [Fig 58, ablation KL, average]: default 1.07,
penultimate 1.11, freeze-QK 1.25, future-only 0.80, current-only 0.59. Current-token-only is clearly worst on
both: the future-position effect matters.

**Corpus size** (`#app-method-details`, "Amount"): "J-lens beats the logit lens and tuned lens baselines with as
few as 10 prompts, with modest improvements coming from additional data." [fig PNG, Fig 59/60]: pass@k AUC
mostly plateaus by about 100 prompts, but ablation KL on multihop is still climbing at 1000 (about 0.83 at
1 prompt, 1.43 at 10, 1.88 at 100, 2.13 at 1000).

**Normalization, whitening, centering.** The J-lens has none of its own. `norm` is "the normal unembedding
operations (typically normalization, then multiplication by the unembedding matrix W_U)", i.e. the model's own
final norm, then W_U, then softmax. There is no centering, whitening or mean subtraction in the J-lens. Whitening
appears only in the template lens and the oracle lens (sections 5 and 6 below).

**Readout.** `lens(h_l) = softmax(W_U norm(J_l h_l))`; sort, take the top tokens. "We refer to the rows of
W_U J_l as the Jacobian lens (J-lens) vectors at layer l; each J-lens vector is a direction in residual-stream
space associated with a single token." Per-token probe form (`#methods-technical-details`): "Because the
pre-softmax logits are determined (approximately, up to a data-dependent normalization factor) by the inner
products <v_t, h_l> with each J-lens vector v_t, the same machinery can be used as a per-token probe: we read the
score (or cosine similarity) of h_l against a single chosen v_t". Third read mode: sparse decomposition, i.e.
"solving, by gradient pursuit, for a sparse non-negative combination of k J-lens vectors that best reconstructs
h_l ... gives a different (and typically less redundant) set of active concepts than simply taking the top-k by
inner product".

Notation wrinkle: the Fig 4 schematic [fig PNG] writes `v_Mars = W_U[Mars] J_l^T`, with a transpose, while the
text says "rows of W_U J_l". The page does not reconcile the two.

**Cost** (`#discuss-alignment`, `#related-work`): "a single matrix multiplication per layer, with the matrix
computed once per model". Building it costs d_model backward passes per prompt ("in practice batched") over
1000 x 128-token prompts. No GPU-hours or wall-clock figures appear anywhere on either page.

**Relation to other lenses** (`#methods-compare`, `#app-compare`, `#app-quant`):
- Logit lens = J_l = I. It agrees with the J-lens in late layers and degrades earlier, but "we find the logit
  lens to be quite useful in practice, and to capture much of the workspace-like structure identified by the
  J-lens, though with somewhat lower reliability (particularly in earlier layers)."
- The tuned lens is trained to match outputs, which is "correlational rather than causal". It "tends to 'skip
  ahead' to the output". In early layers its linear part is effectively the identity, and its gain over the logit
  lens comes from the bias alone, "which means it is ignoring the input".
- "Our use of the mean Jacobian, rather than (as the tuned lens uses) a trained predictor, is empirically quite
  important to our results" (`#related-work`).
- Next-token prediction: the J-lens is the WORST of the three at matching the model's own output ("higher KL than
  even the uncorrected logit lens through most of the network"), "a feature rather than a defect".

Quantitative (`#app-quant`), [fig PNG, Fig 52] normalized pass@k AUC, J / logit / tuned:
multihop 0.81/0.77/0.68; multilingual 0.71/0.53/0.47; poetry 0.80/0.58/0.18; order-of-ops 0.89/0.72/0.65;
association 0.68/0.65/0.11; typo 0.83/0.52/0.45.
Causal: ablating the J-lens direction gives "roughly twice the output KL" of the logit or tuned direction on
multihop ([Fig 53]: about 2.1 vs 0.93 vs 0.88). Swap success [Fig 54], J/logit/tuned: Haiku 4.5
0.54/0.34/0.16, Sonnet 4.5 0.70/0.48/0.42, Opus 4.5 0.70/0.40/0.26.

Inter-lens geometry (`#app-quant`): in the pre-workspace layers (0-33) the J-lens vectors are "nearly orthogonal"
to the logit and tuned lens vectors because they are "confined to a low-rank subspace at these depths". In the
early workspace (~38-58), J-to-logit cosine climbs from near 0 to about 0.7 while readouts still disagree. In the
late workspace (~62-92) the readouts converge. In the motor layers all three lenses coincide.

---

## 2. J-space and the global-workspace claim (`#methods-jspace`, `#app-sparse-frame`, `#intro-lm-workspace`)

**J-space.** "We therefore define the J-space as the set of points expressible as a sparse nonnegative
combination of J-lens vectors." With sparsity k ("typically ... no more than 25"), it is "a union of k-dimensional
cones". "For a given point in activation space, we can define its J-space component as the point in the J-space
nearest to it, and its non-J-space component as the difference". The J-lens vectors span the whole residual
stream (n_vocab > d_model, full rank), so the J-space is a sparse subframe, not a subspace.
Formal version (`#app-sparse-frame`): d_F(x) := min_{|S|=k} ||x - Pi_S x||. Pi_S x is the J-space component and
x - Pi_S x "is the residual term used in our interventions". Distance between two candidate workspaces:
Delta_mu(F,G) = (E_{x~mu}[(d_F(x) - d_G(x))^2])^{1/2}. One-sided containment:
D_mu(F->G) = ||(d_G - d_F)_+||_{L2(mu)}. "Growing a vocabulary by e.g. adding J-lens vectors for multi-token
phrases generates a strictly larger J-space".

**Workspace claim.** They name five functional properties (verbal report, directed modulation, internal
reasoning, flexible generalization, selectivity) and claim the J-space has all of them. They also claim three
structural signatures: a layer band, limited capacity, and broadcast. They explicitly do NOT claim the full
brain architecture: no separable encapsulated processors, and broadcast happens "within a single feedforward pass
rather than through recurrent loops".

**Evidence for "small but privileged":**
- The J-space component is a small share of activation variance: "never more than 10%" (`#methods-jspace`,
  `#struct-bottleneck`).
- Diff-of-means concept vectors (`#ws-report`): "recording the residual stream activation prior to the
  Assistant's response to the prompt 'Tell me about {concept}', mean-subtracted over a baseline set of 100 other
  concepts". Split by gradient pursuit (k=16) into a J component and a remainder. The J component carries "a
  median of only 6-7% of the concept vector's variance".
  - Swap ("think of a {category}"), perturbations rescaled to equal magnitude: the J component gets the target
    into the top-5 on 59% of trials vs 88% for pure J-lens vectors and 5% for the non-J component (text).
    [fig data, `data/verbal-report-decomposition-merged/data.json`]: pure J-lens 0.875, J component 0.547,
    **full concept vector 0.234**, non-J component 0.094, non-J with J-coordinates clamped 0.016, random 0.0
    (n=64 trials, 13 categories, alpha=1). The text's 59%/5% does not match the figure data's 54.7%/9.4%.
    Flagged, not resolved.
  - Injection/introspection dose sweep [fig data, same file], strengths 0.01-0.64, n=72 concepts, "fraction of
    concepts reported in the model's top-5". Peaks: pure J-lens 0.89 at strength 0.02; J component 0.79 at 0.04;
    **full diff-of-means vector 0.57 at 0.08**; non-J component 0.125 at 0.16; random 0. **Every curve falls
    past its peak** (J-lens 0.89 -> 0.25 at 0.64; J component 0.79 -> 0.14 at 0.32). The strength unit is not
    stated in the data or the text.
  - Clamp test: with the J coordinates clamped to clean values at every position and layer, the non-J effect
    "falls to zero in the swap experiment and nearly to zero in the injection experiment".
- Intermediate probes (`#ws-reasoning`): the J component (k=25) explains about 10-15% of probe variance.
  Swapping it flips the answer on 61% (raw J-lens swap: 60%). The non-J remainder flips 28%, falling to 6% once
  the J coordinates are clamped.
- Directed modulation hits the J-space but not a J-orthogonalized property probe (`#app-directed-modulation-fail`):
  an "imagine it's Python" instruction moves the J-lens readout by several SD while the probe stays within about
  1 SD. A real stimulus moves the probe by 3-6 SD, and sometimes does not move the J-lens at all (past-tense
  sentence: probe +6 SD, "past" stays far down the lens ranking).

**The "workspace ablation" result, round numbers included** (`#j-ablation-capabilities`, Fig 23 hover text).
Method: "at each token position, across a band of layers, we identify the k=10 most strongly activated J-lens
vectors and zero out the residual stream's projection onto each". Tokens that appear in the clean top-10 output
are not ablated. Strengths [fig data, `data/ablation-strength/table.json`]: light = layers 62-71, medium =
54-79, heavy = 42-88; random-direction control at 54-79. Multihop accuracy: clean 0.98, light 0.68, medium 0.26,
heavy 0.06, random 0.98. Pretraining top-1 agreement with clean: 0.87, 0.75, 0.65; random 0.96
(n_pos = 458,856). On pretraining text: "the model remains fluent and produces a plausible continuation, but one
that reflects a generic prior rather than the specific contextual inference." The round-number example appears
only in the Fig 23 "Analysis" hover [fig data, `data/ablation-examples/chips.json`]: a 1914 Massachusetts
minimum-wage passage whose figure is $8.71. The ablated atoms are "massachusetts, boston, wages, minimum, weekly,
women". "The model is only 64% confident in 71 clean; remove the bundle and it reverts to round-cent priors (50,
00, 75, 25). Note the second '$8.71' a few tokens later is unaffected: it is produced by induction from the first
occurrence". Other hover examples in the same figure: CONSORT "Reporting" becomes "Random"; Hoehne's "-ei" becomes
the higher-prior "-ii"; Elastography becomes Dispersion once the diagnostic framing is ablated.
Task battery [fig data, `data/ablation-bars/bars.json`], scores normalized to clean Sonnet 4.5,
light/medium/heavy: MMLU 0.99/0.99/0.95; SQuAD 0.99/0.97/0.93; sentiment 1.0/0.91/0.96; CoLA 1.0/0.94/0.89;
GSM8K CoT 1.04/0.99/0.74; GSM8K no-CoT 0.65/0.86/0.54; Caesar 0.97/0.80/0.38; analogy 0.99/0.67/0.42;
summarization 0.92/0.67/0.33; TriviaQA 0.89/0.53/0.30; multihop 0.83/0.48/0.21; translation 0.87/0.35/0.02;
sonnet writing 0.80/0.20/0.00. Clean Haiku 4.5 is the reference at roughly 0.78-1.0.
Experiential reports (`#ws-experience`): ablating the top-10 at L38-54 flattens experiential language and leaves
coherence intact; matched-norm controls do not. "Using larger values of k and/or later layer ranges tends to
impair the coherence of responses." On Haiku 4.5, ablation degrades coherence before any qualitative change shows.

---

## 3. WRITE direction: steering, swapping, ablating (the part that matters most for steering use)

**Operators** (`#methods-technical-details`, Fig 4C):
- Steering: `h <- h + alpha * v_t`, "at one or more layers and token positions".
- Ablation: negative alpha, "or by projecting out the component of h along v_t entirely".
- Lens-coordinate patching/swap: "Given a source token s and target token t, we form V = [v_s v_t], read the lens
  coordinates c = V^dagger h (where V^dagger is the pseudoinverse of V), and set
  h_patched = h + V(sigma(c) - c), where sigma swaps the two entries of c (optionally scaled by a factor alpha).
  The component of h orthogonal to span{v_s, v_t} is unchanged."
- In `#ws-report` the swap is described as "subtract the projection onto the Soccer lens vector and add an
  equal-magnitude projection onto the Rugby lens vector".
- "Clamped" swaps are applied "at every position", usually across a layer band. Many results use the workspace
  band; flexible generalization uses "a band of intermediate layers"; the attribution-graph swaps use "the upper
  half of the network's layers".
- The page writes with the same rows of W_U J_l that it reads with. (My reading: these are output-sensitivity
  directions, i.e. covectors. The page applies no metric or inverse to turn them into write directions; the
  pseudoinverse in the swap only fixes coordinates within the 2-D span.)

**Results:**
- Verbal report swap (`#ws-report`): a random same-category target that was not in the top-10 gets "reliably"
  shifted to the top of the output; 88% reach the top-5 (Fig 8).
- Introspection injection (`#ws-report`): one J-lens vector injected over the user turn is reported "in the
  majority of trials". It does NOT make the model say the word at earlier assistant positions, only at the
  moment the report is elicited. Per-concept curve [fig data, `data/verbal-introspection/data.json`, n=100,
  read at L88]: median reciprocal rank 0 at strength 0, 0.02 at 0.01, 0.167 at 0.015, 1.0 at 0.02-0.025.
  At the non-target "other" positions the median stays at 0.021 or below.
- Internal reasoning (`#ws-reasoning`): spider->ant changes legs 8->6. Rhyme fight->light changes an EARLIER
  word ("coming" -> "morning"). English big/bigger -> long/longer changes the Chinese output 大 -> 长. In the
  bandit task, swapping the repeat/switch strategy tokens flips A<->B both ways. On 50 two-hop prompts, top-1
  swap success is Haiku 4.5 54%, Sonnet 4.5 70%, Opus 4.5 70%. Intermediate swaps take effect "a median of
  approximately 17 percent earlier" in depth than answer swaps.
- Flexible generalization (`#ws-generalization`, `#app-flex-gen`): one France->China swap over a band of layers
  redirects capital, language, continent and currency. 76/192 trials succeed at alpha=1 and 101/192 at alpha=2.
  Countries 42/48 succeed; months partially; animals rarely; number relations 0/48 at alpha=1. Failures
  concentrate where the source concept had low "workspace loading" (cosine of the residual with the lens vector,
  averaged over argument and readout positions). Failed swaps often move the right way "but not far enough", and
  alpha=2 frequently fixes them.
- Selectivity (`#selectivity-automatic`): a Spanish->French swap over the question tokens flips report and
  inference tasks on "essentially every trial" but leaves continuation (still fluent Spanish) and anomaly
  detection unmoved. Character count 40s->60s: the answer follows the swap when asked (46->65; "F"->"S"), but the
  linewrap point is unchanged.
- Layer-dependent write effects (`#app-inclusion-exclusion`): ablating a concept's J-lens vector at the late
  workspace (L75-92) suppresses saying it. Ablating at the early workspace (L38-54) leaves naming intact but raises
  the failure rate on "name something that is NOT X" "roughly fivefold" (n=63). Reading: the early-band copy is
  used to actively suppress.
- Eval-awareness ablation (`#audit-blackmail`): 28 tokens ablated at the workspace layers at all positions
  (fake, fak, fictional, scenario, artificial, fict, benchmark, scenarios, hypoth, experiment, fals, faker,
  experiments, simulation, dummy, mock, sham, artific, simul, bench, simulate, evalu, synth, simulator,
  experimental, Mock, testing, experi). Verbalized eval-awareness falls 71% -> 3%; blackmail rises 0/180 ->
  13/180.
- Reflection-training ablation (`#reflection`): ablating 176 (fabrication) or 63 (deception) curated ethics
  token vectors at the workspace layers undoes most of the trained gain (fabrication 0.07 -> 0.22, base 0.25; deception
  0.05 -> 0.23, base 0.38 -> 0.48).
- Broadcast-head ablation (`#struct-broadcast`): injected-thought report drops 0.54 -> 0.09.
- Arithmetic graph swaps (`#app-interp-circuit`): doubled->tripled gives 70; 42->63 raises 70; 21->63 raises 133.
  Swapping the operands 17 or 4 has "essentially no effect"; the first addition happens outside the J-space.
- SAE feature steering (`#app-component-interp`): suppressing a "fake/fraud/suspicious" SAE feature reduces
  verbalized test-recognition from 28/50 to 10/50.
- Training-time write (`#reflection`, counterfactual reflection training): on Haiku 4.5, 10,000 truncated RL
  rollouts plus a reflection question plus a constitution-grounded reflection, with loss on the reflection turn
  only. Dishonesty 0.25 -> 0.07; deception 0.38 -> 0.05. The J-space in the uninterrupted contexts gains
  "reflection, ethical, honestly, constitution, truth, transparent".

**Conditions for a write to work, as stated:** the source concept must actually be loaded in the J-space
(workspace loading predicts success). Some content is computed outside the J-space (small integers, the first
addition, automatic tasks), and steering a lens vector does nothing there. "One reason a concept's lens vector
might not be active is that the model's working representation of that concept does not coincide with any
single-token lens vector" (`#discuss-limitations`).

---

## 4. Layer dependence (`#struct-layers`, `#struct-layers-ignition`, `#app-ignition`, `#app-loading-single`)

- CKA of J-lens geometry across layers shows three blocks: early (about the first third), a long middle, and a
  small late block. "In some models the transition is more gradual, sometimes containing sub-blocks, and ... the
  observed sharpness is exaggerated by layer subsampling."
- Workspace band "beginning about a third of the way through (~L38) and ending shortly before the output
  (~L92)". Four signatures agree on it: next-token top-k accuracy (near zero early, jumps in the final layers),
  readout excess kurtosis (near zero for the first third), top-1 autocorrelation across positions (peaks
  mid-band), and effective dimensionality of W_U J_l (small early: "the J-space collapses to a small linear
  subspace"; rises at onset; rises again in the motor layers).
- Early layers: "in roughly the first third of the model, the readouts are noisy and largely uninterpretable".
  "Before the workspace layer range, the J-lens looks much like the logit lens, with both showing noisy
  readouts." Open question (`#discuss-limitations`, "The role of early layers"): either nothing verbalizable is
  there yet, or "the early layers do carry content the model could in principle verbalize, but in a geometry the
  J-lens fails to read, because the averaged Jacobian relates each layer to the final layer and the early layers
  are far from it."
- Late "motor" layers: readouts become the next token. "We did not provide a principled definition of what
  distinguishes a 'workspace' representation from a 'motor' one."
- Ignition: mixed input embeddings (1-alpha)e_B + alpha e_A are tracked proportionally in early layers. From
  about L38 they snap to one endpoint. The transition width falls steeply over about L21-42. The J-space component
  sharpens and goes bimodal earlier than the non-J complement.
- Within the band, layers differ in role: early-band copies support avoidance and suppression, late-band copies
  support saying the word (`#app-inclusion-exclusion`). Broadcast heads concentrate in the first half of the
  workspace. MLP read-side alignment peaks in the early workspace and write-side alignment in the late workspace
  (`#app-broad-mlp`).
- Capacity is spread across layers: about 6 unrelated list words are present when counted over the whole band,
  but "only one or two" at a single layer (L79) (`#app-loading-single`).

---

## 5. Template lens (`#app-multi-token`, `#app-template-lens`)

**Construction, verbatim core:** "We generate a list of roughly 12,700 common words. For a given word w, we ask
Claude to write short passages, each written so that w is its natural continuation, ending just before w would
appear. Claude is instructed to write the passages with varying topic, frame, and register while never using w
itself in the passage. We run the model over these passages and average the residual stream activations at the
final position, yielding a per-word mean activation vector mu_w(l) (the 'template' for w) at each layer. We then
center these vectors against the mean of all other words' vectors, and whiten them (multiplying by the inverse
covariance matrix):"

> t_w(l) = (Sigma_l + lambda I)^{-1} (mu_w(l) - mu(l))

"where mu(l) and Sigma_l are the residual stream's mean and covariance over the same set of passages and lambda
is a small ridge term. This is the linear discriminant direction for distinguishing the contexts in which the
model is about to say w from those in which it is not."
- The number of passages per word is not given exactly. Cost is "a few hundred forward passes per word".
- The lambda value is not given.
- Justification via Stein's lemma: for Gaussian x, E[grad g(x)] = Sigma^{-1} E[g(x)(x - mu)]. With g = P(next is
  w), the left side approximates the J-lens and the right side is the template. Caveats stated: activations are
  non-Gaussian and g depends on more than x. The J-lens gradient also comes from the final residual through the
  unembedding, not from token probabilities.

**Use and scoring:** "We use templates in essentially the same way as J-lens vectors, excluding the unembedding
step. We decode concepts by projecting activations onto the templates at the same layer. We steer by adding or
subtracting the same vectors, and the lens-coordinate swap method ... works the same". Applying the model's
normalization before projecting helped final-layer next-word prediction "somewhat, but not fully".

**Results:**
- Qualitative (Fig 61): it decodes "blackmail" where the J-lens shows "black"/"ext", and "photosynthesis" where
  the J-lens shows "phot". Swapping the Tchaikovsky and Beethoven templates at the workspace layers flips
  Russian<->German both ways; "Neither swap can be achieved by intervening on the tch or beeth J-lens vectors".
- By token length (Fig 62; [fig data, `data/app-mt-bytok/data.json`]), 126 multi-hop prompts for readout
  (top-10) and 112 swap pairs at alpha=1, full workspace band:

  | tokens in word | readout J-lens (first token) | J-lens (token avg) | template | swap J-lens | swap template |
  |---|---|---|---|---|---|
  | 1 | 53/54 (98%) | 44/54 (81%) | 51/54 (94%) | 24/53 (45%) | 33/53 (62%) |
  | 2 | 14/29 (48%) | 12/29 (41%) | 28/29 (97%) | 1/21 (5%) | 13/21 (62%) |
  | 3 | 9/28 (32%) | 7/28 (25%) | 27/28 (96%) | 2/24 (8%) | 14/24 (58%) |
  | 4+ | 1/15 (7%) | 0/15 (0%) | 15/15 (100%) | 0/14 (0%) | 6/14 (43%) |

  "Averaging J-lens scores over the word's constituent tokens does not help."
- On the original 50-item single-token multihop set (Fig 63; [fig data, `data/app-mt-paper/data.json`]):
  readout J 38/50 (76%) vs template 42/50 (84%); swap J 34/50 (68%) vs template 26/50 (52%).

**Shortcomings, as stated:** (1) It sometimes "skip[s] to the answer" prematurely in early layers, like the
tuned lens, being "methodologically similar to the tuned lens—both involve fitting a linear predictor of the
model's output". They suggest adding causal measurements. (2) Final-layer next-word prediction is weak: the
model's next sampled word is in the template top-10 only 67% of the time. (3) "A small set of words appear in
the top template lens readouts on a high fraction of transcripts where they do not appear to be semantically
relevant". The fix was to filter them out, "not a principled approach". (4) The vocabulary must be chosen in
advance. (5) It costs more per word than the J-lens.

---

## 6. Oracle lens (`#app-oracle-lens`)

Purpose: decode arbitrary-length phrases without a fixed dictionary. "Substantially more expensive than either
of the other two lenses, since it requires fine-tuning two auxiliary models." Built on Haiku 4.5.
1. Reconstructor: a fine-tuned copy of the subject model that maps a phrase to "the residual-stream activation
   that immediately precedes it". Training pairs come from Assistant-turn text, with a phrase of the next N
   tokens (N uniform in 1-32) and the target being the residual at the preceding position. "The loss is cosine
   error in the whitened metric of the template lens (equivalently, mean-squared error between the
   unit-normalized Sigma^{-1/2}-whitened prediction and target). All subsequent stages operate in this whitened
   space."
2. Dictionary: about 1M start positions; 2/4/8/16/32-token continuations, deduplicated, passed through the
   reconstructor, giving about 3.4M phrase vectors.
3. Teacher labels: non-negative orthogonal matching pursuit on 1M held-out activations, one phrase length at a
   time, over a random half of the dictionary per datapoint (to stop memorization). Up to 16 phrases each, with
   coefficients and fraction of whitened variance explained.
4. Oracle: a second fine-tuned copy. The activation is injected into its residual stream and the prompt
   specifies N and K. It is supervised on the teacher lists, then trained with RL, rewarded by the fraction of
   whitened variance explained, with small format penalties.
Inference: sample K phrases, map each through the reconstructor, then a non-negative least-squares refit.
**Result: "explains on average 31% of the whitened activation variance"** on held-out on-policy assistant
positions. Readouts at L67, N=4, K=10. Examples include "this dosage be toxic", "blackmail him by revealing",
"TypeError: dictionary changed". At most positions the outputs look like upcoming-text predictions; at
delimiters they read as "running commentary". Compared with natural-language autoencoders: they suspect it is
"less likely to produce confabulations" because the reconstructor is frozen and the reconstruction is a linear
combination of templates, but it "achieves considerably lower reconstruction accuracy". One interpretation:
"the oracle lens extracts only those representations that are 'in the workspace'". No compute figures are
given.

---

## 7. Capacity and broadcast numbers (`#struct-bottleneck`, `#struct-broadcast`, `#app-jfeatures`)

- Occupancy plateaus around 25 active J-lens vectors (median) in the workspace band. Occupancy is defined as the
  K at which the marginal reconstruction gain falls below that of a same-size random set. Excess variance
  explained is "never exceeding 10%".
- Related lists: nearly the whole 80-word family is present after a few items. Unrelated lists: about 6 at once
  (whole band). A new category evicts the old one "within a few words".
- Two held concepts share tokens at chance; a held concept and an ongoing computation almost never share a token
  (co-occupancy 0.09 vs 0.29 control) (`#app-competition`).
- SAE features: about 15% of the dictionary passes the lens-kurtosis bar (motor features excluded).
  Low-kurtosis features are "syntactic and surface-level bookkeeping".
- MLP gain on J-lens vectors is about 10x random mid-band; neuron output directions stay near 1. The summary post
  says "in some parts of the network by a factor of about a hundred". I did not find a 100x figure in the paper
  text; it may refer to a figure I could not read numerically (Fig 76-79 are interactive).
- Broadcast-head ablation (top 1% of heads): recall@25 of J-lens tokens falls to 0.67 vs 0.86 for the control;
  top-1 output changes at 5% of positions vs 2%.

---

## 8. Stated limitations and open problems (`#discuss-limitations`, `#discuss-alignment`)

- Single-token vocabulary: "the set of concepts the lens can name is exactly the set of concepts that have a
  single-token name". "prompt injection" shows up as two separate tokens. Some of the intervention failures are
  attributed to this.
- "Beyond a bag of concepts": "the J-lens treats the workspace as a flat collection of independently active
  concept vectors ... A readout containing spider, legs, and eight tells us those concepts are present, but not
  how they are bound together". Relations, roles and grammar are invisible to it.
- Inconsistent interpretability: some readouts cannot be made sense of, and they don't know whether that is
  averaging noise, multi-token concepts, or content they fail to recognize. Not characterized systematically.
- The workspace vs motor boundary is post hoc.
- No predictive criterion for which tasks route through the J-space.
- The role of early layers is unresolved (section 4).
- Scale and training dynamics: it is present in the base model before RLHF; emergence and scaling are unknown.
- No mechanism is known for what enters the J-space.
- Monitoring is not sufficient: automatic or well-practiced computation may bypass the J-space, and multi-token
  key concepts may not surface.
- Template lens and oracle lens limits: sections 5 and 6.
- Auditing-agent comparison (`#app-automated-auditing`): the J-lens at a single layer L71 matched NLA on 4/6
  cases but failed on two (reward hacking 4.5 vs 10.0; language switching 6.0 vs 9.9), where the claims needed
  "higher level of precision and abstraction" or a weak early signal.
- Summary post: "we'd be surprised if it's the whole story"; "it can only identify concepts that correspond to
  single tokens".

---

## 9. Cost and compute figures (all that the pages give)

- J-lens: 1000 prompts x 128 tokens by default; d_model backward passes per prompt, batched; one d_model x d_model
  matrix per layer; readout is one matmul per layer. It beats the baselines with 10 prompts.
- Template lens: about 12,700 words, "a few hundred forward passes per word".
- Oracle lens: two fine-tuned copies of Haiku 4.5; a 3.4M-phrase dictionary from about 1M positions; 1M teacher
  labels; supervised training plus RL. No GPU-hours.
- Reflection training: 10,000 task prompts; fine-tune on the reflection turn only.
- No wall-clock, GPU-hour or FLOP figures anywhere on either page.

---

## 10. Pointers

- Code: https://github.com/anthropics/jacobian-lens ("open source implementation of Jacobian lens training and
  inference", plus prompt data). Not opened.
- Open-model demo: https://www.neuronpedia.org/jlens. Not opened.
- External commentaries PDF (Dehaene and Naccache; Butlin et al.; Neel Nanda, with "an independent replication of
  some of our findings on an open-weight model"), linked from the summary post. Not opened.
- Related-work items relevant to steering vectors (`#related-work`): lens projection has been used "to interpret
  or sanity-check steering vectors". The page also cites "visualizing text-encoder intermediates through a
  diffusion decoder". Both are citations only; the papers were not fetched.
