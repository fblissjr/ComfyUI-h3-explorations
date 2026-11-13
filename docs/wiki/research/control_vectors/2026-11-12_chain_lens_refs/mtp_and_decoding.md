# MTP heads, contrastive decoding, and what can stand in for an MTP head

Notes for reading "The Chain Lens [interim]" (vgel, Raymond Douglas, LessWrong, 2026-11-11). The six papers
were fetched on 2026-11-12 from arXiv HTML (v1), with the PDF used for figures. Raw files are in
a session scratch folder (`refs_fetch_mtp`, not kept).
The LessWrong post itself was not fetched. Anything below about the chain lens comes from the brief
describing it, not from the post.

Sections 1 to 6 report only what each source says. Anything from prior knowledge is labelled
"(from memory, not in the source)". Section 7 is my own analysis.

Notation: "N=k" follows each paper's own convention. The papers do not share one; this is noted where it matters.

---

## 1. Gloeckle et al. 2024, "Better & Faster Large Language Models via Multi-token Prediction" (arXiv 2404.19737, FAIR at Meta)

### Method
- **Loss.** Next-token loss `L1 = -Σ_t log Pθ(x_{t+1} | x_{t:1})` (eq. 1) generalises to
  `Ln = -Σ_t log Pθ(x_{t+n:t+1} | x_{t:1})` (eq. 2). This is factorised through a shared latent
  `z_{t:1}`:
  `Ln = -Σ_t Σ_{i=1..n} log Pθ(x_{t+i} | z_{t:1}) · Pθ(z_{t:1} | x_{t:1})`.
  Each future token is therefore predicted **independently, given only the trunk latent**. Head i never sees tokens t+1..t+i-1.
- **Architecture.** A shared transformer trunk `f_s` produces `z_{t:1}`. There are n independent output
  heads `f_{h_i}`, each "implemented in terms of transformer layers" (a single transformer layer per
  head in the main runs; Appendix B says "a single transformer layer for the heads"). One unembedding
  `f_u` is shared:
  `Pθ(x_{t+i} | x_{t:1}) = softmax(f_u(f_{h_i}(f_s(x_{t:1}))))`, for i = 1..n. Head 1 is the ordinary next-token head.
- **Compute matching.** Adding n-1 head layers removes n-1 layers from the trunk, so parameter counts match.
- **Memory trick.** After the trunk forward pass, each head's forward and backward pass runs in turn,
  accumulating gradient at the trunk, so only one (n, V) logit block is alive at a time. Peak memory
  drops from O(nV+d) to O(V+d) "at no expense in runtime" (Fig. 2, Table S5).
- **Inference.** The basic use is to keep head 1 and discard the rest. Optionally the extra heads
  drive self-speculative decoding (blockwise parallel decoding, Stern et al. 2018; Medusa-style tree
  attention, Cai et al. 2024).
- **Alternative heads tried (Appendix B, Table S4):**
  - *Linear heads*: "a single linear layer without any nonlinearity ... amounting to linear probing of the model's residual representation z".
  - *Causal heads*: `P(x_{t+i}|·) = softmax ∘ f_u ∘ f_{h_i} ∘ f_{h_{i-1}} ∘ … ∘ f_{h_1} ∘ f_s` (each head stacked on the previous one).
  - *Anticausal heads*: the same stack in the reverse order, starting from the furthest token.
  - *Replicated unembeddings*: an n-fold unembedding of shape (d, nV), which the authors call "prohibitive" at scale. Their byte-level models used it.

  Table S4 caption: "no significant improvement with respect to Parallel architecture".

### What it needs
Pretraining from scratch with the auxiliary loss. Models run from 300M to 13B parameters on at least 91B
code tokens (§3.1), plus 7B models on 200B to 1T tokens (Table S13). The paper's training totals about
500K A100/H100 GPU hours ("Environmental impact").

### Measured results (all their own models trained from scratch unless noted)
- **Scaling (Fig. 3, §3.1):** "Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale" (300M to 13B, MBPP and HumanEval).
- **Abstract:** the 13B model solves 12% more HumanEval and 17% more MBPP problems than next-token models.
- **7B, 200B code tokens (Table 1):** n=4 is best on MBPP and HumanEval. MBPP pass@1 is 33.8 vs 30.0 for n=1. On APPS/Intro, n=6 leads.
- **Byte level (§3.3):** a 7B model on 314B bytes. 8-byte prediction solves 67% more MBPP pass@1 and 20% more HumanEval pass@1 than next-byte prediction.
- **Self-speculative decoding (§3.2, Table S2):** 3.0× on code with an average of 2.5 accepted tokens out of 3 suggestions, and 2.7× on text (7B, 4 heads). The 8-byte model reaches 6.4× (Table S3).
- **Linear vs transformer heads, 7B / 200B code tokens (Table S4, MBPP @1/@10/@100):**
  - n=1 baseline: 30.0/53.8/73.7
  - n=4 linear parallel: 33.6/55.0/76.2
  - n=4 transformer parallel: 33.8/55.9/76.9
  - n=4 causal: 31.9/54.9/74.9
  - n=4 anticausal: 30.8/54.8/75.3

  Linear heads came close to transformer heads on this downstream measure.
- **Finetuning an existing model (Llama 2) with 4-token heads (Table S6, 200B code tokens):** MBPP@1 was
  39.6 (n=1), 39.3 (n=4 linear), 38.3 (n=4 transformer) and 42.5 (n=4 transformer, "+Layers" 3). The
  authors' reading: "does not significantly improve performance ... this new loss changes the
  initialization too brutally and never really recovers". This measures the main model's benchmark
  score, not how accurate the added heads are.
- **Natural language (§3.7, Fig. 5):** at 7B / 200B tokens, n=2 matches the baseline on choice
  benchmarks and n=4 regresses a bit. Both n=2 and n=4 improve summarisation ROUGE-L (Fig. 6). On
  GSM8K, n=2 helps at 200B tokens, the pattern reverses at 500B, and n=4 is worse throughout (Fig. S13).
- **Induction (§4.1):** 2-token prediction vastly improves induction at 30M non-embedding parameters
  and below. The advantage disappears at 100M and above.
- **Related work:** the authors cite Pal et al. 2023 (Future Lens). They say it shows "next-token prediction models are able to predict additional consecutive tokens to a certain extent, but less so than our models which are specifically trained for this task."

### Limits the authors state
Benefits show up only at scale. The best n depends on the data distribution. Gains on choice and
likelihood benchmarks are absent or negative for natural language. Adding MTP to an already-trained
model by finetuning (Llama 2) did not clearly help. Their future work: choosing n automatically,
vocabulary size, and losses in embedding space.

---

## 2. DeepSeek-V3 Technical Report, §2.2 Multi-Token Prediction and Figure 3 only (arXiv 2412.19437)

### Method (eqs. 21 to 25)
- D **sequential** modules predict D additional tokens. The report contrasts this with Gloeckle's
  parallel independent heads: "we sequentially predict additional tokens and keep the complete causal chain at each prediction depth."
- Module k consists of:
  - a shared embedding `Emb(·)`
  - a shared output head `OutHead(·)`
  - its own Transformer block `TRM_k(·)`
  - its own projection `M_k ∈ R^{d×2d}`

  Both `Emb` and `OutHead` are "shared with the main model".
- For input position i at depth k:
  - `h'_i^k = M_k [ RMSNorm(h_i^{k-1}) ; RMSNorm(Emb(t_{i+k})) ]` (eq. 21; `[;]` is concatenation).
    For k=1, `h_i^0` is "the representation given by the main model". In Fig. 3 the arrow leaves the
    main model above its stack of L transformer blocks and below its Output Head.
  - `h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})` (eq. 22). The block runs over the whole sequence of combined
    states. In Fig. 3 each MTP module takes a shifted token row (t2..t5 for module 1, t3..t6 for
    module 2), so its attention is ordinary causal attention over the positions at that depth.
  - `P^k_{i+k+1} = OutHead(h_i^k)` (eq. 23). `OutHead` is a linear map to logits followed by a softmax.
- In words: depth k predicts token i+k+1 from the depth-(k-1) state at i plus the **embedding of the
  true token i+k**. Each module is a conditional next-next predictor, and its input includes the
  token the previous depth predicted (the ground-truth token during training).
- **Loss:** `L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1}) = -(1/T) Σ_{i=2+k}^{T+1} log P^k_i[t_i]` (eq. 24), and
  `L_MTP = (λ/D) Σ_{k=1..D} L^k_MTP` (eq. 25). This is added to the main loss.
- **Settings for DeepSeek-V3:** D = 1. λ = 0.3 for the first 10T tokens and 0.1 for the remaining 4.8T tokens (14.8T pre-training tokens in all).
- **Inference:** "we can directly discard the MTP modules and the main model can function independently and normally". The modules can also be repurposed for speculative decoding.
- **Parameter sharing in practice (§3.2):** the shallowest layers (embedding) and deepest layers
  (output head) sit on the same pipeline rank, so the shared embedding and head are physically shared with the MTP module.
- **Lineage:** the report compares its causal-chain principle to EAGLE (Li et al. 2024b), "but its primary objective is speculative decoding".

### Measured results
- **Ablation (Table 4):** a 1-depth MTP module added to two MoE baselines, with inference cost
  identical because the module is discarded.
  - Small MoE (15.7B total parameters, 1.33T tokens): HumanEval 20.7 → 26.8, GSM8K 25.4 → 31.4, MMLU 50.0 → 53.3, NaturalQuestions 22.7 → 22.3.
  - Large MoE (228.7B total parameters, 540B tokens): HumanEval 44.5 → 53.7, MMLU 67.5 → 66.6.

  The report's summary: MTP "consistently enhances the model performance on most of the evaluation benchmarks."
- **§5.4.3:** acceptance rate of the second-token prediction is "between 85% and 90% across various generation topics". This gives 1.8× tokens per second with speculative decoding.

### Limits stated
None specific to MTP in the section read.

---

## 3. Li et al. 2022/2023, "Contrastive Decoding: Open-ended Text Generation as Optimization" (arXiv 2210.15097)

### Objective
- Sequence level: maximise `L_CD = log p_EXP(x_cont | x_pre) - log p_AMA(x_cont | x_pre)`, subject to a plausibility constraint under the expert.
- Token level (eq. 1), searched with beam search (beam 5):
  ```
  CD-score(x_i; x_<i) = log p_EXP(x_i|x_<i) - log p_AMA(x_i|x_<i)   if x_i ∈ V_head(x_<i)
                      = -inf                                          otherwise
  ```
- Adaptive plausibility constraint (eq. 2):
  `V_head(x_<i) = { x_i ∈ V : p_EXP(x_i | x_<i) ≥ α · max_w p_EXP(w | x_<i) }`, with **α = 0.1 throughout**. The authors call it "quite robust and generalizes across various domains."

### Why the constraint is needed (§3.3)
- **False positive.** An implausible token can win the raw contrast. Their example is "NetMessage":
  p_EXP = 3e-9 and p_AMA = 8e-14 give a contrast of 10.6. That beats plausible tokens such as "1961" and "Hawaii".
- **False negative.** An easy token that both models are sure of gets almost no contrast. Example:
  the subword "#orn" after "unic" has p ≈ 0.99 under both, so the contrast is 6e-4. That is lower than
  bad continuations.
- **How the constraint fixes both.** It uses the expert's confidence relative to its own maximum. When
  the expert is confident, the pool can shrink to one token, "which removes the impact of the
  contrastive objective". The expert spreads mass over alternatives for tokens that reflect amateur
  failure modes, so those are not protected.
- **Ablation (§7.4, Table 4, GPT-2 XL):** removing V_head gives MAUVE 0.01, coherence 0.23 and PPL 2e5. With the constraint the figures are MAUVE 0.92, coherence 0.69 and PPL 17.77 ("severe fluency issues").

### Choice of amateur (§3.4, §4.3, §7)
- **Scale.** Use the smallest model of the same family. A larger expert/amateur gap is better (Fig. 2).
  Identical expert and amateur (the diagonal of Fig. 2) gives "highly repetitive text ... because we
  cannot exploit any contrast between two identical LMs". A trigram amateur is poor (MAUVE 0.73)
  because its mistakes "do not highlight failure modes of an expert LM". The amateur must "commit similar failure modes as the expert".
- **Temperature τ of the amateur.** Low τ sharpens the amateur's mode and penalises it more. High τ flattens it toward uniform, so repetition stops being penalised. τ in [0.5, 1.0] is robust. They used 0.5 for GPT-2 and 1.0 for OPT.
- **Context window.** The amateur can be weakened by conditioning it only on the last prompt token.
  Giving the amateur the full prompt "slightly hurts coherence and fluency" (§7.5).
- **Reductions (§4.3):**
  - A uniform amateur gives plain likelihood maximisation.
  - An n-gram amateur fitted to the prefix gives soft n-gram blocking.
  - **The same LM as both expert and amateur, with the amateur's context restricted, "is equivalent to
    the MMI decoding objective"**. That objective maximises the pointwise mutual information between prompt and continuation.
- **Interpretation (§4.1):** the argmax also maximises PMI(x_cont, I=1). This is "text that is most distinguishable as having originated from the expert".
- **Search vs sample (§7.3):** sampling from softmax(CD-score) is worse than searching, but still "reasonably good".

### Results
Experts were GPT-2 XL (1.5B), OPT-6.7B and OPT-13B. Amateurs were GPT-2 small (about 100M) and
OPT-125m. Domains were Wikinews, WikiText-103 and BookCorpus, with 32-word prompts and 256 generated tokens.
- **Automatic metrics (Table 1):** CD beats nucleus, top-k, typical and SimCTG on MAUVE and coherence. Its diversity is comparable to or slightly below nucleus and typical sampling.
- **Human evaluation (Table 2, §6.2):** for coherence, evaluators preferred CD 2.6× more than nucleus and 6.4× more than typical. For fluency the figures are 1.4× and 3.5×.

### Limits stated
No limitations section was found in the HTML (searched). The future-work line names other amateurs, such as an early checkpoint.

---

## 4. Pal et al. 2023, "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State" (arXiv 2311.04897, CoNLL 2023)

### Setup and notation
- The model is GPT-J-6B, with 28 layers, d = 4096 and |V| = 50,400. `G(x) = D(b_L(…b_1(E(x))))`. `h_T^l` is the layer-l state at the last prompt token.
- The goal is to anticipate what G will predict for y_{T+1}..y_{T+N}, using **only** `h_T^l`.
- Index convention: N=0 is the ordinary next token (`y_T` predicts x_{T+1}). **N=1 predicts x_{T+2}**, the
  first token past the next one (caption of Table 3). Targets are the model's own greedy continuation,
  not ground truth.

### The four methods
1. **Direct vocabulary prediction (§2.1, "Vocab").** A linear model `ẑ_{T+N} = g_θ(h_T^l)`, then `ŷ = softmax(ẑ)`. It maps d → 50,400 directly.
2. **Linear model approximation (§2.2, "HS").** A linear model `ĥ^L_{T+N} = f_θ(h_T^l) ≈ h^L_{T+N}`. It predicts the **final-layer hidden state at a future position**, decoded with the pretrained decoder head. It extends tuned lens across positions as well as layers. The authors say reusing the decoder head "marginally aids predictions at the latest layers".

   For both linear methods, one instance is trained per layer (28) and per N (0..3). Training uses 100,000 tokens of Pile context, with an average context of 518.
3. **Fixed-prompt causal intervention (§2.3, "Fixed"). No training.**
   - Run a generic prompt `c = [c_1..c_M]`. Replace its last-position state at the same layer with the source state, `h_M^l := h_T^l`, and let the model continue.
   - For later steps, append the generated tokens: `ŷ_{M+i} = G(c + [x_{T+1}..x_{T+N}] || h_M^l := h_T^l)` (eq. 9).
   - The four prompts tested were:
     - `Hello! Could you please tell me more about "`
     - `The multi-tokens present here are "`
     - `The concepts in this hidden state listed are: (`
     - `<|endoftext|> This state is describing about the following concept:`
   - The authors note: "some prompts are more amenable to this intervention than others".
4. **Learned-prompt causal intervention (§2.4, "Learned"; the main method).**
   - Same patch as Fixed, but the context is a trained soft prefix `c_opt^(l) = [c_1..c_M]`, one per layer l. Training minimises `KL(ŷ_{M+N} ; y_{T+N})` (eq. 10), where
     `ŷ_{M+n} = G([c_1..c_M, x_{T+1}..x_{T+N}] || h_M^l := h_T^l)` (eq. 11).
   - The model is frozen and only the prefix is trained. The gradient goes "through the model instead of a temporary MLP" (unlike prefix tuning), which "produces a significantly better optimized context".
   - Prefix length is 10. There is one prefix per layer for layers 1..28. Training used 10,000 of the 100,000 samples.
   - The main models are trained for N=1. Appendix A: "optimizing for N=1 works best and generalizes surprisingly well to other N, but ... optimizing for other N does not perform well for N=1."
   - The patch is applied once, at the last soft-prompt position. Later tokens see the patched state only through attention.

### Data and selection
Train and test samples come only from positions where GPT-J predicted the next token correctly. The test set is 1,000 tokens with an average context of 535.

### Results (GPT-J-6B)
- **Table 2, best layer for each method, Precision@1 for four columns.** The HTML and PDF header label
  only N=1, N=2, N=3 above four numeric columns. The first column must be N=0: the abstract's "more
  than 48%" matches the second column's 48.4, and §3.4 reports the bigram baseline "at N=1".

  | Method  | N=0  | N=1  | N=2  | N=3  |
  |---------|------|------|------|------|
  | Learned | 97.0 | 48.4 | 43.7 | 46.9 |
  | Fixed   | 97.0 | 20.8 | 30.0 | 36.5 |
  | HS      | 98.0 | 29.2 | 19.0 | 15.8 |
  | Vocab   | 85.7 | 27.5 | 19.4 | 14.7 |

  Surprisal at N=1 is Learned 4.5, Fixed 8.8, HS 14.1 and Vocab 15.3. A bigram baseline built from 900,000 Pile documents gets 20.1% at N=1.
- **Layer profile (Fig. 4, read off the plot, approximate):**
  - For N=0, every method rises toward the last layer.
  - For N=1, the learned prompt sits around 0.4 to 0.48 and peaks at middle layers; Table 3 uses layer 14 as the peak. It falls to about 0.3 by layer 28.
  - The linear models stay around 0.15 to 0.3 and are best in late layers (about 20 to 25).
  - The fixed prompt sits around 0.15 to 0.2 at N=1, and higher at N=2 and N=3 (about 0.3).
  - The authors conclude: "predictive accuracy of the learned prompt model peaks at the middle-layer hidden states ... very different from the immediate next-token N=0, in which accuracy peaks at the last layer."
- **By token type (Table 3, layer 14, N=1):**
  - The learned prompt is better than the linear models in every category except Uppercase No Space (Learned 23.3 vs Vocab 29.2).
  - Its gain is largest when the last context token is lowercase with a leading space (51.9 vs 26.4 or 20.8) or at least 4 characters long (52.1). The authors read this as: "information about how to complete long words may not be immediately accessible by a linear model decoder, but ... can be made accessible by using the parameters of the pretrained model".
  - Note: the prose says "all categories ... are predicted better by the learned prompt than by the linear model", which the Uppercase No Space row contradicts.
- **Calibration (§3.5):** at N=1 the learned prompt's accuracy is 26%, 57%, 77% and 95% for next-token confidence bins of 0–30, 30–60, 60–90 and 90–100%. Hence "predicted future tokens are generally accurate when the model is confident."
- **Named entities alone (§3.5):** 44%, 42% and 37% for N=1, 2, 3, which is similar or slightly lower. "Future information is present broadly, not only for long entity names."
- **Future Lens tool (Fig. 6):** the learned-prompt intervention applied at every (layer, token) cell, showing four anticipated tokens per cell with confidence shading.

### Limits stated (Limitations section)
- A small English training set (100,000 samples) and one model (GPT-J-6B).
- No prior baselines beyond bigram and linear; RNN and non-autoregressive baselines are not tried.
- Only up to 4 tokens ahead (N = 0..3).
- §2.4 adds that a failed fixed-prompt intervention "does not necessarily mean that the hidden state does not encode similar information; it may just be less prominent."

---

## 5. Ghandeharioun et al. 2024, "Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models" (arXiv 2401.06102)

### Framework (§3.1)
- **Source.** Prompt S, position i, model M (L layers), layer ℓ, giving `h_i^ℓ`.
- **A Patchscope is the quintuple `(T, i*, f, M*, ℓ*)`:**
  - T: the target prompt
  - i*: the target position
  - f: a mapping `R^d → R^{d*}`. It can be the identity, a learned linear or affine map, or "even more complex functions".
  - M*: the target model
  - ℓ*: the target layer
- **Operation.** Run M* on T. Replace `h̄_{i*}^{ℓ*} ← f(h_i^ℓ)` and continue the forward pass, generating text if desired.
- **Caveat stated by the authors.** The computation after the patch "can augment the representation with additional information, hence, this approach does not guarantee that the patched representation itself stores all that information". Detaching it from S does guarantee no further information from the source prompt enters.
- **Prior methods as instances (Table 1, §3.2):**
  - Logit lens: ℓ* = L*, f = identity.
  - Tuned lens: ℓ* = L*, f = affine.
  - Linear shortcuts (Din et al.): f = linear.
  - Future Lens: M* = M, ℓ* = ℓ, T = a fixed or learned soft prompt, with T ≠ S.
  - Causal tracing: T = S + noise.
  - Attention knockout: f = 0.
  - Linear caption mapping: a different M*.

### Multi-token decoding setups
- **Token identity Patchscope (§4.1).** T = `tok1 → tok1 ; tok2 → tok2 ; … ; tokk` (few-shot identity demonstrations), with ℓ* = ℓ and f = identity. No training. They used k=3 random tokens. Five random realisations were stable, "particularly notable in the upper layers" (App. A.3).
  - **Results (Fig. 2; LLaMA2-13B, Vicuna-13B, GPT-J-6B, Pythia-12B):** from layer 10 upward it beats logit lens and tuned lens on next-token Precision@1 and surprisal, with "a gain of up to 98% in layers 18-22".
  - In the first 10 layers it is on par with logit lens and worse than tuned lens. Tuned-lens maps were fit by least squares on 10,000 Pile examples, with 2,000 for evaluation.
- **Zero-shot feature extraction (§4.2).** T is a relation verbalisation followed by placeholder x (e.g. "The largest city in x"), with i* = the x position. All (ℓ, ℓ*) pairs are swept, generating up to 20 tokens. A sample counts as correct if any ℓ* yields the object.
  - **Results (GPT-J, 12 tasks, 1,453 datapoints, Table 2):** significantly better than a logistic-regression probe on 6/12 tasks and worse on 1.
- **Entity description (§4.3).** T = `subject1: description1, …, subjectk: descriptionk, x` with k=3, patched at the last position, ℓ* = ℓ, for the first 10 layers of Vicuna-13B and Pythia-12B. Generations are scored with RougeL against Wikipedia.
  - **Results:** similarity rises over the first 5 layers and then slowly falls (Fig. 4).
- **Cross-model (§4.4).**
  - Vicuna 7B → 13B and Pythia 6.9B → 12B, with affine maps learned between layers.
  - Next-token Precision@1 reaches 0.7 (Vicuna) and 0.8 (Pythia). "Patching representations to an early layer of the larger model seems to be the most effective", with a diagonal layer correspondence.
  - Cross-model patching helps entity description for Vicuna but not for Pythia, where the smaller model already outperformed the larger.
- **CoT Patchscope (§5).** Multi-hop correction on Vicuna-13B: 19.57% → 50% on 46 samples.
  - Best target layers ℓ* are 6–16, and ℓ* ≤ ℓ is "more successful on average".
  - Implementation detail (App. E): the configuration is equivalent to S = T with an attention mask adjusted so the two halves cannot see each other. **This is the same attention-restriction device the chain lens uses.**

### Layer choices and failure modes (as stated)
- **Early source layers.** All next-token methods are weak below about layer 10, "as the first layers contextualize the input".
- **Late source layers.** Attribute extraction declines because "the representations shift toward next-token prediction in the later layers", so the attribute "would not be as readily accessible".
- **Late target layers (ℓ*).** These do worse. The authors attribute this to placeholder contamination: the placeholder token's own representation persists in the early layers of the target run, and in multi-token generation later positions still attend to it. In the Vicuna "Paris Hilton" example, from layer 7 onward the output describes "x" itself ("Placeholder for a variable or concept"; App. C.3). They leave its control to future work.
- **Cross-model limits.** Cross-model patching is shown only within one model family, with an affine map.
- **Future work.** Simultaneous multi-token patching, and task-specific vs task-agnostic recipes.
- No separate limitations section exists (searched).

---

## 6. Karvonen et al. 2025, "Activation Oracles" (arXiv 2512.15674): short paragraph

An Activation Oracle (AO) is the target LLM itself, finetuned with LoRA to take activations as input and
answer natural-language questions about them. This is the LatentQA approach, here trained for generality.

**Input.** The oracle prompt is `Layer ℓ: <ACT> … <ACT> question`, with " ?" as the placeholder token.

**Injection.** At layer 1 of the oracle, each placeholder's residual is **added to, norm-matched**:
`h'_i = h_i + ||h_i|| · v_i / ||v_i||` (eq. 1). The authors switched from replacement because replacement
caused activation-norm growth of about 20× at layer 0 and about 100,000× at layer 1 during training
(App. A.5). Layer 1 beat layer 0 by 1–11% on Llama-3.3-70B, and made no difference on Qwen3-8B and Gemma-2-9B.

**Training data.** About 1M examples from three sources:
- system-prompt QA
- 7 binary classification tasks recast as yes/no questions
- a **self-supervised context-prediction task**: given K activations, predict the previous or next J tokens, with K, J ~ U(1, 20), on a 50/50 mix of pretraining and chat text

Activations come from 25%, 50% and 75% depth of the base model, and evaluation uses 50%. Layers at 33%
and 66% also worked; 0% and 10% were "substantially worse" (App. C.5).

**Cost.** LoRA with rank 64, alpha 128, all linear layers, lr 1e-5 and batch 16.
- Qwen3-8B: about 10 H100-hours on one GPU.
- Gemma-2-9B: about 12 H100-hours.
- Llama-3.3-70B: about 90 H200-hours.
- Claude Haiku 3.5 was also trained.

**Results (abstract).** The best AOs "match or exceed prior white-box baselines on all four tasks and are
the best method on 3 out of 4".

**Limits (§8).**
- Black-box answers with no mechanism exposed.
- "Frequently makes incorrect guesses", and is uncalibrated because it is not trained to express uncertainty.
- Small, narrow training data.
- Narrow benchmarks.
- Simple prompting sometimes matches it (hallucination detection, App. F).

---

## 7. My own analysis (not from any source)

Everything in this section is my inference. Labels: [derived] means it follows from the equations
above; [judgement] means an untested opinion.

### 7.0 What an MTP head computes, in the form the chain lens needs
- **The DeepSeek-style module, which is what the post's figure adapts** [derived from eq. 21–23]. It
  computes `p(t_{i+2} | h_i^final, t_{i+1})`. That is a conditional next-next-token distribution. Its
  inputs are a final-layer state and the embedding of a chosen next token. It contains one transformer
  block whose attention normally covers the preceding positions' combined states.
  - Rolled out, the output state `h_i^1` becomes the `h^{k-1}` for the next step. This works for k > D only if the same module is reused past its trained depth.
  - In the brief's description, the chain lens restricts the block's attention to the chain and re-injects the transported state every step. Both are departures from how the module was trained: in training it sees a full causal prefix of real states.
- **The Gloeckle-style head.** It computes `p(x_{t+i} | z)` with no conditioning on intermediate
  tokens. It yields a per-offset marginal, not a chain. With a Gloeckle-style head you cannot ask
  "given the lens picked token A, what follows A".
- **So a substitute must supply two things.** First, a way to read a *future-position* final state from
  one hidden state. Second, ideally, a way to condition on the chosen next token. Methods that supply
  only the first give bags of per-offset tokens, not phrases.

### 7.1 Substitutes for a missing MTP head, ranked by training needed

The reader's setting: a Qwen-family VLM text encoder, a tap at layer 49 of 64, steering vectors added to
conditioning tokens, and an existing J-lens. Whether that encoder ships an MTP layer was not checked.
(From memory, not in the source: Qwen3-Next and some later Qwen releases ship an MTP layer, and the
Qwen2.5-VL and Qwen3-VL releases I know of do not. Confirm by listing the checkpoint's tensor names
for an `mtp`/`nextn` prefix.)

The encoder is still a causal LM with an LM head, so every substitute below can reuse its own weights.

**(a) No training: fixed-prompt patching (Future Lens §2.3 / Patchscopes).**
- **How.** Patch the state at layer 49 (the vector's native layer, so no J-lens transport is needed)
  into the last position of a generic prompt. Run layers 50–64 and the LM head, and decode greedily or
  contrastively. To condition on the J-lens's top token, append it to the target prompt and decode
  onward (eq. 9 does exactly this).
- **What it is.** "The model itself as the MTP head". It is the only zero-training option that conditions on the chosen next token.
- **What it loses vs a native head:**
  - Accuracy. On GPT-J, Fixed reaches 20.8% at N=1 against 48.4% learned; the source does not measure a native head.
  - Sensitivity to the choice of prompt.
  - Placeholder or prompt contamination (Patchscopes App. C.3).
  - The patched state is seen once and then diluted by attention to the prompt. Re-injecting it at
    every step, as the chain lens does, may or may not help here; that is untested.
  - Cost: each step runs 15 layers instead of one block.
- **Steering vectors.** A steering vector is a direction, not a state. [judgement] Patch it as
  `mean_state + v` or `base_state + v`. Activation Oracles' norm-matched additive injection is the closest
  documented recipe, and their reason for addition (replacement blew up norms) is a warning against
  patching a raw direction by replacement.

**(b) No training: token-identity Patchscope continued past one token.**
- **How.** Same mechanics as (a) with the identity few-shot prompt. That prompt is built to make the
  model echo the patched token's identity, so continuing generation tends to give that token's usual
  continuation.
- **[judgement]** It is better suited to "what single token is this" than to "what phrase follows". Patchscopes measured it only for next-token identity.

**(c) A linear fit: Future Lens HS or Vocab maps.**
- **How.** Fit `h_49 → h^final_{T+N}` per offset N, e.g. by least squares as Patchscopes did for tuned lens, then decode with the LM head.
- **Cost.** Cheapest training: one least-squares solve per (layer, N), on about 10^5 tokens in the source.
- **What it loses:**
  - It is a marginal per-offset predictor. It does not condition on the chosen first token, so it gives no chains.
  - It is weak beyond the next token. On GPT-J at N=1 it scores 27.5–29.2% against a 20.1% bigram floor, and falls further at N=2 and N=3.
- **[derived] A conditional linear variant.** Regress `h^final_{T+1}` on `[RMSNorm(h^final_T) ; RMSNorm(Emb(x_{T+1}))]`. This is the DeepSeek module with its transformer block deleted, i.e. just `M_k`. It would give chains at linear cost. No source tests it, and it cannot attend.

**(d) A soft prompt: Future Lens learned-prompt intervention.**
- **How.** Train one 10-vector prefix for layer 49 with the model frozen, using a KL objective against the model's own continuation, on about 10^4 positions.
- **Strengths.**
  - It conditions on the generated tokens (eq. 11), so it yields chains.
  - It uses the model's own remaining 15 layers, so it is the best-measured substitute: 48.4% at N=1 on GPT-J, with accuracy tracking model confidence.
  - Gradients go through 15 layers only, because the patch sits at layer 49.
- **Caveat.** On GPT-J the learned prompt peaks mid-depth and is lower at late layers (about 0.3 at
  layer 28 of 28, read off Fig. 4). Layer 49 of 64 is about 77% depth. The GPT-J analogue is about
  layer 21–22, where the N=1 curve is still near 0.4. This mapping across models is untested.
- **What it loses vs a native head:**
  - It was trained on in-distribution states. Steered states (base + v) are off-distribution.
  - The trunk was not trained to pre-plan. DeepSeek's stated motive is that MTP lets the trunk "pre-plan its representations"; Gloeckle cites Pal et al. as showing next-token-trained models encode the future "less so".

**(e) A small trained head: a DeepSeek-style MTP module on a frozen trunk.**
- **How.** Train one `M (d×2d)` plus one transformer block, sharing the embedding and LM head, with CE
  on `t_{i+2}` given `(h_i^final, Emb(t_{i+1}))` (eq. 21–24). Train it on the encoder's own outputs on a
  text corpus, with the trunk frozen.
- **Strengths.** It reproduces the native interface exactly, so the chain lens recipe would apply unchanged.
- **Cost.** Training one block's worth of parameters on a large token stream. (From memory, not in the
  source: EAGLE and Medusa train such draft heads on frozen trunks cheaply.)
- **What it loses vs a native head:**
  - The trunk was not co-trained, so less future information may be present to read.
  - Gloeckle's Table S6 says nothing about head accuracy on a frozen trunk; it measured something else.
- **Placement.** It is fed the final state, so it needs the J-lens transport from layer 49 as in the chain lens. Alternatively, train it directly on layer-49 states, which removes the transport step.

**(f) Heaviest: an Activation Oracle.**
- **How.** A LoRA over the whole encoder, trained on about 1M examples. Their self-supervised task
  "predict the next J tokens after these activations" is a multi-token future readout. Cost is about
  10 H100-hours at 8B; the source does not give the cost at the reader's model size.
- **Strengths.** It answers free-form questions, not only continuations.
- **What it loses:**
  - Its answers are uncalibrated, by the authors' own account.
  - It is a black box.
  - It gives no token-level distribution comparable to a lens, so contrastive decoding against a mean state is awkward. It does produce token distributions, so the contrast is computable but not what it was trained for.

**Order by training needed:** (a), (b) none < (c) linear fit < (d) soft prompt < (e) small head < (f) LoRA oracle.

**[judgement] Recommended first try for the reader:** (a) at layer 49, then (d) if (a) reads as generic.
Both reuse weights the encoder already has, and both condition on chosen tokens.

### 7.2 The plausibility constraint when the amateur is the same head on a corpus-mean state

1. **This is CD's "same model, restricted context" case, not the "smaller model" case** [derived from §4.3].
   - In CD the MMI reduction is the same LM with the amateur denied the context. Here the amateur is the
     same head denied the state's specific content (the mean state carries none).
   - So `log p(x | h) − log p(x | h_mean)` is a PMI-style score: it ranks how much the state, beyond a
     generic state, raises each token.
   - CD's warning about identical expert and amateur (repetitive output, Fig. 2 diagonal) does not apply,
     because the inputs differ. It does return at later chain steps if the chain tokens come to dominate
     both runs and the two distributions converge.
2. **With a shared linear unembedding, the contrast is nearly a mean-centred lens** [derived].
   - Within one step the log-softmax normalisers are constant across candidate tokens. So ranking by
     `log p_exp − log p_ama` equals ranking by the logit difference `W_U(o_exp − o_ama)`, where o is the
     head's pre-unembedding output after its final norm.
   - If the head were linear, this would be the lens applied to `(h − h_mean)`. The RMSNorm scale and the
     head's nonlinearity break exact equality.
   - Implication: the reader's J-lens readout of a steering vector (itself a difference) is already
     contrastive in this linear sense. What it lacks is the plausibility filter.
3. **The constraint is needed more, not less, with a corpus-mean amateur** [judgement, from §3.3's false-positive mechanism].
   - A corpus-mean state predicts frequent, generic tokens. Rare tokens get tiny amateur probability.
   - So any rare token with a slight lift under the steered state gets a huge ratio. This is the "NetMessage" failure, likely more often than with a small-model amateur, which at least shares the context.
   - CD without the filter scored MAUVE 0.01.
4. **The constraint's weak spot: a flat expert** [derived from eq. 2].
   - `α · max p_exp` is relative. When the steered state gives a flat expert distribution (low max), the
     bar drops and many low-probability tokens pass. A steered or off-manifold state plausibly does this; untested.
   - Options [judgement]: an absolute floor on p_exp, a top-k cap on V_head, or reporting the size of V_head beside each chain so a flat step is visible.
5. **False negatives behave as CD intends, if the amateur also sees the chain tokens** [derived].
   - Subword completions ("unic" → "orn") are near-certain under both. The constraint shrinks V_head to that one token and the chain continues.
   - If the amateur is run on the mean state *without* the chain tokens, those completions get a large contrast instead. They still pass, but the contrast scores at those steps then mean less.
   - Decide explicitly which the post does. The brief says only "decoding contrastively against a corpus-mean hidden state".
6. **Amateur temperature is a free knob** [from §3.4 and §7.2 of the source].
   - A low τ on the mean-state run penalises its mode, the generic tokens, harder. A high τ turns the amateur toward uniform and the method back into plain greedy decoding.
   - CD's robust range is 0.5–1.0, measured with a small-model amateur. It is untested for a mean-state amateur.
7. **The filter cannot catch a wrong expert** [derived]. V_head only guards against tokens the expert
   itself finds implausible. If the substitute head is poor (Fixed at 20.8% at N=1 on GPT-J), contrast plus
   filter will confidently amplify its errors. Future Lens's calibration result suggests reporting the
   expert's max-probability per step as a confidence column.
