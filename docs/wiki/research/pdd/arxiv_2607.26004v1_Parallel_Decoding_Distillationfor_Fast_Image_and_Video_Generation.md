---
title: "Parallel Decoding Distillationfor Fast Image and Video Generation"
source: "https://arxiv.org/html/2607.26004v1"
site: "arxiv.org"
clipped: 2026-08-26
---

## Parallel Decoding Distillation for Fast Image and Video Generation

Neta Shaul    Chao Liu    Arash Vahdat    Julius Berner  
NVIDIA    Weizmann Institute of Science    <sup>†</sup> equal advising

###### Abstract

Generation in video diffusion or flow models is computationally expensive due to the slow and iterative sampling process. Current state-of-the-art (SOTA) acceleration methods heavily rely on variational score distillation (VSD) and adversarial losses to distill diffusion models into few-step generators. Albeit achieving high-quality video generation, these training losses are notoriously hard to optimize and suffer from mode collapse, leading to loss of video diversity and lack of motion. In this paper, we introduce Parallel Decoding Distillation (PDD), a simplified and scalable trajectory-based distillation method for fast inference of diffusion and flow matching models. Our architecture and training procedure are compatible with any pre-trained model and support sampling with a varying number of function evaluations (NFE). PDD accelerates generation by predicting multiple denoising steps per network evaluation. Conceptually, it learns a representation of the mean velocity without regressing its derivative using JVPs or finite-difference approximations. Our method achieves SOTA performance with 4-8 NFE on LTX-2.3 Text-to-Video/Audio, Wan 14B Text-to-Video, and Qwen-Image Text-to-Image. Moreover, PDD presents a significant improvement in generated video diversity. Project page: [https://research.nvidia.com/labs/genair/pdd](https://research.nvidia.com/labs/genair/pdd)

## 1 Introduction

Large-scale diffusion and flow models [^21] [^51] [^29] [^30] [^1] have achieved remarkable capabilities of media generation, including text-to-image [^61] [^26], text-to-video [^25] [^56] [^40], and multi-modal generation [^19]. Yet, the cost of generation remains high, as their inherently iterative sampling algorithms often require hundreds of network evaluations. The resulting computational cost and latency are one of the main bottlenecks for many applications, such as content editing, real-time video generation, and interactive world modeling. Thus, developing distillation methods for few-step media generation models has become an active field of research [^47] [^66] [^64] [^32] [^27] [^7] [^41] [^70] [^55] [^39] [^17] [^63].

The approaches for distilling diffusion and flow models are broadly categorized into two families: i) *trajectory-based* methods [^45] [^52] [^12] [^44] [^14] [^7] in which a student model distills the many-step sequential sampling process of a pre-trained teacher into a few-step process; and ii) *distribution-based* methods [^46] [^59] [^67] [^64], which relax the constraint of following the teacher trajectories and instead align only the marginal distributions of the student and teacher processes. Trajectory-based methods have shown promising results for fast image generation. However, when applied to video models, they are typically bottlenecked by degraded video quality and costly training algorithms. As a result, the current dominant methods for distillation of video models are distribution-based [^11] [^39]. While recent works regularize the training dynamics by additionally incorporating trajectory-based distillation losses [^70] [^17], they still suffer from alternating training objectives, high memory requirements, and mode collapse, leading to reduced diversity and often static videos.

In this paper, we introduce *parallel decoding distillation* (PDD), a trajectory-based distillation method for fast inference of diffusion and flow models. Instead of merging multiple denoising steps into a single larger step [^45] [^12] [^32] [^72], we learn a parallel decoder that predicts multiple denoising steps in a single network evaluation. Specifically, PDD discretizes the time domain of the flow into a fixed sequence of $N$ intervals, which are grouped into blocks of size $L$. During training, the parallel decoder learns to predict the mean velocities in all intervals within a block using a single forward pass, as illustrated in Figure 2. Target mean velocities are approximated using a Runge-Kutta solver (e.g., Euler or Midpoint) applied to the pre-trained teacher model. At generation, by predicting the velocities across $L$ intervals, we obtain a sample with $N/L$ steps. By varying the block size during training, PDD supports sampling with different number of function evaluations (NFEs) during inference.

Figure 2: The sampling trajectory is discretized into $N$ intervals, which are grouped into blocks of size $L$. The parallel decoder predicts the mean velocities for all intervals within a block using a single evaluation.

PDD is conceptually related to Pi-Flow [^7], but provides a simplified training algorithm that removes the need for an additional policy head and supports a variable NFEs at generation time. Additionally, our formulation clarifies the connection to flow map distillation methods, particularly the Lagrangian formulation [^72] [^4]. By avoiding Jacobian-vector products (JVPs) and finite differences, we obtain minimal-cost training algorithm.

We showcase the efficacy of our method on text-to-video tasks with Wan2.1 [^56] 1.3B and 14B models, achieving SOTA video quality generation with $4$ NFE on VBench [^24] while preserving better video diversity compared to distribution-based baselines. Additionally, we test PDD on text-to-image tasks with the Qwen-Image 20B model, achieving SOTA scores with 4 to 8 NFE on OneIG [^6], GenEval [^16], and DPG-Bench [^22] benchmarks.

All in all our main contributions are:

1. Formulate Parallel Decoding Distillation, a scalable, trajectory-based distillation method for fast inference of flow matching and diffusion models.
2. Develop a single, regression-based training objective that avoids the need for JVPs, finite differences, or multi-stage training procedures and produces high-quality and diverse samples without VSD or GAN losses.
3. Propose a simple architecture and training algorithm that are supported by any pre-trained model and allows generation with varying NFE without additional time conditioning.
4. Validate our method on ImageNet-256, Qwen-Image, Wan2.1 1.3B/14B, and LTX-2.3, achieving SOTA performance with improved diversity.

## 2 Generative Flow Models

#### Notation

The state space is denoted by ${\mathcal{X}}$, where ${\mathcal{X}}=\mathbb{R}^{c\times h\times w}$ for images and ${\mathcal{X}}=\mathbb{R}^{f\times c\times h\times w}$ for videos. Random variables taking values in ${\mathcal{X}}$ are denoted by the uppercase letter $X$, and states in ${\mathcal{X}}$ are denoted by the lowercase letter $x$. Fixed integers are denoted by the uppercase letters $N,L\in{\mathbb{N}}$, while running indices or integer-valued random variables are denoted by the lowercase letters $n,k$.

#### Flow Matching and Diffusion Models

The currently dominant approaches for training generative flow models are flow matching [^29] [^30] [^1] and diffusion models [^49] [^21] [^51]. As we are interested only in deterministic processes, for simplicity, we treat both as flows (1).

A flow process $\left(X_{t}\right)_{0\leq t\leq 1}$ taking values in ${\mathcal{X}}$ is defined by a *velocity field* $v:{\mathcal{X}}\times[0,1]\rightarrow{\mathcal{X}}$ and a *source distribution* $p_{0}:{\mathcal{X}}\rightarrow\mathbb{R}_{\geq 0}$, which serves as a boundary condition setting the marginal of the process at time $t=0$:

$$
\frac{d}{dt}X_{t}=v_{t}\left(X_{t}\right),\quad X_{0}\sim p_{0}.
$$

The marginal of the process, called the *probability path*, is a time-dependent density $p:{\mathcal{X}}\times[0,1]\rightarrow\mathbb{R}_{\geq 0}$ such that $X_{t}\sim p_{t}$ for all $t\in[0,1]$.

Assume a dataset of i.i.d. samples in ${\mathcal{X}}$ from a *target distribution* $p_{1}$ and some easy-to-sample *source distribution* $p_{0}$. Flow matching provides a framework for learning a model $v_{t}$ such that the flow $\left(X_{t}\right)_{0\leq t\leq 1}$ defined by equation 1 maps samples from the source $X_{0}\sim p_{0}$ to samples from the target $X_{1}\sim p_{1}$.

Importantly, during training, the marginal $p_{t}$ of a flow matching model can be sampled efficiently using the *interpolant process*. While our method is agnostic to the chosen interpolant process, all pre-trained models used in this work were trained using the linear scheduler,

$$
X_{t}=(1-t)X_{0}+tX_{1},
$$

where $X_{0}\sim p_{0}$, $X_{1}\sim p_{1}$, and $t\in[0,1]$.

#### Sampling with flows

Obtaining a sample $X_{1}\sim p_{1}$ from a trained flow model $v_{t}$ is done by solving the ODE in equation 1. A general numerical approach discretizes the time interval $[0,1]$ into a sequence of $N$ smaller intervals, $0=t_{0}<t_{1}<\cdots<t_{N}=1$. Then, the exact solution on each interval is

$$
X_{n+1}=X_{n}+\left(t_{n+1}-t_{n}\right)u_{n}\left(X_{n}\right),
$$

where we use the simplified notation $X_{n}:=X_{t_{n}}$, and $u_{n}$ denotes the *mean velocity* of the $n$ -th interval $[t_{n},t_{n+1}]$, defined as

$$
u_{n}\left(X_{n}\right)=\frac{1}{t_{n+1}-t_{n}}\int_{t_{n}}^{t_{n+1}}v_{t}\left(X_{t}\right)\,dt.
$$

The solution is obtained by sequentially approximating the integral in equation 4. The simplest numerical method is the Euler solver, which approximates the velocity $v_{t}$ as constant over the interval $t\in[t_{n},t_{n+1}]$, yielding the mean velocity

$$
u_{n}\left(X_{n}\right)\approx v_{t_{n}}\left(X_{n}\right).
$$

Runge-Kutta methods are a family of higher-order solvers that use additional evaluations of the velocity $v_{t}$ in the interval $[t_{n},t_{n+1}]$ to achieve a higher-order approximation. We use the Midpoint method,

$$
u_{n}\left(X_{n}\right)\approx v_{t_{\text{mid}}}\left(X_{\text{mid}}\right),
$$

where $X_{\text{mid}}$ and $t_{\text{mid}}$ are the midpoint state and time, respectively,

$$
X_{\text{mid}}=X_{n}+\frac{t_{n+1}-t_{n}}{2}v_{t_{n}}\left(X_{n}\right),\quad t_{\text{mid}}=\frac{t_{n+1}+t_{n}}{2}.
$$

Figure 3: (left) The PDD student approximates the mean velocity across multiple consecutive intervals in a single evaluation. The pre-trained flow model (teacher) provides the mean velocity of a single interval using an ODE solver step. (right) Illustrate estimation of the PD loss given an initial state $X_{n}\sim p_{t_{n}}$ and a block size $L=4$; i) student predicts the mean velocities $\bar{u}^{\theta}_{n}\left(\cdot|X_{n}\right)\in{\mathcal{X}}^{L}$; ii) Following the student velocities yields the states $\bar{X}_{k}$ in the block $k\in\left\{n,\ldots,n+L-1\right\}$; iii) one of the states is randomly selected and the student’s output velocity is matched to the teacher’s mean velocity in the corresponding interval using the PD loss (11).

## 3 Parallel Decoding Distillation

To accelerate sampling from flow models, we propose to learn a parallel decoding model that predicts multiple integration steps (3) in a single network evaluation, rather than approximating them one step at a time. Figure 2 illustrates our approach.

#### Parallel decoder.

Assume a pre-trained flow model $v_{t}$ with a flow process $\left(X_{t}\right)_{0\leq t\leq 1}$ defined by equation 1 and marginals $p_{t}$. Fix a time discretization of length $N\in{\mathbb{N}}$,

$$
0=t_{0}<t_{1}<\ldots<t_{N}=1.
$$

Then, a *block* of size $L$ starting at step $n$ is the set of indices $\left\{n,\ldots,n+L-1\right\}$.

For a state $X_{n}\sim p_{t_{n}}$ at time step $t_{n}$, a *parallel decoder* $\bar{u}^{\theta}_{n}\left(\cdot\mid X_{n}\right)\in{\mathcal{X}}^{L}$ with block size $L\in{\mathbb{N}}$ is trained to predict the mean velocities of all intervals in the next block using a single network evaluation, 
$$
\bar{u}^{\theta}_{n}\left(k\mid X_{n}\right)\approx u_{k}\left(X_{k}\right),\quad k=n,\ldots,n+L-1.
$$
 (8)

Importantly, our parallel decoder (8) is well-defined, since the discretized flow process in this block, i.e., $X_{k}$, $k\in\left\{n,\ldots,n+L-1\right\}$, is fully specified by the exact solution (3) and the initial state $X_{n}\sim p_{t_{n}}$.

#### The parallelized process

During training, we employ an on-policy optimization algorithm that requires the intra-block process given by the parallel decoding model.

For a state $X_{n}\sim p_{t_{n}}$ at time step $t_{n}$, the *parallelized process* is defined by 
$$
\bar{X}_{k+1}=\bar{X}_{k}+(t_{k+1}-t_{k})\bar{u}^{\theta}_{n}\left(k\mid X_{n}\right),
$$
 (9) for the block $k\in\left\{n,\ldots,n+L-1\right\}$, with the initial condition $\bar{X}_{n}=X_{n}$.

The parallelized process is obtained by substituting the parallel decoder into the update rule of the exact solution (3). Notably, $\bar{u}^{\theta}_{n}\left(\cdot\mid X_{n}\right)\in{\mathcal{X}}^{L}$ depends only on the initial state $X_{n}$. Thus, the parallelized process $\left(\bar{X}_{k}\right)_{n\leq k\leq n+L}$ is simulated using a single evaluation of the parallel decoder.

#### Sampling

While classical algorithms advance a single interval at each ODE step using the recursive rule (3), with the parallel decoder we can advance $L$ intervals simultaneously. For a current state $X_{n}\sim p_{t_{n}}$, we approximate the exact solution (3) in the block $\left\{n,\ldots,n+L-1\right\}$ using the parallelized process (9). Then, solving the recursion over the intra-block index $k$, yields the *block-step rule*

$$
\bar{X}_{n+L}=X_{n}+\sum_{k=n}^{n+L-1}(t_{k+1}-t_{k})\bar{u}^{\theta}_{n}\left(k|X_{n}\right).
$$

By repeating the recursive block-step rule (10) $N/L$ times, in each step approximating $\bar{X}_{n}\approx X_{n}$, we obtain a clean sample from the model $\bar{X}_{N}$. We provide a PyTorch pseudocode in Algorithm 1, where dimension 0 indexes the parallel predictions $k=0,\ldots,N-1$.

Algorithm 1 Parallel decoding sampling

⬇

\# student - parallel decoder model

\# t - time discretization

\# L - block size

\# shape - data shape

\# init noise

x\_n = randn(\*shape)

\# step sizes

h = diff(t, dim=0)

for n in range(0, len(t)-1, step=L):

\# parallel predictions

u = student(x\_n, t\[n\])

\# slice current block

u\_n, h\_n = u\[n:n+L\], h\[n:n+L\]

\# block step

x\_n = x\_n + einsum(’k,k...’, h\_n, u\_n)

return x\_n

(a) Teacher

(b) Student train

(c) Student generation

Figure 4: Architecture of the parallel decoder (b) vs. the pre-trained flow model (a). Notably, the parallel decoder utilizes the same backbone, but with $N$ times the final linear layer. (c) At generation, instead of applying a linear layer per intra-block step (9), we can fuse the layers into a single linear layer that outputs the block’s average velocity, which advances the state across the entire block (14).

#### Training

The parallel decoder is trained by regressing onto a Runge-Kutta approximation of the mean velocity (8) of the pre-trained teacher flow model. In practice, we use a single Euler or Midpoint step. To obtain a tractable loss, we employ on-policy training, estimating the teacher’s mean velocity on student outputs $\bar{X}_{k}$ as defined in (9).

For a time discretization $t_{0}<\ldots<t_{N}$ and a block size $L$, our training objective is 
$$
{\mathcal{L}}_{\text{PD}}\left(\theta\right)=\mathbb{E}\left[\left\|\bar{u}^{\theta}_{n}\left(k|X_{n}\right)-u_{k}\left(\text{sg}\left(\bar{X}_{k}\right)\right)\right\|^{2}\right],
$$
 (11) where block starting indices $n\in\left\{0,L,\ldots,N-L\right\}$ and intra-block indices $k\in\left\{n,\ldots,n+L-1\right\}$ are sampled uniformly, $X_{n}\sim p_{t_{n}}$ is sampled using the interpolant process (2), $\bar{X}_{k}$ is the parallelized process (9) in the block, $\text{sg}(\cdot)$ denotes the stop-gradient operator, and the teacher mean velocity $u_{k}$ is approximated with a Runge-Kutta step.

Importantly, both $\bar{u}^{\theta}_{n}\left(k|X_{n}\right)$ and $\bar{X}_{k}$ are obtained with a single evaluation of the parallel decoder $\bar{u}^{\theta}_{n}$. Approximating the mean velocity $u_{k}$ with a single Euler (5) or Midpoint (6) step requires 1-2 evaluations of the teacher model $v$ (resp.). This makes the parallel decoding (PD) loss (11) tractable even at large scales. Figure 3 illustrates evaluation of the PD loss and we provide PyTorch pseudocode in Algorithm 2. For simplicity, we ignore the batch dimension, however all operations can be batched.

Proposition 1 shows that the PD loss is a valid objective for learning the parallel decoder as defined in (8). In particular, up to a controllable Runge-Kutta approximation error, the minimizer of the PD loss samples exactly the teacher trajectories, i.e., $\bar{X}_{n}=X_{n}$, for $n=0,\dots,N$. The proof is in Appendix D.

###### Proposition 1.

The minimizer of the parallel decoding loss (11) satisfies the parallel decoder condition (8).

#### Data-free training

In cases where data is not available, we propose an online on-policy training scheme that avoids the need to sample $X_{n}\sim p_{t_{n}}$ with the interpolant process (2). Instead, we follow our sampling algorithm 1, while alternating solver and training steps. That is, we sample an initial state $X_{0}\sim p_{0}$, then for the next $N/L-1$ iterations, we utilize the output of the parallel decoder $\bar{u}^{\theta}_{n}\left(\cdot|X_{n}\right)$ both for the optimization step and to advance $\bar{X}_{n}$ to $\bar{X}_{n+L}$; see the modified PyTorch pseudocode for PD loss in Algorithm 3 in the appendix. Importantly, we apply the stop-gradient operation when advancing the state, preventing additional memory or compute cost.

Algorithm 2 Parallel decoding loss

⬇

\# student - parallel decoder model

\# teacher - pre-trained flow model

\# runge\_kutta - approximates teacher’s mean velocity

\# t - time discretization

\# L\_min - minimal block size

\# L\_max - maximal block size

\# x - data

\# grid size

N = len(t) - 1

\# sample block index

n = L\_min \* randint(0, N//L\_min)

\# sample the probability path

x\_n = (1-t\[n\]) \* randn\_like(x) + t\[n\] \* x

\# parallel predictions

u = student(x\_n, t\[n\])

\# step sizes

h = diff(t, dim=0)

\# sample index in the block

k = randint(n, clip(n+L\_max, max=N))

\# step from n to k

x\_k = x\_n + einsum(’l,l...’, h\[n:k\], u\[n:k\])

\# teacher mean velocity

u\_k = runge\_kutta(teacher, x\_k, t\[k\], h\[k\])

\# mse loss

loss = mse\_loss(u\[k\], u\_k.detach())

return loss

#### Architecture and variable block size

Our requirement is an architecture that predicts $L$ mean-velocities $\bar{u}^{\theta}_{n}\left(\cdot|X_{n}\right)\in{\mathcal{X}}^{L}$, i.e., one velocity for each time step in the block, instead of the single instantaneous velocity prediction $v_{t_{n}}\left(X_{t_{n}}\right)\in{\mathcal{X}}$ of the pre-trained flow model. As illustrated in Figure 4, we utilize the same backbone architecture of the pretrained flow model, but with the final linear layer repeated $N$ times, i.e., one for each time step in the grid (7). Formally, we assume the teacher architecture is of the form

$$
v_{t}(x)=WH_{t}(x),
$$

where $H_{t}$ is the backbone that outputs the final hidden state and $W$ is the final linear layer. Then we learn $N$ linear layers $W^{\theta}_{0},\ldots,W^{\theta}_{N-1}$ and, for $k\geq n$, the parallel decoder’s architecture is given by

$$
\bar{u}^{\theta}_{n}(k|x_{n})=W^{\theta}_{k}H^{\theta}_{t_{n}}(x_{n}).
$$

This enables initialization from the final layer of the pretrained flow model. Additionally, the advantage of taking $N$ (grid size) instead of exactly $L$ (block size) linear layers is that it allows us to learn a single model that can predict any block size without the need to introduce a second time coordinate, which is, for instance, required for flow maps [^14] [^44] [^4].

In practice we are not interested in all possible block sizes, but in some subset of block sizes. Thus, we define a minimum and maximum block size $L_{\text{min}}\leq L_{\text{max}}\in{\mathbb{N}}$. Then, during training, we consider multiples of $L_{\text{min}}$ for indices $n$ of initial states and sample $k\in\left\{n,\ldots,n+L_{\text{max}}-1\right\}$ inside each block.

### 3.1 Layer Fusion and connection to flow maps

An important observation emerges when comparing PDD training and generation. For a block size $L$ and starting step $n$, estimating the PD loss (11) requires the intra-block steps of the parallelized process (9), and thus uses all distinct student output directions $W^{\theta}_{k}H^{\theta}_{t_{n}}\left(X_{n}\right)$, for $k=n,\ldots,n+L-1$. In contrast, during generation, we use the block step (10) to skip $L$ intervals. This requires only the weighted-average direction, which, for our architecture (13), yields

$$
\bar{X}_{n+L}=\bar{X}_{n}+(t_{n+L}-t_{n})W^{\theta}_{n:n+L}H^{\theta}_{t_{n}}\left(\bar{X}_{n}\right),
$$

where $W^{\theta}_{n:n+L}$ is a fused linear layer,

$$
W^{\theta}_{n:n+L}=\sum_{k=n}^{n+L-1}\Delta_{k}W^{\theta}_{k},\ \Delta_{k}=\frac{t_{k+1}-t_{k}}{t_{n+L}-t_{n}}.
$$

Thus, our shared backbone $H^{\theta}_{t_{n}}$ learns a representation of the mean-velocity over the interval $[t_{n},t_{n+L}]$. However, instead of using JVPs [^44] [^14] [^4] [^72] or finite differences [^35], we use the learnable linear maps $W_{k}^{\theta}$, for $k=n,\ldots,n+L-1$, to decompose the mean-velocity prediction into parallel sub-interval predictions. Then, during training, the gradients through the shared backbone recover, in expectation, the training signal for learning the full-interval mean-velocity. A discussion of the exact connection to flow maps is provided in Appendix C. We validate that the parallel decoder can learn non-trivial trajectories by comparing the curvature of its trajectories with that of the teacher trajectories in Figure 17 in the appendix.

An additional practical implication is that during inference we can avoid the extra compute of an enlarged final layer and we only need to hold one fused linear layer per block in memory.

Table 1: High-level differences between flow-map distillation methods, Pi-Flow, and our PDD.

|  | Eulerian/Lagrangian Flow Maps | Pi-Flow | PDD (Ours) |
| --- | --- | --- | --- |
| NFE | Variable | Fixed | Variable |
| JVP/finite-diff. | Required | Free | Free |
| Head at inference | Linear | Gaussian mixture | Fused-linear |

## 4 Related works

#### Trajectory-based distillation

The first successful method to distill the trajectories of the flow ODE (1) using a student-teacher scheme to achieve few-step generation is Progressive Distillation [^45]. They gradually increase the step size of the student using multi-phase training. Consistency models [^52] [^34] [^50] [^32] [^13] initialize the student from the teacher and then directly distill a map from any intermediate state along the trajectory to the clean state using a self-consistency condition. More recent works [^54] [^3] [^44] [^55] [^23] [^27] [^53] [^5], distill the mean velocity between any two states on the trajectory.

These methods have shown promising results on image generation. However, on large scale video models they fail to achieve high-quality few-step generation. Additionally, they often rely on JVP or finite differences which are expensive to evaluate on large-scale models or yield unstable training dynamics.

Most related to our method is Pi-Flow [^7]. Similar to us, they make the observation that given a pre-trained flow model, the trajectories in the interval $[t_{n},t_{n+L}]$, are fully specified by the initial state, and use it to delegate the integration in that interval to a small learnable policy head. In contrast, we utilize it to motivate our parallel prediction paradigm, simplifying training and inference, and changing the focus from expressive parametrizations (such as Gaussian mixtures) to improved supervision. While Pi-Flow distills the continuous-time instantaneous velocity (1) $v$, we go beyond by discretizing time and distilling numerical approximations of the mean velocity (4) $u$. Furthermore, using layer fusion (14) we avoid any additional cost at generation. Lastly, our training algorithm naturally enables sampling with different NFEs, whereas Pi-Flow is restricted to fixed NFE. The high-level differences are summarized in Table 1.

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/origami_dancers/pdd_1/frame_t0p5s.jpg)

Refer to caption

#### Training from scratch

There is a growing effort [^12] [^14] [^4] [^60] [^71] [^72] to develop end-to-end training methods for few-step generation models. These methods directly learn the flow map and avoid separate pre-training and distillation phases by combining the flow matching objective [^29] [^30] with a trajectory-based distillation objective. Instead of using a separate student-teacher scheme, they use the current state of the model as the teacher. This line of work has spawned several follow-up methods that aim to improve convergence [^69] [^15] [^18] [^33] [^37] or replace JVPs with finite-difference approximations [^35] [^41].

While PDD could potentially be extended to a self-distillation framework, such methods require a completely different pipeline, including larger datasets and significantly more compute.

#### Distribution-based distillation

Instead of letting a student distill the trajectories of a teacher model, a more flexible but sufficient condition for a few-steps generation model is to align the marginals $p_{t}$ of the two. ADD [^46], LADD [^47], and APT [^28] are using GAN losses to fine-tune a pre-trained diffusion model into a few step generation model. DMD [^67] [^66] incorporates the VSD loss [^59] for distillation. $f$ -distill [^64] generalizes the VSD loss to $f$ -divergences, and SiD and extensions [^74] [^73] [^36] use variants of Fisher divergences. Distribution-based methods established themselves as dominant approaches for distillation of large-scale video models [^39] [^70] [^17] [^11] [^28]. However, they suffer from mode collapse leading to a lack of generation diversity and motion. Recent works try to mitigate that by regularizing the loss with trajectory-based objectives [^8] [^70] [^17].

Moreover, distribution-based methods are known to be sensitive to hyper-parameters, require additional trainable parameters, and their performance can vary significantly across training iterations due to their alternating training objectives. In contrast, we find that the memory-efficient simple PD loss is much more robust to hyper-parameter choices, and presents consistent generation across training iteration.

## 5 Experiments

We empirically validate PDD on three tasks: i) class-conditional image generation on ImageNet-256 [^43] using the SiT-XL+REPA model [^68]; ii) text-to-image generation using Qwen-Image [^61]; and iii) text-to-video generation using the 1.3B and 14B variants of Wan2.1 [^56] as well as LTX-2.3 [^19]. The datasets used for PDD training are described in Appendix B.

#### Training setup.

PDD has three main design choices: i) the time discretization (7), defined by the grid size $N$ and the time reparameterization; ii) the Runge-Kutta method used to approximate the mean velocity (4); and iii) the minimum and maximum block sizes, $L_{\min}$ and $L_{\max}$, used in Algorithms 2 and 3, which determine the set of available NFEs at inference time. For each task, we train two PDD models, each with a different grid size, Runge-Kutta method, and block-size range. Additional training details are provided in Appendix B.

For class-conditional image generation, we set $N=128$ for the Euler method and $N=64$ for the midpoint method. For both models, we use the uniform time discretization (7), $t_{n}=\frac{n}{N}$, and choose block sizes $L_{\min}$ and $L_{\max}$ such that the available NFEs at inference time are $1,2,4,8$.

For both text-to-image and text-to-video generation, we set $N=256$ for the Euler method and $N=128$ for the Midpoint method. All models use the *shift* transformation [^10] [^48] for the time discretization:

$$
t_{n}=\text{shift}_{s}\left(\frac{n}{N}\right),\quad\text{shift}_{s}(t)=\frac{\frac{1}{s}t}{1+\left(\frac{1}{s}-1\right)t}.
$$

Due to the lack of high-quality image and video datasets, we apply data-free training as described in Algorithm 3. The minimum and maximum block sizes, $L_{\min}$ and $L_{\max}$, are chosen such that the available NFEs at inference time are $2,4,8$ for Qwen-Image and Wan2.1, and $4,8$ for LTX-2.3.

Across all tasks, since the midpoint method requires two teacher evaluations, we accumulate the Euler loss over two intervals within each block for a fair comparison. Additionally, details about classifier-free guidance are in Appendix B.

While PDD exhibits stable convergence across hyperparameter, we find that the two most important training hyperparameters are the time reparameterization and the batch size, with larger batch sizes yielding better performance. Additionally, as shown in Tables 3, 4, 5, the midpoint approximation consistently improves performance compared to the Euler approximation.

Figure 6: FID vs. NFE of PDD with Euler and Midpoint methods for approximating the mean velocity (4) on ImageNet-256 using SiT-XL+REPA model as teacher with guidance scale 2.9.

#### Class-conditional ImageNet

As shown in Figure 6, the FID of our PDD models generally improves as the NFE increases, validating that PDD successfully shares weights across different NFEs. Although the $8$ -NFE setting exhibits an increase in FID, we find that this degradation can be mitigated by using a lower guidance scale, at the cost of higher FID at lower NFEs; see Figure 8 in the appendix.

Table 2: FID with NFE $=1$ on ImageNet-256 using SiT-XL+REPA as teacher. Guidance scale $w=2.9$.

| Method | FID |
| --- | --- |
| Pi-Flow [^7] | 2.85 |
| FreeFlow [^55] | 1.45 |
| PDD - Euler | 2.73 |
| PDD - Midpoint | 2.69 |

Table 2 compares PDD with the related Pi-Flow method [^7] and the SOTA FreeFlow method [^55]. Our PDD method achieves a very competitive FID in the single-step setting while (i) having a simpler objective (without Gaussian mixture as Pi-Flow or an additional network as FreeFlow) and (ii) also supporting inference with multiple NFE budgets.

#### Text-to-image Qwen-Image

Table 3: Qwen-Image on the OneIG-EN, DPG-Bench, and GenEval. <sup>∗</sup> denotes our re-evaluation of the official checkpoint. This table reports the overall metrics. The full dimensions of the three benchmarks are in tables 6, 7, and 8 in Appendix B.2.

<table><tbody><tr><td>Method</td><td>NFE</td><td>OneIG-EN <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>DPG-Bench <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>GenEval <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Euler <sup>∗</sup></td><td>50 <math><semantics><mrow><mo>×</mo> <mn>2</mn></mrow> <annotation>\times 2</annotation></semantics></math></td><td>0.537</td><td>88.30</td><td>0.86</td></tr><tr><td>TwinFlow <sup>∗</sup> <sup><a href="#fn:8">8</a></sup></td><td rowspan="3">2</td><td>0.493</td><td>86.67</td><td>0.82</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.508</td><td>88.04</td><td>0.86</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.516</td><td>88.10</td><td>0.86</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step4-v2) <sup><a href="#fn:9">9</a></sup></td><td rowspan="5">4</td><td>0.524</td><td>88.25</td><td>0.85</td></tr><tr><td>TwinFlow <sup>∗</sup> <sup><a href="#fn:8">8</a></sup></td><td>0.502</td><td>86.18</td><td>0.82</td></tr><tr><td>Pi-Flow <sup>∗</sup> <sup><a href="#fn:7">7</a></sup></td><td>0.533</td><td>88.11</td><td>0.85</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.535</td><td>88.45</td><td>0.86</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.538</td><td>88.66</td><td>0.86</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step8-v2) <sup><a href="#fn:9">9</a></sup></td><td rowspan="4">8</td><td>0.526</td><td>88.20</td><td>0.84</td></tr><tr><td>Pi-Flow <sup>∗</sup> <sup><a href="#fn:7">7</a></sup></td><td>0.536</td><td>87.90</td><td>0.84</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.538</td><td>88.51</td><td>0.86</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.541</td><td>88.46</td><td>0.85</td></tr></tbody></table>

Table 4: Qwen-Image on HPSv2, PickScore, and OneIG diversity. <sup>∗</sup> denotes our re-evaluation of the official checkpoint.

<table><tbody><tr><td>Method</td><td>NFE</td><td>HPSv2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>PickScore <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>OneIG diversity <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td>Euler <sup>∗</sup></td><td>50 <math><semantics><mrow><mo>×</mo> <mn>2</mn></mrow> <annotation>\times 2</annotation></semantics></math></td><td>30.83</td><td>22.78</td><td>0.200</td></tr><tr><td>TwinFlow <sup>∗</sup> <sup><a href="#fn:8">8</a></sup></td><td rowspan="3">2</td><td>29.86</td><td>22.26</td><td>0.131</td></tr><tr><td>PDD - Euler (Ours)</td><td>29.59</td><td>22.47</td><td>0.197</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>30.15</td><td>22.66</td><td>0.177</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step4-v2) <sup><a href="#fn:9">9</a></sup></td><td rowspan="5">4</td><td>32.34</td><td>22.98</td><td>0.095</td></tr><tr><td>TwinFlow <sup>∗</sup> <sup><a href="#fn:8">8</a></sup></td><td>30.01</td><td>22.26</td><td>0.150</td></tr><tr><td>Pi-Flow <sup>∗</sup> <sup><a href="#fn:7">7</a></sup></td><td>30.94</td><td>22.67</td><td>0.182</td></tr><tr><td>PDD - Euler (Ours)</td><td>31.05</td><td>22.72</td><td>0.192</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>31.33</td><td>22.86</td><td>0.174</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step8-v2) <sup><a href="#fn:9">9</a></sup></td><td rowspan="4">8</td><td>32.35</td><td>22.95</td><td>0.109</td></tr><tr><td>Pi-Flow <sup>∗</sup> <sup><a href="#fn:7">7</a></sup></td><td>31.09</td><td>22.55</td><td>0.186</td></tr><tr><td>PDD - Euler (Ours)</td><td>31.34</td><td>22.73</td><td>0.198</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>31.56</td><td>22.86</td><td>0.181</td></tr></tbody></table>

We evaluate our PDD-distilled Qwen-Image models on three main benchmarks, OneIG [^6], DPG-Bench [^22], and GenEval [^16], with $\text{NFE}=2,4,8$. For SOTA baselines, we consider QwenLightning-v2 [^9], which employs DMD2 [^66] distillation, Pi-Flow, and TwinFlow [^8]. For all three baselines, we report re-evaluations of the official checkpoints. Table 3 shows that PDD achieves the best performance on the overall metrics of the three benchmarks. The full dimensions are provided in tables 6, 7, and 8 in the appendix. Additionally, we evaluate the models on the HPSv2 [^62] benchmark as well as measure PickScore [^62] on the same generated images. Table 4 shows our PDD is the runner up to DMD2 (Lightning-v2) model in terms of these human preference metrics. However, DMD suffers from mode collapse, resulting in a significant reduction of diversity. This is observed by the OneIG diversity metrics in Table 4 and on many examples in Figures 11 to 16. In contrast, our PDD shows competitive results while better preserving diversity and more closely following the teacher generation.

Table 5: Wan Text-to-Video 1.3B and 14B models performance and diversity metrics on VBench [^24] with Self-Forcing prompt set. Diversity is measured as mean pairwise V-JEPA 2/VideoMAE V2 feature distance across 5 generated videos per prompt. <sup>∗</sup> Our re-evaluation of the official checkpoint. <sup>∗∗</sup> Our re-implementation. <sup>†</sup> Reported numbers on proprietary prompts without diversity evaluation, since rCM [^70] did not release checkpoints.

<table><tbody><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">NFE</td><td colspan="3">VBench</td><td colspan="2">V-JEPA 2</td><td colspan="2">VideoMAE V2</td></tr><tr><td>Overall <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Quality <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Semantic <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Cosine <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>L2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Cosine <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>L2 <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td></tr><tr><td rowspan="6">1.3B</td><td>UniPC <sup>∗</sup> (Teacher)</td><td>50 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 2</td><td>83.77</td><td>84.90</td><td>79.22</td><td>0.1254</td><td>27.07</td><td>0.02681</td><td>2.922</td></tr><tr><td>rCM <sup>†</sup> <sup><a href="#fn:70">70</a></sup></td><td rowspan="5">4</td><td>84.43</td><td>85.38</td><td>80.63</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AnyFlow <sup>∗</sup> <sup><a href="#fn:17">17</a></sup></td><td>84.45</td><td>85.22</td><td>81.34</td><td>0.0704</td><td>19.88</td><td>0.01029</td><td>1.807</td></tr><tr><td>DMD <math><semantics><msup><mn>2</mn> <mrow><mo>∗</mo> <mo>⁣</mo> <mo>∗</mo></mrow></msup> <annotation>2^{**}</annotation></semantics></math> (FastGen <sup><a href="#fn:38">38</a></sup>)</td><td>84.69</td><td>86.14</td><td>78.87</td><td>0.0833</td><td>21.83</td><td>0.01646</td><td>2.278</td></tr><tr><td>PDD - Euler (Ours)</td><td>84.44</td><td>85.99</td><td>78.22</td><td>0.1018</td><td>24.54</td><td>0.01901</td><td>2.489</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>84.94</td><td>86.45</td><td>78.91</td><td>0.1032</td><td>24.63</td><td>0.02054</td><td>2.548</td></tr><tr><td rowspan="9">14B</td><td>UniPC <sup>∗</sup> (Teacher)</td><td>50 <math><semantics><mo>×</mo> <annotation>\times</annotation></semantics></math> 2</td><td>83.90</td><td>84.56</td><td>81.24</td><td>0.1263</td><td>27.27</td><td>0.02497</td><td>2.824</td></tr><tr><td>rCM <sup>†</sup> <sup><a href="#fn:70">70</a></sup></td><td rowspan="5">4</td><td>84.92</td><td>85.43</td><td>82.88</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AnyFlow <sup>∗</sup> <sup><a href="#fn:17">17</a></sup></td><td>84.95</td><td>85.70</td><td>81.92</td><td>0.0786</td><td>20.99</td><td>0.01297</td><td>1.992</td></tr><tr><td>DMD2 <sup>∗∗</sup> (FastGen <sup><a href="#fn:38">38</a></sup>)</td><td>84.40</td><td>85.16</td><td>81.34</td><td>0.0568</td><td>17.67</td><td>0.00945</td><td>1.710</td></tr><tr><td>PDD <math><semantics><msup><mtext>short</mtext></msup> <annotation>{}^{\text{short}}</annotation></semantics></math> - Midpoint (Ours)</td><td>84.92</td><td>85.71</td><td>81.77</td><td>0.0791</td><td>21.27</td><td>0.01247</td><td>2.027</td></tr><tr><td>PDD <math><semantics><msup><mtext>long</mtext></msup> <annotation>{}^{\text{long}}</annotation></semantics></math> - Midpoint (Ours)</td><td>84.69</td><td>85.69</td><td>80.71</td><td>0.0846</td><td>22.13</td><td>0.01264</td><td>2.058</td></tr><tr><td>AnyFlow <sup>∗</sup> <sup><a href="#fn:17">17</a></sup></td><td rowspan="3">8</td><td>85.08</td><td>85.78</td><td>82.28</td><td>0.0765</td><td>20.67</td><td>0.01278</td><td>1.974</td></tr><tr><td>PDD <math><semantics><msup><mtext>short</mtext></msup> <annotation>{}^{\text{short}}</annotation></semantics></math> - Midpoint (Ours)</td><td>84.96</td><td>85.83</td><td>81.44</td><td>0.0816</td><td>21.63</td><td>0.01276</td><td>2.054</td></tr><tr><td>PDD <math><semantics><msup><mtext>long</mtext></msup> <annotation>{}^{\text{long}}</annotation></semantics></math> - Midpoint (Ours)</td><td>84.70</td><td>85.77</td><td>80.41</td><td>0.0868</td><td>22.43</td><td>0.01314</td><td>2.097</td></tr></tbody></table>

#### Text-to-video Wan2.1

Our results on the text-to-video task provide strong evidence that PDD is effective at large scale. We evaluate our distilled models on the VBench benchmark [^24] and compare against the SOTA baselines rCM [^70], AnyFlow [^17], and DMD2 (implemented in FastGen [^38]), which all incorporate a VSD loss. Additionally, we measure diversity by encoding the generated videos for VBench using V-JEPA 2 [^2] and VideoMAE V2 [^57] models, and report average pair-wise cosine and L2 distance across $5$ samples for each prompt. We find that skipping a layer in the backbone for the unconditional term in the CFG (17) can improve the performance of PDD. For the Wan2.1 1.3B model we skip the 10-th layer and for the 14B model we skip the 12-th layer.

As shown in Tables 5, on the Wan2.1 1.3B model PDD ranks first in terms of video quality, overall score, as well as diversity. On the Wan2.1 14B model we report on two checkpoints, *short* which stands for 200 training iterations and *long* which stands for 3k iterations. We find that PDD (short) achieves best video quality and is the runner-up to AnyFlow in the overall metrics with both $4$ and $8$ NFE. We find that, in general, PDD-generated videos exhibit a higher degree of motion (see Figure 5 and Figures 19 to 21) compared to baselines. In particular, while PDD (long) obtains lower VBench scores, we find that motion in the generated videos increases later in training, which is also notable in the dynamic degree score of VBench in Figure 18 in the appendix. Additionally, Table 5 shows that PDD obtains higher diversity scores than the baselines. This is also notable in many examples shown in Figures 19 to 21, generated from the best VBench checkpoints.

#### Text-to-video/audio LTX-2.3

To further show the scalability of PDD, we distill the 22B LTX-2.3 model, demonstrating multimodal few-step generation of 10s videos in 720p resolution with audio. As baseline, we consider the official LTX-2.3 distilled 8-step model [^19]. We apply the PD loss to the audio and video latents separately and average the loss across both modalities. Following the standard guidelines, we apply three guidance methods for the teacher, i.e., standard CFG (scale 4.5 for video and 7 for audio), cross-modal guidance (scale 3), and spatiotemporal skip guidance (scale 2 on layer 29). Due to the higher computational costs, we only consider the Euler method (with $N=256$) over a single interval within each block. For the teacher, this leads to $4\times 30$ NFE per generation. The comparisons in Figure 7 and Figures 22 to 25 in the appendix show that PDD can provide high-quality generations at only $8$ NFE after as few as 250 iterations of training, performing on par or better than the official distilled model without any access to training data.

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/ltx/white_gown_woman/teacher/frame_t0s.jpg)

Refer to caption

## 6 Conclusion

In this work, we introduced Parallel Decoding Distillation (PDD), a simple trajectory-based distillation method for accelerating diffusion and flow models. Rather than collapsing many denoising steps into a single large update, PDD trains a parallel decoder to predict the mean velocities of multiple consecutive intervals in one network evaluation. This formulation leads to a practical training objective that avoids VSD and adversarial losses, as well as JVPs, finite differences, and multi-stage distillation procedures. Across text-to-image/video/audio generation, PDD achieves strong few-step performance on large-scale models such as Qwen-Image, Wan2.1, and LTX-2.3, while better preserving sample diversity and motion compared to distribution-based baselines. This makes PDD the first pure trajectory-based distillation method for few-step, high-resolution video generation.

Because our large-scale text-to-image and video experiments rely on data-free training, investigating PDD in data-dependent settings beyond ImageNet-256 remains future work. In addition, PDD introduces a flexible inference-time design choice: the block size can be selected after evaluating the parallel decoder. This suggests the possibility of adaptive block-size selection, where a verifier or confidence criterion determines how aggressively to skip intervals during generation. Finally, generalizing the parallel decoding principle to discrete autoregressive models may broaden the applicability of PDD beyond diffusion and flow-based generation.

## References

## Appendix A Algorithms

Algorithm 3 Data-free parallel decoding loss

⬇

\# student - parallel decoder model

\# teacher - pre-trained flow model

\# runge\_kutta - approximates teacher’s mean velocity

\# t - time discretization

\# L\_min - minimal block size

\# L\_max - maximal block size

\# x\_n - carried state from previous iteration

\# n - carried time index from previous iteration

\# grid size

N = len(t) - 1

\# reset state and time index

if n==N:

x\_n = randn\_like(x\_n)

n=0

\# parallel predictions

u = student(x\_n, t\[n\])

\# step sizes

h = diff(t, dim=0)

\# sample index in the block

k = randint(n, clip(n+L\_max, max=N))

\# step from n to k

x\_k = x\_n + einsum(’l,l...’, h\[n:k\], u\[n:k\])

\# teacher mean velocity

u\_k = runge\_kutta(teacher, x\_k, t\[k\], h\[k\])

\# mse loss

loss = mse\_loss(u\[k\], u\_k.detach())

\# advance state for next iteration

x\_n = x\_n + einsum(’l,l...’, h\[n:n+L\_min\], u\[n:n+L\_min\])

\# advance time index for next iteration

n = n + L\_min

return loss, x\_n.detach(), n

## Appendix B Experiments

#### Datasets

We use a distinct dataset and VAE for each model for our PDD training. For the Repa-ImageNet-256, we use the ImageNet [^43] dataset and the Stable Diffusion VAE [^42]. For Qwen-Image, we employ the data-free PDD training using the text prompts set provided by Pi-Flow [^7] with native Qwen-Image VAE on resolution $1024\times 1024$. Similarly, for Wan2.1 and LTX-2.3 models we use their native VAEs and data-free PDD training. For the Wan2.1 models, we use a set of prompts extracted from reshuffled ViMix-14M [^65] on resolution $480\times 832$ and $5s$ duration with $16$ FPS. For LTX-2.3, we use a mixture of prompts from ViMix-14M [^65] and VidProm [^58], enhanced to include audio descriptions, on resolution $704\times 1280$ and $10s$ duration with $24$ FPS.

#### Architecture

For all our PDD models we use the exact same backbone as the teacher model. Additionally, as described in Section 3 and Figure 4, we enlarge the final linear layer by repeating the channel dimension $N$ times, i.e., the grid size. Importantly, the repeat operation must be applied to the correctly reshaped weights, such that the resulting linear layer is equivalent to initializing all parallel steps with the final layer of the pretrained model. For LTX-2.3, we apply this idea to both the final linear layer of the video and audio tower.

#### Classifier free guidance (CFG)

Following previous works [^67] [^7], we introduce guidance by replacing the teacher velocity (4) with the guided velocity. Given a condition $c$, the CFG velocity [^20] is

$$
v_{t}^{w}(x|c)=v_{t}(x)+w\left(v_{t}(x|c)-v_{t}(x)\right).
$$

We treat the cross-modal guidance and spatiotemporal skip guidance of LTX-2.3 analogously. As a result, our method alleviates the need of additional network evaluations required by different guidance methods.

### B.1 ImageNet

#### Training details

We use AdamW optimizer [^31] with constant learning rate $5\mathrm{e}{-5}$, and weight decay $0$. Additionally, we use batch size 2048, EMA constant $0.99995$. We train for 300k iterations.

#### Classifier free guidance

For each Runge-Kutta method: Euler, Midpoint, we train three PDD models, each with a distinct guidance scale (17): $w=2.7,2.9,3.2$. As shown in Figure 8, we find that different NFE at inference benefits from different guidance scale. In the main paper choose to show $w=2.9$ as it give the most balanced results.

Figure 8: FID vs. NFE of PDD with Euler and Midpoint methods for approximating the mean velocity (4) on Repa-ImageNet-256.

#### FID vs. Training iteration

We evaluate FID every 10K training iteration with NFE $=1,2,4,8$ and report the results in Figure 9. We observe a steady trend of decreasing FID during training.

| NFE $=1$ | NFE $=2$ |
| --- | --- |
|  |  |
| NFE $=4$ | NFE $=8$ |
|  |  |

Figure 9: FID vs. Training iteration of PDD with Euler and Midpoint methods for approximating the mean velocity (4) on Repa-ImageNet-256.

### B.2 Qwen-Image

#### Training details

We use AdamW optimizer with constant learning rate $1\mathrm{e}{-5}$, and weight decay $0$. Additionally, we use batch size 2048, without EMA. We train for 3k iterations and we evaluate the three benchmarks, i.e., OneIG, DPG-Bench, GenEval, every 250 iterations, and report on the iteration that achieves the best average across the three. Overall metric vs. training iteration of each benchmark is provided in Figure 10. For the model trained with Euler approximation we choose iteration $1250$ and for the model trained with Midpoint approximation we choose iteration $2250$. Both models uses the shift transformation (16) with scale $s=5$ and classifier-free guidance (17) with scale $w=4$ (including the native, per-token rescaling of the guided prediction to match the magnitude of the original conditional prediction).

| Benchmark | NFE=2 | NFE=4 | NFE=8 |
| --- | --- | --- | --- |
| OneIG |  |  |  |
| DPG-Bench |  |  |  |
| GenEval |  |  |  |

Figure 10: Overall metrics of OneIG, DPG-Bench, and GenEval vs. Training iteration of the Qwen-Image PDD model.

#### Additional results

We provide all dimensions of our considered benchmarks in tables 6, 7, and 8. Moreover, we provide additional comparisons between the Qwen-Image teacher, PDD, and DMD2 (Lightning-v2) in Figures 11 to 16.

Table 6: Qwen-Image on the OneIG-EN [^6]. The overall score is the average of the five dimensions. <sup>∗</sup> denotes our re-evaluation of the official checkpoint.

<table><tbody><tr><td>Method</td><td>NFE</td><td>Overall <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Alignment</td><td>Text</td><td>Reasoning</td><td>Style</td><td>Diversity</td></tr><tr><td>UniPC</td><td>-</td><td>0.539</td><td>0.882</td><td>0.891</td><td>0.306</td><td>0.418</td><td>0.197</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td rowspan="3">2</td><td>0.493</td><td>0.863</td><td>0.840</td><td>0.268</td><td>0.363</td><td>0.131</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.508</td><td>0.867</td><td>0.796</td><td>0.279</td><td>0.400</td><td>0.197</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.516</td><td>0.874</td><td>0.834</td><td>0.284</td><td>0.410</td><td>0.177</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step4-v2)</td><td rowspan="5">4</td><td>0.524</td><td>0.887</td><td>0.954</td><td>0.284</td><td>0.398</td><td>0.095</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td>0.502</td><td>0.857</td><td>0.881</td><td>0.265</td><td>0.355</td><td>0.150</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>0.533</td><td>0.877</td><td>0.886</td><td>0.294</td><td>0.427</td><td>0.182</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.535</td><td>0.879</td><td>0.897</td><td>0.292</td><td>0.417</td><td>0.192</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.538</td><td>0.883</td><td>0.911</td><td>0.296</td><td>0.427</td><td>0.174</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step8-v2)</td><td rowspan="4">8</td><td>0.526</td><td>0.888</td><td>0.945</td><td>0.287</td><td>0.403</td><td>0.109</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>0.536</td><td>0.870</td><td>0.907</td><td>0.302</td><td>0.419</td><td>0.186</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.538</td><td>0.877</td><td>0.892</td><td>0.298</td><td>0.422</td><td>0.198</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.541</td><td>0.881</td><td>0.913</td><td>0.300</td><td>0.429</td><td>0.180</td></tr></tbody></table>

Table 7: Qwen-Image on the DPG-Bench [^22]. <sup>∗</sup> denotes our re-evaluation of the official checkpoint.

<table><tbody><tr><td>Method</td><td>NFE</td><td>Overall <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Global</td><td>Entity</td><td>Attribute</td><td>Relation</td><td>Other</td></tr><tr><td>UniPC</td><td>-</td><td>88.32</td><td>91.32</td><td>91.56</td><td>92.02</td><td>94.31</td><td>92.73</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td rowspan="3">2</td><td>86.67</td><td>91.81</td><td>93.24</td><td>91.19</td><td>87.39</td><td>91.14</td></tr><tr><td>PDD - Euler (Ours)</td><td>88.04</td><td>92.65</td><td>92.00</td><td>92.65</td><td>92.06</td><td>91.40</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>88.10</td><td>92.50</td><td>92.93</td><td>91.55</td><td>92.01</td><td>93.04</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step4-v2)</td><td rowspan="5">4</td><td>88.25</td><td>93.28</td><td>90.04</td><td>92.97</td><td>92.49</td><td>93.08</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td>86.18</td><td>90.48</td><td>92.78</td><td>91.16</td><td>89.37</td><td>91.46</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>88.11</td><td>91.06</td><td>91.56</td><td>92.88</td><td>93.48</td><td>92.61</td></tr><tr><td>PDD - Euler (Ours)</td><td>88.45</td><td>93.18</td><td>91.87</td><td>93.21</td><td>92.04</td><td>91.16</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>88.66</td><td>93.19</td><td>93.25</td><td>92.27</td><td>92.04</td><td>93.35</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step8-v2)</td><td rowspan="3">8</td><td>88.20</td><td>90.13</td><td>92.92</td><td>92.17</td><td>93.39</td><td>91.94</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>87.90</td><td>88.05</td><td>91.96</td><td>93.25</td><td>93.33</td><td>89.93</td></tr><tr><td>PDD - Euler (Ours)</td><td>88.51</td><td>94.14</td><td>93.87</td><td>90.50</td><td>92.15</td><td>93.16</td></tr><tr><td>PDD - Midpoint (Ours)</td><td></td><td>88.46</td><td>93.36</td><td>92.74</td><td>92.90</td><td>91.87</td><td>91.37</td></tr></tbody></table>

Table 8: Evaluation of Qwen-Image on the GenEval [^16] benchmark. <sup>∗</sup> denotes our re-evaluation of the official checkpoint.

<table><tbody><tr><td>Model</td><td>NFE</td><td>Overall <math><semantics><mo>↑</mo> <annotation>\uparrow</annotation></semantics></math></td><td>Single Object</td><td>Two Object</td><td>Counting</td><td>Colors</td><td>Position</td><td>Attribute Binding</td></tr><tr><td>UniPC</td><td>-</td><td>0.87</td><td>0.99</td><td>0.92</td><td>0.89</td><td>0.88</td><td>0.76</td><td>0.77</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td rowspan="3">2</td><td>0.82</td><td>0.98</td><td>0.91</td><td>0.75</td><td>0.90</td><td>0.68</td><td>0.72</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.86</td><td>0.98</td><td>0.95</td><td>0.89</td><td>0.86</td><td>0.76</td><td>0.72</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.86</td><td>0.99</td><td>0.94</td><td>0.90</td><td>0.88</td><td>0.71</td><td>0.71</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step4-v2)</td><td rowspan="5">4</td><td>0.85</td><td>0.99</td><td>0.95</td><td>0.85</td><td>0.87</td><td>0.72</td><td>0.74</td></tr><tr><td>TwinFlow <sup>∗</sup></td><td>0.82</td><td>0.98</td><td>0.92</td><td>0.76</td><td>0.87</td><td>0.70</td><td>0.68</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>0.85</td><td>0.98</td><td>0.94</td><td>0.87</td><td>0.89</td><td>0.72</td><td>0.73</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.86</td><td>0.99</td><td>0.94</td><td>0.89</td><td>0.86</td><td>0.76</td><td>0.74</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.86</td><td>0.99</td><td>0.95</td><td>0.90</td><td>0.88</td><td>0.68</td><td>0.74</td></tr><tr><td>DMD2 <sup>∗</sup> (Lightning-step8-v2)</td><td rowspan="4">8</td><td>0.84</td><td>0.99</td><td>0.95</td><td>0.84</td><td>0.84</td><td>0.76</td><td>0.68</td></tr><tr><td>Pi-Flow <sup>∗</sup></td><td>0.84</td><td>0.98</td><td>0.93</td><td>0.87</td><td>0.88</td><td>0.69</td><td>0.71</td></tr><tr><td>PDD - Euler (Ours)</td><td>0.86</td><td>0.98</td><td>0.94</td><td>0.89</td><td>0.86</td><td>0.73</td><td>0.74</td></tr><tr><td>PDD - Midpoint (Ours)</td><td>0.85</td><td>0.99</td><td>0.93</td><td>0.88</td><td>0.88</td><td>0.67</td><td>0.73</td></tr></tbody></table>

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/7/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/60/teacher_0.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/180/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/253/teacher_0.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/659/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/488/teacher_0.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/490/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/586/teacher_0.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/321/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/706/teacher_0.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/716/teacher_0.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/qwenimage/diversity_tables/805/teacher_0.jpg)

Refer to caption

### B.3 Wan Text-to-Video

#### Training details

On both Wan2.1 1.3B and 14B models, we use AdamW optimizer with constant learning rate $1\mathrm{e}{-5}$, and weight decay $0$, batch size 256. Moreover, for the shift transformation (16) $s=6$, and for CFG we use guidance scale $w=5$ and add we skip a single layer in the forward of the unconditional evaluation. On the 1.3B we skip layer 10 and on the 14B we skip layer 12.

On the Wan2.1 1.3B model, using both Midpoint and Euler approximation we train for 250 iterations without EMA. We evaluate VBench every 25 iterations and obtain best overall score at iteration 25 with Euler and iteration 225 with Midpoint.

On the Wan2.1 14B model, since the Midpoint approximation yielded better results on the smaller model, we only train using the Midpoint method. The *short* checkpoint trains for 250 iterations without EMA, and we evaluate VBench every 25 iterations and obtaining best overall score at iteration 200. For the *long* checkpoint we train for 3.5k iteration with EMA and constant coefficient of $0.99$. We evaluate VBench every 250 iterations and achieve best score at 3k.

#### Additional results

We compare the curvature of PDD against the Wan2.1 14B teacher in Figure 17. Moreover, we provide all VBench dimensions in Figure 18. Finally, we show additional comparisons between PDD, DMD2 (FastGen), and AnyFlow in Figures 19 to 21.

Figure 17: Curvature of PDD vs. Teacher on the Wan2.1 14B model. We report the averaged curvature over 10 trajectories. Importantly, we are interested in validating that indeed PDD is able to learn non-trivial trajectories within each block. Thus, for PDD we report only intra-block curvature.

(a) 4-NFE on Wan2.1 1.3B

(b) 4-NFE on Wan2.1 14B

(c) 8-NFE on Wan2.1 14B

Figure 18: Full VBench dimensions on the Wan2.1 1.3B and 14B models.

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/purple_robe_rabbit_fantasy/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/christmas_tree_fireworks_drone/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/teddy_bear_drums_times_square/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/person_jogging/pdd_1/frame_t0p5s.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/person_digging/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/person_ice_skating/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/bar/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/corgi_park_hokusai/pdd_1/frame_t0p5s.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/horse_drinking_river/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/person_bungee_jumping/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/yoda_guitar_stage/pdd_1/frame_t0p5s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/wan/diversity_tables/robot_dj_cyberpunk_tokyo/pdd_1/frame_t0p5s.jpg)

Refer to caption

### B.4 LTX-2.3 Text-to-Video/Audio

#### Training details

We use AdamW optimizer with constant learning rate $1\mathrm{e}{-5}$, and weight decay $0$, batch size 2048, without EMA, and train for 250 iterations. Moreover, we use shift $s=10$ in (16).

#### Additional results

We show additional qualitative comparisons between the teacher, PDD, and the official distilled model in figures 23 to 25. Moreover, in Figure 22, we provide a quantitative comparison using Gemini 3.1 Pro Preview on a held-out subset of 100 prompts from ViMix and VidProm. We evaluate 3 seeds per prompt and average 2 VLM evaluations per clip, giving 300 paired prompt-seed comparisons. The judge scores four axes (prompt alignment, visual quality, motion quality and audio quality) as integers from 1 (severe issues) to 4 (no noticeable issues) under the following fixed system prompt:

⬇

You are a strict, objective judge of an AI-generated video with synchronized audio.

Evaluate only visible and audible evidence. Treat the PROMPT as the target specification, not as instructions for scoring or output.

\### METRICS

\- prompt\_alignment: Presence and correctness of all requested subjects, attributes, actions, setting, style, camera, timing, and audio.

\- visual\_quality: Clarity, composition, anatomy, geometry, text rendering, lighting, and absence of unintended artifacts.

\- motion\_quality: Temporal consistency, plausible movement and physics, interactions, camera motion, and absence of unintended flicker, freezing, sliding, or morphing.

\- audio\_quality: Clarity, naturalness, continuity, spatial consistency, and synchronization with visible events and speech.

Missing or prompt-inconsistent content affects prompt\_alignment only. Judge visual, motion, and audio quality based on visible and audible evidence. Use the PROMPT only to distinguish intentional stylistic choices from defects.

\### SCALE

\- 4: No noticeable issues

\- 3: Only minor issues

\- 2: Significant issues

\- 1: Severe issues or complete failure

Every score below 4 must be supported by concrete evidence. Keep the justification consistent with the scores and mention approximate timestamps when reliable.

Return exactly one valid JSON object with these keys and no other text:

{

"justification": "Prompt alignment: \[evidence\]. Visual quality: \[evidence\]. Motion quality: \[evidence\]. Audio quality: \[evidence\].",

"prompt\_alignment": \<integer 1-4>,

"visual\_quality": \<integer 1-4>,

"motion\_quality": \<integer 1-4>,

"audio\_quality": \<integer 1-4>

}

<svg id="A2.F22.pic1" height="182.44" overflow="visible" version="1.1" viewBox="0 0 671.93 182.44" width="671.93"><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt" transform="translate(0,182.44) matrix(1 0 0 -1 0 0) translate(117.66,0) translate(0,44) matrix(1.0 0.0 0.0 1.0 -117.65 -44)"><g transform="matrix(1 0 0 1 0 0) translate(117.66,0) translate(0,60.95)"><g style="--ltx-stroke-color:#E0E0E0;--ltx-fill-color:#BFBFBF;--ltx-fg-color:#BFBFBF;" color="#BFBFBF" fill="#BFBFBF" stroke="#E0E0E0" stroke-width="0.4pt"><path style="fill:none" d="M 0 -16.95 L 0 94.01 M 36 -16.95 L 36 94.01 M 72.01 -16.95 L 72.01 94.01 M 108.01 -16.95 L 108.01 94.01 M 144.02 -16.95 L 144.02 94.01 M 180.02 -16.95 L 180.02 94.01 M 216.02 -16.95 L 216.02 94.01 M 252.03 -16.95 L 252.03 94.01 M 288.03 -16.95 L 288.03 94.01 M 324.04 -16.95 L 324.04 94.01 M 360.04 -16.95 L 360.04 94.01 M 396.05 -16.95 L 396.05 94.01 M 432.05 -16.95 L 432.05 94.01 M 468.05 -16.95 L 468.05 94.01 M 504.06 -16.95 L 504.06 94.01 M 540.06 -16.95 L 540.06 94.01"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" color="#808080" fill="#808080" stroke="#808080" stroke-width="0.2pt"><path style="fill:none" d="M 0 -22.86 L 0 -16.95 M 36 -22.86 L 36 -16.95 M 72.01 -22.86 L 72.01 -16.95 M 108.01 -22.86 L 108.01 -16.95 M 144.02 -22.86 L 144.02 -16.95 M 180.02 -22.86 L 180.02 -16.95 M 216.02 -22.86 L 216.02 -16.95 M 252.03 -22.86 L 252.03 -16.95 M 288.03 -22.86 L 288.03 -16.95 M 324.04 -22.86 L 324.04 -16.95 M 360.04 -22.86 L 360.04 -16.95 M 396.05 -22.86 L 396.05 -16.95 M 432.05 -22.86 L 432.05 -16.95 M 468.05 -22.86 L 468.05 -16.95 M 504.06 -22.86 L 504.06 -16.95 M 540.06 -22.86 L 540.06 -16.95"></path></g><g style="--ltx-stroke-color:#808080;--ltx-fill-color:#808080;--ltx-fg-color:#808080;" color="#808080" fill="#808080" stroke="#808080" stroke-width="0.2pt"><path style="fill:none" d="M -5.91 0 L 0 0 M -5.91 25.68 L 0 25.68 M -5.91 51.37 L 0 51.37 M -5.91 77.05 L 0 77.05"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" stroke-width="0.4pt"><path style="fill:none" d="M 0 -16.95 L 540.06 -16.95"></path><path style="fill:none" d="M 0 -16.95 L 0 94.01"></path><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -3.2 -35.77)"><foreignObject style="--ltx-fo-width:0.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="6.4"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="0"><semantics><mn mathsize="0.900em">0</mn> <annotation encoding="application/x-tex">0</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 29.6 -35.77)"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="12.8"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="20"><semantics><mn mathsize="0.900em">20</mn> <annotation encoding="application/x-tex">20</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 65.61 -35.77)"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="12.8"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="40"><semantics><mn mathsize="0.900em">40</mn> <annotation encoding="application/x-tex">40</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 101.61 -35.77)"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="12.8"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="60"><semantics><mn mathsize="0.900em">60</mn> <annotation encoding="application/x-tex">60</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 137.62 -35.77)"><foreignObject style="--ltx-fo-width:1em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="12.8"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="80"><semantics><mn mathsize="0.900em">80</mn> <annotation encoding="application/x-tex">80</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 170.42 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="100"><semantics><mn mathsize="0.900em">100</mn> <annotation encoding="application/x-tex">100</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 206.43 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="120"><semantics><mn mathsize="0.900em">120</mn> <annotation encoding="application/x-tex">120</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 242.43 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="140"><semantics><mn mathsize="0.900em">140</mn> <annotation encoding="application/x-tex">140</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 278.43 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="160"><semantics><mn mathsize="0.900em">160</mn> <annotation encoding="application/x-tex">160</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 314.44 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="180"><semantics><mn mathsize="0.900em">180</mn> <annotation encoding="application/x-tex">180</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 350.44 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="200"><semantics><mn mathsize="0.900em">200</mn> <annotation encoding="application/x-tex">200</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 386.45 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="220"><semantics><mn mathsize="0.900em">220</mn> <annotation encoding="application/x-tex">220</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 422.45 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="240"><semantics><mn mathsize="0.900em">240</mn> <annotation encoding="application/x-tex">240</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 458.45 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="260"><semantics><mn mathsize="0.900em">260</mn> <annotation encoding="application/x-tex">260</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 494.46 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="280"><semantics><mn mathsize="0.900em">280</mn> <annotation encoding="application/x-tex">280</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 530.46 -35.77)"><foreignObject style="--ltx-fo-width:1.5em;--ltx-fo-height:0.63em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.03" overflow="visible" transform="matrix(1 0 0 -1 0 8.03)" width="19.2"><math xmlns="http://www.w3.org/1998/Math/MathML" display="inline" data-latex="300"><semantics><mn mathsize="0.900em">300</mn> <annotation encoding="application/x-tex">300</annotation></semantics></math></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -87.59 -3.11)"><foreignObject style="--ltx-fo-width:6.03em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;font-size:9.25pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="77.15"><span id="A2.F22.pic1.1" style="font-size:90%;">Audio quality</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -93.99 22.57)"><foreignObject style="--ltx-fo-width:6.53em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;font-size:9.25pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="83.55"><span id="A2.F22.pic1.2" style="font-size:90%;">Motion quality</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -89.08 48.26)"><foreignObject style="--ltx-fo-width:6.14em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;font-size:9.25pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="78.64"><span id="A2.F22.pic1.3" style="font-size:90%;">Visual quality</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 -113.04 73.94)"><foreignObject style="--ltx-fo-width:8.02em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;font-size:9.25pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="102.6"><span id="A2.F22.pic1.4" style="font-size:90%;">Prompt alignment</span></foreignObject></g> <clipPath id="pgfcp1"><path d="M 0 -16.95 L 540.06 -16.95 L 540.06 94.01 L 0 94.01 Z"></path></clipPath><g clip-path="url(#pgfcp1)"><g style="--ltx-fill-color:#DF9080;" fill="#DF9080"><path style="stroke:none" d="M 385.24 -8.99 h 154.82 v 17.99 h -154.82 Z M 367.24 16.69 h 172.82 v 17.99 h -172.82 Z M 365.44 42.38 h 174.62 v 17.99 h -174.62 Z M 387.04 68.06 h 153.02 v 17.99 h -153.02 Z"></path></g><g></g><g style="--ltx-fill-color:#E1F0D9;" fill="#E1F0D9"><path style="stroke:none" d="M 162.02 -8.99 h 223.23 v 17.99 h -223.23 Z M 192.62 16.69 h 174.62 v 17.99 h -174.62 Z M 176.42 42.38 h 189.02 v 17.99 h -189.02 Z M 201.62 68.06 h 185.42 v 17.99 h -185.42 Z"></path></g><g></g><g style="--ltx-fill-color:#8CB754;" fill="#8CB754"><path style="stroke:none" d="M 0 -8.99 h 162.02 v 17.99 h -162.02 Z M 0 16.69 h 192.62 v 17.99 h -192.62 Z M 0 42.38 h 176.42 v 17.99 h -176.42 Z M 0 68.06 h 201.62 v 17.99 h -201.62 Z"></path></g><g></g></g><g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 433.03 -2.42)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.5" style="font-size:70%;--ltx-fg-color:#333333;">86 (28.7%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 424.03 23.26)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.6" style="font-size:70%;--ltx-fg-color:#333333;">96 (32.0%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 423.13 48.95)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.7" style="font-size:70%;--ltx-fg-color:#333333;">97 (32.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 433.93 74.63)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.8" style="font-size:70%;--ltx-fg-color:#333333;">85 (28.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 240.87 -2.42)"><foreignObject style="--ltx-fo-width:5.23em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="65.52"><span id="A2.F22.pic1.9" style="font-size:70%;--ltx-fg-color:#333333;">124 (41.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 250.31 23.26)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.10" style="font-size:70%;--ltx-fg-color:#333333;">97 (32.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 238.17 48.95)"><foreignObject style="--ltx-fo-width:5.23em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="65.52"><span id="A2.F22.pic1.11" style="font-size:70%;--ltx-fg-color:#333333;">105 (35.0%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 261.57 74.63)"><foreignObject style="--ltx-fo-width:5.23em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="65.52"><span id="A2.F22.pic1.12" style="font-size:70%;--ltx-fg-color:#333333;">103 (34.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 51.38 -2.42)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.13" style="font-size:70%;--ltx-fg-color:#333333;">90 (30.0%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 63.55 23.26)"><foreignObject style="--ltx-fo-width:5.23em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="65.52"><span id="A2.F22.pic1.14" style="font-size:70%;--ltx-fg-color:#333333;">107 (35.7%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 58.59 48.95)"><foreignObject style="--ltx-fo-width:4.73em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="59.25"><span id="A2.F22.pic1.15" style="font-size:70%;--ltx-fg-color:#333333;">98 (32.7%)</span></foreignObject></g> <g style="--ltx-stroke-color:#333333;--ltx-fill-color:#333333;" fill="#333333" stroke="#333333" transform="matrix(1.0 0.0 0.0 1.0 68.05 74.63)"><foreignObject style="--ltx-fo-width:5.23em;--ltx-fo-height:0.58em;--ltx-fo-depth:0.19em;font-size:9.06pt;" height="9.69" overflow="visible" transform="matrix(1 0 0 -1 0 7.26)" width="65.52"><span id="A2.F22.pic1.16" style="font-size:70%;--ltx-fg-color:#333333;">112 (37.3%)</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 178.95 -53.92)"><foreignObject style="--ltx-fo-width:14.26em;--ltx-fo-height:0.68em;--ltx-fo-depth:0.19em;font-size:9.25pt;" height="11.07" overflow="visible" transform="matrix(1 0 0 -1 0 8.65)" width="182.53"><span id="A2.F22.pic1.17" style="font-size:90%;">Paired prompt-seed comparisons</span></foreignObject></g> <g style="--ltx-stroke-color:#000000;--ltx-fill-color:#FFFFFF;" fill="#FFFFFF" stroke="#000000" transform="matrix(1.0 0.0 0.0 1.0 197.27 110.66)"><g transform="matrix(1 0 0 -1 0 16.12)"><g transform="matrix(1 0 0 1 0 8.06)"><g style="--ltx-fill-color:#8CB754;" fill="#8CB754" transform="matrix(1 0 0 -1 0 0) translate(0.28,0)"><path d="M 0 -2.96 h 11.07 v 8.3 h -11.07 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 11.62 0) translate(18.27,0) matrix(1.0 0.0 0.0 1.0 -15.5 -3.22)"><foreignObject style="--ltx-fo-width:2.26em;--ltx-fo-height:0.66em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.51" overflow="visible" transform="matrix(1 0 0 -1 0 8.51)" width="28.87"><span id="A2.F22.pic1.18" style="font-size:90%;">Wins</span></foreignObject></g> <g style="--ltx-fill-color:#E1F0D9;" fill="#E1F0D9" transform="matrix(1 0 0 -1 48.16 0) translate(0.28,0)"><path d="M 0 -2.96 h 11.07 v 8.3 h -11.07 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 59.78 0) translate(15.6,0) matrix(1.0 0.0 0.0 1.0 -12.84 -3.22)"><foreignObject style="--ltx-fo-width:1.84em;--ltx-fo-height:0.66em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.51" overflow="visible" transform="matrix(1 0 0 -1 0 8.51)" width="23.54"><span id="A2.F22.pic1.19" style="font-size:90%;">Ties</span></foreignObject></g> <g style="--ltx-fill-color:#DF9080;" fill="#DF9080" transform="matrix(1 0 0 -1 90.99 0) translate(0.28,0)"><path d="M 0 -2.96 h 11.07 v 8.3 h -11.07 Z"></path></g><g style="--ltx-stroke-color:#000000;--ltx-fill-color:#000000;" fill="#000000" stroke="#000000" transform="matrix(1 0 0 -1 102.61 0) translate(21.45,0) matrix(1.0 0.0 0.0 1.0 -18.69 -3.22)"><foreignObject style="--ltx-fo-width:2.75em;--ltx-fo-height:0.66em;--ltx-fo-depth:0em;font-size:9.25pt;" height="8.51" overflow="visible" transform="matrix(1 0 0 -1 0 8.51)" width="35.24"><span id="A2.F22.pic1.20" style="font-size:90%;">Losses</span></foreignObject></g></g></g></g></g></g></g></svg>

Figure 22: Per-axis judge preference between PDD and the official distilled LTX-2.3 model, both with 8 NFE, as scored by Gemini 3.1 Pro Preview. Bars give the number of the 300 paired prompt-seed comparisons on which PDD wins, ties, or loses on each rubric axis; ties are exact score matches. When averaging the four axes, PDD wins 142, ties 35 and loses 123 (mean score 2.62 against 2.59).

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/tied_man_in_van/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/brad_pitt_joker/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/ltx/laughing_man_park/teacher/frame_t0s.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/underwater_fish/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/kids_show_host/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/ltx/spiderman/teacher/frame_t0s.jpg)

Refer to caption

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/ufo_nyc/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![[Uncaptioned image]](https://arxiv.org/html/2607.26004v1/figures/ltx/smiling_woman_forest/teacher/frame_t0s.jpg)

\[Uncaptioned image\]

![Refer to caption](https://arxiv.org/html/2607.26004v1/figures/ltx/van_gogh_frog/teacher/frame_t0s.jpg)

Refer to caption

## Appendix C Connection to flow maps

In contrast to PDD, flow maps [^44] [^14] [^4] [^72] use continuous time. They define the mean velocity (4) between any two times of the flow process (1) by conditioning on $t$ and $s$. For a state $X_{t}=x_{t}$ at time $t\in[0,1]$, the mean velocity to time $s>t$ is

$$
u_{t,s}(x_{t})=\frac{1}{s-t}\int_{t}^{s}v_{r}(x_{r})dr.
$$

The Lagrangian formulation of flow maps [^4] [^72] shares similarities with our PDD, as it also employs an on-policy approximation during training. However, to learn the mean velocity of a fixed interval $[t_{n},t_{n+L}]$, we discretize it into $L$ intervals (a block) $t_{n}<\ldots<t_{n+L}$ and directly regress onto a numerical integration of the velocity in each interval, whereas Lagrangian flow maps regress the derivative of the displacement onto the velocity.

$$
{\mathcal{L}}(\theta)=\mathbb{E}\left[\left\|\frac{\partial}{\partial s}\left[(s-t_{n})u^{\theta}_{t_{n},s}\left(X_{t_{n}}\right)\right]-v_{s}\left(\text{sg}\left(X_{t_{n}}+(s-t_{n})u^{\theta}_{t_{n},s}\left(X_{t_{n}}\right)\right)\right)\right\|^{2}\right],\quad s\sim U[t_{n},t_{n+L}].
$$

Indeed, integrating the expression inside the norm in equation 19 w.r.t. $s$ yields a PDD-like objective (11) in continuous time.

Importantly, equation 19 is an example on a fixed interval. In practice, flow maps learn the mean velocity of any interval $[t,s]\subseteq[0,1]$ using the training objective

$$
{\mathcal{L}}(\theta)=\mathbb{E}\left[\left\|\frac{\partial}{\partial s}\left[(s-t)u^{\theta}_{t,s}\left(X_{t}\right)\right]-v_{s}\left(\text{sg}\left(X_{t}+(s-t)u^{\theta}_{t,s}\left(X_{t}\right)\right)\right)\right\|^{2}\right].
$$

## Appendix D Proofs

###### Proof of Proposition.

Consider a block starting at step $n\leq N-L$, and let $X_{n}\sim p_{t_{n}}$. Assume that the PD objective is realizable, so that its global minimum is zero, and let $\theta^{\star}$ be a global minimizer of the PD loss (11). Then, for every $k\in\left\{n,\ldots,n+L-1\right\}$ we have,

$$
\left\|\bar{u}^{\theta^{\star}}_{n}\left(k\mid X_{n}\right)-u_{k}\left(\bar{X}_{k}\right)\right\|^{2}=0.
$$

Hence,

$$
\bar{u}^{\theta^{\star}}_{n}\left(k\mid X_{n}\right)=u_{k}\left(\bar{X}_{k}\right).
$$

The parallel decoding condition (8) requires equality along the teacher trajectory $X_{k}$, as defined by the exact solution (3). We show that equation 22 implies $\bar{X}_{k}=X_{k}$ throughout the block $k=n,\ldots,n+L-1$.

Substituting equation 22 into the parallelized process (9) gives

$$
\bar{X}_{k+1}=\bar{X}_{k}+(t_{k+1}-t_{k})u_{k}\left(\bar{X}_{k}\right),\qquad\bar{X}_{n}=X_{n}.
$$

The base case is therefore $\bar{X}_{n}=X_{n}$. Assume that $\bar{X}_{k}=X_{k}$ for some $k\in\left\{n,\ldots,n+L-2\right\}$. Then

$$
\displaystyle\bar{X}_{k+1}
$$
 
$$
\displaystyle=X_{k}+(t_{k+1}-t_{k})u_{k}\left(X_{k}\right)
$$
 
$$
\displaystyle=X_{k+1},
$$

where the last equality follows from the exact solution (3). By induction, $\bar{X}_{k}=X_{k}$ for all $k\in\left\{n,\ldots,n+L-1\right\}$. ∎

[^1]: Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions, 2025. URL [https://arxiv.org/abs/2303.08797](https://arxiv.org/abs/2303.08797).

[^2]: Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL [https://arxiv.org/abs/2506.09985](https://arxiv.org/abs/2506.09985).

[^3]: Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models, 2025a. URL [https://arxiv.org/abs/2406.07507](https://arxiv.org/abs/2406.07507).

[^4]: Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. How to build a consistency model: Learning flow maps via self-distillation, 2025b. URL [https://arxiv.org/abs/2505.18825](https://arxiv.org/abs/2505.18825).

[^5]: Xu Cai, Yang Wu, Qianli Chen, Haoran Wu, Lichuan Xiang, and Hongkai Wen. Shortcutting pre-trained flow matching diffusion models is almost free lunch, 2025. URL [https://arxiv.org/abs/2510.17858](https://arxiv.org/abs/2510.17858).

[^6]: Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation, 2025. URL [https://arxiv.org/abs/2506.07977](https://arxiv.org/abs/2506.07977).

[^7]: Hansheng Chen, Kai Zhang, Hao Tan, Leonidas Guibas, Gordon Wetzstein, and Sai Bi. pi-flow: Policy-based few-step generation via imitation distillation, 2025. URL [https://arxiv.org/abs/2510.14974](https://arxiv.org/abs/2510.14974).

[^8]: Zhenglin Cheng, Peng Sun, Jianguo Li, and Tao Lin. Twinflow: Realizing one-step generation on large models with self-adversarial flows, 2026. URL [https://arxiv.org/abs/2512.05150](https://arxiv.org/abs/2512.05150).

[^9]: LightX2V Contributors. Lightx2v: Light video generation inference framework. [https://github.com/ModelTC/lightx2v](https://github.com/ModelTC/lightx2v), 2025.

[^10]: Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL [https://arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206).

[^11]: Xiangyu Fan, Zesong Qiu, Zhuguanyu Wu, Fanzhou Wang, Zhiqian Lin, Tianxiang Ren, Dahua Lin, Ruihao Gong, and Lei Yang. Phased dmd: Few-step distribution matching distillation via score matching within subintervals, 2026. URL [https://arxiv.org/abs/2510.27684](https://arxiv.org/abs/2510.27684).

[^12]: Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models, 2025. URL [https://arxiv.org/abs/2410.12557](https://arxiv.org/abs/2410.12557).

[^13]: Zhengyang Geng, Ashwini Pokle, William Luo, Justin Lin, and J. Zico Kolter. Consistency models made easy, 2024. URL [https://arxiv.org/abs/2406.14548](https://arxiv.org/abs/2406.14548).

[^14]: Zhengyang Geng, Mingyang Deng, Xingjian Bai, J. Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling, 2025a. URL [https://arxiv.org/abs/2505.13447](https://arxiv.org/abs/2505.13447).

[^15]: Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J. Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models, 2025b. URL [https://arxiv.org/abs/2512.02012](https://arxiv.org/abs/2512.02012).

[^16]: Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment, 2023. URL [https://arxiv.org/abs/2310.11513](https://arxiv.org/abs/2310.11513).

[^17]: Yuchao Gu, Guian Fang, Yuxin Jiang, Weijia Mao, Song Han, Han Cai, and Mike Zheng Shou. Anyflow: Any-step video diffusion model with on-policy flow map distillation, 2026. URL [https://arxiv.org/abs/2605.13724](https://arxiv.org/abs/2605.13724).

[^18]: Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, and Yuxuan Wang. Splitmeanflow: Interval splitting consistency in few-step generative modeling, 2025. URL [https://arxiv.org/abs/2507.16884](https://arxiv.org/abs/2507.16884).

[^19]: Yoav HaCohen, Benny Brazowski, Nisan Chiprut, Yaki Bitterman, Andrew Kvochko, Avishai Berkowitz, Daniel Shalem, Daphna Lifschitz, Dudu Moshe, Eitan Porat, Eitan Richardson, Guy Shiran, Itay Chachy, Jonathan Chetboun, Michael Finkelson, Michael Kupchick, Nir Zabari, Nitzan Guetta, Noa Kotler, Ofir Bibi, Ori Gordon, Poriya Panet, Roi Benita, Shahar Armon, Victor Kulikov, Yaron Inger, Yonatan Shiftan, Zeev Melumian, and Zeev Farbman. Ltx-2: Efficient joint audio-visual foundation model, 2026. URL [https://arxiv.org/abs/2601.03233](https://arxiv.org/abs/2601.03233).

[^20]: Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. URL [https://arxiv.org/abs/2207.12598](https://arxiv.org/abs/2207.12598).

[^21]: Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. URL [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239).

[^22]: Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024. URL [https://arxiv.org/abs/2403.05135](https://arxiv.org/abs/2403.05135).

[^23]: Zheyuan Hu, Chieh-Hsin Lai, Yuki Mitsufuji, and Stefano Ermon. Cmt: Mid-training for efficient learning of consistency, mean flow, and flow map models, 2026. URL [https://arxiv.org/abs/2509.24526](https://arxiv.org/abs/2509.24526).

[^24]: Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models, 2023. URL [https://arxiv.org/abs/2311.17982](https://arxiv.org/abs/2311.17982).

[^25]: Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models, 2025. URL [https://arxiv.org/abs/2412.03603](https://arxiv.org/abs/2412.03603).

[^26]: Black Forest Labs. FLUX.2: Frontier Visual Intelligence. [https://bfl.ai/blog/flux-2](https://bfl.ai/blog/flux-2), 2025.

[^27]: Kyungmin Lee, Sihyun Yu, and Jinwoo Shin. Decoupled meanflow: Turning flow models into flow maps for accelerated sampling, 2025. URL [https://arxiv.org/abs/2510.24474](https://arxiv.org/abs/2510.24474).

[^28]: Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. Diffusion adversarial post-training for one-step video generation, 2025. URL [https://arxiv.org/abs/2501.08316](https://arxiv.org/abs/2501.08316).

[^29]: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747).

[^30]: Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. URL [https://arxiv.org/abs/2209.03003](https://arxiv.org/abs/2209.03003).

[^31]: Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101).

[^32]: Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models, 2025. URL [https://arxiv.org/abs/2410.11081](https://arxiv.org/abs/2410.11081).

[^33]: Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, and Kaiming He. One-step latent-free image generation with pixel mean flows, 2026. URL [https://arxiv.org/abs/2601.22158](https://arxiv.org/abs/2601.22158).

[^34]: Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing high-resolution images with few-step inference, 2023. URL [https://arxiv.org/abs/2310.04378](https://arxiv.org/abs/2310.04378).

[^35]: Tianze Luo, Haotian Yuan, and Zhuang Liu. Soflow: Solution flow models for one-step generative modeling, 2026. URL [https://arxiv.org/abs/2512.15657](https://arxiv.org/abs/2512.15657).

[^36]: Weijian Luo, Zemin Huang, Zhengyang Geng, J. Zico Kolter, and Guo jun Qi. One-step diffusion distillation through score implicit matching, 2024. URL [https://arxiv.org/abs/2410.16794](https://arxiv.org/abs/2410.16794).

[^37]: Anh Nguyen, Viet Nguyen, Duc Vu, Trung Dao, Chi Tran, Toan Tran, and Anh Tran. Improved training technique for shortcut models, 2025. URL [https://arxiv.org/abs/2510.21250](https://arxiv.org/abs/2510.21250).

[^38]: Weili Nie, Julius Berner, Chao Liu, and Arash Vahdat. Nvidia fastgen: Fast generation from diffusion models, 2026a. URL [https://github.com/NVlabs/FastGen](https://github.com/NVlabs/FastGen).

[^39]: Weili Nie, Julius Berner, Nanye Ma, Chao Liu, Saining Xie, and Arash Vahdat. Transition matching distillation for fast video generation, 2026b. URL [https://arxiv.org/abs/2601.09881](https://arxiv.org/abs/2601.09881).

[^40]: NVIDIA. Cosmos 3: Omnimodal world models for physical ai, 2026. URL [https://arxiv.org/abs/2606.02800](https://arxiv.org/abs/2606.02800).

[^41]: Dogyun Park, Yanyu Li, Sergey Tulyakov, and Anil Kag. Eflow: Fast few-step video generator training from scratch via efficient solution flow, 2026. URL [https://arxiv.org/abs/2603.27086](https://arxiv.org/abs/2603.27086).

[^42]: Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022. URL [https://arxiv.org/abs/2112.10752](https://arxiv.org/abs/2112.10752).

[^43]: Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. ImageNet Large Scale Visual Recognition Challenge. *International Journal of Computer Vision (IJCV)*, 115(3):211–252, 2015. [10.1007/s11263-015-0816-y](https://doi.org/10.1007/s11263-015-0816-y).

[^44]: Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation, 2025. URL [https://arxiv.org/abs/2506.14603](https://arxiv.org/abs/2506.14603).

[^45]: Tim Salimans and Jonathan Ho. Progressive distillation for fast sampling of diffusion models, 2022. URL [https://arxiv.org/abs/2202.00512](https://arxiv.org/abs/2202.00512).

[^46]: Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation, 2023. URL [https://arxiv.org/abs/2311.17042](https://arxiv.org/abs/2311.17042).

[^47]: Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast high-resolution image synthesis with latent adversarial diffusion distillation, 2024. URL [https://arxiv.org/abs/2403.12015](https://arxiv.org/abs/2403.12015).

[^48]: Neta Shaul, Uriel Singer, Ricky T. Q. Chen, Matthew Le, Ali Thabet, Albert Pumarola, and Yaron Lipman. Bespoke non-stationary solvers for fast sampling of diffusion and flow models, 2024. URL [https://arxiv.org/abs/2403.01329](https://arxiv.org/abs/2403.01329).

[^49]: Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics, 2015. URL [https://arxiv.org/abs/1503.03585](https://arxiv.org/abs/1503.03585).

[^50]: Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models, 2023. URL [https://arxiv.org/abs/2310.14189](https://arxiv.org/abs/2310.14189).

[^51]: Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021. URL [https://arxiv.org/abs/2011.13456](https://arxiv.org/abs/2011.13456).

[^52]: Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models, 2023. URL [https://arxiv.org/abs/2303.01469](https://arxiv.org/abs/2303.01469).

[^53]: Peng Sun and Tao Lin. Any-step generation via n-th order recursive consistent velocity field estimation. In *The Fourteenth International Conference on Learning Representations*, 2026. URL [https://openreview.net/forum?id=GnawtLKGkP](https://openreview.net/forum?id=GnawtLKGkP).

[^54]: Joshua Tian Jin Tee, Kang Zhang, Hee Suk Yoon, Dhananjaya Nagaraja Gowda, Chanwoo Kim, and Chang D. Yoo. Physics informed distillation for diffusion models, 2024. URL [https://arxiv.org/abs/2411.08378](https://arxiv.org/abs/2411.08378).

[^55]: Shangyuan Tong, Nanye Ma, Saining Xie, and Tommi Jaakkola. Flow map distillation without data, 2025. URL [https://arxiv.org/abs/2511.19428](https://arxiv.org/abs/2511.19428).

[^56]: Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL [https://arxiv.org/abs/2503.20314](https://arxiv.org/abs/2503.20314).

[^57]: Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking, 2023a. URL [https://arxiv.org/abs/2303.16727](https://arxiv.org/abs/2303.16727).

[^58]: Wenhao Wang and Yi Yang. VidProM: A million-scale real prompt-gallery dataset for text-to-video diffusion models, 2024. URL [https://arxiv.org/abs/2403.06098](https://arxiv.org/abs/2403.06098).

[^59]: Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation, 2023b. URL [https://arxiv.org/abs/2305.16213](https://arxiv.org/abs/2305.16213).

[^60]: Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, and Lei Bai. Transition models: Rethinking the generative learning objective, 2025. URL [https://arxiv.org/abs/2509.04394](https://arxiv.org/abs/2509.04394).

[^61]: Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL [https://arxiv.org/abs/2508.02324](https://arxiv.org/abs/2508.02324).

[^62]: Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis, 2023. URL [https://arxiv.org/abs/2306.09341](https://arxiv.org/abs/2306.09341).

[^63]: Howard Xiao, Brian Chao, Lior Yariv, and Gordon Wetzstein. Spectral progressive diffusion for efficient image and video generation, 2026. URL [https://arxiv.org/abs/2605.18736](https://arxiv.org/abs/2605.18736).

[^64]: Yilun Xu, Weili Nie, and Arash Vahdat. One-step diffusion models with $f$ -divergence distribution matching, 2025. URL [https://arxiv.org/abs/2502.15681](https://arxiv.org/abs/2502.15681).

[^65]: Timing Yang, Sucheng Ren, Alan Yuille, and Feng Wang. Vimix-14m: A curated multi-source video-text dataset with long-form, high-quality captions and crawl-free access, 2025. URL [https://arxiv.org/abs/2511.18382](https://arxiv.org/abs/2511.18382).

[^66]: Tianwei Yin, Michaël Gharbi, Taesung Park, Richard Zhang, Eli Shechtman, Fredo Durand, and William T. Freeman. Improved distribution matching distillation for fast image synthesis, 2024a. URL [https://arxiv.org/abs/2405.14867](https://arxiv.org/abs/2405.14867).

[^67]: Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T. Freeman, and Taesung Park. One-step diffusion with distribution matching distillation, 2024b. URL [https://arxiv.org/abs/2311.18828](https://arxiv.org/abs/2311.18828).

[^68]: Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think, 2025. URL [https://arxiv.org/abs/2410.06940](https://arxiv.org/abs/2410.06940).

[^69]: Huijie Zhang, Aliaksandr Siarohin, Willi Menapace, Michael Vasilkovsky, Sergey Tulyakov, Qing Qu, and Ivan Skorokhodov. Alphaflow: Understanding and improving meanflow models, 2025. URL [https://arxiv.org/abs/2510.20771](https://arxiv.org/abs/2510.20771).

[^70]: Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency, 2025. URL [https://arxiv.org/abs/2510.08431](https://arxiv.org/abs/2510.08431).

[^71]: Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching, 2025. URL [https://arxiv.org/abs/2503.07565](https://arxiv.org/abs/2503.07565).

[^72]: Linqi Zhou, Mathias Parger, Ayaan Haque, and Jiaming Song. Terminal velocity matching, 2026. URL [https://arxiv.org/abs/2511.19797](https://arxiv.org/abs/2511.19797).

[^73]: Mingyuan Zhou, Huangjie Zheng, Yi Gu, Zhendong Wang, and Hai Huang. Adversarial score identity distillation: Rapidly surpassing the teacher in one step, 2024a. URL [https://arxiv.org/abs/2410.14919](https://arxiv.org/abs/2410.14919).

[^74]: Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation, 2024b. URL [https://arxiv.org/abs/2404.04057](https://arxiv.org/abs/2404.04057).