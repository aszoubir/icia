# FOUNDATIONAL SPECIFICATION 06: COMPARATIVE ARCHITECTURAL ANALYSIS AND FALSIFICATION PROTOCOL

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture)  
**Status:** Canonical Foundation Document  

---

## 1. Multi-Dimensional Architectural Comparison

| Dimension | 1. Baseline Transformer | 2. Intention-Prompted Transformer | 3. External Agent Wrapper | 4. Candidate A: IDFN | 5. Candidate B: HISSA | 6. Candidate C: DIVRE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Organizing Primitive** | Pairwise Attention $\text{softmax}(QK^T / \sqrt{d})$ | Attention with prepended goal token | String-level prompt loop over LLM | Continuous Lyapunov Energy Gradient $-\nabla E$ | Intentional Spectral State Matrix $A(\mathcal{I})$ | Symplectic Dual-Stream Error Coupling |
| **Mathematical Formulation** | Discrete matrix projection | Discrete prefix conditioning | Discrete Markov chain of prompts | Non-linear ordinary differential equations | Multi-timescale selective continuous SSM | Bifurcated symplectic dynamical recurrence |
| **Context Memory Scaling** | $O(N)$ KV-cache; $O(N^2)$ cross-attn | $O(N)$ KV-cache; $O(N^2)$ cross-attn | $O(N \cdot K)$ cascading prompt tokens | $O(1)$ constant hidden state $S \in \mathbb{R}^d$ | $O(1)$ constant hidden state $h \in \mathbb{R}^{d_h}$ | $O(1)$ dual vectors $T_t, V_t \in \mathbb{R}^d$ |
| **Inference Step Cost** | $O(N \cdot d)$ FLOPs | $O((N + N_g) \cdot d)$ FLOPs | Variable ($10^3 \sim 10^5$ tokens / loop) | $O(N_{\text{steps}} \cdot d^2)$ solver cost | $O(d)$ strictly linear recurrent FLOPs | $O(d^2)$ matrix-vector recurrence |
| **Teleological Fidelity** | Poor: decays with distractor length | Moderate: decays when distractors dominate | Fragile: relies on model self-reflection | High: mathematically bounded attractor | High: spectral filter drops off-target noise | Absolute: closed-loop error verification |
| **Reality Verification** | None (open-loop generation) | None (open-loop generation) | Heuristic (string parsing or regex test) | Energy minimization residual $\| \nabla E \|$ | Residual state norm $\| y_t - y^*_t \|$ | Native Discrepancy Tensor $\mathcal{E}_t = X_t - \hat{V}_t$ |
| **Hardware Alignment** | High (GEMM, Tensor Cores) | High (GEMM, Tensor Cores) | Low (I/O latency, network roundtrips) | Moderate (solver requires sequential steps) | Very High (associative scan & 1D conv) | High (dense matrix multiplications) |
| **Implementation Complexity** | Baseline | Low (add prefix embedding) | Low (Python orchestration script) | High (differentiable ODE integrator) | High (hardware-efficient SSM kernel) | Moderate-High (coupled dual-stream) |

---

## 2. Rigorous Falsification Criteria for ICIA Candidates

A scientific paradigm must be strictly falsifiable. We specify the exact conditions under which each architectural candidate shall be declared falsified:

### 2.1 Falsification of Candidate A (IDFN)
Candidate A shall be **falsified and rejected** if:
1. *Integration Divergence:* The adaptive numerical solver requires $\ge 20$ internal integration steps to achieve attractor convergence, causing latency to exceed standard Transformer attention by $>300\%$ without a Pareto improvement in accuracy.
2. *Gradient Vanishing in Continuous Time:* Adjoint gradient propagation across long temporal horizons ($T > 1,000$) exhibits numerical instability or vanishing gradients comparable to vanilla continuous RNNs.
3. *Empirical Failure:* Accuracy on the Distractor Perturbation Benchmark falls below $95\%$ at $N = 10,000$ distractor tokens.

### 2.2 Falsification of Candidate B (HISSA)
Candidate B shall be **falsified and rejected** if:
1. *Non-Linear Binding Failure:* Spectral modulation of matrix $A(\mathcal{I})$ fails to represent multi-clause compositional constraint satisfaction problems where multiple constraints are non-linearly entangled.
2. *Empirical Failure:* When tested on dynamic sub-goal inversion (where the target goal flips mid-sequence), the recurrent state $h_t$ fails to clear stale goal momentum within 3 tokens.

### 2.3 Falsification of Candidate C (DIVRE)
Candidate C shall be **falsified and rejected** if:
1. *Symplectic Instability:* The dual-stream interaction matrix produces oscillatory or exploding gradients during training due to uncalibrated feedback loops between $T_t$ and $V_t$.
2. *Discrepancy Saturation:* The teleological discrepancy tensor $\mathcal{E}_t$ saturates to near-zero or near-constant values, failing to differentiate between minor sensory noise and fundamental goal deviation.
3. *Empirical Failure:* DIVRE fails to detect an unannounced environmental rule shift within $\le 2$ observation steps.

---

## 3. The Synthetic Convergence: Selecting the Primary Vehicle

While all three candidates present profound mathematical advantages over passive attention, **Candidate C (DIVRE)** combined with **Candidate B’s selective state parameterization** represents the most immediately realizable, mathematically stable, and empirically testable vehicle for our first execution:

- It provides an explicit **closed-loop reality verification tensor** $\mathcal{E}_t$.
- It preserves $O(1)$ constant memory scaling during generation.
- It is fully differentiable end-to-end and can be implemented in pure, highly optimized tensor mathematics.
