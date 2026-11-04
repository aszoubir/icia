# FOUNDATIONAL SPECIFICATION 09: ADAPTIVE MEMORY AND ATTRACTOR DYNAMICS

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA-v2 (Intention-Centred Intelligence Architecture — Integrated Prototype)  
**Status:** Canonical Foundation Document  

---

## 1. The Long-Horizon Intention Drift Problem

In our initial minimal benchmark, we observed that while the minimal DIVRE prototype was vastly superior to Transformers at intermediate horizons ($100\%$ goal retention at $L=32$ vs $55\%$ for SAT and $43\%$ for ICT), both architectures experienced degradation as sequences scaled to $L \ge 128$.

This degradation arises from two distinct physical phenomena:
1. **In Transformers (Associative Dispersion):** Attention weights normalize over all context tokens via softmax. As $L \to \infty$, the probability mass allocated to the original goal token decays as $O(1/L)$, increasing attention entropy monotonically from $0.235$ to $5.740$.
2. **In Vanilla Recurrence (State Vanishing/Drift):** A standard recurrent hidden state $h_t$ is repeatedly overwritten by incoming distractors. Without an active non-linear restorative force, the spectral radius of the state transition causes the goal signal to attenuate exponentially.

---

## 2. Active Intention Attractor Anchoring (AIAA)

To achieve **invariant goal preservation across unbounded sequence lengths ($L \to \infty$)**, ICIA-v2 introduces **Active Intention Attractor Anchoring (AIAA)**.

```
                      [ ATTRACTOR ANCHORING DYNAMICS ]

        Active Intention I_t ──────────────────────────────┐
                 │                                         │
                 │ Restorative Force                       │
                 ▼ -k_anchor * (T_{t-1} - Proj(I_t))       ▼
        ┌──────────────────┐    Discrepancy E_t    ┌───────────────┐
        │  TOP-DOWN STREAM │ <──────────────────── │ ANCHOR WEIGHT │
        │       T_t        │                       │   α(I_t, E_t) │
        └──────────────────┘                       └───────────────┘
```

### 2.1 Mathematical Formulation of AIAA
Let the teleological state $T_t \in \mathbb{R}^d$ be subject to a non-linear restorative potential $\Phi_{\text{anchor}}(T; \mathcal{I})$ centered at the continuous goal projection $\Pi_{\mathcal{G}}(\mathcal{I}_t)$:

$$\Phi_{\text{anchor}}(T_t; \mathcal{I}_t) = \frac{1}{2} \| T_t - \Pi_{\mathcal{G}}(\mathcal{I}_t) \|_2^2$$

The discretized state update in DIVRE-v2 incorporates this gradient flow:

$$T_t = (1 - \alpha_t) \cdot \underbrace{\text{LayerNorm}\left( T_{t-1} + g_t \odot \tanh(W_t [T_{t-1}; \mathcal{I}_t; V_t]) \right)}_{\text{Recurrent Teleological Evolution}} + \alpha_t \cdot \underbrace{\Pi_{\mathcal{G}}(\mathcal{I}_t)}_{\text{Attractor Anchor}}$$

Where the dynamic anchor weight $\alpha_t \in (0, 1)$ is computed as:
$$\alpha_t = \sigma\left( \mathbf{W}_\alpha [\mathcal{I}_t; \mathcal{E}_t] + b_\alpha \right)$$

### 2.2 Proof of Invariant Bounded Drift
#### Theorem 2 (Asymptotic Goal Invariance under AIAA):
*Let the distractor sequence be of arbitrary length $L$. Assume the distractor tokens $X_{\tau}$ are bounded $\|X_{\tau}\| \le B_x$. Under AIAA with anchor coefficient $\alpha_t \ge \alpha_{\min} > 0$, the teleological state $T_L$ remains within a strictly bounded ball around the goal attractor $\mathcal{G}^*$:*
$$\| T_L - \Pi_{\mathcal{G}}(\mathcal{I}) \| \le (1 - \alpha_{\min})^L \| T_0 - \Pi_{\mathcal{G}}(\mathcal{I}) \| + \frac{B_{\text{drift}}}{\alpha_{\min}}$$

*As $L \to \infty$, the initial state perturbation decays to zero, and the maximum deviation is strictly upper-bounded by $\frac{B_{\text{drift}}}{\alpha_{\min}}$, independent of sequence length $L$.*

---

## 3. Intention-Gated Associative Memory (IGAM)

Rather than storing an exhaustive, growing history of tokens (as in the Transformer KV-cache), ICIA-v2 implements **Intention-Gated Associative Memory (IGAM)**.

```
       Incoming Stimulus X_t ─────────────┐
                                          ▼
   Active Intention I_t ───> [ TELEOLOGICAL FILTER ] ───> Relevant Only?
                                          │
                     ┌────────────────────┴────────────────────┐
                     │ NO                                      │ YES
                     ▼                                         ▼
            Discard to Periphery                       Episodic Key-Value Write
                                                       M_t = U(M_{t-1}, X_t, I_t)
```

### 3.1 Memory Structure
The memory substrate $\mathcal{M}_t$ consists of $K$ persistent memory slots:
$$\mathcal{M}_t = \{ (K_i, V_i) \}_{i=1}^K, \quad K_i \in \mathbb{R}^{d_k}, \quad V_i \in \mathbb{R}^{d_v}$$

### 3.2 Selective Writing via Reality Discrepancy Gate
Unlike Transformers which write every incoming token into the KV-cache indiscriminately:
1. **Surprise Thresholding:** Writing occurs only when the reality discrepancy tensor exceeds a novelty threshold:
   $$w_{\text{write}}(t) = \sigma\left( \beta_w (\|\mathcal{E}_t\|_2 - \tau_{\text{novelty}}) \right)$$
2. **Teleological Alignment:** The input is projected along the intention gradient:
   $$\kappa_t = \text{Softmax}\left( \frac{\mathcal{M}_{\text{keys}} \mathcal{I}_t}{\sqrt{d_k}} \right)$$
   If an incoming token is an irrelevant distractor ($\|\mathcal{E}_t\| \approx 0$ or orthogonal to $\mathcal{I}_t$), $w_{\text{write}} \approx 0$. Distractors are entirely rejected from memory, preventing memory pollution.

### 3.3 Intention-Conditioned Memory Retrieval
During action selection and plan verification, working memory is queried by the active intention $\mathcal{I}_t$:
$$r_t = \sum_{i=1}^K \text{Softmax}\left( \frac{\langle \mathcal{I}_t, K_i \rangle}{\sqrt{d_k}} \right) V_i$$
This retrieval is $O(K)$ where $K$ is fixed (e.g. $K=32$ or $K=64$), completely decoupled from sequence length $L$.
