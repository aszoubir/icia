# FOUNDATIONAL SPECIFICATION 07: RIGOROUS EXPERIMENTAL BENCHMARK PROTOCOL

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture)  
**Status:** Canonical Foundation Document  

---

## 1. Experimental Methodology and Standardization

To prevent the common pitfall of comparing disparate models with mismatched parameter budgets or unfair token allocations, all architectures in this experimental campaign adhere to a **strictly standardized benchmark protocol**:

### 1.1 Controlled Invariants
- **Parameter Parity:** All candidate models and baselines are matched within $\pm 5\%$ parameter count ($\approx 150\text{k} - 250\text{k}$ parameters for minimal first-execution validation; scalable to larger parameter classes).
- **Hidden Dimensionality:** Latent embedding dimension $d = 64$ or $128$, feedforward hidden dimension $d_{ff} = 256$.
- **Training Budget:** Identical number of gradient updates (1,000 to 5,000 steps), identical AdamW optimizer settings ($\text{lr} = 10^{-3}, \beta_1 = 0.9, \beta_2 = 0.98, \text{weight\_decay} = 0.01$).
- **Independent Test Sets:** 1,000 synthetic test episodes generated with fixed seeds never seen during training.

---

## 2. Benchmark Task Suite

```
                                  [ THE BENCHMARK BATTERY ]

   1. DISTRACTOR PERTURBATION          2. DYNAMIC CONSTRAINT PIVOT        3. RESOURCE-BOUNDED PLANNING
    (Goal Drift / Null Space)              (Autoregressive Inertia)            (Pareto Efficiency)

   [Goal G0]                           [State S_t -> G_A]                  Target G* in Obstacle Grid
       │                                       │                                       │
       ▼                                       ▼                                       ▼
   [L Distractor Tokens]               [SURPRISE: C_new at t_flip]         Compute within FLOP Cap
       │                                       │                                       │
       ▼                                       ▼                                       ▼
   Verify: Is Final Action             Verify: Does Model Pivot            Verify: E_reality < ε
   faithful to G0?                     within 1 step without drift?        Optimal Path / Budget
```

### Benchmark 1: Distractor Perturbation & Goal Preservation (DPGP)
- **Scientific Question:** Does the model maintain its intended goal as the volume of intervening irrelevant or adversarial context increases?
- **Protocol:**
  1. An explicit intention specification $\mathcal{I}_0 = (G_0, C_0)$ is presented at $t=0$.
  2. A sequence of distractor tokens of length $L \in \{8, 32, 128, 512\}$ is streamed. Distractors contain syntactically valid instructions urging alternate goals $G_{\text{fake}}$.
  3. The model must emit the terminal plan vector $A^*$.
- **Metric:** Goal Fidelity Score $\mathcal{F}_{\text{goal}} = \mathbb{I}[A^* = G_0]$ and Attention/State Drift Metric $\delta_{\text{drift}} = \| S_L - S_0 \|_{\mathcal{G}}$.

### Benchmark 2: Dynamic Constraint Pivot & Runtime Recovery (DCPR)
- **Scientific Question:** How rapidly can the architecture pivot when an unexpected environmental constraint is imposed mid-sequence, without wiping or regenerating context?
- **Protocol:**
  1. The agent executes a 10-step plan toward Goal $G_A$ under constraint $C_{\text{allow}} = \{ \text{Path } 1, 2 \}$.
  2. At step $t = 5$, the environment injects a hard obstacle blocking Path 1.
  3. The model must immediately re-route to Path 2 without emitting any invalid actions along Path 1.
- **Metric:** Pivot Latency $\tau_{\text{pivot}}$ (number of steps to transition to a valid path) and Constraint Violation Count $\kappa_{\text{violations}}$.

### Benchmark 3: Teleological Discrepancy & Verification Efficiency (TDVE)
- **Scientific Question:** Can the architecture continuously detect whether its actions are succeeding in reality, and adaptively scale its computation based on discrepancy?
- **Protocol:**
  1. The agent navigates a stochastic state graph where actions have probabilistic transitions.
  2. The agent monitors its internal discrepancy tensor $\mathcal{E}_t$.
  3. When an action slips or fails, the agent must detect the discrepancy and spend computational recurrence to re-orient.
- **Metric:** Calibration Error of $\mathcal{E}_t$ against ground-truth environmental state difference, and Total Energy/FLOPs spent to achieve verified completion.

---

## 3. Comparison Baselines

1. **Baseline 1: Standard Autoregressive Transformer (SAT)**
   - Causal multi-head self-attention with learned positional embeddings.
2. **Baseline 2: Intention-Conditioned Transformer (ICT)**
   - Transformer augmented with prepended goal token embeddings and cross-attention goal conditioning.
3. **Candidate ICIA Model: DIVRE-SSM Hybrid (Primary Executable Architecture)**
   - Dual-stream top-down teleological recurrent flow coupled with bottom-up reality verification and selective intention state-space modulation.
