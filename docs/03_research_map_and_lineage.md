# FOUNDATIONAL SPECIFICATION 03: RESEARCH MAP, HISTORICAL LINEAGE, AND UNEXPLORED OPPORTUNITIES

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture)  
**Status:** Canonical Foundation Document  

---

## 1. Taxonomic Map of Computational Intelligence Paradigms

```
                                  [ THE INTELLECTUAL GENEALOGY ]

   CYBERNETICS & CONTROL THEORY           SYMBOLIC COGNITION            STATISTICAL MACHINE LEARNING
   (Wiener 1948, Ashby 1952)          (Newell & Simon 1959)             (Rumelhart 1986, LeCun 1989)
         │                                   │                                      │
         ▼                                   ▼                                      ▼
   OPTIMAL CONTROL & LQG               BDI ARCHITECTURES                  RECURRENT NETS & LSTMs
   (Pontryagin 1962, Kalman 1960)     (Rao & Georgeff 1995)               (Elman 1990, Schmidhuber 1997)
         │                                   │                                      │
         ▼                                   ▼                                      ▼
   ACTIVE INFERENCE & FEP             SOAR & ACT-R HYBRIDS                TRANSFORMERS & ATTENTION
   (Friston 2006-2024)                (Laird 2012, Anderson 2007)         (Vaswani et al. 2017)
         │                                   │                                      │
         │                                   │                            SELECTIVE SSMs / MAMBA
         │                                   │                            (Gu & Dao 2023)
         │                                   │                                      │
         └─────────────────────────┬─────────┴──────────────────────────────────────┘
                                   │
                                   ▼
                       [ THE UNEXPLORED SYNTHESIS ]
                ICIA: INTENTION-CENTRED INTELLIGENCE ARCHITECTURE
```

---

## 2. Lineage Analysis: What Each Paradigm Achieved and Where It Failed

### 2.1 Cybernetics, Homeostasis, and Active Inference
- **Core Principle:** An intelligent agent is an open thermodynamic system that resists entropic dissolution by maintaining its internal variables within homeostatic bounds via closed feedback loops (Wiener, Ashby, Friston).
- **Achievements:** Rigorous mathematical formalization of teleology as negative feedback. Active Inference proves that perception (updating beliefs $\mathcal{B}$) and action (transforming reality to match target prior $\mathcal{G}$) minimize the same Variational Free Energy functional:
  $$\mathcal{F} = D_{KL}(q(s) \parallel p(s)) - \mathbb{E}_{q(s)}[\log p(o \mid s)]$$
- **Failure Mode in AI:** Historically relegated to low-dimensional motor control or stylized toy POMDPs due to the computational intractability of exact Bayesian belief updating in high-dimensional semantic spaces.

### 2.2 Symbolic Cognitive Architectures (BDI, SOAR, ACT-R)
- **Core Principle:** Intelligence is physical symbol manipulation guided by explicit goals, production rules, and means-ends analysis (Newell, Simon, Laird, Anderson).
- **Achievements:** First explicit computational representation of intentions. Formalized goal stacks, sub-goal generation, and impasse resolution.
- **Failure Mode in AI:** Complete inability to learn continuous representations from raw sensory data. Extremely brittle; combinatorial explosion under uncertainty; lack of end-to-end differentiability.

### 2.3 Deep Connectionism: Recurrent Models to Transformers
- **Core Principle:** Continuous gradient-based representation learning from large-scale data.
  - *RNN/LSTM:* State compression via recurrence ($h_t = f(h_{t-1}, x_t)$). Suffered from vanishing gradients and bounded memory capacity.
  - *Transformers:* Replaced recurrence entirely with self-attention ($O(N^2)$ global associative routing). Unlocked massive parallel pre-training.
  - *Selective SSMs (Mamba/S4):* Returned to linear recurrence with input-dependent selection ($h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$).
- **Achievements:** Scalable representation learning, emergent world knowledge, and fluent multimodal generation.
- **Failure Mode in AI:** They are **fundamentally teleologically blind**. They model conditional distributions $p(x_t \mid x_{<t})$. They have no concept of goal maintenance, no continuous reality verification loop, and are prone to context drift, hallucination, and adversarial distraction.

### 2.4 External Agent Frameworks (The "Wrapper" Illusion)
- **Current Trend:** AutoGen, CrewAI, LangChain, Tool-calling LLM loops.
- **Critical Flaw:** These systems do not innovate at the architectural or computational level. They place an ad-hoc heuristic string-parsing wrapper around a passive next-token predictor. They suffer from high latency, cascading errors, quadratic token accumulation costs, and zero mathematical stability guarantees.

---

## 3. The Unexplored Opportunity: The ICIA Breakthrough

The unexplored scientific opportunity is the construction of a **natively intention-centred, differentiable computational architecture** that fuses:

1. **Differentiable Teleological Dynamics:** Intention is not a prompt string; it is a continuous-symbolic tensor $\mathcal{I}_t$ that dynamically modulates the state-space matrices ($A(\mathcal{I}), B(\mathcal{I}), C(\mathcal{I})$) or neural vector fields ($f(S; \mathcal{I})$).
2. **Contractive Reality Verification:** An online recurrent loop that continuously computes the Lie bracket or geodesic distance between the intended state trajectory and observed sensory outcomes:
   $$\mathcal{E}_{\text{reality}}(t) = d_{\mathcal{M}}\left(\mathcal{G}_t, \Psi_{\text{belief}}(S_t)\right)$$
3. **Selective Orthogonal Routing:** Input stimuli $X_t$ are decomposed into components along the intention gradient $\nabla_{\mathcal{I}} \mathcal{V}$ (amplified) and components orthogonal to the intention (suppressed or redirected to the background anomaly detector).
4. **Sub-Quadratic Bounded Memory:** Persistent memory organized as an associative attractor basin indexed by intention hashes, eliminating the linear growth of the Transformer KV-cache.
