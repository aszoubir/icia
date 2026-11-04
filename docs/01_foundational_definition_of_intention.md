# FOUNDATIONAL SPECIFICATION 01: A RIGOROUS DEFINITION OF COMPUTATIONAL INTENTION

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture)  
**Status:** Canonical Foundation Document  

---

## 1. The Core Ontological Distinction

In contemporary computational intelligence (specifically Transformers and Large Language Models), the system is fundamentally organized around **probabilistic next-token autoregression** conditioned on a historical prefix:
$$p(x_t \mid x_1, x_2, \dots, x_{t-1})$$

In this existing paradigm:
- A "prompt" or "instruction" is merely passive history in the context buffer.
- A "reward" (as in RL/RLHF) is a scalar feedback signal received ex-post.
- An "attention weight" $\alpha_{ij} = \text{softmax}(Q_i K_j^T / \sqrt{d})$ reflects statistical associational relevance between tokens, not teleological purpose.
- A "prediction" is a statistical estimate of what will occur next given past data.
- An "action" is an emitted token or tool call.

**Intention is none of these.**

### Formal Definition:
> **Computational Intention** is an internal, active, teleological state construct that specifies a desired future configuration of reality (or internal cognitive state), parameterizes the dynamical operators governing information processing, defines the admissible constraint manifold, allocates bounded computational resources, and continuously measures the discrepancy between intended, predicted, and observed states to drive adaptive execution.

| Concept | What It Is | What It Is NOT | Why Transformers Confuse It |
| :--- | :--- | :--- | :--- |
| **Prompt / Instruction** | Passive input tokens sitting in a static context buffer. | An active teleological governing state. | Treated identically to distractors or irrelevant prefix text. |
| **Prediction** | $\mathbb{E}[S_{t+k} \mid S_t, A_t]$: what the system expects *will* happen. | What the system *intends* to achieve. | Passive autoregression conflates plausible continuation with goal accomplishment. |
| **Reward** | Scalar feedback signal $R(s, a) \in \mathbb{R}$ evaluating past state. | Prospective structured goal manifold. | Lacks intrinsic causal structure, multi-scale decomposition, and self-verification. |
| **Action** | An operator applied to the internal state or external environment. | The governing purpose of that operator. | Emitting tokens does not imply intention preservation. |
| **Attention** | Associational routing of past activations based on query-key similarity. | Purpose-driven prioritization of what *should* be computed. | Attention is reactive and quadratic; intention is proactive and selective. |

---

## 2. Mathematical Formalization of the Intention State

An Intention at time step $t$, denoted $\mathcal{I}_t$, is a structured 8-tuple defined on a differentiable, continuous-symbolic manifold:

$$\mathcal{I}_t = \left( \mathcal{G}_t, \mathcal{C}_t, \mathcal{P}_t, \mathcal{B}_t, \mathcal{U}_t, \mathcal{H}_t, \mathcal{R}_t, \mathcal{V}_t \right)$$

### 2.1 Component Definitions

1. **Goal Manifold ($\mathcal{G}_t \in \mathbb{R}^{d_g}$):**
   The target representation specifying the desired terminal state distribution $p^*(\mathcal{S})$. In continuous latent space, $\mathcal{G}_t$ defines an attractor point or manifold $\mathcal{S}^*$ such that the system's objective is to minimize the geodesic distance $d_{\mathcal{M}}(S_t, \mathcal{S}^*)$.

2. **Constraint Field ($\mathcal{C}_t \in \mathbb{R}^{k \times d_c}$):**
   A system of invariant boundaries and inequalities defining the admissible computational and operational subspace:
   $$\Omega_{admissible} = \{ S \in \mathcal{S} \mid c_i(S) \le 0, \quad \forall i \in \{1, \dots, k\} \}$$
   Unlike soft penalties, the constraint field actively projects state transitions onto the null-space of prohibited directions.

3. **Priority Metric ($\mathcal{P}_t \in \Delta^{m-1}$):**
   A simplex vector $\sum_{j=1}^m p_j = 1$ establishing relative resource allocations and trade-off gradients between competing sub-intentions:
   $$\mathcal{P}_t = \text{Softmax}\left( W_p \mathcal{G}_t + b_p \right)$$

4. **Belief State ($\mathcal{B}_t \in \mathbb{R}^{d_b}$):**
   The epistemic internal representation of external and internal reality at step $t$. It encapsulates the agent's current understanding of the environment and its own latent dynamics.

5. **Uncertainty Field ($\mathcal{U}_t \in \mathbb{S}^{d_b}_{++}$):**
   A positive-definite covariance or entropy tensor characterizing both epistemic uncertainty (lack of knowledge about the environment model) and aleatoric uncertainty (stochastic environmental noise):
   $$\mathcal{U}_t = \mathbb{E}\left[ (\mathcal{B}_t - \bar{\mathcal{B}}_t)(\mathcal{B}_t - \bar{\mathcal{B}}_t)^T \right]$$

6. **Temporal Horizon Geometry ($\mathcal{H}_t \in \mathbb{R}^+$):**
   The teleological time-scale over which the intention is defined. It governs the discounting profile $\gamma(\tau) = \exp(-\tau / \mathcal{H}_t)$, determining whether the computation prioritizes immediate reactive actions or multi-stage deliberative plans.

7. **Computational Resource Tensor ($\mathcal{R}_t = (F_t, M_t, L_t)$):**
   The operational budget allocated for achieving the intention:
   - $F_t \in \mathbb{N}$: Max FLOP budget / internal recurrence depth.
   - $M_t \in \mathbb{N}$: Dynamic working memory capacity.
   - $L_t \in \mathbb{R}^+$: Maximum allowable latency before mandatory action emission.

8. **Verification Operator ($\mathcal{V}_t: \mathcal{S} \times \mathcal{G} \to [0, 1]$):**
   A continuous, falsifiable evaluation operator measuring the degree of intention fulfillment:
   $$\mathcal{V}_t(S, G) = \sigma\left( \langle \psi(S), \phi(G) \rangle - \theta_v \right)$$
   Where $\psi$ and $\phi$ map state and goal to a common verification Hilbert space.

---

## 3. The Intention Lifecycle Algebra

An intention is not a static parameter; it undergoes formal state transitions throughout its operational lifecycle:

```
[FORMATION] ──> [DECOMPOSITION] ──> [PRIORITIZATION] ──> [EXECUTION]
     ▲                                                         │
     │                                                         ▼
[REVISION] <──────── [MONITORING & VERIFICATION] <─────────────┘
     │                          │
     ▼                          ▼
[TERMINATION / REJECTION]  [COMPLETION]
```

### 3.1 Lifecycle Phase Operators

1. **Intention Formation ($\Phi_{form}$):**
   Synthesizes a new intention $\mathcal{I}_{new}$ from incoming stimuli $X_t$, higher-level meta-intentions $\mathcal{I}_{meta}$, and current belief $\mathcal{B}_t$:
   $$\mathcal{I}_t = \Phi_{form}(X_t, \mathcal{B}_t, \mathcal{I}_{meta})$$

2. **Intention Decomposition ($\Phi_{decomp}$):**
   Factorizes a macro-intention $\mathcal{I}_t$ into an acyclic directed graph (DAG) of micro-intentions:
   $$\{\mathcal{I}_t^{(1)}, \mathcal{I}_t^{(2)}, \dots, \mathcal{I}_t^{(n)}\} = \Phi_{decomp}(\mathcal{I}_t)$$
   Subject to compositionality: $\bigoplus_{i=1}^n \mathcal{I}_t^{(i)} \cong \mathcal{I}_t$.

3. **Intention Arbitration & Prioritization ($\Phi_{prior}$):**
   Resolves mutual exclusion and resource contention among competing active intentions:
   $$\mathcal{I}_t^* = \arg\max_{\mathcal{I}^{(i)}} \left( \mathcal{P}_t^{(i)} \cdot \text{Utility}(\mathcal{I}^{(i)}) - \lambda \cdot \text{Cost}(\mathcal{I}^{(i)}, \mathcal{R}_t) \right)$$

4. **Intention-Governed Execution ($\Phi_{exec}$):**
   Modulates the computational state transition operator $F_{\theta}$ such that internal information routing is driven strictly by $\mathcal{I}_t$:
   $$S_{t+1} = F_{\theta}\left(S_t, X_t; \mathcal{I}_t\right)$$

5. **Reality Monitoring & Discrepancy Computation ($\Phi_{monit}$):**
   Quantifies the teleological error tensor $\mathcal{E}_t$:
   $$\mathcal{E}_t = \left\| \mathcal{G}_t - \text{Project}_{\mathcal{G}}(\mathcal{B}_t) \right\|_{\mathcal{M}}$$

6. **Intention Revision ($\Phi_{rev}$):**
   When teleological error exceeds an epistemic tolerance threshold $\tau_{drift}$, the system updates the intention without discarding invariant constraints:
   $$\mathcal{I}_{t+1} = \Phi_{rev}(\mathcal{I}_t, \mathcal{E}_t, \mathcal{U}_t)$$

7. **Completion and Termination ($\Phi_{term}$):**
   When $\mathcal{V}_t(S_t, \mathcal{G}_t) \ge 1 - \epsilon$, the intention triggers memory consolidation and safely terminates, returning resources to $\mathcal{R}$.

---

## 4. Fundamental Axioms of Intention-Centred Computation

- **Axiom 1 (Teleological Primacy):** No neural activation, state transition, or memory update shall occur without an explicit conditioning projection from the active intention $\mathcal{I}_t$.
- **Axiom 2 (Active Reality Verification):** An intention is only well-formed if it possesses a computable verification operator $\mathcal{V}_t$ capable of determining whether the intended outcome has materialized in observed reality.
- **Axiom 3 (Bounded Conservation of Resource):** All cognitive computation is bounded by resource tensor $\mathcal{R}_t$. If an intention cannot be fulfilled within budget, it must trigger revision or graceful degradation, never unbounded looping or uncontrolled hallucination.
- **Axiom 4 (Invariance of Constraints):** Environmental and safety constraints $\mathcal{C}_t$ form a hard projection boundary; no optimization of goal $\mathcal{G}_t$ may violate $\Omega_{admissible}$.
