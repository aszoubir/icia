# FOUNDATIONAL SPECIFICATION 05: THREE CANDIDATE COMPUTATIONAL ARCHITECTURES

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture)  
**Status:** Canonical Foundation Document  

---

# CANDIDATE A: IDFN — Intention-Driven Dynamical Field Network

```
                                [ CANDIDATE A: IDFN ]

       Sensory Input X(t) ───────┐
                                 ▼
   Intention I_t ───> Energy Landscape E(S; I_t) ───> Gradient Field -∇_S E(S; I_t)
                                                                 │
                                                                 ▼
                                                  Adaptive Numerical Integrator
                                                  dS/dt = -∇_S E + W_x X(t)
                                                                 │
                                                                 ▼
                                                    Converged Attractor S*
                                                                 │
                                                                 ▼
                                                          Verified Action A_t
```

### 1. Mathematical Definition
The cognitive state $S(t) \in \mathbb{R}^d$ evolves on a continuous Riemannian manifold governed by an energy potential function $E(S; \mathcal{I})$ parameterized directly by the intention $\mathcal{I}_t$:
$$E(S; \mathcal{I}) = \frac{1}{2} (S - \mathcal{G})^T Q(\mathcal{I}) (S - \mathcal{G}) + \sum_{k=1}^K \phi_k(c_k(S)) - \log \sum_{m} \exp\left( \frac{\langle S, M_m \rangle}{\tau} \right)$$
where $Q(\mathcal{I}) \succ 0$ is a positive-definite metric tensor shaping the curvature of the goal attractor, and $\phi_k(z) = \frac{1}{\beta} \log(1 + e^{\beta z})$ is a soft-barrier constraint penalty.

### 2. Core Update Equations (Continuous & Discretized)
$$\frac{dS(t)}{dt} = -\nabla_S E(S(t); \mathcal{I}_t) + \mathbf{W}_{\text{stimulus}} X(t) - \gamma S(t)$$
Using adaptive Runge-Kutta 4th-order (RK4) integration with step size $\delta$:
$$k_1 = f(S_n, X_n; \mathcal{I}_n), \quad k_2 = f(S_n + \frac{\delta}{2} k_1, \dots), \quad S_{n+1} = S_n + \frac{\delta}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

### 3. Neural Computation and Memory Mechanism
- **Neural Computation:** Computation is the physical relaxation of the state into the lowest energy basin defined by the active intention.
- **Memory Mechanism:** Modern Continuous Hopfield energy landscape where stored memories form local minima; the active intention tilts the energy landscape so that only teleologically congruent memories are accessible.

### 4. Learning Algorithm and Training Objectives
$$\mathcal{L}_{\text{IDFN}}(\theta) = \mathbb{E}\left[ \| S_{T_{\text{relax}}} - \mathcal{G}^* \|^2 + \lambda_1 \int_0^T \|\dot{S}(t)\|^2 dt + \lambda_2 \sum_k \max(0, c_k(S_T)) \right]$$
Optimized using the Adjoint State Method (Neural ODE backpropagation), computing gradients through continuous time with $O(1)$ activation memory cost.

### 5. Complexity, Advantages, and Limitations
- **Complexity:** Time: $O(N_{\text{steps}} \cdot d^2)$ per observation; Memory: $O(d)$ (constant with sequence length).
- **Advantages:** Guaranteed Lyapunov convergence; native continuous-time reasoning; zero prompt token degradation over infinite time.
- **Limitations:** Stiff differential equations can cause integration slowdowns; computationally demanding numerical integration on standard GPU SIMD hardware.
- **Falsifiable Prediction 1:** IDFN will demonstrate zero degradation in goal satisfaction accuracy as distractor sequence length increases from $10^2$ to $10^6$ tokens.

---

# CANDIDATE B: HISSA — Hierarchical Intention State-Space Architecture

```
                                [ CANDIDATE B: HISSA ]

   Macro-Intention (Slow: L2) ──────────> Generates A^(1)(I_L2), Δ^(1)(I_L2)
             │                                       │
             ▼                                       ▼
   Sub-Intention (Med: L1)   ──────────> Generates A^(0)(I_L1), Δ^(0)(I_L1)
             │                                       │
             ▼                                       ▼
   Micro-Step   (Fast: L0)   ──X_t──> h_t = A(I) h_{t-1} + B(I) X_t ──> Y_t
```

### 1. Mathematical Definition
HISSA generalizes selective continuous-time state-space models ($h'(t) = A h(t) + B x(t), y(t) = C h(t)$) by structuring the state transition operators into a hierarchical tree where higher levels represent invariant intentions and lower levels execute high-frequency state updates.

### 2. Core Update Equations
For layer $l \in \{0, 1, 2\}$ with timescale decimation factors $\tau_0 < \tau_1 < \tau_2$:
$$\Delta^{(l)}_t = \text{softplus}\left( \mathbf{W}_\Delta^{(l)} \mathcal{I}^{(l+1)}_t + \mathbf{U}_\Delta^{(l)} x^{(l)}_t \right)$$
$$\bar{\mathbf{A}}^{(l)}_t = \exp\left( -\Delta^{(l)}_t \cdot \text{diag}\left( \exp(\mathbf{w}_A^{(l)} \odot \mathcal{I}^{(l+1)}_t) \right) \right)$$
$$\bar{\mathbf{B}}^{(l)}_t = \Delta^{(l)}_t \cdot \left( \mathbf{W}_B^{(l)} \mathcal{I}^{(l+1)}_t \right)$$
$$h^{(l)}_t = \bar{\mathbf{A}}^{(l)}_t h^{(l)}_{t-1} + \bar{\mathbf{B}}^{(l)}_t x^{(l)}_t$$
$$y^{(l)}_t = \mathbf{C}^{(l)}(\mathcal{I}^{(l+1)}_t) h^{(l)}_t + \mathbf{D}^{(l)} x^{(l)}_t$$

### 3. Neural Computation and Memory Mechanism
- **Neural Computation:** State selection is spectral: when intention $\mathcal{I}$ changes, the eigenvalues of $\bar{\mathbf{A}}$ rotate, instantly rendering the model blind to frequencies/patterns orthogonal to the goal while amplifying goal-relevant resonant frequencies.
- **Memory Mechanism:** HiPPO orthogonal polynomial state representation compressed into fixed-size hidden vectors $h_t \in \mathbb{R}^{d_h}$, where intention-gating prevents overwriting critical historical goal states.

### 4. Learning Algorithm and Training Objectives
Hardware-aligned parallel associative scan during training; constant-memory linear recurrent rollout during inference.
$$\mathcal{L}_{\text{HISSA}}(\theta) = \sum_{t=1}^T \mathcal{L}_{\text{task}}(y_t, y^*_t) + \lambda \sum_{l=1}^2 \| \mathcal{I}^{(l)}_t - \text{Downsample}(\mathcal{I}^{(l)}_{t-1}) \|^2$$

### 5. Complexity, Advantages, and Limitations
- **Complexity:** Training: $O(L \cdot d)$ parallel associative scan; Inference: $O(1)$ memory, $O(d)$ FLOPs per step.
- **Advantages:** Blazing fast GPU throughput; strictly linear sequence scaling; mathematically rigorous multi-timescale hierarchy.
- **Limitations:** Less intuitive spatial/topological reasoning compared to field models; linear recurrence can struggle with non-linear symbolic binding without high channel expansion.
- **Falsifiable Prediction 2:** HISSA will achieve $>5\times$ faster inference latency than FlashAttention-2 Transformers while preserving $100\%$ goal retention across 64k distractor tokens.

---

# CANDIDATE C: DIVRE — Dual-Stream Intention-Verification Recurrent Engine

```
                                [ CANDIDATE C: DIVRE ]

         TOP-DOWN STREAM (Teleology / Intention)
         T_t = Ψ_top(T_{t-1}, G_t, C_t) ─────────────┐
                         │                           ▼
                         │               [ SYMPLECTIC INTERACTION ]
                         │               Δ_t = T_t ⊗ V_t - Reality Discrepancy
                         ▼                           ▲
         BOTTOM-UP STREAM (Observation / Reality)    │
         V_t = Ψ_bot(V_{t-1}, X_t, Anomaly_t) ───────┘
                         │
                         ▼
             Verified Output Emission A_t
```

### 1. Mathematical Definition
DIVRE is founded on a closed-loop active-inference architecture consisting of two symmetrically coupled recurrent streams:
1. **The Teleological Stream ($T_t \in \mathbb{R}^{d_t}$):** Propagates top-down goal expectations, invariant constraints, and counterfactual plan projections forward in time.
2. **The Verification Stream ($V_t \in \mathbb{R}^{d_v}$):** Propagates bottom-up sensory evidence, prediction errors, and environmental anomalies.

The two streams interact via a non-linear symplectic cross-coupling operator that forces the state trajectory to minimize the symplectic divergence between what was intended ($T_t$) and what occurred ($V_t$).

### 2. Core Update Equations
1. Top-down teleological prediction:
   $$\hat{V}_t = \mathbf{W}_{\text{pred}} T_{t-1}$$
2. Reality discrepancy tensor:
   $$\mathcal{E}_t = X_t - \hat{V}_t$$
3. Bottom-up verification update:
   $$V_t = \text{GELU}\left( \mathbf{W}_v [V_{t-1}; X_t; \mathcal{E}_t] \right)$$
4. Symplectic coupling & Teleological state correction:
   $$T_t = T_{t-1} + \sigma\left( \mathbf{W}_{\text{gate}} \mathcal{E}_t \right) \odot \tanh\left( \mathbf{W}_t [T_{t-1}; \mathcal{G}_t; V_t] \right) - \nabla_T \mathcal{C}(T_{t-1})$$
5. Final verified action selection:
   $$A_t = \text{Softmax}\left( \mathbf{W}_a [T_t \odot V_t] \right)$$

### 3. Neural Computation and Memory Mechanism
- **Neural Computation:** Computes explicitly on the **discrepancy manifold** $\mathcal{E}_t$. If reality matches intention ($\mathcal{E}_t \to 0$), computational effort collapses to an efficient $O(1)$ identity flow. If discrepancy spikes, computation automatically branches into high-capacity corrective recurrence.
- **Memory Mechanism:** Dual-partition key-value memory where key-value pairs are indexed by intention tags and signed by verification status (success vs. failure trajectory).

### 4. Learning Algorithm and Training Objectives
$$\mathcal{L}_{\text{DIVRE}}(\theta) = \underbrace{\mathbb{E}[ \| X_t - \hat{V}_t \|^2 ]}_{\text{Sensory Predictive Accuracy}} + \alpha \underbrace{\mathbb{E}[ \| \Pi_{\mathcal{G}}(T_t) - \mathcal{G}^* \|^2 ]}_{\text{Goal Attainment}} + \beta \underbrace{\mathbb{E}[ \text{Tr}(\mathcal{E}_t^T \mathcal{E}_t) ]}_{\text{Discrepancy Minimization}}$$

### 5. Complexity, Advantages, and Limitations
- **Complexity:** $O(L \cdot d^2)$ per step; constant inference memory $O(d)$.
- **Advantages:** Directly exposes the reality gap $\mathcal{E}_t$ at every step; impossible to hallucinate without triggering an explicit discrepancy spike; native error recovery.
- **Limitations:** Slightly higher parameter count due to the dual streams; requires balancing the learning rates of the two coupled dynamical streams.
- **Falsifiable Prediction 3:** When presented with unexpected environmental rule shifts mid-execution, DIVRE will detect the failure within 1 step ($\mathcal{E}_t > 3\sigma$) and replan successfully, whereas Transformers will continue emitting invalid tokens along the obsolete path.
