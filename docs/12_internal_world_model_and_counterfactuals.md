# FOUNDATIONAL SPECIFICATION 12: THE INTERNAL WORLD MODEL AND LATENT COUNTERFACTUAL PLANNING

**Project Codename:** IIAY (Intention Is All You Need)  
**Architecture:** ICIA (Intention-Centred Intelligence Architecture — Unified Prototype)  
**Status:** Canonical Foundation Document  

---

## 1. The Necessity of an Internal World Model

In next-token autoregressive Transformers, planning is fundamentally **open-loop**: the model emits a string of tokens representing a chain of thought or a sequence of actions. If an action fails or the environment exhibits stochastic slip, the Transformer cannot "un-emit" the tokens; it must re-process the entire conversational prefix from scratch.

In contrast, biological intelligence and control theory demonstrate that true agency requires an **Internal World Model** ($\mathcal{W}_\theta$):
> An agent must possess a differentiable internal forward simulator capable of predicting the consequences of hypothetical actions *in latent space* before committing to execution in physical reality.

```
                  [ LATENT COUNTERFACTUAL SEARCH (IMAGINATION) ]

                          Active Intention I_t
                                   │
                                   ▼
                       Current Latent State S_t
                                   │
              ┌────────────────────┼────────────────────┐
              ▼ (Candidate A_1)    ▼ (Candidate A_2)    ▼ (Candidate A_3)
          [ WORLD MODEL ]      [ WORLD MODEL ]      [ WORLD MODEL ]
          S_hat_{t+1}^(1)      S_hat_{t+1}^(2)      S_hat_{t+1}^(3)
              │                    │                    │
              ▼                    ▼                    ▼
          Score against I_t    Score against I_t    Score against I_t
          Ψ(S_hat; G_t)        Ψ(S_hat; G_t)        Ψ(S_hat; G_t)
              │                    │                    │
              └────────────────────┼────────────────────┘
                                   │
                                   ▼ Select Optimal Action A*
                          [ EXECUTE IN REALITY ]
                                   │
                                   ▼
                         Observe Actual X_{t+1}
                                   │
                                   ▼
                      Reality Discrepancy E_{t+1}
               Update World Model & Re-anchor Intention
```

---

## 2. Mathematical Formalization of the Differentiable World Model

Let the latent cognitive manifold be $\mathcal{S} \subseteq \mathbb{R}^{d_s}$ and the action space be $\mathcal{A} \subseteq \mathbb{R}^{d_a}$. The World Model $\mathcal{W}_\theta$ is formalized as a coupled transition operator:

$$\mathcal{W}_\theta: \mathcal{S} \times \mathcal{A} \times \mathcal{I} \longrightarrow \mathcal{S} \times \mathcal{X} \times \mathbb{R}^+$$

### 2.1 Forward State Transition Equation
$$\hat{S}_{t+1} = S_t + \mathbf{f}_{\text{trans}}(S_t, A_t; \mathcal{I}_t)$$
Where:
$$\mathbf{f}_{\text{trans}}(S_t, A_t; \mathcal{I}_t) = \text{LayerNorm}\left( W_s S_t + W_a A_t + W_i \mathcal{I}_t \right)$$

### 2.2 Expected Sensory Observation Predictor
$$\hat{X}_{t+1} = \mathbf{g}_{\text{obs}}(\hat{S}_{t+1})$$

### 2.3 Counterfactual Evaluation Function
For any candidate action $A \in \mathcal{A}$, its teleological fitness is evaluated by the **Expected Intention Gradient**:
$$J(A; S_t, \mathcal{I}_t) = - \underbrace{\frac{1}{2} \| \Pi_{\mathcal{G}}(\hat{S}_{t+1}) - \mathcal{G}_t \|_2^2}_{\text{Goal Proximity}} - \lambda_c \underbrace{\sum_{j=1}^k \text{ReLU}(c_j(\hat{S}_{t+1}))}_{\text{Constraint Violation Penalty}} - \lambda_u \underbrace{\text{Var}(\hat{S}_{t+1})}_{\text{Uncertainty Penalty}}$$

The optimal computational or external action is selected via:
$$A_t^* = \arg\max_{A \in \mathcal{A}} J(A; S_t, \mathcal{I}_t)$$

---

## 3. Closed-Loop Reality Feedback and Online Adaptation

When action $A_t^*$ is enacted, the environment produces an actual observation $X_{t+1}$. The system continuously computes the **Reality Discrepancy Tensor**:

$$\mathcal{E}_{t+1} = X_{t+1} - \mathbf{g}_{\text{obs}}(\hat{S}_{t+1})$$

### 3.1 Dual Online Update
1. **World Model Calibration:** The prediction error updates the world model weights via online gradient descent:
   $$\theta_{\text{wm}} \longleftarrow \theta_{\text{wm}} - \eta_{\text{wm}} \nabla_{\theta_{\text{wm}}} \|\mathcal{E}_{t+1}\|_2^2$$
2. **State & Plan Re-anchoring:** The cognitive state is updated through a Kalman-like innovation step:
   $$S_{t+1} = \hat{S}_{t+1} + \mathbf{K}_t \mathcal{E}_{t+1}$$
   Where $\mathbf{K}_t = \sigma(W_k [\hat{S}_{t+1}; \mathcal{E}_{t+1}])$ is the dynamic Kalman gain matrix.

This closed loop ensures that the system is **self-correcting**: even under severe environmental noise or stochastic action failure, the agent detects the discrepancy instantly and re-routes its trajectory.
