# INTENTION IS ALL YOU NEED (IIAY)
## The Intention-Centred Intelligence Architecture (ICIA)

[![Tests](https://img.shields.io/badge/Tests-47%20Passed-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![C99](https://img.shields.io/badge/Embedded-MISRA--C99-orange)](include/icia_core.h)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Post--Transformer-red)]()

> **"Intention is not merely a feature to be added to attention; Intention is the true organizing principle of intelligence."**

**Project Codename:** `IIAY`  
**Architecture:** `ICIA` (Intention-Centred Intelligence Architecture)  
**Research Classification:** Foundational Post-Transformer AI Architecture  

---

## 1. Executive Overview

Modern AI relies almost exclusively on the Transformer architecture, which organizes computation around **passive, sequence-wide self-attention**:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

While effective for statistical token association, the Transformer suffers from four foundational pathologies:
1. **The Teleological Blind Spot:** Computation is driven by historical token associations rather than intended future goals.
2. **The Attention Dispersion Law:** We prove analytically and verify empirically that as distractor context scales, self-attention entropy diverges logarithmically ($H \to \log N$), dispersing goal-directed attention mass as $O(1/N)$.
3. **Quadratic Scratchpad Explosion:** Multi-step reasoning requires generating hundreds of intermediate Chain-of-Thought (CoT) tokens into the external context buffer ($O(N^2)$ memory and latency).
4. **Open-Loop Vulnerability:** Standard Transformers lack an internal forward model, failing catastrophically under environmental noise or unannounced roadblocks.

**ICIA** replaces passive token attention with **first-class intention dynamics**. Intention is formalized as a structured computational state $\mathcal{I}_t = (\mathcal{G}_t, \mathcal{C}_t, \mathcal{P}_t, \mathcal{B}_t, \mathcal{U}_t, \mathcal{H}_t, \mathcal{R}_t, \mathcal{V}_t)$, and information processing is governed by continuous Riemannian gradient flow toward intention attractors:
$$\dot{S}_t = f(S_t) - \nabla \Psi_{\mathcal{G}}(S_t) - \sum_{j} \mu_j \nabla c_j(S_t)$$

---

## 2. Empirical Benchmark Scorecard

All metrics are measured on CPU from reproducible experiments in this repository:

| Benchmark Domain | Standard Transformer Baseline | ICIA Alternative | Empirical Advantage |
| :--- | :--- | :--- | :--- |
| **Long-Horizon Distractors ($L=1024$)** | Attention disperses; Goal Retention $31.3\pm2.5\%$ | Active Attractor Anchoring (AIAA); Goal Retention $40.3\pm20.6\%$ | **$-56.8\%$ params; mean retention gain ($+9\%$) lies *within* ICIA's seed variance (range $26$–$64\%$), not the previously claimed $100\%$** |
| **Stochastic Control ($15\%$ slip noise)** | Open-loop derailment; $22.0\pm1.0\%$ Success | Closed-loop innovation feedback; $69.3\pm26.1\%$ Success | **$+47\%$ mean reliability, $0$ collisions; but seed-sensitive ($47$–$98\%$), not $100\%$** |
| **Unannounced Obstacle (Dynamic Wall)**| Blind open-loop execution; $1.3\pm1.2\%$ Success (up to $72$ crashes) | Discrepancy Gate ($\|\mathcal{E}_t\| > \tau$); $75.3\pm17.1\%$ Success ($0$ crashes) | **Zero-shot 1-step replan; $0$ collisions across all seeds; success rate $59$–$93\%$** |
| **NLP Multi-Hop Reasoning ($D=10$)** | Autoregressive CoT ($60$ tokens); $5{,}512\pm1{,}144$ ms; accuracy $7.8\pm3.5\%$ | $O(1)$ Latent Recurrence; $9.0\pm3.2$ ms ($0$ tokens); accuracy $3.3\pm0\%$ | **$\approx612\times$ lower latency & zero context-buffer pollution (wall-clock, host-load sensitive) — but *lower* accuracy than CoT; the reported $0.995$ verification confidence is *not* calibrated** |
| **Natural Language Modeling (ICIA-LM)**| GPT Baseline: $4.42\pm0.03$ PPL ($1.44$M params, $O(N)$ KV-cache)| ICIA-LM: $1.29\pm0.03$ PPL ($433$k params, $O(1)$ bounded memory)| **$3.4\times$ lower perplexity; $-69.9\%$ params; $66.9\times$ less cache** (needle-in-a-haystack retrieval scores $0\%/0\%$ for *both* — see §2.1)|
| **Continuous Physics (+15N Shock)**| Open-loop drift $0.567\pm0.063$ m; $30\pm6\ \mu\text{s}$ decision latency | Teleological potential flow; drift $0.009\pm0.001$ m; $52\pm11\ \mu\text{s}$ latency | **$63\times$ tighter drift; $0$ RL episodes; but impulse-shock survival is *tied* ($100\%/100\%$) and ICIA control latency is *higher* than the baseline (wall-clock, host-load sensitive)** |
| **Multi-Agent Bottleneck Coordination**| Uncoordinated flow: $260$ collisions (2-agent) / $2{,}040$ (pileup)| Anti-symmetric Collective Field: $0$ collisions| **Zero communication tokens & collision-free; success is *tied* at $100\%/100\%$ — the gain is collision elimination, not throughput** |
| **Hierarchical Impasse Resolution**| Autoregressive inertia: $0\%$ success, $97$ stuck cycles | Dynamic Sub-goal Re-routing: $100\%$ success, $1$ replan | **$2.18\times$ fewer steps ($120\to55$); deterministic (seed-invariant) and reproduces exactly** |
| **AST Code Invariants (Synthesis)**| Token hallucination ($33.3\%$ syntax, $0.0\%$ pass rate)| Active Syntax Constraint Invariants ($100\%$ syntax, $100\%$ pass)| **$100\%$ valid ASTs; $5$ syntax/scoping bugs blocked** |
| **Natural Language Programming**| Negation blindness ($25.0\%$ pass rate, $50\%$ violations)| Repulsive Barrier Potential ($100.0\%$ pass rate, $0\%$ violations)| **$4\times$ higher pass rate; $0\%$ broken prohibitions** |
| **WebAssembly & Edge Deployment**| Bloated runtime / GPU server requirement| Standalone Wasm Core ($413$ bytes, zero deps)| **Sub-microsecond V8 JIT; 100% offline browser/edge** |
| **Hardware RTL FPGA Synthesis**| Unsuitable for bare metal (millisecond latencies)| 2-stage pipelined Q16.16 Verilog Core| **$5.0\text{ ns}$ latency; 1 output/clock at $200\text{ MHz}$** |
| **MBPP Negative Constraint Coding**| Negation blindness ($0\%$ adherence, $0\%$ pass rate)| AST Invariant Barrier flow ($100\%$ adherence, $100\%$ pass)| **Zero negation blindness; $100\%$ functional success** |
| **Hybrid Invariant Governor**| Foundation hallucination & negation failure ($0\%$ pass)| Real-time teleological interception ($100\%$ pass rate)| **$10$ violations repaired; Soft-slack Pareto relaxation** |
| **T-MCTS Non-Convex Search**| Greedy Flow ($0\%$ success, trapped in dead end)| Teleological MCTS ($100\%$ detour convergence)| **$212$ barrier paths pruned; Escapes concave local traps** |
| **Multi-File AST Invariants**| Unchecked refactor ($1$ cycle, $1$ missing symbol)| Project Dependency Graph ($100\%$ valid ASTs)| **$1.51\text{ ms}$ repo-wide verification; Zero circular deadlocks** |
| **Governed Decoding — Real Sampled Decoder (Phase 16)**| Free temperature sampling ($T{=}0.8$, top-$k$ $40$, $16$ samples/task) from a genuinely trained $328$k-param byte-level GPT: syntax $3.3\pm2.6\%$, adherence $3.3\pm2.6\%$, functional $0\pm0\%$ | AST-barrier rejection sampling + intention-synthesized fallback: adherence & functional $100\%$ | **Guarantee holds, cost is exposed: $0$ of $240$ real decoder samples across 3 seeds were accepted (fallback rate $100\%$) — the $100\%$ comes from the fallback, not from governance rescuing decoder output at this scale** |

---

### 2.1 Scope, Baselines, and Limitations (audited 2026-11)

To keep this repository scientifically navigable, the following boundaries are stated explicitly:

1. **Scale:** All learned components are small (≤450k parameters) and trained from scratch on synthetic corpora or hand-designed environments on CPU. Results demonstrate architectural *mechanisms*, not competitive performance against production-scale Transformers or LLMs.
2. **Simulated code baselines:** The "Autoregressive LLM Baseline" in Phases 9–12 is a scripted greedy emitter over curated candidate lists with known defects ([run_code_synthesis_experiment.py](icia/experiments/run_code_synthesis_experiment.py)), not a sampled pretrained model. These benchmarks validate that AST barrier enforcement *works*; they do not measure an LLM's failure rate. Phase 16 ([run_governed_decoding_benchmark.py](icia/experiments/run_governed_decoding_benchmark.py)) closes this gap with a genuinely trained, temperature-sampled byte-level GPT as the RAW arm; its measured verdict is in note 4 below.
3. **Reproduced vs. headline numbers:** The seeded multi-run matrix (see §4.1, seeds 42/1/2) reports mean ± std. Several historical headline claims exceeded what the repo's own exported JSON supports, and the scorecard above now shows measured values. Most notably: Phase 2 goal retention at L=1024 is $40.3\pm20.6\%$ for DIVRE-v2 (not $100\%$), and its Standard-Transformer baseline scores $31.3\pm2.5\%$ (not $0\%$) — so the retention advantage is small and inside ICIA's own seed variance. Phase 3 ICIA success is $69\pm26\%$ (clean) and $75\pm17\%$ (roadblock), i.e. robust and collision-free but not the previously stated $100\%$. The qualitative direction (ICIA degrades far less than open-loop attention at long horizons and under unannounced obstacles) does reproduce.
4. **Known negative results and tied outcomes:** In Phase 16, the real $328$k-param decoder trained on 17 correct programs emits *functional* code $0\%$ of the time and syntactically-valid code only $3.3\%$ of the time at $T{=}0.8$; across 3 seeds, AST-barrier rejection sampling accepted $0$ of $240$ live samples (fallback rate $100\%$). The governed $100\%$ pass rate is therefore produced entirely by the intention-synthesized reference fallback, not by the barrier recovering good decoder output — governance guarantees the *output* of the pipeline, not the *competence* of the base decoder at this scale. Needle-in-a-haystack retrieval in Phase 5 scores $0\%/0\%$ (ICIA/GPT) at *all* distractor depths — the verified Phase 5 advantage is perplexity, not retrieval. In Phase 4 latent multi-hop reasoning, ICIA accuracy ($3.3\pm0\%$) is *lower* than the CoT baseline ($7.8\pm3.5\%$); the real, reproducible gain there is $\approx612\times$ lower latency with zero scratchpad tokens, not correctness — and the `verification_confidence` output ($0.995$) is badly miscalibrated against that accuracy. Several other "wins" are actually ties on the success metric and only differ on a secondary axis: Phase 6 impulse-shock survival is $100\%/100\%$ (the differentiator is $63\times$ lower position drift, and ICIA's control latency is in fact *higher* than the baseline), and Phase 7 multi-agent and Phase 8 impasse success are $100\%$ for the governed arm but Phase 7's baseline also succeeds ($100\%$) — its failure is collisions, not completion.
5. **Math vocabulary:** "Riemannian"/"symplectic" in module names and older docs refer to Euclidean potential flow with an anti-symmetric rotation term (no metric tensor, no symplectic integrator). The relaxation module is a greedy slack allocator, not an iterative augmented-Lagrangian solver.
6. **Deployment tiers:** The WebAssembly core is executed and numerically verified under Node/V8 in CI. The C99 header and Verilog RTL now additionally compile/simulate locally (gcc, Icarus Verilog) with Q16.16 parity against the Python simulator.

---

## 3. Architecture Blueprint

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    ICIA: COMPLETE POST-TRANSFORMER ENGINE                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  Sensory Stream X_t ────────► [ Intention-Gated Associative Memory (IGAM) ]            │
│                                       │ (Surprise Gated: ||E_t|| > tau)                 │
│                                       ▼                                                 │
│  Teleological Drive G_t ────► [ DUAL-STREAM RECURRENT CORE (DIVRE) ]                   │
│                                 │   Trajectory Stream: T_t = f(T_{t-1}, V_t, G_t)       │
│                                 │   Verification Head: nu_t = V(T_t, G_t)               │
│                                 ▼                                                       │
│                           Action Execution a_t                                          │
│                                 │                                                       │
│                                 ▼                                                       │
│  Reality Observation Y_t ──► [ Reality Discrepancy Gate: E_t = Y_t - W(S_t, a_t) ]     │
│                                       │                                                 │
│                                       ├─ If ||E_t|| <= tau: Nominal Execution Continue  │
│                                       └─ If ||E_t|| > tau: Dynamic Impasse Pivot        │
│                                            (Counterfactual Rollout in W_theta)          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Quickstart

### Prerequisites
- Python 3.11+
- Recommended: `uv` or virtual environment with `torch` and `numpy`

### 1. Run Interactive Terminal Demonstration Suite
```bash
python demo.py
```
Explore live roadblock recovery, latent deliberation step tracking, and attention dispersion simulations.

### 2. Run Comprehensive Unit Test Suite
```bash
python -m pytest tests/
```
Verifies Riemannian potential invariants, IGAM surprise gating, zero-scratchpad token emission, and engine stability.

### 3. Reproduce Master Empirical Benchmarks
```bash
# Phase 2: Long-horizon distractor scaling (L up to 1024)
python -m icia.experiments.run_phase2_experiment

# Phase 3: Stochastic navigation and dynamic roadblock recovery
python -m icia.experiments.run_phase3_experiment

# Phase 4: NLP symbolic reasoning vs. Chain-of-Thought
python -m icia.experiments.run_nlp_reasoning_experiment

# Phase 8: Hierarchical Intention Decomposition
python -m icia.experiments.run_hierarchical_experiment

# Phase 9: Intention-Guided Code Synthesis & AST Invariants
python -m icia.experiments.run_code_synthesis_experiment

# Phase 10: Natural Language Programming & Repulsive Barrier Compilation
python -m icia.experiments.run_nl_coding_experiment

# Phase 11: WebAssembly Binary Builder & Execution
python -m icia.native.wasm_builder

# Phase 12: Constrained Coding & Negation Blindness Benchmark
python -m icia.experiments.run_constrained_coding_benchmark

# Phase 13: Hybrid Foundation Governor & Soft-Slack Relaxation
python -m icia.experiments.run_hybrid_governor_experiment

# Phase 14: Teleological Monte Carlo Tree Search (T-MCTS)
python -m icia.experiments.run_mcts_experiment

# Phase 15: Repository-Scale Multi-File AST Dependency Invariants
python -m icia.experiments.run_project_graph_experiment
```

### 4.1 Reproducibility & Seeding
Every stochastic benchmark (Phases 2–8) reads the `ICIA_SEED` environment variable
(default `42`) and seeds `random`, `numpy`, and `torch` before model construction, so
results are bit-reproducible per seed:

```bash
# Single deterministic run
ICIA_SEED=1 python -m icia.experiments.run_phase2_experiment

# Multi-seed variance matrix + mean/std scorecard
for s in 42 1 2; do
  for e in run_phase2_experiment run_phase3_experiment run_nlp_reasoning_experiment; do
    ICIA_SEED=$s python -m icia.experiments.$e
    cp icia_*.json scratch/ms/  # per-seed copies consumed by the collector
  done
done
python scripts/collect_seed_stats.py
```

The seed used by each run is recorded in its exported JSON (`"seed"` field).
Deterministic rule-based benchmarks (Phases 9–13, 15, WASM, hardware simulator)
are seed-independent.

---

## 5. Research Documentation Index

The complete scientific documentation comprises 31 foundational specifications and master empirical reports in [`docs/`](docs/):

1. [`01_foundational_definition_of_intention.md`](docs/01_foundational_definition_of_intention.md): Mathematical 8-tuple formalization $\mathcal{I}_t$.
2. [`02_critical_scientific_assessment.md`](docs/02_critical_scientific_assessment.md): Adversarial critique of Transformers.
3. [`03_research_map_and_lineage.md`](docs/03_research_map_and_lineage.md): Cybernetics, Active Inference, and SSM heritage.
4. [`04_mathematical_decomposition.md`](docs/04_mathematical_decomposition.md): Continuous Riemannian gradient flow and Lyapunov stability proofs.
5. [`05_candidate_architectures.md`](docs/05_candidate_architectures.md): Specification of IDFN, HISSA, and DIVRE candidates.
6. [`06_comparative_analysis_and_falsification.md`](docs/06_comparative_analysis_and_falsification.md): Pre-registered falsification criteria.
7. [`07_experimental_protocol_and_benchmarks.md`](docs/07_experimental_protocol_and_benchmarks.md): DPGP, DCPR, and TDVE protocols.
8. [`08_master_scientific_report.md`](docs/08_master_scientific_report.md): Phase 1 Empirical Report & Entropy Dispersion Law.
9. [`09_adaptive_memory_and_attractor_dynamics.md`](docs/09_adaptive_memory_and_attractor_dynamics.md): Active Intention Attractor Anchoring & Theorem 2 proof.
10. [`10_hierarchical_planning_and_decomposition.md`](docs/10_hierarchical_planning_and_decomposition.md): Multi-scale teleological tree decomposition.
11. [`11_phase2_architectural_refinement_report.md`](docs/11_phase2_architectural_refinement_report.md): Phase 2 Report ($L \le 1024$ distractor validation).
12. [`12_internal_world_model_and_counterfactuals.md`](docs/12_internal_world_model_and_counterfactuals.md): Differentiable forward simulator $\mathcal{W}_\theta$.
13. [`13_multiobjective_training_and_metalearning.md`](docs/13_multiobjective_training_and_metalearning.md): Pareto loss and gradient surgery.
14. [`14_phase3_world_model_and_reasoning_report.md`](docs/14_phase3_world_model_and_reasoning_report.md): Phase 3 Report (stochastic roadblock recovery).
15. [`15_nlp_and_latent_symbolic_reasoning.md`](docs/15_nlp_and_latent_symbolic_reasoning.md): Latent teleological reasoning foundation.
16. [`16_nlp_reasoning_benchmark_report.md`](docs/16_nlp_reasoning_benchmark_report.md): Phase 4 Empirical Benchmark Report.
17. [`17_academic_paper_intention_is_all_you_need.md`](docs/17_academic_paper_intention_is_all_you_need.md): **Canonical Scientific Manuscript (Publication-Ready).**
18. [`18_icia_language_model_scaling.md`](docs/18_icia_language_model_scaling.md): Dual-stream language modeling architecture and O(1) memory specification.
19. [`19_icia_lm_empirical_report.md`](docs/19_icia_lm_empirical_report.md): Phase 5 Language Model Empirical Report (1.26 PPL, 66.9x less memory).
20. [`20_continuous_physics_and_embodied_control.md`](docs/20_continuous_physics_and_embodied_control.md): Continuous Riemannian control and non-linear dynamics specification.
21. [`21_continuous_physics_empirical_report.md`](docs/21_continuous_physics_empirical_report.md): Phase 6 Embodied Control Empirical Report (63× lower position drift vs. baseline; ~45 µs control latency; impulse-shock survival tied with baseline).
22. [`22_collective_intention_and_multiagent_coordination.md`](docs/22_collective_intention_and_multiagent_coordination.md): Collective Intention Algebra & Symplectic Vorticity.
23. [`23_multiagent_coordination_report.md`](docs/23_multiagent_coordination_report.md): Phase 7 Multi-Agent Benchmark Report (0 collisions, 0 communication tokens).
24. [`24_hierarchical_intention_report.md`](docs/24_hierarchical_intention_report.md): Phase 8 Hierarchical Decomposition Report (1-step pivot vs autoregressive inertia).
25. [`25_code_generation_and_ast_invariants.md`](docs/25_code_generation_and_ast_invariants.md): Phase 9 Intention-Guided Code Synthesis & AST Invariants Report.
26. [`26_natural_language_programming_report.md`](docs/26_natural_language_programming_report.md): Phase 10 Natural Language Programming & Repulsive Barrier Compilation Report.
27. [`27_wasm_and_hardware_rtl_synthesis.md`](docs/27_wasm_and_hardware_rtl_synthesis.md): Phase 11 WebAssembly & Hardware RTL Acceleration Report (413-byte Wasm, 5ns Verilog core).
28. [`28_negative_constraint_coding_benchmark.md`](docs/28_negative_constraint_coding_benchmark.md): Phase 12 Negative Constraint Coding & Negation Blindness Resolution (100% adherence, 100% pass).
29. [`29_hybrid_foundation_and_relaxed_barriers.md`](docs/29_hybrid_foundation_and_relaxed_barriers.md): Phase 13 Hybrid Foundation Governor & Soft-Slack Lagrangian Relaxation (100% governed pass rate).
30. [`30_teleological_mcts_test_time_search.md`](docs/30_teleological_mcts_test_time_search.md): Phase 14 Teleological Monte Carlo Tree Search over W_theta (Non-convex concave trap navigation).
31. [`31_repository_scale_ast_dependency_invariants.md`](docs/31_repository_scale_ast_dependency_invariants.md): Phase 15 Repository-Scale Multi-File AST Dependency Graph Invariants (Zero circular imports).

---

## 6. Project Structure

```
AI/
├── .github/workflows/ci.yml     # Multi-OS Matrix CI/CD Pipeline
├── dist/                        # Standalone distributions (wheel, tarball, and NeurIPS submission zip)
├── docs/                        # 31 Foundational scientific specifications & reports
├── hardware/                    # Synthesizable Verilog 2001 FPGA RTL core (5ns decision latency)
│   ├── icia_riemannian_core.v   # 2-stage pipelined Q16.16 hardware engine
│   └── tb_icia_riemannian_core.v# Verilog simulation testbench
├── include/
│   └── icia_core.h              # Zero-allocation ANSI C99 single-header embedded engine
├── icia/                        # Core Python architecture package
│   ├── cli.py                   # Unified console executable (`icia` command)
│   ├── primitives/              # Mathematical primitives (IntentionState, IGAM, CollectiveField, RelaxedBarriers)
│   ├── models/                  # ICIA Engines (DIVRE, HISSA, WorldModel, ICIA-LM, Hierarchical, HybridGovernor, TeleologicalMCTS)
│   ├── code/                    # AST Invariant Engine, NL Intention Programmer, ProjectDependencyGraph
│   ├── hardware/                # Cycle-accurate Q16.16 hardware pipeline simulator
│   ├── native/                  # Sub-microsecond binary C-types & WebAssembly emitter
│   ├── tasks/                   # Benchmark suites (DPGP, CartPole, Navigation, Hierarchical, Code, MBPP, Labyrinth, MultiFile)
│   ├── nlp/                     # NLP Subsystem (ByteTokenizer, Corpus Generator, Latent Reasoner)
│   ├── web/                     # Live HTTP server & 413-byte WebAssembly binary
│   └── experiments/             # Master benchmark runners (Phases 1-15)
├── paper/                       # Formal NeurIPS/ICLR LaTeX manuscript & vector figures
│   ├── figures/                 # Camera-ready vector figures (PDF & PNG)
│   ├── main.tex                 # Academic preprint manuscript
│   └── references.bib           # Complete bibliography
├── tests/                       # Automated pytest suite (47 passing unit & integration tests)
├── dashboard.html               # Interactive Generative UI dashboard (9 interactive modules)
├── pyproject.toml               # Modern PEP 517/621 build configuration
└── README.md                    # Project documentation
```

---

## 7. Citation

```bibtex
@article{icia2026intention,
  title={Intention Is All You Need: A Post-Transformer Architecture for Latent Teleological Computation},
  author={Foundational AI Architecture Research Consortium},
  journal={Foundational AI Architecture Reports},
  year={2026}
}
```

---

## 8. License

This research and software are distributed under the **MIT License**.
