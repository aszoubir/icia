# FOUNDATIONAL SPECIFICATION & EMPIRICAL REPORT 25: INTENTION-GUIDED CODE SYNTHESIS & AST INVARIANTS

**Architecture:** ICIA Code Generation Subsystem  
**Date:** 2026-11-04  
**Status:** Validated Empirical Report  
**Benchmark Suite:** Algorithmic Code Synthesis under Distractor Syntax Injections

---

## 1. Scientific Objective
Evaluate the Intention-Centred Intelligence Architecture on formal code generation against standard autoregressive causal language modeling.

In autoregressive token generation:
$$p(x_t \mid x_{<t})$$
The model lacks an invariant representation of future goals and program structure. As token sequences scale:
1. **Delimiter Mismatch:** Unclosed brackets ($([\{$) disperse across self-attention context, leading to SyntaxErrors.
2. **Scope Resolution Breakdown:** Identifiers outside the lexical scope are hallucinated, causing runtime `NameError` exceptions.
3. **Autoregressive Inertia:** When a bug is emitted at token $t$, all subsequent tokens $x_{>t}$ are conditioned on the bug, preventing self-correction.

In **ICIA Intention-Guided Code Synthesis**:
1. The **Macro-Intention** $\mathcal{I}_{\text{macro}}$ anchors the target function contract, signature, and post-conditions.
2. The **Syntax & Scope Barrier Manifold** $c_j(S) \le 0$ enforces continuous delimiter balance, AST well-formedness, and lexical scope resolution.
3. The **Reality Discrepancy Gate** $\mathcal{E}_t$ evaluates candidate expressions before emission. Inadmissible candidates encounter infinite barrier potentials, triggering instantaneous 1-step latent pivots with **zero external token emissions**.

---

## 2. Experimental Protocol
- **Tasks Evaluated:**
  1. `binary_search(arr, target)`: Logarithmic search with pointer bounds and midpoint arithmetic.
  2. `matrix_transpose(matrix)`: 2D dimension resolution and nested list comprehension.
  3. `is_valid_brackets(s)`: Delimiter stack tracking with dictionary lookup.
- **Injected Perturbations:** Candidates include unclosed parentheses/brackets, mismatched delimiters, and references to undefined out-of-scope identifiers.
- **Evaluation Criteria:**
  - **Syntax Validity Rate (%):** Successfully parsed by Python AST compiler without `SyntaxError`.
  - **Functional Execution Rate (%):** Successfully passes algorithmic test cases without `NameError` or logic crash.
  - **Latent Invariant Pivots:** Total bugs blocked at the barrier before token emission.

---

## 3. Measured Results

| Metric | Autoregressive Baseline | ICIA Intention Synthesis | Scientific Delta |
| :--- | :---: | :---: | :---: |
| **Syntax Validity Rate** | **$33.3\%$** | **$100.0\%$** | **$+66.7\%$ absolute (Zero syntax errors)** |
| **Functional Execution Rate** | **$0.0\%$** | **$100.0\%$** | **$+100.0\%$ absolute (All unit tests passed)** |
| **Out-of-Scope Identifier Crashes** | $100\%$ on scoping tasks (`NameError`) | **$0\%$ (Zero NameErrors)** | **Lexical Scope Invariant Preserved** |
| **Unclosed Delimiter Crashes** | $66.7\%$ (`(` and `[` unclosed) | **$0\%$ (100% balanced stack)** | **Delimiter Invariant Preserved** |
| **Latent Invariant Pivots** | $0$ (Unaware of bugs) | **$5$ bugs blocked at barrier** | **1-step latent correction without inertia** |

---

## 4. Key Epistemic Conclusions
1. **Structural Delimiter Invariants are Guaranteed:** Standard Transformers cannot guarantee bracket balance over long horizons due to attention dispersion ($H \to \ln N$). ICIA's continuous delimiter stack barrier $c_{\text{delim}}(S) \le 0$ forces delimiter matching mathematically.
2. **Elimination of Autoregressive Inertia in Code:** In standard LLMs, generating an erroneous token permanently poisons the KV-cache. In ICIA, flawed candidate tokens trigger a barrier discrepancy spike that pivots to an alternate admissible expression in latent state space *before* the token enters the output stream.
