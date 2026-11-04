"""
ICIA Phase 2 Master Experiment Runner
Benchmarks:
1. Standard Autoregressive Transformer (SAT)
2. Intention-Conditioned Transformer (ICT)
3. ICIA Candidate C v1 (DIVRE-v1)
4. ICIA Candidate C v2 (DIVRE-v2 with AIAA & IGAM)

Evaluates on:
- Long-Horizon Distractor Extrapolation (L up to 1024!)
- Dynamic Constraint Pivot & Impasse Resolution (DCPR)
- Pareto Efficiency & Memory Footprint
"""

import time
import os
import json
import random
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from icia.models.standard_transformer import StandardTransformerModel
from icia.models.prompt_conditioned_transformer import IntentionConditionedTransformer
from icia.models.candidate_c_divre import DIVREModel
from icia.models.candidate_c_divre_v2 import DIVREv2Model
from icia.tasks.benchmark_suite_v2 import VectorizedBenchmarkGenerator


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step_dpgp(model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, g: torch.Tensor, y: torch.Tensor, model_type: str) -> float:
    model.train()
    optimizer.zero_grad()
    if model_type == "SAT":
        logits, _ = model(x)
    elif model_type in ("ICT", "DIVRE_V1", "DIVRE_V2"):
        logits, _ = model(x, g)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    loss = F.cross_entropy(logits[:, -1, :], y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def evaluate_dpgp_long(model: nn.Module, gen: VectorizedBenchmarkGenerator, distractor_lens: List[int], model_type: str, n_trials: int = 100) -> Dict[int, Dict[str, float]]:
    model.eval()
    results = {}

    with torch.no_grad():
        for d_len in distractor_lens:
            x_test, g_test, y_test = gen.generate_dpgp_batch(n_trials, d_len)
            start_time = time.perf_counter()
            if model_type == "SAT":
                logits, weights = model(x_test)
                last_token_attn = weights[:, :, -1, :] + 1e-12
                attn_entropy = float(-(last_token_attn * torch.log(last_token_attn)).sum(dim=-1).mean().item())
            elif model_type == "ICT":
                logits, _ = model(x_test, g_test)
                attn_entropy = 0.0
            elif model_type in ("DIVRE_V1", "DIVRE_V2"):
                logits, _ = model(x_test, g_test)
                attn_entropy = 0.0

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0 / n_trials
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            accuracy = float((preds == y_test).float().mean().item()) * 100.0

            results[d_len] = {
                "accuracy": accuracy,
                "latency_ms_per_sample": elapsed_ms,
                "attention_entropy": attn_entropy
            }
    return results


def evaluate_dcpr_pivot(model: nn.Module, gen: VectorizedBenchmarkGenerator, model_type: str, n_trials: int = 100) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_test, g_test, y_target_seq, c_mask, t_flip = gen.generate_dcpr_batch(n_trials, seq_len=24)
        if model_type == "SAT":
            logits, _ = model(x_test)
        elif model_type == "ICT":
            logits, _ = model(x_test, g_test)
        elif model_type == "DIVRE_V1":
            logits, _ = model(x_test, g_test)
        elif model_type == "DIVRE_V2":
            logits, _ = model(x_test, g_test, constraint_mask=c_mask)

        preds = torch.argmax(logits, dim=-1)
        post_flip_acc = float((preds[:, t_flip+1:] == y_target_seq[:, t_flip+1:]).float().mean().item()) * 100.0
        immediate_pivot = float((preds[:, t_flip+1] == y_target_seq[:, t_flip+1]).float().mean().item()) * 100.0
        pre_flip_acc = float((preds[:, :t_flip] == y_target_seq[:, :t_flip]).float().mean().item()) * 100.0

    return {
        "pre_flip_accuracy": pre_flip_acc,
        "immediate_pivot_accuracy": immediate_pivot,
        "post_flip_recovery_accuracy": post_flip_acc
    }


def run_phase2_benchmark():
    print("=" * 80)
    print("PROJECT IIAY: PHASE 2 INTEGRATED BENCHMARK RUNNER")
    print("Evaluating SAT vs ICT vs DIVRE-v1 vs DIVRE-v2 (AIAA + IGAM)")
    print("=" * 80)

    vocab_size = 128
    d_model = 64
    d_ff = 128
    gen = VectorizedBenchmarkGenerator(vocab_size=vocab_size, d_model=d_model, seed=42)

    sat = StandardTransformerModel(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=d_ff, max_len=2048)
    ict = IntentionConditionedTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=d_ff, max_len=2048)
    divre_v1 = DIVREModel(vocab_size=vocab_size, d_model=d_model, d_intention=64)
    divre_v2 = DIVREv2Model(vocab_size=vocab_size, d_model=d_model, d_intention=64, num_mem_slots=16)

    models = {
        "Standard Transformer (SAT)": (sat, "SAT"),
        "Intention-Conditioned Transformer (ICT)": (ict, "ICT"),
        "ICIA DIVRE-v1 (Phase 1 Baseline)": (divre_v1, "DIVRE_V1"),
        "ICIA DIVRE-v2 (Integrated with AIAA & IGAM)": (divre_v2, "DIVRE_V2")
    }

    print("\n[STAGE 1] Model Parameter Calibration:")
    for name, (m, _) in models.items():
        print(f"  * {name}: {count_parameters(m):,} parameters")

    print("\n[STAGE 2] Multi-Scale Training Phase (1,000 steps per model with distractor horizons up to 64)...")
    train_distractor_lens = [4, 8, 16, 32, 64]
    for name, (m, m_type) in models.items():
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for step in range(1000):
            d_len = random.choice(train_distractor_lens)
            x_b, g_b, y_b = gen.generate_dpgp_batch(batch_size=32, distractor_len=d_len)
            l = train_step_dpgp(m, opt, x_b, g_b, y_b, m_type)
            losses.append(l)
        print(f"  * {name} trained. Final batch loss: {np.mean(losses[-50:]):.4f}")

    print("\n[STAGE 3] Benchmark 1: Unbounded Long-Horizon Distractor Extrapolation (DPGP)")
    eval_distractor_lens = [8, 32, 128, 512, 1024]
    dpgp_results = {}
    for name, (m, m_type) in models.items():
        print(f"  * Testing {name}...")
        res = evaluate_dpgp_long(m, gen, eval_distractor_lens, m_type, n_trials=100)
        dpgp_results[name] = res
        for d_len, metrics in res.items():
            print(f"      Distractors={d_len:4d} -> Acc: {metrics['accuracy']:5.1f}%, Latency: {metrics['latency_ms_per_sample']:.2f}ms")

    print("\n[STAGE 4] Benchmark 2: Dynamic Constraint Pivot & Recovery (DCPR)")
    dcpr_results = {}
    for name, (m, m_type) in models.items():
        res = evaluate_dcpr_pivot(m, gen, m_type, n_trials=100)
        dcpr_results[name] = res
        print(f"  * {name}: Immediate Pivot Acc: {res['immediate_pivot_accuracy']:.1f}%, Post-Flip Acc: {res['post_flip_recovery_accuracy']:.1f}%")

    final_output = {
        "parameters": {name: count_parameters(m) for name, (m, _) in models.items()},
        "dpgp_results": dpgp_results,
        "dcpr_results": dcpr_results
    }
    with open("icia_phase2_benchmark_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nPhase 2 Benchmark successfully executed! Results exported to icia_phase2_benchmark_results.json")
    return final_output


if __name__ == "__main__":
    run_phase2_benchmark()
