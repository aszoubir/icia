"""
ICIA Minimal Experiment Execution Runner
Runs standardized, parameter-matched benchmarks comparing:
1. Standard Autoregressive Transformer (SAT)
2. Intention-Conditioned Transformer (ICT)
3. ICIA Candidate C: Dual-Stream Intention-Verification Recurrent Engine (DIVRE)
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
from icia.tasks.benchmark_suite import SyntheticBenchmarkGenerator


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_step(model: nn.Module, optimizer: optim.Optimizer, x: torch.Tensor, g: torch.Tensor, y: torch.Tensor, model_type: str) -> float:
    model.train()
    optimizer.zero_grad()
    if model_type == "SAT":
        logits, _ = model(x)
    elif model_type == "ICT":
        logits, _ = model(x, g)
    elif model_type == "DIVRE":
        logits, _ = model(x, g)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Cross entropy on final token prediction
    loss = F.cross_entropy(logits[:, -1, :], y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


def evaluate_dpgp(model: nn.Module, gen: SyntheticBenchmarkGenerator, distractor_lens: List[int], model_type: str, n_trials: int = 100) -> Dict[int, Dict[str, float]]:
    """
    Evaluates Distractor Perturbation & Goal Preservation across distractor lengths.
    """
    model.eval()
    results = {}

    with torch.no_grad():
        for d_len in distractor_lens:
            x_test, g_test, y_test = gen.generate_dpgp_batch(n_trials, d_len)
            start_time = time.perf_counter()
            if model_type == "SAT":
                logits, weights = model(x_test)
                # Compute attention entropy to quantify dispersion
                # weights: (B, heads, T, T)
                last_token_attn = weights[:, :, -1, :] + 1e-12
                attn_entropy = float(-(last_token_attn * torch.log(last_token_attn)).sum(dim=-1).mean().item())
            elif model_type == "ICT":
                logits, _ = model(x_test, g_test)
                attn_entropy = 0.0
            elif model_type == "DIVRE":
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


def evaluate_dcpr(model: nn.Module, gen: SyntheticBenchmarkGenerator, model_type: str, n_trials: int = 100) -> Dict[str, float]:
    """
    Evaluates Dynamic Constraint Pivot & Runtime Recovery.
    Tests how many steps the model takes to pivot after mid-sequence constraint flip.
    """
    model.eval()
    with torch.no_grad():
        x_test, g_test, y_target_seq, t_flip = gen.generate_dcpr_batch(n_trials, seq_len=20)
        if model_type == "SAT":
            logits, _ = model(x_test)
        elif model_type == "ICT":
            logits, _ = model(x_test, g_test)
        elif model_type == "DIVRE":
            logits, _ = model(x_test, g_test)

        preds = torch.argmax(logits, dim=-1)  # (B, T)
        
        # Check pivot speed: at t_flip + 1, did model switch to target?
        post_flip_acc = float((preds[:, t_flip+1:] == y_target_seq[:, t_flip+1:]).float().mean().item()) * 100.0
        # Immediate pivot success: at t_flip + 1
        immediate_pivot = float((preds[:, t_flip+1] == y_target_seq[:, t_flip+1]).float().mean().item()) * 100.0
        # Pre-flip fidelity
        pre_flip_acc = float((preds[:, :t_flip] == y_target_seq[:, :t_flip]).float().mean().item()) * 100.0

    return {
        "pre_flip_accuracy": pre_flip_acc,
        "immediate_pivot_accuracy": immediate_pivot,
        "post_flip_recovery_accuracy": post_flip_acc
    }


def run_full_benchmark():
    print("=" * 80)
    print("PROJECT IIAY: MASTER BENCHMARK RUNNER")
    print("Evaluating SAT vs ICT vs ICIA-DIVRE")
    print("=" * 80)

    vocab_size = 128
    d_model = 64
    d_ff = 128
    gen = SyntheticBenchmarkGenerator(vocab_size=vocab_size, d_model=d_model, seed=42)

    # Instantiate models
    sat = StandardTransformerModel(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=d_ff)
    ict = IntentionConditionedTransformer(vocab_size=vocab_size, d_model=d_model, n_heads=4, n_layers=2, d_ff=d_ff)
    divre = DIVREModel(vocab_size=vocab_size, d_model=d_model, d_intention=64)

    models = {
        "Standard Transformer (SAT)": (sat, "SAT"),
        "Intention-Conditioned Transformer (ICT)": (ict, "ICT"),
        "ICIA Candidate C (DIVRE)": (divre, "DIVRE")
    }

    print("\n[STAGE 1] Model Parameter Calibration:")
    for name, (m, _) in models.items():
        print(f"  * {name}: {count_parameters(m):,} parameters")

    print("\n[STAGE 2] Standardized Training Phase (1,200 steps per model)...")
    train_distractor_lens = [4, 8, 16]
    for name, (m, m_type) in models.items():
        opt = optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        losses = []
        for step in range(1200):
            d_len = random.choice(train_distractor_lens)
            x_b, g_b, y_b = gen.generate_dpgp_batch(batch_size=32, distractor_len=d_len)
            l = train_step(m, opt, x_b, g_b, y_b, m_type)
            losses.append(l)
        print(f"  * {name} trained. Final batch loss: {np.mean(losses[-50:]):.4f}")

    print("\n[STAGE 3] Benchmark 1: Distractor Perturbation & Goal Preservation (DPGP)")
    eval_distractor_lens = [8, 32, 128, 512]
    dpgp_results = {}
    for name, (m, m_type) in models.items():
        print(f"  * Testing {name}...")
        res = evaluate_dpgp(m, gen, eval_distractor_lens, m_type, n_trials=100)
        dpgp_results[name] = res
        for d_len, metrics in res.items():
            print(f"      Distractors={d_len:3d} -> Acc: {metrics['accuracy']:5.1f}%, Latency: {metrics['latency_ms_per_sample']:.2f}ms")

    print("\n[STAGE 4] Benchmark 2: Dynamic Constraint Pivot & Runtime Recovery (DCPR)")
    dcpr_results = {}
    for name, (m, m_type) in models.items():
        res = evaluate_dcpr(m, gen, m_type, n_trials=100)
        dcpr_results[name] = res
        print(f"  * {name}: Immediate Pivot Acc: {res['immediate_pivot_accuracy']:.1f}%, Post-Flip Acc: {res['post_flip_recovery_accuracy']:.1f}%")

    # Serialize results to json
    final_output = {
        "parameters": {name: count_parameters(m) for name, (m, _) in models.items()},
        "dpgp_results": dpgp_results,
        "dcpr_results": dcpr_results
    }
    with open("icia_benchmark_results.json", "w") as f:
        json.dump(final_output, f, indent=2)

    print("\nBenchmark successfully executed! Results exported to icia_benchmark_results.json")
    return final_output


if __name__ == "__main__":
    run_full_benchmark()
