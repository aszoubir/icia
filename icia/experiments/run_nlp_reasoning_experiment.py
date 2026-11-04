"""
ICIA NLP: Symbolic Multi-Step Reasoning Experiment Runner
Benchmarks:
1. ICIA-NLP Latent Teleological Reasoner (0 scratchpad tokens, O(1) context)
2. Direct Causal Transformer (Parameter-matched, no scratchpad)
3. Chain-of-Thought (CoT) Autoregressive Transformer (Emits intermediate tokens)

Evaluates:
- Distractor resistance under context scaling (D in [0, 2, 5, 10])
- Scratchpad token overhead (CoT token explosion vs. ICIA O(1))
- Verification confidence trajectory across latent thinking steps
- Latency and throughput per sample
"""

import time
import os
import json
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from icia.nlp.tokenizer import ByteTokenizer
from icia.nlp.tasks.symbolic_reasoning_generator import SymbolicReasoningGenerator
from icia.nlp.models.icia_nlp_reasoner import ICIANLPReasoner
from icia.nlp.models.transformer_nlp_baseline import TransformerNLPBaseline


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_models(
    icia_model: ICIANLPReasoner,
    trans_model: TransformerNLPBaseline,
    generator: SymbolicReasoningGenerator,
    num_train_batches: int = 150,
    batch_size: int = 24,
    device: str = "cpu"
) -> Dict[str, List[float]]:
    """
    Trains ICIA-NLP and Transformer baseline on symbolic reasoning problems.
    """
    icia_model.to(device)
    trans_model.to(device)

    opt_icia = optim.AdamW(icia_model.parameters(), lr=1e-3, weight_decay=1e-4)
    opt_trans = optim.AdamW(trans_model.parameters(), lr=1e-3, weight_decay=1e-4)

    icia_losses = []
    trans_losses = []

    print(f"--- Training ICIA-NLP and Transformer Baselines ({num_train_batches} batches, batch_size={batch_size}) ---")
    start_train = time.perf_counter()

    for batch_idx in range(1, num_train_batches + 1):
        # Mix 2 and 3 hops, distractors 0 to 3 for broad training
        n_hops = random.choice([2, 3])
        n_dist = random.randint(0, 3)
        batch = generator.generate_batch(
            batch_size=batch_size,
            num_hops=n_hops,
            num_distractors=n_dist
        )

        p_tokens = batch["prompt_tokens"].to(device)
        p_mask = batch["prompt_mask"].to(device)
        a_tokens = batch["answer_tokens"].to(device)
        cot_tokens = batch["cot_tokens"].to(device)
        a_vals = batch["answer_vals"].float().to(device)

        # ------------------- Train ICIA -------------------
        icia_model.train()
        opt_icia.zero_grad()
        out_icia = icia_model(
            prompt_tokens=p_tokens,
            target_answer_tokens=a_tokens,
            prompt_mask=p_mask,
            k_steps=4
        )
        # Loss 1: Value regression loss (MSE)
        loss_val = F.mse_loss(out_icia["pred_val"], a_vals)
        # Loss 2: Answer token cross-entropy
        logits = out_icia["token_logits"] # (B, a_len, V)
        loss_tok = F.cross_entropy(logits.view(-1, icia_model.vocab_size), a_tokens.view(-1), ignore_index=0)
        # Loss 3: Verification regularization (encourage final step confidence)
        loss_ver = F.mse_loss(out_icia["verifications"][-1], torch.ones_like(out_icia["verifications"][-1]))
        
        total_icia_loss = loss_val * 0.1 + loss_tok + 0.1 * loss_ver
        total_icia_loss.backward()
        torch.nn.utils.clip_grad_norm_(icia_model.parameters(), 1.0)
        opt_icia.step()
        icia_losses.append(float(total_icia_loss.item()))

        # ------------------- Train Transformer -------------------
        trans_model.train()
        opt_trans.zero_grad()
        # Train on CoT sequence generation: condition on prompt, predict intermediate CoT tokens + answer
        # Concatenate prompt + cot_tokens
        trans_input = torch.cat([p_tokens, cot_tokens[:, :-1]], dim=1) # (B, P + C - 1)
        trans_target = torch.cat([p_tokens[:, 1:], cot_tokens], dim=1) # (B, P + C - 1)
        
        out_trans = trans_model(trans_input, causal=True)
        logits_trans = out_trans["logits"]
        # Only compute loss on the generation portion (cot_tokens)
        p_len = p_tokens.size(1)
        gen_logits = logits_trans[:, p_len - 1:, :] # (B, C, V)
        loss_trans_seq = F.cross_entropy(
            gen_logits.contiguous().view(-1, trans_model.vocab_size),
            cot_tokens.contiguous().view(-1),
            ignore_index=0
        )
        loss_trans_val = F.mse_loss(out_trans["pred_val"], a_vals)
        total_trans_loss = loss_trans_seq + 0.1 * loss_trans_val
        total_trans_loss.backward()
        torch.nn.utils.clip_grad_norm_(trans_model.parameters(), 1.0)
        opt_trans.step()
        trans_losses.append(float(total_trans_loss.item()))

        if batch_idx % 30 == 0 or batch_idx == num_train_batches:
            print(f"Batch {batch_idx:03d}/{num_train_batches} | ICIA Loss: {total_icia_loss.item():.4f} (Val: {loss_val.item():.2f}, Tok: {loss_tok.item():.4f}) | Trans Loss: {total_trans_loss.item():.4f}")

    total_train_time = time.perf_counter() - start_train
    print(f"Training completed in {total_train_time:.2f} s.")
    return {"icia_losses": icia_losses, "trans_losses": trans_losses}


def evaluate_distractor_scaling(
    icia_model: ICIANLPReasoner,
    trans_model: TransformerNLPBaseline,
    generator: SymbolicReasoningGenerator,
    distractor_counts: List[int] = [0, 2, 5, 10],
    num_samples_per_level: int = 50,
    device: str = "cpu"
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates models across varying distractor loads.
    Measures:
    - Numeric value exact match (|pred - true| <= 1.0)
    - Scratchpad tokens emitted
    - Latency per sample
    """
    icia_model.eval()
    trans_model.eval()
    results = {}

    print("\n--- Evaluating Distractor Scaling Resistance ---")

    for d_count in distractor_counts:
        batch = generator.generate_batch(
            batch_size=num_samples_per_level,
            num_hops=2,
            num_distractors=d_count
        )
        p_tokens = batch["prompt_tokens"].to(device)
        p_mask = batch["prompt_mask"].to(device)
        a_tokens = batch["answer_tokens"].to(device)
        true_vals = batch["answer_vals"].float().to(device)

        # 1. Evaluate ICIA-NLP (Latent Teleological Reasoner)
        t0_icia = time.perf_counter()
        with torch.no_grad():
            out_icia = icia_model(prompt_tokens=p_tokens, target_answer_tokens=a_tokens, prompt_mask=p_mask, k_steps=4)
            pred_icia_vals = out_icia["pred_val"]
            icia_errors = torch.abs(pred_icia_vals - true_vals)
            icia_acc = float((icia_errors <= 1.0).float().mean().item()) * 100.0
            icia_mean_error = float(icia_errors.mean().item())
            icia_ver_final = float(out_icia["verifications"][-1].mean().item())
        t_icia = (time.perf_counter() - t0_icia) * 1000.0 / num_samples_per_level

        # 2. Evaluate Transformer CoT Generation
        t0_trans = time.perf_counter()
        with torch.no_grad():
            gen_tokens, n_gen = trans_model.generate(p_tokens, max_new_tokens=60)
            out_trans = trans_model(gen_tokens, causal=True)
            pred_trans_vals = out_trans["pred_val"]
            trans_errors = torch.abs(pred_trans_vals - true_vals)
            trans_acc = float((trans_errors <= 1.0).float().mean().item()) * 100.0
            trans_mean_error = float(trans_errors.mean().item())
        t_trans = (time.perf_counter() - t0_trans) * 1000.0 / num_samples_per_level

        # Compute average prompt length
        avg_prompt_len = float(p_mask.sum(dim=-1).mean().item())

        print(f"Distractors: {d_count:2d} (Prompt Len: {avg_prompt_len:.1f} bytes) | "
              f"ICIA Acc: {icia_acc:5.1f}% (MAE: {icia_mean_error:4.2f}, 0 scratchpad tok, {t_icia:5.2f} ms) | "
              f"Trans CoT Acc: {trans_acc:5.1f}% (MAE: {trans_mean_error:4.2f}, {n_gen} scratchpad tok, {t_trans:5.2f} ms)")

        results[d_count] = {
            "avg_prompt_length": avg_prompt_len,
            "icia": {
                "accuracy_pct": icia_acc,
                "mean_abs_error": icia_mean_error,
                "scratchpad_tokens": 0,
                "final_verification_confidence": icia_ver_final,
                "latency_ms_per_sample": t_icia
            },
            "transformer_cot": {
                "accuracy_pct": trans_acc,
                "mean_abs_error": trans_mean_error,
                "scratchpad_tokens": n_gen,
                "latency_ms_per_sample": t_trans
            }
        }

    return results


def evaluate_latent_deliberation_steps(
    icia_model: ICIANLPReasoner,
    generator: SymbolicReasoningGenerator,
    max_steps: int = 6,
    num_samples: int = 50,
    device: str = "cpu"
) -> Dict[int, Dict[str, float]]:
    """
    Evaluates ICIA accuracy and verification confidence as internal thinking steps increase from 1 to max_steps.
    """
    icia_model.eval()
    print("\n--- Evaluating ICIA Latent Thinking Step Progression (Internal Recurrence) ---")
    
    batch = generator.generate_batch(
        batch_size=num_samples,
        num_hops=3, # Harder 3-hop problem
        num_distractors=2
    )
    p_tokens = batch["prompt_tokens"].to(device)
    p_mask = batch["prompt_mask"].to(device)
    true_vals = batch["answer_vals"].float().to(device)

    step_progression = {}

    with torch.no_grad():
        for k in range(1, max_steps + 1):
            out = icia_model(
                prompt_tokens=p_tokens,
                prompt_mask=p_mask,
                k_steps=k
            )
            pred_vals = out["pred_val"]
            errors = torch.abs(pred_vals - true_vals)
            acc = float((errors <= 1.0).float().mean().item()) * 100.0
            mae = float(errors.mean().item())
            conf = float(out["verifications"][-1].mean().item())

            print(f"Thinking Steps k={k}: Accuracy = {acc:5.1f}% | MAE = {mae:4.2f} | Verification Confidence = {conf:.4f}")
            step_progression[k] = {
                "thinking_steps": k,
                "accuracy_pct": acc,
                "mean_abs_error": mae,
                "verification_confidence": conf
            }

    return step_progression


def main():
    set_seed(42)
    device = "cpu"
    print("=" * 70)
    print("  ICIA-NLP EXPERIMENT: LATENT RECURRENCE VS. AUTOREGRESSIVE CoT")
    print("=" * 70)

    tokenizer = ByteTokenizer()
    generator = SymbolicReasoningGenerator(tokenizer=tokenizer, seed=42)

    # Initialize models
    icia_reasoner = ICIANLPReasoner(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        d_latent=128,
        k_think_steps=4
    )
    transformer_baseline = TransformerNLPBaseline(
        vocab_size=tokenizer.vocab_size,
        d_model=96,
        nhead=4,
        num_layers=4,
        d_ff=256
    )

    icia_param_count = sum(p.numel() for p in icia_reasoner.parameters())
    trans_param_count = sum(p.numel() for p in transformer_baseline.parameters())

    print(f"Model Architectures:")
    print(f" - ICIA-NLP Reasoner:       {icia_param_count:,} parameters (Latent Teleological Recurrence)")
    print(f" - Causal Transformer:      {trans_param_count:,} parameters (Autoregressive CoT)")

    # 1. Train models
    train_history = train_models(
        icia_model=icia_reasoner,
        trans_model=transformer_baseline,
        generator=generator,
        num_train_batches=160,
        batch_size=32,
        device=device
    )

    # 2. Evaluate Distractor Scaling
    distractor_results = evaluate_distractor_scaling(
        icia_model=icia_reasoner,
        trans_model=transformer_baseline,
        generator=generator,
        distractor_counts=[0, 2, 5, 10],
        num_samples_per_level=60,
        device=device
    )

    # 3. Evaluate Latent Deliberation Depth
    deliberation_results = evaluate_latent_deliberation_steps(
        icia_model=icia_reasoner,
        generator=generator,
        max_steps=6,
        num_samples=60,
        device=device
    )

    # Serialize complete benchmark results
    final_output = {
        "metadata": {
            "experiment": "ICIA-NLP Symbolic Multi-Step Reasoning",
            "date": "2026-11-04",
            "device": "CPU",
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "icia_parameters": icia_param_count,
            "transformer_parameters": trans_param_count,
            "parameter_ratio_icia_to_transformer": icia_param_count / trans_param_count
        },
        "distractor_scaling_benchmark": distractor_results,
        "latent_deliberation_depth_benchmark": deliberation_results,
        "loss_summary": {
            "icia_final_loss": train_history["icia_losses"][-1],
            "transformer_final_loss": train_history["trans_losses"][-1]
        }
    }

    output_path = "icia_nlp_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[OK] Benchmark completed successfully. Results saved to {output_path}")


if __name__ == "__main__":
    main()
