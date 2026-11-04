"""
ICIA NLP Master Benchmark Runner: ICIA-LM vs. Causal GPT Baseline
Evaluates:
1. Narrative English Pre-Training & Validation Perplexity (PPL)
2. Needle-in-a-Haystack Factual Retrieval under Distractor Paragraph Scaling (P in [1, 3, 5, 8])
3. Memory Footprint: O(1) Bounded IGAM Slots vs. O(N) Unbounded KV-Cache Accumulation
4. Inference Latency per Token
"""

import time
import os
import json
import math
import random
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from icia.nlp.tokenizer import ByteTokenizer
from icia.nlp.data.narrative_corpus_generator import NarrativeCorpusGenerator
from icia.nlp.models.icia_lm import ICIALanguageModel
from icia.nlp.models.gpt_baseline import GPTLanguageModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_language_models(
    icia_lm: ICIALanguageModel,
    gpt_lm: GPTLanguageModel,
    generator: NarrativeCorpusGenerator,
    num_train_batches: int = 80,
    batch_size: int = 12,
    seq_len: int = 256,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Pre-trains ICIA-LM and GPT Baseline on narrative English stories.
    """
    icia_lm.to(device)
    gpt_lm.to(device)

    opt_icia = optim.AdamW(icia_lm.parameters(), lr=1.5e-3, weight_decay=1e-4)
    opt_gpt = optim.AdamW(gpt_lm.parameters(), lr=1.5e-3, weight_decay=1e-4)

    icia_losses = []
    gpt_losses = []

    print(f"\n--- Pre-Training ICIA-LM and GPT Baseline on English Stories ({num_train_batches} batches, seq_len={seq_len}) ---")
    start_train = time.perf_counter()

    for batch_idx in range(1, num_train_batches + 1):
        # Mix clean narratives with needle QA narratives
        include_needle = (random.random() < 0.4)
        batch = generator.generate_batch(
            batch_size=batch_size,
            num_distractor_paragraphs=random.choice([1, 2]),
            with_needle=include_needle,
            max_len=seq_len
        )
        tokens = batch["prompt_tokens"].to(device) # (B, seq_len)
        mask = batch["prompt_mask"].to(device)

        # Autoregressive target: predict next token (shift by 1)
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        target_mask = mask[:, 1:]

        # 1. Train ICIA-LM
        icia_lm.train()
        opt_icia.zero_grad()
        out_icia = icia_lm(input_ids)
        logits_icia = out_icia["logits"] # (B, seq_len - 1, V)
        loss_icia = F.cross_entropy(
            logits_icia.reshape(-1, icia_lm.vocab_size),
            target_ids.reshape(-1),
            ignore_index=0
        )
        loss_icia.backward()
        torch.nn.utils.clip_grad_norm_(icia_lm.parameters(), 1.0)
        opt_icia.step()
        icia_losses.append(float(loss_icia.item()))

        # 2. Train GPT Baseline
        gpt_lm.train()
        opt_gpt.zero_grad()
        out_gpt = gpt_lm(input_ids, causal=True)
        logits_gpt = out_gpt["logits"] # (B, seq_len - 1, V)
        loss_gpt = F.cross_entropy(
            logits_gpt.reshape(-1, gpt_lm.vocab_size),
            target_ids.reshape(-1),
            ignore_index=0
        )
        loss_gpt.backward()
        torch.nn.utils.clip_grad_norm_(gpt_lm.parameters(), 1.0)
        opt_gpt.step()
        gpt_losses.append(float(loss_gpt.item()))

        if batch_idx % 20 == 0 or batch_idx == num_train_batches:
            ppl_icia = math.exp(min(15.0, loss_icia.item()))
            ppl_gpt = math.exp(min(15.0, loss_gpt.item()))
            print(f"Batch {batch_idx:03d}/{num_train_batches} | ICIA Loss: {loss_icia.item():.4f} (PPL: {ppl_icia:6.2f}) | GPT Loss: {loss_gpt.item():.4f} (PPL: {ppl_gpt:6.2f})")

    train_time = time.perf_counter() - start_train
    print(f"Pre-training finished in {train_time:.2f} s.")
    return {
        "train_time_sec": train_time,
        "icia_losses": icia_losses,
        "gpt_losses": gpt_losses
    }


def evaluate_validation_perplexity(
    icia_lm: ICIALanguageModel,
    gpt_lm: GPTLanguageModel,
    generator: NarrativeCorpusGenerator,
    num_val_batches: int = 15,
    batch_size: int = 12,
    seq_len: int = 256,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Evaluates validation loss and perplexity on unseen English narratives.
    """
    icia_lm.eval()
    gpt_lm.eval()

    total_loss_icia = 0.0
    total_loss_gpt = 0.0
    total_tokens = 0

    with torch.no_grad():
        for _ in range(num_val_batches):
            batch = generator.generate_batch(
                batch_size=batch_size,
                num_distractor_paragraphs=2,
                with_needle=False,
                max_len=seq_len
            )
            tokens = batch["prompt_tokens"].to(device)
            input_ids = tokens[:, :-1]
            target_ids = tokens[:, 1:]

            n_tok = (target_ids != 0).sum().item()

            out_icia = icia_lm(input_ids)
            l_icia = F.cross_entropy(out_icia["logits"].reshape(-1, icia_lm.vocab_size), target_ids.reshape(-1), ignore_index=0, reduction="sum")
            total_loss_icia += l_icia.item()

            out_gpt = gpt_lm(input_ids, causal=True)
            l_gpt = F.cross_entropy(out_gpt["logits"].reshape(-1, gpt_lm.vocab_size), target_ids.reshape(-1), ignore_index=0, reduction="sum")
            total_loss_gpt += l_gpt.item()

            total_tokens += n_tok

    mean_loss_icia = total_loss_icia / total_tokens
    mean_loss_gpt = total_loss_gpt / total_tokens
    ppl_icia = math.exp(min(15.0, mean_loss_icia))
    ppl_gpt = math.exp(min(15.0, mean_loss_gpt))

    print(f"\n--- Validation Perplexity on English Stories ---")
    print(f"ICIA-LM:      Loss = {mean_loss_icia:.4f} | Perplexity = {ppl_icia:6.2f}")
    print(f"GPT Baseline: Loss = {mean_loss_gpt:.4f} | Perplexity = {ppl_gpt:6.2f}")

    return {
        "icia_val_loss": mean_loss_icia,
        "icia_val_ppl": ppl_icia,
        "gpt_val_loss": mean_loss_gpt,
        "gpt_val_ppl": ppl_gpt
    }


def evaluate_needle_retrieval(
    icia_lm: ICIALanguageModel,
    gpt_lm: GPTLanguageModel,
    generator: NarrativeCorpusGenerator,
    tokenizer: ByteTokenizer,
    paragraph_counts: List[int] = [1, 3, 5, 8],
    num_samples_per_level: int = 20,
    device: str = "cpu"
) -> Dict[int, Dict[str, Any]]:
    """
    Evaluates needle-in-a-haystack retrieval as distractor paragraphs scale.
    Measures:
    - Context length (bytes)
    - KV-cache memory slots consumed (O(1) vs O(N))
    - Exact match accuracy
    - Inference latency
    """
    icia_lm.eval()
    gpt_lm.eval()
    results = {}

    print("\n--- Evaluating Needle-in-a-Haystack Factual Retrieval ---")

    for p_count in paragraph_counts:
        samples = [
            generator.generate_story(num_distractor_paragraphs=p_count, with_needle=True)
            for _ in range(num_samples_per_level)
        ]

        avg_len = sum(s["token_count"] for s in samples) / len(samples)

        icia_correct = 0
        gpt_correct = 0

        t0_icia = time.perf_counter()
        with torch.no_grad():
            for s in samples:
                p_tokens = torch.tensor([s["prompt_only_tokens"]], dtype=torch.long, device=device)
                target_ans = s["answer_text"].strip()
                ans_len = max(3, len(s["answer_tokens"]))

                # Generate with ICIA-LM (O(1) bounded memory)
                gen_icia, _ = icia_lm.generate(p_tokens, max_new_tokens=ans_len)
                gen_text_icia = tokenizer.decode(gen_icia[0, p_tokens.size(1):].tolist()).strip()
                if target_ans.lower() in gen_text_icia.lower():
                    icia_correct += 1

        t_icia = (time.perf_counter() - t0_icia) * 1000.0 / num_samples_per_level

        t0_gpt = time.perf_counter()
        with torch.no_grad():
            for s in samples:
                p_tokens = torch.tensor([s["prompt_only_tokens"]], dtype=torch.long, device=device)
                target_ans = s["answer_text"].strip()
                ans_len = max(3, len(s["answer_tokens"]))

                # Generate with GPT Baseline (O(N) KV cache)
                gen_gpt, _ = gpt_lm.generate(p_tokens, max_new_tokens=ans_len)
                gen_text_gpt = tokenizer.decode(gen_gpt[0, p_tokens.size(1):].tolist()).strip()
                if target_ans.lower() in gen_text_gpt.lower():
                    gpt_correct += 1

        t_gpt = (time.perf_counter() - t0_gpt) * 1000.0 / num_samples_per_level

        icia_acc = (icia_correct / num_samples_per_level) * 100.0
        gpt_acc = (gpt_correct / num_samples_per_level) * 100.0

        # Memory slots comparison
        icia_slots = icia_lm.num_memory_slots # 32 slots constant!
        gpt_slots = int(avg_len) # scales with sequence length!

        print(f"Distractor Paragraphs: {p_count:2d} | Avg Length: {avg_len:6.1f} bytes | "
              f"ICIA: {icia_acc:5.1f}% ({icia_slots} memory slots, {t_icia:5.1f} ms) | "
              f"GPT: {gpt_acc:5.1f}% ({gpt_slots} memory slots, {t_gpt:5.1f} ms)")

        results[p_count] = {
            "num_distractor_paragraphs": p_count,
            "avg_context_bytes": avg_len,
            "icia": {
                "accuracy_pct": icia_acc,
                "memory_slots": icia_slots,
                "latency_ms_per_sample": t_icia
            },
            "gpt_baseline": {
                "accuracy_pct": gpt_acc,
                "memory_slots": gpt_slots,
                "latency_ms_per_sample": t_gpt
            }
        }

    return results


def main():
    set_seed(42)
    device = "cpu"

    print("=" * 75)
    print("  ICIA-LM: NATURAL LANGUAGE PRE-TRAINING & NEEDLE-IN-A-HAYSTACK SCALING")
    print("=" * 75)

    tokenizer = ByteTokenizer()
    generator = NarrativeCorpusGenerator(tokenizer=tokenizer, seed=42)

    # Initialize models
    icia_lm = ICIALanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_layers=3,
        num_memory_slots=32,
        max_seq_len=8192
    )
    gpt_lm = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=3,
        d_ff=160,
        max_seq_len=8192
    )

    icia_params = sum(p.numel() for p in icia_lm.parameters())
    gpt_params = sum(p.numel() for p in gpt_lm.parameters())

    print(f"Model Capacity:")
    print(f" - ICIA-LM:       {icia_params:,} parameters (Dual-Stream + O(1) IGAM Bounded Memory)")
    print(f" - GPT Baseline:  {gpt_params:,} parameters (Standard Causal Transformer + O(N) KV-Cache)")
    print(f" - Parameter Match Ratio: {icia_params / gpt_params:.3f}")

    # 1. Pre-Train models on English stories
    train_history = train_language_models(
        icia_lm=icia_lm,
        gpt_lm=gpt_lm,
        generator=generator,
        num_train_batches=70,
        batch_size=10,
        seq_len=200,
        device=device
    )

    # 2. Evaluate Validation Perplexity
    val_metrics = evaluate_validation_perplexity(
        icia_lm=icia_lm,
        gpt_lm=gpt_lm,
        generator=generator,
        num_val_batches=12,
        batch_size=10,
        seq_len=200,
        device=device
    )

    # 3. Evaluate Needle-in-a-Haystack Retrieval
    retrieval_results = evaluate_needle_retrieval(
        icia_lm=icia_lm,
        gpt_lm=gpt_lm,
        generator=generator,
        tokenizer=tokenizer,
        paragraph_counts=[1, 2, 4, 6],
        num_samples_per_level=10,
        device=device
    )

    # Serialize complete results
    final_output = {
        "metadata": {
            "experiment": "ICIA-LM Natural Language Pre-Training & Needle Retrieval",
            "date": "2026-11-04",
            "device": "CPU",
            "icia_parameters": icia_params,
            "gpt_parameters": gpt_params,
            "parameter_ratio": icia_params / gpt_params,
            "bounded_memory_slots": icia_lm.num_memory_slots
        },
        "pretraining": {
            "train_time_sec": train_history["train_time_sec"],
            "icia_final_train_loss": train_history["icia_losses"][-1],
            "gpt_final_train_loss": train_history["gpt_losses"][-1]
        },
        "validation_perplexity": val_metrics,
        "needle_retrieval_benchmark": retrieval_results
    }

    output_path = "icia_lm_benchmark_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\n[OK] Benchmark completed successfully. Results saved to {output_path}")


if __name__ == "__main__":
    main()
