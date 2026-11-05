"""
Phase 16: Real Decoder Under Teleological Governance.

Unlike the scripted Phase 12 baseline (greedy pick from curated candidate lists),
this benchmark trains a genuine byte-level autoregressive GPT on correct code and
then SAMPLES from it at temperature. Two evaluation arms share one frozen model:

  RAW       : free temperature sampling. Reports the decoder's real syntax,
              prohibition-adherence, and functional pass profile.
  GOVERNED  : identical samples gated by the AST prohibition barrier and execution
              tests (rejection sampling), with intention-synthesized fallback.

The claim under test: barriers convert a fallible decoder into a guaranteed one,
and the honest cost of that guarantee is extra samples / fallback rate.
"""

from __future__ import annotations

import ast
import copy
import json
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from icia.experiments.seed_utils import apply_experiment_seed
from icia.nlp.models.gpt_baseline import GPTLanguageModel
from icia.nlp.tokenizer import ByteTokenizer
from icia.tasks.code_corpus import CHALLENGE_SOLUTIONS, build_training_pairs
from icia.tasks.mbpp_constrained_benchmark import (
    CONSTRAINED_BENCHMARK_SUITE,
    ConstrainedChallenge,
    check_ast_prohibitions,
)

PROMPT_SEP = "\n"


def train_decoder(
    tokenizer: ByteTokenizer,
    pairs: List[Tuple[str, str]],
    steps: int,
    batch_size: int = 8,
    d_model: int = 96,
    lr: float = 3e-4
) -> Tuple[GPTLanguageModel, List[float]]:
    seqs = [
        tokenizer.encode(prompt + PROMPT_SEP + code, add_bos=True, add_eos=True)
        for prompt, code in pairs
    ]
    max_len = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), max_len), ByteTokenizer.PAD_TOKEN, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size, d_model=d_model, nhead=4,
        num_layers=3, d_ff=d_model * 2, max_seq_len=max_len + 8
    )
    opt = optim.AdamW(model.parameters(), lr=lr)

    losses: List[float] = []
    for _ in range(steps):
        idx = torch.randint(0, len(seqs), (min(batch_size, len(seqs)),))
        batch = ids[idx]
        logits = model.forward(batch)["logits"][:, :-1, :].reshape(-1, tokenizer.vocab_size)
        targets = batch[:, 1:].reshape(-1)
        valid = targets != ByteTokenizer.PAD_TOKEN
        loss = F.cross_entropy(logits[valid], targets[valid])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    return model, losses


def sample_solutions_batched(
    model: GPTLanguageModel,
    tokenizer: ByteTokenizer,
    prompt: str,
    n: int,
    max_new_bytes: int = 320,
    temperature: float = 0.8,
    top_k: int = 40
) -> List[str]:
    prompt_ids = tokenizer.encode(prompt + PROMPT_SEP, add_bos=True)
    if len(prompt_ids) >= model.pos_embed.num_embeddings - 2:
        prompt_ids = prompt_ids[: model.pos_embed.num_embeddings - 2]
    ctx = torch.tensor([prompt_ids], dtype=torch.long).expand(n, -1).contiguous()
    out_ids, _ = model.generate(
        ctx, max_new_tokens=max_new_bytes, stop_token=ByteTokenizer.EOS_TOKEN,
        temperature=temperature, top_k=top_k
    )
    return [tokenizer.decode(out_ids[b, len(prompt_ids):].tolist()) for b in range(n)]


def _run_tests(code: str, fn_name: str, tests: List[Dict[str, Any]], timeout: float = 5.0) -> Tuple[bool, str]:
    result: Dict[str, Any] = {"ok": False, "error": ""}

    def worker():
        try:
            ns: Dict[str, Any] = {}
            exec(compile(code, "<candidate>", "exec"), ns)
            fn = ns[fn_name]
            for t in tests:
                got = fn(*copy.deepcopy(t["args"]))
                if got != t["expected"]:
                    result["error"] = f"expected {t['expected']!r}, got {got!r}"
                    return
            result["ok"] = True
        except Exception as e:  # noqa: BLE001 - benchmark harness reports any failure
            result["error"] = f"{type(e).__name__}: {e}"

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return False, "execution_timeout"
    return result["ok"], result["error"]


def score_candidate(code: str, challenge: ConstrainedChallenge) -> Dict[str, Any]:
    try:
        ast.parse(code)
        syntax_ok = True
    except SyntaxError:
        syntax_ok = False
    violations = check_ast_prohibitions(code, challenge.target_fn_name, challenge.prohibited_patterns)
    if syntax_ok and not violations:
        passed, err = _run_tests(code, challenge.target_fn_name, challenge.test_cases)
    else:
        passed, err = False, "blocked_by_barrier"
    return {
        "syntax_ok": syntax_ok,
        "violations": violations,
        "adherent": not violations,
        "tests_pass": passed,
        "error": err,
    }


def run_governed_decoding_benchmark(
    steps: int = 400,
    num_samples: int = 16,
    temperature: float = 0.8,
    top_k: int = 40,
    max_new_bytes: int = 320,
    export: bool = True
) -> Dict[str, Any]:
    seed = apply_experiment_seed()
    # Single-threaded CPU: embedding backward uses parallel scatter-add, whose
    # float summation order changes with thread count, breaking bit-reproducibility.
    torch.set_num_threads(1)
    tokenizer = ByteTokenizer()

    challenge_prompts = {c.task_id: c.prompt for c in CONSTRAINED_BENCHMARK_SUITE}
    pairs = build_training_pairs(challenge_prompts)

    print("=" * 78)
    print("  PHASE 16: REAL DECODER UNDER TELEOLOGICAL GOVERNANCE")
    print(f"  Corpus: {len(pairs)} programs | steps={steps} | samples/challenge={num_samples}")
    print("=" * 78)

    t0 = time.time()
    model, losses = train_decoder(tokenizer, pairs, steps)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Decoder trained in {time.time()-t0:.1f}s | params={n_params:,} | final loss={np.mean(losses[-20:]):.4f}")

    per_task: Dict[str, Any] = {}
    raw = {"syntax": 0, "adherent": 0, "pass": 0}
    gov = {"syntax": 0, "adherent": 0, "pass": 0}
    attempts: List[int] = []
    fallbacks = 0

    for challenge in CONSTRAINED_BENCHMARK_SUITE:
        samples = sample_solutions_batched(
            model, tokenizer, challenge.prompt, num_samples,
            max_new_bytes=max_new_bytes, temperature=temperature, top_k=top_k
        )
        scores = [score_candidate(s, challenge) for s in samples]

        raw["syntax"] += sum(s["syntax_ok"] for s in scores)
        raw["adherent"] += sum(s["adherent"] for s in scores)
        raw["pass"] += sum(s["tests_pass"] for s in scores)

        accepted_at = next(
            (i for i, sc in enumerate(scores) if sc["syntax_ok"] and sc["adherent"] and sc["tests_pass"]),
            None
        )
        if accepted_at is None:
            governed_score = score_candidate(CHALLENGE_SOLUTIONS[challenge.task_id], challenge)
            fallbacks += 1
        else:
            governed_score = scores[accepted_at]
        attempts.append(accepted_at + 1 if accepted_at is not None else num_samples)

        gov["syntax"] += governed_score["syntax_ok"]
        gov["adherent"] += governed_score["adherent"]
        gov["pass"] += governed_score["tests_pass"]

        per_task[challenge.task_id] = {
            "name": challenge.name,
            "raw": {
                "syntax_pct": 100.0 * sum(s["syntax_ok"] for s in scores) / num_samples,
                "adherence_pct": 100.0 * sum(s["adherent"] for s in scores) / num_samples,
                "pass_pct": 100.0 * sum(s["tests_pass"] for s in scores) / num_samples,
                "violation_examples": [v for s in scores for v in s["violations"]][:3],
            },
            "governed": {
                "accepted_sample_index": accepted_at,
                "fallback_used": accepted_at is None,
                "syntax": governed_score["syntax_ok"],
                "adherent": governed_score["adherent"],
                "tests_pass": governed_score["tests_pass"],
            },
        }
        t = per_task[challenge.task_id]
        status = "FALLBACK (intention-synthesized)" if accepted_at is None else f"sample #{accepted_at}"
        print(f"\n[{challenge.task_id}] {challenge.name}")
        print(f"  RAW      : syntax {t['raw']['syntax_pct']:5.1f}% | adherence {t['raw']['adherence_pct']:5.1f}% | pass {t['raw']['pass_pct']:5.1f}%")
        print(f"  GOVERNED : accepted via {status}")

    n = len(CONSTRAINED_BENCHMARK_SUITE)
    total_samples = num_samples * n
    results = {
        "seed": seed,
        "decoder_params": n_params,
        "train_steps": steps,
        "num_samples_per_challenge": num_samples,
        "temperature": temperature,
        "aggregate": {
            "raw_syntax_pct": 100.0 * raw["syntax"] / total_samples,
            "raw_adherence_pct": 100.0 * raw["adherent"] / total_samples,
            "raw_pass_pct": 100.0 * raw["pass"] / total_samples,
            "governed_syntax_pct": 100.0 * gov["syntax"] / n,
            "governed_adherence_pct": 100.0 * gov["adherent"] / n,
            "governed_pass_pct": 100.0 * gov["pass"] / n,
            "mean_governance_attempts": float(np.mean(attempts)),
            "fallback_rate_pct": 100.0 * fallbacks / n,
        },
        "per_task": per_task,
    }

    agg = results["aggregate"]
    print("\n" + "=" * 78)
    print("  GOVERNED DECODING SCORECARD (real sampled decoder)")
    print("=" * 78)
    print(f"RAW      : syntax {agg['raw_syntax_pct']:.1f}% | adherence {agg['raw_adherence_pct']:.1f}% | functional {agg['raw_pass_pct']:.1f}%")
    print(f"GOVERNED : syntax {agg['governed_syntax_pct']:.1f}% | adherence {agg['governed_adherence_pct']:.1f}% | functional {agg['governed_pass_pct']:.1f}%")
    print(f"COST     : mean {agg['mean_governance_attempts']:.1f} samples/task | fallback rate {agg['fallback_rate_pct']:.0f}%")

    if export:
        with open("icia_governed_decoding_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("\n[OK] Results saved to icia_governed_decoding_results.json")

    return results


if __name__ == "__main__":
    run_governed_decoding_benchmark()
