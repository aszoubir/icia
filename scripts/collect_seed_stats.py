"""
Aggregate multi-seed benchmark results into mean +/- std per headline metric.

Reads scratch/ms/<name>_s<seed>.json copies produced by the seeded benchmark
matrix (see README quickstart) and prints a table for the README scorecard.
"""

import json
import re
import statistics
from pathlib import Path

MS = Path(__file__).resolve().parent.parent / "scratch" / "ms"
SEEDS = (42, 1, 2)


def load_json_metric(path: Path, keypath):
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in keypath:
        data = data[key]
    return float(data)


def collect_json(name: str, keypath):
    vals = []
    for seed in SEEDS:
        p = MS / f"{name}_s{seed}.json"
        if not p.exists():
            continue
        try:
            vals.append(load_json_metric(p, keypath))
        except (KeyError, TypeError):
            continue
    return vals


def collect_log(name: str, pattern, cast=float):
    vals = []
    for seed in SEEDS:
        p = MS / f"{name}_s{seed}.log"
        if not p.exists():
            continue
        m = re.search(pattern, p.read_text(encoding="utf-8", errors="replace"))
        if m:
            vals.append(cast(m.group(1)))
    return vals


METRICS = [
    # (label, collector)
    ("Phase2 DPGP L=1024 SAT acc%", lambda: collect_json("phase2", ("dpgp_results", "Standard Transformer (SAT)", "1024", "accuracy"))),
    ("Phase2 DPGP L=1024 ICT acc%", lambda: collect_json("phase2", ("dpgp_results", "Intention-Conditioned Transformer (ICT)", "1024", "accuracy"))),
    ("Phase2 DPGP L=1024 DIVRE-v2 acc%", lambda: collect_json("phase2", ("dpgp_results", "ICIA DIVRE-v2 (Integrated with AIAA & IGAM)", "1024", "accuracy"))),
    ("Phase3 clean SAT success%", lambda: collect_json("phase3", ("benchmark_1_clean", "Open-Loop Transformer (SAT)", "success_rate"))),
    ("Phase3 clean ICIA success%", lambda: collect_json("phase3", ("benchmark_1_clean", "ICIA Autonomous Engine", "success_rate"))),
    ("Phase3 roadblock SAT success%", lambda: collect_json("phase3", ("benchmark_2_roadblock", "Open-Loop Transformer (SAT)", "success_rate"))),
    ("Phase3 roadblock ICIA success%", lambda: collect_json("phase3", ("benchmark_2_roadblock", "ICIA Autonomous Engine", "success_rate"))),
    ("Phase4 D=10 ICIA acc%", lambda: collect_json("nlp", ("distractor_scaling_benchmark", "10", "icia", "accuracy_pct"))),
    ("Phase4 D=10 CoT acc%", lambda: collect_json("nlp", ("distractor_scaling_benchmark", "10", "transformer_cot", "accuracy_pct"))),
    ("Phase4 D=10 ICIA latency ms", lambda: collect_json("nlp", ("distractor_scaling_benchmark", "10", "icia", "latency_ms_per_sample"))),
    ("Phase4 D=10 CoT latency ms", lambda: collect_json("nlp", ("distractor_scaling_benchmark", "10", "transformer_cot", "latency_ms_per_sample"))),
    ("Phase4 D=10 ICIA verif-conf", lambda: collect_json("nlp", ("distractor_scaling_benchmark", "10", "icia", "final_verification_confidence"))),
    ("Phase5 ICIA val PPL", lambda: collect_json("lm", ("validation_perplexity", "icia_val_ppl"))),
    ("Phase5 GPT val PPL", lambda: collect_json("lm", ("validation_perplexity", "gpt_val_ppl"))),
    ("Phase5 needle ICIA acc% (depth1)", lambda: collect_json("lm", ("needle_retrieval_benchmark", "1", "icia", "accuracy_pct"))),
    ("Phase5 needle GPT acc% (depth1)", lambda: collect_json("lm", ("needle_retrieval_benchmark", "1", "gpt_baseline", "accuracy_pct"))),
    ("Phase6 ICIA drift m", lambda: collect_log("run_continuous_control_experiment", r"ICIA-Control:.*?Position Drift = ([\d.]+) m")),
    ("Phase6 baseline drift m", lambda: collect_log("run_continuous_control_experiment", r"Baseline Policy:.*?Position Drift = ([\d.]+) m")),
    ("Phase6 ICIA control latency us", lambda: collect_log("run_continuous_control_experiment", r"ICIA Control Decision Latency:\s+([\d.]+) microseconds")),
    ("Phase6 baseline latency us", lambda: collect_log("run_continuous_control_experiment", r"Sequence Baseline Decision Latency:\s+([\d.]+) microseconds")),
    ("Phase16 RAW syntax%", lambda: collect_json("governed", ("aggregate", "raw_syntax_pct"))),
    ("Phase16 RAW adherence%", lambda: collect_json("governed", ("aggregate", "raw_adherence_pct"))),
    ("Phase16 RAW functional%", lambda: collect_json("governed", ("aggregate", "raw_pass_pct"))),
    ("Phase16 GOVERNED functional%", lambda: collect_json("governed", ("aggregate", "governed_pass_pct"))),
    ("Phase16 mean governance attempts", lambda: collect_json("governed", ("aggregate", "mean_governance_attempts"))),
    ("Phase16 fallback rate%", lambda: collect_json("governed", ("aggregate", "fallback_rate_pct"))),
    ("Phase8 AR stuck retries", lambda: collect_log("run_hierarchical_experiment", r"Autoregressive Baseline: Success = [\d.]+%.*?Stuck Retries = ([\d.]+)")),
    ("Phase7 2-agent baseline collisions", lambda: collect_log("run_multiagent_experiment", r"Uncoordinated Baseline: Success Rate = [\d.]+% \| Collisions = (\d+)")),
]


def main():
    print(f"{'metric':42s} {'n':>2s}  {'mean':>10s} {'std':>10s}  values")
    print("-" * 96)
    for label, fn in METRICS:
        vals = fn()
        if not vals:
            print(f"{label:42s}  -- (no data yet)")
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{label:42s} {len(vals):2d}  {mean:10.3f} {std:10.3f}  {[round(v, 2) for v in vals]}")


if __name__ == "__main__":
    main()
