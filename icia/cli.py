"""
ICIA Command Line Interface (CLI)
Global Entry Point for Project IIAY (Intention Is All You Need)
Usage:
  icia demo       - Run interactive terminal demonstration suite
  icia test       - Execute automated pytest test suite
  icia bench      - Run empirical benchmark suite
  icia serve      - Launch live neural dashboard web server
  icia scorecard  - Display empirical falsification scorecard
"""

import os
import sys
import argparse
import subprocess

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_demo():
    import demo
    demo.main()


def run_tests():
    print("Running ICIA Automated Test Suite (pytest)...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/"])
    sys.exit(res.returncode)


def run_benchmark(phase: str):
    print(f"Executing ICIA Benchmark Phase: {phase}")
    script_map = {
        "2": "icia.experiments.run_phase2_experiment",
        "3": "icia.experiments.run_phase3_experiment",
        "4": "icia.experiments.run_nlp_reasoning_experiment",
        "5": "icia.experiments.run_lm_benchmark",
        "6": "icia.experiments.run_continuous_control_experiment",
        "multiagent": "icia.experiments.run_multiagent_experiment"
    }
    if phase == "all":
        for p, mod in script_map.items():
            print(f"\n>>> Running Phase {p} ({mod}) <<<")
            subprocess.run([sys.executable, "-m", mod])
    elif phase in script_map:
        subprocess.run([sys.executable, "-m", script_map[phase]])
    else:
        print(f"Unknown phase: {phase}. Options: 2, 3, 4, 5, 6, multiagent, all")


def run_serve(port: int):
    from icia.web.server import start_server
    start_server(port=port)


def show_scorecard():
    import demo
    demo.demo_falsification_scorecard()


def main():
    parser = argparse.ArgumentParser(
        prog="icia",
        description="ICIA: Intention-Centred Intelligence Architecture CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # demo
    subparsers.add_parser("demo", help="Run interactive terminal demonstration")

    # test
    subparsers.add_parser("test", help="Run automated test suite (pytest)")

    # bench
    bench_parser = subparsers.add_parser("bench", help="Run empirical benchmark experiments")
    bench_parser.add_argument("--phase", default="all", choices=["2", "3", "4", "5", "6", "multiagent", "all"], help="Benchmark phase to execute")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch live neural web server")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")

    # scorecard
    subparsers.add_parser("scorecard", help="Display master scientific falsification scorecard")

    args = parser.parse_args()

    if args.command == "demo":
        run_demo()
    elif args.command == "test":
        run_tests()
    elif args.command == "bench":
        run_benchmark(args.phase)
    elif args.command == "serve":
        run_serve(args.port)
    elif args.command == "scorecard":
        show_scorecard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
