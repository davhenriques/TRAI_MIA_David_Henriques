"""Reproduce every table and figure of the paper with one command.

Runs each experiment script in order with the same number of seeds and saves its printed
output to results/logs/<script>.txt (CSV results go to results/runs/, figures to
results/figures/). The learner pilot study is slow (~10 min) and opt-in.

Usage:  python scripts/run_all.py [--seeds 20] [--with-pilot]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("run_baselines.py", "RQ1 baselines: random vs uncertainty (consensus labels)"),
    ("run_fairness.py", "RQ1 fairness-aware querying: ablation + lambda sweep"),
    ("build_personas.py", "Reviewer personas: fidelity vs real annotators"),
    ("calibrate_flags.py", "Flag rule calibration (held-out seeds 100-109)"),
    ("run_coordinator.py", "RQ2 accountability: coordinator v1 vs v2"),
    ("run_oversight.py", "RQ3/RQ4 oversight and exposure caps"),
    ("run_report_accuracy.py", "RQ3 report accuracy: which estimator to show the supervisor"),
    ("run_final.py", "Headline: final system vs naive HITL vs oracle"),
    ("paper_numbers.py", "Supporting numbers quoted in the paper (disagreement, total exposure)"),
]
NO_SEEDS = {"build_personas.py", "calibrate_flags.py"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--with-pilot", action="store_true")
    args = parser.parse_args()
    steps = ([("pilot_model_selection.py", "Learner pilot study (8 models)")] if args.with_pilot else []) + STEPS
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)

    start = time.time()
    for script, title in steps:
        cmd = [sys.executable, str(ROOT / "scripts" / script)]
        if script not in NO_SEEDS and script != "pilot_model_selection.py":
            cmd += ["--seeds", str(args.seeds)]
        print(f"[{time.time() - start:5.0f}s] {title} ...", flush=True)
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log = ROOT / "results" / "logs" / script.replace(".py", ".txt")
        log.write_text(result.stdout + ("\n--- stderr ---\n" + result.stderr if result.returncode else ""))
        if result.returncode:
            print(f"  FAILED (see {log})", flush=True)
            print(result.stderr[-2000:])
            sys.exit(result.returncode)
    print(f"[{time.time() - start:5.0f}s] done — logs in results/logs/, tables in results/tables/")


if __name__ == "__main__":
    main()
