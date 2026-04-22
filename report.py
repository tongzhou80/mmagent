"""
Report evaluation results across problems and stages for a given run.

Reads results from: runs/<run_name>/problem_<id>/results/stage_<N>.json

Usage:
    python report.py --run_name gpt_oss_s1
    python report.py --run_name gpt_oss_s1 --stage 1
"""

import argparse
import json
import os

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
PROBLEM_IDS = [1, 2, 6, 7]


def load_results(run_name: str, stage: int | None) -> list[dict]:
    run_dir = os.path.join(RUNS_DIR, run_name)
    rows = []
    for problem_id in PROBLEM_IDS:
        results_dir = os.path.join(run_dir, f"problem_{problem_id}", "results")
        if not os.path.isdir(results_dir):
            continue
        for fname in sorted(os.listdir(results_dir)):
            if not fname.endswith(".json"):
                continue
            s = int(fname.replace("stage_", "").replace(".json", ""))
            if stage is not None and s != stage:
                continue
            with open(os.path.join(results_dir, fname)) as f:
                rows.append(json.load(f))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--stage", type=int, default=None,
                        help="Filter to a specific stage (default: all stages)")
    args = parser.parse_args()

    rows = load_results(args.run_name, args.stage)
    if not rows:
        print(f"No results found for run '{args.run_name}'.")
        return

    stages = sorted({r["stage"] for r in rows})
    stage_label = f"stage {args.stage}" if args.stage else f"stages {stages}"
    print(f"Run: {args.run_name}  |  {stage_label}  |  {len(rows)} result(s)")
    print(f"{'─' * 90}")

    header = f"{'Problem':>8}  {'Stage':>5}  {'dtype':>5}  {'Shape':^28}  {'Status':^14}  {'max_diff':>10}  {'Kernel ms':>10}  {'Ref ms':>8}  {'Speedup':>8}"
    print(header)
    print(f"{'─' * 90}")

    correct_rows = []
    for r in rows:
        shape = f"{r['M']}x{r['K']} @ {r['K']}x{r['N']}"
        dtype = r.get("dtype", "fp32")
        compiled = r.get("compiled", False)
        correct = r.get("correctness", False)

        if not compiled:
            print(f"  {r['problem_id']:>6}  {r['stage']:>5}  {dtype:>5}  {shape:<28}  {'COMPILE ERR':^14}")
            continue
        if "runtime_error" in r:
            print(f"  {r['problem_id']:>6}  {r['stage']:>5}  {dtype:>5}  {shape:<28}  {'RUNTIME ERR':^14}")
            continue

        if correct:
            trials = r.get("correct_trials", "")
            status = f"OK {trials}".strip()
            correct_rows.append(r)
        else:
            trials = r.get("correct_trials", "")
            status = f"WRONG {trials}".strip()

        max_diff = f"{r.get('max_diff', float('nan')):.2e}"
        if correct:
            kernel_ms = f"{r.get('kernel_ms', float('nan')):.2f}"
            ref_ms    = f"{r.get('ref_ms', float('nan')):.2f}"
            speedup   = f"{r.get('speedup', float('nan')):.3f}x"
        else:
            kernel_ms = ref_ms = speedup = "N/A"

        print(f"  {r['problem_id']:>6}  {r['stage']:>5}  {dtype:>5}  {shape:<28}  {status:^14}  {max_diff:>10}  {kernel_ms:>10}  {ref_ms:>8}  {speedup:>8}")

    print(f"{'─' * 90}")

    total = len(rows)
    n_compile_err  = sum(1 for r in rows if not r.get("compiled"))
    n_runtime_err  = sum(1 for r in rows if r.get("compiled") and "runtime_error" in r)
    n_wrong        = sum(1 for r in rows if r.get("compiled") and "runtime_error" not in r and not r.get("correctness"))
    n_correct      = len(correct_rows)
    speedups = [r["speedup"] for r in correct_rows if r.get("speedup") is not None]
    faster   = [s for s in speedups if s > 1.0]

    print(f"Compile errors:   {n_compile_err}/{total}")
    print(f"Runtime errors:   {n_runtime_err}/{total}")
    print(f"Wrong output:     {n_wrong}/{total}")
    print(f"OK:               {n_correct}/{total} ({100*n_correct/total:.0f}%)")
    if speedups:
        print(f"Avg speedup:      {sum(speedups)/len(speedups):.3f}x  (among correct)")
        print(f"Faster than PT:   {len(faster)}/{total} ({100*len(faster)/total:.0f}%)")


if __name__ == "__main__":
    main()
