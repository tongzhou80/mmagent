"""
Collect NCU performance metrics for a kernel and compute arithmetic intensity.

Runs the kernel under ncu via kernel_runner.py, parses CSV output, and saves
a JSON profile alongside the evaluation results.

Arithmetic intensity is reported in two flavors:
  - Theoretical: 2*M*N*K FLOPs / bytes assuming each A/B element loaded once
  - Measured:    actual FMA count * 2 / actual DRAM bytes (from hardware counters)

Usage:
    python profile.py --run_name gpt_oss_s1 --problem_id 1 --stage 1
"""

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

PROBLEM_SIZES = {
    1: (4096,   4096,   4096),
    2: (2048,   4096,   8192),
    6: (256,    256,    524288),
    7: (32768,  32768,  64),
}

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

NCU_METRICS = [
    # Occupancy / launch config
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    # DRAM bandwidth
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    # L1 / L2 cache
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    # Compute
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    # Warp stall reasons
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
]


def find_ncu() -> str:
    if shutil.which("ncu"):
        return "ncu"
    for d in ["/usr/local/cuda-12.8/bin", "/usr/local/cuda-12.3/bin",
              "/usr/local/cuda-12/bin", "/usr/local/cuda/bin", "/usr/bin"]:
        p = os.path.join(d, "ncu")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "ncu not found — install Nsight Compute or add it to PATH"
    )


def find_kernel_name(run_name: str, problem_id: int, stage: int) -> str:
    kernel_path = os.path.join(RUNS_DIR, run_name, f"problem_{problem_id}",
                               "kernels", f"stage_{stage}.cu")
    with open(kernel_path) as f:
        src = f.read()
    match = re.search(r"__global__\s+\w+\s+(\w+)\s*\(", src)
    if not match:
        raise ValueError("No __global__ function found in kernel source")
    return match.group(1)


def parse_ncu_csv(csv_text: str, kernel_name: str) -> dict[str, str]:
    """Extract {metric_name: metric_value} for our kernel from NCU CSV output."""
    lines = csv_text.splitlines()
    # NCU prepends informational lines before the CSV header — find it
    header_idx = next(
        (i for i, l in enumerate(lines) if "Metric Name" in l),
        None,
    )
    if header_idx is None:
        return {}

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    metrics = {}
    for row in reader:
        if kernel_name not in row.get("Kernel Name", ""):
            continue
        name = row.get("Metric Name", "").strip()
        val  = row.get("Metric Value", "").strip()
        if name:
            metrics[name] = val
    return metrics


def to_float(val: str) -> float | None:
    try:
        return float(str(val).replace(",", ""))
    except (ValueError, AttributeError):
        return None


def compute_arithmetic_intensity(
    M: int, N: int, K: int, dtype: str, metrics: dict
) -> dict:
    elem_bytes = DTYPE_BYTES[dtype]
    flops_theoretical = 2 * M * N * K

    # Theoretical AI: perfect reuse — every A and B element read exactly once
    mem_bytes_theoretical = (M * K + K * N + M * N) * elem_bytes
    ai_theoretical = flops_theoretical / mem_bytes_theoretical

    # Measured AI: FMA counter × 2 FLOPs each / actual DRAM bytes
    ffma      = to_float(metrics.get("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"))
    dram_read = to_float(metrics.get("dram__bytes_read.sum"))
    dram_write= to_float(metrics.get("dram__bytes_write.sum"))

    ai_measured = None
    if None not in (ffma, dram_read, dram_write):
        actual_dram = dram_read + dram_write
        if actual_dram > 0:
            ai_measured = (ffma * 2) / actual_dram

    return {
        "ai_theoretical":       round(ai_theoretical, 6),
        "ai_measured":          round(ai_measured, 6) if ai_measured is not None else None,
        "flops_theoretical":    flops_theoretical,
        "mem_bytes_theoretical": mem_bytes_theoretical,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--problem_id", type=int, required=True,
                        choices=sorted(PROBLEM_SIZES))
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    args = parser.parse_args()

    default_M, default_N, default_K = PROBLEM_SIZES[args.problem_id]
    M = args.M or default_M
    N = args.N or default_N
    K = args.K or default_K

    kernel_name = find_kernel_name(args.run_name, args.problem_id, args.stage)
    ncu = find_ncu()
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_runner.py")

    cmd = [
        ncu,
        "--metrics", ",".join(NCU_METRICS),
        "--csv",
        "--target-processes", "all",
        "--kernel-name", kernel_name,
        sys.executable, runner,
        "--run_name", args.run_name,
        "--problem_id", str(args.problem_id),
        "--stage", str(args.stage),
        "--dtype", args.dtype,
        "--M", str(M), "--N", str(N), "--K", str(K),
    ]

    print(f"Profiling problem {args.problem_id} stage {args.stage} "
          f"({M}x{K} @ {K}x{N}, {args.dtype})...")
    print(f"  Kernel function: {kernel_name}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0 or not result.stdout.strip():
        if "ERR_NVGPUCTRPERM" in result.stderr:
            print("[NCU ERROR] Permission denied accessing GPU performance counters.")
            print("  Fix (one-time, until reboot):")
            print("    sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'")
        else:
            print(f"[NCU ERROR]\nstdout: {result.stdout[:1000]}\nstderr: {result.stderr[:1000]}")
        sys.exit(1)

    metrics_raw = parse_ncu_csv(result.stdout, kernel_name)
    if not metrics_raw:
        print("[ERROR] No metrics parsed. NCU stdout was:")
        print(result.stdout[:3000])
        sys.exit(1)

    ai = compute_arithmetic_intensity(M, N, K, args.dtype, metrics_raw)

    # Convert all metric values to float where possible, keep string otherwise
    metrics_out = {k: (to_float(v) if to_float(v) is not None else v)
                   for k, v in metrics_raw.items()}

    profile = {
        "stage":        args.stage,
        "problem_id":   args.problem_id,
        "dtype":        args.dtype,
        "M": M, "N": N, "K": K,
        "kernel_name":  kernel_name,
        "arithmetic_intensity": ai,
        "metrics":      metrics_out,
    }

    profiles_dir = os.path.join(RUNS_DIR, args.run_name,
                                f"problem_{args.problem_id}", "profiles")
    os.makedirs(profiles_dir, exist_ok=True)
    out_path = os.path.join(profiles_dir, f"stage_{args.stage}.json")
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)

    # --- Pretty print ---
    print(f"\nProfile saved to {out_path}")

    print(f"\n{'─'*45}")
    print(f"  Arithmetic Intensity")
    print(f"{'─'*45}")
    print(f"  Theoretical  {ai['ai_theoretical']:.4f} FLOPs/byte")
    if ai['ai_measured'] is not None:
        print(f"  Measured     {ai['ai_measured']:.4f} FLOPs/byte")
    else:
        print(f"  Measured     N/A  (missing FMA or DRAM counters)")

    KEY_METRICS = [
        ("Achieved occupancy",    "sm__warps_active.avg.pct_of_peak_sustained_active",           "%"),
        ("Registers/thread",      "launch__registers_per_thread",                                 ""),
        ("Static smem/block",     "launch__shared_mem_per_block_static",                          "B"),
        ("Dynamic smem/block",    "launch__shared_mem_per_block_dynamic",                         "B"),
        ("Occupancy limit",       "launch__occupancy_limit_warps",                                ""),
        ("DRAM read",             "dram__bytes_read.sum",                                         "B"),
        ("DRAM write",            "dram__bytes_write.sum",                                        "B"),
        ("DRAM throughput",       "dram__throughput.avg.pct_of_peak_sustained_elapsed",           "%"),
        ("L1 hit rate",           "l1tex__t_sector_hit_rate.pct",                                 "%"),
        ("L2 hit rate",           "lts__t_sector_hit_rate.pct",                                   "%"),
        ("FMA pipe util",         "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",  "%"),
        ("Memory stall",          "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "%"),
        ("MIO throttle stall",    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",    "%"),
    ]

    print(f"\n{'─'*45}")
    print(f"  Key Metrics")
    print(f"{'─'*45}")
    for label, metric, unit in KEY_METRICS:
        val = metrics_raw.get(metric, "N/A")
        print(f"  {label:<26} {val} {unit}")


if __name__ == "__main__":
    main()
