"""
Evaluate a generated kernel against the PyTorch reference (torch.matmul).

Reads kernel from:   runs/<run_name>/problem_<id>/kernels/stage_<N>.cu
                     runs/<run_name>/problem_<id>/kernels/stage_<N>_launch.py
Saves results to:    runs/<run_name>/problem_<id>/results/stage_<N>.json

Compilation via PyCUDA SourceModule (fast — no PyTorch C++ headers).
Timing via PyTorch CUDA events.

Usage:
    python eval.py --run_name my_run --problem_id 1 --stage 1
"""

import argparse
import json
import os
import re
import sys
import traceback

import numpy as np
import torch

# Ensure nvcc is on PATH for PyCUDA SourceModule compilation
def _ensure_nvcc_in_path():
    import shutil
    if shutil.which("nvcc"):
        return
    for cuda_dir in ["/usr/local/cuda/bin", "/usr/local/cuda-12.8/bin",
                     "/usr/local/cuda-12/bin", "/usr/local/cuda-11.8/bin"]:
        if os.path.isfile(os.path.join(cuda_dir, "nvcc")):
            os.environ["PATH"] = cuda_dir + os.pathsep + os.environ.get("PATH", "")
            return

_ensure_nvcc_in_path()

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

PROBLEM_SIZES = {
    1: (4096,   4096,   4096),
    2: (2048,   4096,   8192),
    6: (256,    256,    524288),
    7: (32768,  32768,  64),
}

NUM_WARMUP = 3
NUM_TRIALS = 20
NUM_CORRECT_TRIALS = 3

# Match KernelBench tolerances (inspired by torchbench)
PRECISION_TOLERANCES = {
    "fp32": 1e-4,
    "fp16": 1e-2,
    "bf16": 1e-2,
}

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_pycuda():
    """Initialize PyCUDA using PyTorch's existing CUDA context."""
    import atexit
    import pycuda.driver as drv
    drv.init()
    device_id = torch.cuda.current_device()
    device = drv.Device(device_id)
    ctx = device.retain_primary_context()
    ctx.push()
    atexit.register(ctx.pop)
    return drv, ctx


def compile_kernel(kernel_src: str):
    """Compile a CUDA kernel string with PyCUDA and return the module."""
    from pycuda.compiler import SourceModule
    return SourceModule(kernel_src, options=["-O3"])


def load_kernel(kernel_path: str, launch_path: str):
    """
    Load and compile a kernel, return a callable that takes PyTorch tensors.
    """
    with open(kernel_path) as f:
        kernel_src = f.read()
    with open(launch_path) as f:
        launch_src = f.read()

    # Extract the kernel function name
    match = re.search(r"__global__\s+\w+\s+(\w+)\s*\(", kernel_src)
    if not match:
        raise ValueError("No __global__ function found in kernel source")
    fn_name = match.group(1)

    drv, ctx = init_pycuda()
    module = compile_kernel(kernel_src)
    kernel_fn = module.get_function(fn_name)

    # Parse the launch config
    ns = {}
    exec(launch_src, ns)
    get_launch_config = ns["get_launch_config"]

    def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        N = B.shape[1]
        C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
        block, grid = get_launch_config(M, N, K)
        kernel_fn(
            np.intp(A.data_ptr()),
            np.intp(B.data_ptr()),
            np.intp(C.data_ptr()),
            np.int32(M), np.int32(N), np.int32(K),
            block=block, grid=grid,
        )
        return C

    return run


def measure_runtime_ms(fn, *args) -> float:
    for _ in range(NUM_WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(NUM_TRIALS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]  # median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--problem_id", type=int, required=True, choices=sorted(PROBLEM_SIZES))
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()

    default_M, default_N, default_K = PROBLEM_SIZES[args.problem_id]
    M = args.M or default_M
    N = args.N or default_N
    K = args.K or default_K

    problem_dir = os.path.join(RUNS_DIR, args.run_name, f"problem_{args.problem_id}")
    kernel_path = os.path.join(problem_dir, "kernels", f"stage_{args.stage}.cu")
    launch_path = os.path.join(problem_dir, "kernels", f"stage_{args.stage}_launch.py")
    results_dir = os.path.join(problem_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, f"stage_{args.stage}.json")

    tol = PRECISION_TOLERANCES[args.dtype]
    torch_dtype = DTYPE_MAP[args.dtype]
    result = {"stage": args.stage, "problem_id": args.problem_id, "dtype": args.dtype,
              "M": M, "N": N, "K": K}

    # Ensure PyTorch CUDA context is live before PyCUDA touches it
    torch.cuda.init()

    # --- Compile ---
    print(f"Compiling stage {args.stage} kernel for problem {args.problem_id} "
          f"({M}x{K} @ {K}x{N})...")
    try:
        run_kernel = load_kernel(kernel_path, launch_path)
        result["compiled"] = True
        print("  Compiled OK")
    except Exception:
        err = traceback.format_exc()
        print(f"[COMPILE ERROR]\n{err}")
        result.update({"compiled": False, "compile_error": err})
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    # --- Correctness ---
    print(f"Checking correctness ({NUM_CORRECT_TRIALS} trials)...")
    pass_count = 0
    max_diffs = []

    try:
        for trial in range(NUM_CORRECT_TRIALS):
            torch.manual_seed(trial)
            A = torch.rand(M, K, device="cuda", dtype=torch_dtype)
            B = torch.rand(K, N, device="cuda", dtype=torch_dtype)
            ref = torch.matmul(A, B)

            out = run_kernel(A, B)
            torch.cuda.synchronize()
            # Compare on CPU to avoid OOM on large output matrices
            ref_cpu = ref.cpu()
            out_cpu = out.cpu()
            torch.cuda.empty_cache()

            max_diff = (out_cpu - ref_cpu).abs().max().item()
            max_diffs.append(max_diff)
            passed = torch.allclose(out_cpu, ref_cpu, atol=tol, rtol=tol)
            status = "PASS" if passed else "FAIL"
            print(f"  trial {trial+1}/{NUM_CORRECT_TRIALS}: [{status}]  max_diff={max_diff:.2e}")
            if passed:
                pass_count += 1

        correct = (pass_count == NUM_CORRECT_TRIALS)
        result.update({
            "correctness": correct,
            "correct_trials": f"{pass_count}/{NUM_CORRECT_TRIALS}",
            "max_diff": round(max(max_diffs), 6),
        })
    except Exception:
        err = traceback.format_exc()
        print(f"[RUNTIME ERROR]\n{err}")
        result.update({"correctness": False, "runtime_error": err})
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(1)

    # --- Performance ---
    if not correct:
        print("Skipping performance measurement (correctness failed).")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(0)

    print(f"Measuring performance ({NUM_TRIALS} trials)...")
    torch.manual_seed(0)
    A = torch.rand(M, K, device="cuda", dtype=torch_dtype)
    B = torch.rand(K, N, device="cuda", dtype=torch_dtype)
    kernel_ms = measure_runtime_ms(run_kernel, A, B)
    ref_ms    = measure_runtime_ms(torch.matmul, A, B)
    speedup   = ref_ms / kernel_ms

    result.update({
        "kernel_ms": round(kernel_ms, 4),
        "ref_ms":    round(ref_ms, 4),
        "speedup":   round(speedup, 4),
    })

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {result_path}")
    print(f"  Kernel:   {kernel_ms:.2f} ms")
    print(f"  PyTorch:  {ref_ms:.2f} ms")
    print(f"  Speedup:  {speedup:.3f}x")


if __name__ == "__main__":
    main()
