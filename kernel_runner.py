"""
Minimal kernel runner used as the NCU profiling target.

Compiles and runs the kernel exactly once against fixed random inputs.
Designed to be invoked by profile.py via ncu — not run directly.
"""

import argparse
import os
import re
import sys

import numpy as np
import torch


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

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--problem_id", type=int, required=True)
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
    torch_dtype = DTYPE_MAP[args.dtype]

    kernel_path = os.path.join(RUNS_DIR, args.run_name, f"problem_{args.problem_id}",
                               "kernels", f"stage_{args.stage}.cu")
    launch_path = os.path.join(RUNS_DIR, args.run_name, f"problem_{args.problem_id}",
                               "kernels", f"stage_{args.stage}_launch.py")

    with open(kernel_path) as f:
        kernel_src = f.read()
    with open(launch_path) as f:
        launch_src = f.read()

    match = re.search(r"__global__\s+\w+\s+(\w+)\s*\(", kernel_src)
    if not match:
        print("ERROR: no __global__ function found", file=sys.stderr)
        sys.exit(1)
    fn_name = match.group(1)

    torch.cuda.init()

    import atexit
    import pycuda.driver as drv
    drv.init()
    device = drv.Device(torch.cuda.current_device())
    ctx = device.retain_primary_context()
    ctx.push()
    atexit.register(ctx.pop)

    from pycuda.compiler import SourceModule
    module = SourceModule(kernel_src, options=["-O3"])
    kernel_fn = module.get_function(fn_name)

    ns = {}
    exec(launch_src, ns)
    get_launch_config = ns["get_launch_config"]

    torch.manual_seed(0)
    A = torch.rand(M, K, device="cuda", dtype=torch_dtype)
    B = torch.rand(K, N, device="cuda", dtype=torch_dtype)
    C = torch.zeros(M, N, device="cuda", dtype=torch_dtype)

    block, grid = get_launch_config(M, N, K)
    kernel_fn(
        np.intp(A.data_ptr()), np.intp(B.data_ptr()), np.intp(C.data_ptr()),
        np.int32(M), np.int32(N), np.int32(K),
        block=block, grid=grid,
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
