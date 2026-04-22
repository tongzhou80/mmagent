"""
Generate a staged optimization prompt for matrix multiplication C = A @ B.

Each stage builds on the previous one, providing the current best implementation
plus a focused optimization guide for the LM to apply.

Prompts are saved to runs/<run_name>/problem_<id>/prompts/stage_<N>.txt

Problem sizes mirror the corresponding KernelBench level-1 problems:
  1 — 4096 x 4096 x 4096  (square)
  2 — 2048 x 8192 x 4096  (large K)
  6 —  256 x 524288 x 256  (very large K, small output)
  7 — 32768 x 64 x 32768  (small K, large output)

Usage:
    python generate_prompt.py --run_name my_run --problem_id 1 --stage 1
    python generate_prompt.py --run_name my_run --problem_id 2 --stage 1 --dtype fp16
"""

import argparse
import os

# ---------------------------------------------------------------------------
# Naive baseline implementation (i-j parallelized, 16x16 thread block)
# ---------------------------------------------------------------------------

NAIVE_KERNEL = """\
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < M && j < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
    }
}

// Launch configuration:
//   dim3 block(16, 16);
//   dim3 grid((M + 15) / 16, (N + 15) / 16);
"""

# ---------------------------------------------------------------------------
# Stage descriptions
# ---------------------------------------------------------------------------

STAGE_1_GUIDE = """\
## Stage 1: Global Memory Coalescing

This stage has two changes. Apply both of them.

### 1a — Swap the thread index to dimension mapping

In the naive kernel, `threadIdx.x` is mapped to row `i` and `threadIdx.y` to
column `j`. This is backwards for coalescing: the x-dimension is the
fastest-changing dimension within a warp, so it should map to the
fastest-changing index in memory.

Change:
```c
// Before (uncoalesced)
int i = blockDim.x * blockIdx.x + threadIdx.x;
int j = blockDim.y * blockIdx.y + threadIdx.y;

// After (coalesced)
int j = blockDim.x * blockIdx.x + threadIdx.x;
int i = blockDim.y * blockIdx.y + threadIdx.y;
```

### 1b — Make the x-dimension a multiple of 32 (warp size)

A warp is 32 threads that execute in lockstep. For coalescing to be fully
effective, the x-dimension of the thread block should be exactly 32 so that
each warp maps to a contiguous 32-element row segment.

Change the thread block from `(16, 16)` to `(32, 8)` and update the grid
accordingly:
```c
// Before
dim3 block(16, 16);
dim3 grid((M + 15) / 16, (N + 15) / 16);

// After
dim3 block(32, 8);
dim3 grid((N + 31) / 32, (M + 7) / 8);
```

Note the grid dimensions also flip to match the swapped i/j mapping.

### What NOT to do in this stage

Do not apply any other optimizations yet — no shared memory tiling,
no register blocking, no vectorized loads. Only apply the two changes above.
"""

# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "fp32": "float",
    "fp16": "__half",
    "bf16": "__nv_bfloat16",
}

def build_prompt(M: int, N: int, K: int, dtype: str, stage: int) -> str:
    c_dtype = DTYPE_MAP.get(dtype, "float")

    header = f"""\
You are an expert CUDA programmer optimizing matrix multiplication step by step.

## Problem

Compute C = A @ B where:
  A : {M} x {K}  ({dtype} / {c_dtype})
  B : {K} x {N}  ({dtype} / {c_dtype})
  C : {M} x {N}  ({dtype} / {c_dtype})

Matrices are stored in row-major order.

## Current implementation (naive baseline)

The kernel below is the starting point. It is correct but unoptimized:

```c
{NAIVE_KERNEL.strip()}
```
"""

    if stage == 1:
        stage_section = f"\n## Your task\n\n{STAGE_1_GUIDE.strip()}\n"
    else:
        raise NotImplementedError(f"Stage {stage} not yet defined.")

    footer = """
## Output format

Output ONLY two delimited blocks — nothing else, no prose, no markdown outside
the blocks.

**Block 1 — the CUDA kernel(s):**

```
<kernel>
// All __global__ functions and any __device__ helpers go here.
// Use float for fp32, __half for fp16, __nv_bfloat16 for bf16.
__global__ void matmul(const float* A, const float* B, float* C,
                        int M, int N, int K) {
    // ...
}
</kernel>
```

**Block 2 — the launch configuration as a Python function:**

```
<launch>
def get_launch_config(M, N, K):
    block = (32, 8, 1)
    grid = ((N + 31) // 32, (M + 7) // 8, 1)
    return block, grid
</launch>
```

`get_launch_config` must return a `(block, grid)` tuple of 3-element tuples.
`M`, `N`, `K` are plain Python ints. Do not import anything inside the function.
"""

    return header + stage_section + footer


RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

# M, N, K sizes mirroring KernelBench level-1 problems
PROBLEM_SIZES: dict[int, tuple[int, int, int]] = {
    1: (4096,   4096,   4096),    # square
    2: (2048,   4096,   8192),    # large K
    6: (256,    256,    524288),  # very large K, small output
    7: (32768,  32768,  64),      # small K, large output
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--problem_id", type=int, required=True, choices=sorted(PROBLEM_SIZES))
    parser.add_argument("--stage", type=int, default=1, help="Optimization stage (1, 2, …)")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    # Allow overriding sizes explicitly if needed
    parser.add_argument("--M", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    args = parser.parse_args()

    default_M, default_N, default_K = PROBLEM_SIZES[args.problem_id]
    M = args.M or default_M
    N = args.N or default_N
    K = args.K or default_K

    prompt = build_prompt(M, N, K, args.dtype, args.stage)

    prompts_dir = os.path.join(RUNS_DIR, args.run_name, f"problem_{args.problem_id}", "prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    out_path = os.path.join(prompts_dir, f"stage_{args.stage}.txt")
    with open(out_path, "w") as f:
        f.write(prompt)
    print(f"Prompt written to {out_path}")


if __name__ == "__main__":
    main()
