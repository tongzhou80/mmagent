# mmagent

An iterative LLM-guided agent for optimizing CUDA matrix multiplication kernels.
The agent walks an LLM through a sequence of optimization stages — from a naive
baseline up through memory coalescing, shared-memory tiling, register blocking,
and beyond — evaluating correctness and performance at each step.

## Overview

```
generate_prompt.py  →  query_llm.py  →  eval.py  →  report.py
```

Each stage builds on the last:

1. **`generate_prompt.py`** — Constructs a prompt that includes the current best
   kernel and a focused optimization guide for the next stage.
2. **`query_llm.py`** — Sends the prompt to an LLM and parses its `<kernel>` and
   `<launch>` output blocks into `.cu` and `_launch.py` files.
3. **`eval.py`** — Compiles the kernel via PyCUDA `SourceModule`, checks
   correctness against `torch.matmul`, and measures median runtime over 20 trials.
4. **`report.py`** — Summarizes correctness and speedup across all problems and stages.

### Problems

The four problems mirror KernelBench level-1 matmul tasks:

| ID | Shape (M × K @ K × N)       | Characteristic         |
|----|------------------------------|------------------------|
| 1  | 4096 × 4096 × 4096           | Square                 |
| 2  | 2048 × 8192 × 4096           | Large K                |
| 6  | 256 × 524288 × 256           | Very large K, tiny C   |
| 7  | 32768 × 64 × 32768           | Small K, large C       |

### Optimization stages

| Stage | Technique                          |
|-------|------------------------------------|
| 1     | Global memory coalescing (32×8 block, swap i/j mapping) |
| 2     | Shared memory tiling *(planned)*   |
| 3     | Register blocking *(planned)*      |
| …     | …                                  |

## Setup

Requires Python 3.10 and a CUDA 12.x GPU.

```bash
cd mmagent
uv sync
```

If `nvcc` is not on your `PATH`, `eval.py` auto-detects common CUDA install
locations (`/usr/local/cuda-12.8/bin`, etc.). You can also set it explicitly:

```bash
PATH=/usr/local/cuda-12.8/bin:$PATH uv sync
```

Copy `.env.example` to `.env` and fill in your API key:

```bash
cp .env.example .env
# edit .env — set OPENROUTER_API_KEY (or whichever provider you use)
```

## Usage

All scripts take `--run_name` to namespace results under `runs/<run_name>/`.

### Run the full pipeline for one problem

```bash
uv run python generate_prompt.py --run_name my_run --problem_id 1 --stage 1
uv run python query_llm.py       --run_name my_run --problem_id 1 --stage 1
uv run python eval.py            --run_name my_run --problem_id 1 --stage 1
```

### Run all four problems

```bash
for pid in 1 2 6 7; do
    uv run python generate_prompt.py --run_name my_run --problem_id $pid --stage 1
    uv run python query_llm.py       --run_name my_run --problem_id $pid --stage 1
    uv run python eval.py            --run_name my_run --problem_id $pid --stage 1
done
```

### Report results

```bash
uv run python report.py --run_name my_run           # all stages
uv run python report.py --run_name my_run --stage 1 # specific stage
```

### Options

**`generate_prompt.py`**
```
--run_name     run namespace (required)
--problem_id   1, 2, 6, or 7 (required)
--stage        optimization stage, default 1
--dtype        fp32 | fp16 | bf16, default fp32
--M/N/K        override problem dimensions
```

**`query_llm.py`**
```
--run_name     run namespace (required)
--problem_id   problem to query (required)
--stage        stage to query, required
--model        LiteLLM model string, default openrouter/openai/gpt-oss-120b:free
--temperature  default 0.7
--max_tokens   default 4096
```

**`eval.py`**
```
--run_name     run namespace (required)
--problem_id   1, 2, 6, or 7 (required)
--stage        stage to evaluate (required)
--dtype        fp32 | fp16 | bf16, default fp32
--M/N/K        override problem dimensions
```

## File layout

```
runs/
  <run_name>/
    problem_<id>/
      prompts/
        stage_1.txt          # prompt sent to LLM
      kernels/
        stage_1.cu           # CUDA kernel source
        stage_1_launch.py    # get_launch_config(M, N, K) → (block, grid)
      results/
        stage_1.json         # correctness + timing results
```

## Correctness and tolerances

Outputs are compared against `torch.matmul` using `torch.allclose` on CPU
(to avoid OOM on large output matrices). Tolerances match KernelBench:

| dtype | atol / rtol |
|-------|-------------|
| fp32  | 1e-4        |
| fp16  | 1e-2        |
| bf16  | 1e-2        |

## Stage 1 baseline results (`gpt_oss_s1`, model: `gpt-oss-120b`)

| Problem | Shape                     | Correct | Kernel ms | PyTorch ms | Speedup |
|---------|---------------------------|---------|-----------|------------|---------|
| 1       | 4096×4096×4096            | YES     | 24.95     | 2.37       | 0.095x  |
| 2       | 2048×8192×4096            | YES     | 26.79     | 2.32       | 0.086x  |
| 6       | 256×524288×256            | YES     | 19.93     | 1.55       | 0.078x  |
| 7       | 32768×64×32768            | YES     | 29.83     | 5.44       | 0.182x  |

Stage 1 kernels are correct but ~5–13x slower than PyTorch's cuBLAS. The next
stage (shared memory tiling) is where significant performance gains begin.
