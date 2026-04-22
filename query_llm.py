"""
Query an LLM with a staged prompt and save the parsed kernel + launch config.

Reads prompt from:   runs/<run_name>/problem_<id>/prompts/stage_<N>.txt
Saves kernel to:     runs/<run_name>/problem_<id>/kernels/stage_<N>.cu
Saves launch config: runs/<run_name>/problem_<id>/kernels/stage_<N>_launch.py

Usage:
    python query_llm.py --run_name my_run --problem_id 1 --stage 1
    python query_llm.py --run_name my_run --problem_id 2 --stage 1 --model openrouter/openai/gpt-oss-120b:free
"""

import argparse
import os
import re
import sys

import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.drop_params = True

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

DEFAULT_MODEL = "openrouter/openai/gpt-oss-120b:free"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096


def query_model(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def parse_response(response: str) -> tuple[str, str]:
    """Extract <kernel>...</kernel> and <launch>...</launch> blocks."""
    kernel_match = re.search(r"<kernel>(.*?)</kernel>", response, re.DOTALL)
    launch_match = re.search(r"<launch>(.*?)</launch>", response, re.DOTALL)

    if not kernel_match:
        raise ValueError("No <kernel>...</kernel> block found in response")
    if not launch_match:
        raise ValueError("No <launch>...</launch> block found in response")

    return kernel_match.group(1).strip(), launch_match.group(1).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--problem_id", type=int, required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    problem_dir = os.path.join(RUNS_DIR, args.run_name, f"problem_{args.problem_id}")
    prompt_path = os.path.join(problem_dir, "prompts", f"stage_{args.stage}.txt")
    kernels_dir = os.path.join(problem_dir, "kernels")
    os.makedirs(kernels_dir, exist_ok=True)

    if not os.path.exists(prompt_path):
        print(f"Prompt not found: {prompt_path}")
        print("Run generate_prompt.py first.")
        sys.exit(1)

    with open(prompt_path) as f:
        prompt = f.read()

    print(f"Querying {args.model} for stage {args.stage}...")
    response = query_model(prompt, args.model, args.temperature, args.max_tokens)

    kernel_src, launch_src = parse_response(response)

    kernel_path = os.path.join(kernels_dir, f"stage_{args.stage}.cu")
    launch_path = os.path.join(kernels_dir, f"stage_{args.stage}_launch.py")

    with open(kernel_path, "w") as f:
        f.write(kernel_src + "\n")
    with open(launch_path, "w") as f:
        f.write(launch_src + "\n")

    print(f"Kernel saved to  {kernel_path}")
    print(f"Launch config saved to  {launch_path}")


if __name__ == "__main__":
    main()
