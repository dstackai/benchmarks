#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_benchmark.sh "<concurrency values>" <input_len> <output_len> <x> <y>
# Example:
# ./run_benchmark.sh "128 64 32 16 8 4 2 1" 2048 2048 1 2

if [ $# -lt 5 ]; then
  echo "Usage: $0 \"<space-separated concurrency values>\" <input_len> <output_len> <x> <y>"
  exit 1
fi

# Convert input string to array
concurrency_values=($1)
input_len=$2
output_len=$3
x=$4
y=$5

result_dir="/root/.cache/benchmark_results"
mkdir -p "${result_dir}"

for concurrency in "${concurrency_values[@]}"; do
  timestamp=$(date +"%Y%m%d_%H%M%S")
  result_filename="${x}p${y}d_conc${concurrency}_in${input_len}_out${output_len}_${timestamp}.json"

  echo "Running benchmark with concurrency=${concurrency}, input_len=${input_len}, output_len=${output_len}, prefix=${x}p${y}d"
  vllm bench serve \
    --backend vllm \
    --model "openai/gpt-oss-120b" \
    --endpoint /v1/completions \
    --dataset-name random \
    --num-prompts 500 \
    --random-input-len "${input_len}" \
    --random-output-len "${output_len}" \
    --max-concurrency "${concurrency}" \
    --save-result \
    --result-dir "${result_dir}" \
    --result-filename "${result_filename}"
done