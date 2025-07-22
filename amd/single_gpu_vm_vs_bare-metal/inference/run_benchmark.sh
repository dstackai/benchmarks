#!/bin/bash
isl=1024
osl=1024
MaxConcurrency="4 8 16 32 64 128 256"
RESULT_DIR="./results_concurrency_sweep"
mkdir -p $RESULT_DIR

for concurrency in $MaxConcurrency; do
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    FILENAME="llama3.3-70B-random-${concurrency}concurrency-${TIMESTAMP}.json"

    python3 /app/vllm/benchmarks/benchmark_serving.py \
        --model meta-llama/Llama-3.3-70B-Instruct \
        --dataset-name random \
        --random-input-len $isl \
        --random-output-len $osl \
        --num-prompts $((10 * $concurrency)) \
        --max-concurrency $concurrency \
        --ignore-eos \
        --percentile-metrics ttft,tpot,e2el \
        --save-result \
        --result-dir "$RESULT_DIR" \
        --result-filename "$FILENAME"
done