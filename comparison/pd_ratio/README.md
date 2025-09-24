# Benchmarking Prefill-Decode disaggregation ratios

## Introduction
This benchmark evaluates how the ratio of Prefill to Decode workers affects inference performance under different workload profiles and concurrency levels. By measuring TTFT, ITL, and throughput, it investigates how Prefill-Decode worker allocation influences both latency and system efficiency.From these results, we want to theorize whether the dynamic ratio adjustment can significantly improve performance, or if a fixed ratio is sufficient when the workload profile is known in advance.

## Benchmark setup
• GPU: Nvidia 8xH200 SXM5
• CPU: Intel Xeon Platinum 8468
• Model: openai/gpt-oss-120b


## Benchmark steps
1. Run a lmsysorg/sglang:latest container
   ```bash
   docker run --gpus all \
       --shm-size 32g \
       --network=host \
       -it \
       --ipc=host \
       lmsysorg/sglang:latest \
       bash
   ```
2. Create Prefill-Decode Worker

   **3 Prefill 1 Decode Configuration:**
   
   Prefill Server 1 (GPUs 0-1, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30000 \
     --disaggregation-bootstrap-port 9000
   ```
   
   Prefill Server 2 (GPUs 2-3, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30001 \
     --disaggregation-bootstrap-port 9001
   ```
   
   Prefill Server 3 (GPUs 4-5, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=4,5 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30002 \
     --disaggregation-bootstrap-port 9002
   ```
   
   Decode Server 1 (GPUs 6-7, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30003
   ```

   **2 Prefill 2 Decode Configuration:**
   
   Prefill Server 1 (GPUs 0-1, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30000 \
     --disaggregation-bootstrap-port 9000
   ```
   
   Prefill Server 2 (GPUs 2-3, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30001 \
     --disaggregation-bootstrap-port 9001
   ```
   
   Decode Server 1 (GPUs 4-5, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=4,5 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30002
   ```
   
   Decode Server 2 (GPUs 6-7, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30003
   ```

   **1 Prefill 3 Decode Configuration:**
   
   Prefill Server 1 (GPUs 0-1, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode prefill \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30000 \
     --disaggregation-bootstrap-port 9000
   ```
   
   Decode Server 1 (GPUs 2-3, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=2,3 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30001
   ```
   
   Decode Server 2 (GPUs 4-5, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=4,5 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30002
   ```
   
   Decode Server 3 (GPUs 6-7, TP=2):
   ```bash
   export MODEL_ID=openai/gpt-oss-120b
   CUDA_VISIBLE_DEVICES=6,7 python -m sglang.launch_server \
     --model-path $MODEL_ID \
     --disaggregation-mode decode \
     --disaggregation-transfer-backend nixl \
     --base-gpu-id 0 \
     --tp-size 2 \
     --port 30003
   ```

3. Start the Router

   **3 Prefill 1 Decode Router:**
   ```bash
   python -m sglang_router.launch_router \
     --pd-disaggregation \
     --prefill http://127.0.0.1:30000 9000 \
     --prefill http://127.0.0.1:30001 9001 \
     --prefill http://127.0.0.1:30002 9002 \
     --decode http://127.0.0.1:30003 \
     --policy round_robin \
     --host 0.0.0.0 \
     --port 8000
   ```

   **2 Prefill 2 Decode Router:**
   ```bash
   python -m sglang_router.launch_router \
     --pd-disaggregation \
     --prefill http://127.0.0.1:30000 9000 \
     --prefill http://127.0.0.1:30001 9001 \
     --decode http://127.0.0.1:30002 \
     --decode http://127.0.0.1:30003 \
     --policy round_robin \
     --host 0.0.0.0 \
     --port 8000
   ```

   **1 Prefill 3 Decode Router:**
   ```bash
   python -m sglang_router.launch_router \
     --pd-disaggregation \
     --prefill http://127.0.0.1:30000 9000 \
     --decode http://127.0.0.1:30001 \
     --decode http://127.0.0.1:30002 \
     --decode http://127.0.0.1:30003 \
     --policy round_robin \
     --host 0.0.0.0 \
     --port 8000
   ```

4. Install vLLM
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv venv
   source .venv/bin/activate
   uv pip install vllm
   ```

5. Run the benchmark
   ```bash
   ./run_benchmark.sh "32 64 128" <ISL> <OSL>
   ```
   Where 32, 64, and 128 are three concurrency levels to test.
   
   See [run_benchmark.sh](scripts/run_benchmark.sh) for the benchmark script.

