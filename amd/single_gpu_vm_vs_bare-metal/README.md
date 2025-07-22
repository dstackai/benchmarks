# VM Vs Bare-metal Benchmark

This provides steps to run inference and training performance benchmark on single AMD MI300X GPUs using VM and Bare-metal.

## Benchmarks

### 1. Inference

**Run VM Benchmark:**

```bash
cd inference
./run_vm_container.sh
# Inside container:
vllm serve meta-llama/Llama-3.3-70B-Instruct --max-model-len 100000
./run_benchmark.sh
```

**Run Bare-metal Benchmark:**

```bash
cd inference
./run_baremetal_container.sh
# Inside container:
vllm serve meta-llama/Llama-3.3-70B-Instruct --max-model-len 100000
./run_benchmark.sh
```

### 2. Training

Using the `rocm/dev-ubuntu-22.04:6.4-complete` image.

**Run VM Benchmark:**

```bash
cd inference
./run_vm_container.sh
# Inside container:
./run_benchmark.sh
```

**Run VM Benchmark:**

```bash
cd inference
./run_baremetal_container.sh
# Inside container:
./run_benchmark.sh
```