import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os

# === Toy transformer-style model ===
class ToyModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

def run(rank, world_size, hidden_size, batch_size, n_iters):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = ToyModel(hidden_size).to(rank)
    model.eval()

    shard_size = hidden_size // world_size
    start = rank * shard_size
    end = (rank + 1) * shard_size

    with torch.no_grad():
        local_weight = model.linear1.weight[start:end, :].contiguous()
        local_bias = model.linear1.bias[start:end].contiguous()

    input_tensor = torch.randn(batch_size, hidden_size, device=rank)
    gathered_output = [torch.zeros(batch_size, shard_size, device=rank) for _ in range(world_size)]

    # Warmup
    for _ in range(5):
        local_out = torch.nn.functional.linear(input_tensor, local_weight, local_bias)
        dist.all_gather(gathered_output, local_out)
        full_out = torch.cat(gathered_output, dim=-1)
        full_out = model.linear2(full_out)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        local_out = torch.nn.functional.linear(input_tensor, local_weight, local_bias)
        dist.all_gather(gathered_output, local_out)
        full_out = torch.cat(gathered_output, dim=-1)
        full_out = model.linear2(full_out)
    torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / n_iters
    throughput = batch_size / latency

    if rank == 0:
        print(f"\n=== Inference Benchmark ===")
        print(f"Hidden Size: {hidden_size}, Batch Size: {batch_size}, GPUs: {world_size}")
        print(f"Average Latency: {latency * 1e3:.3f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec\n")

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size to use")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Model hidden size")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    mp.spawn(
        run,
        args=(world_size, args.hidden_size, args.batch_size, args.iters),
        nprocs=world_size,
    )

if __name__ == "__main__":
    main()