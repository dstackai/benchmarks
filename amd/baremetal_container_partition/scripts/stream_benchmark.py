import torch
import triton
import triton.language as tl
import time
import argparse
import re

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def copy_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr, dtype: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask).to(dtype)
    tl.store(y_ptr + offsets, x, mask=mask)

def parse_size(size_str):
    """Parses a string like 8M, 4G, 2K, etc. into bytes."""
    match = re.match(r"^(\d+(?:\.\d+)?)([KMGTP]?)$", size_str.upper())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")
    num, unit = match.groups()
    num = float(num)
    scale = {
        "": 1,
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4
    }
    return int(num * scale[unit])

def benchmark_stream_copy(total_bytes, block_size, dtype=torch.float32, n_warmups=10, n_iters=100):
    element_size_bytes = torch.tensor([], dtype=dtype).element_size()
    n_elements = total_bytes // element_size_bytes

    x = torch.rand(n_elements, dtype=dtype, device=DEVICE)
    y = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Warm-up
    for _ in range(n_warmups):
        copy_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size, dtype=tl.float32)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(n_iters):
        copy_kernel[grid](x, y, n_elements, BLOCK_SIZE=block_size, dtype=tl.float32)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    total_bytes_copied = 2 * total_bytes * n_iters  # read + write
    gbps = total_bytes_copied / (end - start) / 1e9

    print(f"[Stream Copy Benchmark]")
    print(f"Tensor size: {total_bytes / (1024**2):.2f} MB, DType: {dtype}, Block size: {block_size}")
    print(f"Time per iteration: {(end - start) / n_iters:.6f} sec")
    print(f"Effective Bandwidth: {gbps:.2f} GB/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triton Stream Copy Benchmark")
    parser.add_argument("--size", type=str, required=True, help="Data size to copy (e.g., 8M, 4G)")
    parser.add_argument("--block-size", type=int, default=1024, help="Block size (e.g., 1024, 2048)")
    args = parser.parse_args()

    total_bytes = parse_size(args.size)
    benchmark_stream_copy(total_bytes=total_bytes, block_size=args.block_size)