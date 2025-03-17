#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Benchmark offline inference throughput for TRT-LLM."""

import argparse
import time
import random
import dataclasses
from typing import Optional

from transformers import AutoTokenizer
from tensorrt_llm import LLM, SamplingParams  # Adjust these imports if your TRT-LLM package differs
import os
import json
from benchmark_utils import convert_to_pytorch_benchmark_format, write_to_json

@dataclasses.dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int

def synthesize_requests(tokenizer, input_len: int, output_len: int, num_prompts: int) -> list[SampleRequest]:
    """Generates synthetic prompts with the desired token lengths."""
    requests = []
    vocab_size = tokenizer.vocab_size
    for _ in range(num_prompts):
        # Create a random candidate prompt using random token IDs.
        candidate_ids = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
        # Attempt to decode and ensure the tokenized length equals input_len.
        for _ in range(5):
            candidate_prompt = tokenizer.decode(candidate_ids)
            tokenized_len = len(tokenizer.encode(candidate_prompt))
            if tokenized_len == input_len:
                break
            diff = input_len - tokenized_len
            if diff > 0:
                candidate_ids.extend([random.randint(0, vocab_size - 1) for _ in range(diff)])
            else:
                candidate_ids = candidate_ids[:input_len]
        requests.append(SampleRequest(prompt=candidate_prompt, prompt_len=input_len, expected_output_len=output_len))
    return requests

def save_to_pytorch_benchmark_format(args: argparse.Namespace,
                                     results: dict[str, any]) -> None:
    pt_records = convert_to_pytorch_benchmark_format(
        args=args,
        metrics={
            "requests_per_second": [results["requests_per_second"]],
            "tokens_per_second": [results["tokens_per_second"]],
        },
        extra_info={
            k: results[k]
            for k in ["elapsed_time", "num_requests", "total_num_tokens", "total_output_tokens", "total_input_tokens"]
        })
    if pt_records:
        # Don't use json suffix here as we don't want CI to pick it up
        pt_file = f"{os.path.splitext(args.output_json)[0]}.pytorch.json"
        write_to_json(pt_file, pt_records)

def main():
    parser = argparse.ArgumentParser(description="Benchmark offline inference throughput for TRT-LLM.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the TRT-LLM model directory.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the tokenizer directory for TRT-LLM.")
    parser.add_argument("--input-len", type=int, required=True,
                        help="Input prompt length (in tokens) for each request.")
    parser.add_argument("--output-len", type=int, required=True,
                        help="Output length (in tokens) for each request.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size for TRT-LLM.")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model length (context length). Not used in this script but kept for consistency.")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of synthetic prompts to generate for benchmarking.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for synthetic prompt generation.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save benchmark results in JSON format.")
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Synthesize synthetic requests.
    requests = synthesize_requests(tokenizer, args.input_len, args.output_len, args.num_prompts)
    prompts = [req.prompt for req in requests]
    
    # Instantiate the TRT-LLM engine.
    llm = LLM(model=args.model, tokenizer=args.tokenizer, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
    
    # Create sampling parameters. Adjust these parameters to match your desired configuration. #detokenize False
    sampling_params = SamplingParams(temperature=1, top_p=1, ignore_eos=True, max_tokens=args.output_len, min_tokens=args.output_len, detokenize=False)
    
    # Benchmark the generation.
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    
    elapsed_time = end - start
    total_input_tokens = args.num_prompts * args.input_len
    total_output_tokens = args.num_prompts * args.output_len
    total_tokens = total_input_tokens + total_output_tokens
    
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Throughput: {args.num_prompts / elapsed_time:.2f} requests/s")
    print(f"Input tokens: {total_input_tokens / elapsed_time:.2f} tokens/s")
    print(f"Output tokens: {total_output_tokens / elapsed_time:.2f} tokens/s")
    print(f"Total tokens: {total_tokens / elapsed_time:.2f} tokens/s")

    if args.output_json:
        results = {
            "elapsed_time": elapsed_time,
            "num_requests": args.num_prompts,
            "total_num_tokens": total_tokens,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "requests_per_second": args.num_prompts / elapsed_time,
            "tokens_per_second": total_tokens / elapsed_time
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        save_to_pytorch_benchmark_format(args, results)

if __name__ == "__main__":
    main()