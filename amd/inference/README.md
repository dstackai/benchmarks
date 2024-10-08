# Benchmarking TGI vs vLLM on AMD 8 X MI300x: Performance Insights with LlaMA 3.1 405B Model

## Introduction
In this benchmarking analysis, we compare the performance of two popular inference backends, `TGI` and `vLLM`, using the `LLaMA 3.1 405B` model on the `8 X AMD MI300x`. 

## Benchmark Setup
To validate the tests, you can conduct the tests using [benchmark_serving](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) provided by
`vLLM`. The script provides instructions on both `TGI` and `vLLM`.

### vLLM
```
PyTorch version: 2.4.1+rocm6.1
Is debug build: False
CUDA used to build PyTorch: N/A
ROCM used to build PyTorch: 6.1.40091-a8dbc0c19

OS: Ubuntu 22.04.4 LTS (x86_64)
GCC version: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version: 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.1.0 24103 7db7f5e49612030319346f900c08f474b1f9023a)
CMake version: version 3.26.4
Libc version: glibc-2.35

Python version: 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0] (64-bit runtime)
Python platform: Linux-6.8.0-45-generic-x86_64-with-glibc2.35
Is CUDA available: True
CUDA runtime version: Could not collect
CUDA_MODULE_LOADING set to: LAZY
GPU models and configuration: AMD Instinct MI300X (gfx942:sramecc+:xnack-)
Nvidia driver version: Could not collect
cuDNN version: Could not collect
HIP runtime version: 6.1.40093
MIOpen runtime version: 3.1.0
Is XNNPACK available: True

Versions of relevant libraries:
[pip3] mypy==1.4.1
[pip3] mypy-extensions==1.0.0
[pip3] numpy==1.26.4
[pip3] pytorch-triton-rocm==3.0.0
[pip3] pyzmq==24.0.1
[pip3] torch==2.4.1+rocm6.1
[pip3] torchaudio==2.4.1+rocm6.1
[pip3] torchvision==0.16.1+fdea156
[pip3] transformers==4.45.1
[pip3] triton==3.0.0
[conda] No relevant packages
ROCM Version: 6.1.40091-a8dbc0c19
Neuron SDK Version: N/A
vLLM Version: 0.6.3.dev116+g151ef4ef
vLLM Build Flags:
CUDA Archs: Not Set; ROCm: Disabled; Neuron: Disabled
```



#### Steps
1. Run vLLM server: 
   `ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve meta-llama/Llama-3.1-405B-Instruct 
                                                             --tensor-parallel-size=8 --disable-log-requests 
                                                             --disable-frontend-multiprocessing`

2. To create larger prompt sequence lengths, the text in `sonnet.txt` is repeated in the file. Also, default `--sonnet-prefix-len` is set to 50

3. Run test:
    `python benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-405B-Instruct 
                                 --dataset-name sonnet  --num-prompt=<Batch Size> --dataset-path="sonnet.txt" --sonnet-input-len <sequence length>`


### TGI
TGI Docker Image:`ghcr.io/huggingface/text-generation-inference:sha-11d7af7-rocm`

#### Steps
1. Run TGI server:
   `text-generation-launcher --port 8000 --num-shard 8 --sharded true --max-concurrent-requests 8192 --max-total-tokens 130000 --max-input-tokens 125000`
2. To create larger prompt sequence lengths, the text in `sonnet.txt` is repeated in the file. Also, default `--sonnet-prefix-len` is set to 50
3. Run test:
   `python benchmark_serving.py --backend tgi --model meta-llama/Llama-3.1-405B-Instruct --dataset-name sonnet  
                                --sonnet-input-len <sequence lenght>   --endpoint /generate_stream --dataset-path="sonnet.txt" --num-prompt=<Batch Size>`


## Results
* TGI outperforms vLLM across all batch sizes in terms of token throughput. The performance gap increases
as the batch size increases. For batches larger than 64, there is significant difference in performance. The sequence
lengths of prompts are kept constant at 80 tokens per prompt.
![Chart1](charts_short_seq/throughput_tgi_vllm.png)
* TGI outperforms vLLM in `Time To First Token` across all batch sizes except batch size 2 & 32. Here too the performance
gap is significant at larger batches.
![Chart2](charts_short_seq/ttft_mean_tgi_vllm.png)
* To check the performance in larger prompt sizes we conducted the tests at 10000 tokens per prompt. Here too, 
in terms of token throughput and `TTFT`, `TGI` outperformed `vLLM` significantly.
![Chart3](charts_long_seq/throughput_tgi_vllm.png)
![Chart4](charts_long_seq/mean_ttft_tgi_vllm.png)
* Finally, we also performed the tests with single prompt with sequence length starting from 16000 to 128000 tokens(Max allowed by Llama 3.1).
This too yielded results in favor of TGI.
![Chart5](charts_single_seq/throughput_tgi_vllm.png)
* Another noticeable metric is VRAM consumption with `TGI` and `vLLM`
Below is the `rocm-smi` outputs of TGI & vLLM after they load weights. Notice VRAM consumed is `68%` with TGI while vLLM
consumes `95%`. This might be the reason in significant performance difference.

### TGI (rocm-smi)
```============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                  
==========================================================================================================================
0       2     0x74a1,   55354  47.0°C      139.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
1       3     0x74a1,   41632  40.0°C      135.0W    NPS1, SPX, 0        131Mhz  900Mhz  0%   auto  750.0W  68%    0%    
2       4     0x74a1,   47045  44.0°C      136.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
3       5     0x74a1,   60169  48.0°C      143.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
4       6     0x74a1,   56024  46.0°C      139.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
5       7     0x74a1,   705    42.0°C      136.0W    NPS1, SPX, 0        131Mhz  900Mhz  0%   auto  750.0W  68%    0%    
6       8     0x74a1,   59108  51.0°C      144.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
7       9     0x74a1,   10985  44.0°C      138.0W    NPS1, SPX, 0        132Mhz  900Mhz  0%   auto  750.0W  68%    0%    
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
```
### vLLM (rocm-smi)

```========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)                                                  
====================================================================================================================
0       [0x74a1 : 0x00]       47.0°C      139.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   97%   0%    
        AMD Instinct MI300X                                                                                         
1       [0x74a1 : 0x00]       39.0°C      135.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
2       [0x74a1 : 0x00]       44.0°C      136.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
3       [0x74a1 : 0x00]       48.0°C      143.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
4       [0x74a1 : 0x00]       46.0°C      138.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
5       [0x74a1 : 0x00]       41.0°C      137.0W    NPS1, SPX       131Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
6       [0x74a1 : 0x00]       51.0°C      143.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
7       [0x74a1 : 0x00]       43.0°C      137.0W    NPS1, SPX       132Mhz  900Mhz  0%   auto  750.0W   95%   0%    
        AMD Instinct MI300X                                                                                         
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
```


