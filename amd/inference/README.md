# Benchmarking TGI vs vLLM on AMD 8 X MI300x: Performance Insights with LlaMA 3.1 405B Model

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

## How to Validate
To validate the tests, you can conduct the tests using [benchmark_serving](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py) provided by
`vLLM`. The script provides instructions on both `TGI` and `vLLM`.


