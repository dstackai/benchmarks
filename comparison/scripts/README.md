## Generate charts
```
python mean_tpot_ms.py --input-file ../csv_data/qps_comparison.csv --output-file ../images/mean_tpot_ms_vs_qps_comparison_gpus.png
python mean_ttft_ms.py --input-file ../csv_data/qps_comparison.csv --output-file ../images/mean_ttft_vs_qps_comparison_gpus.png
python request_completed.py --input-file ../csv_data/qps_comparison.csv --output-file ../images/completed_requests_vs_qps_comparison_gpus
python request_throughput.py --input-file ../csv_data/qps_comparison.csv --output-file ../images/request_throughput_vs_qps_comparison_gpus.png
python total_token_throughput.py --input-file ../csv_data/qps_comparison.csv --output-file ../images/total_token_throughput_vs_qps_comparison_gpus.png


python token_vs_batch_ip128_op2048.py --input-file ../csv_data/throughput_comparsion.csv --output-file ../images/tokens_per_second_vs_batch_size_128_2048_comparison_gpus.png
python token_vs_batch_ip2048_op2048.py --input-file ../csv_data/throughput_comparsion.csv --output-file ../images/tokens_per_second_vs_batch_size_2048_2048_comparison_gpus.png
python token_vs_batch_ip32784_op2048.py --input-file ../csv_data/throughput_comparsion.csv --output-file ../images/tokens_per_second_vs_batch_size_32784_2048_comparison_gpus.png

python throughput_latency_curve_128_2048.py --input-file ../csv_data/throughput_latency_curve_128_2048.csv --output-file ../images/tokens_per_second_vs_end_to_end_latency_comparison_gpus_128_2048.png
python throughput_latency_curve_2048_2048.py --input-file ../csv_data/throughput_latency_curve_2048_2048.csv --output-file ../images/tokens_per_second_vs_end_to_end_latency_comparison_gpus.png
python throughput_latency_curve_32784_2048.py --input-file ../csv_data/throughput_latency_curve_32784_2048.csv --output-file ../images/tokens_per_second_vs_end_to_end_latency_comparison_gpus_32784_2048.png

``` 
