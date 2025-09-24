#!/bin/bash
# save this as run_all.sh
# make it executable: chmod +x run_all.sh

concurrency="128 64 32"

# Define the ISL-OSL combinations
combinations=(
  "2048 2048"
  "128 2048"
  "2048 128"
)

for combo in "${combinations[@]}"; do
  isl=$(echo $combo | awk '{print $1}')
  osl=$(echo $combo | awk '{print $2}')
  
  echo "Running benchmark for ISL=$isl, OSL=$osl"
  ./run_benchmark.sh "$concurrency" "$isl" "$osl" 2 2
done