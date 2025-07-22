#!/bin/bash
apt-get update && apt-get install -y git cmake && \
pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4 && \
pip install transformers peft wandb && \
git clone https://github.com/huggingface/trl && \
cd trl && \
pip install .
python3 trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eos_token '<|im_end|>' \
    --eval_strategy steps \
    --eval_steps 100 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16