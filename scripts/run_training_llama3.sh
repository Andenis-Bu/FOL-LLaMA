#!/bin/bash

python run_training.py \
    --model_id "meta-llama/Meta-Llama-3-8B" \
    --data_file "datasets/MALLS-v0.1-train.json" \
    --output_dir "adapters/FOL-Llama-3-8b" \
    --log_file "logs/llama3_training_log.txt" \
    --max_length 1024 \
    --batch_size 1 \
    --grad_accum 8 \
    --epochs 10 \
    --lr 2e-5 \
    --val_split 0.04 \
    --seed 42 
