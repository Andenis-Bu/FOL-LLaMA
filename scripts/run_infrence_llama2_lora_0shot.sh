#!/bin/bash

python run_infrence.py \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --adapters_path "adapters/FOL-Llama-2-7b-hf" \
    --data_file "datasets/folio_parsed.json" \
    --output_file "outputs/llama2_lora_0shot.json" \
    --model_type "lora" \
    --num_shots 0 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --top_p 1.0
