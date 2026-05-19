#!/bin/bash

python run_infrence.py \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --adapters_path "adapters/FOL-Llama-3-8b" \
    --data_file "datasets/folio_parsed.json" \
    --output_file "outputs/llama3_lora_5shot.json" \
    --model_type "lora" \
    --num_shots 5 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --top_p 1.0
