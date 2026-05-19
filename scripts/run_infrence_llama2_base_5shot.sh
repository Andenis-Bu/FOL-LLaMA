#!/bin/bash

python run_infrence.py \
    --model_path "meta-llama/Llama-2-7b-hf" \
    --data_file "datasets/folio_parsed.json" \
    --output_file "outputs/llama2_base_5shot.json" \
    --model_type "base" \
    --num_shots 5 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --top_p 1.0
