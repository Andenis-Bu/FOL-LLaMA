#!/bin/bash

python eval_fol.py \
    --gold "datasets/folio_parsed.json" \
    --pred "outputs/llama3_lora_5shot.json" \
    --max-domain 4
