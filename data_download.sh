#!/bin/bash

# Create target directory if it doesn't exist
mkdir -p datasets

# Download the dataset
curl -L \
  https://huggingface.co/datasets/yuan-yang/MALLS-v0/resolve/main/MALLS-v0.1-train.json \
  -o datasets/MALLS-v0.1-train.json

curl -L \
  https://huggingface.co/datasets/Andenis-Bu/FOL-LLaMA-Inference/resolve/main/folio_parsed.json \
  -o datasets/folio_parsed.json