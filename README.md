# FOL-LLaMA

Fine-tuning LLaMA 2 (7B) and LLaMA 3 (8B) with QLoRA for translating natural language (NL) sentences into first-order logic (FOL) formulas.

The models are trained on the [MALLS-v0.1](https://huggingface.co/datasets/yuan-yang/MALLS-v0) dataset (27K auto-verified NL–FOL pairs) and evaluated on the [FOLIO](https://github.com/Yale-LILY/FOLIO) benchmark using two custom metrics: syntactic validity (SynV) and bounded semantic check (BSC) via Z3.

---

## Install

1. Prepare environment
```bash
conda create -n fol-llama python=3.10 -y
conda activate fol-llama

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install transformers accelerate sentencepiece
pip install -U bitsandbytes
pip install datasets
pip install peft

pip install huggingface_hub
huggingface-cli login
```

2. Download datasets
```bash
sh data_download.sh
```

3. *(Optional)* Install Z3 for the BSC evaluation metric
```bash
pip install z3-solver
```

---

## Pre-trained Adapters

| Model | Base | HuggingFace |
|---|---|---|
| FOL-LLaMA 2 | `meta-llama/Llama-2-7b-hf` | [Andenis-Bu/FOL-Llama-2-7b-hf](https://huggingface.co/Andenis-Bu/FOL-Llama-2-7b-hf) |
| FOL-LLaMA 3 | `meta-llama/Meta-Llama-3-8B` | [Andenis-Bu/FOL-Llama-3-8b](https://huggingface.co/Andenis-Bu/FOL-Llama-3-8b) |

---

## Dataset

The input data follows a simple JSON format:
```json
[
    {
        "NL": "All people who regularly drink coffee are dependent on caffeine.",
        "FOL": "∀x (RegularlyDrinksCoffee(x) → DependentOnCaffeine(x))"
    }
]
```

- **Training**: [MALLS-v0.1](https://huggingface.co/datasets/yuan-yang/MALLS-v0/blob/main/MALLS-v0.1-train.json) (27K pairs)
- **Evaluation**: [FOLIO parsed](https://huggingface.co/datasets/Andenis-Bu/FOL-LLaMA-Inference/blob/main/folio_parsed.json) (1K pairs)

---

## Training

Fine-tune a LLaMA model on MALLS with QLoRA:

```bash
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
```

Or use the provided scripts:
```bash
sh scripts/run_training_llama2.sh
sh scripts/run_training_llama3.sh
```

---

## Inference

### Base model

```bash
python run_infrence.py \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --data_file "datasets/folio_parsed.json" \
    --output_file "outputs/llama3_base_0shot.json" \
    --model_type "base" \
    --num_shots 0
```

Or use the provided scripts:
```bash
sh scripts/run_infrence_llama2_base_0shot.sh
sh scripts/run_infrence_llama3_base_0shot.sh
sh scripts/run_infrence_llama2_base_5shot.sh
sh scripts/run_infrence_llama3_base_5shot.sh
```

### Fine-tuned model with LoRA adapter

```bash
python run_infrence.py \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --adapters_path "adapters/FOL-Llama-3-8b" \
    --data_file "datasets/folio_parsed.json" \
    --output_file "outputs/llama3_lora_5shot.json" \
    --model_type "pretrained" \
    --num_shots 0
```

Or use the provided scripts:
```bash
sh scripts/run_infrence_llama2_lora_0shot.sh
sh scripts/run_infrence_llama3_lora_0shot.sh
sh scripts/run_infrence_llama2_lora_5shot.sh
sh scripts/run_infrence_llama3_lora_5shot.sh
```

All experiment configurations are available as shell scripts in `scripts/`.

---

## Evaluation

Evaluate predictions against gold FOL annotations:

```bash
python eval_fol.py \
    --gold "datasets/folio_parsed.json" \
    --pred "outputs/llama3_lora_5shot.json" \
    --max-domain 4
```

---


## Acknowledgements

This project builds upon the following work:

- [LogicLLaMA](https://github.com/gblackout/LogicLLaMA) — Yang et al., *Harnessing the Power of Large Language Models for Natural Language to First-Order Logic Translation*, 2023. The MALLS dataset used for Fine-tuning.
- [FOLIO](https://github.com/Yale-LILY/FOLIO) — The FOLIO dataset used for evaluation benchmark.
- [Hugging Face PEFT](https://github.com/huggingface/peft) — LoRA / QLoRA adapter training.

---

## Citation

If you use this code, adapters, or results, please cite:

```bibtex
@thesis{bukauskas2026folllama,
    title   = {Possibilities of translating natural language into the language of predicate logic using large language models},
    author  = {Andenis Bukauskas},
    school  = {Vilnius University},
    year    = {2026},
}
```

---

## License

Code in this repository is released under the [Apache License 2.0](LICENSE).

The LoRA adapters are additionally subject to the base model licenses:
- LLaMA 2 adapters: [Meta LLaMA 2 Community License](https://ai.meta.com/llama/license/)
- LLaMA 3 adapters: [Meta LLaMA 3 Community License](https://llama.meta.com/llama3/license/)

The MALLS dataset is CC BY NC 4.0 and subject to the [OpenAI Terms of Use](https://openai.com/policies/terms-of-use).