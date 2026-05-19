import argparse
import json
import os
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from transformers import TrainerCallback
from peft import LoraConfig, get_peft_model

# =============================
# CLI Arguments
# =============================
parser = argparse.ArgumentParser(description="Fine-tune a LLaMA model with QLoRA")
parser.add_argument("--model_id",    type=str,   default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--data_file",   type=str,   default="data/data_sets/folio1000.json")
parser.add_argument("--output_dir",  type=str,   default="adapters/llama2-test")
parser.add_argument("--log_file",    type=str,   default=None, help="Path to write training logs (default: <output_dir>/training_log.txt)")
parser.add_argument("--max_length",  type=int,   default=1024)
parser.add_argument("--batch_size",  type=int,   default=1)
parser.add_argument("--grad_accum",  type=int,   default=8)
parser.add_argument("--epochs",      type=int,   default=10,    help="Upper bound; early stopping will stop earlier")
parser.add_argument("--lr",          type=float, default=2e-5)
parser.add_argument("--val_split",   type=float, default=0.04)
parser.add_argument("--seed",        type=int,   default=42)
args = parser.parse_args()

# =============================
# Configuration
# =============================
MODEL_ID = args.model_id
DATA_FILE = args.data_file
OUTPUT_DIR = args.output_dir
LOG_FILE = args.log_file or os.path.join(OUTPUT_DIR, "training_log.txt")

MAX_LENGTH = args.max_length
BATCH_SIZE = args.batch_size
GRAD_ACCUM = args.grad_accum
EPOCHS = args.epochs
LR = args.lr
VAL_SPLIT = args.val_split
SEED = args.seed

# =============================
# File logging callback
# =============================
class FileLoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(self.log_path, "a") as f:
            f.write(json.dumps(logs) + "\n")

# =============================
# Prompt (training = inference)
# =============================
def build_prompt(nl: str) -> str:
    return (
        "[INST] Translate the following natural language sentence into first-order logic.\n\n"
        f"NL:\n{nl}\n"
        "[/INST]\n"
    )

# =============================
# Load dataset
# =============================
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

assert isinstance(raw_data, list)

def format_example(ex):
    prompt = build_prompt(ex["NL"])
    text = prompt + ex["FOL"] + tokenizer.eos_token
    return {
        "prompt": prompt,
        "text": text,
    }

# =============================
# Tokenizer
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# =============================
# Build dataset
# =============================
dataset = Dataset.from_list(raw_data)
dataset = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)

train_ds = dataset["train"].map(format_example)
val_ds   = dataset["test"].map(format_example)

# =============================
# Tokenization with prompt masking
# =============================
def tokenize(batch):
    full = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    prompt = tokenizer(
        batch["prompt"],
        truncation=True,
        max_length=MAX_LENGTH,
    )

    labels = full["input_ids"].copy()
    prompt_len = len(prompt["input_ids"])

    # Mask prompt tokens
    labels[:prompt_len] = [-100] * prompt_len

    full["labels"] = labels
    return full

train_ds = train_ds.map(tokenize, remove_columns=train_ds.column_names)
val_ds   = val_ds.map(tokenize, remove_columns=val_ds.column_names)

# =============================
# Data collator (dynamic padding)
# =============================
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
)

# =============================
# Model (QLoRA)
# =============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# =============================
# Training arguments
# =============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,

    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    logging_steps=20,
    report_to="none",
    seed=SEED,
)

# =============================
# Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        FileLoggingCallback(LOG_FILE),
    ],
)

# =============================
# Train
# =============================
trainer.train()

# =============================
# Save adapter + tokenizer
# =============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete. Best adapter saved to:", OUTPUT_DIR)
print("Training log saved to:", LOG_FILE)