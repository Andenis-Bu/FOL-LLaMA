import argparse
import json
from pathlib import Path
import torch
from utils import parse_fol_response
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

# =============================
# CLI Arguments
# =============================
parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LLaMA model")
parser.add_argument("--model_path",      type=str,   default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--adapters_path",   type=str,   default="adapters/nl2fol-lora-v2/checkpoint-6500")
parser.add_argument("--data_file",       type=str,   default="data/data_sets/folio100.json")
parser.add_argument("--output_file",     type=str,   default="outputs/folio_predictions.json")
parser.add_argument("--model_type",      type=str,   choices=["base", "lora"], default="base")
parser.add_argument("--num_shots",       type=int,   choices=[0, 5], default=0, help="Number of few-shot examples (0 or 5)")
parser.add_argument("--max_new_tokens",  type=int,   default=256)
parser.add_argument("--temperature",     type=float, default=0.0)
parser.add_argument("--top_p",           type=float, default=1.0)
args = parser.parse_args()

# =============================
# Configuration
# =============================
MODEL_PATH = args.model_path
ADAPTERS_PATH = args.adapters_path

DATA_FILE = args.data_file
OUTPUT_FILE = args.output_file

MODEL_TYPE = args.model_type
NUM_SHOTS = args.num_shots
MAX_NEW_TOKENS = args.max_new_tokens
TEMPERATURE = args.temperature
TOP_P = args.top_p

# =============================
# Few-shot examples
# =============================
FEW_SHOT_EXAMPLES = [
    {
        "NL": "No one playing for Nautico is Brazilian.",
        "FOL": "\u2200x (PlaysFor(x, nautico) \u2192 \u00acBrazilian(x))"
    },
    {
        "NL": "Ailton Silva does not play for a football club.",
        "FOL": "\u2200x (FootballClub(x) \u2192 \u00acPlaysFor(ailtonsilva, x))"
    },
    {
        "NL": "Ailton was not loaned out to a football club.",
        "FOL": "\u2200x (FootballClub(x) \u2192 \u00acLoanedTo(ailton, x))"
    },
    {
        "NL": "Ailton Silva played for Fluminense.",
        "FOL": "PlaysFor(ailtonsilva, fluminense)"
    },
    {
        "NL": "Ailton Silva was loaned out to a football club.",
        "FOL": "\u2203x (FootballClub(x) \u2227 LoanedTo(ailtonsilva, x))"
    },
]

def _format_few_shot_block(examples):
    lines = []
    for ex in examples:
        lines.append(f"NL: {ex['NL']}")
        lines.append(f"FOL: {ex['FOL']}")
        lines.append("")
    return "\n".join(lines)

# =============================
# Prompt builders
# =============================
def build_prompt_base(nl_sentence: str, num_shots: int = 0) -> str:
    few_shot_section = ""
    if num_shots > 0:
        few_shot_section = (
            "Examples:\n"
            + _format_few_shot_block(FEW_SHOT_EXAMPLES[:num_shots])
            + "Now translate the following sentence:\n\n"
        )
    return (
        "You are a formal logic translator.\n"
        "Translate the following natural language sentence into first-order logic.\n\n"
        + few_shot_section
        + f"Sentence:\n\"{nl_sentence}\"\n\n"
        "Constraints:\n"
        "- Use first-order logic with quantifiers ∀, ∃\n"
        "- Use only predicates provided in the sentence\n"
        "- Do NOT explain your answer\n"
        "- Output ONLY a valid FOL formula\n\n"
        "FOL:"
    )

def build_prompt_trained(nl: str, num_shots: int = 0) -> str:
    few_shot_section = ""
    if num_shots > 0:
        few_shot_section = (
            "Here are some examples:\n\n"
            + _format_few_shot_block(FEW_SHOT_EXAMPLES[:num_shots])
        )
    return (
        "[INST] Translate the following natural language sentence into first-order logic.\n\n"
        + few_shot_section
        + f"NL:\n{nl}\n"
        "[/INST]\n"
    )

# =============================
# Load data early (fail fast)
# =============================
data_path = Path(DATA_FILE)
if not data_path.exists():
    candidates = sorted(str(p) for p in Path("data/data_sets").rglob("*.json"))
    raise FileNotFoundError(
        f"Dataset file not found: {DATA_FILE}\n"
        f"Current working directory: {Path.cwd()}\n"
        f"Available JSON files under data/data_sets/:\n- "
        + "\n- ".join(candidates)
    )

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f) 

# =============================
# Load model
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

if MODEL_TYPE == "base":
    print("Using base model")
    model = base_model
else:
    print("Using pretrained model")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTERS_PATH
    )

model.eval()

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

# =============================
# Inference loop
# =============================
Path("outputs").mkdir(exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    fout.write("[\n")
    for idx, ex in enumerate(tqdm(data, desc="Inference"), start=1):
        sentence = ex["NL"]

        if MODEL_TYPE == "base":
            prompt = build_prompt_base(sentence, NUM_SHOTS)
        else:
            prompt = build_prompt_trained(sentence, NUM_SHOTS)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=False,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

        pred_fol = decoded.strip()

        if MODEL_TYPE == "base":
            final_fol = parse_fol_response(pred_fol)
        else:
            final_fol = pred_fol

        entry = {"NL": sentence, "FOL": final_fol}
        suffix = ",\n" if idx < len(data) else "\n"
        fout.write("    " + json.dumps(entry, ensure_ascii=True) + suffix)
        fout.flush()

    fout.write("]\n")

print("Done. Results written to", OUTPUT_FILE)
