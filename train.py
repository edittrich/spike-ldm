"""
Fine-Tuning Pipeline: Flat Interfaces → Logical Data Model
============================================================
Data source : HuggingFace richardr1126/spider-schema
Strategy    : QLoRA 4-bit (16 GB VRAM)
Model       : Qwen/Qwen2.5-7B-Instruct

Usage:
    deepspeed --num_gpus=1 train.py

Requirements:
    uv add torch --index-url https://download.pytorch.org/whl/cu128
    uv add transformers datasets peft trl accelerate sentencepiece protobuf python-dotenv bitsandbytes
"""

import os

# Must be set before importing torch to reduce memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# ─────────────────────────────────────────────────────────────────────────────
# 0. Environment & Authentication
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
token = os.getenv("HF_TOKEN")

torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Spider Schemas from HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
# The dataset contains 166 database schemas with the following columns:
#   db_id                  → database name
#   Schema (values (type)) → "Table1 : col1 (type) , col2 (type) | Table2 ..."
#   Primary Keys           → "col1 , col2"
#   Foreign Keys           → "Table1.col1 = Table2.col2"
# ─────────────────────────────────────────────────────────────────────────────

print("Loading Spider schemas from HuggingFace...")
spider_ds = load_dataset(
    "richardr1126/spider-schema",
    split="train",
    token=token,
)
print(f"  Databases loaded : {len(spider_ds)}")
print(f"  Columns          : {spider_ds.column_names}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Schema Parsing Helpers
# ─────────────────────────────────────────────────────────────────────────────


def parse_schema_text(schema_text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the raw schema string into a structured dict.

    Input : "stadium : Stadium_ID (number) , Name (text) | singer : Singer_ID (number)"
    Output: {"stadium": [("Stadium_ID", "number"), ("Name", "text")],
             "singer":  [("Singer_ID", "number")]}
    """
    tables: dict[str, list[tuple[str, str]]] = {}
    for table_block in schema_text.split("|"):
        table_block = table_block.strip()
        if ":" not in table_block:
            continue
        table_name, cols_text = table_block.split(":", 1)
        table_name = table_name.strip()
        cols: list[tuple[str, str]] = []
        for col in cols_text.split(","):
            col = col.strip()
            if not col:
                continue
            if "(" in col and ")" in col:
                col_name = col[: col.index("(")].strip()
                col_type = col[col.index("(") + 1 : col.index(")")].strip()
            else:
                col_name = col
                col_type = "text"
            cols.append((col_name, col_type))
        tables[table_name] = cols
    return tables


def schema_to_flat_input(row: dict) -> str:
    """
    De-normalize a Spider schema into a flat interface description.
    All columns from all tables are merged into a single field list —
    this simulates the kind of flat API / CSV input the model will see
    at inference time.

    Example output:
        Interface: concert_singer
        Fields: Stadium_ID, Location, Name, Capacity, Singer_ID, Country, ...
    """
    db_id = row["db_id"]
    tables = parse_schema_text(row["Schema (values (type))"])
    all_columns = [col_name for cols in tables.values() for col_name, _ in cols]
    return f"Interface: {db_id}\nFields: {', '.join(all_columns)}"


def schema_to_normalized_output(row: dict) -> str:
    """
    Build the ground-truth normalized logical data model as JSON.

    The output contains:
      - entities : list of tables with typed attributes (PK / FK annotated)
      - relations: list of N:1 foreign-key relationships between tables

    Example output (pretty-printed JSON):
        {
          "entities": [
            {"name": "stadium", "attributes": ["Stadium_ID: number (PK)", ...]},
            ...
          ],
          "relations": [
            {"from": "concert", "to": "stadium", "type": "N:1"},
            ...
          ]
        }
    """
    tables = parse_schema_text(row["Schema (values (type))"])
    pk_text = row["Primary Keys"]
    fk_text = row["Foreign Keys"]

    # Build a lowercase set of primary key column names for fast lookup
    primary_keys: set[str] = set()
    if pk_text and pk_text.strip():
        for pk in pk_text.split(","):
            primary_keys.add(pk.strip().lower())

    # Parse foreign key pairs: [(from_table.col, to_table.col), ...]
    fk_pairs: list[tuple[str, str]] = []
    if fk_text and fk_text.strip():
        for fk in fk_text.split(","):
            fk = fk.strip()
            if "=" in fk:
                left, right = fk.split("=")
                fk_pairs.append((left.strip(), right.strip()))

    # Build a lowercase set of FK column references for fast lookup
    fk_cols: set[str] = {left.lower() for left, _ in fk_pairs if "." in left}

    # Build entity list with annotated attributes
    entities = []
    for table_name, cols in tables.items():
        attributes = []
        for col_name, col_type in cols:
            full_ref = f"{table_name}.{col_name}".lower()
            if col_name.lower() in primary_keys:
                annotation = " (PK)"
            elif full_ref in fk_cols:
                annotation = " (FK)"
            else:
                annotation = ""
            attributes.append(f"{col_name}: {col_type}{annotation}")
        entities.append({"name": table_name, "attributes": attributes})

    # Build relation list from FK pairs
    relations = [
        {"from": left.split(".")[0], "to": right.split(".")[0], "type": "N:1"}
        for left, right in fk_pairs
        if "." in left and "." in right
    ]

    return json.dumps(
        {"entities": entities, "relations": relations},
        indent=2,
        ensure_ascii=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Build Training Pairs
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding training pairs...")
training_pairs: list[dict] = []

for row in spider_ds:
    try:
        training_pairs.append(
            {
                "input": schema_to_flat_input(row),
                "output": schema_to_normalized_output(row),
            }
        )
    except Exception as exc:
        print(f"  Skipping '{row.get('db_id', '?')}': {exc}")

print(f"  Training pairs created: {len(training_pairs)}")
print("\n--- Sample input ---")
print(training_pairs[0]["input"])
print("\n--- Sample output ---")
print(training_pairs[0]["output"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dataset Split
# ─────────────────────────────────────────────────────────────────────────────

hf_dataset = Dataset.from_list(training_pairs)
hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)

print(f"\n  Train size : {len(hf_dataset['train'])}")
print(f"  Test size  : {len(hf_dataset['test'])}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Prompt Formatting (ChatML)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a database architect. "
    "Analyze flat interface descriptions and produce normalized logical data "
    "models as JSON. The JSON must contain 'entities' (with typed attributes "
    "annotated as PK or FK where applicable) and 'relations' "
    "(with from, to, and type fields)."
)


def format_for_training(row: dict) -> dict:
    """Wrap each training pair in the ChatML prompt template."""
    return {
        "text": (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n{row['input']}\n"
            f"<|assistant|>\n{row['output']}"
        )
    }


hf_dataset = hf_dataset.map(format_for_training)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model & Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
# QLoRA: 4-bit quantization via bitsandbytes reduces the 7B model from
# ~14 GB (bfloat16) to ~4.5 GB, leaving enough headroom for gradients
# and activations on a 16 GB VRAM GPU.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"\nLoading model: {MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # nested quantization saves ~0.4 GB
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, token=token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=token,
)
model.config.use_cache = False  # required for gradient checkpointing
model.config.pad_token_id = tokenizer.pad_token_id


# ─────────────────────────────────────────────────────────────────────────────
# 7. LoRA Configuration
# ─────────────────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Training
# ─────────────────────────────────────────────────────────────────────────────
# Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
#                      = 1 × 16 = 16
# ─────────────────────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir="./datamodel-lora",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch size = 16
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",  # set to "wandb" for experiment tracking
    dataset_text_field="text",
    max_length=512,
    gradient_checkpointing=True,  # trades compute for memory savings
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset["train"],
    eval_dataset=hf_dataset["test"],
    processing_class=tokenizer,
)

print("\nStarting training...")
trainer.train()

# Save the LoRA adapter and tokenizer
trainer.model.save_pretrained("./datamodel-lora/final")
tokenizer.save_pretrained("./datamodel-lora/final")
print("Model saved to: ./datamodel-lora/final")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Inference Test
# ─────────────────────────────────────────────────────────────────────────────
# For standalone inference (without retraining), load the adapter like this:
#
#   from peft import PeftModel
#   base  = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
#   model = PeftModel.from_pretrained(base, "./datamodel-lora/final")
# ─────────────────────────────────────────────────────────────────────────────


def predict(flat_input: str) -> str:
    """Run inference on a flat interface description and return the data model JSON."""
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{flat_input}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("<|assistant|>")[-1].strip()


test_input = (
    "Interface: webshop\n"
    "Fields: order_id, customer_name, customer_email, "
    "product_name, product_price, quantity, order_date, "
    "shipping_address, payment_method"
)

print("\n--- Inference Test ---")
print("Input :", test_input)
print("Output:", predict(test_input))
