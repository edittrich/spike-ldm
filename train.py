"""
Fine-Tuning Pipeline: Flat Interfaces → Logical Data Model
============================================================
Data source 1: HuggingFace richardr1126/spider-schema (single flat interface)
Data source 2: Synthetic multi-interface pairs with explicit FK references
Strategy     : QLoRA 4-bit (16 GB VRAM)
Model        : Qwen/Qwen2.5-7B-Instruct

Usage:
    uv run python train.py

Requirements:
    uv add torch --index-url https://download.pytorch.org/whl/cu130
    uv add transformers datasets peft trl accelerate sentencepiece protobuf python-dotenv bitsandbytes
"""

import os

# Must be set before importing torch to reduce memory fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import torch
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from data_wikisql import load_wikisql_pairs
from data_atis import build_atis_pairs


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
# 2. Spider Schema Parsing Helpers
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
    De-normalize a Spider schema into a single flat interface description.
    All columns from all tables are merged into one field list.

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
    """
    tables = parse_schema_text(row["Schema (values (type))"])
    pk_text = row["Primary Keys"]
    fk_text = row["Foreign Keys"]

    primary_keys: set[str] = set()
    if pk_text and pk_text.strip():
        for pk in pk_text.split(","):
            primary_keys.add(pk.strip().lower())

    fk_pairs: list[tuple[str, str]] = []
    if fk_text and fk_text.strip():
        for fk in fk_text.split(","):
            fk = fk.strip()
            if "=" in fk:
                left, right = fk.split("=")
                fk_pairs.append((left.strip(), right.strip()))

    fk_cols: set[str] = {left.lower() for left, _ in fk_pairs if "." in left}

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
# 3. Synthetic Multi-Interface Training Data
# ─────────────────────────────────────────────────────────────────────────────
# These examples teach the model to:
#   - Accept multiple named interfaces as input
#   - Recognize FK references by field name matching table names
#   - Build correct N:1 relations across interfaces
# ─────────────────────────────────────────────────────────────────────────────

# Each entry is a tuple of (input_text, output_json_dict)
# The input uses the multi-interface format the user wants to provide.
MULTI_INTERFACE_EXAMPLES: list[tuple[str, dict]] = [

    # ── E-Commerce ────────────────────────────────────────────────────────────
    (
        "Interface: product\nFields: id, name, price\n"
        "Interface: customer\nFields: id, name, email\n"
        "Interface: order\nFields: id, product, customer, quantity, date",
        {
            "entities": [
                {"name": "product",  "attributes": ["id: number (PK)", "name: text", "price: number"]},
                {"name": "customer", "attributes": ["id: number (PK)", "name: text", "email: text"]},
                {"name": "order",    "attributes": ["id: number (PK)", "product: number (FK)",
                                                    "customer: number (FK)", "quantity: number", "date: text"]},
            ],
            "relations": [
                {"from": "order", "to": "product",  "type": "N:1"},
                {"from": "order", "to": "customer", "type": "N:1"},
            ],
        },
    ),

    # ── Blog Platform ─────────────────────────────────────────────────────────
    (
        "Interface: author\nFields: id, name, email\n"
        "Interface: category\nFields: id, name\n"
        "Interface: post\nFields: id, title, body, author, category, created_at",
        {
            "entities": [
                {"name": "author",   "attributes": ["id: number (PK)", "name: text", "email: text"]},
                {"name": "category", "attributes": ["id: number (PK)", "name: text"]},
                {"name": "post",     "attributes": ["id: number (PK)", "title: text", "body: text",
                                                    "author: number (FK)", "category: number (FK)",
                                                    "created_at: text"]},
            ],
            "relations": [
                {"from": "post", "to": "author",   "type": "N:1"},
                {"from": "post", "to": "category", "type": "N:1"},
            ],
        },
    ),

    # ── Hospital ──────────────────────────────────────────────────────────────
    (
        "Interface: doctor\nFields: id, name, specialty\n"
        "Interface: patient\nFields: id, name, birthdate\n"
        "Interface: appointment\nFields: id, doctor, patient, date, diagnosis",
        {
            "entities": [
                {"name": "doctor",      "attributes": ["id: number (PK)", "name: text", "specialty: text"]},
                {"name": "patient",     "attributes": ["id: number (PK)", "name: text", "birthdate: text"]},
                {"name": "appointment", "attributes": ["id: number (PK)", "doctor: number (FK)",
                                                       "patient: number (FK)", "date: text", "diagnosis: text"]},
            ],
            "relations": [
                {"from": "appointment", "to": "doctor",  "type": "N:1"},
                {"from": "appointment", "to": "patient", "type": "N:1"},
            ],
        },
    ),

    # ── School ────────────────────────────────────────────────────────────────
    (
        "Interface: student\nFields: id, name, birthdate\n"
        "Interface: teacher\nFields: id, name, subject\n"
        "Interface: course\nFields: id, name, teacher\n"
        "Interface: enrollment\nFields: id, student, course, grade, semester",
        {
            "entities": [
                {"name": "student",    "attributes": ["id: number (PK)", "name: text", "birthdate: text"]},
                {"name": "teacher",    "attributes": ["id: number (PK)", "name: text", "subject: text"]},
                {"name": "course",     "attributes": ["id: number (PK)", "name: text", "teacher: number (FK)"]},
                {"name": "enrollment", "attributes": ["id: number (PK)", "student: number (FK)",
                                                      "course: number (FK)", "grade: text", "semester: text"]},
            ],
            "relations": [
                {"from": "course",      "to": "teacher", "type": "N:1"},
                {"from": "enrollment",  "to": "student", "type": "N:1"},
                {"from": "enrollment",  "to": "course",  "type": "N:1"},
            ],
        },
    ),

    # ── Library ───────────────────────────────────────────────────────────────
    (
        "Interface: book\nFields: id, title, isbn, author\n"
        "Interface: author\nFields: id, name, country\n"
        "Interface: member\nFields: id, name, email\n"
        "Interface: loan\nFields: id, book, member, loan_date, return_date",
        {
            "entities": [
                {"name": "book",   "attributes": ["id: number (PK)", "title: text", "isbn: text",
                                                  "author: number (FK)"]},
                {"name": "author", "attributes": ["id: number (PK)", "name: text", "country: text"]},
                {"name": "member", "attributes": ["id: number (PK)", "name: text", "email: text"]},
                {"name": "loan",   "attributes": ["id: number (PK)", "book: number (FK)",
                                                  "member: number (FK)", "loan_date: text", "return_date: text"]},
            ],
            "relations": [
                {"from": "book", "to": "author", "type": "N:1"},
                {"from": "loan", "to": "book",   "type": "N:1"},
                {"from": "loan", "to": "member", "type": "N:1"},
            ],
        },
    ),

    # ── Project Management ────────────────────────────────────────────────────
    (
        "Interface: employee\nFields: id, name, department\n"
        "Interface: project\nFields: id, name, budget, manager\n"
        "Interface: task\nFields: id, title, project, assignee, status, due_date",
        {
            "entities": [
                {"name": "employee", "attributes": ["id: number (PK)", "name: text", "department: text"]},
                {"name": "project",  "attributes": ["id: number (PK)", "name: text", "budget: number",
                                                    "manager: number (FK)"]},
                {"name": "task",     "attributes": ["id: number (PK)", "title: text",
                                                    "project: number (FK)", "assignee: number (FK)",
                                                    "status: text", "due_date: text"]},
            ],
            "relations": [
                {"from": "project", "to": "employee", "type": "N:1"},
                {"from": "task",    "to": "project",  "type": "N:1"},
                {"from": "task",    "to": "employee", "type": "N:1"},
            ],
        },
    ),

    # ── Music Streaming ───────────────────────────────────────────────────────
    (
        "Interface: artist\nFields: id, name, genre\n"
        "Interface: album\nFields: id, title, artist, release_year\n"
        "Interface: track\nFields: id, title, album, duration, plays",
        {
            "entities": [
                {"name": "artist", "attributes": ["id: number (PK)", "name: text", "genre: text"]},
                {"name": "album",  "attributes": ["id: number (PK)", "title: text",
                                                  "artist: number (FK)", "release_year: number"]},
                {"name": "track",  "attributes": ["id: number (PK)", "title: text",
                                                  "album: number (FK)", "duration: number", "plays: number"]},
            ],
            "relations": [
                {"from": "album", "to": "artist", "type": "N:1"},
                {"from": "track", "to": "album",  "type": "N:1"},
            ],
        },
    ),

    # ── Inventory ─────────────────────────────────────────────────────────────
    (
        "Interface: supplier\nFields: id, name, country\n"
        "Interface: warehouse\nFields: id, location, capacity\n"
        "Interface: product\nFields: id, name, supplier, price\n"
        "Interface: stock\nFields: id, product, warehouse, quantity",
        {
            "entities": [
                {"name": "supplier",  "attributes": ["id: number (PK)", "name: text", "country: text"]},
                {"name": "warehouse", "attributes": ["id: number (PK)", "location: text", "capacity: number"]},
                {"name": "product",   "attributes": ["id: number (PK)", "name: text",
                                                     "supplier: number (FK)", "price: number"]},
                {"name": "stock",     "attributes": ["id: number (PK)", "product: number (FK)",
                                                     "warehouse: number (FK)", "quantity: number"]},
            ],
            "relations": [
                {"from": "product", "to": "supplier",  "type": "N:1"},
                {"from": "stock",   "to": "product",   "type": "N:1"},
                {"from": "stock",   "to": "warehouse", "type": "N:1"},
            ],
        },
    ),

    # ── Forum ─────────────────────────────────────────────────────────────────
    (
        "Interface: user\nFields: id, username, email\n"
        "Interface: thread\nFields: id, title, user, category, created_at\n"
        "Interface: reply\nFields: id, thread, user, body, created_at",
        {
            "entities": [
                {"name": "user",   "attributes": ["id: number (PK)", "username: text", "email: text"]},
                {"name": "thread", "attributes": ["id: number (PK)", "title: text", "user: number (FK)",
                                                  "category: text", "created_at: text"]},
                {"name": "reply",  "attributes": ["id: number (PK)", "thread: number (FK)",
                                                  "user: number (FK)", "body: text", "created_at: text"]},
            ],
            "relations": [
                {"from": "thread", "to": "user",   "type": "N:1"},
                {"from": "reply",  "to": "thread", "type": "N:1"},
                {"from": "reply",  "to": "user",   "type": "N:1"},
            ],
        },
    ),

    # ── Real Estate ───────────────────────────────────────────────────────────
    (
        "Interface: agent\nFields: id, name, license_number\n"
        "Interface: property\nFields: id, address, price, agent\n"
        "Interface: buyer\nFields: id, name, email\n"
        "Interface: offer\nFields: id, property, buyer, amount, status, date",
        {
            "entities": [
                {"name": "agent",    "attributes": ["id: number (PK)", "name: text", "license_number: text"]},
                {"name": "property", "attributes": ["id: number (PK)", "address: text", "price: number",
                                                    "agent: number (FK)"]},
                {"name": "buyer",    "attributes": ["id: number (PK)", "name: text", "email: text"]},
                {"name": "offer",    "attributes": ["id: number (PK)", "property: number (FK)",
                                                    "buyer: number (FK)", "amount: number",
                                                    "status: text", "date: text"]},
            ],
            "relations": [
                {"from": "property", "to": "agent",    "type": "N:1"},
                {"from": "offer",    "to": "property", "type": "N:1"},
                {"from": "offer",    "to": "buyer",    "type": "N:1"},
            ],
        },
    ),
]


def build_multi_interface_pairs(
    examples: list[tuple[str, dict]],
    repeat: int = 5,
) -> list[dict]:
    """
    Convert the static multi-interface examples into training pairs.

    Args:
        examples: list of (input_text, output_dict) tuples
        repeat  : how many times to repeat each example (with light shuffling)
                  to increase dataset size and reduce overfitting

    Returns:
        list of {"input": str, "output": str} dicts
    """
    pairs = []
    random.seed(42)

    for flat_input, output_dict in examples:
        for _ in range(repeat):
            # Shuffle entity attribute order slightly so the model generalizes
            shuffled = output_dict.copy()
            shuffled["entities"] = [
                {**e, "attributes": random.sample(e["attributes"], len(e["attributes"]))}
                for e in output_dict["entities"]
            ]
            pairs.append({
                "input":  flat_input,
                "output": json.dumps(shuffled, indent=2, ensure_ascii=False),
            })

    print(f"  Multi-interface pairs generated: {len(pairs)} "
          f"({len(examples)} templates × {repeat} repeats)")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Build Combined Training Dataset
# ─────────────────────────────────────────────────────────────────────────────

print("\nBuilding training pairs from Spider schemas...")
spider_pairs: list[dict] = []

for row in spider_ds:
    try:
        spider_pairs.append({
            "input":  schema_to_flat_input(row),
            "output": schema_to_normalized_output(row),
        })
    except Exception as exc:
        print(f"  Skipping '{row.get('db_id', '?')}': {exc}")

print(f"  Spider pairs: {len(spider_pairs)}")

print("\nBuilding multi-interface training pairs...")
multi_pairs = build_multi_interface_pairs(MULTI_INTERFACE_EXAMPLES, repeat=50)

print("\nLoading WikiSQL pairs...")
wikisql_pairs = load_wikisql_pairs(max_tables=500, token=token)

print("\nBuilding ATIS pairs...")
atis_pairs = build_atis_pairs(repeat=20)

# Combine all sources and shuffle
# Spider:    ~83 pairs  (single flat interface, entity structure)
# Synthetic:  500 pairs (multi-interface, FK detection by field name)
# WikiSQL:   ~500 pairs (single flat interface, varied real-world domains)
# ATIS:       ~60 pairs (multi-table aviation domain, complex FKs)
all_pairs = spider_pairs[::2] + multi_pairs + wikisql_pairs + atis_pairs
random.seed(42)
random.shuffle(all_pairs)

print(f"\n  Spider pairs   : {len(spider_pairs[::2])}")
print(f"  Synthetic pairs: {len(multi_pairs)}")
print(f"  WikiSQL pairs  : {len(wikisql_pairs)}")
print(f"  ATIS pairs     : {len(atis_pairs)}")
print(f"  Total          : {len(all_pairs)}")
print("\n--- Spider sample input ---")
print(spider_pairs[0]["input"])
print("\n--- Multi-interface sample input ---")
print(multi_pairs[0]["input"])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Dataset Split
# ─────────────────────────────────────────────────────────────────────────────

hf_dataset = Dataset.from_list(all_pairs)
hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)

print(f"\n  Train size : {len(hf_dataset['train'])}")
print(f"  Test size  : {len(hf_dataset['test'])}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Prompt Formatting (ChatML)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a database architect. "
    "Analyze flat interface descriptions and produce normalized logical data "
    "models as JSON. The input may contain one or multiple named interfaces. "
    "When a field name matches another interface name, treat it as a foreign key. "
    "The JSON must contain 'entities' (with typed attributes annotated as PK or FK "
    "where applicable) and 'relations' (with from, to, and type fields)."
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
# 7. Model & Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
# QLoRA: 4-bit quantization via bitsandbytes reduces the 7B model from
# ~14 GB (bfloat16) to ~4.5 GB, leaving enough headroom on a 16 GB GPU.
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

print(f"\nLoading model: {MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,        # nested quantization saves ~0.4 GB
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
model.config.use_cache = False             # required for gradient checkpointing
model.config.pad_token_id = tokenizer.pad_token_id


# ─────────────────────────────────────────────────────────────────────────────
# 8. LoRA Configuration
# ─────────────────────────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=16,                                               # LoRA rank
    lora_alpha=32,                                      # scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Training
# ─────────────────────────────────────────────────────────────────────────────
# Effective batch size = per_device_train_batch_size × gradient_accumulation_steps
#                      = 1 × 16 = 16
# ─────────────────────────────────────────────────────────────────────────────

training_args = SFTConfig(
    output_dir="./ldm-expert-lora",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,        # effective batch size = 16
    learning_rate=2e-4,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",                      # set to "wandb" for experiment tracking
    dataset_text_field="text",
    max_length=512,
    gradient_checkpointing=True,           # trades compute for memory savings
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
trainer.model.save_pretrained("./ldm-expert-lora/final")
tokenizer.save_pretrained("./ldm-expert-lora/final")
print("Model saved to: ./ldm-expert-lora/final")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Inference Test
# ─────────────────────────────────────────────────────────────────────────────

def predict(flat_input: str) -> str:
    """Run inference on a flat interface description and return the data model JSON."""
    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{flat_input}\n"
        f"<|assistant|>\n"
    )
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


# Test with the exact format the user reported as failing
test_input = (
    "Interface: product\nFields: id, name, price\n"
    "Interface: customer\nFields: id, name, email\n"
    "Interface: order\nFields: id, product, customer, quantity, date"
)

print("\n--- Inference Test (multi-interface) ---")
print("Input :", test_input)
print("Output:", predict(test_input))
