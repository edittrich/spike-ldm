"""
Inference Script: Flat Interface → Logical Data Model → Mermaid ERD
=====================================================================
Loads the fine-tuned LoRA adapter on top of the base model and
runs inference on flat interface descriptions, then exports the
result as a Mermaid ER diagram file.

Usage:
    uv run python inference.py
    uv run python inference.py --input "Interface: shop\nFields: id, name, price" --output example.mmd
    uv run python inference.py --file ecommerce.txt --output ecommerce.mmd
    uv run python inference.py --examples
    uv run python inference.py --output example.mmd
"""

import argparse
import json
import os

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ─────────────────────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
token = os.getenv("HF_TOKEN")

BASE_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "./ldm-expert-lora/final"

SYSTEM_PROMPT = (
    "You are a database architect. "
    "Analyze flat interface descriptions and produce normalized logical data "
    "models as JSON. The JSON must contain 'entities' (with typed attributes "
    "annotated as PK or FK where applicable) and 'relations' "
    "(with from, to, and type fields)."
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Model & Adapter
# ─────────────────────────────────────────────────────────────────────────────


def load_model():
    print(f"Loading base model  : {BASE_MODEL_ID}")
    print(f"Loading LoRA adapter: {ADAPTER_PATH}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        use_fast=True,
        token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=token,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("Model ready.\n")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 2. Inference
# ─────────────────────────────────────────────────────────────────────────────


def predict(model, tokenizer, flat_input: str) -> dict:
    """
    Run inference on a flat interface description.

    Returns a parsed JSON dict with 'entities' and 'relations',
    or a dict with 'raw' key on parse failure.
    """
    prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{flat_input}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw = decoded.split("<|assistant|>")[-1].strip()

    # Try to parse as JSON, extract JSON block as fallback
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return {"raw": raw}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mermaid ERD Export
# ─────────────────────────────────────────────────────────────────────────────


def to_mermaid(result: dict, title: str = "") -> str:
    """
    Convert a logical data model dict to a Mermaid ER diagram string.

    Mermaid erDiagram syntax:
        erDiagram
            ENTITY {
                type attribute PK
                type attribute FK
                type attribute
            }
            ENTITY_A ||--o{ ENTITY_B : "relation"

    Args:
        result: dict with 'entities' and 'relations' keys
        title : optional diagram title

    Returns:
        Mermaid diagram as a string
    """
    if "raw" in result:
        return f"%% Could not parse model output\n%% Raw: {result['raw'][:200]}"

    lines = []

    if title:
        lines.append(f"---")
        lines.append(f"title: {title}")
        lines.append(f"---")

    lines.append("erDiagram")

    # Entity blocks
    for entity in result.get("entities", []):
        name = entity["name"].upper().replace(" ", "_")
        lines.append(f"    {name} {{")
        for attr in entity.get("attributes", []):
            # attr format: "col_name: type (PK)" or "col_name: type (FK)"
            # Mermaid format: type name PK/FK
            mermaid_attr = _parse_attribute(attr)
            lines.append(f"        {mermaid_attr}")
        lines.append("    }")

    lines.append("")

    # Relation lines
    # Map N:1 → Mermaid crow's foot notation
    # "from N:1 to" means: many (from) relate to one (to)
    # Mermaid: FROM }o--|| TO
    relation_map = {
        "N:1": "}o--||",
        "1:N": "||--o{",
        "1:1": "||--||",
        "N:M": "}o--o{",
    }

    for rel in result.get("relations", []):
        from_e = rel.get("from", "").upper().replace(" ", "_")
        to_e = rel.get("to", "").upper().replace(" ", "_")
        rel_type = rel.get("type", "N:1")
        arrow = relation_map.get(rel_type, "}o--||")
        lines.append(f'    {from_e} {arrow} {to_e} : "references"')

    return "\n".join(lines)


def _parse_attribute(attr: str) -> str:
    """
    Convert a model attribute string to Mermaid attribute syntax.

    Input examples:
        "Stadium_ID: number (PK)"  → "number Stadium_ID PK"
        "Name: text"               → "string Name"
        "Singer_ID: number (FK)"   → "number Singer_ID FK"

    Mermaid type mapping:
        number  → int
        text    → string
        boolean → boolean
        date    → date
        (other) → string
    """
    type_map = {
        "number": "int",
        "integer": "int",
        "int": "int",
        "text": "string",
        "varchar": "string",
        "char": "string",
        "string": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "date": "date",
        "time": "datetime",
        "float": "float",
        "real": "float",
        "decimal": "float",
    }

    annotation = ""
    if "(PK)" in attr:
        annotation = "PK"
        attr = attr.replace("(PK)", "").strip()
    elif "(FK)" in attr:
        annotation = "FK"
        attr = attr.replace("(FK)", "").strip()

    if ":" in attr:
        col_name, col_type = attr.split(":", 1)
        col_name = col_name.strip().replace(" ", "_")
        col_type = col_type.strip().lower()
        mermaid_type = type_map.get(col_type, "string")
    else:
        col_name = attr.strip().replace(" ", "_")
        mermaid_type = "string"

    parts = [mermaid_type, col_name]
    if annotation:
        parts.append(annotation)

    return " ".join(parts)


def save_mermaid(mermaid_str: str, output_path: str):
    """Write the Mermaid diagram to a .mmd file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mermaid_str)
    print(f"Mermaid file saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    (
        "Webshop",
        "Interface: webshop\n"
        "Fields: order_id, customer_name, customer_email, product_name, "
        "product_price, quantity, order_date, shipping_address, payment_method",
    ),
    (
        "Hospital",
        "Interface: hospital\n"
        "Fields: patient_id, patient_name, doctor_name, doctor_specialty, "
        "appointment_date, diagnosis, medication, room_number, ward_name",
    ),
    (
        "School",
        "Interface: school\n"
        "Fields: student_id, student_name, course_name, teacher_name, "
        "grade, semester, classroom, enrollment_date",
    ),
]


def run_single(model, tokenizer, flat_input: str, title: str, output_path: str):
    """Run inference for a single input and save the Mermaid file."""
    print("=== Input ===")
    print(flat_input)

    result = predict(model, tokenizer, flat_input)

    print("\n=== Logical Data Model (JSON) ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    mermaid = to_mermaid(result, title=title)

    print("\n=== Mermaid ERD ===")
    print(mermaid)

    save_mermaid(mermaid, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flat Interface → Logical Data Model → Mermaid ERD"
    )
    parser.add_argument(
        "--input", type=str, help="Flat interface description as string"
    )
    parser.add_argument(
        "--file", type=str, help="Path to a .txt file with the interface"
    )
    parser.add_argument(
        "--output", type=str, default="diagram.mmd", help="Output .mmd file path"
    )
    parser.add_argument("--title", type=str, default="", help="Diagram title")
    parser.add_argument(
        "--examples", action="store_true", help="Run all built-in examples"
    )
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.file:
        with open(args.file, "r") as f:
            flat_input = f.read().strip()
        title = args.title or os.path.splitext(os.path.basename(args.file))[0]
        run_single(model, tokenizer, flat_input, title=title, output_path=args.output)

    elif args.input:
        flat_input = args.input.replace("\\n", "\n")
        run_single(
            model, tokenizer, flat_input, title=args.title, output_path=args.output
        )

    elif args.examples:
        for name, flat_input in EXAMPLES:
            output_path = f"{name.lower()}.mmd"
            print(f"\n{'=' * 60}")
            print(f"Example: {name}")
            print(f"{'=' * 60}")
            run_single(
                model, tokenizer, flat_input, title=name, output_path=output_path
            )

    else:
        # Interactive mode
        print("Interactive mode — type 'quit' to exit.")
        print("Format: provide interface name and fields when prompted.\n")
        while True:
            print("Interface name (or 'quit'):")
            name = input("> ").strip()
            if name.lower() == "quit":
                break
            print("Fields (comma-separated):")
            fields = input("> ").strip()
            flat_input = f"Interface: {name}\nFields: {fields}"
            output_path = (
                args.output if args.output != "diagram.mmd" else f"{name.lower()}.mmd"
            )
            run_single(
                model, tokenizer, flat_input, title=name, output_path=output_path
            )
            print()
