"""
WikiSQL Dataset → Training Pairs
=================================
WikiSQL contains 80,654 examples across 24,241 tables from Wikipedia.
Each table has a flat schema (single table, no FKs) — useful for
teaching the model to recognize entity structure and PK detection.

We load directly from Parquet since the original dataset uses a
loading script which is no longer supported.

Source: https://huggingface.co/datasets/Salesforce/wikisql
"""

from datasets import load_dataset
import json
import random


def load_wikisql_pairs(
    max_tables: int = 500,
    token: str | None = None,
) -> list[dict]:
    """
    Extract unique table schemas from WikiSQL and convert them to
    flat interface → normalized entity training pairs.

    Args:
        max_tables: maximum number of unique tables to extract
        token:      HuggingFace token

    Returns:
        list of {"input": str, "output": str} dicts
    """
    print("Loading WikiSQL from Parquet...")

    # Load directly from Parquet files — avoids the loading script issue
    parquet_url = (
        "https://huggingface.co/datasets/Salesforce/wikisql/"
        "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    )

    try:
        ds = load_dataset(
            "parquet",
            data_files={"train": parquet_url},
            split="train",
            token=token,
        )
        print(f"  Total examples : {len(ds)}")
        print(f"  Columns        : {ds.column_names}")
    except Exception as e:
        print(f"  Warning: could not load WikiSQL Parquet ({e})")
        print("  Falling back to mlx-community/wikisql mirror...")
        ds = load_dataset("mlx-community/wikisql", split="train", token=token)
        print(f"  Total examples : {len(ds)}")

    # WikiSQL Parquet schema:
    # table = {
    #   "header": ["col1", "col2", ...],
    #   "types":  ["text", "real", ...],
    #   "id":     "table_id"
    # }

    # De-duplicate tables by id
    seen_ids: set[str] = set()
    pairs: list[dict] = []

    for row in ds:
        table = row.get("table", row)  # handle both formats

        # Extract table fields depending on format
        if isinstance(table, dict):
            table_id = table.get("id", str(len(pairs)))
            headers  = table.get("header", [])
            types    = table.get("types", ["text"] * len(headers))
        else:
            # mlx-community mirror has flat structure
            continue

        if table_id in seen_ids or not headers:
            continue
        seen_ids.add(table_id)

        # Build readable table name from id
        table_name = f"table_{table_id.replace('-', '_')}"

        # Flat input: all columns as a single interface
        flat_input = (
            f"Interface: {table_name}\n"
            f"Fields: {', '.join(headers)}"
        )

        # Normalized output: single entity, first column as PK
        type_map = {"text": "text", "real": "number", "integer": "number"}
        attributes = []
        for i, (col, col_type) in enumerate(zip(headers, types)):
            mtype      = type_map.get(str(col_type).lower(), "text")
            annotation = " (PK)" if i == 0 else ""
            attributes.append(f"{col}: {mtype}{annotation}")

        output = json.dumps(
            {
                "entities": [{"name": table_name, "attributes": attributes}],
                "relations": [],
            },
            indent=2,
            ensure_ascii=False,
        )

        pairs.append({"input": flat_input, "output": output})

        if len(pairs) >= max_tables:
            break

    print(f"  WikiSQL pairs extracted: {len(pairs)}")
    return pairs


if __name__ == "__main__":
    pairs = load_wikisql_pairs(max_tables=5)
    for p in pairs:
        print("\n--- Input ---")
        print(p["input"])
        print("--- Output ---")
        print(p["output"])
