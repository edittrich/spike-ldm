"""
ATIS Dataset → Training Pairs
===============================
ATIS (Airline Travel Information System) contains queries about
flight information. The underlying database has a well-defined
relational schema with multiple tables and FK relationships —
making it ideal for multi-table relation training.

Source: https://huggingface.co/datasets/tuetschek/atis
Schema: flight, airline, airport, city, aircraft, fare, etc.

Since ATIS only ships with NL queries (not the raw schema), we
encode the known ATIS schema directly as high-quality training pairs.
"""

import json
import random

# ── ATIS Database Schema (well-known, manually encoded) ──────────────────────
# Source: https://github.com/jkkummerfeld/text2sql-data
# The ATIS schema has 25+ tables. We include the most important ones.

ATIS_SCHEMA = {
    "entities": [
        {
            "name": "airline",
            "attributes": [
                "airline_code: text (PK)",
                "airline_name: text",
                "note: text",
            ],
        },
        {
            "name": "airport",
            "attributes": [
                "airport_code: text (PK)",
                "airport_name: text",
                "airport_location: text",
                "country_name: text",
                "time_zone_code: text",
                "minimum_connect_time: number",
            ],
        },
        {
            "name": "city",
            "attributes": [
                "city_code: text (PK)",
                "city_name: text",
                "state_code: text",
                "country_name: text",
                "time_zone_code: text",
            ],
        },
        {
            "name": "aircraft",
            "attributes": [
                "aircraft_code: text (PK)",
                "aircraft_description: text",
                "manufacturer: text",
                "basic_type: text",
                "engines: number",
                "propulsion: text",
                "wide_body: text",
                "pressurized: text",
            ],
        },
        {
            "name": "flight",
            "attributes": [
                "flight_id: number (PK)",
                "airline_code: text (FK)",
                "flight_number: number",
                "from_airport: text (FK)",
                "to_airport: text (FK)",
                "departure_time: number",
                "arrival_time: number",
                "aircraft_code_sequence: text (FK)",
                "meal_code: text",
                "stops: number",
            ],
        },
        {
            "name": "fare",
            "attributes": [
                "fare_id: number (PK)",
                "from_airport: text (FK)",
                "to_airport: text (FK)",
                "fare_basis_code: text",
                "fare_airline: text (FK)",
                "restriction_code: text",
                "one_direction_cost: number",
                "round_trip_cost: number",
            ],
        },
        {
            "name": "flight_fare",
            "attributes": [
                "flight_id: number (FK)",
                "fare_id: number (FK)",
            ],
        },
        {
            "name": "class_of_service",
            "attributes": [
                "booking_class: text (PK)",
                "rank: number",
                "class_description: text",
            ],
        },
        {
            "name": "restriction",
            "attributes": [
                "restriction_code: text (PK)",
                "advance_purchase: number",
                "stopovers: text",
                "saturday_stay_required: text",
                "minimum_stay: number",
                "maximum_stay: number",
                "application: text",
                "no_discounts: text",
            ],
        },
    ],
    "relations": [
        {"from": "flight", "to": "airline", "type": "N:1"},
        {"from": "flight", "to": "airport", "type": "N:1"},
        {"from": "flight", "to": "aircraft", "type": "N:1"},
        {"from": "fare", "to": "airport", "type": "N:1"},
        {"from": "fare", "to": "airline", "type": "N:1"},
        {"from": "flight_fare", "to": "flight", "type": "N:1"},
        {"from": "flight_fare", "to": "fare", "type": "N:1"},
        {"from": "fare", "to": "restriction", "type": "N:1"},
    ],
}


def build_atis_pairs(repeat: int = 20) -> list[dict]:
    """
    Generate training pairs from the ATIS schema.

    Uses the full schema as well as subsets (2-4 tables at a time)
    to teach the model to handle partial schemas correctly.

    Args:
        repeat: number of full-schema and subset repetitions

    Returns:
        list of {"input": str, "output": str} dicts
    """
    pairs: list[dict] = []
    random.seed(42)
    entities = ATIS_SCHEMA["entities"]
    relations = ATIS_SCHEMA["relations"]
    entity_map = {e["name"]: e for e in entities}

    # ── Full schema pair ──────────────────────────────────────────────────────
    for _ in range(repeat):
        shuffled_entities = random.sample(entities, len(entities))
        flat_parts = []
        for e in shuffled_entities:
            fields = [a.split(":")[0].strip() for a in e["attributes"]]
            flat_parts.append(f"Interface: {e['name']}\nFields: {', '.join(fields)}")
        flat_input = "\n".join(flat_parts)

        output_entities = [
            {**e, "attributes": random.sample(e["attributes"], len(e["attributes"]))}
            for e in shuffled_entities
        ]
        pairs.append(
            {
                "input": flat_input,
                "output": json.dumps(
                    {"entities": output_entities, "relations": relations},
                    indent=2,
                    ensure_ascii=False,
                ),
            }
        )

    # ── Subset pairs: 2-4 tables at a time ───────────────────────────────────
    for _ in range(repeat * 2):
        n = random.randint(2, 4)
        subset = random.sample(entities, min(n, len(entities)))
        subset_names = {e["name"] for e in subset}

        # Only keep relations where both tables are in the subset
        subset_relations = [
            r
            for r in relations
            if r["from"] in subset_names and r["to"] in subset_names
        ]

        flat_parts = []
        for e in subset:
            fields = [a.split(":")[0].strip() for a in e["attributes"]]
            flat_parts.append(f"Interface: {e['name']}\nFields: {', '.join(fields)}")

        pairs.append(
            {
                "input": "\n".join(flat_parts),
                "output": json.dumps(
                    {"entities": subset, "relations": subset_relations},
                    indent=2,
                    ensure_ascii=False,
                ),
            }
        )

    print(f"  ATIS pairs generated: {len(pairs)}")
    return pairs


if __name__ == "__main__":
    pairs = build_atis_pairs(repeat=2)
    print(f"\nTotal pairs: {len(pairs)}")
    print("\n--- Sample full schema input (first 3 interfaces) ---")
    print("\n".join(pairs[0]["input"].split("\n")[:6]))
    print("\n--- Sample output (relations) ---")
    result = json.loads(pairs[0]["output"])
    print(json.dumps(result["relations"], indent=2))
