import pandas as pd
import requests
import json


EXPECTED_SCHEMA = [
    "transaction_id",
    "customer_email",
    "purchase_amount",
    "purchase_date"
]


incoming_data = pd.DataFrame({
    "txn_id": ["A1", "A2"],
    "email_address": ["alice@test.com", "bob@test.com"],
    "total_cost": [150.00, 89.50],
    "date": ["2026-05-26", "2026-05-26"]
})


def heal_schema(expected_cols, actual_cols):
    prompt = f"""
    You are a data engineer system.

    Your job is to map actual data columns
    to the expected schema.

    Expected columns: {expected_cols}
    Actual columns: {actual_cols}

    Match the actual columns to the expected columns
    based on semantic meaning.

    Return ONLY a valid JSON object where:
    keys = actual columns
    values = expected columns.
    """

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result_text = response.json().get("response", "{}")

        mapping = json.loads(result_text)

        return mapping

    except Exception as e:
        print(f"CRITICAL: LLM healing failed: {e}")
        return None


def process_data(df, expected_schema):

    actual_cols = list(df.columns)

    if set(actual_cols) == set(expected_schema):
        print("Schema validation passed.")
        return df

    print("WARNING: Schema mismatch detected.")
    print("Starting self-healing...")

    mapping = heal_schema(
        expected_schema,
        actual_cols
    )

    if mapping:

        print(f"Healing mapping: {mapping}")

        df = df.rename(columns=mapping)

        missing_cols = [
            col
            for col in expected_schema
            if col not in df.columns
        ]

        if missing_cols:
            raise KeyError(
                f"Healing incomplete. Missing: {missing_cols}"
            )

        print("Pipeline successfully healed.")

        return df

    raise RuntimeError(
        "Self-healing failed."
    )


healed_df = process_data(
    incoming_data,
    EXPECTED_SCHEMA
)

print("\nFinal DataFrame:")
print(healed_df)