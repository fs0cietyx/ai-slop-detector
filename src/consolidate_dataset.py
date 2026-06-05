import json
import os
from typing import Any, Dict, List

import pandas as pd

HUMAN_DATA_FILE: str = "data/human_data.jsonl"
AI_DATA_FILE: str = "data/ai_data.jsonl"
OUTPUT_FILE: str = "data/training_dataset.csv"


def consolidate() -> None:
    """
    Merges local human and AI data streams into a unified training artifact.

    Pillar I: Architectural Rigor - Ensures data consistency and label mapping.
    """
    if not os.path.exists(HUMAN_DATA_FILE) or not os.path.exists(AI_DATA_FILE):
        print("Error: Required data files not found.")
        return

    data: List[Dict[str, Any]] = []

    # Ingest Human Data
    with open(HUMAN_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append({"text": item["text"], "label": 0})
            except (json.JSONDecodeError, KeyError):
                continue

    human_count = len(data)
    print(f"Loaded {human_count} human paragraphs.")

    # Ingest AI Data
    ai_count = 0
    with open(AI_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                data.append({"text": item["text"], "label": 1})
                ai_count += 1
            except (json.JSONDecodeError, KeyError):
                continue

    print(f"Loaded {ai_count} AI paragraphs.")

    # Conversion and Shuffling
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully consolidated {len(df)} samples into {OUTPUT_FILE}")


if __name__ == "__main__":
    consolidate()
