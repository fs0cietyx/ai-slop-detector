import json
import os
from typing import Any, Dict, List

import pandas as pd

from .core.config import logger


def run_consolidation() -> None:
    """
    Merges distributed human and AI data streams into a validated training artifact.

    Adheres to Pillar I (Architectural Rigor) and Pillar IV (Defensive Programming).
    """
    human_file = "data/human_data.jsonl"
    ai_file = "data/ai_data.jsonl"
    output_file = "data/training_dataset.csv"

    if not os.path.exists(human_file) or not os.path.exists(ai_file):
        logger.error("CONSOLIDATION_HALTED: Source JSONL files missing.")
        return

    logger.info("Initiating Data Consolidation Cycle...")
    records: List[Dict[str, Any]] = []

    def ingest_file(path: str, expected_label: int) -> int:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    text = item.get("text")
                    if text and isinstance(text, str) and len(text) > 100:
                        records.append({
                            "text": text.strip(),
                            "label": expected_label,
                            "source_origin": item.get("source", "unknown")
                        })
                        count += 1
                except Exception:
                    continue
        return count

    # Ingest human data (Label 0)
    h_count = ingest_file(human_file, 0)
    logger.info(f"Ingested {h_count} human-origin samples.")

    # Ingest AI data (Label 1)
    a_count = ingest_file(ai_file, 1)
    logger.info(f"Ingested {a_count} AI-origin samples.")

    if not records:
        logger.warning("PIPELINE_IDLE: No valid records found for consolidation.")
        return

    # [Optimization] Performance-aware DataFrame conversion
    df = pd.DataFrame(records)
    
    # [AppSec] Integrity: Shuffle dataset to prevent bias during epoch boundaries
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Export to CSV for Trainer ingestion
    df.to_csv(output_file, index=False)
    logger.info(f"Successfully consolidated {len(df)} samples into {output_file}")


if __name__ == "__main__":
    try:
        run_consolidation()
    except Exception as e:
        logger.critical(f"FATAL_CONSOLIDATION_FAILURE: {str(e)}")
