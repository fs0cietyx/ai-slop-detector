import json
import logging
import os
import warnings
from typing import Any, Dict, List

from tqdm import tqdm

# Pillar I: Modularity - Use absolute imports for reliability
from slopguard.core.engine import InferenceEngine

# Pillar VI: Secure Observability
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
HUMAN_DATA_FILE: str = os.path.abspath("data/human_data.jsonl")

def run_batch_test() -> None:
    """
    Executes a formal batch evaluation against local domain-specific data.
    """
    if not os.path.exists(HUMAN_DATA_FILE):
        print(f"Error: {HUMAN_DATA_FILE} not found. Please run the crawler first.")
        return

    print("\n--- Batch Validation: Wikipedia Dataset ---")

    try:
        detector = InferenceEngine()
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    correct: int = 0
    total: int = 0
    results: List[Dict[str, Any]] = []

    # Load data for processing
    with open(HUMAN_DATA_FILE, "r", encoding="utf-8") as f:
        lines: List[str] = f.readlines()

    print(f"Processing {len(lines)} human paragraphs...")

    for line in tqdm(lines, desc="Validating"):
        try:
            data: Dict[str, Any] = json.loads(line)
            text: str = str(data.get("text", ""))
            if not text:
                continue

            prediction = detector.predict(text)
            label: str = prediction[0]
            confidence: float = prediction[1]
            # Binary correctness verification
            is_correct: bool = label == "HUMAN-WRITTEN"
            if is_correct:
                correct += 1

            total += 1
            results.append(
                {
                    "source": data.get("source"),
                    "prediction": label,
                    "confidence": confidence,
                    "correct": is_correct,
                }
            )
        except Exception as e:
            print(f"Skipping malformed record: {e}")
            continue

    if total == 0:
        print("No valid data processed.")
        return

    accuracy: float = correct / total

    print("\n" + "=" * 40)
    print("      WIKIPEDIA VALIDATION RESULTS")
    print("=" * 40)
    print(f" TOTAL SAMPLES:  {total}")
    print(f" CORRECT:        {correct}")
    print(f" INCORRECT:      {total - correct}")
    print(f" ACCURACY:       {accuracy:.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    run_batch_test()
