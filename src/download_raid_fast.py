import os
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME: str = "liamdugan/raid"
OUTPUT_FILE: str = "data/raid_subset.csv"
SAMPLES_PER_CLASS: int = 1000


def download_fast() -> None:
    """
    Performs a high-speed, memory-efficient streaming download of the RAID dataset.

    Pillar II: NLP Optimization - Uses streaming to avoid multi-gigabyte memory overhead.
    """
    print(f"Streaming {DATASET_NAME} from Hugging Face...")

    try:
        # Load the dataset in streaming mode
        dataset = load_dataset(DATASET_NAME, split="train", streaming=True, revision="main")

        human_samples: List[Dict[str, Any]] = []
        ai_samples: List[Dict[str, Any]] = []

        print("Collecting samples...")
        with tqdm(total=SAMPLES_PER_CLASS * 2) as pbar:
            for entry in dataset:
                model_name: str = entry.get("model", "")
                generation_text: str = entry.get("generation", "")
                domain_name: str = entry.get("domain", "")

                if model_name == "human" and len(human_samples) < SAMPLES_PER_CLASS:
                    human_samples.append(
                        {
                            "text": generation_text,
                            "binary_label": 0,
                            "model": model_name,
                            "domain": domain_name,
                        }
                    )
                    pbar.update(1)
                elif model_name != "human" and len(ai_samples) < SAMPLES_PER_CLASS:
                    ai_samples.append(
                        {
                            "text": generation_text,
                            "binary_label": 1,
                            "model": model_name,
                            "domain": domain_name,
                        }
                    )
                    pbar.update(1)

                if len(human_samples) >= SAMPLES_PER_CLASS and len(ai_samples) >= SAMPLES_PER_CLASS:
                    break

        df = pd.DataFrame(human_samples + ai_samples)
        df = df.sample(frac=1).reset_index(drop=True)

        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        print(f"Successfully saved {len(df)} samples to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error during streaming download: {e}")


if __name__ == "__main__":
    download_fast()
