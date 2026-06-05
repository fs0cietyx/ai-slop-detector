import os
import random
from typing import List

from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME: str = "liamdugan/raid"
OUTPUT_FILE: str = "data/raid_subset.csv"
SAMPLES_PER_CLASS: int = 25000


def download_and_preprocess() -> None:
    """
    Downloads and prepares a balanced subset of the RAID dataset.
    """
    print(f"Loading {DATASET_NAME} metadata from Hugging Face...")

    try:
        dataset = load_dataset(DATASET_NAME, split="train", revision="main")
        total_records: int = len(dataset)
        print(f"Total records in RAID train split: {total_records}")

        print("Identifying indices for sampling...")
        human_indices: List[int] = []
        ai_indices: List[int] = []

        models: List[str] = dataset["model"]

        for i, model in enumerate(tqdm(models, desc="Finding samples")):
            if model == "human":
                human_indices.append(i)
            else:
                ai_indices.append(i)

        print(f"Found {len(human_indices)} human and {len(ai_indices)} AI samples.")

        # Pillar V: Randomize sampling (marking as safe for non-crypto use)
        sampled_human_indices = random.sample(
            human_indices, min(SAMPLES_PER_CLASS, len(human_indices))
        )  # nosec
        sampled_ai_indices = random.sample(ai_indices, min(SAMPLES_PER_CLASS, len(ai_indices)))  # nosec

        all_sampled_indices = sampled_human_indices + sampled_ai_indices
        random.shuffle(all_sampled_indices)  # nosec

        print(f"Selecting {len(all_sampled_indices)} samples...")
        subset = dataset.select(all_sampled_indices)

        # Optimized conversion to dataframe
        df = subset.to_pandas()

        # Binary label mapping
        df["binary_label"] = df["model"].apply(lambda x: 0 if x == "human" else 1)
        df = df[["generation", "binary_label", "model", "domain"]]
        df = df.rename(columns={"generation": "text"})

        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        print(f"Successfully saved {len(df)} balanced samples to {OUTPUT_FILE}")

    except Exception as e:
        print(f"Error during RAID acquisition: {e}")


if __name__ == "__main__":
    download_and_preprocess()
