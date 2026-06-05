import pandas as pd
from datasets import load_dataset
import os
import random
from tqdm import tqdm

# Configuration
DATASET_NAME = "liamdugan/raid"
OUTPUT_FILE = "data/raid_subset.csv"
SAMPLES_PER_CLASS = 25000 # Total 50,000 samples

def download_and_preprocess():
    print(f"Loading {DATASET_NAME} metadata from Hugging Face...")
    
    try:
        # Load the dataset. We use the 'train' split.
        # RAID is huge, so we avoid loading everything into a DataFrame at once.
        dataset = load_dataset(DATASET_NAME, split='train')
        total_records = len(dataset)
        print(f"Total records in RAID train split: {total_records}")

        print("Identifying indices for sampling (this might take a few minutes)...")
        human_indices = []
        ai_indices = []
        
        # We iterate to find indices. This is more memory efficient than .to_pandas()
        # We use a smaller loop or a more efficient search if possible
        # Actually, dataset['model'] will load the whole column. 
        # For 11M records, this is ~100MB of strings, which is fine.
        models = dataset['model']
        
        for i, model in enumerate(tqdm(models, desc="Finding samples")):
            if model == 'human':
                human_indices.append(i)
            else:
                ai_indices.append(i)
        
        print(f"Found {len(human_indices)} human and {len(ai_indices)} AI samples.")
        
        # Randomly sample
        sampled_human_indices = random.sample(human_indices, min(SAMPLES_PER_CLASS, len(human_indices)))
        sampled_ai_indices = random.sample(ai_indices, min(SAMPLES_PER_CLASS, len(ai_indices)))
        
        all_sampled_indices = sampled_human_indices + sampled_ai_indices
        random.shuffle(all_sampled_indices)
        
        print(f"Selecting {len(all_sampled_indices)} samples...")
        subset = dataset.select(all_sampled_indices)
        
        # Now convert the small subset to pandas
        df = subset.to_pandas()
        
        # Map labels to binary
        df['binary_label'] = df['model'].apply(lambda x: 0 if x == 'human' else 1)
        
        # We only need text, label, model, and domain
        df = df[['text', 'binary_label', 'model', 'domain']]
        
        # Save to data directory
        os.makedirs('data', exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Successfully saved {len(df)} balanced samples to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error during RAID acquisition: {e}")

if __name__ == "__main__":
    download_and_preprocess()
