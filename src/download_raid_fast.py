import os

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DATASET_NAME = "liamdugan/raid"
OUTPUT_FILE = "data/raid_subset.csv"
SAMPLES_PER_CLASS = 1000 

def download_fast():
    print(f"Streaming {DATASET_NAME} from Hugging Face...")
    
    try:
        dataset = load_dataset(DATASET_NAME, split='train', streaming=True, revision="main")
        
        human_samples = []
        ai_samples = []
        
        print("Collecting samples...")
        with tqdm(total=SAMPLES_PER_CLASS * 2) as pbar:
            for entry in dataset:
                if entry['model'] == 'human' and len(human_samples) < SAMPLES_PER_CLASS:
                    human_samples.append({
                        'text': entry['generation'],
                        'binary_label': 0,
                        'model': entry['model'],
                        'domain': entry['domain']
                    })
                    pbar.update(1)
                elif entry['model'] != 'human' and len(ai_samples) < SAMPLES_PER_CLASS:
                    ai_samples.append({
                        'text': entry['generation'],
                        'binary_label': 1,
                        'model': entry['model'],
                        'domain': entry['domain']
                    })
                    pbar.update(1)
                
                if len(human_samples) >= SAMPLES_PER_CLASS and len(ai_samples) >= SAMPLES_PER_CLASS:
                    break
        
        df = pd.DataFrame(human_samples + ai_samples)
        df = df.sample(frac=1).reset_index(drop=True)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"Successfully saved {len(df)} samples to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_fast()
