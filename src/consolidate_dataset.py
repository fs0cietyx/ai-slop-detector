import json
import pandas as pd
import os

HUMAN_DATA_FILE = "data/human_data.jsonl"
AI_DATA_FILE = "data/ai_data.jsonl"
OUTPUT_FILE = "data/training_dataset.csv"

def consolidate():
    if not os.path.exists(HUMAN_DATA_FILE) or not os.path.exists(AI_DATA_FILE):
        print("Error: Required data files not found.")
        return

    data = []
    
    # Read Human Data
    with open(HUMAN_DATA_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({"text": item['text'], "label": 0})
    
    human_count = len(data)
    print(f"Loaded {human_count} human paragraphs.")

    # Read AI Data
    ai_count = 0
    with open(AI_DATA_FILE, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({"text": item['text'], "label": 1})
            ai_count += 1
    
    print(f"Loaded {ai_count} AI paragraphs.")
    
    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully consolidated {len(df)} samples into {OUTPUT_FILE}")

if __name__ == "__main__":
    consolidate()
