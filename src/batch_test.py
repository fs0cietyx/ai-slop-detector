import json
import os
import sys
import logging
import warnings
from tqdm import tqdm
from core.inference import AISlopDetector, DetectorConfig

# Pillar VI: Secure Observability - Silence Verbose External Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
HUMAN_DATA_FILE = os.path.abspath("data/human_data.jsonl")
MODEL_DIR = os.path.abspath("models/ai-slop-detector-v1")

def run_batch_test():
    """
    Evaluates the model against the locally scraped Wikipedia data.
    
    Provides a quantitative assessment of domain-specific generalization (Task A).
    """
    if not os.path.exists(HUMAN_DATA_FILE):
        print(f"Error: {HUMAN_DATA_FILE} not found. Please run the crawler first.")
        return

    print(f"\n--- Batch Validation: Wikipedia Dataset ---")
    
    config = DetectorConfig(model_dir=MODEL_DIR)
    try:
        detector = AISlopDetector(config)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        return

    correct = 0
    total = 0
    results = []

    # Load and process data
    with open(HUMAN_DATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} human paragraphs...")
    
    for line in tqdm(lines, desc="Validating"):
        try:
            data = json.loads(line)
            text = data.get('text', '')
            if not text:
                continue
                
            label, confidence = detector.predict(text)
            
            # Since this is human_data.jsonl, the ground truth is always HUMAN-WRITTEN (Label 0)
            is_correct = (label == "HUMAN-WRITTEN")
            if is_correct:
                correct += 1
            
            total += 1
            results.append({
                "source": data.get('source'),
                "prediction": label,
                "confidence": confidence,
                "correct": is_correct
            })
        except Exception:
            continue

    if total == 0:
        print("No valid data processed.")
        return

    accuracy = correct / total
    
    print("\n" + "="*40)
    print("      WIKIPEDIA VALIDATION RESULTS")
    print("="*40)
    print(f" TOTAL SAMPLES:  {total}")
    print(f" CORRECT:        {correct}")
    print(f" INCORRECT:      {total - correct}")
    print(f" ACCURACY:       {accuracy:.2%}")
    print("="*40 + "\n")

if __name__ == "__main__":
    run_batch_test()
