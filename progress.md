# Project Progress Diary

## [Date - Initial Milestones]
- **RAID Data Acquisition**: Implemented a memory-efficient streaming downloader (`src/download_raid_fast.py`). Successfully sampled 2,000 documents for initial training.
- **Model Training Complete**: Successfully fine-tuned BERT-base + LoRA on a subset of the RAID dataset.
    - **Performance**: 97.5% Accuracy, 97.57% F1 Score.
    - **Training Time**: ~21 minutes (MPS-accelerated).
    - **Artifact**: Model saved to `models/ai-slop-detector-v1`.
- **Inference Setup**: Created `src/predict.py` for real-time AI detection testing.
- **AI Data Generation**: Developed a two-pass approach (Summarize -> Rewrite) to ensure high-quality, Wikipedia-style AI text. Used OpenAI Batch API and GPT-5 Nano for cost-efficient generation.
- **Model Training**: Fine-tuned BERT-base using LoRA and Hugging Face. Achieved 96% accuracy on the internal dataset.
- **Validation**: Created a sanity check script for real-time testing of human vs. AI text.
- **Benchmarking**: Tested the model against the RAID dataset.
    - **Result**: TPR of ~90% on movie reviews (top 10 leaderboard potential).
    - **Result**: Surprising generalization to Llama model families.
    - **Note**: Performance is weaker on poetry and abstracts, likely due to training domain limitations.
