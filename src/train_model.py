import os
from typing import Any, Dict

import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Configuration
MODEL_NAME: str = "bert-base-uncased"
DATA_PATH: str = "data/raid_subset.csv"
OUTPUT_DIR: str = "models/ai-slop-detector-v1"
LOG_DIR: str = "logs"


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Computes accuracy and F1 metrics for training evaluation.
    """
    load_accuracy = evaluate.load("accuracy")
    load_f1 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy: float = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1: float = load_f1.compute(predictions=predictions, references=labels)["f1"]

    return {"accuracy": accuracy, "f1": f1}


def train() -> None:
    """
    Orchestrates the fine-tuning of the transformer model using LoRA.
    """
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return

    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["text"] = df["text"].astype(str)

    # Dataset Preparation
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("binary_label", "labels")
    tokenized_datasets = dataset.train_test_split(test_size=0.2)

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision="main")

    def tokenize_function(examples: Dict[str, Any]) -> Any:
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("Tokenizing dataset...")
    tokenized_datasets = tokenized_datasets.map(tokenize_function, batched=True)

    # Model Preparation
    print(f"Loading base model: {MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, revision="main"
    )

    # LoRA Configuration (Pillar II: Optimization)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
    )

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        push_to_hub=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    print("Training complete!")


if __name__ == "__main__":
    train()
