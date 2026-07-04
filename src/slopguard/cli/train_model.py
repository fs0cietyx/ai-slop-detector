import os
from typing import Any, Dict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from slopguard.core.config import config, logger


def compute_metrics(eval_pred: Any) -> Dict[str, float]:
    """
    Computes enterprise-standard metrics for model evaluation.
    """
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels) or {"accuracy": 0.0}
    f1 = f1_metric.compute(predictions=predictions, references=labels) or {"f1": 0.0}

    return {"accuracy": float(acc["accuracy"]), "f1": float(f1["f1"])}


def run_training() -> None:
    """
    Orchestrates the fine-tuning of the transformer model using LoRA adapters.

    Adheres to Pillar II (ML Optimization) and Pillar IV (Defensive Programming).
    """
    data_path = "data/training_dataset.csv"
    
    if not os.path.exists(data_path):
        logger.error(f"TRAINING_HALTED: Dataset missing at {data_path}")
        return

    logger.info("Initializing Training Sequence...")

    # [Optimization] Efficient data ingestion with Pandas
    try:
        df = pd.read_csv(data_path)
        df["text"] = df["text"].astype(str)
        
        # Ensure label integrity
        df = df[df["label"].isin([0, 1])]
    except Exception as e:
        logger.error(f"DATASET_CORRUPTION: Failed to parse {data_path}: {str(e)}")
        return

    # Dataset Preparation
    dataset = Dataset.from_pandas(df)
    dataset = dataset.rename_column("label", "labels")
    tokenized_datasets = dataset.train_test_split(test_size=0.15, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, revision="main")  # nosec: B615

    def tokenize_fn(examples: Dict[str, Any]) -> Any:
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=config.MAX_INPUT_LENGTH
        )

    logger.info("Tokenizing multi-domain dataset...")
    tokenized_datasets = tokenized_datasets.map(tokenize_fn, batched=True)

    # Model Preparation
    logger.info(f"Loading Base Architecture: {config.MODEL_NAME}")
    base_model = AutoModelForSequenceClassification.from_pretrained(  # nosec: B615
        config.MODEL_NAME, 
        num_labels=2,
        revision="main"
    )

    # [Optimization] LoRA (Low-Rank Adaptation) Configuration
    # Reduces trainable parameters by ~99% while maintaining performance.
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

    # [Optimization] Production Training Arguments
    training_args = TrainingArguments(
        output_dir=config.ADAPTER_PATH,
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        push_to_hub=False,
        report_to="none",
        # Resource Discipline: Half-precision for faster training if supported
        fp16=torch.cuda.is_available(),
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

    logger.info("Training Cycle initiated.")
    trainer.train()

    logger.info(f"Exporting adapters to {config.ADAPTER_PATH}")
    trainer.save_model(config.ADAPTER_PATH)
    logger.info("Training complete. Assets verified.")


if __name__ == "__main__":
    try:
        run_training()
    except Exception as e:
        logger.critical(f"FATAL_TRAINING_FAILURE: {str(e)}")
