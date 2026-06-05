# RAID Dataset Evaluation

## Overview
RAID is a large-scale dataset for human and AI text detection, covering 8 domains and 5 model families. It serves as the primary benchmark for this project.

## Domains & Performance
- **Movie Reviews**: Exceptional performance, True Positive Rate (TPR) ~90% at low False Positive Rate (FPR). Potential top-10 leaderboard placement.
- **Wikipedia**: Decent performance, as expected given the training data.
- **Poetry & Abstracts**: Poor performance; the model struggles with these highly structured or creative domains.
- **Model Generalization**: Although trained on GPT-5 Nano generations, the model performs surprisingly well on **Llama** (Meta) model families.

## Key Metrics
- **True Positive Rate (TPR)**: Probability of correctly identifying AI writing.
- **False Positive Rate (FPR)**: Probability of incorrectly identifying human writing as AI.
- **ROC Curve**: Used to tune sensitivity and compare against industry benchmarks like GPT Zero (which achieves 98% TPR at 5% FPR).
