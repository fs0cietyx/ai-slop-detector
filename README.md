# AI Slop Detector (Production Edition)

[![Security Audit](https://github.com/yourusername/ai-slop-detector/actions/workflows/main.yml/badge.svg)](https://github.com/yourusername/ai-slop-detector/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)

An enterprise-grade, zero-trust AI text classifier designed to identify "AI Slop" with high precision. Built with a production-first mindset for ML Engineers and AppSec Architects.

---

## Technical Features

### Performance & Optimization
- **Modular Inference Engine**: Isolated ML pipeline for independent scaling and testing.
- **Hardware-Aware Execution**: Automated selection of CUDA, Apple MPS, or CPU acceleration.
- **Resource Discipline**: Lazy-loading weights and strict memory boundaries for large payloads.
- **Tensor Optimization**: Zero-overhead evaluation passes using `torch.inference_mode()`.

### Security Hardening
- **Strict Configuration**: Pydantic-enforced environment validation.
- **Input Sanitization**: Payload neutralization to prevent injection and DoS.
- **Secrets Isolation**: Environment-level credential management.
- **Rootless Containerization**: Multi-stage Docker deployment with unprivileged execution.

---

## Execution Guide

### Containerized Deployment
```bash
docker build -t slop-detector -f deploy/Dockerfile .
docker run --rm slop-detector "Sample text for analysis..."
```

### Local Development
```bash
# Environment initialization
python3 -m venv venv
source venv/bin/activate

# Dependency installation
pip install -r requirements.txt

# Execute inference
python3 -m src.predict "Sample text here"
```

## Core Technologies
- **Model**: BERT-base-uncased (LoRA Fine-tuned)
- **Stack**: PyTorch, Hugging Face Transformers, PEFT
- **Quality**: Pydantic, Bandit, Ruff, MyPy

---
*Developed with focus on engineering quality and security.*
