# AI Slop Detector: Text Classifier

[![Security Audit](https://github.com/fs0cietyx/ai-slop-detector/actions/workflows/main.yml/badge.svg)](https://github.com/fs0cietyx/ai-slop-detector/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![Model: BERT+LoRA](https://img.shields.io/badge/Model-BERT+LoRA-blue.svg)](https://huggingface.co/docs/peft/index)

An enterprise-grade, zero-trust machine learning suite designed to classify text as human-written or AI-generated with clinical precision. This project moves beyond "vibe-coded" prototypes, implementing a modular, production-hardened architecture built for high-throughput AI services.

---

## 🏗️ Architectural Deep Dive

### 1. Machine Learning Engine
The core inference engine leverages a fine-tuned **BERT-base-uncased** transformer (110M parameters). To ensure high performance on consumer-grade hardware, we implement **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA (Low-Rank Adaptation)**.

#### 🔧 Hyperparameter Configuration:
- **Architecture**: `bert-base-uncased`
- **Fine-Tuning Method**: LoRA (Low-Rank Adaptation)
- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: `query`, `value`
- **Dropout**: 0.1
- **Learning Rate**: 2e-4
- **Batch Size**: 16 (Effective via Gradient Accumulation)
- **Epochs**: 3
- **Validation Accuracy**: **97.5%** on the RAID Benchmark.

#### 🚀 Optimization Protocols:
- **Hardware-Awareness**: Adaptive selection of **NVIDIA CUDA** or **Apple Silicon MPS** for tensor operations.
- **Inference Mode**: Utilizes `torch.inference_mode()` to eliminate gradient tracking overhead during classification.
- **Lazy Loading**: Weights are mounted into memory only when the engine is initialized, minimizing background resource consumption.

### 2. The Hybrid Data Pipeline
The model’s robustness stems from a dual-layer data acquisition strategy.

- **Global Baseline (RAID)**: Trained on the **Robust AI Detection (RAID)** dataset, providing exposure to **11 different LLMs** (GPT-4, Claude, Llama-2, etc.) across multiple creative and technical domains.
- **Local Validation (Wikipedia)**: A custom **Asynchronous Random Walk Crawler** collected 10,000 paragraphs of human-written Wikipedia text to serve as a domain-specific sanity check.
- **Adversarial Synthesis**: Implements a **Two-Pass Generation** pipeline using **Gemini 1.5 Flash** (Summarize → Rewrite) to create highly coherent AI text that mimics specific human styles.

### 3. Zero-Trust Security & AppSec
The system is architected under the assumption that all input is malicious.

- **Input Neutralization**: Aggressive sanitization strips non-printable ASCII and UTF-8 control characters.
- **DoS Prevention**: Enforced **5,000-character boundary** on all inference payloads to prevent memory exhaustion.
- **Secrets Management**: Configuration is enforced via **Pydantic BaseSettings**. The system rejects execution if security invariants or API keys are missing or malformed.
- **SSRF Protection**: The crawler validates all outbound requests against a strict `BASE_URL` allow-list.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Core ML** | PyTorch, Transformers, PEFT, Accelerate |
| **Data Science** | Pandas, Scikit-learn, Evaluate, NumPy |
| **Backend/DevOps** | Pydantic, Docker (Multi-stage), AsyncIO, Logging |
| **Quality Control** | MyPy (Strict), Ruff, Bandit (SAST), Pytest |

---

## 🚀 Execution Guide

### 🐋 Containerized Deployment (Production)
The project is ready for Docker-based orchestration using a rootless, multi-stage build.

```bash
# Build the production image
docker build -t slop-detector -f deploy/Dockerfile .

# Run secure inference
docker run --rm slop-detector "The industrial revolution was a period of global economic transition..."
```

### 🐍 Local Development
```bash
# Initialize and install
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run inference
python3 -m src.predict "Your text here"
```

---

## 📈 Engineering Quality Gates
Every commit is audited by our **Production CI/CD Pipeline**:
- **MyPy**: Enforces 100% strict static typing.
- **Ruff**: Validates PEP 8 compliance and code formatting.
- **Bandit**: Conducts Static Application Security Testing (SAST).
- **Pytest**: Executes the functional verification suite.

---
*Developed with a commitment to engineering excellence and algorithmic integrity.*
