# 🔍 Vision-Based Industrial Anomaly Detection using Contrastive Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end system for detecting anomalies in industrial product images using **SimCLR contrastive learning**. The system learns robust representations from normal product images, then identifies anomalies as images that deviate from the learned "normal" distribution.

> **BSc Final Year Project** — Electrical & Computer Engineering

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [Results](#-results)
- [References](#-references)

---

## 🎯 Overview

**Problem:** Industrial quality control requires detecting product defects, but labeled anomalous samples are scarce.

**Solution:** Use **self-supervised contrastive learning (SimCLR)** to learn representations from normal product images only, then detect anomalies by measuring distance from the learned normal distribution.

### Key Features
- 🧠 **SimCLR** contrastive learning with NT-Xent loss
- 🏗️ **ResNet-18** backbone encoder pretrained on ImageNet
- 📊 **k-NN anomaly scoring** using feature memory bank
- 🗺️ **Grad-CAM** anomaly localization heatmaps
- 📈 Comprehensive **metrics**: AUROC, F1-Score, precision, recall
- 🌐 Interactive **Streamlit web application**

---

## 🏗️ Architecture

```
Training Phase:
  Image → [SimCLR Augmentations] → [ResNet-18 Encoder] → [Projection Head] → NT-Xent Loss
                                          ↓
                                   Feature Memory Bank (normal samples)

Inference Phase:
  Image → [ResNet-18 Encoder] → Features → k-NN Distance to Memory Bank → Anomaly Score
                                    ↓
                              Grad-CAM Heatmap → Anomaly Localization
```

---

## 📁 Project Structure

```
BSC_PROJECT/
├── configs/
│   └── config.yaml              # All hyperparameters & paths
├── data/
│   └── mvtec_ad/                # MVTec AD dataset (15 categories)
├── src/
│   ├── __init__.py
│   ├── augmentations.py         # SimCLR augmentation pipeline
│   ├── dataset.py               # MVTec AD Dataset & DataLoaders
│   ├── model.py                 # ResNet-18 encoder + projection head
│   ├── losses.py                # NT-Xent contrastive loss
│   ├── trainer.py               # SimCLR training loop
│   ├── memory_bank.py           # Feature memory bank + k-NN scoring
│   ├── evaluator.py             # Metrics & evaluation plots
│   ├── gradcam.py               # Grad-CAM anomaly localization
│   └── utils.py                 # Utilities (config, logging, etc.)
├── scripts/
│   ├── download_dataset.py      # Download MVTec AD from Kaggle
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation entry point
│   └── inference.py             # Single image inference
├── app/
│   └── streamlit_app.py         # Interactive web interface
├── outputs/                     # Checkpoints, logs, results
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd BSC_PROJECT
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MVTec AD Dataset

Option A — Using the download script (requires Kaggle account):
```bash
python scripts/download_dataset.py
```

Option B — Manual download from [Kaggle](https://www.kaggle.com/datasets/alex000kim/mvtec-ad) and extract to `data/mvtec_ad/`.

---

## 📖 Usage

### 1. Train the Model

```bash
# Train on a specific category
python scripts/train.py --category bottle --epochs 100

# Train on all categories
python scripts/train.py

# Quick test (1 epoch)
python scripts/train.py --category bottle --epochs 1
```


### 2. Evaluate Performance

```bash
# Evaluate a specific category
python scripts/evaluate.py --category bottle

# Evaluate all categories
python scripts/evaluate.py
```

### 3. Run Inference

```bash
python scripts/inference.py --image path/to/image.png --category bottle
```

### 4. Launch Web Application

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Results

After training and evaluation, results are saved in `outputs/`:

| Output | Location |
|--------|----------|
| Model checkpoints | `outputs/checkpoints/` |
| Training logs | `outputs/logs/` |
| Metrics & plots | `outputs/results/` |
| Grad-CAM images | `outputs/visualizations/` |

### Metrics Computed
- **AUROC** — Area Under ROC Curve (primary metric)
- **F1-Score** — Harmonic mean of precision and recall
- **Accuracy** — Overall classification accuracy
- **Precision / Recall** — Per-class performance

### Visualizations Generated
- ROC curves per category
- Confusion matrices
- Anomaly score distributions
- Grad-CAM heatmap overlays

---

## 📚 References

1. Chen, T., et al. (2020). **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**. ICML 2020.
2. Bergmann, P., et al. (2019). **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**. CVPR 2019.
3. Selvaraju, R.R., et al. (2017). **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**. ICCV 2017.
4. He, K., et al. (2016). **Deep Residual Learning for Image Recognition (ResNet)**. CVPR 2016.

---

## 📄 License

This project is developed for academic purposes as a BSc final year project.
