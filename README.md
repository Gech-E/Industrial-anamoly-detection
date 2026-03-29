# 🔍 Vision-Based Industrial Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

A production-ready, CPU-optimized pipeline for detecting anomalies in industrial product images. The system learns robust feature representations from normal product images and identifies anomalies using multi-layer feature extraction and Mahalanobis distance scoring.

> **BSc Final Year Project** — Electrical & Computer Engineering

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Setup & Installation](#-setup--installation)
- [Usage (CLI)](#-usage-cli)
- [Usage (API)](#-usage-api)
- [Web Application](#-web-application)
- [Results](#-results)
- [References](#-references)

---

## 🎯 Overview

**Problem:** Industrial quality control requires detecting product defects, but labeled anomalous samples are extremely scarce or unavailable during training.

**Solution:** A highly optimized self-supervised pipeline that extracts multi-layer features (layer2, layer3, and layer4) from a pretrained **ResNet-50** backbone using only normal images. Anomalies are detected by measuring the **Mahalanobis distance** of test features from the normal distribution's regularized covariance matrix.

This approach routinely achieves **> 0.95 AUROC** on the MVTec AD dataset, even on heavily restricted CPU-only platforms.

---

## 🏗️ Architecture

```text
Training/Building Phase:
  Normal Image → [ResNet-50 Encoder] → Multi-Layer Spatial Features
                                              ↓
                                      Average Pooling & L2 Normalization
                                              ↓
                                      [Optional] PCA Reduction & Coreset Subsampling
                                              ↓
                         Fit Mahalanobis Scorer (Mean & Cholesky Covariance Inverse)

Inference Phase:
  Test Image → [ResNet-50 Encoder] → Features → Mahalanobis Distance → Anomaly Score
                                                                          ↓
                                                                   Thresholding (Normal/Anomaly)
                                                                          ↓
                                     [Optional] Grad-CAM Layer 3 Heatmap -> Localization
```

*(Note: The pipeline also fully supports full contrastive pre-training using SimCLR with NT-Xent loss via the `--train-simclr` flag).*

---

## ✨ Key Features

- 🚀 **Production-Ready Inference API**: Clean `AnomalyPredictor` class for single (`predict()`) and batch (`predict_batch()`) inference.
- 🧠 **Multi-Layer Feature Extraction**: Concatenates intermediate ResNet-50 layers (3584-dimensional features) for rich structural representation.
- 📐 **Mahalanobis Distance Scoring**: Robust, statistically grounded anomaly scoring using Ledoit-Wolf shrinkage and Cholesky decomposition.
- ⚡ **CPU-Optimized**: Fully optimized for CPU execution with native multiprocessing fixes (`num_workers=0`), fast vectorized Mahalanobis distance, and PCA dimensionality reduction.
- ⏱️ **Early Stopping & Checkpoint Resume**: Train safely without wasting compute cycles.
- 🗺️ **Grad-CAM Localization**: Automatically generates heatmaps highlighting exactly *where* the model sees an anomaly.
- 🌐 **Interactive Streamlit Web App**: Beautiful, user-friendly interface for uploading photos and getting instant predictions and heatmaps.

---

## 🚀 Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Gech-E/Industrial-anamoly-detection.git
cd Industrial-anamoly-detection
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download MVTec AD Dataset

Option A — Using the download script (requires Kaggle API credentials):
```bash
python scripts/download_dataset.py
```

Option B — Manual download from [Kaggle](https://www.kaggle.com/datasets/alex000kim/mvtec-ad) and extract to `Data/mvtec_ad/`.

---

## 📖 Usage (CLI)

### 1. Train / Build the Pipeline

By default, the pipeline runs in **Feature Extraction Only** mode, which directly utilizes representations from the ImageNet-pretrained backbone. This is the fastest and most accurate approach for most industrial datasets.

```bash
# Build the memory bank and scorer for the 'bottle' category
python scripts/train.py --category bottle

# Run full SimCLR contrastive training (optional)
python scripts/train.py --category bottle --train-simclr

# Resume SimCLR training from the last checkpoint
python scripts/train.py --category bottle --train-simclr --resume
```

### 2. Evaluate Performance

Evaluates the trained scorer against the test set, computing AUROC, F1, Average Precision, and generating ROC/PR curves and score distributions.

```bash
python scripts/evaluate.py --category bottle
```

### 3. Run Inference via CLI

```bash
# Single image inference
python scripts/inference.py --image path/to/image.png --category bottle

# Batch inference on a directory
python scripts/inference.py --image_dir path/to/images/ --category bottle
```

---

## 💻 Usage (API)

The inference API is designed to be cleanly integrated into other production systems:

```python
from src.inference.predictor import AnomalyPredictor
from PIL import Image

# 1. Initialize the predictor (auto-loads model, thresholds, and scorer)
predictor = AnomalyPredictor.from_config(
    config_path="configs/config.yaml", 
    category="bottle"
)

# 2. Run single prediction
img = Image.open("test_image.png")
result = predictor.predict(img)

print(f"Prediction: {result['label']}")         # "Normal" or "Anomaly"
print(f"Anomaly Score: {result['score']:.4f}")  
print(f"Confidence: {result['confidence']:.1%}") 

# 3. Run batch prediction
results = predictor.predict_batch([img1, img2, img3])
```

---

## 🌐 Web Application

Launch the interactive web dashboard to test images and view Grad-CAM heatmaps directly in your browser:

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Results

With the multi-layer feature extraction and Mahalanobis scoring upgrade, the pipeline achieves significant performance on CPU architectures:

**Example Results (MVTec AD - Bottle):**
- **AUROC**: ~0.9587
- **Average Precision**: ~0.9877
- **F1-Score**: ~0.9153

After evaluation, detailed reports and visualizations are saved in `outputs/`:
- `outputs/results/` — JSON metrics, ROC curves, PR curves, Confusion Matrices, and Score Distributions.
- `outputs/visualizations/` — Grad-CAM heatmaps showing anomaly localization.

---

## 📚 References

1. Cohen, N., & Hoshen, Y. (2020). **Sub-Image Anomaly Detection with Deep Pyramid Correspondences (SPADE)**. *Mahalanobis distance & multi-layer features application*.
2. Defard, T., et al. (2021). **PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization**.
3. Chen, T., et al. (2020). **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**. ICML 2020.
4. Bergmann, P., et al. (2019). **MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection**. CVPR 2019.
5. He, K., et al. (2016). **Deep Residual Learning for Image Recognition (ResNet)**. CVPR 2016.

---

## 📄 License

This project is developed for academic purposes as a BSc final year project.
