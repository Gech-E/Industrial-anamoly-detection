"""
Streamlit Web Application for Industrial Anomaly Detection.
Features PatchCore anomaly heatmaps, calibrated confidence scores,
and interactive visualization.

Usage: streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.utils import load_config, get_device, load_checkpoint
from src.models.simclr import SimCLRModel
from src.memory.memory_bank import (
    MemoryBank, AnomalyScorer,
    PatchMemoryBank, PatchAnomalyScorer,
)
from src.scoring.calibration import ScoreCalibrator
from src.visualization.heatmap import AnomalyHeatmapGenerator
from src.training.augmentations import get_eval_transform


# ── Page Configuration ──
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #fff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #b8b8d1;
        font-size: 1.1rem;
    }

    .result-normal {
        background: linear-gradient(135deg, #0d3b26 0%, #1a5c3a 100%);
        border: 2px solid #4CAF50;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-anomaly {
        background: linear-gradient(135deg, #3b0d0d 0%, #5c1a1a 100%);
        border: 2px solid #F44336;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

    .confidence-bar {
        background: #1a1a2e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #F44336; font-weight: 700; }
    .confidence-medium { color: #FF9800; font-weight: 700; }
    .confidence-low { color: #4CAF50; font-weight: 700; }

    .stSidebar { background-color: #0f0c29; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading (Cached) ──
@st.cache_resource
def load_pipeline(config_path, category):
    """Load the full anomaly detection pipeline (cached)."""
    config = load_config(config_path)
    device = get_device()

    checkpoint_dir = config.get("output", {}).get(
        "checkpoints_dir", "outputs/checkpoints"
    )

    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
    if not os.path.exists(checkpoint_path):
        return None, None, None, None, None, None, config, device, False

    # Load model
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    state_dict = checkpoint.get("model_state_dict", {})
    is_resnet50 = any("conv3" in key for key in state_dict.keys())

    if "model" not in config:
        config["model"] = {}
    config["model"]["backbone"] = "resnet50" if is_resnet50 else "resnet18"
    config["model"]["feature_dim"] = 2048 if is_resnet50 else 512

    model = SimCLRModel(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Check if PatchCore pipeline is available
    patch_bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")
    use_patch = os.path.exists(patch_bank_path)

    patch_scorer = None
    global_scorer = None
    calibrator = None
    heatmap_gen = None

    if use_patch:
        # PatchCore pipeline
        patch_bank = PatchMemoryBank()
        patch_bank.load(patch_bank_path)

        scoring_cfg = config.get("scoring", {})
        patch_scorer = PatchAnomalyScorer(
            k_neighbors=scoring_cfg.get("k_neighbors", 3),
            weight_knn=scoring_cfg.get("weight_knn", 1.0),
            weight_mahalanobis=scoring_cfg.get("weight_mahalanobis", 0.0),
            weight_cosine=scoring_cfg.get("weight_cosine", 0.0),
        )
        patch_scorer.fit(patch_bank)

        loc_cfg = config.get("localization", {})
        heatmap_gen = AnomalyHeatmapGenerator(
            sigma=loc_cfg.get("gaussian_sigma", 4.0),
            colormap=loc_cfg.get("colormap", "jet"),
            alpha=loc_cfg.get("overlay_alpha", 0.4),
        )
    else:
        # Global pipeline fallback
        bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
        if os.path.exists(bank_path):
            memory_bank = MemoryBank()
            memory_bank.load(bank_path)

            ad_cfg = config.get("anomaly_detection", {})
            global_scorer = AnomalyScorer(
                method=ad_cfg.get("method", "mahalanobis"),
                k_neighbors=ad_cfg.get("k_neighbors", 5),
            )
            global_scorer.fit(memory_bank)

    # Calibrator
    calibrator = ScoreCalibrator(
        method=config.get("calibration", {}).get("method", "minmax_sigmoid")
    )

    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    metrics_path = os.path.join(results_dir, f"{category}_metrics.json")
    threshold = 0.5

    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            threshold = float(metrics.get("threshold", 0.5))
            if "calibration" in metrics:
                calibrator.load_params(metrics["calibration"])
        except Exception:
            pass

    return (model, patch_scorer, global_scorer, calibrator,
            heatmap_gen, threshold, config, device, use_patch)


def predict_image_patch(image, model, patch_scorer, calibrator,
                        heatmap_gen, threshold, config, device):
    """Run PatchCore prediction."""
    eval_transform = get_eval_transform(config)
    img_size = config.get("dataset", {}).get("image_size", 224)

    img_resized = image.resize((img_size, img_size))
    original_np = np.array(img_resized)
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        patch_features, patch_shape = model.extract_patch_features(input_tensor)

    patches = patch_features[0].cpu()
    patch_scores, image_score = patch_scorer.score_patches(patches)

    is_anomaly = image_score >= threshold
    label = "Anomaly" if is_anomaly else "Normal"
    confidence = calibrator.calibrate(image_score) if calibrator._fitted else 0.5

    heatmap = heatmap_gen.generate(
        patch_scores, patch_shape, image_size=(img_size, img_size)
    )
    overlay = heatmap_gen.overlay(original_np, heatmap)

    return {
        "score": image_score,
        "threshold": threshold,
        "label": label,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original_np,
    }


def predict_image_global(image, model, global_scorer, calibrator,
                         threshold, config, device):
    """Run global feature prediction (fallback)."""
    eval_transform = get_eval_transform(config)
    img_size = config.get("dataset", {}).get("image_size", 224)

    img_resized = image.resize((img_size, img_size))
    original_np = np.array(img_resized)
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, "extract_features"):
            features = model.extract_features(input_tensor)
        else:
            features = model.encode(input_tensor)

    features_cpu = features.cpu()
    anomaly_score = float(global_scorer.score(features_cpu)[0])

    is_anomaly = anomaly_score >= threshold
    label = "Anomaly" if is_anomaly else "Normal"
    confidence = calibrator.calibrate(anomaly_score) if calibrator._fitted else 0.5

    # Generate Grad-CAM heatmap as fallback
    heatmap = None
    overlay = None
    try:
        from src.inference.gradcam import GradCAM
        gradcam = GradCAM(model, target_layer_name="layer3")
        heatmap = gradcam.generate(input_tensor, device)
        import cv2
        heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = np.clip(
            np.float32(heatmap_colored) * 0.4 + np.float32(original_np) * 0.6,
            0, 255
        ).astype(np.uint8)
    except Exception:
        pass

    return {
        "score": anomaly_score,
        "threshold": threshold,
        "label": label,
        "confidence": confidence,
        "confidence_pct": round(confidence * 100, 1),
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original_np,
    }


# ── Main App ──
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Industrial Anomaly Detection</h1>
        <p>PatchCore-based anomaly detection with heatmap localization & calibrated confidence</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        config_path = st.text_input("Config Path", value="configs/config.yaml")

        try:
            config = load_config(config_path)
            categories = config["dataset"]["categories"]
        except Exception:
            st.error("Could not load config file!")
            categories = ["bottle"]
            config = {}

        category = st.selectbox(
            "🏷️ Product Category",
            categories,
            help="Select the MVTec AD product category",
        )

        st.markdown("---")
        st.markdown("### 📊 Model Info")

        checkpoint_dir = config.get("output", {}).get(
            "checkpoints_dir", "outputs/checkpoints"
        )
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        patch_bank_path = os.path.join(checkpoint_dir, f"{category}_patch_bank.pt")

        if os.path.exists(checkpoint_path):
            st.success(f"✅ Model loaded for **{category}**")
            if os.path.exists(patch_bank_path):
                st.info("🧩 PatchCore pipeline active")
            else:
                st.info("📊 Global feature pipeline")
        else:
            st.warning(f"⚠️ No trained model for **{category}**")
            st.info(
                "Run training first:\n```\n"
                f"python scripts/train.py --category {category}\n```"
            )

        st.markdown("---")
        st.markdown(
            "### 🎓 About\n"
            "BSc Final Year Project\n\n"
            "**Method:** PatchCore + Hybrid Scoring\n\n"
            "**Backbone:** ResNet-50 (ImageNet)\n\n"
            "**Dataset:** MVTec AD\n\n"
            "**Features:**\n"
            "- Patch-level anomaly detection\n"
            "- Anomaly heatmap localization\n"
            "- Calibrated confidence scores"
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image for anomaly detection",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a product image to check for anomalies",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width="stretch")

    with col2:
        st.markdown("### 🔬 Detection Results")

        if uploaded_file is not None:
            pipeline = load_pipeline(config_path, category)
            (model, patch_scorer, global_scorer, calibrator,
             heatmap_gen, threshold, config, device, use_patch) = pipeline

            if model is None:
                st.error(
                    f"❌ No trained model found for **{category}**.\n\n"
                    f"Please train first:\n```\n"
                    f"python scripts/train.py --category {category}\n```"
                )
            elif patch_scorer is None and global_scorer is None:
                st.error("❌ No scorer available. Re-train the model.")
            else:
                with st.spinner("🔄 Analyzing image..."):
                    if use_patch and patch_scorer is not None:
                        result = predict_image_patch(
                            image, model, patch_scorer, calibrator,
                            heatmap_gen, threshold, config, device
                        )
                    else:
                        result = predict_image_global(
                            image, model, global_scorer, calibrator,
                            threshold, config, device
                        )

                # Display result
                if result["label"] == "Normal":
                    st.markdown(f"""
                    <div class="result-normal">
                        <h2 style="color: #4CAF50;">✅ NORMAL</h2>
                        <p style="color: #a8d5ba;">
                            Confidence: {result['confidence_pct']:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-anomaly">
                        <h2 style="color: #F44336;">⚠️ ANOMALY DETECTED</h2>
                        <p style="color: #f5a8a8;">
                            Confidence: {result['confidence_pct']:.1f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # Metrics
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Anomaly Score", f"{result['score']:.4f}")
                with m2:
                    st.metric("Threshold", f"{result['threshold']:.4f}")
                with m3:
                    # Color-coded confidence
                    conf = result["confidence_pct"]
                    st.metric("Confidence", f"{conf:.1f}%")

                # Score progress bar
                st.markdown("**Score vs Threshold:**")
                score_norm = min(
                    result["score"] / max(result["threshold"] * 2, 1e-6), 1.0
                )
                st.progress(
                    score_norm,
                    text=(
                        f"Score: {result['score']:.4f} | "
                        f"Threshold: {result['threshold']:.4f}"
                    ),
                )

    # Visualization section
    if uploaded_file is not None and model is not None and (
        patch_scorer is not None or global_scorer is not None
    ):
        st.markdown("---")

        if use_patch:
            st.markdown("### 🗺️ Anomaly Localization (PatchCore Heatmap)")
        else:
            st.markdown("### 🗺️ Anomaly Localization (Grad-CAM)")

        v1, v2, v3 = st.columns(3)

        with v1:
            st.image(
                result["original"],
                caption="Original Image",
                width="stretch",
            )

        with v2:
            if result.get("heatmap") is not None:
                fig, ax = plt.subplots(figsize=(6, 6))
                cmap = config.get("localization", {}).get("colormap", "jet")
                im = ax.imshow(result["heatmap"], cmap=cmap, vmin=0, vmax=1)
                ax.axis("off")
                ax.set_title("Anomaly Heatmap", fontsize=14, fontweight="bold")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No heatmap available")

        with v3:
            if result.get("overlay") is not None:
                st.image(
                    result["overlay"],
                    caption="Overlay Visualization",
                    width="stretch",
                )
            else:
                st.info("No overlay available")

    # Performance dashboard
    st.markdown("---")
    st.markdown("### 📈 Model Performance Dashboard")

    results_dir = config.get("output", {}).get("results_dir", "outputs/results")

    if os.path.exists(results_dir):
        metric_files = [
            f for f in os.listdir(results_dir) if f.endswith("_metrics.json")
        ]

        if metric_files:
            perf_data = []
            for mf in metric_files:
                try:
                    with open(os.path.join(results_dir, mf)) as f:
                        data = json.load(f)
                        perf_data.append(data)
                except Exception:
                    pass

            if perf_data:
                import pandas as pd

                df = pd.DataFrame([
                    {
                        "Category": d["category"],
                        "Pipeline": d.get("pipeline", "global"),
                        "AUROC": d["auroc"],
                        "F1-Score": d["f1_score"],
                        "Accuracy": d["accuracy"],
                        "Precision": d["precision"],
                        "Recall": d["recall"],
                    }
                    for d in perf_data
                ])

                st.dataframe(
                    df.style.format({
                        "AUROC": "{:.4f}",
                        "F1-Score": "{:.4f}",
                        "Accuracy": "{:.4f}",
                        "Precision": "{:.4f}",
                        "Recall": "{:.4f}",
                    }),
                    width="stretch",
                )

                # Mean AUROC highlight
                mean_auroc = df["AUROC"].mean()
                st.markdown(
                    f"**Mean AUROC: {mean_auroc:.4f}** "
                    f"({'✅ Target met!' if mean_auroc >= 0.90 else '⚠️ Below 0.90 target'})"
                )

                # Show ROC curves if available
                roc_files = [
                    f for f in os.listdir(results_dir)
                    if f.endswith("_roc_curve.png")
                ]
                if roc_files:
                    st.markdown("#### ROC Curves")
                    cols = st.columns(min(len(roc_files), 3))
                    for i, rf in enumerate(roc_files[:6]):
                        with cols[i % 3]:
                            st.image(
                                os.path.join(results_dir, rf),
                                width="stretch",
                            )
        else:
            st.info(
                "📋 No evaluation results yet. Run evaluation first:\n"
                "```\npython scripts/evaluate.py\n```"
            )
    else:
        st.info(
            "📋 No results directory found. "
            "Train and evaluate the model first."
        )


if __name__ == "__main__":
    main()
