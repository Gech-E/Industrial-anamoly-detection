"""
Streamlit Web Application for Industrial Anomaly Detection.
Provides an interactive interface for uploading images and getting anomaly predictions.

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
import matplotlib.pyplot as plt

from src.utils.utils import load_config, get_device, load_checkpoint
from src.models.simclr import SimCLRModel
from src.memory.memory_bank import MemoryBank, AnomalyScorer
from src.inference.gradcam import GradCAM
from src.training.augmentations import get_eval_transform


#  Page Configuration 
st.set_page_config(
    page_title="Industrial Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS 
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
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2d2d5e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #64ffda;
    }
    .metric-card .label {
        font-size: 0.9rem;
        color: #9e9eb8;
        margin-top: 4px;
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
    
    .stSidebar { background-color: #0f0c29; }
</style>
""", unsafe_allow_html=True)


#  Session State ─
@st.cache_resource
def load_model_and_scorer(config_path, category):
    """Cache model and scorer to avoid reloading on every interaction."""
    config = load_config(config_path)
    device = get_device()
    
    checkpoint_dir = config.get("output", {}).get("checkpoints_dir", "outputs/checkpoints")
    
    checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
    
    if not os.path.exists(checkpoint_path):
        return None, None, None, None, config, device
        
    # Auto-detect backbone from checkpoint keys
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", {})
    
    # ResNet-50 uses Bottleneck blocks with conv3, ResNet-18 does not
    is_resnet50 = any("conv3" in key for key in state_dict.keys())
    
    if "model" not in config:
        config["model"] = {}
        
    config["model"]["backbone"] = "resnet50" if is_resnet50 else "resnet18"
    config["model"]["feature_dim"] = 2048 if is_resnet50 else 512
    
    # Try to load calibrated threshold from metrics file
    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    metrics_path = os.path.join(results_dir, f"{category}_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            if "threshold" in metrics:
                if "anomaly_detection" not in config:
                    config["anomaly_detection"] = {}
                config["anomaly_detection"]["score_threshold"] = float(metrics["threshold"])
        except Exception:
            pass
    
    # Load model
    model = SimCLRModel(config)
    
    # We can just load the state_dict directly since we already loaded the checkpoint
    model.load_state_dict(state_dict)
    # Also log normal loading
    import logging
    logging.info(
        f"Checkpoint loaded: {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')}, loss {checkpoint.get('loss', '?'):.4f})"
    )
    
    model = model.to(device)
    model.eval()
    
    # Load memory bank and scorer
    bank_path = os.path.join(checkpoint_dir, f"{category}_memory_bank.pt")
    if not os.path.exists(bank_path):
        return model, None, None, None, config, device
    
    memory_bank = MemoryBank()
    memory_bank.load(bank_path)
    
    ad_cfg = config.get("anomaly_detection", {})
    scorer = AnomalyScorer(
        method=ad_cfg.get("method", "knn"),
        k_neighbors=ad_cfg.get("k_neighbors", 5),
    )
    scorer.fit(memory_bank)
    
    # Grad-CAM
    gradcam = GradCAM(model, target_layer_name="layer4")
    
    return model, scorer, gradcam, memory_bank, config, device


def predict_image(image, model, scorer, gradcam, config, device):
    """Run anomaly detection on a single image."""
    eval_transform = get_eval_transform(config)
    
    # Preprocess
    img_resized = image.resize((224, 224))
    original_np = np.array(img_resized)
    input_tensor = eval_transform(image).unsqueeze(0).to(device)
    
    # Feature extraction and scoring
    with torch.no_grad():
        features = model.encode(input_tensor)
    
    features_cpu = features.cpu()
    anomaly_score = float(scorer.score(features_cpu)[0])
    
    # Threshold
    threshold = config.get("anomaly_detection", {}).get("score_threshold")
    if threshold is None:
        threshold = 0.5  # Fallback if uncalibrated
        
    predicted_label = "Anomaly" if anomaly_score >= threshold else "Normal"
    
    # Grad-CAM heatmap
    heatmap = gradcam.generate(input_tensor, device)
    overlay = gradcam.generate_overlay(original_np, heatmap, alpha=0.4)
    
    return {
        "score": anomaly_score,
        "threshold": threshold,
        "label": predicted_label,
        "heatmap": heatmap,
        "overlay": overlay,
        "original": original_np,
    }


#  Main App 
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Industrial Anomaly Detection</h1>
        <p>Vision-based anomaly detection using SimCLR contrastive learning</p>
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
        
        checkpoint_dir = config.get("output", {}).get("checkpoints_dir", "outputs/checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"{category}_best.pt")
        
        if os.path.exists(checkpoint_path):
            st.success(f"✅ Model loaded for **{category}**")
        else:
            st.warning(f"⚠️ No trained model for **{category}**")
            st.info("Run training first:\n```\npython scripts/train.py --category " + category + "\n```")
        
        st.markdown("---")
        st.markdown(
            "### 🎓 About\n"
            "BSc Final Year Project\n\n"
            "**Method:** SimCLR + k-NN\n\n"
            "**Backbone:** ResNet-18\n\n"
            "**Dataset:** MVTec AD"
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
            # Load model
            model, scorer, gradcam, memory_bank, config, device = load_model_and_scorer(
                config_path, category
            )
            
            if model is None or scorer is None:
                st.error(
                    f"❌ No trained model found for **{category}**.\n\n"
                    f"Please train first:\n```\npython scripts/train.py --category {category}\n```"
                )
            else:
                with st.spinner("🔄 Analyzing image..."):
                    result = predict_image(image, model, scorer, gradcam, config, device)
                
                # Display result
                if result["label"] == "Normal":
                    st.markdown(f"""
                    <div class="result-normal">
                        <h2 style="color: #4CAF50;">✅ NORMAL</h2>
                        <p style="color: #a8d5ba;">Score: {result['score']:.6f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-anomaly">
                        <h2 style="color: #F44336;">⚠️ ANOMALY DETECTED</h2>
                        <p style="color: #f5a8a8;">Score: {result['score']:.6f}</p>
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
                    confidence = abs(result['score'] - result['threshold']) / max(result['threshold'], 1e-6) * 100
                    st.metric("Confidence", f"{min(confidence, 100):.1f}%")
    
    # Visualization section
    if uploaded_file is not None and model is not None and scorer is not None:
        st.markdown("---")
        st.markdown("### 🗺️ Anomaly Localization (Grad-CAM)")
        
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.image(result["original"], caption="Original Image", width="stretch")
        
        with v2:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(result["heatmap"], cmap="jet", vmin=0, vmax=1)
            ax.axis("off")
            ax.set_title("Anomaly Heatmap", fontsize=14, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        
        with v3:
            st.image(result["overlay"], caption="Overlay Visualization", width="stretch")
    
    # Performance dashboard
    st.markdown("---")
    st.markdown("### 📈 Model Performance Dashboard")
    
    results_dir = config.get("output", {}).get("results_dir", "outputs/results")
    
    if os.path.exists(results_dir):
        metric_files = [f for f in os.listdir(results_dir) if f.endswith("_metrics.json")]
        
        if metric_files:
            perf_data = []
            for mf in metric_files:
                with open(os.path.join(results_dir, mf)) as f:
                    data = json.load(f)
                    perf_data.append(data)
            
            if perf_data:
                # Summary table
                import pandas as pd
                df = pd.DataFrame([{
                    "Category": d["category"],
                    "AUROC": d["auroc"],
                    "F1-Score": d["f1_score"],
                    "Accuracy": d["accuracy"],
                    "Precision": d["precision"],
                    "Recall": d["recall"],
                } for d in perf_data])
                
                st.dataframe(df.style.format({
                    "AUROC": "{:.4f}",
                    "F1-Score": "{:.4f}",
                    "Accuracy": "{:.4f}",
                    "Precision": "{:.4f}",
                    "Recall": "{:.4f}",
                }), use_container_width="stretch")
                
                # Show ROC curves if available
                roc_files = [f for f in os.listdir(results_dir) if f.endswith("_roc_curve.png")]
                if roc_files:
                    st.markdown("#### ROC Curves")
                    cols = st.columns(min(len(roc_files), 3))
                    for i, rf in enumerate(roc_files[:6]):
                        with cols[i % 3]:
                            st.image(os.path.join(results_dir, rf), use_container_width="stretch")
        else:
            st.info("📋 No evaluation results yet. Run evaluation first:\n```\npython scripts/evaluate.py\n```")
    else:
        st.info("📋 No results directory found. Train and evaluate the model first.")


if __name__ == "__main__":
    main()
