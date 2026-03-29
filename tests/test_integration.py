"""
Integration Tests for the Industrial Anomaly Detection System.
Tests the end-to-end workflow with synthetic data to ensure all modules
work together correctly.

Run: pytest tests/test_integration.py -v
"""

import os
import sys
import pytest
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────
# Shared Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_config():
    """Minimal quick-test configuration."""
    tmpdir = tempfile.mkdtemp(prefix="integ_test_")
    data_dir = os.path.join(tmpdir, "data")
    output_dir = os.path.join(tmpdir, "outputs")
    
    # Create synthetic MVTec-style dataset
    train_good = os.path.join(data_dir, "bottle", "train", "good")
    test_good = os.path.join(data_dir, "bottle", "test", "good")
    test_defect = os.path.join(data_dir, "bottle", "test", "broken_large")
    os.makedirs(train_good)
    os.makedirs(test_good)
    os.makedirs(test_defect)
    
    # Generate synthetic training images (normal)
    np.random.seed(42)
    for i in range(16):
        # Normal images: smooth gradients
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(100, 200, 64).reshape(1, -1).repeat(64, axis=0).astype(np.uint8)
        arr[:, :, 1] = np.linspace(80, 180, 64).reshape(-1, 1).repeat(64, axis=1).astype(np.uint8)
        arr[:, :, 2] = 120
        # Add slight noise
        noise = np.random.randint(-10, 10, (64, 64, 3))
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(train_good, f"normal_{i:03d}.png"))
    
    # Normal test images
    for i in range(6):
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.linspace(100, 200, 64).reshape(1, -1).repeat(64, axis=0).astype(np.uint8)
        arr[:, :, 1] = np.linspace(80, 180, 64).reshape(-1, 1).repeat(64, axis=1).astype(np.uint8)
        arr[:, :, 2] = 120
        noise = np.random.randint(-15, 15, (64, 64, 3))
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(test_good, f"normal_{i:03d}.png"))
    
    # Anomalous test images (very different pattern)
    for i in range(6):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Add a noticeable "defect" patch
        arr[20:40, 20:40, :] = 255  # bright white square
        Image.fromarray(arr).save(os.path.join(test_defect, f"defect_{i:03d}.png"))
    
    config = {
        "dataset": {
            "name": "mvtec_ad",
            "root_dir": data_dir,
            "image_size": 224,
            "num_workers": 0,
            "categories": ["bottle"],
        },
        "model": {
            "backbone": "resnet18",
            "pretrained": False,
            "feature_dim": 512,
            "projection_dim": 128,
            "projection_hidden_dim": 256,
        },
        "training": {
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "temperature": 0.07,
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 1,
            "use_amp": False,
            "gradient_clip_max_norm": 1.0,
            "checkpoint_interval": 1,
            "seed": 42,
        },
        "augmentations": {
            "random_crop": {"enabled": True, "scale": [0.5, 1.0]},
            "horizontal_flip": {"enabled": True, "probability": 0.5},
            "color_jitter": {"enabled": True, "probability": 0.8, "brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1},
            "grayscale": {"enabled": True, "probability": 0.2},
            "gaussian_blur": {"enabled": True, "probability": 0.5, "kernel_size": 23},
            "normalize": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "anomaly_detection": {
            "method": "knn",
            "k_neighbors": 3,
            "score_threshold": 0.5,
        },
        "gradcam": {
            "target_layer": "layer4",
            "colormap": "jet",
        },
        "output": {
            "root_dir": output_dir,
            "checkpoints_dir": os.path.join(output_dir, "checkpoints"),
            "logs_dir": os.path.join(output_dir, "logs"),
            "results_dir": os.path.join(output_dir, "results"),
            "visualizations_dir": os.path.join(output_dir, "visualizations"),
        },
    }
    
    # Ensure output dirs
    for d in config["output"].values():
        os.makedirs(d, exist_ok=True)
    
    yield config
    
    # Cleanup
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────
# Integration Test 1: End-to-End Training Pipeline
# ─────────────────────────────────────────────────────────────────

class TestTrainingPipeline:
    """Integration tests for the full training workflow."""
    
    def test_full_training_loop(self, test_config):
        """
        End-to-end test: create data → train SimCLR → save checkpoint.
        This is the core integration test.
        """
        from src.utils.utils import set_seed
        from src.models.simclr import SimCLRModel
        from src.training.dataset import create_train_dataloader
        from src.training.trainer import SimCLRTrainer
        
        set_seed(42)
        device = torch.device("cpu")
        config = test_config
        
        # Create training dataloader
        train_dl = create_train_dataloader(config, "bottle")
        assert len(train_dl) > 0, "Training dataloader should have batches"
        
        # Initialize model
        model = SimCLRModel(config)
        
        # Initialize trainer
        trainer = SimCLRTrainer(model, config, device)
        
        # Train for a few epochs
        history = trainer.train(train_dl, "bottle")
        
        # Validate training history
        assert "epoch_losses" in history
        assert len(history["epoch_losses"]) == config["training"]["epochs"]
        assert all(loss > 0 for loss in history["epoch_losses"]), "All losses should be positive"
        
        # Verify loss decreased (training is working)
        # With only 3 epochs and random init this may not always decrease,
        # but the loss should at least be finite
        assert all(np.isfinite(loss) for loss in history["epoch_losses"]), "Losses should be finite"
        
        # Verify checkpoint was saved
        checkpoint_dir = config["output"]["checkpoints_dir"]
        best_ckpt = os.path.join(checkpoint_dir, "bottle_best.pt")
        assert os.path.exists(best_ckpt), "Best checkpoint should be saved"
    
    def test_memory_bank_construction(self, test_config):
        """Build memory bank from trained model features."""
        from src.utils.utils import set_seed
        from src.models.simclr import SimCLRModel
        from src.training.dataset import create_feature_dataloader
        from src.memory.memory_bank import MemoryBank
        
        set_seed(42)
        device = torch.device("cpu")
        config = test_config
        
        model = SimCLRModel(config)
        model.eval()
        
        # Create feature dataloader (no augmentation)
        feature_dl = create_feature_dataloader(config, "bottle")
        
        # Build memory bank
        bank = MemoryBank()
        bank.build(model, feature_dl, device)
        
        assert bank.features is not None
        assert bank.count == 16  # 16 training images
        assert bank.features.shape == (16, 512)
        
        # Features should be L2-normalized
        norms = torch.norm(bank.features, dim=1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5), \
            "Memory bank features should be L2-normalized"
        
        # Save and reload
        bank_path = os.path.join(config["output"]["checkpoints_dir"], "bottle_mem_bank.pt")
        bank.save(bank_path)
        
        bank2 = MemoryBank()
        bank2.load(bank_path)
        assert torch.allclose(bank.features, bank2.features)


# ─────────────────────────────────────────────────────────────────
# Integration Test 2: End-to-End Evaluation Pipeline
# ─────────────────────────────────────────────────────────────────

class TestEvaluationPipeline:
    """Integration tests for the full evaluation workflow."""
    
    def test_full_evaluation_pipeline(self, test_config):
        """
        End-to-end: model → features → anomaly scoring → metrics & plots.
        """
        from src.utils.utils import set_seed
        from src.models.simclr import SimCLRModel
        from src.training.dataset import create_test_dataloader, create_feature_dataloader
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        from src.evaluation.evaluator import AnomalyEvaluator
        
        set_seed(42)
        device = torch.device("cpu")
        config = test_config
        
        # Initialize model
        model = SimCLRModel(config)
        model.eval()
        
        # Build memory bank
        feature_dl = create_feature_dataloader(config, "bottle")
        bank = MemoryBank()
        bank.build(model, feature_dl, device)
        
        # Create scorer
        scorer = AnomalyScorer(method="knn", k_neighbors=3)
        scorer.fit(bank)
        
        # Score test set
        test_dl = create_test_dataloader(config, "bottle")
        scores, labels = scorer.score_batch(model, test_dl, device)
        
        assert len(scores) == 12  # 6 normal + 6 anomalous
        assert len(labels) == 12
        assert sum(labels) == 6   # 6 anomalous
        assert all(np.isfinite(scores)), "All scores should be finite"
        
        # Run evaluation
        results_dir = config["output"]["results_dir"]
        evaluator = AnomalyEvaluator(output_dir=results_dir)
        metrics = evaluator.generate_full_report(scores, labels, "bottle")
        
        assert 0 <= metrics["auroc"] <= 1.0
        assert 0 <= metrics["f1_score"] <= 1.0
        assert metrics["total_samples"] == 12
        assert metrics["num_normal"] == 6
        assert metrics["num_anomaly"] == 6
        
        # Check plots were generated
        assert os.path.exists(metrics["plots"]["roc_curve"])
        assert os.path.exists(metrics["plots"]["confusion_matrix"])
        assert os.path.exists(metrics["plots"]["score_distribution"])
        
        # Check JSON was saved
        json_path = os.path.join(results_dir, "bottle_metrics.json")
        assert os.path.exists(json_path)


# ─────────────────────────────────────────────────────────────────
# Integration Test 3: Inference Pipeline
# ─────────────────────────────────────────────────────────────────

class TestInferencePipeline:
    """Integration tests for single-image inference."""
    
    def test_single_image_inference(self, test_config):
        """Simulate full inference pipeline on a single image."""
        from src.models.simclr import SimCLRModel
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        from src.inference.gradcam import GradCAM
        from src.training.augmentations import get_eval_transform
        from src.training.dataset import create_feature_dataloader
        
        device = torch.device("cpu")
        config = test_config
        
        # Setup model
        model = SimCLRModel(config)
        model.eval()
        
        # Build memory bank
        feature_dl = create_feature_dataloader(config, "bottle")
        bank = MemoryBank()
        bank.build(model, feature_dl, device)
        
        scorer = AnomalyScorer(method="knn", k_neighbors=3)
        scorer.fit(bank)
        
        # Setup Grad-CAM
        gradcam = GradCAM(model, target_layer_name="layer4")
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        
        # Process
        eval_transform = get_eval_transform(config)
        input_tensor = eval_transform(test_img).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = model.encode(input_tensor)
        
        # Score
        score = scorer.score(features.cpu())
        assert score.shape == (1,)
        assert np.isfinite(score[0])
        
        # Generate heatmap
        heatmap = gradcam.generate(input_tensor, device)
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1.0
        
        # Generate overlay
        original_np = np.array(test_img.resize((224, 224)))
        overlay = gradcam.generate_overlay(original_np, heatmap)
        assert overlay.shape == (224, 224, 3)
        
        # Save visualization
        vis_dir = config["output"]["visualizations_dir"]
        save_path = os.path.join(vis_dir, "test_inference.png")
        
        threshold = config["anomaly_detection"]["score_threshold"]
        label = "Anomaly" if score[0] >= threshold else "Normal"
        
        gradcam.visualize(
            original_np, heatmap,
            anomaly_score=float(score[0]),
            label=label,
            save_path=save_path,
        )
        assert os.path.exists(save_path), "Visualization should be saved"
    
    def test_batch_inference(self, test_config):
        """Test batch inference with multiple images."""
        from src.models.simclr import SimCLRModel
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        from src.training.dataset import create_test_dataloader, create_feature_dataloader
        
        device = torch.device("cpu")
        config = test_config
        
        model = SimCLRModel(config)
        model.eval()
        
        # Build memory bank
        feature_dl = create_feature_dataloader(config, "bottle")
        bank = MemoryBank()
        bank.build(model, feature_dl, device)
        
        scorer = AnomalyScorer(method="knn", k_neighbors=3)
        scorer.fit(bank)
        
        # Score the entire test set
        test_dl = create_test_dataloader(config, "bottle")
        scores, labels = scorer.score_batch(model, test_dl, device)
        
        assert len(scores) == 12
        assert len(labels) == 12


# ─────────────────────────────────────────────────────────────────
# Integration Test 4: Mahalanobis Distance Alternative
# ─────────────────────────────────────────────────────────────────

class TestMahalanobisPipeline:
    """Tests the Mahalanobis distance scoring alternative."""
    
    def test_mahalanobis_evaluation(self, test_config):
        """End-to-end with Mahalanobis scoring instead of k-NN."""
        from src.models.simclr import SimCLRModel
        from src.training.dataset import create_test_dataloader, create_feature_dataloader
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        
        device = torch.device("cpu")
        config = test_config
        
        model = SimCLRModel(config)
        model.eval()
        
        feature_dl = create_feature_dataloader(config, "bottle")
        bank = MemoryBank()
        bank.build(model, feature_dl, device)
        
        # Use Mahalanobis
        scorer = AnomalyScorer(method="mahalanobis", k_neighbors=3)
        scorer.fit(bank)
        
        test_dl = create_test_dataloader(config, "bottle")
        scores, labels = scorer.score_batch(model, test_dl, device)
        
        assert len(scores) == 12
        assert all(np.isfinite(scores)), "Mahalanobis scores should be finite"


# ─────────────────────────────────────────────────────────────────
# Integration Test 5: Config-Driven Workflow
# ─────────────────────────────────────────────────────────────────

class TestConfigDriven:
    """Test that the config file properly drives all components."""
    
    def test_config_file_loads_correctly(self):
        """The actual config.yaml should load and have required fields."""
        from src.utils.utils import load_config
        config = load_config("configs/config.yaml")
        
        # Required sections
        assert "dataset" in config
        assert "model" in config
        assert "training" in config
        assert "augmentations" in config
        assert "anomaly_detection" in config
        assert "output" in config
        
        # Required fields
        assert config["dataset"]["image_size"] == 224
        assert config["model"]["feature_dim"] == 512
        assert config["model"]["projection_dim"] == 128
        assert config["training"]["temperature"] == 0.07
        assert len(config["dataset"]["categories"]) == 15
    
    def test_all_augmentations_configurable(self, test_config):
        """All augmentation options should be respected."""
        from src.training.augmentations import ContrastiveAugmentation
        
        # Disable all optional augmentations
        config = test_config.copy()
        config["augmentations"] = {
            "random_crop": {"enabled": False},
            "horizontal_flip": {"enabled": False},
            "color_jitter": {"enabled": False},
            "grayscale": {"enabled": False},
            "gaussian_blur": {"enabled": False},
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        }
        
        aug = ContrastiveAugmentation(config)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        v1, v2 = aug(img)
        assert v1.shape == (3, 224, 224)
