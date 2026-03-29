"""
Unit Tests for the Industrial Anomaly Detection System.
Tests each module independently with controlled inputs.

Run: pytest tests/test_unit.py -v
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
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    """Minimal test configuration."""
    return {
        "dataset": {
            "name": "mvtec_ad",
            "root_dir": "data/mvtec_ad",
            "image_size": 224,
            "num_workers": 0,
            "categories": ["bottle"],
        },
        "model": {
            "backbone": "resnet18",
            "pretrained": False,  # No ImageNet download in tests
            "feature_dim": 512,
            "projection_dim": 128,
            "projection_hidden_dim": 256,
        },
        "training": {
            "epochs": 2,
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
            "root_dir": "outputs",
            "checkpoints_dir": "outputs/checkpoints",
            "logs_dir": "outputs/logs",
            "results_dir": "outputs/results",
            "visualizations_dir": "outputs/visualizations",
        },
    }


@pytest.fixture
def sample_image():
    """Create a random RGB PIL image."""
    arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def sample_batch():
    """Create a random batch of image tensors."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def tmp_output_dir():
    """Create a temporary output directory that's cleaned up after tests."""
    tmpdir = tempfile.mkdtemp(prefix="anomaly_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def fake_dataset_dir():
    """Create a fake MVTec-style directory structure with synthetic images."""
    tmpdir = tempfile.mkdtemp(prefix="mvtec_test_")
    
    # Create category/train/good
    train_good = os.path.join(tmpdir, "bottle", "train", "good")
    os.makedirs(train_good)
    for i in range(10):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(os.path.join(train_good, f"img_{i:03d}.png"))
    
    # Create category/test/good
    test_good = os.path.join(tmpdir, "bottle", "test", "good")
    os.makedirs(test_good)
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(os.path.join(test_good, f"img_{i:03d}.png"))
    
    # Create category/test/broken_large
    test_defect = os.path.join(tmpdir, "bottle", "test", "broken_large")
    os.makedirs(test_defect)
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(os.path.join(test_defect, f"img_{i:03d}.png"))
    
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────
# Test: src/utils.py
# ─────────────────────────────────────────────────────────────────

class TestUtils:
    """Unit tests for src/utils.py"""
    
    def test_load_config(self):
        """Test YAML config loading."""
        from src.utils.utils import load_config
        config = load_config("configs/config.yaml")
        assert isinstance(config, dict)
        assert "dataset" in config
        assert "model" in config
        assert "training" in config
        assert config["model"]["backbone"] == "resnet18"
    
    def test_set_seed_reproducibility(self):
        """Test that set_seed produces reproducible results."""
        from src.utils.utils import set_seed
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2), "set_seed should produce reproducible tensors"
    
    def test_get_device(self):
        """Test device selection."""
        from src.utils.utils import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")
    
    def test_ensure_dirs(self, config, tmp_output_dir):
        """Test directory creation."""
        from src.utils.utils import ensure_dirs
        config["output"]["root_dir"] = os.path.join(tmp_output_dir, "outputs")
        config["output"]["checkpoints_dir"] = os.path.join(tmp_output_dir, "outputs", "checkpoints")
        config["output"]["logs_dir"] = os.path.join(tmp_output_dir, "outputs", "logs")
        config["output"]["results_dir"] = os.path.join(tmp_output_dir, "outputs", "results")
        config["output"]["visualizations_dir"] = os.path.join(tmp_output_dir, "outputs", "vis")
        ensure_dirs(config)
        assert os.path.isdir(config["output"]["checkpoints_dir"])
        assert os.path.isdir(config["output"]["logs_dir"])
    
    def test_count_parameters(self, config):
        """Test parameter counting."""
        from src.utils.utils import count_parameters
        from src.models.simclr import SimCLRModel
        model = SimCLRModel(config)
        info = count_parameters(model)
        assert info["total"] > 0
        assert info["trainable"] > 0
        assert info["total_millions"] > 0
    
    def test_format_time(self):
        """Test time formatting."""
        from src.utils.utils import format_time
        assert "s" in format_time(30)
        assert "m" in format_time(120)
        assert "h" in format_time(7200)
    
    def test_save_and_load_checkpoint(self, config, tmp_output_dir):
        """Test checkpoint save/load roundtrip."""
        from src.utils.utils import save_checkpoint, load_checkpoint
        from src.models.simclr import SimCLRModel
        
        model = SimCLRModel(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        save_path = os.path.join(tmp_output_dir, "test_checkpoint.pt")
        save_checkpoint(model, optimizer, epoch=5, loss=0.123, save_path=save_path)
        assert os.path.exists(save_path)
        
        model2 = SimCLRModel(config)
        checkpoint = load_checkpoint(model2, save_path)
        assert checkpoint["epoch"] == 5
        assert abs(checkpoint["loss"] - 0.123) < 1e-5
        
        # Verify model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Loaded model weights should match saved model"


# ─────────────────────────────────────────────────────────────────
# Test: src/augmentations.py
# ─────────────────────────────────────────────────────────────────

class TestAugmentations:
    """Unit tests for src/augmentations.py"""
    
    def test_contrastive_augmentation_returns_two_views(self, config, sample_image):
        """ContrastiveAugmentation should return two tensors."""
        from src.training.augmentations import ContrastiveAugmentation
        aug = ContrastiveAugmentation(config)
        view1, view2 = aug(sample_image)
        assert isinstance(view1, torch.Tensor)
        assert isinstance(view2, torch.Tensor)
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
    
    def test_two_views_are_different(self, config, sample_image):
        """The two augmented views should be different (stochastic aug)."""
        from src.training.augmentations import ContrastiveAugmentation
        aug = ContrastiveAugmentation(config)
        view1, view2 = aug(sample_image)
        # They could theoretically be the same, but with all these random augs it's extremely rare
        assert not torch.equal(view1, view2), "Two views should differ due to random augmentations"
    
    def test_eval_transform_output_shape(self, config, sample_image):
        """Eval transform should produce a correctly-shaped tensor."""
        from src.training.augmentations import get_eval_transform
        transform = get_eval_transform(config)
        result = transform(sample_image)
        assert result.shape == (3, 224, 224)
    
    def test_inverse_normalize(self, config):
        """Inverse normalize should approximately reverse normalization."""
        from src.training.augmentations import get_eval_transform, get_inverse_normalize
        transform = get_eval_transform(config)
        inv_norm = get_inverse_normalize(config)
        
        img = Image.fromarray(np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8))
        tensor = transform(img)
        recovered = inv_norm(tensor)
        # Values should be approximately in [0, 1] range
        assert recovered.min() >= -0.5
        assert recovered.max() <= 1.5


# ─────────────────────────────────────────────────────────────────
# Test: src/model.py
# ─────────────────────────────────────────────────────────────────

class TestModel:
    """Unit tests for src/model.py"""
    
    def test_resnet_encoder_output_shape(self, sample_batch):
        """ResNet encoder should output (B, 512) features."""
        from src.models.simclr import ResNetEncoder
        encoder = ResNetEncoder(pretrained=False)
        features = encoder(sample_batch)
        assert features.shape == (4, 512)
    
    def test_projection_head_output_shape(self):
        """Projection head should map 512 → 128."""
        from src.models.simclr import ProjectionHead
        head = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=128)
        x = torch.randn(4, 512)
        out = head(x)
        assert out.shape == (4, 128)
    
    def test_simclr_model_forward(self, config, sample_batch):
        """SimCLR forward should return (features, projections)."""
        from src.models.simclr import SimCLRModel
        model = SimCLRModel(config)
        features, projections = model(sample_batch)
        assert features.shape == (4, 512)
        assert projections.shape == (4, 128)
    
    def test_simclr_model_encode(self, config, sample_batch):
        """SimCLR encode should return features only (no projections)."""
        from src.models.simclr import SimCLRModel
        model = SimCLRModel(config)
        features = model.encode(sample_batch)
        assert features.shape == (4, 512)
    
    def test_model_train_eval_modes(self, config):
        """Model should switch between train and eval modes."""
        from src.models.simclr import SimCLRModel
        model = SimCLRModel(config)
        model.train()
        assert model.training
        model.eval()
        assert not model.training
    
    def test_model_gradient_flow(self, config, sample_batch):
        """Gradients should flow through the model."""
        from src.models.simclr import SimCLRModel
        model = SimCLRModel(config)
        model.train()
        features, projections = model(sample_batch)
        loss = projections.sum()
        loss.backward()
        # Check that encoder conv1 has gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should flow through the model"


# ─────────────────────────────────────────────────────────────────
# Test: src/losses.py
# ─────────────────────────────────────────────────────────────────

class TestLosses:
    """Unit tests for src/losses.py"""
    
    def test_ntxent_loss_returns_scalar(self):
        """NT-Xent loss should return a scalar tensor."""
        from src.training.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        loss = loss_fn(z_i, z_j)
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() > 0, "Loss should be positive"
    
    def test_ntxent_loss_identical_pairs(self):
        """Loss should be low when positive pairs are identical."""
        from src.training.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.randn(8, 128)
        loss_identical = loss_fn(z, z)
        
        z_diff = torch.randn(8, 128)
        loss_different = loss_fn(z, z_diff)
        
        assert loss_identical.item() < loss_different.item(), \
            "Loss should be lower for identical pairs than random pairs"
    
    def test_ntxent_loss_gradient(self):
        """Loss should produce valid gradients."""
        from src.training.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)
        z_i = torch.randn(8, 128, requires_grad=True)
        z_j = torch.randn(8, 128, requires_grad=True)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        assert z_i.grad is not None
        assert z_j.grad is not None
        assert not torch.isnan(z_i.grad).any(), "Gradients should not be NaN"
    
    def test_ntxent_loss_temperature_effect(self):
        """Lower temperature should produce higher loss values."""
        from src.training.losses import NTXentLoss
        z_i = torch.randn(8, 128)
        z_j = torch.randn(8, 128)
        
        loss_low_temp = NTXentLoss(temperature=0.01)(z_i, z_j)
        loss_high_temp = NTXentLoss(temperature=1.0)(z_i, z_j)
        
        # With random inputs, lower temp = sharper distribution = higher loss
        assert loss_low_temp.item() > loss_high_temp.item()
    
    def test_ntxent_loss_batch_size_invariance(self):
        """Loss should work with different batch sizes."""
        from src.training.losses import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)
        
        for bs in [2, 4, 8, 16]:
            z_i = torch.randn(bs, 128)
            z_j = torch.randn(bs, 128)
            loss = loss_fn(z_i, z_j)
            assert not torch.isnan(loss), f"Loss should not be NaN for batch_size={bs}"
            assert loss.item() > 0, f"Loss should be positive for batch_size={bs}"


# ─────────────────────────────────────────────────────────────────
# Test: src/dataset.py
# ─────────────────────────────────────────────────────────────────

class TestDataset:
    """Unit tests for src/dataset.py"""
    
    def test_mvtec_dataset_train(self, config, fake_dataset_dir):
        """Train dataset should load only normal images."""
        from src.training.dataset import MVTecDataset
        from src.training.augmentations import get_eval_transform
        
        transform = get_eval_transform(config)
        dataset = MVTecDataset(
            root_dir=fake_dataset_dir,
            category="bottle",
            split="train",
            transform=transform,
        )
        assert len(dataset) == 10
        assert all(l == 0 for l in dataset.labels), "All train labels should be 0 (normal)"
        
        img, label, idx = dataset[0]
        assert img.shape == (3, 224, 224)
        assert label == 0
    
    def test_mvtec_dataset_test(self, config, fake_dataset_dir):
        """Test dataset should load both normal and anomalous images."""
        from src.training.dataset import MVTecDataset
        from src.training.augmentations import get_eval_transform
        
        transform = get_eval_transform(config)
        dataset = MVTecDataset(
            root_dir=fake_dataset_dir,
            category="bottle",
            split="test",
            transform=transform,
        )
        assert len(dataset) == 10  # 5 good + 5 defect
        assert sum(dataset.labels) == 5, "Should have 5 anomalous samples"
        assert dataset.labels.count(0) == 5, "Should have 5 normal samples"
    
    def test_mvtec_dataset_contrastive(self, config, fake_dataset_dir):
        """Contrastive mode should return two views."""
        from src.training.dataset import MVTecDataset
        from src.training.augmentations import ContrastiveAugmentation
        
        aug = ContrastiveAugmentation(config)
        dataset = MVTecDataset(
            root_dir=fake_dataset_dir,
            category="bottle",
            split="train",
            transform=aug,
            is_contrastive=True,
        )
        view1, view2, label, idx = dataset[0]
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
        assert label == 0
    
    def test_dataset_defect_type(self, config, fake_dataset_dir):
        """Test that defect types are correctly recorded."""
        from src.training.dataset import MVTecDataset
        dataset = MVTecDataset(
            root_dir=fake_dataset_dir,
            category="bottle",
            split="test",
        )
        defect_types = set(dataset.defect_types)
        assert "good" in defect_types
        assert "broken_large" in defect_types
    
    def test_create_dataloaders(self, config, fake_dataset_dir):
        """Test DataLoader creation functions."""
        from src.training.dataset import create_train_dataloader, create_test_dataloader, create_feature_dataloader
        
        config["dataset"]["root_dir"] = fake_dataset_dir
        config["dataset"]["num_workers"] = 0
        config["training"]["batch_size"] = 4
        
        train_dl = create_train_dataloader(config, "bottle")
        assert len(train_dl) > 0
        batch = next(iter(train_dl))
        assert len(batch) == 4  # view1, view2, labels, indices
        
        test_dl = create_test_dataloader(config, "bottle")
        assert len(test_dl) > 0
        
        feat_dl = create_feature_dataloader(config, "bottle")
        assert len(feat_dl) > 0
    
    def test_dataset_not_found_raises_error(self, config):
        """Requesting a nonexistent dataset directory should raise FileNotFoundError."""
        from src.training.dataset import MVTecDataset
        with pytest.raises(FileNotFoundError):
            MVTecDataset(
                root_dir="/nonexistent/path",
                category="bottle",
                split="train",
            )


# ─────────────────────────────────────────────────────────────────
# Test: src/memory_bank.py
# ─────────────────────────────────────────────────────────────────

class TestMemoryBank:
    """Unit tests for src/memory_bank.py"""
    
    def test_memory_bank_save_load(self, tmp_output_dir):
        """Memory bank save/load roundtrip."""
        from src.memory.memory_bank import MemoryBank
        
        bank = MemoryBank()
        bank.features = F.normalize(torch.randn(20, 512), dim=1)
        bank.count = 20
        
        save_path = os.path.join(tmp_output_dir, "test_bank.pt")
        bank.save(save_path)
        assert os.path.exists(save_path)
        
        bank2 = MemoryBank()
        bank2.load(save_path)
        assert bank2.count == 20
        assert torch.allclose(bank.features, bank2.features)
    
    def test_knn_scorer_normal_vs_anomaly(self):
        """k-NN scorer should assign higher scores to anomalous features."""
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        
        # Create a memory bank of "normal" features (clustered near [1, 0, 0, ...])
        normal_features = torch.randn(50, 512) * 0.1
        normal_features[:, 0] += 3.0  # Shift normal features in one direction
        
        bank = MemoryBank()
        bank.features = F.normalize(normal_features, dim=1)
        bank.count = 50
        
        scorer = AnomalyScorer(method="knn", k_neighbors=3)
        scorer.fit(bank)
        
        # Normal test features (similar to bank)
        normal_test = torch.randn(5, 512) * 0.1
        normal_test[:, 0] += 3.0
        normal_scores = scorer.score(normal_test)
        
        # Anomalous test features (far from bank)
        anomaly_test = torch.randn(5, 512) * 0.1
        anomaly_test[:, 0] -= 3.0  # Opposite direction
        anomaly_scores = scorer.score(anomaly_test)
        
        assert np.mean(anomaly_scores) > np.mean(normal_scores), \
            "Anomalous features should have higher scores than normal features"
    
    def test_mahalanobis_scorer(self):
        """Mahalanobis scorer should produce valid scores."""
        from src.memory.memory_bank import MemoryBank, AnomalyScorer
        
        bank = MemoryBank()
        bank.features = F.normalize(torch.randn(50, 512), dim=1)
        bank.count = 50
        
        scorer = AnomalyScorer(method="mahalanobis", k_neighbors=3)
        scorer.fit(bank)
        
        test_features = torch.randn(5, 512)
        scores = scorer.score(test_features)
        assert scores.shape == (5,)
        assert all(np.isfinite(scores)), "Scores should be finite"


# ─────────────────────────────────────────────────────────────────
# Test: src/evaluator.py
# ─────────────────────────────────────────────────────────────────

class TestEvaluator:
    """Unit tests for src/evaluator.py"""
    
    def test_evaluate_perfect_separation(self, tmp_output_dir):
        """Evaluator should return AUROC=1.0 for perfectly separated scores."""
        from src.evaluation.evaluator import AnomalyEvaluator
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        
        scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        metrics = evaluator.evaluate(scores, labels, "test_category")
        assert metrics["auroc"] == 1.0, "AUROC should be 1.0 for perfect separation"
        assert metrics["f1_score"] > 0.9
    
    def test_evaluate_random_scores(self, tmp_output_dir):
        """Evaluator should handle random scores without errors."""
        from src.evaluation.evaluator import AnomalyEvaluator
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        
        np.random.seed(42)
        scores = np.random.rand(100)
        labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        
        metrics = evaluator.evaluate(scores, labels, "random_test")
        assert 0 <= metrics["auroc"] <= 1.0
        assert 0 <= metrics["f1_score"] <= 1.0
        assert metrics["total_samples"] == 100
    
    def test_plot_roc_curve(self, tmp_output_dir):
        """ROC curve plot should be saved."""
        from src.evaluation.evaluator import AnomalyEvaluator
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        
        path = evaluator.plot_roc_curve(scores, labels, "test_category")
        assert path is not None
        assert os.path.exists(path)
    
    def test_plot_confusion_matrix(self, tmp_output_dir):
        """Confusion matrix plot should be saved."""
        from src.evaluation.evaluator import AnomalyEvaluator
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([0, 0, 1, 1])
        
        path = evaluator.plot_confusion_matrix(scores, labels, "test_category", threshold=0.5)
        assert path is not None
        assert os.path.exists(path)
    
    def test_save_metrics_json(self, tmp_output_dir):
        """Metrics should be saved as valid JSON."""
        from src.evaluation.evaluator import AnomalyEvaluator
        import json
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        metrics = {"auroc": 0.95, "f1_score": 0.88, "category": "test"}
        evaluator.save_metrics(metrics, "test")
        
        json_path = os.path.join(tmp_output_dir, "test_metrics.json")
        assert os.path.exists(json_path)
        with open(json_path) as f:
            loaded = json.load(f)
        assert loaded["auroc"] == 0.95
    
    def test_generate_full_report(self, tmp_output_dir):
        """Full report generation should produce metrics and plots."""
        from src.evaluation.evaluator import AnomalyEvaluator
        
        evaluator = AnomalyEvaluator(output_dir=tmp_output_dir)
        scores = np.array([0.1, 0.2, 0.15, 0.8, 0.9, 0.85])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        metrics = evaluator.generate_full_report(scores, labels, "test_full")
        assert "plots" in metrics
        assert metrics["plots"]["roc_curve"] is not None
        assert metrics["plots"]["confusion_matrix"] is not None
        assert metrics["plots"]["score_distribution"] is not None


# ─────────────────────────────────────────────────────────────────
# Test: src/gradcam.py
# ─────────────────────────────────────────────────────────────────

class TestGradCAM:
    """Unit tests for src/gradcam.py"""
    
    def test_gradcam_heatmap_shape(self, config):
        """Grad-CAM should produce a heatmap matching the input spatial dimensions."""
        from src.models.simclr import SimCLRModel
        from src.inference.gradcam import GradCAM
        
        model = SimCLRModel(config)
        gradcam = GradCAM(model, target_layer_name="layer4")
        
        input_tensor = torch.randn(1, 3, 224, 224)
        heatmap = gradcam.generate(input_tensor, device=torch.device("cpu"))
        
        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
    
    def test_gradcam_overlay(self, config):
        """Grad-CAM overlay should produce a valid image."""
        from src.models.simclr import SimCLRModel
        from src.inference.gradcam import GradCAM
        
        model = SimCLRModel(config)
        gradcam = GradCAM(model, target_layer_name="layer4")
        
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        
        overlay = gradcam.generate_overlay(original, heatmap, alpha=0.4)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8
    
    def test_gradcam_visualize_saves_file(self, config, tmp_output_dir):
        """Grad-CAM visualization should save to file."""
        from src.models.simclr import SimCLRModel
        from src.inference.gradcam import GradCAM
        
        model = SimCLRModel(config)
        gradcam = GradCAM(model, target_layer_name="layer4")
        
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        
        save_path = os.path.join(tmp_output_dir, "test_gradcam.png")
        gradcam.visualize(original, heatmap, anomaly_score=0.75, label="Anomaly", save_path=save_path)
        assert os.path.exists(save_path)
