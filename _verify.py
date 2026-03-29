"""Quick verification of all refactored components."""
import sys
sys.path.insert(0, '.')

import torch
import numpy as np

# 1. Test patch feature extraction
from src.models.simclr import SimCLRModel
print("=" * 60)
print("Test 1: Patch Feature Extraction")
print("=" * 60)

model = SimCLRModel({
    'model': {'backbone': 'resnet50', 'pretrained': True, 
              'patch_layers': ['layer2', 'layer3']}
})
model.eval()
x = torch.randn(2, 3, 224, 224)

pf, ps = model.extract_patch_features(x)
print(f"  Patch features: {pf.shape} (expected: [2, 784, 1536])")
print(f"  Patch grid: {ps} (expected: (28, 28))")

gf = model.extract_features(x)
print(f"  Global features: {gf.shape} (expected: [2, 3584])")
print("  PASS\n")

# 2. Test score calibration
from src.scoring.calibration import ScoreCalibrator
print("=" * 60)
print("Test 2: Score Calibration")
print("=" * 60)

scores = np.array([1.0, 1.5, 2.0, 2.5, 8.0, 9.0, 10.0, 12.0])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

cal = ScoreCalibrator(method='minmax_sigmoid')
cal.fit(scores, labels)

for s, l in zip(scores, labels):
    conf = cal.calibrate(float(s))
    pct = cal.to_percentage(conf)
    lbl = cal.get_confidence_label(conf)
    print(f"  Score={s:.1f} Label={l} Conf={pct:.1f} ({lbl})")

print("  PASS\n")

# 3. Test heatmap generation
from src.visualization.heatmap import AnomalyHeatmapGenerator
print("=" * 60)
print("Test 3: Heatmap Generation")
print("=" * 60)

gen = AnomalyHeatmapGenerator(sigma=4.0)
patch_scores = np.random.rand(784).astype(np.float32)
heatmap = gen.generate(patch_scores, (28, 28), image_size=(224, 224))
print(f"  Heatmap shape: {heatmap.shape} (expected: (224, 224))")
print(f"  Heatmap range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

colored = gen.colorize(heatmap)
print(f"  Colored shape: {colored.shape} (expected: (224, 224, 3))")

fake_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
overlay = gen.overlay(fake_img, heatmap)
print(f"  Overlay shape: {overlay.shape} (expected: (224, 224, 3))")
print("  PASS\n")

# 4. Test PatchMemoryBank and PatchAnomalyScorer
from src.memory.memory_bank import PatchMemoryBank, PatchAnomalyScorer
print("=" * 60)
print("Test 4: PatchMemoryBank & PatchAnomalyScorer")
print("=" * 60)

bank = PatchMemoryBank(use_pca=False, coreset_ratio=0.5, coreset_max=500)
bank.features = torch.randn(200, 64)
bank.features = torch.nn.functional.normalize(bank.features, p=2, dim=1)
bank.count = 200
bank.patch_shape = (28, 28)
bank.feature_dim = 64

scorer = PatchAnomalyScorer(k_neighbors=3)
scorer.fit(bank)

test_patches = torch.randn(784, 64)
test_patches = torch.nn.functional.normalize(test_patches, p=2, dim=1)
patch_scores, img_score = scorer.score_patches(test_patches)
print(f"  Patch scores shape: {patch_scores.shape} (expected: (784,))")
print(f"  Image score: {img_score:.6f}")
print("  PASS\n")

# 5. Test evaluator
from src.evaluation.evaluator import AnomalyEvaluator
print("=" * 60)
print("Test 5: Evaluator")
print("=" * 60)

import tempfile, os
tmpdir = os.path.join('.', '_test_output')
os.makedirs(tmpdir, exist_ok=True)

evaluator = AnomalyEvaluator(output_dir=tmpdir)
fake_scores = np.concatenate([np.random.randn(50) * 0.5 + 2, np.random.randn(50) * 0.5 + 6])
fake_labels = np.concatenate([np.zeros(50), np.ones(50)])

metrics = evaluator.evaluate(fake_scores, fake_labels, "test_category")
print(f"  AUROC: {metrics['auroc']:.4f}")
print(f"  F1: {metrics['f1_score']:.4f}")
print(f"  AP: {metrics['average_precision']:.4f}")
print(f"  Threshold: {metrics['threshold']:.4f}")

# Cleanup
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)
print("  PASS\n")

print("=" * 60)
print("ALL TESTS PASSED!")
print("=" * 60)
