"""
Research-grade evaluation module for anomaly detection.

Computes comprehensive metrics at both image-level and pixel-level:
    - AUROC (image-level + pixel-level)
    - Average Precision (AP)
    - F1-Score (optimal threshold + fixed threshold)
    - PRO score (Per-Region Overlap) for localization quality
    - Precision, Recall, Accuracy
    - Cross-category aggregation with mean ± std

Supports ablation study result logging and visualization.
"""

import os
import json
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from typing import Dict, Tuple, Optional, List

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Research-grade evaluator for anomaly detection.

    Computes standard metrics, generates visualizations, and supports
    cross-category aggregation with statistical reporting.

    Threshold selection strategies:
        - youden: Maximize Youden's J statistic (TPR - FPR)
        - f1: Maximize F1 score
        - percentile: Use a fixed percentile of scores
    """

    def __init__(
        self,
        output_dir: str = "outputs/results",
        threshold_method: str = "youden",
        percentile_value: float = 97.0,
    ):
        self.output_dir = output_dir
        self.threshold_method = threshold_method
        self.percentile_value = percentile_value
        os.makedirs(output_dir, exist_ok=True)

    def evaluate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        category: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Compute all evaluation metrics.

        Args:
            scores: Anomaly scores, shape (N,). Higher = more anomalous.
            labels: Ground truth labels, shape (N,). 0=normal, 1=anomaly.
            category: Category name for logging.
            threshold: Classification threshold. If None, auto-optimized.

        Returns:
            Dict with all metrics.
        """
        # Compute AUROC
        auroc = roc_auc_score(labels, scores)

        # Compute Average Precision
        ap = average_precision_score(labels, scores)

        # Auto-optimize threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(scores, labels)

        # Binary predictions
        predictions = (scores >= threshold).astype(int)

        # Compute metrics
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, zero_division=0)
        precision = precision_score(labels, predictions, zero_division=0)
        rec = recall_score(labels, predictions, zero_division=0)

        # F1 at fixed thresholds for ablation
        f1_at_fixed = {}
        for pct in [90, 95, 97, 99]:
            fixed_thresh = float(np.percentile(scores, pct))
            fixed_preds = (scores >= fixed_thresh).astype(int)
            f1_at_fixed[f"f1_at_p{pct}"] = float(
                f1_score(labels, fixed_preds, zero_division=0)
            )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        # Classification report
        report = classification_report(
            labels,
            predictions,
            target_names=["Normal", "Anomaly"],
            output_dict=True,
        )

        metrics = {
            "category": category,
            "auroc": float(auroc),
            "average_precision": float(ap),
            "f1_score": float(f1),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(rec),
            "threshold": float(threshold),
            "threshold_method": self.threshold_method,
            "f1_at_fixed_thresholds": f1_at_fixed,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_normal": int((labels == 0).sum()),
            "num_anomaly": int((labels == 1).sum()),
            "total_samples": len(labels),
            "score_stats": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "normal_mean": float(np.mean(scores[labels == 0])) if (labels == 0).any() else 0.0,
                "normal_std": float(np.std(scores[labels == 0])) if (labels == 0).any() else 0.0,
                "anomaly_mean": float(np.mean(scores[labels == 1])) if (labels == 1).any() else 0.0,
                "anomaly_std": float(np.std(scores[labels == 1])) if (labels == 1).any() else 0.0,
            },
        }

        logger.info(
            f"[{category}] AUROC: {auroc:.4f} | AP: {ap:.4f} | F1: {f1:.4f} | "
            f"Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {rec:.4f} | "
            f"Threshold: {threshold:.4f} ({self.threshold_method})"
        )

        return metrics

    def evaluate_pixel_level(
        self,
        anomaly_maps: List[np.ndarray],
        ground_truth_masks: List[np.ndarray],
        category: str,
    ) -> Dict:
        """
        Compute pixel-level metrics for anomaly localization.

        Args:
            anomaly_maps: List of predicted anomaly heatmaps (H, W) in [0, 1]
            ground_truth_masks: List of binary GT masks (H, W) in {0, 1}
            category: Category name

        Returns:
            Dict with pixel-level AUROC, AP, and PRO score
        """
        # Flatten all maps and masks
        all_preds = []
        all_gt = []

        for amap, gt_mask in zip(anomaly_maps, ground_truth_masks):
            # Ensure same size
            if amap.shape != gt_mask.shape:
                import cv2
                amap = cv2.resize(amap, (gt_mask.shape[1], gt_mask.shape[0]))

            all_preds.append(amap.flatten())
            all_gt.append(gt_mask.flatten())

        all_preds = np.concatenate(all_preds)
        all_gt = np.concatenate(all_gt)

        # Binarize GT
        all_gt = (all_gt > 0.5).astype(int)

        pixel_metrics = {}

        if all_gt.sum() > 0 and all_gt.sum() < len(all_gt):
            pixel_auroc = roc_auc_score(all_gt, all_preds)
            pixel_ap = average_precision_score(all_gt, all_preds)
            pixel_metrics["pixel_auroc"] = float(pixel_auroc)
            pixel_metrics["pixel_ap"] = float(pixel_ap)

            logger.info(
                f"[{category}] Pixel-AUROC: {pixel_auroc:.4f} | "
                f"Pixel-AP: {pixel_ap:.4f}"
            )
        else:
            pixel_metrics["pixel_auroc"] = 0.0
            pixel_metrics["pixel_ap"] = 0.0
            logger.warning(
                f"[{category}] Cannot compute pixel metrics — "
                f"no positive/negative pixels"
            )

        # PRO score (Per-Region Overlap)
        pro = self._compute_pro_score(anomaly_maps, ground_truth_masks)
        pixel_metrics["pro_score"] = float(pro)
        logger.info(f"[{category}] PRO score: {pro:.4f}")

        return pixel_metrics

    def _compute_pro_score(
        self,
        anomaly_maps: List[np.ndarray],
        ground_truth_masks: List[np.ndarray],
        num_thresholds: int = 200,
        fpr_limit: float = 0.3,
    ) -> float:
        """
        Compute Per-Region Overlap (PRO) score.

        PRO measures localization quality by computing the mean overlap
        between predicted and ground truth anomaly regions across multiple
        thresholds, normalized by the area under FPR limit.

        Research reference:
            Bergmann et al., "Improving Unsupervised Defect Segmentation by
            Applying Structural Similarity to Autoencoders", 2019
        """
        try:
            from scipy.ndimage import label as scipy_label
        except ImportError:
            logger.warning("scipy not available for PRO score — returning 0")
            return 0.0

        all_preds = np.concatenate([m.flatten() for m in anomaly_maps])
        thresholds = np.linspace(
            all_preds.max(), all_preds.min(), num_thresholds
        )

        pro_values = []
        fpr_values = []

        for threshold in thresholds:
            total_overlap = 0.0
            total_regions = 0
            total_fp = 0
            total_neg = 0

            for amap, gt_mask in zip(anomaly_maps, ground_truth_masks):
                if amap.shape != gt_mask.shape:
                    import cv2
                    amap = cv2.resize(amap, (gt_mask.shape[1], gt_mask.shape[0]))

                binary_pred = (amap >= threshold).astype(int)
                gt_binary = (gt_mask > 0.5).astype(int)

                # Count FP pixels
                total_fp += int(np.sum(binary_pred * (1 - gt_binary)))
                total_neg += int(np.sum(1 - gt_binary))

                # Find connected regions in ground truth
                labeled, n_regions = scipy_label(gt_binary)

                for region_id in range(1, n_regions + 1):
                    region_mask = (labeled == region_id)
                    region_area = region_mask.sum()
                    if region_area > 0:
                        overlap = (binary_pred * region_mask).sum() / region_area
                        total_overlap += overlap
                        total_regions += 1

            # FPR
            fpr = total_fp / max(total_neg, 1)
            fpr_values.append(fpr)

            # Mean per-region overlap
            pro = total_overlap / max(total_regions, 1)
            pro_values.append(pro)

        # Compute area under PRO-FPR curve up to fpr_limit
        fpr_values = np.array(fpr_values)
        pro_values = np.array(pro_values)

        # Filter to FPR <= limit
        valid = fpr_values <= fpr_limit
        if valid.sum() < 2:
            return 0.0

        # Sort by FPR
        sorted_idx = np.argsort(fpr_values[valid])
        fpr_sorted = fpr_values[valid][sorted_idx]
        pro_sorted = pro_values[valid][sorted_idx]

        # Area under curve, normalized by fpr_limit
        pro_score = float(np.trapz(pro_sorted, fpr_sorted) / fpr_limit)

        return pro_score

    def cross_category_summary(
        self,
        all_metrics: Dict[str, Dict],
    ) -> Dict:
        """
        Compute cross-category summary with mean ± std.

        Args:
            all_metrics: Dict mapping category -> metrics dict

        Returns:
            Summary dict with aggregated statistics
        """
        if not all_metrics:
            return {}

        metric_keys = ["auroc", "average_precision", "f1_score",
                       "accuracy", "precision", "recall"]

        summary = {}
        for key in metric_keys:
            values = [m[key] for m in all_metrics.values() if key in m]
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": {cat: m[key] for cat, m in all_metrics.items() if key in m},
                }

        # Log summary table
        logger.info(f"\n{'=' * 78}")
        logger.info("Cross-Category Summary (mean +/- std)")
        logger.info(f"{'=' * 78}")
        for key in metric_keys:
            if key in summary:
                s = summary[key]
                logger.info(
                    f"  {key:<20s}: {s['mean']:.4f} +/- {s['std']:.4f} "
                    f"(min={s['min']:.4f}, max={s['max']:.4f})"
                )
        logger.info(f"{'=' * 78}")

        return summary

    def _find_optimal_threshold(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find optimal threshold based on configured method."""
        if self.threshold_method == "f1":
            return self._find_f1_optimal_threshold(scores, labels)
        elif self.threshold_method == "percentile":
            return float(np.percentile(scores, self.percentile_value))
        else:
            return self._find_youden_threshold(scores, labels)

    def _find_youden_threshold(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find threshold that maximizes Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return float(np.median(scores))

    def _find_f1_optimal_threshold(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find threshold that maximizes F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0,
        )
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])

    # ─── Plotting Methods ───────────────────────────────────────────

    def plot_roc_curve(
        self, scores: np.ndarray, labels: np.ndarray,
        category: str, save: bool = True,
    ) -> Optional[str]:
        """Plot and save ROC curve."""
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"AUROC = {auroc:.4f}")
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
        ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve - {category}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_roc_curve.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"ROC curve saved: {save_path}")
            return save_path
        plt.close(fig)
        return None

    def plot_precision_recall_curve(
        self, scores: np.ndarray, labels: np.ndarray,
        category: str, save: bool = True,
    ) -> Optional[str]:
        """Plot and save Precision-Recall curve."""
        precisions, recalls, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(recalls, precisions, color="#FF5722", linewidth=2, label=f"AP = {ap:.4f}")
        ax.fill_between(recalls, precisions, alpha=0.15, color="#FF5722")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve - {category}", fontsize=14, fontweight="bold")
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_pr_curve.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"PR curve saved: {save_path}")
            return save_path
        plt.close(fig)
        return None

    def plot_f1_vs_threshold(
        self, scores: np.ndarray, labels: np.ndarray,
        category: str, optimal_threshold: float, save: bool = True,
    ) -> Optional[str]:
        """Plot F1 score as a function of threshold."""
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1_scores = np.where(
            (precisions[:-1] + recalls[:-1]) > 0,
            2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1]),
            0,
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(thresholds, f1_scores, color="#9C27B0", linewidth=2)
        ax.axvline(
            x=optimal_threshold, color="#FF9800", linestyle="--",
            linewidth=2, label=f"Selected threshold = {optimal_threshold:.4f}",
        )
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"F1 vs Threshold - {category}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_f1_threshold.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        plt.close(fig)
        return None

    def plot_confusion_matrix(
        self, scores: np.ndarray, labels: np.ndarray,
        category: str, threshold: float, save: bool = True,
    ) -> Optional[str]:
        """Plot and save confusion matrix."""
        predictions = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, predictions)

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax, annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix - {category}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_confusion_matrix.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        plt.close(fig)
        return None

    def plot_score_distribution(
        self, scores: np.ndarray, labels: np.ndarray,
        category: str, threshold: float, save: bool = True,
    ) -> Optional[str]:
        """Plot anomaly score distribution for normal vs anomalous samples."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]

        ax.hist(normal_scores, bins=50, alpha=0.6, color="#4CAF50",
                label=f"Normal (n={len(normal_scores)})", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#F44336",
                label=f"Anomaly (n={len(anomaly_scores)})", density=True)
        ax.axvline(x=threshold, color="#FF9800", linestyle="--", linewidth=2,
                   label=f"Threshold = {threshold:.4f}")

        ax.set_xlabel("Anomaly Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Score Distribution - {category}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save:
            save_path = os.path.join(self.output_dir, f"{category}_score_distribution.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            return save_path
        plt.close(fig)
        return None

    def save_metrics(self, metrics: Dict, category: str):
        """Save metrics to JSON file."""
        save_path = os.path.join(self.output_dir, f"{category}_metrics.json")
        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {save_path}")

    def generate_full_report(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        category: str,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Generate complete evaluation report with metrics and all plots."""
        metrics = self.evaluate(scores, labels, category, threshold)
        threshold_used = metrics["threshold"]

        # Generate all plots
        roc_path = self.plot_roc_curve(scores, labels, category)
        pr_path = self.plot_precision_recall_curve(scores, labels, category)
        f1t_path = self.plot_f1_vs_threshold(scores, labels, category, threshold_used)
        cm_path = self.plot_confusion_matrix(scores, labels, category, threshold_used)
        dist_path = self.plot_score_distribution(scores, labels, category, threshold_used)

        metrics["plots"] = {
            "roc_curve": roc_path,
            "pr_curve": pr_path,
            "f1_threshold": f1t_path,
            "confusion_matrix": cm_path,
            "score_distribution": dist_path,
        }

        self.save_metrics(metrics, category)
        return metrics

    def log_ablation_result(
        self,
        config_name: str,
        category: str,
        metrics: Dict,
        config_params: Dict,
    ):
        """
        Log ablation study result for comparison.

        Appends result to a JSON log file for later analysis.
        """
        ablation_path = os.path.join(self.output_dir, "ablation_results.json")

        # Load existing results
        existing = []
        if os.path.exists(ablation_path):
            try:
                with open(ablation_path, "r") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        result = {
            "config_name": config_name,
            "category": category,
            "auroc": metrics.get("auroc", 0),
            "f1_score": metrics.get("f1_score", 0),
            "average_precision": metrics.get("average_precision", 0),
            "config_params": config_params,
        }

        existing.append(result)

        with open(ablation_path, "w") as f:
            json.dump(existing, f, indent=2)

        logger.info(
            f"Ablation result logged: {config_name} | "
            f"AUROC={metrics.get('auroc', 0):.4f}"
        )
