"""
Evaluation module for anomaly detection.
Computes AUROC, F1-Score, accuracy, and generates classification reports.
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
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Evaluates anomaly detection performance.
    Computes standard metrics and generates visualizations.
    """
    
    def __init__(self, output_dir: str = "outputs/results"):
        self.output_dir = output_dir
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
        f1 = f1_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Classification report
        report = classification_report(
            labels, predictions,
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
            "recall": float(recall),
            "threshold": float(threshold),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "num_normal": int((labels == 0).sum()),
            "num_anomaly": int((labels == 1).sum()),
            "total_samples": len(labels),
        }
        
        logger.info(
            f"[{category}] AUROC: {auroc:.4f} | F1: {f1:.4f} | "
            f"Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
        )
        
        return metrics
    
    def _find_optimal_threshold(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find threshold that maximizes Youden's J statistic."""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        
        j_scores = tpr - fpr
        
        best_idx = np.argmax(j_scores)
        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return float(np.median(scores))
    
    def plot_roc_curve(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        category: str,
        save: bool = True,
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
        ax.set_title(f"ROC Curve — {category}", fontsize=14, fontweight="bold")
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
    
    def plot_confusion_matrix(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        category: str,
        threshold: float,
        save: bool = True,
    ) -> Optional[str]:
        """Plot and save confusion matrix."""
        predictions = (scores >= threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
            ax=ax,
            annot_kws={"size": 16},
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title(f"Confusion Matrix — {category}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f"{category}_confusion_matrix.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Confusion matrix saved: {save_path}")
            return save_path
        
        plt.close(fig)
        return None
    
    def plot_score_distribution(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        category: str,
        threshold: float,
        save: bool = True,
    ) -> Optional[str]:
        """Plot anomaly score distribution for normal vs anomalous samples."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        ax.hist(normal_scores, bins=50, alpha=0.6, color="#4CAF50", label="Normal", density=True)
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#F44336", label="Anomaly", density=True)
        ax.axvline(x=threshold, color="#FF9800", linestyle="--", linewidth=2, label=f"Threshold = {threshold:.4f}")
        
        ax.set_xlabel("Anomaly Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Score Distribution — {category}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            save_path = os.path.join(self.output_dir, f"{category}_score_distribution.png")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Score distribution saved: {save_path}")
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
        """
        Generate complete evaluation report with metrics and all plots.
        
        Returns:
            Dict with metrics and paths to generated plots.
        """
        metrics = self.evaluate(scores, labels, category, threshold)
        threshold_used = metrics["threshold"]
        
        # Generate plots
        roc_path = self.plot_roc_curve(scores, labels, category)
        cm_path = self.plot_confusion_matrix(scores, labels, category, threshold_used)
        dist_path = self.plot_score_distribution(scores, labels, category, threshold_used)
        
        metrics["plots"] = {
            "roc_curve": roc_path,
            "confusion_matrix": cm_path,
            "score_distribution": dist_path,
        }
        
        # Save metrics
        self.save_metrics(metrics, category)
        
        return metrics
