"""
Common metrics utilities for anomaly detection evaluation
"""

import numpy as np
import cv2
from sklearn import metrics
from skimage import measure
import torch
import time
from typing import Dict, List, Tuple, Optional


def compute_image_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Area Under ROC Curve for image-level anomaly detection.

    Args:
        y_true: Ground truth labels (0: normal, 1: anomaly)
        y_scores: Anomaly scores for each image

    Returns:
        AUROC score
    """
    return metrics.roc_auc_score(y_true, y_scores)


def compute_pixel_auroc(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray
) -> float:
    """
    Compute pixel-wise AUROC for anomaly segmentation.

    Args:
        anomaly_maps: Predicted anomaly heat maps [N, H, W]
        ground_truth_masks: Ground truth masks [N, H, W]

    Returns:
        Pixel-wise AUROC score
    """
    if isinstance(anomaly_maps, list):
        anomaly_maps = np.stack(anomaly_maps)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_maps = anomaly_maps.ravel()
    flat_gt_masks = ground_truth_masks.ravel()

    return metrics.roc_auc_score(
        flat_gt_masks.astype(int),
        flat_anomaly_maps
    )


def compute_pro_score(
    anomaly_maps: np.ndarray,
    ground_truth_masks: np.ndarray,
    num_thresholds: int = 200,
    max_fpr: float = 0.3
) -> float:
    """
    Compute Per-Region Overlap (PRO) score.

    Args:
        anomaly_maps: Predicted anomaly heat maps [N, H, W]
        ground_truth_masks: Ground truth masks [N, H, W]
        num_thresholds: Number of thresholds to evaluate
        max_fpr: Maximum false positive rate to consider

    Returns:
        PRO score
    """
    if isinstance(anomaly_maps, list):
        anomaly_maps = np.stack(anomaly_maps)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    pros = []
    fprs = []

    min_th = anomaly_maps.min()
    max_th = anomaly_maps.max()
    delta = (max_th - min_th) / num_thresholds

    # Morphological kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for threshold in np.arange(min_th, max_th, delta):
        # Threshold the anomaly maps
        binary_maps = (anomaly_maps > threshold).astype(np.uint8)

        pro_items = []
        for binary_map, gt_mask in zip(binary_maps, ground_truth_masks):
            # Dilate the binary anomaly map
            binary_map = cv2.dilate(binary_map, kernel)

            # For each ground truth region
            for region in measure.regionprops(measure.label(gt_mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]

                # Compute true positives for this region
                tp_pixels = binary_map[axes0_ids, axes1_ids].sum()
                pro_items.append(tp_pixels / region.area)

        if len(pro_items) > 0:
            pros.append(np.mean(pro_items))
        else:
            pros.append(0.0)

        # Compute false positive rate
        inverse_masks = 1 - ground_truth_masks
        fp_pixels = np.logical_and(inverse_masks, binary_maps).sum()
        fpr = fp_pixels / (inverse_masks.sum() + 1e-10)
        fprs.append(fpr)

    pros = np.array(pros)
    fprs = np.array(fprs)

    # Filter by maximum FPR
    valid_indices = fprs < max_fpr
    if valid_indices.sum() == 0:
        return 0.0

    fprs = fprs[valid_indices]
    pros = pros[valid_indices]

    # Normalize FPR
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min() + 1e-10)

    # Compute area under the PRO curve
    pro_score = metrics.auc(fprs, pros)

    return pro_score


def compute_average_precision(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> float:
    """
    Compute Average Precision score.

    Args:
        y_true: Ground truth labels
        y_scores: Anomaly scores

    Returns:
        Average Precision score
    """
    return metrics.average_precision_score(y_true, y_scores)


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute F1 score and find optimal threshold.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted scores

    Returns:
        Tuple of (f1_score, precision, recall, threshold)
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    return best_f1, best_precision, best_recall, best_threshold


class MetricsComputer:
    """Compute and store metrics for anomaly detection evaluation."""

    def __init__(self, metrics_to_compute: Optional[List[str]] = None):
        """
        Initialize metrics computer.

        Args:
            metrics_to_compute: List of metrics to compute.
                               Default: ['image_auroc', 'pixel_auroc', 'pro_score']
        """
        if metrics_to_compute is None:
            metrics_to_compute = ['image_auroc', 'pixel_auroc', 'pro_score']

        self.metrics_to_compute = metrics_to_compute
        self.results = {}
        self.timings = {}

    def compute(
        self,
        image_labels: np.ndarray,
        image_scores: np.ndarray,
        pixel_labels: Optional[np.ndarray] = None,
        pixel_scores: Optional[np.ndarray] = None,
        category: str = "overall"
    ) -> Dict[str, float]:
        """
        Compute specified metrics.

        Args:
            image_labels: Ground truth image labels
            image_scores: Predicted image anomaly scores
            pixel_labels: Ground truth pixel masks (optional)
            pixel_scores: Predicted pixel anomaly maps (optional)
            category: Category name for storing results

        Returns:
            Dictionary of computed metrics
        """
        results = {}

        # Image-level metrics
        if 'image_auroc' in self.metrics_to_compute:
            start_time = time.time()
            results['image_auroc'] = compute_image_auroc(image_labels, image_scores)
            self.timings['image_auroc'] = time.time() - start_time

        if 'image_ap' in self.metrics_to_compute:
            start_time = time.time()
            results['image_ap'] = compute_average_precision(image_labels, image_scores)
            self.timings['image_ap'] = time.time() - start_time

        if 'image_f1' in self.metrics_to_compute:
            start_time = time.time()
            f1, prec, rec, thresh = compute_f1_score(image_labels, image_scores)
            results['image_f1'] = f1
            results['image_precision'] = prec
            results['image_recall'] = rec
            results['image_threshold'] = thresh
            self.timings['image_f1'] = time.time() - start_time

        # Pixel-level metrics (if masks are provided)
        if pixel_labels is not None and pixel_scores is not None:
            if 'pixel_auroc' in self.metrics_to_compute:
                start_time = time.time()
                results['pixel_auroc'] = compute_pixel_auroc(pixel_scores, pixel_labels)
                self.timings['pixel_auroc'] = time.time() - start_time

            if 'pixel_ap' in self.metrics_to_compute:
                start_time = time.time()
                flat_labels = pixel_labels.ravel()
                flat_scores = pixel_scores.ravel()
                results['pixel_ap'] = compute_average_precision(flat_labels, flat_scores)
                self.timings['pixel_ap'] = time.time() - start_time

            if 'pro_score' in self.metrics_to_compute:
                start_time = time.time()
                results['pro_score'] = compute_pro_score(pixel_scores, pixel_labels)
                self.timings['pro_score'] = time.time() - start_time

        # Store results by category
        self.results[category] = results

        return results

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary of all computed metrics.

        Returns:
            Dictionary with average metrics across all categories
        """
        if not self.results:
            return {}

        summary = {}
        all_metrics = set()

        # Collect all metric names
        for category_results in self.results.values():
            all_metrics.update(category_results.keys())

        # Compute averages
        for metric in all_metrics:
            values = []
            for category_results in self.results.values():
                if metric in category_results:
                    values.append(category_results[metric])

            if values:
                summary[f"{metric}_mean"] = np.mean(values)
                summary[f"{metric}_std"] = np.std(values)

        return summary

    def print_results(self, category: Optional[str] = None):
        """
        Print formatted results.

        Args:
            category: Specific category to print. If None, print all.
        """
        if category:
            if category in self.results:
                print(f"\n{category} Results:")
                for metric, value in self.results[category].items():
                    print(f"  {metric}: {value:.4f}")
        else:
            # Print all categories
            for cat, results in self.results.items():
                print(f"\n{cat} Results:")
                for metric, value in results.items():
                    if not metric.endswith('_threshold'):
                        print(f"  {metric}: {value:.4f}")

            # Print summary
            summary = self.get_summary()
            if summary:
                print("\nOverall Summary:")
                for metric, value in summary.items():
                    print(f"  {metric}: {value:.4f}")