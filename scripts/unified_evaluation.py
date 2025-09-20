#!/usr/bin/env python3
"""
Unified Evaluation Script for MVTec-AD Dataset
Runs all models (GLASS, DDAD, DiffusionAD, Dinomaly) on MVTec-AD dataset
"""

import os
import sys
import time
import json
import yaml
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common utilities
from scripts.utils.metrics import MetricsComputer

warnings.filterwarnings("ignore")


class UnifiedEvaluator:
    """Unified evaluator for all anomaly detection models."""

    def __init__(self, config_path: str, data_path: Optional[str] = None):
        """
        Initialize the unified evaluator.

        Args:
            config_path: Path to configuration file
            data_path: Override path to MVTec dataset
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Override data path if provided
        if data_path:
            self.config['dataset']['path'] = data_path

        # Setup paths
        self.data_path = Path(self.config['dataset']['path'])
        self.results_dir = Path(self.config['evaluation']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup device
        self.device = self._setup_device()

        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(
            metrics_to_compute=self.config['evaluation']['metrics']
        )

        # Results storage
        self.all_results = {}
        self.timing_results = {}

        print(f"Unified Evaluator initialized")
        print(f"Data path: {self.data_path}")
        print(f"Results directory: {self.results_dir}")
        print(f"Device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        device_config = self.config['runtime']['device']

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                print("GPU not available, using CPU")
        elif device_config == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                raise RuntimeError("CUDA requested but not available")
        else:
            device = torch.device('cpu')

        return device

    def evaluate_all_models(self) -> Dict:
        """
        Evaluate all enabled models on MVTec-AD dataset.

        Returns:
            Dictionary containing all evaluation results
        """
        results = {}

        # Check which models are enabled
        models_to_evaluate = []
        for model_name, model_config in self.config['models'].items():
            if model_config.get('enabled', False):
                models_to_evaluate.append(model_name)

        print(f"\nModels to evaluate: {models_to_evaluate}")

        # Evaluate each model
        for model_name in models_to_evaluate:
            print(f"\n{'='*50}")
            print(f"Evaluating {model_name.upper()}")
            print(f"{'='*50}")

            try:
                model_results = self._evaluate_model(model_name)
                results[model_name] = model_results
                print(f"\n{model_name.upper()} evaluation completed successfully")
            except Exception as e:
                print(f"\nError evaluating {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        # Save all results
        self._save_results(results)

        return results

    def _evaluate_model(self, model_name: str) -> Dict:
        """
        Evaluate a specific model.

        Args:
            model_name: Name of the model to evaluate

        Returns:
            Dictionary containing evaluation results
        """
        # Import the appropriate model wrapper
        try:
            if model_name == 'glass':
                from scripts.model_wrappers.glass_wrapper import GLASSWrapper
                model_wrapper = GLASSWrapper(self.config, self.device)
            elif model_name == 'ddad':
                from scripts.model_wrappers.ddad_wrapper import DDADWrapper
                model_wrapper = DDADWrapper(self.config, self.device)
            elif model_name == 'diffusion_ad':
                from scripts.model_wrappers.diffusion_wrapper import DiffusionADWrapper
                model_wrapper = DiffusionADWrapper(self.config, self.device)
            elif model_name == 'dinomaly':
                from scripts.model_wrappers.dinomaly_wrapper import DinomalyWrapper
                model_wrapper = DinomalyWrapper(self.config, self.device)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except ImportError as e:
            print(f"Warning: Could not import wrapper for {model_name}: {e}")
            print(f"Using base wrapper with placeholder results")
            from scripts.model_wrappers.base_wrapper import BaseModelWrapper
            model_wrapper = BaseModelWrapper(self.config, self.device)
            model_wrapper.model_name = model_name

        # Evaluate on all categories
        categories = self.config['dataset']['categories']
        model_results = {}

        for category in tqdm(categories, desc=f"Evaluating {model_name}"):
            print(f"\n  Category: {category}")

            # Get data loaders
            train_loader, test_loader = self._get_data_loaders(category)

            # Train/load model
            start_time = time.time()
            model_wrapper.train_or_load(category, train_loader)
            train_time = time.time() - start_time

            # Evaluate model
            start_time = time.time()
            category_results = model_wrapper.evaluate(test_loader)
            eval_time = time.time() - start_time

            # Add timing information
            category_results['train_time'] = train_time
            category_results['eval_time'] = eval_time

            # Store results
            model_results[category] = category_results

            # Print category results
            self._print_category_results(model_name, category, category_results)

        # Compute overall statistics
        model_results['overall'] = self._compute_overall_stats(model_results)

        return model_results

    def _get_data_loaders(self, category: str) -> Tuple[DataLoader, DataLoader]:
        """
        Get data loaders for a specific category.

        Args:
            category: MVTec category name

        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Import dataset class
        from models.DinomalyV2.dataset import MVTecDataset, get_data_transforms
        from torchvision.datasets import ImageFolder

        # Get transforms
        image_size = self.config['dataset']['image_size']
        crop_size = self.config['dataset']['crop_size']
        data_transform, gt_transform = get_data_transforms(image_size, crop_size)

        # Paths
        category_path = self.data_path / category
        train_path = category_path / 'train'
        test_path = category_path

        # Create datasets
        train_dataset = ImageFolder(root=str(train_path), transform=data_transform)
        test_dataset = MVTecDataset(
            root=str(test_path),
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test"
        )

        # Create loaders
        batch_size = self.config['evaluation']['batch_size']
        num_workers = self.config['evaluation']['num_workers']

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, test_loader

    def _compute_overall_stats(self, model_results: Dict) -> Dict:
        """
        Compute overall statistics across all categories.

        Args:
            model_results: Results for all categories

        Returns:
            Dictionary with overall statistics
        """
        all_metrics = {}

        for category, results in model_results.items():
            if category == 'overall' or 'error' in results:
                continue

            for metric, value in results.items():
                if metric not in ['train_time', 'eval_time']:
                    if metric not in all_metrics:
                        all_metrics[metric] = []
                    all_metrics[metric].append(value)

        # Compute mean and std for each metric
        overall_stats = {}
        for metric, values in all_metrics.items():
            overall_stats[f"{metric}_mean"] = np.mean(values)
            overall_stats[f"{metric}_std"] = np.std(values)
            overall_stats[f"{metric}_min"] = np.min(values)
            overall_stats[f"{metric}_max"] = np.max(values)

        return overall_stats

    def _print_category_results(self, model_name: str, category: str, results: Dict):
        """Print results for a category."""
        print(f"    Results for {model_name} on {category}:")
        for metric, value in results.items():
            if metric not in ['train_time', 'eval_time']:
                if isinstance(value, float):
                    print(f"      {metric}: {value:.4f}")

    def _save_results(self, results: Dict):
        """
        Save evaluation results to files.

        Args:
            results: All evaluation results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as JSON
        json_path = self.results_dir / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\nResults saved to {json_path}")

        # Create summary DataFrame
        summary_data = []
        for model_name, model_results in results.items():
            if 'overall' in model_results:
                row = {'model': model_name}
                row.update(model_results['overall'])
                summary_data.append(row)

        if summary_data:
            df_summary = pd.DataFrame(summary_data)

            # Save as CSV
            csv_path = self.results_dir / f"summary_{timestamp}.csv"
            df_summary.to_csv(csv_path, index=False)
            print(f"Summary saved to {csv_path}")

            # Print summary table
            print("\n" + "="*80)
            print("EVALUATION SUMMARY")
            print("="*80)
            print(df_summary.to_string(index=False))

        # Save detailed results by category
        detailed_path = self.results_dir / f"detailed_{timestamp}.csv"
        detailed_data = []

        for model_name, model_results in results.items():
            if isinstance(model_results, dict):
                for category, cat_results in model_results.items():
                    if category != 'overall' and isinstance(cat_results, dict) and 'error' not in cat_results and 'status' not in cat_results:
                        row = {
                            'model': model_name,
                            'category': category
                        }
                        row.update(cat_results)
                        detailed_data.append(row)

        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            df_detailed.to_csv(detailed_path, index=False)
            print(f"\nDetailed results saved to {detailed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation script for anomaly detection models on MVTec-AD"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/unified_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Override path to MVTec dataset"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["glass", "ddad", "diffusion_ad", "dinomaly", "all"],
        default=["all"],
        help="Models to evaluate"
    )

    args = parser.parse_args()

    # Load and potentially modify config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Enable/disable models based on arguments
    if "all" not in args.models:
        for model_name in config['models']:
            config['models'][model_name]['enabled'] = model_name in args.models
    else:
        for model_name in config['models']:
            config['models'][model_name]['enabled'] = True

    # Save modified config temporarily
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        temp_config_path = f.name

    # Create evaluator
    evaluator = UnifiedEvaluator(temp_config_path, args.data_path)

    # Run evaluation
    print("\nStarting unified evaluation...")
    results = evaluator.evaluate_all_models()

    # Clean up temp config
    os.unlink(temp_config_path)

    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()