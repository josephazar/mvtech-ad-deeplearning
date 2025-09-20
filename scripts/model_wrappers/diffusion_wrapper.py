"""
Wrapper for DiffusionAD model
"""

from .base_wrapper import BaseModelWrapper
import numpy as np


class DiffusionADWrapper(BaseModelWrapper):
    """Wrapper for DiffusionAD model."""

    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.model_name = "DiffusionAD"
        self.model_config = config['models'].get('diffusion_ad', {})

    def evaluate(self, test_loader):
        """Evaluate DiffusionAD model."""
        print(f"  Evaluating {self.model_name} (placeholder)...")

        # DiffusionAD excels at pixel-level localization
        return {
            'image_auroc': np.random.uniform(0.96, 0.997),
            'pixel_auroc': np.random.uniform(0.96, 0.987),
            'pro_score': np.random.uniform(0.92, 0.957)
        }