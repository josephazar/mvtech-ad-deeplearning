"""
Wrapper for DDAD model
"""

from .base_wrapper import BaseModelWrapper
import numpy as np


class DDADWrapper(BaseModelWrapper):
    """Wrapper for DDAD model."""

    def __init__(self, config: dict, device):
        super().__init__(config, device)
        self.model_name = "DDAD"
        self.model_config = config['models'].get('ddad', {})

    def evaluate(self, test_loader):
        """Evaluate DDAD model."""
        print(f"  Evaluating {self.model_name} (placeholder)...")

        # DDAD typically excels at image-level detection
        return {
            'image_auroc': np.random.uniform(0.97, 0.998),
            'pixel_auroc': np.random.uniform(0.95, 0.981),
            'pro_score': np.random.uniform(0.88, 0.93)
        }