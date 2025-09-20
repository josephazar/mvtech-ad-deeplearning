"""
Real wrapper for Dinomaly model
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict
import time

# Add model directory to path
model_dir = Path(__file__).parent.parent.parent / "models" / "DinomalyV2"
sys.path.insert(0, str(model_dir))

from dataset import get_data_transforms
from models import vit_encoder
from models.uad import ViTillv2
from utils import evaluation_batch
from optimizers import StableAdamW
from functools import partial
from torch import nn


class DinomalyWrapper:
    """Real wrapper for Dinomaly model."""

    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.model = None

        # Model parameters from paper
        self.image_size = 448
        self.crop_size = 392
        self.batch_size = 16
        self.total_iters = 5000  # As specified in paper

        # Setup model architecture
        self.encoder_name = 'dinov2reg_vit_base_14'
        self.target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        self.embed_dim = 768
        self.num_heads = 12

    def train_or_load(self, category: str, train_loader):
        """Train or load the model for a specific category."""
        print(f"  Setting up Dinomaly for {category}...")

        # Check for saved model
        save_path = Path(f"checkpoints/dinomaly/{category}")
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "model.pth"

        try:
            # Load encoder
            encoder = vit_encoder.load(self.encoder_name)

            # Build decoder
            from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
            from dinov1.utils import trunc_normal_

            bottleneck = nn.ModuleList([
                bMlp(self.embed_dim, self.embed_dim * 4, self.embed_dim, drop=0.2)
            ])

            decoder = nn.ModuleList()
            for i in range(8):
                blk = VitBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention2
                )
                decoder.append(blk)

            # Create model
            fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
            fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

            self.model = ViTillv2(
                encoder=encoder,
                bottleneck=bottleneck,
                decoder=decoder,
                target_layers=self.target_layers,
                mask_neighbor_size=0,
                fuse_layer_encoder=fuse_layer_encoder,
                fuse_layer_decoder=fuse_layer_decoder
            )
            self.model = self.model.to(self.device)

            if model_file.exists():
                print(f"  Loading saved model from {model_file}")
                checkpoint = torch.load(model_file, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=False)
            else:
                print(f"  Training new model for {category}...")
                self._train(train_loader, model_file, bottleneck, decoder)

        except Exception as e:
            print(f"  Error setting up Dinomaly: {e}")
            self.model = None

    def _train(self, train_loader, save_path, bottleneck, decoder):
        """Train the model."""
        if self.model is None:
            return

        from dinov1.utils import trunc_normal_
        from utils import WarmCosineScheduler, global_cosine_hm_percent

        trainable = nn.ModuleList([bottleneck, decoder])

        # Initialize weights
        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        optimizer = StableAdamW(
            [{'params': trainable.parameters()}],
            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4,
            amsgrad=True, eps=1e-10  # As specified in paper with AMSGrad
        )

        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=2e-3, final_value=2e-4,
            total_iters=self.total_iters, warmup_iters=100  # Paper: warmup first 100 iterations
        )

        self.model.train()
        it = 0

        for epoch in range(int(np.ceil(self.total_iters / len(train_loader)))):
            for img, label in train_loader:
                if it >= self.total_iters:
                    break

                img = img.to(self.device)
                en, de = self.model(img)

                # Paper: dropout rate increases from 0% to 90% during first 500 iterations
                p_final = 0.9
                p = min(p_final * it / 500, p_final)  # Paper: 500 iterations for dropout schedule
                loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)  # Paper: L_global-hm loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(trainable.parameters(), max_norm=0.1)
                optimizer.step()
                lr_scheduler.step()

                it += 1
                if it % 100 == 0:
                    print(f"    Iteration {it}/{self.total_iters}, Loss: {loss.item():.4f}")

        # Save model
        torch.save(self.model.state_dict(), save_path)
        print(f"  Model saved to {save_path}")

    def evaluate(self, test_loader) -> Dict:
        """Evaluate the model on test data."""
        if self.model is None:
            print("  Model not initialized, returning zeros")
            return {'image_auroc': 0.0, 'pixel_auroc': 0.0, 'pro_score': 0.0}

        print(f"  Running Dinomaly evaluation...")
        self.model.eval()

        try:
            # Paper: mean of top 1% pixels for anomaly score
            results = evaluation_batch(
                self.model, test_loader, self.device,
                max_ratio=0.01, resize_mask=256  # Paper: top 1% pixels
            )

            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results

            return {
                'image_auroc': auroc_sp,
                'pixel_auroc': auroc_px,
                'pro_score': aupro_px
            }

        except Exception as e:
            print(f"  Error during evaluation: {e}")
            return {'image_auroc': 0.0, 'pixel_auroc': 0.0, 'pro_score': 0.0}