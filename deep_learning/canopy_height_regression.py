from torchgeo.trainers import PixelwiseRegressionTask
from typing import Any, Optional, Union, List, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum
import segmentation_models_pytorch as smp
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

import math
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

from height_regression_losses import RelativeBalancedFocalLoss, RangeSpecificFocalLoss

def get_balanced_universal_loss():
    return RelativeBalancedFocalLoss(
        gamma = 2.0,
        rel_reference = 0.25,
        min_height = 1.0,
        weight_href = 2.0,
        beta = 1.5,
        reduction = 'none'
    )
        
def get_very_low_vegetation_loss():
    return RangeSpecificFocalLoss(
        loss_range=(-0.5, 3.5),
        alpha=1.5,
        abs_reference=0.1,
    )

def get_low_medium_vegetation_loss():
    return RangeSpecificFocalLoss(
        loss_range=(1.0, 10.0),
        alpha=4.0, 
        abs_reference=1.5,
    )

def get_medium_high_vegetation_loss():
    return RangeSpecificFocalLoss(
        loss_range=(6, 18),
        alpha=4.0,
        abs_reference=2.0,
    )

def get_tall_vegetation_loss():
    return RangeSpecificFocalLoss(
        loss_range=(14, float('inf')),
        alpha=3.5,
        abs_reference=3.0,
    )

# Loss functions for relative focal weights on L1 loss.
def get_very_low_vegetation_loss_relative():
    return RangeSpecificFocalLoss(
        loss_range=(-0.5, 3.5),
        alpha=3.5,
        rel_reference=0.25
    )

def get_low_medium_vegetation_loss_relative():
    return RangeSpecificFocalLoss(
        loss_range=(1.0, 10.0),
        alpha=3.5, 
        rel_reference=0.25
    )

def get_medium_high_vegetation_loss_relative():
    return RangeSpecificFocalLoss(
        loss_range=(6, 18),
        alpha=3.5,
        rel_reference=0.25,
    )

def get_tall_vegetation_loss_relative():
    return RangeSpecificFocalLoss(
        loss_range=(14, 40),
        alpha=3.5,
        rel_reference=0.25,
    )

class HeightRangeMetrics:
    """Height range specific metrics calculator for canopy height regression"""
    
    def __init__(self):
        self.height_ranges = [
            (0, 1), (1, 2), (2, 4), (4, 8), 
            (8, 12), (12, 16), (16, 20), (20, 25), 
            (25, float('inf'))
        ]
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        input_mask: torch.Tensor,
        nan_value: float
    ) -> Dict[str, float]:
        """Calculate metrics for each height range."""
        metrics = {}
            
        # Create valid mask
        valid_mask = (targets != nan_value) & input_mask
        
        for min_h, max_h in self.height_ranges:
            # Create range mask
            range_mask = valid_mask & (targets >= min_h) & (targets < max_h)
            
            if range_mask.any():
                range_preds = predictions[range_mask]
                range_targets = targets[range_mask]
                
                # Calculate metrics
                abs_errors = torch.abs(range_preds - range_targets)
                signed_diff = range_preds - range_targets
                
                range_name = f"h_{min_h}_{max_h}"
                metrics.update({
                    f"test_{range_name}_mae": abs_errors.mean(),
                    f"test_{range_name}_std": abs_errors.std(),
                    f"test_{range_name}_signed_mean": signed_diff.mean(),
                    f"test_{range_name}_signed_std": signed_diff.std(),
                    f"test_{range_name}_count": range_mask.sum()
                })
            else:
                range_name = f"h_{min_h}_{max_h}"
                metrics.update({
                    f"test_{range_name}_mae": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_std": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_signed_mean": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_signed_std": torch.tensor(0.0, device=predictions.device),
                    f"test_{range_name}_count": torch.tensor(0, device=predictions.device)
                })
        
        return metrics

class CanopyHeightRegressionTask(PixelwiseRegressionTask):
    """LightningModule for range-specific canopy height regression."""
    target_key = 'mask'
    
    def __init__(
        self,
            model: str = 'unet',#'deeplabv3+',
        backbone: str = "efficientnet-b2",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 10,
        num_outputs: int = 1,
        target_range: str = "universal",  # One of: "very_low", "low_medium", "medium_high", "tall"
        nan_value_target: float = -1.0,
        nan_value_input: float = -9999.0,
        lr: float = 1e-4,
        patience: int = 15,
        **kwargs: Any
    ) -> None:

        self.nan_value_target = nan_value_target
        self.nan_value_input = nan_value_input
        self.nan_counter = 0
        self.target_range = target_range
        
        # Configure range-specific parameters and metrics
        self.range_configs = {
            "very_low": {
                "loss_fn": get_very_low_vegetation_loss_relative(),
                "metric_ranges": [(0, 1), (1, 2), (2, 3)]
            },
            "low_medium": {
                "loss_fn": get_low_medium_vegetation_loss_relative(),
                "metric_ranges": [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
            },
            "medium_high": {
                "loss_fn": get_medium_high_vegetation_loss_relative(),
                "metric_ranges": [(6, 8), (8, 12), (12, 16), (16, 18)]
            },
            "tall": {
                "loss_fn": get_tall_vegetation_loss_relative(),
                "metric_ranges": [(14, 16), (16, 20), (20, 25), (25, float('inf'))]
            },
            "universal": {
                "loss_fn" : get_balanced_universal_loss(),
                "metric_ranges": [(0,1),(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 18), (18, 20), (20,25), (25,30)]
            }
        }
        
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_outputs=num_outputs,
            lr=lr,
            patience=patience,
            **kwargs
        )
        
    def _create_input_mask(self, x: Tensor) -> Tensor:
        """Create mask for valid input values across all bands"""
        # Returns True for valid pixels (not -9999.0) across all bands, False otherwise
        return ~(x == self.nan_value_input).any(dim=1, keepdim=True)  # Mask where any band has nodata

    def _handle_nan_inputs(self, x: Tensor) -> Tensor:
        """Handle NaN values in input tensor."""
        input_mask = self._create_input_mask(x)
        x_filled = x.clone()
        x_filled[x == self.nan_value_input] = 0.0  # Fill nodata with zeros
        return x_filled, input_mask

    def configure_losses(self) -> None:
        """Configure the loss function based on target range."""
        if self.target_range not in self.range_configs:
            raise ValueError(f"Unknown target range: {self.target_range}")
        self.criterion = self.range_configs[self.target_range]["loss_fn"]

    def _nan_robust_loss_reduction(self, loss: Tensor, y: Tensor, nan_value_target: float, input_mask: Tensor) -> Tensor:
        target_mask = y != nan_value_target
        valid_mask = input_mask & target_mask    
        loss = torch.masked_select(loss, valid_mask)
        return torch.mean(loss) if loss.numel() > 0 else torch.tensor(1e5, device=loss.device)

    def _calculate_range_metrics(self, y_hat: Tensor, y: Tensor, input_mask: Tensor) -> Dict[str, float]:
        """Calculate metrics for specific height ranges."""
        metrics = {}
        current_ranges = self.range_configs[self.target_range]["metric_ranges"]
        
        for min_val, max_val in current_ranges:
            # Define range name
            range_name = f"h_{min_val}_{max_val}".replace('.', '_')
            if max_val == float('inf'):
                range_mask = (y >= min_val)
            else:
                range_mask = (y >= min_val) & (y < max_val)
            
            # Combine masks
            valid_mask = range_mask & (y != self.nan_value_target) & input_mask
            
            if valid_mask.any():
                range_preds = y_hat[valid_mask]
                range_targets = y[valid_mask]
                
                # Calculate metrics
                mae = torch.nn.functional.l1_loss(range_preds, range_targets).item()
                rmse = torch.sqrt(torch.nn.functional.mse_loss(range_preds, range_targets)).item()
                bias = (range_preds - range_targets).mean().item()
                
                metrics[f"{range_name}_mae"] = mae
                metrics[f"{range_name}_rmse"] = rmse
                metrics[f"{range_name}_bias"] = bias
                metrics[f"{range_name}_count"] = valid_mask.sum().item()
            else:
                metrics[f"{range_name}_mae"] = 0.0
                metrics[f"{range_name}_rmse"] = 0.0
                metrics[f"{range_name}_bias"] = 0.0
                metrics[f"{range_name}_count"] = 0
                
        return metrics

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        # stats = {}
        
        # # Overall prediction statistics
        # stats.update({
        #     'pred_mean': y_hat.mean().item(),
        #     'pred_std': y_hat.std().item(),
        #     'pred_min': y_hat.min().item(),
        #     'pred_max': y_hat.max().item(),
        #     'pred_median': y_hat.median().item(),
        #     'pred_unique_values': len(torch.unique(y_hat))
        # })
        
        # # Add percentile statistics
        # percentiles = [1, 5, 25, 50, 75, 95, 99]
        # for p in percentiles:
        #     stats[f'pred_percentile_{p}'] = torch.quantile(y_hat, p/100).item()

        # print('Prediction statistics:')
        # for key in stats:
        #     print(f'{key}: {stats[key]}')

        # stats = {}    
        # # Overall prediction statistics                                                                                                                  
        # stats.update({
        #     'pred_mean': y.mean().item(),
        #     'pred_std': y.std().item(),
        #     'pred_min': y.min().item(),
        #     'pred_max': y.max().item(),
        #     'pred_median': y.median().item(),
        #     'pred_unique_values': len(torch.unique(y))
        # })

        # # Add percentile statistics                                                                                                                           
        # percentiles = [1, 5, 25, 50, 75, 95, 99]
        # for p in percentiles:
        #     stats[f'pred_percentile_{p}'] = torch.quantile(y, p/100).item()

        # print('Target statistics:')
        # for key in stats:
        #     print(f'{key}: {stats[key]}')
    
            
        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target, input_mask)

        # Handle potential NaN loss
        if torch.isnan(loss):
            self.nan_counter += 1
            loss = torch.tensor(1e5, device=loss.device, requires_grad=True)

        # Log current parameters
        self.log("train_loss", loss)
        self.log("nan_count", self.nan_counter)
        
        #Log range-specific metrics
       # metrics = self._calculate_range_metrics(y_hat * input_mask, y, input_mask)
       # for name, value in metrics.items():
       #     self.log(f"train_{name}", value)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)

        
        # Handle NaN inputs and get mask
        x_filled, input_mask = self._handle_nan_inputs(x)

        # Forward pass with filled inputs
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

    
        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target,input_mask)


        # Handle potential NaN loss
        if torch.isnan(loss):
            loss = torch.tensor(1e5, device=loss.device)
    
        # Log overall loss
        self.log("val_loss", loss)
        
        metrics = self._calculate_range_metrics(y_hat * input_mask, y, input_mask)
        for name, value in metrics.items():
            self.log(f"val_{name}", value)
    
            
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)

        # Handle NaN inputs and get mask
        x_filled, input_mask = self._handle_nan_inputs(x)
        
        # Forward pass with filled inputs
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target, input_mask)
        
        # Log overall loss
        self.log("test_loss", loss)
        
        # Calculate and log range-specific metrics
        if not hasattr(self, 'height_metrics'):
            self.height_metrics = HeightRangeMetrics()
        
        range_metrics = self.height_metrics.calculate_metrics(
            y_hat * input_mask,  # Apply input mask to predictions
            y,
            input_mask,
            self.nan_value_target
        )
        
        # Log all range metrics
        for metric_name, value in range_metrics.items():
            self.log(metric_name, value)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
        if batch_idx > -1:
            x = batch['image']
            y_hat: Tensor = self(x)
            return y_hat    
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,  
            weight_decay=0.01
        )
        
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            epochs=self.trainer.max_epochs,
            total_steps=total_steps,
            pct_start=0.30,
            div_factor=15,        
            final_div_factor=60, 
            three_phase=False,
            anneal_strategy='cos'
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }
