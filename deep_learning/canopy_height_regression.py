from torchgeo.trainers import PixelwiseRegressionTask
from typing import Any, Optional, Union, List, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum

from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau

class AdaptiveCanopyHeightLoss(nn.Module):
    def __init__(self, alpha=3, beta=0.7, base_weight=0.05):
        super().__init__()
        self.alpha = alpha      # Increased power law exponent for stronger high-value emphasis
        self.beta = beta        # Increased beta to favor power law over log scaling
        self.base_weight = base_weight  # Smaller base weight since low values are already well-learned
    
    def forward(self, predictions, targets):
        # Small base weights for zeros
        base_weights = torch.ones_like(targets) * self.base_weight
        
        # Stronger power law transformation
        power_law_weights = torch.pow(targets + self.base_weight, self.alpha)
        
        # Less emphasis on log scaling
        log_weights = torch.log(targets + 1 + self.base_weight)
        
        # Hybrid weighting with increased emphasis on power law component
        weights = (self.beta * power_law_weights + 
                  (1 - self.beta) * log_weights + 
                  base_weights)
        
        # Normalize weights
        weights = weights / (weights.mean() + 1e-8)
        
        # Weighted MSE        
        return weights * ((predictions - targets) ** 2)


class CanopyHeightRegressionTask(PixelwiseRegressionTask):

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet18",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 10,
        num_outputs: int = 1,
        num_filters: int = 3,
        loss: str = "weighted",
        nan_value: float = -1.0,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        value_ranges: List[Tuple[float, float]] = None,
        **kwargs
    ) -> None:
        self.nan_value = nan_value
        self.alpha = kwargs.get('alpha', 2)
        self.beta = kwargs.get('beta', 0.7)
        self.base_weight = kwargs.get('base_weight',0.05)
        
        # Define default value ranges if not provided
        self.value_ranges = value_ranges or [
            (0, 1.5),     # Very low canopy
            (1.5, 5),     # Low canopy
            (5, 10),      # Medium canopy
            (10, 20),     # High canopy
            (20, float('inf'))  # Very high canopy
        ]
        
        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_outputs=num_outputs,
            num_filters=num_filters,
            loss=loss,
            lr=lr,
            patience=patience,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder
        )
        
        # Initialize range metrics
        self.range_metrics = {
            "very_low": {"count": 0, "mse": [], "range": self.value_ranges[0]},
            "low": {"count": 0, "mse": [], "range": self.value_ranges[1]},
            "medium": {"count": 0, "mse": [], "range": self.value_ranges[2]},
            "high": {"count": 0, "mse": [], "range": self.value_ranges[3]},
            "very_high": {"count": 0, "mse": [], "range": self.value_ranges[4]}
        }

    def configure_losses(self) -> None:
        loss: str = self.hparams["loss"]
        if loss == "mse":
            self.criterion = nn.MSELoss(reduction='none')
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss == "weighted":
            self.criterion = AdaptiveCanopyHeightLoss(alpha=self.alpha, beta=self.beta)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently supports 'mse', 'mae', or 'weighted' loss."
            )

    def _nan_robust_loss_reduction(self, loss: Tensor, y: Tensor, nan_value: float) -> Tensor:
        y = y != nan_value
        loss = torch.masked_select(loss, y)
        return torch.mean(loss) 

    def _update_range_metrics(self, y_hat: Tensor, y: Tensor) -> Dict:
        """Update metrics for each value range"""
        metrics = {}
        
        for range_name, range_data in self.range_metrics.items():
            min_val, max_val = range_data['range']
            mask = (y >= min_val) & (y < max_val) & (y != self.nan_value)
            
            if mask.any():
                range_preds = y_hat[mask]
                range_targets = y[mask]
                
                mse = torch.nn.functional.mse_loss(range_preds, range_targets).item()
                
                metrics[f"{range_name}/mse"] = mse
                metrics[f"{range_name}/count"] = mask.sum().item()
        
        return metrics

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value)
        
        # Log overall loss
        self.log("train_loss", loss)
        
        # Log range-specific metrics
        range_metrics = self._update_range_metrics(y_hat, y)
        for metric_name, value in range_metrics.items():
            self.log(f"train_{metric_name}", value)
        
        return loss

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value)
        
        # Log overall loss
        self.log("val_loss", loss)
        
        # Log range-specific metrics
        range_metrics = self._update_range_metrics(y_hat, y)
        for metric_name, value in range_metrics.items():
            self.log(f"val_{metric_name}", value)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)
        loss = self._nan_robust_loss_reduction(loss, y, self.nan_value)
        
        # Log overall loss
        self.log("test_loss", loss)
        
        # Log range-specific metrics
        range_metrics = self._update_range_metrics(y_hat, y)
        for metric_name, value in range_metrics.items():
            self.log(f"test_{metric_name}", value)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
        if batch_idx > -1:
            x = batch['image']
            y_hat: Tensor = self(x)
            return y_hat    
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.hparams.patience,
            verbose=True,
            min_lr=1e-6,              # Don't reduce LR below this value
            cooldown=2                 # Wait this ma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
