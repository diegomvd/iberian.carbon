from torchgeo.trainers import RegressionTask
from typing import Any, Optional, Union, List, Tuple, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import WeightsEnum
import segmentation_models_pytorch as smp
from pathlib import Path

from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR

from height_regression_losses import RangeAwareL1Loss

import math
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

MAX_LR = 1e-4
PCT_START = 0.30
DIV_FACTOR = 20
FINAL_DIV_FACTOR = 100

class AttentionPixelWiseRegressionTask(RegressionTask):
    """LightningModule for pixelwise regression with SCSE attention."""
    target_key = 'mask'

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights
        if self.hparams['model'] == 'unet':
            # Create UNet with SCSE attention
            self.model = smp.Unet(
                encoder_name=self.hparams['backbone'],
                encoder_weights='imagenet' if weights is True else None,
                in_channels=self.hparams['in_channels'],
                classes=1,
                decoder_attention_type='scse'  # This enables SCSE attention in the decoder
            )
        elif self.hparams['model'] == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams['backbone'],
                encoder_weights='imagenet' if weights is True else None,
                in_channels=self.hparams['in_channels'],
                classes=1,
            )
        elif self.hparams['model'] == 'fcn':
            self.model = FCN(
                in_channels=self.hparams['in_channels'],
                classes=1,
                num_filters=self.hparams['num_filters'],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        # Handle pretrained weights
        if self.hparams['model'] != 'fcn':
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone if specified
        if self.hparams.get('freeze_backbone', False) and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder if specified
        if self.hparams.get('freeze_decoder', False) and self.hparams['model'] in [
            'unet',
            'deeplabv3+',
        ]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

class CanopyHeightRegressionTask(AttentionPixelWiseRegressionTask):
    def __init__(
        self,
        model: str = 'unet',
        backbone: str = "efficientnet-b4",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 10,
        num_outputs: int = 1,
        num_filters: int = 3,
        loss: str = "range-reg-l1",
        nan_value_target: float = -1.0,
        nan_value_input: float = 0.0,
        lr: float = 1e-4,
        patience: int = 15,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        value_ranges: List[Tuple[float, float]] = None,
        log_target: bool = True,
        **kwargs
    ) -> None:
        self.nan_value_target = nan_value_target
        self.nan_value_input = nan_value_input
        self.nan_counter = 0
        self.log_target = log_target
        
        # Define updated value ranges if not provided
        self.value_ranges = value_ranges or [
            (0, 1),       # Very low canopy (shrubs, saplings)
            (1, 5),       # Low canopy (young trees)
            (5, 10),  
            (10, 15),    # Medium canopy (established trees)
            (15, 20),     # High canopy (mature trees)
            (20, 25),     # Very high canopy (old growth)
            (25, 30)  # Exceptional height (special cases)
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
        
        # Initialize range metrics with new ranges
        self.range_metrics = {
            f"{self.value_ranges[0][0]}_{self.value_ranges[0][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[0]},
            f"{self.value_ranges[1][0]}_{self.value_ranges[1][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[1]},
            f"{self.value_ranges[2][0]}_{self.value_ranges[2][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[2]},
            f"{self.value_ranges[3][0]}_{self.value_ranges[3][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[3]},
            f"{self.value_ranges[4][0]}_{self.value_ranges[4][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[4]},
            f"{self.value_ranges[5][0]}_{self.value_ranges[5][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[5]},
            f"{self.value_ranges[6][0]}_{self.value_ranges[6][1]}m": {"count": 0, "bias":[], "mae": [], "range": self.value_ranges[6]}
        }

    def _transform_target(self, y: Tensor) -> Tensor:
        return torch.where(y != self.nan_value_target, torch.expm1(y), y)

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

    def _handle_nan_gradients(self) -> None:
        """Handle NaN gradients after backward pass."""
        # Replace NaN gradients with zeros
        for param in self.parameters():
            if param.grad is not None:
                nan_mask = torch.isnan(param.grad)
                if nan_mask.any():
                    param.grad[nan_mask] = 0.0
                    self.nan_counter += nan_mask.sum().item()


    def configure_losses(self) -> None:
        loss: str = self.hparams["loss"]
        if loss == "mse":
            self.criterion = nn.MSELoss(reduction='none')
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction='none')
        elif loss == "range-reg-l1":          
            self.criterion = RangeAwareL1Loss(nan_value = self.nan_value_target)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently supports 'mse', 'mae', or 'weighted' loss."
            )

    def _nan_robust_loss_reduction(self, loss: Tensor, y: Tensor, nan_value_target: float, input_mask: Tensor) -> Tensor:
        target_mask = y != nan_value_target
        valid_mask = input_mask & target_mask
        
        # Add debug info
        total_pixels = valid_mask.numel()
        valid_pixels = valid_mask.sum().item()
        if valid_pixels == 0:
            print(f"Warning: No valid pixels found! Total pixels: {total_pixels}")
            print(f"Input mask valid: {input_mask.sum().item()}")
            print(f"Target mask valid: {target_mask.sum().item()}")
        
        loss = torch.masked_select(loss, valid_mask)
        return torch.mean(loss) if loss.numel() > 0 else torch.tensor(1e5, device=loss.device)

    def _update_range_metrics(self, y_hat: Tensor, y: Tensor) -> Dict:
        """Update metrics for each value range"""
        metrics = {}
        
        for range_name, range_data in self.range_metrics.items():
            min_val, max_val = range_data['range']

            if self.log_target:
                y_natural = self._transform_target(y)
                mask = (y_natural >= min_val) & (y_natural < max_val) & (y_natural != self.nan_value_target)
            else:
                mask = (y >= min_val) & (y < max_val) & (y != self.nan_value_target)
                
            if mask.any():
                range_preds = y_hat[mask]
                range_targets = y[mask]
                
                if self.log_target:
                    range_preds = self._transform_target(range_preds)
                    range_targets = self._transform_target(range_targets)

                mae = torch.nn.functional.l1_loss(range_preds, range_targets).item()
                bias = torch.mean(range_preds - range_targets)    

                metrics[f"{range_name}_mae"] = mae
                metrics[f"{range_name}_bias"] = bias
                metrics[f"{range_name}_count"] = mask.sum().item()

            else:
                metrics[f"{range_name}_mae"] = -1.0
                metrics[f"{range_name}_bias"] = -1.0
                metrics[f"{range_name}_count"] = 0
            
        return metrics

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        x = batch["image"]
        y = batch[self.target_key].to(torch.float)

        x_filled, input_mask = self._handle_nan_inputs(x)
        
        y_hat = self(x_filled)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        
        # Calculate loss 
        loss: Tensor = self.criterion(y_hat, y)
        #loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target, input_mask)

        # Handle potential NaN loss
        if torch.isnan(loss):
            self.nan_counter += 1
            loss = torch.tensor(1e5, device=loss.device, requires_grad=True)

        # Log current parameters
        self.log("train_loss", loss)
        self.log("nan_count", self.nan_counter)
        
        # Log range-specific metrics
        # range_metrics = self._update_range_metrics(y_hat*input_mask, y)
        # for metric_name, value in range_metrics.items():
        #     self.log(f"train_{metric_name}", value)
        
        return loss

    def on_after_backward(self) -> None:
        """Handle NaN gradients after backward pass."""
        self._handle_nan_gradients()

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
        # loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target,input_mask)

        # Handle potential NaN loss
        if torch.isnan(loss):
            loss = torch.tensor(1e5, device=loss.device)
        
        # Log overall loss
        self.log("val_loss", loss)
        
        # Log range-specific metrics
        range_metrics = self._update_range_metrics(y_hat*input_mask, y)
        for metric_name, value in range_metrics.items():
            self.log(f"val_{metric_name}", value)

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
        # loss = self._nan_robust_loss_reduction(loss, y, self.nan_value_target,input_mask)
        
        # Log overall loss
        self.log("test_loss", loss)
        
        # Log range-specific metrics
        range_metrics = self._update_range_metrics(y_hat*input_mask, y)
        for metric_name, value in range_metrics.items():
            self.log(f"test_{metric_name}", value)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Optional[Tensor]:
        if batch_idx > -1:
            x = batch['image']
            y_hat: Tensor = self(x)
            if self.log_target:
                y_hat = self._transform_target(y_hat)
            return y_hat    
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=MAX_LR,  
            weight_decay=0.01
        )
        
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            epochs=self.trainer.max_epochs,
            total_steps=total_steps,
            pct_start=PCT_START,
            div_factor=DIV_FACTOR,        
            final_div_factor=FINAL_DIV_FACTOR, 
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
