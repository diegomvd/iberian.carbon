
import torch.nn as nn
import torch
from typing import Optional

class RangeAwareDistilledL1Loss(nn.Module):
    """
    Loss function for height prediction in log1p space that encourages range diversity.
    Range is calculated in natural space, which naturally handles log space predictions.
    
    Args:
        percentile_range: Tuple of (lower, upper) percentiles for range calculation
        lambda_reg: Weight for range diversity term
        eps: Small constant for numerical stability
        nan_value: Value to be treated as no-data
    """
    def __init__(
        self,
        teacher_model,
        alpha = 0.4,
        height_threshold: float = 15.0,
        percentile_range: tuple[float, float] = (5.0, 95.0),
        lambda_reg: float = 0.35,
        eps: float = 1e-6,
        nan_value: float = -1.0
    ):
        super().__init__()
        self.teacher = teacher_model
        self.alpha = alpha
        self.height_threshold = height_threshold
        self.percentile_range = percentile_range
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.nan_value = nan_value
        
    def forward(
        self,
        pred: torch.Tensor,  # Predictions in log1p space
        target: torch.Tensor,  # Targets in log1p space
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        teacher_pred = self.teacher(X)

        if mask is None:
            mask = target != self.nan_value
            
        valid_pred = pred[mask]
        valid_target = target[mask]
        valid_teacher_pred = teacher_pred[mask]
        
        if len(valid_pred) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Base L1 loss in log1p space
        base_loss = torch.abs(valid_pred - valid_target)
        mean_base_loss = torch.mean(base_loss)
        
        height_mask = valid_target <= self.height_threshold
        if height_mask.any():
            distill_loss = torch.mean(torch.abs(valid_pred[height_mask] - valid_teacher_pred[height_mask]))
        else:
            distill_loss = torch.tensor(0.0, device=pred.device)    

        # Convert to natural space for range calculation
        natural_pred = torch.expm1(valid_pred)
        natural_target = torch.expm1(valid_target)
        
        # Calculate percentile ranges
        p_low, p_high = self.percentile_range
        percentiles = torch.tensor([p_low/100.0, p_high/100.0], device=pred.device)
        
        pred_percentiles = torch.quantile(natural_pred, percentiles)
        target_percentiles = torch.quantile(natural_target, percentiles)
        
        pred_range = pred_percentiles[1] - pred_percentiles[0]
        target_range = target_percentiles[1] - target_percentiles[0]
        
        # Range diversity penalty
        range_penalty = torch.relu(1.0 - pred_range/(target_range + self.eps))
        
        return (1-self.alpha)*mean_base_loss + self.alpha*distill_loss + self.lambda_reg * range_penalty

class RangeAwareL1Loss(nn.Module):
    """
    Loss function for height prediction in log1p space that encourages range diversity.
    Range is calculated in natural space, which naturally handles log space predictions.
    
    Args:
        percentile_range: Tuple of (lower, upper) percentiles for range calculation
        lambda_reg: Weight for range diversity term
        eps: Small constant for numerical stability
        nan_value: Value to be treated as no-data
    """
    def __init__(
        self,
        percentile_range: tuple[float, float] = (10.0, 90.0),
        lambda_reg: float = 0.35,
        eps: float = 1e-6,
        nan_value: float = -1.0
    ):
        super().__init__()
        self.percentile_range = percentile_range
        self.lambda_reg = lambda_reg
        self.eps = eps
        self.nan_value = nan_value
        
    def forward(
        self,
        pred: torch.Tensor,  # Predictions in log1p space
        target: torch.Tensor,  # Targets in log1p space
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if mask is None:
            mask = target != self.nan_value
            
        valid_pred = pred[mask]
        valid_target = target[mask]
        
        if len(valid_pred) == 0:
            return torch.tensor(0.0, device=pred.device)
        
        # Base L1 loss in log1p space
        base_loss = torch.abs(valid_pred - valid_target)
        mean_base_loss = torch.mean(base_loss)
        
        # Convert to natural space for range calculation
        natural_pred = torch.expm1(valid_pred)
        natural_target = torch.expm1(valid_target)
        
        # Calculate percentile ranges
        p_low, p_high = self.percentile_range
        percentiles = torch.tensor([p_low/100.0, p_high/100.0], device=pred.device)
        
        pred_percentiles = torch.quantile(natural_pred, percentiles)
        target_percentiles = torch.quantile(natural_target, percentiles)
        
        pred_range = pred_percentiles[1] - pred_percentiles[0]
        target_range = target_percentiles[1] - target_percentiles[0]
        
        # Range diversity penalty
        range_penalty = torch.relu(1.0 - pred_range/(target_range + self.eps))
        
        return mean_base_loss + self.lambda_reg * range_penalty

class RangePenaltyBalancedMSELoss(nn.Module):
    def __init__(
        self,
        height_power: float = 2.5,    
        max_reference_height = 25, # not use in log space
        range_penalty_weight: float = 0.1,#0.1,
        range_target_factor: float = 0.0,#0.9,
        nan_value: float = -1.0
    ):
        super().__init__()
        self.height_power = height_power
        # Precompute scale factor based on max reference height
        self.scale_factor = 1 / (pow(1 + max_reference_height, height_power))
        self.range_penalty_weight = range_penalty_weight
        self.range_target_factor = range_target_factor
        self.nan_value = nan_value

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid_mask = target != self.nan_value
        
        # Power-law height weighting with fixed scaling
        heights = torch.clamp(torch.expm1(target), min=0.0)
        #weights = heights*self.height_power
        weights = torch.pow(0.1 + heights, self.height_power)# * self.scale_factor
        
        # MSE base loss with power-law weighting
        squared_error = torch.pow(pred - target, 2)
        base_loss = squared_error * weights
        
        # Range penalty using only valid data
        valid_pred = pred[valid_mask]
        valid_target = target[valid_mask]
        
        if valid_pred.numel() > 0:
            pred_p95 = torch.quantile(valid_pred, 0.95)
            target_p95 = torch.quantile(valid_target, 0.95)

            p95_penalty = torch.nn.functional.relu(target_p95 - pred_p95)
           #  pred_range = torch.quantile(valid_pred, 0.95) - torch.quantile(valid_pred, 0.05)
#             target_range = torch.quantile(valid_target, 0.95) - torch.quantile(valid_target, 0.05)
            
#  #           print(f'Loss calculation, pred_range: {pred_range}, target_range : {target_range}')

#             range_diff = target_range - pred_range
#             range_penalty = torch.nn.functional.relu(range_diff)

#             # neg penalty should prevent from expanding into negative space.
#             neg_penalty = torch.mean(torch.nn.functional.relu(-torch.quantile(valid_pred, 0.05)))
            
#             #range_ratio = pred_range / (target_range + 1e-6)
#             #range_penalty = torch.pow(torch.clamp(self.range_target_factor - range_ratio, min=0), 2)

# #            print(f'Base loss: {base_loss[valid_mask].mean()}, range_penalty: {range_penalty}, neg_penalty: {neg_penalty}')
            
            total_loss = base_loss + self.range_penalty_weight * p95_penalty
        else:
            total_loss = base_loss
            
        return total_loss



class RelativeBalancedFocalLoss(nn.Module):

    """                                                                                             Relative focal loss for vegetation height regression that adapts to different height ranges.  
    Args:                                                                                                alpha (float): Controls the rate at which the focal term grows with relative error                                                                   
        rel_reference (float): Reference relative error (e.g., 0.25 for 25%)                                                                                 
        min_height (float): Minimum height for relative error calculation                                                                                
        eps (float): Small value for numerical stability                                                                                  
        reduction (str): Reduction method ('mean', 'sum', or 'none')                                                                                    
    """
    def __init__(
        self,
        gamma: float = 2.0,
        rel_reference: float = 0.25,
        min_height: float = 1.0,
        weight_href: float = 3.0,
        beta: float = 2.0,
        reduction: str = 'none'
    ):
        super().__init__()
        self.gamma = gamma
        self.rel_reference = rel_reference
        self.min_height = min_height
        self.reduction = reduction
        self.weight_href = weight_href
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
                                                                                                   
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
                    
        abs_error = torch.abs(pred - target)

        # Use max(height, min_height) to avoid division by small numbers                                                                                     
        relative_error = abs_error / torch.clamp(torch.abs(target), min=self.min_height)

        # Calculate focal weight using log1p for stability                                                                                  
        focal_weight = torch.log1p(relative_error / self.rel_reference)
        focal_term = torch.exp(self.gamma * focal_weight)

        scaled_height = (target - self.weight_href) / (self.weight_href)
        weights = torch.sigmoid(self.beta * scaled_height)  # Range [0.1, 1.0]

        #weights = torch.pow(0.1 + target/self.weight_href, self.beta)
                                         
        loss = focal_term * abs_error * weights

        if self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Reduction {self.reduction} is not supported.")



class RangeSpecificFocalLoss(nn.Module):
    """
    Range-specific focal loss for height regression with extended loss consideration ranges.
    Returns unreduced loss tensor for external nan handling.
    
    Args:
        loss_range (tuple): (min_height, max_height) for loss consideration
        alpha (float): Focal term power
        abs_reference (float): Reference value for normalizing the focal term
    """
    def __init__(
        self,
        loss_range: tuple,
        alpha: float,
        abs_reference: Optional[float] = None,
        rel_reference: Optional[float] = None    
    ):
        super().__init__()
        self.loss_range = loss_range
        self.alpha = alpha
        self.abs_reference = abs_reference
        self.rel_reference = rel_reference

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the range-specific focal loss.
        Returns unreduced loss tensor.
        """
        loss_range_mask = (target >= self.loss_range[0]) & (target <= self.loss_range[1])
        abs_error = torch.abs(pred - target)

        if self.abs_reference is not None:
            focal_term = torch.pow(abs_error / self.abs_reference, self.alpha)
        elif self.rel_reference is not None:
            relative_error = abs_error / (target + 1e-6)
            focal_term = torch.pow(relative_error / self.rel_reference, self.alpha)
            
        return focal_term * abs_error * loss_range_mask.float()
