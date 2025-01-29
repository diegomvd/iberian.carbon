
import torch.nn as nn
import torch
from typing import Optional

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
        lambda_reg: float = 0.1,
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


