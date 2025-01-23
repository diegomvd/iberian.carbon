
import torch.nn as nn
import torch
from typing import Optional

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

        weights = torch.pow(0.1 + target/self.weight_href, self.beta)
                                         
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
