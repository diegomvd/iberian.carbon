from typing import Any, Optional, Sequence, Union

import torch
from torch import Tensor, tensor

from typing import Tuple

from torchmetrics.functional.regression.mae import _mean_absolute_error_compute
from torchmetrics.metric import Metric

from torchmetrics.functional.regression.mape import _mean_absolute_percentage_error_compute

from torchmetrics.utilities.checks import _check_same_shape

from torchmetrics.functional.regression.mse import _mean_squared_error_compute


def _nan_robust_mean_absolute_error_update(preds: Tensor, target: Tensor, nan_value: float) -> Tuple[Tensor, int]:
    """Update and returns variables required to compute Mean Absolute Error removing NaN values before reduction.

    Check for same shape of input tensors.

    Args:
        preds: Predicted tensor
        target: Ground truth tensor
        num_outputs: Number of outputs in multioutput setting

    """
    _check_same_shape(preds, target)
    
    preds = preds.view(-1)
    target = target.view(-1)
    
    preds = preds if preds.is_floating_point else preds.float()  # type: ignore[truthy-function] # todo
    target = target if target.is_floating_point else target.float()  # type: ignore[truthy-function] # todo
    
    abs_error = torch.abs(preds - target)
    target_nan_mask = target!=nan_value # create nan mask
    abs_error = torch.masked_select(abs_error,target_nan_mask) 
    
    sum_abs_error = torch.sum(abs_error)
    return sum_abs_error, target.shape[0]

def _nan_robust_mean_absolute_percentage_error_update(
    preds: Tensor,
    target: Tensor,
    nan_value: float,
    epsilon: float = 1.17e-06
) -> Tuple[Tensor, int]:
    
    _check_same_shape(preds, target)

    abs_diff = torch.abs(preds - target)
    abs_per_error = abs_diff / torch.clamp(torch.abs(target), min=epsilon)

    target_nan_mask = target!=nan_value # create nan mask
    abs_per_error = torch.masked_select(abs_per_error,target_nan_mask) 

    sum_abs_per_error = torch.sum(abs_per_error)

    num_obs = target.numel()

    return sum_abs_per_error, num_obs

def _nan_robust_mean_squared_error_update(preds: Tensor, target: Tensor, nan_value: float) -> Tuple[Tensor, int]:
    
    _check_same_shape(preds, target)
    preds = preds.view(-1)
    target = target.view(-1)
 
    diff = preds - target

    target_nan_mask = target!=nan_value # create nan mask
    diff = torch.masked_select(diff,target_nan_mask) 

    sum_squared_error = torch.sum(diff * diff, dim=0)

    return sum_squared_error, target.shape[0]    

class NanRobustMeanAbsoluteError(Metric):

    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_error: Tensor
    total: Tensor

    def __init__(
        self,
        nan_value: float = -32767.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.nan_value = nan_value
        self.add_state("sum_abs_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""

        sum_abs_error, num_obs = _nan_robust_mean_absolute_error_update(preds, target, nan_value=self.nan_value)

        self.sum_abs_error += sum_abs_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)

class NanRobustMeanAbsolutePercentageError(Metric):
   
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    sum_abs_per_error: Tensor
    total: Tensor

    def __init__(
        self,
        nan_value: float = -32767.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.nan_value = nan_value
        self.add_state("sum_abs_per_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_abs_per_error, num_obs = _nan_robust_mean_absolute_percentage_error_update(preds, target, self.nan_value)

        self.sum_abs_per_error += sum_abs_per_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean absolute percentage error over state."""
        return _mean_absolute_percentage_error_compute(self.sum_abs_per_error, self.total)

class NanRobustMeanSquaredError(Metric):
    
    is_differentiable = True
    higher_is_better = False
    full_state_update = False
    plot_lower_bound: float = 0.0

    sum_squared_error: Tensor
    total: Tensor

    def __init__(
        self,
        squared: bool = True,
        nan_value: float = -32767.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if not isinstance(squared, bool):
            raise ValueError(f"Expected argument `squared` to be a boolean but got {squared}")
        self.squared = squared

        self.nan_value = nan_value

        self.add_state("sum_squared_error", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        sum_squared_error, num_obs = _nan_robust_mean_squared_error_update(preds, target, nan_value = self.nan_value)

        self.sum_squared_error += sum_squared_error
        self.total += num_obs

    def compute(self) -> Tensor:
        """Compute mean squared error over state."""
        return _mean_squared_error_compute(self.sum_squared_error, self.total, squared=self.squared)
