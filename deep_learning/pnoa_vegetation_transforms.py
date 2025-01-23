import kornia.augmentation as K
import torch
from torch import Tensor
from typing import Dict, Optional

class PNOAVegetationRemoveAbnormalHeight(K.IntensityAugmentationBase2D):
    """Remove height outliers by assigning a no-data value equal to -1.0 meters"""

    hmax = 60.0 # Conserative height threshold

    def __init__(self, hmax: float = 60.0, nan_target: float = -1.0 ) -> None:
        super().__init__(p=1)
        self.nan_target = nan_target
        self.flags = {"hmax" : torch.tensor(hmax).view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        input[(input > flags['hmax'].to(torch.device(input.device)))] = self.nan_target
        return input   

class PNOAVegetationInput0InArtificialSurfaces(K.IntensityAugmentationBase2D):
    """Artificial surfaces are associated with a value of -32767.0, input a value of 0 meters so the NN learns to identify absence of vegetation because of artificial surface"""

    def __init__(self, nan_value: float = -32767.0) -> None:
        super().__init__(p=1)
        self.flags = {"nodata" : torch.tensor(nan_value).view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Set it to nodata
        input[(input - flags['nodata'].to(torch.device(input.device))) == 0] = 0.0
        return input