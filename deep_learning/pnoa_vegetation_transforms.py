import kornia.augmentation as K
import torch
from torch import Tensor
from typing import Dict, Optional

class S2Scaling(K.IntensityAugmentationBase2D):
    """Remove height outliers by assigning a no-data value equal to -1.0 meters"""

    def __init__(self, nan_input: float = 0.0 ) -> None:
        super().__init__(p=1)
        self.nan_input = nan_input

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.where(input != self.nan_input, input/10000, input)
       

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
        input[(input - flags['nodata'].to(torch.device(input.device))) == 0] = -1.0
        return input

class PNOAVegetationLogTransform(K.IntensityAugmentationBase2D):
    def __init__(self, nodata: float = -1.0):
        super().__init__(p=1.0)  # Always apply
        self.nodata = nodata

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
#        print(f'Input max: {input.max()}')
        ret = torch.where(input!=self.nodata, torch.log1p(input), input)    
 #       print(f'Input max after log: {ret.max()}')
        return  ret
