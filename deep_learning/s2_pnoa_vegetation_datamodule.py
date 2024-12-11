from pnoa_vegetation import PNOAVegetation
from s2_mosaic import S2Mosaic

from lightning import LightningDataModule

from kornia_intersection_dataset import KorniaIntersectionDataset
import kornia.augmentation as K

import torch
from torch import Tensor
from torch.utils.data import DataLoader, _utils

from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import GeoDataset
from torchgeo.datasets.splits import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datamodules import GeoDataModule

from rasterio.crs import CRS

from typing import Dict, Optional, Union, Type, Tuple, Callable

class Config:
    NODATA = {
        'red': 0, 'green': 0, 'blue': 0, 'nir': 0,
        'swir16': 0, 'swir22': 0, 'rededge1': 0,
        'rededge2': 0, 'rededge3': 0, 'nir08': 0
    }
    
    MINS = {k: 1 for k in NODATA.keys()}
    MAXS = {k: 10000 for k in NODATA.keys()}
    
    DEFAULT_PATCH_SIZE = 352
    DEFAULT_BATCH_SIZE = 128
    DEFAULT_PREDICT_PATCH_SIZE = 2048
    DEFAULT_NAN_TARGET = -1.0
    DEFAULT_SEED = 43554578


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
    
class S2MinMaxNormalize(K.IntensityAugmentationBase2D):
    """Normalize S2 bands."""

    def __init__(self, mins: Tensor, maxs: Tensor) -> None:
        super().__init__(p=1)
        self.flags = {"min": mins.view(-1,1,1), "max": maxs.view(-1,1,1)}

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, int],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        return (input - flags["min"].to(torch.device(input.device))) / (flags["max"].to(torch.device(input.device)) - flags["min"].to(torch.device(input.device)) + 1e-8)        

class S2PNOAVegetationDataModule(GeoDataModule):

    def __init__(
        self,
        data_dir: str = "path/to/dir",
        patch_size: int = Config.DEFAULT_PATCH_SIZE,
        batch_size: int = Config.DEFAULT_BATCH_SIZE,
        length: int | None = None, 
        num_workers: int = 0, 
        seed: int = Config.DEFAULT_SEED, 
        predict_patch_size: int = Config.DEFAULT_PREDICT_PATCH_SIZE,
        nan_target: float = -1.0
    ):

        super().__init__(
            GeoDataset, batch_size, patch_size, length, num_workers
        )
        
        self.save_hyperparameters()
        self._setup_augmentations()

    def _setup_augmentations(self) -> None:
        """Initialize all augmentation pipelines with enhanced transforms."""
        nodata = torch.tensor([Config.NODATA[b] for b in S2Mosaic.all_bands])
        mins = torch.tensor([Config.MINS[b] for b in S2Mosaic.all_bands])
        maxs = torch.tensor([Config.MAXS[b] for b in S2Mosaic.all_bands])
        
        # Base normalization and mask transforms remain the same
        base_mask_transforms = K.AugmentationSequential(
            PNOAVegetationRemoveAbnormalHeight(nan_target=self.hparams.nan_target),
            PNOAVegetationInput0InArtificialSurfaces(),
            data_keys=None,
            keepdim=True
        )
        
        self.train_aug = {
            'general': K.AugmentationSequential(
                # Geometric augmentations (coordinate-preserving)
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.5),  # Only 90-degree rotations to preserve pixel grid
                
                # Intensity augmentations
                K.RandomGaussianNoise(mean=0.0, std=0.01, p=0.3),
                K.RandomBrightness(brightness=0.1, p=0.3),
                K.RandomContrast(contrast=0.1, p=0.3),
                  
                # Weather/Atmospheric simulation
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5), p=0.2),
                
                data_keys=None,
                keepdim=True,
                same_on_batch=False,
            ),
            
            'image': K.AugmentationSequential(
                # Base normalization
                S2MinMaxNormalize(mins, maxs),
                data_keys=None,
                keepdim=True
            ),
            
            'mask': K.AugmentationSequential(
                base_mask_transforms,
                
                # Careful noise addition to target
                K.RandomGaussianNoise(
                mean=0.0,
                std=0.005,  # Very small noise for regularization
                p=0.2
            ),
            
            data_keys=None,
            keepdim=True
            )
        }

        # Validation/Test augmentations - keep only normalization
        self.val_aug = self.test_aug = {
            'image': K.AugmentationSequential(
                S2MinMaxNormalize(mins, maxs),
                data_keys=None,
                keepdim=True
            ),
            'mask': base_mask_transforms
        }
        
        # Prediction augmentations
        self.predict_aug = {
            'image': self.val_aug['image']
        }

    def setup(self, stage: str) -> None:
        """Set up datasets for each stage."""
        s2 = S2Mosaic(self.hparams.data_dir)
        
        if stage == 'predict':
            self._setup_predict(s2)
        else:
            self._setup_train_val_test(s2)

    def _setup_predict(self, s2: S2Mosaic) -> None:
        """Set up prediction dataset and sampler."""
        self.predict_dataset = s2
        size = self.hparams.predict_patch_size
        stride = size - 256
        self.predict_sampler = GridGeoSampler(self.predict_dataset, size, stride)

    def _setup_train_val_test(self, s2: S2Mosaic) -> None:
        """Set up training, validation and test datasets and samplers."""
        pnoa_vegetation = PNOAVegetation(self.hparams.data_dir)
        dataset = KorniaIntersectionDataset(s2, pnoa_vegetation)
        
        # Split dataset
        splits = random_grid_cell_assignment(
            dataset,
            [0.8, 0.1, 0.1],
            grid_size=6,
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )
        self.train_dataset, self.val_dataset, self.test_dataset = splits
        
        # Set up samplers based on stage
        if self.trainer.training:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length
            )
        
        if self.trainer.training or self.trainer.validating:
            self.val_sampler = GridGeoSampler(
                self.val_dataset,
                self.hparams.patch_size,
                self.hparams.patch_size
            )
        
        if self.trainer.testing:
            self.test_sampler = GridGeoSampler(
                self.test_dataset,
                self.hparams.patch_size,
                self.hparams.patch_size
            )

    def on_after_batch_transfer(
        self, batch: Dict[str, Tensor], dataloader_idx: int
    ) -> Dict[str, Tensor]:

        """Apply augmentations after batch transfer."""
        if not self.trainer:
            return batch
            
        # Determine split and get corresponding augmentations
        split = self._get_current_split()
        aug = self._valid_attribute(f"{split}_aug", "aug")
        
        # Remove geo information
        batch = {k: v for k, v in batch.items() if k not in ['crs', 'bbox']}
        
        # Apply augmentations
        batch['image'] = aug['image']({'image': batch['image']})['image']
        if 'mask' in aug and 'mask' in batch:
            batch['mask'] = aug['mask']({'image': batch['mask']})['image']
        
        if 'general' in aug:
            batch = aug['general'](batch)
            batch['image'] = batch['image'].to('mps:0')
        
        return batch

    def _get_current_split(self) -> str:
        """Determine current split based on trainer state."""
        if self.trainer.training:
            return 'train'
        elif self.trainer.validating or self.trainer.sanity_checking:
            return 'val'
        elif self.trainer.testing:
            return 'test'
        elif self.trainer.predicting:
            return 'predict'
        return 'val'  # default



    @staticmethod        
    def collate_crs_fn(batch,*,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        return batch[0]

    @staticmethod
    def collate_bbox_fn(batch, *,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None ):
        return batch[0]

    @staticmethod
    def collate_geo(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        collate_map = {
            torch.Tensor : _utils.collate.collate_tensor_fn,
            CRS : S2PNOAVegetationDataModule.collate_crs_fn,
            BoundingBox : S2PNOAVegetationDataModule.collate_bbox_fn}
        return _utils.collate.collate(batch, collate_fn_map=collate_map)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        
        if isinstance(batch, dict):
            batch['image'] = batch['image'].to(device)
            if not self.trainer.predicting:
                batch['mask'] = batch['mask'].float().to(device)

        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
        


    








