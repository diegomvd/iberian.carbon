from pnoa_vegetation import PNOAVegetation
from s2_mosaic import S2Mosaic

from lightning import LightningDataModule

from kornia_intersection_dataset import KorniaIntersectionDataset
import kornia.augmentation as K

import torch
from torch import Tensor, Generator
from torch.utils.data import DataLoader, _utils

from balanced_geo_samplers import WeightedHeightComplexitySampler

from pnoa_vegetation_transforms import PNOAVegetationInput0InArtificialSurfaces, PNOAVegetationRemoveAbnormalHeight, PNOAVegetationLogTransform, S2Scaling

from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler

from torchgeo.datasets import GeoDataset
from torchgeo.datasets.splits import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from torchgeo.datamodules import GeoDataModule

from rasterio.crs import CRS

from typing import Dict, Optional, Union, Type, Tuple, Callable, List, Iterator

class Config:
    DEFAULT_NAN_INPUT = 0.0 #-9999.0
    DEFAULT_PATCH_SIZE = 256
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_PREDICT_PATCH_SIZE = 2048
    DEFAULT_NAN_TARGET = -1.0
    DEFAULT_SEED = 43554578

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
        nan_target: float = Config.DEFAULT_NAN_TARGET,
        nan_input: float = Config.DEFAULT_NAN_INPUT,
        phase_epochs: List[int] = [50,50,50,75,75]  
    ):

        super().__init__(
            GeoDataset, batch_size, patch_size, length, num_workers
        )
        
        self.save_hyperparameters()
        self._setup_augmentations()
        self.phase_epochs = np.cumsum(phase_epochs)
        self.current_epoch = 0

    def get_current_phase(self, epoch:int) -> int:
        self.current_epoch = epoch
        return torch.searchsorted(self.phase_epochs, epoch)

    def _setup_augmentations(self) -> None:
        """Initialize all augmentation pipelines with enhanced transforms."""
        
        # Base normalization and mask transforms remain the same
        base_mask_transforms = K.AugmentationSequential(
            PNOAVegetationRemoveAbnormalHeight(nan_target=self.hparams.nan_target),
            PNOAVegetationInput0InArtificialSurfaces(),
            PNOAVegetationLogTransform(),
            data_keys=None,
            keepdim=True
        )
        
        self.train_aug = {
            'general': K.AugmentationSequential(
                # Geometric augmentations only
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomRotation(degrees=90, p=0.5),
                data_keys=None,
                keepdim=True,
                same_on_batch=False,
            ),
            'image': K.AugmentationSequential(
                S2Scaling(),
                # Reduced intensity augmentations for mosaiced data
                K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.3),
                K.RandomBrightness(brightness=0.1, p=0.3),
                K.RandomContrast(contrast=0.1, p=0.3),
                data_keys=None,
                keepdim=True,
            ),
            'mask': K.AugmentationSequential(
                base_mask_transforms,
                K.RandomGaussianNoise(
                    mean=0.0,
                    std=0.002,
                    p=0.2
                ),
                data_keys=None,
                keepdim=True
            )
        }

        # Validation/Test augmentations - keep only normalization
        self.val_aug = self.test_aug = {
            'image':K.AugmentationSequential( S2Scaling(), data_keys=None, keepdim=True),
            'mask': base_mask_transforms
        }
        
        # Prediction augmentations
        self.predict_aug = {
            'image':K.AugmentationSequential( S2Scaling(), data_keys=None, keepdim=True)
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        
        if isinstance(batch, dict):
            batch['image'] = batch['image'].to(device)
            if not self.trainer.predicting:
                batch['mask'] = batch['mask'].float().to(device)

        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

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
        if 'image' in aug and 'image' in batch:
            batch['image'] = aug['image']({'image': batch['image']})['image']
        if 'mask' in aug and 'mask' in batch:
            batch['mask'] = aug['mask']({'image': batch['mask']})['image']
        
        if 'general' in aug:
            batch = aug['general'](batch)
            batch['image'] = batch['image'].to('mps:0')
        
        return batch

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

    def train_dataloader(self) -> DataLoader:
        phase = self.get_current_phase(self.trainer.current_epoch)
        
        if phase == 0:  # Phase 1: slight emphasis on taller trees while keeping some complexity
            sampler = WeightedHeightComplexitySampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length,
                height_exp = 1.5,         
            )
        elif phase == 1:  # Phase 2: aggressive emphasis on tall trees while keeping some complexity
            sampler = WeightedHeightComplexitySampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length,
                height_exp = 3.0,   
            )
        elif phase == 2:  # Phase 3: less agressive emphasis on tall trees
            sampler = WeightedHeightComplexitySampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length,
                height_exp = 1.5,   
            )
        elif phase == 3:  # Phase 4 mild emphasis on tall trees
            sampler = WeightedHeightComplexitySampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length,
                height_exp = 0.5,   
            )   
        elif phase == 4:  # Phase 4 only focuses on complexity
            sampler = WeightedHeightComplexitySampler(
                self.train_dataset,
                self.hparams.patch_size,
                self.hparams.batch_size,
                self.hparams.length,
                height_exp = 0.0,   
            )    

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )

    def _setup_train_val_test(self, s2: S2Mosaic) -> None:
        """Set up training, validation and test datasets and samplers."""
        pnoa_vegetation = PNOAVegetation(self.hparams.data_dir)
        dataset = KorniaIntersectionDataset(s2, pnoa_vegetation)
        
        # Split dataset
        splits = random_grid_cell_assignment(
            dataset,
            [0.8, 0.1, 0.1],
            grid_size=8,
            generator=torch.Generator().manual_seed(self.hparams.seed)
        )
        self.train_dataset, self.val_dataset, self.test_dataset = splits
        
        # Set up samplers based on stage
        # if self.trainer.training:
        #     self.train_batch_sampler = TallVegetationSampler(
        #         self.train_dataset,
        #         self.hparams.patch_size,
        #         self.hparams.batch_size,
        #         self.hparams.length
        #     )

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

    
        


    








