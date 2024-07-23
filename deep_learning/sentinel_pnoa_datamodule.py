from lightning import LightningDataModule
from pnoa_vegetation_nDSM import PNOAnDSMV
from sentinel_worldcover_composites import Sentinel1,Sentinel2NDVI,Sentinel2RGBNIR,Sentinel2SWIR
import torch
from torch.utils.data import DataLoader
from torchgeo.samplers import RandomBatchGeoSampler, RandomGeoSampler
from torchgeo.datasets import IntersectionDataset
from torchgeo.datasets.splits import random_grid_cell_assignment
from torchgeo.datasets.utils import BoundingBox
from rasterio.crs import CRS
from torch.utils.data import _utils
from typing import Callable, Dict, Optional, Tuple, Type, Union

class SentinelPNOADataModule(LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", sample_size: int = 256, batch_size: int = 128, length: int = 10000, transform = None, seed: int = 42):
        super().__init__()
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.seed = seed

    def setup(self, stage: str):

        swir_dataset = Sentinel2SWIR(self.data_dir)
        rgbnir_dataset = Sentinel2RGBNIR(self.data_dir)
        ndvi_dataset = Sentinel2NDVI(self.data_dir)
        vvvhratio_dataset = Sentinel1(self.data_dir)

        pnoa_dataset = PNOAnDSMV(self.data_dir)

        # SWIR dataset is put at the end to upsample it from 20m to 10m resolution instead of downsampling the rest
        sentinel = rgbnir_dataset & ndvi_dataset & vvvhratio_dataset & swir_dataset
        
        # Perform identical  splits by fixing the seed at fit and test stages to ensure that we are not training on test set. 
        if stage == 'fit' or stage == 'test':
            
            # This will downsample the canopy height data from 2,5m to 10m resolution.
            # sentinel_pnoa = IntersectionDataset(sentinel, pnoa_dataset)
            sentinel_pnoa = IntersectionDataset(sentinel, pnoa_dataset, transforms=self.transform)
            
            self.set_train, self.set_val, self.set_test = random_grid_cell_assignment(
                sentinel_pnoa, [0.8,0.1,0.1], grid_size = 6, generator=torch.Generator().manual_seed(self.seed)
            )
        
        # Do not perform image augmentations for predicting.
        if stage == 'predict':
            self.set_predict = IntersectionDataset(sentinel, pnoa_dataset)

    @staticmethod        
    def collate_crs_fn(batch,*,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        return batch[0]

    @staticmethod
    def collate_bbox_fn(batch, *,collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None ):
        return batch[0]

    @staticmethod
    def collate_geo(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        collate_map = {torch.Tensor : _utils.collate.collate_tensor_fn, CRS : SentinelPNOADataModule.collate_crs_fn, BoundingBox : SentinelPNOADataModule.collate_bbox_fn}
        return _utils.collate.collate(batch, collate_fn_map=collate_map)

    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_train, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        dataloader = DataLoader(self.set_train, batch_sampler=sampler, num_workers=0, collate_fn = self.collate_geo)
        return dataloader

    def val_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_val, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        dataloader = DataLoader(self.set_val, batch_sampler=sampler, num_workers=0, collate_fn = self.collate_geo)
        return dataloader
    
    def test_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_test, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        return  DataLoader(self.set_test, batch_sampler=sampler, num_workers=0, collate_fn = self.collate_geo)

    def predict_dataloader(self):
        return DataLoader(self.set_predict, batch_size=self.batch_size, num_workers=0)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict):
            # move all tensors in your custom data structure to the device
            batch['image'] = batch['image'].to(device)
            batch['mask'] = batch['mask'].to(device)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch
