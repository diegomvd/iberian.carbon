from lightning import LightningDataModule
from pnoa_vegetation_nDSM import PNOAnDSMV
from sentinel_worldcover_composites import Sentinel1,Sentinel2NDVI,Sentinel2RGBNIR,Sentinel2SWIR
import torch
from torch.utils.data import random_split, DataLoader
from torchgeo.samplers import RandomBatchGeoSampler
from torchgeo.datasets import IntersectionDataset

class SentinelPNOADataModule(LightningDataModule):

    def __init__(self, data_dir: str = "path/to/dir", sample_size: int = 256, batch_size: int = 128, length: int = 10000, transform = transform, seed: int = 42):
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
            sentinel_pnoa = IntersectionDataset(sentinel, pnoa_dataset, transform=self.transform)
            
            self.set_train, self.set_val, self.set_test = random_split(
                sentinel_pnoa, [0.8,0.1,0.1], generator=torch.Generator().manual_seed(self.seed)
            )
        
        # Do not perform image augmentations for predicting.
        if stage == 'predict':
            self.set_predict = IntersectionDataset(sentinel, pnoa_dataset)


    def train_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_train, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        return DataLoader(self.set_train, sampler=sampler, num_workers=0)

    def val_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_val, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        return DataLoader(self.set_val, sampler=sampler, num_workers=0)

    def test_dataloader(self):
        sampler = RandomBatchGeoSampler(self.set_test, size = self.sample_size, batch_size = self.batch_size, length = self.length)
        return DataLoader(self.set_test, sampler=sampler, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.set_predict, batch_size=self.batch_size, num_workers=0)

    # def transfer_batch_to_device(self, batch, device, dataloader_idx):
    #     if isinstance(batch, dict):
    #         # move all tensors in your custom data structure to the device
    #         batch['image'] = batch['image'].to(device)
    #     else:
    #         batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
    #     return batch